#!/usr/bin/env python3
"""storage_rounds.py -- T4: does the single-round advantage survive a MEMORY?

THE QUESTION.  Every fidelity in the paper is a SINGLE round: noise, syndrome
projection, recovery, decode.  A quantum memory is not one round.  What a reader
actually needs to know is whether VSCR's certified non-Pauli headroom (1.9e-4 to
3.7e-3 over the rigid Pauli decoder) compounds into a longer storage lifetime or
decays away after a few rounds -- and whether the non-Pauli recovery, which does
not map syndromes to Pauli corrections, is more fragile under mid-circuit readout
error than the decoder is.  Neither question is answerable from the single-round
numbers, because the round map is not a scalar.

THE ROUND MAP.  One round of noise N, syndrome measurement and
syndrome-conditioned recovery R_s is the instrument

    E(rho) = sum_s R_s P_s N(rho) P_s R_s^dagger ,

which is CP and trace-preserving, and a memory of R rounds is E^R.  Writing
P_s = W_s W_s^dagger and G_s = V^dagger R_s W_s, the action restricted to the code
space collapses to a map on the 2x2 LOGICAL density matrix,

    Lam(sigma) = sum_s G_s ( sum_k A_k^(s) sigma A_k^(s)^dagger ) G_s^dagger ,
    A_k^(s)    = W_s^dagger K_k V ,

which is exactly the reduced-branch formalism `vscr_paper_abl` and
`vscr_general` already use for the single-round ceilings -- so the multi-round
simulation is built from the same audited 2x2 blocks rather than from a new
density-matrix code path.  Because Lam is linear in sigma and
F = <psi|Lam^R(|psi><psi|)|psi> is therefore degree 2 in the Bloch vector, the
existing 3x7 Gauss-Legendre/azimuthal `haar_quadrature` is EXACT for every R, not
just for R=1: there is no Monte-Carlo error anywhere in this module.

TWO SIMULATIONS, CROSS-CHECKED.  `Lam` above projects back with V^dagger and so
silently discards whatever R_s pushes OUT of the code space.  `_round_full`
therefore also evolves the whole dim x dim density matrix, using the identity
R_s P_s rho P_s R_s^dagger = (R_s W_s)(W_s^dagger rho W_s)(R_s W_s)^dagger, which
costs O(dim^2) per branch instead of O(dim^3) and makes [[9,1,3]] affordable.
Comparing the two measures the LEAKAGE, i.e. exactly the term the reduced map
drops; `--selftest` requires them to agree to the measured unitarity deviation.

READOUT ERROR.  With per-bit readout error eta the true syndrome is s but the
recovery applied is R_s~, so the round map needs the CROSS-branch blocks
G~[s~,s] = V^dagger R_s~ W_s, which `StabCode.cross_G` supplies.  For a Pauli
decoder every off-diagonal block vanishes identically (the paper's exact
suppression law), but a learned non-Pauli R_s has no such protection -- so this
module measures whether VSCR's advantage survives realistic mid-circuit readout
over many rounds instead of assuming it does.

ANCHORS (--validate).  The R=1 reduced map must reproduce, bit-for-bit where the
code path is shared and to the unitarity deviation otherwise:
  * `abl.exact_F(phi, noise, p)` for the VSCR/VQR-ind tables;
  * `opt_unitary_ceiling(noise, p)['F_dec']` for the Pauli decoder;
  * `ssvr_qec.vscr_fidelity` on a full 32x32 density matrix.
A multi-round number that does not reduce to the audited single-round number at
R=1 is meaningless, so this is a gate, not a diagnostic.

Usage:
    ./qenv/bin/python storage_rounds.py --selftest
    ./qenv/bin/python storage_rounds.py --validate
    ./qenv/bin/python storage_rounds.py                  # full T4 sweep
    ./qenv/bin/python storage_rounds.py --device cuda    # GPU full-space path
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Single-threaded BLAS BEFORE numpy: see the note in scaling_analysis.py.  The
# thread count changes which eigenvector basis LAPACK returns for StabCode's
# degenerate stabilizer eigenspaces, and with it W_s and every reduced block.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('OPENBLAS_CORETYPE', 'HASWELL')

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
import ssvr_qec as m                                            # noqa: E402
import vscr_general as g                                        # noqa: E402
import vscr_paper_abl as abl                                    # noqa: E402
# `vscr_paper` is deliberately NOT imported here: nothing in this module uses it,
# and vscr_paper_abl already pulls it in (which is where PHI_DEC and the warm
# schedules come from).  An unused direct import would only imply a dependency
# that does not exist.

OUT_JSON = 'storage_rounds.json'
CHANNELS = ('depolarizing', 'amplitude_damping', 'mixed', 'coherent')
CODE_ORDER = ('5,1,3', '7,1,3', '9,1,3')
SHORT = {'depolarizing': 'dep', 'amplitude_damping': 'ad', 'mixed': 'mixed',
         'coherent': 'coh'}
# Recovery tables at n=5, produced by the audited paper runs.
WARM_NPZ = {'depolarizing': 'vscr_angles_paper_dep.npz',
            'amplitude_damping': 'vscr_angles_paper_ad.npz',
            'mixed': 'vscr_angles_paper_mixed.npz',
            'coherent': 'vscr_angles_paper_coh.npz'}
IND_NPZ = {'depolarizing': 'vscr_angles_abl_ind_dep.npz',
           'amplitude_damping': 'vscr_angles_abl_ind_ad.npz',
           'mixed': 'vscr_angles_abl_ind_mixed.npz',
           'coherent': 'vscr_angles_abl_ind_coh.npz'}
# 'mixed' is dense-Kraus only up to n=5 (see vscr_general.branch_A), so the
# larger codes run the three channels that scale.
CHANNELS_BY_CODE = {'5,1,3': CHANNELS,
                    '7,1,3': ('depolarizing', 'amplitude_damping', 'coherent'),
                    '9,1,3': ('depolarizing', 'amplitude_damping', 'coherent')}
# Readout error per syndrome bit per round.  scaling_analysis.py uses 0.005 as a
# realistic mid-circuit value and a few percent for cloud devices.
READOUT_GRID = (0.0, 0.005, 0.01, 0.02, 0.05)
ROUND_GRID = (1, 2, 3, 5, 8, 12, 16, 20, 30, 40)
# The exact 3x7 Haar rule: 21 states, exact for any degree-2 Bloch polynomial,
# hence exact for Lam^R at every R (see the module docstring).
HAAR_N_U, HAAR_N_PHI = 3, 7


# ---------------------------------------------------------------------
# The round map as a 4x4 superoperator on the 2x2 LOGICAL space
# ---------------------------------------------------------------------
_CODE_CACHE = {}


def get_code(spec_key):
    """Cached StabCode; construction runs ~40 asserts, so build each once."""
    if spec_key not in _CODE_CACHE:
        _CODE_CACHE[spec_key] = g.StabCode(g.CODES[spec_key])
    return _CODE_CACHE[spec_key]


def _vec_rowmajor(X):
    """vec(X) in ROW-MAJOR order, the convention for which
    vec(A X B) = (A kron B^T) vec(X)."""
    return np.asarray(X, dtype=complex).reshape(-1)


def branch_kraus_stacked(code, channel, p):
    """{A_k^(s)} as a list of (K_s, 2, 2) stacks, one per syndrome branch.

    Stacking lets the branch map sigma -> sum_k A sigma A^dagger be one batched
    einsum instead of a Python loop over up to ~1000 Kraus operators per branch
    at n=9, which is the difference between a round costing milliseconds and
    minutes.
    """
    A = g.branch_A(code, channel, p)
    return [np.stack([np.asarray(a, dtype=complex) for a in br])
            if len(br) else np.zeros((0, 2, 2), dtype=complex) for br in A]


def branch_p_weights(Astack):
    """p_s = Tr[sum_k A_k^dagger A_k] / 2, the branch probability.

    Recomputed here rather than taken from a second code path so that the
    probabilities and the map used to evolve the state cannot disagree.
    """
    ps = []
    for A in Astack:
        if A.shape[0] == 0:
            ps.append(0.0)
            continue
        acc = np.einsum('kij,kil->jl', A.conj(), A)     # sum_k A_k^dag A_k
        ps.append(float(np.real(np.trace(acc)) / 2.0))
    return np.asarray(ps, dtype=float)


def _branch_superop(A):
    """sum_k kron(A_k, conj(A_k)) for a (K,2,2) stack, as one vectorised einsum.

    kron(X,Y)[2i+a, 2j+b] = X[i,j] Y[a,b], so the whole Kraus sum is
    einsum('kij,kab->iajb').reshape(4,4).  At [[9,1,3]] a depolarizing round has
    4^9 = 262144 Kraus operators spread over 256 branches, so doing this as K
    separate np.kron calls per branch costs seconds where the einsum costs
    milliseconds -- and it is the same expression, not an approximation.
    """
    A = np.asarray(A, dtype=complex)
    if A.shape[0] == 0:
        return np.zeros((4, 4), dtype=complex)
    return np.einsum('kij,kab->iajb', A, A.conj()).reshape(4, 4)


def round_superop(Astack, G, readout=None):
    """The 4x4 superoperator S of one storage round, vec row-major.

        Lam(sigma) = sum_s G_s ( sum_k A_k sigma A_k^dag ) G_s^dag
        S          = sum_s (G_s kron conj(G_s)) sum_k (A_k kron conj(A_k))

    `readout`, when given, is the (nsyn~, nsyn) conditional table C(s~|s) and
    `G` must then be the CROSS-branch block table G~[s~, s] = V^dag R_s~ W_s, so
    that the recovery applied is the one the noisy readout reported rather than
    the one the state actually had:

        Lam_eta(sigma) = sum_{s,s~} C(s~|s) G~[s~,s] B_s(sigma) G~[s~,s]^dag

    R rounds is then just S^R -- which is why an arbitrarily long memory costs no
    more than one round, and why the asymptotic decay rate is an eigenvalue of S
    rather than something fitted from a long simulation.
    """
    nsyn = len(Astack)
    G = np.asarray(G, dtype=complex)
    # B_s as one (nsyn,4,4) stack, so applying the whole round to a matrix is a
    # single batched matmul rather than a Python loop over branches.
    Bs = np.stack([_branch_superop(A) for A in Astack])
    # Build S column by column: column j is vec(Lam(E_j)) for the four matrix
    # units E_j.  Four vectorised passes replace nsyn (or nsyn^2 with readout
    # error) small kron/matmul calls, which at [[9,1,3]] is 256 or 65536 of them.
    S = np.zeros((4, 4), dtype=complex)
    for j in range(4):
        e = np.zeros(4, dtype=complex)
        e[j] = 1.0
        allB = (Bs @ e).reshape(nsyn, 2, 2)              # B_s(E_j) for every s
        if readout is None:
            out = np.einsum('sij,sjk,slk->il', G, allB, G.conj())
        else:
            C = np.asarray(readout, dtype=float)
            out = np.einsum('tsij,sjk,tslk,ts->il', G, allB, G.conj(), C)
        S[:, j] = out.reshape(4)
    return S


def readout_confusion(nsyn, n_bits, eta):
    """C(s~|s): independent bit-flip probability eta on each of n_bits bits.

    Row s~ column s, rows summing to 1.  Built from the Hamming distance between
    syndrome labels, so it is exactly the product channel the readout law in
    `scaling_analysis.py` assumes.
    """
    C = np.zeros((nsyn, nsyn), dtype=float)
    for s in range(nsyn):
        for st in range(nsyn):
            d = bin(s ^ st).count('1')
            C[st, s] = eta ** d * (1.0 - eta) ** (n_bits - d)
    assert np.abs(C.sum(0) - 1.0).max() < 1e-12, C.sum(0)
    return C


# ---------------------------------------------------------------------
# Fidelity after R rounds, exactly
# ---------------------------------------------------------------------
def haar_probe_set():
    """The audited 3x7 exact Haar rule: 21 logical states and their weights.

    Reused rather than reimplemented so the multi-round average is the SAME
    quadrature the single-round paper numbers come from.  It is exact for any
    degree-2 polynomial in the Bloch vector, and F(R) is such a polynomial for
    every R because Lam^R is linear in |psi><psi|.
    """
    states, w = m.haar_quadrature(HAAR_N_U, HAAR_N_PHI)
    st = np.asarray(states, dtype=complex)
    ww = np.asarray(w, dtype=float)
    assert st.shape[1] == 2 and abs(ww.sum() - 1.0) < 1e-12, (st.shape, ww.sum())
    return st, ww


_PROBE = None


def probes():
    global _PROBE
    if _PROBE is None:
        st, ww = haar_probe_set()
        # Pre-vectorise: sigma_j and the bra/ket contraction for each probe, so a
        # whole (R, channel, recovery) sweep is matrix products with no Python
        # loop over the 21 states.
        sig = np.stack([_vec_rowmajor(np.outer(p, p.conj())) for p in st])
        _PROBE = (st, ww, sig)
    return _PROBE


def fidelity_after_rounds(S, R):
    """F(R) = sum_j w_j <psi_j| Lam^R(|psi_j><psi_j|) |psi_j>, exact.

    This is the UNNORMALISED convention the paper uses everywhere (the same one
    `abl.exact_F` and `vscr_fidelity_quad` report), so F(R) is directly
    comparable to the single-round audited numbers and F(0) = 1 exactly.
    """
    st, ww, sig = probes()
    SR = np.linalg.matrix_power(np.asarray(S, dtype=complex), int(R))
    out = sig @ SR.T                                  # (nq, 4)
    tot = 0.0
    for j in range(st.shape[0]):
        o = out[j].reshape(2, 2)
        tot += ww[j] * float(np.real(st[j].conj() @ o @ st[j]))
    return tot


def fidelity_curve(S, rounds):
    """F(R) for every R in `rounds`, reusing the eigendecomposition of S.

    Diagonalising once and raising eigenvalues to R is both faster than R matrix
    powers and numerically cleaner for large R, where repeated multiplication
    would accumulate.  Falls back to matrix_power if S is defective.
    """
    S = np.asarray(S, dtype=complex)
    st, ww, sig = probes()
    try:
        ev, U = np.linalg.eig(S)
        cond = np.linalg.cond(U)
        if not np.isfinite(cond) or cond > 1e10:
            raise np.linalg.LinAlgError('ill-conditioned eigenbasis')
        Ui = np.linalg.inv(U)
        base = sig @ Ui.T                              # (nq, 4)
        out = []
        for R in rounds:
            lamR = ev ** R
            o = (base * lamR[None, :]) @ U.T           # (nq, 4)
            tot = 0.0
            for j in range(st.shape[0]):
                oj = o[j].reshape(2, 2)
                tot += ww[j] * float(np.real(st[j].conj() @ oj @ st[j]))
            out.append(tot)
        return out
    except np.linalg.LinAlgError:
        return [fidelity_after_rounds(S, R) for R in rounds]


def asymptotic_decay(S):
    """Spectral data of the round map, with an explicit warning about what it is.

    Lam is trace-preserving whenever every G_s is unitary, so 1 is an eigenvalue
    and F(R) tends to the fidelity of Lam's fixed point rather than to zero.  The
    modulus of the largest eigenvalue strictly below 1 then sets the R -> INFINITY
    tail -- and only the tail.  At the round counts anyone actually simulates the
    decay is usually dominated by a different eigenvalue that carries more weight
    in the observable, so quoting this rate as "the memory time" overstates it by
    orders of magnitude: on amplitude damping p=0.10 the warm recovery gives
    lam = 1 - 2.1e-9 (4.8e8 rounds) while F falls from 0.986 to 0.668 within 40
    rounds.  Reported for completeness; `storage_lifetime` is what to quote.
    """
    ev = np.linalg.eigvals(np.asarray(S, dtype=complex))
    mods = np.sort(np.abs(ev))[::-1]
    below = [x for x in mods if x < 1.0 - 1e-9]
    lam = float(below[0]) if below else 0.0
    return {'spectrum_moduli': [float(x) for x in mods],
            'trace_preserving_eigenvalue_one': bool(abs(mods[0] - 1.0) < 1e-9),
            'lam_logical': lam,
            # None rather than float('inf'): json.dump would emit the
            # non-standard token `Infinity`, which strict parsers reject.  A
            # missing value plus an explicit flag says the same thing legally.
            'lam_logical_efold_rounds': _finite_or_none(
                -1.0 / math.log(lam) if 0.0 < lam < 1.0 else float('inf')),
            'efold_capped': not (0.0 < lam < 1.0),
            'lam_logical_note': ('governs the R->infinity tail only; see '
                                 'storage_lifetime for the observable memory time')}


def _finite_or_none(x):
    """x as a float, or None if it is infinite/NaN, so the artifact stays
    strictly valid JSON (json.dump writes `Infinity` and `NaN`, which are not
    JSON).  Callers that need to distinguish 'unbounded' from 'missing' pair this
    with an explicit boolean flag."""
    x = float(x)
    return x if math.isfinite(x) else None


def fixed_point(S):
    """The trace-1 fixed point sigma_inf of the round map, from S's eigenvector.

    This replaces pushing `matrix_power` to R=10^6 for F_inf.  At that depth every
    sub-dominant eigenvalue has underflowed and the result is dominated by
    round-off accumulated in the squaring, which measured slightly ABOVE F(1000)
    and so contradicted the monotonicity of F(R) -- a numerical artifact that
    would have been reported as physics.  The eigenvector is exact instead.

    Returns (sigma_inf, eigenvalue).  sigma_inf is Hermitised after normalising by
    its trace: an eigenvector is only defined up to a complex scale, and the
    physical fixed point of a trace-preserving map is a real density matrix.
    """
    S = np.asarray(S, dtype=complex)
    ev, U = np.linalg.eig(S)
    k = int(np.argmin(np.abs(ev - 1.0)))
    sig = U[:, k].reshape(2, 2)
    tr = np.trace(sig)
    assert abs(tr) > 1e-12, 'fixed-point eigenvector is traceless: %r' % (sig,)
    sig = sig / tr
    sig = 0.5 * (sig + sig.conj().T)
    return sig, float(np.real(ev[k]))


def storage_lifetime(S, r_cap=10 ** 6):
    """The observable memory time: F_inf and the round count R_half.

    Both are read off F(R) rather than fitted to a decay model, so a reader can
    check them against the stored curve:

      F_inf   the fixed-point fidelity, the exact Haar average of
              <psi|sigma_inf|psi> for the round map's trace-1 fixed point;
      R_half  the first round at which F has fallen halfway from F(0) = 1 to
              F_inf, found by doubling then bisection.

    R_half is the number to quote for "how long is this memory", because unlike a
    single decay rate it does not assume the decay is exponential -- and it is not,
    whenever more than one eigenvalue carries weight in the observable.
    """
    sig_inf, ev1 = fixed_point(S)
    st, ww = haar_probe_set()
    F_inf = float(sum(w * float(np.real(p.conj() @ sig_inf @ p))
                      for p, w in zip(st, ww)))
    target = 0.5 * (1.0 + F_inf)
    if fidelity_after_rounds(S, 1) <= target:
        R_half = 1
    else:
        lo, hi = 1, 2
        while fidelity_after_rounds(S, hi) > target and hi < r_cap:
            lo, hi = hi, hi * 2
        if hi >= r_cap and fidelity_after_rounds(S, hi) > target:
            R_half = float('inf')
        else:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if fidelity_after_rounds(S, mid) > target:
                    lo = mid
                else:
                    hi = mid
            R_half = hi
    return {'F_inf': F_inf, 'R_half': _finite_or_none(R_half),
            # R_half is None exactly when the search hit r_cap: F had not fallen
            # halfway to F_inf within 10^6 rounds.  That is a LOWER bound on the
            # memory time, not a measurement of infinity, and it is what happens
            # for the learned recovery on coherent noise.
            'R_half_capped': not math.isfinite(R_half),
            'R_half_cap': r_cap,
            'fixed_point_eigenvalue': ev1,
            'fixed_point_purity': float(np.real(np.trace(sig_inf @ sig_inf))),
            'R_half_note': ('rounds until F falls halfway from 1 to F_inf; '
                            'model-free, so it stays meaningful when the decay is '
                            'not a single exponential')}


# ---------------------------------------------------------------------
# Recoveries: the per-branch blocks G_s = V^dagger R_s W_s
# ---------------------------------------------------------------------
RECOVERIES = ('decoder', 'warm', 'ind', 'ceiling', 'identity')


def _load_phi(kind, channel):
    """The audited (16, 60) angle table for a learned recovery at n=5."""
    path = (WARM_NPZ if kind == 'warm' else IND_NPZ)[channel]
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        raise FileNotFoundError(
            '%s missing -- run vscr_paper.py / vscr_paper_abl.py first' % path)
    phi = np.load(full)['phi']
    assert phi.shape == (16, m.PHI_DIM), phi.shape
    return np.asarray(phi, dtype=float)


def recovery_blocks(code, kind, channel=None, p=None):
    """Return (G, R_mats, meta) for the requested recovery.

    G       : (nsyn, 2, 2) diagonal blocks V^dagger R_s W_s, used at ideal readout
    R_mats  : (nsyn, dim, dim) full-space unitaries, or None when the recovery is
              known only as a 2x2 block table (the ceiling).  R_mats is what makes
              readout error simulable, because the cross-branch blocks
              V^dagger R_s~ W_s need the physical operator, not just its diagonal.
    meta    : diagnostics, including the unitarity deviation of G -- how far this
              recovery is from mapping each syndrome subspace isometrically into
              the code space, and hence how much the reduced map may drop.
    """
    nsyn, dim = code.nsyn, code.dim
    if kind == 'decoder':
        R = np.asarray(code.C_mat, dtype=complex)
        assert R.shape == (nsyn, dim, dim), R.shape
    elif kind == 'identity':
        R = np.stack([np.eye(dim, dtype=complex)] * nsyn)
    elif kind in ('warm', 'ind'):
        if code.n != 5:
            raise ValueError('%s recovery exists only at n=5 (never trained at '
                             'n=%d)' % (kind, code.n))
        with torch.no_grad():
            R = m.recovery_unitary_batch(
                torch.tensor(_load_phi(kind, channel),
                             dtype=torch.float64)).numpy().astype(complex)
    elif kind == 'ceiling':
        if code.n != 5:
            raise ValueError('the n=5 ceiling solver is what is audited; use '
                             'scaling_analysis for larger-code ceilings')
        ceil = abl.opt_unitary_ceiling(channel, p)
        Ga = np.asarray(ceil['G_opt'], dtype=complex)
        assert Ga.shape == (nsyn, 2, 2), Ga.shape
        # BASIS CONVERSION, and the one place this module can silently go wrong.
        # `opt_unitary_ceiling` optimises G_s = V^dag R_s W_s against
        # vscr_paper_abl's own syndrome basis W_BASIS, while the branch operators
        # A_k used here come from vscr_general's code.W.  Both are orthonormal
        # bases of im P_s, so W_basis = code.W @ T_s for a 2x2 unitary T_s.  The
        # physical round map is basis-independent (G^a B^a G^a^dag == G^g B^g
        # G^g^dag when A^a = T^dag A^g and G^a = G^g T), but plugging the ceiling's
        # G^a into the gen-basis A^g is NOT that map -- it is a different, worse
        # recovery, and it made the ceiling look 0.15 BELOW the decoder.  The
        # correct transport is G^g = G^a T_s^dag; the decoder pins it down, since
        # G^a_dec = T_s must convert to exactly the identity.
        T = np.stack([code.W[s].conj().T @ abl.W_BASIS[s] for s in range(nsyn)])
        G = np.einsum('sij,slj->sil', Ga, T.conj())      # G^g = G^a T^dag
        Tdev = max(float(np.abs(T[s].conj().T @ T[s] - np.eye(2)).max())
                   for s in range(nsyn))
        dev = max(float(np.abs(G[s].conj().T @ G[s] - np.eye(2)).max())
                  for s in range(nsyn))
        return G, None, {'kind': kind, 'F_unit': float(ceil['F_unit']),
                         'F_dec': float(ceil['F_dec']), 'unitarity_dev': dev,
                         'basis_transport_dev': Tdev,
                         'note': ('2x2 block table only, no full-space '
                                  'realisation, so readout error is refused '
                                  'rather than faked; G is transported from '
                                  'vscr_paper_abl\'s syndrome basis into '
                                  'vscr_general\'s')}
    else:
        raise ValueError('unknown recovery %r' % (kind,))

    # G_s = V^dagger R_s W_s, built from StabCode's own V and W so that any
    # disagreement with vscr_paper_abl's n=5 bindings is caught by --selftest
    # rather than silently moving the anchor.
    G = np.matmul(code.V.conj().T[None, :, :], R)      # (nsyn, 2, dim)
    G = np.matmul(G, code.W)                           # (nsyn, 2, 2)
    dev = max(float(np.abs(G[s].conj().T @ G[s] - np.eye(2)).max())
              for s in range(nsyn))
    return G, R, {'kind': kind, 'unitarity_dev': dev}


def cross_blocks(code, R_mats):
    """G~[s~, s] = V^dagger R_s~ W_s, the full cross-branch block table.

    This is what readout error actually acts on: with a misread syndrome the
    recovery applied is R_s~ while the state was projected onto branch s.  For a
    PAULI recovery every off-diagonal block must vanish identically, because
    C_s~ C_s carries syndrome s~ XOR s != 0 and so maps the code space orthogonal
    to itself -- the paper's exact readout suppression law.  A learned non-Pauli
    R_s has no such protection, and this is where the difference shows up.
    """
    if R_mats is None:
        return None
    return code.cross_G(R_mats)


def no_recovery_superop(code, channel, p):
    """The 'Raw' baseline: sigma -> V^dag N(V sigma V^dag) V.  No measurement, no
    correction.

    Computed by restricting the FULL-space channel to the code space on the four
    matrix units, which is exact by construction for every channel and every n.
    It is deliberately NOT built as a product of per-stage restricted maps: for
    'mixed' (depolarizing then damping) the intermediate state has already left
    the code space, and restricting before the second stage discards exactly the
    part that stage can bring back -- the composition of restrictions is not the
    restriction of the composition.  `_no_recovery_superop_analytic` is the
    single-stage closed form, kept only so `--selftest` can cross-check this
    against an independent expression where that expression is valid.

    The map is trace-DECREASING: only the syndrome-0 part of the noise keeps the
    state inside the code space, and the rest is lost.  That is the correct
    physics of an uncorrected memory, not a defect.
    """
    V = np.asarray(code.V, dtype=complex)
    Vd = V.conj().T
    S = np.zeros((4, 4), dtype=complex)
    for j in range(4):
        E = np.zeros((2, 2), dtype=complex)
        E.flat[j] = 1.0                       # row-major vec(E) == e_j
        out = apply_channel_full(torch.tensor(V @ E @ Vd), code, channel, p)
        S[:, j] = _vec_rowmajor(Vd @ out.detach().numpy() @ V)
    return S


def _no_recovery_superop_analytic(code, channel, p):
    """Closed form of the 'Raw' map for a SINGLE-STAGE channel.

    Kept private and used only by `--selftest`: for depolarizing it sums
    V^dag E V over the centralizer (every other Pauli maps the code space
    orthogonal to itself and contributes exactly zero), for amplitude damping and
    coherent it contracts the Kraus operators with V directly.  It is invalid for
    'mixed', which is why the public function above does not use it.
    """
    def _sup(Ahat):
        S = np.zeros((4, 4), dtype=complex)
        for A in Ahat:
            A = np.asarray(A, dtype=complex)
            S += np.kron(A, A.conj())
        return S

    Vd = code.V.conj().T
    if channel == 'depolarizing':
        _idx, amp = g.depolarizing_paulis(code, p)
        return _sup([float(amp[r]) * code.lam(code.pauli[r])
                     for r in code.cz if amp[r] != 0.0])
    if channel == 'amplitude_damping':
        KV = g.amp_damping_kraus_V(code, p)
        return _sup([Vd @ KV[k] for k in range(KV.shape[0])])
    if channel == 'coherent':
        U = g.coherent_unitary(code, p)
        return _sup([Vd @ U @ code.V])
    raise ValueError('analytic Raw map is single-stage only, got %r' % (channel,))


def build_round(code, channel, p, kind, eta=0.0):
    """Assemble the superoperator of ONE storage round.

    Returns (S, meta).  `meta` carries the branch probabilities, the unitarity
    deviation, the exact single-round fidelity (so the R=1 anchor is computed from
    the very object being iterated, not from a parallel path), and -- when eta > 0
    -- the measured norm of the off-diagonal cross-blocks, which is the quantity
    that decides whether a misread syndrome merely suppresses the fidelity or
    actively rotates the logical state.
    """
    Astack = branch_kraus_stacked(code, channel, p)
    ps = branch_p_weights(Astack)
    meta = {'channel': channel, 'p': float(p), 'recovery': kind, 'eta': float(eta),
            'branch_prob_sum': float(ps.sum()), 'n_branches': int(len(ps))}
    if kind == 'none':
        S = no_recovery_superop(code, channel, p)
        meta.update({'kind': 'none', 'unitarity_dev': 0.0,
                     'cross_block_norm': 0.0})
        meta['F_R1'] = fidelity_after_rounds(S, 1)
        return S, meta

    G, R, gmeta = recovery_blocks(code, kind, channel=channel, p=p)
    meta.update(gmeta)
    if eta > 0.0:
        Gx = cross_blocks(code, R)
        if Gx is None:
            raise ValueError(
                'recovery %r has no full-space realisation, so readout error '
                'cannot be simulated for it; refusing to invent one' % kind)
        C = readout_confusion(code.nsyn, code.m, eta)
        S = round_superop(Astack, Gx, readout=C)
        # The off-diagonal part of the cross-block table is exactly what a Pauli
        # recovery does not have.  Reporting its norm makes the difference between
        # the decoder and a learned recovery a measured number.
        off = np.asarray(Gx, dtype=complex).copy()
        for s in range(code.nsyn):
            off[s, s] = 0.0
        meta['cross_block_norm'] = float(np.abs(off).max())
        meta['diag_block_dev'] = float(max(
            np.abs(Gx[s, s] - G[s]).max() for s in range(code.nsyn)))
    else:
        S = round_superop(Astack, G)
        meta['cross_block_norm'] = 0.0
    meta['F_R1'] = fidelity_after_rounds(S, 1)
    return S, meta


# ---------------------------------------------------------------------
# FULL-space round map: the same physics without projecting back with V^dagger
# ---------------------------------------------------------------------
def _apply_one_qubit_kraus(rho_t, ops, q, n):
    """rho -> sum_k K_q rho K_q^dagger for 2x2 operators `ops` acting on site q.

    `rho_t` is the density matrix as a tensor of shape (2,)*n + (2,)*n, indices
    0..n-1 the ket side and n..2n-1 the bra side.  Contracting a 2x2 operator
    against one index costs 4 * dim^2 rather than the dim^3 of forming and
    multiplying the embedded dim x dim operator, which is what keeps the
    [[9,1,3]] full-space simulation (dim=512) inside a fraction of a second.
    """
    out = None
    for K in ops:
        K = torch.as_tensor(K).to(rho_t.device)
        # ket side: K acts on index q
        t = torch.tensordot(K, rho_t, dims=([1], [q])).movedim(0, q)
        # bra side: conj(K) acts on index n+q, giving rho -> K rho K^dagger
        t = torch.tensordot(K.conj(), t, dims=([1], [n + q])).movedim(0, n + q)
        out = t if out is None else out + t
    return out


def channel_kraus_per_qubit(channel, p):
    """Per-qubit 2x2 Kraus sets, matching ssvr_qec._channel_stages exactly.

    Returns a list of STAGES, where each stage is a list of length ONE holding
    the 2x2 operator set applied to EVERY qubit (all four channels here are
    identical across qubits, so a per-qubit list would be n copies of the same
    thing).  Stages compose -- 'mixed' is depolarizing(p) THEN amplitude damping
    (p/2) -- and within a stage the qubits are disjoint so their order is
    irrelevant.  That is the same structure the audited n=5 channel uses, which is
    what `--selftest` checks `apply_channel_full` against.
    """
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    stages = []
    if channel in ('depolarizing', 'mixed'):
        stages.append([[math.sqrt(1 - p) * I2, math.sqrt(p / 3.0) * X,
                        math.sqrt(p / 3.0) * Y, math.sqrt(p / 3.0) * Z]])
    if channel in ('amplitude_damping', 'mixed'):
        gam = p if channel == 'amplitude_damping' else 0.5 * p
        stages.append([[np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gam)]],
                                 dtype=complex),
                        np.array([[0.0, math.sqrt(gam)], [0.0, 0.0]],
                                 dtype=complex)]])
    if channel == 'coherent':
        stages.append([[np.array(
            [[math.cos(p / 2), -1j * math.sin(p / 2)],
             [-1j * math.sin(p / 2), math.cos(p / 2)]], dtype=complex)]])
    if not stages:
        raise ValueError(channel)
    # Every stage must be a one-element list of operator lists, or the caller
    # would silently iterate over the ROWS of a 2x2 matrix.
    for st in stages:
        assert len(st) == 1 and all(np.asarray(o).shape == (2, 2) for o in st[0]),\
            'malformed stage %r' % (st,)
    return stages


def apply_channel_full(rho, code, channel, p):
    """N(rho) on the FULL dim x dim space, for any operator rho."""
    n, dim = code.n, code.dim
    rt = rho.reshape(*([2] * (2 * n)))
    for stage in channel_kraus_per_qubit(channel, p):
        for q in range(n):
            rt = _apply_one_qubit_kraus(rt, stage[0], q, n)
    return rt.reshape(dim, dim)


def code_tensors(code, device):
    """W and V of `code` as torch tensors on `device`, cached per (code, device).

    Cached on the code object rather than in a module dict so the cache cannot
    outlive a code built with a different thread count.
    """
    cache = getattr(code, '_torch_cache', None)
    if cache is None:
        cache = code._torch_cache = {}
    key = str(device)
    if key not in cache:
        W = torch.tensor(np.asarray(code.W, dtype=complex), device=device)
        V = torch.tensor(np.asarray(code.V, dtype=complex), device=device)
        cache[key] = (W, W.conj().transpose(1, 2).contiguous(), V)
    return cache[key]


def apply_round_batch(rhos, code, channel, p, RW, W, Wdag):
    """One storage round applied to a BATCH of density matrices.

    `rhos` is (nq, dim, dim).  Batching over the 21 Haar probes is what makes the
    full-space path worth running at all: the branch einsum then contracts
    nq x nsyn blocks in one call instead of 21 separate ones, which on the GPU is
    the difference between launch-bound and arithmetic-bound.
    """
    out = apply_channel_full_batch(rhos, code, channel, p)
    # Wdag is (nsyn, 2, dim), so its dim index is the THIRD subscript: Z_s is
    # W_s^dagger rho W_s.  Getting this wrong is dimensionally legal at 2x2 and
    # would silently scramble the branches, hence the shape assert below.
    Z = torch.einsum('sai,qij,sjb->qsab', Wdag, out, W)      # (nq, nsyn, 2, 2)
    assert Z.shape == (rhos.shape[0], W.shape[0], 2, 2), Z.shape
    return torch.einsum('sia,qsab,sjb->qij', RW, Z, RW.conj())


def apply_channel_full_batch(rhos, code, channel, p):
    """N(.) on a batch of full density matrices, per-qubit Kraus by tensor contraction."""
    n, dim = code.n, code.dim
    q = rhos.shape[0]
    rt = rhos.reshape(*([q] + [2] * (2 * n)))
    for stage in channel_kraus_per_qubit(channel, p):
        for site in range(n):
            rt = _apply_one_qubit_kraus_batch(rt, stage[0], site, n)
    return rt.reshape(q, dim, dim)


def _apply_one_qubit_kraus_batch(rho_t, ops, site, n):
    """sum_k K rho K^dagger on a (q, 2,...,2) tensor; site index is 1+site."""
    ket, bra = 1 + site, 1 + n + site
    out = None
    for K in ops:
        # Move to rho's device: `torch.as_tensor` on a numpy array yields a CPU
        # tensor, which is correct for the CPU path and a hard error on CUDA.
        K = torch.as_tensor(K).to(rho_t.device)
        t = torch.tensordot(K, rho_t, dims=([1], [ket])).movedim(0, ket)
        t = torch.tensordot(K.conj(), t, dims=([1], [bra])).movedim(0, bra)
        out = t if out is None else out + t
    return out


def _measure_batch(rhos, Vpsi, P_code, w):
    """Weighted Haar average of <psi|rho|psi>, plus trace and code-space fraction.

    `leakage` is 1 - Tr[P rho]/Tr[rho], the fraction of the PHYSICAL state sitting
    outside the code space.  It is a property of rho alone, not of the probe, and
    it is the quantity that explains any disagreement between this full-space
    curve and the reduced 2x2 one: the reduced map computes (V^dag E V)^R while
    this computes V^dag E^R V, and the two differ exactly by paths that left the
    code space and came back.
    """
    f = torch.einsum('qi,qij,qj->q', Vpsi.conj(), rhos, Vpsi).real
    tr = torch.einsum('qii->q', rhos).real
    inc = torch.einsum('ij,qji->q', P_code, rhos).real
    F = float((w * f).sum())
    TR = float((w * tr).sum())
    INC = float((w * inc).sum())
    return {'F': F, 'trace': TR, 'in_code': INC,
            'leakage': (1.0 - INC / TR) if TR > 0 else 0.0}


def storage_full(code, channel, p, R_mats, rounds, device='cpu'):
    """F(R) from the FULL dim x dim density matrix, leakage included.

    Returns (rounds_sorted, curve, leakage, trace).  `device` is passed straight
    to torch, which is the point of this path: at [[9,1,3]] the branch einsum
    contracts 256 blocks of (512,2) columns, i.e. real arithmetic rather than the
    launch overhead that made the GPU lose on the T3 gradient stage.
    """
    dev = torch.device(device)
    W, Wdag, V = code_tensors(code, dev)
    R_t = torch.tensor(np.asarray(R_mats, dtype=complex), device=dev)
    RW = torch.matmul(R_t, W)                                # (nsyn, dim, 2)
    P_code = torch.matmul(V, V.conj().T)                     # (dim, dim)
    st, ww = haar_probe_set()
    Vpsi = torch.stack([V @ torch.tensor(np.asarray(s, dtype=complex),
                                         device=dev) for s in st])
    w = torch.tensor(ww, dtype=torch.float64, device=dev)
    rhos = torch.einsum('qi,qj->qij', Vpsi, Vpsi.conj()).contiguous()

    want = sorted(set(int(r) for r in rounds) | {0})
    rmax = want[-1]
    meas = {}
    for R in range(0, rmax + 1):
        if R in want:
            meas[R] = _measure_batch(rhos, Vpsi, P_code, w)
        if R < rmax:
            rhos = apply_round_batch(rhos, code, channel, p, RW, W, Wdag)
    return (want, [meas[R]['F'] for R in want],
            [meas[R]['leakage'] for R in want],
            [meas[R]['trace'] for R in want])


# ---------------------------------------------------------------------
# Self-test: every claim this module makes, checked against audited code
# ---------------------------------------------------------------------
def selftest(log=print, heavy=True):
    """Assert the multi-round machinery reduces to the audited single round.

    Ordered from cheapest to most expensive.  The first four groups are the ones
    that matter: a multi-round number that does not collapse onto
    `abl.exact_F` / `opt_unitary_ceiling` at R=1 is measuring something other
    than the paper's quantity, and everything downstream would be unverifiable.
    """
    fails = []
    n = [0]

    def ck(name, cond, detail=''):
        n[0] += 1
        if cond:
            log('  [ok]   %s' % name)
        else:
            log('  [FAIL] %s %s' % (name, detail))
            fails.append(name)

    code5 = get_code('5,1,3')
    rng = np.random.RandomState(0)

    # --- 1. the channel: bit-exact against the audited n=5 implementation ---
    worst = 0.0
    for noise in CHANNELS:
        for p in (0.05, 0.10, 0.20):
            psi = m.encode(m.random_logical_state(1, rng))
            rho = torch.outer(psi, psi.conj())
            ref = m.apply_channel(rho, noise, p).numpy()
            got = apply_channel_full(rho, code5, noise, p).detach().numpy()
            worst = max(worst, float(np.abs(ref - got).max()))
    ck('apply_channel_full is BIT-EXACT vs ssvr_qec.apply_channel '
       '(4 channels x 3 strengths)', worst == 0.0, worst)
    for noise in CHANNELS:
        st = channel_kraus_per_qubit(noise, 0.1)
        ck('channel_kraus_per_qubit(%s) is well-formed stages' % noise,
           all(len(s) == 1 and all(np.asarray(o).shape == (2, 2) for o in s[0])
               for s in st))
    ck("channel stages: 'mixed' composes depolarizing THEN damping (2 stages)",
       len(channel_kraus_per_qubit('mixed', 0.1)) == 2)
    ck("channel stages: 'coherent' has ONE unitary Kraus operator",
       len(channel_kraus_per_qubit('coherent', 0.1)[0][0]) == 1)

    # --- 2. R=1 anchor: the reduced map IS the audited single-round number ---
    for noise in CHANNELS:
        S, meta = build_round(code5, noise, 0.10, 'decoder')
        ref = abl.opt_unitary_ceiling(noise, 0.10)['F_dec']
        ck('R=1 decoder anchor on %s == opt_unitary_ceiling[F_dec]' % noise,
           abs(meta['F_R1'] - ref) < 1e-12, (meta['F_R1'], ref))
        ck('R=1 branch probabilities on %s sum to 1' % noise,
           abs(meta['branch_prob_sum'] - 1.0) < 1e-12, meta['branch_prob_sum'])
        phi = _load_phi('warm', noise)
        refw = float(abl.exact_F(phi, noise, 0.10)[0])
        Sw, metaw = build_round(code5, noise, 0.10, 'warm')
        ck('R=1 warm-VSCR anchor on %s == abl.exact_F' % noise,
           abs(metaw['F_R1'] - refw) < 1e-12, (metaw['F_R1'], refw))

    # --- 2b. the 'Raw' baseline against a direct full-space restriction ---
    # `no_recovery_superop` builds V^dag K_k V from the centralizer for Pauli
    # channels, which is a different code path from the branch decomposition.  It
    # must equal the brute-force restriction of the audited full-space channel to
    # the code space, or the 'none' curves would be measuring a different channel
    # from the recovered ones.  Only centralizer Paulis contribute: any other
    # Pauli maps the code space orthogonal to itself, so V^dag E V = 0 and the
    # restricted map is legitimately trace-DECREASING (uncorrected errors leave the
    # code space).  `code.lam` asserts unitarity and would trip on those, which is
    # the bug this check now guards.
    for noise in CHANNELS:
        Sn = no_recovery_superop(code5, noise, 0.10)
        for trial in range(3):
            # `random_logical_state` returns a torch tensor; take plain numpy so
            # that `.conj()` below is numpy's and not a lazy conjugate bit.
            psi = np.asarray(m.random_logical_state(
                1, np.random.RandomState(trial)).numpy(), dtype=complex)
            sig = np.outer(psi, psi.conj())
            Vsig = torch.tensor(code5.V @ sig @ code5.V.conj().T)
            out = m.apply_channel(Vsig, noise, 0.10).numpy()
            ref = code5.V.conj().T @ out @ code5.V
            got = (Sn @ _vec_rowmajor(sig)).reshape(2, 2)
            ck("'none' == V^dag N(V sigma V^dag) V on %s (trial %d)"
               % (noise, trial),
               float(np.abs(got - ref).max()) < 1e-14,
               float(np.abs(got - ref).max()))
        # Tr[X] for a row-major vec is the pairing with [1,0,0,1].  Applying the
        # 'Raw' map to I/2 (trace 1) must return trace strictly BELOW 1: the
        # syndrome-0 Paulis are only a (1-p)^n-ish fraction of the mixture, and the
        # rest has left the code space.
        half = np.array([0.5, 0.0, 0.0, 0.5], dtype=complex)      # vec(I/2)
        tr_out = float(np.real(np.array([1.0, 0.0, 0.0, 1.0]) @ (Sn @ half)))
        ck("'none' on %s is trace-DECREASING (uncorrected errors leave the code "
           'space)' % noise, tr_out < 1.0 - 1e-9, tr_out)
        # Two independent expressions for the same map.  For a single-stage
        # channel the closed form (centralizer sum / Kraus contraction) must equal
        # the full-space restriction.  For 'mixed' the closed form is REFUSED,
        # because the composition of code-space restrictions is not the
        # restriction of the composition -- the trap this pair of checks pins down.
        if noise != 'mixed':
            Sa = _no_recovery_superop_analytic(code5, noise, 0.10)
            ck("'none' full-space restriction == closed form on %s" % noise,
               float(np.abs(Sn - Sa).max()) < 1e-14,
               float(np.abs(Sn - Sa).max()))
        else:
            try:
                _no_recovery_superop_analytic(code5, 'mixed', 0.10)
                ck("'mixed' refuses the single-stage closed form", False,
                   'it returned instead of raising')
            except ValueError:
                ck("'mixed' refuses the single-stage closed form (restrictions "
                   'do not compose)', True)

    # --- 3. trivial round count and trace preservation ---
    S, _ = build_round(code5, 'amplitude_damping', 0.10, 'decoder')
    # Not `== 1.0`: F(0) is the weighted sum of 21 quadrature norms, and the
    # weights sum to 1 only to round-off, so F(0) = 1 - 2e-16 is the exact answer.
    ck('F(R=0) == 1 to round-off (nothing has happened yet)',
       abs(fidelity_after_rounds(S, 0) - 1.0) < 1e-14,
       fidelity_after_rounds(S, 0))
    ck('eigendecomposed curve == repeated matrix power',
       max(abs(a - b) for a, b in zip(
           fidelity_curve(S, (1, 2, 5, 10, 20)),
           [fidelity_after_rounds(S, r) for r in (1, 2, 5, 10, 20)])) < 1e-14)
    _, _, _, tr = storage_full(code5, 'amplitude_damping', 0.10,
                               np.asarray(code5.C_mat, dtype=complex),
                               (1, 2, 5, 10))
    ck('full-space round map is trace-preserving to 1e-12 over 10 rounds',
       max(abs(t - 1.0) for t in tr) < 1e-12, tr)
    # `storage_lifetime` is the number to quote for a memory, so it has to be
    # self-consistent with the curve it is read off rather than merely plausible.
    life = storage_lifetime(S)
    F_inf, R_h = life['F_inf'], life['R_half']
    ck('storage_lifetime: F_inf lies strictly below F(1) for a noisy round map',
       F_inf < fidelity_after_rounds(S, 1) - 1e-9, (F_inf,))
    ck('storage_lifetime: F(R) stays above F_inf at every finite R',
       all(fidelity_after_rounds(S, r) > F_inf - 1e-12
           for r in (1, 2, 5, 12, 40, 1000)))
    tgt = 0.5 * (1.0 + F_inf)
    ck('storage_lifetime: R_half brackets the halfway point exactly',
       fidelity_after_rounds(S, R_h) <= tgt
       < fidelity_after_rounds(S, max(1, R_h - 1)),
       (R_h, fidelity_after_rounds(S, R_h), tgt))
    # Amplitude damping is not unital, so its fixed point is biased toward |0> and
    # F_inf need NOT be 1/2 -- asserting 1/2 here would be asserting a property
    # the channel does not have.  Depolarizing IS unital, so for it 1/2 is exact.
    Sd, _ = build_round(code5, 'depolarizing', 0.10, 'decoder')
    ck("storage_lifetime: a unital channel's fixed point gives F_inf = 1/2",
       abs(storage_lifetime(Sd)['F_inf'] - 0.5) < 1e-9,
       storage_lifetime(Sd)['F_inf'])
    # The fixed point must satisfy its own defining equation and be a physical
    # state, otherwise F_inf and R_half are read off nothing at all.
    sig_inf, ev1 = fixed_point(S)
    got = (S @ _vec_rowmajor(sig_inf)).reshape(2, 2)
    ck('fixed_point: Lam(sigma_inf) == sigma_inf (the defining equation)',
       float(np.abs(got - sig_inf).max()) < 1e-12,
       float(np.abs(got - sig_inf).max()))
    ck('fixed_point: eigenvalue is 1 to round-off', abs(ev1 - 1.0) < 1e-12, ev1)
    ck('fixed_point: trace 1 and Hermitian',
       abs(float(np.real(np.trace(sig_inf))) - 1.0) < 1e-12
       and float(np.abs(sig_inf - sig_inf.conj().T).max()) < 1e-14)
    ck('fixed_point: positive semidefinite (a physical state)',
       float(np.linalg.eigvalsh(sig_inf).min()) > -1e-12,
       float(np.linalg.eigvalsh(sig_inf).min()))
    sig_d, _ = fixed_point(Sd)
    ck("fixed_point: a unital channel's fixed point is exactly I/2",
       float(np.abs(sig_d - 0.5 * np.eye(2)).max()) < 1e-12
       and abs(storage_lifetime(Sd)['fixed_point_purity'] - 0.5) < 1e-12,
       sig_d)

    # --- 4. _branch_superop == the explicit kron sum it replaces ---
    A = (rng.randn(7, 2, 2) + 1j * rng.randn(7, 2, 2))
    ref = np.zeros((4, 4), dtype=complex)
    for k in range(7):
        ref += np.kron(A[k], A[k].conj())
    ck('_branch_superop == explicit sum of kron(A_k, conj(A_k))',
       float(np.abs(_branch_superop(A) - ref).max()) < 1e-12)
    ck('_branch_superop of an empty stack is the zero map',
       float(np.abs(_branch_superop(np.zeros((0, 2, 2)))).max()) == 0.0)

    _selftest_readout(ck, code5)
    _selftest_ceiling_basis(ck, code5)
    _selftest_full_vs_reduced(ck, code5, heavy=heavy)
    log('[selftest] %d checks, %d failures' % (n[0], len(fails)))
    return fails


def _selftest_ceiling_basis(ck, code5):
    """The basis transport for the ceiling's G, pinned down by the decoder.

    `abl.opt_unitary_ceiling` optimises G_s in vscr_paper_abl's syndrome basis
    (W_BASIS) while this module's branch operators A_k come from vscr_general's
    (code.W).  The transport G^g = G^a T^dag with T_s = code.W_s^dag W_BASIS_s is
    what makes them compatible, and it is checkable WITHOUT reference to the
    ceiling: the decoder's block in the abl basis IS T_s, so transporting it must
    return this module's decoder block.  Skipping the transport made the ceiling
    look 0.15 BELOW the decoder, which is impossible by construction -- so this
    check is what turns that impossibility into an assertion.
    """
    nsyn = code5.nsyn
    T = np.stack([code5.W[s].conj().T @ abl.W_BASIS[s] for s in range(nsyn)])
    ck('the syndrome-basis transport T_s is unitary',
       max(float(np.abs(T[s].conj().T @ T[s] - np.eye(2)).max())
           for s in range(nsyn)) < 1e-12)
    Gdec_gen, _, _ = recovery_blocks(code5, 'decoder',
                                     channel='amplitude_damping', p=0.10)
    Gdec_abl = np.stack([abl.V_ISO.conj().T @ m.C_SYNDS[s].numpy()
                         @ abl.W_BASIS[s] for s in range(nsyn)])
    Gt = np.einsum('sij,slj->sil', Gdec_abl, T.conj())
    ck('transporting the DECODER from the abl basis reproduces this module\'s '
       'decoder blocks', float(np.abs(Gt - Gdec_gen).max()) < 1e-12,
       float(np.abs(Gt - Gdec_gen).max()))
    ck('this module\'s decoder blocks are a phase times the identity '
       '(W_s = C_s V, so V^dag C_s W_s = I)',
       max(float(np.abs(Gdec_gen[s] - Gdec_gen[s][0, 0] * np.eye(2)).max())
           for s in range(nsyn)) < 1e-12)
    for noise in CHANNELS:
        for p in (0.05, 0.10):
            ceil = abl.opt_unitary_ceiling(noise, p)
            _S, meta = build_round(code5, noise, p, 'ceiling')
            _Sd, metad = build_round(code5, noise, p, 'decoder')
            ck('ceiling R=1 == opt_unitary_ceiling[F_unit] on %s p=%.2f'
               % (noise, p),
               abs(meta['F_R1'] - ceil['F_unit']) < 1e-9,
               (meta['F_R1'], ceil['F_unit']))
            ck('ceiling >= decoder at R=1 on %s p=%.2f (it is a ceiling)'
               % (noise, p),
               meta['F_R1'] >= metad['F_R1'] - 1e-12,
               (meta['F_R1'], metad['F_R1']))
            ck('the transported ceiling G stays unitary on %s p=%.2f'
               % (noise, p), meta['unitarity_dev'] < 1e-12,
               meta['unitarity_dev'])


def _selftest_readout(ck, code5):
    """The readout law, single round AND multi-round, plus why it is exact.

    A Pauli recovery has identically vanishing cross-branch blocks, so with
    readout error eta the round map is exactly (1-eta)^m times the ideal one and
    the R-round law F(eta,R) = (1-eta)^(mR) F(0,R) follows by iteration.  That is
    a PREDICTION, not a fit, and it is checked to round-off here.  A learned
    non-Pauli recovery has no such vanishing, so the same law must FAIL for it --
    asserting only the decoder case would hide the difference that matters.
    """
    m_bits = code5.m
    G, R, _ = recovery_blocks(code5, 'decoder', channel='amplitude_damping',
                              p=0.10)
    Gx = cross_blocks(code5, R)
    off = np.asarray(Gx, dtype=complex).copy()
    for s in range(code5.nsyn):
        off[s, s] = 0.0
    ck('decoder cross-branch blocks vanish identically (the readout law\'s '
       'premise)', float(np.abs(off).max()) < 1e-14, float(np.abs(off).max()))
    ck('decoder diagonal cross-blocks equal the ideal-readout blocks',
       float(max(np.abs(Gx[s, s] - G[s]).max()
                 for s in range(code5.nsyn))) == 0.0)

    Gw, Rw, _ = recovery_blocks(code5, 'warm', channel='amplitude_damping',
                                p=0.10)
    Gxw = cross_blocks(code5, Rw)
    offw = np.asarray(Gxw, dtype=complex).copy()
    for s in range(code5.nsyn):
        offw[s, s] = 0.0
    ck('the learned non-Pauli recovery does NOT have vanishing cross-blocks '
       '(so it is not protected by the law)',
       float(np.abs(offw).max()) > 1e-6, float(np.abs(offw).max()))

    S0, m0 = build_round(code5, 'amplitude_damping', 0.10, 'decoder')
    for eta in (0.005, 0.01, 0.02, 0.05):
        Se, me = build_round(code5, 'amplitude_damping', 0.10, 'decoder',
                             eta=eta)
        pred = (1.0 - eta) ** m_bits * m0['F_R1']
        ck('readout law at R=1, eta=%.3f: F == (1-eta)^m F(0)' % eta,
           abs(me['F_R1'] - pred) < 1e-14 * max(1.0, pred),
           (me['F_R1'], pred))
        # the multi-round extension, which the paper does not state
        rounds = (1, 2, 5, 10)
        got = fidelity_curve(Se, rounds)
        want = [(1.0 - eta) ** (m_bits * r) * f
                for r, f in zip(rounds, fidelity_curve(S0, rounds))]
        ck('readout law at R=1..10, eta=%.3f: F(eta,R) == (1-eta)^(mR) F(0,R)'
           % eta,
           max(abs(a - b) for a, b in zip(got, want)) < 1e-13,
           (got, want))
    # eta=0 through the readout path must equal the ideal path exactly.
    C0 = readout_confusion(code5.nsyn, m_bits, 0.0)
    ck('readout_confusion(eta=0) is the identity',
       float(np.abs(C0 - np.eye(code5.nsyn)).max()) == 0.0)
    ck('readout_confusion columns are probability distributions',
       float(np.abs(C0.sum(0) - 1.0).max()) < 1e-14)
    S_id, _ = build_round(code5, 'amplitude_damping', 0.10, 'decoder')
    A = branch_kraus_stacked(code5, 'amplitude_damping', 0.10)
    S_x = round_superop(A, Gx, readout=C0)
    ck('the readout code path at eta=0 reproduces the ideal round map',
       float(np.abs(S_x - S_id).max()) < 1e-15, float(np.abs(S_x - S_id).max()))
    # The warm recovery must VIOLATE the multiplicative law, or the decoder check
    # above would be vacuous.  But the violation must also be shown to be
    # NEGLIGIBLE, which is the scientifically interesting part: the learned table
    # has cross-branch blocks of order 1e-3, yet they change the fidelity at
    # eta=0.02 by only ~5e-11, because they enter weighted by the confusion
    # probability and largely cancel in the Haar average.  So a non-Pauli recovery
    # is not protected by the law in principle, and is protected to 8 decimal
    # places in practice -- both halves are asserted rather than assumed.
    Sw0, mw0 = build_round(code5, 'amplitude_damping', 0.10, 'warm')
    Swe, mwe = build_round(code5, 'amplitude_damping', 0.10, 'warm', eta=0.02)
    pred = (1.0 - 0.02) ** m_bits * mw0['F_R1']
    viol = abs(mwe['F_R1'] - pred)
    ck('the non-Pauli recovery genuinely violates the multiplicative law '
       '(so the decoder check is not vacuous)', viol > 1e-14, viol)
    ck('... but only at the 1e-11 level, i.e. negligibly against the (1-eta)^m '
       'suppression itself', viol < 1e-8 * pred, (viol, pred))
    ck('the cross-blocks that cause it are ~1e-3, so the tiny effect is a '
       'cancellation, not a missing term',
       1e-5 < mwe['cross_block_norm'] < 1e-2, mwe['cross_block_norm'])


def _selftest_full_vs_reduced(ck, code5, heavy=True):
    """The reduced 2x2 map against the FULL dim x dim density-matrix evolution.

    This is the check that licenses using the cheap map at all.  The reduced map
    computes (V^dag E V)^R while the full one computes V^dag E^R V; they coincide
    exactly when every R_s maps range(P_s) isometrically into the code space, and
    differ by the leakage otherwise.  For the Pauli decoder that leakage is
    round-off, so agreement must be at 1e-14.  For the learned table it is the
    measured unitarity deviation, so agreement is asserted at that scale instead
    of at round-off -- asserting 1e-14 there would be asserting something false.
    """
    rounds = (1, 2, 3, 5, 8, 12) if heavy else (1, 2, 5)
    for noise in CHANNELS:
        # 1e-12 for the decoder rather than round-off: 'mixed' composes two
        # stages into 8^5 = 32768 Kraus operators and iterating that 12 times
        # accumulates to ~2.5e-13.  Twelve correct decimal places is still an
        # exactness claim, not a tolerance that could hide a real discrepancy.
        for kind, tol in (('decoder', 1e-12), ('warm', 1e-6)):
            S, meta = build_round(code5, noise, 0.10, kind)
            G, R, gm = recovery_blocks(code5, kind, channel=noise, p=0.10)
            w, cf, leak, tr = storage_full(code5, noise, 0.10, R, rounds)
            red = fidelity_curve(S, rounds)
            dmax = max(abs(cf[w.index(r)] - red[i])
                       for i, r in enumerate(rounds))
            ck('full-space == reduced over R<=%d, %s/%s (tol %.0e)'
               % (rounds[-1], noise, kind, tol), dmax < tol, dmax)
            if kind == 'decoder':
                ck('decoder leakage is round-off at every R, %s' % noise,
                   max(abs(x) for x in leak) < 1e-13, max(abs(x) for x in leak))
            else:
                # Leakage must SATURATE, not compound: each round re-projects onto
                # the syndrome subspaces, so what the learned table leaks is set
                # by its own unitarity deviation and does not accumulate.  If it
                # grew linearly with R the reduced map would be unusable as a
                # memory model and this is where that would show up.
                l1 = abs(leak[w.index(1)])
                lmax = max(abs(x) for x in leak)
                ck('warm leakage saturates rather than compounding, %s '
                   '(R=%d max %.2e vs R=1 %.2e)' % (noise, rounds[-1], lmax, l1),
                   lmax < 5.0 * l1 + 1e-15, (lmax, l1))
                # Floor the bound at 1e-13: on depolarizing and mixed the refine
                # stage picks the decoder, so unitarity_dev is itself round-off
                # (1e-16) and the leakage is round-off too (1e-15) -- comparing
                # two round-off quantities without a floor tests nothing.  Where
                # the table really is non-Pauli (amplitude damping, coherent) the
                # floor is inactive and the bound is the real one.
                bound = max(gm['unitarity_dev'], 1e-13)
                ck('warm leakage is bounded by max(unitarity_dev, round-off), '
                   '%s' % noise, lmax < bound, (lmax, bound))
            ck('full-space trace stays 1 over R<=%d, %s/%s'
               % (rounds[-1], noise, kind),
               max(abs(t - 1.0) for t in tr) < 1e-12, tr)
    if heavy and torch.cuda.is_available():
        G, R, _ = recovery_blocks(code5, 'decoder', channel='amplitude_damping',
                                  p=0.10)
        a = storage_full(code5, 'amplitude_damping', 0.10, R, (1, 5, 12),
                         device='cpu')[1]
        b = storage_full(code5, 'amplitude_damping', 0.10, R, (1, 5, 12),
                         device='cuda')[1]
        ck('full-space curve is device-independent (cpu vs cuda)',
           max(abs(x - y) for x, y in zip(a, b)) < 1e-12, (a, b))
    elif heavy:
        # SAY SO rather than quietly running one fewer check.  run_selftests.py
        # sets CUDA_VISIBLE_DEVICES='' so the suite is CPU-only and deterministic,
        # which legitimately skips this; a reader comparing check counts would
        # otherwise see 125 vs 126 and have no way to tell a skip from a deletion.
        # Device independence stays locked regardless: storage_rounds.py
        # --selftest run directly performs it, and audit_numbers.py section 14b
        # asserts the production device_benchmark's CPU/CUDA curves agree to 1e-12.
        ck('CUDA unavailable (CUDA_VISIBLE_DEVICES=%r): device-independence check '
           'SKIPPED, not silently dropped -- covered by `storage_rounds.py '
           '--selftest` and by the audited device_benchmark'
           % os.environ.get('CUDA_VISIBLE_DEVICES'), True)


# ---------------------------------------------------------------------
# The T4 sweep
# ---------------------------------------------------------------------
P_GRID = (0.05, 0.10)
ROUNDS = (0, 1, 2, 3, 5, 8, 12, 16, 20, 30, 40)
# Only n=5 has trained tables and an audited ceiling solver; the larger codes get
# the decoder, which is constructible from StabCode alone.
RECOVERIES_BY_CODE = {
    '5,1,3': ('none', 'decoder', 'warm', 'ind', 'ceiling'),
    '7,1,3': ('none', 'decoder'),
    '9,1,3': ('none', 'decoder'),
}
# Readout error is only simulable where a full-space R_s exists (see
# `recovery_blocks`); the ceiling is a 2x2 block table and is refused.
ETA_GRID = (0.0, 0.005, 0.02)
ETA_OK = ('decoder', 'warm', 'ind')
# Below this, a single-round "advantage" is round-off rather than physics, and its
# ratio to the R-round advantage is 0/0.  Placed five orders of magnitude below the
# smallest genuine headroom in the paper (5.4e-6, coherent at p=0.06) and five
# above double-precision noise on a fidelity near 1, so it separates the two
# regimes with margin on both sides.
SURVIVAL_FLOOR = 1e-9


def sweep(log=print):
    """F(R) for every code x channel x strength x recovery x readout rate."""
    out = []
    for spec in CODE_ORDER:
        code = get_code(spec)
        for channel in CHANNELS_BY_CODE[spec]:
            for p in P_GRID:
                for kind in RECOVERIES_BY_CODE[spec]:
                    for eta in ETA_GRID:
                        if eta > 0.0 and kind not in ETA_OK:
                            continue
                        t0 = time.time()
                        S, meta = build_round(code, channel, p, kind, eta=eta)
                        curve = fidelity_curve(S, ROUNDS)
                        rec = {'code': spec, 'n': code.n, 'channel': channel,
                               'p': p, 'recovery': kind, 'eta': eta,
                               'rounds': list(ROUNDS),
                               'F': [float(x) for x in curve],
                               'wall_s': time.time() - t0}
                        rec.update(asymptotic_decay(S))
                        rec.update(storage_lifetime(S))
                        for k in ('F_R1', 'unitarity_dev', 'cross_block_norm',
                                  'branch_prob_sum', 'picked'):
                            if k in meta:
                                rec[k] = meta[k]
                        out.append(rec)
                        log('  [%s|%s|p=%.2f|%-8s|eta=%.3f] F(1)=%.9f '
                            'F(40)=%.9f lam=%.6f (%.1fs)'
                            % (spec, channel, p, kind, eta, curve[1], curve[-1],
                               rec['lam_logical'], rec['wall_s']))
    return out


def enrich_records(records, log=print):
    """Recompute the spectral and lifetime fields of stored records in place.

    `--reaggregate` reads an artifact whose curves are already correct but whose
    lifetime fields may predate a change in how they are defined or reported.
    Rebuilding the 4x4 round map is cheap (0.1 s at [[9,1,3]], and the ceiling
    solver dominates at ~1 s per point), so this is far better than re-running the
    full-space sweep and the device benchmark, which together cost minutes.  The
    stored F(R) curve is re-derived and asserted to match, which is what makes
    this an enrichment rather than a silent rewrite.
    """
    changed = 0
    for r in records:
        S, _meta = build_round(get_code(r['code']), r['channel'], r['p'],
                               r['recovery'], eta=r.get('eta', 0.0))
        curve = fidelity_curve(S, r['rounds'])
        dmax = max(abs(a - b) for a, b in zip(curve, r['F']))
        assert dmax < 1e-12, ('stored curve does not reproduce: %s/%s/p=%s/%s '
                              'max|d|=%.2e' % (r['code'], r['channel'], r['p'],
                                               r['recovery'], dmax))
        stale = [k for k in ('rounds_to_e_fold',) if k in r]
        for k in stale:
            del r[k]
        r.update(asymptotic_decay(S))
        r.update(storage_lifetime(S))
        r['curve_reproduced_max_abs_diff'] = float(dmax)
        changed += 1
    log('  [enrich] recomputed lifetime fields for %d records; every stored '
        'curve re-derived and matched' % changed)
    return records


def summarise(records, log=print):
    """The headline comparison: does the single-round advantage SURVIVE rounds?

    For each (code, channel, p, eta) the advantage of a learned recovery over the
    decoder is tabulated at R=1 and at the longest R.  If the ratio stays near 1
    the advantage is a property of the round map and the memory inherits it; if it
    decays, the single-round headline overstates what a real memory would deliver.
    """
    idx = {(r['code'], r['channel'], r['p'], r['recovery'], r['eta']): r
           for r in records}
    rows = []
    for spec in CODE_ORDER:
        for channel in CHANNELS_BY_CODE[spec]:
            for p in P_GRID:
                for eta in ETA_GRID:
                    dec = idx.get((spec, channel, p, 'decoder', eta))
                    if dec is None:
                        continue
                    for kind in ('warm', 'ind', 'ceiling'):
                        alt = idx.get((spec, channel, p, kind, eta))
                        if alt is None:
                            continue
                        a1 = alt['F'][1] - dec['F'][1]
                        aR = alt['F'][-1] - dec['F'][-1]
                        # A single-round advantage at round-off level means the two
                        # recoveries are the SAME recovery here (on depolarizing and
                        # mixed the decoder already is the family optimum).  Their
                        # ratio is then 0/0 -- reporting "survival = 28x" for two
                        # numbers of size 1e-16 would be inventing a trend out of
                        # floating-point noise, so it is reported as undefined.
                        degenerate = abs(a1) <= SURVIVAL_FLOOR
                        rows.append({
                            'code': spec, 'channel': channel, 'p': p, 'eta': eta,
                            'recovery': kind,
                            'advantage_R1': a1, 'advantage_R%d' % ROUNDS[-1]: aR,
                            'advantage_degenerate': degenerate,
                            'survival_ratio': (None if degenerate else aR / a1),
                            'F_dec_R1': dec['F'][1], 'F_alt_R1': alt['F'][1],
                            'F_dec_Rlast': dec['F'][-1],
                            'F_alt_Rlast': alt['F'][-1],
                            'lam_dec': dec['lam_logical'],
                            'lam_alt': alt['lam_logical'],
                            'R_half_dec': dec.get('R_half'),
                            'R_half_alt': alt.get('R_half'),
                            'F_inf_dec': dec.get('F_inf'),
                            'F_inf_alt': alt.get('F_inf'),
                            # An infinite R_half means F never reached the
                            # halfway point within the search cap (10^6 rounds),
                            # which happens for the learned recovery on coherent
                            # noise.  A ratio with an infinite numerator is not a
                            # gain, so it is reported as a lower bound instead of
                            # as "inf x".
                            'R_half_gain': (
                                None if not dec.get('R_half')
                                or alt.get('R_half') is None
                                else alt['R_half'] / dec['R_half']),
                            'R_half_capped': bool(
                                alt.get('R_half') is None
                                or dec.get('R_half') is None)})
    log('\n%-9s %-18s %5s %5s %-8s %11s %11s %8s %9s'
        % ('code', 'channel', 'p', 'eta', 'recov.', 'adv@R=1', 'adv@R=%d'
           % ROUNDS[-1], 'survival', 'R_half x'))
    log('-' * 94)
    for r in rows:
        gain = r['R_half_gain']
        if gain is None and r.get('R_half_capped') and r['R_half_dec']:
            # The learned recovery did not reach halfway within the 10^6-round
            # search cap, so its gain over the decoder is a LOWER bound.
            gtxt = '>%.0f' % (1e6 / r['R_half_dec'])
        elif gain is None:
            gtxt = 'n/a'
        else:
            gtxt = '%.2f' % gain
        log('%-9s %-18s %5.2f %5.3f %-8s %11.3e %11.3e %8s %9s'
            % (r['code'], r['channel'], r['p'], r['eta'], r['recovery'],
               r['advantage_R1'], r['advantage_R%d' % ROUNDS[-1]],
               'n/a' if r['survival_ratio'] is None
               else '%.2f' % r['survival_ratio'], gtxt))
    log('-' * 94)
    log('survival = advantage(R=%d) / advantage(R=1).  Near 1 means the'
        % ROUNDS[-1])
    log('           single-round headroom is merely inherited by a long memory;')
    log('           ABOVE 1 means it compounds, which is the interesting case;')
    log('           below 1 means it washes out and the headline overstates.')
    log('R_half x = R_half(recovery) / R_half(decoder), where R_half is the round')
    log('           count at which F has fallen halfway from 1 to its fixed point')
    log('           F_inf.  This is the memory-time gain, and unlike a decay rate')
    log('           it is read straight off the curve, so it stays meaningful when')
    log('           the decay is not a single exponential.')
    log('n/a      = the single-round advantage is at round-off (|adv| <= %.0e),'
        % SURVIVAL_FLOOR)
    log('           i.e. the two recoveries ARE the same recovery on that channel')
    log('           (the decoder is already the family optimum there), so the ratio')
    log('           is 0/0 and would otherwise print a trend made of noise.')
    return rows


# ---------------------------------------------------------------------
# Validation against the audited artifacts
# ---------------------------------------------------------------------
# main.tex (Methods) documents a 48-sample Monte-Carlo SEM of 1.1e-4 at amplitude
# damping p=0.10, and the paper's *_p010 tables are n_test=200 sampled estimates.
# The exact curves computed here must land inside that estimator noise -- a much
# tighter tolerance would be testing the sampler, not this module.
MC_TOL = 5.0e-4


def validate(log=print):
    """Cross-check R=1 against the numbers the paper already reports.

    Two independent anchors.  The EXACT one (`abl.exact_F`,
    `opt_unitary_ceiling`) is checked to 1e-12 in `selftest`; what is checked here
    is the PUBLISHED one, `paper_numbers.json[*_p010]`, a 200-state Monte-Carlo
    estimate that agrees only to its own sampling error.  Passing both means the
    multi-round machinery is anchored to the paper's quantity at bit level and to
    the paper's printed table at estimator level.
    """
    with open(os.path.join(ROOT, 'paper_numbers.json')) as fh:
        pn = json.load(fh)
    key = {'depolarizing': 'dep_p010', 'amplitude_damping': 'ad_p010',
           'mixed': 'mx_p010', 'coherent': 'coh_p010'}
    code5 = get_code('5,1,3')
    fails, checks = [], 0
    log('\n=== validation: R=1 vs paper_numbers.json[*_p010] (n_test=200) ===')
    for noise, kk in key.items():
        if kk not in pn:
            fails.append('%s: no %s table in paper_numbers.json' % (noise, kk))
            continue
        tab = pn[kk]
        for kind, label in (('decoder', 'Perfect-code decoder'),
                            ('warm', 'VSCR warm (ours)'),
                            ('none', 'Raw')):
            if label not in tab:
                continue
            if kind == 'none':
                S = no_recovery_superop(code5, noise, 0.10)
                got = fidelity_after_rounds(S, 1)
            else:
                _S, meta = build_round(code5, noise, 0.10, kind)
                got = meta['F_R1']
            ref = float(tab[label])
            checks += 1
            ok = abs(got - ref) <= MC_TOL
            log('  %-18s %-8s exact R=1=%.12f  paper=%.12f  |d|=%.2e  %s'
                % (noise, kind, got, ref, abs(got - ref),
                   'OK' if ok else 'FAIL'))
            if not ok:
                fails.append('%s/%s: %.12f vs paper %.12f (|d|=%.2e > %.0e)'
                             % (noise, kind, got, ref, abs(got - ref), MC_TOL))
    log('  %d comparisons, %d failures (tolerance %.0e = the paper\'s own '
        'Monte-Carlo scale)' % (checks, len(fails), MC_TOL))
    return {'n_comparisons': checks, 'n_failures': len(fails),
            'failures': fails, 'tolerance': MC_TOL, 'passed': not fails}


def full_space_curves(wanted, device='cpu', log=print):
    """Full dim x dim runs for the listed (code, channel, p, recovery) tuples.

    This is the expensive path, and the one where the device choice was measured
    rather than assumed: at [[9,1,3]] it costs 148 s on a pinned CPU core against
    18 s on the RTX 5070 Ti (8.3x), while at [[5,1,3]] the GPU is 3.8x SLOWER
    because 32x32 work is launch-bound -- the mirror image of the T3 gradient
    stage result, and the same underlying reason.
    """
    out = []
    for spec, channel, p, kind in wanted:
        code = get_code(spec)
        _G, R, gm = recovery_blocks(code, kind, channel=channel, p=p)
        t0 = time.time()
        w, cf, leak, tr = storage_full(code, channel, p, R, ROUNDS,
                                       device=device)
        S, _meta = build_round(code, channel, p, kind)
        red = fidelity_curve(S, ROUNDS)
        dmax = max(abs(a - b) for a, b in zip(cf, red))
        rec = {'code': spec, 'channel': channel, 'p': p, 'recovery': kind,
               'device': device, 'rounds': list(w),
               'F_full': [float(x) for x in cf],
               'F_reduced': [float(x) for x in red],
               'max_abs_full_vs_reduced': float(dmax),
               'leakage': [float(x) for x in leak],
               'leakage_max': float(max(abs(x) for x in leak)),
               'trace_dev_max': float(max(abs(t - 1.0) for t in tr)),
               'unitarity_dev': gm['unitarity_dev'],
               'wall_s': time.time() - t0}
        out.append(rec)
        log('  [full|%s|%s|p=%.2f|%-8s|%s] %6.1fs  max|full-reduced|=%.2e  '
            'leakage_max=%.2e' % (spec, channel, p, kind, device, rec['wall_s'],
                                  dmax, rec['leakage_max']))
    return out


def device_benchmark(log=print):
    """Measure the full-space path on CPU and CUDA and record which wins where.

    Stored in the artifact so the device choice is a measurement a reader can
    check rather than an assumption.  CUDA is skipped cleanly when unavailable
    (this host's NVML is broken by a driver/library mismatch -- 595.84 kernel
    module against a 595.91.07 userspace -- but the CUDA runtime itself works, so
    `nvidia-smi` failing is not evidence that the GPU is unusable).
    """
    devs = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    if len(devs) == 1:
        log('  [bench] CUDA unavailable; benchmarking cpu only')
    res = {'torch': torch.__version__,
           'device_name': (torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else None),
           'nvml_note': ('nvidia-smi fails with a driver/library version '
                         'mismatch while torch.cuda works; NVML, not the CUDA '
                         'runtime, is what is broken'),
           'rounds': list(ROUNDS)}
    # BOTH ends of the crossover, not just the end that flatters the GPU.  At
    # n=5 (dim 32) the full-space path is launch-bound and the GPU LOSES; at n=9
    # (dim 512) the branch einsum is real arithmetic and it wins by ~8x.  Quoting
    # only one of the two would make the device policy look arbitrary.
    res['points'] = []
    for spec, channel, p, kind in (('5,1,3', 'amplitude_damping', 0.10, 'warm'),
                                   ('9,1,3', 'amplitude_damping', 0.05,
                                    'decoder')):
        code = get_code(spec)
        _G, R, _gm = recovery_blocks(code, kind, channel=channel, p=p)
        pt = {'code': spec, 'channel': channel, 'p': p, 'recovery': kind,
              'dim': code.dim}
        for dev in devs:
            t0 = time.time()
            cf = storage_full(code, channel, p, R, ROUNDS, device=dev)[1]
            pt[dev] = {'seconds': time.time() - t0, 'F': [float(x) for x in cf]}
            log('  [bench] [[%s]] %-5s %7.2fs' % (spec, dev, pt[dev]['seconds']))
        if len(devs) == 2:
            pt['cuda_speedup'] = (pt['cpu']['seconds'] / pt['cuda']['seconds'])
            pt['max_abs_curve_diff'] = float(
                max(abs(x - y) for x, y in zip(pt['cpu']['F'],
                                               pt['cuda']['F'])))
            log('  [bench] [[%s]] CUDA speedup %.2fx; curves agree to %.2e'
                % (spec, pt['cuda_speedup'], pt['max_abs_curve_diff']))
        res['points'].append(pt)
    # Backwards-compatible flat keys for the n=9 point, which is what the prose
    # and the ED table quote as "the" GPU win.
    n9 = [q for q in res['points'] if q['code'] == '9,1,3']
    if n9:
        q = n9[0]
        res.update({'code': q['code'], 'channel': q['channel'], 'p': q['p'],
                    'recovery': q['recovery']})
        for dev in devs:
            res[dev] = q[dev]
        if len(devs) == 2:
            res['cuda_speedup_n9'] = q['cuda_speedup']
            res['max_abs_curve_diff'] = q['max_abs_curve_diff']
    return res


def device_for(n, override=None):
    """The measured-best device for a full-space run on an n-qubit code.

    The crossover is not a guess: the n=5/7/9 timings put the GPU 3.8x BEHIND at
    dim=32 (launch-bound), 3.6x ahead at dim=128 and 8.3x ahead at dim=512.  So
    the rule is dim >= 128, i.e. n >= 7 -- the mirror image of the T3 gradient
    stage, where the same launch overhead made the GPU lose at every seed count
    that mattered.  `override` wins, and CUDA is never requested when absent.
    """
    if override:
        return override
    if n >= 7 and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='T4: multi-round logical-storage simulations')
    ap.add_argument('--selftest', action='store_true',
                    help='assert every reduction and law against audited code')
    ap.add_argument('--validate', action='store_true',
                    help='cross-check R=1 against paper_numbers.json[*_p010]')
    ap.add_argument('--device', choices=('cpu', 'cuda'), default=None,
                    help='override the measured per-code device choice')
    ap.add_argument('--no-full-space', action='store_true',
                    help='reduced 2x2 sweep only (fast, no leakage measurement)')
    ap.add_argument('--no-bench', action='store_true',
                    help='skip the ~3 min CPU-vs-CUDA benchmark at [[9,1,3]]')
    ap.add_argument('--bench-only', action='store_true',
                    help='run only the device benchmark and merge it into '
                         '--out, keeping the stored sweep.  The benchmark is the '
                         'slow part of a full run (a CPU [[9,1,3]] curve alone '
                         'costs ~140 s) and the only part whose result depends on '
                         'what else the host is doing, so it is the part worth '
                         're-measuring on its own.')
    ap.add_argument('--quick', action='store_true',
                    help='n=5 only, one strength, decoder+warm')
    ap.add_argument('--reaggregate', metavar='ARTIFACT',
                    help='recompute the summary/invariants from the records '
                         'already stored in ARTIFACT without re-running the '
                         'sweep.  The reduced curves are deterministic, so '
                         're-simulating to fix a summary column would be waste.')
    ap.add_argument('--out', default=OUT_JSON)
    args = ap.parse_args(argv)

    if args.selftest:
        return 1 if selftest(heavy=not args.quick) else 0
    if args.reaggregate:
        with open(args.reaggregate) as fh:
            old = json.load(fh)
        print('=== re-aggregating %d records from %s ==='
              % (len(old['records']), args.reaggregate), flush=True)
        recs = enrich_records(old['records'])
        rows = summarise(recs)
        return _finish(args, recs, rows, old.get('full_space'),
                       old.get('device_benchmark'), old.get('validation'),
                       old.get('meta', {}).get('wall_s', 0.0))
    if args.bench_only:
        if not os.path.exists(args.out):
            raise SystemExit('--bench-only needs an existing artifact at %s; '
                             'run the sweep first' % args.out)
        with open(args.out) as fh:
            old = json.load(fh)
        print('=== re-measuring the device benchmark into %s ===' % args.out,
              flush=True)
        old['device_benchmark'] = device_benchmark()
        tmp = args.out + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(old, fh, indent=1)
        os.replace(tmp, args.out)
        print('wrote %s (%.1f KB)'
              % (args.out, os.path.getsize(args.out) / 1024.0))
        return 0

    t0 = time.time()
    print('=== T4 multi-round logical storage ===', flush=True)
    print('  codes: %s   rounds up to: %d' % (', '.join(CODE_ORDER),
                                               ROUNDS[-1]), flush=True)
    print('  strengths: %s   readout eta: %s' % (P_GRID, ETA_GRID), flush=True)

    val = validate() if args.validate else None

    if args.quick:
        recs = []
        code5 = get_code('5,1,3')
        for channel in CHANNELS:
            for kind in ('decoder', 'warm'):
                S, meta = build_round(code5, channel, 0.10, kind)
                rec = {'code': '5,1,3', 'n': 5, 'channel': channel, 'p': 0.10,
                       'recovery': kind, 'eta': 0.0, 'rounds': list(ROUNDS),
                       'F': [float(x) for x in fidelity_curve(S, ROUNDS)]}
                rec.update({k: meta[k] for k in ('F_R1', 'unitarity_dev')
                            if k in meta})
                rec.update(asymptotic_decay(S))
                rec.update(storage_lifetime(S))
                recs.append(rec)
        rows = summarise(recs)
        full = bench = None
    else:
        print('\n--- reduced 2x2 sweep (exact, all codes) ---', flush=True)
        recs = sweep()
        rows = summarise(recs)
        full = bench = None
        if not args.no_full_space:
            print('\n--- full dim x dim sweep (leakage; device chosen per code) '
                  '---', flush=True)
            wanted = ([('5,1,3', c, p, k) for c in CHANNELS for p in P_GRID
                       for k in ('decoder', 'warm')]
                      + [(s, c, p, 'decoder')
                         for s in ('7,1,3', '9,1,3')
                         for c in CHANNELS_BY_CODE[s] for p in P_GRID])
            full = []
            for spec, channel, p, kind in wanted:
                dev = device_for(get_code(spec).n, args.device)
                full += full_space_curves([(spec, channel, p, kind)],
                                          device=dev)
        if not args.no_bench:
            print('\n--- device benchmark ([[9,1,3]] full space) ---', flush=True)
            bench = device_benchmark()
    return _finish(args, recs, rows, full, bench, val, time.time() - t0)


def _finish(args, recs, rows, full, bench, val, wall_s):
    """Write the artifact and enforce the invariants every curve must satisfy."""
    payload = {
        'meta': {
            'task': 'T4 multi-round logical-storage simulations',
            'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'host': os.uname().nodename, 'python': sys.version.split()[0],
            'numpy': np.__version__, 'torch': torch.__version__,
            'cuda_available': bool(torch.cuda.is_available()),
            'device_name': (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else None),
            'wall_s': wall_s,
            'reaggregated_from': args.reaggregate,
            'round_map': ('E(rho) = sum_s R_s P_s N(rho) P_s R_s^dagger, '
                          'iterated as a 4x4 superoperator on the 2x2 logical '
                          'space, so R rounds costs one eigendecomposition'),
            'estimator': ('exact 3x7 Haar quadrature, exact for every R because '
                          'the round map is linear in |psi><psi| -- there is no '
                          'Monte-Carlo error in any curve here'),
            'device_policy': ('cpu at n=5 (launch-bound), cuda at n>=7 '
                              '(measured 3.6x and 8.3x faster); see '
                              'device_benchmark'),
        },
        'config': {'codes': list(CODE_ORDER), 'p_grid': list(P_GRID),
                   'rounds': list(ROUNDS), 'eta_grid': list(ETA_GRID),
                   'recoveries_by_code': {k: list(v) for k, v in
                                          RECOVERIES_BY_CODE.items()},
                   'channels_by_code': {k: list(v) for k, v in
                                        CHANNELS_BY_CODE.items()}},
        'records': recs,
        'summary': rows,
        'full_space': full,
        'device_benchmark': bench,
        'validation': val,
    }
    tmp = args.out + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, args.out)
    print('wrote %s (%.1f KB)' % (args.out, os.path.getsize(args.out) / 1024.0))

    bad = []
    if val is not None and not val['passed']:
        bad.append('validation against paper_numbers.json failed: %s'
                   % val['failures'][:3])
    for r in recs:
        # Not `== 1.0`: F(0) is the weighted sum of 21 quadrature norms and the
        # weights sum to 1 only to round-off, so F(0) = 1 - 2e-16 is exact.
        if abs(r['F'][0] - 1.0) > 1e-14:
            bad.append('F(R=0) != 1 for %s/%s/p=%.2f/%s/eta=%.3f (got %.17g)'
                       % (r['code'], r['channel'], r['p'], r['recovery'],
                          r['eta'], r['F'][0]))
        if not all(r['F'][i] >= r['F'][i + 1] - 1e-12
                   for i in range(len(r['F']) - 1)):
            # Fidelity must be monotone non-increasing in the number of rounds:
            # every round applies a noisy channel, and the recovery cannot undo
            # information the channel destroyed.  A rise means the round map is
            # not a channel.
            bad.append('F(R) is NOT monotone for %s/%s/p=%.2f/%s/eta=%.3f'
                       % (r['code'], r['channel'], r['p'], r['recovery'],
                          r['eta']))
        if abs(r['branch_prob_sum'] - 1.0) > 1e-12:
            bad.append('branch probabilities do not sum to 1 for %s/%s/p=%.2f'
                       % (r['code'], r['channel'], r['p']))
    if full:
        worst = max(r['max_abs_full_vs_reduced'] for r in full)
        print('\n  worst |full-space - reduced| over %d runs: %.2e'
              % (len(full), worst))
        for r in full:
            # 1e-11 for the decoder rather than round-off: the reduced map is
            # EXACT there, but 'mixed' composes 8^5 = 32768 Kraus operators and
            # iterating it to R=40 accumulates round-off linearly (measured
            # 1.7e-12).  Eleven correct decimal places after 40 rounds of a
            # 32768-operator channel is still an exactness claim, not a tolerance
            # loose enough to hide a real discrepancy.
            tol = (1e-11 if r['recovery'] == 'decoder'
                   else max(1e-6, 10.0 * r['unitarity_dev']))
            if r['max_abs_full_vs_reduced'] > tol:
                bad.append('%s/%s/p=%.2f/%s: full vs reduced %.2e > tol %.0e'
                           % (r['code'], r['channel'], r['p'], r['recovery'],
                              r['max_abs_full_vs_reduced'], tol))
            if r['trace_dev_max'] > 1e-11:
                bad.append('%s/%s: round map not trace-preserving (%.2e)'
                           % (r['code'], r['channel'], r['trace_dev_max']))
    if bad:
        print('\nFAILURES (%d):' % len(bad))
        for b in bad:
            print('  - %s' % b)
        return 1
    print('\nall T4 invariants hold: F(0)=1, F(R) monotone, branch weights '
          'normalised, trace preserved, and the reduced map validated against '
          'the full density matrix on every configuration')
    return 0


if __name__ == '__main__':
    sys.exit(main())

