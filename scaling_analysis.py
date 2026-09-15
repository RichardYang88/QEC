#!/usr/bin/env python3
"""scaling_analysis.py -- does VSCR's advantage survive a larger code?

THE QUESTION.  Every number in the paper is measured on [[5,1,3]].  A reviewer
is entitled to ask whether the non-Pauli headroom VSCR captures is a property of
the method or an accident of a five-qubit code, and whether the mid-circuit
measurement errors of real hardware wash it out before the code can grow.
Neither question needs a bigger *training run* to answer.

WHY IT CAN BE ANSWERED EXACTLY.  The reduced-branch formalism of
`vscr_general.py` expresses the decoder fidelity, the per-branch-unitary ceiling
(the optimum of exactly the family VSCR occupies) and the optimal-CPTP ceiling as
functions of 2x2 blocks A_k = W_s^dagger K_k V alone.  No density matrix, no
ansatz and no optimizer enters, so [[7,1,3]] and [[9,1,3]] cost the same 4x4
Rayleigh quotient and the same 4x4 Choi program per branch as [[5,1,3]]; only
the construction of A_k grows, and `vscr_general` collapses that through the
Pauli/centralizer identity.  Everything reported here is therefore EXACT rather
than extrapolated: the ceilings are the same certified quantities the paper
reports at n=5, evaluated on bigger codes.

WHAT IS REPORTED, per code x channel x strength
  F_dec    exact min-weight-lookup-decoder fidelity (Haar-averaged, closed form)
  F_unit   optimum over per-syndrome UNITARY recoveries = the VSCR family optimum
  F_cptp   optimum over per-syndrome CPTP recoveries (4x4 Choi SDP)
  headroom F_unit - F_dec and F_cptp - F_dec
  F_petz   the noise-adapted Petz channel, exact Haar average (no Monte Carlo)
  readout  the exact suppression law under per-ancilla readout error eta

THE READOUT RESULT.  A readout error maps the true syndrome s to an observed s~
and the instrument then applies R_s~.  For a PAULI recovery every cross-branch
block V^dagger C_s~ W_s vanishes identically when s~ != s, because C_s~ C_s
carries syndrome s~ XOR s != 0 and so maps the code space orthogonal to itself.
`cross_G` verifies this numerically over all 16x16 / 64x64 / 256x256 pairs, and
it makes readout error an exact multiplicative suppression

      F(eta) = (1 - eta)^(n-k) F(0)      for the decoder AND for any branch
                                          recovery completed off the code space,

so the RELATIVE advantage of VSCR over the decoder is invariant under readout
error while the ABSOLUTE headroom shrinks by (1-eta)^(n-k).  `_selftest_readout`
checks the law against an independent full-density-matrix simulation at n=5.  A
recovery retrained *with* noisy syndromes can only beat the fixed
noiseless-optimal branches used here, so the reported headroom at eta is a
CONSERVATIVE (lower) estimate of VSCR's advantage.

CROSS-VALIDATION (the point of `--validate`).  At n=5 this script must reproduce
the audited production numbers: F_dec / F_unit / F_cptp against
`paper_numbers.json['abl']['sdp']`, the branch weights p_s entry by entry, and
the Petz fidelities against ED Table 9.  If those agree, the n=7 and n=9 rows
come from the same validated code path and can be trusted.

Usage:
    ./qenv/bin/python scaling_analysis.py --validate      # n=5 only
    ./qenv/bin/python scaling_analysis.py                 # full n=5/7/9 run
    ./qenv/bin/python scaling_analysis.py --no-sdp        # skip the CPTP SDP
    ./qenv/bin/python scaling_analysis.py --json out.json # write results
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Single-threaded BLAS, set BEFORE numpy is imported.  OpenBLAS fixes its worker
# pool at library initialisation, so setting these after `import numpy` leaves the
# process multi-threaded even though vscr_general also calls omp_set_num_threads(1)
# afterwards.  That is not cosmetic here: StabCode builds its logical basis from
# np.linalg.eigh on operators whose eigenspaces are 2^(n-1)-fold DEGENERATE, so the
# returned eigenvector basis -- and with it W_s and every reduced block A_k --
# depends on the LAPACK blocking, which depends on the thread count.  A
# multi-threaded run was reproducibly ~7e-4 off the single-threaded value for
# n=5 amplitude damping p=0.05, and repeated the p=0.10 Petz number at p=0.05 for
# n=7.  Importing vscr_general first is what makes these take effect.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import numpy as np                                              # noqa: E402
import vscr_general as g                                        # noqa: E402
# The ceiling SOLVERS are reused from the production ablation module rather than
# reimplemented, so a scaling number and a paper number come from identical code.
import vscr_paper_abl as ab                                     # noqa: E402

CODE_ORDER = ('5,1,3', '7,1,3', '9,1,3')

# (channel, strength) grid.  Amplitude damping and coherent over-rotation are the
# two channels on which the paper certifies real non-Pauli headroom; depolarizing
# is the zero-headroom control and mixed the Pauli-plus-damping control.
POINTS = (('amplitude_damping', 0.05), ('amplitude_damping', 0.10),
          ('coherent', 0.05), ('coherent', 0.10),
          ('depolarizing', 0.10), ('mixed', 0.10))

# Readout errors spanning current superconducting hardware: state-of-the-art
# mid-circuit readout is ~0.5%, typical cloud devices a few percent, and the


# ---------------------------------------------------------------------------
# channels, applied qubit-sequentially exactly as ssvr_qec does
# ---------------------------------------------------------------------------
def _qubit_kraus(channel, p):
    """Single-qubit Kraus stages of `channel` at strength p.

    Matches ssvr_qec exactly: depolarizing gives I with weight 1-p and X/Y/Z with
    p/3; amplitude damping gives diag(1, sqrt(1-p)) and sqrt(p)|0><1|; 'mixed' is
    depolarizing(p) FOLLOWED BY amplitude damping(p/2) (apply_mixed); 'coherent'
    is a single Rx(p).  Returns a list of stages, each a list of 2x2 operators."""
    dep = [math.sqrt(1 - p) * np.eye(2, dtype=complex)] + \
          [math.sqrt(p / 3.0) * g._MAT[v] for v in (g.X_, g.Y_, g.Z_)]
    gam = 0.5 * p if channel == 'mixed' else p
    ad = [np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gam)]], dtype=complex),
          np.array([[0.0, math.sqrt(gam)], [0.0, 0.0]], dtype=complex)]
    rx = np.array([[math.cos(p / 2), -1j * math.sin(p / 2)],
                   [-1j * math.sin(p / 2), math.cos(p / 2)]], dtype=complex)
    if channel == 'depolarizing':
        return [dep]
    if channel == 'amplitude_damping':
        return [ad]
    if channel == 'mixed':
        return [dep, ad]
    if channel == 'coherent':
        return [[rx]]
    raise ValueError(channel)


def apply_channel(rho, code, channel, p, dagger=False):
    """Apply the noise channel (or its Hilbert-space adjoint) to a dim x dim rho.

    Qubit-sequential, as ssvr_qec.apply_depolarizing / apply_amplitude_damping
    are, so the cost is O(n * dim^2 * |ops|) rather than the O(4^n * dim^3) of
    forming the full Kraus set.  That is what makes the Petz map -- which needs
    both N and N^dagger acting on dim x dim operators -- affordable at n=9.

    The adjoint of a COMPOSED channel reverses the stage order, exactly as
    ssvr_qec.apply_channel_adjoint does: (AD o Dep)^dagger = Dep^dagger o
    AD^dagger.  Getting this wrong is invisible on the three single-stage
    channels and shows up only on 'mixed', which is why `_selftest_petz`
    compares against production on all four."""
    rho = np.asarray(rho, dtype=complex)
    n = code.n
    T = rho.reshape([2] * (2 * n))
    stages = _qubit_kraus(channel, p)
    if dagger:
        stages = list(reversed(stages))
    for stage in stages:
        ops = [np.asarray(o, dtype=complex) for o in stage]
        if dagger:                      # N^dagger(y) = sum_k K_k^dagger y K_k
            ops = [o.conj().T for o in ops]

        for q in range(n):
            acc = np.zeros_like(T)
            for op in ops:
                A = np.moveaxis(np.tensordot(op, T, axes=([1], [q])), 0, q)
                A = np.moveaxis(
                    np.tensordot(op.conj(), A, axes=([1], [n + q])), 0, n + q)
                acc = acc + A
            T = acc
    return T.reshape(rho.shape)

# paper's own late-measure identity branch on WK_C180 retains only 2.4% of shots.
ETAS = (0.005, 0.01, 0.02, 0.05, 0.10)



# ---------------------------------------------------------------------------
# exact fidelities and ceilings
# ---------------------------------------------------------------------------
def branch_bundle(code, channel, p):
    """Per-branch (Q_s, const_s, p_s, M_s, G_dec_s, {A_k}) for a channel point."""
    A = g.branch_A(code, channel, p)
    out = []
    for s in range(code.nsyn):
        Q, const, ps = g.qform(A[s])
        M, pm = g.branch_M(A[s])
        assert abs(pm - ps) < 1e-12 * max(1.0, ps), (s, pm, ps)
        # with W_s = C_s V the decoder's reduced block is exactly the identity,
        # which is the basis-independent statement "C_s undoes the branch".
        Gd = code.V.conj().T @ code.C_mat[s] @ code.W[s]
        out.append(dict(Q=Q, const=const, p_s=ps, M=M, G_dec=Gd, A=A[s]))
    assert np.abs(out[0]['G_dec'] - np.eye(2)).max() < 1e-12
    return out


def fidelities(code, channel, p, do_sdp=True, n_start=8, log=None):
    """F_dec, F_unit (the VSCR family optimum) and F_cptp, with duality asserts."""
    B = branch_bundle(code, channel, p)
    F_dec = F_unit = F_cptp = 0.0
    ps_l, cfd_l, cfu_l, cfm_l, gap_l = [], [], [], [], []
    for s, b in enumerate(B):
        ps = b['p_s']
        ps_l.append(ps)
        if ps < 1e-13:
            cfd_l.append(1.0); cfu_l.append(1.0); cfm_l.append(1.0)
            gap_l.append(0.0)
            F_dec += ps; F_unit += ps; F_cptp += ps
            continue
        gd = b['G_dec'].reshape(4)
        Jd = float(np.real(gd @ b['Q'] @ gd.conj()))
        single = b['A'][0] if len(b['A']) == 1 else None
        Ju, Gu = ab._max_unitary_J(b['Q'], n_start=n_start, seed=0,
                                   single=single, G_dec=b['G_dec'])
        # Certified upper bound on the unitary family: ||vec(G)||^2 = 2 for any
        # G in U(2), so J <= 2 lambda_max(Q).  Reporting the gap makes any
        # under-convergence of the multi-start solver visible, not silent.
        bound = 2.0 * float(np.linalg.eigvalsh(b['Q'])[-1])
        assert Ju <= bound + 1e-9, (s, Ju, bound)
        gap_l.append(bound - Ju)
        cfd_l.append((Jd + b['const']) / 6 / ps)
        cfu_l.append((Ju + b['const']) / 6 / ps)
        F_dec += ps * cfd_l[-1]
        F_unit += ps * cfu_l[-1]
        if do_sdp:
            fu, _ = ab._sdp_branch(b['M'], b['G_dec'], G_ref=Gu)
            cfm_l.append(fu / ps)
            F_cptp += fu
        else:
            cfm_l.append(None)
    if do_sdp:
        # weak duality: the same check production's `sdp_ceiling` performs at n=5
        assert F_cptp >= F_unit - 1e-9, (F_cptp, F_unit)
        assert F_cptp >= F_dec - 1e-9, (F_cptp, F_dec)
    tot = float(sum(ps_l))
    assert abs(tot - 1.0) < 1e-9, f'branch weights sum to {tot}, not 1'
    if log:
        print(f'    {log}', flush=True)
    return dict(F_dec=F_dec, F_unit=F_unit,
                F_cptp=(F_cptp if do_sdp else None),
                headroom_unit=F_unit - F_dec,
                headroom_cptp=(F_cptp - F_dec if do_sdp else None),
                p_s=ps_l, cf_dec=cfd_l, cf_unit=cfu_l, cf_cptp=cfm_l,
                max_unitary_gap_to_bound=max(gap_l))


# ---------------------------------------------------------------------------
# the noise-adapted Petz channel, Haar-averaged exactly
# ---------------------------------------------------------------------------
def _petz_map(code, channel, p):
    """Build the code's noise-adapted Petz recovery as a callable on dim x dim.

    Same construction as ssvr_qec._petz_build: reference state sigma = P_code/2,
    so sigma^{1/2} = P_code/sqrt(2), and N(sigma)^{-1/2} pseudo-inverted on the
    support of N(sigma).  Returns (rec, E) with E the encoding isometry."""
    E = code.V
    Pc = E @ E.conj().T
    sig_half = Pc / math.sqrt(2.0)
    nsig = apply_channel(Pc / 2.0, code, channel, p)
    nsig = 0.5 * (nsig + nsig.conj().T)
    ev, U = np.linalg.eigh(nsig)
    tol = float(ev.max()) * 1e-12
    inv = np.where(ev > tol, 1.0 / np.sqrt(np.clip(ev, tol, None)), 0.0)
    Ainv = (U * inv) @ U.conj().T

    def rec(y):
        return sig_half @ apply_channel(Ainv @ y @ Ainv, code, channel, p,
                                        dagger=True) @ sig_half
    return rec, E


def haar_average_of_map(code, rec, E):
    """Exact E_psi<psi_E| rec(|psi_E><psi_E|) |psi_E> over the Haar ensemble.

    Builds the 2x2 superoperator Phi_(ac),(bd) = <a| E^dag rec(E |b><d| E^dag) E |c>
    on the four matrix units and applies the degree-2 moment on CP^1,

        E_psi <psi|Phi(|psi><psi|)|psi>
            = ( sum_{a,c} Phi_(ac),(ac)  +  sum_{a,b} Phi_(aa),(bb) ) / 6
            = ( sum_mu |Tr B_mu|^2 + sum_mu ||B_mu||_F^2 ) / 6   in Kraus form,

    the same identity the paper's cf_s closed form rests on.  `_selftest_haar`
    checks both the Kraus-form equality and the trivial case Phi = id -> 1,
    because the two index pairings above are easy to transpose and the identity
    channel does not distinguish them."""
    S = np.zeros((4, 4), dtype=complex)
    for b in range(2):
        for d in range(2):
            X = np.zeros((2, 2), dtype=complex)
            X[b, d] = 1.0
            Y = E.conj().T @ rec(E @ X @ E.conj().T) @ E
            for a in range(2):
                for c in range(2):
                    S[a * 2 + c, b * 2 + d] = Y[a, c]
    t1 = sum(S[a * 2 + c, a * 2 + c] for a in range(2) for c in range(2))
    t2 = sum(S[a * 2 + a, b * 2 + b] for a in range(2) for b in range(2))
    return float(np.real(t1 + t2)) / 6.0


def petz_round_trip(code, channel, p):
    """The composed map R_Petz o N acting on dim x dim operators, plus E.

    This composition is the object whose Haar-averaged pure-state fidelity is
    the reported Petz number, so it is factored out and used by both the exact
    closed form and the Monte-Carlo cross-check."""
    rec, E = _petz_map(code, channel, p)

    def rn(y):
        return rec(apply_channel(y, code, channel, p))
    return rn, E


def petz_fidelity(code, channel, p):
    """Exact Haar-averaged fidelity of the code's noise-adapted Petz recovery."""
    rn, E = petz_round_trip(code, channel, p)
    return haar_average_of_map(code, rn, E)



def _selftest_haar(code5, n_states=4000, seed=3):
    """The closed form must equal a Monte-Carlo Haar average of the SAME map.

    Guards against transposing the two index pairings in `haar_average_of_map`:
    the identity channel gives 1 under either, so it is checked separately from
    the Petz map, whose average is sampled here directly."""
    # (a) Phi = identity  ->  exactly 1
    E = code5.V
    one = haar_average_of_map(code5, lambda y: y, E)
    assert abs(one - 1.0) < 1e-12, one
    # (b) Kraus cross-check of the formula on a random CP map
    rng = np.random.RandomState(seed)
    for _ in range(3):
        B = [rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
             for _ in range(3)]
        sc = 1.0 / max(float(np.linalg.eigvalsh(
            sum(b.conj().T @ b for b in B)).max()), 1.0)
        B = [b * math.sqrt(sc) for b in B]

        def phi(Y, B=B):
            return sum(b @ Y @ b.conj().T for b in B)
        exact = haar_average_of_map(code5, lambda y: E @ phi(
            E.conj().T @ y @ E) @ E.conj().T, E)
        kraus = sum(abs(np.trace(b)) ** 2
                    + float(np.real(np.trace(b @ b.conj().T))) for b in B) / 6.0
        assert abs(exact - kraus) < 1e-12, (exact, kraus)
    # (c) Petz round trip: closed form vs Monte-Carlo over Haar-random states
    worst = 0.0
    for ch, p in (('depolarizing', 0.10), ('amplitude_damping', 0.10),
                  ('coherent', 0.15)):
        rn, _E = petz_round_trip(code5, ch, p)
        exact = haar_average_of_map(code5, rn, _E)
        rs = np.random.RandomState(seed)
        vals = []
        for _ in range(n_states):
            z = rs.normal(size=2) + 1j * rs.normal(size=2)
            psi = z / np.linalg.norm(z)
            pe = _E @ psi
            vals.append(float(np.real(pe.conj() @ rn(np.outer(pe, pe.conj()))
                                      @ pe)))
        mc = float(np.mean(vals))
        sem = float(np.std(vals) / math.sqrt(n_states))
        print(f'    [haar] {ch:18s} p={p}: closed form {exact:.9f} vs MC '
              f'{mc:.9f} +- {sem:.1e}   diff {abs(exact-mc):.2e}', flush=True)
        assert abs(exact - mc) < 5 * sem + 1e-4, (ch, exact, mc, sem)
        worst = max(worst, abs(exact - mc))
    return worst




# ---------------------------------------------------------------------------
# readout error
# ---------------------------------------------------------------------------
def readout_crossblocks(code):
    """(max |V^dag C_s~ W_s| over s~ != s,  max |V^dag C_s W_s - I|).

    The first must be numerically zero: C_s~ C_s carries syndrome s~ XOR s != 0
    and therefore maps the code space orthogonal to itself.  That single fact is
    what makes readout error an exact multiplicative suppression instead of a
    channel-dependent perturbation, and it is checked on every pair."""
    Gx = code.cross_G(code.C_mat)
    off = np.abs(Gx).copy()
    for s in range(code.nsyn):
        off[s, s] = 0.0
    diag = max(float(np.abs(Gx[s, s] - np.eye(2)).max())
               for s in range(code.nsyn))
    return float(off.max()), diag


def readout_curve(F0, m, etas=ETAS):
    """Exact F(eta) = (1-eta)^m F(0) for a Pauli recovery on m = n-k ancillas."""
    return {f'{e:g}': F0 * (1.0 - e) ** m for e in etas}


def fidelity_at_eta(code, channel, p, eta, R_mats=None):
    """Exact F(eta) INCLUDING any cross-branch leakage, from the reduced blocks.

        F(eta) = sum_{s,s~} T(s~|s) sum_k [ |Tr(G~_{s->s~} A_k)|^2
                                           + Tr(G~ A_k A_k^dag G~^dag) ] / 6 ,
        T(s~|s) = eta^d (1-eta)^(m-d),  d = Hamming(s, s~),

    with G~_{s->s~} = V^dagger R_s~ W_s.  This is the general form; the
    multiplicative law `readout_curve` is the special case in which every
    off-diagonal G~ vanishes.  Pairs whose block is numerically zero are skipped,
    which is what keeps the cost at O(nsyn) rather than O(nsyn^2) for a Pauli
    recovery -- and the number of skipped pairs is itself the evidence for the
    law, so `_selftest_readout` counts them."""
    A = g.branch_A(code, channel, p)
    Rm = code.C_mat if R_mats is None else np.asarray(R_mats)
    Gx = code.cross_G(Rm)                       # (nsyn~, nsyn, 2, 2)
    m = code.m
    tot = 0.0
    used = 0
    for s in range(code.nsyn):
        for st in range(code.nsyn):
            if np.abs(Gx[st, s]).max() < 1e-15:
                continue
            d = bin(s ^ st).count('1')
            T = (eta ** d) * ((1.0 - eta) ** (m - d))
            if T == 0.0:
                continue
            used += 1
            tot += T * g.cf_unnormalised(Gx[st, s], A[s])
    return tot, used



def simulate_readout_n5(code5, channel, p, eta, n_states=120, seed=7):
    """Independent full-density-matrix check of the suppression law at n=5.

    Shares NO code with the reduced-branch path: it forms P_s as dim x dim
    operators from ssvr_qec, applies the channel to explicit encoded pure states,
    and averages the post-selected fidelity over random logical states while
    mixing over observed syndromes with T(s~|s) = eta^d (1-eta)^(m-d)."""
    import ssvr_qec as m5
    rng = np.random.RandomState(seed)
    Vm = code5.V
    P = [m5.P_SYNDS[s].numpy() for s in range(16)]
    C = [code5.C_mat[s] for s in range(16)]
    # the two decoder tables must be the same Paulis
    for s in range(16):
        ref = m5.pauli_string(m5.SYND_TABLE[m5.SYND_BITS[s]]).numpy()
        assert np.abs(C[s] - ref).max() < 1e-14, f'decoder mismatch at s={s}'
    m = code5.m
    tot = 0.0
    for _ in range(n_states):
        z = rng.normal(size=2) + 1j * rng.normal(size=2)
        psi = z / np.linalg.norm(z)
        pe = Vm @ psi
        rho = apply_channel(np.outer(pe, pe.conj()), code5, channel, p)
        acc = 0.0
        for s in range(16):
            pr = P[s] @ rho @ P[s]
            if np.abs(pr).max() < 1e-18:
                continue
            for st in range(16):
                d = bin(s ^ st).count('1')
                T = (eta ** d) * ((1 - eta) ** (m - d))
                if T == 0.0:
                    continue
                v = C[st] @ pe
                acc += T * float(np.real(v.conj() @ pr @ v))
        tot += acc
    return tot / n_states


# ---------------------------------------------------------------------------
# cross-validation against the audited production artifacts
# ---------------------------------------------------------------------------
def _selftest_adjoint(code5, n_rand=6, seed=13):
    """Tr[X^dag N(Y)] == Tr[N^dag(X)^dag Y] on random operators, all channels.

    This is the identity ssvr_qec._selftest_petz uses to validate its own
    adjoint; it is the sharp test for the composed-stage reversal, because a
    wrong stage order still yields a CP map and only breaks this pairing."""
    rng = np.random.RandomState(seed)
    worst = 0.0
    for ch, p in (('depolarizing', 0.10), ('amplitude_damping', 0.10),
                  ('mixed', 0.10), ('coherent', 0.15)):
        for _ in range(n_rand):
            d = code5.dim
            X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
            Y = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
            lhs = np.trace(X.conj().T @ apply_channel(Y, code5, ch, p))
            rhs = np.trace(apply_channel(X, code5, ch, p, dagger=True).conj().T
                           @ Y)
            worst = max(worst, abs(lhs - rhs) / max(abs(lhs), 1e-30))
    print(f'    [adjoint] worst relative Tr[X^dag N(Y)] - Tr[N^dag(X)^dag Y] '
          f'= {worst:.2e}', flush=True)
    assert worst < 1e-12, worst
    return worst


def _selftest_petz(code5, n_states=400, seed=11):
    """Two independent agreements with the production Petz implementation.

    (a) `apply_channel` must reproduce `ssvr_qec.noisy_state` bit-for-bit on all
        four channels -- this is the channel-definition check, and it is what
        licenses comparing anything below;
    (b) the exact Haar average must equal a Monte-Carlo average of
        `ssvr_qec.petz_recovery_fidelity`, which is how ED Table 9 was produced.
    """
    import ssvr_qec as m5
    import torch
    rng = np.random.RandomState(seed)
    worst = 0.0
    for ch, p in (('depolarizing', 0.10), ('amplitude_damping', 0.10),
                  ('mixed', 0.10), ('coherent', 0.15)):
        # (a) channel agreement, on encoded pure states
        dmax = 0.0
        for _ in range(8):
            z = rng.normal(size=2) + 1j * rng.normal(size=2)
            psi = torch.tensor(z / np.linalg.norm(z), dtype=m5.DTYPE)
            pe = m5.encode(psi)
            pe_np = pe.numpy()
            ref = m5.noisy_state(pe, p, ch).numpy()
            got = apply_channel(np.outer(pe_np, pe_np.conj()), code5, ch, p)
            dmax = max(dmax, float(np.abs(ref - got).max()))
        assert dmax < 1e-14, (ch, dmax)
        # (b) exact vs production Monte-Carlo Petz fidelity
        exact = petz_fidelity(code5, ch, p)
        vals = []
        for _ in range(n_states):
            z = rng.normal(size=2) + 1j * rng.normal(size=2)
            psi = torch.tensor(z / np.linalg.norm(z), dtype=m5.DTYPE)
            pe = m5.encode(psi)
            rho = m5.noisy_state(pe, p, ch)
            vals.append(m5.petz_recovery_fidelity(pe, rho, ch, p))
        mc = float(np.mean(vals))
        sem = float(np.std(vals) / math.sqrt(len(vals)))
        d = abs(exact - mc)
        worst = max(worst, d)
        print(f'    [petz] {ch:18s} p={p}: exact {exact:.9f} vs production '
              f'MC {mc:.9f} +- {sem:.1e}  (channel dev {dmax:.1e})  '
              f'diff {d:.2e}', flush=True)
        assert d < 5 * sem + 1e-4, (ch, exact, mc, sem)
    return worst



def _selftest_readout(code5):
    """Three checks of the exact readout-error treatment.

    (a) `cf_unnormalised` must reduce to p_s * cf_of when G~ is unitary -- the
        identity that licenses using the general form for vanishing blocks;
    (b) every off-diagonal cross-branch block of the Pauli decoder must vanish,
        and the exact `fidelity_at_eta` must then equal (1-eta)^m F(0) to
        machine precision while touching only the nsyn diagonal pairs;
    (c) an INDEPENDENT full-density-matrix Monte-Carlo simulation must agree
        with that law within its own sampling error."""
    m = code5.m
    A = g.branch_A(code5, 'depolarizing', 0.10)
    Q, const, ps = g.qform(A[0])
    Gr = np.linalg.qr(np.random.RandomState(0).normal(size=(2, 2))
                      + 1j * np.random.RandomState(1).normal(size=(2, 2)))[0]
    lhs = g.cf_unnormalised(Gr, A[0])
    rhs = ps * g.cf_of(Gr, Q, const, ps)
    print(f'    [cf] cf_unnormalised vs p_s*cf_of for unitary G: '
          f'{lhs:.15f} vs {rhs:.15f}  diff {abs(lhs-rhs):.2e}', flush=True)
    assert abs(lhs - rhs) < 1e-12, (lhs, rhs)

    out = {}
    for ch, p in (('depolarizing', 0.10), ('amplitude_damping', 0.10)):
        F0 = fidelities(code5, ch, p, do_sdp=False)['F_dec']
        off, diag = readout_crossblocks(code5)
        assert off < 1e-14, f'cross-branch block did not vanish: {off}'
        assert diag < 1e-14, f'diagonal block is not the identity: {diag}'
        for eta in (0.0, 0.02, 0.05):
            ex, used = fidelity_at_eta(code5, ch, p, eta)
            law = F0 * (1.0 - eta) ** m
            assert used == code5.nsyn, (used, code5.nsyn)
            print(f'    [readout-exact] {ch:18s} eta={eta:.3f}: {ex:.12f} vs '
                  f'law {law:.12f}  diff {abs(ex-law):.2e}  '
                  f'({used}/{code5.nsyn**2} pairs nonzero)', flush=True)
            assert abs(ex - law) < 1e-12, (ch, eta, ex, law)
            sim = simulate_readout_n5(code5, ch, p, eta)
            print(f'    [readout-sim  ] {ch:18s} eta={eta:.3f}: {sim:.9f} '
                  f'(independent density-matrix MC)', flush=True)
            assert abs(sim - law) < 5e-4, (ch, eta, sim, law)
        out[ch] = dict(F_dec_0=F0, cross_offdiag=off, cross_diag=diag)
    return out



def validate_n5(code5, do_sdp=True):
    """Reproduce every audited n=5 ceiling in paper_numbers.json['abl']['sdp']."""
    P = json.load(open(os.path.join(ROOT, 'paper_numbers.json')))
    sdp = P['abl']['sdp']
    keymap = {'dep_0.05': ('depolarizing', 0.05), 'dep_0.1': ('depolarizing', 0.10),
              'dep_0.15': ('depolarizing', 0.15), 'ad_0.1': ('amplitude_damping', 0.10),
              'mixed_0.1': ('mixed', 0.10), 'coh_0.1': ('coherent', 0.10),
              'coh_0.15': ('coherent', 0.15), 'coh_0.3': ('coherent', 0.30)}
    worst = dict(F_dec=0.0, F_unit=0.0, F_cptp=0.0, p_s=0.0)
    for k, (ch, p) in keymap.items():
        ref = sdp[k]
        got = fidelities(code5, ch, p, do_sdp=do_sdp)
        d_dec = abs(got['F_dec'] - ref['F_dec_exact'])
        d_uni = abs(got['F_unit'] - ref['F_unit'])
        d_ps = float(np.abs(np.array(got['p_s']) - np.array(ref['p_s'])).max())
        d_cpt = (abs(got['F_cptp'] - ref['F_cptp']) if do_sdp else 0.0)
        worst['F_dec'] = max(worst['F_dec'], d_dec)
        worst['F_unit'] = max(worst['F_unit'], d_uni)
        worst['F_cptp'] = max(worst['F_cptp'], d_cpt)
        worst['p_s'] = max(worst['p_s'], d_ps)
        print(f'    [{k:10s}] F_dec {got["F_dec"]:.9f} (ref {ref["F_dec_exact"]:.9f}, '
              f'd {d_dec:.1e})  F_unit d {d_uni:.1e}  F_cptp d {d_cpt:.1e}  '
              f'p_s d {d_ps:.1e}', flush=True)
        assert d_dec < 1e-9 and d_uni < 1e-8 and d_ps < 1e-12
        if do_sdp:
            assert d_cpt < 1e-8, (k, d_cpt)
    print(f'    worst deviations: {worst}', flush=True)
    return worst


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
def run_code(code, name, do_sdp=True, n_start=8, resolvable=1e-3):
    """Full sweep for one code: ceilings, Petz, and the exact readout law."""
    off, diag = readout_crossblocks(code)
    assert off < 1e-13, f'cross-branch block did not vanish on [[{name}]]: {off}'
    assert diag < 1e-13, f'decoder block is not the identity on [[{name}]]: {diag}'
    out = {'code': name, 'n': code.n, 'n_ancilla': code.m,
           'nsyn': code.nsyn, 'distance': code.distance,
           'readout_cross_offdiag': off, 'readout_cross_diag_dev': diag,
           'resolvable_scale': resolvable, 'points': {}}
    print(f'\n=== [[{name}]]  n={code.n}  ancillas={code.m}  '
          f'syndromes={code.nsyn}  d={code.distance} ===', flush=True)
    print(f'    readout cross-branch blocks: max |V^dag C_s~ W_s| (s~!=s) = '
          f'{off:.2e}  -> exact (1-eta)^{code.m} suppression', flush=True)
    for ch, p in POINTS:
        if ch == 'mixed' and code.n > 5:
            # 'mixed' composes a Pauli channel with amplitude damping, so its
            # Kraus set is 8^n and has no Pauli shortcut; see branch_A's guard.
            print(f'  {ch:18s} p={p:<5g} skipped (dense 8^n Kraus set, n>5)',
                  flush=True)
            continue
        t0 = time.time()
        r = fidelities(code, ch, p, do_sdp=do_sdp, n_start=n_start)
        r['F_petz'] = petz_fidelity(code, ch, p)
        r['eta_curve'] = {
            'F_dec': readout_curve(r['F_dec'], code.m),
            'F_unit': readout_curve(r['F_unit'], code.m),
            'headroom': readout_curve(r['headroom_unit'], code.m)}
        # re-derive the law from the exact cross-branch sum on THIS code, so the
        # suppression factor is verified at n=7 and n=9 and not just at n=5
        r['eta_exact_check'] = {}
        for eta in (0.01, 0.05):
            ex, used = fidelity_at_eta(code, ch, p, eta)
            law = r['F_dec'] * (1.0 - eta) ** code.m
            assert used == code.nsyn, (ch, used, code.nsyn)
            assert abs(ex - law) < 1e-11, (ch, eta, ex, law)
            r['eta_exact_check'][f'{eta:g}'] = dict(
                exact=ex, law=law, dev=abs(ex - law), pairs_nonzero=used,
                pairs_total=code.nsyn ** 2)

        d0 = r['headroom_unit']
        # eta at which the ABSOLUTE headroom falls below the resolution scale of
        # a benchmark: (1-eta)^m d0 = resolvable.  None if it starts below.
        r['eta_headroom_below_resolvable'] = (
            None if d0 <= resolvable
            else 1.0 - (resolvable / d0) ** (1.0 / code.m))
        r['headroom_rel_to_logical_error'] = (
            d0 / (1.0 - r['F_dec']) if r['F_dec'] < 1 else None)
        r['seconds'] = time.time() - t0
        out['points'][f'{ch}_{p:g}'] = r
        cpt = ('%.9f' % r['F_cptp']) if do_sdp else '   --    '
        print(f'  {ch:18s} p={p:<5g} F_dec {r["F_dec"]:.9f}  '
              f'F_unit {r["F_unit"]:.9f}  F_cptp {cpt}  '
              f'headroom {d0:+.3e}  Petz {r["F_petz"]:.9f}  '
              f'eta* {("n/a" if r["eta_headroom_below_resolvable"] is None else "%.4f" % r["eta_headroom_below_resolvable"])}'
              f'  ({r["seconds"]:.0f}s)', flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--validate', action='store_true',
                    help='only cross-check n=5 against the audited artifacts')
    ap.add_argument('--no-sdp', action='store_true',
                    help='skip the per-branch optimal-CPTP Choi program')
    ap.add_argument('--no-selftest', action='store_true',
                    help='skip the Petz / readout-law self-tests')
    ap.add_argument('--codes', default=';'.join(CODE_ORDER),
                    help='semicolon-separated subset, e.g. "5,1,3;7,1,3"')
    ap.add_argument('--starts', type=int, default=8,
                    help='multi-starts for the per-branch unitary ceiling')
    ap.add_argument('--json', default='scaling_results.json')
    a = ap.parse_args(argv)
    do_sdp = not a.no_sdp
    codes = [c.strip() for c in a.codes.split(';') if c.strip()]
    for c in codes:
        assert c in g.CODES, f'unknown code {c!r}; known: {sorted(g.CODES)}'
    if a.validate:
        codes = ['5,1,3']          # validation is an n=5-only exercise


    t0 = time.time()
    print(f'=== scaling_analysis (pinned cpu {g.PINNED_CPU}, sdp={do_sdp}) ===',
          flush=True)
    built = {}
    for name in codes:
        tb = time.time()
        built[name] = g.StabCode(g.CODES[name])
        c = built[name]
        print(f'  built [[{name}]]: n={c.n} dim={c.dim} syndromes={c.nsyn} '
              f'd={c.distance} in {time.time()-tb:.1f}s', flush=True)

    res = {'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'pinned_cpu': g.PINNED_CPU, 'sdp': do_sdp, 'codes': {}}

    c5 = built.get('5,1,3') or g.StabCode(g.CODES['5,1,3'])
    if not a.no_selftest:
        print('\n--- self-test: exact Haar closed form vs Monte Carlo ---',
              flush=True)
        _selftest_haar(c5)
        print('--- self-test: exact Petz vs ssvr_qec sampled Petz (n=5) ---',
              flush=True)
        _selftest_adjoint(c5)
        _selftest_petz(c5)
        print('--- self-test: readout suppression law vs density-matrix sim ---',
              flush=True)
        _selftest_readout(c5)

    if a.validate or '5,1,3' in built:
        print('\n--- cross-validation: n=5 vs paper_numbers.json[abl][sdp] ---',
              flush=True)
        res['validation_n5'] = validate_n5(c5, do_sdp=do_sdp)
    if a.validate:
        print('\nVALIDATION ONLY: all n=5 checks reproduced the audited numbers.')
        return 0

    for name in codes:
        res['codes'][name] = run_code(built[name], name, do_sdp=do_sdp,
                                      n_start=a.starts)
    res['runtime_s'] = time.time() - t0
    with open(os.path.join(ROOT, a.json), 'w') as f:
        json.dump(res, f, indent=1)
    print(f'\nwrote {a.json}  ({res["runtime_s"]:.0f}s total)')
    return 0


if __name__ == '__main__':
    sys.exit(main())



