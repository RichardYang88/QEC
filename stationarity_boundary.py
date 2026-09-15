#!/usr/bin/env python3
"""stationarity_boundary.py -- where exactly does the decoder stop being stationary?

THE CLAIM UNDER TEST.  The paper's central structural result is that the Pauli
decoder point phi^dec is an EXACT stationary point of the label-free objective,
hence a saddle wherever certified headroom sits above it.  Methods proves it for
Pauli-mixture noise via a logical two-design twirl, then states that the same
stationarity holds numerically on amplitude damping and on the coherent channels,
attributing it to "the logical twirl induced by the Haar ensemble".  That
attribution is a conjecture, and it is load-bearing: the warm-start story rests on
the decoder point being stationary on ALL FOUR reported channels, not just the
Pauli ones.

WHAT THIS SCRIPT DOES.  It replaces the conjecture with a measurement over a
designed two-axis family and reports where the gradient is exactly zero.

  axis 1 -- the CHANNEL.  Nine noise models spanning every structural class that
    matters: Pauli-mixture and unital (depolarizing), non-unital with non-Pauli
    Kraus operators (amplitude damping), a composition of the two (mixed), a
    non-diagonal coherent over-rotation (Rx), a DIAGONAL coherent over-rotation
    (Rz, a different symmetry class), a general single-qubit channel that is
    neither unital nor Pauli-diagonal (damping + dephasing, and a seeded random
    CPTP map), and two genuinely CORRELATED channels that are not product
    channels at all (collective dephasing exp(-i theta sum Z_q) and a coherent
    ZZ ring exp(-i theta sum Z_q Z_q+1)).

  axis 2 -- the ENSEMBLE / LOSS.  The Haar objective is a complex 2-design
    average, so the family crosses three ensembles that ARE complex 2-designs on
    CP^1 (the exact Haar closed form, the 4-state tetrahedral SIC, the 6-state
    octahedron) against three that are NOT (the single fixed states |0_L> and
    |+_L>, and the two-state computational ensemble {|0_L>,|1_L>}).

The hypothesis under test is that the boundary is the ENSEMBLE, not the channel:
stationarity should hold for every channel under every 2-design ensemble and fail
for every channel under every non-design ensemble.  If some channel breaks that
pattern, the script reports it rather than hiding it.

METHOD.  F(phi) is evaluated from the reduced 2x2 branch blocks
A_k = W_s^dagger K_k V via the exact degree-2 moment,

    F_haar(phi)      = sum_{s,k} ( |Tr(G_s A_k)|^2 + ||G_s A_k||_F^2 ) / 6 ,
    F_ensemble(phi)  = sum_i w_i sum_{s,k} |<psi_i| G_s A_k |psi_i>|^2 ,
    G_s              = V^dagger R(phi_s) W_s ,

the same expression the production quadrature objective uses.  It is invariant
under W_s -> W_s U (G_s A_k is), so the choice of branch basis cannot affect any
gradient reported here.  R(phi_s) is the production 60-gate ansatz applied to two
columns by ssvr_qec.recovery_action_cols, so all 960 angles are differentiated by
autograd AND independently by central finite differences.

For every channel the script also reports the certified non-Pauli headroom
F_unit - F_dec, so a channel on which the gradient vanishes can be classified as a
genuine minimum (zero headroom) or as a SADDLE (positive headroom).

Usage:
    ./qenv/bin/python stationarity_boundary.py                # full grid
    ./qenv/bin/python stationarity_boundary.py --quick        # FD on 60 angles
    ./qenv/bin/python stationarity_boundary.py --json out.json
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
# Thread/coretype pins BEFORE numpy: see the note in vscr_general.py.  ssvr_qec
# and vscr_paper_abl both pin these at their own module level, but importing
# numpy first would initialise OpenBLAS multi-threaded and make those pins no-ops.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('OPENBLAS_CORETYPE', 'HASWELL')

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
torch.set_num_threads(1)

import ssvr_qec as m                                            # noqa: E402
import vscr_paper as vp                                         # noqa: E402
import vscr_paper_abl as ab                                     # noqa: E402
import vscr_general as g                                        # noqa: E402

CODE = g.StabCode(g.CODES['5,1,3'])
W_np = ab.W_BASIS                       # (16, 32, 2) production branch bases
V_np = ab.V_ISO                         # (32, 2) encoding isometry
W_t = torch.tensor(W_np, dtype=torch.complex128)
V_t = torch.tensor(V_np, dtype=torch.complex128)
I2 = np.eye(2, dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)



# ---------------------------------------------------------------------------
# the channel family.  Every builder returns K V, i.e. a (K, dim, 2) stack.
# ---------------------------------------------------------------------------
def _bits(b, n=5):
    """Per-qubit bit values of a basis index, q0 = most significant (kron order)."""
    return [(b >> (n - 1 - q)) & 1 for q in range(n)]


def _sz(b):
    """sum_q <Z_q> = sum_q (1 - 2 b_q) for the computational basis state b."""
    return sum(1 - 2 * x for x in _bits(b))


def _zz_ring(b):
    """sum_q <Z_q Z_{q+1}> on a ring, for the computational basis state b."""
    v = [1 - 2 * x for x in _bits(b)]
    return sum(v[q] * v[(q + 1) % 5] for q in range(5))


def _ad_ops(gam):
    return [np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gam)]], dtype=complex),
            np.array([[0.0, math.sqrt(gam)], [0.0, 0.0]], dtype=complex)]


def _dep_ops(p):
    return [math.sqrt(1 - p) * I2] + [math.sqrt(p / 3.0) * g._MAT[v]
                                      for v in (g.X_, g.Y_, g.Z_)]


def _deph_ops(q):
    return [math.sqrt(1 - q) * I2, math.sqrt(q) * PZ]


def _diag_unitary_V(fn):
    """(1, dim, 2): the diagonal unitary exp(i f(b)) acting on the codewords."""
    ph = np.exp(1j * np.array([fn(b) for b in range(CODE.dim)]))
    return (ph[:, None] * CODE.V)[None, :, :]


def random_cptp_qubit(seed=20240915):
    """A seeded, genuinely general single-qubit CPTP channel (4 Kraus ops).

    A random Kraus set rescaled by S^{-1/2}, S = sum B^dag B, which makes it
    trace preserving.  It is deliberately NOT unital, NOT Pauli-diagonal and NOT
    unitary, so it probes a class none of the four paper channels covers.  Trace
    preservation is asserted."""
    rs = np.random.RandomState(seed)
    B = [rs.normal(size=(2, 2)) + 1j * rs.normal(size=(2, 2)) for _ in range(4)]
    S = sum(b.conj().T @ b for b in B)
    ev, U = np.linalg.eigh(0.5 * (S + S.conj().T))
    Sih = U @ np.diag(1.0 / np.sqrt(np.clip(ev, 1e-14, None))) @ U.conj().T
    ops = [b @ Sih for b in B]
    assert np.abs(sum(o.conj().T @ o for o in ops) - I2).max() < 1e-12
    return ops


def kv_depolarizing():
    return CODE.product_kraus_V([_dep_ops(0.10)] * 5)


def kv_amplitude_damping():
    return CODE.product_kraus_V([_ad_ops(0.10)] * 5)


def kv_mixed():
    # ssvr_qec.apply_mixed: depolarizing(p) THEN amplitude damping(p/2)
    return CODE.apply_product_ops(
        CODE.product_kraus_V([_dep_ops(0.10)] * 5), [_ad_ops(0.05)] * 5)


def kv_coherent_rx():
    return (g.coherent_unitary(CODE, 0.15) @ CODE.V)[None, :, :]


def kv_coherent_rz():
    # exp(-i eps/2 sum_q Z_q): a DIAGONAL coherent over-rotation, i.e. a unitary
    # that commutes with every Z and so sits in a different symmetry class from Rx
    return _diag_unitary_V(lambda b: -0.15 / 2.0 * _sz(b))


def kv_damp_dephase():
    return CODE.apply_product_ops(
        CODE.product_kraus_V([_ad_ops(0.10)] * 5), [_deph_ops(0.10)] * 5)


def kv_random_cptp():
    return CODE.product_kraus_V([random_cptp_qubit()] * 5)


def kv_collective_dephasing():
    # exp(-i theta sum_q Z_q): CORRELATED, not a product of single-qubit channels
    return _diag_unitary_V(lambda b: -0.15 * _sz(b))


def kv_coherent_zz():
    # exp(-i theta sum_q Z_q Z_q+1): CORRELATED two-body coherent error
    return _diag_unitary_V(lambda b: -0.15 * _zz_ring(b))


CHANNELS = [
    ('depolarizing',         'Pauli mixture, unital',        kv_depolarizing),
    ('amplitude damping',    'non-unital, non-Pauli Kraus',  kv_amplitude_damping),
    ('mixed',                'Pauli mixture then damping',   kv_mixed),
    ('coherent Rx',          'unitary, non-diagonal',        kv_coherent_rx),
    ('coherent Rz',          'unitary, diagonal',            kv_coherent_rz),
    ('damping + dephasing',  'general single-qubit',         kv_damp_dephase),
    ('random CPTP',          'general single-qubit, seeded', kv_random_cptp),
    ('collective dephasing', 'correlated, not a product',    kv_collective_dephasing),
    ('coherent ZZ ring',     'correlated two-body coherent', kv_coherent_zz),
]


# ---------------------------------------------------------------------------
# ensembles: three complex 2-designs on CP^1, three that are not
# ---------------------------------------------------------------------------
def _bloch(r):
    """Logical-basis spinor |psi> = cos(t/2)|0_L> + e^{i ph} sin(t/2)|1_L>."""
    n = math.sqrt(sum(x * x for x in r))
    x, y, z = r[0] / n, r[1] / n, r[2] / n
    th = math.acos(max(-1.0, min(1.0, z)))
    ph = math.atan2(y, x)
    return np.array([math.cos(th / 2),
                     complex(math.cos(ph), math.sin(ph)) * math.sin(th / 2)],
                    dtype=complex)


_SKEW = np.array([math.cos(0.3),
                  complex(math.cos(1.1), math.sin(1.1)) * math.sin(0.3)],
                 dtype=complex)
_OCT = [_bloch(v) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                            (0, -1, 0), (0, 0, 1), (0, 0, -1))]
_TET = [_bloch(v) for v in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]
_Z0 = np.array([1, 0], dtype=complex)
F_, O_ = 'fidelity', 'observable_Z'
ENSEMBLES = [
    # --- the fidelity functional: three complex 2-designs -----------------
    ('haar (exact closed form)', '2-design', None, F_),
    ('tetrahedral SIC (4 states)', '2-design', (_TET, np.full(4, 0.25)), F_),
    ('octahedral (6 states)', '2-design', (_OCT, np.full(6, 1.0 / 6.0)), F_),
    # --- the fidelity functional: five ensembles that are NOT designs -----
    ('fixed |0_L>', 'not a 2-design', ([_Z0], np.ones(1)), F_),
    ('fixed |+_L>', 'not a 2-design', ([_bloch((1, 0, 0))], np.ones(1)), F_),
    ('fixed skewed state', 'not a 2-design', ([_SKEW], np.ones(1)), F_),
    ('two-state {|0_L>,|1_L>}', 'not a 2-design',
     ([_Z0, np.array([0, 1], dtype=complex)], np.full(2, 0.5)), F_),
    # the octahedron with deliberately UNEQUAL weights: same six states, so the
    # only thing broken is the design property itself
    ('biased octahedron (6 states)', 'not a 2-design',
     (_OCT, np.array([0.45, 0.05, 0.20, 0.10, 0.15, 0.05])), F_),
    # --- a DIFFERENT loss functional: what a hardware Pauli benchmark returns
    ('logical-<Z_L> on skewed state', 'different functional',
     ([_SKEW], np.ones(1)), O_),
    ('logical-<Z_L> on octahedron', 'different functional',
     (_OCT, np.full(6, 1.0 / 6.0)), O_),
]




def branch_A_from_KV(KV):
    """A[s][k] = W_s^dagger K_k V, using the production branch bases ab.W_BASIS."""
    Wd = W_np.conj().transpose(0, 2, 1)                # (16, 2, dim)
    KV = np.asarray(KV, dtype=complex)                 # (K, dim, 2)
    A = np.matmul(Wd[:, None, :, :], KV[None, :, :, :])   # (16, K, 2, 2)
    assert A.shape[0] == 16 and A.shape[2:] == (2, 2), A.shape
    assert A.shape[1] == KV.shape[0], (A.shape, KV.shape)
    return [list(A[s]) for s in range(16)]



def _G_of(phi):
    """G_s = V^dagger R(phi_s) W_s for all 16 branches, differentiable in phi."""
    RW = m.recovery_action_cols(phi.reshape(16, -1), W_t)     # (16, dim, 2)
    return torch.matmul(V_t.conj().T.unsqueeze(0), RW)        # (16, 2, 2)


def make_objective(A_t, ens, loss='fidelity'):
    """Exact label-free objective F(phi) for a branch-operator stack and ensemble.

    loss='fidelity'  the paper's objective.  `ens is None` selects the exact Haar
                     closed form; otherwise `ens` is (states, conj, weights) and
                     the average is that discrete ensemble.  Both are the same
                     degree-2 expression, so they agree to machine precision
                     exactly when the discrete ensemble is a complex 2-design.
    loss='observable_Z'
                     a HARDWARE-STYLE benchmark loss: the recovered logical
                     <Z_L> on the ensemble, instead of the state fidelity.  This
                     is what a device actually returns when it measures one Pauli
                     on a prepared state, and it is a different functional -- the
                     sum_k |<psi|G A_k|psi>|^2 structure that the fidelity has is
                     replaced by a single sesquilinear form.  It exists here to
                     locate the boundary of the stationarity result: if the
                     vanishing gradient were an artefact of the fidelity
                     functional rather than of the decoder point, this is where it
                     would show up."""
    Zt = torch.tensor(np.array([[1, 0], [0, -1]], dtype=complex),
                      dtype=torch.complex128)

    def F(phi):
        G = _G_of(phi)
        GA = torch.matmul(G.unsqueeze(1), A_t)                # (16, K, 2, 2)
        if loss == 'observable_Z':
            ps_t, _ps_c, pw_t = ens
            # GA.unsqueeze(2) broadcasts the Kraus index against the ensemble
            # index exactly as the fidelity branch does below; without it the
            # batch dims are (16,K) vs (1,nE) and any ensemble with nE != K
            # raises a size mismatch (it silently worked for the single-state
            # ensembles, which is why the nE=1 self-test passed).
            v = torch.matmul(GA.unsqueeze(2),
                             ps_t.view(1, 1, -1, 2, 1))        # (16,K,nE,2,1)
            q = torch.matmul(Zt, v)
            val = (v.conj() * q).real.sum(dim=(3, 4))          # (16, K, nE)
            # sum_k A^dag A = V^dag (sum_k K_k^dag K_k) V = I, so this unnormalised
            # sum over branches IS the instrument's expectation value, not a
            # post-selected ratio -- verified against Tr[Z_L P_code rec(rho)].
            return (val.sum(dim=(0, 1)) * pw_t).sum()
        if ens is None:
            tr = GA.diagonal(dim1=-2, dim2=-1).sum(-1)        # (16, K)
            fro = (GA.real ** 2 + GA.imag ** 2).sum((-2, -1))
            return ((tr.real ** 2 + tr.imag ** 2) + fro).sum() / 6.0
        ps_t, ps_c, pw_t = ens
        v = ps_t.view(1, 1, -1, 2, 1)
        u = ps_c.view(1, 1, -1, 1, 2)
        amp = torch.matmul(torch.matmul(u, GA.unsqueeze(2)), v).reshape(
            16, GA.shape[1], -1)                              # (16, K, nE)
        per_e = (amp.real ** 2 + amp.imag ** 2).sum(dim=(0, 1))
        return (per_e * pw_t).sum()
    return F



def prepare_ensemble(ens):
    if ens is None:
        return None
    states, w = ens
    ps = np.stack(states)
    assert np.abs(np.linalg.norm(ps, axis=1) - 1).max() < 1e-12
    assert abs(float(w.sum()) - 1.0) < 1e-12
    return (torch.tensor(ps, dtype=torch.complex128),
            torch.tensor(ps.conj(), dtype=torch.complex128),
            torch.tensor(np.asarray(w, dtype=float), dtype=torch.float64))


def grad_at(A_t, ens, phi, fd_angles=None, h=1e-6, loss='fidelity'):
    """(F(phi), max |dF/dphi| by autograd, max |dF/dphi| by central FD).

    `phi` is a (16, PHI_DIM) real tensor.  Both estimators are returned because
    they fail differently: autograd can silently propagate a detached subgraph
    (returning exactly 0), while a too-large FD step rounds a genuine zero into
    a nonzero value.  Agreement between them at ~1e-18 with FD exactly 0 is the
    signature of a true stationary point, not of a broken gradient.

    The maximum is over ALL 960 angles, i.e. over all 16 branches separately, so
    a small value establishes PER-BRANCH stationarity and rules out a
    cancellation between branches."""
    F = make_objective(A_t, ens, loss=loss)
    leaf = phi.detach().clone().requires_grad_(True)
    val = F(leaf)
    val.backward()
    g_auto = float(leaf.grad.abs().max())
    base = phi.detach().clone()
    g_fd = 0.0
    idx = range(base.numel()) if fd_angles is None else fd_angles
    with torch.no_grad():
        for i in idx:
            pp = base.clone().view(-1)
            pm = base.clone().view(-1)
            pp[i] += h
            pm[i] -= h
            g_fd = max(g_fd, abs(float(F(pp.reshape(16, -1)))
                                 - float(F(pm.reshape(16, -1)))) / (2 * h))
    return float(val), g_auto, g_fd


# A fixed CONTROL point, shared by every channel and ensemble: a large random
# angle table, far from phi^dec.  The gradient there must be O(1).  Without this
# control a vanishing gradient at phi^dec is indistinguishable from a broken or
# silently detached gradient, which is exactly the failure mode this study exists
# to rule out.
_CONTROL_PHI = torch.tensor(
    np.random.RandomState(20260915).normal(size=(16, m.PHI_DIM)) * 1.5,
    dtype=torch.float64)


def grad_control(A_t, ens, fd_angles=None, loss='fidelity'):
    """max |dF/dphi| at the shared random control point -- must be O(1)."""
    _, g_auto, g_fd = grad_at(A_t, ens, _CONTROL_PHI, fd_angles=fd_angles,
                              loss=loss)
    return g_auto, g_fd



def headroom(A_list):
    """(F_dec, F_unit, certified non-Pauli headroom) from the same solvers the
    paper uses, so a zero gradient can be classified as a minimum or a saddle."""
    F_dec = F_unit = 0.0
    for s in range(16):
        Q, const, ps = g.qform(A_list[s])
        if ps < 1e-14:
            continue
        Gd = V_np.conj().T @ m.C_SYNDS[s].numpy() @ W_np[s]
        gd = Gd.reshape(4)
        Jd = float(np.real(gd @ Q @ gd.conj()))
        single = A_list[s][0] if len(A_list[s]) == 1 else None
        Ju, _ = ab._max_unitary_J(Q, n_start=8, seed=0, single=single, G_dec=Gd)
        F_dec += (Jd + const) / 6.0
        F_unit += (Ju + const) / 6.0
    return F_dec, F_unit, F_unit - F_dec


# ---------------------------------------------------------------------------
# self-tests that anchor the objective to the audited production numbers
# ---------------------------------------------------------------------------
def _selftest_objective():
    """(a) the three 2-design ensembles must agree with the exact Haar closed
    form at an ARBITRARY angle table, not just at phi^dec -- this validates both
    the objective code and the ensembles; (b) F_haar(phi^dec) must reproduce the
    audited decoder fidelity in paper_numbers.json for all four paper channels."""
    P = json.load(open(os.path.join(ROOT, 'paper_numbers.json')))
    sdp = P['abl']['sdp']
    rs = np.random.RandomState(5)
    phi_rand = torch.tensor(rs.normal(size=(16, m.PHI_DIM)) * 1.5,
                            dtype=torch.float64)
    KV = kv_depolarizing()
    A_t = torch.tensor(np.stack([np.stack(x)
                                 for x in branch_A_from_KV(KV)]),
                       dtype=torch.complex128)
    Fh = make_objective(A_t, None)(phi_rand)
    devs = []
    for ename, eclass, ens, loss in ENSEMBLES:
        if loss != 'fidelity' or eclass != '2-design' or ens is None:
            continue
        Fe = make_objective(A_t, prepare_ensemble(ens))(phi_rand)
        devs.append(abs(float(Fe) - float(Fh)))
        print(f'    [2-design] {ename:26s} vs exact Haar at a random phi: '
              f'{float(Fe):.12f} vs {float(Fh):.12f}  diff {devs[-1]:.2e}',
              flush=True)
    assert max(devs) < 1e-12, devs

    # (a2) the observable_Z loss against a direct density-matrix evaluation:
    #      Tr[Z_L P_code sum_s R_s P_s rho P_s R_s^dag], with rho built from the
    #      same K_k V columns.  Independent of every reduced-branch identity.
    #      The codespace projector is REQUIRED: the reduced block G_s A_k is
    #      V^dag R_s P_s K_k V, which measures Z_L on the codespace component of
    #      R_s P_s K_k |psi_E>.  At phi^dec the decoder maps each syndrome
    #      subspace exactly into the codespace so the projector is redundant, but
    #      at a generic angle table R_s leaks out of it and the two differ -- the
    #      reduced form is the post-selected logical observable, which is what a
    #      hardware Z_L benchmark on a codespace readout returns.
    KV = kv_amplitude_damping()
    A_l = branch_A_from_KV(KV)
    A_o = torch.tensor(np.stack([np.stack(x) for x in A_l]),
                       dtype=torch.complex128)
    ZL = m.ZL.numpy()
    Pcode = m.P_code.numpy()
    Ps = [m.P_SYNDS[s].numpy() for s in range(16)]
    e_o = prepare_ensemble(([_SKEW], np.ones(1)))
    worst_o = worst_o_nop = 0.0
    for phi_try in (vp.decoder_angles(), phi_rand):
        got = float(make_objective(A_o, e_o, loss='observable_Z')(phi_try))
        col = KV @ _SKEW                                   # (K, dim)
        rho = np.einsum('ki,kj->ij', col, col.conj())
        R = m.recovery_unitary_batch(phi_try).detach().numpy()
        rec = sum(R[s] @ (Ps[s] @ rho @ Ps[s]) @ R[s].conj().T
                  for s in range(16))
        ref = float(np.real(np.trace(ZL @ Pcode @ rec)))
        ref_nop = float(np.real(np.trace(ZL @ rec)))
        worst_o = max(worst_o, abs(got - ref))
        worst_o_nop = max(worst_o_nop, abs(got - ref_nop))
    print(f'    [observ. ] logical-<Z_L> loss vs direct density-matrix trace '
          f'(codespace-projected): worst diff {worst_o:.2e}; '
          f'unprojected would differ by {worst_o_nop:.2e}', flush=True)
    assert worst_o < 1e-12, worst_o



    refs = {'depolarizing': 'dep_0.1', 'amplitude damping': 'ad_0.1',
            'mixed': 'mixed_0.1', 'coherent Rx': 'coh_0.15'}
    for cname, _cls, build in CHANNELS:
        if cname not in refs:
            continue
        A_list = branch_A_from_KV(build())
        At = torch.tensor(np.stack([np.stack(x) for x in A_list]),
                          dtype=torch.complex128)
        F0, _, _ = grad_at(At, None, vp.decoder_angles(),
                           fd_angles=range(0, 960, 240))
        ref = sdp[refs[cname]]['F_dec_exact']
        print(f'    [anchor ] {cname:20s} F_haar(phi^dec) {F0:.9f} vs audited '
              f'F_dec_exact {ref:.9f}  diff {abs(F0-ref):.2e}', flush=True)
        assert abs(F0 - ref) < 1e-9, (cname, F0, ref)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--quick', action='store_true',
                    help='central differences on 60 of the 960 angles')
    ap.add_argument('--no-selftest', action='store_true')
    ap.add_argument('--json', default='stationarity_boundary.json')
    a = ap.parse_args(argv)
    fd_angles = list(range(0, 960, 16)) if a.quick else None

    print(f'=== stationarity_boundary (pinned cpu {g.PINNED_CPU}, '
          f'quick={a.quick}) ===', flush=True)
    if not a.no_selftest:
        print('\n--- self-test: objective anchored to audited production values ---',
              flush=True)
        _selftest_objective()

    rows = []
    t0 = time.time()
    for cname, cclass, build in CHANNELS:
        A_list = branch_A_from_KV(build())
        A_t = torch.tensor(np.stack([np.stack(x) for x in A_list]),
                           dtype=torch.complex128)
        F_dec, F_unit, hr = headroom(A_list)
        print(f'\n[{cname}]  ({cclass}; {A_t.shape[1]} Kraus)')
        print(f'    F_dec {F_dec:.9f}   F_unit {F_unit:.9f}   '
              f'certified headroom {hr:+.3e}', flush=True)
        for ename, eclass, ens, loss in ENSEMBLES:
            e = prepare_ensemble(ens)
            F0, ga, gf = grad_at(A_t, e, vp.decoder_angles(), fd_angles,
                                 loss=loss)
            ca, cf = grad_control(A_t, e, fd_angles, loss=loss)
            assert ca > 1e-6, (
                f'CONTROL FAILED on {cname}/{ename}: the gradient at a random '
                f'angle table is {ca:.2e}, so this measurement cannot '
                f'distinguish a stationary point from a broken gradient')
            kind = ('minimum' if hr <= 1e-12 else 'SADDLE')
            rows.append(dict(channel=cname, channel_class=cclass,
                             ensemble=ename, ensemble_class=eclass, loss=loss,
                             F=F0, grad_autograd=ga, grad_fd=gf,
                             control_grad_autograd=ca, control_grad_fd=cf,
                             F_dec=F_dec, F_unit=F_unit, headroom=hr,
                             landscape=kind))
            print(f'    {ename:30s} [{eclass:22s}] F={F0:+.9f}  '
                  f'grad@dec autograd {ga:.2e}  FD {gf:.2e}  '
                  f'(control {ca:.2e}/{cf:.2e})  -> '
                  f'{"stationary" if ga < 1e-12 else "NOT stationary"} '
                  f'({kind})', flush=True)


    # the boundary, stated as a measurement rather than an attribution
    summary = dict(
        n_channels=len(CHANNELS), n_ensembles=len(ENSEMBLES),
        max_grad_at_decoder=max(max(r['grad_autograd'], r['grad_fd'])
                                for r in rows),
        min_control_grad=min(r['control_grad_autograd'] for r in rows),
        n_stationary_2designs=sum(1 for r in rows
                                  if r['ensemble_class'] == '2-design'
                                  and r['grad_autograd'] < 1e-12),
        n_2designs=sum(1 for r in rows if r['ensemble_class'] == '2-design'),
        n_stationary_non_designs=sum(1 for r in rows
                                     if r['ensemble_class'] != '2-design'
                                     and r['grad_autograd'] < 1e-12),
        n_non_designs=sum(1 for r in rows
                          if r['ensemble_class'] != '2-design'),
        runtime_s=time.time() - t0)

    print('\n================ SUMMARY ================')
    for k, v in summary.items():
        print(f'  {k:32s} {v}')
    clean = (summary['max_grad_at_decoder'] < 1e-12
             and summary['min_control_grad'] > 1e-4)
    print(f'\n  every channel x ensemble pair is stationary at phi^dec while '
          f'the control gradient stays O(1): {clean}')
    print(f'  stationary on non-2-design ensembles too: '
          f'{summary["n_stationary_non_designs"]}/{summary["n_non_designs"]}')
    out = dict(generated_utc=time.strftime('%Y-%m-%dT%H:%M:%S'),
               pinned_cpu=g.PINNED_CPU, quick=a.quick, rows=rows,
               summary=summary, stationary_on_all_tested=clean)

    with open(os.path.join(ROOT, a.json), 'w') as f:
        json.dump(out, f, indent=1)
    print(f'\nwrote {a.json}  ({summary["runtime_s"]:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

