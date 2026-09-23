#!/usr/bin/env python3
"""mixed_input_boundary.py -- the mixed-state edge of the decoder-stationarity theorem.

THE HOLE THIS CLOSES.  Methods proves that the Pauli decoder point phi^dec is an
exact stationary point of the label-free objective for EVERY ensemble of logical
inputs, via nine l = 0,1,2 sector gradients that are ensemble-independent
(stationarity_boundary.py).  The proof has one structural assumption that is easy
to miss and that a device does not satisfy: "ensemble" means a distribution over
PURE logical states, because the orthogonality step uses |n_i| = 1.  A real
processor delivers a MIXED logical input -- noisy preparation, idle decay and
leakage back into the codespace all shrink the Bloch vector -- so the object that
actually gets scored on hardware sits outside the theorem as stated.  This script
measures exactly how far outside, and finds that the answer is: not at all, for a
reason that is algebra plus one extra measured zero.

THE ALGEBRA.  For rho_n = (I + n.sigma)/2 with |n| = r <= 1 -- pure iff r = 1 --
nothing in the derivation of stationarity_boundary.py used purity, so

    Tr[M rho_n] = a + b.n,      sum_sk |Tr[M_sk rho_n]|^2 = c0 + c1.n + n^T C n

holds for every r.  The mixed-input score is therefore the SAME quadratic
polynomial in the Bloch vector with the constraint |n| = 1 dropped, and writing
grad C = grad C0 + (Tr grad C / 3) I with S0 = grad[c0 + Tr C / 3],
S1 = grad c1, S2 = grad C0,

    grad F_mix(n) = grad c0 + (grad c1).n + n^T (grad C) n
                  = S0 + S1.n + n^T S2 n - (1 - r^2) (S0 - grad c0),          (*)

an explicit linear combination of the nine sector gradients plus ONE extra
coefficient gradient, grad c0, that the pure-state orthogonality argument never
had to control (it only ever needed the sum S0 = grad c0 + Tr grad C / 3).  So
mixed inputs are stationary at phi^dec whenever all ten vanish, and all ten are
measured here: max|grad c0| and max|grad F_mix| over a radius grid r in [0,1],
four Bloch directions and all nine channels, against the usual O(1) random-angle
control.  (*) itself is validated where it is NONZERO -- at random angle tables,
where both sides are ~1e-2 -- because an identity checked only at a common zero
proves nothing.

TWO FURTHER FACTS, ONE OF WHICH NEEDS NO ALGEBRA AT ALL.

  * A LINEAR score cannot distinguish a mixed input from a pure ensemble with the
    same Bloch vector.  Both hardware benchmarks are linear in rho: the branch
    population F_s = P(data = 00000 | s) and the logical-<Z_L> observable.  For
    any decomposition rho = sum_i w_i |psi_i><psi_i|, F_lin(rho) = sum_i w_i
    F_lin(psi_i) EXACTLY, so the pure-ensemble certificate covers them with no
    extension.  Verified numerically against an explicit two-state decomposition,
    and the mixed-input gradient of the <Z_L> functional is measured directly.

  * The QUADRATIC training loss can distinguish them, and by an exactly
    computable amount.  For the two-state decomposition n_+/- = n +/- delta with
    delta orthogonal to n and |delta|^2 = 1 - r^2,

        1/2 [F(n_+) + F(n_-)] - F(n) = delta^T C delta >= 0,

    so a pure-ensemble score OVER-READS a genuinely mixed input by exactly that
    amount (C is a Gram matrix, hence PSD, so the sign is not a convention).
    Averaged over directions at fixed radius the mixed score is
    c0 + (r^2/3) Tr C, so the over-read against the Haar number is
    (1 - r^2) (F_haar - c0).  This is a benchmark-comparability statement, not a
    gradient statement: mixedness moves the VALUE the device reports, while (*)
    says it cannot move the ARGMIN at first order.

DEVICE RELEVANCE, from the run's own numbers.  hw_error_budget.json reports the
implied mean per-instruction error of the executed circuit; feeding that rate
through the compiled 36-gate encoder as a depolarizing channel after every
instruction gives the Bloch radius r of the logical state that actually reaches
the recovery, per logical input state, together with its codespace weight.  That
r is then used to price the over-read above, so the hardware implication is
quoted at a mixedness the run itself implies rather than at a chosen one.

SECOND ORDER.  Stationarity is not optimality, so the script also asks whether a
mixed-input score can be improved away from phi^dec at all: multi-start Adam on
the isotropic mixed score, reporting the best gain over the decoder point and
what that gain costs on the audited pure-ensemble (Haar) score.  Read the gains as
an existence statement, not as reproducible numbers: this is a non-convex search,
and 400 Adam steps amplify the last bits of whichever BLAS kernel the pinned core
selects, so WHICH start wins on a given channel moves between hosts (two full runs
agreed bit for bit on amplitude damping and differed by three orders of magnitude
on the coherent channel).  What is reproducible, and what paper/audit_numbers.py
locks, is the invariant set: the warm-start gradient is zero to round-off, no start
ever does worse than the decoder point, depolarizing (where the decoder is the
certified optimum) does not move at all, and the largest gain is of the order of
the certified headroom rather than of the fidelity.

Usage:
    ./qenv/bin/python mixed_input_boundary.py
    ./qenv/bin/python mixed_input_boundary.py --quick      # fewer FD angles/starts
    ./qenv/bin/python mixed_input_boundary.py --json mixed_input_boundary.json
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
# Thread/coretype pins BEFORE numpy -- see the identical note in
# stationarity_boundary.py: importing numpy first would initialise OpenBLAS
# multi-threaded and make these no-ops, which changes the last bits of every
# number below.
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
import stationarity_boundary as sb                              # noqa: E402

CODE = g.StabCode(g.CODES['5,1,3'])
V_np = ab.V_ISO                                                 # (32, 2)
I2 = np.eye(2, dtype=complex)
PAULI = [np.asarray(g._MAT[v], dtype=complex) for v in (g.X_, g.Y_, g.Z_)]

# radius grid: r = 1 is the pure-state theorem, r = 0 the maximally mixed logical
# input, and the values in between are what a device delivers.
RADII = (1.0, 0.95, 0.9, 0.8, 0.6, 0.3, 0.0)
_dirs = {
    'z': np.array([0.0, 0.0, 1.0]),
    'x': np.array([1.0, 0.0, 0.0]),
    'skew': np.array([0.4, 0.5, 0.76811]),
    'random': np.array([-0.31, 0.62, 0.72]),
}
DIRS = {k: v / np.linalg.norm(v) for k, v in _dirs.items()}
# the four channels the paper reports headline fidelities for
PAPER4 = ('depolarizing', 'amplitude damping', 'mixed', 'coherent Rx')


def A_of(build):
    """(16, K, 2, 2) reduced branch blocks, identical to stationarity_boundary."""
    return torch.tensor(np.stack([np.stack(x)
                                  for x in sb.branch_A_from_KV(build())]),
                        dtype=torch.complex128)


def F_mix_coeffs(A_t, phi, n):
    """The mixed-input score c0 + c1.n + n^T C n, differentiable in phi.

    Valid for EVERY |n| <= 1; |n| = 1 is the pure-state case the theorem covers."""
    c0, c1, C = sb.harmonic_coeffs(A_t, phi)
    nt = torch.tensor(np.asarray(n, dtype=float), dtype=torch.float64)
    return c0 + (c1 * nt).sum() + nt @ C @ nt


def F_iso_coeffs(A_t, phi, r):
    """Direction-averaged mixed score c0 + (r^2/3) Tr C: the isotropic mixed input.

    At r = 1 this is exactly the Haar objective c0 + Tr C / 3, which is what makes
    it the right object to descend on -- it interpolates between the audited
    pure-ensemble score and the maximally mixed one without picking a direction."""
    c0, c1, C = sb.harmonic_coeffs(A_t, phi)
    return c0 + (r ** 2) / 3.0 * torch.diagonal(C).sum()


def grad_c0(A_t, phi):
    """(grad_phi c0, grad_phi Tr C / 3) -- the two halves of the l = 0 sector.

    The pure-state proof only ever needed their SUM, S0.  A mixed input needs each
    of them separately, which is why this is measured rather than assumed."""
    leaf = phi.detach().clone().requires_grad_(True)
    c0, c1, C = sb.harmonic_coeffs(A_t, leaf)
    gj, = torch.autograd.grad(c0, leaf, retain_graph=True, allow_unused=True)
    gt, = torch.autograd.grad(torch.diagonal(C).sum() / 3.0, leaf,
                              allow_unused=True)
    z = torch.zeros_like(leaf).reshape(-1)
    return (z if gj is None else gj.reshape(-1)), (z if gt is None
                                                   else gt.reshape(-1))


def grad_mixed(A_t, phi, n=None, fd_angles=(), h=1e-6, iso_r=None):
    """(F, max|dF/dphi| autograd, max|dF/dphi| central FD) for a mixed input.

    `iso_r` scores the direction-averaged mixed input instead of the fixed n.  Both
    estimators are returned for the same reason stationarity_boundary returns both:
    autograd can silently propagate a detached subgraph (returning exactly 0), and
    an FD step that is too large rounds a genuine zero into a nonzero value.

    `fd_angles` is the iterable of angle indices to difference; EMPTY (the default)
    skips the FD pass and returns None for it, because a full 960-angle FD costs
    ~4 s per call and the radius x direction scan only needs it as a cross-check on
    a subset."""
    leaf = phi.detach().clone().requires_grad_(True)
    F = (F_iso_coeffs(A_t, leaf, iso_r) if iso_r is not None
         else F_mix_coeffs(A_t, leaf, n))
    F.backward()
    g_auto = float(leaf.grad.abs().max())
    val = float(F.detach())
    idx = list(fd_angles)
    if not idx:
        return val, g_auto, None
    base = phi.detach().clone()
    g_fd = 0.0
    nt = (None if n is None
          else torch.tensor(np.asarray(n, dtype=float), dtype=torch.float64))
    with torch.no_grad():
        for i in idx:
            pp, pm = base.clone().view(-1), base.clone().view(-1)
            pp[i] += h
            pm[i] -= h
            c0p, c1p, Cp = sb.harmonic_coeffs(A_t, pp.reshape(16, -1))
            c0m, c1m, Cm = sb.harmonic_coeffs(A_t, pm.reshape(16, -1))
            if iso_r is None:
                fp = c0p + (c1p * nt).sum() + nt @ Cp @ nt
                fm = c0m + (c1m * nt).sum() + nt @ Cm @ nt
            else:
                fp = c0p + (iso_r ** 2) / 3.0 * torch.diagonal(Cp).sum()
                fm = c0m + (iso_r ** 2) / 3.0 * torch.diagonal(Cm).sum()
            g_fd = max(g_fd, abs(float(fp - fm)) / (2 * h))
    return val, g_auto, g_fd


def _S2_quad(S2, n):
    """n^T S2 n from the five packed components of the traceless l = 2 block.

    The packing of stationarity_boundary._pack is
    (C0[0,0], C0[1,1], C0[0,1], C0[0,2], C0[1,2]), and tracelessness forces
    C0[2,2] = -C0[0,0] - C0[1,1], so
    n^T C0 n = a(nx^2-nz^2) + b(ny^2-nz^2) + 2c nx ny + 2d nx nz + 2e ny nz."""
    a, b, c, d, e = [S2[k].reshape(-1) for k in range(5)]
    nx, ny, nz = [float(x) for x in n]
    return (a * (nx * nx - nz * nz) + b * (ny * ny - nz * nz)
            + 2 * c * nx * ny + 2 * d * nx * nz + 2 * e * ny * nz)


def identity_residual(A_t, phi, n, verbose=False):
    """Validate (*) where it is NONZERO: grad F_mix vs the sector combination.

    Returns (max abs residual, |lhs|, |rhs|).  Called at random angle tables, where
    all three quantities are O(1e-2), so a residual at machine epsilon is evidence
    about the algebra rather than about a shared zero."""
    r = float(np.linalg.norm(n))
    leaf = phi.detach().clone().requires_grad_(True)
    lhs, = torch.autograd.grad(F_mix_coeffs(A_t, leaf, n), leaf)
    lhs = lhs.reshape(-1)
    S0, S1, S2 = sb.sector_grads(A_t, phi)
    g0, _ = grad_c0(A_t, phi)
    S0 = S0.reshape(-1)
    rhs = (S0
           + sum(float(n[j]) * S1[j].reshape(-1) for j in range(3))
           + _S2_quad(S2, n)
           - (1.0 - r * r) * (S0 - g0))
    res = float((lhs - rhs).abs().max())
    if verbose:
        print(f'      |lhs| {float(lhs.abs().max()):.3e}  '
              f'|rhs| {float(rhs.abs().max()):.3e}  residual {res:.3e}',
              flush=True)
    return res, float(lhs.abs().max()), float(rhs.abs().max())




# audited anchors: the same four channels stationarity_boundary anchors to, with
# the same keys, so a drift in either script fails both.
ANCHOR = {'depolarizing': 'dep_0.1', 'amplitude damping': 'ad_0.1',
          'mixed': 'mixed_0.1', 'coherent Rx': 'coh_0.15'}


def _decompose(n):
    """(n_plus, n_minus, delta): an explicit two-pure-state decomposition of the
    mixed input with Bloch vector n.  delta is orthogonal to n with
    |delta|^2 = 1 - r^2, so n_+/- are unit vectors that average back to n."""
    n = np.asarray(n, dtype=float)
    r = float(np.linalg.norm(n))
    assert r <= 1.0 + 1e-12, r
    if r < 1e-12:
        d = np.array([1.0, 0.0, 0.0])
    else:
        t = (np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 * max(r, 1e-30)
             else np.array([0.0, 1.0, 0.0]))
        d = t - n * (np.dot(t, n) / np.dot(n, n))
        d /= np.linalg.norm(d)
    delta = d * math.sqrt(max(0.0, 1.0 - r * r))
    return n + delta, n - delta, delta


def selftests(A_dep, phi_dec):
    """Anchor every new object to something already audited before it is used.

    (a) the isotropic mixed score at r = 1 IS the Haar objective, so it must
        reproduce the audited exact decoder fidelity in paper_numbers.json;
    (b) for a genuinely MIXED rho the expansion c0 + c1.n + n^T C n must equal a
        graph-free density-matrix trace sum_sk |Tr[M_sk rho]|^2 -- this is the step
        that extends stationarity_boundary's pure-state check to |n| < 1, and
        everything below rests on it;
    (c) C is a Gram matrix, hence PSD, which is what fixes the SIGN of the
        mixed-vs-pure over-read;
    (d) a LINEAR score is exactly decomposable over pure states (so the hardware
        population and <Z_L> benchmarks need no extension), while the QUADRATIC
        training loss is not, and its discrepancy is exactly delta^T C delta;
    (e) identity (*) holds at random angle tables, where both sides are O(1e-2).
    """
    P = json.load(open(os.path.join(ROOT, 'paper_numbers.json')))
    sdp = P['abl']['sdp']
    builds = dict((c, b) for c, _cl, b in sb.CHANNELS)
    print('  [selftest] anchoring the mixed-input score to audited numbers',
          flush=True)
    for cname, ref in ANCHOR.items():
        A_t = A_of(builds[cname])
        r_haar = float(F_iso_coeffs(A_t, phi_dec, 1.0).detach())
        got = float(sdp[ref]['F_dec_exact'])
        assert abs(r_haar - got) < 1e-9, (cname, r_haar, got)
        with torch.no_grad():
            _, _, C = sb.harmonic_coeffs(A_t, phi_dec)
            ev = np.linalg.eigvalsh(C.numpy())
        assert ev.min() > -1e-15, (cname, ev)
        print(f'    {cname:20s} F_iso(r=1) = {r_haar:.9f} == audited '
              f'F_dec_exact {got:.9f}   min eig C = {ev.min():.1e} >= 0',
              flush=True)

    # (b) mixed rho against a graph-free density-matrix trace
    rs = np.random.RandomState(20260919)
    worst_dm = worst_r = 0.0
    for cname in ('amplitude damping', 'random CPTP'):
        A_l = sb.branch_A_from_KV(builds[cname]())
        A_t = torch.tensor(np.stack([np.stack(x) for x in A_l]),
                           dtype=torch.complex128)
        G_np = sb._G_of(phi_dec).detach().numpy()
        M_np = np.matmul(G_np[:, None], np.stack([np.stack(x) for x in A_l]))
        for _ in range(30):
            r = float(rs.uniform(0.0, 1.0))
            d = rs.normal(size=3)
            n = r * d / np.linalg.norm(d)
            rho = 0.5 * (I2 + sum(n[j] * sb._SIG[j] for j in range(3)))
            evr = np.linalg.eigvalsh(rho)
            assert evr.min() > -1e-14 and abs(np.trace(rho).real - 1) < 1e-14, evr
            worst_r = max(worst_r, float(np.linalg.norm(n)))
            Fdm = float(sum(abs(np.trace(M_np[s, k] @ rho)) ** 2
                            for s in range(16) for k in range(M_np.shape[1])))
            Fex = float(F_mix_coeffs(A_t, phi_dec, n).detach())
            worst_dm = max(worst_dm, abs(Fdm - Fex))
    assert worst_dm < 1e-12, worst_dm     # same tolerance the pure-state check in
    # stationarity_boundary._selftest_harmonic uses for the identical comparison
    print(f'    mixed density-matrix trace vs the expansion over 60 states with '
          f'r <= {worst_r:.3f}: worst diff {worst_dm:.2e}', flush=True)


    # (d) linear vs quadratic under an explicit two-state decomposition
    worst_lin = worst_quad = max_over = 0.0
    for r in (0.95, 0.8, 0.5, 0.0):
        n = r * DIRS['skew']
        n_p, n_m, delta = _decompose(n)
        assert abs(np.linalg.norm(n_p) - 1) < 1e-14
        assert abs(np.linalg.norm(n_m) - 1) < 1e-14
        assert abs(0.5 * (n_p + n_m) - n).max() < 1e-14
        with torch.no_grad():
            z0, z1 = sb.observable_coeffs(A_dep, phi_dec)
            _, _, C = sb.harmonic_coeffs(A_dep, phi_dec)
        z1n = z1.numpy()
        lin_mix = float(z0) + float(z1n @ n)
        lin_ens = (0.5 * (float(z0) + float(z1n @ n_p))
                   + 0.5 * (float(z0) + float(z1n @ n_m)))
        worst_lin = max(worst_lin, abs(lin_mix - lin_ens))
        q_mix = float(F_mix_coeffs(A_dep, phi_dec, n).detach())
        q_ens = (0.5 * float(F_mix_coeffs(A_dep, phi_dec, n_p).detach())
                 + 0.5 * float(F_mix_coeffs(A_dep, phi_dec, n_m).detach()))
        pred = float(delta @ C.numpy() @ delta)
        assert pred >= -1e-18, pred
        max_over = max(max_over, pred)
        worst_quad = max(worst_quad, abs((q_ens - q_mix) - pred))
    assert worst_lin < 1e-14, worst_lin
    assert worst_quad < 1e-14, worst_quad
    print(f'    linear score over a two-state decomposition: residual '
          f'{worst_lin:.2e}  (EXACT: hardware populations and <Z_L> need no '
          f'extension)', flush=True)
    print(f'    quadratic score vs delta^T C delta: residual {worst_quad:.2e}, '
          f'largest over-read {max_over:.3e} >= 0', flush=True)

    # (e) the algebra of (*), tested where it is nonzero
    worst_res, worst_mag = 0.0, None
    for cname, _cl, build in sb.CHANNELS:
        A_c = A_of(build)
        for seed in range(2):
            phi_r = torch.tensor(
                np.random.RandomState(1000 + seed).normal(size=(16, m.PHI_DIM))
                * 1.5, dtype=torch.float64)
            for r in (1.0, 0.8, 0.0):
                res, lmag, rmag = identity_residual(A_c, phi_r, r * DIRS['random'])
                worst_res = max(worst_res, res)
                worst_mag = (min(lmag, rmag) if worst_mag is None
                             else min(worst_mag, lmag, rmag))
    assert worst_res < 1e-12, worst_res
    assert worst_mag > 1e-4, worst_mag
    print(f'    identity (*) at random angle tables: worst residual '
          f'{worst_res:.2e}, both sides >= {worst_mag:.2e} (so the check is not '
          f'vacuous at a shared zero)', flush=True)
    return dict(linear_decomp_residual=worst_lin,
                quadratic_delta_residual=worst_quad,
                quadratic_max_overread=max_over,
                mixed_density_matrix_residual=worst_dm,
                identity_residual=worst_res,
                identity_min_magnitude=worst_mag)


def grad_obs_mixed(A_t, phi, n):
    """max|grad| of the LINEAR logical-<Z_L> score z0 + z1.n at a mixed input.

    Included because linearity already covers mixed inputs exactly; measuring it is
    a consistency check on that argument rather than a new claim."""
    leaf = phi.detach().clone().requires_grad_(True)
    z0, z1 = sb.observable_coeffs(A_t, leaf)
    nt = torch.tensor(np.asarray(n, dtype=float), dtype=torch.float64)
    (z0 + (z1 * nt).sum()).backward()
    return float(leaf.grad.abs().max())


def mixed_scan(phi_dec, fd_angles, verbose=True):
    """Per channel: the extra l = 0 coefficient grad c0, the mixed-input gradient
    over the whole radius grid and four directions, a central-FD cross-check of the
    same, the linear-score gradient, the random-angle controls, and the coefficient
    VALUES (c0, Tr C / 3) that price the mixed-vs-pure over-read at any radius."""
    rows = []
    for cname, cclass, build in sb.CHANNELS:
        A_t = A_of(build)
        g0, gtr = grad_c0(A_t, phi_dec)
        g0c, gtrc = grad_c0(A_t, sb._CONTROL_PHI)
        worst_mix = worst_fd = 0.0
        worst_ctrl = 0.0
        per_r = {}
        for r in RADII:
            worst_r = 0.0
            for dname, d in DIRS.items():
                n = r * d
                _, ga, _ = grad_mixed(A_t, phi_dec, n=n)
                worst_r = max(worst_r, ga)
                if r in (0.8, 0.0):
                    _, _, gf = grad_mixed(A_t, phi_dec, n=n, fd_angles=fd_angles)
                    worst_fd = max(worst_fd, gf if gf is not None else 0.0)
                _, ca, _ = grad_mixed(A_t, sb._CONTROL_PHI, n=n)
                worst_ctrl = max(worst_ctrl, ca)
            per_r['%.2f' % r] = worst_r
            worst_mix = max(worst_mix, worst_r)
        # isotropic (direction-averaged) mixed score, same treatment
        iso = {}
        for r in RADII:
            _, ga, gf = grad_mixed(A_t, phi_dec, iso_r=r, fd_angles=fd_angles)
            iso['%.2f' % r] = dict(grad_autograd=ga, grad_fd=gf)
            worst_mix = max(worst_mix, ga)
            worst_fd = max(worst_fd, gf if gf is not None else 0.0)
        z_mix = grad_obs_mixed(A_t, phi_dec, 0.8 * DIRS['skew'])
        with torch.no_grad():
            c0, c1, C = sb.harmonic_coeffs(A_t, phi_dec)
            c0v = float(c0)
            trc3 = float(torch.diagonal(C).sum()) / 3.0
        rec = dict(channel=cname, channel_class=cclass,
                   max_grad_c0=float(g0.abs().max()),
                   max_grad_trC3=float(gtr.abs().max()),
                   max_grad_mixed=worst_mix, max_grad_mixed_fd=worst_fd,
                   control_grad_c0=float(g0c.abs().max()),
                   control_grad_trC3=float(gtrc.abs().max()),
                   control_grad_mixed=worst_ctrl,
                   max_grad_Z_mixed=z_mix,
                   c0=c0v, tr_C_over_3=trc3, F_haar=c0v + trc3,
                   per_radius=per_r, isotropic=iso)
        rows.append(rec)
        if verbose:
            print(f'    {cname:22s} grad c0 {rec["max_grad_c0"]:.2e}  '
                  f'mixed {worst_mix:.2e} (FD {worst_fd:.2e})  '
                  f'<Z_L> {z_mix:.2e}  ctrl {worst_ctrl:.2e}  '
                  f'c0 {c0v:.6f} TrC/3 {trc3:.6f}', flush=True)
    return rows


def device_prep(p_instr, n_gates_expect=36, verbose=True):
    """Bloch radius of the logical state the DEVICE delivers, from the run's own
    implied instruction error.

    Model, stated exactly so it can be argued with: the compiled Clifford encoder
    (the same 36-gate list the hardware circuits are built from), with a
    depolarizing channel of rate `p_instr` applied to every qubit an instruction
    touches, immediately after that instruction.  `p_instr` is read from
    hw_error_budget.json -- the mean per-instruction error implied by the executed
    circuit's measured loss -- so this is an estimate built on the run's own data
    and not a vendor calibration, and it is quoted as an order of magnitude.  Only
    the encoder section is treated as PREPARATION: everything downstream (frame
    injection, extraction, idle, recovery) is the CHANNEL in the paper's formalism,
    not part of the input state.

    The logical state is rho_L = V^dagger rho V, reported as its codespace weight
    and, after normalisation, its Bloch radius r and purity (1 + r^2)/2."""
    import qcloud_vscr_new as q
    gates = q.synthesize_encoder()
    assert len(gates) == n_gates_expect, (len(gates), n_gates_expect)

    def _kraus(p):
        return ([math.sqrt(1.0 - p) * I2]
                + [math.sqrt(p / 3.0) * s for s in PAULI])

    out = {}
    for label, b in (('0_L', 0), ('1_L', 1)):
        psi = np.zeros(CODE.dim, dtype=complex)
        psi[b] = 1.0
        rho = np.outer(psi, psi.conj())
        for gt in gates:
            if gt[0] == 'CNOT':
                U = q.embed2(q.CNOT4, gt[1], gt[2], n=5)
                qs = [gt[1], gt[2]]
            else:
                U = q.embed1(q._gate_matrix(gt), gt[1], n=5)
                qs = [gt[1]]
            rho = U @ rho @ U.conj().T
            for qb in qs:
                K = [q.embed1(k, qb, n=5) for k in _kraus(p_instr)]
                rho = sum(k @ rho @ k.conj().T for k in K)
        rhoL = V_np.conj().T @ rho @ V_np
        w = float(np.trace(rhoL).real)
        assert 0.0 < w <= 1.0 + 1e-12, w
        rl = rhoL / w
        nvec = np.array([float(np.trace(Pm @ rl).real) for Pm in PAULI])
        r = float(np.linalg.norm(nvec))
        purity = float(np.trace(rl @ rl).real)
        assert abs(purity - (1.0 + r * r) / 2.0) < 1e-12, (purity, r)
        ev = np.linalg.eigvalsh(rl)
        assert ev.min() > -1e-12 and abs(ev.sum() - 1) < 1e-12, ev
        assert r <= 1.0 + 1e-12, r
        out[label] = dict(radius=r, codespace_weight=w, purity=purity,
                          leakage=1.0 - w, bloch=nvec.tolist())
        if verbose:
            print(f'    |{label}>  r = {r:.6f}  purity = {purity:.6f}  '
                  f'codespace weight = {w:.6f}  (leakage {1 - w:.6f})',
                  flush=True)
    radii = [v['radius'] for v in out.values() if isinstance(v, dict)]
    out['p_instr'] = float(p_instr)
    out['n_encoder_instructions'] = len(gates)
    out['radius_max'] = max(radii)
    out['radius_min'] = min(radii)
    return out


def descent(phi_dec, r, n_starts=8, steps=400, lr=0.05, seed=20260919,
            verbose=True):
    """Can a MIXED-input score be improved away from phi^dec at all?

    Stationarity is not optimality, so this asks the second-order question with
    multi-start Adam on the isotropic mixed score c0 + (r^2/3) Tr C -- which at
    r = 1 is exactly the audited Haar objective, so the same code path reproduces
    the paper's warm-start behaviour there.  Start 0 is phi^dec itself (the warm
    start, where the gradient is exactly zero); the rest are symmetry-broken.
    Reported per channel: the warm-start gradient, the gain of the warm start, the
    best gain over all starts, what that best point costs on the pure-ensemble
    (Haar) score, and how far its angles move."""
    builds = dict((c, b) for c, _cl, b in sb.CHANNELS)
    out = {}
    for cname in PAPER4:
        A_t = A_of(builds[cname])
        base = float(F_iso_coeffs(A_t, phi_dec, r).detach())
        base_haar = float(F_iso_coeffs(A_t, phi_dec, 1.0).detach())
        best = dict(F=base, phi=phi_dec.clone(), start='decoder')
        warm_F = None
        for k in range(n_starts):
            if k == 0:
                phi = phi_dec.clone()
            else:
                rs = np.random.RandomState(seed + 100 * k)
                phi = phi_dec + torch.tensor(
                    rs.normal(size=tuple(phi_dec.shape)) * 1.5,
                    dtype=torch.float64)
            phi = phi.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([phi], lr=lr)
            for _ in range(steps):
                opt.zero_grad()
                (-F_iso_coeffs(A_t, phi, r)).backward()
                opt.step()
            Fk = float(F_iso_coeffs(A_t, phi.detach(), r).detach())
            if k == 0:
                warm_F = Fk
            if Fk > best['F']:
                best = dict(F=Fk, phi=phi.detach().clone(), start=k)
        phi_b = best['phi']
        haar_b = float(F_iso_coeffs(A_t, phi_b, 1.0).detach())
        leaf = phi_dec.detach().clone().requires_grad_(True)
        (-F_iso_coeffs(A_t, leaf, r)).backward()
        rec = dict(channel=cname, r=r, F_decoder=base, F_best=best['F'],
                   gain=best['F'] - base, best_start=best['start'],
                   warm_gain=warm_F - base,
                   F_haar_decoder=base_haar, F_haar_best=haar_b,
                   haar_cost=base_haar - haar_b,
                   max_angle_move=float((phi_b - phi_dec).abs().max()),
                   warm_start_grad=float(leaf.grad.abs().max()),
                   n_starts=n_starts, steps=steps, lr=lr, seed=seed)
        out[cname] = rec
        if verbose:
            print(f'    {cname:20s} r={r:.2f}  warm-start grad '
                  f'{rec["warm_start_grad"]:.2e} (gain {rec["warm_gain"]:.1e})  '
                  f'best gain {rec["gain"]:.3e} (start {best["start"]})  '
                  f'Haar cost {rec["haar_cost"]:.3e}  '
                  f'move {rec["max_angle_move"]:.3f}', flush=True)
    return out



def main():
    ap = argparse.ArgumentParser(
        description='the mixed-state edge of the decoder-stationarity theorem')
    ap.add_argument('--json', default='mixed_input_boundary.json',
                    help='artifact path (default: mixed_input_boundary.json)')
    ap.add_argument('--quick', action='store_true',
                    help='central FD on 24 angles and 3 descent starts')
    ap.add_argument('--no-descent', action='store_true',
                    help='skip the second-order multi-start descent')
    ap.add_argument('--budget', default='hw_error_budget.json',
                    help='source of the implied per-instruction error')
    a = ap.parse_args()
    t0 = time.time()
    phi_dec = vp.decoder_angles()
    builds = dict((c, b) for c, _cl, b in sb.CHANNELS)
    fd_angles = range(0, 960, 40) if a.quick else range(0, 960, 10)

    print('=== self-tests: anchor the mixed-input objects before using them ===',
          flush=True)
    st = selftests(A_of(builds['depolarizing']), phi_dec)

    print('\n=== mixed-input gradient scan: nine channels, r in [0, 1] ===',
          flush=True)
    rows = mixed_scan(phi_dec, fd_angles)

    print('\n=== device preparation: how mixed is the logical input on the run '
          '===', flush=True)
    hb = json.load(open(os.path.join(ROOT, a.budget)))
    p_instr = float(hb['headline']['implied_mean_instruction_error_D'])
    print(f'    implied mean per-instruction error from {a.budget}: '
          f'{p_instr:.6f}', flush=True)
    dev = device_prep(p_instr)
    r_dev = dev['radius_min']           # the more mixed of the two logical states

    print('\n=== the over-read a pure-ensemble benchmark carries at that r ===',
          flush=True)
    over = {}
    for rec in rows:
        val = (1.0 - r_dev ** 2) * (rec['F_haar'] - rec['c0'])
        over[rec['channel']] = dict(F_haar=rec['F_haar'], c0=rec['c0'],
                                    overread=val, r=r_dev)
        if rec['channel'] in PAPER4:
            print(f'    {rec["channel"]:20s} F_haar {rec["F_haar"]:.9f}  '
                  f'c0 {rec["c0"]:.9f}  over-read at r={r_dev:.4f}: '
                  f'{val:.3e}', flush=True)

    desc = {}
    if not a.no_descent:
        print('\n=== second order: multi-start descent on the mixed score ===',
              flush=True)
        desc = descent(phi_dec, r_dev, n_starts=3 if a.quick else 8,
                       steps=200 if a.quick else 400)

    max_mix = max(r['max_grad_mixed'] for r in rows)
    max_fd = max(r['max_grad_mixed_fd'] for r in rows)
    max_c0 = max(r['max_grad_c0'] for r in rows)
    min_ctrl = min(r['control_grad_mixed'] for r in rows)
    max_z = max(r['max_grad_Z_mixed'] for r in rows)
    max_gain = max((v['gain'] for v in desc.values()), default=None)
    max_warm_grad = max((v['warm_start_grad'] for v in desc.values()),
                        default=None)
    summary = dict(
        n_channels=len(rows), n_radii=len(RADII), n_dirs=len(DIRS),
        n_scores=len(rows) * len(RADII) * (len(DIRS) + 1),
        n_angles=int(phi_dec.numel()), n_fd_angles=len(list(fd_angles)),
        max_grad_c0=max_c0, max_grad_mixed=max_mix, max_grad_mixed_fd=max_fd,
        min_control_grad_mixed=min_ctrl, max_grad_Z_mixed=max_z,
        device_p_instr=p_instr, device_radius_min=dev['radius_min'],
        device_radius_max=dev['radius_max'],
        device_codespace_weight_min=min(
            v['codespace_weight'] for v in dev.values()
            if isinstance(v, dict)),
        max_overread_at_device_r=max(v['overread'] for v in over.values()),
        descent_max_gain=max_gain, descent_max_warm_grad=max_warm_grad,
        runtime_s=time.time() - t0)
    summary.update({('selftest_' + k): v for k, v in st.items()})

    # The claim, as a boolean over measured quantities rather than a narrative:
    # every mixed input is stationary at phi^dec because the nine sector gradients
    # (stationarity_boundary.py) AND the one extra coefficient grad c0 vanish, the
    # algebra (*) connecting them is validated where it is nonzero, the mixed score
    # itself is validated against a graph-free density-matrix trace, and the
    # controls are O(1).
    proved_mixed = bool(
        max_mix < 1e-14 and max_fd < 1e-8 and max_c0 < 1e-15
        and min_ctrl > 1e-4 and max_z < 1e-14
        and st['identity_residual'] < 1e-12
        and st['identity_min_magnitude'] > 1e-4
        and st['mixed_density_matrix_residual'] < 1e-12)

    print('\n================ SUMMARY ================')
    for k, v in summary.items():
        print(f'  {k:34s} {v}')
    print(f'\n  mixed-input gradient at phi^dec <= {max_mix:.2e} over '
          f'{summary["n_scores"]} (channel, radius, direction) scores, '
          f'FD <= {max_fd:.2e}, controls >= {min_ctrl:.2e}')
    print(f'  the extra l=0 coefficient grad c0 vanishes to {max_c0:.2e}')
    print(f'  device-implied logical input radius r = {dev["radius_min"]:.4f}'
          f'--{dev["radius_max"]:.4f} at p_instr = {p_instr:.4f}; a '
          f'pure-ensemble benchmark over-reads such an input by up to '
          f'{summary["max_overread_at_device_r"]:.3e}')
    print(f'  warm-start gradient on the mixed score <= {max_warm_grad}, '
          f'best multi-start gain {max_gain}')
    print(f'  STATIONARY FOR MIXED INPUTS TOO (algebra + ten measured zeros): '
          f'{proved_mixed}')

    out = dict(generated_utc=time.strftime('%Y-%m-%dT%H:%M:%S'),
               pinned_cpu=g.PINNED_CPU, quick=a.quick,
               radii=list(RADII), directions=list(DIRS),
               selftests=st, rows=rows, device_prep=dev,
               overread=over, descent=desc, summary=summary,
               stationary_for_mixed_inputs=proved_mixed)
    with open(os.path.join(ROOT, a.json), 'w') as f:
        json.dump(out, f, indent=1)
    print(f'\nwrote {a.json}  ({summary["runtime_s"]:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

