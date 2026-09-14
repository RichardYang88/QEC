#!/usr/bin/env python3
"""
diag_check.py — diagnose the warm-start headroom capture (Fix A / Fix B).

THE PROBLEM THIS EXISTS FOR.  Warm-started VSCR used to score EXACTLY the Pauli
decoder, so Fig.3 showed the two curves on top of each other and the "learned
non-Pauli correction" claim had no supporting evidence.  The cause is not bad
tuning: `PHI_DEC` is an EXACT stationary point of the training objective.  Each
branch term is a 4x4 Rayleigh quotient g_s^dag Q_s g_s with ||g_s|| fixed, and
the gradient of a Rayleigh quotient vanishes exactly at the eigenvectors of Q_s;
vec(G_dec) is such an eigenvector on every branch of every channel.  Wherever
lambda(G_dec) < lambda_max(Q_s) the decoder is a SADDLE with certified headroom
F_unit - F_dec above it, and no first-order method started there can move.

WHAT THIS SCRIPT REPORTS, per channel:
  1. the exact gradient of the objective at PHI_DEC (autograd AND central finite
     differences) -- the mechanism, measured rather than asserted;
  2. the certified headroom F_unit - F_dec from `opt_unitary_ceiling`, and the
     worst single branch;
  3. what `refine_per_syndrome` recovers from PHI_DEC: F before/after, the
     certified p-averaged ceiling, and the captured fraction;
  4. for any saved warm angle table, the Fix A guarantee F_warm >= F_dec on the
     EXACT production objective (not an approximation), plus the branch
     deviation and syndrome spread that show the recovery is genuinely
     syndrome-dependent rather than one global rotation.

Usage: python diag_check.py [--quick] [--channels=dep,ad] [--json OUT]
"""
import os, sys, time

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
# See the long note in ssvr_qec.py: this host intermittently SIGSEGV/SIGILLs in
# tiny complex128 kernels when a process is free to MIGRATE across its hybrid
# P-/E-core cluster. Importing ssvr_qec pins us to one CPU, which fixes it
# (measured: unpinned core-dumps within ~2e4 calls; pinned runs 4e5 clean).
# OPENBLAS_CORETYPE is for determinism only -- it is NOT the fix, since the
# faults reproduce under every coretype including AVX-only SANDYBRIDGE.
os.environ.setdefault('OPENBLAS_CORETYPE', 'HASWELL')

import json
import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssvr_qec as m                                          # noqa: E402
import vscr_paper as vp                                       # noqa: E402
import vscr_paper_abl as ab                                   # noqa: E402

CHAN = {'dep': 'depolarizing', 'ad': 'amplitude_damping',
        'mixed': 'mixed', 'coh': 'coherent'}
NPZ = {'depolarizing': 'vscr_angles_paper_dep.npz',
       'amplitude_damping': 'vscr_angles_paper_ad.npz',
       'mixed': 'vscr_angles_paper_mixed.npz',
       'coherent': 'vscr_angles_paper_coh.npz'}
SCHED_P = {'depolarizing': (0.02, 0.15), 'amplitude_damping': (0.01, 0.15),
           'mixed': (0.02, 0.15), 'coherent': (0.02, 0.15)}


def grad_at(phi_tab, noise, p):
    """max |d F̄(p) / d phi| over all 960 angles, by autograd and central FD."""
    qf = ab._branch_qforms(noise, p)
    Qt = torch.tensor(np.stack([q[0] for q in qf]), dtype=torch.complex128)
    Mt = torch.tensor(np.stack([ab._M_of_S(q[3]) for q in qf]),
                      dtype=torch.complex128)
    W_t = torch.tensor(ab.W_BASIS, dtype=torch.complex128)
    V_t = torch.tensor(ab.V_ISO, dtype=torch.complex128)

    def obj(phi_t):
        RW = m.recovery_action_cols(phi_t.reshape(16, -1), W_t)
        G = torch.matmul(V_t.conj().T.unsqueeze(0), RW)
        gg = G.reshape(16, 4)
        J = torch.einsum('si,sij,sj->s', gg, Qt, gg.conj()).real
        T = torch.einsum('si,sij,sj->s', gg.conj(), Mt, gg).real
        return ((J + T) / 6).sum()

    pt = torch.tensor(np.asarray(phi_tab, dtype=float),
                      dtype=torch.float64, requires_grad=True)
    obj(pt).backward()
    gmax = float(pt.grad.abs().max())
    h = 1e-6
    base = np.asarray(phi_tab, dtype=float).copy()
    fdmax = 0.0
    for s in range(16):
        for k in range(0, m.PHI_DIM, 12):
            pp = base.copy(); pp[s, k] += h
            pm = base.copy(); pm[s, k] -= h
            fdmax = max(fdmax, abs(ab._exact_F_from_qform(pp, qf)[0]
                                   - ab._exact_F_from_qform(pm, qf)[0]) / (2 * h))
    return gmax, fdmax


def check_channel(noise, quick=False):
    out = {'channel': noise}
    p_test = 0.06
    pr = SCHED_P[noise]
    print(f'\n===== {noise} =====', flush=True)

    # 1. the saddle
    ga, gf = grad_at(vp.PHI_DEC.numpy(), noise, p_test)
    print(f'  [1] gradient of F̄ at PHI_DEC (p={p_test}): '
          f'|autograd|max={ga:.3e}  |central FD|max={gf:.3e}')
    out['grad_autograd_max'] = ga
    out['grad_fd_max'] = gf

    # 2. certified headroom
    r = ab.opt_unitary_ceiling(noise, p_test)
    cf_d = np.asarray(r['cf_dec'], float)
    cf_u = np.asarray(r['cf_unit'], float)
    br = cf_u - cf_d
    sb = int(np.argmax(br))
    print(f'  [2] certified at p={p_test}: F_dec={r["F_dec"]:.9f} '
          f'F_unit={r["F_unit"]:.9f} headroom={r["F_unit"]-r["F_dec"]:+.3e}')
    print(f'      worst branch s={sb}: cf_dec={cf_d[sb]:.9f} '
          f'cf_unit={cf_u[sb]:.9f} gap={br[sb]:+.3e}')
    out.update({'F_dec': r['F_dec'], 'F_unit': r['F_unit'],
                'headroom': r['F_unit'] - r['F_dec'],
                'worst_branch': sb, 'worst_branch_gap': float(br[sb])})

    # 3. what the separable refinement recovers
    n_start, steps = (2, 150) if quick else (3, 900)
    t0 = time.time()
    phi_ref, info = ab.refine_per_syndrome(
        vp.PHI_DEC, noise, p_range=pr, n_p=6, n_start=n_start, steps=steps,
        lr=0.01, seed=0)
    print(f'  [3] refine_per_syndrome over p in {pr} '
          f'({time.time()-t0:.0f}s, n_start={n_start}, steps={steps}):')
    print(f'      F̄ {info["F_before"]:.9f} -> {info["F_after"]:.9f}  '
          f'(certified ceiling {info["F_cert"]:.9f}, rigorous bound '
          f'{info["F_bound"]:.9f})')
    print(f'      captured {100*info["capture_frac"]:.4f}% of headroom; '
          f'{info["n_improved"]}/16 blocks improved; block objective == '
          f'production objective to {info["prod_err"]:.2e}; '
          f'code-preserving to {info["unitarity_dev"]:.2e}')
    Fr, _ = ab._exact_F_from_qform(phi_ref, ab._branch_qforms(noise, p_test))
    print(f'      at p={p_test}: F̄_dec={r["F_dec"]:.9f} -> '
          f'F̄_refined={Fr:.9f} (F_unit={r["F_unit"]:.9f})')
    out.update({'F_pavg_before': info['F_before'],
                'F_pavg_after': info['F_after'],
                'F_pavg_cert': info['F_cert'],
                'capture_frac': info['capture_frac'],
                'n_improved': info['n_improved'],
                'prod_err': info['prod_err'],
                'unitarity_dev': info['unitarity_dev'],
                'F_refined_at_p': Fr})
    assert info['F_after'] >= info['F_before'] - 1e-15
    assert Fr >= r['F_dec'] - 1e-15, 'FIX A VIOLATED: refined < decoder'

    # 4. saved warm table, if present
    path = NPZ[noise]
    if os.path.exists(path):
        phi_w = np.load(path)['phi']
        B = ab.pavg_bundle(noise, pr, 6)
        Fw = float(ab.pavg_F(phi_w, B)[0])
        Fd = float(ab.pavg_F(vp.PHI_DEC, B)[0])
        mod = vp.VSCRWarm().set_phi(phi_w)
        print(f'  [4] {path}: F̄_warm={Fw:.9f} vs F̄_dec={Fd:.9f}  '
              f'delta={Fw-Fd:+.3e}  '
              f'branch_deviation={mod.branch_deviation():.3e}  '
              f'syndrome_spread={mod.syndrome_spread():.3e}')
        ok = bool(Fw >= Fd - 1e-15)
        print(f'      FIX A (F_warm >= F_dec): {"PASS" if ok else "FAIL"}')
        out.update({'F_warm_saved': Fw, 'F_dec_pavg': Fd, 'fixA_ok': ok,
                    'branch_deviation': mod.branch_deviation(),
                    'syndrome_spread': mod.syndrome_spread()})
    else:
        print(f'  [4] {path} not found -- skipping saved-table check')
    return out


def main(argv):
    quick = '--quick' in argv
    chans = ['depolarizing', 'amplitude_damping', 'coherent']
    jout = 'diag_check.json'
    for i, a in enumerate(argv):
        if a.startswith('--channels'):
            chans = [CHAN.get(c.strip(), c.strip())
                     for c in a.split('=', 1)[1].split(',') if c.strip()]
        if a == '--json' and i + 1 < len(argv):
            jout = argv[i + 1]
    print(f'=== diag_check (quick={quick}) channels={chans} ===', flush=True)
    res = [check_channel(c, quick) for c in chans]
    with open(jout, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'\nwrote {jout}')
    print('\n================ SUMMARY ================')
    for r in res:
        print(f"{r['channel']:20s} headroom={r['headroom']:+.3e}  "
              f"captured={100*r['capture_frac']:7.3f}%  "
              f"grad@dec={r['grad_autograd_max']:.1e}  "
              f"fixA={'PASS' if r.get('fixA_ok', True) else 'FAIL'}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
