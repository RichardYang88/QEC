"""Is the Pauli decoder a critical point of the warm-start objective?

If grad_phi F(PHI_DEC) ~ 0 then Adam cannot find the non-Pauli headroom from the
warm start, no matter how exact the gradient estimator is.  Also runs explicit
gradient ASCENT on the exact p=0.06 objective to see how much headroom is
reachable at all from the decoder.
"""
import sys

sys.path.insert(0, '/home/yqc/github/QEC')
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import ssvr_qec as m                                        # noqa: E402
import vscr_paper as vp                                     # noqa: E402
import vscr_paper_abl as ab                                 # noqa: E402

NOISE = sys.argv[1] if len(sys.argv) > 1 else 'amplitude_damping'
P = 0.06
print('channel =', NOISE, ' p =', P, flush=True)

u = ab.opt_unitary_ceiling(NOISE, P)
f_dec = ab.exact_F(vp.PHI_DEC, NOISE, P)[0]
print('F(dec)      = %.12f' % f_dec)
print('F(unit ceil)= %.12f   headroom = %.4e' % (u['F_unit'],
                                                 u['F_unit'] - f_dec), flush=True)


def F_of(phi_np, p=P, pr=None):
    """Exact Haar-averaged F of a phi table, via the quadrature objective."""
    Q = m.quad_cache((p, p), NOISE, n_p=1) if pr is None \
        else m.quad_cache(pr, NOISE)
    _loss, F, _ = m.vscr_fidelity_quad(
        torch.tensor(np.asarray(phi_np), dtype=torch.float64), Q)
    return float(F)


# ---- 1. gradient of the exact objective at the decoder -------------------
for label, pr in (('p=0.06 only', (P, P)), ('curriculum (0.01,0.15)',
                                            (0.01, 0.15))):
    Q = m.quad_cache(pr, NOISE, n_p=1) if pr[0] == pr[1] else m.quad_cache(pr,
                                                                          NOISE)
    phi = torch.tensor(np.asarray(vp.PHI_DEC, dtype=float),
                       dtype=torch.float64).clone().requires_grad_(True)
    loss, F, _ = m.vscr_fidelity_quad(phi, Q)
    loss.backward()
    g = phi.grad.detach().numpy()
    print('\n[%s]  F = %.12f' % (label, float(F)))
    print('   ||grad_phi loss||_max = %.4e   ||.||_2 = %.4e'
          % (np.abs(g).max(), np.linalg.norm(g)))
    print('   per-syndrome ||grad_s||_2 = %s'
          % np.array2string(np.linalg.norm(g.reshape(16, -1), axis=1),
                            precision=2, suppress_small=False))

# ---- 2. explicit full-precision gradient ascent on F at p = 0.06 ---------
Q = m.quad_cache((P, P), NOISE, n_p=1)
phi = torch.tensor(np.asarray(vp.PHI_DEC, dtype=float),
                   dtype=torch.float64).clone().requires_grad_(True)
opt = torch.optim.Adam([phi], lr=1e-2)
print('\nexplicit gradient ASCENT on the exact p=%.2f objective:' % P,
      flush=True)
for step in range(4001):
    opt.zero_grad()
    loss, F, _ = m.vscr_fidelity_quad(phi, Q)
    (-loss).backward()                       # ascend
    opt.step()
    if step % 500 == 0 or step == 4000:
        fv = float(F)
        print('   step %5d  F = %.12f  (F-F_dec = %+.4e, captured = %+.2f%%)'
              % (step, fv, fv - f_dec,
                 100 * (fv - f_dec) / (u['F_unit'] - f_dec)), flush=True)

with torch.no_grad():
    phi_best = phi.detach().cpu().numpy()
print('\nbest ascent F(exact_F)  = %.12f' % ab.exact_F(phi_best, NOISE, P)[0])
print('unitary ceiling         = %.12f' % u['F_unit'])
print('max|phi_ascent - PHI_DEC| = %.4e'
      % np.abs(phi_best - np.asarray(vp.PHI_DEC)).max())
np.save('/tmp/phi_ascent_%s.npy' % NOISE, phi_best)
