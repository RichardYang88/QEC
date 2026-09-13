"""Why does warm training capture 0% of the headroom?  Full-precision probe."""
import sys

sys.path.insert(0, '/home/yqc/github/QEC')
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import vscr_paper as vp                                     # noqa: E402
import vscr_paper_abl as ab                                 # noqa: E402

vp.WARM_SCHEDULES.setdefault('coherent', [(400, 3e-3, (0.02, 0.08)),
                                          (500, 3e-3, (0.02, 0.15)),
                                          (800, 3e-4, (0.02, 0.15))])
NOISE = sys.argv[1] if len(sys.argv) > 1 else 'amplitude_damping'
PS = (0.01, 0.03, 0.06, 0.10, 0.15)

print('channel =', NOISE, flush=True)
model, hist, infos = vp.train_warm_best(NOISE, seeds=(1234,), verbose=True,
                                        quad=True)
with torch.no_grad():
    phi_tr = model().detach().cpu().numpy()
phi_dec = np.asarray(vp.PHI_DEC, dtype=float)
print('\nphi shape trained=%s decoder=%s' % (phi_tr.shape, phi_dec.shape))
print('max|phi_trained - PHI_DEC| = %.6e' % np.abs(phi_tr - phi_dec).max())
print('per-syndrome max|dphi| = %s'
      % np.array2string(np.abs(phi_tr - phi_dec).max(axis=1), precision=3))
print('hist fidelity: first=%.10f last=%.10f  (delta=%+.3e)'
      % (hist['fidelity'][0], hist['fidelity'][-1],
         hist['fidelity'][-1] - hist['fidelity'][0]))
print('hist loss    : first=%.10f last=%.10f' % (hist['loss'][0],
                                                 hist['loss'][-1]))
print('infos =', infos, flush=True)

u = ab.opt_unitary_ceiling(NOISE, 0.06)
print('\nopt_unitary_ceiling(0.06): F_unit=%.10f F_dec=%.10f'
      % (u['F_unit'], u['F_dec']))

print('\n%-6s %14s %14s %14s %14s %10s' % ('p', 'F(dec)', 'F(trained)',
                                           'F(unit ceil)', 'headroom',
                                           'captured'))
for p in PS:
    f_d = ab.exact_F(phi_dec, NOISE, p)[0]
    f_t = ab.exact_F(phi_tr, NOISE, p)[0]
    f_u = ab.opt_unitary_ceiling(NOISE, p)['F_unit']
    head = f_u - f_d
    cap = (f_t - f_d) / head if head > 1e-12 else float('nan')
    print('%-6.2f %14.10f %14.10f %14.10f %14.3e %9.1f%%'
          % (p, f_d, f_t, f_u, head, 100 * cap), flush=True)

# Does the quadrature objective (what training actually minimises) agree with
# exact_F at p = 0.06?  If these differ, the two estimators are inconsistent.
import ssvr_qec as m                                         # noqa: E402
for nm, phi in (('dec', phi_dec), ('trained', phi_tr)):
    Q = m.quad_cache((0.06, 0.06), NOISE, n_p=1)
    loss, F, _ = m.vscr_fidelity_quad(torch.tensor(phi, dtype=torch.float64), Q)
    print('quad-objective F(%s) = %.10f   (exact_F = %.10f)'
          % (nm, float(F), ab.exact_F(phi, NOISE, 0.06)[0]), flush=True)
