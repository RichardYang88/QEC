"""Does the warm-start VSCR actually capture the non-Pauli recovery headroom?

Compares, per channel at p = P (Fig.4's bar-chart point):
  F_dec   the rigid Pauli decoder warm start (starting point)
  F_mc    train_warm_best with the old 48-sample Monte-Carlo gradient
  F_quad  train_warm_best with the exact zero-variance quadrature gradient
  F_unit  the EXACT optimum over per-syndrome unitaries (ceiling)
and reports the fraction of the available headroom each run captures.

All four fidelities are evaluated with the SAME noiseless closed-form
second-moment estimator `ab.exact_F`, so the comparison carries zero
Monte-Carlo error and headrooms of order 1e-5 are resolvable.
"""
import sys
import time

sys.path.insert(0, '/home/yqc/github/QEC')
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import vscr_paper as vp                                     # noqa: E402
import vscr_paper_abl as ab                                 # noqa: E402

# `coherent` is not in the stock curriculum (vscr_paper_coh.py injects it);
# use exactly the same schedule so this diagnostic matches the paper driver.
vp.WARM_SCHEDULES.setdefault('coherent', [(400, 3e-3, (0.02, 0.08)),
                                          (500, 3e-3, (0.02, 0.15)),
                                          (800, 3e-4, (0.02, 0.15))])

SEEDS = (1234,)
P = float(sys.argv[1]) if len(sys.argv) > 1 else 0.06
N_EPOCHS = sum(ep for ep, _, _ in vp.WARM_SCHEDULES['depolarizing'])

print('curriculum = %d epochs/stage-set (%d total), seeds = %s, p = %.3f'
      % (len(vp.WARM_SCHEDULES['depolarizing']), N_EPOCHS, SEEDS, P),
      flush=True)


def _phi(model):
    with torch.no_grad():
        return model().detach().cpu().numpy()


rows = []
for noise in ab.CHANNELS:
    t0 = time.time()
    f_dec = float(ab.exact_F(vp.PHI_DEC, noise, P)[0])
    f_unit = float(ab.opt_unitary_ceiling(noise, P)['F_unit'])

    fids = {}
    for tag, quad in (('mc', False), ('quad', True)):
        model, _hist, _infos = vp.train_warm_best(noise, seeds=SEEDS,
                                                  verbose=False, quad=quad)
        fids[tag] = float(ab.exact_F(_phi(model), noise, P)[0])

    head = f_unit - f_dec
    cap = {t: ((fids[t] - f_dec) / head if head > 1e-12 else float('nan'))
           for t in fids}
    rows.append((noise, f_dec, fids['mc'], fids['quad'], f_unit,
                 cap['mc'], cap['quad']))
    print('%-20s F_dec=%.6f  F_mc=%.6f  F_quad=%.6f  F_unit=%.6f | '
          'headroom=%.2e  captured: mc=%+.1f%% quad=%+.1f%%  (%.0fs)'
          % (noise, f_dec, fids['mc'], fids['quad'], f_unit, head,
             100 * cap['mc'], 100 * cap['quad'], time.time() - t0), flush=True)

print('\n%-20s %10s %10s %10s %10s %9s %9s'
      % ('channel', 'F_dec', 'F_mc', 'F_quad', 'F_unit', 'cap_mc', 'cap_quad'))
for r in rows:
    print('%-20s %10.6f %10.6f %10.6f %10.6f %8.1f%% %8.1f%%'
          % (r[0], r[1], r[2], r[3], r[4], 100 * r[5], 100 * r[6]))
np.savez('/tmp/headroom_check.npz',
         channels=np.array([r[0] for r in rows]),
         F_dec=np.array([r[1] for r in rows]),
         F_mc=np.array([r[2] for r in rows]),
         F_quad=np.array([r[3] for r in rows]),
         F_unit=np.array([r[4] for r in rows]),
         p=np.array([P]))
print('\nsaved /tmp/headroom_check.npz', flush=True)

