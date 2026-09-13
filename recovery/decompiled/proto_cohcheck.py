"""coherent channel only -- imported via vscr_paper_coh, which injects the
missing vp.WARM_SCHEDULES['coherent'] entry exactly as production does."""
import sys
import time

sys.path.insert(0, '/home/yqc/github/QEC')
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import vscr_paper as vp                                     # noqa: E402
import vscr_paper_abl as ab                                 # noqa: E402
import vscr_paper_coh                                       # noqa: E402,F401  (injects)

assert 'coherent' in vp.WARM_SCHEDULES, 'schedule injection failed'

P = 0.06
noise = 'coherent'
t0 = time.time()
model, _h, _i = vp.train_warm_best(noise, seeds=(1234,), verbose=False,
                                   quad=True)
with torch.no_grad():
    phi = model().detach().cpu().numpy()
f_warm = float(ab.exact_F(phi, noise, P)[0])
u = ab.opt_unitary_ceiling(noise, P)
f_dec, f_unit = u['F_dec'], u['F_unit']
head = f_unit - f_dec
print('%-18s p=%.2f  F_dec=%.10f  F_warm=%.10f  F_unit=%.10f'
      % (noise, P, f_dec, f_warm, f_unit))
print('  headroom=%.4e  captured=%+.4f%%  F_warm-F_dec=%+.3e  ordering %s '
      '[%.0fs]'
      % (head,
         100 * (f_warm - f_dec) / head if head > 1e-15 else float('nan'),
         f_warm - f_dec,
         'OK' if f_dec - 1e-12 <= f_warm <= f_unit + 1e-12 else 'VIOLATED',
         time.time() - t0), flush=True)
