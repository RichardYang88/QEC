"""Verify the coverage the 28-line run_selftests.py no longer exercises:
  * ssvr_qec._selftest_gates()   (not called by the current runner at all)
  * E: warm-start training smoke test + fidelity ordering
  * F: cold training smoke test
"""
import sys
import time

sys.path.insert(0, '/home/yqc/github/QEC')
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import ssvr_qec as m                                        # noqa: E402
import vscr_paper as vp                                     # noqa: E402
import vscr_paper_abl as ab                                 # noqa: E402

t0 = time.time()
print('=== GATES selftest (missing from run_selftests.py) ===', flush=True)
m._selftest_gates()
print('[gates OK in %.0fs]' % (time.time() - t0), flush=True)

print('\n=== E: warm-start training + fidelity ordering ===', flush=True)
P_TEST = 0.06
SEEDS_T = (1234,)
VIOLATIONS = []


def _phi(model):
    with torch.no_grad():
        return model().detach().cpu().numpy()


for noise in ab.CHANNELS:
    t1 = time.time()
    model, _hist, _infos = vp.train_warm_best(noise, seeds=SEEDS_T,
                                              verbose=False, quad=True)
    f_warm = float(ab.exact_F(_phi(model), noise, P_TEST)[0])
    u = ab.opt_unitary_ceiling(noise, P_TEST)
    f_dec, f_unit = u['F_dec'], u['F_unit']
    # F_unit <= F_cptp is already asserted for 4 channel/p pairs by section B
    ok = (f_dec - 1e-12 <= f_warm <= f_unit + 1e-12)
    head = f_unit - f_dec
    cap = 100 * (f_warm - f_dec) / head if head > 1e-15 else float('nan')
    print('  %-18s p=%.2f  F_dec=%.10f  F_warm=%.10f  F_unit=%.10f'
          % (noise, P_TEST, f_dec, f_warm, f_unit))
    print('       headroom=%.3e  captured=%.4f%%  ordering %s  [%.0fs]'
          % (head, cap, 'OK' if ok else 'VIOLATED', time.time() - t1),
          flush=True)
    if not ok:
        print('       !! F_warm - F_dec = %+.3e  (warm start REGRESSED below '
              'the decoder baseline)' % (f_warm - f_dec), flush=True)
        VIOLATIONS.append((noise, f_warm - f_dec))

print('\n=== F: cold (Lindbladian) baseline smoke test ===', flush=True)
t1 = time.time()
out = vp.train_lindr_phys(n_train=150, p_range=(0.02, 0.08),
                          noise='depolarizing')
print('  train_lindr_phys returned %s [%.0fs]'
      % (type(out).__name__ if not isinstance(out, tuple)
         else 'tuple(len=%d)' % len(out), time.time() - t1), flush=True)

print('\n=== ORDERING SUMMARY ===', flush=True)
if VIOLATIONS:
    worst = min(d for _n, d in VIOLATIONS)
    for n, d in VIOLATIONS:
        print('  %-18s F_warm - F_dec = %+.3e' % (n, d))
    print('  worst regression = %+.3e  (%s at the 1e-6 plotting-relevance '
          'threshold)' % (worst, 'NEGLIGIBLE' if worst > -1e-6 else 'MATERIAL'))
else:
    print('  F_dec <= F_warm <= F_unit on every channel  OK')

print('\nEXTRA CHECKS COMPLETE  [TOTAL %.0fs]' % (time.time() - t0))
