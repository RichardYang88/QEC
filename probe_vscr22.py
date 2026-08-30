"""Probe 22: mixed-noise VSCR with curriculum schedule + gated selection."""
import time, numpy as np, torch
import ssvr_qec as m

t0 = time.time()
model, hist, infos = m.train_vscr_best('mixed')
print(f"train {time.time()-t0:.0f}s, val_fid={hist['fidelity'][-1]:.4f}")
A = m.train_lindr(n_train=1500, p_range=(0.01, 0.12), noise='mixed')
p_values = np.array([0.005, 0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.20])
res = m.evaluate_methods(model, A, p_values, 'mixed', n_test=30)
print("p        Raw    PerfDec   ZNE      VD      LinDR   VSCR")
for i, p in enumerate(p_values):
    print(f"{p:.3f}  {res['Raw']['F'][i]:.4f} {res['Perfect-code']['F'][i]:.4f} "
          f"{res['ZNE']['F'][i]:.4f} {res['Virtual Distillation']['F'][i]:.4f} "
          f"{res['LinDR']['F'][i]:.4f} {res['VSCR (ours)']['F'][i]:.4f}")
print("PROBE22 DONE")
