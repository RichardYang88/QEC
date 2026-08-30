"""Focused quick evaluation: train VSCR (curriculum + seed selection) on
depolarizing + amplitude damping, evaluate all methods, print tables. ~8 min."""
import time, numpy as np, torch
import ssvr_qec as m

torch.manual_seed(m.SEED); np.random.seed(m.SEED)
p_values = np.array([0.005, 0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.20])

def run(noise, p_lindr):
    t0 = time.time()
    model, hist, infos = m.train_vscr_best(noise, seeds=(2024, 777), verbose=False)
    A = m.train_lindr(n_train=1500, p_range=p_lindr, noise=noise)
    res = m.evaluate_methods(model, A, p_values, noise, n_test=30)
    best_cf = max(i['min_cf'] for i in infos)
    print(f"\n=== {noise}  (train {time.time()-t0:.1f}s, "
          f"final val_fid={hist['fidelity'][-1]:.4f}, best-seed min_cf={best_cf:.4f}) ===")
    print("p        " + "  ".join(f"{k[:9]:>10s}" for k in res))
    for i, p in enumerate(p_values):
        print(f"{p:.3f}    " + "  ".join(f"{res[k]['F'][i]:10.4f}" for k in res))
    return res

r_dep = run('depolarizing', (0.01, 0.12))
r_ad = run('amplitude_damping', (0.01, 0.08))
print("\nDONE")
