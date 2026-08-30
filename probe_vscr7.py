"""Probe 7: does longer training escape the plateau? (depolarizing, hypernet VSCR)"""
import time, numpy as np, torch
import ssvr_qec as m

torch.manual_seed(m.SEED); np.random.seed(m.SEED)
t0 = time.time()
model = m.VSCR()
stages = [(600, 3e-2, (0.02, 0.08)),
          (600, 3e-2, (0.02, 0.15)),
          (3000, 3e-3, (0.02, 0.15)),
          (3000, 1e-3, (0.02, 0.15))]
for ep, lr, pr in stages:
    h = m.train_vscr(model, n_epochs=ep, batch=48, p_range=pr, noise='depolarizing',
                     lr=lr, lam=0.0, verbose=False)
    with torch.no_grad():
        R_all = m.recovery_unitary_batch(model())
    nov = sum(float(abs(torch.trace(m.C_SYNDS[s].conj().T @ R_all[s])) / m.DIM > 0.95)
              for s in range(16))
    print(f"stage {ep}@{lr}: val={h['fidelity'][-1]:.4f} locked_syndromes={nov}/16 ({time.time()-t0:.0f}s)")

rng = np.random.RandomState(11)
for p in [0.04, 0.10, 0.20]:
    f_v, f_d = [], []
    for _ in range(25):
        psi = m.random_logical_state(1, rng); pe = m.encode(psi)
        rho = m.noisy_state(pe, p, 'depolarizing')
        _, Fv = m.vscr_fidelity(pe, rho, R_all, lam=0.0)
        f_v.append(float(Fv))
        f_d.append(float(m.perfect_code_decoder_fidelity(pe, rho)))
    print(f"p={p}: VSCR {np.mean(f_v):.4f}   decoder {np.mean(f_d):.4f}")
print("PROBE7 DONE")
