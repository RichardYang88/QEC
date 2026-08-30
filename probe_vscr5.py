"""Probe 5: overlap diagnostic + finer fine-tune (3e-3) + instrument upper-bound check."""
import time, numpy as np, torch
import ssvr_qec as m

noise = 'depolarizing'
torch.manual_seed(m.SEED); np.random.seed(m.SEED)
t0 = time.time()
model = m.VSCR()
for ep, lr, pr in [(600, 3e-2, (0.02, 0.08)), (400, 3e-2, (0.02, 0.15)),
                   (1500, 3e-3, (0.02, 0.15))]:
    h = m.train_vscr(model, n_epochs=ep, batch=48, p_range=pr, noise=noise,
                     lr=lr, lam=0.0, verbose=False)
    print(f"stage {ep}@{lr}: val={h['fidelity'][-1]:.4f}  ({time.time()-t0:.0f}s)")

with torch.no_grad():
    R_all = m.recovery_unitary_batch(model())
print("per-syndrome overlap |Tr[C_s^dag R_s]|/32 with decoder correction:")
for s in range(16):
    ov = abs(torch.trace(m.C_SYNDS[s].conj().T @ R_all[s])) / m.DIM
    print(f"  s={s:04b}  {float(ov):.4f}")

rng = np.random.RandomState(11)
C_all = torch.stack(m.C_SYNDS)
for p in [0.04, 0.10, 0.20]:
    f_v, f_d, f_i = [], [], []
    for _ in range(25):
        psi = m.random_logical_state(1, rng); pe = m.encode(psi)
        rho = m.noisy_state(pe, p, noise)
        _, Fv = m.vscr_fidelity(pe, rho, R_all, lam=0.0)
        _, Fi = m.vscr_fidelity(pe, rho, C_all, lam=0.0)
        f_v.append(float(Fv)); f_i.append(float(Fi))
        f_d.append(float(m.perfect_code_decoder_fidelity(pe, rho)))
    print(f"p={p}: VSCR {np.mean(f_v):.4f}   decoder {np.mean(f_d):.4f}   "
          f"instrument(C_s) {np.mean(f_i):.4f}")
print("PROBE5 DONE")