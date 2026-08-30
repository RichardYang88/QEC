"""Probe 6: capacity test -- per-syndrome free parameters (no shared hypernet)."""
import time, numpy as np, torch
import ssvr_qec as m


class FreePhi(torch.nn.Module):
    """Ablation: independent phi_s per syndrome (no parameter sharing)."""
    def __init__(self):
        super().__init__()
        self.phi = torch.nn.Parameter(
            torch.randn(16, m.PHI_DIM, dtype=torch.float64) * 0.05)

    def forward(self):
        return self.phi


torch.manual_seed(m.SEED); np.random.seed(m.SEED)
t0 = time.time()
model = FreePhi()
for ep, lr, pr in [(600, 3e-2, (0.02, 0.08)), (400, 3e-2, (0.02, 0.15)),
                   (1500, 3e-3, (0.02, 0.15))]:
    h = m.train_vscr(model, n_epochs=ep, batch=48, p_range=pr, noise='depolarizing',
                     lr=lr, lam=0.0, verbose=False)
    print(f"stage {ep}@{lr}: val={h['fidelity'][-1]:.4f}  ({time.time()-t0:.0f}s)")

with torch.no_grad():
    R_all = m.recovery_unitary_batch(model())
print("per-syndrome overlap with decoder correction:")
for s in range(16):
    ov = abs(torch.trace(m.C_SYNDS[s].conj().T @ R_all[s])) / m.DIM
    print(f"  s={s:04b}  {float(ov):.4f}")

rng = np.random.RandomState(11)
for p in [0.04, 0.10, 0.20]:
    f_v, f_d = [], []
    for _ in range(25):
        psi = m.random_logical_state(1, rng); pe = m.encode(psi)
        rho = m.noisy_state(pe, p, 'depolarizing')
        _, Fv = m.vscr_fidelity(pe, rho, R_all, lam=0.0)
        f_v.append(float(Fv))
        f_d.append(float(m.perfect_code_decoder_fidelity(pe, rho)))
    print(f"p={p}: FreePhi {np.mean(f_v):.4f}   decoder {np.mean(f_d):.4f}")
print("PROBE6 DONE")