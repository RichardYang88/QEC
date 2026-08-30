"""Probe 9: larger batch (higher per-syndrome gradient SNR) both noises."""
import time, numpy as np, torch
import ssvr_qec as m

def locked_frac(model):
    with torch.no_grad():
        R_all = m.recovery_unitary_batch(model())
    return sum(float(abs(torch.trace(m.C_SYNDS[s].conj().T @ R_all[s])) / m.DIM > 0.95)
               for s in range(16)), R_all

for noise, ptr in [('depolarizing', (0.02, 0.15)), ('amplitude_damping', (0.01, 0.12))]:
    print(f"\n########## {noise} ##########")
    torch.manual_seed(m.SEED); np.random.seed(m.SEED)
    t0 = time.time()
    model = m.VSCR()
    stages = [(800, 3e-2, 192), (1200, 3e-3, 192)]
    for ep, lr, b in stages:
        h = m.train_vscr(model, n_epochs=ep, batch=b, p_range=ptr, noise=noise,
                         lr=lr, lam=0.0, verbose=False)
        nov, R_all = locked_frac(model)
        print(f"  stage {ep}@{lr} b={b}: val={h['fidelity'][-1]:.4f} locked={nov}/16 ({time.time()-t0:.0f}s)")
    rng = np.random.RandomState(11)
    for p in [0.04, 0.10, 0.20]:
        f_v, f_d, f_z = [], [], []
        for _ in range(25):
            psi = m.random_logical_state(1, rng); pe = m.encode(psi)
            rho = m.noisy_state(pe, p, noise)
            _, Fv = m.vscr_fidelity(pe, rho, R_all, lam=0.0)
            f_v.append(float(Fv))
            f_d.append(float(m.perfect_code_decoder_fidelity(pe, rho)))
            f_z.append(float(m.zne_fidelity(pe, p, noise)))
        print(f"  p={p}: VSCR {np.mean(f_v):.4f}   dec {np.mean(f_d):.4f}   zne {np.mean(f_z):.4f}")
print("PROBE9 DONE")
