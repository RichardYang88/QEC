"""Probe: validate LinDR fix + sweep VSCR hyperparameters (lr, lam) vs decoder/ZNE."""
import time, numpy as np, torch
import ssvr_qec as m

for noise, ptr in [('depolarizing', (0.02, 0.15)), ('amplitude_damping', (0.01, 0.12))]:
    print(f"\n########## {noise} ##########")
    # ---- LinDR fix check ----
    t0 = time.time()
    A = m.train_lindr(n_train=1500, p_range=ptr, noise=noise)
    print(f"[LinDR fit {time.time()-t0:.1f}s]")
    rng = np.random.RandomState(7)
    f_lin, f_raw, f_dec = [], [], []
    for _ in range(30):
        psi = m.random_logical_state(1, rng); pe = m.encode(psi)
        rho = m.noisy_state(pe, 0.07, noise)
        f_lin.append(float(m.lindr_fidelity(pe, rho, A)))
        f_raw.append(float(m.baseline_raw(pe, rho)))
        f_dec.append(float(m.perfect_code_decoder_fidelity(pe, rho)))
    print(f"  p=0.07:  LinDR {np.mean(f_lin):.4f}   raw {np.mean(f_raw):.4f}   decoder {np.mean(f_dec):.4f}")
    # ---- VSCR hyperparameter sweep ----
    for lr, lam in [(1e-2, 0.1), (3e-2, 0.1), (3e-2, 0.0)]:
        torch.manual_seed(m.SEED); np.random.seed(m.SEED)
        t0 = time.time()
        model = m.VSCR()
        h = m.train_vscr(model, n_epochs=500, batch=48, p_range=ptr, noise=noise,
                         lr=lr, lam=lam, verbose=False)
        with torch.no_grad():
            R_all = m.recovery_unitary_batch(model())
        rng = np.random.RandomState(11)
        rows = {'VSCR': [], 'dec': [], 'zne': []}
        for p in [0.04, 0.10, 0.20]:
            a, b, c = [], [], []
            for _ in range(25):
                psi = m.random_logical_state(1, rng); pe = m.encode(psi)
                rho = m.noisy_state(pe, p, noise)
                _, F = m.vscr_fidelity(pe, rho, R_all, lam=0.0)
                a.append(float(F))
                b.append(float(m.perfect_code_decoder_fidelity(pe, rho)))
                c.append(float(m.zne_fidelity(pe, p, noise)))
            rows['VSCR'].append(np.mean(a)); rows['dec'].append(np.mean(b)); rows['zne'].append(np.mean(c))
        fmt = lambda v: ' '.join(f'{x:.4f}' for x in v)
        print(f"  lr={lr} lam={lam}: {time.time()-t0:.0f}s  val_fid={h['fidelity'][-1]:.4f}")
        print(f"    p=0.04/0.10/0.20  VSCR {fmt(rows['VSCR'])}")
        print(f"                      dec  {fmt(rows['dec'])}")
        print(f"                      zne  {fmt(rows['zne'])}")
print("PROBE DONE")