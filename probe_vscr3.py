"""Probe 3: curriculum training (easy noise -> full range) + long fine-tune."""
import time, numpy as np, torch
import ssvr_qec as m

p_probe = [0.04, 0.07, 0.10, 0.15, 0.20]
PLANS = {
    'depolarizing': [
        (600, 3e-2, (0.02, 0.08)),
        (400, 3e-2, (0.02, 0.15)),
        (600, 1e-3, (0.02, 0.15))],
    'amplitude_damping': [
        (600, 3e-2, (0.01, 0.06)),
        (400, 3e-2, (0.01, 0.12)),
        (600, 1e-3, (0.01, 0.12))],
}
for noise, stages in PLANS.items():
    print(f"\n########## {noise} ##########")
    torch.manual_seed(m.SEED); np.random.seed(m.SEED)
    t0 = time.time()
    model = m.VSCR()
    for ep, lr, ptr in stages:
        h = m.train_vscr(model, n_epochs=ep, batch=48, p_range=ptr, noise=noise,
                         lr=lr, lam=0.0, verbose=False)
        print(f"  stage ep={ep} lr={lr} range={ptr}: {time.time()-t0:.0f}s val={h['fidelity'][-1]:.4f}")
    with torch.no_grad():
        R_all = m.recovery_unitary_batch(model())
    rng = np.random.RandomState(11)
    rows = {'VSCR': [], 'dec': [], 'zne': [], 'vd': []}
    for p in p_probe:
        a, b, c, d = [], [], [], []
        for _ in range(25):
            psi = m.random_logical_state(1, rng); pe = m.encode(psi)
            rho = m.noisy_state(pe, p, noise)
            _, F = m.vscr_fidelity(pe, rho, R_all, lam=0.0)
            a.append(float(F)); b.append(float(m.perfect_code_decoder_fidelity(pe, rho)))
            c.append(float(m.zne_fidelity(pe, p, noise)))
            d.append(float(m.virtual_distillation_fidelity(pe, rho)))
        rows['VSCR'].append(np.mean(a)); rows['dec'].append(np.mean(b))
        rows['zne'].append(np.mean(c)); rows['vd'].append(np.mean(d))
    fmt = lambda v: ' '.join(f'{x:.4f}' for x in v)
    print("    p:    " + ' '.join(f'{x:7.3f}' for x in p_probe))
    for k in ['VSCR', 'dec', 'zne', 'vd']:
        print(f"    {k:5s} {fmt(rows[k])}")
print("PROBE3 DONE")