"""Probe 20: extended high-LR exploration before freezing (3 seeds, both noises)."""
import time, numpy as np, torch
import ssvr_qec as m

def cond_fid(model, noise, p=0.07, M=150, seed=5):
    with torch.no_grad():
        R_all = m.recovery_unitary_batch(model())
    rng = np.random.RandomState(seed)
    Fs = np.zeros(16); Ws = np.zeros(16)
    for _ in range(M):
        psi = m.random_logical_state(1, rng); pe = m.encode(psi)
        rho = m.noisy_state(pe, p, noise)
        rho16 = rho.unsqueeze(0).expand(16, m.DIM, m.DIM)
        PRP = torch.bmm(torch.bmm(m.P_SYNDS_STACK, rho16), m.P_SYNDS_STACK)
        Y = torch.bmm(torch.bmm(R_all, PRP), R_all.conj().transpose(1, 2))
        Fs += torch.einsum('i,sij,j->s', pe.conj(), Y, pe).real.numpy()
        Ws += torch.einsum('sij,ji->s', m.P_SYNDS_STACK, rho).real.numpy()
    return Fs / np.maximum(Ws, 1e-12), Ws / M

PLANS = {
 'depolarizing': [(600, 3e-2, (0.02, 0.08)), (1200, 3e-2, (0.02, 0.15)),
                  (1200, 1e-3, (0.02, 0.15))],
 'amplitude_damping': [(1600, 3e-2, (0.01, 0.12)), (1200, 1e-3, (0.01, 0.12))],
}
for noise, stages in PLANS.items():
    print(f"\n########## {noise} ##########")
    results = []
    for seed in [2024, 777, 1234]:
        torch.manual_seed(seed); np.random.seed(seed)
        t0 = time.time()
        model = m.VSCR()
        for ep, lr, pr in stages:
            h = m.train_vscr(model, n_epochs=ep, batch=48, p_range=pr, noise=noise,
                             lr=lr, lam=0.0, verbose=False, rng_seed=seed)
        cf, w = cond_fid(model, noise)
        mask = w > 1e-3
        mincf = float(np.where(mask, cf, 10.0).min())
        results.append((mincf, h['fidelity'][-1], seed, model))
        print(f"  seed {seed}: val={h['fidelity'][-1]:.4f} min_cf={mincf:.4f} ({time.time()-t0:.0f}s)")
    results.sort(key=lambda r: (-r[0], -r[1]))
    best = results[0][3]
    print(f"  -> selected seed with min_cf={results[0][0]:.4f}")
    with torch.no_grad():
        R_all = m.recovery_unitary_batch(best)
    rng = np.random.RandomState(11)
    for p in [0.04, 0.07, 0.10, 0.15, 0.20]:
        f_v, f_d = [], []
        for _ in range(30):
            psi = m.random_logical_state(1, rng); pe = m.encode(psi)
            rho = m.noisy_state(pe, p, noise)
            _, Fv = m.vscr_fidelity(pe, rho, R_all, lam=0.0)
            f_v.append(float(Fv))
            f_d.append(float(m.perfect_code_decoder_fidelity(pe, rho)))
        print(f"  p={p}: VSCR {np.mean(f_v):.4f}   dec {np.mean(f_d):.4f}")
print("PROBE20 DONE")
