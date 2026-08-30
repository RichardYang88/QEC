"""Probe 11: seed dependence of the broken syndrome."""
import time, numpy as np, torch
import ssvr_qec as m

def worst_gap(model, noise='depolarizing', p=0.07, M=150, seed=5):
    with torch.no_grad():
        R_all = m.recovery_unitary_batch(model())
    C_all = torch.stack(m.C_SYNDS)
    rng = np.random.RandomState(seed)
    gap = np.zeros(16)
    for _ in range(M):
        psi = m.random_logical_state(1, rng); pe = m.encode(psi)
        rho = m.noisy_state(pe, p, noise)
        rho16 = rho.unsqueeze(0).expand(16, m.DIM, m.DIM)
        PRP = torch.bmm(torch.bmm(m.P_SYNDS_STACK, rho16), m.P_SYNDS_STACK)
        pc = pe.conj()
        def Fs(R):
            Y = torch.bmm(torch.bmm(R, PRP), R.conj().transpose(1, 2))
            return torch.einsum('i,sij,j->s', pc, Y, pe).real
        gap += (Fs(C_all) - Fs(R_all)).numpy()
    return gap / M

for seed in [1234, 2024, 777]:
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    model = m.VSCR()
    for ep, lr, pr in [(600, 3e-2, (0.02, 0.08)), (400, 3e-2, (0.02, 0.15)),
                       (1200, 1e-3, (0.02, 0.15))]:
        m.train_vscr(model, n_epochs=ep, batch=48, p_range=pr, noise='depolarizing',
                     lr=lr, lam=0.0, verbose=False, rng_seed=seed)
    g = worst_gap(model)
    order = np.argsort(-g)[:3]
    print(f"seed {seed}: val-gap total={g.sum():.4f}  worst syndromes: "
          + ', '.join(f'{s:04b}:{g[s]:.4f}' for s in order) + f"  ({time.time()-t0:.0f}s)")
print("PROBE11 DONE")
