"""Probe 13: SGD repair pass (Adam normalizes away the syndrome weights)."""
import time, numpy as np, torch
import ssvr_qec as m

def gap_vec(model, noise='depolarizing', p=0.07, M=150, seed=5):
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

def sgd_stage(model, lr, n_ep, w, ptr=(0.02, 0.15), batch=48, seed=1234):
    rng = np.random.RandomState(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for ep in range(n_ep):
        opt.zero_grad()
        R_all = m.recovery_unitary_batch(model())
        tot = 0.0
        for _ in range(batch):
            psi = m.random_logical_state(1, rng); pe = m.encode(psi)
            p = rng.uniform(*ptr)
            rho = m.noisy_state(pe, p, 'depolarizing')
            loss, _ = m.vscr_fidelity(pe, rho, R_all, lam=0.0, synd_w=w)
            tot = tot + loss
        (tot / batch).backward(); opt.step()

torch.manual_seed(1234); np.random.seed(1234)
t0 = time.time()
model = m.VSCR()
for ep, lr, pr in [(600, 3e-2, (0.02, 0.08)), (400, 3e-2, (0.02, 0.15)),
                   (1200, 1e-3, (0.02, 0.15))]:
    m.train_vscr(model, n_epochs=ep, batch=48, p_range=pr, noise='depolarizing',
                 lr=lr, lam=0.0, verbose=False)
g = gap_vec(model)
print(f"base: gap_sum={g.sum():.4f} worst={np.argmax(g):04b}:{g.max():.4f} ({time.time()-t0:.0f}s)")
w = torch.tensor(1.0 + 50.0 * g / max(g.max(), 1e-9), dtype=torch.float64)
for lr in [1e-3, 3e-3, 1e-2]:
    sgd_stage(model, lr, 300, w)
    g = gap_vec(model)
    print(f"sgd 300@{lr}: gap_sum={g.sum():.4f} worst={np.argmax(g):04b}:{g.max():.4f} ({time.time()-t0:.0f}s)")
print("PROBE13 DONE")
