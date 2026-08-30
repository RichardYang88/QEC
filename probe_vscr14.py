"""Probe 14: measure gradient magnitude on the broken-syndrome landscape."""
import time, numpy as np, torch
import ssvr_qec as m

torch.manual_seed(1234); np.random.seed(1234)
model = m.VSCR()
for ep, lr, pr in [(600, 3e-2, (0.02, 0.08)), (400, 3e-2, (0.02, 0.15)),
                   (1200, 1e-3, (0.02, 0.15))]:
    m.train_vscr(model, n_epochs=ep, batch=48, p_range=pr, noise='depolarizing',
                 lr=lr, lam=0.0, verbose=False)

w = torch.ones(16, dtype=torch.float64); w[14] = 51.0   # syndrome 1110
rng = np.random.RandomState(9)
before = {n: p.detach().clone() for n, p in model.named_parameters()}
for step in range(3):
    model.zero_grad()
    R_all = m.recovery_unitary_batch(model())
    tot = 0.0
    for _ in range(96):
        psi = m.random_logical_state(1, rng); pe = m.encode(psi)
        p = rng.uniform(0.02, 0.15)
        rho = m.noisy_state(pe, p, 'depolarizing')
        loss, _ = m.vscr_fidelity(pe, rho, R_all, lam=0.0, synd_w=w)
        tot = tot + loss
    (tot / 96).backward()
    gnorm = float(sum(float(p.grad.norm()**2) for p in model.parameters() if p.grad is not None)) ** 0.5
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    opt.step()
    print(f"step {step}: loss={float(tot/96):.5f} grad_norm={gnorm:.3e}")
delta = max(float((p - before[n]).abs().max()) for n, p in model.named_parameters())
print(f"max |param change| after 3 SGD steps @1e-2: {delta:.3e}")
print("PROBE14 DONE")
