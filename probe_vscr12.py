"""Probe 12: self-diagnostic repair pass -- reweight toward worst syndromes."""
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

torch.manual_seed(1234); np.random.seed(1234)
t0 = time.time()
model = m.VSCR()
for ep, lr, pr in [(600, 3e-2, (0.02, 0.08)), (400, 3e-2, (0.02, 0.15)),
                   (1200, 1e-3, (0.02, 0.15))]:
    m.train_vscr(model, n_epochs=ep, batch=48, p_range=pr, noise='depolarizing',
                 lr=lr, lam=0.0, verbose=False)
g = gap_vec(model)
print(f"before repair: gap_sum={g.sum():.4f} worst={np.argmax(g):04b}:{g.max():.4f} ({time.time()-t0:.0f}s)")

w = torch.tensor(1.0 + 50.0 * g / max(g.max(), 1e-9), dtype=torch.float64)
print("repair weights:", np.round(w.numpy(), 1))
for ep, lr in [(400, 1e-3), (800, 3e-4)]:
    h = m.train_vscr(model, n_epochs=ep, batch=48, p_range=(0.02, 0.15), noise='depolarizing',
                     lr=lr, lam=0.0, verbose=False, synd_w=w)
    g = gap_vec(model)
    print(f"repair {ep}@{lr}: val={h['fidelity'][-1]:.4f} gap_sum={g.sum():.4f} "
          f"worst={np.argmax(g):04b}:{g.max():.4f} ({time.time()-t0:.0f}s)")

with torch.no_grad():
    R_all = m.recovery_unitary_batch(model())
rng = np.random.RandomState(11)
for p in [0.04, 0.07, 0.10, 0.15, 0.20]:
    f_v, f_d, f_z = [], [], []
    for _ in range(30):
        psi = m.random_logical_state(1, rng); pe = m.encode(psi)
        rho = m.noisy_state(pe, p, 'depolarizing')
        _, Fv = m.vscr_fidelity(pe, rho, R_all, lam=0.0)
        f_v.append(float(Fv))
        f_d.append(float(m.perfect_code_decoder_fidelity(pe, rho)))
        f_z.append(float(m.zne_fidelity(pe, p, 'depolarizing')))
    print(f"p={p}: VSCR {np.mean(f_v):.4f}   dec {np.mean(f_d):.4f}   zne {np.mean(f_z):.4f}")
print("PROBE12 DONE")
