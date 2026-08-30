"""Probe 16: decisive -- optimize ONLY the stuck syndrome's params (frozen rest)."""
import time, numpy as np, torch
import ssvr_qec as m

class FreePhi(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = torch.nn.Parameter(
            torch.randn(16, m.PHI_DIM, dtype=torch.float64) * 0.05)
    def forward(self):
        return self.phi

def cond_fid(R_all, noise='depolarizing', p=0.07, M=150, seed=5):
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

torch.manual_seed(1234); np.random.seed(1234)
model = FreePhi()
m.train_vscr(model, n_epochs=800, batch=48, p_range=(0.02, 0.15), noise='depolarizing',
             lr=3e-2, lam=0.0, verbose=False)
with torch.no_grad():
    R_all = m.recovery_unitary_batch(model())
cf, w = cond_fid(R_all)
bad = [s for s in range(16) if w[s] > 1e-3 and cf[s] < 0.85]
print("bad syndromes after joint training:", [f'{s:04b}:{cf[s]:.3f}' for s in bad])

# fix each bad syndrome alone: freeze all rows except one
rng = np.random.RandomState(3)
for s in bad:
    row = torch.nn.Parameter(model.phi.data[s].clone())
    opt = torch.optim.Adam([row], lr=5e-2)
    wv = torch.zeros(16, dtype=torch.float64); wv[s] = 1.0
    for ep in range(400):
        opt.zero_grad()
        phi_all = model.phi.data.clone()
        phi_all[s] = row
        R_all2 = m.recovery_unitary_batch(phi_all)
        tot = 0.0
        for _ in range(32):
            psi = m.random_logical_state(1, rng); pe = m.encode(psi)
            p = rng.uniform(0.02, 0.15)
            rho = m.noisy_state(pe, p, 'depolarizing')
            loss, _ = m.vscr_fidelity(pe, rho, R_all2, lam=0.0, synd_w=wv)
            tot = tot + loss
        (tot / 32).backward()
        opt.step()
    with torch.no_grad():
        model.phi.data[s] = row.data
        R_all = m.recovery_unitary_batch(model())
    cf, w = cond_fid(R_all)
    print(f"  repaired s={s:04b}: cf now {cf[s]:.4f}")

rng = np.random.RandomState(11)
tot_v = tot_d = 0.0
for p in [0.07]:
    f_v, f_d = [], []
    for _ in range(100):
        psi = m.random_logical_state(1, rng); pe = m.encode(psi)
        rho = m.noisy_state(pe, p, 'depolarizing')
        _, Fv = m.vscr_fidelity(pe, rho, R_all, lam=0.0)
        f_v.append(float(Fv))
        f_d.append(float(m.perfect_code_decoder_fidelity(pe, rho)))
    print(f"p={p}: FreePhi+repair {np.mean(f_v):.4f}   dec {np.mean(f_d):.4f}")
print("PROBE16 DONE")
