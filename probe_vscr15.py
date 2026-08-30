"""Probe 15: label-free per-syndrome diagnostic + embedding restart of the worst syndrome."""
import time, numpy as np, torch
import ssvr_qec as m

def cond_fid(model, noise='depolarizing', p=0.07, M=150, seed=5):
    """Label-free per-syndrome conditional fidelity E[F_s]/E[Tr P_s rho]."""
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

torch.manual_seed(1234); np.random.seed(1234)
t0 = time.time()
model = m.VSCR()
for ep, lr, pr in [(600, 3e-2, (0.02, 0.08)), (400, 3e-2, (0.02, 0.15)),
                   (1200, 1e-3, (0.02, 0.15))]:
    m.train_vscr(model, n_epochs=ep, batch=48, p_range=pr, noise='depolarizing',
                 lr=lr, lam=0.0, verbose=False)
cf, w = cond_fid(model)
print(f"base: per-syndrome cond fid (weight>1e-3):  ({time.time()-t0:.0f}s)")
for s in range(16):
    if w[s] > 1e-3: print(f"  s={s:04b}  cf={cf[s]:.4f}  w={w[s]:.4f}")

# restart the worst syndrome embedding, then fine-tune whole model
bad = int(np.argmin(np.where(w > 1e-3, cf, 10.0)))
print(f"worst syndrome: {bad:04b}")
with torch.no_grad():
    model.synd_emb[bad] = torch.randn(24, dtype=torch.float64) * 0.3
for ep, lr in [(400, 1e-2), (600, 1e-3)]:
    h = m.train_vscr(model, n_epochs=ep, batch=48, p_range=(0.02, 0.15), noise='depolarizing',
                     lr=lr, lam=0.0, verbose=False)
    cf, w = cond_fid(model)
    worst_cf = float(np.where(w > 1e-3, cf, 10.0).min())
    print(f"finetune {ep}@{lr}: val={h['fidelity'][-1]:.4f} worst_cf={worst_cf:.4f} ({time.time()-t0:.0f}s)")

with torch.no_grad():
    R_all = m.recovery_unitary_batch(model())
rng = np.random.RandomState(11)
for p in [0.04, 0.07, 0.10, 0.15, 0.20]:
    f_v, f_d = [], []
    for _ in range(30):
        psi = m.random_logical_state(1, rng); pe = m.encode(psi)
        rho = m.noisy_state(pe, p, 'depolarizing')
        _, Fv = m.vscr_fidelity(pe, rho, R_all, lam=0.0)
        f_v.append(float(Fv))
        f_d.append(float(m.perfect_code_decoder_fidelity(pe, rho)))
    print(f"p={p}: VSCR {np.mean(f_v):.4f}   dec {np.mean(f_d):.4f}")
print("PROBE15 DONE")
