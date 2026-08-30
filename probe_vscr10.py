"""Probe 10: correct diagnostic -- per-syndrome fidelity gap & subspace action distance."""
import time, numpy as np, torch
import ssvr_qec as m

torch.manual_seed(m.SEED); np.random.seed(m.SEED)
t0 = time.time()
model = m.VSCR()
for ep, lr, pr in [(600, 3e-2, (0.02, 0.08)), (400, 3e-2, (0.02, 0.15)),
                   (1500, 1e-3, (0.02, 0.15))]:
    h = m.train_vscr(model, n_epochs=ep, batch=48, p_range=pr, noise='depolarizing',
                     lr=lr, lam=0.0, verbose=False)
print(f"trained: val={h['fidelity'][-1]:.4f} ({time.time()-t0:.0f}s)")

with torch.no_grad():
    R_all = m.recovery_unitary_batch(model())
C_all = torch.stack(m.C_SYNDS)

# subspace action distance: ||(R_s - C_s) P_s||_F / ||P_s||_F
print("per-syndrome subspace action distance (0 = acts like decoder on Im P_s):")
for s in range(16):
    d = torch.norm((R_all[s] - C_all[s]) @ m.P_SYNDS[s]) / torch.norm(m.P_SYNDS[s])
    print(f"  s={s:04b}  {float(d):.4f}")

# per-syndrome mean fidelity gap at p=0.07 over 200 samples
rng = np.random.RandomState(5)
gap = np.zeros(16); tot_v = 0.0; tot_d = 0.0
M = 200
for _ in range(M):
    psi = m.random_logical_state(1, rng); pe = m.encode(psi)
    rho = m.noisy_state(pe, 0.07, 'depolarizing')
    rho16 = rho.unsqueeze(0).expand(16, m.DIM, m.DIM)
    PRP = torch.bmm(torch.bmm(m.P_SYNDS_STACK, rho16), m.P_SYNDS_STACK)
    pc = pe.conj()
    def Fs(R):
        Y = torch.bmm(torch.bmm(R, PRP), R.conj().transpose(1, 2))
        return torch.einsum('i,sij,j->s', pc, Y, pc.conj() if False else pe).real
    fv = Fs(R_all); fd = Fs(C_all)
    gap += (fd - fv).numpy()
    tot_v += float(fv.sum()); tot_d += float(fd.sum())
print(f"\np=0.07  mean F: VSCR {tot_v/M:.4f}  decoder {tot_d/M:.4f}")
print("per-syndrome mean gap (decoder - VSCR):")
for s in range(16):
    print(f"  s={s:04b}  {gap[s]/M:+.5f}")
print("PROBE10 DONE")
