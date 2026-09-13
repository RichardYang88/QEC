import sys, time
sys.path.insert(0, '/home/yqc/github/QEC')
import numpy as np, torch
import ssvr_qec as m
import vscr_paper as vp
import vscr_paper_abl as ab

R = m.recovery_unitary_batch(vp.PHI_DEC)
print('per-sample fidelity spread at the DECODER warm start:')
for noise in ab.CHANNELS:
    for p in (0.05, 0.10):
        acc = []
        for k in range(64):
            psi = m.random_logical_state(1, np.random.RandomState(k))
            pe = m.encode(psi)
            rn = m.noisy_state(pe, p, noise)
            acc.append(float(m.vscr_fidelity(pe, rn, R, lam=0.0)[1]))
        acc = np.array(acc)
        print('  %-20s p=%.2f  mean=%.8f  std=%.3e  sem48=%.2e'
              % (noise, p, acc.mean(), acc.std(ddof=1), acc.std(ddof=1)/np.sqrt(48)))

print()
print('recovery_action_cols speed vs recovery_unitary_batch:')
phi = vp.PHI_DEC.clone().detach().requires_grad_(True)
psi, rho, w, meta = m.quad_cache((0.02, 0.15), 'amplitude_damping')
print('  quad_cache nodes:', psi.shape, rho.shape, 'sum w =', float(w.sum()))
for _ in range(3):
    m.vscr_fidelity_quad(phi, psi, rho, w)[0].backward(); phi.grad = None
torch.cuda.synchronize() if torch.cuda.is_available() else None
t0 = time.time()
for _ in range(50):
    m.vscr_fidelity_quad(phi, psi, rho, w)[0].backward(); phi.grad = None
print('  vscr_fidelity_quad fwd+bwd: %.2f ms' % ((time.time()-t0)*20))
t0 = time.time()
for _ in range(3):
    Ra = m.recovery_unitary_batch(phi)
    tot = 0.0
    for j in range(psi.shape[0]):
        tot = tot + float(w[j]) * m.vscr_fidelity(psi[j], rho[j], Ra, lam=0.0)[0]
print('  old quad loop (42 x vscr_fidelity): %.0f ms' % ((time.time()-t0)/3*1000))
