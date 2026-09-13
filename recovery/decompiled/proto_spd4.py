import sys, os, time
sys.path.insert(0, '/home/yqc/github/QEC')
nt = int(os.environ.get('NT', '24'))
import torch
torch.set_num_threads(nt)
import numpy as np
import ssvr_qec as m
import vscr_paper as vp

def bench(fn, n=20):
    for _ in range(3): fn()
    t0 = time.time()
    for _ in range(n): fn()
    return (time.time() - t0) / n * 1e3

rng = np.random.RandomState(0)
psi = m.random_logical_state(1, rng); pe = m.encode(psi)
out = []
for noise in ('depolarizing', 'amplitude_damping', 'coherent', 'mixed'):
    out.append('%s noisy_state %.0f ms' % (noise[:4], bench(lambda: m.noisy_state(pe, 0.06, noise), 10)))
R = m.recovery_unitary_batch(vp.PHI_DEC)
mc = bench(lambda: [m.vscr_fidelity(pe, m.noisy_state(pe, 0.06, 'coherent'), R, lam=0.0)], 50)
Q = m.quad_cache((0.02, 0.15), 'amplitude_damping')
phi = vp.PHI_DEC.clone().detach().requires_grad_(True)
q = bench(lambda: (m.vscr_fidelity_quad(phi, Q)[0].backward(), setattr(phi, 'grad', None)), 100)
print('threads=%2d | %s | vscr_fidelity %.2f ms | quad loss fwd+bwd %.2f ms'
      % (nt, ' | '.join(out), mc, q), flush=True)
