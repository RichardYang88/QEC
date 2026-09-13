import sys, os, time
sys.path.insert(0, '/home/yqc/github/QEC')
nt = int(os.environ.get('NT', '24'))
import torch
torch.set_num_threads(nt)
import numpy as np
import ssvr_qec as m
import vscr_paper as vp
W = m.syndrome_bases()
phi = (torch.randn(16, m.PHI_DIM, dtype=torch.float64) * 0.5).requires_grad_(True)
Q = m.quad_cache((0.02, 0.15), 'amplitude_damping')

def bench(fn, n=100):
    for _ in range(5): fn()
    t0 = time.time()
    for _ in range(n): fn()
    return (time.time() - t0) / n * 1e3

def step():
    m.vscr_fidelity_quad(phi, Q)[0].backward()
    phi.grad = None

print('threads=%2d  action_cols no_grad %6.2f ms | full loss fwd+bwd %6.2f ms'
      % (nt, bench(lambda: m.recovery_action_cols(phi.detach(), W), 200), bench(step)),
      flush=True)
