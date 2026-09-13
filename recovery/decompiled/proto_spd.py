import sys, time
sys.path.insert(0, '/home/yqc/github/QEC')
import numpy as np, torch
import ssvr_qec as m
import vscr_paper as vp

phi = vp.PHI_DEC.clone().detach().requires_grad_(True)
for noise in ('depolarizing', 'amplitude_damping', 'coherent', 'mixed'):
    t0 = time.time(); Q = m.quad_cache((0.02, 0.15), noise); tb = time.time() - t0
    for _ in range(3):
        m.vscr_fidelity_quad(phi, Q)[0].backward(); phi.grad = None
    t0 = time.time()
    for _ in range(20):
        m.vscr_fidelity_quad(phi, Q)[0].backward(); phi.grad = None
    tf = (time.time() - t0) / 20
    # reference: full 32x32 unitaries + 42 vscr_fidelity calls (the old quad path)
    t0 = time.time()
    Ra = m.recovery_unitary_batch(phi)
    tot = 0.0
    for j in range(Q['psi'].shape[0]):
        tot = tot + float(Q['w'][j]) * m.vscr_fidelity(Q['psi'][j], Q['rho'][j], Ra, lam=0.0)[0]
    tr = time.time() - t0
    F = float(m.vscr_fidelity_quad(phi.detach(), Q)[1])
    print('%-20s cache build %5.2fs | fast epoch %7.2f ms | old quad epoch %7.0f ms '
          '(%5.1fx) | F=%.8f' % (noise, tb, tf*1e3, tr*1e3, tr/tf, F), flush=True)
