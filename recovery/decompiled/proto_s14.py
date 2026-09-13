import os, sys, time, faulthandler
faulthandler.enable()
sys.path.insert(0, '/home/yqc/github/QEC')
import numpy as np
from scipy.optimize import minimize
import ssvr_qec as m
import vscr_paper_abl as b

NOISE, P, S = 'coherent', 0.10, 14
M, p_s = b._branch_M(NOISE, P, S)
M = np.asarray(M, dtype=np.complex128)
G_dec = b.V_ISO.conj().T @ m.C_SYNDS[S].numpy() @ b.W_BASIS[S]
Ju, G_u = b.opt_unitary_branch(NOISE, P, S)
_, const, _ = b._branch_qform(NOISE, P, S)
D = b._sdp_dual_bound(M, ntry=8)
print(f'branch s={S} p_s={p_s:.12e}')
print(f'  dual bound      = {D:.12e}   -> F_s_max <= {D/p_s:.9f}')
print(f'  Ju={Ju:.12f} const={const:.12f} -> F_unit = {(Ju+const)/6/p_s:.9f}')
print(f'  F_dec          = ?')
gd = G_dec.reshape(4); Q, _, _ = b._branch_qform(NOISE, P, S)
print(f'  F_dec          = {(float(np.real(gd@Q@gd.conj()))+const)/6/p_s:.9f}')

fu, x = b._sdp_branch(M, G_dec, G_ref=G_u)
print(f'  _sdp_branch    = {fu:.12e}   -> F_s = {fu/p_s:.9f}   gap={D-fu:+.3e}')


# score a candidate L directly
def score(L):
    xv = np.concatenate([np.asarray(L).real.reshape(16), np.asarray(L).imag.reshape(16)])
    o, _, ctr, _, cdet, _ = b._sdp_valgrad(xv, M)
    return o, ctr, cdet


def L_of_unitary(G):
    v = np.zeros(4, dtype=complex)
    for a in range(2):
        for bb in range(2):
            v[a * 2 + bb] = G[bb, a] / np.sqrt(2)
    L = np.zeros((4, 4), dtype=complex)
    L[:, 0] = v
    return L


print('\n  start-point scores (obj, c_tr, c_det):')
for nm, L in (('identity', L_of_unitary(np.eye(2, dtype=complex))),
              ('G_dec', L_of_unitary(G_dec)), ('G_u', L_of_unitary(G_u))):
    o, ctr, cdet = score(L)
    print(f'    {nm:9s} obj={o:.12e} c_tr={ctr:+.3e} c_det={cdet:+.3e} '
          f'{"FEASIBLE" if ctr > -1e-8 and cdet > -1e-10 else "infeasible"}')

# how many random restarts does it take to reach the certified optimum?
print('\n  random-restart search (feasibility-projected):')
best = -np.inf
cons = None
for trial in range(200):
    rr = np.random.RandomState(trial)
    L0 = (rr.randn(4, 4) + 1j * rr.randn(4, 4))
    # project to the largest feasible scaling: Tr_out(a^2 C) <= I/2
    C = L0 @ L0.conj().T
    S2 = np.array([[C[0, 0] + C[1, 1], C[0, 2] + C[1, 3]],
                   [C[2, 0] + C[3, 1], C[2, 2] + C[3, 3]]]).real
    a2 = 0.5 / max(float(np.linalg.eigvalsh(S2)[-1]), 1e-300)
    L0 = L0 * np.sqrt(a2) * 0.999
    o, ctr, cdet = score(L0)
    if ctr < -1e-9 or cdet < -1e-11:
        continue
    best = max(best, o)
    if trial < 5 or o > 6.0e-6:
        print(f'    trial{trial:3d} feasible-start obj={o:.9e}')
print(f'  best FEASIBLE-START objective over 200 random draws = {best:.12e}'
      f'  (dual {D:.12e})')
