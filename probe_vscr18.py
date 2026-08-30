"""Probe 18: correct ceiling -- max_U E[<psi|U P_s rho P_s U^dag|psi>] via trace norm."""
import numpy as np, torch
import ssvr_qec as m

def avg_ops(noise, p=0.07, M=600, seed=5):
    rng = np.random.RandomState(seed)
    Xs = torch.zeros((16, m.DIM, m.DIM), dtype=m.DTYPE)   # E[ P_s rho P_s ]
    Qs = torch.zeros((m.DIM, m.DIM), dtype=m.DTYPE)       # E[ |psi_enc><psi_enc| ]
    for _ in range(M):
        psi = m.random_logical_state(1, rng); pe = m.encode(psi)
        rho = m.noisy_state(pe, p, noise)
        Qs += torch.outer(pe, pe.conj())
        for s in range(16):
            Ps = m.P_SYNDS[s]
            Xs[s] += Ps @ rho @ Ps
    return Xs / M, Qs / M

for noise in ['depolarizing', 'amplitude_damping']:
    X, Q = avg_ops(noise, 0.07)
    print(f"\n=== {noise} p=0.07 ===")
    tot_c = tot_d = tot_w = 0.0
    for s in range(16):
        w = float(torch.trace(X[s]).real)
        if w < 1e-4: continue
        Mm = Q.sqrt() @ X[s] @ Q.sqrt()
        ceil = float(torch.linalg.norm(Mm, ord='nuc'))          # max_U Tr[U X U^dag Q]
        C = m.C_SYNDS[s]
        dec = float(torch.trace(C @ X[s] @ C.conj().T @ m.P_code).real)
        tot_c += ceil; tot_d += dec; tot_w += w
        print(f"  s={s:04b}  w={w:.4f}  ceiling={ceil/w:.4f}  decoder={dec/w:.4f}")
    print(f"  TOTAL exp fidelity: ceiling={tot_c:.4f}  decoder={tot_d:.4f}  (w_sum={tot_w:.4f})")
print("PROBE18 DONE")
