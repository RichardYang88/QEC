"""Probe 17: fundamental ceiling -- best-possible unitary per syndrome via SVD."""
import numpy as np, torch
import ssvr_qec as m

def avg_operators(noise='depolarizing', p=0.07, M=600, seed=5):
    rng = np.random.RandomState(seed)
    A = torch.zeros((16, m.DIM, m.DIM), dtype=m.DTYPE)   # E[P_s rho P_s]
    B = torch.zeros((16, m.DIM, m.DIM), dtype=m.DTYPE)   # E[|psi><psi| on code]
    Bs = torch.zeros((16, m.DIM, m.DIM), dtype=m.DTYPE)  # per-syndrome weighted
    for _ in range(M):
        psi = m.random_logical_state(1, rng); pe = m.encode(psi)
        rho = m.noisy_state(pe, p, noise)
        Pc = torch.outer(pe, pe.conj())
        for s in range(16):
            Ps = m.P_SYNDS[s]
            A[s] += Ps @ rho @ Ps
            Bs[s] += torch.trace(Ps @ rho).real * Pc
    return A / M, Bs / M

for noise in ['depolarizing', 'amplitude_damping']:
    A, Bs = avg_operators(noise, 0.07)
    print(f"\n=== {noise}, p=0.07 ===")
    tot_opt = tot_dec = 0.0
    for s in range(16):
        w = float(torch.trace(A[s]).real)
        if w < 1e-4: continue
        # max_U Tr[U A U^dag B] = sum_i sqrt(eig(A)) sqrt(eig(B)) (sorted)
        ea = torch.linalg.eigvalsh(A[s]).clamp(min=0).sqrt()
        eb = torch.linalg.eigvalsh(Bs[s]).clamp(min=0).sqrt()
        opt = float((ea * eb.flip(0)).sum())
        C = m.C_SYNDS[s]
        dec = float(torch.trace(C @ A[s] @ C.conj().T @ m.P_code).real)
        tot_opt += opt; tot_dec += dec
        print(f"  s={s:04b}  w={w:.4f}  ceiling={opt/w:.4f}  decoder={dec/w:.4f}")
    print(f"  TOTAL: ceiling={tot_opt:.4f}  decoder={tot_dec:.4f}")
print("PROBE17 DONE")
