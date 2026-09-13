"""Prototype of the fix: per-syndrome symmetry-broken warm start.

F̄ = Σ_s p_s · cf_s(G_s) and cf_s depends ONLY on R_s = ansatz(phi_s), so the
exact objective is SEPARABLE across the 16 syndromes.  Each block is a 4x4
Rayleigh quotient whose Pauli warm start is an eigenvector (grad = 0, a saddle).
Optimising each block independently from a few random starts therefore reaches
the global unitary optimum without ever touching the CPTP SDP.

Reports F̄(decoder), F̄(fixed), F̄(certified unitary ceiling) at p.
"""
import sys
import time

sys.path.insert(0, '/home/yqc/github/QEC')
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import ssvr_qec as m                                        # noqa: E402
import vscr_paper as vp                                     # noqa: E402
import vscr_paper_abl as ab                                 # noqa: E402

NOISE = sys.argv[1] if len(sys.argv) > 1 else 'amplitude_damping'
P = float(sys.argv[2]) if len(sys.argv) > 2 else 0.06
NSTART = int(sys.argv[3]) if len(sys.argv) > 3 else 3
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 1500

PHI_DEC = np.asarray(vp.PHI_DEC, dtype=float)
VcT = torch.tensor(np.asarray(ab.V_ISO).conj().T, dtype=torch.complex128)
Wc_all = torch.tensor(np.asarray(ab.W_BASIS), dtype=torch.complex128)

t0 = time.time()
u_all = ab.opt_unitary_ceiling(NOISE, P)
CF_CEIL = [float(x) for x in u_all['cf_unit']]
phi_new = PHI_DEC.copy()
tot_old = tot_new = tot_ceil = 0.0
print('channel=%s p=%.3f  (%d starts x %d steps per syndrome)'
      % (NOISE, P, NSTART, STEPS), flush=True)
print('\n%-3s %9s %14s %14s %14s %9s'
      % ('s', 'p_s', 'cf(dec)', 'cf(fixed)', 'cf(ceil)', 'captured'))
for s in range(16):
    Q_np, const, p_s = ab._branch_qform(NOISE, P, s)
    const = float(const); p_s = float(p_s)
    if p_s < 1e-13:
        print('%-3d %9.2e %14s %14s %14s %9s' % (s, p_s, '-', '-', '-', '-'))
        continue
    Qt = torch.tensor(np.asarray(Q_np, dtype=complex), dtype=torch.complex128)
    Wc = Wc_all[s]

    def cf(phi_t):
        G = VcT @ m.recovery_unitary(phi_t) @ Wc
        g = G.reshape(4)
        return (torch.vdot(g, Qt @ g).real + const) / 6.0 / p_s

    cf_dec = float(cf(torch.tensor(PHI_DEC[s], dtype=torch.float64)))
    # exact optimum of this block, from the production (dual-certified) routine
    cf_ceil = CF_CEIL[s]

    best, best_phi = cf_dec, PHI_DEC[s].copy()
    rng = np.random.RandomState(1000 + s)
    for t in range(NSTART):
        init = (PHI_DEC[s] if t == 0
                else PHI_DEC[s] + rng.randn(m.PHI_DIM) * (0.5 if t % 2 else 1.5))
        phi = torch.tensor(np.asarray(init, dtype=float),
                           dtype=torch.float64).clone().requires_grad_(True)
        opt = torch.optim.Adam([phi], lr=1e-2)
        for _ in range(STEPS):
            opt.zero_grad()
            (-cf(phi)).backward()
            opt.step()
        with torch.no_grad():
            v = float(cf(phi))
        if v > best:
            best, best_phi = v, phi.detach().cpu().numpy()
    phi_new[s] = best_phi
    head = cf_ceil - cf_dec
    cap = 100 * (best - cf_dec) / head if head > 1e-15 else float('nan')
    tot_old += p_s * cf_dec
    tot_new += p_s * best
    tot_ceil += p_s * cf_ceil
    print('%-3d %9.2e %14.10f %14.10f %14.10f %8.2f%%'
          % (s, p_s, cf_dec, best, cf_ceil, cap), flush=True)

f_dec = ab.exact_F(PHI_DEC, NOISE, P)[0]
f_new = ab.exact_F(phi_new, NOISE, P)[0]
f_ceil = float(u_all['F_unit'])
print('\nF̄(decoder)        = %.12f' % f_dec)
print('F̄(per-syn fixed)  = %.12f' % f_new)
print('F̄(unit ceiling)   = %.12f' % f_ceil)
print('headroom = %.4e ; captured by the fix = %.4f%%'
      % (f_ceil - f_dec,
         100 * (f_new - f_dec) / (f_ceil - f_dec) if f_ceil - f_dec > 1e-15
         else float('nan')))
print('sum_p_s check: %.12f vs %.12f' % (tot_old, f_dec))
np.save('/tmp/phi_persyn_%s.npy' % NOISE, phi_new)
print('runtime %.0fs ; saved /tmp/phi_persyn_%s.npy' % (time.time() - t0, NOISE))
