"""Decisive test: is the headroom reachable by the ansatz at all?

The warm start sits at an exact saddle (grad = 0), so instead of perturbing it
we optimise ONE branch's 60 ansatz angles from RANDOM starts and compare with
the certified closed-form optimum for that branch.  If a random start reaches
the optimum, the ansatz is expressive and the ONLY obstacle is the saddle
initialisation -- which makes "warm start from the closed-form optimum" a
complete fix.
"""
import sys

sys.path.insert(0, '/home/yqc/github/QEC')
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import ssvr_qec as m                                        # noqa: E402
import vscr_paper as vp                                     # noqa: E402
import vscr_paper_abl as ab                                 # noqa: E402

NOISE = sys.argv[1] if len(sys.argv) > 1 else 'amplitude_damping'
P = float(sys.argv[2]) if len(sys.argv) > 2 else 0.06
NTRY = int(sys.argv[3]) if len(sys.argv) > 3 else 12
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3000

u = ab.opt_unitary_ceiling(NOISE, P)
cf_dec, cf_unit = u['cf_dec'], u['cf_unit']
R_dec = m.recovery_unitary_batch(
    torch.tensor(np.asarray(vp.PHI_DEC, dtype=float), dtype=torch.float64))

# the branch with the largest certified unitary headroom
s_star = int(np.argmax(np.array(cf_unit) - np.array(cf_dec)))
print('channel=%s p=%.3f -> worst branch s=%d' % (NOISE, P, s_star))
print('  cf_dec[s]=%.12f  cf_unit[s]=%.12f  branch headroom=%.4e'
      % (cf_dec[s_star], cf_unit[s_star],
         cf_unit[s_star] - cf_dec[s_star]), flush=True)

Q_np, const, p_s = ab._branch_qform(NOISE, P, s_star)
Qt = torch.tensor(np.asarray(Q_np, dtype=complex), dtype=torch.complex128)
VcT = torch.tensor(np.asarray(ab.V_ISO).conj().T, dtype=torch.complex128)
Wc = torch.tensor(np.asarray(ab.W_BASIS[s_star]), dtype=torch.complex128)
const = float(const); p_s = float(p_s)


def branch_cf(phi_t):
    """Exact conditional fidelity of branch s as a differentiable fn of phi_s."""
    R = m.recovery_unitary(phi_t)
    G = VcT @ R @ Wc
    g = G.reshape(4)
    return (torch.vdot(g, Qt @ g).real + const) / 6.0 / p_s


# sanity: the decoder's own angles must reproduce cf_dec[s]
with torch.no_grad():
    chk = float(branch_cf(torch.tensor(
        np.asarray(vp.PHI_DEC, dtype=float)[s_star], dtype=torch.float64)))
print('  check: branch_cf(PHI_DEC[%d]) = %.12f  vs cf_dec = %.12f  (diff %.2e)'
      % (s_star, chk, cf_dec[s_star], chk - cf_dec[s_star]), flush=True)

print('\n%-6s %16s %16s %12s' % ('try', 'cf(ansatz)', 'cf_unit optimum',
                                 'reached'))
best = (-np.inf, None)
rng = np.random.RandomState(0)
for t in range(NTRY):
    phi0 = (np.asarray(vp.PHI_DEC, dtype=float)[s_star]
            + (0.0 if t == 0 else rng.randn(m.PHI_DIM) * (0.3 if t % 2 else 1.0)))
    phi = torch.tensor(np.asarray(phi0, dtype=float),
                       dtype=torch.float64).clone().requires_grad_(True)
    opt = torch.optim.Adam([phi], lr=1e-2)
    for _ in range(STEPS):
        opt.zero_grad()
        (-branch_cf(phi)).backward()          # ascend the branch fidelity
        opt.step()
    with torch.no_grad():
        cf = float(branch_cf(phi))
    if cf > best[0]:
        best = (cf, phi.detach().cpu().numpy())
    tgt = cf_unit[s_star]
    print('%-6d %16.12f %16.12f %11.4f%%'
          % (t, cf, tgt, 100 * (cf - cf_dec[s_star]) / max(tgt - cf_dec[s_star],
                                                           1e-300)), flush=True)

print('\nBEST ansatz cf = %.12f' % best[0])
print('certified unitary optimum = %.12f   (diff %+.3e)'
      % (cf_unit[s_star], best[0] - cf_unit[s_star]))
head = cf_unit[s_star] - cf_dec[s_star]
print('branch headroom captured from a RANDOM start = %.2f%%'
      % (100 * (best[0] - cf_dec[s_star]) / head if head > 1e-15 else float('nan')))
np.save('/tmp/phi_branch_%s_%d.npy' % (NOISE, s_star), best[1])
