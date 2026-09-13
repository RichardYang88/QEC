"""Is grad_phi F(PHI_DEC) == 0 REAL, or is autograd broken?

Central finite differences are the ground truth.  Also inspects what PHI_DEC
actually contains (Pauli angles are multiples of pi/2, where an even objective
has vanishing derivative -- a genuine plateau, not a bug).
"""
import sys

sys.path.insert(0, '/home/yqc/github/QEC')
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import ssvr_qec as m                                        # noqa: E402
import vscr_paper as vp                                     # noqa: E402

NOISE = sys.argv[1] if len(sys.argv) > 1 else 'amplitude_damping'
P = float(sys.argv[2]) if len(sys.argv) > 2 else 0.06
PHI = np.asarray(vp.PHI_DEC, dtype=float)
print('channel=%s p=%s  PHI_DEC shape=%s' % (NOISE, P, PHI.shape), flush=True)

# ---- what is in PHI_DEC? -------------------------------------------------
u = PHI / (np.pi / 2)
print('PHI_DEC / (pi/2): min=%.6f max=%.6f' % (u.min(), u.max()))
print('  max deviation from an INTEGER multiple of pi/2 = %.3e'
      % np.abs(u - np.round(u)).max())
vals, cnts = np.unique(np.round(PHI, 9), return_counts=True)
print('  distinct values: %s' % np.array2string(vals, precision=6))
print('  counts         : %s' % np.array2string(cnts), flush=True)


def F_of(phi_np):
    Q = m.quad_cache((P, P), NOISE, n_p=1)
    _l, F, _ = m.vscr_fidelity_quad(
        torch.tensor(np.asarray(phi_np), dtype=torch.float64), Q)
    return float(F)


# ---- autograd vs central differences ------------------------------------
phi_t = torch.tensor(PHI, dtype=torch.float64).clone().requires_grad_(True)
Q = m.quad_cache((P, P), NOISE, n_p=1)
loss, F, _ = m.vscr_fidelity_quad(phi_t, Q)
loss.backward()
g_auto = phi_t.grad.detach().numpy()
print('\nF = %.15f   loss = %.15f   (1-F = %.15f)' % (float(F), float(loss),
                                                      1 - float(F)))
print('autograd ||g||_max = %.4e' % np.abs(g_auto).max(), flush=True)

rng = np.random.RandomState(0)
idx = [(s, k) for s in range(16) for k in rng.choice(PHI.shape[1], 6,
                                                    replace=False)]
print('\n%-10s %14s %14s %14s' % ('(s,k)', 'central-diff', 'autograd', 'ratio'))
worst = 0.0
for (s, k) in idx[:36]:
    for h in (1e-4,):
        pp = PHI.copy(); pp[s, k] += h
        pm = PHI.copy(); pm[s, k] -= h
        fd = (F_of(pp) - F_of(pm)) / (2 * h)
    ga = -g_auto[s, k]                       # g is d(loss)/dphi = -dF/dphi
    r = abs(fd) / abs(ga) if abs(ga) > 1e-30 else float('inf')
    worst = max(worst, abs(fd - ga))
    print('%-10s %14.6e %14.6e %14.3e' % ('(%d,%d)' % (s, k), fd, ga, r),
          flush=True)
print('\nworst |central-diff - autograd| = %.4e' % worst)

# ---- can F be improved at all by a single angle perturbation? -----------
print('\nsingle-angle scan (is the decoder really a plateau?):')
best = (0.0, None)
for (s, k) in idx[:24]:
    for d in (-1e-2, -1e-3, 1e-3, 1e-2):
        pp = PHI.copy(); pp[s, k] += d
        df = F_of(pp) - float(F)
        if df > best[0]:
            best = (df, (s, k, d))
print('  best single-angle gain = %+.6e at %s' % (best[0], best[1]))
