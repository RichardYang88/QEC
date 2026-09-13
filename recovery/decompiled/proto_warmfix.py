"""Can the ansatz + exact gradient reach the certified headroom at all?

PHI_DEC is an exact stationary point, so the warm start must be symmetry-broken.
Tests, for a range of perturbation scales:
  init = PHI_DEC + eps * randn  ->  minimise the exact quadrature loss (= 1 - F)
and reports how much of the CERTIFIED unitary headroom is captured.
Also checks the ansatz can express the closed-form optimum at all, by fitting
the per-branch optimal unitary G_opt and evaluating it.
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
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
PHI_DEC = np.asarray(vp.PHI_DEC, dtype=float)

f_dec = ab.exact_F(PHI_DEC, NOISE, P)[0]
u = ab.opt_unitary_ceiling(NOISE, P)
f_unit = u['F_unit']
head = f_unit - f_dec
print('channel=%s p=%.3f  F_dec=%.12f  F_unit=%.12f  headroom=%.4e'
      % (NOISE, P, f_dec, f_unit, head), flush=True)

Q = m.quad_cache((P, P), NOISE, n_p=1)


def train(init, steps=STEPS, lr=1e-2, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    phi = torch.tensor(np.asarray(init, dtype=float),
                       dtype=torch.float64).clone().requires_grad_(True)
    opt = torch.optim.Adam([phi], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss, _F, _ = m.vscr_fidelity_quad(phi, Q)
        loss.backward()                      # loss = 1 - F exactly => minimise
        opt.step()
    with torch.no_grad():
        return phi.detach().cpu().numpy()


print('\n%-10s %14s %14s %10s' % ('eps', 'F(final)', 'F-F_dec', 'captured'))
best_row = None
for eps in (0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1):
    rng = np.random.RandomState(0)
    init = PHI_DEC + eps * rng.randn(*PHI_DEC.shape)
    phi_f = train(init)
    f = ab.exact_F(phi_f, NOISE, P)[0]
    cap = (f - f_dec) / head if head > 1e-12 else float('nan')
    print('%-10.0e %14.12f %14.4e %9.2f%%' % (eps, f, f - f_dec, 100 * cap),
          flush=True)
    if best_row is None or f > best_row[1]:
        best_row = (eps, f, phi_f)

print('\nbest eps=%.0e -> F=%.12f (captured %.2f%% of the certified headroom)'
      % (best_row[0], best_row[1],
         100 * (best_row[1] - f_dec) / head if head > 1e-12 else float('nan')))

# ---- can the ansatz express the closed-form per-branch optimum? ----------
print('\nansatz expressivity: fit R_s so that V^dag R_s W_s == G_opt[s]')
G_opt = u['G_opt']
print('  G_opt type/shape: %s' % (np.shape(G_opt),))
# V_ISO is (32,2) and W_BASIS is (16,32,2), so G[s] = V^dag R_s W_s is (2,2).
Vc = torch.tensor(np.asarray(ab.V_ISO).conj(), dtype=torch.complex128)
Wc = torch.tensor(np.asarray(ab.W_BASIS), dtype=torch.complex128)
# least-squares fit of the 60 angles to reproduce G_opt on the syndrome block
rng = np.random.RandomState(1)
best_fit, best_err = None, np.inf
for trial in range(6):
    phi = torch.tensor(rng.randn(16, PHI_DEC.shape[1]) * 0.3,
                       dtype=torch.float64).requires_grad_(True)
    opt = torch.optim.Adam([phi], lr=3e-2)
    tgt = torch.tensor(np.asarray(G_opt), dtype=torch.complex128)
    for _ in range(3000):
        opt.zero_grad()
        R = m.recovery_unitary_batch(phi)
        G = torch.einsum('ia,sij,sjb->sab', Vc, R, Wc)
        err = torch.abs(G - tgt).pow(2).sum()
        err.backward(); opt.step()
    e = float(err)
    if e < best_err:
        best_err, best_fit = e, phi.detach().cpu().numpy()
print('  best ||G_fit - G_opt||^2 = %.4e over 16 branches' % best_err)
f_fit = ab.exact_F(best_fit, NOISE, P)[0]
print('  F(fitted G_opt) = %.12f   vs F_unit = %.12f   (diff %+.3e)'
      % (f_fit, f_unit, f_fit - f_unit))
np.save('/tmp/phi_best_%s.npy' % NOISE, best_row[2])
