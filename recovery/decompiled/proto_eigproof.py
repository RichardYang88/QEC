"""PROOF of why the warm start cannot move: J(G) = g^dag Q g is a Rayleigh
quotient, so its stationary points are EXACTLY the eigenvectors of Q.  Verify
that the Pauli decoder's vec(G_dec) is an eigenvector of every branch's Q.
"""
import sys

sys.path.insert(0, '/home/yqc/github/QEC')
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import vscr_paper as vp                                     # noqa: E402
import vscr_paper_abl as ab                                 # noqa: E402

R_dec = ab.m.recovery_unitary_batch(
    torch.tensor(np.asarray(vp.PHI_DEC, dtype=float), dtype=torch.float64))

for noise in ab.CHANNELS:
    p = 0.06
    print('=== %s p=%.2f ===' % (noise, p))
    n_eig = 0
    for s in range(16):
        Q, const, p_s = ab._branch_qform(noise, p, s)
        if p_s < 1e-13:
            continue
        Q = np.asarray(Q, dtype=complex)
        G_dec = ab.V_ISO.conj().T @ R_dec[s].detach().numpy() @ ab.W_BASIS[s]
        g = np.asarray(G_dec, dtype=complex).reshape(4)
        g = g / np.linalg.norm(g)
        Qg = Q @ g
        # residual of the eigen equation, normalised by the Rayleigh scale
        res = np.linalg.norm(Qg - (g.conj() @ Qg) * g) / max(np.linalg.norm(Qg), 1e-300)
        ev, evec = np.linalg.eigh(Q)
        ray = float(np.real(g.conj() @ Q @ g))
        top = float(ev.max())
        which = int(np.argmin(np.abs(ev - ray)))
        is_top = abs(ray - top) <= 1e-12 * max(1.0, abs(top))
        n_eig += int(res < 1e-9)
        if s < 6 or not is_top:
            print('  s=%2d  eig-residual=%.3e  lambda(dec)=%.6e  '
                  'lambda_max=%.6e  is_top=%s  gap=%.3e'
                  % (s, res, ray, top, is_top, top - ray), flush=True)
    print('  -> %d/16 branches have vec(G_dec) an exact eigenvector of Q'
          % n_eig, flush=True)
