"""Train (or reuse) the depolarizing VSCR model and export its 16x60 angle
matrix phi to vscr_angles_dep.npz for the quantum-cloud hardware experiment."""
import os, torch, numpy as np
import ssvr_qec as m

OUT = 'vscr_angles_dep.npz'
if os.path.exists(OUT):
    print('exists:', OUT)
else:
    torch.manual_seed(0)
    model, hist, infos = m.train_vscr_best('depolarizing')
    with torch.no_grad():
        phi = model().cpu().numpy()
    np.savez(OUT, phi=phi, val_fidelity=np.array([hist['fidelity'][-1]]))
    print('saved', OUT, 'phi', phi.shape, 'val F', hist['fidelity'][-1], infos)
