"""vscr_paper_coh.py — coherent-control-error channel supplement.

Trains the warm-started VSCR on a systematic coherent over-rotation channel
(identical Rx(eps) on all five data qubits; NOT a Pauli mixture) and
benchmarks it against the perfect-code Pauli decoder, the cold-started v1
model and the raw state.  For incoherent (Pauli-mixture) noise the decoder
point is a stationary point of the Haar-averaged branch-fidelity loss, so
VSCR reproduces it exactly; for coherent errors the learned non-Pauli
corrections can and do exceed rigid Pauli decoding.

Outputs: vscr_angles_paper_coh.npz, paper/figures/fig_coherent.pdf|png,
and merges coh_* fields into paper_numbers.json.
"""
import json, os, time

# Pin the native thread pools BEFORE numpy/torch are imported; see the note on
# `ssvr_qec._pin_blas_threads` for why this matters on this host.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'OPENBLAS_MAIN_FREE'):
    os.environ.setdefault(_v, '1')
# See the long note in ssvr_qec.py: this host intermittently SIGSEGV/SIGILLs in
# tiny complex128 kernels when a process is free to MIGRATE across its hybrid
# P-/E-core cluster. Importing ssvr_qec pins us to one CPU, which fixes it
# (measured: unpinned core-dumps within ~2e4 calls; pinned runs 4e5 clean).
# OPENBLAS_CORETYPE is for determinism only -- it is NOT the fix, since the
# faults reproduce under every coretype including AVX-only SANDYBRIDGE.
os.environ.setdefault('OPENBLAS_CORETYPE', 'HASWELL')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ssvr_qec as m
import vscr_paper as vp

vp.WARM_SCHEDULES['coherent'] = [(400, 3e-3, (0.02, 0.08)),
                                 (500, 3e-3, (0.02, 0.15)),
                                 (800, 3e-4, (0.02, 0.15))]


def main():
    t0 = time.time()
    print('=== coherent: training warm VSCR ===', flush=True)
    model, hist, infos = vp.train_warm_best('coherent')
    with torch.no_grad():
        phi = model().cpu().numpy()
    np.savez('vscr_angles_paper_coh.npz', phi=phi,
             val_fidelity=np.array([hist['fidelity'][-1]]),
             infos=json.dumps(infos))

    print('  fitting physical LinDR (coherent) ...', flush=True)
    A = vp.train_lindr_phys(noise='coherent', p_range=(0.01, 0.10))
    cold_phi = np.load('vscr_angles_dep.npz')['phi']
    with torch.no_grad():
        cold_R = m.recovery_unitary_batch(torch.tensor(cold_phi, dtype=torch.float64))
    print('  evaluating (coherent, n_test=200) ...', flush=True)
    res, audit = vp.evaluate_paper(model, A, vp.P_VALUES, 'coherent',
                                   cold_R=cold_R)
    # FIX C audit: this is the channel where the UNPROJECTED Richardson-ZNE
    # estimator exceeds 1, which is why ZNE is reported as 'ZNE-phys'.
    zne_over = vp.zne_unphysical_overshoot(vp.P_VALUES, 'coherent')

    with torch.no_grad():
        R_warm = m.recovery_unitary_batch(model())
    R_dec = m.recovery_unitary_batch(vp.PHI_DEC)
    cf_dec, w = vp.cf_table(R_dec, 'coherent', 0.15)
    cf_cold, _ = vp.cf_table(cold_R, 'coherent', 0.15)
    cf_warm, _ = vp.cf_table(R_warm, 'coherent', 0.15)

    # ---- figure ----
    # FIX C: y-axis capped at the physical ceiling.  VD is absent from `res`
    # because on a unitary channel rho is pure, rho^2 = rho, and the k=2
    # virtual-distillation estimator is algebraically identical to Raw --
    # `evaluate_paper` asserts that identity and drops the duplicate curve.
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    ax.axhline(1.0, color='k', lw=0.7, ls=':', zorder=0)
    for nm in res:
        F = np.array(res[nm]['F']); E = np.array(res[nm]['F_sem'])
        assert F.max() <= 1.0 + 1e-9, (nm, float(F.max()))
        ax.plot(vp.P_VALUES, F, marker=vp.PAPER_MARKERS[nm], ms=3.5, lw=1.4,
                color=vp.PAPER_COLORS[nm], label=nm)
        if E.max() > 0.002:
            ax.fill_between(vp.P_VALUES, F - E, F + E,
                            color=vp.PAPER_COLORS[nm], alpha=0.25)
    ax.set_xlabel('coherent over-rotation $\\varepsilon$ (rad)')
    ax.set_ylabel('average recovery fidelity $\\bar{F}$')
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_ylim(0.25, 1.0)
    ax.annotate('physical ceiling $\\bar{F}=1$', xy=(0.985, 1.0),
                xycoords=('axes fraction', 'data'), ha='right', va='top',
                fontsize=6.5, color='k')
    ax.legend(fontsize=6.5, loc='lower left')
    for tag in ('pdf', 'png'):
        fig.savefig(f'paper/figures/fig_coherent.{tag}')
    plt.close(fig)
    print('  figure: paper/figures/fig_coherent.pdf', flush=True)

    # ---- merge numbers ----
    num = json.load(open('paper_numbers.json'))
    i10 = int(np.where(vp.P_VALUES == 0.10)[0][0])
    num['infos_coh'] = infos
    num['coh_eval_audit'] = audit
    num['coh_zne_unphysical_overshoot'] = zne_over
    num['coh_p010'] = {nm: res[nm]['F'][i10] for nm in res}
    num['coh_full_curves'] = {nm: res[nm]['F'] for nm in res}
    num['coh_full_sem'] = {nm: res[nm]['F_sem'] for nm in res}
    num['min_cf_coh_p015'] = {'decoder': float(np.min(cf_dec[w > 1e-3])),
                              'cold_v1': float(np.min(cf_cold[w > 1e-3])),
                              'warm': float(np.min(cf_warm[w > 1e-3]))}
    json.dump(num, open('paper_numbers.json', 'w'), indent=2)

    print('\n=== COHERENT NUMBERS ===')
    print('p=0.10:', json.dumps({k: round(v, 4) for k, v in num['coh_p010'].items()}))
    print('warm-dec per p:', np.round(np.array(res['VSCR warm (ours)']['F']) -
                                      np.array(res['Perfect-code decoder']['F']), 5))
    print('min_cf_coh_p015:', num['min_cf_coh_p015'])
    print(f'runtime {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
