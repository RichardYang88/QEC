# Source Generated with Decompyle++
# File: vscr_paper.cpython-311.pyc (Python 3.11)

'''
vscr_paper.py — Paper-grade VSCR benchmark: decoder-warm-started variational
syndrome-conditioned recovery for the [[5,1,3]] code.

Improvements over the prototype run (full_run.log):
  1. WARM START: each syndrome branch R_s is initialised at the exact Pauli
     decoder correction C_s (hypernetwork output layer zero-initialised), so
     training starts AT perfect-decoder performance and can only refine it
     with continuous, non-Pauli corrections.  Verified |F_warm(0)-F_dec|<1e-12.
  2. PHYSICAL LinDR: the vnCDR-style linear regression baseline is projected
     onto the nearest physical (PSD, trace-1) state before scoring, removing
     the unphysical F > 1 artefacts of the prototype.
  3. ERROR BARS: evaluation over 200 Haar-random logical states per point.
  4. Cold-start VSCR (v1 hardware snapshot, vscr_angles_dep.npz) included as
     an ablation on the depolarizing channel.

Outputs:
  vscr_paper_results.npz, vscr_angles_paper_{dep,ad,mixed}.npz,
  paper_numbers.json, paper/figures/*.pdf|png

NOTE: vscr_angles_dep.npz (v1, used for the WK_C180 hardware feasibility
runs) is NEVER overwritten.
'''
import json
import math
import os
import time
import numpy as np
import torch
from torch.nn import nn
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import pyplot as plt
import ssvr_qec as m
os.makedirs('paper/figures', exist_ok = True)
FIGDIR = 'paper/figures'
RESULTS_NPZ = 'vscr_paper_results.npz'
SEEDS = (1234, 2024)
P_VALUES = np.array([
    0.005,
    0.01,
    0.02,
    0.04,
    0.07,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3])
N_TEST = 200
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'font.family': 'sans-serif',
    'axes.linewidth': 0.8,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight' })

def decoder_angles():
    '''(16, PHI_DIM) angles realising R_s = C_s (Pauli decoder) up to phase.
    Layer-1 slots per qubit q: [Rz(3q), Rx(3q+1), Rz(3q+2)]; X -> Rx(pi),
    Z -> Rz(pi), Y -> Rz(pi)Rx(pi) (phase-irrelevant).'''
    phi = np.zeros((16, m.PHI_DIM))
    for s, bits in enumerate(m.SYND_BITS):
        corr = m.SYND_TABLE[bits]
        for q, ch in enumerate(corr):
            if ch == 'X':
                phi[(s, 3 * q + 1)] = np.pi
                continue
            if ch == 'Z':
                phi[(s, 3 * q)] = np.pi
                continue
            if ch == 'Y':
                phi[(s, 3 * q)] = np.pi
                phi[(s, 3 * q + 1)] = np.pi
            return torch.tensor(phi, dtype = torch.float64)

PHI_DEC = decoder_angles()

def _verify_warm_start():
    R = m.recovery_unitary_batch(PHI_DEC)
    C = torch.stack(m.C_SYNDS).to(torch.complex128)
# WARNING: Decompyle incomplete


class VSCRWarm(nn.Module):
    pass
# WARNING: Decompyle incomplete


def _selftest_hypernet_trainable(noise, steps, batch = ('amplitude_damping', 60, 12)):
    '''Guard against re-introducing the dead-hypernetwork bug.

    Asserts (a) the warm start is exactly the decoder, (b) EVERY trainable
    parameter of the syndrome pathway moves under the real label-free loss, and
    (c) the 16 branch angle sets become mutually distinct.  Without (b)/(c) the
    model can only apply one global rotation, and both the "syndrome-conditioned
    recovery" claim and the K-dependence of the low-data ablation are vacuous.
    '''
    pass
# WARNING: Decompyle incomplete

WARM_SCHEDULES = {
    'depolarizing': [
        (400, 0.003, (0.02, 0.08)),
        (500, 0.003, (0.02, 0.15)),
        (800, 0.0003, (0.02, 0.15))],
    'amplitude_damping': [
        (400, 0.003, (0.01, 0.08)),
        (500, 0.003, (0.01, 0.15)),
        (800, 0.0003, (0.01, 0.15))],
    'mixed': [
        (400, 0.003, (0.02, 0.08)),
        (500, 0.003, (0.02, 0.12)),
        (800, 0.0003, (0.02, 0.15))] }

def train_warm_best(noise, seeds, verbose, quad = (SEEDS, True, True)):
    """Multi-seed warm training with the same LABEL-FREE selection as the
    prototype: rank by worst per-syndrome conditional fidelity.

    quad=True (default) makes the whole pipeline exact and fast:

      * every epoch's objective is the deterministic Haar x noise-strength
        quadrature of `ssvr_qec.quad_cache`, propagated through only the
        2-dimensional syndrome subspace -- 1.5 ms/epoch against 67 ms for the
        equivalent full-32x32 reference loop (47x), and the gradient has zero
        variance, which is what makes a 1.9e-4 non-Pauli headroom resolvable
        instead of being buried in a 48-sample Monte-Carlo SEM of 1.1e-4;
      * the seed-selection diagnostic uses
        `ssvr_qec.syndrome_conditional_fidelity_exact` rather than the M=150
        Monte-Carlo version, whose ~1e-3 sem is larger than the effect.

    quad=False restores the original Monte-Carlo path bit-for-bit.
    """
    pass
# WARNING: Decompyle incomplete


def _psd_project(rho):
    rho = 0.5 * (rho + rho.conj().T)
    (ev, V) = torch.linalg.eigh(rho)
    ev = ev.clamp_min(0)
    tr = ev.sum()
    if float(tr) < 1e-12:
        return rho
    return (None * ev @ V.conj().T) / tr


def train_lindr_phys(n_train, p_range, noise, n_val, rng_seed = (1500, (0.01, 0.12), 'depolarizing', 200, m.SEED + 99)):
    '''Same ridge fit as ssvr_qec.train_lindr but the ridge strength is
    selected on PHYSICALLY PROJECTED (PSD, trace-1) validation fidelity.'''
    pass
# WARNING: Decompyle incomplete


def lindr_phys_fidelity(psi_enc, rho_noisy, A):
    y = A @ m._pauli_features(rho_noisy)
    y = y.clone()
    y[0] = 1
    rho = m._reconstruct_state(y.to(torch.complex128))
    rho = _psd_project(rho)
    return float((psi_enc.conj() @ rho @ psi_enc).real)

PAPER_METHODS = [
    'Raw',
    'Perfect-code decoder',
    'ZNE-phys (oracle)',
    'VD (oracle)',
    'LinDR-phys (oracle)',
    'VSCR cold-start',
    'VSCR warm (ours)']
VD_DEGENERATE_FOR = ('coherent',)
PAPER_COLORS = {
    'Raw': '#9e9e9e',
    'Perfect-code decoder': '#e41a1c',
    'ZNE-phys (oracle)': '#ff7f00',
    'VD (oracle)': '#984ea3',
    'LinDR-phys (oracle)': '#4daf4a',
    'VSCR cold-start': '#a65628',
    'VSCR warm (ours)': '#1f4e9c' }
PAPER_MARKERS = {
    'Raw': 'v',
    'Perfect-code decoder': '^',
    'ZNE-phys (oracle)': 's',
    'VD (oracle)': 'D',
    'LinDR-phys (oracle)': 'P',
    'VSCR cold-start': 'x',
    'VSCR warm (ours)': 'o' }

def zne_phys_fidelity(psi_enc, p, noise):
    '''Richardson ZNE (scales 1,2,3) with the extrapolated operator projected
    onto the nearest physical (PSD, trace-1) state before scoring.  Identical
    treatment to LinDR-phys; without it the reported "fidelity" is an unbounded
    estimator artefact that exceeds 1 on the coherent channel.'''
    rho_zne = (3 * m.noisy_state(psi_enc, p * 1, noise) - 3 * m.noisy_state(psi_enc, p * 2, noise)) + m.noisy_state(psi_enc, p * 3, noise)
    return float((psi_enc.conj() @ _psd_project(rho_zne) @ psi_enc).real)


def zne_unphysical_overshoot(p_values, noise, n_test, seed = (60, m.SEED + 11)):
    '''Audit of the raw (unprojected) ZNE estimator: returns {p: max F} where
    F = <psi|rho_zne|psi> may exceed 1.  Recorded in paper_numbers.json so the
    reader can see exactly why the physical projection was introduced.'''
    pass
# WARNING: Decompyle incomplete


def evaluate_paper(model_warm, A_lindr, p_values, noise, n_test, seed, cold_R = (N_TEST, m.SEED + 7, None)):
    """method -> {'F': mean, 'F_sem': sem, 'LER': mean, 'LER_sem': sem}.

    Baselines are all scored on PHYSICAL states (ZNE-phys, LinDR-phys) so that
    F <= 1 for every method.  On channels listed in VD_DEGENERATE_FOR the VD
    baseline is omitted after asserting F_VD == F_Raw to machine precision
    (rho^2 = rho for a unitary channel), instead of plotting a duplicate curve.
    Returns (out, audit) where audit records the VD/Raw degeneracy check.
    """
    pass
# WARNING: Decompyle incomplete


def cf_table(R_all, noise, p, M, rng_seed = (400, m.SEED + 5)):
    '''Label-free per-syndrome conditional fidelity cf_s and weights w_s.'''
    rng = np.random.RandomState(rng_seed)
    Fs = np.zeros(16)
    Ws = np.zeros(16)
    torch.no_grad()
    for _ in range(M):
        psi = m.random_logical_state(1, rng)
        pe = m.encode(psi)
        rho = m.noisy_state(pe, p, noise)
        rho16 = rho.unsqueeze(0).expand(16, m.DIM, m.DIM)
        PRP = torch.bmm(torch.bmm(m.P_SYNDS_STACK, rho16), m.P_SYNDS_STACK)
        Y = torch.bmm(torch.bmm(R_all, PRP), R_all.conj().transpose(1, 2))
        Fs += torch.einsum('i,sij,j->s', pe.conj(), Y, pe).real.numpy()
        Ws += torch.einsum('sij,ji->s', m.P_SYNDS_STACK, rho).real.numpy()
        None(None, None)
    with None:
        if not None:
            pass
    return (Fs / np.maximum(Ws, 1e-12), Ws / M)


def save_fig(fig, name):
    fig.savefig(f'''{FIGDIR}/{name}.pdf''')
    fig.savefig(f'''{FIGDIR}/{name}.png''')
    plt.close(fig)
    print(f'''  figure: {FIGDIR}/{name}.pdf''', flush = True)


def plot_benchmark(curves_by_noise, fname, p_vals = ('fig_sim_benchmark', None)):
    pass
# WARNING: Decompyle incomplete


def plot_ler(curves_by_noise, fname, p_vals = ('fig_sim_ler', None)):
    pass
# WARNING: Decompyle incomplete


def plot_training(hist_warm, fname = ('fig_training',)):
    '''Warm curve (this run) + cold real-grad + cold random-walk (v1 npz).'''
    (fig, ax) = plt.subplots(figsize = (3.6, 2.8))
    old = np.load('vscr_results.npz', allow_pickle = True)
    ax.plot(old['hist_rand_fid'], color = '#a65628', lw = 1.2, ls = '--', label = 'cold start, random updates')
    ax.plot(old['hist_real_fid'], color = '#e08214', lw = 1.2, label = 'cold start, Adam gradients (v1)')
    ax.plot(hist_warm['fidelity'], color = '#1f4e9c', lw = 1.6, label = 'decoder warm start (ours)')
    ax.set_xlabel('epoch')
    ax.set_ylabel('validation fidelity ($p=0.06$)')
    ax.set_ylim(0.5, 1.02)
    ax.grid(alpha = 0.3, lw = 0.5)
    ax.legend(loc = 'lower right', fontsize = 7)
    save_fig(fig, fname)


def plot_branch_cf(cf_dec, cf_cold, cf_warm, fname = ('fig_branch_cf',)):
    labels = range(16)()
    (fig, ax) = plt.subplots(figsize = (7.2, 2.8))
    x = np.arange(16)
    bw = 0.27
    ax.bar(x - bw, cf_dec, bw, color = '#e41a1c', label = 'perfect-code decoder')
# WARNING: Decompyle incomplete


def hardware_frame_numbers(phi_warm_dep):
    '''Exact ideal reference for the future full QPU benchmark, computed with
    the NEW warm angles: for psi in {0,+} and all 16 single-error Pauli
    frames, the per-frame circuit fidelity F = sum_s F_s and the per-branch
    F_s (the quantity the hardware post-selection estimates).  Also
    re-verifies circuit == density-matrix simulator to <1e-9.'''
    import qcloud_vscr_new as qc
    qc.setup()
    phi = np.asarray(phi_warm_dep, dtype = float)
    frames = m.SYND_BITS()
    out = {
        'frames': frames,
        'per_frame': { },
        'worst_circuit_vs_simulator': 0 }
    worst = 0
# WARNING: Decompyle incomplete

SHORT = {
    'depolarizing': 'dep',
    'amplitude_damping': 'ad',
    'mixed': 'mixed' }
LINDR_RANGES = {
    'depolarizing': (0.01, 0.12),
    'amplitude_damping': (0.01, 0.08),
    'mixed': (0.01, 0.1) }

def main():
    pass
# WARNING: Decompyle incomplete


def _save_all(curves, hists, hist_rw, cf, infos_all, hw, t0):
    pass
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    main()
    return None
