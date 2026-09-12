"""
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
"""
import json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ssvr_qec as m

os.makedirs('paper/figures', exist_ok=True)
FIGDIR = 'paper/figures'
RESULTS_NPZ = 'vscr_paper_results.npz'
SEEDS = (1234, 2024)
P_VALUES = np.array([0.005, 0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30])
N_TEST = 200

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'font.family': 'sans-serif', 'axes.linewidth': 0.8,
    'figure.dpi': 150, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

# ---------------------------------------------------------------------
# Decoder warm-start angles
# ---------------------------------------------------------------------
def decoder_angles():
    """(16, PHI_DIM) angles realising R_s = C_s (Pauli decoder) up to phase.
    Layer-1 slots per qubit q: [Rz(3q), Rx(3q+1), Rz(3q+2)]; X -> Rx(pi),
    Z -> Rz(pi), Y -> Rz(pi)Rx(pi) (phase-irrelevant)."""
    phi = np.zeros((16, m.PHI_DIM))
    for s, bits in enumerate(m.SYND_BITS):
        corr = m.SYND_TABLE[bits]
        for q, ch in enumerate(corr):
            if ch == 'X':
                phi[s, 3 * q + 1] = np.pi
            elif ch == 'Z':
                phi[s, 3 * q] = np.pi
            elif ch == 'Y':
                phi[s, 3 * q] = np.pi
                phi[s, 3 * q + 1] = np.pi
    return torch.tensor(phi, dtype=torch.float64)


PHI_DEC = decoder_angles()


def _verify_warm_start():
    R = m.recovery_unitary_batch(PHI_DEC)
    C = torch.stack(m.C_SYNDS).to(torch.complex128)
    for s in range(16):
        nz = C[s].abs() > 0.5
        ph = R[s][nz] / C[s][nz]
        assert torch.allclose(ph, ph[0].expand_as(ph), atol=1e-12), s
    rng = np.random.RandomState(0)
    for noise in ('depolarizing', 'amplitude_damping', 'mixed'):
        for p in (0.05, 0.1, 0.3):
            psi = m.encode(m.random_logical_state(1, rng))
            rho = m.noisy_state(psi, p, noise)
            Fd = float(m.perfect_code_decoder_fidelity(psi, rho))
            Fw = float(m.vscr_fidelity(psi, rho, R, lam=0.0)[1])
            assert abs(Fd - Fw) < 1e-12, (noise, p, Fd, Fw)
    print('[warm-start] R_s == C_s (phase-exact); fidelity == decoder: VERIFIED',
          flush=True)



class VSCRWarm(nn.Module):
    """VSCR with decoder warm start: phi_s = PHI_DEC[s] + phi_base + hnet(emb_s).
    The hypernetwork output layer is zero-initialised so the model starts
    exactly at the Pauli decoder; training learns continuous refinements."""

    def __init__(self, n_synd=16, phi_dim=m.PHI_DIM, hidden=64, ctx=24):
        super().__init__()
        # clone: register_buffer stores a REFERENCE; zeroing this buffer in an
        # ablation must never mutate the global PHI_DEC.
        self.register_buffer('phi_dec', PHI_DEC.clone())
        self.synd_emb = nn.Parameter(torch.randn(n_synd, ctx, dtype=torch.float64) * 0.1)
        self.hnet = nn.Sequential(
            nn.Linear(ctx, hidden, dtype=torch.float64), nn.Tanh(),
            nn.Linear(hidden, phi_dim, dtype=torch.float64))
        nn.init.zeros_(self.hnet[0].weight); nn.init.zeros_(self.hnet[0].bias)
        nn.init.zeros_(self.hnet[2].weight); nn.init.zeros_(self.hnet[2].bias)
        self.phi_base = nn.Parameter(torch.zeros(phi_dim, dtype=torch.float64))

    def forward(self):
        return self.phi_dec + self.phi_base + self.hnet(self.synd_emb)


# Warm-start curriculum (gentler lr than cold start: the model begins exactly
# at the Pauli decoder and only needs continuous refinement).
WARM_SCHEDULES = {
    'depolarizing': [(400, 3e-3, (0.02, 0.08)),
                     (500, 3e-3, (0.02, 0.15)),
                     (800, 3e-4, (0.02, 0.15))],
    'amplitude_damping': [(400, 3e-3, (0.01, 0.08)),
                          (500, 3e-3, (0.01, 0.15)),
                          (800, 3e-4, (0.01, 0.15))],
    'mixed': [(400, 3e-3, (0.02, 0.08)),
              (500, 3e-3, (0.02, 0.12)),
              (800, 3e-4, (0.02, 0.15))],
}


def train_warm_best(noise, seeds=SEEDS, verbose=True):
    """Multi-seed warm training with the same LABEL-FREE selection as the
    prototype: rank by worst per-syndrome conditional fidelity."""
    schedule = WARM_SCHEDULES[noise]
    trs, infos = [], []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = VSCRWarm()
        hist_all = {'epoch': [], 'fidelity': [], 'loss': []}
        off = 0
        for ep, lr, pr in schedule:
            hist = m.train_vscr(model, n_epochs=ep, batch=48, p_range=pr,
                                noise=noise, lr=lr, lam=0.0, rng_seed=seed,
                                use_real_grad=True, verbose=False)
            hist_all['epoch'] += [off + e for e in hist['epoch']]
            hist_all['fidelity'] += hist['fidelity']
            hist_all['loss'] += hist['loss']
            off += ep
        cf, w = m.syndrome_conditional_fidelity(model, noise)
        mask = w > 1e-3
        min_cf = float(np.where(mask, cf, 10.0).min())
        val_fid = hist_all['fidelity'][-1]
        infos.append({'seed': seed, 'min_cf': min_cf, 'val_fid': val_fid})
        if verbose:
            print(f'    [{noise}] seed {seed}: val_fid={val_fid:.4f} '
                  f'min syndrome cf={min_cf:.4f}', flush=True)
        trs.append((min_cf, val_fid, seed, model, hist_all))
    pool = [t for t in trs if t[1] >= 0.85] or [max(trs, key=lambda t: t[1])]
    best = max(pool, key=lambda t: (t[0], t[1]))
    return best[3], best[4], infos



# ---------------------------------------------------------------------
# Physical (PSD-projected) LinDR baseline
# ---------------------------------------------------------------------
def _psd_project(rho):
    rho = 0.5 * (rho + rho.conj().T)
    ev, V = torch.linalg.eigh(rho)
    ev = ev.clamp_min(0.0)
    tr = ev.sum()
    if float(tr) < 1e-12:
        return rho
    return (V * ev) @ V.conj().T / tr


def train_lindr_phys(n_train=1500, p_range=(0.01, 0.12), noise='depolarizing',
                     n_val=200, rng_seed=m.SEED + 99):
    """Same ridge fit as ssvr_qec.train_lindr but the ridge strength is
    selected on PHYSICALLY PROJECTED (PSD, trace-1) validation fidelity."""
    rng = np.random.RandomState(rng_seed)

    def gen(n):
        Xs, Ys, PS = [], [], []
        for _ in range(n):
            psi = m.random_logical_state(1, rng)
            pe = m.encode(psi)
            p = rng.uniform(*p_range)
            Xs.append(m._pauli_features(m.noisy_state(pe, p, noise)).numpy())
            Ys.append(m._pauli_features(torch.outer(pe, pe.conj())).numpy())
            PS.append(pe)
        return np.stack(Xs), np.stack(Ys), torch.stack(PS)

    X, Y, _ = gen(n_train)
    Vx, _, Vpsi = gen(n_val)
    XtX, XtY = X.T @ X, X.T @ Y
    best_f, best_B = -1.0, None
    for a in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        B = np.linalg.solve(XtX + (a * n_train + 1e-6) * np.eye(m.N_PAULI), XtY)
        Yh = Vx @ B
        Yh[:, 0] = 1.0
        coeffs = torch.tensor(Yh, dtype=torch.float64).to(torch.complex128)
        rho = torch.einsum('mk,kij->mij', coeffs, m.PAULI_STACK) / m.INV_DIM
        F = [float((Vpsi[i].conj() @ _psd_project(rho[i]) @ Vpsi[i]).real)
             for i in range(rho.shape[0])]
        f = float(np.mean(F))
        if f > best_f:
            best_f, best_B = f, B
    return torch.tensor(best_B.T.copy(), dtype=torch.float64)


def lindr_phys_fidelity(psi_enc, rho_noisy, A):
    y = A @ m._pauli_features(rho_noisy)
    y = y.clone(); y[0] = 1.0
    rho = m._reconstruct_state(y.to(torch.complex128))
    rho = _psd_project(rho)
    return float((psi_enc.conj() @ rho @ psi_enc).real)


# ---------------------------------------------------------------------
# Paper evaluation: all methods, 200 test states, mean +/- sem
# ---------------------------------------------------------------------
PAPER_METHODS = ['Raw', 'Perfect-code decoder', 'ZNE (oracle)', 'VD (oracle)',
                 'LinDR-phys (oracle)', 'VSCR cold-start', 'VSCR warm (ours)']
PAPER_COLORS = {
    'Raw': '#9e9e9e', 'Perfect-code decoder': '#e41a1c', 'ZNE (oracle)': '#ff7f00',
    'VD (oracle)': '#984ea3', 'LinDR-phys (oracle)': '#4daf4a',
    'VSCR cold-start': '#a65628', 'VSCR warm (ours)': '#1f4e9c',
}
PAPER_MARKERS = {
    'Raw': 'v', 'Perfect-code decoder': '^', 'ZNE (oracle)': 's',
    'VD (oracle)': 'D', 'LinDR-phys (oracle)': 'P',
    'VSCR cold-start': 'x', 'VSCR warm (ours)': 'o',
}


def evaluate_paper(model_warm, A_lindr, p_values, noise, n_test=N_TEST,
                   seed=m.SEED + 7, cold_R=None):
    """method -> {'F': mean, 'F_sem': sem, 'LER': mean, 'LER_sem': sem}."""
    rng = np.random.RandomState(seed)
    test_states = [m.random_logical_state(1, rng) for _ in range(n_test)]
    names = [x for x in PAPER_METHODS if not (x == 'VSCR cold-start' and cold_R is None)]
    out = {nm: {'F': [], 'F_sem': [], 'LER': [], 'LER_sem': []} for nm in names}
    with torch.no_grad():
        R_all = m.recovery_unitary_batch(model_warm())
    for p in p_values:
        fids = {nm: [] for nm in names}
        for psi in test_states:
            pe = m.encode(psi)
            rho_n = m.noisy_state(pe, p, noise)
            fids['Raw'].append(float(m.baseline_raw(pe, rho_n)))
            fids['Perfect-code decoder'].append(
                float(m.perfect_code_decoder_fidelity(pe, rho_n)))
            fids['ZNE (oracle)'].append(float(m.zne_fidelity(pe, p, noise)))
            fids['VD (oracle)'].append(
                float(m.virtual_distillation_fidelity(pe, rho_n)))
            fids['LinDR-phys (oracle)'].append(lindr_phys_fidelity(pe, rho_n, A_lindr))
            if cold_R is not None:
                with torch.no_grad():
                    _, Fc = m.vscr_fidelity(pe, rho_n, cold_R, lam=0.0)
                fids['VSCR cold-start'].append(float(Fc))
            with torch.no_grad():
                _, Fw = m.vscr_fidelity(pe, rho_n, R_all, lam=0.0)
            fids['VSCR warm (ours)'].append(float(Fw))
        for nm in names:
            arr = np.array(fids[nm])
            ler = (arr < 0.9).astype(float)
            out[nm]['F'].append(float(arr.mean()))
            out[nm]['F_sem'].append(float(arr.std(ddof=1) / math.sqrt(n_test)))
            out[nm]['LER'].append(float(ler.mean()))
            out[nm]['LER_sem'].append(float(ler.std(ddof=1) / math.sqrt(n_test)))
    return out



# ---------------------------------------------------------------------
# Per-syndrome conditional-fidelity tables
# ---------------------------------------------------------------------
def cf_table(R_all, noise, p, M=400, rng_seed=m.SEED + 5):
    """Label-free per-syndrome conditional fidelity cf_s and weights w_s."""
    rng = np.random.RandomState(rng_seed)
    Fs = np.zeros(16); Ws = np.zeros(16)
    with torch.no_grad():
        for _ in range(M):
            psi = m.random_logical_state(1, rng)
            pe = m.encode(psi)
            rho = m.noisy_state(pe, p, noise)
            rho16 = rho.unsqueeze(0).expand(16, m.DIM, m.DIM)
            PRP = torch.bmm(torch.bmm(m.P_SYNDS_STACK, rho16), m.P_SYNDS_STACK)
            Y = torch.bmm(torch.bmm(R_all, PRP), R_all.conj().transpose(1, 2))
            Fs += torch.einsum('i,sij,j->s', pe.conj(), Y, pe).real.numpy()
            Ws += torch.einsum('sij,ji->s', m.P_SYNDS_STACK, rho).real.numpy()
    return Fs / np.maximum(Ws, 1e-12), Ws / M


def save_fig(fig, name):
    fig.savefig(f'{FIGDIR}/{name}.pdf')
    fig.savefig(f'{FIGDIR}/{name}.png')
    plt.close(fig)
    print(f'  figure: {FIGDIR}/{name}.pdf', flush=True)


def plot_benchmark(curves_by_noise, fname='fig_sim_benchmark', p_vals=None):
    p_vals = P_VALUES if p_vals is None else p_vals
    titles = {'depolarizing': 'Depolarizing', 'amplitude_damping': 'Amplitude damping',
              'mixed': 'Mixed (depol. + AD)'}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.05), sharey=True)
    for ax, noise in zip(axes, ['depolarizing', 'amplitude_damping', 'mixed']):
        res = curves_by_noise[noise]
        for nm in res:
            F = np.array(res[nm]['F']); E = np.array(res[nm]['F_sem'])
            ax.plot(p_vals, F, marker=PAPER_MARKERS[nm], ms=3.5, lw=1.4,
                    color=PAPER_COLORS[nm], label=nm)
            if E.max() > 0.002:
                ax.fill_between(p_vals, F - E, F + E, color=PAPER_COLORS[nm], alpha=0.25)
        ax.set_xlabel('physical error rate $p$')
        ax.set_title(titles[noise])
        ax.grid(alpha=0.3, lw=0.5)
        ax.set_ylim(0.25, 1.03)
    axes[0].set_ylabel('average recovery fidelity $\\bar{F}$')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.145), columnspacing=1.4,
               handlelength=1.8, fontsize=7.5)
    save_fig(fig, fname)


def plot_ler(curves_by_noise, fname='fig_sim_ler', p_vals=None):
    p_vals = P_VALUES if p_vals is None else p_vals
    titles = {'depolarizing': 'Depolarizing', 'amplitude_damping': 'Amplitude damping',
              'mixed': 'Mixed (depol. + AD)'}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, noise in zip(axes, ['depolarizing', 'amplitude_damping', 'mixed']):
        res = curves_by_noise[noise]
        for nm in res:
            L = np.array(res[nm]['LER'])
            ax.semilogy(p_vals, np.maximum(L, 3e-4), marker=PAPER_MARKERS[nm],
                        ms=3.5, lw=1.4, color=PAPER_COLORS[nm], label=nm)
        ax.set_xlabel('physical error rate $p$')
        ax.set_title(titles[noise])
        ax.grid(alpha=0.3, lw=0.5, which='both')
    axes[0].set_ylabel('logical error rate  ($F<0.9$)')
    save_fig(fig, fname)


def plot_training(hist_warm, fname='fig_training'):
    """Warm curve (this run) + cold real-grad + cold random-walk (v1 npz)."""
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    old = np.load('vscr_results.npz', allow_pickle=True)
    ax.plot(old['hist_rand_fid'], color='#a65628', lw=1.2, ls='--',
            label='cold start, random updates')
    ax.plot(old['hist_real_fid'], color='#e08214', lw=1.2,
            label='cold start, Adam gradients (v1)')
    ax.plot(hist_warm['fidelity'], color='#1f4e9c', lw=1.6,
            label='decoder warm start (ours)')
    ax.set_xlabel('epoch'); ax.set_ylabel('validation fidelity ($p=0.06$)')
    ax.set_ylim(0.5, 1.02); ax.grid(alpha=0.3, lw=0.5)
    ax.legend(loc='lower right', fontsize=7)
    save_fig(fig, fname)


def plot_branch_cf(cf_dec, cf_cold, cf_warm, fname='fig_branch_cf'):
    labels = [f'{s}\n{m.SYND_TABLE[m.SYND_BITS[s]]}' for s in range(16)]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    x = np.arange(16); bw = 0.27
    ax.bar(x - bw, cf_dec, bw, color='#e41a1c', label='perfect-code decoder')
    if cf_cold is not None:
        ax.bar(x, cf_cold, bw, color='#a65628', label='VSCR cold-start (v1)')
    ax.bar(x + bw, cf_warm, bw, color='#1f4e9c', label='VSCR warm (ours)')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6)
    ax.set_xlabel('syndrome $s$ / decoder correction $C_s$')
    ax.set_ylabel('conditional fidelity $\\mathrm{cf}_s$')
    ax.set_ylim(0, 1.08); ax.grid(alpha=0.3, lw=0.5, axis='y')
    ax.legend(fontsize=7, loc='lower left', ncol=3)
    save_fig(fig, fname)



# ---------------------------------------------------------------------
# Hardware-facing exact numbers with the NEW (paper) depolarizing angles
# ---------------------------------------------------------------------
def hardware_frame_numbers(phi_warm_dep):
    """Exact ideal reference for the future full QPU benchmark, computed with
    the NEW warm angles: for psi in {0,+} and all 16 single-error Pauli
    frames, the per-frame circuit fidelity F = sum_s F_s and the per-branch
    F_s (the quantity the hardware post-selection estimates).  Also
    re-verifies circuit == density-matrix simulator to <1e-9."""
    import qcloud_vscr_new as qc
    qc.setup()                                   # synthesise + verify encoder
    phi = np.asarray(phi_warm_dep, dtype=float)
    frames = [m.SYND_TABLE[b] for b in m.SYND_BITS]   # IIIII + 15 single errors
    out = {'frames': frames, 'per_frame': {}, 'worst_circuit_vs_simulator': 0.0}
    worst = 0.0
    for psi in ['0', '+']:
        for fr in frames:
            Fc, per_s = qc.circuit_fidelity_exact(psi, fr, phi)
            Fs = qc.simulator_fidelity_frame(psi, fr, phi)
            worst = max(worst, abs(Fc - Fs))
            out['per_frame'][f'{psi}|{fr}'] = {'F_circuit': Fc, 'F_sim': Fs,
                                               'per_s': {str(k): v for k, v in per_s.items()}}
    out['worst_circuit_vs_simulator'] = worst
    assert worst < 1e-9, worst
    print(f'[hardware-ref] circuit==simulator verified, worst diff {worst:.2e}',
          flush=True)
    return out



SHORT = {'depolarizing': 'dep', 'amplitude_damping': 'ad', 'mixed': 'mixed'}
LINDR_RANGES = {'depolarizing': (0.01, 0.12),
                'amplitude_damping': (0.01, 0.08),
                'mixed': (0.01, 0.10)}


def main():
    t0 = time.time()
    _verify_warm_start()

    cold_phi = np.load('vscr_angles_dep.npz')['phi']
    with torch.no_grad():
        cold_R = m.recovery_unitary_batch(torch.tensor(cold_phi, dtype=torch.float64))
    R_dec = m.recovery_unitary_batch(PHI_DEC)

    curves, hists, models, infos_all = {}, {}, {}, {}
    for noise in ['depolarizing', 'amplitude_damping', 'mixed']:
        print(f'\n=== {noise}: training warm VSCR ({len(SEEDS)} seeds) ===', flush=True)
        model, hist, infos = train_warm_best(noise)
        models[noise], hists[noise], infos_all[noise] = model, hist, infos
        with torch.no_grad():
            phi = model().cpu().numpy()
        np.savez(f'vscr_angles_paper_{SHORT[noise]}.npz', phi=phi,
                 val_fidelity=np.array([hist['fidelity'][-1]]),
                 infos=json.dumps(infos))
        print(f'  fitting physical LinDR ({noise}) ...', flush=True)
        A = train_lindr_phys(noise=noise, p_range=LINDR_RANGES[noise])
        print(f'  evaluating ({noise}, n_test={N_TEST}) ...', flush=True)
        curves[noise] = evaluate_paper(
            model, A, P_VALUES, noise,
            cold_R=cold_R if noise == 'depolarizing' else None)

    # random-walk ablation (depolarizing, warm init, same schedule)
    print('\n=== ablation: random-walk updates from warm init ===', flush=True)
    torch.manual_seed(1234); np.random.seed(1234)
    rw_model = VSCRWarm()
    hist_rw = {'epoch': [], 'fidelity': [], 'loss': []}
    off = 0
    for ep, lr, pr in WARM_SCHEDULES['depolarizing']:
        h = m.train_vscr(rw_model, n_epochs=ep, batch=48, p_range=pr,
                         noise='depolarizing', lr=lr, lam=0.0, rng_seed=1234,
                         use_real_grad=False, verbose=False)
        hist_rw['epoch'] += [off + e for e in h['epoch']]
        hist_rw['fidelity'] += h['fidelity']
        hist_rw['loss'] += h['loss']
        off += ep

    # per-syndrome conditional fidelities (depolarizing)
    with torch.no_grad():
        R_warm_dep = m.recovery_unitary_batch(models['depolarizing']())
    cf = {}
    for tag, p in [('05', 0.05), ('15', 0.15)]:
        cf[f'dec_{tag}'], w = cf_table(R_dec, 'depolarizing', p)
        cf[f'cold_{tag}'], _ = cf_table(cold_R, 'depolarizing', p)
        cf[f'warm_{tag}'], _ = cf_table(R_warm_dep, 'depolarizing', p)
        if tag == '15':
            cf['w15'] = w

    print('\n=== figures ===', flush=True)
    plot_benchmark(curves)
    plot_ler(curves)
    plot_training(hists['depolarizing'])
    plot_branch_cf(cf['dec_15'], cf['cold_15'], cf['warm_15'])

    hw = hardware_frame_numbers(models['depolarizing']().detach().cpu().numpy())
    _save_all(curves, hists, hist_rw, cf, infos_all, hw, t0)


def _save_all(curves, hists, hist_rw, cf, infos_all, hw, t0):
    np.savez(RESULTS_NPZ,
             p_values=P_VALUES,
             curves={n: curves[n] for n in curves},
             hist_warm_dep=np.array(hists['depolarizing']['fidelity']),
             hist_warm_ad=np.array(hists['amplitude_damping']['fidelity']),
             hist_warm_mx=np.array(hists['mixed']['fidelity']),
             hist_randwalk=np.array(hist_rw['fidelity']),
             cf_dec_05=cf['dec_05'], cf_cold_05=cf['cold_05'], cf_warm_05=cf['warm_05'],
             cf_dec_15=cf['dec_15'], cf_cold_15=cf['cold_15'], cf_warm_15=cf['warm_15'],
             syndrome_weights_15=cf['w15'], infos=json.dumps(infos_all))

    i10 = int(np.where(P_VALUES == 0.10)[0][0])
    numbers = {
        'runtime_s': time.time() - t0,
        'infos': infos_all,
        'dep_p010': {nm: curves['depolarizing'][nm]['F'][i10]
                     for nm in curves['depolarizing']},
        'ad_p010': {nm: curves['amplitude_damping'][nm]['F'][i10]
                    for nm in curves['amplitude_damping']},
        'mx_p010': {nm: curves['mixed'][nm]['F'][i10]
                    for nm in curves['mixed']},
        'dep_full_curves': {nm: curves['depolarizing'][nm]['F']
                            for nm in curves['depolarizing']},
        'ad_full_curves': {nm: curves['amplitude_damping'][nm]['F']
                           for nm in curves['amplitude_damping']},
        'mx_full_curves': {nm: curves['mixed'][nm]['F'] for nm in curves['mixed']},
        'min_cf_dep_p015': {'decoder': float(np.min(cf['dec_15'][cf['w15'] > 1e-3])),
                            'cold_v1': float(np.min(cf['cold_15'][cf['w15'] > 1e-3])),
                            'warm': float(np.min(cf['warm_15'][cf['w15'] > 1e-3]))},
        'min_cf_dep_p005': {'decoder': float(np.min(cf['dec_05'])),
                            'cold_v1': float(np.min(cf['cold_05'])),
                            'warm': float(np.min(cf['warm_05']))},
        'hardware_frame_numbers': hw,
    }
    with open('paper_numbers.json', 'w') as f:
        json.dump(numbers, f, indent=2)

    print('\n================ PAPER NUMBERS ================')
    for k in ['dep_p010', 'ad_p010', 'mx_p010']:
        print(k, json.dumps({a: round(b, 4) for a, b in numbers[k].items()}))
    print('min_cf_dep_p015', numbers['min_cf_dep_p015'])
    print('min_cf_dep_p005', numbers['min_cf_dep_p005'])
    print(f'total runtime {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
