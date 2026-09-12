"""
vscr_paper_abl.py — Same-family baselines, optimality ceilings and ablations
for the VSCR [[5,1,3]] benchmark (addresses the critique that the baseline
matrix lacks a *learned variational unitary recovery* comparator).

Experiments (label-free / simulation-only; same evaluation protocol as
vscr_paper.py: fresh Haar states per point, mean +/- sem):

  E1  VQR-ind: per-syndrome INDEPENDENT variational unitary recovery
      (16 x 60 parameter table, NO hypernetwork), decoder-warm-started,
      same unsupervised loss / optimizer / curriculum as VSCR.
      -> tests the necessity of the shared hypernetwork architecture.
  E2  Low-data regime: VSCR (hypernetwork) vs VQR-ind trained on a FIXED
      pool of K logical states (K = 2,4,8,16), evaluated on fresh states.
      -> tests whether cross-syndrome sharing gives statistical strength.
  E3  SDP ceiling: optimal CPTP recovery per syndrome branch (exact 4x4
      Choi SDP, Reimpell-Werner / Watrous form) = information-theoretic
      best over ALL recoveries (unitary or not, Pauli or not) given the
      projective syndrome readout.
      -> answers "is there ANY gain from continuous non-Pauli correction?"
  E4  No-projection ablation: a single GLOBAL variational unitary recovery
      (QVECTOR-style, Johnson et al. 2017), no syndrome measurement.
      -> quantifies the value of the projective syndrome readout.
  E5  Pauli-restricted learned recovery: exhaustive unsupervised search over
      the Pauli coset acting on each branch (4 logical classes/syndrome).
      -> proves the best Pauli-restricted recovery == perfect-code decoder,
         so VSCR strictly generalises the decoder baseline.

Outputs: vscr_paper_abl_results.npz, paper/figures/fig_ablation.{pdf,png},
merges 'abl_*' fields into paper_numbers.json.
"""
import itertools, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ssvr_qec as m
import vscr_paper as vp

SMOKE = os.environ.get('ABL_SMOKE', '0') == '1'   # tiny budgets for testing

# coherent curriculum (same as vscr_paper_coh.py)
SCHED = dict(vp.WARM_SCHEDULES)
SCHED['coherent'] = [(400, 3e-3, (0.02, 0.08)),
                     (500, 3e-3, (0.02, 0.15)),
                     (800, 3e-4, (0.02, 0.15))]
CHANNELS = ['depolarizing', 'amplitude_damping', 'mixed', 'coherent']
SHORT = {'depolarizing': 'dep', 'amplitude_damping': 'ad',
         'mixed': 'mixed', 'coherent': 'coh'}
if SMOKE:
    SCHED = {k: [(30, 3e-3, v[0][2])] for k, v in SCHED.items()}
    VP_P = np.array([0.05, 0.10, 0.15])
    N_TEST = 40
else:
    VP_P = vp.P_VALUES
    N_TEST = 200

plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10, 'savefig.dpi': 600,
                     'savefig.bbox': 'tight', 'figure.dpi': 150})


# =====================================================================
# Models
# =====================================================================
class VQRInd(nn.Module):
    """Per-syndrome independent angle table (no hypernetwork, no sharing).
    960 parameters vs 5944 for the hypernetwork model at n=5."""
    def __init__(self, warm=True):
        super().__init__()
        base = vp.PHI_DEC.clone() if warm else torch.zeros(16, m.PHI_DIM,
                                                           dtype=torch.float64)
        self.phi = nn.Parameter(base)

    def forward(self):
        return self.phi


class GlobalRec(nn.Module):
    """Single global variational unitary (NO syndrome projection);
    QVECTOR-style variational recovery baseline. 60 parameters."""
    def __init__(self):
        super().__init__()
        self.phi = nn.Parameter(torch.zeros(m.PHI_DIM, dtype=torch.float64))

    def forward(self):
        return self.phi


N_PARAMS = {'VSCR hypernet': 16 * 24 + (24 * 64 + 64) + (64 * 60 + 60) + 60,
            'VQR-ind table': 16 * m.PHI_DIM,
            'global (no proj.)': m.PHI_DIM}


# =====================================================================
# E1: independent-table training (same protocol as vp.train_warm_best)
# =====================================================================
def train_ind(noise, seeds=(1234,)):
    schedule = SCHED[noise]
    trs, infos = [], []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = VQRInd(warm=True)
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
        min_cf = float(np.where(w > 1e-3, cf, 10.0).min())
        infos.append({'seed': seed, 'min_cf': min_cf,
                      'val_fid': hist_all['fidelity'][-1]})
        print(f'    [ind/{noise}] seed {seed}: val_fid={infos[-1]["val_fid"]:.4f}'
              f' min cf={min_cf:.4f}', flush=True)
        trs.append((min_cf, infos[-1]['val_fid'], model, hist_all))
    best = max(trs, key=lambda t: (t[0], t[1]))
    return best[2], best[3], infos

# =====================================================================
# Generic curve evaluation for an arbitrary phi table / global unitary
# =====================================================================
def eval_phi_table(phi_table, noise, p_values, n_test=None, seed=m.SEED + 7):
    """F̄ ± sem of the syndrome-projected recovery R_s = ansatz(phi_s)."""
    n_test = n_test or N_TEST
    rng = np.random.RandomState(seed)
    states = [m.random_logical_state(1, rng) for _ in range(n_test)]
    with torch.no_grad():
        R_all = m.recovery_unitary_batch(torch.as_tensor(phi_table,
                                                         dtype=torch.float64))
    outF, outE = [], []
    for p in p_values:
        fs = []
        with torch.no_grad():
            for psi in states:
                pe = m.encode(psi)
                rho = m.noisy_state(pe, p, noise)
                _, F = m.vscr_fidelity(pe, rho, R_all, lam=0.0)
                fs.append(float(F))
        a = np.array(fs)
        outF.append(float(a.mean()))
        outE.append(float(a.std(ddof=1) / math.sqrt(n_test)))
    return outF, outE


def eval_global(phi, noise, p_values, n_test=None, seed=m.SEED + 7):
    """F̄ ± sem of a single global unitary (no projection)."""
    n_test = n_test or N_TEST
    rng = np.random.RandomState(seed)
    states = [m.random_logical_state(1, rng) for _ in range(n_test)]
    with torch.no_grad():
        U = m.recovery_unitary(torch.as_tensor(phi, dtype=torch.float64))
    outF, outE = [], []
    for p in p_values:
        fs = []
        with torch.no_grad():
            for psi in states:
                pe = m.encode(psi)
                rho = m.noisy_state(pe, p, noise)
                out = U @ rho @ U.conj().T
                fs.append(float((pe.conj() @ out @ pe).real))
        a = np.array(fs)
        outF.append(float(a.mean()))
        outE.append(float(a.std(ddof=1) / math.sqrt(n_test)))
    return outF, outE


# =====================================================================
# E4: global (no-projection) variational recovery, QVECTOR-style
# =====================================================================
def train_global(noise, seed=1234):
    epochs = 60 if SMOKE else 600
    lr = 1e-2
    pr = SCHED[noise][0][2]
    torch.manual_seed(seed); np.random.seed(seed)
    model = GlobalRec()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.RandomState(seed)
    for ep in range(epochs):
        opt.zero_grad()
        U = m.recovery_unitary(model())
        loss = 0.0
        for _ in range(48):
            psi = m.random_logical_state(1, rng)
            pe = m.encode(psi)
            p = rng.uniform(*pr)
            rho = m.noisy_state(pe, p, noise)
            out = U @ rho @ U.conj().T
            loss = loss - (pe.conj() @ out @ pe).real
        (loss / 48).backward()
        opt.step()
        if (ep + 1) % 200 == 0:
            print(f'    [global/{noise}] epoch {ep+1}: '
                  f'F={-float(loss)/48:.4f}', flush=True)
    return model.phi.detach().clone()


# =====================================================================
# E2: low-data regime (fixed pool of K logical states for training)
# =====================================================================
def train_pool(make_model, pool, noise, seed=1234, epochs=None, lr=3e-3,
               p_range=(0.02, 0.15)):
    epochs = epochs or (60 if SMOKE else 400)
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.RandomState(seed + 1)
    K = len(pool)
    for ep in range(epochs):
        opt.zero_grad()
        R_all = m.recovery_unitary_batch(model())
        loss = 0.0
        for _ in range(48):
            psi = pool[rng.randint(K)]
            pe = m.encode(psi)
            p = rng.uniform(*p_range)
            rho = m.noisy_state(pe, p, noise)
            l, _ = m.vscr_fidelity(pe, rho, R_all, lam=0.0)
            loss = loss + l
        (loss / 48).backward()
        opt.step()
    return model

# =====================================================================
# E3: SDP optimal CPTP recovery ceiling (exact, per branch)
#   For branch s the projected+recovered process reduces to 2x2 operators
#   A_k = W_s^† K_k V  (K_k: channel Kraus, V: encoding isometry, W_s:
#   orthonormal basis of image(P_s)).  The unnormalised average branch
#   fidelity of a CP trace-non-increasing map G_s is
#       E_psi Tr[G_s(Σ_k A_k psi psi^† A_k^†) psi psi^†] = 2 Tr[C_s M_s]
#   with Choi matrix C_s = (id ⊗ G_s)(|Ω><Ω|) and the EXACT Haar average
#       M_s = [ kron(S1, I2) + 2 Σ_k w_k w_k^† ] / 6,
#       S1 = Σ_k A_k* A_k^T,   w_k = vec(A_k*)/√2 = (I ⊗ A_k^†)|Ω>.
#   Max 2 Tr[C M] s.t. C ⪰ 0, Tr_out(C) ⪯ I/2  (trace-non-increasing;
#   any TNI map extends to a CPTP map on the full 32-dim space without
#   changing the branch fidelity).  F_s^max = 2Tr[C*M]/p_s, p_s = Tr[S1]/2;
#   ceiling F̄^CPTP = Σ_s p_s F_s^max  (directly comparable to our F̄).
# =====================================================================
def _kraus_dep(p):
    """1024 Kraus ops of the per-qubit-sequential depolarizing channel."""
    per_q = []
    for q in range(m.N):
        ops = [math.sqrt(1 - p) * m.I2] + \
              [math.sqrt(p / 3.0) * G for G in (m.Xp, m.Yp, m.Zp)]
        per_q.append([m.embed_gate(o, [q]).numpy() for o in ops])
    out = []
    for combo in itertools.product(*per_q):
        K = combo[0]
        for c in combo[1:]:
            K = c @ K
        out.append(K)
    return out


def _kraus_ad(gamma):
    per_q = [tuple(x.numpy() for x in m.amp_kraus(gamma, q)) for q in range(m.N)]
    out = []
    for combo in itertools.product(*per_q):
        K = combo[0]
        for c in combo[1:]:
            K = c @ K
        out.append(K)
    return out


def _kraus_coherent(eps):
    U = np.eye(m.DIM, dtype=complex)
    Rx2 = np.array([[math.cos(eps / 2), -1j * math.sin(eps / 2)],
                    [-1j * math.sin(eps / 2), math.cos(eps / 2)]])
    for q in range(m.N):
        U = U @ m.embed_gate(torch.tensor(Rx2, dtype=m.DTYPE), [q]).numpy()
    return [U]


_KRAUS_CACHE = {}


def channel_kraus(noise, p):
    key = (noise, round(float(p), 10))
    if key in _KRAUS_CACHE:
        return _KRAUS_CACHE[key]
    if noise == 'depolarizing':
        K = _kraus_dep(p)
    elif noise == 'amplitude_damping':
        K = _kraus_ad(p)
    elif noise == 'coherent':
        K = _kraus_coherent(p)
    else:
        raise ValueError(noise)   # 'mixed' handled lazily in _branch_A
    _KRAUS_CACHE[key] = K
    return K



def _W_basis():
    """(16, 32, 2) orthonormal bases of the syndrome subspaces."""
    Ws = np.zeros((16, m.DIM, 2), dtype=complex)
    for s in range(16):
        P = m.P_SYNDS_STACK[s].numpy()
        ev, evec = np.linalg.eigh(0.5 * (P + P.conj().T))
        cols = evec[:, ev > 0.5]
        assert cols.shape[1] == 2, (s, ev)
        Ws[s] = cols
    return Ws


W_BASIS = _W_basis()
V_ISO = m.E_CODE.numpy()                       # (32,2) encoding isometry


def _branch_A(noise, p, s):
    """2x2 reduced branch operators {A_k = W_s^† K_k V}."""
    W = W_BASIS[s].conj().T                    # (2,32)
    if noise == 'mixed':
        key = ('mixedKV', round(float(p), 10))
        if key not in _KRAUS_CACHE:
            _KRAUS_CACHE[key] = (
                [K @ V_ISO for K in _kraus_dep(p)], _kraus_ad(0.5 * p))
        Kd_V, Kad = _KRAUS_CACHE[key]
        return [W @ (Kj @ KV) for KV in Kd_V for Kj in Kad]
    return [W @ (K @ V_ISO) for K in channel_kraus(noise, p)]


def _branch_M(noise, p, s):
    """Exact Haar-averaged objective matrix M_s and branch weight p_s."""
    S1 = np.zeros((2, 2), dtype=complex)
    VV = np.zeros((4, 4), dtype=complex)
    for A in _branch_A(noise, p, s):
        S1 += A.conj() @ A.T
        w = A.conj().reshape(4) / math.sqrt(2)     # = (I ⊗ A^†)|Ω>
        VV += np.outer(w, w.conj())
    M = (np.kron(S1, np.eye(2)) + 2.0 * VV) / 6.0
    p_s = float(np.real(np.trace(S1))) / 2.0
    return M, p_s

def _sdp_branch(M, G_dec):
    """Max 2Tr[CM] s.t. C=LL^† ⪰ 0, I/2 − Tr_out C ⪰ 0.  Returns F_unnorm."""
    def unpack(x):
        return (x[:16] + 1j * x[16:]).reshape(4, 4)

    def trace_out(C):
        return np.array([[C[0, 0] + C[1, 1], C[0, 2] + C[1, 3]],
                         [C[2, 0] + C[3, 1], C[2, 2] + C[3, 3]]])

    def neg_obj(x):
        L = unpack(x)
        C = L @ L.conj().T
        return -2.0 * float(np.real(np.trace(C @ M)))

    def c_tr(x):
        L = unpack(x)
        T = 0.5 * np.eye(2) - trace_out(L @ L.conj().T)
        return float(np.real(np.trace(T)))

    def c_det(x):
        L = unpack(x)
        T = 0.5 * np.eye(2) - trace_out(L @ L.conj().T)
        return float(np.real(np.linalg.det(T)))

    def L_of_unitary(G):
        v = np.zeros(4, dtype=complex)      # (I⊗G)|Ω> : v[a*2+b] = G[b,a]/√2
        for a in range(2):
            for b in range(2):
                v[a * 2 + b] = G[b, a] / math.sqrt(2)
        L = np.zeros((4, 4), dtype=complex)
        L[:, 0] = v
        return L

    starts = [L_of_unitary(np.eye(2, dtype=complex)), L_of_unitary(G_dec)]
    rs = np.random.RandomState(0)
    for _ in range(2):
        starts.append((rs.randn(4, 4) + 1j * rs.randn(4, 4)) / 2)
    best, best_x = -1.0, None
    cons = [{'type': 'ineq', 'fun': c_tr}, {'type': 'ineq', 'fun': c_det}]
    for L0 in starts:
        x0 = np.concatenate([L0.real.reshape(16), L0.imag.reshape(16)])
        r = minimize(neg_obj, x0, constraints=cons, method='SLSQP',
                     options={'maxiter': 500, 'ftol': 1e-12})
        if r.success and c_tr(r.x) > -1e-8 and c_det(r.x) > -1e-10:
            if -r.fun > best:
                best, best_x = -r.fun, r.x
    return best, best_x


def sdp_ceiling(noise, p):
    """F̄^CPTP ceiling and per-branch data for a channel at strength p."""
    Fs, ps = [], []
    for s in range(16):
        M, p_s = _branch_M(noise, p, s)
        G_dec = V_ISO.conj().T @ m.C_SYNDS[s].numpy() @ W_BASIS[s]   # 2x2
        fu, _ = _sdp_branch(M, G_dec)
        Fs.append(fu / max(p_s, 1e-15) if p_s > 1e-12 else 1.0)
        ps.append(p_s)
    F_bar = float(np.dot(ps, Fs))
    return {'F_cptp': F_bar, 'F_s_max': [float(x) for x in Fs],
            'p_s': [float(x) for x in ps]}


def _selftest_sdp():
    # (a) identity channel: ceiling == 1 exactly
    r = sdp_ceiling('depolarizing', 0.0)
    assert abs(r['F_cptp'] - 1.0) < 1e-6, r['F_cptp']
    # (b) depolarizing p=0.05: ceiling >= decoder F̄ (MC) and close to it
    r = sdp_ceiling('depolarizing', 0.05)
    rng = np.random.RandomState(5)
    Fdec = []
    for _ in range(300):
        psi = m.random_logical_state(1, rng)
        pe = m.encode(psi)
        rho = m.noisy_state(pe, 0.05, 'depolarizing')
        Fdec.append(float(m.perfect_code_decoder_fidelity(pe, rho)))
    Fdec = float(np.mean(Fdec))
    assert r['F_cptp'] >= Fdec - 2e-4, (r['F_cptp'], Fdec)
    assert r['F_cptp'] <= Fdec + 2e-3, (r['F_cptp'], Fdec)
    print(f'[sdp selftest] identity ceiling=1 OK; dep(0.05): '
          f'ceiling={r["F_cptp"]:.6f} vs decoder={Fdec:.6f} OK', flush=True)

# =====================================================================
# E5: Pauli-restricted learned recovery (exhaustive per-branch search)
# =====================================================================
def pauli_lookup(noise, p, M=200, seed=m.SEED + 5):
    """Unsupervised exhaustive search over Pauli recoveries per branch.
    Any Pauli P with synd(P) != s maps image(P_s) orthogonal to the code
    space (fidelity 0); the 64 Paulis with synd(P) = s fall into 4 logical
    classes C_s · {I, X_L, Z_L, X_L Z_L} acting identically (up to phase)
    on the branch.  So the search is exactly 4 candidates per branch, and
    a brute-force check over random Pauli strings verifies the reduction."""
    classes = [('I', m.I_full), ('XL', m.XL), ('ZL', m.ZL),
               ('XLZL', m.XL @ m.ZL)]
    rng = np.random.RandomState(seed)
    states = [m.encode(m.random_logical_state(1, rng)) for _ in range(M)]
    rhos = [m.noisy_state(pe, p, noise) for pe in states]

    def cf_of(R, s):
        Ps = m.P_SYNDS_STACK[s]
        f = w = 0.0
        with torch.no_grad():
            for pe, rho in zip(states, rhos):
                pr = Ps @ rho @ Ps
                f += float((pe.conj() @ (R @ pr @ R.conj().T) @ pe).real)
                w += float(torch.trace(Ps @ rho).real)
        return f / max(w, 1e-12)

    out = {'matches_decoder': True, 'best_class': [], 'gain_over_I': [],
           'cf_class': []}
    for s in range(16):
        cfs = [cf_of(m.C_SYNDS[s].to(m.DTYPE) @ L, s) for _, L in classes]
        best = int(np.argmax(cfs))
        out['best_class'].append(classes[best][0])
        out['gain_over_I'].append(float(cfs[best] - cfs[0]))
        out['cf_class'].append([float(c) for c in cfs])
        if best != 0:
            out['matches_decoder'] = False
    # brute-force verification of the 4-class reduction
    rs = np.random.RandomState(7)
    worst = 0.0
    for _ in range(60):
        sstr = ''.join('IXYZ'[i] for i in rs.randint(0, 4, m.N))
        P = m.pauli_string(sstr)
        bits = tuple(int(m._anticommute(P, g)) for g in m.STAB)
        s = m.SYND_BITS.index(bits)
        cf = cf_of(P, s)
        # identify P's logical class: G = V† P W_s vs class reps (phase-free)
        G = V_ISO.conj().T @ P.numpy() @ W_BASIS[s]
        G = G / max(abs(np.linalg.det(G)), 1e-12) ** 0.5
        diffs = []
        for _, L in classes:
            Gr = V_ISO.conj().T @ (m.C_SYNDS[s].to(m.DTYPE) @ L).numpy() @ W_BASIS[s]
            Gr = Gr / max(abs(np.linalg.det(Gr)), 1e-12) ** 0.5
            ph = np.trace(G @ Gr.conj().T) / 2
            diffs.append(float(np.abs(G - ph * Gr).max()))
        cls = int(np.argmin(diffs))
        worst = max(worst, abs(cf - out['cf_class'][s][cls]), diffs[cls])
    out['max_bruteforce_dev'] = float(worst)
    assert worst < 1e-8, worst
    return out


# =====================================================================
# Figure
# =====================================================================
def plot_ablation(bar_vals, lowdata, fname='fig_ablation'):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    labels = list(bar_vals.keys())
    vals = [bar_vals[k] for k in labels]
    cols = ['#e41a1c', '#1f4e9c', '#377eb8', '#a65628', '#bbbbbb']
    bars = ax.bar(range(len(vals)), vals, 0.62, color=cols[:len(vals)])
    bars[-1].set_hatch('//')
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=6.5)
    ax.set_ylabel('$\\bar{F}$ at $p=0.10$ (depolarizing)')
    ax.set_ylim(0, 1.12); ax.grid(alpha=0.3, lw=0.5, axis='y')
    ax.set_title('same-family baselines & CPTP ceiling', fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f'{v:.3f}',
                ha='center', fontsize=6.5)

    ax = axes[1]
    Ks = lowdata['K']
    ax.plot(Ks, lowdata['cold_hyper_F'], 'o-', color='#1f4e9c', lw=1.5, ms=5,
            label='VSCR hypernetwork $\\bar{F}$ (cold)')
    ax.plot(Ks, lowdata['cold_ind_F'], 's--', color='#377eb8', lw=1.5, ms=5,
            label='VQR-ind, no sharing $\\bar{F}$ (cold)')
    ax.plot(Ks, lowdata['cold_hyper_mincf'], 'o:', color='#1f4e9c', lw=1.0,
            ms=4, alpha=0.7, label='hypernetwork min cf$_s$')
    ax.plot(Ks, lowdata['cold_ind_mincf'], 's:', color='#377eb8', lw=1.0,
            ms=4, alpha=0.7, label='VQR-ind min cf$_s$')
    ax.set_xscale('log', base=2); ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel('training pool size $K$ (logical states)')
    ax.set_ylabel('depolarizing $p=0.10$')
    ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=6.5, loc='lower right')
    ax.set_title('low-data regime (cold start)', fontsize=9)
    fig.tight_layout()
    vp.save_fig(fig, fname)

# =====================================================================
# Main
# =====================================================================
WARM_NPZ = {'depolarizing': 'vscr_angles_paper_dep.npz',
            'amplitude_damping': 'vscr_angles_paper_ad.npz',
            'mixed': 'vscr_angles_paper_mixed.npz',
            'coherent': 'vscr_angles_paper_coh.npz'}


def _decoder_ref_F(noise, p, n=200, seed=m.SEED + 7):
    rng = np.random.RandomState(seed)
    fs = []
    for _ in range(n):
        psi = m.random_logical_state(1, rng)
        pe = m.encode(psi)
        fs.append(float(m.perfect_code_decoder_fidelity(
            pe, m.noisy_state(pe, p, noise))))
    return float(np.mean(fs))


def main():
    t0 = time.time()
    print('=== E3 prelude: SDP self-test ===', flush=True)
    _selftest_sdp()

    # ---------------- E1: per-syndrome independent recovery --------------
    ind = {}
    for noise in CHANNELS:
        print(f'\n=== E1: VQR-ind training ({noise}) ===', flush=True)
        model, hist, infos = train_ind(noise)
        phi = model().detach().cpu().numpy()
        np.savez(f'vscr_angles_abl_ind_{SHORT[noise]}.npz', phi=phi,
                 infos=json.dumps(infos))
        F, E = eval_phi_table(phi, noise, VP_P)
        # warm-VSCR reference on the same grid
        phi_w = np.load(WARM_NPZ[noise])['phi']
        Fw, _ = eval_phi_table(phi_w, noise, VP_P)
        R_ind = m.recovery_unitary_batch(torch.tensor(phi, dtype=torch.float64))
        R_dec = m.recovery_unitary_batch(vp.PHI_DEC)
        cf_ind, w = vp.cf_table(R_ind, noise, 0.15, M=200)
        cf_dec, _ = vp.cf_table(R_dec, noise, 0.15, M=200)
        ind[noise] = {'F': F, 'F_sem': E, 'infos': infos,
                      'max_abs_diff_vs_warm':
                          float(np.max(np.abs(np.array(F) - np.array(Fw)))),
                      'min_cf_15': float(np.min(cf_ind[w > 1e-3])),
                      'min_cf_dec_15': float(np.min(cf_dec[w > 1e-3]))}
        print(f'  ind[{noise}] F(p=0.10)={F[np.where(VP_P==0.10)[0][0]]:.6f} '
              f'max|dF vs warm VSCR|={ind[noise]["max_abs_diff_vs_warm"]:.2e} '
              f'min cf@0.15: ind={ind[noise]["min_cf_15"]:.6f} '
              f'dec={ind[noise]["min_cf_dec_15"]:.6f}', flush=True)

    # ---------------- E4: global no-projection recovery ------------------
    glob = {}
    for noise in CHANNELS:
        print(f'\n=== E4: global (no-projection) recovery ({noise}) ===',
              flush=True)
        phi_g = train_global(noise)
        F, E = eval_global(phi_g.numpy(), noise, VP_P)
        glob[noise] = {'F': F, 'F_sem': E}
        print(f'  global[{noise}] F(p=0.10)='
              f'{F[np.where(VP_P==0.10)[0][0]]:.6f}', flush=True)

    # ---------------- E2: low-data regime --------------------------------
    print('\n=== E2: low-data regime (depolarizing) ===', flush=True)
    Ks = [2, 4, 8] if SMOKE else [2, 4, 8, 16]
    lowdata = {'K': Ks}
    for init in ('warm', 'cold'):
        for tag in ('hyper', 'ind'):
            lowdata[f'{init}_{tag}_F'] = []
            lowdata[f'{init}_{tag}_mincf'] = []
        for K in Ks:
            rp = np.random.RandomState(555)
            pool = [m.random_logical_state(1, rp) for _ in range(K)]
            if init == 'warm':
                mk_h = lambda: vp.VSCRWarm()
                mk_i = lambda: VQRInd(True)
                ep, lr = (60 if SMOKE else 400), 3e-3
            else:
                def mk_h():
                    mod = vp.VSCRWarm()
                    # cold: no decoder knowledge (buffer assignment, never
                    # in-place, so no other consumer of PHI_DEC can be hit)
                    mod._buffers['phi_dec'] = torch.zeros_like(mod.phi_dec)
                    # NOTE: VSCRWarm zero-inits the hypernetwork (required for
                    # the exact warm start), but an all-zero tanh-MLP is a dead
                    # gradient point; for the COLD comparison we restore a
                    # standard random init so the shared model has the same
                    # trainable capacity as the independent table.
                    nn.init.normal_(mod.hnet[0].weight, std=0.1)
                    nn.init.zeros_(mod.hnet[0].bias)
                    nn.init.normal_(mod.hnet[2].weight, std=0.1)
                    nn.init.zeros_(mod.hnet[2].bias)
                    return mod
                mk_i = lambda: VQRInd(False)
                ep, lr = (60 if SMOKE else 600), 1e-2
            mh = train_pool(mk_h, pool, 'depolarizing', epochs=ep, lr=lr)
            mi = train_pool(mk_i, pool, 'depolarizing', epochs=ep, lr=lr)
            for tag, mod in (('hyper', mh), ('ind', mi)):
                phi = mod().detach()
                Fq, _ = eval_phi_table(phi.numpy(), 'depolarizing',
                                       np.array([0.10]), n_test=200)
                cf, w = vp.cf_table(m.recovery_unitary_batch(phi),
                                    'depolarizing', 0.10, M=200)
                lowdata[f'{init}_{tag}_F'].append(Fq[0])
                lowdata[f'{init}_{tag}_mincf'].append(
                    float(np.min(cf[w > 1e-3])))
            print(f'  {init} K={K}: hyper F={lowdata[f"{init}_hyper_F"][-1]:.4f}'
                  f' mincf={lowdata[f"{init}_hyper_mincf"][-1]:.4f} | '
                  f'ind F={lowdata[f"{init}_ind_F"][-1]:.4f} '
                  f'mincf={lowdata[f"{init}_ind_mincf"][-1]:.4f}', flush=True)

    # ---------------- E3: CPTP ceilings ----------------------------------
    print('\n=== E3: SDP CPTP ceilings ===', flush=True)
    CEIL_POINTS = [('depolarizing', 0.05), ('depolarizing', 0.10),
                   ('depolarizing', 0.15), ('amplitude_damping', 0.10),
                   ('mixed', 0.10), ('coherent', 0.15), ('coherent', 0.30)]
    sdp = {}
    for noise, p in CEIL_POINTS:
        r = sdp_ceiling(noise, p)
        Fdec = _decoder_ref_F(noise, p)
        cf_dec, w_dec = vp.cf_table(m.recovery_unitary_batch(vp.PHI_DEC),
                                    noise, p, M=200)
        gaps = [r['F_s_max'][s] - cf_dec[s]
                for s in range(16) if w_dec[s] > 1e-3]
        r['F_dec'] = Fdec
        r['max_branch_gap'] = float(max(gaps))
        r['F_gap_bar'] = r['F_cptp'] - Fdec
        sdp[f'{SHORT[noise]}_{p}'] = r
        print(f'  {noise} p={p}: CPTP={r["F_cptp"]:.6f} dec={Fdec:.6f} '
              f'dF={r["F_gap_bar"]:+.2e} max branch gap='
              f'{r["max_branch_gap"]:.2e}', flush=True)

    # ---------------- E5: Pauli-restricted lookup -------------------------
    print('\n=== E5: Pauli-restricted learned recovery ===', flush=True)
    pl = {'dep_0.15': pauli_lookup('depolarizing', 0.15),
          'coh_0.15': pauli_lookup('coherent', 0.15)}
    for k, v in pl.items():
        print(f'  {k}: matches decoder={v["matches_decoder"]} '
              f'max gain over I-class={max(v["gain_over_I"]):.2e} '
              f'bruteforce dev={v["max_bruteforce_dev"]:.1e}', flush=True)

    # ---------------- figure + save ---------------------------------------
    i10 = int(np.where(VP_P == 0.10)[0][0])
    F_warm_010 = eval_phi_table(np.load(WARM_NPZ['depolarizing'])['phi'],
                                'depolarizing', np.array([0.10]))[0][0]
    bar_vals = {'Perfect-code\ndecoder': _decoder_ref_F('depolarizing', 0.10),
                'VSCR warm\n(hypernet)': F_warm_010,
                'VQR-ind\n(no sharing)': ind['depolarizing']['F'][i10],
                'Global unitary\n(no projection)':
                    glob['depolarizing']['F'][i10],
                'CPTP ceiling\n(SDP)': sdp['dep_0.1']['F_cptp']}
    if not SMOKE:
        plot_ablation(bar_vals, lowdata)

    np.savez('vscr_paper_abl_results.npz',
             p_values=VP_P,
             ind={n: ind[n] for n in ind},
             glob={n: glob[n] for n in glob},
             lowdata=lowdata, sdp=sdp, pauli=pl,
             bar_vals=bar_vals, n_params=N_PARAMS,
             runtime=np.array([time.time() - t0]))
    num = json.load(open('paper_numbers.json'))
    num['abl'] = {'ind': ind, 'global': glob, 'lowdata': lowdata, 'sdp': sdp,
                  'pauli_lookup': {k: {kk: vv for kk, vv in v.items()
                                       if kk != 'cf_class'}
                                   for k, v in pl.items()},
                  'bar_dep_p010': bar_vals, 'n_params': N_PARAMS,
                  'runtime_s': time.time() - t0}
    json.dump(num, open('paper_numbers.json', 'w'), indent=2)

    print('\n================ ABLATION SUMMARY ================')
    for noise in CHANNELS:
        print(f'{noise:20s} ind F(p=.10)={ind[noise]["F"][i10]:.6f} '
              f'glob F(p=.10)={glob[noise]["F"][i10]:.6f} '
              f'max|dF(ind-warm)|={ind[noise]["max_abs_diff_vs_warm"]:.2e}')
    print('ceilings:', {k: round(v['F_cptp'], 6) for k, v in sdp.items()})
    print('decoder :', {k: round(v['F_dec'], 6) for k, v in sdp.items()})
    print(f'total runtime {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
