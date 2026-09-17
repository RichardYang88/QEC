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

# Pin the native thread pools BEFORE numpy/torch/scipy are imported; see the note
# on `ssvr_qec._pin_blas_threads` for why this matters on this host.
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
import torch.nn as nn
from scipy.optimize import minimize
from scipy.linalg import expm
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
                                use_real_grad=True, verbose=False, quad=True)
            hist_all['epoch'] += [off + e for e in hist['epoch']]
            hist_all['fidelity'] += hist['fidelity']
            hist_all['loss'] += hist['loss']
            off += ep
        # same EXACT selection diagnostic as `vp.train_warm_best(quad=True)`, so
        # the E1 comparison differs in the architecture only
        with torch.no_grad():
            phi_now = model().detach().cpu()
        cf, w = m.syndrome_conditional_fidelity_exact(
            phi_now, m.quad_cache((0.07, 0.07), noise, n_p=1))
        min_cf = float(np.where(w > 0.001, cf, 10.0).min())
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

def _sdp_branch(M, G_dec, G_ref=None):
    """Max 2Tr[CM] s.t. C=LL^† ⪰ 0 and Tr_out C = I/2 (trace preservation).

    Returns F_unnorm.  The trace-preservation condition is imposed as four real
    EQUALITIES rather than as the inequality I/2 − Tr_out C ⪰ 0; see the second
    bug fix recorded on `feasible` below for the measured cost of the inequality
    form (it under-reported the [[9,1,3]] amplitude-damping ceiling by 3.7x at
    p=0.05 and 16x at p=0.10, because SLSQP was free to stop in the interior of
    the sub-trace-preserving set).

    BUG FIX.  This used to initialise `best = -1.0` and to accept a start's
    result only when scipy reported `success=True`.  SLSQP reports
    `success=False` at many perfectly converged KKT points -- and always once it
    hits `maxiter` -- so a branch whose solves all "failed" returned -1.0, which
    `sdp_ceiling` then divided by p_s.  That is how the amplitude-damping
    p=0.30 ceiling came out as F_CPTP = -0.1113, a NEGATIVE fidelity, making
    every headroom derived from it meaningless.  The same gate also makes the
    ceiling silently UNDER-report: on amplitude damping p=0.10 the old code
    returned 0.985471 while the exact unitary-family optimum is 0.986532 -- a
    "ceiling" below a feasible point, which cannot happen for a valid upper
    bound, and which is exactly why Fig.4's ceiling bar could sit below the
    method it is supposed to bound.

    Two changes fix it.  (i) Every start is scored directly and the running best
    is seeded with the best FEASIBLE start, so the return value can never fall
    below a point already verified feasible -- in particular never below the
    exact unitary optimum when it is supplied as `G_ref`.  (ii) `r.success` is
    no longer trusted; feasibility is checked explicitly instead.
    `sdp_ceiling` then asserts F̄^CPTP >= max(F̄^unitary, F̄^decoder).

    Returns (F_unnorm, x_star), x_star being the length-32 real parameter vector
    attaining F_unnorm so callers can re-verify feasibility."""
    def unpack(x):
        return (x[:16] + 1j * x[16:]).reshape(4, 4)

    def trace_out(C):
        return np.array([[C[0, 0] + C[1, 1], C[0, 2] + C[1, 3]],
                         [C[2, 0] + C[3, 1], C[2, 2] + C[3, 3]]])

    def obj(x):
        L = unpack(x)
        C = L @ L.conj().T
        return 2.0 * float(np.real(np.trace(C @ M)))

    def neg_obj(x):
        return -obj(x)

    def tp_residual(x):
        """The four real residuals of Tr_out(C) = I/2, i.e. of sum_j B_j^dag B_j = I.

        Tr_out(C) is Hermitian, so its two diagonal entries are real and its
        off-diagonal pair is one complex number: exactly four real equations."""
        T = trace_out(unpack(x) @ unpack(x).conj().T)
        return [float(np.real(T[0, 0]) - 0.5), float(np.real(T[1, 1]) - 0.5),
                float(np.real(T[0, 1])), float(np.imag(T[0, 1]))]

    def L_of_unitary(G):
        v = np.zeros(4, dtype=complex)      # (I⊗G)|Ω> : v[a*2+b] = G[b,a]/√2
        for a in range(2):
            for b in range(2):
                v[a * 2 + b] = G[b, a] / math.sqrt(2)
        L = np.zeros((4, 4), dtype=complex)
        L[:, 0] = v
        return L

    def feasible(x):
        # SECOND BUG FIX, found by `ancilla_recovery.py`.  The constraint used to
        # be the INEQUALITY I/2 - Tr_out(C) >= 0, tested one-sidedly as
        # `c_tr(x) > -1e-8 and c_det(x) > -1e-10`.  That admits the whole interior
        # of the sub-trace-preserving set, and SLSQP then stops at interior points
        # that are strictly worse than the true optimum -- not because the
        # objective is wrong but because nothing forces the iterate back onto the
        # trace-preserving manifold where the maximum lives.  Measured on
        # [[9,1,3]] amplitude damping p=0.05, branch s=15: the inequality form
        # returns 1.986802e-05 with a rank-1 Choi matrix (it never leaves the
        # unitary family at all) and only reaches 2.036127e-05 with 24 random
        # starts, while the equality form below returns 2.046582e-05 from two
        # random starts and reproduces it to 3e-13 from three and from six.  That
        # last value is not a solver artefact: it is attained by an explicitly
        # trace-preserving Kraus pair (defect 4e-16) whose 4x4 Stinespring dilation
        # is unitary to 4e-16, scored by `vscr_general.cf_unnormalised`, and
        # confirmed by a 20000-state brute-force Haar quadrature of the physical
        # estimator to within 0.9 sigma.  Imposing the four real equations of
        # Tr_out(C) = I/2 is also simply more correct: the ceiling is over
        # recovery CHANNELS, which are trace preserving by definition.
        #
        # A unitary Choi vector satisfies the equality exactly, so the
        # `L_of_unitary` starts below are feasible and the running-best seeding
        # still guarantees F_cptp >= F_unit (weak duality) by construction.
        return max(abs(v) for v in tp_residual(x)) < 1e-7

    starts = [L_of_unitary(np.eye(2, dtype=complex)), L_of_unitary(G_dec)]
    if G_ref is not None:
        starts.append(L_of_unitary(np.asarray(G_ref, dtype=complex)))
    rs = np.random.RandomState(0)
    for _ in range(2):
        starts.append((rs.randn(4, 4) + 1j * rs.randn(4, 4)) / 2)
    best, best_x = -np.inf, None
    cons = [{'type': 'eq', 'fun': (lambda x, i=i: tp_residual(x)[i])}
            for i in range(4)]
    for L0 in starts:
        x0 = np.concatenate([L0.real.reshape(16), L0.imag.reshape(16)])
        if feasible(x0) and obj(x0) > best:      # (i) seed with the start
            best, best_x = obj(x0), x0
        r = minimize(neg_obj, x0, constraints=cons, method='SLSQP',
                     options={'maxiter': 500, 'ftol': 1e-12})
        if feasible(r.x) and -float(r.fun) > best:   # (ii) ignore r.success
            best, best_x = -float(r.fun), r.x
    if best_x is None:
        raise RuntimeError('branch SDP found no feasible point')
    return best, best_x


# =====================================================================
# EXACT per-branch unitary ceiling, and the separable p-averaged objective
#
# The Haar-averaged conditional fidelity of branch s has the closed form
#
#     cf_s(G_s) = [ J_s(G_s) + const_s ] / 6 / p_s ,
#     J_s(G_s)  = sum_k |Tr(G_s A_k)|^2 = g_s^dagger Q_s g_s ,
#     g_s       = vec_rowmajor(G_s) ,   G_s = V^dagger R_s W_s  (2x2) ,
#     const_s   = Tr(sum_k A_k^dagger A_k) = Tr(Q_s) ,   p_s = const_s / 2 ,
#
# obtained from the degree-2 Haar identity
# int |<psi|X|psi>|^2 dpsi = (|Tr X|^2 + Tr(X X^dagger))/6 on CP^1 and the
# unitary invariance of the trace.  Two facts follow, and they drive
# everything below:
#
#   (a) cf_s depends on R_s ONLY through the 2x2 block G_s, so the exact
#       fidelity of a whole (16, PHI_DIM) table costs 16 small quadratic
#       forms -- no Monte-Carlo, no 32x32 channel applications;
#   (b) p_s cancels in p_s * cf_s, so the p-averaged objective
#       F_avg = sum_p w_p sum_s p_s(p) cf_s(G_s; p) = sum_s obj_s(phi_s)
#       is SEPARABLE across syndromes.
#
# (b) is what makes `refine_per_syndrome` able to reach the global optimum of
# the VSCR family block by block; (a) is what makes the certified headroom
# `F_unit - F_dec` computable to machine precision instead of to the ~1e-4
# Monte-Carlo SEM that the headroom itself is the size of.
# =====================================================================
_PAULI4 = [np.eye(2, dtype=complex),
           np.array([[0, 1], [1, 0]], dtype=complex),
           np.array([[0, -1j], [1j, 0]], dtype=complex),
           np.array([[1, 0], [0, -1]], dtype=complex)]


def _branch_qform(noise, p, s):
    """Q_s (4x4), the G-independent constant Tr(sum_k A_k^dagger A_k), and p_s."""
    Q = np.zeros((4, 4), dtype=complex)
    const = 0.0
    for A in _branch_A(noise, p, s):
        A = np.asarray(A, dtype=complex)
        v = A.T.reshape(4)
        Q += np.outer(v, v.conj())
        const += float(np.real(np.trace(A.conj().T @ A)))
    Q = 0.5 * (Q + Q.conj().T)
    return Q, const, float(np.real(np.trace(Q))) / 2


def _G_of_h(h):
    """2x2 unitary exp(i * sum_a h_a sigma_a) from 4 real Lie-algebra coords.

    u(2) = span{i I, i X, i Y, i Z}, so this parametrisation is surjective onto
    U(2) and unconstrained in h -- which is what lets L-BFGS-B / Nelder-Mead be
    used directly, with no unitarity constraint to enforce and no projection
    step to differentiate through.

    Uses the CLOSED FORM rather than scipy.linalg.expm.  Writing h = (h0, r)
    with r = (h1, h2, h3) and rho = |r|, the Pauli algebra (r.sigma)^2 = rho^2 I
    collapses the exponential series to

        exp(i h.sigma) = e^{i h0} [ cos(rho) I + i sinc(rho) (r.sigma) ] ,
        sinc(rho) = sin(rho)/rho ,   sinc(0) = 1 .

    That is exact to machine precision (asserted against `expm` in
    `_selftest_liealg`), ~40x cheaper per call, and -- what matters for a
    multi-start solve that evaluates it tens of thousands of times from inside
    scipy's finite-difference Jacobian -- it avoids the LAPACK Schur / Pade code
    path `expm` takes, which is where the intermittent segfaults in
    `opt_unitary_branch` originated."""
    h = np.asarray(h, dtype=float)
    r = h[1], h[2], h[3]
    rho = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    sinc = 1.0 if rho < 1e-12 else math.sin(rho) / rho
    rds = r[0] * _PAULI4[1] + r[1] * _PAULI4[2] + r[2] * _PAULI4[3]
    G = math.cos(rho) * _PAULI4[0] + 1j * sinc * rds
    return G * complex(math.cos(h[0]), math.sin(h[0]))


def _selftest_liealg(n=400, seed=0):
    """Assert the closed form of `_G_of_h` equals scipy's `expm` and is unitary."""
    rs = np.random.RandomState(seed)
    worst_e = worst_u = 0.0
    for scale in (1e-14, 1e-6, 1.0, 10.0):
        for _ in range(n):
            h = rs.normal(size=4) * scale
            G = _G_of_h(h)
            worst_e = max(worst_e, float(np.abs(G - expm(1j * sum(
                float(c) * Pm for c, Pm in zip(h, _PAULI4)))).max()))
            worst_u = max(worst_u,
                          float(np.abs(G.conj().T @ G - np.eye(2)).max()))
    assert worst_e < 1e-12, f'closed form != expm: {worst_e:.2e}'
    assert worst_u < 1e-12, f'_G_of_h not unitary: {worst_u:.2e}'
    # surjectivity spot-check: the identity and every Pauli are reachable
    assert np.abs(_G_of_h(np.zeros(4)) - np.eye(2)).max() < 1e-15
    print(f'[lie-algebra selftest] closed form == scipy.expm to {worst_e:.2e} '
          f'and unitary to {worst_u:.2e} over {4 * n} samples spanning '
          f'scales 1e-14..10  VERIFIED', flush=True)


# discrete seeds for `opt_unitary_branch`: the 4 Paulis plus 2 phase matrices.
# J is invariant under a global phase on G, so these are redundant but cheap.
_PHASE_SEEDS = [np.array([[0, 1j], [1j, 0]], dtype=complex),      # i X
                np.array([[1j, 0], [0, 1]], dtype=complex)]        # diag(i, 1)


def _max_unitary_J(Q, n_start=16, seed=0, single=None, G_dec=None):
    """max over 2x2 UNITARIES G of  J(G) = g^dagger Q g,  g = vec_rowmajor(G).

    Q must be Hermitian.  Returns (J_max, G_opt).

    `single` : if not None the branch has exactly one Kraus operator A, so
        max_G |Tr(G A)| = ||A||_* (nuclear norm) is attained by the polar factor
        of A and is PROVABLY global; it is asserted against the SVD and used as
        a seed.  `G_dec` : the decoder's 2x2 block, always worth seeding with.
    Otherwise the search is multi-start L-BFGS-B over the Lie algebra u(2) plus
    a derivative-free Nelder-Mead polish.  The result is a strong local optimum
    and is CERTIFIED from above by `2 * lambda_max(Q)`, since J is a Rayleigh
    quotient and ||vec(G)||^2 = Tr(G^dagger G) = 2 for every 2x2 unitary."""
    def J(G):
        g = np.asarray(G, dtype=complex).reshape(4)
        return float(np.real(g @ Q @ g.conj()))

    def negJ(h):
        return -J(_G_of_h(h))

    best_G, best_v = None, -np.inf
    if single is not None:
        U, S, Vh = np.linalg.svd(single)
        G0 = Vh.conj().T @ U.conj().T
        assert abs(J(G0) - float(S.sum()) ** 2) < 1e-9, (J(G0), S)
        best_G, best_v = G0, J(G0)
    seeds = list(_PAULI4) + list(_PHASE_SEEDS)
    if G_dec is not None:
        seeds.append(G_dec)
    for Pm in seeds:
        if J(Pm) > best_v:
            best_G, best_v = Pm, J(Pm)
    rs = np.random.RandomState(seed)
    starts = [np.zeros(4)] + [rs.normal(size=4) * 1.5 for _ in range(n_start)]
    for h0 in starts:
        r = minimize(negJ, h0, method='L-BFGS-B',
                     options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14})
        if -float(r.fun) > best_v:
            best_G, best_v = _G_of_h(r.x), -float(r.fun)
    for _ in range(4):
        r = minimize(negJ, rs.normal(size=4) * 1.5, method='Nelder-Mead',
                     options={'maxiter': 6000, 'fatol': 1e-16, 'xatol': 1e-14})
        if -float(r.fun) > best_v:
            best_G, best_v = _G_of_h(r.x), -float(r.fun)
    return best_v, best_G


def opt_unitary_branch(noise, p, s, n_start=16):
    """Exact max over 2x2 unitaries of J(G) = sum_k |Tr(G A_k)|^2 for branch s.

    Returns (J_max, G_opt).  Single-Kraus branches use the polar-factor closed
    form, cross-checked against the nuclear norm, so they are provably global;
    multi-Kraus branches use multi-start L-BFGS-B over the Lie algebra plus a
    derivative-free Nelder-Mead polish, seeded with the identity, the decoder
    unitary, the 6 Pauli/phase matrices and random points.

    On the amplitude-damping p=0.06 branch-2 case the ansatz reaches this
    optimum to -1.0e-15 (see `_selftest_refine`)."""
    Q, const, p_s = _branch_qform(noise, p, s)
    As = list(_branch_A(noise, p, s))
    G_dec = V_ISO.conj().T @ m.C_SYNDS[s].numpy() @ W_BASIS[s]
    return _max_unitary_J(Q, n_start=n_start, seed=0,
                          single=(As[0] if len(As) == 1 else None), G_dec=G_dec)


def opt_unitary_ceiling(noise, p):
    """F̄ of the best PER-BRANCH UNITARY recovery (the VSCR family optimum).

    Returns {'F_unit', 'F_dec', 'cf_unit', 'cf_dec', 'p_s', 'G_opt'}.  F_dec is
    the analytic decoder fidelity (validated against Monte-Carlo in
    `_selftest_sdp`), so F_unit - F_dec is the exact, noiseless non-Pauli
    headroom available to a trained warm start."""
    cf_u, cf_d, ps, Gs = [], [], [], []
    for s in range(16):
        Q, const, p_s = _branch_qform(noise, p, s)
        ps.append(p_s)
        if p_s < 1e-13:
            cf_u.append(1.0)
            cf_d.append(1.0)
            Gs.append(np.eye(2, dtype=complex))
            continue
        Ju, Gu = opt_unitary_branch(noise, p, s)
        gd = (V_ISO.conj().T @ m.C_SYNDS[s].numpy() @ W_BASIS[s]).reshape(4)
        Jd = float(np.real(gd @ Q @ gd.conj()))
        cf_u.append((Ju + const) / 6 / p_s)
        cf_d.append((Jd + const) / 6 / p_s)
        Gs.append(Gu)
    ps = np.array(ps)
    return {'F_unit': float(np.dot(ps, cf_u)),
            'F_dec': float(np.dot(ps, cf_d)),
            'cf_unit': cf_u, 'cf_dec': cf_d, 'p_s': ps.tolist(),
            'G_opt': [np.asarray(g).tolist() for g in Gs]}


_BRANCH_S_CACHE = {}


def _branch_S(noise, p, s):
    """S_s = sum_k A_k A_k^dagger  (2x2 Hermitian), cached.

    The Haar second moment of the branch amplitude is
        int |<psi|G A|psi>|^2 dpsi = [ |Tr(G A)|^2 + Tr(G A A^dagger G^dagger) ] / 6 ,
    so summing over k gives
        F_s(G_s) = [ J_s(G_s) + Tr(G_s S_s G_s^dagger) ] / 6 .
    `Tr(G S G^dagger) == Tr(S)` holds EXACTLY when G_s is unitary, and G_s =
    V^dagger R_s W_s is unitary precisely when R_s maps range(P_s) into the code
    space -- true for the Pauli decoder and for every branch optimum, but NOT
    for an arbitrary angle table (measured error 3.0e-1 on a random table).
    Carrying S_s explicitly therefore makes `exact_F` / `pavg_F` exact for
    ARBITRARY tables instead of only code-preserving ones, which is what lets
    `_selftest_refine` test 1 be a real cross-check against the production
    objective rather than an identity that holds only at PHI_DEC."""
    key = (noise, round(float(p), 12), s)
    S = _BRANCH_S_CACHE.get(key)
    if S is None:
        S = np.zeros((2, 2), dtype=complex)
        for A in _branch_A(noise, p, s):
            A = np.asarray(A, dtype=complex)
            S += A @ A.conj().T
        S = 0.5 * (S + S.conj().T)
        _BRANCH_S_CACHE[key] = S
    return S


def _M_of_S(S):
    """4x4 matrix M with  g^dagger M g = Tr(G S G^dagger),  g = vec_rowmajor(G).

    Tr(G S G^dagger) = sum_{a,b,c} G[a,b] S[b,c] conj(G[a,c]), and with
    g[2a+b] = G[a,b] this is sum conj(g[2a+c]) S[b,c] g[2a+b], i.e.
    M[2a+c, 2a+b] = S[b,c] = kron(I_2, S^T).  Hermitian whenever S is."""
    return np.kron(np.eye(2), np.asarray(S, dtype=complex).T)


def _branch_qforms(noise, p):
    """[(Q_s, const_s, p_s, S_s) for s in 16] -- built once, reused by every
    evaluation.  `_branch_qform` rebuilds the branch Kraus operators (~0.3 s per
    call on depolarizing), so callers that score many tables at one (noise, p)
    must hoist this out of the loop."""
    out = []
    for s in range(16):
        Q, const, p_s = _branch_qform(noise, p, s)
        out.append((Q, const, p_s, _branch_S(noise, p, s)))
    return out


def _exact_F_from_qform(phi_table, qf):
    """`exact_F` against pre-built branch data `qf` = `_branch_qforms(noise, p)`."""
    phi = np.asarray(phi_table, dtype=float)
    R_all = m.recovery_unitary_batch(
        torch.tensor(phi, dtype=torch.float64)).detach().numpy()
    cfs = []
    Ftot = 0.0
    for s in range(16):
        Q, const, p_s, S = qf[s]
        if p_s < 1e-13:
            cfs.append(1.0)
            continue
        G = V_ISO.conj().T @ R_all[s] @ W_BASIS[s]
        g = np.asarray(G, dtype=complex).reshape(4)
        J = float(np.real(g @ Q @ g.conj()))
        # exact G-dependent second moment; == const when G is unitary
        T = float(np.real(g.conj() @ _M_of_S(S) @ g))
        cf = (J + T) / 6 / p_s
        cfs.append(cf)
        Ftot += p_s * cf
    return float(Ftot), cfs


def _unitarity_dev(phi_table):
    """max_s ||G_s^dagger G_s - I||_max for G_s = V^dagger R_s W_s.

    Zero iff every branch maps range(P_s) isometrically into the code space,
    which is exactly the condition under which the `const` shortcut of
    `_branch_qform` is valid and `opt_unitary_ceiling` is the true ceiling."""
    phi = np.asarray(phi_table, dtype=float)
    R_all = m.recovery_unitary_batch(
        torch.tensor(phi, dtype=torch.float64)).detach().numpy()
    dev = 0.0
    for s in range(16):
        G = V_ISO.conj().T @ R_all[s] @ W_BASIS[s]
        dev = max(dev, float(np.abs(G.conj().T @ G - np.eye(2)).max()))
    return dev


def exact_F(phi_table, noise, p):
    """Noiseless Haar-averaged F̄ of an arbitrary per-syndrome unitary table.

    phi_table is (16, PHI_DIM) of ansatz angles; R_s = ansatz(phi_s).  Uses the
    closed second-moment identity
        F_s = [ sum_k |Tr(G_s A_k)|^2 + Tr(sum_k A_k^† A_k) ] / 6 / p_s ,
        G_s = V^† R_s W_s ,
    so the result has ZERO Monte-Carlo error -- gains of order 1e-5 (the exact
    coherent-over-rotation headroom) are resolvable, which the n_test=400
    protocol in `eval_phi_table` cannot do.  Cross-checked against that MC
    evaluator and against `sdp_ceiling`'s branch data in `_selftest_sdp`."""
    return _exact_F_from_qform(phi_table, _branch_qforms(noise, p))


def _pavg_qbar(B):
    """(Qbar, Mbar): the p-AVERAGED 4x4 branch forms.

    The Gauss-Legendre p-average of a sum of quadratic forms is again a sum of
    quadratic forms, so each of the 16 blocks of the training objective is

        obj_s(G_s) = g_s^T Qbar_s conj(g_s) + g_s^dagger Mbar_s g_s ,

    ONE pair of 4x4 forms -- which is both why the blocks are separable and why
    the decoder's eigenvector property survives the p-average.  The Mbar term is
    the G-dependent half of the Haar second moment (see `_branch_S`); for unitary
    G_s it collapses to the constant `coff[s]`, so `pavg_unitary_ceiling` needs
    only Qbar."""
    Qbar = np.zeros((16, 4, 4), dtype=complex)
    Mbar = np.zeros((16, 4, 4), dtype=complex)
    for s in range(16):
        for ip in range(B['n_p']):
            wip = float(B['w6'][ip])
            Qbar[s] += wip * np.asarray(B['Q'][s][ip], dtype=complex)
            Mbar[s] += wip * np.asarray(B['M'][s][ip], dtype=complex)
    return Qbar, Mbar


_PAVG_CEIL_CACHE = {}


def pavg_unitary_ceiling(B, n_start=16):
    """(F̄_cert, obj_cert[16]): the exact p-averaged VSCR-family optimum.

    sum_s max over 2x2 unitaries of obj_s, each block solved by the same
    `_max_unitary_J` search `opt_unitary_ceiling` uses at a single p.  This is
    the certified ceiling against which `refine_per_syndrome`'s captured
    headroom -- and Fig.4's gap-to-ceiling panel -- are measured.  Cached."""
    key = (B['noise'], tuple(B['p_range']), int(B['n_p']))
    if key not in _PAVG_CEIL_CACHE:
        Qbar, _Mbar = _pavg_qbar(B)
        Jm = np.array([_max_unitary_J(Qbar[s], n_start=n_start, seed=0)[0]
                       for s in range(16)])
        obj_cert = np.asarray(B['coff'], dtype=float) + Jm
        _PAVG_CEIL_CACHE[key] = (float(obj_cert.sum()), obj_cert)
    return _PAVG_CEIL_CACHE[key]


def pavg_bundle(noise, p_range=(0.02, 0.15), n_p=6):
    """Precompute the separable p-averaged objective for one channel.

    The training objective is the Gauss-Legendre p-average of F̄,
        F̄_avg = sum_p w_p sum_s p_s(p) cf_s(G_s; p) ,
    and because `cf_s = (J_s + const_s)/6/p_s` the branch weight CANCELS:
        p_s(p) cf_s(G_s; p) = (J_s(G_s; p) + const_s(p)) / 6 .
    Two consequences drive everything below.

    (1) SEPARABILITY.  The term for syndrome s involves only G_s, hence only
        R_s = ansatz(phi_s).  So F̄_avg = sum_s obj_s(phi_s) with the 16 blocks
        mutually independent -- even after the p-average, since reordering the
        two sums leaves each block a function of phi_s alone.  Optimising the
        blocks separately therefore reaches the GLOBAL optimum of the VSCR
        family, with no cross-syndrome coupling to coordinate.

    (2) CONDITIONING.  Dividing by p_s is what makes cf_s explode on rare
        syndromes; the cancelled form has no such division and stays exact even
        where p_s underflows.

    `_branch_qform` rebuilds the branch Kraus operators (about 0.3 s per call
    for depolarizing), so the 16 x n_p grid is materialised ONCE here and reused
    by every refinement start.  `_selftest_refine` asserts sum_s obj_s reproduces
    `ssvr_qec.vscr_fidelity_quad` on `quad_cache(p_range, noise)` to ~1e-14."""
    (p_nodes, p_w) = m.p_quadrature(p_range, n_p)
    p_nodes = np.asarray(p_nodes, dtype=float)
    p_w = np.asarray(p_w, dtype=float) / 6
    coff = []
    Q = []
    M = []
    for s in range(16):
        consts = []
        rows = []
        mrows = []
        for p in p_nodes:
            (Qs, cs, _ps) = _branch_qform(noise, float(p), s)
            rows.append(np.asarray(Qs, dtype=complex))
            mrows.append(_M_of_S(_branch_S(noise, float(p), s)))
            consts.append(float(cs))
        Q.append(np.stack(rows))
        M.append(np.stack(mrows))
        coff.append(float(np.dot(p_w, np.asarray(consts))))
    return {'noise': noise,
            'n_p': int(n_p),
            'p_range': (float(p_range[0]), float(p_range[1])),
            'p_nodes': p_nodes,
            'w6': p_w,
            'Q': Q,
            'coff': coff,
            'M': M,
            'Q_t': [torch.tensor(q, dtype=torch.complex128) for q in Q],
            'M_t': [torch.tensor(x, dtype=torch.complex128) for x in M],
            'coff_t': torch.tensor(coff, dtype=torch.float64)}


def pavg_F(phi_table, B):
    """Exact p-averaged F̄ and its 16 separable branch contributions.

    Returns (F̄_avg, obj) with obj[s] = the whole contribution of syndrome s, so
    sum(obj) == F̄_avg exactly.  `B` is a `pavg_bundle`.

    EXACT for an ARBITRARY (16, PHI_DIM) table: both halves of the Haar second
    moment are carried (the `Q` half and the G-dependent `M` half from
    `_branch_S`), so this equals `ssvr_qec.vscr_fidelity_quad` on
    `quad_cache(p_range, noise)` to ~1e-14 for any table, not only for
    code-preserving ones.  `_selftest_refine` test 1 asserts exactly that."""
    phi = np.asarray(phi_table, dtype=float).reshape(16, m.PHI_DIM)
    R_all = m.recovery_unitary_batch(
        torch.tensor(phi, dtype=torch.float64)).detach().numpy()
    obj = np.empty(16, dtype=float)
    for s in range(16):
        G = V_ISO.conj().T @ R_all[s] @ W_BASIS[s]
        g = np.asarray(G, dtype=complex).reshape(4)
        tot = 0.0
        for ip in range(B['n_p']):
            wip = float(B['w6'][ip])
            Qs = B['Q'][s][ip]
            Ms = B['M'][s][ip]
            tot += wip * (float(np.real(g @ Qs @ g.conj()))
                          + float(np.real(g.conj() @ Ms @ g)))
        obj[s] = tot
    return float(obj.sum()), obj


_P_SLOTS_NP = None


def _pauli_slots_np():
    """(PHI_DIM, 32, 32) complex128 numpy copy of `ssvr_qec.P_SLOTS`, built once."""
    global _P_SLOTS_NP
    if _P_SLOTS_NP is None:
        _P_SLOTS_NP = np.stack([P.numpy() for P in m.P_SLOTS])
    return _P_SLOTS_NP


def _block_obj_grad(phi_s, s, Qb, Mb):
    """EXACT obj_s and d obj_s / d phi_s in closed form -- no autograd.

        obj_s = Re(g^T Qb conj(g)) + Re(conj(g)^T Mb g),   g = vec_rowmajor(G),
        G = V^dagger R(phi) W_s ,   R(phi) = G_{K-1} ... G_0 ,
        G_k = cos(phi_k/2) I - i sin(phi_k/2) P_k .

    Introduce the prefix and suffix column blocks

        Y_k = G_{k-1}...G_0 W_s        (32x2),  Y_0 = W_s
        Z_k = V^dagger G_{K-1}...G_k   (2x32),  Z_K = V^dagger

    so that G = Z_k Y_k for every k (Z_k carries G_k, Y_k does not), and

        dG/dphi_k = Z_{k+1} (dG_k/dphi_k) Y_k
                  = -sin(phi_k/2)/2 * (Z_{k+1} Y_k)
                    - i cos(phi_k/2)/2 * (Z_{k+1} P_k Y_k).

    Both Qb and Mb are Hermitian, so both quadratic forms are real and their
    derivatives collapse to

        d obj_s/dphi_k = 2 Re[g^T Qb conj(dg_k)] + 2 Re[conj(g)^T Mb dg_k].

    The whole gradient is ~4e5 complex multiply-adds (about 1 ms) and it is
    verified against torch autograd to 1e-12 in `_selftest_refine`.  It was
    written to get autograd out of the refinement hot loop -- this host
    intermittently raised SIGSEGV and SIGILL inside
    `torch.autograd._engine_run_backward` after tens of thousands of tiny
    backward passes -- and it is also ~4x faster here.

    It does NOT remove the fault, and an earlier revision of this docstring
    wrongly claimed that pure numpy has no such failure mode.  This function is
    in fact now the most frequently observed crash frame.  Measured with
    `diag_native_fault.py` (400k calls of this function alone, single-threaded,
    nothing else running): SIGSEGV within the first 2e4 calls with
    OPENBLAS_CORETYPE=HASWELL, and SIGILL at ~2.5e5 calls with CORETYPE unset --
    while 2e7 bare (2,32)@(32,32) complex matmuls ran clean.  So the trigger is
    this operation mix, not GEMM and not autograd.  Read the long note in
    `ssvr_qec.py` before "fixing" this.  The mitigation that works is subprocess
    isolation with retry (`vscr_paper._refine_isolated`, plus per-test isolation
    in `run_selftests.py`), which absorbs a fault at the cost of one retry."""
    P_np = _pauli_slots_np()
    K = P_np.shape[0]
    phi = np.asarray(phi_s, dtype=float)
    half = 0.5 * phi
    c = np.cos(half)
    sn = np.sin(half)
    W_s = W_BASIS[s]
    Y = np.empty((K + 1, m.DIM, 2), dtype=complex)
    Y[0] = W_s
    for k in range(K):
        Y[k + 1] = c[k] * Y[k] - 1j * sn[k] * (P_np[k] @ Y[k])
    Z = np.empty((K + 1, 2, m.DIM), dtype=complex)
    Z[K] = V_ISO.conj().T
    for k in range(K - 1, -1, -1):
        Z[k] = c[k] * Z[k + 1] - 1j * sn[k] * (Z[k + 1] @ P_np[k])
    G = Z[0] @ Y[0]
    g = G.reshape(4)
    gc = g.conj()
    obj = float(np.real(g @ Qb @ gc) + np.real(gc @ Mb @ g))
    grad = np.empty(K, dtype=float)
    for k in range(K):
        Zk1 = Z[k + 1]
        dG = (-0.5 * sn[k]) * (Zk1 @ Y[k]) \
            + (-0.5j * c[k]) * ((Zk1 @ P_np[k]) @ Y[k])
        dg = dG.reshape(4)
        grad[k] = 2.0 * float(np.real(g @ Qb @ dg.conj())
                              + np.real(gc @ Mb @ dg))
    return obj, grad


def _adam_numpy(f_grad, x0, steps, lr, b1=0.9, b2=0.999, eps=1e-8):
    """Plain Adam in numpy, matching torch's default betas/eps and update rule
    `x -= lr * mhat / (sqrt(vhat) + eps)`, so `steps`/`lr` keep their meaning."""
    x = np.array(x0, dtype=float)
    mnt = np.zeros_like(x)
    vnt = np.zeros_like(x)
    for t in range(1, int(steps) + 1):
        _f, g = f_grad(x)
        mnt = b1 * mnt + (1.0 - b1) * g
        vnt = b2 * vnt + (1.0 - b2) * (g * g)
        mh = mnt / (1.0 - b1 ** t)
        vh = vnt / (1.0 - b2 ** t)
        x = x - lr * mh / (np.sqrt(vh) + eps)
    return x


# Large symmetry-breaking offsets for `refine_per_syndrome`.  Offsets in
# [1e-4, 1e-1] were measured to capture -0.52% .. -0.03% of the headroom -- a
# small random 60-angle offset lies almost entirely in the ansatz's null
# directions and Adam then random-walks -- so the offsets here are O(1) radians.
_REFINE_SCALES = (1.0, 2.0, 0.5)


def refine_per_syndrome(phi_init, noise, p_range=(0.02, 0.15), n_p=6, n_start=3,
                        steps=1500, lr=0.01, seed=0, bundle=None, verbose=False):
    """Escape the decoder saddle by optimising each syndrome INDEPENDENTLY.

    WHY THIS IS NEEDED.  Each branch term is a 4x4 Rayleigh quotient,
        obj_s ∝ g_s^† Q_s g_s ,   g_s = vec(V^† R_s W_s) ,  ||g_s|| fixed,
    and the gradient of a Rayleigh quotient vanishes EXACTLY at the eigenvectors
    of Q_s.  The Pauli decoder's vec(G_dec) is such an eigenvector on every
    branch of every channel (measured eigen-residual ~1e-16 on depolarizing; on
    coherent the decoder sits in Q's null space, i.e. an eigenvector with
    eigenvalue ~0).  So `PHI_DEC` is an exact stationary point of the warm-start
    objective -- not a small-gradient region but a true zero, confirmed by
    central finite differences as well as autograd.  Wherever
    λ(G_dec) < λ_max(Q_s) the decoder is a SADDLE, and the certified headroom
    F̄_unit − F̄_dec is precisely that eigenvalue gap.  No first-order method
    started at PHI_DEC can move off it, which is why warm training captured 0%
    of the headroom while `Adam` still visibly moved the parameters: it was
    normalising round-off-level gradients and drifting, not descending.

    Small symmetry-breaking perturbations do NOT fix this -- measured captured
    headroom of −0.52% to −0.03% for eps in [1e-4, 1e-1], because a random
    60-angle offset is almost entirely in the ansatz's null directions and Adam
    then random-walks.  What DOES work is exploiting separability: optimise each
    of the 16 blocks on its own from a few LARGE symmetry-broken starts.  The
    ansatz is provably expressive enough -- on amplitude damping p=0.06 branch 2
    it reaches `opt_unitary_ceiling`'s certified optimum to −1.0e−15 -- and the
    full table then captures essentially all of the certified headroom.

    Start 0 is always `phi_init[s]` and a candidate is only accepted if it
    strictly improves that block's objective, so the result can NEVER be worse
    than the input table -- the refinement is monotone by construction.

    Returns (phi, info); phi is (16, PHI_DIM).
    """
    phi_init = np.asarray(phi_init, dtype=float).reshape(16, m.PHI_DIM)
    B = pavg_bundle(noise, p_range, n_p) if bundle is None else bundle
    # the p-average collapses each block to ONE pair of 4x4 quadratic forms
    Qbar, Mbar = _pavg_qbar(B)

    F0, obj0 = pavg_F(phi_init, B)
    phi = phi_init.copy()
    obj1 = obj0.copy()
    rng = np.random.RandomState(seed)
    n_improved = 0
    for s in range(16):
        Qb, Mb = Qbar[s], Mbar[s]

        def neg_fg(x, _Qb=Qb, _Mb=Mb, _s=s):
            """-obj_s and its exact gradient; `_block_obj_grad` is closed-form,
            so the whole refinement runs without a single autograd backward."""
            ov, gr = _block_obj_grad(x, _s, _Qb, _Mb)
            return -ov, -gr

        best_val, best_phi = obj0[s], phi_init[s].copy()
        for k in range(max(1, int(n_start))):
            if k == 0:
                init = phi_init[s].copy()
            else:
                init = (phi_init[s]
                        + rng.normal(size=m.PHI_DIM)
                        * _REFINE_SCALES[(k - 1) % len(_REFINE_SCALES)])
            cur = _adam_numpy(neg_fg, init, steps, lr)
            val = _block_obj_grad(cur, s, Qb, Mb)[0]
            if val > best_val + 1e-15:              # STRICT -> monotone
                best_val, best_phi = val, cur.copy()
        if best_val > obj0[s] + 1e-15:
            n_improved += 1
        phi[s] = best_phi
        obj1[s] = best_val
        if verbose:
            print(f'    [refine {noise}] s={s:2d}  obj {obj0[s]:.12f} -> '
                  f'{obj1[s]:.12f}  (+{obj1[s] - obj0[s]:.3e})', flush=True)

    F1 = float(obj1.sum())
    # the block-summed objective must equal the direct evaluation of the table
    Fchk, objchk = pavg_F(phi, B)
    assert abs(Fchk - F1) < 1e-12, (Fchk, F1)
    assert float(np.abs(objchk - obj1).max()) < 1e-12
    # MONOTONE by construction: start 0 is always phi_init[s] and a candidate is
    # accepted only on strict improvement, so this can never regress.
    assert F1 >= F0 - 1e-15, f'refinement lowered F̄: {F0} -> {F1}'
    # cross-check against the PRODUCTION objective on the same p-grid
    Qc = m.quad_cache(B['p_range'], noise, n_p=B['n_p'])
    _, Fprod, _ = m.vscr_fidelity_quad(torch.tensor(phi, dtype=torch.float64), Qc)
    prod_err = abs(F1 - float(Fprod))
    assert prod_err < 1e-12, \
        f'block objective != production objective: {prod_err:.2e}'
    # rigorous upper bound: obj_s = g^dagger (Qbar^T + Mbar) g and ||g||^2 <= 2
    A_op = Qbar.transpose(0, 2, 1) + Mbar
    F_bound = 0.0
    for s in range(16):
        Ah = 0.5 * (A_op[s] + A_op[s].conj().T)
        F_bound += 2.0 * float(np.linalg.eigvalsh(Ah).max())
    assert F1 <= F_bound + 1e-12, f'F̄ above rigorous bound: {F1} > {F_bound}'
    F_cert, obj_cert = pavg_unitary_ceiling(B)
    # `F_cert` is the exact optimum of the CODE-PRESERVING (unitary-G) family, so
    # it is a meaningful ceiling only while the refined table stays in that
    # family; `unit_dev` measures how far it has left it.  With a converged
    # budget unit_dev is ~1e-5 or better (asserted in `_selftest_refine`); with a
    # deliberately tiny budget (ABL_SMOKE) the blocks are simply unconverged and
    # F_cert need not bound F1, so this is REPORTED rather than asserted.  The
    # bound that always holds is the rigorous `F_bound`, asserted above, and the
    # guarantee that actually matters -- F1 >= F0 on the TRUE production
    # objective -- is asserted too.
    unit_dev = _unitarity_dev(phi)
    cert_valid = bool(unit_dev < 1e-4)
    if cert_valid:
        assert F1 <= F_cert + 1e-9, f'F̄ above certified ceiling: {F1} > {F_cert}'
    info = {'noise': noise, 'p_range': tuple(B['p_range']), 'n_p': int(B['n_p']),
            'F_before': float(F0), 'F_after': float(F1), 'F_cert': float(F_cert),
            'F_prod': float(Fprod), 'prod_err': float(prod_err),
            'F_bound': float(F_bound), 'unitarity_dev': float(unit_dev),
            'cert_valid': cert_valid,
            'obj_before': obj0, 'obj_after': obj1, 'obj_cert': obj_cert,
            'headroom': float(F_cert - F0), 'captured': float(F1 - F0),
            'capture_frac': (float(F1 - F0) / float(F_cert - F0)
                             if F_cert > F0 + 1e-18 else 1.0),
            'n_improved': int(n_improved), 'n_start': int(n_start),
            'steps': int(steps), 'lr': float(lr), 'seed': int(seed),
            'phi': phi.copy()}
    if verbose:
        print(f'    [refine {noise}] F̄ {F0:.9f} -> {F1:.9f}  (certified ceiling '
              f'{F_cert:.9f}, captured {100 * info["capture_frac"]:.4f}% of '
              f'headroom, {n_improved}/16 blocks improved)', flush=True)
    return phi, info


def _selftest_refine(noise='amplitude_damping', n_start=3, steps=900,
                     p_range=(0.02, 0.15), p_test=0.06):
    """Prove the separable refinement is EXACT and captures the headroom.

    Test 1 (identity): `pavg_F` -- which cancels p_s against cf_s and splits the
    objective into 16 independent blocks -- must reproduce the PRODUCTION
    training objective `ssvr_qec.vscr_fidelity_quad` on `quad_cache(p_range,
    noise)`.  If these disagreed, the refinement would be optimising something
    other than what training and Fig.3/Fig.4 report.

    Test 2 (stationarity): the exact gradient of the p=0.06 training objective
    w.r.t. all 960 warm-start angles must vanish at PHI_DEC to round-off.  This
    is the mechanism behind 0% headroom capture -- the decoder is a stationary
    point, not a merely flat region -- so it is asserted, not just observed.

    Test 3 (monotone + capture): `refine_per_syndrome` must never lower F̄, must
    stay at or below the certified ceiling F̄_unit, and must capture most of the
    certified headroom F̄_unit - F̄_dec.
    """
    # ---- Test 1: the separable objective IS the production objective ----
    B = pavg_bundle(noise, p_range, 6)
    Q = m.quad_cache(p_range, noise, n_p=6)
    g = torch.Generator().manual_seed(7)
    tabs = {'PHI_DEC': vp.PHI_DEC.numpy(),
            'random': (torch.randn(16, m.PHI_DIM, dtype=torch.float64,
                                   generator=g) * 0.6).numpy()}
    worst = worst_s = 0.0
    for tag, tab in tabs.items():
        Fa, obj = pavg_F(tab, B)
        _, Fq, F_s = m.vscr_fidelity_quad(
            torch.tensor(tab, dtype=torch.float64), Q)
        worst = max(worst, abs(Fa - float(Fq)))
        worst_s = max(worst_s, float(np.abs(obj - F_s.numpy()).max()))
    assert worst < 1e-13, f'pavg_F != vscr_fidelity_quad: {worst:.2e}'
    assert worst_s < 1e-13, f'per-block obj_s != F_s: {worst_s:.2e}'
    print(f'[refine selftest 1] pavg_F == production vscr_fidelity_quad on '
          f'quad_cache{tuple(p_range)} [{noise}] for the decoder AND a random '
          f'table: dF={worst:.2e}, dF_s={worst_s:.2e}  VERIFIED', flush=True)
    # the G-independent `const` shortcut is valid exactly on the code-preserving
    # family; pin down where that approximation stops being exact.
    dev_dec = _unitarity_dev(vp.PHI_DEC)
    dev_rnd = _unitarity_dev(tabs['random'])
    assert dev_dec < 1e-12, dev_dec
    assert dev_rnd > 1e-3, dev_rnd
    print(f'[refine selftest 1b] decoder table is code-preserving to '
          f'{dev_dec:.2e} (random table: {dev_rnd:.2e}), so the `const` term is '
          f'exact for the decoder and for every branch optimum but NOT for '
          f'arbitrary tables -- which is why `pavg_F` carries the full '
          f'Tr(G S G^dag)', flush=True)

    # ---- Test 1c: the CLOSED-FORM block gradient == autograd == FD ----
    # The refinement hot loop no longer calls torch autograd at all (see
    # `_block_obj_grad`), so its value and gradient must be pinned against both
    # an independent autograd path and central finite differences.
    Qb_all, Mb_all = _pavg_qbar(B)
    g2 = torch.Generator().manual_seed(11)
    tab = (torch.randn(16, m.PHI_DIM, dtype=torch.float64,
                       generator=g2) * 0.7).numpy()
    W_t = torch.tensor(W_BASIS, dtype=torch.complex128)
    V_t = torch.tensor(V_ISO, dtype=torch.complex128)
    Qt_t = torch.tensor(Qb_all, dtype=torch.complex128)
    Mt_t = torch.tensor(Mb_all, dtype=torch.complex128)
    _, obj_tab = pavg_F(tab, B)
    worst_v = worst_g = worst_fd = scale = 0.0
    for s in (0, 2, 7, 15):
        Qb, Mb = Qb_all[s], Mb_all[s]
        ov, gr = _block_obj_grad(tab[s], s, Qb, Mb)
        worst_v = max(worst_v, abs(ov - float(obj_tab[s])))
        scale = max(scale, float(np.abs(gr).max()))
        # independent reference: autograd through the production column path
        pt = torch.tensor(tab[s], dtype=torch.float64, requires_grad=True)
        RW = m.recovery_action_cols(pt.reshape(1, -1), W_t[s:s + 1])[0]
        Gt = (V_t.conj().T @ RW).reshape(4)
        valt = (torch.real(Gt @ Qt_t[s] @ Gt.conj())
                + torch.real(Gt.conj() @ Mt_t[s] @ Gt))
        valt.backward()
        worst_v = max(worst_v, abs(float(valt) - ov))
        worst_g = max(worst_g, float(np.abs(pt.grad.numpy() - gr).max()))
        # central finite differences on a subset of the 60 angles
        h = 1e-6
        for k in range(0, m.PHI_DIM, 11):
            pp = tab[s].copy(); pp[k] += h
            pm = tab[s].copy(); pm[k] -= h
            fd = (_block_obj_grad(pp, s, Qb, Mb)[0]
                  - _block_obj_grad(pm, s, Qb, Mb)[0]) / (2 * h)
            worst_fd = max(worst_fd, abs(fd - gr[k]))
    assert worst_v < 1e-13, f'closed-form obj != pavg_F/autograd: {worst_v:.2e}'
    assert worst_g < 1e-11 * max(scale, 1.0), \
        f'closed-form grad != autograd: {worst_g:.2e} (scale {scale:.2e})'
    assert worst_fd < 1e-6 * max(scale, 1.0), \
        f'closed-form grad != central FD: {worst_fd:.2e} (scale {scale:.2e})'
    print(f'[refine selftest 1c] closed-form block objective and gradient match '
          f'pavg_F and torch autograd to {worst_v:.2e} / {worst_g:.2e} '
          f'(gradient scale {scale:.2e}), and central FD to {worst_fd:.2e}, on '
          f'4 blocks x 6 angles  VERIFIED', flush=True)

    # ---- Test 2: PHI_DEC is an EXACT stationary point (a saddle) ----
    qf = _branch_qforms(noise, p_test)
    Qt = torch.tensor(np.stack([q[0] for q in qf]), dtype=torch.complex128)
    Mt = torch.tensor(np.stack([_M_of_S(q[3]) for q in qf]),
                      dtype=torch.complex128)
    W_t = torch.tensor(W_BASIS, dtype=torch.complex128)
    V_t = torch.tensor(V_ISO, dtype=torch.complex128)

    def obj_at_p(phi_t):
        """F̄(p_test) = sum_s (J_s + Tr(G_s S_s G_s^†))/6, differentiable."""
        RW = m.recovery_action_cols(phi_t.reshape(16, -1), W_t)
        G = torch.matmul(V_t.conj().T.unsqueeze(0), RW)
        gg = G.reshape(16, 4)
        J = torch.einsum('si,sij,sj->s', gg, Qt, gg.conj()).real
        T = torch.einsum('si,sij,sj->s', gg.conj(), Mt, gg).real
        return ((J + T) / 6).sum()

    phi_t = vp.PHI_DEC.clone().requires_grad_(True)
    obj_at_p(phi_t).backward()
    gmax = float(phi_t.grad.abs().max())
    Fex, _ = _exact_F_from_qform(vp.PHI_DEC, qf)
    assert abs(float(obj_at_p(phi_t.detach())) - Fex) < 1e-12
    h = 1e-6
    base = vp.PHI_DEC.numpy().copy()
    fdmax = 0.0
    for s in range(16):
        for k in range(0, m.PHI_DIM, 7):
            pp = base.copy(); pp[s, k] += h
            pm = base.copy(); pm[s, k] -= h
            fdmax = max(fdmax, abs(_exact_F_from_qform(pp, qf)[0]
                                   - _exact_F_from_qform(pm, qf)[0]) / (2 * h))
    assert gmax < 1e-11, f'autograd gradient at PHI_DEC is {gmax:.2e}, not zero'
    assert fdmax < 1e-9, f'finite-difference gradient at PHI_DEC is {fdmax:.2e}'
    ouc = opt_unitary_ceiling(noise, p_test)
    assert abs(ouc['F_dec'] - Fex) < 1e-10, (ouc['F_dec'], Fex)
    print(f'[refine selftest 2] PHI_DEC is an EXACT stationary point of F̄ at '
          f'{noise} p={p_test}: |autograd|max={gmax:.2e}, |central FD|max='
          f'{fdmax:.2e}  VERIFIED', flush=True)
    hr = ouc['F_unit'] - ouc['F_dec']
    verdict = ('the decoder is a SADDLE, not an optimum' if hr > 1e-12 else
               'headroom is EXACTLY zero: the Pauli decoder is already the '
               'optimal unitary recovery for this channel, so warm == decoder '
               'is correct physics here rather than an optimisation failure')
    print(f'[refine selftest 2b] F̄_dec={Fex:.9f}  F̄_unit={ouc["F_unit"]:.9f}  '
          f'certified headroom={hr:.3e} -> {verdict}', flush=True)

    # ---- Test 3: monotone, below the ceiling, captures the headroom ----
    phi_ref, info = refine_per_syndrome(vp.PHI_DEC, noise, p_range=p_range,
                                        n_start=n_start, steps=steps, seed=0,
                                        bundle=B)
    assert info['F_after'] >= info['F_before'] - 1e-15, info
    assert info['F_after'] <= info['F_cert'] + 1e-12, info
    assert info['F_after'] <= info['F_bound'] + 1e-12, info
    # with a CONVERGED budget the refined table must still be code-preserving,
    # which is what makes F_cert a valid ceiling and capture_frac meaningful
    assert info['unitarity_dev'] < 1e-3, \
        (f"converged refinement left the code-preserving regime: "
         f"{info['unitarity_dev']:.2e}")
    assert info['cert_valid'], info['unitarity_dev']
    cap = info['capture_frac']
    assert cap > 0.5, f'refinement captured only {100 * cap:.3f}% of headroom'
    F_ref, _ = _exact_F_from_qform(phi_ref, qf)
    assert F_ref >= Fex - 1e-15, (F_ref, Fex)
    print(f'[refine selftest 3] {noise}: F̄ {info["F_before"]:.9f} -> '
          f'{info["F_after"]:.9f} (certified ceiling {info["F_cert"]:.9f}, '
          f'rigorous bound {info["F_bound"]:.9f}), captured '
          f'{100 * cap:.4f}% of headroom, {info["n_improved"]}/16 blocks '
          f'improved', flush=True)
    print(f'[refine selftest 3b] block objective == production objective to '
          f'{info["prod_err"]:.2e}; refined table stays code-preserving to '
          f'{info["unitarity_dev"]:.2e}; at p={p_test}: {Fex:.9f} -> '
          f'{F_ref:.9f} (F̄_unit={ouc["F_unit"]:.9f})  VERIFIED', flush=True)
    return info


def sdp_ceiling(noise, p):
    """F̄^CPTP ceiling and per-branch data for a channel at strength p.

    Each branch's primal is now seeded with the EXACT unitary-family optimum from
    `opt_unitary_branch`, and the result is asserted to satisfy

        F̄^CPTP >= max(F̄^unitary, F̄^decoder) ,

    the weak-duality sanity check the previous version lacked (see the
    `_sdp_branch` docstring for the negative-ceiling and
    ceiling-below-a-feasible-point failures it produced).  The unitary-family
    numbers are returned alongside so callers can report how much of the CPTP gap
    a per-syndrome UNITARY recovery can actually close -- which is the quantity
    VSCR is able to reach, as opposed to what an unrestricted CPTP map could
    reach in principle."""
    ouc = opt_unitary_ceiling(noise, p)
    Fs, ps = [], []
    for s in range(16):
        M, p_s = _branch_M(noise, p, s)
        G_dec = V_ISO.conj().T @ m.C_SYNDS[s].numpy() @ W_BASIS[s]   # 2x2
        G_ref = np.asarray(ouc['G_opt'][s], dtype=complex)
        fu, _ = _sdp_branch(M, G_dec, G_ref=G_ref)
        Fs.append(fu / max(p_s, 1e-15) if p_s > 1e-12 else 1.0)
        ps.append(p_s)
    F_bar = float(np.dot(ps, Fs))
    assert F_bar >= ouc['F_unit'] - 1e-9, \
        (f'CPTP ceiling {F_bar:.12f} is below the unitary-family optimum '
         f'{ouc["F_unit"]:.12f} for {noise} p={p}')
    assert F_bar >= ouc['F_dec'] - 1e-9, \
        (f'CPTP ceiling {F_bar:.12f} is below the decoder '
         f'{ouc["F_dec"]:.12f} for {noise} p={p}')
    return {'F_cptp': F_bar, 'F_s_max': [float(x) for x in Fs],
            'p_s': [float(x) for x in ps],
            'F_unit': float(ouc['F_unit']),
            'F_dec_exact': float(ouc['F_dec']),
            'cf_unit': [float(x) for x in ouc['cf_unit']]}


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
GAP_METHODS = ['Perfect-code decoder', 'VSCR warm (ours)',
               'VQR-ind (no sharing)', 'Global unitary (no proj.)']
GAP_COLORS = {'Perfect-code decoder': '#e41a1c', 'VSCR warm (ours)': '#1f4e9c',
              'VQR-ind (no sharing)': '#377eb8',
              'Global unitary (no proj.)': '#a65628'}
CHAN_SHORT = {'depolarizing': 'depol.', 'amplitude_damping': 'amp.\ndamping',
              'mixed': 'mixed', 'coherent': 'coherent'}
GAP_FLOOR = 3e-8          # log-axis stand-in for an EXACTLY zero gap


def plot_ablation(bar_vals, lowdata, gap=None, fname='fig_ablation'):
    """Fig.4: same-family baselines, gap to the certified ceiling, low data.

    FIX D -- why panel (a) changed channel.  It used to be the depolarizing
    p=0.10 comparison, where the decoder, VSCR-warm, VQR-ind and the SDP CPTP
    ceiling were 0.9469945679012, 0.9469945678940, 0.9469945678990 and
    0.9469945679015 -- four bars agreeing to 7e-12.  That is neither a plotting
    bug nor a failure of the method: `sdp_ceiling` reports max_branch_gap = 0.0
    EXACTLY on depolarizing, i.e. for that channel the rigid Pauli decoder is
    already the optimal CPTP recovery, so nothing in the family can be separated
    from it and the panel is non-discriminative by physics.

    Panel (a) therefore now uses AMPLITUDE DAMPING at p=0.10, where the
    certified ceiling exceeds the decoder by 7.2e-4 (and the worst single branch
    by 0.195), so the bars actually differ.  Panel (b) is NEW: the gap to the
    certified CPTP ceiling, log scale, for all four channels.  A log gap is the
    only presentation in which an exact 0, a 1e-6, a 7e-4 and a 3e-3 are
    simultaneously legible, and it states the real claim -- VSCR closes the gap
    where one exists and matches the ceiling exactly where none does.  Bars
    sitting on the floor line are EXACTLY zero (hatched, labelled '0')."""
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 2.95))

    # ---- (a) discriminative bar chart ----
    ax = axes[0]
    labels = list(bar_vals.keys())
    vals = [bar_vals[k] for k in labels]
    cols = ['#e41a1c', '#1f4e9c', '#377eb8', '#a65628', '#bbbbbb']
    bars = ax.bar(range(len(vals)), vals, 0.62, color=cols[:len(vals)])
    bars[-1].set_hatch('//')
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=6.5)
    ax.set_ylabel('$\\bar{F}$ at $p=0.10$ (amplitude damping)')
    lo = min(vals)
    ax.set_ylim(max(0.0, lo - 0.06), max(vals) + 0.03)
    ax.grid(alpha=0.3, lw=0.5, axis='y')
    ax.set_title('same-family baselines & CPTP ceiling', fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f'{v:.5f}',
                ha='center', fontsize=6.0)

    # ---- (b) log-scale gap to the certified ceiling ----
    ax = axes[1]
    if gap:
        chans = [c for c in ('depolarizing', 'amplitude_damping', 'mixed',
                             'coherent') if c in gap]
        x = np.arange(len(chans))
        bw = 0.8 / max(len(GAP_METHODS), 1)
        for j, mth in enumerate(GAP_METHODS):
            if not all(mth in gap[c] for c in chans):
                continue
            v = np.array([max(float(gap[c][mth]), 0.0) for c in chans])
            vz = np.maximum(v, GAP_FLOOR)
            xx = x + (j - (len(GAP_METHODS) - 1) / 2) * bw
            bb = ax.bar(xx, vz, bw, color=GAP_COLORS[mth], label=mth)
            for xi, bi, vv in zip(xx, bb, v):
                if vv <= 0.0:                       # exactly optimal
                    bi.set_hatch('///'); bi.set_edgecolor('k')
                    ax.text(xi, GAP_FLOOR * 1.25, '0', ha='center', va='bottom',
                            fontsize=5.5, rotation=90, color=GAP_COLORS[mth])
        ax.set_yscale('log')
        ax.set_ylim(GAP_FLOOR / 4, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([CHAN_SHORT.get(c, c) for c in chans], fontsize=6.5)
        ax.set_ylabel('$\\bar{F}^{\\mathrm{CPTP}} - \\bar{F}$  (log)')
        ax.grid(alpha=0.3, lw=0.5, which='both', axis='y')
        ax.legend(fontsize=5.8, loc='upper left', ncol=1, framealpha=0.9)
        ax.set_title('gap to certified ceiling, $p=0.10$\n(hatched = exactly 0)',
                     fontsize=9)
    else:
        ax.text(0.5, 0.5, 'no ceiling data', ha='center', va='center',
                transform=ax.transAxes, fontsize=8)
        ax.set_title('gap to certified ceiling', fontsize=9)

    # ---- (c) low-data regime ----
    ax = axes[2]
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
    ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=6.0, loc='lower right')
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
        phi_grad = model().detach().cpu().numpy()
        # Apply the SAME Fix A/B post-processing as the warm-VSCR pipeline.  The
        # refinement is architecture-blind -- it acts on the (16, PHI_DIM) table
        # -- so without this the ablation would compare a refined hypernetwork
        # model against an UNrefined table and credit the architecture with the
        # refinement's gain (9.9e-3 on amplitude damping, 3.1e-3 on coherent).
        phi, rinfo = vp.refine_with_floor(
            phi_grad, noise, tuple(SCHED[noise][-1][2]), verbose=True,
            tag='/ind', n_start=(2 if SMOKE else None),
            steps=(250 if SMOKE else None))
        phi = np.asarray(phi, dtype=float)
        infos.append({'stage': 'refine', 'picked': rinfo['picked'],
                      'F_decoder': float(rinfo['scores']['decoder']),
                      'F_grad': float(rinfo['scores']['grad']),
                      'F_refined': float(rinfo['scores']['refined']),
                      'capture_frac': float(rinfo['capture_frac'])})
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
                    # NOTE: VSCRWarm now zero-inits ONLY the hypernetwork OUTPUT
                    # layer hnet[2] -- that is what makes the warm start exactly
                    # the decoder.  hnet[0] keeps a non-degenerate
                    # std = 1/sqrt(ctx) init, because a double-zero tanh-MLP is a
                    # dead gradient point: tanh(0) = 0 makes hnet[2]'s gradient
                    # vanish too, so neither layer could ever move (see the
                    # VSCRWarm docstring and `_selftest_hypernet_trainable`).
                    # For the COLD comparison we randomise hnet[2] as well, so
                    # the model starts away from the decoder with the same
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
                   ('mixed', 0.10), ('coherent', 0.10), ('coherent', 0.15),
                   ('coherent', 0.30)]
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

    # ---------------- E3b: gap to the certified ceiling (Fig.4b) -----------
    i10 = int(np.where(VP_P == 0.10)[0][0])
    P_GAP = float(VP_P[i10])
    print(f'\n=== E3b: gap to the certified CPTP ceiling (p={P_GAP}) ===',
          flush=True)
    gap = {}
    for noise in CHANNELS:
        key = f'{SHORT[noise]}_{P_GAP}'
        if key not in sdp:
            r = sdp_ceiling(noise, P_GAP)
            Fdec = _decoder_ref_F(noise, P_GAP)
            cf_dec, w_dec = vp.cf_table(m.recovery_unitary_batch(vp.PHI_DEC),
                                        noise, P_GAP, M=200)
            bg = [r['F_s_max'][s] - cf_dec[s]
                  for s in range(16) if w_dec[s] > 1e-3]
            r['F_dec'] = Fdec
            r['max_branch_gap'] = float(max(bg)) if bg else 0.0
            r['F_gap_bar'] = r['F_cptp'] - Fdec
            sdp[key] = r
        Fc = float(sdp[key]['F_cptp'])
        Fw = float(eval_phi_table(np.load(WARM_NPZ[noise])['phi'], noise,
                                  np.array([P_GAP]))[0][0])
        # A NEGATIVE gap would mean the "ceiling" is below a feasible recovery,
        # i.e. the SDP primal stalled -- exactly the failure the `G_ref` seeding
        # in `_sdp_branch` was added to prevent.  Fail loudly here rather than
        # letting `plot_ablation` clamp it to zero and draw a bogus "exactly
        # optimal" bar.
        for nm, Fm in (('decoder', float(sdp[key]['F_dec'])), ('warm', Fw),
                       ('ind', float(ind[noise]['F'][i10])),
                       ('global', float(glob[noise]['F'][i10]))):
            assert Fm <= Fc + 1e-9, \
                (f'{noise} p={P_GAP}: {nm} F={Fm:.12f} exceeds the certified '
                 f'CPTP ceiling {Fc:.12f} -- the SDP primal is not optimal')
        gap[noise] = {
            'F_cptp': Fc, 'F_dec': float(sdp[key]['F_dec']),
            'F_warm': Fw,
            'Perfect-code decoder': Fc - float(sdp[key]['F_dec']),
            'VSCR warm (ours)': Fc - Fw,
            'VQR-ind (no sharing)': Fc - float(ind[noise]['F'][i10]),
            'Global unitary (no proj.)': Fc - float(glob[noise]['F'][i10]),
            'max_branch_gap': float(sdp[key]['max_branch_gap'])}
        print(f'  {noise:20s} F_cptp={Fc:.9f}  ' + '  '.join(
            f'{k}={gap[noise][k]:+.2e}' for k in GAP_METHODS), flush=True)

    # ---------------- figure + save ---------------------------------------
    # FIX D: panel (a) moves to AMPLITUDE DAMPING, where the certified ceiling
    # exceeds the decoder by 7.2e-4.  On depolarizing the two are equal to
    # ~7e-12 because the rigid Pauli decoder is already exactly CPTP-optimal
    # there (max_branch_gap == 0.0), so no member of the family can be
    # separated from it and the old panel was non-discriminative by physics.
    # The depolarizing numbers are still computed and saved as `bar_dep_p010`
    # so the claim "decoder == ceiling on depolarizing" stays auditable.
    def _bars(noise):
        Fw = float(eval_phi_table(np.load(WARM_NPZ[noise])['phi'], noise,
                                  np.array([P_GAP]))[0][0])
        return {'Perfect-code\ndecoder': gap[noise]['F_dec'],
                'VSCR warm\n(hypernet)': Fw,
                'VQR-ind\n(no sharing)': ind[noise]['F'][i10],
                'Global unitary\n(no projection)': glob[noise]['F'][i10],
                'CPTP ceiling\n(SDP)': gap[noise]['F_cptp']}

    bar_vals = _bars('amplitude_damping')
    bar_vals_dep = _bars('depolarizing')
    if not SMOKE:
        plot_ablation(bar_vals, lowdata, gap=gap)

    np.savez('vscr_paper_abl_results.npz',
             p_values=VP_P,
             ind={n: ind[n] for n in ind},
             glob={n: glob[n] for n in glob},
             lowdata=lowdata, sdp=sdp, pauli=pl,
             bar_vals=bar_vals, bar_vals_dep=bar_vals_dep, gap=gap,
             n_params=N_PARAMS,
             runtime=np.array([time.time() - t0]))
    num = json.load(open('paper_numbers.json'))
    num['abl'] = {'ind': ind, 'global': glob, 'lowdata': lowdata, 'sdp': sdp,
                  'pauli_lookup': {k: {kk: vv for kk, vv in v.items()
                                       if kk != 'cf_class'}
                                   for k, v in pl.items()},
                  # FIX D: `bar_ad_p010` is what Fig.4(a) now plots; the
                  # depolarizing set is kept so the "decoder == CPTP ceiling"
                  # identity that motivated the change stays auditable.
                  'bar_ad_p010': bar_vals, 'bar_dep_p010': bar_vals_dep,
                  'gap_to_ceiling': gap, 'n_params': N_PARAMS,
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
