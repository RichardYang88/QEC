# Source Generated with Decompyle++
# File: vscr_paper_abl.cpython-311.pyc (Python 3.11)

'''
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
merges \'abl_*\' fields into paper_numbers.json.
'''
import itertools
import json
import math
import os
import time
import numpy as np
import torch
from torch.nn import nn
from scipy.optimize import minimize
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import pyplot as plt
import ssvr_qec as m
import vscr_paper as vp
SMOKE = os.environ.get('ABL_SMOKE', '0') == '1'
SCHED = dict(vp.WARM_SCHEDULES)
SCHED['coherent'] = [
    (400, 0.003, (0.02, 0.08)),
    (500, 0.003, (0.02, 0.15)),
    (800, 0.0003, (0.02, 0.15))]
CHANNELS = [
    'depolarizing',
    'amplitude_damping',
    'mixed',
    'coherent']
SHORT = {
    'depolarizing': 'dep',
    'amplitude_damping': 'ad',
    'mixed': 'mixed',
    'coherent': 'coh' }
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'figure.dpi': 150 })

class VQRInd(nn.Module):
    '''Per-syndrome independent angle table (no hypernetwork, no sharing).
    960 parameters vs 5944 for the hypernetwork model at n=5.'''
    
    def __init__(self = None, warm = None):
        pass
    # WARNING: Decompyle incomplete

    
    def forward(self):
        return self.phi

    __classcell__ = None


class GlobalRec(nn.Module):
    '''Single global variational unitary (NO syndrome projection);
    QVECTOR-style variational recovery baseline. 60 parameters.'''
    
    def __init__(self = None):
        pass
    # WARNING: Decompyle incomplete

    
    def forward(self):
        return self.phi

    __classcell__ = None

N_PARAMS = {
    'VSCR hypernet': 5944,
    'VQR-ind table': 16 * m.PHI_DIM,
    'global (no proj.)': m.PHI_DIM }

def train_ind(noise, seeds = ((1234,),)):
    schedule = SCHED[noise]
    infos = []
    trs = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = VQRInd(warm = True)
        hist_all = {
            'epoch': [],
            'fidelity': [],
            'loss': [] }
        off = 0
        for ep, lr, pr in schedule:
            hist = m.train_vscr(model, n_epochs = ep, batch = 48, p_range = pr, noise = noise, lr = lr, lam = 0, rng_seed = seed, use_real_grad = True, verbose = False)
            off += ep = None
            (cf, w) = m.syndrome_conditional_fidelity(model, noise)
            min_cf = float(np.where(w > 0.001, cf, 10).min())
            infos.append({
                'seed': seed,
                'min_cf': min_cf,
                'val_fid': hist_all['fidelity'][-1] })
            print(f'''    [ind/{noise}] seed {seed}: val_fid={infos[-1]['val_fid']:.4f} min cf={min_cf:.4f}''', flush = True)
            trs.append((min_cf, infos[-1]['val_fid'], model, hist_all))
            best = max(trs, key = (lambda t: (t[0], t[1])))
            return (best[2], best[3], infos)


def eval_phi_table(phi_table, noise, p_values, n_test, seed = (None, m.SEED + 7)):
    '''F̄ ± sem of the syndrome-projected recovery R_s = ansatz(phi_s).'''
    if not n_test:
        pass
    n_test = N_TEST
    rng = np.random.RandomState(seed)
    states = range(n_test)()
    torch.no_grad()
    R_all = m.recovery_unitary_batch(torch.as_tensor(phi_table, dtype = torch.float64))
    None(None, None)


def eval_global(phi, noise, p_values, n_test, seed = (None, m.SEED + 7)):
    '''F̄ ± sem of a single global unitary (no projection).'''
    if not n_test:
        pass
    n_test = N_TEST
    rng = np.random.RandomState(seed)
    states = range(n_test)()
    torch.no_grad()
    U = m.recovery_unitary(torch.as_tensor(phi, dtype = torch.float64))
    None(None, None)


def train_global(noise, seed = (1234,)):
    epochs = 60 if SMOKE else 600
    lr = 0.01
    pr = SCHED[noise][0][2]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GlobalRec()
    opt = torch.optim.Adam(model.parameters(), lr = lr)
    rng = np.random.RandomState(seed)
# WARNING: Decompyle incomplete


def train_pool(make_model, pool, noise, seed, epochs, lr, p_range = (1234, None, 0.003, (0.02, 0.15))):
    epochs = 60 if epochs or SMOKE else 400
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr = lr)
    rng = np.random.RandomState(seed + 1)
    K = len(pool)
# WARNING: Decompyle incomplete


def _kraus_dep(p):
    '''1024 Kraus ops of the per-qubit-sequential depolarizing channel.'''
    per_q = []
# WARNING: Decompyle incomplete


def _kraus_ad(gamma):
    per_q = range(m.N)()
    out = []
# WARNING: Decompyle incomplete


def _kraus_coherent(eps):
    U = np.eye(m.DIM, dtype = complex)
    Rx2 = np.array([
        [
            math.cos(eps / 2),
            (-0+-1j) * math.sin(eps / 2)],
        [
            (-0+-1j) * math.sin(eps / 2),
            math.cos(eps / 2)]])
    for q in range(m.N):
        U = U @ m.embed_gate(torch.tensor(Rx2, dtype = m.DTYPE), [
            q]).numpy()
        return [
            U]

_KRAUS_CACHE = { }

def channel_kraus(noise, p):
    key = (noise, round(float(p), 10))
    if key in _KRAUS_CACHE:
        return _KRAUS_CACHE[key]
    if None == 'depolarizing':
        K = _kraus_dep(p)
    elif noise == 'amplitude_damping':
        K = _kraus_ad(p)
    elif noise == 'coherent':
        K = _kraus_coherent(p)
    else:
        raise ValueError(noise)
    _KRAUS_CACHE[key] = K
    return K


def _W_basis():
    '''(16, 32, 2) orthonormal bases of the syndrome subspaces.'''
    Ws = np.zeros((16, m.DIM, 2), dtype = complex)
    for s in range(16):
        P = m.P_SYNDS_STACK[s].numpy()
        (ev, evec) = np.linalg.eigh(0.5 * (P + P.conj().T))
        cols = evec[(:, ev > 0.5)]
        if not cols.shape[1] == 2:
            raise (s, ev)()
        Ws[s] = cols
        return Ws

W_BASIS = _W_basis()
V_ISO = m.E_CODE.numpy()

def _branch_A(noise, p, s):
    '''2x2 reduced branch operators {A_k = W_s^† K_k V}.'''
    W = W_BASIS[s].conj().T
    if noise == 'mixed':
        key = ('mixedKV', round(float(p), 10))
        if key not in _KRAUS_CACHE:
            _KRAUS_CACHE[key] = (_kraus_dep(p)(), _kraus_ad(0.5 * p))
        (Kd_V, Kad) = _KRAUS_CACHE[key]
        return Kd_V()
    return channel_kraus(noise, p)()


def _branch_M(noise, p, s):
    '''Exact Haar-averaged objective matrix M_s and branch weight p_s.'''
    S1 = np.zeros((2, 2), dtype = complex)
    VV = np.zeros((4, 4), dtype = complex)
    for A in _branch_A(noise, p, s):
        S1 += A.conj() @ A.T
        w = A.conj().reshape(4) / math.sqrt(2)
        VV += np.outer(w, w.conj())
        M = (np.kron(S1, np.eye(2)) + 2 * VV) / 6
        p_s = float(np.real(np.trace(S1))) / 2
        return (M, p_s)


def _sdp_valgrad_autograd(x, M_t):
    '''Reference implementation of `_sdp_valgrad` via torch autograd.

    Kept ONLY as an independent cross-check: `_selftest_sdp` (g) asserts the
    closed-form gradients below agree with these to 1e-12.  Autograd is the
    definition-correct way to differentiate the three functions but costs
    ~1.9 ms per call, which dominates a solve that does hundreds of them, so
    the production path uses the closed form (~20 us).
    '''
    xr = torch.as_tensor(np.asarray(x[:16], dtype = np.float64).copy(), dtype = torch.float64).requires_grad_(True)
    xi = torch.as_tensor(np.asarray(x[16:], dtype = np.float64).copy(), dtype = torch.float64).requires_grad_(True)
    L = torch.complex(xr, xi).reshape(4, 4)
    C = L @ L.conj().T
    T = 0.5 * torch.eye(2, dtype = torch.complex128) - torch.stack([
        torch.stack([
            C[(0, 0)] + C[(1, 1)],
            C[(0, 2)] + C[(1, 3)]]),
        torch.stack([
            C[(2, 0)] + C[(3, 1)],
            C[(2, 2)] + C[(3, 3)]])])
    f_obj = 2 * torch.real(torch.trace(C @ M_t))
    f_tr = torch.real(T[(0, 0)] + T[(1, 1)])
    f_det = torch.real(T[(0, 0)] * T[(1, 1)] - T[(0, 1)] * T[(1, 0)])
    out = []
    for f in (f_obj, f_tr, f_det):
        gx = torch.autograd.grad(f, (xr, xi), retain_graph = True, allow_unused = True)
        np.concatenate((lambda .0: pass# WARNING: Decompyle incomplete
)(gx()))
        return (float(f_obj), out[0], float(f_tr), out[1], float(f_det), out[2])

_I2 = np.eye(2)

def _sdp_valgrad(x, M):
    """EXACT values and gradients of the three functions `_sdp_branch` optimises.

        L     = (x[:16] + i x[16:]).reshape(4,4)      Choi factor, C = L L^dag
        obj   = 2 Re Tr[L L^dag M]                                 (maximised)
        T     = I/2 - Tr_out(C)   (2x2 Hermitian)
        c_tr  = Re Tr[T]                                           (>= 0)
        c_det = Re det[T]                                          (>= 0)

    Closed forms.  Writing L = X + iY and P = A L, the identity
    `(L^dag A)_{ba} = conj((A^dag L)_{ab})` gives, for any fixed matrix A,

        d Re Tr[L^dag A dL] = Re sum_ab conj((A^dag L)_ab) dL_ab ,

    and with dL = dX + i dY the real part of conj(P) dL is
    Re(P) dX + Im(P) dY.  Applying this three times:

    * obj = 2 Re Tr[L^dag M dL]|_{dL->L}: since M is Hermitian, A^dag = M and
          grad_X obj = 4 Re(ML),   grad_Y obj = 4 Im(ML).
    * c_tr = 1 - Tr[C] = 1 - ||L||_F^2 exactly (Tr[I/2] = 1 and the trace of a
      partial trace is the trace), hence
          grad_X c_tr = -2X,       grad_Y c_tr = -2Y.
    * c_det: d det[T] = sum_ab adj(T)^T_ab dT_ab and dT = -dS with
      S = Tr_out(C), so with Ghat = adj(T)^T and K = Ghat (x) I_2 (the 4x4 lift
      satisfying sum_ab Ghat_ab dS_ab = Tr[K^T dC]) one gets
          d c_det = -2 Re sum_ab conj((K^T L)_ab) dL_ab ,
      i.e. grad_X c_det = -2 Re(K^T L), grad_Y c_det = -2 Im(K^T L).
      (K is Hermitian because T is, so K^* = K^T and the two dL and dL^dag
      terms combine into twice the real part.)

    x is the length-32 real parameter vector.  Returns
    (obj, g_obj, c_tr, g_c_tr, c_det, g_c_det), each g_* a length-32 float64
    array in the SAME layout as x.

    Why this exists at all: scipy was differentiating these three scalars by
    finite differences, i.e. 3 x 33 = 99 extra evaluations per Jacobian, inside
    ~22 solves per syndrome x 16 syndromes x every (noise, p) point -- about
    10 min per `sdp_ceiling` call and ~70 min for the E3 sweep in `main`.  The
    exact gradient is both ~50x faster and more reliable: SLSQP's default
    finite-difference step is at the level where the trace/det constraints are
    already saturated, so the numerical gradient routinely lost feasibility and
    forced the solver to `maxiter`.
    """
    x = np.asarray(x, dtype = np.float64)
    X = x[:16].reshape(4, 4)
    Y = x[16:].reshape(4, 4)
    L = X + (0+1j) * Y
    C = L @ L.conj().T
    T = 0.5 * _I2 - np.array([
        [
            C[(0, 0)] + C[(1, 1)],
            C[(0, 2)] + C[(1, 3)]],
        [
            C[(2, 0)] + C[(3, 1)],
            C[(2, 2)] + C[(3, 3)]]])
    f_obj = 2 * float(np.real(np.trace(C @ M)))
    f_tr = float(np.real(T[(0, 0)] + T[(1, 1)]))
    f_det = float(np.real(T[(0, 0)] * T[(1, 1)] - T[(0, 1)] * T[(1, 0)]))
    ML = M @ L
    g_obj = np.concatenate([
        4 * ML.real.reshape(16),
        4 * ML.imag.reshape(16)])
    g_tr = np.concatenate([
        -2 * X.reshape(16),
        -2 * Y.reshape(16)])
    Ghat = np.array([
        [
            T[(1, 1)],
            -T[(1, 0)]],
        [
            -T[(0, 1)],
            T[(0, 0)]]])
    P = np.kron(Ghat.T, _I2) @ L
    g_det = np.concatenate([
        -2 * P.real.reshape(16),
        -2 * P.imag.reshape(16)])
    return (f_obj, g_obj, f_tr, g_tr, f_det, g_det)

_Y_BASIS = (np.array([
    [
        1,
        0],
    [
        0,
        1]], dtype = complex), np.array([
    [
        0,
        1],
    [
        1,
        0]], dtype = complex), np.array([
    [
        0,
        (0+1j)],
    [
        (-0+-1j),
        0]], dtype = complex), np.array([
    [
        1,
        0],
    [
        0,
        -1]], dtype = complex))
_Y_LIFT2 = _Y_BASIS
_Y_LIFT4 = (lambda .0: pass# WARNING: Decompyle incomplete
)(_Y_BASIS())

def _lmin_grad(A, lifts):
    '''lambda_min(A) and d lambda_min / d y_k = Re Tr[v v^dag  dA/dy_k].

    lambda_min is a concave function of a Hermitian matrix, and its subgradient
    at a simple eigenvalue is the rank-one projector onto the eigenvector.  Both
    constraints of the dual below are of this form, so each feasible set
    {y : lambda_min(A(y)) >= 0} is CONVEX and SLSQP sees exact gradients.
    '''
    (w, V) = np.linalg.eigh(A)
    G = np.outer(V[(:, 0)], V[(:, 0)].conj())
    return (None, (lambda .0 = None: pass# WARNING: Decompyle incomplete
)(lifts()))


def _sdp_dual_bound(M, ntry = (8,)):
    '''RIGOROUS upper bound on `_sdp_branch`\'s optimum, via LP/SDP duality.

    The branch problem is, in terms of the Choi matrix C = L L^dag directly,

        (P)  max 2 <M, C>   s.t.  C >= 0,   Tr_out(C) <= I/2 .

    Its Lagrangian dual, with multiplier Y >= 0 on the trace constraint and
    using Tr[Y Tr_out(C)] = <kron(Y, I_2), C>, is

        (D)  min (1/2) Tr[Y]  s.t.  Y >= 0,   kron(Y, I_2) - 2M >= 0 .

    Weak duality gives <P> <= <D> for every DUAL-FEASIBLE Y, no matter how (P)
    was solved.  So this is a certificate, not a second opinion: if the
    multi-start SLSQP primal had stalled in a local trap, (P) < (D) by a visible
    margin.  Slater\'s condition holds for (P) (C = eps*I is strictly feasible),
    so strong duality applies and the true gap is zero -- measured worst gap over
    25 branches x 5 channels is < 1e-9.

    This replaces the trust-constr "polish" that used to run on every branch.
    trust-constr needed Hessians, and `c_det = Re det[I/2 - Tr_out(L L^dag)]` is
    QUARTIC in L (det of a matrix that is itself quadratic in L), not quadratic
    as an earlier version of this file assumed -- so the "constant" Hessian it
    was handed was simply wrong, off by exactly 2.0 at a non-zero point.  A
    wrong Hessian is worse than none: it silently biases the trust region.  It
    also never improved on SLSQP (both returned F = 0.014085885 on
    depolarizing p=0.05, branch s=7) while costing ~7x more per solve, and the
    finite-difference Hessian path scipy falls back on is what segfaulted
    Section B.  Dual certification is both cheaper and mathematically stronger.
    '''
    M = np.asarray(M, dtype = np.complex128)
    
    def _Y(y):
        return (lambda .0 = None: pass# WARNING: Decompyle incomplete
)(range(4)())

    
    def c1(y = None):
        pass
    # WARNING: Decompyle incomplete

    
    def c2(y = None):
        pass
    # WARNING: Decompyle incomplete

    cons = [
        None,
        {
            'type': None,
            'fun': None,
            'jac': (lambda y = None: pass# WARNING: Decompyle incomplete
) }]
    rs = np.random.RandomState(0)
    best = np.inf
    for t in range(ntry):
        y0 = np.array([
            2 * np.linalg.norm(M, 2) + 1,
            0,
            0,
            0]) if t == 0 else np.array([
            rs.rand() * 2 + 0.5,
            rs.randn() * 0.3,
            rs.randn() * 0.3,
            rs.randn() * 0.3])
        r = minimize((lambda y: y[0]), y0, jac = (lambda y: np.array([
1,
0,
0,
0])), constraints = cons, method = 'SLSQP', options = {
            'maxiter': 500,
            'ftol': 1e-14 })
    except Exception:
        continue
    if c1(r.x) > -1e-10 and c2(r.x) > -1e-11 and float(r.x[0]) < best:
        best = float(r.x[0])
    continue
    return best


def _sdp_branch(M, G_dec, G_ref = (None,)):
    '''Max 2Tr[CM] s.t. C=LL^† ⪰ 0, I/2 − Tr_out C ⪰ 0.  Returns F_unnorm.

    All three functions are supplied to scipy together with their EXACT
    closed-form gradients from `_sdp_valgrad` (see its docstring for the ~40x
    speedup and the feasibility argument); `fun` and `jac` share one memoised
    evaluation because scipy calls them back to back at the same point.

    Returns (F_unnorm, x_star) where x_star is the length-32 real parameter
    vector attaining F_unnorm, so callers can re-verify feasibility.
    '''
    M_np = np.asarray(M, dtype = np.complex128)
    _memo = { }
    
    def vg(x = None):
        '''(obj, g_obj, c_tr, g_c_tr, c_det, g_c_det), memoised over a few x.'''
        pass
    # WARNING: Decompyle incomplete

    
    def obj(x = None):
        pass
    # WARNING: Decompyle incomplete

    
    def neg_obj(x = None, *a):
        pass
    # WARNING: Decompyle incomplete

    
    def neg_obj_jac(x = None, *a):
        pass
    # WARNING: Decompyle incomplete

    
    def c_tr(x = None):
        pass
    # WARNING: Decompyle incomplete

    
    def c_tr_jac(x = None, *a):
        pass
    # WARNING: Decompyle incomplete

    
    def c_det(x = None):
        pass
    # WARNING: Decompyle incomplete

    
    def c_det_jac(x = None, *a):
        pass
    # WARNING: Decompyle incomplete

    
    def feasible(x = None):
        pass
    # WARNING: Decompyle incomplete

    
    def L_of_unitary(G):
        v = np.zeros(4, dtype = complex)
        for a in range(2):
            for b in range(2):
                v[a * 2 + b] = G[(b, a)] / math.sqrt(2)
                L = np.zeros((4, 4), dtype = complex)
                L[(:, 0)] = v
                return L

    starts = [
        L_of_unitary(np.eye(2, dtype = complex)),
        L_of_unitary(G_dec)]
# WARNING: Decompyle incomplete


def _branch_qform(noise, p, s):
    '''Q_s (4x4), the G-independent constant Tr(sum_k A_k^† A_k), and p_s.'''
    Q = np.zeros((4, 4), dtype = complex)
    const = 0
    for A in _branch_A(noise, p, s):
        A = np.asarray(A, dtype = complex)
        v = A.T.reshape(4)
        Q += np.outer(v, v.conj())
        const += float(np.real(np.trace(A.conj().T @ A)))
        Q = 0.5 * (Q + Q.conj().T)
        return (Q, const, float(np.real(np.trace(Q))) / 2)

_PAULI4 = [
    np.eye(2, dtype = complex),
    np.array([
        [
            0,
            1],
        [
            1,
            0]], dtype = complex),
    np.array([
        [
            0,
            (-0+-1j)],
        [
            (0+1j),
            0]], dtype = complex),
    np.array([
        [
            1,
            0],
        [
            0,
            -1]], dtype = complex)]

def _G_of_h(h):
    return None(sum * (lambda .0: pass# WARNING: Decompyle incomplete
)(zip(h, _PAULI4)()))


def opt_unitary_branch(noise, p, s, n_start = (16,)):
    '''Exact max over 2x2 unitaries of J(G) = sum_k |Tr(G A_k)|^2 for branch s.

    Returns (J_max, G_opt).  Single-Kraus branches use the polar-factor closed
    form, cross-checked against the nuclear norm, so they are provably global;
    multi-Kraus branches use multi-start L-BFGS-B over the Lie algebra plus a
    derivative-free Nelder-Mead polish, seeded with the identity, the decoder
    unitary, the 6 Pauli/phase matrices and random points.
    '''
    (Q, const, p_s) = _branch_qform(noise, p, s)
    
    def J(G = None):
        pass
    # WARNING: Decompyle incomplete

    best_G = None
    best_v = -(np.inf)
    As = _branch_A(noise, p, s)()
    if len(As) == 1:
        (U, S, Vh) = np.linalg.svd(As[0])
        G0 = Vh.conj().T @ U.conj().T
        if not abs(J(G0) - float(S.sum()) ** 2) < 1e-09:
            raise (J(G0), S)()
        best_G = G0
        best_v = J(G0)
    for P in _PAULI4 + [
        np.array([
            [
                0,
                (0+1j)],
            [
                (0+1j),
                0]], dtype = complex),
        np.array([
            [
                (0+1j),
                0],
            [
                0,
                1]], dtype = complex)]:
        if J(P) > best_v:
            best_G = P
            best_v = J(P)
        G_dec = V_ISO.conj().T @ m.C_SYNDS[s].numpy() @ W_BASIS[s]
        if J(G_dec) > best_v:
            best_G = G_dec
            best_v = J(G_dec)
    rs = np.random.RandomState(0)
    for h0 in (lambda .0 = None: pass# WARNING: Decompyle incomplete
) + range(n_start)():
        r = None((lambda h = None: pass# WARNING: Decompyle incomplete
), h0, method = 'L-BFGS-B', options = {
            'maxiter': 1000,
            'ftol': 1e-16,
            'gtol': 1e-14 })
        if -float(r.fun) > best_v:
            best_G = _G_of_h(r.x)
            best_v = -float(r.fun)
        for _ in range(4):
            r = None((lambda h = None: pass# WARNING: Decompyle incomplete
), rs.normal(size = 4) * 1.5, method = 'Nelder-Mead', options = {
                'maxiter': 6000,
                'fatol': 1e-16,
                'xatol': 1e-14 })
            if -float(r.fun) > best_v:
                best_G = _G_of_h(r.x)
                best_v = -float(r.fun)
            return (best_v, best_G)


def opt_unitary_ceiling(noise, p):
    """F̄ of the best PER-BRANCH UNITARY recovery (the VSCR family optimum).

    Returns {'F_unit', 'F_dec', 'cf_unit', 'cf_dec', 'p_s', 'G_opt'}.  F_dec is
    the analytic decoder fidelity (validated against Monte-Carlo in
    `_selftest_sdp`), so F_unit - F_dec is the exact, noiseless non-Pauli
    headroom available to a trained warm start.
    """
    (cf_u, cf_d, ps, Gs) = ([], [], [], [])
    for s in range(16):
        (Q, const, p_s) = _branch_qform(noise, p, s)
        ps.append(p_s)
        if p_s < 1e-13:
            cf_u.append(1)
            cf_d.append(1)
            Gs.append(np.eye(2, dtype = complex))
            continue
        (Ju, Gu) = opt_unitary_branch(noise, p, s)
        gd = (V_ISO.conj().T @ m.C_SYNDS[s].numpy() @ W_BASIS[s]).reshape(4)
        Jd = float(np.real(gd @ Q @ gd.conj()))
        cf_u.append((Ju + const) / 6 / p_s)
        cf_d.append((Jd + const) / 6 / p_s)
        Gs.append(Gu)
        ps = np.array(ps)
        return {
            'F_unit': float(np.dot(ps, cf_d)),
            'F_dec': cf_u,
            'cf_unit': cf_d,
            'cf_dec': ps.tolist(),
            'p_s': (lambda .0: [ g.tolist() for g in .0 ]),
            'G_opt': Gs() }


def exact_F(phi_table, noise, p):
    """Noiseless Haar-averaged F̄ of an arbitrary per-syndrome unitary table.

    phi_table is (16, PHI_DIM) of ansatz angles; R_s = ansatz(phi_s).  Uses the
    closed second-moment identity
        F_s = [ sum_k |Tr(G_s A_k)|^2 + Tr(sum_k A_k^† A_k) ] / 6 / p_s ,
        G_s = V^† R_s W_s ,
    so the result has ZERO Monte-Carlo error -- gains of order 1e-5 (the exact
    coherent-over-rotation headroom) are resolvable, which the n_test=400
    protocol in `eval_phi_table` cannot do.  Cross-checked against that MC
    evaluator and against `sdp_ceiling`'s branch data in `_selftest_sdp`.
    """
    phi = np.asarray(phi_table, dtype = float)
    R_all = m.recovery_unitary_batch(torch.tensor(phi, dtype = torch.float64))
    cfs = []
    Ftot = 0
    for s in range(16):
        (Q, const, p_s) = _branch_qform(noise, p, s)
        if p_s < 1e-13:
            cfs.append(1)
            continue
        R_s = R_all[s].detach().numpy()
        G = V_ISO.conj().T @ R_s @ W_BASIS[s]
        g = np.asarray(G, dtype = complex).reshape(4)
        J = float(np.real(g @ Q @ g.conj()))
        cf = (J + const) / 6 / p_s
        cfs.append(cf)
        Ftot += p_s * cf
        return (float(Ftot), cfs)


def pavg_bundle(noise, p_range, n_p = ((0.02, 0.15), 6)):
    '''Precompute the separable p-averaged objective for one channel.

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
    `ssvr_qec.vscr_fidelity_quad` on `quad_cache(p_range, noise)` to ~1e-14.
    '''
    (p_nodes, p_w) = m.p_quadrature(p_range, n_p)
    p_nodes = np.asarray(p_nodes, dtype = float)
    p_w = np.asarray(p_w, dtype = float) / 6
    coff = []
    Q = []
    for s in range(16):
        consts = []
        rows = []
        for p in p_nodes:
            (Qs, cs, _ps) = _branch_qform(noise, float(p), s)
            rows.append(np.asarray(Qs, dtype = complex))
            consts.append(float(cs))
            Q.append(np.stack(rows))
            coff.append(float(np.dot(p_w, np.asarray(consts))))
            return {
                'noise': (float(p_range[0]), float(p_range[1])),
                'n_p': p_nodes,
                'p_range': p_w,
                'p_nodes': Q,
                'w6': coff,
                'Q': (lambda .0: [ torch.tensor(q, dtype = torch.complex128) for q in .0 ]),
                'coff': Q(),
                'Q_t': (lambda .0: [ torch.tensor(c, dtype = torch.float64) for c in .0 ]),
                'coff_t': coff() }


def pavg_F(phi_table, B):
    '''Exact p-averaged F̄ and its 16 separable branch contributions.

    Returns (F̄_avg, obj) with obj[s] = the whole contribution of syndrome s, so
    sum(obj) == F̄_avg exactly.  `B` is a `pavg_bundle`.
    '''
    phi = np.asarray(phi_table, dtype = float).reshape(16, m.PHI_DIM)
    R_all = m.recovery_unitary_batch(torch.tensor(phi, dtype = torch.float64)).detach().numpy()
    obj = np.empty(16, dtype = float)
    for s in range(16):
        G = V_ISO.conj().T @ R_all[s] @ W_BASIS[s]
        g = np.asarray(G, dtype = complex).reshape(4)
        tot = B['coff'][s]
        for ip in range(B['n_p']):
            Qs = B['Q'][s][ip]
            tot += float(B['w6'][ip]) * float(np.real(g @ Qs @ g.conj()))
            obj[s] = tot
            return (float(obj.sum()), obj)


def refine_per_syndrome(phi_init, noise, p_range, n_p, n_start, steps, lr, seed, bundle, verbose = ((0.02, 0.15), 6, 3, 1500, 0.01, 0, None, False)):
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
    full table then captures 99.9999% of the certified headroom
    (F̄ 0.994310805 → 0.994994328 against F̄_unit = 0.994994328).

    Start 0 is always `phi_init[s]` and a candidate is only accepted if it
    strictly improves that block's objective, so the result can NEVER be worse
    than the input table -- the refinement is monotone by construction.

    Returns (phi, info); phi is (16, PHI_DIM).
    """
    phi_init = np.asarray(phi_init, dtype = float).reshape(16, m.PHI_DIM)
# WARNING: Decompyle incomplete


def sdp_ceiling(noise, p, ntry_dual = (4,)):
    '''F̄^CPTP ceiling and per-branch data for a channel at strength p.

    BUG FIX: this used to take `fu, _ = _sdp_branch(...)` unconditionally and
    divide by p_s.  `_sdp_branch` returned -1.0 whenever SLSQP reported
    `success=False` (which it does at many perfectly converged KKT points, and
    always once it hits `maxiter`), so `F_s_max` became -1/p_s -- e.g. the
    amplitude-damping p=0.30 ceiling was printed as F_CPTP = -0.1113, a
    negative fidelity, and the derived "non-Pauli headroom" was meaningless.
    The solver now never returns below its feasible starts, and this function
    additionally seeds it with the exact unitary-family optimum and asserts
    F̄^CPTP >= max(F̄^unitary, F̄^decoder).

    Every branch is now also CERTIFIED globally optimal by `_sdp_dual_bound`:
    `gap_max` in the returned dict is the largest |dual − primal| over the 16
    branches and must be ~0.  A large POSITIVE gap means the multi-start SLSQP
    stalled in a local trap (the primal is not provably optimal); a large
    NEGATIVE one would violate weak duality and means a dual-infeasible Y was
    accepted.  Both are failures, so the reported quantity is the absolute gap;
    the per-branch weak-duality direction is asserted separately below.
    '''
    (Fs, ps, Funits, Fdecs, gaps) = ([], [], [], [], [])
    for s in range(16):
        (M, p_s) = _branch_M(noise, p, s)
        G_dec = V_ISO.conj().T @ m.C_SYNDS[s].numpy() @ W_BASIS[s]
        if p_s <= 1e-12:
            Fs.append(1)
            Funits.append(1)
            Fdecs.append(1)
            ps.append(p_s)
            continue
        (Ju, G_u) = opt_unitary_branch(noise, p, s)
        (_, const, _) = _branch_qform(noise, p, s)
        (fu, _) = _sdp_branch(M, G_dec, G_ref = G_u)
        dual = _sdp_dual_bound(M, ntry = ntry_dual)
        if not fu <= dual + 1e-07 * max(1, abs(dual)):
            raise (f'''weak duality violated: primal {fu} > dual {dual}''', noise, p, s)()
        gaps.append(dual - fu)
        Fu = fu / p_s
        Funit = (Ju + const) / 6 / p_s
        gd = G_dec.reshape(4)
        (Q, _, _) = _branch_qform(noise, p, s)
        Fdec = (float(np.real(gd @ Q @ gd.conj())) + const) / 6 / p_s
        if not Fu >= Funit - 1e-06:
            raise (noise, p, s, Fu, Funit)()
        if not Fu >= Fdec - 1e-06:
            raise (noise, p, s, Fu, Fdec)()
        Fs.append(float(Fu))
        Funits.append(float(Funit))
        Fdecs.append(float(Fdec))
        ps.append(p_s)
        F_bar = float(np.dot(ps, Fs))
        return {
            'F_cptp': ps(),
            'F_s_max': float(np.dot(ps, Funits)),
            'p_s': float(np.dot(ps, Fdecs)),
            'F_unit': None,
            'F_dec_analytic': float,
            'gap_max': max((lambda .0: pass# WARNING: Decompyle incomplete
)(gaps(), default = 0)) }


def _selftest_sdp():
    r = sdp_ceiling('depolarizing', 0)
    if not abs(r['F_cptp'] - 1) < 1e-06:
        raise r['F_cptp']()
    r = sdp_ceiling('depolarizing', 0.05)
    rng = np.random.RandomState(5)
    Fdec = []
    for _ in range(300):
        psi = m.random_logical_state(1, rng)
        pe = m.encode(psi)
        rho = m.noisy_state(pe, 0.05, 'depolarizing')
        Fdec.append(float(m.perfect_code_decoder_fidelity(pe, rho)))
        Fdec = float(np.mean(Fdec))
        if not r['F_cptp'] >= Fdec - 0.0002:
            raise (r['F_cptp'], Fdec)()
        if not r['F_cptp'] <= Fdec + 0.002:
            raise (r['F_cptp'], Fdec)()
        for noise, p in (('depolarizing', 0.05), ('amplitude_damping', 0.1), ('coherent', 0.1), ('mixed', 0.08)):
            for s in (0, 1, 5, 9, 15):
                (_, p_M) = _branch_M(noise, p, s)
                (Q, const, p_Q) = _branch_qform(noise, p, s)
                if not abs(p_Q - p_M) < 1e-12:
                    raise (noise, p, s, p_Q, p_M)()
                fr = (lambda .0: pass# WARNING: Decompyle incomplete
)(_branch_A(noise, p, s)())
                if not abs(const - fr) < 1e-12:
                    raise (noise, p, s, const, fr)()
                (Fe, _) = exact_F(vp.PHI_DEC, noise, p)
                (Fm, Fsem) = eval_phi_table(vp.PHI_DEC, noise, [
                    p], n_test = 400)
                if not abs(Fe - Fm[0]) < 6 * Fsem[0] + 1e-09:
                    raise (noise, p, Fe, Fm, Fsem)()
            raise (noise, p, Fe)()
            print(f'''[sdp selftest c] {noise:20s} p={p:.2f}: exact_F(decoder)={Fe:.8f}  MC(400)={Fm[0]:.8f}+-{Fsem[0]:.1e} ({abs(Fe - Fm[0]) / max(Fsem[0], 1e-18):.1f} sigma)  p_s/Kraus-norm OK''', flush = True)
            for noise, p in (('amplitude_damping', 0.1), ('coherent', 0.1)):
                for s in range(16):
                    As = _branch_A(noise, p, s)()
                    if len(As) != 1 or p <= 0:
                        continue
                    (Ju, _) = opt_unitary_branch(noise, p, s)
                    ref = float(np.linalg.svd(As[0])[1].sum()) ** 2
                    if not abs(Ju - ref) < 1e-09:
                        raise (noise, p, s, Ju, ref)()
                    print('[sdp selftest d] single-Kraus branches hit ||A||_*^2 exactly  OK', flush = True)
                    for noise, p in (('amplitude_damping', 0.3), ('amplitude_damping', 0.06), ('depolarizing', 0.3), ('coherent', 0.1), ('mixed', 0.15)):
                        u = opt_unitary_ceiling(noise, p)
                        c = sdp_ceiling(noise, p)
                        raise (noise, p, c['F_cptp'])()
                        if not c['F_cptp'] >= u['F_unit'] - 1e-09:
                            raise (noise, p, c, u)()
                        if not u['F_unit'] >= u['F_dec'] - 1e-12:
                            raise (noise, p, u)()
                        if not c['gap_max'] < 1e-07:
                            raise (noise, p, c['gap_max'])()
                        print(f'''[sdp selftest e] {noise:20s} p={p:.2f}: F̄_dec={u['F_dec']:.8f} <= F̄_unit={u['F_unit']:.8f} <= F̄_CPTP={c['F_cptp']:.8f}  headroom(unit)={u['F_unit'] - u['F_dec']:+.2e}  worst |primal-dual gap|={c['gap_max']:.1e}  OK''', flush = True)
                        if not abs(u['F_unit'] - 1) < 1e-09:
                            raise u['F_unit']()
                        print(f'''[sdp selftest f] coherent(0.10): F̄_unit = {u['F_unit']:.12f} (=1 exactly, unitary noise is perfectly correctable)  OK''', flush = True)
                        print(f'''[sdp selftest] identity ceiling=1 OK; dep(0.05): ceiling={r['F_cptp']:.6f} vs decoder={Fdec:.6f} OK''', flush = True)
                        print('[sdp selftest g] closed-form Jacobians of the branch SDP', flush = True)
                        (lambda x, Mm: L = (np.asarray(x[:16]) + (0+1j) * np.asarray(x[16:])).reshape(4, 4)2 * float(np.real(np.trace(L @ L.conj().T @ Mm)))) = opt_unitary_ceiling('coherent', 0.1)
                        
                        def _np_T(x):
                            L = (np.asarray(x[:16]) + (0+1j) * np.asarray(x[16:])).reshape(4, 4)
                            C = L @ L.conj().T
                            S = np.array([
                                [
                                    C[(0, 0)] + C[(1, 1)],
                                    C[(0, 2)] + C[(1, 3)]],
                                [
                                    C[(2, 0)] + C[(3, 1)],
                                    C[(2, 2)] + C[(3, 3)]]])
                            return 0.5 * np.eye(2) - S

                        _S0 = 12
                        (Mh, _) = _branch_M('depolarizing', 0.15, _S0)
                        Mnp = np.asarray(Mh, dtype = np.complex128)
                        _Gd = V_ISO.conj().T @ m.C_SYNDS[_S0].numpy() @ W_BASIS[_S0]
                        (_, _Gref) = opt_unitary_branch('depolarizing', 0.15, _S0)
                        Mt = torch.as_tensor(Mnp, dtype = torch.complex128)
                        rng = np.random.RandomState(0)
                        _w_ag = 0
                        for _trial in range(3):
                            xt = np.zeros(32)
                            xt[[
                                0,
                                5,
                                10,
                                15]] = 1 / math.sqrt(2)
                            xt = xt * 0.7 + rng.randn(32) * 0.3
                            (o, go, t, gt, d, gd) = _sdp_valgrad(xt, Mnp)
                            (oa, goa, ta, gta, da, gda) = _sdp_valgrad_autograd(xt, Mt)
                            Tm = _np_T(xt)
                            if not abs(o - _np_obj(xt, Mnp)) < 1e-12:
                                raise (o, _np_obj(xt, Mnp))()
                            if not abs(o - oa) < 1e-12:
                                raise (o, oa)()
                            if not abs(t - float(np.real(np.trace(Tm)))) < 1e-12:
                                raise AssertionError
                            if not abs(t - ta) < 1e-12:
                                raise AssertionError
                            if not abs(d - float(np.real(np.linalg.det(Tm)))) < 1e-12:
                                raise AssertionError
                            if not abs(d - da) < 1e-12:
                                raise AssertionError
                            for _a, _c in ((go, goa), (gt, gta), (gd, gda)):
                                _w_ag = max(_w_ag, float(np.max(np.abs(_a - _c))))
                                h = 1e-06
                                for j in range(32):
                                    e = np.zeros(32)
                                    e[j] = h
                                    T2 = _np_T(xt - e)
                                    T1 = _np_T(xt + e)
                                    for got, want, nm in ((go[j], (_np_obj(xt + e, Mnp) - _np_obj(xt - e, Mnp)) / (2 * h), 'obj'), (gt[j], float(np.real(np.trace(T1) - np.trace(T2))) / (2 * h), 'tr'), (gd[j], float(np.real(np.linalg.det(T1) - np.linalg.det(T2))) / (2 * h), 'det')):
                                        sc = max(1, abs(want))
                                        if not abs(got - want) < 1e-05 * sc:
                                            raise f'''grad[{nm}][{j}]: closed form {got} vs FD {want}'''()
                                        if not _w_ag < 1e-12:
                                            raise f'''closed-form gradient disagrees with autograd: {_w_ag}'''()
                                        print(f'''        all 3x32 gradient components match torch autograd to {_w_ag:.1e} and central differences to <1e-5 rel''', flush = True)
                                        _CERT_CASES = [
                                            ('depolarizing', 0.15, 0),
                                            ('depolarizing', 0.15, 7),
                                            ('depolarizing', 0.15, 12),
                                            ('depolarizing', 0.15, 15),
                                            ('amplitude_damping', 0.3, 1),
                                            ('coherent', 0.1, 14)]
                                        _wgap = 0
                                        for _nz, _p, _sn in _CERT_CASES:
                                            (Mn, pn) = _branch_M(_nz, _p, _sn)
                                            Mn = np.asarray(Mn, dtype = np.complex128)
                                            if pn <= 1e-12:
                                                continue
                                            (_, _Gu) = opt_unitary_branch(_nz, _p, _sn)
                                            (Fp, _) = _sdp_branch(Mn, V_ISO.conj().T @ m.C_SYNDS[_sn].numpy() @ W_BASIS[_sn], G_ref = _Gu)
                                            D = _sdp_dual_bound(Mn, ntry = 8)
                                            if not Fp <= D + 1e-09:
                                                raise f'''weak duality violated on {_nz} p={_p} branch {_sn}: primal {Fp} > dual {D}'''()
                                            if not D - Fp < 1e-08:
                                                raise f'''primal-dual gap {D - Fp:.2e} on {_nz} p={_p} branch {_sn}: the multi-start solve is NOT certified globally optimal (primal {Fp:.12e}, dual {D:.12e})'''()
                                            _wgap = max(_wgap, abs(D - Fp))
                                            print(f'''        {_nz:18s} p={_p:.2f} s={_sn:2d}: F_s={Fp / pn:.9f}  primal={Fp:.12e} <= dual={D:.12e}  gap={D - Fp:+.2e}  CERTIFIED''', flush = True)
                                            print(f'''        {len(_CERT_CASES)} branches certified by SDP duality; worst |gap| = {_wgap:.2e} (no trust-constr needed)''', flush = True)
                                            t0 = time.time()
                                            (_f_jac, _x_jac) = _sdp_branch(Mnp, _Gd, G_ref = _Gref)
                                            t_jac = time.time() - t0
                                            if not np.all(np.isfinite(_x_jac)):
                                                raise AssertionError
                                            _Tj = _np_T(_x_jac)
                                            if not float(np.real(np.trace(_Tj))) > -1e-08:
                                                raise 'returned x is not trace-feasible'()
                                            if not float(np.real(np.linalg.det(_Tj))) > -1e-10:
                                                raise 'returned x is not det-feasible'()
                                            if not abs(_f_jac - _np_obj(_x_jac, Mnp)) < 1e-09:
                                                raise (_f_jac, _np_obj(_x_jac, Mnp))()
                                            
                                            def _fd_branch(Mm = None, G_dec = None, G_ref = None):
                                                '''The pre-fix path: same starts/loop, but scipy differentiates by FD.

        Deliberately SLSQP-only.  The old loop ALSO ran trust-constr on every
        candidate; with no analytic Hessian scipy finite-differences the
        objective gradient inside `ScalarFunction.hess`, and that path is what
        hard-segfaulted Section B (killing the interpreter rather than raising),
        so re-running it here would put the whole self-test suite at risk.  It
        also never changed the answer -- measured on this very branch, SLSQP-only
  