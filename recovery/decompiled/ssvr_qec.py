# Source Generated with Decompyle++
# File: ssvr_qec.cpython-311.pyc (Python 3.11)

'''
SSVR-QEC : Soft-Syndrome Variational Recovery for Quantum Error Correction
==========================================================================
A variational quantum-classical machine-learning approach to quantum error
correction, demonstrated on the [[5,1,3]] perfect code.

NOTE ON DESIGN EVOLUTION:  The first prototype ("pure SSVR") applied a single
variational unitary R(phi(soft syndrome)) directly to rho_noisy.  Benchmarking
proved it fundamentally too weak: a unitary cannot project/purify, so it
converged to R = I (fidelity == raw).  The method was therefore upgraded to a
variational quantum INSTRUMENT (VSCR core), which keeps the quantum-classical
variational loop but makes it competitive:

METHOD (VSCR):
  1. Projective stabiliser syndrome measurement -> 16 syndrome projectors P_s.
     This preserves the purification/projection power of real decoders.
  2. A LEARNED variational unitary R_s is applied per syndrome.  The 16
     unitaries share parameters through a classical hypernetwork (learnable
     syndrome embedding -> rotation angles phi_s).  Corrections are CONTINUOUS
     and can be NON-PAULI (essential for amplitude-damping / mixed noise),
     unlike rigid Pauli lookup decoders.
  3. Trained UNSUPERVISED, end-to-end, on the expected recovery fidelity
     sum_s <psi| R_s P_s rho P_s R_s^dag |psi> plus a small code-space
     (manifold-consistency) penalty.  No error labels are required.

ENGINE: fully-differentiable density-matrix simulator in PyTorch
(complex128, n=5 physical qubits -> 32x32 matrices).  Real gradients (Adam);
no random-walk / fake gradient in the training loop (cf. the broken reference
ACE-QEC in main.py whose train_*_step use np.random.randn as "gradient").

BASELINES: Raw, Perfect-code lookup decoder (optimal single-error QEC),
Zero-Noise Extrapolation (ZNE), Virtual Distillation (VD),
Linear Data-driven Recovery (LinDR -- vnCDR-style), and VSCR (ours).

Author: generated research prototype.  Reproducible (seeds fixed).
'''
import math
import os
import time
import warnings
if not os.environ.get('QEC_ALLOW_CUDA'):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from torch.nn import nn
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import pyplot as plt
warnings.filterwarnings('ignore')
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(int(os.environ.get('QEC_TORCH_THREADS', '1')))
DTYPE = torch.complex128
N = 5
DIM = 2 ** N
LOG_DIM = N
I2 = torch.eye(2, dtype = DTYPE)
Xp = torch.tensor([
    [
        0,
        1],
    [
        1,
        0]], dtype = DTYPE)
Yp = torch.tensor([
    [
        0,
        (-0+-1j)],
    [
        (0+1j),
        0]], dtype = DTYPE)
Zp = torch.tensor([
    [
        1,
        0],
    [
        0,
        -1]], dtype = DTYPE)
Hp = (1 / math.sqrt(2)) * torch.tensor([
    [
        1,
        1],
    [
        1,
        -1]], dtype = DTYPE)

def Rz(t):
    '''Rz(t) = diag(e^{-it/2}, e^{it/2}).  t: real scalar tensor.'''
    return torch.diag(torch.stack([
        ((-0+-1j) * t / 2).exp(),
        ((0+1j) * t / 2).exp()]))


def Rx(t):
    c = (t / 2).cos()
    s = (t / 2).sin()
    return c * I2 - (0+1j) * s * Xp


def Ry(t):
    c = (t / 2).cos()
    s = (t / 2).sin()
    return c * I2 - (0+1j) * s * Yp


def Rzz(t):
    '''exp(-i t/2 Z⊗Z) = diag(e^{-it/2}, e^{it/2}, e^{it/2}, e^{-it/2}).'''
    e = ((-0+-1j) * t / 2).exp()
    ep = ((0+1j) * t / 2).exp()
    return torch.diag(torch.stack([
        e,
        ep,
        ep,
        e]))

CNOT = torch.tensor([
    [
        1,
        0,
        0,
        0],
    [
        0,
        1,
        0,
        0],
    [
        0,
        0,
        0,
        1],
    [
        0,
        0,
        1,
        0]], dtype = DTYPE)

def embed_gate(U, targets, n = (N,)):
    targets = sorted(targets)
    k = len(targets)
    others = range(n)()
    No = len(others)
    D = torch.eye(2 ** No, dtype = DTYPE).reshape([
        2] * No + [
        2] * No)
    Ur = U.reshape([
        2] * k + [
        2] * k)
    G = torch.tensordot(D, Ur, dims = 0)
    other_pos = enumerate(others)()
    targ_pos = enumerate(targets)()
    perm_in = []
    perm_out = []
    for q in range(n):
        if q in other_pos:
            perm_out.append(other_pos[q])
            perm_in.append(No + other_pos[q])
            continue
        perm_out.append(2 * No + targ_pos[q])
        perm_in.append(2 * No + k + targ_pos[q])
        G = G.permute(perm_out + perm_in)
        return G.reshape(2 ** n, 2 ** n)


def pauli_string(string, n = (N,)):
    """Build 2^n x 2^n Pauli from a string like 'XZZXI' (q0 = leftmost = MSB)."""
    table = {
        'I': I2,
        'X': Xp,
        'Y': Yp,
        'Z': Zp }
    G = table[string[0]]
    for ch in string[1:]:
        G = torch.kron(G, table[ch])
        return G


def _selftest_gates():
    '''Sanity checks for embed_gate / pauli_string conventions.'''
    G = embed_gate(Xp, [
        0], n = 2)
    if not torch.allclose(G, torch.kron(Xp, I2)):
        raise 'embed q0 failed'()
    G = embed_gate(Xp, [
        1], n = 2)
    if not torch.allclose(G, torch.kron(I2, Xp)):
        raise 'embed q1 failed'()
    G = embed_gate(CNOT, [
        0,
        1], n = 2)
    if not torch.allclose(G, CNOT):
        raise 'embed CNOT failed'()
    if not torch.allclose(pauli_string('XI', n = 2), torch.kron(Xp, I2)):
        raise 'pauli_string failed'()
    if not torch.allclose(embed_gate(Xp, [
        0], n = 2), pauli_string('XI', n = 2)):
        raise AssertionError

_selftest_gates()
STAB_STR = [
    'XZZXI',
    'IXZZX',
    'XIXZZ',
    'ZXIXZ']
STAB = STAB_STR()
XL = pauli_string('XXXXX')
ZL = pauli_string('ZZZZZ')
I_full = torch.eye(DIM, dtype = DTYPE)
P_code = I_full
for g in STAB:
    P_code = P_code @ 0.5 * (I_full + g)
    P_code = 0.5 * (P_code + P_code.conj().T)
    _rank = torch.linalg.matrix_rank(P_code.real.double())
    if not int(_rank) == 2:
        raise f'''code-space rank must be 2, got {_rank}'''()
    for i in range(4):
        for j in range(i + 1, 4):
            if not torch.allclose(STAB[i] @ STAB[j], STAB[j] @ STAB[i]):
                raise 'stabilisers do not commute'()
            
            def _logical_basis():
                '''Return |0_L>, |1_L> as 32-dim complex vectors (columns of E_code).

    Convention: |0_L> is the +1 eigenstate of Z_L inside the code space and
    |1_L> = X_L |0_L>  (so X_L flips 0<->1 exactly, removing global-phase
    ambiguity that would otherwise break an equality assertion).'''
                M = P_code @ ZL @ P_code
                Mh = 0.5 * (M + M.conj().T)
                (evals, evecs) = torch.linalg.eigh(Mh)
                zeroL = evecs[(:, evals.argmax())]
                j = zeroL.abs().argmax()
                zeroL = zeroL * torch.exp((-0+-1j) * torch.angle(zeroL[j]))
                oneL = XL @ zeroL
                if not torch.allclose(ZL @ zeroL, zeroL, atol = 1e-06):
                    raise '|0_L> not +1 of Z_L'()
                if not torch.allclose(ZL @ oneL, -oneL, atol = 1e-06):
                    raise '|1_L> not -1 of Z_L'()
                for g in STAB:
                    if not torch.allclose(g @ zeroL, zeroL, atol = 1e-06):
                        raise 'stab not +1 on |0_L>'()
                    if not torch.allclose(g @ oneL, oneL, atol = 1e-06):
                        raise 'stab not +1 on |1_L>'()
                    if not torch.allclose(XL @ zeroL, oneL, atol = 1e-06):
                        raise 'X_L does not flip 0->1'()
                    E = torch.stack([
                        zeroL,
                        oneL], dim = 1)
                    if not torch.allclose(E.conj().T @ E, I2, atol = 1e-06):
                        raise 'E not isometry'()
                    return E

            E_CODE = _logical_basis()
            ZERO_L = E_CODE[(:, 0)]
            ONE_L = E_CODE[(:, 1)]
            
            def encode(psi):
                '''psi: length-2 complex tensor (logical state) -> 32-dim encoded vector.'''
                return E_CODE @ psi.to(DTYPE)

            
            def fidelity(psi_enc, rho):
                '''Fidelity of mixed state rho w.r.t. pure |psi_enc> in the 32-dim space.'''
                return (psi_enc.conj() @ rho @ psi_enc).real

            
            def _anticommute(P_full, g):
                '''True if P_full and g anticommute.'''
                return not torch.allclose(P_full @ g, g @ P_full)

            
            def _syndrome_table():
                table = {
                    (0, 0, 0, 0): 'I' * N }
                for q in range(N):
                    for name, Pl in (('X', Xp), ('Y', Yp), ('Z', Zp)):
                        targets = [
                            'I'] * N
                        targets[q] = name
                        E = pauli_string(''.join(targets))
                        bits = (lambda .0 = None: Nonefor g in .0:
int(_anticommute(E, g))None)(STAB())
                        if not bits != (0, 0, 0, 0):
                            raise 'single error has zero syndrome?!'()
                        if not bits not in table:
                            raise f'''syndrome clash -> not a perfect code ({bits})'''()
                        table[bits] = ''.join(targets)
                        if not len(table) == 16:
                            raise f'''need 16 syndromes, got {len(table)}'''()
                        return table

            SYND_TABLE = _syndrome_table()
            SYND_BITS = range(16)()
            
            def _synd_projectors():
                '''16 syndrome projectors  P_s = prod_i (I + (-1)^{s_i} g_i)/2  (rank 2 each).'''
                out = []
                for bits in SYND_BITS:
                    P = I_full
                    for i, b in enumerate(bits):
                        P = P @ 0.5 * (I_full + -1 ** b * STAB[i])
                        out.append(P)
                        return out

            P_SYNDS = _synd_projectors()
            P_SYNDS_STACK = torch.stack(P_SYNDS)
            C_SYNDS = SYND_BITS()
            Xq = range(N)()
            Yq = range(N)()
            Zq = range(N)()
            
            def apply_depolarizing(rho, p):
                '''Independent single-qubit depolarizing of rate p on every qubit.'''
                for q in range(N):
                    rho = (1 - p) * rho + (p / 3) * ((Xq[q] @ rho @ Xq[q]) + (Yq[q] @ rho @ Yq[q]) + (Zq[q] @ rho @ Zq[q]))
                    return rho

            
            def amp_kraus(gamma, q):
                '''Two Kraus ops for amplitude damping on qubit q (rate gamma).'''
                K0_2 = torch.tensor([
                    [
                        1,
                        0],
                    [
                        0,
                        math.sqrt(1 - gamma)]], dtype = DTYPE)
                K1_2 = torch.tensor([
                    [
                        0,
                        math.sqrt(gamma)],
                    [
                        0,
                        0]], dtype = DTYPE)
                return (embed_gate(K0_2, [
                    q]), embed_gate(K1_2, [
                    q]))

            
            def apply_amplitude_damping(rho, gamma):
                for q in range(N):
                    (K0, K1) = amp_kraus(gamma, q)
                    rho = (K0 @ rho @ K0.conj().T) + (K1 @ rho @ K1.conj().T)
                    return rho

            
            def apply_mixed(rho, p, gamma):
                rho = apply_depolarizing(rho, p)
                rho = apply_amplitude_damping(rho, gamma)
                return rho

            
            def apply_coherent(rho, eps):
                '''Systematic coherent over-rotation: identical Rx(eps) on every qubit
    (calibration-type control error).  NOT a Pauli mixture.'''
                U = I_full
                for q in range(N):
                    U = U @ embed_gate(Rx(torch.tensor(eps, dtype = DTYPE)), [
                        q])
                    return U @ rho @ U.conj().T

            
            def noisy_state(psi_enc, p, noise = ('depolarizing',)):
                '''Build the noisy density matrix from a pure encoded state.'''
                rho = torch.outer(psi_enc, psi_enc.conj())
                if noise == 'depolarizing':
                    rho = apply_depolarizing(rho, p)
                elif noise == 'amplitude_damping':
                    rho = apply_amplitude_damping(rho, p)
                elif noise == 'mixed':
                    rho = apply_mixed(rho, p, 0.5 * p)
                elif noise == 'coherent':
                    rho = apply_coherent(rho, p)
                else:
                    raise ValueError(noise)
                return rho

            N_LAYERS = 3
            PARAMS_PER_LAYER = 3 * N + N
            PHI_DIM = PARAMS_PER_LAYER * N_LAYERS
            
            def _build_recovery_slots(n, n_layers = (N, N_LAYERS)):
                slots = []
                for _ in range(n_layers):
                    for q in range(n):
                        slots += [
                            Zq[q],
                            Xq[q],
                            Zq[q]]
                        for q in range(n):
                            b = (q + 1) % n
                            (a, b2) = (q, b) if q < b else (b, q)
                            slots.append(Zq[a] @ Zq[b2])
                            return slots

            P_SLOTS = _build_recovery_slots()
            MP_SLOTS = P_SLOTS()
            
            def recovery_unitary(phi):
                '''phi: real tensor of length PHI_DIM -> 32x32 complex unitary R.'''
                R = I_full.clone()
                half = phi / 2
                for k in range(len(P_SLOTS)):
                    G = torch.cos(half[k]) * I_full + torch.sin(half[k]) * MP_SLOTS[k]
                    R = G @ R
                    return R

            OBS_LIST = (lambda .0: [ embed_gate(Zp, [
q]) for q in .0 ]) + range(N)()
            N_FEAT = len(OBS_LIST)
            OBS_STACK = torch.stack(OBS_LIST)
            
            def features_of(rho):
                '''rho: 32x32 complex -> float64 feature tensor (vectorised, no grad).'''
                return torch.einsum('kij,ji->k', OBS_STACK, rho).real

            
            def recovery_unitary_batch(phi_batch):
                '''phi_batch: (B, PHI_DIM) real -> (B, 32, 32) complex unitaries (batched).'''
                B = phi_batch.shape[0]
                R = I_full.unsqueeze(0).expand(B, DIM, DIM).clone()
                half = phi_batch / 2
                for k in range(len(P_SLOTS)):
                    c = torch.cos(half[(:, k)]).view(B, 1, 1)
                    s = torch.sin(half[(:, k)]).view(B, 1, 1)
                    G = c * I_full.unsqueeze(0) + s * MP_SLOTS[k].unsqueeze(0)
                    R = torch.bmm(G, R)
                    return R

            P_EXP = (lambda .0: [ P.to(DTYPE).contiguous() for P in .0 ])(P_SLOTS())
            
            def recovery_action_cols(phi_batch, X):
                '''(B, PHI_DIM) real angles, (B, DIM, r) complex columns -> R(phi_b) X_b.

    Identical to `recovery_unitary_batch(phi_batch) @ X` but O(PHI_DIM * DIM^2 * r)
    instead of O(PHI_DIM * DIM^3), and differentiable w.r.t. phi_batch.
    '''
                Y = X
                half = phi_batch / 2
                for k in range(len(P_SLOTS)):
                    c = torch.cos(half[(:, k)]).to(DTYPE).view(-1, 1, 1)
                    s = torch.sin(half[(:, k)]).to(DTYPE).view(-1, 1, 1)
                    Y = c * Y - (0+1j) * s * torch.matmul(P_EXP[k], Y)
                    return Y

            
            def recovery_action_cols_dag(phi_batch, X):
                '''(B, PHI_DIM), (B, DIM, r) -> R(phi_b)^† X_b  (adjoint column propagation).

    R(phi) = G_{59} ... G_0 with G_k = exp(-i phi_k P_k / 2), so
    R^† = G_0^† G_1^† ... G_59^† and the gates must be applied in REVERSE order
    with the sign of the sin term flipped.  Verified against
    `recovery_unitary_batch(phi).conj().transpose(1,2) @ X` in
    `_selftest_action_cols`.
    '''
                Y = X
                half = phi_batch / 2
                for k in reversed(range(len(P_SLOTS))):
                    c = torch.cos(half[(:, k)]).to(DTYPE).view(-1, 1, 1)
                    s = torch.sin(half[(:, k)]).to(DTYPE).view(-1, 1, 1)
                    Y = c * Y + (0+1j) * s * torch.matmul(P_EXP[k], Y)
                    return Y

            W_SYND = None
            
            def syndrome_bases():
                '''W[s] = (32, 2) orthonormal basis of range(P_s), the rank-2 syndrome space.

    Only needed by the fast quadrature loss.  The final objective is invariant
    under the choice of basis because it enters exclusively through
    W_s W_s^† = P_s (see `vscr_fidelity_quad`).
    '''
                global W_SYND
                if W_SYND is not None:
                    W = np.zeros((16, DIM, 2), dtype = complex)
                    for s in range(16):
                        (ev, evec) = np.linalg.eigh(P_SYNDS[s].numpy())
                        if not abs(ev[-1] - 1) < 1e-12 or abs(ev[-2] - 1) < 1e-12:
                            raise AssertionError
                        if not abs(ev[-3]) < 1e-12:
                            raise 'syndrome projector is not rank 2'()
                        W[s] = evec[(:, -2:)]
                        if not np.allclose(W[s].conj().T @ W[s], np.eye(2), atol = 1e-12):
                            raise AssertionError
                        if not np.allclose(W[s] @ W[s].conj().T, P_SYNDS[s].numpy(), atol = 1e-12):
                            raise AssertionError
                        W_SYND = torch.tensor(W, dtype = DTYPE)
                        return W_SYND

            
            def vscr_fidelity_quad(phi_batch, Q, synd_w = (None,)):
                '''EXACT Haar-quadrature recovery fidelity, vectorised over syndromes AND
    probe states, propagating only TWO columns per syndrome.

        F = sum_j w_j sum_s <psi_j| R_s P_s rho_j P_s R_s^dag |psi_j>

    Writing P_s = W_s W_s^dag and G_s = R_s W_s (32x2),

        u_{sj} = P_s R_s^dag psi_j = W_s G_s^dag psi_j = W_s z_{sj},
        term   = u^dag rho_j u = z_{sj}^dag B_{sj} z_{sj},   B_{sj} = W_s^dag rho_j W_s,

    so the 60-gate ansatz is applied to the 2 columns of W_s ONLY, and every
    rho_j-dependent quantity (B) is a constant precomputed once per training
    stage by `quad_cache`.  This is the whole point: the exact objective costs
    ~1/20 of the Monte-Carlo path per epoch *and* has zero gradient variance,
    whereas the non-Pauli headroom it is meant to resolve (1.9e-4 .. 3.4e-3
    over the Pauli decoder) sits right at the 48-sample Monte-Carlo SEM
    (1.1e-4 for amplitude damping at p = 0.10).

    Returns (loss, F, F_s) with F_s the exact per-syndrome averaged fidelity
    (used for the label-free worst-syndrome model-selection criterion).

    NOTE: the code-population regulariser (`lam` in `vscr_fidelity`) needs the
    full 32x32 R_s and is not supported here; every paper configuration trains
    with lam = 0, which `train_vscr` asserts.
    '''
                G = recovery_action_cols(phi_batch, Q['W'])
                z = torch.einsum('sib,ni->snb', G.conj(), Q['psi'])
                T = torch.einsum('snb,snbc,snc->sn', z.conj(), Q['BS'], z).real
                F_s = (T * Q['w']).sum(dim = 1)
                F = F_s.sum() if synd_w is not None else (synd_w * F_s).sum()
                return (1 - F, F, F_s)

            
            def _selftest_action_cols():
                g = torch.Generator().manual_seed(11)
                phi = torch.randn(16, PHI_DIM, dtype = torch.float64, generator = g) * 0.7
                Xr = torch.randn(16, DIM, 2, dtype = torch.float64, generator = g)
                Xi = torch.randn(16, DIM, 2, dtype = torch.float64, generator = g)
                X = (Xr + (0+1j) * Xi).to(DTYPE)
                R = recovery_unitary_batch(phi)
                err = float((torch.bmm(R, X) - recovery_action_cols(phi, X)).abs().max())
                errd = float((torch.bmm(R.conj().transpose(1, 2), X) - recovery_action_cols_dag(phi, X)).abs().max())
                if not err < 1e-12:
                    raise err()
                if not errd < 1e-12:
                    raise errd()
                p1 = phi.clone().requires_grad_(True)
                p2 = phi.clone().requires_grad_(True)
                torch.bmm(recovery_unitary_batch(p1), X).abs().sum().backward()
                recovery_action_cols(p2, X).abs().sum().backward()
                gerr = float((p1.grad - p2.grad).abs().max())
                if not gerr < 1e-11:
                    raise gerr()
                eye = torch.eye(DIM, dtype = DTYPE).unsqueeze(0).expand(16, DIM, DIM)
                err2 = float((R - recovery_action_cols(phi, eye.contiguous())).abs().max())
                if not err2 < 1e-12:
                    raise err2()
                for noise in ('depolarizing', 'amplitude_damping', 'coherent', 'mixed'):
                    Q = quad_cache((0.05, 0.05), noise, n_u = 2, n_phi = 3, n_p = 1)
                    wts = Q['w']
                    rho = Q['rho']
                    psi = Q['psi']
                    (loss_q, F_q, F_s_q) = vscr_fidelity_quad(phi.detach(), Q)
                    R_all = recovery_unitary_batch(phi.detach())
                    F_ref = 0
                    F_s_ref = torch.zeros(16, dtype = torch.float64)
                    for j in range(psi.shape[0]):
                        (_, Fj) = vscr_fidelity(psi[j], rho[j], R_all, lam = 0)
                        F_ref += float(wts[j]) * float(Fj)
                        for s in range(16):
                            u = P_SYNDS[s] @ R_all[s].conj().T @ psi[j]
                            if not abs(float(F_q) - F_ref) < 1e-11:
                                raise (noise, float(F_q), F_ref)()
                            if not float((F_s_q - F_s_ref).abs().max()) < 1e-11:
                                raise AssertionError
                            if not abs(float(F_q) - float(F_s_q.sum())) < 1e-12:
                                raise AssertionError
                            Q['W'].clone() = None
                            th = torch.tensor(0.7, dtype = torch.float64)
                            rot = torch.tensor([
                                [
                                    th.cos(),
                                    -th.sin()],
                                [
                                    th.sin(),
                                    th.cos()]], dtype = DTYPE)
                            W2 = torch.matmul(W2, rot)
                            Q2 = dict(Q)
                            Q2['W'] = W2
                            Q2['BS'] = torch.einsum('sia,nij,sjb->snab', W2.conj(), rho, W2)
                            d = abs(float(vscr_fidelity_quad(phi.detach(), Q2)[1]) - float(F_q))
                            if not d < 1e-12:
                                raise d()
                            print(f'''[action_cols selftest] {noise:20s}: vectorised quad loss {float(F_q):.12f} == reference loop {F_ref:.12f}, basis-rotation invariant to {d:.1e}  VERIFIED''', flush = True)
                            print(f'''[action_cols selftest] values agree to {err:.2e} (dag {errd:.2e}), gradients to {gerr:.2e}, r=DIM case to {err2:.2e}  VERIFIED''', flush = True)
                            return (err, gerr)

            
            class VSCR(nn.Module):
                
                def __init__(self = None, n_synd = None, phi_dim = None, hidden = None, ctx = None):
                    super().__init__()
                    self.synd_emb = nn.Parameter(torch.randn(n_synd, ctx, dtype = torch.float64) * 0.1)
                    self.hnet = nn.Sequential(nn.Linear(ctx, hidden, dtype = torch.float64), nn.Tanh(), nn.Linear(hidden, phi_dim, dtype = torch.float64))
                    self.phi_base = nn.Parameter(torch.randn(phi_dim, dtype = torch.float64) * 0.05)

                
                def forward(self):
                    return self.phi_base + self.hnet(self.synd_emb)

                __classcell__ = None

            
            def vscr_fidelity(psi_enc, rho_noisy, R_all, lam, synd_w = (0, None)):
                '''Expected recovery fidelity given the 16 recovery unitaries R_all:

        F = sum_s <psi| R_s P_s rho P_s R_s^dag |psi>     (exact expectation
        over the projective syndrome outcome).  Vectorised over 16 syndromes.
        R_all can be SHARED across samples (built once per epoch/eval).

        synd_w: optional length-16 weight vector rescaling the per-syndrome
        terms (training-time gradient balancing; the optimum is unchanged
        because each R_s only appears in its own syndrome term).'''
                rho16 = rho_noisy.unsqueeze(0).expand(16, DIM, DIM)
                PR = torch.bmm(P_SYNDS_STACK, rho16)
                PRP = torch.bmm(PR, P_SYNDS_STACK)
                RPR = torch.bmm(R_all, PRP)
                RPRd = torch.bmm(RPR, R_all.conj().transpose(1, 2))
                pc = psi_enc.conj()
                F_s = torch.einsum('i,sij,j->s', pc, RPRd, psi_enc).real
                F = F_s.sum() if synd_w is not None else (synd_w * F_s).sum()
                if lam == 0:
                    return (1 - F, F)
                pop_s = None.einsum('ij,sji->s', P_code, RPRd).real
                code_pop = pop_s.sum() if synd_w is not None else (synd_w * pop_s).sum()
                loss = (1 - F) + lam * (1 - code_pop)
                return (loss, F)

            
            def vscr_forward(model, psi_enc, rho_noisy, lam = (0.1,)):
                '''Convenience wrapper: build R_all from the model, then score.'''
                R_all = recovery_unitary_batch(model())
                return vscr_fidelity(psi_enc, rho_noisy, R_all, lam = lam)

            
            def baseline_raw(psi_enc, rho_noisy):
                return fidelity(psi_enc, rho_noisy)

            
            def perfect_code_decoder_fidelity(psi_enc, rho_noisy):
                '''Exact expected fidelity of the optimal single-error lookup decoder:

        F = sum_s <psi| C_s P_s rho P_s C_s^dag |psi>     with P_s the
    syndrome-s projector (rank 2) and C_s the rigid Pauli correction.
    Uses precomputed P_SYNDS / C_SYNDS.'''
                val = torch.zeros((), dtype = torch.float64)
                for s in range(16):
                    block = C_SYNDS[s] @ P_SYNDS[s] @ rho_noisy @ P_SYNDS[s] @ C_SYNDS[s].conj().T
                    val = val + fidelity(psi_enc, block)
                    return val

            
            def zne_fidelity(psi_enc, p, noise):
                '''Richardson ZNE at noise scales 1,2,3 (state-level extrapolation).'''
                r1 = noisy_state(psi_enc, p * 1, noise)
                r2 = noisy_state(psi_enc, p * 2, noise)
                r3 = noisy_state(psi_enc, p * 3, noise)
                rho_zne = (3 * r1 - 3 * r2) + r3
                return fidelity(psi_enc, rho_zne)

            
            def virtual_distillation_fidelity(psi_enc, rho_noisy, k = (2,)):
                '''rho_VD ~ rho^k (unnormalised) projected toward the purest component.'''
                r = rho_noisy
                for _ in range(k - 1):
                    r = r @ rho_noisy
                    rho_vd = 0.5 * (r + r.conj().T)
                    trc = torch.trace(rho_vd).real
                    if trc.abs() < 1e-12:
                        return fidelity(psi_enc, rho_noisy)
                    rho_vd = None / trc
                    return fidelity(psi_enc, rho_vd)

            
            def _pauli_index_set(n, max_weight = (N, N)):
                import itertools
                idx = []
                for w in range(0, max_weight + 1):
                    for qs in itertools.combinations(range(n), w):
                        for paulis in itertools.product('XYZ', repeat = w):
                            s = [
                                'I'] * n
                            for qi, pch in zip(qs, paulis):
                                s[qi] = pch
                                idx.append(''.join(s))
                                return idx

            PAULI_IDX = _pauli_index_set()
            N_PAULI = len(PAULI_IDX)
            PAULI_MATS = PAULI_IDX()
            PAULI_STACK = torch.stack(PAULI_MATS)
            
            def _pauli_features(rho):
                '''Vectorised Pauli-expectation feature vector (no grad).'''
                return torch.einsum('kij,ji->k', PAULI_STACK, rho).real

            
            def _reconstruct_state(coeffs):
                '''rho = (1/2^n) sum_P c_P P  (c_I forced to 1 by the caller).  Vectorised.'''
                return torch.einsum('k,kij->ij', coeffs.to(DTYPE), PAULI_STACK) / INV_DIM

            INV_DIM = 2 ** N
            
            def train_lindr(n_train, p_range, noise, n_val, rng_seed = (1500, (0.01, 0.12), 'depolarizing', 200, SEED + 99)):
                '''Ridge least-squares fit  A : noisy-Pauli-features -> ideal-Pauli-features.

    The ridge strength is chosen on a held-out validation set by
    reconstruction fidelity.  Returns A with the column convention y = A @ x.'''
                rng = np.random.RandomState(rng_seed)
                
                def gen(m = None):
                    PS = []
                    Ys = []
                    Xs = []
                    for _ in range(m):
                        psi = random_logical_state(1, rng)
                        psi_enc = encode(psi)
                        p = rng.uniform(*p_range)
                        rho_n = noisy_state(psi_enc, p, noise)
                        Xs.append(_pauli_features(rho_n).numpy())
                        rho_id = torch.outer(psi_enc, psi_enc.conj())
                        Ys.append(_pauli_features(rho_id).numpy())
                        PS.append(psi_enc)
                        return (np.stack(Xs), np.stack(Ys), torch.stack(PS))

                (X, Y, _) = gen(n_train)
                (Vx, _, Vpsi) = gen(n_val)
                XtX = X.T @ X
                XtY = X.T @ Y
                (best_f, best_B) = (-1, None)
                for a in (0, 0.0001, 0.001, 0.01, 0.1, 1):
                    B = np.linalg.solve(XtX + (a * n_train + 1e-06) * np.eye(N_PAULI), XtY)
                    Yh = Vx @ B
                    Yh[(:, 0)] = 1
                    coeffs = torch.tensor(Yh, dtype = torch.float64).to(DTYPE)
                    rho_b = torch.einsum('mk,kij->mij', coeffs, PAULI_STACK) / INV_DIM
                    F = torch.einsum('mi,mij,mj->m', Vpsi.conj(), rho_b, Vpsi).real
                    if float(F.mean()) > best_f:
                        best_B = B
                        best_f = float(F.mean())
                    return torch.tensor(best_B.T.copy(), dtype = torch.float64)

            
            def lindr_fidelity(psi_enc, rho_noisy, A):
                x = _pauli_features(rho_noisy)
                y = A @ x
                y = y.clone()
                y[0] = 1
                rho_lindr = _reconstruct_state(y)
                return fidelity(psi_enc, rho_lindr)

            
            def random_logical_state(n, rng = (1, None)):
                '''Uniform-Haar random n-qubit logical pure state as a length-2^n complex tensor.'''
                if not rng:
                    pass
                rng = np.random
                d = 2 ** n
                g = rng.standard_normal(d) + (0+1j) * rng.standard_normal(d)
                g = g / np.linalg.norm(g)
                return torch.tensor(g, dtype = DTYPE)

            
            def clifford_logical_states():
                '''The 6 single-qubit Pauli eigenstates (|0>,|1>,|+>,|->,|+i>,|-i>).'''
                s = []
                for vec in (np.array([
                    1,
                    0], dtype = complex), np.array([
                    0,
                    1], dtype = complex), np.array([
                    1,
                    1], dtype = complex) / np.sqrt(2), np.array([
                    1,
                    -1], dtype = complex) / np.sqrt(2), np.array([
                    1,
                    (0+1j)], dtype = complex) / np.sqrt(2), np.array([
                    1,
                    (-0+-1j)], dtype = complex) / np.sqrt(2)):
                    s.append(torch.tensor(vec, dtype = DTYPE))
                    return s

            
            def haar_quadrature(n_u, n_phi = (3, 7)):
                '''Deterministic exact Haar quadrature over single-logical-qubit states.

    Returns (states, weights): states is (n_u*n_phi, 2) complex128, weights is
    (n_u*n_phi,) float64 summing to 1.  For any observable polynomial of degree
    <= 2 in the Bloch vector, sum_i w_i <psi_i|O|psi_i> equals the Haar average.
    '''
                (u, wu) = np.polynomial.legendre.leggauss(n_u)
                phis = 2 * np.pi * np.arange(n_phi) / n_phi
                weights = []
                states = []
                for iu in range(n_u):
                    th = math.acos(float(np.clip(u[iu], -1, 1)))
                    for ip in range(n_phi):
                        states.append(np.array([
                            math.cos(th / 2),
                            np.exp((0+1j) * phis[ip]) * math.sin(th / 2)], dtype = complex))
                        weights.append(wu[iu] / 2 / n_phi)
                        return (torch.tensor(np.array(states), dtype = DTYPE), torch.tensor(np.array(weights), dtype = torch.float64))

            
            def p_quadrature(p_range, n = (6,)):
                '''Gauss-Legendre nodes/weights for averaging over a noise-strength interval.

    n = 6 is exact to machine precision for all four channels used here (see the
    convergence table in the comment block above `haar_quadrature`); the
    integrand is analytic in p but NOT polynomial -- `apply_depolarizing`
    composes five single-qubit maps (degree 5), amplitude damping enters through
    sqrt(1-gamma), and `apply_coherent` is trigonometric -- so a low-order rule
    such as n = 2 leaves a bias of up to 5.6e-5.
    '''
                (x, w) = np.polynomial.legendre.leggauss(n)
                hi = float(p_range[1])
                lo = float(p_range[0])
                half = 0.5 * (hi - lo)
                mid = 0.5 * (lo + hi)
                return (mid + half * x, w / 2)

            
            def quad_cache(p_range, noise, n_u, n_phi, n_p = (3, 7, 6)):
                """Precompute everything the exact quadrature objective needs, ONCE.

    Returns a dict Q with
      'psi'  (nq, 32) complex128   encoded probe states V|psi_j>   (constant)
      'rho'  (nq, 32, 32)          E_p(V|psi_j><psi_j|V^†)         (constant)
      'w'    (nq,) float64         Haar x noise-strength weights, sum = 1
      'W'    (16, 32, 2)           orthonormal syndrome bases      (constant)
      'BS'   (16, nq, 2, 2)        W_s^† rho_j W_s                 (constant)
      'p_syn' (16,) float64        syndrome probabilities sum_j w_j Tr[B_sj]
      'meta' dict                  node bookkeeping
    nq = n_u * n_phi * n_p = 126 for the defaults.  The channel is therefore
    applied exactly once per training stage instead of once per epoch per
    sample, and `BS` folds the density matrices down to the 2x2 syndrome
    blocks so that a training epoch only propagates 2 columns per syndrome
    through the 60-gate ansatz.
    """
                (states, sw) = haar_quadrature(n_u, n_phi)
                (p_nodes, pw) = p_quadrature(p_range, n_p)
                ws = []
                rhos = []
                psis = []
                for ip, p in enumerate(p_nodes):
                    for iq in range(states.shape[0]):
                        pe = encode(states[iq])
                        psis.append(pe.detach())
                        rhos.append(noisy_state(pe, float(p), noise).detach())
                        ws.append(float(pw[ip]) * float(sw[iq]))
                        psi = torch.stack(psis)
                        rho = torch.stack(rhos)
                        w = torch.tensor(ws, dtype = torch.float64)
                        w = w / float(w.sum())
                        W = syndrome_bases()
                        BS = torch.einsum('sia,nij,sjb->snab', W.conj(), rho, W)
                        p_syn = torch.einsum('sn,n->s', BS.diagonal(dim1 = 2, dim2 = 3).sum(-1).real, w)
                        if not p_syn.dtype == w.dtype or torch.all(p_syn >= -1e-15):
                            raise p_syn()
                        if not abs(float(p_syn.sum()) - 1) < 1e-12:
                            raise float(p_syn.sum())()
                        meta = {
                            'p_nodes': (lambda .0: [ float(x) for x in .0 ]),
                            'p_weights': pw(),
                            'n_u': n_u,
                            'n_phi': n_phi,
                            'n_p': n_p,
                            'noise': noise,
                            'p_range': (float(p_range[0]), float(p_range[1])) }
                        return {
                            'psi': psi,
                            'rho': rho,
                            'w': w,
                            'W': W,
                            'BS': BS,
                            'p_syn': p_syn,
                            'meta': meta }

            
            def syndrome_conditional_fidelity_exact(phi, Q):
                '''EXACT (zero-variance) label-free per-syndrome conditional fidelity.

        cf_s = sum_j w_j <psi_j| R_s P_s rho_j P_s R_s^dag |psi_j> / p_s
             = F_s / p_s ,

    i.e. the same quantity as `syndrome_conditional_fidelity` but with the
    Haar average and the syndrome probabilities both replaced by the exact
    quadrature in `Q`.  This matters: the Monte-Carlo version uses M = 150
    probe states, so its sem is ~1e-3 -- an order of magnitude LARGER than the
    1.9e-4 non-Pauli headroom the warm start is being selected on, which makes
    seed selection essentially a coin flip.  Returns (cf, p_s) as float64
    arrays of shape (16,).
    '''
                torch.no_grad()
                (_, _, F_s) = vscr_fidelity_quad(torch.as_tensor(phi, dtype = torch.float64).reshape(-1, PHI_DIM), Q)
                None(None, None)

            
            def _selftest_quadrature(n_u, n_phi, n_mc, seed = (3, 7, None, 7)):
                """Prove the quadrature is EXACT for the objective's algebraic degree.

    Test 1 (algebraic, machine precision): for random 2x2 operators A, B the
    Haar second moment is  E[<psi|A|psi><psi|B|psi>] = (TrA TrB + TrAB)/6.
    Test 2 (end to end): the quadrature-averaged recovery fidelity equals
    (i) the independent analytic branch sum built from `_branch_M`'s exact Haar
    matrix M_s, and (ii) `vscr_paper_abl.exact_F`'s quadratic-form route, both
    to machine precision, and (iii) a Monte-Carlo average to within 6 sigma.
    Test 3: the noise-strength rule CONVERGES.  rho(p) is not polynomial (see
    the comment block above), so the test measures the Gauss-Legendre error
    against a 40-node reference and pins n_p = 6 to machine precision while
    recording how badly the naive 2-node rule is biased.

    n_mc=None picks a per-channel sample count: the depolarizing channel builds
    1024 Kraus operators per call (~0.3 s), so a large MC average there would
    dominate the runtime while adding nothing -- the analytic comparisons are
    already exact to 1e-11.
    """
                import vscr_paper as vp
                import vscr_paper_abl as ab
                if n_mc is not None:
                    n_mc = {
                        'depolarizing': 60,
                        'amplitude_damping': 300,
                        'coherent': 300,
                        'mixed': 300 }
                (states, w) = haar_quadrature(n_u, n_phi)
                if not abs(float(w.sum()) - 1) < 1e-14:
                    raise AssertionError
                rng = np.random.RandomState(seed)
                worst = 0
                for _ in range(200):
                    A = torch.tensor(rng.normal(size = (2, 2)) + (0+1j) * rng.normal(size = (2, 2)), dtype = DTYPE)
                    B = torch.tensor(rng.normal(size = (2, 2)) + (0+1j) * rng.normal(size = (2, 2)), dtype = DTYPE)
                    q = (lambda .0 = None: Nonefor i in .0:
float(w[i]) * complex(states[i].conj() @ A @ states[i]) * complex(states[i].conj() @ B @ states[i])None)(range(states.shape[0])())
                    exact = (torch.trace(A) * torch.trace(B) + torch.trace(A @ B)) / 6
                    worst = max(worst, abs(q - complex(exact)))
                    if not worst < 1e-12:
                        raise f'''Haar second moment not exact: {worst:.3e}'''()
                    print(f'''[quad selftest 1] E[<A><B>] exact to {worst:.2e} ({states.shape[0]} states)  VERIFIED''', flush = True)
                    R = recovery_unitary_batch(vp.PHI_DEC)
                    for Fq in (('depolarizing', 0.1), ('amplitude_damping', 0.1), ('coherent', 0.1), ('mixed', 0.1)):
                        (noise, p) = None
                        acc = []
                        Fmc = 0
                        nmc = n_mc[noise] if isinstance(n_mc, dict) else n_mc
                        rg = np.random.RandomState(seed + 1)
                        for _ in range(nmc):
                            pe = encode(random_logical_state(1, rg))
                            acc.append(float(vscr_fidelity(pe, noisy_state(pe, p, noise), R, lam = 0)[1]))
                            acc = np.array(acc)
                            Fmc = float(acc.mean())
                            sem = float(acc.std(ddof = 1) / math.sqrt(nmc))
                            (Fb, _) = ab.exact_F(vp.PHI_DEC, noise, p)
                            Fa = 0
                            for s in range(16):
                                (M_s, p_s) = ab._branch_M(noise, p, s)
                                if p_s < 1e-14:
                                    continue
                                G = ab.V_ISO.conj().T @ C_SYNDS[s].numpy() @ ab.W_BASIS[s]
                                v = np.zeros(4, dtype = complex)
                                for a in range(2):
                                    for b in range(2):
                                        v[a * 2 + b] = G[(b, a)] / math.sqrt(2)
                                        L = np.zeros((4, 4), dtype = complex)
                                        L[(:, 0)] = v
                                        Fa += float(np.real(2 * np.trace(L @ L.conj().T @ M_s)))
                                        if not abs(Fq - Fa) < 1e-11:
                                            raise (noise, p, Fq, Fa)()
                                        if not abs(Fq - Fb) < 1e-11:
                                            raise (noise, p, Fq, Fb)()
                                        if not abs(Fq - Fmc) < 6 * sem + 1e-09:
                                            raise (noise, p, Fq, Fmc, sem)()
                                        sem48 = float(acc.std(ddof = 1) / math.sqrt(48))
                                        print(f'''[quad selftest 2] {noise:20s} p={p}: quad={Fq:.10f} M-route={Fa:.10f}  Q-route={Fb:.10f}  MC({nmc})={Fmc:.7f}+-{sem:.1e} (6-sigma OK);  sem of a 48-sample batch = {sem48:.2e}''', flush = True)
                                        for Favg in ('depolarizing', 'amplitude_damping', 'coherent', 'mixed'):
                                            noise = None
                                            
                                            def Fav(pv = None, _n = None):
                                                return (lambda .0 = None: Nonefor i in .0:
float(w[i]) * float(vscr_fidelity(encode(states[i]), noisy_state(encode(states[i]), float(pv), _n), R, lam = 0)[1])None)(range(states.shape[0])())

                                            ref = Favg(40)
                                            e6 = abs(Favg(6) - ref)
                                            e2 = abs(Favg(2) - ref)
                                            if not e6 < 1e-13:
                                                raise (noise, e6)()
                                            if not noise == 'depolarizing' and e2 > 1e-09:
                                                raise (noise, e2)()
                                            Qc = quad_cache((0.02, 0.15), noise)
                                            if not Qc['psi'].shape[0] == 126:
                                                raise Qc['psi'].shape()
                                            if not abs(float(Qc['w'].sum()) - 1) < 1e-14:
                                                raise AssertionError
                                            print(f'''[quad selftest 3] {noise:20s} p-average on [0.02,0.15]: 2-node err={e2:.2e} (biased, as documented)   6-node err={e6:.2e} vs 40-node ref={ref:.12f}  VERIFIED''', flush = True)
                                            import time as _time
                                            phi_t = torch.randn(16, PHI_DIM, dtype = torch.float64, generator = torch.Generator().manual_seed(5)) * 0.6
                                            for Q in ('depolarizing', 'amplitude_damping', 'coherent', 'mixed'):
                                                noise = None
                                                t0 = _time.time()
                                                for _ in range(20):
                                                    (_l, Fv, F_s) = vscr_fidelity_quad(phi_t, Q)
                                                    t_fast = (_time.time() - t0) / 20
                                                    R_all = recovery_unitary_batch(phi_t)
                                                    t0 = _time.time()
                                                    F_ref = 0
                                                    F_s_ref = torch.zeros(16, dtype = torch.float64)
                                                    for j in range(Q['psi'].shape[0]):
                                                        (_, Fj) = vscr_fidelity(Q['psi'][j], Q['rho'][j], R_all, lam = 0)
                                                        F_ref += float(Q['w'][j]) * float(Fj)
                                                        for s in range(16):
                                                            for j in range(Q['psi'].shape[0]):
                                                                u = P_SYNDS[s] @ R_all[s].conj().T @ Q['psi'][j]
                                                                _time.time() - t0 = None
                                                                if not abs(float(Fv) - F_ref) < 1e-11:
                                                                    raise (noise, float(Fv), F_ref)()
                                                                if not float((F_s - F_s_ref).abs().max()) < 1e-11:
                                                                    raise AssertionError
                                                                print(f'''[quad selftest 4] {noise:20s} vectorised F={float(Fv):.12f} == reference {F_ref:.12f};  {t_fast * 1000:.1f} ms/epoch vs {t_ref * 1000:.0f} ms for the reference loop ({t_ref / max(t_fast, 1e-09):.0f}x faster)  VERIFIED''', flush = True)
                                                                for Qa in ('depolarizing', 'amplitude_damping'):
                                                                    noise = None
                                                                    Qb = quad_cache((0.02, 0.15), noise, n_u = 3, n_phi = 5)
                                                                    a = float(vscr_fidelity_quad(phi_t, Qa)[1])
                                                                    b = float(vscr_fidelity_quad(phi_t, Qb)[1])
                                                                    print(f'''[quad selftest 5] {noise:20s} {Qa['psi'].shape[0]}-node F={a:.12f}  {Qb['psi'].shape[0]}-node F={b:.12f}  (diff {a - b:+.2e} -- deterministic, no Monte-Carlo noise)''', flush = True)
                                                                    
                                                                    class _Probe:
                                                                        
                                                                        def __init__(self, phi_):
                                                                            self._phi = phi_

                                                                        
                                                                        def __call__(self):
                                                                            return self._phi


                                                                    for Q1 in ('depolarizing', 'amplitude_damping', 'mixed'):
                                                                        noise = None
                                                                        p = 0.07
                                                                        ps_def = torch.einsum('n,nij,sji->s', Q1['w'].to(Q1['rho'].dtype), Q1['rho'], P_SYNDS_STACK.to(Q1['rho'].dtype)).real.numpy()
                                                                        if not np.abs(ps_def - Q1['p_syn'].numpy()).max() < 1e-13:
                                                                            raise AssertionError
                                                                        if not abs(float(Q1['p_syn'].sum()) - 1) < 1e-12:
                                                                            raise AssertionError
                                                                        (cf_q, _) = syndrome_conditional_fidelity_exact(vp.PHI_DEC, Q1)
                                                                        (cf_a, w_a) = syndrome_conditional_fidelity(_Probe(vp.PHI_DEC), noise, p = p, M = 1500, rng_seed = SEED + 5)
                                                                        (cf_b, _) = syndrome_conditional_fidelity(_Probe(vp.PHI_DEC), noise, p = p, M = 1500, rng_seed = SEED + 91)
                                                                        keep = w_a > 0.001
                                                                        dev = float(max(np.abs(cf_q[keep] - cf_a[keep]).max(), np.abs(Q1['p_syn'].numpy()[keep] - w_a[keep]).max()))
                                                                        spread = float(np.abs(cf_a - cf_b)[keep].max())
                                                                        if not dev < 6 * spread + 1e-12:
                                                                            raise (noise, dev, spread)()
                                                                        print(f'''[quad selftest 6] {noise:20s} p={p}: exact cf_s matches MC(1500) to {dev:.2e} (MC run-to-run spread {spread:.2e}, {int(keep.sum())}/16 syndromes with p_s>1e-3; p_syn exact to {np.abs(ps_def - Q1['p_syn'].numpy()).max():.1e}, sums to {float(Q1['p_syn'].sum()):.12f})  VERIFIED''', flush = True)
                                                                        return True

            
            def train_vscr(model, n_epochs, batch, p_range, noise, lr, lam, rng_seed, use_real_grad, verbose, synd_w, quad = (400, 48, (0.02, 0.15), 'depolarizing', 0.01, 0.1, SEED, True, True, None, False)):
                '''Train VSCR.  If use_real_grad=False, parameters get RANDOM Gaussian
    updates (replicating the broken ACE-QEC reference in main.py) to ablate
    the importance of genuine gradients.  synd_w: optional per-syndrome loss
    weights (gradient balancing; see vscr_fidelity).

    quad=True replaces the Monte-Carlo probe-state average by the EXACT
    deterministic Haar/noise-strength quadrature of `haar_quadrature` /
    `p_quadrature` (see the comment block above those functions), evaluated by
    `vscr_fidelity_quad`.  Two reasons this matters for Fig.3/Fig.4:

      * Accuracy.  The signal those figures are meant to resolve -- the
        non-Pauli headroom of the optimal syndrome-conditioned unitary over the
        rigid Pauli decoder -- is 1.9e-4 (amplitude damping, p=0.06) to 3.4e-3
        (coherent / depolarizing).  The measured per-sample spread of the
        recovery fidelity gives a 48-sample Monte-Carlo SEM of 1.1e-4 at
        amplitude damping p=0.10 and 3.2e-5 at p=0.05, i.e. the estimator noise
        is the same order as the effect for exactly the channel where the effect
        is smallest.  The quadrature gradient is zero-variance and zero-bias.
      * Speed.  The 126 (psi, rho) pairs are cached once per call, and because
        <psi|R_s P_s rho P_s R_s^dag|psi> only ever probes the 2-dimensional
        syndrome subspace, the 60-gate ansatz is applied to 2 columns instead of
        a full 32x32 unitary (see `vscr_fidelity_quad`).

    The default remains False so that pre-existing Monte-Carlo runs reproduce
    bit-for-bit.

    The p = 0.06 model-selection criterion (`best_val`) is evaluated with the
    exact quadrature in BOTH modes, so MC and quad runs are selected by the
    same deterministic, label-free number.
    '''
                rng = np.random.RandomState(rng_seed)
                opt = torch.optim.Adam(model.parameters(), lr = lr) if use_real_grad else None
                history = {
                    'epoch': [],
                    'fidelity': [],
                    'loss': [] }
                val_p = 0.06
                QV = quad_cache((val_p, val_p), noise, n_p = 1)
                for epoch in range(n_epochs):
                    model.train()
                    if use_real_grad:
                        opt.zero_grad()
                        if QT is None:
                            (loss, F, _) = vscr_fidelity_quad(model(), QT, synd_w = synd_w)
                            loss.backward()
                            opt.step()
                            avg_loss = float(loss.detach())
                            tr_f = float(F)
                        else:
                            tot_loss = 0
                            tot_f = 0
                            R_all = recovery_unitary_batch(model())
                            for _ in range(batch):
                                psi = random_logical_state(1, rng)
                                psi_enc = encode(psi)
                                p = rng.uniform(*p_range)
                                rho_n = noisy_state(psi_enc, p, noise)
                                (loss, F) = vscr_fidelity(psi_enc, rho_n, R_all, lam = lam, synd_w = synd_w)
                                tot_loss = tot_loss + loss
                                tot_f = tot_f + F
                                tot_loss = tot_loss / batch
                                tot_loss.backward()
                                opt.step()
                                avg_loss = float(tot_loss.detach())
                                tr_f = float(tot_f / batch)
                            torch.no_grad()
                            for p_ in model.parameters():
                                torch.tensor(None(rng.randn(*p_.shape) * 0.001, dtype = p_.dtype))
                                None(None, None)
                            with None:
                                if not p_.add_:
                                    pass
                            None if quad else (lambda .0 = None: [ random_logical_state(1, rng) for _ in .0 ])
                            avg_loss = float('nan')
                    model.eval()
                    torch.no_grad()
                    (_, vf_t, _) = vscr_fidelity_quad(model(), QV)
                    vf = float(vf_t)
                    None(None, None)
                with None:
                    if not None:
                        pass
                history['epoch'].append(epoch)
                history['fidelity'].append(vf)
                history['loss'].append(avg_loss)
                if verbose and (epoch + 1) % 50 == 0:
                    print(f'''    epoch {epoch + 1:3d}/{n_epochs}: val_fid={vf:.4f}''')
                continue
                return history

            
            def syndrome_conditional_fidelity(model, noise, p, M, rng_seed = (0.07, 150, SEED + 5)):
                '''LABEL-FREE per-syndrome diagnostic: the mean recovery fidelity
    CONDITIONED on each syndrome occurring,

        cf_s = E[ <psi| R_s P_s rho P_s R_s^dag |psi> ] / E[ Tr[P_s rho] ].

    A healthy instrument has cf_s close to 1 for every syndrome that occurs
    with non-negligible probability.  Used to detect the occasional syndrome
    that falls into a bad local optimum and to select the best seed.  No
    error labels or decoder table are involved.'''
                torch.no_grad()
                R_all = recovery_unitary_batch(model())
                None(None, None)

            VSCR_SCHEDULES = {
                'depolarizing': [
                    (600, 0.03, (0.02, 0.08)),
                    (400, 0.03, (0.02, 0.15)),
                    (1200, 0.001, (0.02, 0.15))],
                'amplitude_damping': [
                    (800, 0.03, (0.01, 0.12)),
                    (1200, 0.001, (0.01, 0.12))],
                'mixed': [
                    (600, 0.03, (0.02, 0.08)),
                    (400, 0.03, (0.02, 0.12)),
                    (1200, 0.001, (0.02, 0.12))] }
            
            def train_vscr_best(noise, seeds, schedule, batch, lam, verbose, val_gate = ((1234, 2024, 777), None, 48, 0, True, 0.85)):
                '''Train VSCR for several seeds and keep the best model.

    Selection is LABEL-FREE: candidates that actually trained (validation
    fidelity >= val_gate) are ranked by the worst per-syndrome conditional
    fidelity (tie-break: validation fidelity); if no candidate passes the
    gate, the highest validation fidelity is used.  Rationale: training
    occasionally leaves ONE syndrome in a bad local optimum (which syndrome
    depends on the seed); with >=3 seeds, a candidate that is healthy on ALL
    syndromes is available.  The gate rejects degenerate runs whose cf values
    are uniformly mediocre (flat), which would otherwise inflate min_cf.'''
                if not schedule:
                    pass
                schedule = VSCR_SCHEDULES[noise]
                trs = []
                infos = []
                for seed in seeds:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    model = VSCR()
                    hist_all = {
                        'epoch': [],
                        'fidelity': [],
                        'loss': [] }
                    off = 0
                    for ep, lr, pr in schedule:
                        hist = train_vscr(model, n_epochs = ep, batch = batch, p_range = pr, noise = noise, lr = lr, lam = lam, rng_seed = seed, use_real_grad = True, verbose = False)
                        off += ep = None
                        (cf, w) = syndrome_conditional_fidelity(model, noise)
                        mask = w > 0.001
                        min_cf = float(np.where(mask, cf, 10).min())
                        val_fid = hist_all['fidelity'][-1]
                        infos.append({
                            'seed': seed,
                            'min_cf': min_cf,
                            'val_fid': val_fid })
                        if verbose:
                            print(f'''    seed {seed}: val_fid={val_fid:.4f}  min syndrome cf={min_cf:.4f}''')
                    trs.append((min_cf, val_fid, seed, model, hist_all))
                    pool = trs()
                    if not pool:
                        pool = [
                            max(trs, key = (lambda t: t[1]))]
                best = max(pool, key = (lambda t: (t[0], t[1])))
                return (best[3], best[4], infos)

            METHOD_COLORS = {
                'Raw': '#9e9e9e',
                'Perfect-code': '#e41a1c',
                'ZNE': '#ff7f00',
                'Virtual Distillation': '#984ea3',
                'LinDR': '#4daf4a',
                'VSCR (ours)': '#377eb8',
                'VSCR-RandGrad': '#a65628' }
            
            def evaluate_methods(model, A_lindr, p_values, noise, n_test, seed = (40, SEED + 7)):
                """Return dict: method -> {'F': list, 'LER': list}."""
                rng = np.random.RandomState(seed)
                test_states = range(n_test)()
                names = [
                    'Raw',
                    'Perfect-code',
                    'ZNE',
                    'Virtual Distillation',
                    'LinDR',
                    'VSCR (ours)']
                results = names()
                torch.no_grad()
                R_all = recovery_unitary_batch(model())
                None(None, None)

            
            def plot_curves(p_values, results, title, fname, ylabel = ('Avg. fidelity',)):
                (fig, ax) = plt.subplots(figsize = (7.2, 5))
                for m, d in results.items():
                    ax.plot(p_values, d['F'], 'o-', color = METHOD_COLORS.get(m, 'k'), label = m, linewidth = 2, markersize = 6)
                    ax.set_xlabel('Physical error rate $p$', fontsize = 12)
                    ax.set_ylabel(ylabel, fontsize = 12)
                    ax.set_title(title, fontsize = 13)
                    ax.set_ylim(-0.05, 1.05)
                    ax.grid(True, alpha = 0.3)
                    ax.legend(fontsize = 9, loc = 'best')
                    plt.tight_layout()
                    plt.savefig(fname, dpi = 150)
                    plt.close()
                    return None

            
            def plot_ler(p_values, results, title, fname):
                (fig, ax) = plt.subplots(figsize = (7.2, 5))
                for m, d in results.items():
                    ax.plot(p_values, d['LER'], 's-', color = METHOD_COLORS.get(m, 'k'), label = m, linewidth = 2, markersize = 6)
                    ax.set_xlabel('Physical error rate $p$', fontsize = 12)
                    ax.set_ylabel('Logical error rate (F<0.9)', fontsize = 12)
                    ax.set_title(title, fontsize = 13)
                    ax.set_ylim(-0.02, 1.02)
                    ax.grid(True, alpha = 0.3)
                    ax.legend(fontsize = 9, loc = 'best')
                    plt.tight_layout()
                    plt.savefig(fname, dpi = 150)
                    plt.close()
                    return None

            
            def plot_training(hist_real, hist_rand, fname):
                (fig, ax) = plt.subplots(figsize = (7.2, 5))
                ax.plot(hist_real['epoch'], hist_real['fidelity'], 'b-', linewidth = 2, label = 'VSCR (real Adam gradients)')
                ax.plot(hist_rand['epoch'], hist_rand['fidelity'], 'r--', linewidth = 2, label = 'ACE-style random update (broken ref.)')
                ax.set_xlabel('Epoch', fontsize = 12)
                ax.set_ylabel('Validation fidelity at $p=0.06$', fontsize = 12)
                ax.set_title('Training dynamics: real gradients vs random walk', fontsize = 13)
                ax.grid(True, alpha = 0.3)
                ax.legend(fontsize = 10, loc = 'best')
                plt.tight_layout()
                plt.savefig(fname, dpi = 150)
                plt.close()

            
            def plot_summary_bar(p_values, results, p_idx, fname, noise_label):
                p = p_values[p_idx]
                methods = list(results.keys())
                vals = methods()
                colors = methods()
                (fig, ax) = plt.subplots(figsize = (8.5, 5))
                bars = ax.bar(methods, vals, color = colors)
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'''{v:.3f}''', ha = 'center', fontsize = 9)
                    ax.set_ylabel('Avg. fidelity', fontsize = 12)
                    ax.set_title(f'''Fidelity at $p={p:.3f}$ ({noise_label})''', fontsize = 13)
                    ax.set_ylim(0, 1.08)
                    ax.grid(True, alpha = 0.3, axis = 'y')
                    plt.xticks(rotation = 15, fontsize = 9)
                    plt.tight_layout()
                    plt.savefig(fname, dpi = 150)
                    plt.close()
                    return None

            
            def main():
                t0 = time.time()
                print('========================================================================')
                print('VSCR-QEC : Variational Syndrome-Conditioned Recovery (variational')
                print('           quantum-classical ML for quantum error correction)')
                print('========================================================================')
                print(f'''Code: [[{N},1,3]] perfect code  (Hilbert dim {DIM})''')
                print('Recovery: projective syndrome measurement + LEARNED variational unitary')
                print(f'''          R_s per syndrome ({PHI_DIM} rotation params, hypernetwork-shared)''')
                print('\n[1] Training VSCR (depolarizing): curriculum + 3 seeds,')
                print('    label-free per-syndrome selection...')
                (model_dep, hist_real, infos_dep) = train_vscr_best('depolarizing')
                print('\n[1b] Ablation: random-walk updates (replicates broken ACE-QEC)...')
                torch.manual_seed(SEED)
                np.random.seed(SEED)
                model_rand = VSCR()
                hist_rand = train_vscr(model_rand, n_epochs = 2200, batch = 48, p_range = (0.02, 0.15), noise = 'depolarizing', lr = 0.01, lam = 0, use_real_grad = False, verbose = False)
                os.makedirs('figures', exist_ok = True)
                plot_training(hist_real, hist_rand, 'figures/fig_training_curves.png')
                print('\n[2] Fitting LinDR (vnCDR-style linear) baseline...')
                A_dep = train_lindr(n_train = 1500, p_range = (0.01, 0.12), noise = 'depolarizing')
                p_values = np.array([
                    0.005,
                    0.01,
                    0.02,
                    0.04,
                    0.07,
                    0.1,
                    0.15,
                    0.2])
                print('\n[3] Evaluating all methods (depolarizing)...')
                res_dep = evaluate_methods(model_dep, A_dep, p_values, 'depolarizing', n_test = 40)
                plot_curves(p_values, res_dep, 'Depolarizing noise: recovery fidelity', 'figures/fig_fidelity_depolarizing.png')
                plot_ler(p_values, res_dep, 'Depolarizing noise: logical error rate', 'figures/fig_ler_depolarizing.png')
                plot_summary_bar(p_values, res_dep, 5, 'figures/fig_summary_depolarizing_p010.png', 'depolarizing')
                print('\n[4] Training VSCR (amplitude damping): 3 seeds + selection...')
                (model_ad, hist_ad, infos_ad) = train_vscr_best('amplitude_damping')
                A_ad = train_lindr(n_train = 1500, p_range = (0.01, 0.08), noise = 'amplitude_damping')
                res_ad = evaluate_methods(model_ad, A_ad, p_values, 'amplitude_damping', n_test = 40)
                plot_curves(p_values, res_ad, 'Amplitude-damping noise: recovery fidelity', 'figures/fig_fidelity_amplitude_damping.png')
                plot_summary_bar(p_values, res_ad, 5, 'figures/fig_summary_ad_p010.png', 'amplitude-damping')
                print('\n[5] Training VSCR (mixed depol+AD): 3 seeds + selection...')
                (model_mx, hist_mx, infos_mx) = train_vscr_best('mixed')
                A_mx = train_lindr(n_train = 1500, p_range = (0.01, 0.1), noise = 'mixed')
                res_mx = evaluate_methods(model_mx, A_mx, p_values, 'mixed', n_test = 40)
                plot_curves(p_values, res_mx, 'Mixed (depol+AD) noise: recovery fidelity', 'figures/fig_fidelity_mixed.png')
                plot_summary_bar(p_values, res_mx, 5, 'figures/fig_summary_mixed_p010.png', 'mixed')
                _print_and_save(res_dep, res_ad, res_mx, p_values, hist_real, hist_rand, t0)
                return (res_dep, res_ad, res_mx, hist_real, hist_rand)

            
            def _print_and_save(res_dep, res_ad, res_mx, p_values, hist_real, hist_rand, t0):
                print('\n========================================================================')
                print('RESULTS  (avg fidelity at each p)')
                print('========================================================================')
                for None in (('DEPOLARIZING', res_dep), ('AMPLITUDE-DAMPING', res_ad), ('MIXED', res_mx)):
                    (label, res) = None
                    'p        '('  '.join + (lambda .0: Nonefor m in .0:
f'''{m[:9]:>10s}'''None)(res()))
                    for i, p in enumerate(p_values):
                        None(None + (lambda .0 = None: Nonefor m in .0:
f'''{res[m]['F'][i]:10.4f}'''None)(res()))
                        print('\nImprovement of VSCR over baselines (depolarizing, p=0.10):')
                        i = list(p_values).index(0.1)
                        base = res_dep['VSCR (ours)']['F'][i]
                        for m in ('Raw', 'Perfect-code', 'ZNE', 'Virtual Distillation', 'LinDR'):
                            b = res_dep[m]['F'][i]
                            print(f'''  vs {m:22s}: {b:.4f} -> {base:.4f}  (abs {base - b:+.4f}, rel {100 * (base - b) / max(b, 1e-09):+.1f}%)''')
                            print('\nImprovement of VSCR over baselines (amplitude-damping, p=0.10):')
                            base = res_ad['VSCR (ours)']['F'][i]
                            for m in ('Raw', 'Perfect-code', 'ZNE', 'Virtual Distillation', 'LinDR'):
                                b = res_ad[m]['F'][i]
                                print(f'''  vs {m:22s}: {b:.4f} -> {base:.4f}  (abs {base - b:+.4f}, rel {100 * (base - b) / max(b, 1e-09):+.1f}%)''')
                                print(f'''\nFinal training fidelity (real grad): {hist_real['fidelity'][-1]:.4f}''')
                                print(f'''Final training fidelity (random walk): {hist_rand['fidelity'][-1]:.4f}''')
                                print(f'''Total runtime: {time.time() - t0:.1f}s''')
                                None(None, p_values = None, dep_F = None, ad_F = None, mx_F = (lambda .0 = None: pass# WARNING: Decompyle incomplete
), dep_LER = res_dep(), hist_real_fid = np.array(hist_real['fidelity']), hist_rand_fid = np.array(hist_rand['fidelity']))
                                print('\nFigures saved in figures/: fig_training_curves.png, fig_fidelity_depolarizing.png,')
                                print('  fig_ler_depolarizing.png, fig_summary_depolarizing_p010.png,')
                                print('  fig_fidelity_amplitude_damping.png, fig_summary_ad_p010.png,')
                                print('  fig_fidelity_mixed.png, fig_summary_mixed_p010.png')
                                print('Numerical data: vscr_results.npz')
                                return None

            if __name__ == '__main__':
                main()
                return None
            return (lambda .0: [ pauli_string(s) for s in .0 ])
