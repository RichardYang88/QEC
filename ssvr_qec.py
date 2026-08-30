"""
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
"""

import math, os, time, warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

SEED = 1234
torch.manual_seed(SEED); np.random.seed(SEED)
DTYPE = torch.complex128
N = 5                 # physical qubits
DIM = 2 ** N          # 32
LOG_DIM = N

# ---------------------------------------------------------------------
# Primitive 2x2 / 4x4 gates (differentiable in angle tensors)
# ---------------------------------------------------------------------
I2 = torch.eye(2, dtype=DTYPE)
Xp = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
Yp = torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE)
Zp = torch.tensor([[1, 0], [0, -1]], dtype=DTYPE)
Hp = (1 / math.sqrt(2)) * torch.tensor([[1, 1], [1, -1]], dtype=DTYPE)


def Rz(t):
    """Rz(t) = diag(e^{-it/2}, e^{it/2}).  t: real scalar tensor."""
    return torch.diag(torch.stack([(-1j * t / 2).exp(), (1j * t / 2).exp()]))


def Rx(t):
    c = (t / 2).cos(); s = (t / 2).sin()
    return c * I2 - 1j * s * Xp


def Ry(t):
    c = (t / 2).cos(); s = (t / 2).sin()
    return c * I2 - 1j * s * Yp


def Rzz(t):
    """exp(-i t/2 Z⊗Z) = diag(e^{-it/2}, e^{it/2}, e^{it/2}, e^{-it/2})."""
    e = (-1j * t / 2).exp(); ep = (1j * t / 2).exp()
    return torch.diag(torch.stack([e, ep, ep, e]))


CNOT = torch.tensor([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=DTYPE)


# ---------------------------------------------------------------------
# Embed a k-qubit gate into the full 2^n Hilbert space.
# Convention: qubit 0 is the MOST significant bit (leftmost in kron).
# ---------------------------------------------------------------------
def embed_gate(U, targets, n=N):
    targets = sorted(targets)
    k = len(targets)
    others = [q for q in range(n) if q not in targets]
    No = len(others)
    D = torch.eye(2 ** No, dtype=DTYPE).reshape([2] * No + [2] * No)
    Ur = U.reshape([2] * k + [2] * k)
    # tensordot(dims=0) == outer product: axes = [o_out(No), o_in(No), t_out(k), t_in(k)]
    G = torch.tensordot(D, Ur, dims=0)
    other_pos = {q: i for i, q in enumerate(others)}
    targ_pos = {q: i for i, q in enumerate(targets)}
    perm_out, perm_in = [], []
    for q in range(n):
        if q in other_pos:
            perm_out.append(other_pos[q])           # o_out
            perm_in.append(No + other_pos[q])       # o_in
        else:
            perm_out.append(2 * No + targ_pos[q])    # t_out
            perm_in.append(2 * No + k + targ_pos[q]) # t_in
    G = G.permute(perm_out + perm_in)
    return G.reshape(2 ** n, 2 ** n)


def pauli_string(string, n=N):
    """Build 2^n x 2^n Pauli from a string like 'XZZXI' (q0 = leftmost = MSB)."""
    table = {'I': I2, 'X': Xp, 'Y': Yp, 'Z': Zp}
    G = table[string[0]]
    for ch in string[1:]:
        G = torch.kron(G, table[ch])
    return G


def _selftest_gates():
    """Sanity checks for embed_gate / pauli_string conventions."""
    # X on qubit 0 (MSB) of 2 qubits  ==  X ⊗ I
    G = embed_gate(Xp, [0], n=2)
    assert torch.allclose(G, torch.kron(Xp, I2)), "embed q0 failed"
    # X on qubit 1  ==  I ⊗ X
    G = embed_gate(Xp, [1], n=2)
    assert torch.allclose(G, torch.kron(I2, Xp)), "embed q1 failed"
    # CNOT(0->1) on 2 qubits
    G = embed_gate(CNOT, [0, 1], n=2)
    assert torch.allclose(G, CNOT), "embed CNOT failed"
    # pauli_string 'XI' on 2 qubits == X ⊗ I
    assert torch.allclose(pauli_string('XI', n=2), torch.kron(Xp, I2)), "pauli_string failed"
    # consistency: embed X on q0 n=2 == pauli_string 'XI'
    assert torch.allclose(embed_gate(Xp, [0], n=2), pauli_string('XI', n=2))


_selftest_gates()

# ---------------------------------------------------------------------
# The [[5,1,3]] perfect code
#   Stabilizers (Laflamme-Miquel-Paz 1996 / DiVincenzo-Shor):
#       g1 = X Z Z X I
#       g2 = I X Z Z X
#       g3 = X I X Z Z
#       g4 = Z X I X Z
#   Logical operators:  X_L = XXXXX ,  Z_L = ZZZZZ
#   Distance 3 -> corrects any single-qubit error (15 non-trivial Paulis).
# ---------------------------------------------------------------------
STAB_STR = ['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']
STAB = [pauli_string(s) for s in STAB_STR]
XL = pauli_string('XXXXX')
ZL = pauli_string('ZZZZZ')
I_full = torch.eye(DIM, dtype=DTYPE)

# code-space projector  P_code = prod_i (I + g_i)/2   (rank 2)
P_code = I_full
for g in STAB:
    P_code = P_code @ (0.5 * (I_full + g))
# symmetrise away numerical noise
P_code = 0.5 * (P_code + P_code.conj().T)
_rank = torch.linalg.matrix_rank(P_code.real.double())
assert int(_rank) == 2, f"code-space rank must be 2, got {_rank}"

# mutual commutation of stabilisers
for i in range(4):
    for j in range(i + 1, 4):
        assert torch.allclose(STAB[i] @ STAB[j], STAB[j] @ STAB[i]), "stabilisers do not commute"


def _logical_basis():
    """Return |0_L>, |1_L> as 32-dim complex vectors (columns of E_code).

    Convention: |0_L> is the +1 eigenstate of Z_L inside the code space and
    |1_L> = X_L |0_L>  (so X_L flips 0<->1 exactly, removing global-phase
    ambiguity that would otherwise break an equality assertion)."""
    M = P_code @ ZL @ P_code
    Mh = 0.5 * (M + M.conj().T)
    evals, evecs = torch.linalg.eigh(Mh)
    zeroL = evecs[:, evals.argmax()]               # +1 eigenstate of Z_L
    j = zeroL.abs().argmax()                      # fix global phase
    zeroL = zeroL * torch.exp(-1j * torch.angle(zeroL[j]))
    oneL = XL @ zeroL                             # unitary -> normalized; flips Z_L sign
    # checks
    assert torch.allclose(ZL @ zeroL, zeroL, atol=1e-6), "|0_L> not +1 of Z_L"
    assert torch.allclose(ZL @ oneL, -oneL, atol=1e-6), "|1_L> not -1 of Z_L"
    for g in STAB:
        assert torch.allclose(g @ zeroL, zeroL, atol=1e-6), "stab not +1 on |0_L>"
        assert torch.allclose(g @ oneL, oneL, atol=1e-6), "stab not +1 on |1_L>"
    assert torch.allclose(XL @ zeroL, oneL, atol=1e-6), "X_L does not flip 0->1"
    E = torch.stack([zeroL, oneL], dim=1)          # (32,2) isometry
    assert torch.allclose(E.conj().T @ E, I2, atol=1e-6), "E not isometry"
    return E


E_CODE = _logical_basis()          # (32,2)  columns |0_L>, |1_L>
ZERO_L = E_CODE[:, 0]
ONE_L = E_CODE[:, 1]


def encode(psi):
    """psi: length-2 complex tensor (logical state) -> 32-dim encoded vector."""
    return E_CODE @ psi.to(DTYPE)


def fidelity(psi_enc, rho):
    """Fidelity of mixed state rho w.r.t. pure |psi_enc> in the 32-dim space."""
    return (psi_enc.conj() @ rho @ psi_enc).real


# ---------------------------------------------------------------------
# Syndrome table for the [[5,1,3]] code (perfect: 15 distinct non-zero
# syndromes, one per single-qubit Pauli error).
#   syndrome bit i = 1  iff  {g_i, E} != 0  (E anticommutes with g_i)
# ---------------------------------------------------------------------
def _anticommute(P_full, g):
    """True if P_full and g anticommute."""
    return not torch.allclose(P_full @ g, g @ P_full)


def _syndrome_table():
    table = { (0, 0, 0, 0): 'I' * N }     # no error -> identity
    for q in range(N):
        for name, Pl in [('X', Xp), ('Y', Yp), ('Z', Zp)]:
            targets = ['I'] * N
            targets[q] = name
            E = pauli_string(''.join(targets))
            bits = tuple(int(_anticommute(E, g)) for g in STAB)
            assert bits != (0, 0, 0, 0), "single error has zero syndrome?!"
            assert bits not in table, f"syndrome clash -> not a perfect code ({bits})"
            table[bits] = ''.join(targets)
    assert len(table) == 16, f"need 16 syndromes, got {len(table)}"
    return table


SYND_TABLE = _syndrome_table()
SYND_BITS = [((s >> 3) & 1, (s >> 2) & 1, (s >> 1) & 1, s & 1) for s in range(16)]


def _synd_projectors():
    """16 syndrome projectors  P_s = prod_i (I + (-1)^{s_i} g_i)/2  (rank 2 each)."""
    out = []
    for bits in SYND_BITS:
        P = I_full
        for i, b in enumerate(bits):
            P = P @ (0.5 * (I_full + ((-1) ** b) * STAB[i]))
        out.append(P)
    return out


P_SYNDS = _synd_projectors()
P_SYNDS_STACK = torch.stack(P_SYNDS)          # (16, 32, 32) constant
C_SYNDS = [pauli_string(SYND_TABLE[b]) for b in SYND_BITS]   # Pauli corrections (decoder)


# ---------------------------------------------------------------------
# Noise channels (exact, differentiable, density-matrix level)
# ---------------------------------------------------------------------
# Pre-embed per-qubit Paulis and amplitude-damping Kraus operators
Xq = [embed_gate(Xp, [q]) for q in range(N)]
Yq = [embed_gate(Yp, [q]) for q in range(N)]
Zq = [embed_gate(Zp, [q]) for q in range(N)]


def apply_depolarizing(rho, p):
    """Independent single-qubit depolarizing of rate p on every qubit."""
    for q in range(N):
        rho = (1 - p) * rho + (p / 3.0) * (
            Xq[q] @ rho @ Xq[q] + Yq[q] @ rho @ Yq[q] + Zq[q] @ rho @ Zq[q])
    return rho


def amp_kraus(gamma, q):
    """Two Kraus ops for amplitude damping on qubit q (rate gamma)."""
    K0_2 = torch.tensor([[1.0, 0.0], [0.0, math.sqrt(1 - gamma)]], dtype=DTYPE)
    K1_2 = torch.tensor([[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=DTYPE)
    return embed_gate(K0_2, [q]), embed_gate(K1_2, [q])


def apply_amplitude_damping(rho, gamma):
    for q in range(N):
        K0, K1 = amp_kraus(gamma, q)
        rho = K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T
    return rho


def apply_mixed(rho, p, gamma):
    rho = apply_depolarizing(rho, p)
    rho = apply_amplitude_damping(rho, gamma)
    return rho


def noisy_state(psi_enc, p, noise='depolarizing'):
    """Build the noisy density matrix from a pure encoded state."""
    rho = torch.outer(psi_enc, psi_enc.conj())
    if noise == 'depolarizing':
        rho = apply_depolarizing(rho, p)
    elif noise == 'amplitude_damping':
        rho = apply_amplitude_damping(rho, p)
    elif noise == 'mixed':
        rho = apply_mixed(rho, p, 0.5 * p)
    else:
        raise ValueError(noise)
    return rho

# ---------------------------------------------------------------------
# Variational recovery unitary R(phi)  (differentiable, fast)
#   per layer:  Rz-Rx-Rz on every qubit  +  Rzz on a ring (q -> q+1 mod N)
#   Each gate = exp(-i theta/2 P) = cos(theta/2) I - i sin(theta/2) P, so we
#   precompute the full-space Pauli P (and the constant -iP) for every slot;
#   at runtime a gate is just two scalars + a rank-2 combination of two
#   precomputed matrices (no tensordot / permute / reshape -> ~10x faster).
# ---------------------------------------------------------------------
N_LAYERS = 3
PARAMS_PER_LAYER = 3 * N + N                 # 15 + 5 = 20
PHI_DIM = PARAMS_PER_LAYER * N_LAYERS         # 60


def _build_recovery_slots(n=N, n_layers=N_LAYERS):
    slots = []
    for _ in range(n_layers):
        for q in range(n):
            slots += [Zq[q], Xq[q], Zq[q]]                 # Rz, Rx, Rz
        for q in range(n):
            b = (q + 1) % n
            a, b2 = (q, b) if q < b else (b, q)
            slots.append(Zq[a] @ Zq[b2])                    # Rzz ~ Z_a Z_b
    return slots


P_SLOTS = _build_recovery_slots()                     # list of full Paulis (constant)
MP_SLOTS = [(-1j * P).contiguous() for P in P_SLOTS]  # -iP per slot (constant)


def recovery_unitary(phi):
    """phi: real tensor of length PHI_DIM -> 32x32 complex unitary R."""
    R = I_full.clone()
    half = phi / 2
    for k in range(len(P_SLOTS)):
        G = torch.cos(half[k]) * I_full + torch.sin(half[k]) * MP_SLOTS[k]
        R = G @ R
    return R


# ---------------------------------------------------------------------
# Classical feature vector (soft syndrome + logical/single-qubit signatures)
#   features = Re Tr[O rho_noisy]  for O in
#     {g1..g4, X_L, Z_L, Z_0..Z_4}   -> 11 features
# ---------------------------------------------------------------------
OBS_LIST = STAB + [XL, ZL] + [embed_gate(Zp, [q]) for q in range(N)]
N_FEAT = len(OBS_LIST)                        # 11
OBS_STACK = torch.stack(OBS_LIST)             # (N_FEAT, 32, 32) complex, constant


def features_of(rho):
    """rho: 32x32 complex -> float64 feature tensor (vectorised, no grad)."""
    return torch.einsum('kij,ji->k', OBS_STACK, rho).real


# ---------------------------------------------------------------------
# VSCR model : Variational Syndrome-Conditioned Recovery
#   A parameterised quantum INSTRUMENT: projective stabiliser syndrome
#   measurement (projector P_s) FOLLOWED BY a LEARNED variational unitary
#   R_s, one per syndrome, shared through a classical hypernetwork
#   (syndrome embedding -> phi).  Trained UNSUPERVISED end-to-end on
#   recovery fidelity (no error labels).  Unlike a rigid Pauli-lookup
#   decoder, R_s can be ANY unitary -> learns non-Pauli corrections
#   (essential for amplitude damping / mixed noise) and partial multi-error
#   recovery, while keeping the projection (purification) power of a
#   measurement-based decoder.
# ---------------------------------------------------------------------
def recovery_unitary_batch(phi_batch):
    """phi_batch: (B, PHI_DIM) real -> (B, 32, 32) complex unitaries (batched)."""
    B = phi_batch.shape[0]
    R = I_full.unsqueeze(0).expand(B, DIM, DIM).clone()
    half = phi_batch / 2
    for k in range(len(P_SLOTS)):
        c = torch.cos(half[:, k]).view(B, 1, 1)
        s = torch.sin(half[:, k]).view(B, 1, 1)
        G = c * I_full.unsqueeze(0) + s * MP_SLOTS[k].unsqueeze(0)
        R = torch.bmm(G, R)
    return R


class VSCR(nn.Module):
    def __init__(self, n_synd=16, phi_dim=PHI_DIM, hidden=64, ctx=24):
        super().__init__()
        self.synd_emb = nn.Parameter(torch.randn(n_synd, ctx, dtype=torch.float64) * 0.1)
        self.hnet = nn.Sequential(
            nn.Linear(ctx, hidden, dtype=torch.float64), nn.Tanh(),
            nn.Linear(hidden, phi_dim, dtype=torch.float64))
        self.phi_base = nn.Parameter(torch.randn(phi_dim, dtype=torch.float64) * 0.05)

    def forward(self):
        # phi for all 16 syndromes: (16, PHI_DIM), requires grad
        return self.phi_base + self.hnet(self.synd_emb)


def vscr_fidelity(psi_enc, rho_noisy, R_all, lam=0.0, synd_w=None):
    """Expected recovery fidelity given the 16 recovery unitaries R_all:

        F = sum_s <psi| R_s P_s rho P_s R_s^dag |psi>     (exact expectation
        over the projective syndrome outcome).  Vectorised over 16 syndromes.
        R_all can be SHARED across samples (built once per epoch/eval).

        synd_w: optional length-16 weight vector rescaling the per-syndrome
        terms (training-time gradient balancing; the optimum is unchanged
        because each R_s only appears in its own syndrome term)."""
    rho16 = rho_noisy.unsqueeze(0).expand(16, DIM, DIM)
    PR = torch.bmm(P_SYNDS_STACK, rho16)         # P_s rho
    PRP = torch.bmm(PR, P_SYNDS_STACK)           # P_s rho P_s
    RPR = torch.bmm(R_all, PRP)                  # R_s P_s rho P_s
    RPRd = torch.bmm(RPR, R_all.conj().transpose(1, 2))     # ... R_s^dag
    pc = psi_enc.conj()
    F_s = torch.einsum('i,sij,j->s', pc, RPRd, psi_enc).real
    F = F_s.sum() if synd_w is None else (synd_w * F_s).sum()
    if lam == 0.0:
        return (1.0 - F), F
    pop_s = torch.einsum('ij,sji->s', P_code, RPRd).real
    code_pop = pop_s.sum() if synd_w is None else (synd_w * pop_s).sum()
    loss = (1.0 - F) + lam * (1.0 - code_pop)
    return loss, F


def vscr_forward(model, psi_enc, rho_noisy, lam=0.1):
    """Convenience wrapper: build R_all from the model, then score."""
    R_all = recovery_unitary_batch(model())
    return vscr_fidelity(psi_enc, rho_noisy, R_all, lam=lam)

# =====================================================================
# BASELINES
# =====================================================================
def baseline_raw(psi_enc, rho_noisy):
    return fidelity(psi_enc, rho_noisy)


def perfect_code_decoder_fidelity(psi_enc, rho_noisy):
    """Exact expected fidelity of the optimal single-error lookup decoder:

        F = sum_s <psi| C_s P_s rho P_s C_s^dag |psi>     with P_s the
    syndrome-s projector (rank 2) and C_s the rigid Pauli correction.
    Uses precomputed P_SYNDS / C_SYNDS."""
    val = torch.zeros((), dtype=torch.float64)
    for s in range(16):
        block = C_SYNDS[s] @ (P_SYNDS[s] @ rho_noisy @ P_SYNDS[s]) @ C_SYNDS[s].conj().T
        val = val + fidelity(psi_enc, block)
    return val


def zne_fidelity(psi_enc, p, noise):
    """Richardson ZNE at noise scales 1,2,3 (state-level extrapolation)."""
    r1 = noisy_state(psi_enc, p * 1.0, noise)
    r2 = noisy_state(psi_enc, p * 2.0, noise)
    r3 = noisy_state(psi_enc, p * 3.0, noise)
    rho_zne = 3.0 * r1 - 3.0 * r2 + r3                  # Richardson for {1,2,3}
    return fidelity(psi_enc, rho_zne)


def virtual_distillation_fidelity(psi_enc, rho_noisy, k=2):
    """rho_VD ~ rho^k (unnormalised) projected toward the purest component."""
    r = rho_noisy
    for _ in range(k - 1):
        r = r @ rho_noisy
    rho_vd = 0.5 * (r + r.conj().T)
    trc = torch.trace(rho_vd).real
    if trc.abs() < 1e-12:
        return fidelity(psi_enc, rho_noisy)
    rho_vd = rho_vd / trc
    return fidelity(psi_enc, rho_vd)


# ---------------------------------------------------------------------
# LinDR : vnCDR-style linear data-driven recovery on the FULL Pauli basis.
#   Features = all 4^N Pauli expectations (weight 0..5, = 1024).  A linear
#   map A : noisy-Pauli-features -> ideal-Pauli-features is fit by least
#   squares over a range of physical error rates; the (full) state is
#   reconstructed and its fidelity reported.  This is the faithful linear
#   analogue of PEC / vnCDR with full single-shot tomography: strong on
#   Pauli (depolarizing) noise where the channel is linear, but limited by
#   linearity on non-Pauli (amplitude-damping) noise.
# ---------------------------------------------------------------------
def _pauli_index_set(n=N, max_weight=N):
    import itertools
    idx = []
    for w in range(0, max_weight + 1):
        for qs in itertools.combinations(range(n), w):
            for paulis in itertools.product('XYZ', repeat=w):
                s = ['I'] * n
                for qi, pch in zip(qs, paulis):
                    s[qi] = pch
                idx.append(''.join(s))
    return idx


PAULI_IDX = _pauli_index_set()
N_PAULI = len(PAULI_IDX)
PAULI_MATS = [pauli_string(s) for s in PAULI_IDX]
PAULI_STACK = torch.stack(PAULI_MATS)         # (N_PAULI, 32, 32) constant


def _pauli_features(rho):
    """Vectorised Pauli-expectation feature vector (no grad)."""
    return torch.einsum('kij,ji->k', PAULI_STACK, rho).real


def _reconstruct_state(coeffs):
    """rho = (1/2^n) sum_P c_P P  (c_I forced to 1 by the caller).  Vectorised."""
    return torch.einsum('k,kij->ij', coeffs.to(DTYPE), PAULI_STACK) / INV_DIM


INV_DIM = 2 ** N


def train_lindr(n_train=1500, p_range=(0.01, 0.12), noise='depolarizing',
                n_val=200, rng_seed=SEED + 99):
    """Ridge least-squares fit  A : noisy-Pauli-features -> ideal-Pauli-features.

    The ridge strength is chosen on a held-out validation set by
    reconstruction fidelity.  Returns A with the column convention y = A @ x."""
    rng = np.random.RandomState(rng_seed)

    def gen(m):
        Xs, Ys, PS = [], [], []
        for _ in range(m):
            psi = random_logical_state(1, rng)
            psi_enc = encode(psi)
            p = rng.uniform(*p_range)
            rho_n = noisy_state(psi_enc, p, noise)
            Xs.append(_pauli_features(rho_n).numpy())
            rho_id = torch.outer(psi_enc, psi_enc.conj())
            Ys.append(_pauli_features(rho_id).numpy())
            PS.append(psi_enc)
        return np.stack(Xs), np.stack(Ys), torch.stack(PS)

    X, Y, _ = gen(n_train)
    Vx, _, Vpsi = gen(n_val)
    XtX = X.T @ X
    XtY = X.T @ Y
    best_f, best_B = -1.0, None
    for a in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        B = np.linalg.solve(XtX + (a * n_train + 1e-6) * np.eye(N_PAULI), XtY)  # x@B
        Yh = Vx @ B
        Yh[:, 0] = 1.0
        coeffs = torch.tensor(Yh, dtype=torch.float64).to(DTYPE)
        rho_b = torch.einsum('mk,kij->mij', coeffs, PAULI_STACK) / INV_DIM
        F = torch.einsum('mi,mij,mj->m', Vpsi.conj(), rho_b, Vpsi).real
        if float(F.mean()) > best_f:
            best_f, best_B = float(F.mean()), B
    return torch.tensor(best_B.T.copy(), dtype=torch.float64)   # col conv: y = A@x


def lindr_fidelity(psi_enc, rho_noisy, A):
    x = _pauli_features(rho_noisy)
    y = A @ x
    y = y.clone(); y[0] = 1.0                         # force identity coeff = 1
    rho_lindr = _reconstruct_state(y)
    return fidelity(psi_enc, rho_lindr)

# ---------------------------------------------------------------------
# Random logical state generator (single logical qubit on the Bloch sphere)
# ---------------------------------------------------------------------
def random_logical_state(n=1, rng=None):
    """Uniform-Haar random n-qubit logical pure state as a length-2^n complex tensor."""
    rng = rng or np.random
    d = 2 ** n
    g = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    g = g / np.linalg.norm(g)
    return torch.tensor(g, dtype=DTYPE)


def clifford_logical_states():
    """The 6 single-qubit Pauli eigenstates (|0>,|1>,|+>,|->,|+i>,|-i>)."""
    s = []
    for vec in [np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex),
                np.array([1, 1], dtype=complex) / np.sqrt(2),
                np.array([1, -1], dtype=complex) / np.sqrt(2),
                np.array([1, 1j], dtype=complex) / np.sqrt(2),
                np.array([1, -1j], dtype=complex) / np.sqrt(2)]:
        s.append(torch.tensor(vec, dtype=DTYPE))
    return s


# ---------------------------------------------------------------------
# VSCR training  (real Adam gradients through the differentiable simulator)
# ---------------------------------------------------------------------
def train_vscr(model, n_epochs=400, batch=48, p_range=(0.02, 0.15),
               noise='depolarizing', lr=1e-2, lam=0.1, rng_seed=SEED,
               use_real_grad=True, verbose=True, synd_w=None):
    """Train VSCR.  If use_real_grad=False, parameters get RANDOM Gaussian
    updates (replicating the broken ACE-QEC reference in main.py) to ablate
    the importance of genuine gradients.  synd_w: optional per-syndrome loss
    weights (gradient balancing; see vscr_fidelity)."""
    rng = np.random.RandomState(rng_seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr) if use_real_grad else None
    history = {'epoch': [], 'fidelity': [], 'loss': []}
    val_psi = [random_logical_state(1, rng) for _ in range(30)]
    val_p = 0.06
    for epoch in range(n_epochs):
        model.train()
        if use_real_grad:
            opt.zero_grad()
            tot_loss = 0.0; tot_f = 0.0
            R_all = recovery_unitary_batch(model())    # build ONCE per epoch
            for _ in range(batch):
                psi = random_logical_state(1, rng)
                psi_enc = encode(psi)
                p = rng.uniform(*p_range)
                rho_n = noisy_state(psi_enc, p, noise)
                loss, F = vscr_fidelity(psi_enc, rho_n, R_all, lam=lam,
                                        synd_w=synd_w)
                tot_loss = tot_loss + loss; tot_f = tot_f + F
            tot_loss = tot_loss / batch
            avg_loss = float(tot_loss.detach())
            tot_loss.backward()
            opt.step()
            tr_f = float(tot_f / batch)
        else:
            # BROKEN reference behaviour: random-walk "gradient"
            with torch.no_grad():
                tr_f = 0.0
                R_all = recovery_unitary_batch(model())
                for _ in range(batch):
                    psi = random_logical_state(1, rng)
                    psi_enc = encode(psi)
                    p = rng.uniform(*p_range)
                    rho_n = noisy_state(psi_enc, p, noise)
                    _, F = vscr_fidelity(psi_enc, rho_n, R_all, lam=lam)
                    tr_f += float(F)
                tr_f /= batch
                for p_ in model.parameters():
                    p_.add_(torch.tensor(rng.randn(*p_.shape) * 0.001, dtype=p_.dtype))
            avg_loss = float('nan')
        model.eval()
        vf = 0.0
        with torch.no_grad():
            R_all = recovery_unitary_batch(model())
            for psi in val_psi:
                psi_enc = encode(psi)
                rho_n = noisy_state(psi_enc, val_p, noise)
                _, F = vscr_fidelity(psi_enc, rho_n, R_all, lam=lam)
                vf += float(F)
        vf /= len(val_psi)
        history['epoch'].append(epoch)
        history['fidelity'].append(vf)
        history['loss'].append(avg_loss)
        if verbose and (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch+1:3d}/{n_epochs}: val_fid={vf:.4f}")
    return history


# ---------------------------------------------------------------------
# Label-free per-syndrome diagnostic + robust multi-seed training
# ---------------------------------------------------------------------
def syndrome_conditional_fidelity(model, noise, p=0.07, M=150, rng_seed=SEED + 5):
    """LABEL-FREE per-syndrome diagnostic: the mean recovery fidelity
    CONDITIONED on each syndrome occurring,

        cf_s = E[ <psi| R_s P_s rho P_s R_s^dag |psi> ] / E[ Tr[P_s rho] ].

    A healthy instrument has cf_s close to 1 for every syndrome that occurs
    with non-negligible probability.  Used to detect the occasional syndrome
    that falls into a bad local optimum and to select the best seed.  No
    error labels or decoder table are involved."""
    with torch.no_grad():
        R_all = recovery_unitary_batch(model())
    rng = np.random.RandomState(rng_seed)
    Fs = np.zeros(16); Ws = np.zeros(16)
    for _ in range(M):
        psi = random_logical_state(1, rng)
        pe = encode(psi)
        rho = noisy_state(pe, p, noise)
        rho16 = rho.unsqueeze(0).expand(16, DIM, DIM)
        PRP = torch.bmm(torch.bmm(P_SYNDS_STACK, rho16), P_SYNDS_STACK)
        Y = torch.bmm(torch.bmm(R_all, PRP), R_all.conj().transpose(1, 2))
        Fs += torch.einsum('i,sij,j->s', pe.conj(), Y, pe).real.numpy()
        Ws += torch.einsum('sij,ji->s', P_SYNDS_STACK, rho).real.numpy()
    return Fs / np.maximum(Ws, 1e-12), Ws / M


# Curriculum schedules (epochs, lr, p_range) per noise channel.
VSCR_SCHEDULES = {
    'depolarizing': [(600, 3e-2, (0.02, 0.08)),
                     (400, 3e-2, (0.02, 0.15)),
                     (1200, 1e-3, (0.02, 0.15))],
    'amplitude_damping': [(800, 3e-2, (0.01, 0.12)),
                          (1200, 1e-3, (0.01, 0.12))],
    'mixed': [(600, 3e-2, (0.02, 0.08)),
              (400, 3e-2, (0.02, 0.12)),
              (1200, 1e-3, (0.02, 0.12))],
}


def train_vscr_best(noise, seeds=(1234, 2024, 777), schedule=None, batch=48,
                    lam=0.0, verbose=True, val_gate=0.85):
    """Train VSCR for several seeds and keep the best model.

    Selection is LABEL-FREE: candidates that actually trained (validation
    fidelity >= val_gate) are ranked by the worst per-syndrome conditional
    fidelity (tie-break: validation fidelity); if no candidate passes the
    gate, the highest validation fidelity is used.  Rationale: training
    occasionally leaves ONE syndrome in a bad local optimum (which syndrome
    depends on the seed); with >=3 seeds, a candidate that is healthy on ALL
    syndromes is available.  The gate rejects degenerate runs whose cf values
    are uniformly mediocre (flat), which would otherwise inflate min_cf."""
    schedule = schedule or VSCR_SCHEDULES[noise]
    trs = []
    infos = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = VSCR()
        hist_all = {'epoch': [], 'fidelity': [], 'loss': []}
        off = 0
        for ep, lr, pr in schedule:
            hist = train_vscr(model, n_epochs=ep, batch=batch, p_range=pr,
                              noise=noise, lr=lr, lam=lam, rng_seed=seed,
                              use_real_grad=True, verbose=False)
            hist_all['epoch'] += [off + e for e in hist['epoch']]
            hist_all['fidelity'] += hist['fidelity']
            hist_all['loss'] += hist['loss']
            off += ep
        cf, w = syndrome_conditional_fidelity(model, noise)
        mask = w > 1e-3
        min_cf = float(np.where(mask, cf, 10.0).min())
        val_fid = hist_all['fidelity'][-1]
        infos.append({'seed': seed, 'min_cf': min_cf, 'val_fid': val_fid})
        if verbose:
            print(f"    seed {seed}: val_fid={val_fid:.4f}  "
                  f"min syndrome cf={min_cf:.4f}")
        trs.append((min_cf, val_fid, seed, model, hist_all))
    pool = [t for t in trs if t[1] >= val_gate]
    if not pool:                      # all runs degenerated: salvage the best
        pool = [max(trs, key=lambda t: t[1])]
    best = max(pool, key=lambda t: (t[0], t[1]))
    return best[3], best[4], infos

# ---------------------------------------------------------------------
# Evaluation: fidelity & logical error rate vs physical error rate
# ---------------------------------------------------------------------
METHOD_COLORS = {
    'Raw': '#9e9e9e', 'Perfect-code': '#e41a1c', 'ZNE': '#ff7f00',
    'Virtual Distillation': '#984ea3', 'LinDR': '#4daf4a', 'VSCR (ours)': '#377eb8',
    'VSCR-RandGrad': '#a65628'}


def evaluate_methods(model, A_lindr, p_values, noise, n_test=40, seed=SEED + 7):
    """Return dict: method -> {'F': list, 'LER': list}."""
    rng = np.random.RandomState(seed)
    test_states = [random_logical_state(1, rng) for _ in range(n_test)]
    names = ['Raw', 'Perfect-code', 'ZNE', 'Virtual Distillation', 'LinDR', 'VSCR (ours)']
    results = {m: {'F': [], 'LER': []} for m in names}
    with torch.no_grad():
        R_all = recovery_unitary_batch(model())        # build ONCE for whole sweep
    for p in p_values:
        fids = {m: [] for m in names}
        for psi in test_states:
            psi_enc = encode(psi)
            rho_n = noisy_state(psi_enc, p, noise)
            fids['Raw'].append(float(baseline_raw(psi_enc, rho_n)))
            fids['Perfect-code'].append(float(perfect_code_decoder_fidelity(psi_enc, rho_n)))
            fids['ZNE'].append(float(zne_fidelity(psi_enc, p, noise)))
            fids['Virtual Distillation'].append(float(virtual_distillation_fidelity(psi_enc, rho_n)))
            fids['LinDR'].append(float(lindr_fidelity(psi_enc, rho_n, A_lindr)))
            with torch.no_grad():
                _, F = vscr_fidelity(psi_enc, rho_n, R_all, lam=0.0)
            fids['VSCR (ours)'].append(float(F))
        for m in results:
            arr = np.array(fids[m])
            results[m]['F'].append(float(arr.mean()))
            results[m]['LER'].append(float(np.mean(arr < 0.9)))
    return results


def plot_curves(p_values, results, title, fname, ylabel='Avg. fidelity'):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for m, d in results.items():
        ax.plot(p_values, d['F'], 'o-', color=METHOD_COLORS.get(m, 'k'),
                label=m, linewidth=2, markersize=6)
    ax.set_xlabel('Physical error rate $p$', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc='best')
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()


def plot_ler(p_values, results, title, fname):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for m, d in results.items():
        ax.plot(p_values, d['LER'], 's-', color=METHOD_COLORS.get(m, 'k'),
                label=m, linewidth=2, markersize=6)
    ax.set_xlabel('Physical error rate $p$', fontsize=12)
    ax.set_ylabel('Logical error rate (F<0.9)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc='best')
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()


def plot_training(hist_real, hist_rand, fname):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(hist_real['epoch'], hist_real['fidelity'], 'b-', linewidth=2,
            label='VSCR (real Adam gradients)')
    ax.plot(hist_rand['epoch'], hist_rand['fidelity'], 'r--', linewidth=2,
            label='ACE-style random update (broken ref.)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation fidelity at $p=0.06$', fontsize=12)
    ax.set_title('Training dynamics: real gradients vs random walk', fontsize=13)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10, loc='best')
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()


def plot_summary_bar(p_values, results, p_idx, fname, noise_label):
    p = p_values[p_idx]
    methods = list(results.keys())
    vals = [results[m]['F'][p_idx] for m in methods]
    colors = [METHOD_COLORS.get(m, 'k') for m in methods]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars = ax.bar(methods, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.3f}',
                ha='center', fontsize=9)
    ax.set_ylabel('Avg. fidelity', fontsize=12)
    ax.set_title(f'Fidelity at $p={p:.3f}$ ({noise_label})', fontsize=13)
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15, fontsize=9)
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()

# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 72)
    print("VSCR-QEC : Variational Syndrome-Conditioned Recovery (variational")
    print("           quantum-classical ML for quantum error correction)")
    print("=" * 72)
    print(f"Code: [[{N},1,3]] perfect code  (Hilbert dim {DIM})")
    print("Recovery: projective syndrome measurement + LEARNED variational unitary")
    print(f"          R_s per syndrome ({PHI_DIM} rotation params, hypernetwork-shared)")

    # ---- 1. Train VSCR on depolarizing noise (real gradients) ----
    print("\n[1] Training VSCR (depolarizing): curriculum + 3 seeds,")
    print("    label-free per-syndrome selection...")
    model_dep, hist_real, infos_dep = train_vscr_best('depolarizing')

    # ---- 1b. Ablation: same architecture, random updates (broken ref.) ----
    print("\n[1b] Ablation: random-walk updates (replicates broken ACE-QEC)...")
    torch.manual_seed(SEED); np.random.seed(SEED)
    model_rand = VSCR()
    hist_rand = train_vscr(model_rand, n_epochs=2200, batch=48,
                          p_range=(0.02, 0.15), noise='depolarizing',
                          lr=1e-2, lam=0.0, use_real_grad=False, verbose=False)
    os.makedirs('figures', exist_ok=True)
    plot_training(hist_real, hist_rand, 'figures/fig_training_curves.png')

    # ---- 2. Train LinDR baseline (depolarizing) ----
    print("\n[2] Fitting LinDR (vnCDR-style linear) baseline...")
    A_dep = train_lindr(n_train=1500, p_range=(0.01, 0.12), noise='depolarizing')

    # ---- 3. Evaluate on depolarizing ----
    p_values = np.array([0.005, 0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.20])
    print("\n[3] Evaluating all methods (depolarizing)...")
    res_dep = evaluate_methods(model_dep, A_dep, p_values, 'depolarizing', n_test=40)
    plot_curves(p_values, res_dep, 'Depolarizing noise: recovery fidelity',
                'figures/fig_fidelity_depolarizing.png')
    plot_ler(p_values, res_dep, 'Depolarizing noise: logical error rate',
             'figures/fig_ler_depolarizing.png')
    plot_summary_bar(p_values, res_dep, 5, 'figures/fig_summary_depolarizing_p010.png',
                      'depolarizing')

    # ---- 4. Amplitude-damping noise ----
    print("\n[4] Training VSCR (amplitude damping): 3 seeds + selection...")
    model_ad, hist_ad, infos_ad = train_vscr_best('amplitude_damping')
    A_ad = train_lindr(n_train=1500, p_range=(0.01, 0.08), noise='amplitude_damping')
    res_ad = evaluate_methods(model_ad, A_ad, p_values, 'amplitude_damping', n_test=40)
    plot_curves(p_values, res_ad, 'Amplitude-damping noise: recovery fidelity',
                'figures/fig_fidelity_amplitude_damping.png')
    plot_summary_bar(p_values, res_ad, 5, 'figures/fig_summary_ad_p010.png',
                      'amplitude-damping')

    # ---- 5. Mixed noise ----
    print("\n[5] Training VSCR (mixed depol+AD): 3 seeds + selection...")
    model_mx, hist_mx, infos_mx = train_vscr_best('mixed')
    A_mx = train_lindr(n_train=1500, p_range=(0.01, 0.10), noise='mixed')
    res_mx = evaluate_methods(model_mx, A_mx, p_values, 'mixed', n_test=40)
    plot_curves(p_values, res_mx, 'Mixed (depol+AD) noise: recovery fidelity',
                'figures/fig_fidelity_mixed.png')
    plot_summary_bar(p_values, res_mx, 5, 'figures/fig_summary_mixed_p010.png', 'mixed')

    _print_and_save(res_dep, res_ad, res_mx, p_values, hist_real, hist_rand, t0)
    return res_dep, res_ad, res_mx, hist_real, hist_rand

def _print_and_save(res_dep, res_ad, res_mx, p_values, hist_real, hist_rand, t0):
    print("\n" + "=" * 72)
    print("RESULTS  (avg fidelity at each p)")
    print("=" * 72)
    for label, res in [('DEPOLARIZING', res_dep),
                       ('AMPLITUDE-DAMPING', res_ad), ('MIXED', res_mx)]:
        print(f"\n--- {label} ---")
        print("p        " + "  ".join(f"{m[:9]:>10s}" for m in res))
        for i, p in enumerate(p_values):
            print(f"{p:.3f}    " + "  ".join(f"{res[m]['F'][i]:10.4f}" for m in res))
    print("\nImprovement of VSCR over baselines (depolarizing, p=0.10):")
    i = list(p_values).index(0.10)
    base = res_dep['VSCR (ours)']['F'][i]
    for m in ['Raw', 'Perfect-code', 'ZNE', 'Virtual Distillation', 'LinDR']:
        b = res_dep[m]['F'][i]
        print(f"  vs {m:22s}: {b:.4f} -> {base:.4f}  "
              f"(abs {(base - b):+.4f}, rel {100 * (base - b) / max(b, 1e-9):+.1f}%)")
    print("\nImprovement of VSCR over baselines (amplitude-damping, p=0.10):")
    base = res_ad['VSCR (ours)']['F'][i]
    for m in ['Raw', 'Perfect-code', 'ZNE', 'Virtual Distillation', 'LinDR']:
        b = res_ad[m]['F'][i]
        print(f"  vs {m:22s}: {b:.4f} -> {base:.4f}  "
              f"(abs {(base - b):+.4f}, rel {100 * (base - b) / max(b, 1e-9):+.1f}%)")
    print(f"\nFinal training fidelity (real grad): {hist_real['fidelity'][-1]:.4f}")
    print(f"Final training fidelity (random walk): {hist_rand['fidelity'][-1]:.4f}")
    print(f"Total runtime: {time.time() - t0:.1f}s")

    np.savez('vscr_results.npz',
             p_values=p_values,
             dep_F={m: res_dep[m]['F'] for m in res_dep},
             ad_F={m: res_ad[m]['F'] for m in res_ad},
             mx_F={m: res_mx[m]['F'] for m in res_mx},
             dep_LER={m: res_dep[m]['LER'] for m in res_dep},
             hist_real_fid=np.array(hist_real['fidelity']),
             hist_rand_fid=np.array(hist_rand['fidelity']))
    print("\nFigures saved in figures/: fig_training_curves.png, fig_fidelity_depolarizing.png,")
    print("  fig_ler_depolarizing.png, fig_summary_depolarizing_p010.png,")
    print("  fig_fidelity_amplitude_damping.png, fig_summary_ad_p010.png,")
    print("  fig_fidelity_mixed.png, fig_summary_mixed_p010.png")
    print("Numerical data: vscr_results.npz")


if __name__ == "__main__":
    main()

