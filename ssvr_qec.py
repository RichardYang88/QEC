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

# Pin the native thread pools BEFORE numpy/torch are imported -- both read these
# variables at load time.
#
# WHY: every hot loop in this project is small-matrix bound (2x2, 4x4 and 32x32
# blocks), so single-threaded execution costs nothing measurable and buys
# bit-reproducibility, which a paper pipeline needs anyway.
#
# It is NOT a fix for the native faults described below. An earlier revision of
# this comment claimed the faults "did not reproduce single-threaded"; that was
# wrong -- they reproduce readily with every thread pool pinned to 1.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

# ---------------------------------------------------------------------------
# THE NATIVE FAULT, what was ruled out, and what actually works
# ---------------------------------------------------------------------------
# Symptom: long compute runs intermittently die with SIGSEGV or SIGILL. The
# faulting frame MOVES between unrelated libraries (`torch.autograd.
# _engine_run_backward`, scipy's L-BFGS-B `approx_derivative`, a bare
# `numpy.matmul`, an 8x8 matmul in `vscr_paper_abl.c_tr`, and most often
# `vscr_paper_abl._block_obj_grad`). Every affected routine returns numerically
# correct results whenever it completes, and the operations are (2,32), (32,32)
# and (8,8) complex128 -- far too small to corrupt memory from a logic error.
#
# RULED OUT by measurement (re-run `diag_native_fault.py` to re-check):
#   * Multi-threading. Faults reproduce with every pool pinned to 1.
#   * OpenBLAS AVX-512 mis-dispatch. This CPU (Intel Core Ultra 9 285K, Arrow
#     Lake) has NO AVX-512 at all, so there is no such kernel to select. And
#     forcing the most conservative kernel set still faults, with SIGILL:
#         OPENBLAS_CORETYPE=HASWELL     -> SIGSEGV within 2e4 calls
#         OPENBLAS_CORETYPE=SANDYBRIDGE -> SIGILL   at ~1.5e5 calls
#         (unset)                       -> SIGILL   at ~2.5e5 calls
#     An illegal instruction cannot come from a correctly compiled AVX-only
#     kernel on a CPU that implements AVX.
#   * The GEMM itself. 2e7 bare (2,32)@(32,32) complex matmuls ran clean while
#     `_block_obj_grad` died within 2e4 calls.
#
# WHAT ACTUALLY CORRELATES: CPU affinity. In repeated 3-way trials run
# concurrently (identical ambient load, so only affinity differs):
#     free to migrate across cores -> FAULT, every trial
#     pinned to one P-core (cpu0)  -> clean, 1.5e5-4e5 calls, every trial
#     pinned to one E-core (cpu16) -> clean, 1.5e5 calls, every trial
# So migration across this hybrid P-/E-core cluster is the trigger, not a
# library and not one bad core. Hence `_pin_single_cpu()` below.
#
# OPENBLAS_CORETYPE is still set, but ONLY for determinism: it makes kernel
# selection independent of the host, so results are bit-comparable across
# machines. It is a no-op for the fault and must not be described as a fix.
os.environ.setdefault('OPENBLAS_CORETYPE', 'HASWELL')


def _pin_single_cpu(prefer_p_cores=8):
    """Restrict this process to ONE logical CPU. Returns the cpu id, or None.

    This is the fix for the native fault documented above: every arm left free to
    migrate across the hybrid cluster faulted, while arms pinned to a single core
    -- P-core or E-core -- completed 4e5 calls clean, repeatedly. Affinity changes
    scheduling only, never a computed value, so this cannot alter any number in
    the paper; `paper/audit_numbers.py` re-verifies the artifacts regardless.

    Opt out with `QEC_PIN_CPU=0` (e.g. to reproduce the fault, or on a host where
    pinning is undesirable). The core is chosen from the PID within the first
    `prefer_p_cores` allowed CPUs, which are the P-cores on Arrow Lake-S, so that
    several concurrent processes (`run_selftests.py`'s isolated children, or
    `vscr_paper._refine_isolated`) spread out instead of piling onto cpu0 -- and
    so long runs get a fast core rather than an E-core.
    """
    if os.environ.get('QEC_PIN_CPU', '1') == '0':
        return None
    if not hasattr(os, 'sched_setaffinity'):
        return None                       # non-Linux: nothing to do
    try:
        allowed = sorted(os.sched_getaffinity(0))
        if not allowed:
            return None
        pool = allowed[:max(1, min(prefer_p_cores, len(allowed)))]
        cpu = pool[os.getpid() % len(pool)]
        os.sched_setaffinity(0, {cpu})
        return cpu
    except OSError:
        return None                       # restricted container, cgroup, etc.


PINNED_CPU = _pin_single_cpu()

import numpy as np
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass        # inter-op pool already initialised by an earlier parallel op
# Run the autograd engine in the CALLING thread instead of its own worker pool.
# For the graphs here (a 60-gate chain on two columns) the engine has no
# parallelism to exploit anyway, so this costs nothing and makes backward
# bit-deterministic. An earlier revision of this comment called it "the switch
# that removes" the native faults; it does not -- the faults were later
# reproduced in pure numpy with no autograd anywhere in the call stack (see the
# block above and `diag_native_fault.py`).
torch.autograd.set_multithreading_enabled(False)


def _pin_blas_threads(n=1):
    """Best-effort runtime pin of the OpenMP pool; returns what was pinned.

    OpenBLAS/MKL themselves are controlled by the ``*_NUM_THREADS`` environment
    variables set at the top of this module and of every entry-point script
    *before* numpy is imported -- that is the only safe mechanism, because
    those libraries read the variables once at load time.

    We deliberately do NOT dlopen the BLAS and call its ``*_set_num_threads``
    symbols by name.  An earlier version of this function did, and it segfaulted
    the interpreter at import time: ``scipy-openblas64`` exports several
    near-identically named setters (``openblas_set_num_threads64_``,
    ``openblas_set_num_threads_64_``, ``openblas_set_num_threads_local``, ...)
    with differing ABIs, and calling the wrong one through
    ``ctypes`` with a guessed signature jumps into garbage.  That is a worse
    failure mode than the one being guarded against, so only ``omp_set_num_threads``
    -- whose signature is fixed by the OpenMP standard -- is called here.

    Rationale for pinning at all: on this host (Intel Core Ultra 9 285K, Arrow
    Lake) sustained complex128 GEMM work crashed intermittently with SIGSEGV /
    SIGILL at a *moving* site -- torch's autograd engine, scipy's
    finite-difference Jacobian, and plain numpy ``matmul`` -- the signature of a
    native kernel/threading fault rather than a logic bug.  All matrices here
    are 2x2, 4x4 or 32x32, so a threaded BLAS gains nothing and only adds risk."""
    import ctypes
    done = []
    try:
        with open('/proc/self/maps') as fh:
            paths = sorted({ln.split()[-1] for ln in fh
                            if ln.split() and 'gomp' in ln.split()[-1]
                            and '.so' in ln.split()[-1]})
    except OSError:
        return done
    for p in paths:
        try:
            lib = ctypes.CDLL(p)
            fn = lib.omp_set_num_threads
        except (OSError, AttributeError):
            continue
        try:
            fn.argtypes = [ctypes.c_int]
            fn(n)
            done.append(f'{os.path.basename(p)}:omp_set_num_threads')
        except Exception:                                       # pragma: no cover
            pass
    return done


_BLAS_PINS = _pin_blas_threads(1)
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


def apply_coherent(rho, eps):
    """Systematic coherent over-rotation: identical Rx(eps) on every qubit
    (calibration-type control error).  NOT a Pauli mixture."""
    U = I_full
    for q in range(N):
        U = U @ embed_gate(Rx(torch.tensor(eps, dtype=DTYPE)), [q])
    return U @ rho @ U.conj().T


def noisy_state(psi_enc, p, noise='depolarizing'):
    """Build the noisy density matrix from a pure encoded state."""
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


# ---------------------------------------------------------------------
# Column-propagation form of the ansatz.
#   R(phi) = G_{59} ... G_0 with G_k = cos(phi_k/2) I - i sin(phi_k/2) P_k,
#   so applying R to a (DIM, r) block of columns costs O(PHI_DIM*DIM^2*r)
#   instead of O(PHI_DIM*DIM^3) -- a 16x saving for r = 2, which is what makes
#   the EXACT quadrature objective below cheap enough to use every epoch.
# ---------------------------------------------------------------------
P_EXP = [P.to(DTYPE).contiguous() for P in P_SLOTS]


def recovery_action_cols(phi_batch, X):
    """(B, PHI_DIM) real angles, (B, DIM, r) complex columns -> R(phi_b) X_b.

    Identical to `recovery_unitary_batch(phi_batch) @ X` but
    O(PHI_DIM * DIM^2 * r) instead of O(PHI_DIM * DIM^3), and differentiable
    w.r.t. phi_batch.  Verified against the explicit product in
    `_selftest_action_cols`."""
    Y = X
    half = phi_batch / 2
    for k in range(len(P_SLOTS)):
        c = torch.cos(half[:, k]).to(DTYPE).view(-1, 1, 1)
        s = torch.sin(half[:, k]).to(DTYPE).view(-1, 1, 1)
        Y = c * Y - 1j * s * torch.matmul(P_EXP[k], Y)
    return Y


def recovery_action_cols_dag(phi_batch, X):
    """(B, PHI_DIM), (B, DIM, r) -> R(phi_b)^dagger X_b  (adjoint column propagation).

    R(phi) = G_{59} ... G_0 with G_k = exp(-i phi_k P_k / 2), so
    R^dagger = G_0^dagger G_1^dagger ... G_59^dagger and the gates must be
    applied in REVERSE order with the sign of the sin term flipped.  Verified
    against `recovery_unitary_batch(phi).conj().transpose(1,2) @ X` in
    `_selftest_action_cols`."""
    Y = X
    half = phi_batch / 2
    for k in reversed(range(len(P_SLOTS))):
        c = torch.cos(half[:, k]).to(DTYPE).view(-1, 1, 1)
        s = torch.sin(half[:, k]).to(DTYPE).view(-1, 1, 1)
        Y = c * Y + 1j * s * torch.matmul(P_EXP[k], Y)
    return Y


W_SYND = None


def syndrome_bases():
    """W[s] = (32, 2) orthonormal basis of range(P_s), the rank-2 syndrome space.

    Only needed by the fast quadrature loss.  The final objective is invariant
    under the choice of basis because it enters exclusively through
    W_s W_s^dagger = P_s (see `vscr_fidelity_quad`)."""
    global W_SYND
    if W_SYND is None:
        W = np.zeros((16, DIM, 2), dtype=complex)
        for s in range(16):
            P = P_SYNDS[s].numpy()
            ev, evec = np.linalg.eigh(0.5 * (P + P.conj().T))
            assert abs(ev[-1] - 1) < 1e-12 and abs(ev[-2] - 1) < 1e-12, \
                f"syndrome {s} projector is not rank 2: {ev}"
            assert abs(ev[-3]) < 1e-12, f"syndrome {s} projector is not rank 2: {ev}"
            W[s] = evec[:, -2:]
            assert np.allclose(W[s].conj().T @ W[s], np.eye(2), atol=1e-12)
            assert np.allclose(W[s] @ W[s].conj().T, P, atol=1e-12)
        W_SYND = torch.tensor(W, dtype=DTYPE)
    return W_SYND


def vscr_fidelity_quad(phi_batch, Q, synd_w=None):
    """EXACT Haar-quadrature recovery fidelity, vectorised over syndromes AND
    probe states, propagating only TWO columns per syndrome.

        F = sum_j w_j sum_s <psi_j| R_s P_s rho_j P_s R_s^dag |psi_j>

    Writing P_s = W_s W_s^dag and G_s = R_s W_s (32x2),

        u_{sj} = P_s R_s^dag psi_j = W_s G_s^dag psi_j = W_s z_{sj},
        term   = u^dag rho_j u = z_{sj}^dag B_{sj} z_{sj},   B_{sj} = W_s^dag rho_j W_s,

    so the 60-gate ansatz is applied to the 2 columns of W_s ONLY, and every
    rho_j-dependent quantity (B) is a constant precomputed once per training
    stage by `quad_cache`.  This is the whole point: the exact objective costs
    ~1/20 of the Monte-Carlo path per epoch *and* has zero gradient variance,
    whereas the non-Pauli headroom it is meant to resolve spans 5.4e-6
    (coherent, p=0.06) to 3.1e-3 (coherent, p=0.30) over the Pauli decoder:
    the small end lies well BELOW the 48-sample Monte-Carlo SEM of 1.1e-4 at
    amplitude damping p=0.10, and even the large end is only a few times it.

    Returns (loss, F, F_s) with F_s the exact per-syndrome averaged fidelity
    (used for the label-free worst-syndrome model-selection criterion).

    NOTE: the code-population regulariser (`lam` in `vscr_fidelity`) needs the
    full 32x32 R_s and is not supported here; every paper configuration trains
    with lam = 0, which `train_vscr` asserts."""
    G = recovery_action_cols(phi_batch, Q['W'])
    z = torch.einsum('sib,ni->snb', G.conj(), Q['psi'])
    T = torch.einsum('snb,snbc,snc->sn', z.conj(), Q['BS'], z).real
    F_s = (T * Q['w']).sum(dim=1)
    F = F_s.sum() if synd_w is None else (synd_w * F_s).sum()
    return (1 - F), F, F_s


def _selftest_action_cols():
    """Assert the column-propagation form equals the explicit 32x32 product,
    for R and for R^dagger, and that both give identical gradients."""
    g = torch.Generator().manual_seed(11)
    phi = torch.randn(16, PHI_DIM, dtype=torch.float64, generator=g) * 0.7
    Xr = torch.randn(16, DIM, 2, dtype=torch.float64, generator=g)
    Xi = torch.randn(16, DIM, 2, dtype=torch.float64, generator=g)
    X = (Xr + 1j * Xi).to(DTYPE)
    R = recovery_unitary_batch(phi)
    err = float((torch.bmm(R, X) - recovery_action_cols(phi, X)).abs().max())
    errd = float((torch.bmm(R.conj().transpose(1, 2), X)
                  - recovery_action_cols_dag(phi, X)).abs().max())
    assert err < 1e-12, f"action_cols mismatch {err:.2e}"
    assert errd < 1e-12, f"action_cols_dag mismatch {errd:.2e}"
    p1 = phi.clone().requires_grad_(True)
    p2 = phi.clone().requires_grad_(True)
    torch.bmm(recovery_unitary_batch(p1), X).abs().sum().backward()
    recovery_action_cols(p2, X).abs().sum().backward()
    gerr = float((p1.grad - p2.grad).abs().max())
    assert gerr < 1e-12, f"action_cols gradient mismatch {gerr:.2e}"
    print(f"[action-cols selftest] R err {err:.1e}, R^dag err {errd:.1e}, "
          f"grad err {gerr:.1e}  VERIFIED", flush=True)


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
# Petz / noise-adapted recovery map  (RECENT COMPARATOR)
#
#   R_{sigma,N}(X) = sigma^{1/2} N^dag( N(sigma)^{-1/2} X N(sigma)^{-1/2} ) sigma^{1/2}
#
# with the reference state sigma = P_code / 2 (maximally mixed on the code
# space) and N the noise channel.  This is the Petz recovery map / transpose
# channel of Petz (1986), in the noise-adapted form used for QEC by
# Kishore, Radhakrishnan, Balasubramanian & Pandey, Phys. Rev. Research 6,
# 043034 (2024), and by Gilyén, Lloyd, Marvian, Quek & Song,
# Phys. Rev. X 13, 041045 (2023).
#
# WHY IT IS THE RIGHT RECENT BASELINE.  It is a closed-form, training-free,
# noise-adapted CPTP recovery that is provably near-optimal whenever the
# Knill-Laflamme conditions hold approximately -- exactly the regime of this
# code.  Unlike ZNE/VD/LinDR (which are error-*mitigation* estimators scored on
# a state), Petz is a genuine recovery CHANNEL and therefore sits in the same
# category as the decoder and VSCR, and can be scored with the identical exact
# fidelity estimator <psi_enc| R(rho_noisy) |psi_enc>.  A claim that a learned
# syndrome-conditioned unitary beats a classical lookup decoder is much weaker
# than a claim that it beats the best known *noise-adapted* recovery, so this is
# the comparison that actually carries weight.
#
# It is also the honest worst case for VSCR: Petz uses the exact channel and the
# exact code-space reference, and is allowed to be NON-unitary and
# NON-syndrome-resolved (it may leak outside the code space, which a
# measurement-based decoder never does).  Where VSCR still wins, the win is
# attributable to the projective syndrome readout.
#
# Implementation.  The channel is stored as an ordered list of STAGES, each
# stage being a list over qubits of the 32x32 Kraus operators acting on that
# qubit ALONE.  Then N(rho) is a handful of 32x32 products, and the adjoint
# N^dagger(Y) = sum_k K_k^dagger Y K_k is the same loop in reverse with K -> K^dagger.
# This deliberately avoids the Liouville/superoperator route: the tempting
# identity  S_N = kron_q S_q  with  S_q = sum_k kron(K, conj(K))  is FALSE,
# because kron(A0 (x) A1, B0 (x) B1) and kron(A0,B0) (x) kron(A1,B1) carry the
# same entries under DIFFERENT index groupings ((i0,i1,j0,j1) vs (i0,j0,i1,j1))
# and so differ by a permutation -- measured error 7.2e-2 on depolarizing.  It
# also avoids materialising the mixed channel's 4^5 * 2^5 = 32768 full-space
# Kraus operators.  `_selftest_petz` checks the adjoint against the
# implementation-independent Hilbert-Schmidt identity Tr[X^dag N(Y)] =
# Tr[N^dag(X)^dag Y].
# ---------------------------------------------------------------------
_PETZ_CACHE = {}


def _channel_stages(noise, p):
    """Ordered list of STAGES; each stage is a list over qubits of the 32x32
    Kraus operators acting on that qubit alone.

    N = stage[-1] o ... o stage[0].  Within a stage the per-qubit maps act on
    disjoint qubits and therefore commute, so the order inside a stage is
    irrelevant.  `mixed` gets two stages (depolarizing then amplitude damping),
    matching `apply_mixed` exactly."""
    if noise not in ('depolarizing', 'amplitude_damping', 'mixed', 'coherent'):
        raise ValueError(noise)
    I2c = torch.eye(2, dtype=DTYPE)
    stages = []
    if noise in ('depolarizing', 'mixed'):
        ks2 = [math.sqrt(1 - p) * I2c, math.sqrt(p / 3.0) * Xp,
               math.sqrt(p / 3.0) * Yp, math.sqrt(p / 3.0) * Zp]
        stages.append([[embed_gate(K, [q]) for K in ks2] for q in range(N)])
    if noise in ('amplitude_damping', 'mixed'):
        g = p if noise == 'amplitude_damping' else 0.5 * p
        stages.append([list(amp_kraus(g, q)) for q in range(N)])
    if noise == 'coherent':
        U2 = Rx(torch.tensor(p, dtype=DTYPE))
        stages.append([[embed_gate(U2, [q])] for q in range(N)])
    return stages


def apply_channel(rho, noise, p):
    """N(rho) for an arbitrary 32x32 OPERATOR rho (not necessarily a state).

    Same channel as `noisy_state`, but callable on any operator -- which is what
    the Petz map needs in order to form N(sigma)."""
    for stage in _channel_stages(noise, p):
        for ks in stage:
            out = torch.zeros_like(rho)
            for K in ks:
                out = out + K @ rho @ K.conj().T
            rho = out
    return rho


def apply_channel_adjoint(Y, noise, p):
    """N^dagger(Y) = sum_k K_k^dagger Y K_k : stages in REVERSE, K -> K^dagger.

    Verified against Tr[X^dag N(Y)] == Tr[N^dag(X)^dag Y] in `_selftest_petz`."""
    for stage in reversed(_channel_stages(noise, p)):
        for ks in stage:
            out = torch.zeros_like(Y)
            for K in ks:
                out = out + K.conj().T @ Y @ K
            Y = out
    return Y


def _petz_build(noise, p):
    """(A, sigma_half, sigma) for the Petz map at (noise, p); cached.

    A          : N(sigma)^{-1/2}, pseudo-inverted on the support of N(sigma)
    sigma_half : sigma^{1/2} = P_code / sqrt(2)
    sigma      : P_code / 2, the code-space reference state
    """
    key = (noise, round(float(p), 12))
    if key in _PETZ_CACHE:
        return _PETZ_CACHE[key]
    sigma = P_code / 2.0
    sig_half = P_code / math.sqrt(2.0)
    nsig = apply_channel(sigma, noise, p)
    nsig = 0.5 * (nsig + nsig.conj().T)
    ev, V = torch.linalg.eigh(nsig)
    # pseudo-inverse square root on the support of N(sigma)
    tol = float(ev.max()) * 1e-12
    inv = torch.where(ev > tol, 1.0 / torch.sqrt(ev.clamp_min(tol)),
                      torch.zeros_like(ev))
    A = (V * inv) @ V.conj().T
    _PETZ_CACHE[key] = (A, sig_half, sigma)
    return _PETZ_CACHE[key]


def petz_recovery(rho, noise, p):
    """Apply the Petz / noise-adapted recovery map to rho (32x32)."""
    A, sig_half, _sigma = _petz_build(noise, float(p))
    return sig_half @ apply_channel_adjoint(A @ rho @ A, noise, p) @ sig_half


def petz_recovery_fidelity(psi_enc, rho_noisy, noise, p):
    """<psi_enc| Petz(rho_noisy) |psi_enc> -- scored EXACTLY like every other
    recovery in `vscr_paper.evaluate_paper`, so it is directly comparable."""
    return float((psi_enc.conj() @ petz_recovery(rho_noisy, noise, p)
                  @ psi_enc).real)


def _explicit_kraus(noise, p):
    """Explicit full-space 32x32 Kraus set, built the SLOW independent way
    (product over qubits in the same sequential order as `noisy_state`).  Only
    used by `_selftest_petz` to cross-check `apply_channel`."""
    I_full_c = torch.eye(DIM, dtype=DTYPE)
    if noise == 'amplitude_damping':
        per_q = [list(amp_kraus(p, q)) for q in range(N)]
    elif noise == 'depolarizing':
        ks2 = [math.sqrt(1 - p) * torch.eye(2, dtype=DTYPE),
               math.sqrt(p / 3.0) * Xp, math.sqrt(p / 3.0) * Yp,
               math.sqrt(p / 3.0) * Zp]
        per_q = [[embed_gate(K, [q]) for K in ks2] for q in range(N)]
    elif noise == 'coherent':
        U2 = Rx(torch.tensor(p, dtype=DTYPE))
        per_q = [[embed_gate(U2, [q])] for q in range(N)]
    else:
        raise ValueError(f'no explicit Kraus set for {noise}')
    ops = [I_full_c]
    for layer in per_q:
        ops = [Kq @ op for op in ops for Kq in layer]
    return ops


def _selftest_petz():
    """Validate the Petz / noise-adapted recovery implementation.

    (a) `apply_channel` reproduces `noisy_state` exactly on all four channels,
        and matches an INDEPENDENT brute-force full-space Kraus sum wherever
        that is affordable (coherent 1 op, amplitude damping 2^5, depolarizing
        4^5 = 1024);
    (b) `apply_channel_adjoint` is the true Hilbert-Schmidt adjoint, i.e.
        Tr[X^dag N(Y)] == Tr[N^dag(X)^dag Y] for random X and Y -- an
        implementation-independent characterisation that does not care how the
        Kraus operators were grouped into stages;
    (c) the Petz map preserves trace and Hermiticity, returns a physical
        fidelity (F_raw <= F_petz <= 1), and reduces to the identity on the code
        space at p = 0.
    """
    rng = np.random.RandomState(3)
    CH = ('depolarizing', 'amplitude_damping', 'mixed', 'coherent')
    PS = (0.03, 0.10, 0.25)

    # (a) apply_channel == noisy_state
    worst = 0.0
    for noise in CH:
        for p in PS:
            psi = encode(random_logical_state(1, rng))
            rho0 = torch.outer(psi, psi.conj())
            worst = max(worst, float((noisy_state(psi, p, noise)
                                      - apply_channel(rho0, noise, p)).abs().max()))
    assert worst < 1e-13, f'apply_channel != noisy_state: {worst:.2e}'
    print(f'[petz selftest a1] apply_channel == noisy_state on all 4 channels x '
          f'p in {PS} to {worst:.2e}  VERIFIED', flush=True)

    # (a2) independent brute-force Kraus sum
    worst_bf = 0.0
    for noise in ('coherent', 'amplitude_damping', 'depolarizing'):
        for p in (0.05, 0.15):
            psi = encode(random_logical_state(1, rng))
            rho0 = torch.outer(psi, psi.conj())
            Ks = _explicit_kraus(noise, p)
            ref = sum(K @ rho0 @ K.conj().T for K in Ks)
            worst_bf = max(worst_bf,
                           float((apply_channel(rho0, noise, p) - ref).abs().max()))
    assert worst_bf < 1e-12, f'apply_channel != Kraus sum: {worst_bf:.2e}'
    print(f'[petz selftest a2] apply_channel == independent full-space Kraus sum '
          f'(1 / 32 / 1024 operators) to {worst_bf:.2e}  VERIFIED', flush=True)

    # (b) Hilbert-Schmidt adjoint identity
    worst_ad = 0.0
    for noise in CH:
        for p in (0.05, 0.15):
            for _ in range(3):
                Xr = torch.randn(DIM, DIM, dtype=torch.float64)
                X = (Xr + 1j * torch.randn(DIM, DIM, dtype=torch.float64)).to(DTYPE)
                Yr = torch.randn(DIM, DIM, dtype=torch.float64)
                Y = (Yr + 1j * torch.randn(DIM, DIM, dtype=torch.float64)).to(DTYPE)
                lhs = complex(torch.trace(X.conj().T @ apply_channel(Y, noise, p)))
                Nad = apply_channel_adjoint(X, noise, p)
                rhs = complex(torch.trace(Nad.conj().T @ Y))
                scale = max(abs(lhs), abs(rhs), 1e-30)
                worst_ad = max(worst_ad, abs(lhs - rhs) / scale)
    assert worst_ad < 1e-13, f'adjoint identity violated: {worst_ad:.2e}'
    print(f'[petz selftest b] Tr[X^dag N(Y)] == Tr[N^dag(X)^dag Y] on all 4 '
          f'channels to {worst_ad:.2e} relative  VERIFIED', flush=True)

    # (c) Petz map properties
    worst_tr = 0.0
    for noise in CH:
        for p in PS:
            psi = encode(random_logical_state(1, rng))
            rho_n = noisy_state(psi, p, noise)
            out = petz_recovery(rho_n, noise, p)
            worst_tr = max(worst_tr,
                           abs(float(out.trace().real) - float(rho_n.trace().real)),
                           float((out - out.conj().T).abs().max()))
            Fpetz = petz_recovery_fidelity(psi, rho_n, noise, p)
            Fraw = float(baseline_raw(psi, rho_n))
            assert Fpetz <= 1.0 + 1e-9, (noise, p, Fpetz)
            assert Fpetz >= Fraw - 1e-9, (noise, p, Fpetz, Fraw)
    assert worst_tr < 1e-10, f'Petz not trace/Hermiticity consistent: {worst_tr:.2e}'
    print(f'[petz selftest c] Petz preserves trace and Hermiticity to '
          f'{worst_tr:.2e}; F_raw <= F_petz <= 1 on all 4 channels x '
          f'p in {PS}  VERIFIED', flush=True)

    psi = encode(random_logical_state(1, rng))
    rho0 = torch.outer(psi, psi.conj())
    F0 = petz_recovery_fidelity(psi, rho0, 'depolarizing', 0.0)
    assert abs(F0 - 1.0) < 1e-12, F0
    print(f'[petz selftest d] noiseless channel: Petz fidelity = {F0:.12f} '
          f'(identity on the code space)  VERIFIED', flush=True)


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
# EXACT quadrature for the training/selection objective.
#
# Both averages in the objective are replaced by deterministic quadrature
# rules that are exact for the algebraic degree of the integrand, so the
# loss and its gradient have ZERO variance and ZERO bias:
#
#   * Haar average over single-logical-qubit probe states.  With
#     |psi><psi| = (I + r.sigma)/2 the integrand is a polynomial of degree
#     <= 2 in the Bloch vector r, i.e. degree <= 2 in u = cos(theta) and
#     trigonometric degree <= 2 in phi.  Gauss-Legendre with n_u = 3 is
#     exact through degree 5 in u, and an equispaced phi grid with
#     n_phi = 7 is exact through trigonometric degree 6.
#   * Noise-strength average over p in p_range.  Analytic in p but NOT
#     polynomial (depolarizing composes five single-qubit maps, amplitude
#     damping enters through sqrt(1-gamma), coherent is trigonometric), so
#     Gauss-Legendre converges geometrically rather than exactly; n_p = 6
#     leaves a residual bias below 1e-15 for all four channels, whereas a
#     low-order rule such as n_p = 2 leaves a bias of up to 5.6e-5 -- larger
#     than the entire effect being measured.
#
# Why this matters quantitatively: the non-Pauli headroom over the Pauli
# decoder spans 5.4e-6 .. 3.1e-3 (measured by diag_optunit.py), while the
# batch = 48 Monte-Carlo protocol has a SEM of ~1.1e-4 at p = 0.10 on amplitude
# damping.  The small end of that range is an order of magnitude BELOW the noise
# floor of the stochastic estimator, so it can only be resolved -- and only be
# *optimised* -- with an exact one.
# ---------------------------------------------------------------------
def haar_quadrature(n_u=3, n_phi=7):
    """Deterministic exact Haar quadrature over single-logical-qubit states.

    Returns (states, weights): states is (n_u*n_phi, 2) complex128, weights is
    (n_u*n_phi,) float64 summing to 1.  For any observable polynomial of degree
    <= 2 in the Bloch vector, sum_i w_i <psi_i|O|psi_i> equals the Haar average."""
    (u, wu) = np.polynomial.legendre.leggauss(n_u)
    phis = 2 * np.pi * np.arange(n_phi) / n_phi
    weights = []
    states = []
    for iu in range(n_u):
        th = math.acos(float(np.clip(u[iu], -1, 1)))
        for ip in range(n_phi):
            states.append(np.array([math.cos(th / 2),
                                    np.exp(1j * phis[ip]) * math.sin(th / 2)],
                                   dtype=complex))
            weights.append(wu[iu] / 2 / n_phi)
    return (torch.tensor(np.array(states), dtype=DTYPE),
            torch.tensor(np.array(weights), dtype=torch.float64))


def p_quadrature(p_range, n=6):
    """Gauss-Legendre nodes/weights for averaging over a noise-strength interval.

    n = 6 is exact to machine precision for all four channels used here (see the
    convergence note above `haar_quadrature`); the integrand is analytic in p but
    NOT polynomial -- `apply_depolarizing` composes five single-qubit maps
    (degree 5), amplitude damping enters through sqrt(1-gamma), and
    `apply_coherent` is trigonometric -- so a low-order rule such as n = 2 leaves
    a bias of up to 5.6e-5.  Weights sum to 1, i.e. this is a proper average."""
    (x, w) = np.polynomial.legendre.leggauss(n)
    hi = float(p_range[1])
    lo = float(p_range[0])
    half = 0.5 * (hi - lo)
    mid = 0.5 * (lo + hi)
    return (mid + half * x, w / 2)


def quad_cache(p_range, noise, n_u=3, n_phi=7, n_p=6):
    """Precompute everything the exact quadrature objective needs, ONCE.

    Returns a dict Q with
      'psi'   (nq, 32) complex128  encoded probe states V|psi_j>   (constant)
      'rho'   (nq, 32, 32)         E_p(V|psi_j><psi_j|V^dagger)    (constant)
      'w'     (nq,) float64        Haar x noise-strength weights, sum = 1
      'W'     (16, 32, 2)          orthonormal syndrome bases      (constant)
      'BS'    (16, nq, 2, 2)       W_s^dagger rho_j W_s            (constant)
      'p_syn' (16,) float64        syndrome probabilities sum_j w_j Tr[B_sj]
      'meta'  dict                 node bookkeeping
    nq = n_u * n_phi * n_p = 126 for the defaults.  The channel is therefore
    applied exactly once per training stage instead of once per epoch per
    sample, and `BS` folds the density matrices down to the 2x2 syndrome
    blocks so that a training epoch only propagates 2 columns per syndrome
    through the 60-gate ansatz."""
    (states, sw) = haar_quadrature(n_u, n_phi)
    (p_nodes, pw) = p_quadrature(p_range, n_p)
    ws = []
    rhos = []
    psis = []
    with torch.no_grad():
        for ip, p in enumerate(p_nodes):
            for iq in range(states.shape[0]):
                pe = encode(states[iq])
                psis.append(pe.detach())
                rhos.append(noisy_state(pe, float(p), noise).detach())
                ws.append(float(pw[ip]) * float(sw[iq]))
    psi = torch.stack(psis)
    rho = torch.stack(rhos)
    w = torch.tensor(ws, dtype=torch.float64)
    w = w / float(w.sum())
    W = syndrome_bases()
    BS = torch.einsum('sia,nij,sjb->snab', W.conj(), rho, W)
    p_syn = torch.einsum('sn,n->s',
                         BS.diagonal(dim1=2, dim2=3).sum(-1).real, w)
    assert p_syn.dtype == w.dtype and bool(torch.all(p_syn >= -1e-15)), p_syn
    assert abs(float(p_syn.sum()) - 1) < 1e-12, float(p_syn.sum())
    meta = {'p_nodes': [float(x) for x in p_nodes],
            'p_weights': pw, 'n_u': n_u, 'n_phi': n_phi, 'n_p': n_p,
            'noise': noise,
            'p_range': (float(p_range[0]), float(p_range[1]))}
    return {'psi': psi, 'rho': rho, 'w': w, 'W': W, 'BS': BS,
            'p_syn': p_syn, 'meta': meta}


def syndrome_conditional_fidelity_exact(phi, Q):
    """EXACT (zero-variance) label-free per-syndrome conditional fidelity.

        cf_s = sum_j w_j <psi_j| R_s P_s rho_j P_s R_s^dag |psi_j> / p_s
             = F_s / p_s ,

    i.e. the same quantity as `syndrome_conditional_fidelity` but with the Haar
    average and the syndrome probabilities both replaced by the exact
    quadrature in `Q`.  This matters: the Monte-Carlo version uses M = 150
    probe states, so its sem is ~1e-3 -- comparable to or LARGER than the
    6.8e-4 non-Pauli headroom (amplitude damping, p=0.06) that the warm start is
    being selected on, and two orders of magnitude larger than the 5.4e-6
    coherent one, which makes seed selection essentially a coin flip.
    Returns (cf, p_s) as float64 arrays
    of shape (16,); syndromes with p_s <= 1e-13 are reported as cf = 1
    (unreachable branch, same convention as `vscr_paper_abl.exact_F`)."""
    with torch.no_grad():
        (_, _, F_s) = vscr_fidelity_quad(
            torch.as_tensor(phi, dtype=torch.float64).reshape(-1, PHI_DIM), Q)
        p_s = Q['p_syn']
        cf = torch.where(p_s > 1e-13, F_s / p_s.clamp_min(1e-300),
                         torch.ones_like(F_s))
    return cf.numpy().astype(float), p_s.numpy().astype(float)


def _selftest_quadrature(n_u=3, n_phi=7, n_mc=1500, seed=7):
    """Prove the quadrature is EXACT for the objective's algebraic degree.

    Test 1: the Haar rule reproduces the analytic average Tr(O)/2 of <psi|O|psi>
            over CP^1 for random Hermitian O, to machine precision.
    Test 2: `vscr_fidelity_quad` equals an independent loop over the same nodes
            using the original full-32x32 `vscr_fidelity`, on all four channels.
    Test 3: the syndrome probabilities p_s are consistent with Tr(P_s rho).
    Test 4: the exact cf_s matches a Monte-Carlo estimate on every syndrome with
            p_s > 1e-3."""
    states, sw = haar_quadrature(n_u, n_phi)
    st = states.numpy(); wt = sw.numpy()
    assert abs(wt.sum() - 1) < 1e-14, wt.sum()
    rng = np.random.RandomState(seed)
    worst = 0.0
    for _ in range(6):
        A = rng.randn(2, 2) + 1j * rng.randn(2, 2)
        O = 0.5 * (A + A.conj().T)
        q = float(np.real(np.einsum('n,ni,ij,nj->', wt, st.conj(), O, st)))
        worst = max(worst, abs(q - float(np.real(np.trace(O))) / 2))
    assert worst < 1e-14, f"Haar quadrature not exact: {worst:.2e}"
    print(f"[quad selftest 1] Haar rule exact for degree-2 Bloch polynomials "
          f"to {worst:.2e}  VERIFIED", flush=True)

    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(16, PHI_DIM, dtype=torch.float64, generator=g) * 0.6
    worst2 = worst3 = 0.0
    for noise in ('depolarizing', 'amplitude_damping', 'mixed', 'coherent'):
        Q = quad_cache((0.07, 0.07), noise, n_u=n_u, n_phi=n_phi, n_p=1)
        (_, Fq, _) = vscr_fidelity_quad(phi, Q)
        R_all = recovery_unitary_batch(phi)
        Fref = 0.0
        with torch.no_grad():
            for n in range(Q['psi'].shape[0]):
                (_, Fn) = vscr_fidelity(Q['psi'][n], Q['rho'][n], R_all)
                Fref += float(Q['w'][n]) * float(Fn)
        worst2 = max(worst2, abs(float(Fq) - Fref))
        ps_def = torch.einsum('n,nij,sji->s', Q['w'].to(Q['rho'].dtype),
                              Q['rho'],
                              P_SYNDS_STACK.to(Q['rho'].dtype)).real.numpy()
        worst3 = max(worst3, float(np.abs(ps_def - Q['p_syn'].numpy()).max()))
    assert worst2 < 1e-12, f"vscr_fidelity_quad != vscr_fidelity: {worst2:.2e}"
    assert worst3 < 1e-13, f"p_syn inconsistent: {worst3:.2e}"
    print(f"[quad selftest 2] vscr_fidelity_quad == full 32x32 reference on all "
          f"4 channels to {worst2:.2e}  VERIFIED", flush=True)
    print(f"[quad selftest 3] p_syn == sum_n w_n Tr(P_s rho_n) to {worst3:.2e}, "
          f"sums to 1  VERIFIED", flush=True)

    if n_mc:
        # Compare the UNNORMALISED branch quantities F_s and p_s, which are plain
        # means of bounded variables, rather than the ratio cf_s = F_s / p_s.
        # The ratio estimator blows up on rare syndromes (p_s ~ 1e-3 gets a
        # handful of MC hits), so a fixed tolerance on it is meaningless; a
        # z-test against the empirical MC standard error is.
        R_all = recovery_unitary_batch(phi)
        worst_z, worst_at = 0.0, None
        n_ok = n_rare = 0
        for noise in ('depolarizing', 'amplitude_damping'):
            Q1 = quad_cache((0.07, 0.07), noise, n_u=n_u, n_phi=n_phi, n_p=1)
            cf_ex, ps_ex = syndrome_conditional_fidelity_exact(phi, Q1)
            with torch.no_grad():
                _, _, F_s_ex = vscr_fidelity_quad(phi, Q1)
            F_s_ex = F_s_ex.numpy()
            rng2 = np.random.RandomState(seed)
            fs = np.zeros((n_mc, 16)); ws = np.zeros((n_mc, 16))
            with torch.no_grad():
                for j in range(n_mc):
                    pe = encode(random_logical_state(1, rng2))
                    rho = noisy_state(pe, 0.07, noise)
                    Rdpe = torch.einsum('sij,j->si',
                                        R_all.conj().transpose(1, 2), pe)
                    u_all = torch.einsum('sij,sj->si', P_SYNDS_STACK, Rdpe)
                    fs[j] = torch.einsum('si,ij,sj->s', u_all.conj(), rho,
                                         u_all).real.numpy()
                    ws[j] = torch.einsum('sij,ji->s', P_SYNDS_STACK,
                                         rho).real.numpy()
            # sum_s Tr(P_s rho) == Tr(rho) == 1 for EVERY sample: an exact
            # identity, so this validates the MC estimator itself rather than
            # relying on statistics.
            assert float(np.abs(ws.sum(axis=1) - 1.0).max()) < 1e-12
            assert abs(float(ps_ex.sum()) - 1.0) < 1e-12
            # the aggregate fidelity is well sampled -> z-test unconditionally
            tot_mc = fs.sum(axis=1)
            d = abs(float(tot_mc.mean()) - float(F_s_ex.sum()))
            if d > 1e-12 + 1e-9 * abs(float(F_s_ex.sum())):
                sem_t = float(tot_mc.std(ddof=1)) / math.sqrt(n_mc)
                z_t = d / max(sem_t, 1e-300)
                if z_t > worst_z:
                    worst_z, worst_at = z_t, (noise, 'totalF', '-')
            n_ok += 1
            # per-syndrome: only where the MC actually has enough hits.  A
            # syndrome with p_s ~ 3e-4 gets 0.2 samples at n_mc = 600, so its
            # mean carries no information and a z-score built from a near-zero
            # empirical spread is meaningless rather than alarming.
            for s in range(16):
                if ps_ex[s] * n_mc < 30.0:
                    n_rare += 1
                    continue
                for arr, ref, tag in ((fs[:, s], F_s_ex[s], 'F_s'),
                                      (ws[:, s], ps_ex[s], 'p_s')):
                    d = abs(float(arr.mean()) - float(ref))
                    n_ok += 1
                    # Agreement at round-off needs no statistics.  This matters:
                    # for the depolarizing channel Tr(P_s rho) is INDEPENDENT of
                    # the logical state, so the MC spread is pure floating-point
                    # noise (~1.6e-17) and a z-score formed from it reports
                    # z = 58 for a discrepancy of 2e-17.
                    if d <= 1e-12 + 1e-9 * abs(float(ref)):
                        continue
                    sem = float(arr.std(ddof=1)) / math.sqrt(n_mc)
                    z = d / max(sem, 1e-300)
                    if z > worst_z:
                        worst_z, worst_at = z, (noise, s, tag)
        assert worst_z < 6.0, \
            f"exact quadrature disagrees with MC({n_mc}) at {worst_at}: z={worst_z:.2f}"
        print(f"[quad selftest 4] exact quadrature matches MC({n_mc}) on "
              f"depolarizing + amplitude_damping: {n_ok} z-tests (aggregate F "
              f"plus every syndrome with >= 30 expected MC hits), worst "
              f"z = {worst_z:.2f} at {worst_at}; {n_rare} syndrome/channel "
              f"pairs too rare for MC (p_s*n < 30) skipped  VERIFIED", flush=True)
        print(f"[quad selftest 4b] sum_s Tr(P_s rho_j) == Tr(rho_j) == 1 exactly "
              f"for every MC sample, and sum_s p_s == 1  VERIFIED", flush=True)
        print(f"[quad selftest 4c] cf_s == F_s / p_s reproduced to "
              f"{np.abs(cf_ex - np.where(ps_ex > 1e-13, F_s_ex / np.maximum(ps_ex, 1e-300), 1.0)).max():.2e}",
              flush=True)


# ---------------------------------------------------------------------
# VSCR training  (real Adam gradients through the differentiable simulator)
# ---------------------------------------------------------------------
def train_vscr(model, n_epochs=400, batch=48, p_range=(0.02, 0.15),
               noise='depolarizing', lr=1e-2, lam=0.1, rng_seed=SEED,
               use_real_grad=True, verbose=True, synd_w=None, quad=False):
    """Train VSCR.  If use_real_grad=False, parameters get RANDOM Gaussian
    updates (replicating the broken ACE-QEC reference in main.py) to ablate
    the importance of genuine gradients.  synd_w: optional per-syndrome loss
    weights (gradient balancing; see vscr_fidelity).

    quad=True replaces the Monte-Carlo probe-state average by the EXACT
    deterministic Haar/noise-strength quadrature of `haar_quadrature` /
    `p_quadrature` (see the comment block above those functions), evaluated by
    `vscr_fidelity_quad`.  Two reasons this matters for Fig.3/Fig.4:

      * Accuracy.  The signal those figures are meant to resolve -- the
        non-Pauli headroom of the optimal syndrome-conditioned unitary over the
        rigid Pauli decoder -- spans 5.4e-6 (coherent, p=0.06) to 3.1e-3
        (coherent, p=0.30), with 6.8e-4 on amplitude damping at p=0.06.  The
        measured per-sample spread of the
        recovery fidelity gives a 48-sample Monte-Carlo SEM of 1.1e-4 at
        amplitude damping p=0.10 and 3.2e-5 at p=0.05, i.e. the estimator noise
        is the same order as the effect for exactly the channel where the effect
        is smallest.  The quadrature gradient is zero-variance and zero-bias.
      * Speed.  The 126 (psi, rho) pairs are cached once per call, and because
        <psi|R_s P_s rho P_s R_s^dag|psi> only ever probes the 2-dimensional
        syndrome subspace, the 60-gate ansatz is applied to 2 columns instead of
        a full 32x32 unitary (see `vscr_fidelity_quad`).

    The default remains False so that pre-existing Monte-Carlo runs reproduce
    bit-for-bit -- including the RNG stream, which is why `val_psi` is still
    drawn (and discarded) in MC mode.

    The p = 0.06 model-selection criterion (`history['fidelity']`) is evaluated
    with the exact quadrature in BOTH modes, so MC and quad runs are selected by
    the same deterministic, label-free number."""
    rng = np.random.RandomState(rng_seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr) if use_real_grad else None
    history = {'epoch': [], 'fidelity': [], 'loss': []}
    val_p = 0.06
    if quad:
        assert lam == 0.0, ('the quadrature path does not support the '
                            'code-population regulariser; use lam=0.0')
    QT = quad_cache(p_range, noise) if quad else None
    QV = quad_cache((val_p, val_p), noise, n_p=1)
    val_psi = None if quad else [random_logical_state(1, rng) for _ in range(30)]
    for epoch in range(n_epochs):
        model.train()
        if use_real_grad:
            opt.zero_grad()
            if QT is not None:
                # EXACT quadrature objective: deterministic, zero-variance
                loss, F, _ = vscr_fidelity_quad(model(), QT, synd_w=synd_w)
                loss.backward()
                opt.step()
                avg_loss = float(loss.detach())
                tr_f = float(F)
            else:
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
                if QT is not None:
                    _, Fq, _ = vscr_fidelity_quad(model(), QT, synd_w=synd_w)
                    tr_f = float(Fq)
                else:
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
        # model selection uses the EXACT quadrature in BOTH modes, so MC and
        # quad runs are compared on the same deterministic label-free number
        with torch.no_grad():
            _, vf_t, _ = vscr_fidelity_quad(model(), QV)
            vf = float(vf_t)
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

