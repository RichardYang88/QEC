#!/usr/bin/env python3
"""vscr_general.py -- the reduced-branch VSCR machinery for ARBITRARY stabilizer codes.

WHY THIS FILE EXISTS
--------------------
`ssvr_qec.py` hard-codes the [[5,1,3]] perfect code at module level (N=5,
DIM=32, 16 syndromes, and a `SYND_TABLE` whose construction *asserts*
perfectness).  That is the right choice for the production training pipeline,
but it leaves the natural reviewer question -- "does any of this survive a
larger code?" -- unanswerable from the existing artifacts.

The observation that makes it answerable is that the *entire* exact evaluation
stack of the paper is code-size independent once it is written on the
two-dimensional syndrome subspaces.  With

    V   : (DIM, 2)   encoding isometry (columns |0_L>, |1_L>),
    W_s : (DIM, 2)   orthonormal basis of image(P_s),
    K_k : (DIM, DIM) Kraus operators of the noise channel,
    A_k : (2, 2)  =  W_s^dagger K_k V          <-- the reduced branch operator,

the Haar-averaged branch fidelity of any recovery whose physical unitary R_s
satisfies V^dagger R_s W_s = G_s is the CLOSED FORM used throughout the paper

    cf_s(G_s) = ( J_s(G_s) + const_s ) / 6 / p_s ,
    J_s(G_s)  = sum_k |Tr(G_s A_k)|^2 = g_s^dagger Q_s g_s ,   g_s = vec(G_s) ,
    const_s   = sum_k Tr(A_k^dagger A_k) = Tr(Q_s) ,           p_s = const_s/2 ,

so *everything* the paper certifies -- the exact decoder fidelity, the
per-branch-unitary ceiling (the optimum of the family VSCR occupies), and the
optimal-CPTP ceiling -- is a function of the 2x2 blocks {A_k} alone.  Neither a
DIM x DIM density matrix, nor a training run, nor the 60-gate ansatz enters.
DIM = 32 / 128 / 512 therefore costs the same 4x4 Rayleigh quotient and the
same 4x4 Choi program per branch; only the construction of A_k grows.

WHAT IS COMPUTED EXACTLY (no Monte Carlo anywhere)
--------------------------------------------------
  decoder fidelity       F_dec  = sum_s p_s cf_s(G_dec,s)
  unitary-family ceiling F_unit = sum_s p_s cf_s(argmax_{G in U(2)} J_s)
  CPTP ceiling           F_CPTP = sum_s p_s * 2 Tr[C*_s M_s]   (4x4 Choi SDP)
  Petz recovery          F_Petz = exact Haar average of the noise-adapted map
  readout error eta      exact, via the cross-branch blocks (see below)

SCALING TRICKS (both asserted against the brute-force path at n=5)
------------------------------------------------------------------
* Pauli-type channels never form a DIM x DIM Kraus operator.  Writing a Pauli
  as an index vector over {I,X,Y,Z} = {0,1,2,3}, the product of two Paulis has
  index `a XOR b` per qubit, and a Pauli E with syndrome s satisfies
  P_s E V = E V, hence

      A_E = W_s^dagger (sqrt(w_E) E) V = sqrt(w_E) * Lam_{C_s XOR E} ,
      Lam_F = V^dagger F V   for F in the centralizer C(S)  (syndrome 0).

  There are only 2^(n+k) distinct Lam_F, shared by ALL 2^(n-k) branches, and
  weight(C_s * E) = #{q : (C_s)_q != E_q}.  So the 4^n Kraus operators of
  depolarizing noise collapse to 2^(n+k) cheap O(dim) permutations plus a
  vectorised weight histogram -- which is what makes [[9,1,3]] affordable.
  Every quantity built this way is invariant under the global phase of Lam_F
  (Q_s, const_s and |Tr(G A_k)|^2 all are), so dropping the product phase is
  exact, not an approximation.
* Non-Pauli channels (amplitude damping, coherent) use the direct path
  A = W_all^dagger (K_k V), applying K_k to the two codeword columns only.

READOUT ERROR, EXACTLY
----------------------
A mid-circuit measurement error on any of the n-k ancillas maps the true
syndrome s to an observed s~ with probability T(s~|s), and the instrument then
applies R_s~ instead of R_s.  The reduced cross-branch block is

      G~_{s->s~} = V^dagger R_s~ W_s ,
      F(eta) = sum_{s,s~} T(s~|s) sum_k [ |Tr(G~ A_k)|^2
                                          + Tr(G~ A_k A_k^dag G~^dag) ] / 6 ,

which reduces to the paper's cf_s when s~ = s and G~ is unitary.  For a PAULI
decoder the cross terms vanish identically, because C_s~ C_s carries syndrome
s~ XOR s != 0 and therefore maps the code space orthogonal to itself:
G~ = Lam_{C_s~ XOR C_s} = 0 unless s~ = s.  Readout error is then an exact
multiplicative suppression,

      F_dec(eta) = (1-eta)^(n-k) * F_dec(0) ,

and the same holds for ANY branch recovery completed so that it does not map
image(W_s~) into the code space for s != s~.  `_selftest_readout` verifies the
vanishing of every cross block numerically, and `scaling_analysis.py` checks
the law against an independent full-density-matrix simulation at n=5.

This module deliberately duplicates ~30 lines of quadratic-form bookkeeping
from `vscr_paper_abl.py` rather than importing it, because that module binds
`W_BASIS` / `V_ISO` / `_branch_A` to n=5 at import time.  `_selftest_against_paper`
is the guard: at n=5 the Q_s, const_s, p_s and M_s built here must equal the
production ones entry by entry, and the resulting decoder / unitary / CPTP
ceilings must reproduce the audited values in `paper_numbers.json`.
"""
import itertools
import math
import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
# Fixed coretype, for the same reason and with the same value as
# vscr_paper_abl.py: StabCode's logical basis comes from np.linalg.eigh on
# operators with 2^(n-1)-fold DEGENERATE eigenspaces, so the eigenvector basis
# returned inside a degenerate subspace depends on the LAPACK kernel selected --
# and with it W_s and every reduced block A_k.  Pinning the coretype makes the
# degenerate-subspace basis bit-reproducible across runs.  It is NOT a fix for
# the native faults below (those reproduce under every coretype); it is purely
# for determinism of the scaling numbers.
os.environ.setdefault('OPENBLAS_CORETYPE', 'HASWELL')


def _pin_single_cpu(prefer_p_cores=8):
    """Pin this process to ONE cpu before numpy or any native kernel loads.

    Same mitigation, same rationale and same core-selection rule as
    `ssvr_qec._pin_single_cpu` -- see the long "THE NATIVE FAULT" note at the
    top of that module.  Duplicated rather than imported so that this file has
    no dependency on torch/ssvr_qec (it is pure numpy), which matters because
    the scaling analysis is a long-running native workload and this host
    intermittently SIGSEGV/SIGILLs when a process is free to migrate across its
    hybrid P-/E-core cluster."""
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

import numpy as np                                              # noqa: E402

# I=0, X=1, Y=2, Z=3 : the product of two Paulis has index a XOR b per qubit.
I_, X_, Y_, Z_ = 0, 1, 2, 3
_MAT = {
    I_: np.eye(2, dtype=complex),
    X_: np.array([[0, 1], [1, 0]], dtype=complex),
    Y_: np.array([[0, -1j], [1j, 0]], dtype=complex),
    Z_: np.array([[1, 0], [0, -1]], dtype=complex),
}
_CHR = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}


def _str_to_idx(s):
    return np.array([{'I': I_, 'X': X_, 'Y': Y_, 'Z': Z_}[c] for c in s],
                    dtype=np.int8)


def _idx_to_str(a):
    return ''.join(_CHR[int(v)] for v in a)


def _mat_of(a):
    """Full 2^n x 2^n Pauli matrix from an index vector (q0 = leftmost = MSB)."""
    M = _MAT[int(a[0])]
    for v in a[1:]:
        M = np.kron(M, _MAT[int(v)])
    return M


# ---------------------------------------------------------------------------
# the three codes used in the scaling analysis
# ---------------------------------------------------------------------------
CODES = {
    # [[5,1,3]] perfect code -- generators identical to ssvr_qec.STAB_STR
    '5,1,3': dict(stab=['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'],
                  xl='XXXXX', zl='ZZZZZ'),
    # [[7,1,3]] Steane code from the self-dual [7,4,3] Hamming code: the X- and
    # Z-generators share the three parity-check supports {0,2,4,6}, {1,2,5,6},
    # {3,4,5,6}, which pairwise overlap evenly (H H^T = 0), so all six commute.
    '7,1,3': dict(stab=['XIXIXIX', 'IXXIIXX', 'IIIXXXX',
                        'ZIZIZIZ', 'IZZIIZZ', 'IIIZZZZ'],
                  xl='XXXXXXX', zl='ZZZZZZZ'),
    # [[9,1,3]] Shor code: three GHZ blocks {0,1,2},{3,4,5},{6,7,8}; two Z-type
    # generators inside each block (Z0Z1,Z1Z2 | Z3Z4,Z4Z5 | Z6Z7,Z7Z8) and two
    # X-type generators comparing adjacent blocks (X0..X5, X3..X8).  Every X-type
    # generator meets every Z-type one on an even number of qubits, so the eight
    # commute; Z_L = Z0 Z3 Z6 has weight 3, fixing the distance.
    '9,1,3': dict(stab=['ZZIIIIIII', 'IZZIIIIII', 'IIIZZIIII', 'IIIIZZIII',
                        'IIIIIIZZI', 'IIIIIIIZZ',
                        'XXXXXXIII', 'IIIXXXXXX'],
                  xl='XXXXXXXXX', zl='ZIIZIIZII'),
}


class StabCode:
    """An [[n,1,3]] stabilizer code plus its exact reduced-branch data.

    Construction asserts, in order: mutual commutation of the generators, that
    X_L/Z_L commute with them and anticommute with each other, that the code
    space has rank 2, that the distance is exactly 3, that every W_s = C_s V is
    an orthonormal basis of image(P_s), and that the minimum-weight decoder
    table reproduces each syndrome it is indexed by.  A malformed code therefore
    fails loudly here rather than silently biasing a ceiling."""

    def __init__(self, spec):
        stab = [_str_to_idx(s) for s in spec['stab']]
        self.n = len(stab[0])
        self.dim = 2 ** self.n
        self.m = len(stab)                 # n-k stabilizer generators
        assert self.m == self.n - 1, 'this module handles k=1 codes only'
        self.stab = np.stack(stab)         # (m, n)
        self.xl = _str_to_idx(spec['xl'])
        self.zl = _str_to_idx(spec['zl'])
        self.nsyn = 2 ** self.m

        # ---- every Pauli on n qubits, as index vectors + syndromes --------
        P = np.array(list(itertools.product((0, 1, 2, 3), repeat=self.n)),
                     dtype=np.int8)                    # (4^n, n)
        self.pauli = P
        self.weight = (P != 0).sum(1).astype(np.int8)
        self.syn = self._syndromes(P)                  # (4^n,) integer syndrome
        self._check_code()

        # ---- code space isometry V (dim, 2) ------------------------------
        self.V = self._logical_basis()

        # ---- minimum-weight lookup decoder C_s, and W_s = C_s V ----------
        self.C_idx, self.C_mat = self._decoder()
        self.W = np.stack([self.C_mat[s] @ self.V for s in range(self.nsyn)])
        for s in range(self.nsyn):          # W_s^dagger W_s = I_2
            d = np.abs(self.W[s].conj().T @ self.W[s] - np.eye(2)).max()
            assert d < 1e-12, (s, d)
        # Wall_dag stacks the W_s^dagger row-blocks: rows 2s, 2s+1 = W_s^dagger.
        # NB dim == 2*nsyn for every code here, so a mis-ordered reshape is
        # dimensionally legal and would silently scramble the branches; the
        # layout assert below is what catches that.
        Wall = self.W.transpose(1, 0, 2).reshape(self.dim, 2 * self.nsyn)
        self.Wall_dag = Wall.conj().T                  # (2*nsyn, dim)
        assert np.abs(self.Wall_dag.reshape(self.nsyn, 2, self.dim)
                      .transpose(0, 2, 1) - self.W.conj()).max() < 1e-14, \
            'Wall_dag row-block layout is wrong'


        # ---- centralizer C(S) = syndrome-0 Paulis, and Lam_F = V^dag F V --
        cz = np.flatnonzero(self.syn == 0)
        assert cz.size == 2 ** (self.n + 1), (cz.size, 2 ** (self.n + 1))
        self.cz = cz
        self._lam_cache = {}

    # -- Pauli algebra ----------------------------------------------------
    def _syndromes(self, P):
        """Integer syndrome per Pauli: bit i = 1 iff P anticommutes with g_i.

        Two Paulis anticommute on qubit q iff both are non-identity and
        different, so the bit is the parity of #{q : P_q != 0, g_q != 0,
        P_q != g_q}.  Generator 0 is the most significant bit, matching
        ssvr_qec.SYND_BITS so the n=5 decoder tables compare entry by entry.
        Vectorised over the rows of P; a 1-D row is returned as a scalar."""
        P = np.asarray(P)
        one = (P.ndim == 1)
        if one:
            P = P[None, :]
        out = np.zeros(P.shape[0], dtype=np.int64)
        for i in range(self.m):
            g = self.stab[i]
            anti = ((P != 0) & (g != 0) & (P != g)).sum(1) & 1
            out |= anti.astype(np.int64) << (self.m - 1 - i)
        return int(out[0]) if one else out

    def _check_code(self):
        """Assert the Pauli data really describes a distance-3 [[n,1,3]] code."""
        S = self.stab
        for i in range(self.m):                      # (a) generators commute
            for j in range(i + 1, self.m):
                a = ((S[i] != 0) & (S[j] != 0) & (S[i] != S[j])).sum()
                assert a % 2 == 0, f'generators {i},{j} anticommute'
        for L in (self.xl, self.zl):                 # (b) logicals commute ...
            for i in range(self.m):
                a = ((L != 0) & (S[i] != 0) & (L != S[i])).sum()
                assert a % 2 == 0, 'logical does not commute with a generator'
        a = ((self.xl != 0) & (self.zl != 0) & (self.xl != self.zl)).sum()
        assert a % 2 == 1, 'X_L and Z_L must anticommute'
        # (c) distance exactly 3: min weight over C(S) \\ S, identity excluded.
        # S is the whole GROUP, not just the m generators -- products such as
        # (Z0Z1)(Z1Z2) = Z0Z2 have weight 2 and are stabilizers, so testing only
        # against the generator list would report d=2 for Shor's code.
        grp = self.stabilizer_group()
        assert len(grp) == 2 ** self.m, (len(grp), 2 ** self.m)
        rows = np.flatnonzero(self.syn == 0)
        w = [int(self.weight[r]) for r in rows
             if tuple(int(v) for v in self.pauli[r]) not in grp]
        self.distance = min(w)
        assert self.distance == 3, f'expected distance 3, got {self.distance}'
        # the logical coset structure: |C(S)| = 4 |S| for k=1
        assert rows.size == 4 * len(grp), (rows.size, len(grp))

    def stabilizer_group(self):
        """The full stabilizer group as a set of Pauli index tuples (phase-free).

        Built by GF(2) linear algebra on the symplectic representation
        (x-part | z-part), where x_q = 1 for X/Y and z_q = 1 for Z/Y.  All 2^m
        subsets of the generators are XORed, which is exact and cheap (2^8 = 256
        for Shor's code).  Phase is irrelevant here: weight is, and every subset
        product of a valid abelian stabilizer set is a group element."""
        n = self.n
        def to_symp(a):
            a = np.asarray(a)
            x = ((a == X_) | (a == Y_)).astype(np.int8)
            z = ((a == Z_) | (a == Y_)).astype(np.int8)
            return np.concatenate([x, z])

        def from_symp(v):
            out = np.zeros(n, dtype=np.int8)
            for q in range(n):
                x, z = int(v[q]), int(v[n + q])
                out[q] = (0 if (x, z) == (0, 0) else
                          1 if (x, z) == (1, 0) else
                          2 if (x, z) == (1, 1) else 3)
            return tuple(int(t) for t in out)

        Gm = np.stack([to_symp(self.stab[i]) for i in range(self.m)])
        subs = np.array(list(itertools.product((0, 1), repeat=self.m)),
                        dtype=np.int8)                     # (2^m, m)
        elems = (subs.astype(np.int64) @ Gm.astype(np.int64)) % 2
        got = {from_symp(e) for e in elems}
        # every group element must carry the trivial syndrome
        for t in got:
            assert self._syndromes(np.array(t, dtype=np.int8)) == 0, t
        return got


    def _logical_basis(self):
        """|0_L>, |1_L> as columns of a (dim,2) isometry, same convention as
        ssvr_qec._logical_basis: |0_L> the +1 eigenstate of Z_L in the code
        space, |1_L> = X_L |0_L>, so X_L flips the columns exactly and no global
        phase is left ambiguous."""
        dim = self.dim
        I = np.eye(dim, dtype=complex)
        Pc = I.copy()
        for i in range(self.m):
            Pc = Pc @ (0.5 * (I + _mat_of(self.stab[i])))
        Pc = 0.5 * (Pc + Pc.conj().T)
        assert np.linalg.matrix_rank(Pc) == 2, 'code space is not rank 2'
        ZL, XL = _mat_of(self.zl), _mat_of(self.xl)
        Mh = Pc @ ZL @ Pc
        Mh = 0.5 * (Mh + Mh.conj().T)
        ev, evec = np.linalg.eigh(Mh)
        z0 = evec[:, int(ev.argmax())]
        j = int(np.abs(z0).argmax())
        z0 = z0 * np.exp(-1j * np.angle(z0[j]))
        z0 = z0 / np.linalg.norm(z0)
        x0 = XL @ z0
        x0 = x0 / np.linalg.norm(x0)
        V = np.stack([z0, x0], axis=1)
        assert np.abs(V.conj().T @ V - np.eye(2)).max() < 1e-12
        assert np.abs(XL @ V[:, 0] - V[:, 1]).max() < 1e-12
        assert np.abs(ZL @ V[:, 0] - V[:, 0]).max() < 1e-12
        assert np.abs(Pc @ V - V).max() < 1e-12, 'logical basis left the code space'
        return V

    def _decoder(self):
        """Minimum-weight Pauli lookup decoder C_s, one per syndrome.

        Ties (inevitable on degenerate codes -- on Shor's, Z_0, Z_1 and Z_2 all
        give the same syndrome) are broken by `np.argsort(kind='stable')` over
        the lexicographically ordered Pauli list, so the table is deterministic.
        A degenerate tie is harmless: two Paulis with the same syndrome differ by
        an element of C(S), and the branch quantities below are invariant under
        that choice, which `scaling_analysis.py` re-checks by re-decoding with
        the tie order reversed."""
        C_idx = np.zeros((self.nsyn, self.n), dtype=np.int8)
        order = np.argsort(self.weight, kind='stable')
        seen = {}
        for r in order:
            s = int(self.syn[r])
            if s not in seen:
                seen[s] = int(r)
            if len(seen) == self.nsyn:
                break
        assert len(seen) == self.nsyn, f'only {len(seen)} syndromes reachable'
        for s, r in seen.items():
            C_idx[s] = self.pauli[r]
        assert np.array_equal(self._syndromes(C_idx), np.arange(self.nsyn)), \
            'decoder table does not reproduce its own syndrome index'
        C_mat = np.stack([_mat_of(C_idx[s]) for s in range(self.nsyn)])
        return C_idx, C_mat

    # -- reduced branch operators -----------------------------------------
    def lam(self, F):
        """Lam_F = V^dagger F V for a syndrome-0 Pauli index vector F.

        F acts on the computational basis as F|b> = phi(b)|b XOR f>, so F V is a
        signed row-permutation of V and Lam_F costs O(dim), not O(dim^3).
        phi(b) = i^{#Y} * (-1)^{sum of b_q over q with F_q in {Y,Z}}, using
        q0 = most significant bit (the kron order of `_mat_of`)."""
        key = tuple(int(v) for v in F)
        hit = self._lam_cache.get(key)
        if hit is not None:
            return hit
        f_int, nY, zbits = 0, 0, []
        for q, v in enumerate(key):
            bit = 1 << (self.n - 1 - q)
            if v in (X_, Y_):
                f_int |= bit
            if v in (Y_, Z_):
                zbits.append(bit)
            if v == Y_:
                nY += 1
        idx = np.arange(self.dim)
        par = np.zeros(self.dim, dtype=np.int64)
        for bit in zbits:
            par ^= ((idx & bit) != 0).astype(np.int64)
        phi = (1j ** nY) * np.where(par.astype(bool), -1.0, 1.0)
        FV = np.empty_like(self.V)
        FV[idx ^ f_int] = phi[:, None] * self.V
        L = self.V.conj().T @ FV
        # F preserves the code space, so Lam_F must be unitary
        assert np.abs(L.conj().T @ L - np.eye(2)).max() < 1e-10, (key, L)
        self._lam_cache[key] = L
        return L

    def branch_A_pauli(self, pauli_idx, amp):
        """{A_k} per branch for a Pauli-mixture channel.

        `pauli_idx` is a (K, n) index array and `amp` a (K,) array of Kraus
        amplitudes (the sqrt of each probability).  A_E = amp_E * Lam_{C_s XOR E}
        lands on branch s = syndrome(E) only, so the cost is K cheap 2x2 blocks
        rather than K * nsyn.  Exact for any Pauli-mixture channel."""
        pauli_idx = np.asarray(pauli_idx)
        amp = np.asarray(amp, dtype=float)
        syn = self._syndromes(pauli_idx)
        A = [[] for _ in range(self.nsyn)]
        C_idx, lam = self.C_idx, self.lam
        for r in np.flatnonzero(amp != 0.0):
            s = int(syn[r])
            A[s].append(float(amp[r]) * lam(np.bitwise_xor(C_idx[s],
                                                          pauli_idx[r])))
        return A

    def apply_product_ops(self, cols, per_q):
        """Apply every product of the per-qubit operator lists to each column block.

        `cols` is (K0, dim, r); `per_q` is a list of n lists of 2x2 operators.
        Returns (K0 * prod_q |per_q[q]|, dim, r).  Each single-qubit operator is
        applied to the r columns by tensor contraction, so the cost is
        O(K * n * dim * r) instead of the O(K * dim^3) of forming full-space
        Kraus operators -- and it composes, which is what the depolarizing-then-
        damping 'mixed' channel needs."""
        n = self.n
        cur = [np.asarray(cols[i], dtype=complex) for i in range(len(cols))]
        for q in range(n):
            out = []
            for op2 in per_q[q]:
                T = np.asarray(op2, dtype=complex)
                for C in cur:
                    R = C.reshape([2] * n + [-1])
                    R = np.moveaxis(np.tensordot(T, R, axes=([1], [q])), 0, q)
                    out.append(R.reshape(self.dim, -1))
            cur = out
        return np.stack(cur)

    def product_kraus_V(self, per_q):
        """K_k V for a product channel, from n lists of single-qubit Kraus ops.

        Thin wrapper over `apply_product_ops` starting from the two codeword
        columns.  Returns (K, dim, 2)."""
        return self.apply_product_ops(self.V[None, :, :], per_q)


    def branch_A_dense(self, KV):
        """{A_k} per branch from precomputed K_k V columns (non-Pauli channels).

        KV is (K, dim, 2).  `Wall_dag @ KV` broadcasts to (K, 2*nsyn, 2) -- rows
        2s and 2s+1 are the two rows of W_s^dagger -- so reshaping to
        (K, nsyn, 2, 2) puts the whole block A_k(s) = W_s^dagger K_k V at
        [k, s].  Returns (list-of-nsyn lists of (2,2), full)."""
        KV = np.asarray(KV, dtype=complex)
        K = KV.shape[0]
        full = np.matmul(self.Wall_dag, KV)            # (K, 2*nsyn, 2)
        assert full.shape == (K, 2 * self.nsyn, 2), full.shape
        full = full.reshape(K, self.nsyn, 2, 2)
        return [[full[k, s] for k in range(K)] for s in range(self.nsyn)], full

    def cross_G(self, R_mats):
        """Cross-branch blocks G~[s~, s] = V^dagger R_s~ W_s for physical R_s~.

        `R_mats` is (nsyn, dim, dim); the result is (nsyn~, nsyn, 2, 2).
        Contracting V^dagger against R_s~ FIRST keeps the cost at
        O(nsyn * dim^2) instead of the O(nsyn^2 * dim^2) of forming every
        R_s~ W_s, which is what makes the full 256x256 cross-block table of
        [[9,1,3]] affordable.  For Pauli recoveries every off-diagonal block
        must vanish identically, because C_s~ C_s carries syndrome s~ XOR s != 0
        and so maps the code space orthogonal to itself; that is what makes
        readout error an exact multiplicative suppression (module docstring)."""
        VR = np.matmul(self.V.conj().T, np.asarray(R_mats))   # (nsyn~, 2, dim)
        G = np.matmul(VR[:, None, :, :], self.W[None, :, :, :])
        assert G.shape == (self.nsyn, self.nsyn, 2, 2), G.shape
        return G

# ---------------------------------------------------------------------------
# quadratic forms -- identical bookkeeping to vscr_paper_abl._branch_qform /
# _branch_M, duplicated here because those bind W_BASIS/V_ISO to n=5 at import.
# `_selftest_against_paper` asserts the two agree entry by entry at n=5.
# ---------------------------------------------------------------------------
def qform(A_list):
    """(Q_s, const_s, p_s) from the reduced branch operators."""
    Q = np.zeros((4, 4), dtype=complex)
    const = 0.0
    for A in A_list:
        A = np.asarray(A, dtype=complex)
        v = A.T.reshape(4)
        Q += np.outer(v, v.conj())
        const += float(np.real(np.trace(A.conj().T @ A)))
    Q = 0.5 * (Q + Q.conj().T)
    return Q, const, float(np.real(np.trace(Q))) / 2


def branch_M(A_list):
    """(M_s, p_s) for the optimal-CPTP 4x4 Choi program, as in _branch_M."""
    S1 = np.zeros((2, 2), dtype=complex)
    VV = np.zeros((4, 4), dtype=complex)
    for A in A_list:
        A = np.asarray(A, dtype=complex)
        S1 += A.conj() @ A.T
        w = A.conj().reshape(4) / math.sqrt(2)
        VV += np.outer(w, w.conj())
    M = (np.kron(S1, np.eye(2)) + 2.0 * VV) / 6.0
    p_s = float(np.real(np.trace(S1))) / 2.0
    return M, p_s


def cf_of(G, Q, const, p_s):
    """Exact Haar-averaged conditional fidelity of a 2x2 branch recovery G."""
    g = np.asarray(G, dtype=complex).reshape(4)
    J = float(np.real(g @ Q @ g.conj()))
    return (J + const) / 6.0 / p_s


def cf_unnormalised(Gt, A_list):
    """Exact unnormalised branch fidelity for a possibly NON-unitary block Gt.

    E_psi |<psi| G A_k |psi>|^2 = (|Tr(G A_k)|^2 + Tr(G A_k A_k^dag G^dag))/6,
    the degree-2 Haar moment on CP^1.  For unitary Gt the second term collapses
    to Tr(A_k^dag A_k) and p_s * this equals `cf_of`; `_selftest_readout` asserts
    that identity, which is what licenses using this form for the vanishing
    cross-branch blocks produced by a readout error."""
    tot = 0.0
    for A in A_list:
        G = np.asarray(Gt, dtype=complex) @ np.asarray(A, dtype=complex)
        tot += (abs(np.trace(G)) ** 2
                + float(np.real(np.trace(G @ G.conj().T)))) / 6.0
    return tot


# ---------------------------------------------------------------------------
# noise channels, as (Pauli index array, amplitudes) or as per-qubit Kraus lists
# ---------------------------------------------------------------------------
def depolarizing_paulis(code, p):
    """Pauli-mixture data of per-qubit-sequential depolarizing noise of rate p.

    Exactly the channel ssvr_qec.apply_depolarizing implements: each qubit gets
    I with weight 1-p and X/Y/Z with weight p/3, so a Pauli of weight w carries
    (1-p)^(n-w) (p/3)^w."""
    w = code.weight.astype(float)
    prob = (1.0 - p) ** (code.n - w) * (p / 3.0) ** w
    return code.pauli, np.sqrt(prob)


def amp_damping_kraus_V(code, gamma):
    """K_k V for independent amplitude damping of rate gamma on every qubit."""
    K0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma)]], dtype=complex)
    K1 = np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
    return code.product_kraus_V([[K0, K1] for _ in range(code.n)])


def coherent_unitary(code, eps):
    """The single Kraus operator of a systematic Rx(eps) over-rotation.

    Identical to ssvr_qec.apply_coherent: the same Rx(eps) on every qubit.
    Those act on disjoint qubits and so commute, hence the ordered product of
    embedded rotations that `apply_coherent` forms is exactly the plain tensor
    product -- which is what is built here, at O(dim^2) instead of the
    O(n * dim^3) of multiplying n embedded full-space rotations."""
    Rx = np.array([[math.cos(eps / 2), -1j * math.sin(eps / 2)],
                   [-1j * math.sin(eps / 2), math.cos(eps / 2)]], dtype=complex)
    U = np.array([[1.0 + 0j]])
    for _ in range(code.n):
        U = np.kron(U, Rx)
    assert U.shape == (code.dim, code.dim)
    return U


def branch_A(code, channel, p):
    """Dispatch to the reduced branch operators {A_k} of a channel at rate p.

    channel in {'depolarizing', 'amplitude_damping', 'mixed', 'coherent'}.
    'mixed' is ssvr_qec.apply_mixed: depolarizing(p) THEN amplitude damping
    (p/2), so its Kraus set is the composition of the two."""
    if channel == 'depolarizing':
        idx, amp = depolarizing_paulis(code, p)
        return code.branch_A_pauli(idx, amp)
    if channel == 'amplitude_damping':
        return code.branch_A_dense(amp_damping_kraus_V(code, p))[0]
    if channel == 'coherent':
        KV = coherent_unitary(code, p) @ code.V
        return code.branch_A_dense(KV[None, :, :])[0]
    if channel == 'mixed':
        # N = AD(p/2) o Dep(p), i.e. ssvr_qec.apply_mixed.  The composed Kraus
        # set has 8^n elements and cannot be factorised through the Pauli trick
        # (only the depolarizing half is Pauli), so this path is dense and is
        # deliberately restricted to the sizes where it fits: 8^n * dim * 2
        # complex128 entries for the K_k V stack.
        Ktot = 8 ** code.n
        assert Ktot * code.dim * 2 <= 4_000_000, (
            f"'mixed' needs a dense {Ktot}-element Kraus set ({code.n} qubits); "
            f"supported up to n=5. The scaling sweep therefore omits it.")
        dep_q = [[math.sqrt(1 - p) * np.eye(2, dtype=complex)]
                 + [math.sqrt(p / 3.0) * _MAT[v] for v in (X_, Y_, Z_)]
                 for _ in range(code.n)]
        gam = 0.5 * p
        ad_q = [[np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gam)]],
                          dtype=complex),
                 np.array([[0.0, math.sqrt(gam)], [0.0, 0.0]], dtype=complex)]
                for _ in range(code.n)]
        KVd = code.product_kraus_V(dep_q)              # Dep(p) applied to V
        KV = code.apply_product_ops(KVd, ad_q)         # then AD(p/2)
        assert KV.shape[0] == Ktot, (KV.shape, Ktot)
        return code.branch_A_dense(KV)[0]
    raise ValueError(channel)






