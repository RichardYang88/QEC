#!/usr/bin/env python3
"""ancilla_recovery.py -- how much of the CPTP headroom does ONE ancilla buy?

THE QUESTION, AND WHY IT MATTERS.  `scaling_analysis.py` certifies three numbers
per code x channel x strength: the decoder fidelity F_dec, the optimum over
per-syndrome UNITARY branch recoveries F_unit (the family VSCR compiles to), and
the optimum over ALL per-syndrome CPTP branch recoveries F_cptp (a 4x4 Choi
program).  On amplitude damping those three order themselves differently as the
code grows.  At n=5 and n=7 the unitary family already attains the CPTP ceiling
(F_unit == F_cptp to 1e-9), so a gate-level branch instrument captures the whole
non-Pauli advantage.  At n=9 it does not: F_unit == F_dec to 1.1e-16 -- the
unitary headroom is EXACTLY zero -- while F_cptp sits above both by 7.3e-6 at
p=0.05 and 2.1e-5 at p=0.10.  The paper's discussion concludes that "the residual
gain requires a non-unitary branch recovery that a gate-level instrument cannot
express".  This module tests that conclusion instead of asserting it.

THE OBSERVATION.  "Non-unitary" is not the same as "not gate-level".  A CPTP map
Phi(rho) = sum_j B_j rho B_j^dag on the two-dimensional logical space is
realised physically by a Stinespring dilation: append an ancilla of dimension r,
apply a UNITARY, and discard the ancilla.  The minimal ancilla dimension equals
the Kraus rank of Phi, which equals rank(C) for the Choi matrix
C = sum_j |B_j>><<B_j|.  So the families already in the codebase are the first
two rungs of a single ladder indexed by ancilla count:

    rank(C) = 1  <->  Phi unitary             <->  ZERO ancillas  (= F_unit)
    rank(C) = 2  <->  one-ancilla instrument   <->  ONE ancilla    (new)
    rank(C) = 3  <->  ...                      <->  rank-3 dilation
    rank(C) = 4  <->  arbitrary CPTP           <->  TWO ancillas   (= F_cptp)

`vscr_paper_abl._sdp_branch` parameterises the Choi matrix as C = L L^dag with
L a 4x4 complex matrix and maximises 2Tr[CM] subject to I/2 - Tr_out(C) >= 0.
Restricting L to 4 x r columns restricts rank(C) to r and nothing else changes,
so the whole ladder costs one extra integer argument.  That this is the right
restriction is not an assumption: `_selftest_conventions` asserts that r=1
reproduces the production unitary optimum `ab._max_unitary_J` and that r=4
reproduces the production CPTP optimum `ab._sdp_branch`, branch by branch.

WHAT THE OBJECTIVE MEANS, EXACTLY.  With |B>> = (I ox B)|Omega> the Choi
objective expands, by the degree-2 Haar moment on CP^1, to

    2 Tr[C M_s] = sum_{j,k} ( |Tr(B_j A_k)|^2
                              + Tr(B_j A_k A_k^dag B_j^dag) ) / 6 ,

which is the Haar average of <psi| Phi_s(sum_k A_k |psi><psi| A_k^dag) |psi> --
the production estimator, summed over the branch's Kraus operators.  For a single
unitary B the second term collapses to const_s and this becomes exactly
`vscr_general.cf_of`'s (J_s + const_s)/6.  `_selftest_conventions` checks the
identity against `vscr_general.cf_unnormalised`, which is production code, so
the multi-Kraus bookkeeping is validated against a function that was already
audited rather than against a re-derivation of itself.
"""
# ---------------------------------------------------------------------------
# TRACE-PRESERVING, NOT MERELY TRACE-NON-INCREASING -- AND WHY THAT IS A RESULT.
# The production ceiling used to impose the inequality I/2 - Tr_out(C) >= 0.  The
# objective is monotone in C, so the optimum saturates it, but only to the
# optimiser's tolerance -- and worse, the inequality lets SLSQP stop in the
# INTERIOR of the sub-trace-preserving set at points strictly worse than the true
# optimum.  Measured on [[9,1,3]] amplitude damping p=0.05 that under-reported the
# ceiling by 3.7x, and by 16x at p=0.10: the returned Choi matrix had rank 1, so
# the solve never left the unitary family at all.  Twenty-four random starts still
# miss the optimum by 1.1e-7, whereas the four real EQUALITIES Tr_out(C) = I/2
# reach it from two starts and reproduce it to 3e-13 from three and from six.
# `vscr_paper_abl._sdp_branch` has been fixed accordingly.  This module imposes the
# equality throughout, because a physical instrument needs sum_j B_j^dag B_j = I
# exactly, and that equality is precisely what makes the Stinespring isometry
# extendable to a unitary.  U is returned, not just its fidelity: that is the
# circuit.
#
# COST OF THE ANSWER.  One ancilla qubit per branch, one (n+1)-qubit unitary, and a
# discarded ancilla -- no mid-circuit measurement, no post-selection, no
# feed-forward.  Existence is constructive and `_selftest_physical_dilation` builds
# it on the real 2^n-dimensional register: T_i = V B_i W_s^dag satisfies
# sum_i T_i^dag T_i = P_s, so |0>_a -> sum_i |i>_a T_i is an isometry out of
# image(W_s) and extends to a unitary on ancilla x data whose reduced blocks
# reproduce the Kraus operators through the paper's own V^dagger R_s W_s = G_s map.
#
# FIVE SELF-TESTS, each anchored either to production code or to a route that
# shares nothing with the Choi program:
#   _selftest_conventions       r=1 == ab._max_unitary_J, r=4 == ab._sdp_branch,
#                               2Tr[CM] == g.cf_unnormalised, ladder monotone
#   _selftest_tp                exact trace preservation, exactly unitary
#                               dilation, partial trace reproduces Phi, and the
#                               independent unconstrained exp(iH) circuit route
#                               lands on the same value
#   _selftest_quadrature        brute-force Haar average of the physical
#                               estimator, using neither the Choi matrix nor the
#                               degree-2 moment identity
#   _selftest_physical_dilation the (n+1)-qubit unitary exists on the physical
#                               register, not only on the logical space
#   _selftest_controls          coherent and depolarizing must give a FLAT ladder
#                               -- the negative control against solver slack
#
# Usage:
#     ./qenv/bin/python ancilla_recovery.py --quick      # n=5/7 amplitude damping
#     ./qenv/bin/python ancilla_recovery.py              # full n=5/7/9 ladder
#     ./qenv/bin/python ancilla_recovery.py --selftest   # conventions only
# ---------------------------------------------------------------------------
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Single-threaded BLAS BEFORE numpy, and import vscr_general first so that its
# single-CPU pin takes effect.  Same reason as in scaling_analysis.py: the
# degenerate eigenspaces that define W_s are LAPACK-blocking dependent, and
# multi-threading has previously moved an n=5 amplitude-damping number by 7e-4.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
# OPENBLAS_CORETYPE is for determinism only -- it is NOT the fix for this host's
# intermittent SIGSEGV/SIGILL (see the long note in ssvr_qec.py); CPU pinning is.
# HASWELL matches every other module in this repository, which matters because a
# differing coretype here would make the LAPACK blocking, and hence the degenerate
# eigenspaces that define W_s, differ from the production runs this module is
# asserted to reproduce.
os.environ.setdefault('OPENBLAS_CORETYPE', 'HASWELL')

import numpy as np                                              # noqa: E402
from scipy.linalg import expm                                   # noqa: E402
from scipy.optimize import minimize                             # noqa: E402

import vscr_general as g                                        # noqa: E402
import vscr_paper_abl as ab                                     # noqa: E402
import scaling_analysis as sa                                   # noqa: E402

I2 = np.eye(2, dtype=complex)

# The ladder.  r=1 and r=4 must reproduce the audited F_unit and F_cptp; r=2 is
# the one-ancilla instrument this module exists to measure; r=3 is included so
# that if one ancilla qubit does NOT suffice, the honest statement is how much of
# the remaining gap the second one closes.
RANKS = (1, 2, 3, 4)

# (channel, strength) points.  Amplitude damping is where the unitary family runs
# out at n=9; coherent is the control where a perfect unitary recovery exists
# (the ladder must therefore be flat at 1.0) and depolarizing the control where
# the decoder is already CPTP-optimal (the ladder must be flat at F_dec).
# Without those two controls a rank-2 "gain" could be solver slack, not physics.
POINTS = (('amplitude_damping', 0.05), ('amplitude_damping', 0.10),
          ('coherent', 0.10), ('depolarizing', 0.10))

QUICK_POINTS = (('amplitude_damping', 0.05), ('amplitude_damping', 0.10))
QUICK_CODES = ('5,1,3', '7,1,3')
FULL_CODES = ('5,1,3', '7,1,3', '9,1,3')
# ---------------------------------------------------------------------------
# Choi bookkeeping -- conventions copied verbatim from vscr_paper_abl._sdp_branch
# and vscr_general.branch_M, so that a number from this module and a number from
# the audited production ceiling are the same number.
# ---------------------------------------------------------------------------
def trace_out(C):
    """Tr over the second factor: Tr_out(C)[a,a'] = sum_b C[(a,b),(a',b)].

    For C = (id ox Phi)(|Omega><Omega|) with Phi(rho) = sum_j B_j rho B_j^dag this
    equals (sum_j B_j^dag B_j)^T / 2, so Tr_out(C) = I/2 is EXACTLY trace
    preservation of Phi.  Identical index arithmetic to ab._sdp_branch.trace_out.
    """
    return np.array([[C[0, 0] + C[1, 1], C[0, 2] + C[1, 3]],
                     [C[2, 0] + C[3, 1], C[2, 2] + C[3, 3]]])


def choi_vector(B):
    """|B>> = (I ox B)|Omega>, i.e. v[a*2+b] = B[b,a]/sqrt(2).

    Same mapping as ab._sdp_branch.L_of_unitary, which is why the r=1 rung of the
    ladder is the unitary family rather than some other rank-1 set."""
    v = np.zeros(4, dtype=complex)
    for a in range(2):
        for b in range(2):
            v[a * 2 + b] = B[b, a] / math.sqrt(2)
    return v


def operator_of_vector(v):
    """Inverse of choi_vector: B[b,a] = sqrt(2) * v[a*2+b]."""
    B = np.zeros((2, 2), dtype=complex)
    for a in range(2):
        for b in range(2):
            B[b, a] = math.sqrt(2) * v[a * 2 + b]
    return B


def kraus_of_L(L):
    """Kraus operators of C = L L^dag, one per non-negligible column."""
    out = []
    for j in range(L.shape[1]):
        v = L[:, j]
        if float(np.linalg.norm(v)) > 1e-12:
            out.append(operator_of_vector(v))
    return out


def L_of_kraus(Bs):
    """Choi factor whose j-th column is |B_j>>."""
    L = np.zeros((4, len(Bs)), dtype=complex)
    for j, B in enumerate(Bs):
        L[:, j] = choi_vector(np.asarray(B, dtype=complex))
    return L


def choi_obj(L, M):
    """2 Tr[C M] with C = L L^dag: the unnormalised Haar-averaged branch fidelity.

    Divide by p_s to get the conditional fidelity cf_s, exactly as
    scaling_analysis.fidelities does with ab._sdp_branch's return value."""
    return 2.0 * float(np.real(np.trace(L @ L.conj().T @ M)))


def kraus_fidelity(Bs, A_list):
    """INDEPENDENT route to the same scalar, through production code.

    `g.cf_unnormalised` implements (|Tr(GA)|^2 + Tr(GAA^dagG^dag))/6, the degree-2
    Haar moment, for one operator.  Summing it over the Kraus set gives the
    multi-Kraus objective without touching the Choi matrix at all, so agreement
    with `choi_obj` validates the conventions rather than restating them."""
    return float(sum(g.cf_unnormalised(B, A_list) for B in Bs))


def tp_defect(Bs):
    """max|sum_j B_j^dag B_j - I|: zero iff the instrument is trace preserving."""
    D = np.zeros((2, 2), dtype=complex)
    for B in Bs:
        D += np.asarray(B, dtype=complex).conj().T @ B
    return float(np.abs(D - I2).max())


def repair_tp(Bs):
    """Exact trace-preserving repair: B_i -> B_i Delta^{-1/2}, Delta = sum B^dag B.

    Delta is positive definite whenever the solve landed near the TP manifold, and
    the repaired family satisfies sum_i B_i^dag B_i = Delta^{-1/2} Delta
    Delta^{-1/2} = I to machine precision.  The fidelity changes only at second
    order in (Delta - I), which `_selftest_tp` measures rather than assumes."""
    D = np.zeros((2, 2), dtype=complex)
    for B in Bs:
        D += np.asarray(B, dtype=complex).conj().T @ B
    w, V = np.linalg.eigh(0.5 * (D + D.conj().T))
    assert float(w.min()) > 1e-6, 'TP repair needs a positive definite defect'
    Dh = V @ np.diag(1.0 / np.sqrt(w)) @ V.conj().T
    return [np.asarray(B, dtype=complex) @ Dh for B in Bs]
# ---------------------------------------------------------------------------
# the rank-constrained Choi program
# ---------------------------------------------------------------------------
def _pack(L):
    r = L.shape[1]
    return np.concatenate([L.real.reshape(4 * r), L.imag.reshape(4 * r)])


def _unpack(x, r):
    return (x[:4 * r] + 1j * x[4 * r:]).reshape(4, r)


def solve_rank_choi(M, r, G_dec, G_ref=None, L_upper=None, n_rand=6, seed=0,
                    tp_equal=True, maxiter=500, ftol=1e-12):
    """max 2Tr[LL^dag M] over L in C^{4 x r}, with the CPTP constraint on C=LL^dag.

    This is `ab._sdp_branch` with the Choi factor restricted to r columns, hence
    rank(C) <= r, hence (by Stinespring) an instrument realisable with an ancilla
    of dimension r.  Everything else -- objective, constraints, multi-start
    discipline, and the refusal to trust scipy's `success` flag -- is unchanged,
    because `_sdp_branch`'s docstring records what trusting it cost.

    tp_equal=True imposes Tr_out(C) = I/2 as four real EQUALITIES, which is what
    a physical instrument needs; False reproduces the production inequality form.

    `L_upper` supplies the optimum at a higher rank.  Its top-r eigenvectors are a
    far better warm start for the non-convex r<4 programs than random draws, and
    seeding from them is what makes the reported rank-2 value a maximum rather
    than a stationary point found by luck.

    Returns (F_unnorm, L)."""
    assert 1 <= r <= 4, r

    def obj(x):
        L = _unpack(x, r)
        return choi_obj(L, M)

    def to_res(x):
        T = trace_out(_unpack(x, r) @ _unpack(x, r).conj().T)
        return np.array([float(np.real(T[0, 0]) - 0.5),
                         float(np.real(T[1, 1]) - 0.5),
                         float(np.real(T[0, 1])), float(np.imag(T[0, 1]))])

    def c_tr(x):
        T = 0.5 * I2 - trace_out(_unpack(x, r) @ _unpack(x, r).conj().T)
        return float(np.real(np.trace(T)))

    def c_det(x):
        T = 0.5 * I2 - trace_out(_unpack(x, r) @ _unpack(x, r).conj().T)
        return float(np.real(np.linalg.det(T)))

    if tp_equal:
        cons = [{'type': 'eq', 'fun': (lambda x, i=i: to_res(x)[i])}
                for i in range(4)]

        def feasible(x):
            return float(np.abs(to_res(x)).max()) < 1e-6
    else:
        cons = [{'type': 'ineq', 'fun': c_tr}, {'type': 'ineq', 'fun': c_det}]

        def feasible(x):
            return c_tr(x) > -1e-8 and c_det(x) > -1e-10

    def L_of_unitary(G):
        L = np.zeros((4, r), dtype=complex)
        L[:, 0] = choi_vector(np.asarray(G, dtype=complex))
        return L

    starts = [L_of_unitary(I2), L_of_unitary(G_dec)]
    if G_ref is not None:
        starts.append(L_of_unitary(np.asarray(G_ref, dtype=complex)))
    if L_upper is not None and L_upper.shape[1] >= r:
        # top-r spectral truncation of the higher-rank Choi matrix
        C = L_upper @ L_upper.conj().T
        w, V = np.linalg.eigh(0.5 * (C + C.conj().T))
        keep = np.clip(w[-r:], 0.0, None)
        starts.append(V[:, -r:] @ np.diag(np.sqrt(keep)))
    rs = np.random.RandomState(seed)
    for _ in range(n_rand):
        starts.append((rs.randn(4, r) + 1j * rs.randn(4, r)) / 2)

    best, best_x = -np.inf, None
    for L0 in starts:
        x0 = _pack(L0)
        # (i) seed with the start itself if it is feasible, so the return value
        # can never fall below a point already verified feasible
        if feasible(x0) and obj(x0) > best:
            best, best_x = obj(x0), x0
        res = minimize(lambda x: -obj(x), x0, constraints=cons, method='SLSQP',
                       options={'maxiter': maxiter, 'ftol': ftol})
        # (ii) ignore res.success -- see ab._sdp_branch's docstring for why
        if feasible(res.x) and -float(res.fun) > best:
            best, best_x = -float(res.fun), res.x
    if best_x is None:
        raise RuntimeError('rank-%d Choi program found no feasible point' % r)
    return best, _unpack(best_x, r)


def stinespring_of_kraus(Bs):
    """The unitary dilation of a Kraus family, built explicitly.

    Ordering is ancilla-then-logical, flat index a*2+b, matching choi_vector.  The
    isometry W|l> = sum_i |i>_a ox B_i|l> has W^dag W = sum_i B_i^dag B_i, so it is
    an isometry exactly when the family is trace preserving -- which is why
    `repair_tp` runs first.  The remaining columns are an orthonormal basis of the
    orthogonal complement of image(W), taken from the SVD of its projector (whose
    eigenvalue-1 eigenspace IS that complement).  Returns (U, isometry_defect) with
    U unitary to machine precision and <i|_a U |0>_a = B_i."""
    r = len(Bs)
    assert r >= 1
    da = max(2, r)                       # ancilla dimension: a qubit iff r <= 2
    W = np.zeros((da * 2, 2), dtype=complex)
    for i, B in enumerate(Bs):
        W[i * 2:(i + 1) * 2, :] = np.asarray(B, dtype=complex)
    dev = float(np.abs(W.conj().T @ W - I2).max())
    assert dev < 1e-6, 'Kraus family is not trace preserving (defect %.2e)' % dev
    P = np.eye(da * 2) - W @ W.conj().T
    u, sv, _ = np.linalg.svd(0.5 * (P + P.conj().T))
    # P is a projector, so its singular values are 1 exactly on the complement of
    # image(W) and 0 on image(W).  Asserting that split is what licenses taking the
    # first da*2-2 columns of u as an orthonormal basis of the complement; without
    # it a numerically rank-deficient W would silently produce a non-unitary U.
    assert abs(sv[da * 2 - 3] - 1.0) < 1e-9, sv
    assert abs(sv[da * 2 - 2]) < 1e-9, sv
    N = u[:, :da * 2 - 2]
    U = np.concatenate([W, N], axis=1)
    return U, dev


def kraus_of_stinespring(U, r=2):
    """B_i = <i|_a U |0>_a, the Kraus family a dilation unitary implements."""
    return [U[i * 2:(i + 1) * 2, 0:2] for i in range(r)]


def phi_of_stinespring(U, rho):
    """Tr_a[ U (|0><0|_a ox rho) U^dag ] -- the instrument, evaluated as a circuit.

    The ancilla dimension is read off U rather than passed in, so this works for
    any rung of the ladder."""
    da = U.shape[0] // 2
    inp = np.zeros((da * 2, da * 2), dtype=complex)
    inp[0:2, 0:2] = np.asarray(rho, dtype=complex)   # ancilla in |0>, top-left
    out = U @ inp @ U.conj().T
    red = np.zeros((2, 2), dtype=complex)
    for i in range(da):
        red += out[i * 2:(i + 1) * 2, i * 2:(i + 1) * 2]
    return red
# ---------------------------------------------------------------------------
# the same family, optimised directly as a CIRCUIT
#
# A one-ancilla branch instrument is U on ancilla(2) x logical(2) with the ancilla
# prepared in |0> and discarded, so its Kraus operators are B_i = <i|U|0> and it
# is automatically trace preserving with Kraus rank <= 2.  Parameterising
# U = expm(i H) with H Hermitian (4 real diagonal + 6 complex off-diagonal = 16
# real parameters) makes the constraint set EMPTY: every x is feasible.  That is
# worth the extra code because it turns the rank-2 ceiling into an unconstrained
# smooth optimisation that L-BFGS-B solves, and because agreement with
# solve_rank_choi(M, 2, ...) then cross-validates two parameterisations of one
# family that share nothing but the answer -- different variables, different
# constraints, different optimiser, different objective expression.
#
# expm is used rather than the closed form vscr_paper_abl._G_of_h exploits, because
# that closed form is specific to u(2).  The module pins a single CPU and forces
# single-threaded BLAS at import, which is the condition under which the expm
# segfaults recorded in _G_of_h's docstring were avoided.
# ---------------------------------------------------------------------------
_H_IDX = [(i, j) for i in range(4) for j in range(i + 1, 4)]


def hermitian_of_x(x):
    """H(x) Hermitian 4x4 from 16 reals: x[0:4] diagonal, x[4:16] off-diagonal."""
    H = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        H[i, i] = x[i]
    for k, (i, j) in enumerate(_H_IDX):
        H[i, j] = x[4 + k] + 1j * x[10 + k]
        H[j, i] = np.conj(H[i, j])
    return H


def stinespring_obj(x, M):
    """2Tr[CM] for the instrument U = expm(iH(x)) implements, via its Kraus pair."""
    U = expm(1j * hermitian_of_x(x))
    Bs = kraus_of_stinespring(U, 2)
    return choi_obj(L_of_kraus(Bs), M)


def _seed_x_of_unitary(G, tail=None):
    """The 16-vector whose dilation implements B_0 = G, B_1 = 0.

    Any unitary G is expm(i h) for a Hermitian h -- eigendecompose G and take the
    phases -- so H = h (+) tail gives U = expm(iH) = G (+) expm(i tail), whose
    ancilla-column-0 blocks are exactly (G, 0).  Building the seed this way avoids
    a matrix logarithm of a unitary, which is the one step here that could be
    ill-conditioned."""
    G = np.asarray(G, dtype=complex)
    ev, EV = np.linalg.eig(G)
    ev = ev / np.abs(ev)                        # G unitary => |ev| = 1
    h = EV @ np.diag(np.angle(ev)) @ np.linalg.inv(EV)
    h = 0.5 * (h + h.conj().T)                  # kill the eig round-off
    H = np.zeros((4, 4), dtype=complex)
    H[0:2, 0:2] = h
    if tail is not None:
        H[2:4, 2:4] = np.asarray(tail, dtype=complex)
    x = np.zeros(16)
    for i in range(4):
        x[i] = float(np.real(H[i, i]))
    for k, (i, j) in enumerate(_H_IDX):
        x[4 + k] = float(np.real(H[i, j]))
        x[10 + k] = float(np.imag(H[i, j]))
    U = expm(1j * hermitian_of_x(x))
    assert float(np.abs(U[0:2, 0:2] - G).max()) < 1e-9, 'seed does not realise G'
    assert float(np.abs(U[2:4, 0:2]).max()) < 1e-9, 'seed leaked into B_1'
    return x


def solve_stinespring_direct(M, G_dec, n_start=8, seed=0, maxiter=2000):
    """Unconstrained multi-start L-BFGS-B over the one-ancilla circuit family.

    Returns (F_unnorm, U).  Seeds are the dilations of I and of the decoder block
    (both rank-1, hence feasible rank-2 points) plus random Hermitians.  The r=2
    Choi solution is deliberately NOT used as a seed, so agreement between this
    route and `solve_rank_choi(M, 2, ...)` is evidence rather than a tautology."""
    best, best_x = -np.inf, None
    starts = [_seed_x_of_unitary(I2), _seed_x_of_unitary(G_dec)]
    rs = np.random.RandomState(seed)
    for _ in range(n_start):
        starts.append(rs.randn(16) * 1.5)
    for x0 in starts:
        v = stinespring_obj(x0, M)
        if v > best:
            best, best_x = v, x0
        res = minimize(lambda y: -stinespring_obj(y, M), x0, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-15,
                                'gtol': 1e-12})
        if -float(res.fun) > best:
            best, best_x = -float(res.fun), res.x
    if best_x is None:
        raise RuntimeError('direct Stinespring solve found no point')
    return best, expm(1j * hermitian_of_x(best_x))
# ---------------------------------------------------------------------------
# the ladder, per branch and per code x channel x strength
# ---------------------------------------------------------------------------
def branch_ladder(b, ranks=RANKS, n_rand=3, seed=0):
    """{r: unnormalised branch fidelity} over the Kraus-rank ladder, plus factors.

    Ranks are solved in DESCENDING order so that each rung can warm-start from the
    spectral truncation of the one above it.  For the non-convex r<4 programs that
    truncation seed is what makes the answer a maximum rather than a lucky
    stationary point, and it costs one eigendecomposition of a 4x4 matrix.

    `b` is one element of `scaling_analysis.branch_bundle`, so the reduced branch
    operators {A_k}, the objective matrix M_s, the weight p_s and the decoder block
    G_dec are all the audited production objects."""
    M, ps, Gd = b['M'], b['p_s'], b['G_dec']
    Q, const = b['Q'], b['const']
    if ps < 1e-13:
        # zero-weight branch: the conditional fidelity is 1 by convention and the
        # branch contributes exactly p_s, as in scaling_analysis.fidelities
        return dict(p_s=ps, F={r: ps for r in ranks}, cf={r: 1.0 for r in ranks},
                    L={r: None for r in ranks}, skipped=True)
    single = b['A'][0] if len(b['A']) == 1 else None
    Ju, Gu = ab._max_unitary_J(Q, n_start=8, seed=seed, single=single, G_dec=Gd)
    out_F, out_L = {}, {}
    L_up = None
    for r in sorted(ranks, reverse=True):
        fu, L = solve_rank_choi(M, r, Gd, G_ref=Gu, L_upper=L_up,
                                n_rand=n_rand, seed=seed)
        out_F[r], out_L[r] = fu, L
        L_up = L
    return dict(p_s=ps, F=out_F, cf={r: out_F[r] / ps for r in ranks},
                L=out_L, skipped=False, J_unit=Ju, const=const, G_unit=Gu)


def witness(b, bl, rank=2, s=0, seed_base=0):
    """Hardware witness for one branch: Kraus pair, dilation unitary, TP checks.

    Everything returned here is measured, not assumed: the TP defect before and
    after `repair_tp`, the unitarity defect of U, the agreement of the independent
    `kraus_fidelity` route with the Choi objective, and the agreement of the
    unconstrained circuit optimiser with the constrained Choi optimiser."""
    L = bl['L'][rank]
    Bs_raw = kraus_of_L(L)
    f_choi = choi_obj(L, b['M'])
    f_kraus = kraus_fidelity(Bs_raw, b['A'])
    dev_raw = tp_defect(Bs_raw)
    Bs = repair_tp(Bs_raw)
    dev_rep = tp_defect(Bs)
    f_rep = kraus_fidelity(Bs, b['A'])
    U, iso_dev = stinespring_of_kraus(Bs)
    unit_dev = float(np.abs(U.conj().T @ U - np.eye(U.shape[0])).max())
    Bs_back = kraus_of_stinespring(U, len(Bs))
    back_dev = float(max(np.abs(x - y).max()
                         for x, y in zip(Bs, Bs_back)))
    # the instrument, evaluated as a circuit on random states, vs the Kraus sum
    rs = np.random.RandomState(12345)
    worst_map = 0.0
    for _ in range(8):
        v = rs.randn(2) + 1j * rs.randn(2)
        v /= np.linalg.norm(v)
        rho = np.outer(v, v.conj())
        got = phi_of_stinespring(U, rho)
        want = sum(B @ rho @ B.conj().T for B in Bs)
        worst_map = max(worst_map, float(np.abs(got - want).max()))
    f_direct, U_direct = solve_stinespring_direct(b['M'], b['G_dec'], n_start=6,
                                                 seed=seed_base + int(s))
    Bs_direct = kraus_of_stinespring(U_direct, 2)
    return dict(syndrome=int(s), p_s=bl['p_s'], rank=rank,
                # nested per OPERATOR then per row: kraus[j][i][k] = (re, im) of
                # B_j[i, k].  An earlier version flattened this to a list of rows
                # (`for B in Bs for row in B`), which lost the operator grouping
                # and made the ED Table that renders B_0 and B_1 ambiguous.
                kraus=[[[[float(x.real), float(x.imag)] for x in row]
                        for row in np.asarray(B, dtype=complex)] for B in Bs],
                stinespring_U=[[[float(x.real), float(x.imag)] for x in row]
                               for row in U],
                F_choi=f_choi, F_kraus=f_kraus, F_repaired=f_rep,
                F_direct=f_direct,
                cf_choi=f_choi / bl['p_s'], cf_direct=f_direct / bl['p_s'],
                tp_defect_raw=dev_raw, tp_defect_repaired=dev_rep,
                isometry_defect=iso_dev, unitarity_defect=unit_dev,
                kraus_roundtrip_defect=back_dev, map_defect=worst_map,
                direct_unitarity_defect=float(np.abs(
                    U_direct.conj().T @ U_direct - np.eye(4)).max()),
                direct_tp_defect=tp_defect(Bs_direct))


def run_point(code, name, channel, p, ranks=RANKS, n_rand=3, n_witness=4,
              max_branches=None):
    """The full rank ladder at one code x channel x strength, plus witnesses.

    Returns F_r for every rung r, the headroom of each rung over the decoder, the
    fraction of the r=4 (full CPTP) headroom that one ancilla captures, and a
    hardware witness for each of the `n_witness` branches that gain most from
    going rank-1 -> rank-2.  That last choice is deliberate: the witnesses are
    taken where the answer is hardest, not where it is prettiest."""
    t0 = time.time()
    B = sa.branch_bundle(code, channel, p)
    nsyn = len(B)
    idx = range(nsyn) if max_branches is None else range(min(nsyn, max_branches))
    lad = {}
    for s in idx:
        lad[s] = branch_ladder(B[s], ranks=ranks, n_rand=n_rand, seed=0)
    F = {r: 0.0 for r in ranks}
    gain21 = []
    for s in idx:
        bl = lad[s]
        for r in ranks:
            F[r] += bl['F'][r]
        if not bl['skipped'] and 1 in ranks and 2 in ranks:
            gain21.append((bl['F'][2] - bl['F'][1], s))
    # decoder fidelity, from the production quadratic form -- not re-derived here
    F_dec = 0.0
    for s in idx:
        b = B[s]
        gd = b['G_dec'].reshape(4)
        Jd = float(np.real(gd @ b['Q'] @ gd.conj()))
        F_dec += (Jd + b['const']) / 6.0
    out = dict(code=name, n=code.n, nsyn=nsyn, channel=channel, p=p,
               F_dec=F_dec,
               F={str(r): F[r] for r in ranks},
               headroom={str(r): F[r] - F_dec for r in ranks},
               n_branches_solved=len(lad),
               seconds=time.time() - t0)
    if '1' in out['headroom'] and '4' in out['headroom'] \
            and '2' in out['headroom']:
        h1 = out['headroom']['1']
        h2 = out['headroom']['2']
        h4 = out['headroom']['4']
        # The quantity that answers the paper's question: of the headroom the
        # UNITARY family cannot reach (h4 - h1), what fraction does ONE ancilla
        # qubit reach (h2 - h1)?  The denominator vanishes identically on both
        # control channels and at n=5,7 amplitude damping -- there the unitary
        # family already attains the CPTP ceiling, so the ratio is 0/0 and is
        # reported as None rather than silently as 1.0 or as a division error.
        out['nonunitary_headroom'] = h4 - h1
        out['one_ancilla_gain'] = h2 - h1
        out['one_ancilla_fraction_of_nonunitary_headroom'] = (
            None if abs(h4 - h1) < 1e-12 else (h2 - h1) / (h4 - h1))
        out['one_ancilla_gap_to_cptp'] = h4 - h2
    # witnesses: the branches where rank 2 buys the most over rank 1
    gain21.sort(reverse=True)
    out['witness_branches'] = []
    for _, s in gain21[:n_witness]:
        w = witness(B[s], lad[s], rank=(2 if 2 in ranks else max(ranks)), s=s)
        w['gain_rank2_over_rank1'] = (lad[s]['F'][2] - lad[s]['F'][1]
                                      if 1 in ranks and 2 in ranks else None)
        out['witness_branches'].append(w)
    out['seconds'] = time.time() - t0
    return out, lad, B
# ---------------------------------------------------------------------------
# self-tests.  Each one compares against PRODUCTION code (ab._max_unitary_J,
# ab._sdp_branch, g.cf_unnormalised) rather than against a restatement of this
# module's own conventions, so passing them is evidence about the conventions.
# ---------------------------------------------------------------------------
_ST_POINTS = (('amplitude_damping', 0.10), ('depolarizing', 0.10),
              ('coherent', 0.10))
_ST_BRANCHES = (0, 1, 3, 4, 7, 11, 15)


def _selftest_conventions(code, branches=_ST_BRANCHES, verbose=True):
    """r=1 == the production unitary optimum; r=4 == the production CPTP optimum."""
    worst_r1 = worst_r4 = worst_r4eq = worst_kraus = worst_mono = 0.0
    nchecked = 0
    for channel, p in _ST_POINTS:
        B = sa.branch_bundle(code, channel, p)
        for s in branches:
            if s >= len(B) or B[s]['p_s'] < 1e-13:
                continue
            b = B[s]
            bl = branch_ladder(b, ranks=RANKS, n_rand=3, seed=0)
            # (a) r=1 IS the unitary family, so it must equal ab._max_unitary_J
            want_r1 = (bl['J_unit'] + bl['const']) / 6.0
            worst_r1 = max(worst_r1, abs(bl['F'][1] - want_r1))
            # (b) r=4 under the production INEQUALITY constraint must equal
            #     ab._sdp_branch, i.e. the audited F_cptp machinery
            fu_prod, _ = ab._sdp_branch(b['M'], b['G_dec'], G_ref=bl['G_unit'])
            fu_ineq = solve_rank_choi(b['M'], 4, b['G_dec'], G_ref=bl['G_unit'],
                                      n_rand=3, seed=0, tp_equal=False)[0]
            worst_r4 = max(worst_r4, abs(fu_ineq - fu_prod))
            # (c) the EQUALITY-constrained optimum must match the inequality one:
            #     the objective is monotone in C, so the optimum saturates TP
            worst_r4eq = max(worst_r4eq, abs(bl['F'][4] - fu_prod))
            # (d) the multi-Kraus objective equals production cf_unnormalised,
            #     rung by rung -- this is what licenses the Kraus/C dictionary
            for r in RANKS:
                L = bl['L'][r]
                worst_kraus = max(worst_kraus, abs(
                    kraus_fidelity(kraus_of_L(L), b['A']) - choi_obj(L, b['M'])))
            # (e) the ladder is monotone: rank-r Choi matrices are a subset of
            #     rank-(r+1) ones, so F_r <= F_{r+1} and any violation is a
            #     solver failure rather than physics
            for r_lo, r_hi in zip(RANKS[:-1], RANKS[1:]):
                worst_mono = max(worst_mono, bl['F'][r_lo] - bl['F'][r_hi])
            nchecked += 1
    assert nchecked > 0, 'no branch had non-negligible weight'
    assert worst_r1 < 1e-9, 'r=1 != production unitary optimum: %.3e' % worst_r1
    assert worst_r4 < 1e-9, 'r=4 != production CPTP optimum: %.3e' % worst_r4
    assert worst_r4eq < 1e-9, (
        'equality-TP != inequality-TP at r=4: %.3e' % worst_r4eq)
    assert worst_kraus < 1e-11, (
        'Choi objective != production cf_unnormalised: %.3e' % worst_kraus)
    assert worst_mono < 1e-9, 'ladder not monotone in rank: %.3e' % worst_mono
    if verbose:
        print('  [conventions] %d branches x %d channels (n=%d):'
              % (nchecked, len(_ST_POINTS), code.n))
        print('    r=1  == ab._max_unitary_J optimum     to %.2e' % worst_r1)
        print('    r=4  == ab._sdp_branch optimum        to %.2e' % worst_r4)
        print('    r=4  equality-TP == inequality-TP     to %.2e' % worst_r4eq)
        print('    2Tr[CM] == g.cf_unnormalised(Kraus)   to %.2e' % worst_kraus)
        print('    ladder monotone in rank (worst viol.) %.2e' % worst_mono)
        print('  [conventions] VERIFIED', flush=True)
    return dict(worst_r1_vs_production=worst_r1, worst_r4_vs_production=worst_r4,
                worst_r4_equality_vs_inequality=worst_r4eq,
                worst_choi_vs_kraus=worst_kraus, worst_monotonicity=worst_mono,
                n_branches=nchecked)
def _selftest_tp(code, branches=_ST_BRANCHES, verbose=True):
    """Every witness is exactly TP, and its dilation is exactly unitary.

    This is the self-test that carries the hardware claim.  A rank-2 Choi matrix
    is only an *instrument* if sum_i B_i^dag B_i = I; the unconstrained circuit
    route is only an independent check if it lands on the same value; and the
    dilation is only a circuit if U is unitary and tracing the ancilla out of
    U(rho ox |0><0|)U^dag reproduces sum_i B_i rho B_i^dag."""
    worst = dict(tp_raw=0.0, tp_rep=0.0, iso=0.0, unit=0.0, rt=0.0,
                 map_=0.0, direct=0.0, direct_unit=0.0, direct_tp=0.0)
    n = 0
    for channel, p in _ST_POINTS:
        B = sa.branch_bundle(code, channel, p)
        for s in branches:
            if s >= len(B) or B[s]['p_s'] < 1e-13:
                continue
            bl = branch_ladder(B[s], ranks=(1, 2, 4), n_rand=3, seed=0)
            w = witness(B[s], bl, rank=2, s=s)
            worst['tp_raw'] = max(worst['tp_raw'], w['tp_defect_raw'])
            worst['tp_rep'] = max(worst['tp_rep'], w['tp_defect_repaired'])
            worst['iso'] = max(worst['iso'], w['isometry_defect'])
            worst['unit'] = max(worst['unit'], w['unitarity_defect'])
            worst['rt'] = max(worst['rt'], w['kraus_roundtrip_defect'])
            worst['map_'] = max(worst['map_'], w['map_defect'])
            worst['direct_unit'] = max(worst['direct_unit'],
                                       w['direct_unitarity_defect'])
            worst['direct_tp'] = max(worst['direct_tp'], w['direct_tp_defect'])
            # the circuit route optimises the SAME family as the rank-2 Choi
            # program, so it may not beat it; a shortfall above solver noise means
            # one of the two parameterisations is wrong
            worst['direct'] = max(worst['direct'], w['F_choi'] - w['F_direct'])
            n += 1
    assert n > 0
    assert worst['tp_rep'] < 1e-12, 'repaired Kraus set not TP: %.3e' % worst['tp_rep']
    assert worst['unit'] < 1e-12, (
        'Stinespring dilation not unitary: %.3e' % worst['unit'])
    assert worst['direct_unit'] < 1e-12, (
        'expm(iH) dilation not unitary: %.3e' % worst['direct_unit'])
    assert worst['direct_tp'] < 1e-12, (
        'circuit-route Kraus pair not TP: %.3e' % worst['direct_tp'])
    assert worst['iso'] < 1e-9, 'isometry defect too large: %.3e' % worst['iso']
    assert worst['rt'] < 1e-12, 'Kraus roundtrip through U failed: %.3e' % worst['rt']
    assert worst['map_'] < 1e-12, (
        'Tr_a[U(rho ox |0><0|)U^dag] != sum B rho B^dag: %.3e' % worst['map_'])
    assert worst['direct'] < 1e-9, (
        'direct circuit route below the Choi rank-2 optimum by %.3e'
        % worst['direct'])
    if verbose:
        print('  [tp/stinespring] %d branches (n=%d):' % (n, code.n))
        print('    raw solve TP defect (eq-constraint slack) %.2e'
              % worst['tp_raw'])
        print('    after repair_tp, TP defect                %.2e'
              % worst['tp_rep'])
        print('    Stinespring isometry defect               %.2e'
              % worst['iso'])
        print('    dilation unitarity defect                 %.2e'
              % worst['unit'])
        print('    Kraus <-> U roundtrip defect              %.2e'
              % worst['rt'])
        print('    Tr_a[U(rho|0><0|)U^dag] vs sum B rho B^dag %.2e'
              % worst['map_'])
        print('    circuit route shortfall vs Choi rank-2    %.2e'
              % worst['direct'])
        print('    circuit-route unitarity / TP defects      %.2e / %.2e'
              % (worst['direct_unit'], worst['direct_tp']))
        print('  [tp/stinespring] VERIFIED', flush=True)
    return dict(n_branches=n, **worst)


def _selftest_quadrature(code, branches=(0, 1, 4, 7), n_states=20000,
                         seed=20260916, verbose=True):
    """Brute-force Haar quadrature of the physical estimator: a THIRD route.

    `choi_obj` goes through the Choi matrix and `kraus_fidelity` through the
    degree-2 Haar moment identity.  This goes through neither.  It draws random
    pure states and averages

        <psi| Phi( sum_k A_k |psi><psi| A_k^dag ) |psi>
            = sum_{j,k} |<psi| B_j A_k |psi>|^2 ,

    which is exactly the estimator the paper reports,
    <psi_enc| R(rho_noisy) |psi_enc>, restricted to one branch.  Its expectation is
    the unnormalised branch fidelity, so agreement within Monte-Carlo error ties
    both closed forms to the quantity the manuscript actually quotes rather than to
    each other.  Agreement within Monte-Carlo error ties both closed forms to the
    quantity the manuscript actually quotes rather than to each other.

    One subtlety is worth stating rather than hiding, because it makes the check
    STRONGER on some branches and would otherwise look like a failure.  On
    Pauli-type and perfectly-correctable branches the estimator is exactly
    psi-INDEPENDENT: for coherent noise the optimal branch recovery is the inverse
    unitary, so the sum collapses to |<psi|psi>|^2 = 1; for depolarizing noise the
    branch operators are the whole Pauli group, so the sum is a Pauli twirl.  The
    quadrature then has zero variance and is a deterministic identity check rather
    than a statistical one -- measured standard error 7e-18 against a value of
    0.59, i.e. agreement to fifteen digits, which a naive "within 5 sigma" test
    would report as a 144-sigma failure.  So the tolerance is 5 sigma PLUS a
    relative round-off floor, and the two regimes are reported separately."""
    rs = np.random.RandomState(seed)
    V = rs.randn(n_states, 2) + 1j * rs.randn(n_states, 2)
    V /= np.linalg.norm(V, axis=1)[:, None]
    worst_z = 0.0
    worst_rel = 0.0
    worst_sem = 0.0
    n = n_degenerate = 0
    for channel, p in _ST_POINTS:
        B = sa.branch_bundle(code, channel, p)
        for s in branches:
            if s >= len(B) or B[s]['p_s'] < 1e-13:
                continue
            b = B[s]
            bl = branch_ladder(b, ranks=(2,), n_rand=3, seed=0)
            Bs = repair_tp(kraus_of_L(bl['L'][2]))
            exact = kraus_fidelity(Bs, b['A'])
            prods = [Bi @ A for Bi in Bs for A in b['A']]
            samp = np.zeros(n_states)
            for C in prods:
                q = np.einsum('na,ab,nb->n', V.conj(), C, V)
                samp += np.abs(q) ** 2
            mc = float(samp.mean())
            sem = float(samp.std(ddof=1) / np.sqrt(n_states))
            scale = max(abs(exact), 1e-300)
            # the estimator is psi-independent exactly when its standard error is
            # round-off rather than sampling noise
            degenerate = sem < 1e-13 * scale
            if degenerate:
                n_degenerate += 1
                # zero-variance branches: this is a round-off identity check
                worst_rel = max(worst_rel, abs(mc - exact) / scale)
            else:
                worst_z = max(worst_z, abs(mc - exact) / sem)
            worst_sem = max(worst_sem, sem)
            assert abs(mc - exact) <= 5.0 * sem + 1e-11 * scale, (
                'branch %d of %s p=%g: quadrature %.12e vs closed form %.12e '
                '(sem %.2e)' % (s, channel, p, mc, exact, sem))
            n += 1
    assert n > 0
    assert worst_z < 5.0, (
        'brute-force Haar quadrature disagrees with the closed form at %.1f sigma'
        % worst_z)
    assert worst_rel < 1e-11, (
        'worst relative quadrature deviation %.2e' % worst_rel)
    if verbose:
        print('  [quadrature] %d branches, %d states each:' % (n, n_states))
        print('    worst z on the %d branches with sampling noise   %.2f'
              % (n - n_degenerate, worst_z))
        print('    %d branches are psi-INDEPENDENT (zero-variance identity '
              'check)' % n_degenerate)
        print('    worst relative |MC - closed form| on those    %.2e'
              % worst_rel)
        print('    largest standard error seen                    %.2e'
              % worst_sem)
        print('  [quadrature] VERIFIED', flush=True)
    return dict(n_branches=n, n_states=n_states, worst_z=worst_z,
                worst_rel=worst_rel, n_psi_independent=n_degenerate,
                max_sem=worst_sem)


def physical_dilation(code, Bs, s):
    """The (n+1)-qubit unitary that realises branch s's instrument PHYSICALLY.

    Everything above lives on the two-dimensional logical space.  A reviewer is
    entitled to ask whether the branch recovery is realisable on the actual
    2^n-dimensional register, in the same sense that the paper's unitary recoveries
    are: there a physical R_s satisfies V^dagger R_s W_s = G_s.  The analogue for an
    instrument is a physical unitary Rt_s on ancilla x data satisfying

        (<i|_a ox V^dagger) Rt_s (|0>_a ox W_s) = B_i ,

    i.e. the same reduced-block map, applied to the Kraus operators.  Note that
    Rt_s's ancilla block <i|_a Rt_s |0>_a is a dim x dim matrix determined only on
    image(W_s) -- off that subspace it is the arbitrary complement completion, which
    is irrelevant because the syndrome measurement has already projected onto it.

    It exists constructively.  With T_i = V B_i W_s^dagger (a dim x dim operator of
    rank <= 2) and W_s an isometry onto image(P_s),

        sum_i T_i^dagger T_i = W_s (sum_i B_i^dagger B_i) W_s^dagger = P_s ,

    so |0>_a x |x>  ->  sum_i |i>_a x T_i|x> is an isometry on image(W_s) and
    extends to a unitary on ancilla x data by appending any orthonormal basis of
    the complement.  Trace preservation of {B_i} is exactly what makes the two
    columns orthonormal, which is the second reason the equality constraint matters
    -- a merely sub-trace-preserving solution would give a contraction here, not an
    isometry, and no unitary extension.  Returns (U, defects)."""
    dim = code.dim
    V, Ws = code.V, code.W[s]
    T = [V @ np.asarray(B, dtype=complex) @ Ws.conj().T for B in Bs]
    # sum_i T_i^dag T_i must be the syndrome projector P_s = W_s W_s^dag
    S = np.zeros((dim, dim), dtype=complex)
    for Ti in T:
        S += Ti.conj().T @ Ti
    P_s = Ws @ Ws.conj().T
    iso_defect = float(np.abs(S - P_s).max())
    da = max(2, len(Bs))
    N = da * dim
    # The two orthonormal families the unitary has to map between:
    #   Vin[:, l]  = |0>_a ox W_s e_l        (what the branch hands us)
    #   Wout[:, l] = sum_i |i>_a ox V B_i e_l (what the instrument must produce)
    # Wout^dag Wout = sum_i B_i^dag B_i = I, so trace preservation is exactly the
    # condition that makes this a basis-to-basis map and hence unitary-extendable.
    Vin = np.zeros((N, 2), dtype=complex)
    Wout = np.zeros((N, 2), dtype=complex)
    Vin[0:dim, :] = Ws
    for i, B in enumerate(Bs):
        Wout[i * dim:(i + 1) * dim, :] = V @ np.asarray(B, dtype=complex)
    ortho = max(float(np.abs(Vin.conj().T @ Vin - I2).max()),
                float(np.abs(Wout.conj().T @ Wout - I2).max()))

    def _complement(M):
        """Orthonormal basis of the orthogonal complement of image(M)."""
        Pr = np.eye(N) - M @ M.conj().T
        u, sv, _ = np.linalg.svd(0.5 * (Pr + Pr.conj().T))
        # Pr is a projector of rank N-2: singular values 1 then 0.  Asserting the
        # split is what licenses taking the first N-2 columns.
        assert abs(sv[N - 3] - 1.0) < 1e-8, sv[:4]
        assert abs(sv[N - 2]) < 1e-8, sv[:4]
        return u[:, :N - 2]

    # Rt maps an orthonormal basis to an orthonormal basis, so it is unitary.
    U = (Wout @ Vin.conj().T) + (_complement(Wout) @ _complement(Vin).conj().T)
    unit_defect = float(np.abs(U.conj().T @ U - np.eye(N)).max())
    # The reduced blocks of the PHYSICAL unitary must be the Kraus operators, in the
    # paper's own reduced-block form: (bra i|_a ox V^dagger) Rt (|0>_a ox W_s) = B_i.
    # U[:, 0:dim] is Rt |0>_a, so contracting it with W_s applies Rt to the branch
    # subspace and nothing else -- the arbitrary complement completion never enters.
    X = U[:, 0:dim] @ Ws                       # Rt (|0>_a ox W_s), (N, 2)
    B_back = [V.conj().T @ X[i * dim:(i + 1) * dim, :] for i in range(len(Bs))]
    block_defect = float(max(np.abs(x - np.asarray(y, dtype=complex)).max()
                             for x, y in zip(B_back, Bs)))
    return U, dict(isometry_to_projector=iso_defect, column_orthonormality=ortho,
                   unitarity=unit_defect, reduced_block=block_defect,
                   dim=dim, ancilla_dim=da)


def _selftest_physical_dilation(code, branches=(0, 1, 4, 7), verbose=True):
    """The one-ancilla instrument exists on the PHYSICAL register, not just logically.

    Runs `physical_dilation` on the rank-2 optimum of several branches and asserts
    all four defects vanish: that sum_i T_i^dag T_i is the syndrome projector, that
    the two isometry columns are orthonormal, that the completed (n+1)-qubit
    operator is unitary, and that its reduced blocks V^dag (bra i R ket 0) W_s are
    exactly the Kraus operators the Choi program returned."""
    worst = dict(isometry_to_projector=0.0, column_orthonormality=0.0,
                 unitarity=0.0, reduced_block=0.0)
    n = 0
    for channel, p in _ST_POINTS:
        B = sa.branch_bundle(code, channel, p)
        for s in branches:
            if s >= len(B) or B[s]['p_s'] < 1e-13:
                continue
            b = B[s]
            bl = branch_ladder(b, ranks=(2,), n_rand=3, seed=0)
            Bs = repair_tp(kraus_of_L(bl['L'][2]))
            _, d = physical_dilation(code, Bs, s)
            for k in worst:
                worst[k] = max(worst[k], d[k])
            n += 1
    assert n > 0
    for k, v in worst.items():
        assert v < 1e-10, 'physical dilation %s defect %.3e' % (k, v)
    if verbose:
        print('  [physical dilation] %d branches, dim=%d, ancilla dim=2:'
              % (n, code.dim))
        print('    sum_i T_i^dag T_i == P_s              %.2e'
              % worst['isometry_to_projector'])
        print('    isometry columns orthonormal          %.2e'
              % worst['column_orthonormality'])
        print('    completed (n+1)-qubit U unitary       %.2e'
              % worst['unitarity'])
        print('    V^dag (bra i U ket 0) W_s == B_i      %.2e'
              % worst['reduced_block'])
        print('  [physical dilation] VERIFIED', flush=True)
    return dict(n_branches=n, dim=code.dim, **worst)
def _selftest_controls(code, verbose=True):
    """The two control channels must give a FLAT ladder.

    On coherent over-rotation a perfect unitary branch recovery exists, so every
    rung must equal 1.000000000 and the one-ancilla gain must be exactly zero.
    On depolarizing noise the min-weight decoder is already CPTP-optimal, so every
    rung must equal F_dec.  A solver that invents rank-2 slack would fail here,
    which is the point: these are the negative controls for the n=9 result."""
    out = {}
    for channel, p in (('coherent', 0.10), ('depolarizing', 0.10)):
        res, _, _ = run_point(code, 'st', channel, p, ranks=(1, 2, 4),
                              n_rand=2, n_witness=0)
        h = res['headroom']
        spread = max(h['1'], h['2'], h['4']) - min(h['1'], h['2'], h['4'])
        if channel == 'coherent':
            assert abs(res['F']['1'] - 1.0) < 1e-9, (
                'coherent: unitary rung is not perfect recovery: %.12f'
                % res['F']['1'])
        assert abs(h['1']) < 1e-9 or channel == 'coherent', (
            '%s: decoder is not CPTP-optimal (headroom %.3e)' % (channel, h['1']))
        assert spread < 1e-9, (
            '%s: ladder is not flat (spread %.3e)' % (channel, spread))
        out[channel] = dict(F=res['F'], headroom=h, ladder_spread=spread)
        if verbose:
            print('  [control %-13s p=%.2f] F_dec=%.10f  rungs 1/2/4 = '
                  '%.10f / %.10f / %.10f  spread %.2e  FLAT'
                  % (channel, p, res['F_dec'], res['F']['1'], res['F']['2'],
                     res['F']['4'], spread), flush=True)
    return out
def _print_point(res, ranks):
    """One code x channel x strength row, as the rank ladder."""
    print('  %-18s p=%-5g n=%d  F_dec = %.10f' %
          (res['channel'], res['p'], res['n'], res['F_dec']), flush=True)
    lab = {1: 'rank 1  (0 ancilla, unitary)', 2: 'rank 2  (ONE ancilla qubit)',
           3: 'rank 3  (rank-3 dilation)', 4: 'rank 4  (full CPTP)'}
    for r in ranks:
        k = str(r)
        if k not in res['F']:
            continue
        h = res['headroom'][k]
        print('      %-28s F = %.10f   headroom %+.4e'
              % (lab[r], res['F'][k], h), flush=True)
    frac = res.get('one_ancilla_fraction_of_nonunitary_headroom')
    if frac is None:
        print('      one-ancilla share of the non-unitary headroom: n/a '
              '(the unitary family already attains the CPTP ceiling here, '
              'so the ratio is 0/0)', flush=True)
    else:
        print('      one-ancilla share of the non-unitary headroom: %.8f\n'
              '        one ancilla buys %+.4e of the %+.4e that the unitary\n'
              '        family cannot reach; residual gap to CPTP %+.4e'
              % (frac, res['one_ancilla_gain'], res['nonunitary_headroom'],
                 res['one_ancilla_gap_to_cptp']), flush=True)
    print('      (%d branches, %.0fs)' % (res['n_branches_solved'],
                                          res['seconds']), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--quick', action='store_true',
                    help='n=5,7 amplitude damping only')
    ap.add_argument('--selftest', action='store_true',
                    help='run the convention / TP / control self-tests and stop')
    ap.add_argument('--no-selftest', action='store_true',
                    help='skip the self-tests before the ladder run')
    ap.add_argument('--codes', default=None,
                    help='semicolon-separated subset, e.g. "5,1,3;9,1,3"')
    ap.add_argument('--points', default=None,
                    help='semicolon-separated channel:p, e.g. '
                         '"amplitude_damping:0.10;coherent:0.10"')
    ap.add_argument('--ranks', default='1,2,3,4')
    ap.add_argument('--n-witness', type=int, default=4,
                    help='how many branches to emit a hardware witness for')
    ap.add_argument('--max-branches', type=int, default=None,
                    help='cap the syndrome count (debug only: truncates F)')
    ap.add_argument('--json', default='ancilla_recovery.json')
    a = ap.parse_args(argv)

    ranks = tuple(sorted({int(x) for x in a.ranks.split(',') if x.strip()}))
    assert ranks and max(ranks) <= 4 and min(ranks) >= 1, ranks
    codes = (QUICK_CODES if a.quick else FULL_CODES)
    if a.codes:
        codes = tuple(c.strip() for c in a.codes.split(';') if c.strip())
    points = (QUICK_POINTS if a.quick else POINTS)
    if a.points:
        points = tuple((s.split(':')[0], float(s.split(':')[1]))
                       for s in a.points.split(';') if s.strip())
    for c in codes:
        assert c in g.CODES, 'unknown code %r' % c

    t0 = time.time()
    print('=== ancilla_recovery (pinned cpu %s, ranks %s) ==='
          % (g.PINNED_CPU, ','.join(str(r) for r in ranks)), flush=True)
    built = {name: g.StabCode(g.CODES[name]) for name in codes}
    for name, c in built.items():
        print('  built [[%s]]: n=%d dim=%d syndromes=%d d=%d'
              % (name, c.n, c.dim, c.nsyn, c.distance), flush=True)

    res = {'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'pinned_cpu': g.PINNED_CPU, 'ranks': list(ranks),
           'codes': list(codes),
           'points': [(ch, p) for ch, p in points], 'selftests': {},
           'ladder': []}

    st_code = built.get('5,1,3') or g.StabCode(g.CODES['5,1,3'])
    if not a.no_selftest:
        print('\n--- self-test: Choi/Kraus conventions vs production ceilings ---',
              flush=True)
        res['selftests']['conventions'] = _selftest_conventions(st_code)
        print('--- self-test: trace preservation and the Stinespring dilation ---',
              flush=True)
        res['selftests']['tp_stinespring'] = _selftest_tp(st_code)
        print('--- self-test: brute-force Haar quadrature of the estimator ---',
              flush=True)
        res['selftests']['quadrature'] = _selftest_quadrature(st_code)
        print('--- self-test: the (n+1)-qubit dilation exists PHYSICALLY ---',
              flush=True)
        res['selftests']['physical_dilation'] = \
            _selftest_physical_dilation(st_code)
        print('--- self-test: control channels must give a FLAT ladder ---',
              flush=True)
        res['selftests']['controls'] = _selftest_controls(st_code)
    if a.selftest:
        print('\nSELFTEST ONLY: all conventions, TP and control checks passed.')
        return 0

    print('\n--- the Kraus-rank ladder ---', flush=True)
    for name in codes:
        for ch, p in points:
            out, _, _ = run_point(built[name], name, ch, p, ranks=ranks,
                                  n_witness=a.n_witness,
                                  max_branches=a.max_branches)
            _print_point(out, ranks)
            res['ladder'].append(out)

    res['runtime_s'] = time.time() - t0
    if a.max_branches is None:
        with open(os.path.join(ROOT, a.json), 'w') as f:
            json.dump(res, f, indent=1)
        print('\nwrote %s  (%.0fs total)' % (a.json, res['runtime_s']))
    else:
        print('\n--max-branches set: results are truncated, NOT writing json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
PLACEHOLDER_LADDER = None
