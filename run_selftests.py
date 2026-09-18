#!/usr/bin/env python3
"""
run_selftests.py — run every numerical self-test in the pipeline and report.

These are the assertions that make the paper's claims checkable rather than
asserted.  Each one pins a property that, if it silently broke, would invalidate
a specific figure or table:

  ssvr_qec._selftest_gates            gate/embedding conventions
  ssvr_qec._selftest_action_cols      the O(DIM^2 r) column-propagation form of
                                      the ansatz equals the explicit 32x32
                                      product, in value AND gradient
  ssvr_qec._selftest_petz             the Petz/noise-adapted recovery baseline
                                      (Fix E): channel, adjoint and map
  ssvr_qec._selftest_quadrature       the exact Haar x noise-strength quadrature
                                      is exact for the objective's algebraic
                                      degree, and equals the full 32x32
                                      reference and Monte-Carlo
  vscr_paper._verify_warm_start       R_s == C_s phase-exactly, so the warm
                                      start IS the decoder at step 0
  vscr_paper._selftest_hypernet_trainable
                                      the hypernetwork is not dead: every
                                      syndrome-pathway parameter moves and the
                                      16 branches become distinct
  vscr_paper_abl._selftest_liealg     the closed form of exp(i h.sigma) equals
                                      scipy's expm and is unitary
  vscr_paper_abl._selftest_sdp        the CPTP ceiling SDP
  vscr_paper_abl._selftest_refine     Fix A/B: PHI_DEC is an EXACT stationary
                                      point, the separable objective IS the
                                      production objective, and the refinement
                                      is monotone and captures the headroom
  stationarity_boundary._selftest_certificate
                                      the analytic nine-sector certificate: the
                                      objective is an exact quadratic in the input
                                      Bloch vector whose l=0,1,2 coefficients are
                                      ensemble-independent, so vanishing sector
                                      gradients PROVE phi^dec is stationary for
                                      every ensemble, upgrading the 90-pair sweep
                                      from evidence to corollary
  ancilla_recovery (conventions / tp / quadrature /
                      physical dilation / controls)
                                      the Kraus-rank ladder that counts ancillas:
                                      rank(C)=1 reproduces the production unitary
                                      ceiling, rank(C)=4 the production CPTP
                                      ceiling, the Choi objective equals production
                                      cf_unnormalised, the rank-2 optimum is exactly
                                      trace preserving with an exactly unitary
                                      Stinespring dilation, an independent
                                      unconstrained circuit optimisation lands on
                                      the same value, and both control channels
                                      give a FLAT ladder
  multiseed_stats.selftest            the T3 statistics layer: ddof=1 std and sem
                                      against hand-computed values, the two
                                      refine-record shapes normalising to one
                                      canonical record, production's label-free
                                      selection reproduced (its "best of two" bias
                                      can be negative), single-seed std 0.0 rather
                                      than NaN, and degenerate ratios reported as
                                      undefined rather than as large numbers
  storage_rounds.selftest             the T4 multi-round reductions: full dim x dim
                                      channel bit-exact against apply_channel, the
                                      2x2 round map at R=1 reproducing exact_F and
                                      F_dec, the 'Raw' restriction and its
                                      trace-decrease, the ceiling's syndrome-basis
                                      transport, the exact R-round Pauli readout law
                                      and its measured violation by the learned
                                      recovery, and reduced-vs-full agreement with
                                      saturating leakage

Usage:
    python run_selftests.py            # everything (slow: ~20-40 min)
    python run_selftests.py --quick    # skip the slow refinement/SDP tests
    python run_selftests.py --list
    python run_selftests.py refine petz
"""
import os, subprocess, sys, time, traceback

# Pin threading BEFORE importing numpy/torch.  The native BLAS/OpenMP layer in
# this environment has intermittently segfaulted under sustained multi-threaded
# complex128 work; single-threaded runs are deterministic and, because every
# hot loop here is small-matrix bound, no slower in practice.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
# See the long note in ssvr_qec.py: this host intermittently SIGSEGV/SIGILLs in
# tiny complex128 kernels when a process is free to MIGRATE across its hybrid
# P-/E-core cluster. Importing ssvr_qec pins us to one CPU, which fixes it
# (measured: unpinned core-dumps within ~2e4 calls; pinned runs 4e5 clean).
# OPENBLAS_CORETYPE is for determinism only -- it is NOT the fix, since the
# faults reproduce under every coretype including AVX-only SANDYBRIDGE.
os.environ.setdefault('OPENBLAS_CORETYPE', 'HASWELL')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import numpy as np                                        # noqa: E402
import torch                                              # noqa: E402
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ssvr_qec as m                                      # noqa: E402
import vscr_paper as vp                                   # noqa: E402
import vscr_paper_abl as ab                               # noqa: E402
import stationarity_boundary as sb                        # noqa: E402

def _t_ancilla():
    """The Kraus-rank ladder: rank(C) counts the ancillas a branch recovery needs.

    Runs all five of `ancilla_recovery`'s self-tests.  Each is anchored either to
    production code or to a route that shares nothing with the Choi program:
    conventions (r=1 against `ab._max_unitary_J`, r=4 against `ab._sdp_branch`,
    the multi-Kraus Choi objective against `g.cf_unnormalised`, ladder monotone in
    rank); trace preservation and the Stinespring dilation (exactly TP Kraus pair,
    exactly unitary U, partial trace reproduces Phi, and an unconstrained
    `expm(iH)` optimisation sharing no variables or constraints lands on the same
    value); a brute-force Haar quadrature of the physical estimator, using neither
    the Choi matrix nor the degree-2 moment identity; the (n+1)-qubit dilation
    assembled on the physical register and checked against the paper's own
    V^dagger R_s W_s = G_s reduced-block map; and both control channels against a
    FLAT ladder, which is the negative control against solver slack."""
    import ancilla_recovery as ar
    assert ar.main(['--selftest']) == 0


def _t_multiseed():
    """T3 statistics layer: the arithmetic that turns 16 seed runs into a claim.

    Checked against hand-computed values rather than against itself: the sample
    std really is ddof=1 and the sem really is std/sqrt(n); the two differently
    shaped refine records that `train_warm_best` and `refine_with_floor` produce
    normalise to ONE canonical record, without which warm and VQR-ind would not be
    comparable; production's label-free selection is reproduced, and because it
    ranks on worst-branch conditional fidelity rather than on F, the bias of
    "best of two" is asserted to be able to come out NEGATIVE; a single-seed group
    reports std 0.0 rather than NaN (a bare np.std(ddof=1) would poison the JSON
    artifact); and the two degenerate ratios -- a sigma count when every seed
    lands on the same optimum, and a signal-to-noise when the certified headroom
    is itself zero -- must come back undefined rather than as large meaningless
    numbers.  The physics is covered separately by `--verify-reproduction`, which
    re-runs the production seeds and requires bit-exact agreement with
    paper_numbers.json."""
    import multiseed_stats as ms
    assert ms.main(['--selftest']) == 0


def _t_storage():
    """T4 multi-round storage: every reduction this module makes, measured.

    A multi-round number is only meaningful if it collapses onto the audited
    single-round number at R=1, so that is asserted first and hardest: the full
    dim x dim channel is BIT-EXACT against `ssvr_qec.apply_channel` on all four
    channels, the reduced 2x2 round map at R=1 reproduces `abl.exact_F` and
    `opt_unitary_ceiling[F_dec]`, and the 'Raw' baseline equals the direct
    code-space restriction of the audited channel (and is trace-decreasing, since
    uncorrected errors leave the code space).

    Then the three traps that were each found and fixed while building this:
    `code.lam` is only defined on the centralizer, so a Pauli-mixture 'Raw' map
    built over all 4^n Paulis aborts on its unitarity assert; the composition of
    code-space restrictions is NOT the restriction of the composition, so the
    two-stage 'mixed' channel is refused by the closed form; and the ceiling's
    optimal G lives in `vscr_paper_abl`'s syndrome basis while the branch
    operators live in `vscr_general`'s, so without the transport G^g = G^a T^dag
    the ceiling appears 0.15 BELOW the decoder -- impossible, and now asserted
    impossible.

    Finally the physics claims: the Pauli readout law extends exactly to
    R rounds as F(eta,R) = (1-eta)^(mR) F(0,R) while the learned non-Pauli
    recovery violates it (at only ~5e-11, which is asserted too, so the decoder
    check is not vacuous); and the reduced map tracks the full density matrix over
    12 rounds with leakage that SATURATES rather than compounds."""
    import storage_rounds as sr
    assert sr.main(['--selftest']) == 0


# name -> (callable, slow?, description)
TESTS = [
    ('gates',       m._selftest_gates, False,
     'gate / embedding conventions'),
    ('warmstart',   vp._verify_warm_start, False,
     'warm start == Pauli decoder, phase-exact'),
    ('actioncols',  m._selftest_action_cols, False,
     'column-propagation ansatz == explicit 32x32 (value + gradient)'),
    ('liealg',      ab._selftest_liealg, False,
     'closed-form exp(i h.sigma) == scipy.expm, unitary'),
    ('petz',        m._selftest_petz, False,
     'Petz / noise-adapted recovery baseline (Fix E)'),
    ('quadrature',  lambda: m._selftest_quadrature(n_mc=600), True,
     'exact Haar x p quadrature == full reference == Monte-Carlo'),
    ('hypernet',    vp._selftest_hypernet_trainable, True,
     'hypernetwork is trainable (not the dead double-zero init)'),
    ('sdp',         ab._selftest_sdp, True,
     'CPTP ceiling SDP'),
    ('refine',      lambda: ab._selftest_refine(n_start=3, steps=900), True,
     'Fix A/B: decoder is an exact saddle; refinement is monotone + captures'),
    ('stationarity', sb._selftest_certificate, True,
     'nine-sector certificate: phi^dec is stationary for EVERY pure-state input '
     'ensemble and for both loss functionals, not just the sampled ones'),
    ('ancilla',     _t_ancilla, True,
     'Kraus-rank ladder, five self-tests: rank(C)=1 reproduces the production '
     'unitary ceiling and rank(C)=4 the production CPTP ceiling, the Choi '
     'objective equals production cf_unnormalised, the rank-2 optimum is exactly '
     'trace preserving with an exactly unitary one-ancilla Stinespring dilation, '
     'an independent unconstrained exp(iH) circuit optimisation lands on the '
     'same value, a brute-force Haar quadrature of the physical estimator '
     'agrees, the (n+1)-qubit dilation exists on the PHYSICAL register, and '
     'both control channels give a FLAT ladder'),
    ('multiseed',   _t_multiseed, False,
     'T3 statistics layer: sample std (ddof=1) and sem against hand-computed '
     'values, the two refine-record shapes normalise to one canonical record so '
     'warm and VQR-ind are comparable, production\'s label-free selection is '
     'reproduced (and because it ranks on worst-branch cf the "best of two" bias '
     'is asserted able to come out negative), a single-seed group reports std 0.0 '
     'not NaN, and the two degenerate ratios come back undefined rather than as '
     'large meaningless numbers'),
    ('storage',     _t_storage, True,
     'T4 multi-round storage: the full dim x dim channel is BIT-EXACT against '
     'ssvr_qec.apply_channel, the reduced 2x2 round map at R=1 reproduces '
     'abl.exact_F and opt_unitary_ceiling[F_dec] so an R-round number is a '
     'statement about the paper\'s quantity, the \'Raw\' baseline equals the '
     'direct code-space restriction and is trace-decreasing, the ceiling\'s G is '
     'transported between the two syndrome bases (without which it sits 0.15 '
     'BELOW the decoder), the Pauli readout law extends exactly to R rounds as '
     '(1-eta)^(mR) while the learned recovery violates it only at 1e-11, and the '
     'reduced map tracks the full density matrix over 12 rounds with leakage '
     'that saturates rather than compounds'),
]


def run_named(name):
    """Entry point for the isolated subprocess: run exactly one test by name."""
    for n, fn, _slow, _d in TESTS:
        if n == name:
            fn()
            return
    raise SystemExit(f'unknown test {name!r}')


def _run_one_isolated(name, attempts=4):
    """Run one test in a FRESH interpreter, retrying on native faults.

    This host intermittently SIGSEGVs / SIGILLs inside numpy's complex128 GEMM
    (see the long note in `ssvr_qec.py`).  It has been observed on an 8x8 matmul
    inside `vscr_paper_abl.c_tr`, which cannot be a logic error.  Such a fault
    kills the whole process and would abort every remaining test, so each test
    gets its own subprocess and is retried.  `OPENBLAS_CORETYPE=HASWELL` cuts the
    fault rate ~5x but does not remove it, hence the retry rather than one shot.
    Returns True on success; child stdout/stderr are echoed as they arrive.
    """
    code = (f"import sys; sys.path.insert(0, {HERE!r});"
            f"import run_selftests as R; R.run_named({name!r})")
    for k in range(attempts):
        r = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', code],
                           cwd=HERE, capture_output=True, text=True)
        if r.stdout:
            sys.stdout.write(r.stdout)
            sys.stdout.flush()
        if r.returncode == 0:
            if k:
                print(f'    [isolated] {name}: succeeded on attempt {k + 1}',
                      flush=True)
            return True
        for ln in (r.stderr or '').splitlines():
            if not ln.startswith('Extension modules:'):
                print('    ' + ln, flush=True)
        print(f'    [isolated] {name}: exit={r.returncode} '
              f'(attempt {k + 1}/{attempts})', flush=True)
    return False


def main(argv):
    if '--list' in argv:
        for name, _fn, slow, desc in TESTS:
            print(f'  {name:12s} {"[slow]" if slow else "      "} {desc}')
        return 0
    quick = '--quick' in argv
    in_proc = '--in-process' in argv
    picked = [a for a in argv if not a.startswith('-')]
    todo = [(n, f, s, d) for (n, f, s, d) in TESTS
            if (not picked or n in picked) and (not quick or not s)]
    print(f'=== run_selftests: {len(todo)} test(s), '
          f'torch threads={torch.get_num_threads()}, '
          f'{"in-process" if in_proc else "isolated subprocesses"} ===\n',
          flush=True)
    npass, nfail = [], []
    t_all = time.time()
    for name, fn, slow, desc in todo:
        print(f'--- [{name}] {desc} ---', flush=True)
        t0 = time.time()
        if not in_proc:
            if _run_one_isolated(name):
                npass.append(name)
                print(f'--- [{name}] passed in {time.time()-t0:.1f}s\n',
                      flush=True)
            else:
                nfail.append(name)
                print(f'--- [{name}] FAILED (native fault) after '
                      f'{time.time()-t0:.1f}s\n', flush=True)
            continue
        try:
            fn()
        except Exception:
            traceback.print_exc()
            nfail.append(name)
            print(f'--- [{name}] FAILED after {time.time()-t0:.1f}s\n', flush=True)
        else:
            npass.append(name)
            print(f'--- [{name}] passed in {time.time()-t0:.1f}s\n', flush=True)
    print('=' * 66)
    print(f'SELFTESTS: {len(npass)} passed, {len(nfail)} failed '
          f'({time.time()-t_all:.0f}s total)')
    if nfail:
        print('failed:', ', '.join(nfail))
    print('=' * 66)
    return 1 if nfail else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
