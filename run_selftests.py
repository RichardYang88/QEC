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
