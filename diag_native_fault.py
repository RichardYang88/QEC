#!/usr/bin/env python3
"""Measure the host's intermittent native fault rate, and localise it.

WHY THIS EXISTS
---------------
Long runs of this repo intermittently die with SIGSEGV or SIGILL. The crash frame
*moves* between unrelated libraries (`torch.autograd._engine_run_backward`,
scipy's L-BFGS-B `approx_derivative`, a bare `numpy.matmul`, an 8x8 matmul in
`vscr_paper_abl.c_tr`, and most often `vscr_paper_abl._block_obj_grad`). Every
affected routine returns numerically correct results whenever it completes, and
the workloads are (2,32), (32,32) and (8,8) complex128 operations that cannot
corrupt memory from a logic error.

The first hypothesis -- OpenBLAS `DYNAMIC_ARCH` dispatching an AVX-512 kernel this
CPU lacks -- is REFUTED by the measurements below, and this script exists so that
can be re-checked rather than re-assumed:

  * this CPU (Intel Core Ultra 9 285K, Arrow Lake) has no AVX-512 at all
    (`grep avx512 /proc/cpuinfo` is empty; Arrow Lake ships AVX2/FMA3/AVX-VNNI
    only), so there is no AVX-512 kernel to mis-select;
  * forcing the most conservative kernel set still faults with SIGILL. An illegal
    instruction cannot be produced by a correctly compiled AVX-only kernel on a
    CPU that implements AVX.

An illegal instruction in a valid binary, at a moving site, with correct
arithmetic, is a hardware-level symptom (CPU or memory), not a library bug. This
script does not fix that -- it quantifies it, so the retry budget in
`vscr_paper._refine_isolated` and `run_selftests.py` can be sized from evidence,
and so the rate can be compared before/after a BIOS, microcode or memory change.

MEASURED ON THIS HOST (400k calls of `_block_obj_grad`, single-threaded, idle box)
---------------------------------------------------------------------------------
  HASWELL     / all cores   SIGSEGV  within the first 2e4 calls
  unset       / all cores   SIGILL   at ~2.5e5 calls
  SANDYBRIDGE / all cores   SIGILL   at ~1.5e5 calls
and, for contrast, 2e7 bare (2,32)@(32,32) complex matmuls ran clean -- so the
trigger is the operation mix inside `_block_obj_grad`, not GEMM as such. Note
HASWELL faulted *earliest*: pinning the coretype does not help, and an earlier
claim that it cut the rate ~5x was sampling noise on a 6-run sample.

USAGE
-----
  diag_native_fault.py                 # run every arm, report the fault per arm
  diag_native_fault.py --calls 200000  # shorter/longer exposure per arm
  diag_native_fault.py --bare          # also hammer bare matmuls for contrast
  diag_native_fault.py --probe N LABEL # internal: the child workload

Exit status is always 0 (a faulting host is the expected finding here); read the
table. Run it on an otherwise idle machine -- background load changes the rate.
"""
import os
import sys

# Pin threading BEFORE numpy/torch, as everywhere else in this repo.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PY = sys.executable
SELF = os.path.abspath(__file__)


def _child_probe(calls, label):
    """Tight loop over the REAL `_block_obj_grad`, verifying every 20k calls.

    Runs nothing else: no autograd, no SLSQP, no Adam, no printing in the loop.
    Correctness is checked against a reference so that a *silent* miscompute is
    caught too -- that would also be a hardware symptom, and a subtler one.
    """
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import ssvr_qec as m
    import vscr_paper as vp
    import vscr_paper_abl as ab

    try:
        aff = ','.join(str(c) for c in sorted(os.sched_getaffinity(0)))
    except Exception as exc:
        aff = '? (%s)' % exc
    print(f'[probe {label}] coretype={os.environ.get("OPENBLAS_CORETYPE", "<unset>")} '
          f'affinity={aff} calls={calls}', flush=True)

    B = ab.pavg_bundle('amplitude_damping', (0.02, 0.15), 6)
    Qbar, Mbar = ab._pavg_qbar(B)
    PHI = np.asarray(vp.PHI_DEC.detach().cpu().numpy(),
                     dtype=float).reshape(16, m.PHI_DIM)
    slots = ab._pauli_slots_np()
    print(f'[probe {label}] Qbar{Qbar.shape} Mbar{Mbar.shape} PHI{PHI.shape} '
          f'pauli_slots{slots.shape}', flush=True)

    ref_o = np.array([ab._block_obj_grad(PHI[s], s, Qbar[s], Mbar[s])[0]
                      for s in range(16)], dtype=float)
    ref_g = np.array([ab._block_obj_grad(PHI[s], s, Qbar[s], Mbar[s])[1]
                      for s in range(16)], dtype=float)

    bad = 0
    for i in range(calls):
        s = i % 16
        obj, gr = ab._block_obj_grad(PHI[s], s, Qbar[s], Mbar[s])
        if i % 20000 == 0:
            if not (abs(obj - ref_o[s]) <= 1e-12
                    and np.max(np.abs(gr - ref_g[s])) <= 1e-12):
                bad += 1
                print(f'[probe {label}] *** MISMATCH at call {i} block {s}: '
                      f'dobj={obj - ref_o[s]:.3e} '
                      f'dgrad={np.max(np.abs(gr - ref_g[s])):.3e} ***', flush=True)
            print(f'[probe {label}] heartbeat call={i}', flush=True)
    print(f'[probe {label}] completed {calls} calls, mismatches={bad}', flush=True)
    return 1 if bad else 0


def _child_memcheck(iters, label):
    """Contrast arm: ONLY bare complex matmuls -- no allocation churn, no loops.

    This is the control that shows the fault is not "any complex matmul": 2e7 of
    these ran clean on this host while `_block_obj_grad` died within 2e4 calls.
    """
    import numpy as np
    print(f'[memcheck {label}] iters={iters}', flush=True)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((2, 32)) + 1j * rng.standard_normal((2, 32))
    Pm = rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))
    B8 = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    ref_big, ref_8 = A @ Pm, B8 @ B8
    bad = 0
    for i in range(iters):
        if not (np.array_equal(A @ Pm, ref_big) and np.array_equal(B8 @ B8, ref_8)):
            bad += 1
            print(f'[memcheck {label}] *** MISMATCH at iter {i} ***', flush=True)
        if i % 2000000 == 0:
            print(f'[memcheck {label}] heartbeat iter={i}', flush=True)
    print(f'[memcheck {label}] completed {iters} iters, mismatches={bad}',
          flush=True)
    return 1 if bad else 0


def _hostinfo():
    """Facts that decide whether an ISA-dispatch explanation is even possible."""
    flags, model, micro = set(), '?', '?'
    try:
        for ln in open('/proc/cpuinfo'):
            if ln.startswith('model name') and model == '?':
                model = ln.split(':', 1)[1].strip()
            elif ln.startswith('microcode') and micro == '?':
                micro = ln.split(':', 1)[1].strip()
            elif ln.startswith('flags'):
                flags = set(ln.split(':', 1)[1].split())
                break
    except OSError as exc:
        print(f'  (could not read /proc/cpuinfo: {exc})')
    print(f'  cpu          : {model}')
    print(f'  microcode    : {micro}')
    print(f'  logical cpus : {os.cpu_count()}')
    for isa in ('avx512f', 'avx512bw', 'avx512vl', 'avx2', 'fma', 'amx-tile'):
        print(f'  {isa:13s}: {"yes" if isa in flags else "NO"}')
    if not any(f.startswith('avx512') for f in flags):
        print('  -> no AVX-512 on this part, so an AVX-512 mis-dispatch cannot')
        print('     explain a SIGILL and OPENBLAS_CORETYPE cannot be the fix.')


def main(argv):
    if '--probe' in argv:
        i = argv.index('--probe')
        return _child_probe(int(argv[i + 1]),
                            argv[i + 2] if len(argv) > i + 2 else 'p')
    if '--memcheck' in argv:
        i = argv.index('--memcheck')
        return _child_memcheck(int(argv[i + 1]),
                               argv[i + 2] if len(argv) > i + 2 else 'm')

    import signal as _sig
    import subprocess
    import time

    calls = 400000
    if '--calls' in argv:
        calls = int(argv[argv.index('--calls') + 1])
    bare = '--bare' in argv

    print('=== host ===')
    _hostinfo()
    print(f'\n=== arms: {calls} calls of _block_obj_grad each, single-threaded ===',
          flush=True)

    arms = [
        ('HASWELL / all cores',     {'OPENBLAS_CORETYPE': 'HASWELL'},     None),
        ('unset / all cores',       {},                                   None),
        ('SANDYBRIDGE / all cores', {'OPENBLAS_CORETYPE': 'SANDYBRIDGE'}, None),
        ('HASWELL / pinned cpu0',   {'OPENBLAS_CORETYPE': 'HASWELL'},     '0'),
        ('HASWELL / pinned cpu8',   {'OPENBLAS_CORETYPE': 'HASWELL'},     '8'),
    ]
    results = []
    for label, extra, core in arms:
        env = dict(os.environ)
        env.pop('OPENBLAS_CORETYPE', None)
        env.update(extra)
        for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                  'NUMEXPR_NUM_THREADS'):
            env[v] = '1'
        env['CUDA_VISIBLE_DEVICES'] = ''
        cmd = [PY, SELF, '--probe', str(calls), label]
        if core:
            cmd = ['taskset', '-c', core] + cmd
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        dt = time.time() - t0
        if r.returncode < 0:
            verdict = ('FAULT %s (sig %d)'
                       % (_sig.Signals(-r.returncode).name, -r.returncode))
        elif r.returncode == 0:
            verdict = 'clean'
        else:
            verdict = 'MISMATCH/err (rc=%d)' % r.returncode
        hb = [ln for ln in r.stdout.splitlines()
              if 'heartbeat' in ln or 'completed' in ln or 'MISMATCH' in ln]
        print('  %-24s %7.1fs  %-24s %s'
              % (label, dt, verdict, hb[-1].strip() if hb else '(no heartbeat)'),
              flush=True)
        results.append((label, verdict))

    if bare:
        print('\n=== contrast: 2e7 bare complex matmuls ===', flush=True)
        env = dict(os.environ)
        env['OPENBLAS_CORETYPE'] = 'HASWELL'
        t0 = time.time()
        r = subprocess.run([PY, SELF, '--memcheck', '20000000', 'bare-matmul'],
                           capture_output=True, text=True, env=env)
        tail = [ln for ln in r.stdout.splitlines()
                if 'completed' in ln or 'MISMATCH' in ln]
        verdict = ('clean' if r.returncode == 0
                   else 'FAULT/err rc=%d' % r.returncode)
        print('  %-24s %7.1fs  %-24s %s'
              % ('bare (2,32)@(32,32)', time.time() - t0, verdict,
                 tail[-1].strip() if tail else ''))

    nf = sum(1 for _, v in results if v.startswith('FAULT'))
    nm = sum(1 for _, v in results if v.startswith('MISMATCH'))
    print('\n=== %d arms: %d native faults, %d silent mismatches, %d clean ==='
          % (len(results), nf, nm, len(results) - nf - nm))
    print('Reading this:')
    print('  - faults under EVERY coretype, incl. AVX-only SANDYBRIDGE')
    print('      -> not an ISA-dispatch problem; OPENBLAS_CORETYPE is not a fix')
    print('  - a SIGILL at all, on a CPU with no AVX-512')
    print('      -> the CPU executed an invalid instruction: hardware-level')
    print('  - faults on only SOME pinned cores -> suspect that core/cluster')
    print('  - silent MISMATCH -> arithmetic corruption; distrust the results')
    print('  - all arms clean -> the host may have been fixed (BIOS/microcode/')
    print('    memory); consider lowering the retry budgets')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
