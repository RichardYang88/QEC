#!/usr/bin/env python3
"""multiseed_stats.py -- T3: how much of the reported VSCR number is seed luck?

THE PROBLEM THIS CLOSES.  Every trained quantity in the paper comes from
`vscr_paper.train_warm_best(noise, seeds=(1234, 2024))`, which trains two seeds
and keeps the one with the better label-free worst-branch conditional fidelity --
main.tex says so outright ("VSCR reports the best of two").  Reporting a maximum
over draws without the spread they were drawn from is a real methodological gap:
a reviewer cannot tell whether the non-Pauli headroom the paper claims (1.9e-4 to
3.7e-3) is a property of the method or the top of a wide seed distribution.  And
the VQR-ind baseline is trained on a SINGLE seed (1234) while being compared
against a best-of-two VSCR, which is not an even comparison.

WHAT IS MEASURED.  S seeds x 4 channels x 2 protocols, each seed run through the
*unmodified production path* -- `vp.train_warm_best(noise, seeds=(seed,))`, and
`abl.train_ind(noise, seeds=(seed,))` followed by the identical
`vp.refine_with_floor` post-processing that `abl.main` applies.  Per seed we
record the exact quadrature objective at every stage (F_decoder, F_grad,
F_refined, F_used, F_cert), the certified headroom and capture fraction, the
label-free selection diagnostic, and the exact Haar-averaged fidelity curve
`abl.exact_F(phi, noise, p)` on the paper's p grid.  Every one of those is
DETERMINISTIC, so the spread reported here is seed variability and nothing else
-- no Monte-Carlo estimator noise is folded into it.

Then, per (protocol, channel): mean, sample std (ddof=1), sem, min, max, and the
three diagnostics that are the actual point of the exercise:
  * `reproduction`         the seed production selected must come back
                           bit-for-bit from this driver, which is what licenses
                           everything else;
  * `selection_bias`       best-of-two minus the S-seed mean, in units of the
                           S-seed std -- how much the headline is inflated;
  * `signal_to_seed_noise` certified headroom over std(F_used) -- how many sigma
                           the claimed effect sits above the seed lottery.

WHY THIS RUNS ON CPU, NOT CUDA -- MEASURED, NOT ASSUMED.  `--gpu-check` measures
the exact quadrature objective's forward+backward on this host (Intel Ultra 9
285K / RTX 5070 Ti, torch 2.13.0+cu130).  Per-epoch work is 60 gates applied to
(16,32,2) columns, ~2 MFLOP, so at one seed the GPU is 2.0x SLOWER than a pinned
CPU core (10.4 ms vs 5.2 ms per step): kernel launch, not arithmetic, dominates.
Batching does amortise on the GPU (64 seeds cost 3.0x one seed, against 33.6x on
CPU), but 16 seeds batched (10.6 ms/step) only matches 8-way CPU process
parallelism (~21 s for the whole curriculum), and getting there means
reimplementing the audited hypernetwork parameterisation in batched form.  Since
this task's entire value rests on seeds 1234/2024 reproducing
`paper_numbers.json` bit-for-bit, that trade is bad: no wall-clock win, and it
costs the reproduction guarantee.  So production runs are process-parallel over 8
pinned P-cores, and the GPU is used for what it does add -- an INDEPENDENT device
cross-check showing the same curriculum lands on the same objective to 3.6e-9 on
a different backend, 390x below the 1.4e-3 headroom, which rules out the seed
spread being a CPU/BLAS artifact.  CUDA is reserved for
`storage_rounds.py` (T4), where 512x512 density matrices make the arithmetic
real rather than launch-bound.

PARALLELISM NOTE.  Workers are spawned with `subprocess.Popen` and pinned to one
CPU each via `preexec_fn`, never forked.  `ssvr_qec._pin_single_cpu()` runs at
IMPORT time and restricts the process to a single core (the documented fix for
this host's intermittent SIGSEGV/SIGILL under cross-cluster migration); a forked
child would inherit that one-core mask and every worker would pile onto the same
core, silently serialising the run.  Affinity is scheduling only and cannot
change a computed value, which `--verify-reproduction` proves rather than asserts.

Usage:
    ./qenv/bin/python multiseed_stats.py --seeds 16              # full T3 run
    ./qenv/bin/python multiseed_stats.py --quick                 # smoke budget
    ./qenv/bin/python multiseed_stats.py --verify-reproduction   # lock only
    ./qenv/bin/python multiseed_stats.py --gpu-check             # device bench
    ./qenv/bin/python multiseed_stats.py --selftest              # invariants
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Single-threaded BLAS, set BEFORE numpy is imported -- see the note in
# scaling_analysis.py for why the ordering matters (StabCode's degenerate
# eigenspaces make W_s, and with it every number, thread-count dependent).
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('OPENBLAS_CORETYPE', 'HASWELL')

# numpy is imported only AFTER the thread-count variables above are set: OpenBLAS
# fixes its worker pool at library initialisation, so importing earlier would
# leave this process multi-threaded, and StabCode's degenerate eigenspaces make
# W_s -- hence every reduced block and every number -- thread-count dependent.
import numpy as np                                              # noqa: E402

CHANNELS = ('depolarizing', 'amplitude_damping', 'mixed', 'coherent')
PROTOCOLS = ('warm', 'ind')
# The paper's grid.  exact_F is closed-form, so the whole grid stays cheap.
P_VALUES = (0.005, 0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30)
# Seeds 1234 and 2024 MUST come first: they are the production seeds, and the
# reproduction check compares exactly those against paper_numbers.json.
PROD_SEEDS = (1234, 2024)
SEED_POOL = PROD_SEEDS + tuple(1000 + 37 * i for i in range(2, 80))
# CPUs 0-7 are the P-cores on Arrow Lake-S; ssvr_qec._pin_single_cpu prefers the
# same first eight, so workers land on the same core class as production runs.
DEFAULT_CPUS = tuple(range(8))
OUT_JSON = 'multiseed_results.json'
# Scalar fields carried out of the refine stage for every seed.
REFINE_FIELDS = ('picked', 'F_decoder', 'F_grad', 'F_refined', 'F_used',
                 'F_cert', 'F_bound', 'headroom', 'capture_frac',
                 'headroom_vs_decoder', 'headroom_is_zero', 'capture_vs_decoder',
                 'n_improved', 'unitarity_dev', 'cert_valid', 'prod_err')
# Tolerances.  Reproduction must be BIT-exact: same code, same seed, same core
# class -- anything looser would hide a real nondeterminism.
REPRO_TOL = 0.0
# A seed run is only comparable if the refine stage stayed inside the
# code-preserving regime; production asserts cert_valid, so assert it here too.
UNITARITY_TOL = 1e-4


# ---------------------------------------------------------------------
# Worker: one (protocol, channel, seed) through the UNMODIFIED production path
# ---------------------------------------------------------------------
# `refine_with_floor` reports the three candidate scores under `scores` keyed by
# candidate name, while `train_warm_best` re-publishes them as F_decoder/F_grad/
# F_refined.  Normalise both shapes to the latter so a warm seed and an ind seed
# are directly comparable -- which is the whole point of running both protocols.
_SCORE_KEY = {'F_decoder': 'decoder', 'F_grad': 'grad', 'F_refined': 'refined'}


def _flatten_refine(rinfo):
    """The scalar refine diagnostics of one seed, in one canonical shape."""
    out = {}
    scores = rinfo.get('scores') or {}
    for k in REFINE_FIELDS:
        if k == 'picked':
            out[k] = str(rinfo.get('picked', ''))
            continue
        if k in rinfo:
            v = rinfo[k]
        elif k in _SCORE_KEY and _SCORE_KEY[k] in scores:
            v = scores[_SCORE_KEY[k]]
        elif k == 'F_used' and rinfo.get('picked') in scores:
            v = scores[rinfo['picked']]
        else:
            continue
        if k in ('headroom_is_zero', 'cert_valid'):
            out[k] = bool(v)
        elif k == 'n_improved':
            out[k] = int(v)
        else:
            out[k] = float(v)
    return out


def run_one(protocol, noise, seed, p_values=P_VALUES):
    """One seed of one protocol on one channel, via the production functions.

    Deliberately calls `vp.train_warm_best(noise, seeds=(seed,))` and
    `abl.train_ind(noise, seeds=(seed,))` rather than re-deriving the curriculum:
    with a single seed the best-of-N selection inside those functions is the
    identity, so the returned model is exactly the one production would have
    trained for that seed.  That is what makes the reproduction check below a
    real check and not a comparison of two different implementations.
    """
    import numpy as np
    import torch
    import ssvr_qec as m                       # noqa: F401  pins this CPU on import
    import vscr_paper as vp
    import vscr_paper_abl as abl
    import vscr_paper_coh                      # noqa: F401 registers 'coherent'

    rec = {'protocol': protocol, 'noise': noise, 'seed': int(seed),
           'pinned_cpu': (sorted(os.sched_getaffinity(0))[0]
                          if hasattr(os, 'sched_getaffinity') else None)}
    t0 = time.time()
    if protocol == 'warm':
        model, _hist, infos = vp.train_warm_best(noise, seeds=(seed,),
                                                 verbose=False)
        with torch.no_grad():
            phi = np.asarray(model().detach().cpu().numpy(), dtype=float).copy()
        sinfo = [x for x in infos if 'seed' in x][0]
        rinfo = [x for x in infos if x.get('stage') == 'refine'][0]
        p_range = tuple(vp.WARM_SCHEDULES[noise][-1][2])
    elif protocol == 'ind':
        model, _hist, infos = abl.train_ind(noise, seeds=(seed,))
        phi_grad = model().detach().cpu().numpy()
        p_range = tuple(abl.SCHED[noise][-1][2])
        # The identical post-processing `abl.main` applies, so the ablation stays
        # architecture-only rather than architecture-plus-refinement.
        phi, rinfo = vp.refine_with_floor(phi_grad, noise, p_range,
                                          verbose=False, tag='/ind')
        phi = np.asarray(phi, dtype=float)
        sinfo = [x for x in infos if 'seed' in x][0]
    else:
        raise ValueError('unknown protocol %r' % (protocol,))

    rec['min_cf'] = float(sinfo['min_cf'])
    rec['val_fid'] = float(sinfo['val_fid'])
    rec.update(_flatten_refine(rinfo))
    rec['p_range'] = list(p_range)
    # Exact, zero-Monte-Carlo-error fidelity curve via the closed second-moment
    # identity, so the spread across seeds below is seed variability alone.
    # NOTE `abl.exact_F` returns (F_total, per_branch_conditional_fidelities);
    # the second element is kept because it shows WHETHER the seed spread is
    # uniform or concentrated in one bad branch -- which is exactly the failure
    # mode the label-free worst-branch selection criterion exists to catch.
    curves = [abl.exact_F(phi, noise, p) for p in p_values]
    rec['F_exact'] = [float(f) for f, _cfs in curves]
    rec['cf_worst_exact'] = [float(np.min(cfs)) for _f, cfs in curves]
    rec['p_values'] = [float(p) for p in p_values]
    # phi itself is NOT comparable across seeds -- the same F is realised by a
    # whole manifold of angle tables (the device cross-check moves phi by 2.9e-2
    # while moving F by 3.6e-9), so record a digest for provenance only.
    rec['phi_sha256'] = hashlib.sha256(
        np.ascontiguousarray(phi, dtype=np.float64).tobytes()).hexdigest()
    rec['unitarity_dev'] = float(rinfo.get('unitarity_dev', np.nan))
    rec['wall_s'] = time.time() - t0
    assert rec.get('cert_valid', False), \
        'seed %d left the code-preserving regime: unitarity_dev=%.3e' % (
            seed, rec['unitarity_dev'])
    assert rec['unitarity_dev'] < UNITARITY_TOL, rec['unitarity_dev']
    return rec


# ---------------------------------------------------------------------
# Child entry point (one job per fresh interpreter)
# ---------------------------------------------------------------------
def _child(spec_path, out_path):
    with open(spec_path) as fh:
        spec = json.load(fh)
    rec = run_one(spec['protocol'], spec['noise'], spec['seed'],
                  tuple(spec.get('p_values', P_VALUES)))
    tmp = out_path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(rec, fh)
    os.replace(tmp, out_path)
    return 0


def _child_env():
    """Worker environment: single-threaded BLAS, deterministic coretype, and CUDA
    hidden -- this path is CPU by decision (module docstring), and hiding the
    device makes that decision impossible to violate by accident."""
    env = dict(os.environ)
    env['PYTHONPATH'] = ROOT + os.pathsep + env.get('PYTHONPATH', '')
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'OPENBLAS_MAIN_FREE'):
        env.setdefault(v, '1')
    env.setdefault('OPENBLAS_CORETYPE', 'HASWELL')
    env.setdefault('CUDA_VISIBLE_DEVICES', '')
    return env


def _spawn_async(spec, out_path, cpu, spec_dir):
    """Write `spec` and launch one job in a fresh interpreter pinned to `cpu`.

    A fresh interpreter per job is not isolation for its own sake: this host
    intermittently SIGSEGVs/SIGILLs in tiny complex128 kernels (documented at the
    top of ssvr_qec.py, diagnosed by diag_native_fault.py), and
    `vscr_paper._refine_isolated` already relies on retrying in a clean child.
    Doing the same one level up means a fault in the gradient stage costs one
    retry rather than the whole sweep.

    The pin is applied in the child via `preexec_fn`, BEFORE exec, so that
    `ssvr_qec._pin_single_cpu()` -- which runs at import time and picks
    `pool[pid % 8]` out of whatever affinity it finds -- sees a one-element mask
    and lands on exactly `cpu`.  Forking instead would inherit the parent's
    already-narrowed mask and pile every worker onto a single core.
    """
    sp = os.path.join(spec_dir, 'spec_%s_%s_%d.json'
                      % (spec['protocol'], spec['noise'], spec['seed']))
    with open(sp, 'w') as fh:
        json.dump(spec, fh)
    if os.path.exists(out_path):
        os.remove(out_path)

    def _pin():
        os.sched_setaffinity(0, {cpu})

    return subprocess.Popen(
        [sys.executable, '-X', 'faulthandler',
         os.path.join(ROOT, 'multiseed_stats.py'), '--child', sp, out_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        env=_child_env(), cwd=ROOT,
        preexec_fn=(_pin if hasattr(os, 'sched_setaffinity') else None))


def _run_single(spec, cpu, timeout):
    """Blocking one-job run, used by --verify-reproduction."""
    with tempfile.TemporaryDirectory(prefix='ms1_') as td:
        op = os.path.join(td, 'r.json')
        proc = _spawn_async(spec, op, cpu, td)
        t0 = time.time()
        while proc.poll() is None:
            if time.time() - t0 > timeout:
                proc.kill()
                raise RuntimeError('timeout on %s' % (spec,))
            time.sleep(0.3)
        if proc.returncode != 0 or not os.path.exists(op):
            raise RuntimeError('job %s failed: %s'
                               % (spec, (proc.stderr.read() or '')[-600:]))
        with open(op) as fh:
            return json.load(fh)


def _run_sweep(jobs, cpus, timeout, max_attempts=3, log=print):
    """Execute `jobs` across `cpus`, at most one job per CPU at a time.

    Failures are re-queued with a BOUNDED attempt count: the native faults this
    host produces are transient and move around, so a retry in a clean
    interpreter is the established mitigation, but an unbounded retry would turn
    a genuine bug into a hang.
    """
    out = []
    pending = [(j, 1) for j in jobs]
    busy = {}                     # cpu -> (spec, out_path, proc, t_start, attempt)
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix='msres_') as td:
        idx = 0
        while pending or busy:
            for cpu in cpus:
                if cpu in busy or not pending:
                    continue
                spec, att = pending.pop(0)
                op = os.path.join(td, 'r%d.json' % idx)
                idx += 1
                log('  [launch] %-4s %-18s seed %-5d -> cpu%d (attempt %d; '
                    '%d queued, %.0fs elapsed)'
                    % (spec['protocol'], spec['noise'], spec['seed'], cpu, att,
                       len(pending), time.time() - t0))
                busy[cpu] = (spec, op, _spawn_async(spec, op, cpu, td),
                             time.time(), att)
            finished = []
            for cpu, (spec, op, proc, ts, att) in busy.items():
                alive = proc.poll() is None
                if alive and time.time() - ts < timeout:
                    continue
                if alive:                        # over budget: kill, then retry
                    proc.kill()
                    proc.wait()
                    err = 'timeout after %ss' % timeout
                else:
                    err = (proc.stderr.read() or '').strip()
                finished.append(cpu)
                if proc.returncode == 0 and os.path.exists(op):
                    with open(op) as fh:
                        rec = json.load(fh)
                    rec['attempts'] = att
                    out.append(rec)
                    log('  [done  ] %-4s %-18s seed %-5d %6.1fs  '
                        'F_used=%.12f' % (spec['protocol'], spec['noise'],
                                          spec['seed'], rec['wall_s'],
                                          rec.get('F_used', float('nan'))))
                elif att < max_attempts:
                    tail = (err.splitlines() or ['(no stderr)'])[-1]
                    log('  [RETRY ] %-4s %-18s seed %-5d attempt %d failed: %s'
                        % (spec['protocol'], spec['noise'], spec['seed'], att,
                           tail))
                    pending.insert(0, (spec, att + 1))
                else:
                    raise RuntimeError(
                        'job %s/%s/seed %d failed %d attempts; last: %s'
                        % (spec['protocol'], spec['noise'], spec['seed'], att,
                           err[-600:]))
            for cpu in finished:
                del busy[cpu]
            if busy:
                time.sleep(0.4)
    if len(out) != len(jobs):
        raise RuntimeError('only %d/%d jobs produced results'
                           % (len(out), len(jobs)))
    out.sort(key=lambda r: (r['protocol'], r['noise'], r['seed']))
    return out


# ---------------------------------------------------------------------
# Aggregation: mean +/- std over seeds
# ---------------------------------------------------------------------
SCALARS = ('min_cf', 'val_fid', 'F_decoder', 'F_grad', 'F_refined', 'F_used',
           'F_cert', 'F_bound', 'headroom', 'headroom_vs_decoder',
           'capture_frac', 'capture_vs_decoder', 'n_improved', 'unitarity_dev',
           'prod_err')


def _stats(vals):
    """mean / sample std (ddof=1) / sem / min / max, plus the raw values.

    ddof=1 because these are S independent draws, not a population: the
    quantity of interest is the spread a *future* seed would show.
    """
    a = [float(v) for v in vals]
    n = len(a)
    mu = sum(a) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in a) / (n - 1)) if n > 1 else 0.0
    return {'n': n, 'mean': mu, 'std': sd,
            'sem': (sd / math.sqrt(n) if n > 1 else 0.0),
            'min': min(a), 'max': max(a), 'values': a}


def _curve_stats(curves):
    """Per-index mean/std/sem/min/max over a list of equal-length curves.

    The std is the sample std (ddof=1) and is 0.0 -- not NaN -- for a single
    curve, matching `_stats`.  Pure Python rather than np.std(ddof=1) so the n=1
    case cannot emit a RuntimeWarning or silently poison a JSON artifact with NaN,
    which `json.dump` would write as the non-standard token `NaN`.
    """
    n = len(curves)
    out = {'n': n, 'mean': [], 'std': [], 'sem': [], 'min': [], 'max': []}
    for i in range(len(curves[0])):
        col = [float(c[i]) for c in curves]
        mu = sum(col) / n
        sd = (math.sqrt(sum((x - mu) ** 2 for x in col) / (n - 1))
              if n > 1 else 0.0)
        out['mean'].append(mu)
        out['std'].append(sd)
        out['sem'].append(sd / math.sqrt(n) if n > 1 else 0.0)
        out['min'].append(min(col))
        out['max'].append(max(col))
    return out


def aggregate(records, p_values=P_VALUES):
    """Group per-seed records into per-(protocol, channel) mean +/- std."""
    groups = {}
    for r in records:
        groups.setdefault((r['protocol'], r['noise']), []).append(r)
    agg = {}
    for (proto, noise), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r['seed'])
        g = {'n_seeds': len(rs), 'seeds': [r['seed'] for r in rs],
             'p_values': [float(p) for p in p_values]}
        g['scalars'] = {k: _stats([r[k] for r in rs])
                        for k in SCALARS if all(k in r for r in rs)}
        # The fidelity curve, per p: mean +/- std across seeds.  exact_F is
        # closed-form, so this spread contains no estimator noise at all.
        # `_curve_stats` rather than a bare np.std(ddof=1) because a single-seed
        # group (e.g. --seeds 1) would otherwise yield NaN and a RuntimeWarning,
        # disagreeing with `_stats`, which reports 0.0 for n=1.
        g['F_exact'] = _curve_stats([r['F_exact'] for r in rs])
        if all('cf_worst_exact' in r for r in rs):
            g['cf_worst_exact'] = _curve_stats([r['cf_worst_exact'] for r in rs])
        g['picked_counts'] = {}
        for r in rs:
            g['picked_counts'][r.get('picked', '?')] = \
                g['picked_counts'].get(r.get('picked', '?'), 0) + 1
        g['wall_s'] = _stats([r['wall_s'] for r in rs])
        g['per_seed'] = rs
        agg['%s/%s' % (proto, noise)] = g
    return agg


# ---------------------------------------------------------------------
# Reproduction lock: the production seeds must come back bit-for-bit
# ---------------------------------------------------------------------
def _production_infos(noise):
    """The per-seed + refine records the paper was built from."""
    with open(os.path.join(ROOT, 'paper_numbers.json')) as fh:
        pn = json.load(fh)
    warm = (pn['infos_coh'] if noise == 'coherent'
            else pn['infos'].get(noise))
    ind = pn['abl']['ind'][noise]['infos']
    return warm, ind


def _select_prod(rs):
    """Replicate production's label-free selection: max by (min_cf, val_fid)."""
    return max(rs, key=lambda r: (r['min_cf'], r['val_fid']))


def verify_reproduction(by_key, scope=None, log=print):
    """Require BIT-exact agreement with the numbers the paper already reports.

    This is the check that licenses the whole artifact.  It runs after the sweep
    has been distributed over pinned cores in fresh interpreters, so it
    simultaneously proves that (a) this driver calls the same code path
    production did and (b) CPU affinity does not perturb a single computed bit --
    the claim `ssvr_qec._pin_single_cpu` makes in a comment, here measured.

    `scope` is the set of (protocol, channel) pairs actually present in this run.
    Configurations outside it are counted as SKIPPED, never as failures: a
    --quick run over one channel must not be reported as having failed to
    reproduce the other three, or the check would be useless as a gate.
    """
    fails = []
    skipped = []
    checked = 0
    for noise in CHANNELS:
        prod_warm, prod_ind = _production_infos(noise)
        for proto, prod in (('warm', prod_warm), ('ind', prod_ind)):
            if scope is not None and (proto, noise) not in scope:
                skipped.append('%s/%s' % (proto, noise))
                continue
            if not prod:
                skipped.append('%s/%s (no production record)' % (proto, noise))
                continue
            seed_recs = [x for x in prod if 'seed' in x]
            ref_recs = [x for x in prod if x.get('stage') == 'refine']
            if not seed_recs or not ref_recs:
                skipped.append('%s/%s (incomplete production record)'
                               % (proto, noise))
                continue
            sel = _select_prod(seed_recs)
            ref = ref_recs[0]
            got = by_key.get((proto, noise, sel['seed']))
            tag = '%s/%s seed %d' % (proto, noise, sel['seed'])
            if got is None:
                # In scope but absent: the sweep should have produced it.
                fails.append('%s: in scope but missing from this run' % tag)
                continue
            n_here = 0
            # (1) the seed record: min_cf and val_fid, bit-exact
            for k in ('min_cf', 'val_fid'):
                checked += 1
                n_here += 1
                if float(got[k]) != float(sel[k]):
                    fails.append('%s: %s %.17g != production %.17g'
                                 % (tag, k, got[k], sel[k]))
            # (2) the refine record: production's F_* are the SELECTED seed's
            want = {'F_grad': ref.get('F_grad'),
                    'F_refined': ref.get('F_refined'),
                    'F_decoder': ref.get('F_decoder'),
                    'capture_frac': ref.get('capture_frac')}
            for k, wv in want.items():
                if wv is None:
                    continue
                checked += 1
                n_here += 1
                if float(got[k]) != float(wv):
                    fails.append('%s: %s %.17g != production %.17g'
                                 % (tag, k, got[k], wv))
            log('  [repro] %-30s production-selected; %d fields bit-compared'
                % (tag, n_here))
    return {'n_fields_compared': checked, 'n_failures': len(fails),
            'failures': fails, 'bit_exact': not fails and checked > 0,
            'n_configs_checked': len(scope) if scope else None,
            'skipped': skipped}


# ---------------------------------------------------------------------
# Diagnostics: the actual point of the exercise
# ---------------------------------------------------------------------
# main.tex (Methods) quotes a 48-sample Monte-Carlo SEM of 1.1e-4 at amplitude
# damping p=0.10.  It is the natural yardstick for the seed spread: if seeds move
# the answer by far less than the estimator the paper deliberately avoided, the
# seed lottery is not what limits any reported number.
MC_SEM_REFERENCE = 1.1e-4


def diagnostics(agg):
    """Per (protocol, channel): selection bias and signal-to-seed-noise.

    Three numbers matter and they answer three different reviewer questions.

    `selection_bias_*`  -- the paper reports the best of two seeds.  How much
        does that overstate the typical seed?  Computed as the production-selected
        seed's F_used minus the S-seed mean, both absolutely and in units of the
        S-seed std, so "0.03 sigma" and "2 sigma" are distinguishable at a glance.

    `max_over_seeds_*`  -- the same quantity for a best-of-S rule, i.e. the bias
        one would incur by reporting the top of the full sweep instead of the top
        of two.  Reported so the choice of S is visible rather than hidden.

    `signal_to_seed_noise` -- the certified non-unitary headroom over std(F_used).
        This is the number that decides whether the effect is real: the headroom
        is a property of the channel and the code, and if it sits hundreds of
        sigma above the seed spread then no seed could have produced it by luck.

    Also records that the headroom itself has ZERO seed spread (F_cert and
    F_decoder are both functions of the channel alone), and that every seed
    respects the decoder floor and the certified ceiling.
    """
    out = {}
    for key, g in sorted(agg.items()):
        sc = g['scalars']
        if 'F_used' not in sc:
            continue
        fu = sc['F_used']
        std = fu['std']
        mean = fu['mean']
        recs = g['per_seed']
        # Which seed production would have selected, and what it reports.
        prod_seed_recs = [r for r in recs if r['seed'] in PROD_SEEDS]
        sel = (_select_prod(prod_seed_recs) if len(prod_seed_recs) == 2
               else _select_prod(recs))
        best_of_two = float(sel['F_used'])
        top = max(recs, key=lambda r: r['F_used'])
        head = sc.get('headroom_vs_decoder')
        d = {
            'n_seeds': g['n_seeds'],
            'production_seed': int(sel['seed']),
            'F_used_best_of_two': best_of_two,
            'F_used_mean': mean,
            'F_used_std': std,
            'F_used_sem': fu['sem'],
            'F_used_min': fu['min'],
            'F_used_max': fu['max'],
            'selection_bias_abs': best_of_two - mean,
            'selection_bias_sigma': ((best_of_two - mean) / std
                                     if std > 0 else float('inf')
                                     if best_of_two != mean else 0.0),
            'max_over_seeds_abs': float(top['F_used']) - mean,
            'max_over_seeds_sigma': ((float(top['F_used']) - mean) / std
                                     if std > 0 else float('inf')
                                     if float(top['F_used']) != mean else 0.0),
            'max_over_seeds_seed': int(top['seed']),
            'seed_std_vs_mc_sem': std / MC_SEM_REFERENCE,
            'all_seeds_pick_same_candidate': len(g['picked_counts']) == 1,
            'picked_counts': g['picked_counts'],
        }
        if head is not None:
            d['headroom_mean'] = head['mean']
            # F_cert and F_decoder depend on the channel and the p-range only, so
            # the headroom must be seed-independent to machine precision.  A
            # nonzero value here would mean the certification is seed-contaminated.
            d['headroom_std'] = head['std']
            d['headroom_is_seed_independent'] = bool(head['std'] <= 1e-15)
            # On depolarizing and mixed the decoder IS the family optimum, so the
            # certified headroom is zero up to round-off (~1e-14, and signed).
            # Dividing that by a std that is itself at machine epsilon produces a
            # large but MEANINGLESS ratio -- it is 0/0 dressed up as a signal
            # strength.  Report it as None and say why, rather than let a number
            # like "-149 sigma" reach the paper.
            zero_head = all(bool(r.get('headroom_is_zero', False)) for r in recs)
            d['headroom_is_zero'] = zero_head
            d['signal_to_seed_noise'] = (None if zero_head
                                         else (head['mean'] / std if std > 0
                                               else float('inf')))
        # A std at double-precision resolution means every seed converged to the
        # SAME optimum, in which case "bias in units of std" is a ratio of two
        # round-off quantities and must not be read as a sigma count.
        d['std_at_machine_precision'] = bool(std <= 1e-14 * max(1.0, abs(mean)))
        if d['std_at_machine_precision']:
            d['selection_bias_sigma'] = None
            d['max_over_seeds_sigma'] = None
        # Is the seed spread uniform across branches, or concentrated in one bad
        # branch?  The worst-branch conditional fidelity is what the label-free
        # selection ranks on, so if its spread dwarfs the spread of the
        # branch-averaged F, then seed luck lives in a single branch rather than
        # in the reported answer -- which is the failure mode the selection
        # criterion exists to catch, and worth naming explicitly.
        if 'cf_worst_exact' in g:
            sw = g['cf_worst_exact']['std']
            sf = g['F_exact']['std']
            d['max_curve_std_F'] = max(sf)
            d['max_curve_std_worst_cf'] = max(sw)
            d['worst_p_for_seed_spread'] = g['p_values'][int(np.argmax(sf))]
            d['spread_concentration'] = (
                max(sw) / max(sf) if max(sf) > 0
                else (float('inf') if max(sw) > 0 else 0.0))
        # Invariants that must hold for EVERY seed, not just on average.
        if 'F_decoder' in sc:
            d['n_below_decoder_floor'] = sum(
                1 for r in recs if r['F_used'] < r['F_decoder'] - 1e-15)
        if 'F_cert' in sc:
            d['n_above_certified_ceiling'] = sum(
                1 for r in recs
                if r.get('cert_valid') and r['F_used'] > r['F_cert'] + 1e-9)
        out[key] = d
    return out


# ---------------------------------------------------------------------
# --gpu-check: the CUDA device cross-check, and the measurement that decided
# the sweep itself runs on CPU
# ---------------------------------------------------------------------
def _curriculum(model, dev, noise):
    """The production 3-stage curriculum with the quadrature cache on `dev`.

    Replicates the inner loop of `ssvr_qec.train_vscr(quad=True)` exactly --
    Adam, zero_grad, vscr_fidelity_quad, backward, step -- because that function
    builds its own cache on the CPU and offers no way to hand it a device.  The
    per-epoch arithmetic is identical, so any disagreement between the two
    devices below is floating-point association order, not a different
    computation.
    """
    import torch
    import ssvr_qec as m
    import vscr_paper as vp
    t0 = time.time()
    fid = None
    for ep, lr, pr in vp.WARM_SCHEDULES[noise]:
        Q = {k: (v.to(dev) if torch.is_tensor(v) else v)
             for k, v in m.quad_cache(pr, noise).items()}
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(ep):
            opt.zero_grad()
            loss, F, _fs = m.vscr_fidelity_quad(model(), Q)
            loss.backward()
            opt.step()
            fid = float(F)
    dt = time.time() - t0
    with torch.no_grad():
        phi = model().detach().cpu().numpy().copy()
    return fid, phi, dt


def _batched_step_ms(B, dev, noise, reps=200):
    """Milliseconds for one fwd+bwd of the 60-gate column propagation at batch B.

    B=16 is one seed's worth of syndrome branches, so B=16*S is what a
    batched-over-seeds trainer would cost per epoch.  This is the measurement
    that decides whether the GPU's launch overhead can be amortised.
    """
    import torch
    import ssvr_qec as m
    P_CPU = m.P_EXP
    W = (m.quad_cache((0.07, 0.07), noise, n_p=1)['W']
         .repeat(B // 16, 1, 1).to(dev))
    g = torch.Generator().manual_seed(7)
    phi = (torch.randn(B, m.PHI_DIM, dtype=torch.float64, generator=g) * 0.3
           ).to(dev).requires_grad_(True)
    m.P_EXP = [P.to(dev) for P in P_CPU]
    try:
        for _ in range(5):                       # warm up
            Y = m.recovery_action_cols(phi, W)
            (Y.real ** 2 + Y.imag ** 2).sum().backward()
            phi.grad = None
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            Y = m.recovery_action_cols(phi, W)
            (Y.real ** 2 + Y.imag ** 2).sum().backward()
            phi.grad = None
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        return (time.time() - t0) / reps * 1e3
    finally:
        m.P_EXP = P_CPU


def gpu_check(noise='amplitude_damping', seed=1234, log=print):
    """Does an independent backend land on the same objective, and is it faster?

    Q1 is scientific: if the seed spread this task reports were an artifact of
    one CPU's BLAS association order, it would not survive a change of device,
    so agreement here rules that explanation out.  Q2/Q3 are the engineering
    record: their answer is what justifies running the sweep on pinned CPU cores
    rather than the GPU, and it is stored in the artifact so the decision can be
    re-examined instead of taken on faith.
    """
    import torch
    import numpy as np
    import ssvr_qec as m
    import vscr_paper as vp
    import vscr_paper_coh                              # noqa: F401  (schedules)
    if not torch.cuda.is_available():
        return {'available': False,
                'note': 'torch.cuda.is_available() is False on this host'}
    res = {'available': True, 'torch': torch.__version__,
           'cuda_build': torch.version.cuda,
           'device': torch.cuda.get_device_name(0), 'noise': noise,
           'seed': int(seed)}
    log('[gpu-check] %s  torch %s+cu%s'
        % (res['device'], torch.__version__, torch.version.cuda))

    P_CPU = m.P_EXP
    torch.manual_seed(seed); np.random.seed(seed)
    fid_c, phi_c, t_c = _curriculum(vp.VSCRWarm(), torch.device('cpu'), noise)
    m.P_EXP = [P.to(torch.device('cuda')) for P in P_CPU]
    try:
        torch.manual_seed(seed); np.random.seed(seed)
        fid_g, phi_g, t_g = _curriculum(vp.VSCRWarm().to('cuda'),
                                        torch.device('cuda'), noise)
    finally:
        m.P_EXP = P_CPU

    res.update({'cpu_final_F': fid_c, 'cuda_final_F': fid_g,
                'dF_cpu_vs_cuda': abs(fid_c - fid_g),
                'dphi_cpu_vs_cuda': float(np.abs(phi_c - phi_g).max()),
                'cpu_seconds': t_c, 'cuda_seconds': t_g,
                'cuda_speedup_at_one_seed': t_c / t_g})
    log('  Q1 agreement   dF = %.3e   dphi = %.3e'
        % (res['dF_cpu_vs_cuda'], res['dphi_cpu_vs_cuda']))
    log('     (dphi >> dF is expected: one F is realised by a whole manifold of')
    log('      angle tables, so only F is a meaningful coordinate to compare)')
    log('  Q2 one seed    CPU %.2fs vs CUDA %.2fs  -> %.2fx'
        % (t_c, t_g, res['cuda_speedup_at_one_seed']))

    rows = {}
    for B in (16, 64, 256, 1024):
        tc = _batched_step_ms(B, torch.device('cpu'), noise)
        tg = _batched_step_ms(B, torch.device('cuda'), noise)
        rows[str(B)] = {'seeds': B // 16, 'cpu_ms': tc, 'cuda_ms': tg,
                        'cuda_over_cpu': tg / tc}
        log('  Q3 B=%5d (%2d seeds)  CPU %7.3f ms  CUDA %7.3f ms  ratio %.2f'
            % (B, B // 16, tc, tg, tg / tc))
    res['batched_step'] = rows
    res['cpu_scaling_1024_over_16'] = rows['1024']['cpu_ms'] / rows['16']['cpu_ms']
    res['cuda_scaling_1024_over_16'] = (rows['1024']['cuda_ms']
                                        / rows['16']['cuda_ms'])
    res['verdict'] = {
        'gpu_slower_at_one_seed': bool(res['cuda_speedup_at_one_seed'] < 1.0),
        'gpu_amortises_batching': bool(
            res['cuda_scaling_1024_over_16']
            < 0.5 * res['cpu_scaling_1024_over_16']),
        'sweep_device': 'cpu',
        'reason': ('per-epoch work is ~2 MFLOP, so kernel launch rather than '
                   'arithmetic sets the cost and the GPU is slower at one seed; '
                   'batching does amortise on the GPU, but 16 seeds batched only '
                   'matches 8-way CPU process parallelism, and buying that would '
                   'mean reimplementing the audited hypernetwork '
                   'parameterisation -- forfeiting the bit-exact reproduction of '
                   'paper_numbers.json that this whole task rests on'),
    }
    log('  verdict: the sweep runs on pinned CPU cores; CUDA is kept for the '
        'device cross-check above and for storage_rounds.py (T4)')
    return res


# ---------------------------------------------------------------------
# Self-test: the statistics machinery, on synthetic records
# ---------------------------------------------------------------------
def _selftest_stats(ck):
    """Sections 1-3: the primitives the whole aggregation rests on."""
    # --- 1. _stats against hand-computed values -------------------------
    s = _stats([1.0, 2.0, 3.0, 4.0])
    ck('_stats mean', abs(s['mean'] - 2.5) < 1e-15, s['mean'])
    ck('_stats std is the SAMPLE std (ddof=1)',
       abs(s['std'] - math.sqrt(5.0 / 3.0)) < 1e-15, s['std'])
    ck('_stats sem = std/sqrt(n)',
       abs(s['sem'] - math.sqrt(5.0 / 3.0) / 2.0) < 1e-15, s['sem'])
    ck('_stats ddof=1 genuinely differs from ddof=0',
       abs(s['std'] - 1.118033988749895) > 1e-9, s['std'])
    one = _stats([7.0])
    ck('_stats n=1 has zero std and zero sem',
       one['std'] == 0.0 and one['sem'] == 0.0 and one['mean'] == 7.0)

    # --- 2. _flatten_refine normalises BOTH production shapes -----------
    # `train_warm_best` republishes the three candidate scores as F_decoder /
    # F_grad / F_refined, while `refine_with_floor` (the ind path) leaves them in
    # a `scores` dict.  Both must normalise to the same canonical record or the
    # two protocols would not be comparable.
    warm_shape = {'picked': 'refined', 'F_decoder': 0.5, 'F_grad': 0.6,
                  'F_refined': 0.7, 'F_used': 0.7, 'F_cert': 0.8,
                  'F_bound': 0.9, 'headroom': 0.3, 'capture_frac': 0.66,
                  'headroom_vs_decoder': 0.3, 'headroom_is_zero': False,
                  'capture_vs_decoder': 0.66, 'n_improved': 5,
                  'unitarity_dev': 1e-6, 'cert_valid': True, 'prod_err': 1e-16}
    ind_shape = {'picked': 'refined',
                 'scores': {'decoder': 0.5, 'grad': 0.6, 'refined': 0.7},
                 'F_cert': 0.8, 'F_bound': 0.9, 'headroom': 0.3,
                 'capture_frac': 0.66, 'headroom_vs_decoder': 0.3,
                 'headroom_is_zero': False, 'capture_vs_decoder': 0.66,
                 'n_improved': 5, 'unitarity_dev': 1e-6, 'cert_valid': True,
                 'prod_err': 1e-16}
    fw, fi = _flatten_refine(warm_shape), _flatten_refine(ind_shape)
    ck('_flatten_refine: warm and ind shapes agree on every field', fw == fi,
       '\n     warm=%s\n     ind =%s' % (fw, fi))
    ck('_flatten_refine: F_used comes from scores[picked] on the ind shape',
       fi['F_used'] == 0.7, fi.get('F_used'))
    ck('_flatten_refine: booleans stay booleans',
       isinstance(fi['cert_valid'], bool)
       and isinstance(fi['headroom_is_zero'], bool))
    ck('_flatten_refine: n_improved stays an int',
       isinstance(fi['n_improved'], int))

    # --- 3. _select_prod replicates production's label-free rule --------
    a = {'seed': 1234, 'min_cf': 0.50, 'val_fid': 0.99}
    b = {'seed': 2024, 'min_cf': 0.60, 'val_fid': 0.90}
    c = {'seed': 1074, 'min_cf': 0.55, 'val_fid': 0.999}
    ck('_select_prod ranks by min_cf first, not by val_fid',
       _select_prod([a, b, c])['seed'] == 2024)
    ck('_select_prod breaks ties on val_fid',
       _select_prod([a, c])['seed'] == 1074)


def _synth_rec(seed, fu, hrd=0.3, fdec=0.5, fcert=0.99, noise='mixed',
               min_cf=0.5, hrd_zero=False):
    """A minimal synthetic per-seed record, shaped like `run_one`'s output.

    Defaults are chosen so the record is SELF-CONSISTENT: fcert=0.99 sits above
    every fu used below, and fdec=0.50 below it, so the decoder-floor and
    certified-ceiling invariants hold by default and a test that breaks one is
    breaking it deliberately.  min_cf is a parameter because production's
    label-free selection ranks on it, and leaving it identical across seeds makes
    the selection tie-break silently on val_fid instead.
    """
    return {'protocol': 'warm', 'noise': noise, 'seed': seed,
            'min_cf': min_cf, 'val_fid': 0.9, 'F_decoder': fdec,
            'F_grad': fu, 'F_refined': fu, 'F_used': fu, 'F_cert': fcert,
            'F_bound': 1.0, 'headroom': hrd, 'capture_frac': 0.5,
            'headroom_vs_decoder': hrd, 'headroom_is_zero': hrd_zero,
            'capture_vs_decoder': 0.5, 'n_improved': 4,
            'unitarity_dev': 1e-6, 'cert_valid': True, 'prod_err': 1e-16,
            'picked': 'refined', 'wall_s': 1.0,
            'F_exact': [fu] * len(P_VALUES),
            'p_values': [float(p) for p in P_VALUES]}


def _selftest_aggregation(ck):
    """Sections 4 and 6: grouping, and the seed pool the sweep draws from."""
    recs = [_synth_rec(1234, 0.90), _synth_rec(2024, 0.91),
            _synth_rec(1074, 0.92), _synth_rec(1111, 0.93)]
    agg = aggregate(recs)
    ck('aggregate: one group per (protocol, channel)',
       list(agg) == ['warm/mixed'], list(agg))
    g = agg['warm/mixed']
    ck('aggregate: n_seeds', g['n_seeds'] == 4, g['n_seeds'])
    ck('aggregate: mean(F_used)',
       abs(g['scalars']['F_used']['mean'] - 0.915) < 1e-12,
       g['scalars']['F_used']['mean'])
    ck('aggregate: std(F_used) is the sample std',
       abs(g['scalars']['F_used']['std'] - math.sqrt(5.0 / 3.0) / 100.0) < 1e-12,
       g['scalars']['F_used']['std'])
    ck('aggregate: curve mean/std are per-p lists of the right length',
       len(g['F_exact']['mean']) == len(P_VALUES)
       and len(g['F_exact']['std']) == len(P_VALUES))
    ck('aggregate: seeds sorted ascending', g['seeds'] == sorted(g['seeds']))
    two = aggregate(recs + [_synth_rec(1234, 0.95, noise='depolarizing')])
    ck('aggregate: a second channel forms a second group',
       sorted(two) == ['warm/depolarizing', 'warm/mixed'], sorted(two))
    # Regression lock: the depolarizing group above holds ONE seed.  A bare
    # np.std(ddof=1) there returns NaN and warns, which would poison the artifact.
    single = two['warm/depolarizing']
    ck('aggregate: a single-seed group reports std 0.0, never NaN',
       single['scalars']['F_used']['std'] == 0.0
       and single['F_exact']['std'][0] == 0.0
       and single['F_exact']['sem'][0] == 0.0
       and not math.isnan(single['scalars']['F_used']['std']),
       single['F_exact']['std'][:2])
    ck('aggregate: a single-seed group still reports the right mean and min/max',
       single['scalars']['F_used']['mean'] == 0.95
       and single['F_exact']['min'][0] == 0.95
       and single['F_exact']['max'][0] == 0.95)

    ck('SEED_POOL starts with the two production seeds',
       SEED_POOL[:2] == PROD_SEEDS, SEED_POOL[:4])
    ck('SEED_POOL has no duplicates', len(set(SEED_POOL)) == len(SEED_POOL))
    ck('SEED_POOL is long enough for --seeds up to 64', len(SEED_POOL) >= 64,
       len(SEED_POOL))
    ck('every channel has a production warm record to reproduce against',
       all(_production_infos(n)[0] for n in CHANNELS))
    ck('every channel has a production ind record to reproduce against',
       all(_production_infos(n)[1] for n in CHANNELS))


def _selftest_diagnostics(ck):
    """Section 5: the diagnostics, including the cases that could hide a bug."""
    # min_cf is DISTINCT here on purpose: production's label-free selection ranks
    # on it, and identical values would make the choice fall through to val_fid
    # and silently test the wrong rule.
    recs = [_synth_rec(1234, 0.90, min_cf=0.50), _synth_rec(2024, 0.91,
                                                            min_cf=0.60),
            _synth_rec(1074, 0.92, min_cf=0.55), _synth_rec(1111, 0.93,
                                                            min_cf=0.52)]
    g = aggregate(recs)['warm/mixed']
    d = diagnostics(aggregate(recs))['warm/mixed']
    ck('diagnostics: production seed is the min_cf winner among 1234/2024',
       d['production_seed'] == 2024, d['production_seed'])
    # Seed 2024 carries F_used=0.91 while the 4-seed mean is 0.915: the
    # label-free selection ranks on min_cf, NOT on F_used, so the bias of
    # "best of two" can be negative.  A diagnostic that could only ever report a
    # positive bias would be measuring the wrong thing.
    ck('diagnostics: selection bias can be NEGATIVE (selection is on min_cf)',
       d['selection_bias_abs'] < 0, d['selection_bias_abs'])
    ck('diagnostics: selection_bias_sigma carries the sign of the bias',
       d['selection_bias_sigma'] < 0, d['selection_bias_sigma'])
    ck('diagnostics: best-of-S exceeds the mean and names the top seed',
       d['max_over_seeds_abs'] > 0 and d['max_over_seeds_seed'] == 1111)
    ck('diagnostics: |best-of-two bias| <= |best-of-S bias|',
       abs(d['selection_bias_abs']) <= d['max_over_seeds_abs'] + 1e-15)
    ck('diagnostics: headroom is seed-independent when F_cert/F_decoder are',
       d['headroom_std'] <= 1e-15 and d['headroom_is_seed_independent'])
    ck('diagnostics: signal_to_seed_noise = headroom / std(F_used)',
       abs(d['signal_to_seed_noise']
           - 0.3 / g['scalars']['F_used']['std']) < 1e-9,
       d['signal_to_seed_noise'])
    ck('diagnostics: seed std is reported against the MC SEM reference',
       abs(d['seed_std_vs_mc_sem']
           - g['scalars']['F_used']['std'] / MC_SEM_REFERENCE) < 1e-12)
    ck('diagnostics: no seed below the decoder floor',
       d['n_below_decoder_floor'] == 0)
    ck('diagnostics: no seed above the certified ceiling',
       d['n_above_certified_ceiling'] == 0)
    # Degenerate case 1: every seed lands on the SAME optimum, so std is exactly
    # zero.  A sigma count would be 0/0 and must be suppressed, while a genuine
    # nonzero headroom over zero spread is correctly +inf.
    flat = diagnostics(aggregate([_synth_rec(1234, 0.9, min_cf=0.5),
                                 _synth_rec(2024, 0.9, min_cf=0.6)]))['warm/mixed']
    ck('diagnostics: zero seed spread gives std 0, not a ZeroDivisionError',
       flat['F_used_std'] == 0.0)
    ck('diagnostics: a nonzero headroom over zero spread is +inf',
       flat['signal_to_seed_noise'] == float('inf'))
    ck('diagnostics: sigma counts are SUPPRESSED when std sits at machine '
       'precision',
       flat['std_at_machine_precision'] is True
       and flat['selection_bias_sigma'] is None
       and flat['max_over_seeds_sigma'] is None)
    # Degenerate case 2, and the one the real sweep actually hits on depolarizing
    # and mixed: the decoder IS the family optimum, so the certified headroom is
    # signed round-off (~-7e-14) flagged headroom_is_zero.  Dividing that by a
    # machine-epsilon std yields a large but meaningless number -- it must come
    # back as None, never as "-149 sigma" in a table.
    zero = [_synth_rec(1234, 0.9, hrd=-6.8e-14, hrd_zero=True, min_cf=0.5),
            _synth_rec(2024, 0.9, hrd=-6.8e-14, hrd_zero=True, min_cf=0.6)]
    dz = diagnostics(aggregate(zero))['warm/mixed']
    ck('diagnostics: zero certified headroom -> SNR is None, not a fake ratio',
       dz['headroom_is_zero'] is True and dz['signal_to_seed_noise'] is None)
    # ... and the non-degenerate group must still report real numbers.
    ck('diagnostics: sigma counts are present when std is resolvable',
       d['std_at_machine_precision'] is False
       and d['selection_bias_sigma'] is not None
       and d['signal_to_seed_noise'] is not None)
    # A headroom that DID move with the seed must be flagged, not averaged away.
    vary = [_synth_rec(1234, 0.90, hrd=0.30), _synth_rec(2024, 0.91, hrd=0.31),
            _synth_rec(1074, 0.92, hrd=0.32), _synth_rec(1111, 0.93, hrd=0.33)]
    dv = diagnostics(aggregate(vary))['warm/mixed']
    ck('diagnostics: a seed-dependent headroom is FLAGGED',
       dv['headroom_is_seed_independent'] is False and dv['headroom_std'] > 1e-15)
    # A floor-violating seed must be counted, not hidden inside the mean.
    bad = [_synth_rec(1234, 0.90), _synth_rec(2024, 0.49),
           _synth_rec(1074, 0.92), _synth_rec(1111, 0.93)]
    db = diagnostics(aggregate(bad))['warm/mixed']
    ck('diagnostics: a floor-violating seed is COUNTED',
       db['n_below_decoder_floor'] == 1, db['n_below_decoder_floor'])
    # ... and a seed above the certified ceiling likewise.
    over = [_synth_rec(1234, 0.90), _synth_rec(2024, 0.81, fcert=0.80),
            _synth_rec(1074, 0.92), _synth_rec(1111, 0.93)]
    do = diagnostics(aggregate(over))['warm/mixed']
    ck('diagnostics: a ceiling-violating seed is COUNTED',
       do['n_above_certified_ceiling'] == 1, do['n_above_certified_ceiling'])


def selftest(log=print):
    """Assert the aggregation/diagnostic layer is correct without training.

    These are the parts that could silently produce a wrong mean, a wrong std, or
    a bias diagnostic with the wrong sign, so each is checked against a
    hand-computed value.  The physics is covered by --verify-reproduction, which
    is a far stronger check and needs no synthetic data.
    """
    fails = []
    n = [0]

    def ck(name, cond, detail=''):
        n[0] += 1
        if cond:
            log('  [ok]   %s' % name)
        else:
            log('  [FAIL] %s %s' % (name, detail))
            fails.append(name)

    _selftest_stats(ck)
    _selftest_aggregation(ck)
    _selftest_diagnostics(ck)
    log('[selftest] %d checks, %d failures' % (n[0], len(fails)))
    return fails


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------
def _summary(agg, diag, log=print):
    log('\n%-24s %4s %14s %11s %11s %10s %11s'
        % ('protocol/channel', 'S', 'mean F_used', 'std', 'headroom',
           'bias(sig)', 'SNR'))
    log('-' * 94)
    for key in sorted(agg):
        d = diag.get(key)
        if d is None:
            continue
        snr = d.get('signal_to_seed_noise')
        log('%-24s %4d %.12f %11.3e %11.3e %10s %11s'
            % (key, agg[key]['n_seeds'], d['F_used_mean'], d['F_used_std'],
               d.get('headroom_mean', float('nan')),
               ('n/a*' if d['selection_bias_sigma'] is None
                else '%.2f' % d['selection_bias_sigma']),
               ('n/a*' if snr is None
                else ('inf' if snr == float('inf') else '%.3g' % snr))))
    log('-' * 94)
    log('bias(sig) = (best-of-two F_used - S-seed mean) / std.  Near zero means')
    log('            the reported number is not a lucky draw.')
    log('SNR       = certified headroom / std(F_used): the effect size measured')
    log('            in units of the seed lottery.')
    log('n/a*      = undefined, NOT zero.  On channels where the decoder is')
    log('            already the family optimum the certified headroom vanishes')
    log('            and every seed converges to the same table, so both ratios')
    log('            are 0/0 at machine precision; std_at_machine_precision marks')
    log('            them and headroom_is_zero explains why.\n')


def _write_artifact(path, payload):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(payload, fh, indent=1, sort_keys=False)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='T3: multi-seed mean +/- std for the trained VSCR pipeline')
    ap.add_argument('--seeds', type=int, default=None,
                    help='number of seeds (default 16; 4 with --quick; 2 with '
                         '--verify-reproduction).  1234 and 2024 are always in.')
    ap.add_argument('--channels', default=','.join(CHANNELS))
    ap.add_argument('--protocols', default=','.join(PROTOCOLS))
    ap.add_argument('--cpus', default=','.join(str(c) for c in DEFAULT_CPUS),
                    help='CPU ids to pin workers to (default 0-7, the P-cores)')
    ap.add_argument('--timeout', type=float, default=1800.0,
                    help='per-job wall-clock budget before kill-and-retry')
    ap.add_argument('--quick', action='store_true',
                    help='smoke budget: 4 seeds, amplitude damping, warm only')
    ap.add_argument('--gpu-check', action='store_true',
                    help='run the CUDA device cross-check and benchmark, exit')
    ap.add_argument('--verify-reproduction', action='store_true',
                    help='require the production seeds to come back bit-exact')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--reaggregate', metavar='ARTIFACT',
                    help='recompute aggregate/diagnostics/reproduction from the '
                         'per-seed records already stored in ARTIFACT, without '
                         're-running the sweep.  Use this when only the reporting '
                         'layer changed: a full sweep costs ~22 min and its '
                         'records are deterministic, so re-training to fix a '
                         'summary column would be waste.')
    ap.add_argument('--out', default=OUT_JSON)
    ap.add_argument('--child', nargs=2, metavar=('SPEC', 'OUT'),
                    help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.child:
        return _child(*args.child)
    if args.selftest:
        return 1 if selftest() else 0
    if args.gpu_check:
        res = gpu_check()
        print(json.dumps({k: v for k, v in res.items() if k != 'batched_step'},
                         indent=1))
        # Persist the measurement into the artifact when there is one: the device
        # decision is part of the record, not a side conversation, so a reader of
        # multiseed_results.json can see WHY the sweep ran on pinned CPU cores.
        if os.path.exists(args.out):
            with open(args.out) as fh:
                payload = json.load(fh)
            payload['gpu_check'] = res
            _write_artifact(args.out, payload)
            print('stored gpu_check into %s' % args.out)
        return 0
    if args.reaggregate:
        with open(args.reaggregate) as fh:
            old = json.load(fh)
        records = old['records']
        seeds = sorted({r['seed'] for r in records})
        channels = tuple(dict.fromkeys(r['noise'] for r in records))
        protocols = tuple(dict.fromkeys(r['protocol'] for r in records))
        print('=== re-aggregating %d stored records from %s ==='
              % (len(records), args.reaggregate), flush=True)
        print('  %d seeds, channels: %s, protocols: %s'
              % (len(seeds), ', '.join(channels), ', '.join(protocols)),
              flush=True)
        # Re-aggregation is exactly when the reproduction lock matters most: it
        # proves the reporting layer still describes the stored numbers.
        args.verify_reproduction = True
        return _run(args, None, seeds, channels, protocols, (), records=records)

    channels = tuple(c.strip() for c in args.channels.split(',') if c.strip())
    protocols = tuple(p.strip() for p in args.protocols.split(',') if p.strip())
    cpus = tuple(int(c) for c in args.cpus.split(',') if c.strip() != '')
    for c in channels:
        if c not in CHANNELS:
            ap.error('unknown channel %r' % c)
    for p in protocols:
        if p not in PROTOCOLS:
            ap.error('unknown protocol %r' % p)
    if not cpus:
        ap.error('--cpus resolved to an empty list')

    n = args.seeds
    if n is None:
        n = 4 if args.quick else (2 if args.verify_reproduction else 16)
    if args.quick and args.seeds is None:
        channels, protocols = ('amplitude_damping',), ('warm',)
    n = max(n, len(PROD_SEEDS))
    seeds = (list(PROD_SEEDS)
             + [s for s in SEED_POOL if s not in PROD_SEEDS])[:n]

    jobs = [{'protocol': p, 'noise': c, 'seed': s, 'p_values': list(P_VALUES)}
            for p in protocols for c in channels for s in seeds]
    print('=== T3 multi-seed sweep ===', flush=True)
    print('  %d protocol(s) x %d channel(s) x %d seeds = %d jobs on %d pinned '
          'CPU(s)' % (len(protocols), len(channels), len(seeds), len(jobs),
                      len(cpus)), flush=True)
    print('  channels : %s' % ', '.join(channels), flush=True)
    print('  protocols: %s' % ', '.join(protocols), flush=True)
    print('  seeds    : %s%s' % (seeds[:8], ' ...' if len(seeds) > 8 else ''),
          flush=True)
    print('  est. wall: %.1f min at ~54 s/job'
          % (len(jobs) * 54.0 / len(cpus) / 60.0), flush=True)
    return _run(args, jobs, seeds, channels, protocols, cpus)


def _preserved_gpu_check(out_path):
    """The gpu_check block already stored at `out_path`, if any.

    `--gpu-check` costs ~60 s (two full 1700-epoch curricula, one per device) and
    its verdict is what justifies running the sweep on CPU.  Rewriting the artifact
    from a `--reaggregate` used to drop it silently, which the audit then caught as
    a missing measurement.  Carrying it forward makes an expensive, still-valid
    measurement sticky instead of something a reporting-only rerun destroys.
    """
    try:
        with open(out_path) as fh:
            return json.load(fh).get('gpu_check')
    except (OSError, ValueError):
        return None


def _run(args, jobs, seeds, channels, protocols, cpus, records=None):
    """Execute the sweep (or accept stored records), aggregate, verify, write."""
    if records is None:
        t0 = time.time()
        records = _run_sweep(jobs, cpus, args.timeout)
        wall = time.time() - t0
        # 54 s is the measured serial cost of one job (13 s gradient curriculum +
        # 41 s separable refinement), so this is the achieved speedup over serial.
        speedup = 54.0 * len(jobs) / wall if wall > 0 else float('inf')
        print('  sweep done: %d records in %.1fs (%.1f s/job effective, '
              '%.2fx serial on %d pinned core(s))'
              % (len(records), wall, wall / max(1, len(jobs)), speedup,
                 len(cpus)), flush=True)
    else:
        wall = 0.0
    n_jobs = len(jobs) if jobs is not None else len(records)

    agg = aggregate(records)
    diag = diagnostics(agg)
    repro = None
    if args.verify_reproduction:
        scope = {(r['protocol'], r['noise']) for r in records}
        repro = verify_reproduction(
            {(r['protocol'], r['noise'], r['seed']): r for r in records},
            scope=scope)
        print('\n=== reproduction lock ===', flush=True)
        print('  %d configs in scope, %d fields bit-compared against '
              'paper_numbers.json, %d failures, bit_exact=%s'
              % (len(scope), repro['n_fields_compared'], repro['n_failures'],
                 repro['bit_exact']), flush=True)
        if repro['skipped']:
            print('  skipped (outside this run): %s'
                  % ', '.join(repro['skipped']), flush=True)
        for f in repro['failures']:
            print('    FAIL %s' % f, flush=True)
    _summary(agg, diag)

    payload = {
        'meta': {
            'task': 'T3 multi-seed statistics (mean +/- std over training seeds)',
            'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'host': os.uname().nodename,
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'wall_s': wall,
            'n_jobs': n_jobs,
            'reaggregated_from': args.reaggregate,
            'sweep_device': 'cpu -- see gpu_check.verdict for the measurement',
            'estimator': ('every per-seed fidelity is a deterministic quadrature '
                          '(abl.exact_F / pavg_F / exact conditional fidelity), '
                          'so the spread recorded here is seed variability with '
                          'no Monte-Carlo estimator noise folded in'),
        },
        'config': {'seeds': seeds, 'channels': list(channels),
                   'protocols': list(protocols), 'cpus': list(cpus),
                   'p_values': list(P_VALUES), 'timeout_s': args.timeout,
                   'prod_seeds': list(PROD_SEEDS),
                   'mc_sem_reference': MC_SEM_REFERENCE},
        'records': records,
        'aggregate': agg,
        'diagnostics': diag,
        'reproduction': repro,
        # Read BEFORE the write below replaces the file, so a reporting-only rerun
        # cannot destroy the device measurement (see _preserved_gpu_check).
        'gpu_check': _preserved_gpu_check(args.out),
    }
    _write_artifact(args.out, payload)
    print('wrote %s (%.1f KB)' % (args.out,
                                   os.path.getsize(args.out) / 1024.0))

    bad = []
    if repro is not None and not repro['bit_exact']:
        bad.append('reproduction is NOT bit-exact (%d field mismatches)'
                   % repro['n_failures'])
    for key in sorted(diag):
        d = diag[key]
        if d.get('n_below_decoder_floor'):
            bad.append('%s: %d seed(s) below the decoder floor'
                       % (key, d['n_below_decoder_floor']))
        if d.get('n_above_certified_ceiling'):
            bad.append('%s: %d seed(s) above the certified ceiling'
                       % (key, d['n_above_certified_ceiling']))
        if d.get('headroom_is_seed_independent') is False:
            bad.append('%s: certified headroom varies with the seed' % key)
        if d.get('all_seeds_pick_same_candidate') is False:
            # Not fatal, but it means the refine stage's winner is seed-dependent
            # for this channel, which changes what "F_used" averages over.
            print('  NOTE %s: refine picked different candidates across seeds %s'
                  % (key, d['picked_counts']))
    if bad:
        print('\nINVARIANT FAILURES:')
        for b in bad:
            print('  - %s' % b)
        return 1
    print('\nall invariants hold across every seed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
