"""
vscr_paper.py — Paper-grade VSCR benchmark: decoder-warm-started variational
syndrome-conditioned recovery for the [[5,1,3]] code.

Improvements over the prototype run (full_run.log):
  1. WARM START: each syndrome branch R_s is initialised at the exact Pauli
     decoder correction C_s (hypernetwork output layer zero-initialised), so
     training starts AT perfect-decoder performance and can only refine it
     with continuous, non-Pauli corrections.  Verified |F_warm(0)-F_dec|<1e-12.
  2. PHYSICAL LinDR: the vnCDR-style linear regression baseline is projected
     onto the nearest physical (PSD, trace-1) state before scoring, removing
     the unphysical F > 1 artefacts of the prototype.
  3. ERROR BARS: evaluation over 200 Haar-random logical states per point.
  4. Cold-start VSCR (v1 hardware snapshot, vscr_angles_dep.npz) included as
     an ablation on the depolarizing channel.

Outputs:
  vscr_paper_results.npz, vscr_angles_paper_{dep,ad,mixed}.npz,
  paper_numbers.json, paper/figures/*.pdf|png

NOTE: vscr_angles_dep.npz (v1, used for the WK_C180 hardware feasibility
runs) is NEVER overwritten.
"""
import json, math, os, subprocess, sys, tempfile, time

# Pin the native thread pools BEFORE numpy/torch are imported -- both read these
# variables once, at load time, so setting them later has no effect on OpenBLAS.
# See the note on `ssvr_qec._pin_blas_threads` for why this matters on this host.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'OPENBLAS_MAIN_FREE'):
    os.environ.setdefault(_v, '1')
# See the long note in ssvr_qec.py: this host intermittently SIGSEGV/SIGILLs in
# tiny complex128 kernels when a process is free to MIGRATE across its hybrid
# P-/E-core cluster. Importing ssvr_qec pins us to one CPU, which fixes it
# (measured: unpinned core-dumps within ~2e4 calls; pinned runs 4e5 clean).
# OPENBLAS_CORETYPE is for determinism only -- it is NOT the fix, since the
# faults reproduce under every coretype including AVX-only SANDYBRIDGE.
os.environ.setdefault('OPENBLAS_CORETYPE', 'HASWELL')

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ssvr_qec as m

os.makedirs('paper/figures', exist_ok=True)
FIGDIR = 'paper/figures'
RESULTS_NPZ = 'vscr_paper_results.npz'
SEEDS = (1234, 2024)
P_VALUES = np.array([0.005, 0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30])
N_TEST = 200
SMOKE = os.environ.get('ABL_SMOKE', '0') == '1'   # tiny budgets for testing

# Budget for the per-syndrome refinement stage of `train_warm_best`.  The
# p-averaged objective is EXACTLY separable across syndromes, so `steps` is per
# BLOCK (there are 16 of them), not per epoch.
REFINE_N_P = 6          # Gauss-Legendre nodes over the curriculum p-range
REFINE_N_START = 3      # start 0 is always the incumbent table (monotonicity)
REFINE_STEPS = 1500
REFINE_LR = 0.01
if SMOKE:
    P_VALUES = np.array([0.05, 0.10, 0.15])
    N_TEST = 40
    # 250 rather than 60: below ~200 Adam steps the 16 blocks are visibly
    # unconverged and the refined table drifts out of the code-preserving
    # regime, which makes the reported capture fraction meaningless even though
    # the monotonicity guarantee still holds.
    REFINE_N_START, REFINE_STEPS = 2, 250

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'font.family': 'sans-serif', 'axes.linewidth': 0.8,
    'figure.dpi': 150, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

# ---------------------------------------------------------------------
# Decoder warm-start angles
# ---------------------------------------------------------------------
def decoder_angles():
    """(16, PHI_DIM) angles realising R_s = C_s (Pauli decoder) up to phase.
    Layer-1 slots per qubit q: [Rz(3q), Rx(3q+1), Rz(3q+2)]; X -> Rx(pi),
    Z -> Rz(pi), Y -> Rz(pi)Rx(pi) (phase-irrelevant)."""
    phi = np.zeros((16, m.PHI_DIM))
    for s, bits in enumerate(m.SYND_BITS):
        corr = m.SYND_TABLE[bits]
        for q, ch in enumerate(corr):
            if ch == 'X':
                phi[s, 3 * q + 1] = np.pi
            elif ch == 'Z':
                phi[s, 3 * q] = np.pi
            elif ch == 'Y':
                phi[s, 3 * q] = np.pi
                phi[s, 3 * q + 1] = np.pi
    return torch.tensor(phi, dtype=torch.float64)


PHI_DEC = decoder_angles()


def _verify_warm_start():
    R = m.recovery_unitary_batch(PHI_DEC)
    C = torch.stack(m.C_SYNDS).to(torch.complex128)
    for s in range(16):
        nz = C[s].abs() > 0.5
        ph = R[s][nz] / C[s][nz]
        assert torch.allclose(ph, ph[0].expand_as(ph), atol=1e-12), s
    rng = np.random.RandomState(0)
    for noise in ('depolarizing', 'amplitude_damping', 'mixed'):
        for p in (0.05, 0.1, 0.3):
            psi = m.encode(m.random_logical_state(1, rng))
            rho = m.noisy_state(psi, p, noise)
            Fd = float(m.perfect_code_decoder_fidelity(psi, rho))
            Fw = float(m.vscr_fidelity(psi, rho, R, lam=0.0)[1])
            assert abs(Fd - Fw) < 1e-12, (noise, p, Fd, Fw)
    print('[warm-start] R_s == C_s (phase-exact); fidelity == decoder: VERIFIED',
          flush=True)



class VSCRWarm(nn.Module):
    """VSCR with decoder warm start: phi_s = PHI_DEC[s] + phi_base + hnet(emb_s)
    + phi_extra[s].

    The hypernetwork OUTPUT layer is zero-initialised so the model starts
    exactly at the Pauli decoder; training learns continuous refinements.

    INITIALISATION BUG FIX (this used to be a dead hypernetwork).  The previous
    version zero-initialised BOTH `hnet[0]` and `hnet[2]`.  With
    `hnet[0].weight = 0` and `hnet[0].bias = 0` the hidden activation is
    `tanh(0) = 0` for every syndrome, so the gradient of the loss w.r.t.
    `hnet[2].weight` -- which is proportional to that activation -- is exactly
    zero as well.  Both layers therefore have identically zero gradient at the
    warm start and stay zero forever: the only trainable part of the syndrome
    pathway was the single global `phi_base` vector, i.e. the model could apply
    at most ONE global rotation shared by all 16 syndromes.  That silently
    invalidated both the "syndrome-conditioned recovery" claim and the
    K-dependence of the low-data ablation.

    The fix keeps the exact-warm-start property (zero `hnet[2]` still makes
    `forward() == PHI_DEC` at step 0) while giving `hnet[0]` a non-degenerate
    `std = 1/sqrt(ctx)` init, so `tanh(hnet[0] e_s)` is syndrome-dependent and
    non-zero and `hnet[2]` receives a real gradient from step 1.  `synd_emb` is
    drawn from a PRIVATE generator (seed 20240607) so the embedding is
    reproducible independently of the caller's global torch seed -- `train_warm_best`
    reseeds the global stream per seed and must not perturb the embedding.

    `_selftest_hypernet_trainable` asserts the fix stays fixed."""

    def __init__(self, n_synd=16, phi_dim=m.PHI_DIM, hidden=64, ctx=24,
                 emb_scale=0.5):
        super().__init__()
        # clone: register_buffer stores a REFERENCE; zeroing this buffer in an
        # ablation must never mutate the global PHI_DEC.
        self.register_buffer('phi_dec', PHI_DEC.clone())
        g = torch.Generator().manual_seed(20240607)
        self.synd_emb = nn.Parameter(
            torch.randn(n_synd, ctx, dtype=torch.float64, generator=g) * emb_scale)
        self.hnet = nn.Sequential(
            nn.Linear(ctx, hidden, dtype=torch.float64), nn.Tanh(),
            nn.Linear(hidden, phi_dim, dtype=torch.float64))
        # hidden layer: NON-DEGENERATE (see class docstring); output layer: zero
        nn.init.normal_(self.hnet[0].weight, std=1.0 / math.sqrt(ctx))
        nn.init.zeros_(self.hnet[0].bias)
        nn.init.zeros_(self.hnet[2].weight); nn.init.zeros_(self.hnet[2].bias)
        self.phi_base = nn.Parameter(torch.zeros(phi_dim, dtype=torch.float64))
        # residual buffer: lets `set_phi` install an arbitrary refined table
        # that need not lie in the image of `hnet` (see `set_phi`).
        self.register_buffer(
            'phi_extra', torch.zeros(n_synd, phi_dim, dtype=torch.float64))

    def forward(self):
        return (self.phi_dec + self.phi_base + self.hnet(self.synd_emb)
                + self.phi_extra)

    @torch.no_grad()
    def syndrome_spread(self):
        """max_{s,s'} ||hnet(e_s) - hnet(e_s')||_inf : the size of the learned
        SYNDROME-DEPENDENT deviation from the decoder.  Exactly 0 for a dead
        hypernetwork (the old double-zero-init bug), O(1e-3..1e-1) when healthy.
        NOTE: this deliberately excludes `phi_dec`, which already differs across
        branches, and `phi_base`, which is syndrome-independent by construction."""
        h = self.hnet(self.synd_emb) + self.phi_extra
        return float((h.unsqueeze(0) - h.unsqueeze(1)).abs().max())

    @torch.no_grad()
    def branch_deviation(self):
        """max_s ||phi_s - phi_dec[s]||_inf : total learned correction size."""
        d = self.forward() - self.phi_dec
        return float(d.abs().max())

    @torch.no_grad()
    def set_phi(self, phi_table):
        """Install an explicit per-syndrome angle table via `phi_extra`.

        The refined table produced by `refine_per_syndrome` is a free
        (16, PHI_DIM) array, which need not lie in the image of `hnet`; storing
        the difference in the residual buffer makes `forward()` return it
        exactly, so `evaluate_paper`, `cf_table`, `hardware_frame_numbers` and
        the plotting code all see the refined recovery unchanged."""
        t = torch.as_tensor(phi_table, dtype=torch.float64).reshape(
            tuple(self.phi_extra.shape))
        base = self.phi_dec + self.phi_base + self.hnet(self.synd_emb)
        self.phi_extra.copy_(t - base)
        err = float((self.forward() - t).abs().max())
        assert err < 1e-12, f'set_phi round-trip error {err:.3e}'
        return self


def _selftest_hypernet_trainable(noise='amplitude_damping', steps=60, batch=12):
    """Guard against re-introducing the dead-hypernetwork bug.

    Asserts (a) the warm start is exactly the decoder, (b) EVERY trainable
    parameter of the syndrome pathway moves under the real label-free loss, and
    (c) the 16 branch angle sets become mutually distinct.  Without (b)/(c) the
    model can only apply one global rotation, and both the "syndrome-conditioned
    recovery" claim and the K-dependence of the low-data ablation are vacuous.

    The bug this catches: zero-initialising BOTH `hnet[0]` and `hnet[2]` makes
    tanh(hnet[0] e_s) == 0 for every syndrome, so the gradient of `hnet[2]`
    (proportional to that activation) is also identically zero.  Neither layer
    can ever move, `syndrome_spread()` stays exactly 0, and the model collapses
    to a single global rotation on top of `phi_base`."""
    torch.manual_seed(1234); np.random.seed(1234)
    model = VSCRWarm()
    # (a) exact decoder warm start
    warm_err = float((model() - PHI_DEC).abs().max())
    assert warm_err == 0.0, f'warm start is not exactly the decoder: {warm_err:.3e}'
    assert model.syndrome_spread() == 0.0
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    m.train_vscr(model, n_epochs=steps, batch=batch, p_range=(0.02, 0.15),
                 noise=noise, lr=3e-3, lam=0.0, rng_seed=1234,
                 use_real_grad=True, verbose=False, quad=True)
    # (b) every trainable parameter of the syndrome pathway moved
    moved, dead = {}, []
    for n, p in model.named_parameters():
        d = float((p.detach() - before[n]).abs().max())
        moved[n] = d
        if d <= 0.0:
            dead.append(n)
    assert not dead, f'parameters that never moved (dead pathway): {dead}'
    # (c) the 16 branch angle sets became mutually distinct
    spread = model.syndrome_spread()
    assert spread > 0.0, 'syndrome_spread == 0: the hypernetwork is still dead'
    phi = model().detach().numpy()
    distinct = min(float(np.abs(phi[i] - phi[j]).max())
                   for i in range(16) for j in range(i + 1, 16))
    assert distinct > 0.0, 'branch angle sets are not mutually distinct'
    print(f'[hypernet selftest] warm start == decoder exactly; after {steps} '
          f'quad epochs on {noise}: every parameter moved '
          f'(min |dphi|_max = {min(moved.values()):.2e}), syndrome_spread = '
          f'{spread:.3e}, closest pair of branch angle sets differs by '
          f'{distinct:.3e}  VERIFIED', flush=True)
    for n in sorted(moved):
        print(f'    moved {n:18s} |d|max = {moved[n]:.3e}', flush=True)
    return moved


# Warm-start curriculum (gentler lr than cold start: the model begins exactly
# at the Pauli decoder and only needs continuous refinement).
WARM_SCHEDULES = {
    'depolarizing': [(400, 3e-3, (0.02, 0.08)),
                     (500, 3e-3, (0.02, 0.15)),
                     (800, 3e-4, (0.02, 0.15))],
    'amplitude_damping': [(400, 3e-3, (0.01, 0.08)),
                          (500, 3e-3, (0.01, 0.15)),
                          (800, 3e-4, (0.01, 0.15))],
    'mixed': [(400, 3e-3, (0.02, 0.08)),
              (500, 3e-3, (0.02, 0.12)),
              (800, 3e-4, (0.02, 0.15))],
}
if SMOKE:
    # one short stage per channel; keeps the curriculum's first p-range so the
    # refinement stage still has a meaningful p-interval to average over
    WARM_SCHEDULES = {k: [(30, 3e-3, v[0][2])] for k, v in WARM_SCHEDULES.items()}


# ---------------------------------------------------------------------
# Running the refinement stage in a fresh interpreter.
#
# This is a workaround for a DEFECT IN THE NATIVE STACK ON THIS HOST, not for
# anything in the algorithm, and it does not change a single reported number.
#
# Symptom: sustained small complex128 work dies with SIGSEGV or SIGILL, and the
# crash site MOVES between runs -- observed inside
# `torch.autograd.graph._engine_run_backward`, inside scipy's L-BFGS-B
# finite-difference Jacobian (`approx_derivative` -> `numpy.any`), inside an 8x8
# matmul in `vscr_paper_abl.c_tr`, and most often inside the plain numpy
# `_block_obj_grad`. A moving crash site across otherwise identical runs, in
# operations far too small to corrupt memory from a logic error, is the signature
# of a native-level fault, not of a bug in this code.
#
# TWO EARLIER HYPOTHESES WRITTEN HERE WERE WRONG, and are recorded so they are
# not re-adopted (re-measure with `diag_native_fault.py`):
#   * "only after a long torch-autograd session" -- false. It reproduces in pure
#     numpy with no autograd anywhere in the call stack.
#   * "torch's bundled libgomp vs scipy-openblas `omp_*` symbol resolution" --
#     false. It reproduces with every thread pool pinned to 1 and no OpenMP work
#     in flight, and under every OPENBLAS_CORETYPE including AVX-only
#     SANDYBRIDGE -- which cannot raise SIGILL on a CPU that implements AVX, and
#     this Arrow Lake part has no AVX-512 at all, so there is no unsupported
#     kernel for DYNAMIC_ARCH to select either.
#
# What the measurements DO implicate is CPU affinity: every arm left free to
# migrate across this hybrid P-/E-core cluster faulted, while arms pinned to a
# single core completed 4e5 calls clean, repeatedly. `ssvr_qec` now pins every
# process on import (`_pin_single_cpu`, opt out with QEC_PIN_CPU=0), and that is
# the actual fix -- with it the 9-test suite and both pipelines run clean with no
# retry consumed.
#
# This subprocess isolation is nevertheless KEPT as defence in depth: it is what
# let the pipeline complete on runs where the fault still fired (one suite run
# absorbed a SIGSEGV and then a SIGILL and passed on attempt 3).
#
# So the one stage that does sustained GEMM work is run in a CHILD process with
# a clean native heap, and retried.  `refine_per_syndrome` is seeded, closed-form
# and deterministic, so parent and child agree bit-for-bit and the retry cannot
# change the result -- it can only avoid the fault.  Set
# VSCR_REFINE_INPROCESS=1 to run it in-process when debugging.
# ---------------------------------------------------------------------
_REFINE_BOOTSTRAP = (
    "import sys;"
    "sys.path.insert(0, sys.argv[1]);"
    "import numpy as np;"
    "import vscr_paper_abl as a;"
    "d = np.load(sys.argv[2]);"
    "phi, info = a.refine_per_syndrome("
    "d['phi'], str(d['noise']),"
    "p_range=tuple(float(x) for x in d['p_range']),"
    "n_p=int(d['n_p']), n_start=int(d['n_start']), steps=int(d['steps']),"
    "lr=float(d['lr']), seed=int(d['seed']),"
    "verbose=bool(int(d['verbose'])));"
    "info.pop('phi', None);"
    "np.savez(sys.argv[3], phi=phi,"
    "**{k: np.asarray(v) for k, v in info.items()})"
)


def _refine_isolated(phi_init, noise, p_range, n_p, n_start, steps, lr, seed,
                     verbose=False, retries=5, timeout=7200):
    """`vscr_paper_abl.refine_per_syndrome` executed in a fresh interpreter.

    Returns the same ``(phi, info)`` pair.  See the comment block above for why
    this indirection exists; the computation itself is unchanged and
    deterministic, so the child's answer is bit-identical to the parent's."""
    if os.environ.get('VSCR_REFINE_INPROCESS', '0') == '1':
        import vscr_paper_abl as abl
        return abl.refine_per_syndrome(
            phi_init, noise, p_range=p_range, n_p=n_p, n_start=n_start,
            steps=steps, lr=lr, seed=seed, verbose=verbose)
    root = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ)
    env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'OPENBLAS_MAIN_FREE'):
        env.setdefault(v, '1')
    env.setdefault('CUDA_VISIBLE_DEVICES', '')
    last = None
    for attempt in range(1, retries + 1):
        with tempfile.TemporaryDirectory(prefix='vscr_refine_') as td:
            src = os.path.join(td, 'in.npz')
            dst = os.path.join(td, 'out.npz')
            np.savez(src, phi=np.asarray(phi_init, dtype=float), noise=noise,
                     p_range=np.asarray(p_range, dtype=float), n_p=n_p,
                     n_start=n_start, steps=steps, lr=lr, seed=seed,
                     verbose=int(bool(verbose)))
            try:
                r = subprocess.run(
                    [sys.executable, '-X', 'faulthandler', '-c',
                     _REFINE_BOOTSTRAP, root, src, dst],
                    capture_output=True, text=True, timeout=timeout, env=env,
                    cwd=root)
            except subprocess.TimeoutExpired as exc:
                last = f'timeout after {timeout}s'
                r = None
            if r is not None and r.returncode == 0 and os.path.exists(dst):
                out = np.load(dst)
                info = {k: (out[k].item() if out[k].ndim == 0 else out[k])
                        for k in out.files if k != 'phi'}
                info['phi'] = out['phi']
                info['isolated_attempts'] = attempt
                if verbose and r.stdout.strip():
                    print(r.stdout.rstrip(), flush=True)
                return info['phi'], info
            last = (f'exit={getattr(r, "returncode", None)} '
                    f'{(getattr(r, "stderr", "") or "")[-400:]}') or last
            if r is not None and r.returncode != 0:
                # keep the FULL child output: the faulthandler traceback at the
                # top is what identifies the faulting native frame, and the
                # trailing "Extension modules" list alone is useless.
                dump = os.path.join(
                    tempfile.gettempdir(),
                    f'qec_refine_child_{noise}_attempt{attempt}.log')
                try:
                    with open(dump, 'w') as fh:
                        fh.write(f'--- child stdout ---\n{r.stdout}\n'
                                 f'--- child stderr ---\n{r.stderr}\n')
                    print(f'    [{noise}] child output saved to {dump}',
                          flush=True)
                except OSError:
                    pass
        print(f'    [{noise}] refinement attempt {attempt}/{retries} failed in '
              f'the child process ({last}); retrying with a clean interpreter',
              flush=True)
    raise RuntimeError(
        f'refine_per_syndrome failed {retries} times for {noise}: {last}')


def refine_with_floor(phi_grad, noise, p_range, n_p=None, n_start=None, steps=None,
                      lr=None, seed=0, verbose=False, tag=''):
    """FIX A + FIX B as one reusable stage.

    (B) `vscr_paper_abl.refine_per_syndrome` escapes the decoder saddle by
        re-optimising each of the 16 blocks independently -- the p-averaged
        objective is exactly separable across syndromes, so block-wise
        optimisation reaches the global optimum of the per-syndrome-unitary
        family.
    (A) The result is FLOORED: among {decoder, input, refined} we keep whichever
        maximises the exact production objective, so both
        ``F_out >= F_dec`` and ``F_out >= F_input`` hold BY CONSTRUCTION.

    Returns ``(phi, info)``; `info` carries the three candidate scores, which one
    was picked, and every diagnostic `refine_per_syndrome` reports.

    This is a shared stage rather than inline code in `train_warm_best` so that
    the VQR-ind ablation in `vscr_paper_abl` can run the identical
    post-processing.  Without that, the ablation would compare a refined
    hypernetwork model against an unrefined table and attribute the refinement's
    gain to the architecture -- the two would differ by 9.9e-3 on amplitude
    damping and 3.1e-3 on coherent noise purely because of post-processing."""
    import vscr_paper_abl as abl              # lazy: abl imports this module
    n_p = REFINE_N_P if n_p is None else n_p
    n_start = REFINE_N_START if n_start is None else n_start
    steps = REFINE_STEPS if steps is None else steps
    lr = REFINE_LR if lr is None else lr
    phi_grad = np.asarray(phi_grad, dtype=float)
    p_range = tuple(float(x) for x in p_range)
    B = abl.pavg_bundle(noise, p_range, n_p)
    phi_ref, ref = _refine_isolated(phi_grad, noise, p_range, n_p, n_start,
                                    steps, lr, seed, verbose=verbose)
    cands = {'decoder': PHI_DEC.numpy().copy(), 'grad': phi_grad,
             'refined': np.asarray(phi_ref, dtype=float)}
    scores = {k: float(abl.pavg_F(v, B)[0]) for k, v in cands.items()}
    pick = max(scores, key=scores.get)
    assert scores[pick] >= scores['decoder'] - 1e-15, \
        f'decoder floor violated: {scores}'
    assert scores[pick] >= scores['grad'] - 1e-15, \
        f'input floor violated: {scores}'
    info = {k: v for k, v in ref.items() if k != 'phi'}
    info.update({'picked': pick, 'scores': scores, 'p_range': p_range})
    # `refine_per_syndrome` reports the capture fraction relative to the INPUT
    # table.  The scientifically meaningful denominator is the certified headroom
    # above the DECODER, and when that headroom is exactly zero (depolarizing,
    # mixed) a ratio of two ~1e-16 quantities is meaningless -- which is why the
    # raw figure printed e.g. "33% of headroom" for a channel where refined ==
    # decoder == ceiling to all nine digits.  Report both, and flag the
    # degenerate case.
    F_dec = scores['decoder']
    head = float(ref['F_cert']) - F_dec
    info['headroom_vs_decoder'] = head
    info['headroom_is_zero'] = bool(head <= 1e-12)
    info['capture_vs_decoder'] = (
        1.0 if head <= 1e-12
        else max(0.0, (scores['refined'] - F_dec) / head))
    if verbose:
        print(f'    [{noise}{tag}] FIX A/B: F̄ decoder={scores["decoder"]:.9f} '
              f'grad={scores["grad"]:.9f} refined={scores["refined"]:.9f} -> '
              f'using "{pick}" ({scores[pick]:.9f}); certified ceiling '
              f'{ref["F_cert"]:.9f}, certified headroom over the decoder '
              f'{head:+.3e}'
              + ('  (zero: the decoder IS the optimum here)'
                 if info['headroom_is_zero'] else
                 f', captured {100 * info["capture_vs_decoder"]:.4f}% of it')
              + f', {int(ref["n_improved"])}/16 blocks improved; '
              f'F_out >= max(F_dec, F_input) enforced', flush=True)
    return cands[pick], info


def train_warm_best(noise, seeds=SEEDS, verbose=True, quad=True):
    """Multi-seed warm training with the same LABEL-FREE selection as the
    prototype: rank by worst per-syndrome conditional fidelity.

    quad=True (default) makes the whole pipeline exact and fast:

      * every epoch's objective is the deterministic Haar x noise-strength
        quadrature of `ssvr_qec.quad_cache`, propagated through only the
        2-dimensional syndrome subspace -- 1.5 ms/epoch against 67 ms for the
        equivalent full-32x32 reference loop (47x), and the gradient has zero
        variance, which is what makes a non-Pauli headroom as small as 5.4e-6
        resolvable instead of being buried in a 48-sample Monte-Carlo SEM of
        1.1e-4;
      * the seed-selection diagnostic uses
        `ssvr_qec.syndrome_conditional_fidelity_exact` rather than the M=150
        Monte-Carlo version, whose ~1e-3 sem is larger than the effect.

    quad=False restores the original Monte-Carlo path bit-for-bit.
    """
    schedule = WARM_SCHEDULES[noise]
    trs, infos = [], []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = VSCRWarm()
        hist_all = {'epoch': [], 'fidelity': [], 'loss': []}
        off = 0
        for ep, lr, pr in schedule:
            hist = m.train_vscr(model, n_epochs=ep, batch=48, p_range=pr,
                                noise=noise, lr=lr, lam=0.0, rng_seed=seed,
                                use_real_grad=True, verbose=False, quad=quad)
            hist_all['epoch'] += [off + e for e in hist['epoch']]
            hist_all['fidelity'] += hist['fidelity']
            hist_all['loss'] += hist['loss']
            off += ep
        if quad:
            with torch.no_grad():
                phi_now = model().detach().cpu()
            cf, w = m.syndrome_conditional_fidelity_exact(
                phi_now, m.quad_cache((0.07, 0.07), noise, n_p=1))
        else:
            cf, w = m.syndrome_conditional_fidelity(model, noise)
        mask = w > 0.001
        min_cf = float(np.where(mask, cf, 10.0).min())
        val_fid = hist_all['fidelity'][-1]
        infos.append({'seed': seed, 'min_cf': min_cf, 'val_fid': val_fid,
                      'quad': bool(quad)})
        if verbose:
            print(f'    [{noise}] seed {seed}: val_fid={val_fid:.6f} '
                  f'min syndrome cf={min_cf:.6f}'
                  f'{" (exact)" if quad else " (MC)"}', flush=True)
        trs.append((min_cf, val_fid, seed, model, hist_all))
    pool = [t for t in trs if t[1] >= 0.85] or [max(trs, key=lambda t: t[1])]
    best = max(pool, key=lambda t: (t[0], t[1]))
    model, hist_all = best[3], best[4]

    # -----------------------------------------------------------------
    # FIX A (decoder floor) + FIX B (separable per-syndrome refinement).
    # See `refine_with_floor` for the mechanism.  In short: PHI_DEC is an EXACT
    # stationary point of this objective -- measured |autograd| 8.3e-19 and
    # |central finite difference| 0.0 by `vscr_paper_abl._selftest_refine` -- i.e.
    # a SADDLE with the certified headroom F_unit - F_dec sitting above it, so no
    # first-order method started there can move off it.  That is why the
    # warm-start curve used to lie exactly on top of the decoder baseline in
    # Fig.3 and why the "0.0000" headroom capture was structural rather than bad
    # luck.  The refinement escapes the saddle via the exact separability of the
    # p-averaged objective across syndromes, and the floor makes
    # F_warm >= F_dec hold by construction.
    # -----------------------------------------------------------------
    p_range = tuple(schedule[-1][2])
    with torch.no_grad():
        phi_grad = model().detach().cpu().numpy().copy()
    phi_best, ref = refine_with_floor(phi_grad, noise, p_range, verbose=verbose)
    model.set_phi(phi_best)
    scores = ref['scores']
    infos.append({
        'stage': 'refine', 'p_range': list(p_range), 'picked': ref['picked'],
        'F_decoder': scores['decoder'], 'F_grad': scores['grad'],
        'F_refined': scores['refined'], 'F_used': scores[ref['picked']],
        'F_cert': float(ref['F_cert']), 'F_bound': float(ref['F_bound']),
        'headroom': float(ref['headroom']),
        'capture_frac': float(ref['capture_frac']),
        'headroom_vs_decoder': float(ref['headroom_vs_decoder']),
        'headroom_is_zero': bool(ref['headroom_is_zero']),
        'capture_vs_decoder': float(ref['capture_vs_decoder']),
        'n_improved': int(ref['n_improved']),
        'unitarity_dev': float(ref['unitarity_dev']),
        'cert_valid': bool(ref['cert_valid']),
        'prod_err': float(ref['prod_err']),
        'branch_deviation': model.branch_deviation(),
        'syndrome_spread': model.syndrome_spread()})
    return model, hist_all, infos



# ---------------------------------------------------------------------
# Physical (PSD-projected) LinDR baseline
# ---------------------------------------------------------------------
def _psd_project(rho):
    rho = 0.5 * (rho + rho.conj().T)
    ev, V = torch.linalg.eigh(rho)
    ev = ev.clamp_min(0.0)
    tr = ev.sum()
    if float(tr) < 1e-12:
        return rho
    return (V * ev) @ V.conj().T / tr


def train_lindr_phys(n_train=1500, p_range=(0.01, 0.12), noise='depolarizing',
                     n_val=200, rng_seed=m.SEED + 99):
    """Same ridge fit as ssvr_qec.train_lindr but the ridge strength is
    selected on PHYSICALLY PROJECTED (PSD, trace-1) validation fidelity."""
    rng = np.random.RandomState(rng_seed)

    def gen(n):
        Xs, Ys, PS = [], [], []
        for _ in range(n):
            psi = m.random_logical_state(1, rng)
            pe = m.encode(psi)
            p = rng.uniform(*p_range)
            Xs.append(m._pauli_features(m.noisy_state(pe, p, noise)).numpy())
            Ys.append(m._pauli_features(torch.outer(pe, pe.conj())).numpy())
            PS.append(pe)
        return np.stack(Xs), np.stack(Ys), torch.stack(PS)

    X, Y, _ = gen(n_train)
    Vx, _, Vpsi = gen(n_val)
    XtX, XtY = X.T @ X, X.T @ Y
    best_f, best_B = -1.0, None
    for a in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        B = np.linalg.solve(XtX + (a * n_train + 1e-6) * np.eye(m.N_PAULI), XtY)
        Yh = Vx @ B
        Yh[:, 0] = 1.0
        coeffs = torch.tensor(Yh, dtype=torch.float64).to(torch.complex128)
        rho = torch.einsum('mk,kij->mij', coeffs, m.PAULI_STACK) / m.INV_DIM
        F = [float((Vpsi[i].conj() @ _psd_project(rho[i]) @ Vpsi[i]).real)
             for i in range(rho.shape[0])]
        f = float(np.mean(F))
        if f > best_f:
            best_f, best_B = f, B
    return torch.tensor(best_B.T.copy(), dtype=torch.float64)


def lindr_phys_fidelity(psi_enc, rho_noisy, A):
    y = A @ m._pauli_features(rho_noisy)
    y = y.clone(); y[0] = 1.0
    rho = m._reconstruct_state(y.to(torch.complex128))
    rho = _psd_project(rho)
    return float((psi_enc.conj() @ rho @ psi_enc).real)


# ---------------------------------------------------------------------
# PHYSICAL Zero-Noise Extrapolation
#
# The raw Richardson ZNE estimator
#     F = <psi| 3 rho(p) - 3 rho(2p) + rho(3p) |psi>
# is an UNBOUNDED linear functional of the extrapolated operator: nothing in its
# definition keeps that operator inside the state space, so nothing keeps F <= 1.
# On the coherent channel it does not -- `zne_unphysical_overshoot` measures the
# overshoot directly -- and there the extrapolated "state" wins the Fig.3
# comparison only by exceeding physical fidelity.  That is an estimator artefact,
# not a result.  So ZNE is projected onto the nearest physical (PSD, trace-1)
# operator before scoring, exactly as LinDR-phys already is, and the fidelity
# axis is capped at 1.  The unprojected value is retained solely for the audit
# recorded in paper_numbers.json.
# ---------------------------------------------------------------------
def zne_raw_fidelity(psi_enc, p, noise):
    """UNPROJECTED Richardson ZNE estimator.  Kept only for the overshoot audit;
    it can exceed 1 and is never plotted."""
    rho_zne = (3.0 * m.noisy_state(psi_enc, p * 1.0, noise)
               - 3.0 * m.noisy_state(psi_enc, p * 2.0, noise)
               + m.noisy_state(psi_enc, p * 3.0, noise))
    return float((psi_enc.conj() @ rho_zne @ psi_enc).real)


def zne_phys_fidelity(psi_enc, p, noise):
    """Richardson ZNE (scales 1,2,3) with the extrapolated operator projected
    onto the nearest physical (PSD, trace-1) state before scoring.  Identical
    treatment to LinDR-phys; without it the reported "fidelity" is an unbounded
    estimator artefact that exceeds 1 on the coherent channel."""
    rho_zne = (3.0 * m.noisy_state(psi_enc, p * 1.0, noise)
               - 3.0 * m.noisy_state(psi_enc, p * 2.0, noise)
               + m.noisy_state(psi_enc, p * 3.0, noise))
    return float((psi_enc.conj() @ _psd_project(rho_zne) @ psi_enc).real)


def zne_unphysical_overshoot(p_values, noise, n_test=60, seed=m.SEED + 11):
    """Audit of the raw (unprojected) ZNE estimator: returns {p: max F} where
    F = <psi|rho_zne|psi> may exceed 1.  Recorded in paper_numbers.json so the
    reader can see exactly why the physical projection was introduced."""
    rng = np.random.RandomState(seed)
    states = [m.encode(m.random_logical_state(1, rng)) for _ in range(n_test)]
    out = {}
    for p in p_values:
        vals = [zne_raw_fidelity(pe, float(p), noise) for pe in states]
        out[f'{float(p):g}'] = {'max_F_raw_zne': float(max(vals)),
                               'mean_F_raw_zne': float(np.mean(vals)),
                               'n_over_1': int(sum(v > 1.0 + 1e-12 for v in vals)),
                               'n_test': int(n_test)}
    return out


# ---------------------------------------------------------------------
# Paper evaluation: all methods, 200 test states, mean +/- sem
# ---------------------------------------------------------------------
PAPER_METHODS = ['Raw', 'Perfect-code decoder', 'ZNE-phys (oracle)',
                 'VD (oracle)', 'LinDR-phys (oracle)', 'Petz recovery (2024)',
                 'VSCR cold-start', 'VSCR warm (ours)']
# On a UNITARY (coherent) channel rho is pure, so rho^2 = rho and the k = 2
# virtual-distillation estimator is algebraically identical to Raw.  Plotting it
# as a separate baseline would draw the same curve twice and imply a comparison
# that does not exist; `evaluate_paper` asserts the identity to machine
# precision and then omits the curve, recording the check in its audit dict.
VD_DEGENERATE_FOR = ('coherent',)
PAPER_COLORS = {
    'Raw': '#9e9e9e', 'Perfect-code decoder': '#e41a1c',
    'ZNE-phys (oracle)': '#ff7f00', 'VD (oracle)': '#984ea3',
    'LinDR-phys (oracle)': '#4daf4a', 'Petz recovery (2024)': '#377eb8',
    'VSCR cold-start': '#a65628', 'VSCR warm (ours)': '#1f4e9c',
}
PAPER_MARKERS = {
    'Raw': 'v', 'Perfect-code decoder': '^', 'ZNE-phys (oracle)': 's',
    'VD (oracle)': 'D', 'LinDR-phys (oracle)': 'P',
    'Petz recovery (2024)': '*', 'VSCR cold-start': 'x',
    'VSCR warm (ours)': 'o',
}


def evaluate_paper(model_warm, A_lindr, p_values, noise, n_test=N_TEST,
                   seed=m.SEED + 7, cold_R=None):
    """method -> {'F': mean, 'F_sem': sem, 'LER': mean, 'LER_sem': sem}.

    Baselines are all scored on PHYSICAL states (ZNE-phys, LinDR-phys) so that
    F <= 1 for every method.  On channels listed in VD_DEGENERATE_FOR the VD
    baseline is omitted after asserting F_VD == F_Raw to machine precision
    (rho^2 = rho for a unitary channel), instead of plotting a duplicate curve.
    Returns (out, audit) where audit records the VD/Raw degeneracy check."""
    rng = np.random.RandomState(seed)
    test_states = [m.random_logical_state(1, rng) for _ in range(n_test)]
    vd_degenerate = noise in VD_DEGENERATE_FOR
    names = [x for x in PAPER_METHODS
             if not (x == 'VSCR cold-start' and cold_R is None)
             and not (x == 'VD (oracle)' and vd_degenerate)]
    out = {nm: {'F': [], 'F_sem': [], 'LER': [], 'LER_sem': []} for nm in names}
    audit = {'noise': noise, 'vd_degenerate': bool(vd_degenerate),
             'vd_raw_max_absdiff': 0.0, 'max_F_over_all_methods': 0.0,
             'n_test': int(n_test), 'methods': list(names)}
    with torch.no_grad():
        R_all = m.recovery_unitary_batch(model_warm())
    for p in p_values:
        fids = {nm: [] for nm in names}
        vd_raw_diff = 0.0
        for psi in test_states:
            pe = m.encode(psi)
            rho_n = m.noisy_state(pe, p, noise)
            f_raw = float(m.baseline_raw(pe, rho_n))
            fids['Raw'].append(f_raw)
            fids['Perfect-code decoder'].append(
                float(m.perfect_code_decoder_fidelity(pe, rho_n)))
            fids['ZNE-phys (oracle)'].append(zne_phys_fidelity(pe, p, noise))
            f_vd = float(m.virtual_distillation_fidelity(pe, rho_n))
            if vd_degenerate:
                vd_raw_diff = max(vd_raw_diff, abs(f_vd - f_raw))
            else:
                fids['VD (oracle)'].append(f_vd)
            fids['LinDR-phys (oracle)'].append(
                lindr_phys_fidelity(pe, rho_n, A_lindr))
            fids['Petz recovery (2024)'].append(
                m.petz_recovery_fidelity(pe, rho_n, noise, p))
            if cold_R is not None:
                with torch.no_grad():
                    _, Fc = m.vscr_fidelity(pe, rho_n, cold_R, lam=0.0)
                fids['VSCR cold-start'].append(float(Fc))
            with torch.no_grad():
                _, Fw = m.vscr_fidelity(pe, rho_n, R_all, lam=0.0)
            fids['VSCR warm (ours)'].append(float(Fw))
        if vd_degenerate:
            audit['vd_raw_max_absdiff'] = max(audit['vd_raw_max_absdiff'],
                                              vd_raw_diff)
        for nm in names:
            arr = np.array(fids[nm])
            ler = (arr < 0.9).astype(float)
            out[nm]['F'].append(float(arr.mean()))
            out[nm]['F_sem'].append(float(arr.std(ddof=1) / math.sqrt(n_test)))
            out[nm]['LER'].append(float(ler.mean()))
            out[nm]['LER_sem'].append(float(ler.std(ddof=1) / math.sqrt(n_test)))
            audit['max_F_over_all_methods'] = max(
                audit['max_F_over_all_methods'], float(arr.max()))
    if vd_degenerate:
        # rho^2 = rho for a pure (unitary-channel) state, so the k=2 VD
        # estimator IS the Raw estimator; assert it rather than assume it.
        assert audit['vd_raw_max_absdiff'] < 1e-12, \
            (f"VD != Raw on unitary channel {noise}: "
             f"{audit['vd_raw_max_absdiff']:.2e}")
    # every method is now scored on a physical state, so no F may exceed 1
    assert audit['max_F_over_all_methods'] <= 1.0 + 1e-9, audit
    return out, audit



# ---------------------------------------------------------------------
# Per-syndrome conditional-fidelity tables
# ---------------------------------------------------------------------
def cf_table(R_all, noise, p, M=400, rng_seed=m.SEED + 5):
    """Label-free per-syndrome conditional fidelity cf_s and weights w_s."""
    rng = np.random.RandomState(rng_seed)
    Fs = np.zeros(16); Ws = np.zeros(16)
    with torch.no_grad():
        for _ in range(M):
            psi = m.random_logical_state(1, rng)
            pe = m.encode(psi)
            rho = m.noisy_state(pe, p, noise)
            rho16 = rho.unsqueeze(0).expand(16, m.DIM, m.DIM)
            PRP = torch.bmm(torch.bmm(m.P_SYNDS_STACK, rho16), m.P_SYNDS_STACK)
            Y = torch.bmm(torch.bmm(R_all, PRP), R_all.conj().transpose(1, 2))
            Fs += torch.einsum('i,sij,j->s', pe.conj(), Y, pe).real.numpy()
            Ws += torch.einsum('sij,ji->s', m.P_SYNDS_STACK, rho).real.numpy()
    return Fs / np.maximum(Ws, 1e-12), Ws / M


def save_fig(fig, name):
    fig.savefig(f'{FIGDIR}/{name}.pdf')
    fig.savefig(f'{FIGDIR}/{name}.png')
    plt.close(fig)
    print(f'  figure: {FIGDIR}/{name}.pdf', flush=True)


def plot_benchmark(curves_by_noise, fname='fig_sim_benchmark', p_vals=None):
    """Fig.3: average recovery fidelity vs p, all methods, 3 incoherent channels.

    FIX C.  The y-axis is capped at exactly 1.0 and the physical ceiling is
    drawn as a reference line.  Every baseline is now scored on a PHYSICAL state
    (ZNE-phys, LinDR-phys), so F <= 1 for all of them and 1.0 is the true
    ceiling rather than an arbitrary frame.  The previous 1.03 upper limit
    existed only to accommodate the UNPROJECTED Richardson-ZNE estimator, which
    reaches 1.0025 on the coherent channel: the axis had been stretched to fit
    an estimator artefact, and ZNE appeared to beat VSCR there purely by
    overshooting physical fidelity.  `evaluate_paper` asserts max F <= 1 + 1e-9
    for every method and records the raw overshoot in paper_numbers.json."""
    p_vals = P_VALUES if p_vals is None else p_vals
    titles = {'depolarizing': 'Depolarizing', 'amplitude_damping': 'Amplitude damping',
              'mixed': 'Mixed (depol. + AD)'}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.05), sharey=True)
    for ax, noise in zip(axes, ['depolarizing', 'amplitude_damping', 'mixed']):
        res = curves_by_noise[noise]
        ax.axhline(1.0, color='k', lw=0.7, ls=':', zorder=0)
        for nm in res:
            F = np.array(res[nm]['F']); E = np.array(res[nm]['F_sem'])
            assert F.max() <= 1.0 + 1e-9, (noise, nm, float(F.max()))
            ax.plot(p_vals, F, marker=PAPER_MARKERS[nm], ms=3.5, lw=1.4,
                    color=PAPER_COLORS[nm], label=nm)
            if E.max() > 0.002:
                ax.fill_between(p_vals, F - E, F + E, color=PAPER_COLORS[nm], alpha=0.25)
        ax.set_xlabel('physical error rate $p$')
        ax.set_title(titles[noise])
        ax.grid(alpha=0.3, lw=0.5)
        ax.set_ylim(0.25, 1.0)
    axes[0].set_ylabel('average recovery fidelity $\\bar{F}$')
    axes[-1].annotate('physical ceiling $\\bar{F}=1$', xy=(0.985, 1.0),
                      xycoords=('axes fraction', 'data'), ha='right', va='top',
                      fontsize=6.5, color='k')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.145), columnspacing=1.4,
               handlelength=1.8, fontsize=7.5)
    save_fig(fig, fname)


def plot_ler(curves_by_noise, fname='fig_sim_ler', p_vals=None):
    p_vals = P_VALUES if p_vals is None else p_vals
    titles = {'depolarizing': 'Depolarizing', 'amplitude_damping': 'Amplitude damping',
              'mixed': 'Mixed (depol. + AD)'}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, noise in zip(axes, ['depolarizing', 'amplitude_damping', 'mixed']):
        res = curves_by_noise[noise]
        for nm in res:
            L = np.array(res[nm]['LER'])
            ax.semilogy(p_vals, np.maximum(L, 3e-4), marker=PAPER_MARKERS[nm],
                        ms=3.5, lw=1.4, color=PAPER_COLORS[nm], label=nm)
        ax.set_xlabel('physical error rate $p$')
        ax.set_title(titles[noise])
        ax.grid(alpha=0.3, lw=0.5, which='both')
    axes[0].set_ylabel('logical error rate  ($F<0.9$)')
    save_fig(fig, fname)


def plot_training(hist_warm, fname='fig_training'):
    """Warm curve (this run) + cold real-grad + cold random-walk (v1 npz)."""
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    old = np.load('vscr_results.npz', allow_pickle=True)
    ax.plot(old['hist_rand_fid'], color='#a65628', lw=1.2, ls='--',
            label='cold start, random updates')
    ax.plot(old['hist_real_fid'], color='#e08214', lw=1.2,
            label='cold start, Adam gradients (v1)')
    ax.plot(hist_warm['fidelity'], color='#1f4e9c', lw=1.6,
            label='decoder warm start (ours)')
    ax.set_xlabel('epoch'); ax.set_ylabel('validation fidelity ($p=0.06$)')
    ax.set_ylim(0.5, 1.02); ax.grid(alpha=0.3, lw=0.5)
    ax.legend(loc='lower right', fontsize=7)
    save_fig(fig, fname)


def plot_branch_cf(cf_dec, cf_cold, cf_warm, fname='fig_branch_cf'):
    labels = [f'{s}\n{m.SYND_TABLE[m.SYND_BITS[s]]}' for s in range(16)]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    x = np.arange(16); bw = 0.27
    ax.bar(x - bw, cf_dec, bw, color='#e41a1c', label='perfect-code decoder')
    if cf_cold is not None:
        ax.bar(x, cf_cold, bw, color='#a65628', label='VSCR cold-start (v1)')
    ax.bar(x + bw, cf_warm, bw, color='#1f4e9c', label='VSCR warm (ours)')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6)
    ax.set_xlabel('syndrome $s$ / decoder correction $C_s$')
    ax.set_ylabel('conditional fidelity $\\mathrm{cf}_s$')
    ax.set_ylim(0, 1.08); ax.grid(alpha=0.3, lw=0.5, axis='y')
    ax.legend(fontsize=7, loc='lower left', ncol=3)
    save_fig(fig, fname)



# ---------------------------------------------------------------------
# Hardware-facing exact numbers with the NEW (paper) depolarizing angles
# ---------------------------------------------------------------------
def hardware_frame_numbers(phi_warm_dep):
    """Exact ideal reference for the future full QPU benchmark, computed with
    the NEW warm angles: for psi in {0,+} and all 16 single-error Pauli
    frames, the per-frame circuit fidelity F = sum_s F_s and the per-branch
    F_s (the quantity the hardware post-selection estimates).  Also
    re-verifies circuit == density-matrix simulator to <1e-9."""
    import qcloud_vscr_new as qc
    qc.setup()                                   # synthesise + verify encoder
    phi = np.asarray(phi_warm_dep, dtype=float)
    frames = [m.SYND_TABLE[b] for b in m.SYND_BITS]   # IIIII + 15 single errors
    out = {'frames': frames, 'per_frame': {}, 'worst_circuit_vs_simulator': 0.0}
    worst = 0.0
    for psi in ['0', '+']:
        for fr in frames:
            Fc, per_s = qc.circuit_fidelity_exact(psi, fr, phi)
            Fs = qc.simulator_fidelity_frame(psi, fr, phi)
            worst = max(worst, abs(Fc - Fs))
            out['per_frame'][f'{psi}|{fr}'] = {'F_circuit': Fc, 'F_sim': Fs,
                                               'per_s': {str(k): v for k, v in per_s.items()}}
    out['worst_circuit_vs_simulator'] = worst
    assert worst < 1e-9, worst
    print(f'[hardware-ref] circuit==simulator verified, worst diff {worst:.2e}',
          flush=True)
    return out



SHORT = {'depolarizing': 'dep', 'amplitude_damping': 'ad', 'mixed': 'mixed'}
LINDR_RANGES = {'depolarizing': (0.01, 0.12),
                'amplitude_damping': (0.01, 0.08),
                'mixed': (0.01, 0.10)}


def main():
    t0 = time.time()
    _verify_warm_start()

    cold_phi = np.load('vscr_angles_dep.npz')['phi']
    with torch.no_grad():
        cold_R = m.recovery_unitary_batch(torch.tensor(cold_phi, dtype=torch.float64))
    R_dec = m.recovery_unitary_batch(PHI_DEC)

    curves, hists, models, infos_all, audits = {}, {}, {}, {}, {}
    for noise in ['depolarizing', 'amplitude_damping', 'mixed']:
        print(f'\n=== {noise}: training warm VSCR ({len(SEEDS)} seeds) ===', flush=True)
        model, hist, infos = train_warm_best(noise)
        models[noise], hists[noise], infos_all[noise] = model, hist, infos
        with torch.no_grad():
            phi = model().cpu().numpy()
        np.savez(f'vscr_angles_paper_{SHORT[noise]}.npz', phi=phi,
                 val_fidelity=np.array([hist['fidelity'][-1]]),
                 infos=json.dumps(infos))
        print(f'  fitting physical LinDR ({noise}) ...', flush=True)
        A = train_lindr_phys(noise=noise, p_range=LINDR_RANGES[noise])
        print(f'  evaluating ({noise}, n_test={N_TEST}) ...', flush=True)
        curves[noise], audits[noise] = evaluate_paper(
            model, A, P_VALUES, noise,
            cold_R=cold_R if noise == 'depolarizing' else None)

    # random-walk ablation (depolarizing, warm init, same schedule)
    print('\n=== ablation: random-walk updates from warm init ===', flush=True)
    torch.manual_seed(1234); np.random.seed(1234)
    rw_model = VSCRWarm()
    hist_rw = {'epoch': [], 'fidelity': [], 'loss': []}
    off = 0
    for ep, lr, pr in WARM_SCHEDULES['depolarizing']:
        h = m.train_vscr(rw_model, n_epochs=ep, batch=48, p_range=pr,
                         noise='depolarizing', lr=lr, lam=0.0, rng_seed=1234,
                         use_real_grad=False, verbose=False, quad=True)
        hist_rw['epoch'] += [off + e for e in h['epoch']]
        hist_rw['fidelity'] += h['fidelity']
        hist_rw['loss'] += h['loss']
        off += ep

    # per-syndrome conditional fidelities (depolarizing)
    with torch.no_grad():
        R_warm_dep = m.recovery_unitary_batch(models['depolarizing']())
    cf = {}
    for tag, p in [('05', 0.05), ('15', 0.15)]:
        cf[f'dec_{tag}'], w = cf_table(R_dec, 'depolarizing', p)
        cf[f'cold_{tag}'], _ = cf_table(cold_R, 'depolarizing', p)
        cf[f'warm_{tag}'], _ = cf_table(R_warm_dep, 'depolarizing', p)
        if tag == '15':
            cf['w15'] = w

    print('\n=== figures ===', flush=True)
    plot_benchmark(curves)
    plot_ler(curves)
    plot_training(hists['depolarizing'])
    plot_branch_cf(cf['dec_15'], cf['cold_15'], cf['warm_15'])

    hw = hardware_frame_numbers(models['depolarizing']().detach().cpu().numpy())
    _save_all(curves, hists, hist_rw, cf, infos_all, hw, t0, audits=audits)


def _save_all(curves, hists, hist_rw, cf, infos_all, hw, t0, audits=None):
    np.savez(RESULTS_NPZ,
             p_values=P_VALUES,
             curves={n: curves[n] for n in curves},
             hist_warm_dep=np.array(hists['depolarizing']['fidelity']),
             hist_warm_ad=np.array(hists['amplitude_damping']['fidelity']),
             hist_warm_mx=np.array(hists['mixed']['fidelity']),
             hist_randwalk=np.array(hist_rw['fidelity']),
             cf_dec_05=cf['dec_05'], cf_cold_05=cf['cold_05'], cf_warm_05=cf['warm_05'],
             cf_dec_15=cf['dec_15'], cf_cold_15=cf['cold_15'], cf_warm_15=cf['warm_15'],
             syndrome_weights_15=cf['w15'], infos=json.dumps(infos_all))

    i10 = int(np.where(P_VALUES == 0.10)[0][0])
    numbers = {
        'runtime_s': time.time() - t0,
        'infos': infos_all,
        'eval_audits': audits or {},
        # FIX C audit: the UNPROJECTED Richardson-ZNE estimator is an unbounded
        # linear functional and exceeds 1 on the coherent channel, which is why
        # ZNE is now reported projected ('ZNE-phys') and the fidelity axis is
        # capped at 1.  Recorded here so the reader can see the raw artefact.
        'zne_unphysical_overshoot': {
            n: zne_unphysical_overshoot(P_VALUES, n)
            for n in ['depolarizing', 'amplitude_damping', 'mixed']},
        'dep_p010': {nm: curves['depolarizing'][nm]['F'][i10]
                     for nm in curves['depolarizing']},
        'ad_p010': {nm: curves['amplitude_damping'][nm]['F'][i10]
                    for nm in curves['amplitude_damping']},
        'mx_p010': {nm: curves['mixed'][nm]['F'][i10]
                    for nm in curves['mixed']},
        'dep_full_curves': {nm: curves['depolarizing'][nm]['F']
                            for nm in curves['depolarizing']},
        'ad_full_curves': {nm: curves['amplitude_damping'][nm]['F']
                           for nm in curves['amplitude_damping']},
        'mx_full_curves': {nm: curves['mixed'][nm]['F'] for nm in curves['mixed']},
        'min_cf_dep_p015': {'decoder': float(np.min(cf['dec_15'][cf['w15'] > 1e-3])),
                            'cold_v1': float(np.min(cf['cold_15'][cf['w15'] > 1e-3])),
                            'warm': float(np.min(cf['warm_15'][cf['w15'] > 1e-3]))},
        'min_cf_dep_p005': {'decoder': float(np.min(cf['dec_05'])),
                            'cold_v1': float(np.min(cf['cold_05'])),
                            'warm': float(np.min(cf['warm_05']))},
        'hardware_frame_numbers': hw,
    }
    with open('paper_numbers.json', 'w') as f:
        json.dump(numbers, f, indent=2)

    print('\n================ PAPER NUMBERS ================')
    for k in ['dep_p010', 'ad_p010', 'mx_p010']:
        print(k, json.dumps({a: round(b, 4) for a, b in numbers[k].items()}))
    print('min_cf_dep_p015', numbers['min_cf_dep_p015'])
    print('min_cf_dep_p005', numbers['min_cf_dep_p005'])
    print(f'total runtime {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
