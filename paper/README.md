# Manuscript package — VSCR [[5,1,3]] QEC

`main.tex` + `refs.bib` + `figures/` compose the manuscript
*"Variational syndrome-conditioned recovery for quantum error correction:
from unsupervised learning to a superconducting quantum processor"*.
All bibliography entries were verified against Crossref (DOIs in `refs.bib`;
verification dumps: `../refs_verification*.json`).

## Build

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
(Any TeX distribution with `natbib`, `graphicx`, `booktabs`, `siunitx`.)
Number placeholders (`@TOKEN@`) are filled from the analysis artifacts by
`python fill_numbers.py` **after** `../vscr_paper.py` has finished.

## Regenerating everything (free, local)

```bash
cd ..                                  # repo root
./qenv/bin/python run_selftests.py     # numerical self-tests (see below); --quick to skip slow ones
./qenv/bin/python vscr_paper.py        # warm-start training + benchmarks + figs
./qenv/bin/python vscr_paper_coh.py    # coherent-channel supplement (Fig. 3c)
./qenv/bin/python vscr_paper_abl.py    # same-family baselines + SDP ceiling + ablations
./qenv/bin/python scaling_analysis.py  # exact n=5/7/9 ceilings + readout law
./qenv/bin/python ancilla_recovery.py  # Kraus-rank ladder -> ancilla_recovery.json
./qenv/bin/python hw_verify_analysis.py# hardware feasibility fig + numbers
./qenv/bin/python make_schematic.py    # Fig. 1
cd paper && ./qenv/bin/python fill_numbers.py && ./qenv/bin/python make_ed.py
./qenv/bin/python audit_numbers.py     # consistency gate: numbers <-> figures <-> LaTeX
```

Diagnostics (not part of the figure pipeline, but they produce the numbers the
manuscript quotes about the decoder saddle and the certified headroom):

```bash
./qenv/bin/python diag_optunit.py      # exact optimal unitary recovery per branch
                                       #   -> diag_optunit.json (+ --headroom
                                       #      rewrites recovery/headroom_table.json)
./qenv/bin/python diag_check.py        # Fix A/B audit per channel: gradient at
                                       #   PHI_DEC, certified headroom, captured
                                       #   fraction, F_warm >= F_dec  -> diag_check.json
```

`ancilla_recovery.py` answers "how many ancillas does the non-unitary headroom
need?" exactly. Because the Kraus rank of a branch recovery equals the rank of its
Choi matrix, restricting the ceiling program's Choi factor `C = L L^dag` to
`L in C^{4 x r}` restricts the recovery to what an ancilla of dimension `r` can
realise as a unitary followed by a discard. `r=1` is therefore the unitary family
VSCR compiles to, `r=2` a one-ancilla gate-level instrument, and `r=4` the
unrestricted CPTP ceiling — one integer turns the ceiling into a count of ancillas.
The end rungs are asserted to reproduce `ab._max_unitary_J` and `ab._sdp_branch`,
the Choi objective is asserted to equal `g.cf_unnormalised`, the `r=2` optimum is
repaired to exact trace preservation and dilated to an explicit `4x4` unitary that
is checked by tracing the ancilla out, an independent unconstrained `exp(iH)`
optimisation is asserted to land on the same value, and coherent/depolarizing are
run through the same ladder as negative controls that must come out flat.
Use `--quick` for n=5,7 amplitude damping only, `--selftest` for the checks alone.

`run_selftests.py --list` shows what each test pins. All of them are assertions,
not plots: if any one fails, a specific claim or figure in the manuscript is
unsupported.

Verification is deliberately two-layered, because the two layers fail differently:

- `run_selftests.py` (11 tests) checks that the **code** computes what the
  manuscript claims — the warm start is phase-exactly the Pauli decoder, the
  Haar×p quadrature equals the full 32×32 reference, the decoder is an exact
  stationary point, the SDP ceiling bounds it, and the refinement is monotone.
  It runs each test in its own subprocess with retries (see the host note
  below), so one native fault cannot abort the rest of the suite.
- `audit_numbers.py` (543 checks, <1 s, exit 0 iff clean) checks that the
  **artifacts** are mutually consistent and physical: every reported fidelity
  lies in [0,1], `F_warm >= F_decoder` at *every* p on *every* channel, no
  Fig. 4(b) gap-to-ceiling bar is negative or above its own bound, the SDP
  ceiling satisfies weak duality at all 8 operating points, every `\cite` and
  `\ref` resolves, no `\label` is orphaned, and every `\includegraphics` target
  exists. It recomputes nothing — it reads `paper_numbers.json` and the `.tex`
  files — so it is a cheap pre-submission gate. Run it after any regeneration.

Artifacts consumed by the manuscript:
`vscr_paper_results.npz`, `vscr_angles_paper_{dep,ad,mixed,coh}.npz`,
`vscr_paper_abl_results.npz`, `vscr_angles_abl_ind_{dep,ad,mixed,coh}.npz`,
`paper_numbers.json`, `hw_feasibility_numbers.json`,
`paper/figures/fig_{schematic,training,branch_cf,sim_benchmark,sim_ler,coherent,ablation,hw_feasibility}.pdf`.

**Never overwrite `vscr_angles_dep.npz`** — it is the v1 snapshot whose
branch circuits produced the WK_C180 feasibility data (Fig. 4).

## Journal strategy (SCI Q1, review speed)

| venue | quartile | typical 1st decision | fit assessment |
|---|---|---|---|
| **Nature Communications** | Q1 | ~4–8 wk | best with the FULL 128-circuit hardware benchmark; feasibility-only hardware section risks desk rejection |
| **PRX Quantum** | Q1 | ~4–6 wk | strong if hardware section upgraded; rigorous-reviewer crowd values the exact circuit-level verification |
| **npj Quantum Information** | Q1 | ~3–6 wk | **recommended target for the current data**: method + benchmarks + honest hardware feasibility is squarely in scope |
| **Quantum Sci. Technol.** | Q1 | ~4–8 wk | good fallback; hardware-protocol emphasis welcome |
| **Phys. Rev. Applied** | Q1 | ~2–4 wk | fastest solid option; device/benchmark framing |
| **Phys. Rev. Lett.** | Q1 | ~2–4 wk | only if condensed to 4+3 pages; broad-interest bar high |

Decision rule: if QPU time is purchased and the full benchmark lands with
$F$ within a few percent of the ideal reference → Nature Communications or
PRX Quantum. Otherwise submit to npj Quantum Information now and offer the
full benchmark as a revision/strengthener.

## When QPU time arrives (full benchmark integration)

1. `export VSCR_ANGLES_FILE=$PWD/vscr_angles_paper_dep.npz` (new CLI hook in
   `qcloud_vscr_new.load_angles`) so the QPU runs the paper model.
2. Optional probe: `./qenv/bin/python -u qcloud_probe_point.py`.
3. `bash launch_full.sh` → `qcloud_raw_full.json`,
   `qcloud_results_full_late_1000.json`,
   `figures/fig_qcloud_hardware_full_late_1000.png` (incremental dumps).
4. Manuscript updates: promote Fig. 4 to a full-benchmark figure
   (hardware $F$ vs exact per-frame reference, per state/$p$/frame), move the
   4-circuit feasibility characterization to Supplementary, update abstract
   and Results 2.5 numbers, re-run `fill_numbers.py`.
   Ideal references for every branch are already tabulated in
   `extended_data.tex` (ED Table 1) from `paper_numbers.json`.

## Honesty notes (keep in any revision)

- ZNE / VD / LinDR-phys are **oracle** mitigation bounds (exact noise-scaled
  channels / copies / full tomography); they are not like-for-like QEC
  competitors and are labelled "(oracle)" everywhere.
- **ZNE is reported projected, as `ZNE-phys`.** The raw Richardson estimator
  $\langle\psi|3\rho(p)-3\rho(2p)+\rho(3p)|\psi\rangle$ is an unbounded linear
  functional, so nothing keeps it $\le1$: on the coherent channel its mean
  reaches **1.130** at $\varepsilon=0.15$ (and exceeds 1 from $\varepsilon=0.04$).
  An earlier version of Fig. 3 plotted that unprojected curve with the axis
  stretched to 1.03, so ZNE "won" the coherent panel purely by leaving the
  physical state space. All fidelity axes are now capped at exactly 1.0, the
  physical ceiling is drawn as a reference line, `evaluate_paper` asserts
  $\bar F\le1+10^{-9}$ for every method, and the raw overshoot per noise
  strength is recorded as `zne_unphysical_overshoot` /
  `coh_zne_unphysical_overshoot` in `paper_numbers.json`.
- **VD is omitted from the coherent panel.** On a unitary channel $\rho$ is pure,
  so $\rho^2=\rho$ and the $k{=}2$ VD estimator is *algebraically identical* to
  Raw (verified bit-identical at all ten noise strengths). `evaluate_paper`
  asserts the identity to machine precision and drops the duplicate curve rather
  than plotting two indistinguishable lines; `VD_DEGENERATE_FOR = ('coherent',)`.
- **`Petz recovery (2024)` is a real competing recovery, not a mitigation
  bound.** It is the noise-adapted recovery map / transpose channel
  (Biswas et al., Phys. Rev. Research **6**, 043034 (2024)), a closed-form CPTP
  channel scored with the *same* estimator as the decoder and VSCR, given the
  exact channel and the exact code-space reference, and allowed to be
  non-unitary and non-syndrome-resolved. It is the strongest recent comparator
  available for this problem class, and it is the honest worst case for VSCR.
- Fig. 4 hardware data used the v1 angle snapshot; branch structure,
  extraction and post-selection protocol are identical to the paper model.
- The identity branch ($s=0$) failure on WK_C180 (2.4% expected bitstring) is
  reported, not hidden; it motivates the readout-mitigation design of the
  full benchmark.
- Same-family ablations (`vscr_paper_abl.py`, ED Tables 5–7): the per-syndrome
  independent parameter table (VQR-ind) now receives the **same** separable
  refinement as the warm hypernetwork, so the comparison isolates the
  architecture rather than the post-processing. Without that, VQR-ind would have
  appeared worse by 9.9e-3 on amplitude damping and 3.1e-3 on coherent noise
  purely because it was left unrefined. With it, the two agree to ≤2.3e-9 in F̄
  across the full p-grid on all four channels (bit-identical on depolarizing and
  mixed), and in the *warm* low-data regime they are indistinguishable at every
  K ∈ {2,4,8,16} (F̄ = 0.9470, min cf_s = 0.8717). In the *cold*-start low-data
  regime neither is monotone in K, and the ordering flips: at K=2 the two tie
  (0.90574 versus 0.90570), at K=4 and K=16 the independent table is clearly
  stronger (0.94699 versus 0.90605 / 0.90593), but at K=8 it is *weaker*
  (0.87885 versus 0.90610). The cold regime is therefore too noisy in K to
  support any architectural claim in either direction; only the warm regime is
  interpretable, and there the two are identical. Those cold
  numbers are themselves an improvement on the pre-fix values (the cold
  hypernetwork used to stall at 0.879–0.892), i.e. part of that gap was the
  dead-hypernetwork bug rather than a property of sharing. The hypernetwork's
  justification remains asymptotic ($2^{n-k}$ branch scaling) and deployment
  (single model, exact per-branch decoder warm-start construction), not superior
  fidelity at this code size. No claim of the paper relies on the hypernetwork.
- **`sdp_ceiling` was under-reporting the ceiling, and could report a negative
  one.** `_sdp_branch` seeded `best = -1.0` and accepted a start only when scipy
  reported `success=True`; SLSQP reports `success=False` at many perfectly
  converged KKT points and always at `maxiter`, so a stalled branch contributed
  `-1/p_s`. That is how amplitude damping at p=0.30 once printed
  `F_CPTP = -0.1113`. It also made the ceiling silently *lower than a feasible
  point*: at amplitude damping p=0.10 the old value was 0.985115 against an exact
  unitary-family optimum of 0.986538, so Fig. 4's "ceiling" bar sat **below** the
  method it is supposed to bound, and the derived gap for VSCR came out negative
  (−1.2e-3). Fixed by (i) scoring every start directly and seeding the running
  best with the best *feasible* start, (ii) no longer trusting `r.success`, and
  (iii) seeding each branch with the exact unitary optimum and asserting
  `F̄^CPTP >= max(F̄^unitary, F̄^decoder)`. Corrected values: ad p=0.10
  0.985115 → **0.986538**; coh p=0.15 0.999916 → **1.000000** (a perfect recovery
  does exist for a known unitary). `vscr_paper_abl.main()` now also asserts that
  no method exceeds the ceiling, so a negative gap fails loudly instead of being
  clamped to zero and drawn as a bogus "exactly optimal" bar.
- The SDP ceiling shows the Pauli decoder is optimal among ALL CPTP
  recoveries (given the projective readout) on the Pauli-type channels
  (depolarizing: ≤4e-13; mixed: +9e-7). Amplitude damping (+7.2e-4) and
  coherent over-rotation (+1e-4 … +3.1e-3, ceiling exactly 1.0 at ε=0.30)
  leave small non-Pauli headroom concentrated in low-weight branches.
  **That headroom is now captured, not merely reported.** `PHI_DEC` is an exact
  stationary point of the objective — a *saddle*, not a minimum: every branch
  term is a 4×4 Rayleigh quotient and `vec(G_dec)` is an eigenvector of `Q_s`,
  so the gradient is identically zero (measured |autograd| 8.3e-19 and |central
  FD| exactly 0.0, asserted by `_selftest_refine`). Plain Adam therefore cannot
  move off it, and small symmetry-breaking offsets make things *worse*
  (−0.52 % … −0.03 % capture for eps in [1e-4, 1e-1], because a random 60-angle
  offset lies almost entirely in the ansatz's null directions). Because the
  branch weight cancels in `p_s·cf_s`, the p-averaged objective is *exactly
  separable* across the 16 syndromes, so `refine_per_syndrome` re-optimises each
  block independently from large symmetry-broken starts and reaches the
  certified optimum. The decoder floor in `train_warm_best` then keeps whichever
  of {decoder, gradient-trained, refined} maximises the exact production
  objective, so **`F_warm >= F_dec` holds by construction** on every channel —
  which matters, because gradient training was observed to drift *below* the
  decoder on exactly the channels whose headroom is zero. VSCR's claim on
  Pauli-type noise remains *ceiling saturation without labels*; on
  amplitude-damping and coherent noise it is now a measured gain over the
  decoder, bounded above by the certified ceiling. (The Pauli-class search still
  finds discrete +0.65–0.68 cf gains on weight-3e-5 branches that no continuous
  method can reach; that limit is unchanged.)
- `exact_F` / `pavg_F` carry **both** halves of the Haar second moment. The
  G-independent shortcut `Tr(G S G†) → Tr(S)` is exact only when
  `G_s = V† R_s W_s` is unitary, i.e. when `R_s` maps the syndrome subspace into
  the code space — true for the decoder and for every branch optimum, but false
  for an arbitrary angle table (measured error **3.0e-1** on a random table).
  Both evaluators therefore carry the full `Tr(G S G†)`, which is what makes
  `_selftest_refine` test 1 a genuine cross-check against the production
  objective for *arbitrary* tables rather than an identity that holds only at
  `PHI_DEC`. `_unitarity_dev` reports how far a table has left that regime.
- QVECTOR-style global unitary recovery (no projection) partially inverts
  purely coherent errors (F̄=0.9962 at ε=0.10) — the one tested regime where
  a measurement-free variational recovery is competitive; it collapses to
  raw-state level on stochastic channels (0.52–0.77 vs 0.925–0.985).
- Reproducibility note: `abl_fix_gaps.py` repaired the E3 branch-gap column
  after a `register_buffer` aliasing bug (zeroing a cold-start model's
  `phi_dec` buffer mutated the global `PHI_DEC`); root cause fixed in
  `vscr_paper.py` (clone on register). All other run outputs were
  unaffected (they consume `m.C_SYNDS` or pre-corruption artifacts).
- **A dead-hypernetwork bug was fixed.** `VSCRWarm` previously zero-initialised
  *both* `hnet[0]` and `hnet[2]`. With a zero hidden layer the activation
  `tanh(hnet[0] e_s)` is identically 0, so `hnet[2]`'s gradient — which is
  proportional to that activation — is identically 0 as well: neither layer
  could ever move, and the only trainable part of the syndrome pathway was the
  single global `phi_base`. The model could therefore apply at most ONE rotation
  shared by all 16 branches, which silently invalidated the
  "syndrome-conditioned recovery" claim and the K-dependence of the low-data
  ablation. Only the *output* layer is now zero-initialised (that alone is what
  makes the warm start exactly the decoder); the hidden layer keeps a
  non-degenerate `std = 1/sqrt(ctx)` init, and `synd_emb` is drawn from a private
  generator (seed 20240607) so it is reproducible independently of the training
  seed. `_selftest_hypernet_trainable` asserts that every syndrome-pathway
  parameter moves under the real label-free loss and that the 16 branch angle
  sets become mutually distinct (`syndrome_spread() > 0`).
- **Native threads are pinned to 1** at the top of `ssvr_qec.py` and the autograd
  engine is switched to single-threaded
  (`torch.autograd.set_multithreading_enabled(False)`). Every hot loop here is
  small-matrix bound (2×2, 4×4, 32×32), so pinning costs nothing measurable and
  additionally makes runs bit-reproducible. `scipy.linalg.expm` was also replaced
  by the closed form `exp(i h·σ) = e^{i h_0}[cos ρ·I + i·sinc(ρ)(r·σ)]`, asserted
  equal to `expm` to 1e-14 by `_selftest_liealg` and ~40× cheaper inside the
  multi-start solves.
- **Known host defect: intermittent native faults — root-caused to CPU migration,
  and now fixed by pinning.** Long compute runs on this machine intermittently die
  with `SIGSEGV`/`SIGILL`. The crash site *moves* between unrelated libraries
  (`torch.autograd._engine_run_backward`, scipy's L-BFGS-B `approx_derivative`, a
  bare `numpy.matmul`, an **8×8** complex matmul in `vscr_paper_abl.c_tr`, and most
  often `vscr_paper_abl._block_obj_grad`), yet every affected routine returns
  numerically correct results whenever it completes. An 8×8 matmul cannot corrupt
  memory from a logic error, so this was never a bug in the science code.

  Two earlier hypotheses were **wrong**; both are recorded here so they are not
  re-adopted, and both are re-checkable with `diag_native_fault.py`:

  | hypothesis | refuted by |
  |---|---|
  | multi-threaded BLAS/OpenMP races | faults reproduce with every pool pinned to 1 |
  | OpenBLAS `DYNAMIC_ARCH` picking an AVX-512 kernel this CPU lacks | this Arrow Lake part has **no AVX-512 at all** (`grep avx512 /proc/cpuinfo` is empty), and forcing AVX-only `SANDYBRIDGE` *still* raises `SIGILL` — an illegal instruction cannot come from a valid AVX kernel on a CPU that implements AVX |
  | the GEMM itself | 2×10⁷ bare `(2,32)@(32,32)` complex matmuls ran **clean**, while `_block_obj_grad` died within 2×10⁴ calls |

  Fault rate by `OPENBLAS_CORETYPE` (400k calls of `_block_obj_grad`,
  single-threaded, idle box): `HASWELL` → SIGSEGV within 2×10⁴ calls;
  `SANDYBRIDGE` → SIGILL at ~1.5×10⁵; unset → SIGILL at ~2.5×10⁵. The coretype pin
  does **not** help — an earlier "cuts the rate ~5×" claim here was sampling noise
  on a 6-run sample. `OPENBLAS_CORETYPE=HASWELL` is still set, but only for
  cross-machine determinism, and must not be described as a fix.
  **What actually correlates is CPU affinity.** In three-way trials run
  *concurrently* — so ambient load is identical across arms and only affinity
  differs — the pattern held every trial:

  | arm | result |
  |---|---|
  | free to migrate across cores | **FAULT** (SIGSEGV), every trial |
  | pinned to one P-core (cpu0) | clean, 1.5×10⁵–4×10⁵ calls, every trial |
  | pinned to one E-core (cpu16) | clean, 1.5×10⁵ calls, every trial |

  A direct A/B on the same binary, seconds apart, confirms it: `QEC_PIN_CPU=0`
  core-dumped before 2×10⁴ calls, while the default (pinned) completed 1.5×10⁵
  clean. So migration across this hybrid P-/E-core cluster is the trigger — not a
  library, and not one bad core, since both a P-core and an E-core are clean.
  Consistently, `numpy.show_config()` reports OpenBLAS built with `NO_AFFINITY`, so
  nothing else in the stack was holding the thread on a core.

  **The fix**: `ssvr_qec._pin_single_cpu()` runs at import time, so every entry
  point that imports it (`vscr_paper*.py`, `run_selftests.py`, `diag_*.py`) is
  restricted to one logical CPU — a P-core, chosen from the PID so concurrent
  subprocesses spread across cpu0–7 instead of colliding. Opt out with
  `QEC_PIN_CPU=0`. Affinity changes scheduling only, never a computed value. With
  it the 9-test suite runs **9/9 in 45s consuming zero retries** (the same suite
  unpinned needed 3 attempts and 71s).

  **Kept as defence in depth**: `vscr_paper._refine_isolated` and
  `run_selftests.py` still run heavy stages in fresh subprocesses with retries
  (`--in-process` opts out). That is what let an earlier suite run absorb a
  SIGSEGV *and* a SIGILL and still finish 9/9 on attempt 3. Worth keeping even
  with the pin, because the pin is a host-specific workaround, not a guarantee.

  Neither mechanism changes a single reported number: the refinement is
  deterministic given its seed, and `audit_numbers.py` re-verifies all artifacts
  afterwards. CUDA stays disabled (`CUDA_VISIBLE_DEVICES=''`) because of an
  unrelated host driver/library mismatch.

  **On another machine** the pin is harmless (it only narrows affinity) but
  probably unnecessary. Run `diag_native_fault.py` to see whether that host faults
  at all; if it does not, set `QEC_PIN_CPU=0` and let the scheduler place threads.
