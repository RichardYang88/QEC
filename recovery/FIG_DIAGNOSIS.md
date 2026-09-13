# Diagnosis of the three reported figure problems

All numbers below are read from the **surviving** `paper_numbers.json` (the file that
actually generated `paper/figures/fig_sim_benchmark.pdf` = Fig. 3 and
`fig_ablation.pdf` = Fig. 4), so the diagnosis is from data, not from the image.

---

## 1. Fig. 3, last subplot: "ZNE is best" — CONFIRMED BUG (unphysical estimator)

The last subplot is the **coherent** channel (`coh_full_curves`). Its ZNE row is:

```
p:      0.01     0.02     0.03     0.04     0.05     0.06     0.08     0.10     0.12     0.15
ZNE:  1.00000  1.00000  1.00000  1.00007  1.00061  1.00249  1.01176  1.03380  1.07303  1.13044
```

**Fidelity > 1 is impossible.** ZNE "wins" the last subplot purely by *overshooting*
the physical range: Richardson extrapolation with the raw (unprojected) estimator is
unbounded, and on a purely coherent channel the extrapolated operator is not a valid
density matrix, so the estimator runs away past 1.0 and reaches **1.13** at p = 0.15.

The same subplot also has a second dead comparison:

```
Raw : 0.999969 0.999875 0.999500 0.998002 0.993892 0.987573 0.972241 0.951150 0.924660 0.893218
VD  : 0.999969 0.999875 0.999500 0.998002 0.993892 0.987573 0.972241 0.951150 0.924660 0.893218
```

`VD (oracle)` is **bit-identical to `Raw`** at all ten noise strengths. Virtual
distillation suppresses *stochastic* noise; on a purely coherent rotation it has
nothing to distil, so it is a no-op. Physically correct, but plotted as an
indistinguishable duplicate line.

**Fix.** The oracle module already contains the two functions needed:
* `zne_phys_fidelity(psi_enc, p, noise)` — Richardson ZNE (scales 1,2,3) with the
  extrapolated operator **projected back onto the physical set**, i.e. clamped;
* `zne_unphysical_overshoot(p_values, noise, n_test=60, seed=1245)` — an explicit
  audit returning `{p: max F}` for the raw estimator.

So: plot the **physical (projected) ZNE** as the competitor, cap the y-axis at 1.0,
and report the raw overshoot as an audit annotation/table entry rather than letting a
non-physical 1.13 masquerade as the best method. Give VD its own linestyle/marker and
state in the caption that VD == Raw on coherent by construction.

---

## 2. Fig. 3: several baselines coincide exactly — CONFIRMED, two distinct causes

```
dep_full_curves:
  Perfect-code decoder : 0.999835 0.999348 0.997450 0.990251 0.972136 0.946995 0.894240 0.833900 0.771605 0.711680
  VSCR warm (ours)     : 0.999835 0.999348 0.997450 0.990251 0.972136 0.946995 0.894240 0.833900 0.771605 0.711680
```

Identical on **depolarizing, amplitude_damping and mixed** — all ten points, every
channel. This is *not* plotting sloppiness: warm training genuinely returns the
decoder.

**Root cause (structural, established earlier this session).** `PHI_DEC` is an **exact
stationary point** of the branch Rayleigh-quotient objective: for each syndrome the
decoder's 2x2 block `G_dec` is an eigenvector of the 4x4 Hermitian form `Q_s`, so
`grad cf_s = 0` exactly. First-order (Adam) training started at `PHI_DEC` therefore has
zero gradient and cannot move — small symmetry-breaking perturbations do not fix it
because the saddle's stable manifold is wide. Two consequences:

* **Fix A (regression guard):** keep the untrained decoder as a *retained candidate*
  in `train_warm_best`, so `F_warm >= F_dec` holds by construction.
* **Fix B (escape the saddle):** per-syndrome multi-start refinement
  (`refine_per_syndrome`). Because the p-averaged exact objective is **separable
  across the 16 syndromes** (branch weights cancel in the conditional fidelity), each
  block can be optimised independently from symmetry-broken starts and reaches the
  certified unitary optimum.

Note that on depolarizing/mixed this coincidence is *unavoidable and correct*: the
headroom table shows certified headroom ~1e-16 there, i.e. the decoder is already
optimal. Only amplitude_damping and coherent have real headroom to capture.

---

## 3. Fig. 4(a): four of five bars read 0.947 — NOT a calculation error; the *figure* is wrong

`bar_dep_p010` (depolarizing, p = 0.10) verbatim:

```
Perfect-code decoder        0.9469945679012346
VSCR warm (hypernet)        0.9469945678939504
VQR-ind (no sharing)        0.9469945678989581
Global unitary (no proj.)   0.5908069135802474
CPTP ceiling (SDP)          0.9469945679014865
```

Four bars agree to **~1e-10**. That is a genuine, *provable* result, not a bug:

```
opt_unitary_ceiling('depolarizing', p)  ->  F_dec == F_unit
    p=0.05  headroom = 1.1102e-15
    p=0.06  headroom = 7.7716e-16
    p=0.10  headroom = 8.8818e-16
    p=0.15  headroom = 7.7716e-16
opt_unitary_ceiling('mixed', p)         ->  headroom <= 1.22e-15  (all p)
```

and the SDP block reports `max_branch_gap = 5.10e-13`, `F_gap_bar = 4.14e-13`: on
depolarizing the Pauli decoder is **exactly CPTP-optimal**, so every method that
reaches the optimum must return the same number. The four-way tie *is* the statement of
the theorem, and the bar chart is simply the wrong way to show it — a linear-scale bar
chart of quantities differing by 1e-10 has, by construction, zero visual discrimination.

**Fix (two parts).**

1. **Move the discriminating panel to a channel that has headroom.** Certified
   headroom (`headroom_table.json`):

   | channel | p=0.05 | p=0.06 | p=0.10 | p=0.15 |
   |---|---|---|---|---|
   | depolarizing | 1.1e-15 | 7.8e-16 | 8.9e-16 | 7.8e-16 |
   | **amplitude_damping** | **4.82e-4** | **6.84e-4** | **1.78e-3** | **3.67e-3** |
   | mixed | 1.2e-15 | 8.9e-16 | 7.8e-16 | 4.4e-16 |
   | **coherent** | **2.60e-6** | **5.39e-6** | **4.14e-5** | **2.08e-4** |

   On amplitude damping the bars genuinely separate, and after Fix B the warm model
   must land measurably above the decoder — that is the figure that demonstrates the
   method.

2. **Replace the tied linear bars with a log-scale "gap to the certified ceiling".**
   Plot `F_ceiling - F_method` on a log axis. The depolarizing tie then becomes a
   *quantitative* claim (decoder optimal to 1e-13; VQR-ind to 1e-10) instead of an
   uninformative visual tie, and the amplitude-damping/coherent separations become
   visible on the same axis.

---

## Summary of required code changes

| # | change | file |
|---|---|---|
| A | retain the untrained decoder as a candidate so `F_warm >= F_dec` | `vscr_paper.py` (`train_warm_best`) |
| B | per-syndrome symmetry-broken refinement to capture certified headroom | `vscr_paper_abl.py` (`refine_per_syndrome`, `pavg_bundle`, `pavg_F`) + `vscr_paper.py` (`VSCRWarm.phi_extra`, `set_phi`) |
| C | use `zne_phys_fidelity` (projected) in the coherent panel; y-cap 1.0; report `zne_unphysical_overshoot` as an audit | `vscr_paper.py` (`evaluate_paper`, `plot_benchmark`) |
| D | Fig. 4(a) -> amplitude-damping panel + log-scale gap-to-ceiling panel | `vscr_paper_abl.py` (`plot_ablation`, `run_ablations`) |
| E | add the 2024 noise-adapted (Petz) recovery baseline | `vscr_paper_abl.py` + `paper/refs.bib` |
| F | restore `run_selftests.py`, `diag_check.py`, `diag_optunit.py` | new files |

