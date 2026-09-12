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
./qenv/bin/python vscr_paper.py        # warm-start training + benchmarks + figs
./qenv/bin/python vscr_paper_coh.py    # coherent-channel supplement (Fig. 3c)
./qenv/bin/python vscr_paper_abl.py    # same-family baselines + SDP ceiling + ablations
./qenv/bin/python hw_verify_analysis.py# hardware feasibility fig + numbers
./qenv/bin/python make_schematic.py    # Fig. 1
cd paper && ./qenv/bin/python fill_numbers.py && ./qenv/bin/python make_ed.py
```

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
- Fig. 4 hardware data used the v1 angle snapshot; branch structure,
  extraction and post-selection protocol are identical to the paper model.
- The identity branch ($s=0$) failure on WK_C180 (2.4% expected bitstring) is
  reported, not hidden; it motivates the readout-mitigation design of the
  full benchmark.
- Same-family ablations (`vscr_paper_abl.py`, ED Tables 5–7): a per-syndrome
  independent parameter table (VQR-ind) matches the warm hypernetwork to
  ≤1.1e-9 at n=5 and is **stronger** in the cold-start low-data regime —
  the hypernetwork's justification is asymptotic ($2^{n-k}$ branch scaling)
  and deployment (single model, exact per-branch decoder warm-start
  construction), not superior fidelity at this code size. No claim of the
  paper relies on the hypernetwork.
- The SDP ceiling shows the Pauli decoder is optimal among ALL CPTP
  recoveries (given the projective readout) on the Pauli-type channels
  (depolarizing: ≤4e-13; mixed: +9e-7). Amplitude damping (+7.2e-4) and
  coherent over-rotation (+1e-4 … +3.1e-3, ceiling exactly 1.0 at ε=0.30)
  leave small non-Pauli headroom concentrated in low-weight branches that
  warm-start gradient training does not exploit (decoder point stationary;
  Pauli-class search finds discrete +0.65-0.68 cf gains on weight-3e-5
  branches). VSCR's claim is *ceiling saturation without labels* on
  Pauli-type noise, never "beating the decoder".
- QVECTOR-style global unitary recovery (no projection) partially inverts
  purely coherent errors (F̄=0.9962 at ε=0.10) — the one tested regime where
  a measurement-free variational recovery is competitive; it collapses to
  raw-state level on stochastic channels (0.52–0.77 vs 0.925–0.985).
- Reproducibility note: `abl_fix_gaps.py` repaired the E3 branch-gap column
  after a `register_buffer` aliasing bug (zeroing a cold-start model's
  `phi_dec` buffer mutated the global `PHI_DEC`); root cause fixed in
  `vscr_paper.py` (clone on register). All other run outputs were
  unaffected (they consume `m.C_SYNDS` or pre-corruption artifacts).
