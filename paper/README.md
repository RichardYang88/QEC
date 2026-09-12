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
./qenv/bin/python hw_verify_analysis.py# hardware feasibility fig + numbers
./qenv/bin/python make_schematic.py    # Fig. 1
cd paper && ./qenv/bin/python fill_numbers.py && ./qenv/bin/python make_ed.py
```

Artifacts consumed by the manuscript:
`vscr_paper_results.npz`, `vscr_angles_paper_{dep,ad,mixed}.npz`,
`paper_numbers.json`, `hw_feasibility_numbers.json`,
`paper/figures/fig_{schematic,training,branch_cf,sim_benchmark,sim_ler,hw_feasibility}.pdf`.

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
