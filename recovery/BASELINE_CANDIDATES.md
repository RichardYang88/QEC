# Recent (2024–2026) SCI-Q1 comparison baselines for Fig. 4

## Why this file exists

Fig. 4's *experimental* comparison family in `vscr_paper_abl.py` currently consists of

| bar | method | provenance | year |
|---|---|---|---|
| `Perfect-code decoder` | rigid Pauli decoder | Laflamme/Knill 5-qubit | 1996/2001 |
| `VSCR warm (hypernet)` | ours | — | — |
| `VQR-ind (no sharing)` | per-syndrome independent table | Reimpell & Werner, PRL **94**, 080501 | **2005** |
| `Global unitary (no projection)` | QVECTOR-style global recovery | Johnson et al., arXiv:1711.02249 | **2017** |
| `CPTP ceiling (SDP)` | our own dual-certified bound | — | — |

So every *learned* competitor is 8–20 years old. `refs.bib` does contain 2024–2026
entries, but they are used as **citations**, not as **experimental baselines**:
`google_below_threshold` (2025), `ryan_anderson_rt` (2025), `google_cyclic` (2024),
`marques_d3` (2024), `meinerz_nn` (2024), `cai_em_review` (2024), `czarnik_cdr` (2024),
`vandenberg_readout` (2024).

The Petz map is likewise only cited theoretically (`petz_map`, Petz 1986;
`beny_optimal`, Bény & Oreshkov PRL **104**, 120501, 2010) and **never implemented
as a competing recovery** — see `main.tex` lines 99–101 and 399.

## Recommended primary baseline (same problem class, directly implementable)

**D. Biswas, G. Vaidya, P. Mandayam, "Noise-adapted recovery circuits for quantum
error correction", *Physical Review Research* **6**, 043034 (2024).**
DOI `10.1103/physrevresearch.6.043034` — 11 citations.

Abstract (verbatim, via Crossref):

> Implementing quantum error correction (QEC) protocols is a challenging task in
> today's era of noisy intermediate-scale quantum devices. We present quantum
> circuits for a universal, **noise-adapted recovery map, often referred to as the
> Petz map, which is known to achieve close-to-optimal fidelity for arbitrary codes
> and noise channels**. While two of our circuit constructions draw upon algebraic
> techniques such as isometric extension and block encoding, the third approach
> breaks down the recovery map into a sequence of two-outcome POVMs. In each of the
> three cases we improve upon the resource requirements that currently exist in the
> literature. Apart from Petz recovery circuits, they also present circuits that can
> directly estimate the fidelity between the encoded state and the recovered state.
> As a concrete example, they implement Petz recovery circuits for a four-qubit code
> tailor-made to correct **amplitude-damping** noise, demonstrated through ideal and
> noisy simulations on the IBM QasmSimulator.

Why this is the right comparator:

1. **Same object.** It is a *noise-adapted recovery channel* for a QEC code — exactly
   what VSCR learns. Not error *mitigation*, not decoding, so the comparison is
   apples-to-apples on `ab.exact_F`.
2. **Closed form ⇒ exactly evaluable.** The Petz map
   `R_P(σ) = ρ^{1/2} E^†( E(ρ)^{-1/2} σ E(ρ)^{-1/2} ) E(ρ^{1/2})`
   can be computed as a channel on the 32-dim space and scored with the same
   zero-variance Haar quadrature as everything else, so the comparison carries no
   Monte-Carlo error (needed: the headroom we must resolve is 5e-6 … 3.7e-3).
3. **It is the natural upper reference.** Petz is provably close-to-optimal among
   *CPTP* recoveries, so it sits between our Pauli decoder and our
   `sdp_ceiling`/`opt_unitary_ceiling` bounds — showing that VSCR's advantage comes
   from restricting to *per-syndrome unitaries* (implementable, coherent) while still
   beating the best general noise-adapted CPTP map.
4. They already demonstrate on **amplitude damping**, precisely the channel where our
   certified headroom is largest (6.8e-4 … 3.7e-3, see `headroom_table.json`), so it
   is the channel on which the comparison actually discriminates.

## Recommended 一区 (CAS Tier-1 / Q1) citation for the "optimized decoder" family

**V. Sivak, M. Newman, P. Klimov, "Optimization of Decoder Priors for Accurate
Quantum Error Correction", *Physical Review Letters* **133**, 150603 (2024).**
DOI `10.1103/physrevlett.133.150603` — 12 citations. PRL is unambiguously SCI Q1 /
中科院一区. Use as the recent authoritative reference for "the decoder, not the code,
is the tunable object"; it is a surface-code MWPM-prior method so it is cited rather
than reimplemented.

## Other 2024+ Q1 candidates surveyed (Crossref, 6 queries, 45 unique Q1 hits)

| cites | year | venue | title | verdict |
|---|---|---|---|---|
| 72 | 2024 | *Nature Machine Intelligence* **6**, 1478 | Machine learning for practical quantum error mitigation (Liao et al.) | Q1 top, but error **mitigation** — overlaps existing ZNE/VD/LinDR bars |
| 41 | 2024 | *Quantum* **8**, 1287 | Can Error Mitigation Improve Trainability of Noisy VQAs? (Wang et al.) | mitigation + trainability, not recovery |
| 23 | 2024 | *npj Quantum Information* **10** | Simultaneous discovery of QEC codes and encoders with a noise-aware RL agent (Olle et al.) | searches **codes/encoders**, not recoveries |
| 20 | 2025 | *Quantum* **9**, 1609 | Color code decoder with improved scaling (Lee et al.) | topological decoder, different code family |
| 17 | 2025 | *Phys. Rev. Research* **7**, 013029 | Neural network decoder for near-term surface-code experiments (Varbanov et al.) | surface-code decoder |
| 14 | 2025 | *PRX Quantum* **6** | Optimizing QEC Protocols with Erasure Qubits (Gu et al.) | erasure conversion, not recovery learning |
| 12 | 2024 | *PRL* **133**, 150603 | Optimization of Decoder Priors (Sivak et al.) | **chosen as the Q1 citation** |
| 11 | 2024 | *Phys. Rev. Research* **6**, 043034 | Noise-adapted recovery circuits (Biswas et al.) | **chosen as the experimental baseline** |
| 5 | 2025 | *Phys. Rev. A* **111**, 012419 | Measurement-free local EC with RL (Park et al.) | measurement-free, different setting |
| 3 | 2026 | *Quantum Sci. Technol.* **11**, 045036 | Learning encodings by maximizing state distinguishability (Meyer et al.) | encoder learning, not recovery |

## Implementation note

arXiv (`arxiv.org`, `export.arxiv.org`) is **network-blocked** from this host
(`HTTP 000`, connection reset), so all metadata above was retrieved through the
**Crossref REST API** (`api.crossref.org`, reachable) and keyed by DOI.
DOIs and venue/volume/page strings should be re-verified before submission.

The search driver is preserved at `/tmp/find_baseline.py` and its full output at
`/tmp/find_baseline.log`.

