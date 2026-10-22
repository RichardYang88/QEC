# Two corrections found while clearing the last gate before the paid A/B

Recorded 2026-10-20. Neither is a change to any estimator; both are changes to
what the repository ASSERTS, which is the category this evidence directory exists
for.

## 1. A retracted finding was still asserted in the present tense

`check_cal9_transferability`'s docstring (hw_mixedness.py, ~4805) said that on the
2026-10-12 warmcold run the transferability bound "fires": cal9 measured contrast
-0.156 at logical q3, bounding P(read 0) to [0.432, 0.589], while the science
circuits read 0.708 and 0.665 there, so cal9 characterised a DIFFERENT physical
channel and could not validate their estimator.

That is retracted, and was already retracted elsewhere in the same file
(stage_circuits' CAL-FIRST comment, ~985): the comparisons were made at the wrong
key position -- the classical bit the MEAS map assigns rather than the analysed
position -- so they compared cal9's channel against a different science channel
and reported the mismatch as vendor transpiler re-routing.

Verified against the frozen 2026-10-12 data, not taken on trust. hw_mixedness.json
records, for that same run:

    warmcold.cal9_transferability = {available: true, tol_sigma: 5.0,
        n_comparisons: 36, n_violations: 0, violations: [],
        cal9_transferable: true}
    warmcold.verdict.cal9_transferable = true

selftest (m3) locks both directions: a marginal of 0.80 against that channel's
bound is flagged, 0.50 and 0.97 are not.

Why this was worth fixing now rather than as tidiness: it contradicted the
artifact in the same repository, and read as a reason to DISTRUST the WK_C180_2
screen that had just cost 2 hours of queue. If cal9 could not see the science
circuits, that screen would not have cleared the block the A/B uses, and the whole
gate-1 step would have been decoration. It is not decoration. The dead channel
itself was never in doubt -- it is measured, not inferred -- only the claim that
cal9 could not see the science circuits.

## 2. The regression suite intermittently SIGSEGVs, and that is a host fault

A full `run_selftests.py` run failed 2/16 with `[refine]` and `[storage]` both
dying as "native fault": the harness ran each in a fresh interpreter 4 times and
all 4 attempts died, exit=-11 (SIGSEGV) and exit=-4 (SIGILL). Crash sites moved
between attempts -- scipy `_dense_difference`, `vscr_paper_abl._branch_qform:517`,
`_adam_numpy:993` -- which is memory corruption, not a deterministic bug.

This is the fault already documented at run_selftests.py:88-101: this host
intermittently SIGSEGV/SIGILLs in tiny complex128 kernels when a process is free
to migrate across its hybrid P-/E-core cluster. Two mitigations are already in
place there (thread env pinned to 1 via setdefault, CPU affinity via importing
ssvr_qec) and neither fully removes it.

Established, rather than assumed:

  * `vscr_paper_abl.py` is untouched (git diff empty; last modified 2026-09-16,
    committed 4cb02d2 2026-09-17), and numpy/scipy were installed 2026-08-30.
    Nothing in the tree changed, so this is not a regression from recent edits.
  * Not OOM: 17.5 GB available at the time.
  * Not scipy in general: a standalone L-BFGS-B minimise succeeds at 1 and 4
    threads.
  * A small workload (`_selftest_refine(n_start=1, steps=5)`) does not crash at
    all; it finishes selftests 1/1b/1c/2/2b and raises a clean Python
    AssertionError. Only the full workload (n_start=3, steps=900) hits it.
  * Re-running the identical plain command passed, and a subsequent FULL suite
    passed 16/16 with 0 crashes in 211 s. So the failure is intermittent, which
    is the documented behaviour.

Note the thread pinning uses `setdefault`, and the shell here has none of those
variables preset, so the failing run was ALREADY single-threaded. Pinning threads
is therefore not sufficient on its own; the affinity pin is the load-bearing one.

## Does any of this touch the paid A/B? No, and that is checked, not asserted

The warm-vs-cold A/B is analysed by hw_mixedness.py, which:

  * loads scipy but NOT `scipy.optimize`, and does NOT load `vscr_paper_abl`
    (checked against sys.modules after import);
  * has no occurrence of `minimize`, `scipy`, `opt_`, `refine` or `torch` in
    `analyze_warmcold`'s source;
  * uses PRE-TRAINED angles from vscr_angles_paper_dep.npz, so it never calls the
    unitary-ceiling refinement that crashes.

`[hwmixedness] passed` in the 16/16 run above, which was executed AFTER the
docstring edit in section 1, so that edit is verified safe too.

The crashing code path belongs to the VSCR paper ablation, a different pipeline.
