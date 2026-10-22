# WK_C180_2 paid warm/cold DETECTION A/B: FAILED, and why that is the informative outcome

Run: 2026-10-21T01:04:03Z submitted -> 3 submissions FINISHED (3920 s / 3935 s / 4026 s),
11883 s wall, 6/6 circuits, 120000 shots billed.
Log: `hw_evidence/20261021T010403Z_WK_C180_2_warmcold.log`
Dump: `qcloud_raw_warmcold_WK_C180_2_20261021T010404Z.json` (+ `.meta.json` sidecar)
Record: `hw_mixedness.json` -> `warmcold` (previous 2026-10-12 record preserved under
`hardware_run_archive['warmcold@qcloud_raw_warmcold.json']`, not overwritten)

Bought as: `CHIP=WK_C180_2 BUDGET=389.136 MSPS=1.864 SHOTS=20000
BLOCK=39,48,49,57,58,66,67,75,85 CAL_FIRST=1`.  `MSPS` was set to the most
conservative measured upper bound available at submission time, not the optimistic
one, so the budget gate was passed under the least favourable honest assumption.

## 1. Verdict: FAILED, not INVALID -- and that distinction is the point

`ab_test_passed = signal_detected AND signal_direction_correct AND null_consistent_with_zero`

| pre-registered criterion | value | pass |
|---|---|---|
| `signal_detected` (\|z\| >= 3.0) | z = **-19.06** | yes |
| `signal_direction_correct` (gap > 0) | gap = **-0.04685** | **NO** |
| `null_consistent_with_zero` (\|z\| < 3.0) | null z = **-33.91**, gap -0.08030 | **NO** |

Two of three failed.  `ab_test_result = FAILED`.

It is FAILED and **not INVALID** because every apparatus gate cleared:
`primary_estimator_valid=true`, `cal9_transferable=true` (36 comparisons, 0
violations), `vendor_rerouting_detected=false`, `layout_derived=true`,
`layout_agrees_with_detect_layout=true`, cal9 screen 0 broken channels
(0 data, 0 ancilla), mean readout fidelity 0.9996.

`_warmcold_verdict` draws this line deliberately: INVALID would mean a
compilation/apparatus fault that cannot be read as physics.  FAILED means the
apparatus was characterised, the test was interpretable, and the pre-registered
prediction did not survive it.  This run is the latter.

## 2. The `CAL_FIRST=1` choice paid for itself

The screen that gated this purchase completed 2026-10-19T16:42Z; the run was
submitted 32.4 h later, and the q3 fault is documented as a property of a given
day's calibration.  Submission 1 was therefore the two cal9 circuits at 20000
shots -- a **same-day re-screen at 20x the shot count of the gate that authorised
the spend**, at no extra queue time.  It passed: `data3 logical q3 e01=0.0001
e10=0.0003 fid=0.9998 contrast=+0.9995 ok`, all 9 channels ok, 0 broken.

So the block was healthy on 2026-10-21 as well as 2026-10-19, and the failure
below is **not** a dead channel.  Had submission 1 shown a dead q3, 1/3 of the
spend would have stopped the run before any science circuit was bought.

## 3. Root cause, quantified: the two arms do not share one `w`

`warmcold_power` models a device that transmits a fraction `w` of the data
register as `p_warm = w + (1-w)/32` and `p_cold = 0.5w + (1-w)/32`, with a single
`w` for both arms and a uniform floor of `1/32 = 0.03125` from the randomised
weight.  Inverting that model on this run's four circuits:

| branch | P(data=00000) | implied w | vs uniform floor |
|---|---|---|---|
| signal\|warm | 0.0418 | **+0.0109** | 1.34x |
| signal\|cold | 0.0886 | **+0.1224** | 2.84x |
| null\|warm   | 0.0213 | **-0.0102** | **0.68x** |
| null\|cold   | 0.1016 | **+0.1502** | 3.25x |

Warm arm mean w = **+0.0003**.  Cold arm mean w = **+0.1363**.  Not one shared
`w` -- roughly a factor of 10 apart, and the warm arm transmits essentially
nothing.

The decisive single number is `null|warm = 0.0213`, which is **below** the
0.03125 uniform floor.  A purely randomising/depolarising register cannot fall
below the floor it randomises toward, so the warm arm's noise is not
depolarising: it is **biased toward |1>**, i.e. residual excitation from the warm
initialisation that does not relax away over 206 instructions.  That bias is
present in both branches (warm below cold in signal and in null alike), and it is
what moved the null by -0.0803.

The null bias is **1.71x larger** than the signal gap it was supposed to
control for.  A gap smaller than the contamination in its own control carries no
information about the hypothesis, which is exactly why the pre-registered rule
treats a dirty null as disqualifying rather than correctable.

## 4. Power was NOT the limitation, so more shots cannot fix this

The forward projection at the 2026-09 constant predicted `sigma_z = 15.7` at
20000 shots/circuit.  The run measured `|signal_z| = 19.1` and `|null_z| = 33.9`.
The design had **ample** power -- it detected its effect at 19 sigma and its
contamination at 34 sigma.

This is the operationally important consequence: `S* = 20000` was the right
purchase, and it is also the top rung of the ladder, so there is no more
precision to buy.  The failure is **systematic, not statistical**.  Spending the
remaining balance on more shots of the same design would shrink error bars on a
biased quantity and change nothing.

## 5. What this run did establish (it is not a null result)

The 2026-10-12 run on this same chip and block produced
`WARMCOLD_P_DATA_ZERO_2026_10_12 = 0.0407`, but that run was INVALID as an A/B
(one data channel dead at contrast -0.156), and the constant's own comment says
its `P(data=00000)` was "the only measurement that exists of how much of the data
register survives 206 instructions WITH vendor options on".

This run supersedes it, and reproduces it:

| | 2026-10-12 | 2026-10-21 |
|---|---|---|
| chip / block | WK_C180_2 / [39,48,49,57,58,66,67,75,85] | **identical** |
| shots | 3000 | 19994 |
| A/B validity | INVALID (dead data channel) | **VALID** (0 broken) |
| signal\|warm P(data=00000) | 0.0407 +- 0.0036 | 0.0418 +- 0.0014 |

Difference `+0.0011 +- 0.0039` = **0.28 sigma**: statistically indistinguishable.
The warm arm's collapse to ~0.041 reproduces across 9 days on the same chip and
block at 6.7x the shot count, and this is now the first VALID measurement of
206-instruction data-register survival with vendor options on.

Also newly established: the shallow cal9 circuits give data all-zero = **0.9996**
while the 206-instruction science circuits give **0.021-0.102**.  The readout is
essentially perfect; the depth is what destroys the state.  A cal9 screen is a
readout/channel screen and cannot certify coherence at depth -- this run is the
direct demonstration of the gap between the two.

Cross-check from the frozen 2026-10-12 dump (`estimator_family`): it too had
`max_abs_null_z = 16.96` (member `P_bit0_zero`) and
`members_both_clean_and_significant: []`.  **The null contamination is not new to
this run** -- no estimator member in either run has been simultaneously
null-clean and signal-significant.

## 6. Recorded but explicitly NOT a result: the post-hoc null-corrected gap

`signal_gap - null_gap = -0.04685 - (-0.08030) = +0.03345`, which has the
pre-registered sign and equals **6.7%** of the ideal `+0.500000`.

This is recorded as a diagnostic **only**.  It is not the pre-registered test, it
was constructed after seeing that the null moved, and the code writing
`cal9_transferability` / `posthoc_healthy_subset` carries a comment stating that
those alternates are recorded "precisely so the pre-registered one cannot be
quietly replaced by whichever of the three is most flattering".  Substituting a
post-hoc null subtraction for a failed null check is that move, and 6.7% of the
ideal would not rescue it anyway.  `ab_test_passed` remains **false**.

## 7. Billing settled: third and most precise rate sample

```
dashboard_before = 389.136 s   (2026-10-21T01:04Z, at submission)
dashboard_after  = 352.062 s   (2026-10-21T14:36Z, all 3 FINISHED)
delta            =  37.074 s   over 120000 shots  (6/6 circuits, denominator CERTAIN)
rate             =   0.3090 ms/shot
```

Unlike the earlier samples the denominator here is not ambiguous -- all 6 circuits
are confirmed FINISHED -- so this is the tightest of the three measurements.  It
is below even the most optimistic earlier bound (0.373) and falsifies the
pessimistic `MS_PER_SHOT = 14` by **45.3x**.

The A/B actually cost **37.074 s**; the 1.864 ms/shot upper bound had reserved
223.7 s for it.  Remaining balance **352.062 s**, which at the measured rate buys
~1.14 million shots.  **Budget is no longer a constraint on any future design**,
which leaves the systematic in section 3 as the only thing limiting this
experiment.

Measured-rate history: 14 (assumed, falsified) -> 1.864 (worst-case upper bound)
-> 0.373-0.621 (probe attribution range) -> **0.3090** (certain denominator).


## 8. Two code defects found by this run, and one I introduced while fixing them

**(a) `--mode analyze` could not read a `--cal-first` dump.**  The analyze branch
rebuilt the expected circuit order with `stage_circuits(stage, phi_warm,
phi_cold)`, omitting `cal_first`, so for a cal-first submission the rebuild put
`signal|warm` at position 0 while the sidecar correctly recorded `cal9/zero`
there.  The sidecar comparison then raised `AssertionError` and re-analysis was
impossible -- on the very run that most needed re-analysis.  Worse, the recovery
command `launch_mixedness.sh` prints at submission time (`:319-320`) omitted the
flag too, so following the tool's own instructions could not work.

Fixed by passing `cal_first=a.cal_first` to the rebuild.  This does **not** weaken
the check: a genuinely reordered dump still fails it, and a cal-first dump
analysed without the flag still fails it -- now with a message that names the fix
(the `AssertionError` detects the `{'cal9','warmcold'}` kind swap and says so).
The launcher's printed command now includes `--cal-first` exactly when the run was
submitted with it.

Verified end-to-end: `--mode analyze --stage warmcold --cal-first --dump-raw
qcloud_raw_warmcold_WK_C180_2_20261021T010404Z.json` -> exit 0, `meta VERIFIED
against the submission-time sidecar (2026-10-21T01:04:04Z, 6 circuits, block
[39, 48, 49, 57, 58, 66, 67, 75, 85])`, and `verdict`, `pairs`, `per_circuit`,
`cal9`, `cal9_transferability`, `layout`, `posthoc_healthy_subset` all
**byte-identical** to the paid run's record.  Re-analysis is deterministic and
reproduces the verdict.

The failed first attempt did **not** corrupt the evidence: the artifact write only
happens on success, and `hw_mixedness.json` was verified unchanged against a
pre-attempt backup.

**(b) `power_prediction.measured_w` was a misnomer that nearly cancels the real
result.**  That key is a FORWARD projection at `warmcold_power`'s default
`w = WARMCOLD_W_MEASURED_2026_09 = +0.09357` -- the frozen 2026-09 constant, not
anything this run measured.  This run's own estimate is
`verdict.signal_transmission_w = -0.09370`: the same magnitude to 3 decimal
places with the **opposite sign**.  Two near-cancelling numbers in one artifact
under names that both read as "measured" is precisely how a manuscript ends up
quoting a historical constant as its result.

Confirmed no code consumes `power_prediction.measured_w` (the only readers,
`run_selftests.py:294,303`, use the differently-named `power_at_measured_w`), so a
strictly additive `provenance` string was added rather than renaming the key -- a
rename could break a manuscript that cites the JSON path.  The note states that it
is a projection, names the constant and its value, and points at
`verdict.signal_transmission_w` for the measurement.

**(c) Introduced and fixed:** the launcher's new `--cal-first` hint was first
written as `[ -n "$CF_HINT" ] && echo ...` as the script's LAST command.  A
short-circuited `&&` would have made `launch_mixedness.sh` exit **1** whenever
`CAL_FIRST=0`, so any caller checking `$?` would read a successful submission as a
failure.  Replaced with an `if` block plus an explicit `exit 0`.  Both paths
verified: `CAL_FIRST=1` prints the hint and exits 0, `CAL_FIRST=0` omits it and
exits 0.

## 9. What this means for the next step

The hypothesis was not merely undetected -- the pre-registered direction was
falsified at 19 sigma, and the control that was supposed to be blind to the
effect moved at 34 sigma in the same direction.  Combined with section 3, the
reading is that on this hardware the warm/cold **contrast itself** is dominated by
residual excitation in the warm arm that survives 206 instructions, and the null
branch's cancellation -- which holds in simulation at `p = 0` -- does not hold on
a device whose warm arm sits below the uniform floor.

Consequences, in order of how much they constrain:

1. **Do not re-buy this design.**  Power is ample (section 4) and the balance is
   ample (section 7); neither is the binding constraint.  More shots or another
   chip would reproduce the same systematic -- and note that the warm-arm figure
   already reproduced across 9 days on the same chip.
2. **The fix has to be in the design, not the statistics.**  Either the warm arm's
   residual excitation must be removed/measured and subtracted by a pre-registered
   rule declared *before* the next purchase, or the null branch must be rebuilt so
   that it genuinely cancels a biased-toward-|1> channel rather than a
   depolarising one.  A post-hoc subtraction is section 6 and is not admissible.
3. **The screen's scope should be stated wherever it is cited.**  cal9 cleared the
   readout at 0.9996 and cleared the channels as transferable, and the A/B still
   failed.  Gate 1 certifies channels, not coherence at depth; the 0.9996 -> ~0.04
   collapse across 9/18 vs 206 instructions is the evidence.
4. The `WK_C180` fallback (`w = +0.0118`, threshold ~0.0117, `z ~ 3.01`) is **not**
   a remedy for this failure, and its margin was already marginal.  Be precise
   about what is and is not known here: the 0.28-sigma reproduction in section 5
   is on the **same** chip (WK_C180_2) and the same block, 9 days apart, so it
   establishes day-to-day stability, **not** chip-independence.  There is no
   measurement of warm-arm survival under vendor options on `WK_C180` at all, so
   whether the arm asymmetry is chip-specific is currently **unknown** -- it is
   neither demonstrated nor excluded.  Moving chips would be an unpriced gamble on
   an untested hypothesis about the systematic, not a fallback with a known
   margin.


## 10. Why the run was bought as DETECTION when the realised w landed in BOUND territory

This is the part that matters for the *next* decision, so it is stated with the
reconciliation made explicitly -- an earlier reading of these two records as being
in conflict about `w` was **wrong**, and is recorded here as wrong.

**The two verdicts do not disagree about the physics.** Both used the same probe
point estimate `w = 0.016897` [95% CI 0.004900, 0.032515]. They disagree about
the **billing rate**, and therefore about how many shots the budget could buy:

| record | ms/shot | budget | rungs fitting | S* | sigma_z @ w_point | verdict |
|---|---|---|---|---|---|---|
| `wcprobe` (as analysed) | **14.0** | 377.0 | 6 of 13 | 4000 | 1.854 | BOUND EXPERIMENT (exit 5) |
| `wcpick` (20261020T140501Z) | **1.864** | 389.136 | 13 of 13 | 20000 | 4.147 | DETECTION (exit 0) |

At the pessimistic 14 ms/shot the budget only reaches S\*=4000, worth 1.85 sigma,
so the probe correctly called itself a bounding experiment.  At the corrected
1.864 ms/shot the same budget reaches the top rung S\*=20000, worth 4.15 sigma.
**The DETECTION classification was purchased entirely by falsifying the 14 ms/shot
rate** -- which section 7 has now confirmed a third time, at 0.3090 ms/shot.  That
part of the gate chain worked exactly as intended.

**What was NOT robust is the dependence on the point estimate.**
`pick_wc_chip`'s rule (`hw_mixedness.py:3593-3595`) selects on
`max(usable, key=w['point'])` and compares that point value to
`min_w_for_detection(S*)`; `wc_probe_power.detection_affordable` likewise tests
`sigma_z_at_w_point >= WC_PROBE_MIN_SIGMA`.  Neither keys on the lower confidence
bound, although `wc_probe_power` computes and stores `sigma_z_at_w_lo95` for every
rung.  At `w_lo95 = 0.0049`:

| shots | z @ w_point | z @ w_lo95 |
|---|---|---|
| 12000 | 3.212 | 1.036 |
| 15000 | 3.591 | 1.158 |
| **20000 (S\*)** | **4.147** | **1.337** |

**No rung of the ladder reaches 3 sigma at the probe's own 95% lower bound**, and
the record itself carries `w_bound95_at_s_star = 0.007987`, which is *below*
`w_min_for_detection = 0.011711`.  The probe CI spans a factor of 6.6 across that
threshold.

The realised warm-arm survival was `w = +0.0109` (signal branch) and `-0.0102`
(null branch) -- both below 0.011711, on the lo95 side of the CI, and consistent
with the only prior with-options figure on record (`w_measured_2026_10_12 =
0.0098`, worth 1.0 sigma per selftest (l)).  So the A/B was in fact a **BOUND**
regime experiment that had been labelled DETECTION on the strength of a point
estimate 55% higher than what the device delivered.

**A minor provenance wart found on the way:** `wc_probe_power` sets
`ms_per_shot_source = 'dashboard-measured'` purely on the test
`ms_per_shot is not None`.  The probe record above therefore labels its `14.0`
as "dashboard-measured" when 14.0 is the pessimistic constant `MS_PER_SHOT`, not a
dashboard reading.  The value is what drove that record's BOUND verdict, so a
reader trusting the label would mis-attribute the reason.

**No rule was changed here, deliberately.** `pick_wc_chip`'s point-estimate rule
is pre-registered, and selftest (p) asserts that it must follow the declared rule
even when another chip measured a higher w -- "which is what a rule declared
before the data was bought has to do".  Rewriting the selection rule to use
`lo95` *after* seeing that the point estimate failed would be the same
outcome-driven move section 6 declines, applied to the decision procedure instead
of the estimator.  If the rule is to require a lower confidence bound rather than
a point estimate, that change must be declared and selftested **before** the next
purchase, and the probe must then be bought with enough shots for its CI to
actually straddle-or-clear the threshold -- at 987 returned shots the CI was a
factor 6.6 wide, which is too weak an input to carry a DETECTION label either
way.

