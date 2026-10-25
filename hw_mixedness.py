"""
hw_mixedness.py -- MEASURE, on a real superconducting QPU, the mixedness of the
logical state the device actually delivers, and test on hardware the one half of
the mixed-input boundary that hardware can resolve.

Why this file exists
--------------------
`mixed_input_boundary.device_prep` prices the quadratic training loss's over-read
at a device Bloch radius r = 0.9375 with 39% of the encoded weight outside the
code space.  Both numbers are DERIVED, not measured: they come from feeding the
mean per-instruction error that `hw_error_budget.py` implies from the WK_C180
feasibility counts through the compiled 36-gate encoder as an isotropic
depolarizing channel.  The manuscript says so, and ED Table 10d is labelled
"model-implied" for exactly that reason.  This script replaces the label with a
measurement, on the device's own noise and with no injected error.

What is measured (plan: QCLOUD_SETUP.md section 7)
--------------------------------------------------
M1  device mixedness.  A 5-qubit tomography circuit
        prep(psi) on q4 -> U_enc (36) -> U_enc^dag (36) -> R_B on q4 -> MEAS q0..q4
    is run for psi in {0, 1, +, -} and B in {X, Y, Z}.  U_enc^dag maps the code
    space back onto |0000> (x) q4, so "q0..q3 all zero" IS the code-space
    projector V V^dag that device_prep applies analytically, and q4 carries the
    logical state.  Hence, from the same counts:
        w   = P(q0..q3 = 0000)                      code-space weight
        <s_B>_cs = P(q4=0|cs) - P(q4=1|cs)          conditioned Bloch component
        r   = |(<s_X>, <s_Y>, <s_Z>)|               Bloch radius, purity (1+r^2)/2
    Every one of these is reported raw AND after readout mitigation with an
    assignment matrix measured by this run itself (M4), because the vendor
    calibration API returns 401 for this account and the manuscript records that
    refusal verbatim.

M2  the LINEAR score decomposes exactly.  rho_in = I/2 has two different pure
    ensembles, {|0>,|1>} and {|+>,|->}.  The unconditional logical <Z_L> is a
    LINEAR functional of rho_in, so
        L_Z = (<Z_L>_|0> + <Z_L>_|1>)/2 = Tr[Z_L M(I/2)] = L_X
    holds for ANY channel M -- exactly, not approximately.  The individual terms
    span ~1.7 while the two averages must agree, so this is a hardware statement
    with real content, and it is the half of `mixed_input_boundary.selftests`
    item (d) that licences the existing population / <Z_L> benchmarks.  The
    conditional (post-selected) version is NOT linear and is reported only as a
    leakage-asymmetry diagnostic, never as the M2 claim.

M3  the isotropic-depolarizing surrogate is tested at the hardware's own depth.
    `model_mixedness` re-implements device_prep's model on an ARBITRARY compiled
    gate list (independently, as a 32x32 density matrix, so the audit can
    re-derive it).  One parameter, p_instr, is fitted to the MEASURED code-space
    weight; the model then PREDICTS r with nothing left to fit.  Agreement inside
    2 sigma supports the surrogate; disagreement is reported as a measurement of
    where it fails, which is also a result and is consistent with the manuscript
    already calling r an order of magnitude.

M4  the 5-bit readout assignment matrix, measured by this run: e01 from |00000>,
    e10 from X^(x5).  Bracketed by a repeat of the |00000> circuit so the drift
    is a recorded systematic rather than an assumption.

M5  depth scan.  The same circuit with rounds=2 (enc,dec,enc,dec = 150
    instructions, 72 CNOT) separates a PER-GATE loss law (one p_instr explains
    both depths) from a PER-TIME one (it does not).  Two of its four circuits are
    null controls: <X> for psi=|0> and <Z> for psi=|+> must vanish at any depth,
    so a significant value is evidence of a phase/calibration problem and not of
    mixedness.

What this run deliberately does NOT measure
-------------------------------------------
The over-read delta^T C delta itself.  At the predicted radius it is
(1-r^2)(F_Haar - c0) ~ 0.007 on the depolarizing channel, while a branch-summed
hardware fidelity has sigma(F) ~ sqrt(F(1-F)/N_eff) with N_eff ~ 0.15 x shots on
this device's syndrome readout, i.e. sigma(F) ~ 0.033 at 1000 shots and
sigma(F^2) ~ 0.05 -- seven times the effect.  Resolving 0.001 would need
N_eff ~ 4e5 per branch, ~5e7 shots in total.  So the run measures the RADIUS that
prices the over-read, and the linear exactness; the over-read number itself stays
with the analytic result that is audited to 1e-14.  This is the same shot-budget
bound ED Table 3's note already states.

Stages (four are implemented; under the confirmed ~300 s allocation only ONE is
affordable and it is `tomo`.  The run plan, the confirmed cost model and the
salvage ladder are QCLOUD_SETUP.md 8 and `--mode plan`.)
----------------------------------------------------------------
    tomo   : 19 circuits x 1000 shots, 5 qubits -- M1 M2 M3 M5, plus its OWN three
              calibration circuits, so it needs no canary and never assumes that an
              earlier job's block is still its own;  19 000 shots = ~266 s, i.e. the
              WHOLE remaining allocation
    probe  :  8 circuits x  300 shots, 5 qubits -- 2+1 readout calibration (the
              repeat brackets the drift) and 5 single-X fingerprints that determine
              the returned bit order uniquely;  2 400 shots = ~34 s.  DEFERRED as
              redundant: tomo carries the calibration and recovers the bit order
              from its own circuits, which selftest (f) proves is sufficient
    bench  : 34 circuits x 2000 shots, 9 qubits -- mid/late A-B, then the full
              16-branch sum at p=0 for psi in {0,+} (2 x 16), late-measure, on the
              paper's warm-start angles;  68 000 shots = ~952 s.  Does not fit
    ladder :  4 circuits x 4000 shots, 5 qubits -- the DEPTH_SCAN grid at rounds=4
              (293 instructions / 144 CNOT), analysed together with tomo as one
              depth curve;  16 000 shots = ~224 s.  Does not fit alongside tomo
All four together would be 65 circuits, 105 400 shots, ~1476 s -- 4.9x the budget.

The metering unit IS now known.  The account holder confirmed (2026-10) that QPU
time is billed in real time, that exhausting it immediately disables the account,
and that a job cannot overrun its allocation.  So the old ~50x two-model hedge is
closed in favour of the per-shot branch -- 14 ms/shot from the 2026-09 timeline --
and ~300 s buys ~21 400 shots in total.  Two consequences shape the code:

  * A MID-JOB CUT-OFF is the expected failure mode rather than a rare one, and a
    cut-off job returns nothing.  So tomo is submitted in chunks of
    STAGE_CHUNK['tomo'] circuits, submit_cloud writes the raw counts to disk after
    every chunk, and stage_circuits orders the stage calibration-first, then
    depth-1 grouped by input state, then depth-2.  Every chunk boundary is
    therefore an analysable prefix, and each adds a complete increment: 3 circuits
    buy the assignment matrix, 6 buy M1 for |0>_L and M3 at depth 1, 15 buy M1 and
    M2 in full, 19 add M5.  selftest (j) asserts that ordering, --mode analyze
    accepts the truncated dump, reports exactly which inputs and depths are
    missing, and refuses to present a subset as though it were the whole claim.
  * `--mode cloud --stage all` is rejected: with one allocation and four stages,
    running them unattended would bill the first and lose the other three.

Shot sizing is not a guess: `--mode simulate` runs the whole chain on synthetic
counts from a device built out of quoted vendor fidelities, through the SAME
analysis path the cloud data takes (including the tomo job's own calibration), so
the achievable error bars are known before any shots are bought.  At 8000 shots on
a 0.9993/0.9831/0.9612 device it recovers w = 0.3415 (truth 0.3396) and r =
0.83-0.87 (truth 0.854-0.881) with a 95% CI half-width of 0.0278 on r, every null
control inside 1 sigma, and |L_Z - L_X| bounded to 4.7% of the term span; over 6
sampling seeds the M2 z-score has mean +0.50 and rms 1.21, i.e. pure shot noise.
The same study on 0.9988/0.9790/0.9260 gives a CI half-width of 0.0489 and needs
47 918 shots for +-0.02 -- 3.1x more -- which is the arithmetic behind the backend
choice.  Halving the shots widens the CI on r by sqrt(2); that is the whole
trade-off, it is a CLI flag, and `--mode plan` prints the table.

At the 1000 shots the confirmed budget actually allows, that scaling gives a CI
half-width of 0.079 on r -- but the significance of "r < 1" also depends on how far
below 1 the device actually sits, and the FREE preflight of 2026-10-11 MEASURED that
on WK_C180_2's own best block: f2q = 0.99275 against the quoted table's 0.9831.
Better gates mean a purer state, so the predicted radius is 0.9478 rather than
0.8550 and "r < 1" is a ~0.6 sigma statement, not the ~1.8 sigma the quoted table
implied.  Reaching 5 sigma would need ~70 000 shots per circuit, ~18 700 s of QPU,
62x the budget, so on this block the mixedness headline is not purchasable at any
realistic allocation -- the binding constraint is the device being better than its
datasheet, not the ~300 s.  M3's predicted -3.4 sigma refutation of the isotropic
surrogate weakens correspondingly and may vanish: a device closer to depolarizing is
harder to distinguish from it.  M2 does NOT weaken, because it is a null test whose
precision depends on the shot count alone.  Those are the honest error bars; the
analysis reports them as measured, and nothing is rescued by choosing a more
favourable shot count -- or a more favourable BLOCK -- after the fact (see the
BLOCK_SELECTION_RULE comment and selftest (k) for why significance-ranked block
choice is forbidden).  What the run still buys is the upgrade from the paper's
4-circuit x 400-shot feasibility set to a genuine 19-circuit tomography on
WK_C180_2, with its own measured readout assignment matrix.

Run -- everything through preflight costs zero QPU time:
    ./qenv/bin/python hw_mixedness.py --mode plan         # the budget, no account
    ./qenv/bin/python hw_mixedness.py --mode verify       # exact, no account
    ./qenv/bin/python hw_mixedness.py --mode selftest     # the 10 anchors, no account
    ./qenv/bin/python hw_mixedness.py --mode sample       # CPUQVM dry run of the
                                                          # whole analysis chain
    ./qenv/bin/python hw_mixedness.py --mode simulate \
        --vendor 0.9993,0.9831,0.9612 --shots 1000 --seeds 6  # at the real budget
    ./qenv/bin/python hw_mixedness.py --mode preflight \
        --chip-ids WK_C180_2 --vendor 0.9993,0.9831,0.9612   # FREE cloud queries
    bash launch_mixedness.sh tomo    # THE ONLY AFFORDABLE JOB: ~266 s of the ~300 s
    ./qenv/bin/python hw_mixedness.py --mode analyze --stage all   # offline, no token
Deferred until the allocation is topped up; each is a separate job and none of them
fits in what is left after tomo:
    bash launch_mixedness.sh probe   # redundant anyway, tomo self-calibrates
    bash launch_mixedness.sh bench   # ~952 s
    bash launch_mixedness.sh ladder  # ~224 s
Outputs:
    hw_mixedness.json, qcloud_raw_{probe,mixedness,bench,ladder}.json,
    figures/fig_hw_mixedness_<tag>.png
"""
import argparse
import itertools
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import qcloud_vscr_new as q                                       # noqa: E402

NQ5 = 5
DIM5 = 2 ** NQ5
PAULI5 = [q._gate_matrix(('X', 0)), q._gate_matrix(('Y', 0)),
          q._gate_matrix(('Z', 0))]

# tomography grid: 4 inputs x 3 bases = 12 circuits, plus the depth-2 scan
TOMO_STATES = ('0', '1', '+', '-')
BASES = ('X', 'Y', 'Z')
BASIS_ROT = {'Z': [], 'X': [('H', 4)],
             'Y': [('RZ', 4, -math.pi / 2), ('H', 4)]}
# (psi, basis) pairs run at rounds=2.  The first two are the SIGNAL (for psi=0 the
# Bloch vector is along z so <Z> alone gives r; for psi=+ it is along x so <X>
# does), the last two are NULL CONTROLS that must vanish at any depth.
DEPTH_SCAN = (('0', 'Z'), ('+', 'X'), ('0', 'X'), ('+', 'Z'))
IDEAL_BLOCH = {'0': (0.0, 0.0, 1.0), '1': (0.0, 0.0, -1.0),
               '+': (1.0, 0.0, 0.0), '-': (-1.0, 0.0, 0.0),
               '+i': (0.0, 1.0, 0.0), '-i': (0.0, -1.0, 0.0)}
BASIS_AXIS = {'X': 0, 'Y': 1, 'Z': 2}   # which Bloch component basis B reads

STAGE_SHOTS = {'probe': 300, 'tomo': 1000, 'bench': 2000, 'ladder': 4000,
               'warmcold': 3000, 'suppl': 20000}
# THE COST MODEL IS NOW MEASURED, NOT HYPOTHESISED.  The account holder confirmed
# (2026-10) that QPU time is billed in real time, that exhausting it immediately
# disables the account, and that a job CANNOT overrun its allocation.  So the
# pessimistic branch of the old two-model hedge is the operative one:
#     14 ms/shot  (1800 s / 128 000 shots, the 2026-09 timeline)
#     ~300 s left  =>  ~21 400 shots for the WHOLE experiment.
# That number is flat in circuit length on purpose: 14 ms/shot over a 204-
# instruction circuit is ~68 us per instruction, ~2000x a superconducting gate
# time, so the bill is dominated by per-shot overhead (readout, reset, control
# dead time) and not by how many gates a circuit has.  Shortening the circuits
# would therefore not buy shots, and only the shot count is worth optimising.
#
# Consequence for the stage budgets: tomo at 1000 shots is 19 000 shots = 266 s,
# i.e. the ENTIRE remaining allocation.  probe (+34 s) would leave no margin, and
# bench (952 s) and ladder (224 s) are simply unaffordable.  probe is also nearly
# redundant now that tomo carries its own calibration and can recover the bit
# order from its own circuits (analyze_stage, selftest (f)), so the budget goes
# to the headline measurement alone.  The other stages stay implemented and stay
# correct -- they are what to run if the allocation is ever topped up.
#
# "Cannot overrun" makes a MID-JOB KILL the expected failure mode rather than a
# rare one, and a killed job returns nothing, so the whole stage would be lost
# along with the time it had already billed.  Two things defend against that:
# chunking (submit_cloud dumps the raw counts after every chunk, so a kill costs
# only the chunk in flight) and ordering (below).  A refusal is still handled by
# halving --shots down to SHOTS_FLOOR with every attempt recorded, but under a
# hard real-time budget the chunked prefix is the mechanism that matters.
#
# The probe stays on FIVE qubits and therefore cannot carry the mid/late A/B pair:
# a submission has one block, and on a 9-qubit block the returned bitstrings are
# 9 bits wide, which would silently invalidate the 5-position layout likelihood and
# the 5x5 assignment matrix.  Whether a 5-qubit circuit submitted inside a 9-qubit
# block comes back 5 or 9 bits wide is NOT verifiable offline, so the plan never
# depends on it.  The A/B pair is circuits 1-2 of the bench stage instead.
SHOTS_FLOOR = 1000
QPU_REFUSAL_MARKERS = ('qpu time', 'insufficient', 'not enough', 'quota')
# Circuits per submission.  For tomo this is chosen so that EVERY chunk boundary
# lands on a scientifically complete prefix, because with a hard budget the
# truncated dump is the likely outcome and it has to be analysable:
#     chunk 1 ( 1- 3) the 3 calibration circuits -> assignment matrix
#     chunk 2 ( 4- 6) the |0>_L triple           -> M1 for |0>_L and M3 at depth 1
#     chunk 3 ( 7- 9) the |1>_L triple           -> M1 for two inputs
#     chunk 4 (10-12) the |+>_L triple           -> M1 for three, L_Z ensemble
#     chunk 5 (13-15) the |->_L triple           -> M1 and M2 COMPLETE (4 states)
#     chunk 6 (16-18) depth-2 (0,Z),(+,X),(0,X)  -> M3 at depth 2, M5 signals
#     chunk 7 (19   ) depth-2 (+,Z)              -> M5 with both null controls
# analyze_tomo requires all three bases of a state to report that state's radius,
# and all four states for M2, so a state-grouped order means no chunk boundary can
# leave a half-measured state behind.  3 is the finest chunking with that property.
STAGE_CHUNK = {'probe': 8, 'tomo': 3, 'bench': 4, 'ladder': 4, 'warmcold': 2,
               # suppl: chunk 1 = the two LATE circuits (the decisive S1 pair that
               # reproduces the 2026-09 device point), chunk 2 = the two MID ones
               # (the S2 re-test of "dead mid-circuit measurement").  A cut-off
               # therefore costs the re-test, never the reproduction.
               'suppl': 2}
# Order the depth-1 tomography by input state so |0>_L -- the state the paper's
# headline mixedness number is about, and the one whose Z circuit M3 needs -- is
# measured first and survives a kill that happens later.
TOMO_STATE_PRIORITY = ('0', '1', '+', '-')
# The remaining QPU allocation, in seconds, as confirmed by the account holder.
# Every cost statement in --mode plan is derived from this one number rather than
# restating it, so the plan cannot drift away from the budget it is sized for.
BUDGET_S = 300.0
# CONFIRMED cost model (account holder, 2026-10): QPU time is deducted in real
# time, exhausting it disables the account immediately, and a job cannot overrun
# its allocation.  14 ms/shot is 1800 s / 128 000 shots from the 2026-09 timeline,
# so BUDGET_S buys ~21 400 shots for the WHOLE experiment.  Promoted to a module
# constant because selftest (k) needs the same number print_plan uses -- a rate
# that lives in one function cannot be checked against the budget by another.
MS_PER_SHOT = 14.0
# BUT 14 ms/shot IS NOT REPRODUCIBLE FROM ANYTHING ON DISK, and the direction of
# the error matters, so the evidence is recorded rather than buried.  What
# qcloud_run2.log actually timestamps for the 2026-09 run is:
#   trial  16 circuits x 2000 shots =  32 000 shots, "all results received in 77s"
#   full   64 circuits x 4000 shots = 256 000 shots delivered in ~208 s, then
#          RuntimeError: QPU time is insufficient. Please purchase more.
# i.e. 288 000 shots in ~285 s of WALL time = ~1.0 ms/shot, 14x faster than this
# constant.  The two are not necessarily in conflict -- wall time is not billed
# time, and a vendor counter that includes queueing, compilation and per-circuit
# setup can legitimately run several times faster than the job's wall clock -- but
# the 1800 s in the derivation above is an account-dashboard figure that no file in
# this repository records, so the multiplier between wall and billed time is
# UNVERIFIED and could be anywhere from 1x to 14x.
# Consequence, and why the plan is still safe: MS_PER_SHOT is the pessimistic end,
# so every cost printed by --mode plan is an UPPER bound.  If the true billed rate
# is nearer 1-6 ms/shot then the ~300 s allocation buys 47 000-300 000 shots
# rather than 21 400, and the warmcold stage would cost 18-108 s instead of 252 s.
# Nothing here should be "corrected" by raising a stage's shot count on the
# strength of this note alone: the rate is unverified, a job cannot be extended
# once submitted, and the cheapest way to settle it is to read the account's own
# remaining-QPU-time figure before and after the first chunk.  Until then the
# conservative constant stands and the margin it leaves is the point.
MS_PER_SHOT_WALL_OBSERVED_2026_09 = 1.0

# --- THE RATE IS NOW MEASURED.  The 14 ms/shot above is FALSIFIED by 45x. -------
# The note directly above this line said the multiplier between wall and billed
# time was UNVERIFIED and that "the cheapest way to settle it is to read the
# account's own remaining-QPU-time figure before and after the first chunk."
# That has now been done, three independent times, on two chips, and the answer
# is the same each time within a factor of three:
#
#   date        chip        shots      dashboard delta   ms/shot   denominator
#   2026-10-15  WK_C180       1 000     ~14 s reserved    --       (reserved at 14.0, never settled)
#   2026-10-15  WK_C180_2     1 000     0.373-0.621 s    0.373-0.621  attribution RANGE (two probes shared a window)
#   2026-10-21  WK_C180_2   120 000     37.074 s         0.3090   CERTAIN: 389.136 -> 352.062 over
#                                                                   6/6 FINISHED circuits, so the shot
#                                                                   denominator is not in doubt
#
# The 0.3090 figure is the one to plan with: it is the only reading whose shot
# denominator is certain, and it is the LARGEST sample by 120x.  14.0 is retained
# above under its own name purely as the historical pessimistic bound that
# check_budget() still defaults to, so that an offline cost printed without
# --ms-per-shot remains an upper bound rather than silently becoming an estimate.
# Nothing should read 14.0 as a live figure any more.
MS_PER_SHOT_MEASURED = 0.3090
MS_PER_SHOT_MEASURED_PROVENANCE = (
    'dashboard delta 389.136 s -> 352.062 s = 37.074 s over 120 000 shots, '
    '2026-10-21 WK_C180_2 warmcold A/B, 3 submissions all FINISHED, 6/6 circuits '
    'returned, so the shot denominator is certain.  Corroborated at 0.373-0.621 '
    'by the two 2026-10-15 probes.  The 14.0 constant it replaces is 45x larger '
    'and is retained only as a historical upper bound.')


def ms_per_shot_source_label(ms_per_shot):
    """An honest provenance string for a billing rate.

    The label used to be a two-way branch on `ms_per_shot is None`, which stamped
    'dashboard-measured' onto ANY explicitly passed value -- including 14.0, the
    assumed constant nobody had measured.  That is how probe A of 2026-10-15 came
    to record the pessimistic constant as a dashboard reading, and the value drove
    that record's BOUND verdict.  A rate's provenance is a fact about where the
    number came from, not about whether the caller bothered to pass it, so it is
    resolved by VALUE against the constants this module actually owns."""
    if ms_per_shot is None:
        return ('pessimistic constant MS_PER_SHOT=%.1f (assumed, FALSIFIED by '
                '%.0fx; NOT a dashboard reading)'
                % (MS_PER_SHOT, MS_PER_SHOT / MS_PER_SHOT_MEASURED))
    ms = float(ms_per_shot)
    if ms == MS_PER_SHOT:
        return ('explicit value equal to the pessimistic constant MS_PER_SHOT=%.1f'
                '; NOT a dashboard reading' % MS_PER_SHOT)
    if abs(ms - MS_PER_SHOT_MEASURED) < 1e-9:
        return 'dashboard-measured 2026-10-21: %s' % MS_PER_SHOT_MEASURED_PROVENANCE
    return 'caller-supplied dashboard delta (%.4f ms/shot)' % ms

# --- measured calibration of the block the vendor's own optimiser picks ---------
# Recorded from the FREE preflight of 2026-10-11T01:00:25Z on WK_C180_2 -- the
# first time ChipInfo answered at all, having been 401 Unauthorized since
# 2026-09-21.  Per-qubit figures of merit for best_qubit_blocks(5):
#   q39 read 0.96870 1q 0.99860 T1 32.2us T2  2.1us
#   q48 read 0.96610 1q 0.99940 T1 36.5us T2  6.2us
#   q49 read 0.94730 1q 0.99930 T1 33.8us T2  3.0us
#   q58 read 0.92280 1q 0.99950 T1 41.6us T2  8.9us
#   q67 read 0.94990 1q 0.99930 T1 48.2us T2  5.3us
#   CZ edges 39-48 0.9922, 39-49 0.9936, 49-58 0.9875, 58-67 0.9977  (a tree, so
#   the block is connected and 5-qubit stages can run on it as submitted)
# giving f1q=0.99922, f2q=0.99275, fread=0.95096.
#
# THESE ARE WHAT THE DEVICE ACTUALLY DOES, and the two-qubit figure is BETTER than
# the quoted aggregate table (0.9831) that every power estimate in
# QCLOUD_SETUP.md sec.7-8 was built on.  Through vendor_prediction that raises the
# predicted radius 0.8550 -> 0.9478, so the headline signal 1-r falls
# 0.1450 -> 0.0522 (2.8x smaller) and r<1 becomes a 0.60 sigma statement at
# STAGE_SHOTS['tomo']=1000 shots rather than the 1.8 sigma the quoted table
# implied.  More shots do not rescue it: 5 sigma on this block needs ~70 000
# shots/circuit = 1.33 M shots = ~5.2 h of QPU.  The ~300 s budget is therefore NOT
# the binding constraint on that particular claim -- the device being better than
# its datasheet is.  Held here so the arithmetic is reproducible and so nobody
# re-derives the 1.8 sigma figure from the quoted table a third time.
VENDOR_BLOCK_WK_C180_2 = [39, 48, 49, 58, 67]
MEASURED_CALIB_UTC = '2026-10-11T01:00:25Z'
MEASURED_F1Q = 0.99922
MEASURED_F2Q = 0.99275
MEASURED_FREAD = 0.95096
QUOTED_F1Q, QUOTED_F2Q, QUOTED_FREAD = 0.9993, 0.9831, 0.9612
CI_HALF_WIDTH_AT_1000 = 0.079      # bootstrap half-width, from the power study

# The nine-qubit block the 2026-10-12 warmcold job ran on, as recorded in its
# submission-time sidecar.  It is the vendor's own best_qubit_blocks(9) pick, and
# it is REJECTED for any rerun: its own in-job calibration measured logical q3 --
# a DATA qubit -- at e01=0.5679, e10=0.5885, contrast -0.156, i.e. worse than a
# coin flip and negative.  A negative-contrast channel inside the data register
# makes the primary estimator uninterpretable, and it biases a warm-minus-cold
# difference with its sign flipped (the five-channel null reads +2.84 sigma and
# drops to -0.10 sigma once q3 is excluded).  Re-running science circuits on this
# block would reproduce that fault.
#
# It is recorded as a constant rather than re-derived because the rejection must
# survive the vendor's ranking changing: best_qubit_blocks(9) may well return this
# same block again, and ChipInfo cannot be used to rule it out -- on this block it
# reported the channel that measured 0.4218 at 0.9612, and its own worst qubit
# (0.9014) measured perfect.  Metadata quality and day-of readout are essentially
# uncorrelated here, so the only gate that works is a paid cal9 screen.
WARMCOLD_BLOCK_REJECTED_2026_10_12 = [39, 48, 49, 57, 58, 66, 67, 75, 85]
# ... and that rejection is a fact about ONE CHIP.  The block is a list of physical
# qubit indices on WK_C180_2; the same indices on WK_C180 are a different device
# with a different day-of readout, so excluding them there would reject a block on
# evidence measured on another chip.  Keyed by chip, with the constant above kept
# as the WK_C180_2 entry so every existing reference still resolves.
REJECTED_BLOCKS = {'WK_C180_2': [WARMCOLD_BLOCK_REJECTED_2026_10_12]}


def rejected_blocks_for(chip_id):
    """Blocks rejected on MEASURED grounds on this chip (empty if none)."""
    return tuple(tuple(b) for b in REJECTED_BLOCKS.get(chip_id, ()))

# Cost of the screen itself.  Billing is per SHOT, not per instruction, so the two
# calibration circuits cost what any two circuits cost at this shot count:
# 2 x 1000 = 2000 shots = 28 s at the confirmed pessimistic 14 ms/shot.  That fits
# inside even the pessimistic reading of the remaining allocation (~48 s after the
# 252 s the 2026-10-12 run consumed of BUDGET_S), which is the property that makes
# it safe to run before the remaining budget has been read off the dashboard.
# 1000 shots is ample for e01/e10: the 2026-10-12 screen at 3000 shots resolved
# the broken channel at contrast -0.156 against a 0.5 threshold, a margin no
# plausible shot noise at 1000 closes.
CAL9_SCREEN_SHOTS = 1000
CAL9_SCREEN_RAW = 'qcloud_raw_cal9screen.json'

# The block is the vendor's own best_qubit_blocks(5) pick and MUST stay that way.
# Ranking all 96 connected 5-qubit sub-blocks of WK_C180_2 by the significance of
# r<1 puts this block LAST, 96/96, and puts [40, 49, 50, 58, 68] first at 6.3
# sigma -- because a worse device is more mixed, so significance is maximised by
# the worst qubits on the chip.  Picking a block from that ranking would be
# choosing the apparatus so the hypothesis test passes, which is not a measurement
# and would not survive a referee asking why that block.  The only defensible
# criterion is one fixed before any outcome is known; the vendor's optimiser is
# such a criterion.  selftest (k) locks the rule and the trap in together.
BLOCK_SELECTION_RULE = ('vendor best_qubit_blocks(nq); never ranked by the '
                        'significance of the result it would produce')

# seeds for the power study's repeat draws; the first is the one every other
# function defaults to, so a single-seed run and a scan share their first draw
SEED_BANK = (20260921, 11, 202, 3003, 40004, 777777, 5, 606, 8080, 91,
             1234, 24680, 13579, 99991, 424242, 8888)
STAGE_NQ = {'probe': NQ5, 'tomo': NQ5, 'bench': q.NQ, 'ladder': NQ5,
            'warmcold': q.NQ, 'suppl': q.NQ}
STAGE_RAW = {'probe': 'qcloud_raw_probe.json',
             'tomo': 'qcloud_raw_mixedness.json',
             'bench': 'qcloud_raw_bench.json',
             'ladder': 'qcloud_raw_ladder.json',
             'warmcold': 'qcloud_raw_warmcold.json',
             'suppl': 'qcloud_raw_suppl.json'}
# The ladder stage re-runs the DEPTH_SCAN grid at deeper rounds, so the measured
# radius becomes a curve in depth rather than two points.  rounds=4 is 293
# instructions / 144 CNOT and rounds=8 is 581 / 288 (both verified constructible);
# 8 is off by default because it is the most expensive circuit in the whole plan
# and the least essential to any of M1-M5.
LADDER_ROUNDS = (4,)
# Every stage is submitted on ONE fixed physical block, passed with --block, so the
# readout assignment matrix measured by the canary applies to the tomography.  The
# 5-qubit stages and the 9-qubit bench necessarily use different blocks (a
# submission has one block), and which two were used is recorded in the artifact
# rather than assumed to be comparable.
STAGE_JOB = {'probe': 1, 'tomo': 2, 'bench': 3, 'ladder': 4, 'warmcold': 5,
             'suppl': 6}
# the bench stage reuses the frame the WK_C180 feasibility run used, so the new
# full-branch point and the old 4-circuit point are the same physical situation
BENCH_FRAME = 'IIIII'
BENCH_S_INJECTED = 12
BENCH_FRAME_INJECTED = 'IIXII'

# ---------------------------------------------------------------------
# dump paths: every paid run writes its OWN file and can never overwrite evidence
# ---------------------------------------------------------------------
# WHY THIS EXISTS.  run_stage and run_cal9_screen defaulted their dump path to
# STAGE_RAW[stage] / CAL9_SCREEN_RAW, i.e. to qcloud_raw_warmcold.json and
# qcloud_raw_cal9screen.json -- and BOTH ARE COMMITTED.  They are the raw counts
# behind the manuscript's 2026-10-12 INVALID paragraph (logical q3 / physical 85
# at e01=0.5679, e10=0.5885, contrast -0.156) and its 2026-10-13 screen (the same
# channel back at e01=0.0000, e10=0.0020, contrast +0.9980).  A rerun would have
# opened them with 'w' and destroyed the only record those numbers can be
# recomputed from -- silently, because submit_cloud writes the dump incrementally
# and unconditionally after every chunk.  The legacy names are therefore FROZEN: a
# new run gets its own timestamped file, and writing to a frozen name raises.
FROZEN_EVIDENCE = frozenset(
    [CAL9_SCREEN_RAW, 'raw_trial_batch.json', 'raw_verify_options.json',
     # the 2026-10-11 free-query snapshot QCLOUD_SETUP.md 8.8 cites for B5's
     # measured calibration; --mode preflight now writes a timestamped file
     'hw_mixedness_preflight.json']
    + list(STAGE_RAW.values()))
# Run logs are evidence too: the wall time, the returned shot counts and the
# per-chunk record quoted in the hardware section exist only in them, and *.log is
# gitignored, so logs go here and this directory is tracked (see .gitignore).
RUN_LOG_DIR = os.path.join(ROOT, 'hw_evidence')


def _utc_stamp():
    return time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())


def unique_dump_path(kind, chip_id, root=ROOT):
    """A dump path for a NEW run: qcloud_raw_<kind>_<chip>_<UTC>.json.

    Uniqueness is asserted rather than assumed -- the file must not already exist
    and the name must not be frozen evidence -- so a rerun can never land on the
    counts a manuscript number was computed from."""
    base = 'qcloud_raw_%s_%s_%s.json' % (kind, chip_id, _utc_stamp())
    path, n = os.path.join(root, base), 1
    while os.path.exists(path):
        path = os.path.join(root, '%s_%d.json' % (base[:-5], n))
        n += 1
    return assert_writable_dump(path)


def assert_writable_dump(path):
    """Refuse to write a dump over committed evidence, or over another dump."""
    name = os.path.basename(path)
    if name in FROZEN_EVIDENCE:
        raise RuntimeError(
            '%s is FROZEN EVIDENCE: it holds the raw counts a manuscript number '
            'is computed from.  A new run must write its own timestamped dump -- '
            'omit --dump-raw to let unique_dump_path choose one.' % name)
    if os.path.exists(path):
        raise RuntimeError(
            '%s already exists; refusing to overwrite a dump that may already be '
            'cited.  Choose another --dump-raw.' % path)
    return path


# ---------------------------------------------------------------------
# the QPU-time ledger: cost a submission BEFORE it is submitted, and record it
# ---------------------------------------------------------------------
# A job cannot overrun its allocation and a killed job returns nothing, so the
# only place a budget can be enforced is here, before the spend.  Two numbers go
# in: how many shots this submission asks for, and how much time the account
# holder's dashboard says is LEFT.  The second cannot be read from the API --
# QCloudService exposes backend/backends/best_qubit_blocks/setup_logging and
# ChipInfo only topology and calibration, with no balance, quota or usage query
# anywhere in the SDK -- so it is a required argument for any paid submission and
# the human reads it off the dashboard.  That is not a inconvenience to design
# away: it is the only measurement of the billing rate that exists, and the
# difference between two readings across a known number of shots is what finally
# replaces MS_PER_SHOT's unverified 14x.
BUDGET_MARGIN_S = 30.0


class BudgetError(RuntimeError):
    """Raised instead of submitting a job that cannot fit the allocation."""


class NoEligibleBlock(RuntimeError):
    """Every block the vendor ranks is rejected on measured grounds, so there is
    nothing to run on.  Distinct from a failed query: a caller may tolerate the
    second (the server then chooses qubits) but must never tolerate this one."""


def cost_s(n_shots, ms_per_shot=None):
    """Seconds of billed QPU time for `n_shots` at `ms_per_shot` (default: the
    pessimistic MS_PER_SHOT, so every cost stated offline is an upper bound)."""
    return n_shots * (MS_PER_SHOT if ms_per_shot is None else ms_per_shot) / 1000.0


def check_budget(kind, n_shots, budget_remaining, ms_per_shot=None,
                 margin_s=BUDGET_MARGIN_S, verbose=True):
    """Refuse (BudgetError) unless `n_shots` fits the remaining allocation.

    Returns the ledger entry for this submission.  `budget_remaining=None` is a
    refusal rather than a pass: an unread budget is not an infinite one, and the
    2026-10-12 run spent 18 000 shots against a remainder nobody had read."""
    if budget_remaining is None:
        raise BudgetError(
            '%s: --budget-remaining is REQUIRED for a paid submission.  The '
            'remaining allocation cannot be queried (no balance/quota/usage call '
            'exists in the SDK), so read it off the OriginQ dashboard and pass it '
            'in seconds.  Without it this file cannot tell a 28 s screen from a '
            'job that will be cut off.' % kind)
    ms = MS_PER_SHOT if ms_per_shot is None else float(ms_per_shot)
    c, c_pes = cost_s(n_shots, ms), cost_s(n_shots)
    allowed = float(budget_remaining) - margin_s
    if c > allowed:
        raise BudgetError(
            '%s: %d shots cost %.1f s at %.3f ms/shot, but only %.1f s remain and '
            '%.0f s of that is held back as cut-off margin.  %s'
            % (kind, n_shots, c, ms, float(budget_remaining), margin_s,
               ('The margin alone exceeds the remainder, so NOTHING is affordable: '
                'top up the allocation.' if allowed <= 0 else
                'Reduce --shots (the largest that fits is %d) or top up the '
                'allocation.' % int(allowed * 1000.0 / ms))))
    if verbose:
        print('[budget] %s: %d shots = %.1f s at %.3f ms/shot%s; %.1f s remain, '
              '%.0f s margin held back -> %.1f s left after this submission'
              % (kind, n_shots, c, ms,
                 '' if ms_per_shot is None
                 else ' (%s)' % ms_per_shot_source_label(ms_per_shot).split(';')[0],
                 float(budget_remaining), margin_s, allowed - c), flush=True)
        if c_pes > allowed:
            print('[budget] WARNING: at the pessimistic %.0f ms/shot this same '
                  'submission would cost %.1f s and NOT fit.  It is being allowed '
                  'only because the rate was measured from a dashboard delta; if '
                  'that measurement is wrong the job is cut off and only the '
                  'completed chunks survive on disk.'
                  % (MS_PER_SHOT, c_pes), flush=True)
    return dict(kind=kind, n_shots=int(n_shots), ms_per_shot=ms,
                ms_per_shot_source=ms_per_shot_source_label(ms_per_shot),
                cost_s=c, cost_s_pessimistic=c_pes,
                budget_remaining_before=float(budget_remaining),
                budget_remaining_after=float(budget_remaining) - c,
                margin_s=margin_s, utc=_utc_stamp())


def record_billing(art, entry):
    """Append one ledger entry to the artifact, so the billed rate stops being a
    hypothesis: requested vs returned shots, wall time and the dashboard deltas
    that bracket each submission are what MS_PER_SHOT should eventually be
    replaced by."""
    art.setdefault('qpu_billing', []).append(entry)
    return entry



# ---------------------------------------------------------------------
# The warm-vs-cold recovery A/B -- the only hardware claim this budget can buy
# ---------------------------------------------------------------------
# WHY THIS STAGE EXISTS.  The tomo stage's headline is r<1, and selftest (k)
# proved that on the vendor's own best block the measured calibration predicts
# r = 0.9478, i.e. r<1 is a 0.60 sigma statement at 1000 shots and needs ~5.2 h
# of QPU for 5 sigma.  That claim is not purchasable at any plausible budget, so
# spending the allocation on it would buy a null result.  This stage spends the
# same allocation on a claim that IS purchasable: the paper's variational recovery
# is trained twice -- warm-started from the analytic decoder (the angles the
# manuscript reports) and cold-started from zero -- and the two angle sets differ
# by a LARGE, exactly computable amount on exactly one of the sixteen syndrome
# branches.  Measuring that difference on the device tests the paper's central
# methodological claim rather than the device's purity.
#
# THE EFFECT IS LOCALISED AND PRE-COMPUTED, not searched for.  Enumerating all 16
# branches at p=0 with the exact 512-dim state-vector simulator:
#     branch s=5  (frame IZIII, psi=+):  warm F=1.000000000  cold F=0.499999625
#                                        gap = +0.5000003748
#     every other branch:                |gap| < 1e-5
# So s=5 is the ONLY branch carrying a warm/cold difference, and s=12 (frame
# IIXII, gap +0.000003) is the matched null control that must come out at zero.
# Both were fixed before any hardware data was taken.
#
# psi='+' IS LOAD-BEARING, NOT A FREE CHOICE.  The same enumeration over input
# states gives gap(s=5) = +0.500000 for psi in {+, -, +i, -i} and +0.000001 for
# psi in {0, 1}: the cold recovery's failure on branch 5 is a LOGICAL Z, which is
# invisible in the Z basis and maximal in the X/Y bases.  Running this A/B at
# psi='0' would return a null result by construction and look like a failed
# experiment.  selftest (l) asserts the state dependence so it cannot be silently
# edited into a Z-basis run.
#
# p=0 (NO INJECTED STOCHASTIC ERROR).  The frame is the deterministic one whose
# stabilizer syndrome is s, so every shot is the branch of interest and there is
# nothing to post-select on.  That is not a convenience: see WARMCOLD_PRIMARY.
WARMCOLD_PSI = '+'
WARMCOLD_SIGNAL_S, WARMCOLD_SIGNAL_FRAME = 5, 'IZIII'
WARMCOLD_NULL_S, WARMCOLD_NULL_FRAME = 12, 'IIXII'
WARMCOLD_GAP_IDEAL_SIGNAL = 0.5000003748512316
WARMCOLD_GAP_IDEAL_NULL = 0.000003
WARMCOLD_GAP_TOL = 1e-3          # |gap| below this counts as "no warm/cold effect"
# The two angle sets are the ones ALREADY ON DISK, so nothing is retrained and
# the hardware run cannot drift from the manuscript: 'warm' is exactly the file
# the paper's figures and numbers are built from.
WARMCOLD_ANGLES = {'warm': 'vscr_angles_paper_dep.npz',
                   'cold': 'vscr_angles_dep.npz'}
# THE PRIMARY ESTIMATOR IS UNCONDITIONED P(data=00000), NOT POST-SELECTED F_s.
# This is a deliberate, pre-registered change of estimator and the reason is
# measured rather than aesthetic.  The 2026-09 WK_C180 run (raw_trial_batch.json,
# 16 branches x ~1985 shots) has an ancilla marginal that is very nearly uniform:
# Shannon entropy 3.67-3.97 bits out of a maximum of 4.0 under the layout that run
# itself recorded, modal syndrome share 0.10-0.38 against a uniform 0.0625, and on
# the 4-circuit feasibility set the
# mid-circuit variant returned P(expected key) = 0.000 with n_sel = 0 out of 369
# and a total-variation distance to ideal of 1.0000.  Conditioning on a bit that
# carries no information about the injected frame therefore discards ~94% of the
# shots to buy nothing, and inflates the error bar by ~4x.  The unconditioned
# data-register probability keeps every shot, and at p=0 it IS the branch
# quantity: with a deterministic frame there is no frame average to undo.
# Both estimators are reported; this one is primary.  selftest (l) RE-DERIVES the
# entropy and the survival from raw_trial_batch.json and fails if the marginal ever
# stops being near-uniform -- that is the only condition under which the
# post-selected F_s would become the better primary estimator again.
WARMCOLD_PRIMARY = 'P_data_zero_unconditioned'
WARMCOLD_SECONDARY = 'F_s_postselected'
# MEASURED DATA-REGISTER SURVIVAL, and the arithmetic that turns a measured
# PROBABILITY into the w the power model needs.  Two committed measurements exist
# and they disagree by an order of magnitude, so w is carried as a PAIR (the
# measured P and the implied w) and the relation between them is a function rather
# than a mental note:
#     warm P(data=00000) ~ w + (1-w)/32     cold ~ w/2 + (1-w)/32
#     observable gap = w * 0.500            =>  w = (P - 1/32) / (1 - 1/32)
# The 1/32 subtraction is not cosmetic, and omitting it was a real error here.  The
# constant that used to live on this line was the measured PROBABILITY 0.1219
# passed to warmcold_power AS IF it were w, which overstates the gap by 30%
# (0.0610 instead of 0.0468) and the significance with it: the "~7.3 sigma" that
# every earlier version of this file, of print_plan and of launch_mixedness.sh
# quoted is 6.1 sigma once P is converted to w.  Both figures also came from the
# 2026-09 run, which submitted WITHOUT QCloudOptions, while the only with-options
# measurement on record is ten times smaller.  The honest statement is therefore a
# range spanning two platforms and two compiler configurations, not a point:
#     2026-09  WK_C180    no options  P = 0.1219 -> w = 0.0936, 6.1 sigma at 3000
#     2026-10  WK_C180_2  options     P = 0.0407 -> w = 0.0098, 1.0 sigma at 3000
# Which world today resembles is not knowable offline, and it is the difference
# between a detection and a bound.  That is exactly what --wc-probe measures for
# 1000 shots (14 s) before the six-circuit A/B is bought.
WARMCOLD_W_FLOOR_UNIFORM = 1.0 / 32


def w_from_p_data_zero(p, floor=WARMCOLD_W_FLOOR_UNIFORM):
    """Data-register survival w implied by a measured P(data=00000).

    Inverts P = w + (1-w)*floor: the ideal outcome contributes w and a fully
    randomised register contributes the uniform floor on the remaining weight.  One
    function so that the probe, the power model and these constants cannot each
    carry their own idea of what the floor does."""
    return (p - floor) / (1.0 - floor)


WARMCOLD_P_DATA_ZERO_2026_09 = 0.1219    # mean over 16 branches, ~1985 shots each
WARMCOLD_W_MEASURED_2026_09 = w_from_p_data_zero(WARMCOLD_P_DATA_ZERO_2026_09)
# The with-options measurement: the 2026-10-12 signal|warm circuit's P(data=00000)
# on the CORRECTED data window (the 0.0091 figure quoted in an earlier version of
# build_warmcold's note came from the complement window and is withdrawn there).
# That run is INVALID as an A/B -- one data channel was dead at contrast -0.156 --
# but its P(data=00000) is still the only measurement that exists of how much of
# the data register survives 206 instructions WITH vendor options on, which is the
# configuration any rerun will use.  selftest (n) re-derives it from the frozen
# dump, so this constant cannot drift from the data it summarises.
WARMCOLD_P_DATA_ZERO_2026_10_12 = 0.0407
WARMCOLD_W_MEASURED_2026_10_12 = w_from_p_data_zero(WARMCOLD_P_DATA_ZERO_2026_10_12)
# "Pessimistic" means HALF the larger measured survival, so selftest (l)'s claim
# that the design still resolves the effect at half of w is literally what is
# computed.  Note that the with-options world sits far BELOW even this rung, which
# is why selftest (l) carries a third power row for it instead of implying that
# 0.0468 is the floor of the observed range.
WARMCOLD_W_PESSIMISTIC = 0.5 * WARMCOLD_W_MEASURED_2026_09
# 4 science circuits + 2 nine-qubit readout-calibration circuits, all at
# STAGE_SHOTS['warmcold'] = 3000 -> 18 000 shots -> 252 s at the confirmed
# 14 ms/shot, leaving 48 s of margin against a budget that cannot overrun.
WARMCOLD_N_SCIENCE = 4
WARMCOLD_N_CAL = 2
OUT_JSON = os.path.join(ROOT, 'hw_mixedness.json')
LAYOUT_FLOOR = 1e-9        # floor on ideal probabilities in the layout likelihood


# ---------------------------------------------------------------------
# 1) circuits
# ---------------------------------------------------------------------
def setup():
    """Synthesise and phase-verify the encoder once (delegates to qcloud_vscr_new)."""
    return q.setup()


def enc_dec_gates(rounds=1):
    """U_enc then U_enc^dag, `rounds` times: 72 unitary instructions per round."""
    enc = q.ENC5
    dec = q.invert_gates(enc)
    out = []
    for _ in range(rounds):
        out += list(enc) + list(dec)
    return out


def build_tomo(psi, basis, rounds=1):
    """The M1 tomography circuit: prep(psi) -> (enc,dec)^rounds -> R_B -> MEAS q0..q4.

    Deliberately does NOT apply TEST_STATES' validation rotation: `val` maps the
    ideal output to |0>, which would erase the sign that M2's <Z_L> needs."""
    prep = q.TEST_STATES[psi][0]
    gates = list(prep) + enc_dec_gates(rounds) + list(BASIS_ROT[basis])
    gates += [('MEAS', i, i) for i in range(NQ5)]
    return gates


def build_cal(kind):
    """Readout-assignment circuits: 'zero' = |00000>, 'one' = X^(x5)|00000>."""
    gates = [('X', i) for i in range(NQ5)] if kind == 'one' else []
    return gates + [('MEAS', i, i) for i in range(NQ5)]


def build_fingerprint(b):
    """X on qubit b only, then measure: an ideal outcome with a single '1', whose
    observed position identifies b.

    These five circuits exist because of a real degeneracy the self-tests caught.
    Every tomography circuit ideally ends in |0000> (x) q4, so its ideal
    distribution is supported on just {'00000','00001'}: the likelihood pins WHICH
    observed position carries q4, and says nothing at all about how q0..q3 are
    arranged among the remaining four.  That is harmless for every estimator in
    this file -- the code-space condition is "the other four are all zero", which
    is permutation invariant, and readout mitigation is applied per observed
    position -- but it is not a determined layout, and an undetermined layout
    should not be written into an artifact as if it were one.  Five depth-2
    circuits remove the ambiguity outright."""
    return [('X', b)] + [('MEAS', i, i) for i in range(NQ5)]


def build_bench_ab(phi):
    """The mid/late A-B pair: the exact circuit whose mid-circuit measurement
    returned 0 hits out of 735 on WK_C180, rebuilt for this chip."""
    out, meta = [], []
    for kind, late in (('mid', False), ('late', True)):
        out.append(q.build_branch_circuit(q.ENC5, '0', BENCH_FRAME_INJECTED,
                                          BENCH_S_INJECTED, phi, late_measure=late))
        meta.append(dict(kind='ab', measure=kind, psi='0', p=0.0,
                         frame=BENCH_FRAME_INJECTED, s=BENCH_S_INJECTED))
    return out, meta


def build_bench(phi, states=('0', '+')):
    """Full 16-branch sum at p=0 (no injected error) for each state, late-measure.

    p=0 rather than the paper's 0.02/0.08 sweep because (i) it is deterministic --
    a sampled Pauli frame depends on RNG call order -- (ii) it pairs with the tomo
    stage, whose mixedness is also the device's own, and (iii) the ideal reference
    is then exactly `simulator_fidelity_frame(psi, 'IIIII', phi)`, so the gap is
    pure device noise."""
    out, meta = [], []
    for psi in states:
        for s in range(16):
            out.append(q.build_branch_circuit(q.ENC5, psi, BENCH_FRAME, s, phi,
                                              late_measure=True))
            meta.append(dict(kind='bench', measure='late', psi=psi, p=0.0,
                             frame=BENCH_FRAME, s=s))
    return out, meta


def build_cal9(kind):
    """Readout-assignment circuits on ALL NINE qubits: 'zero' = |0^9>,
    'one' = X^(x9)|0^9>.

    build_cal is five qubits wide, and the warmcold job runs on a nine-qubit
    block.  Whether a five-qubit circuit submitted inside a nine-qubit block
    comes back five or nine bits wide is NOT verifiable offline (see the
    STAGE_CHUNK comment), and the whole point of these two circuits is to be
    interpretable, so they are built at the block's own width.  They cost 9 and
    18 instructions against the science circuits' 206, and they buy the block's
    per-qubit readout error measured on the day, which is what decomposes the
    observed data-register survival into a readout part and a circuit part."""
    gates = [('X', i) for i in range(q.NQ)] if kind == 'one' else []
    return gates + [('MEAS', i, i) for i in range(q.NQ)]


def load_angle_set(name):
    """One of WARMCOLD_ANGLES' two committed files, loaded from disk.

    Refuses to fall back to training: the warm arm is the exact angle set the
    manuscript's figures and numbers are built from, so retraining it would make
    the hardware run a DIFFERENT experiment from the one the paper reports, and
    that substitution must be impossible to do silently."""
    path = os.path.join(ROOT, WARMCOLD_ANGLES[name])
    if not os.path.exists(path):
        raise SystemExit(
            'ERROR: the %s angle file %s is missing.  The warmcold A/B compares '
            'two EXISTING trainings and will not retrain, because a retrained '
            'warm arm would no longer be the recovery the manuscript reports.'
            % (name, path))
    return np.load(path)['phi']


def load_warmcold_angles():
    """(warm, cold) angle arrays, both shape (16, 60)."""
    return load_angle_set('warm'), load_angle_set('cold')


def warmcold_frame_of(s):
    """The Pauli frame whose stabilizer syndrome is branch `s`.

    Taken from the same SYND_TABLE the training, the paper's figures and
    qcloud_vscr_new's own smoke test use, reached through q.m (ssvr_qec) rather
    than restated here.  Restating a 16-entry map in a second place is how a
    hardware branch silently stops being the branch the simulation predicted, and
    that would invalidate the whole A/B without anything looking wrong."""
    return q.m.SYND_TABLE[q.m.SYND_BITS[s]]


def warmcold_s_of(frame):
    """The inverse of warmcold_frame_of (the map is a bijection on 16 branches)."""
    for s in range(16):
        if warmcold_frame_of(s) == frame:
            return s
    raise KeyError('no syndrome branch corresponds to frame %r' % frame)


def build_warmcold(phi_warm, phi_cold):
    """The warm-vs-cold recovery A/B: one signal branch, one null branch, two
    angle sets, plus the block's own nine-qubit readout calibration.

    ORDER IS THE SALVAGE PLAN.  STAGE_CHUNK['warmcold'] = 2 puts the submission
    boundaries after circuits 2 and 4, so chunk 1 is the SIGNAL PAIR and is a
    complete, analysable headline on its own; chunk 2 adds the null control;
    chunk 3 adds calibration.  That is deliberately the opposite of the tomo
    stage's cal-first order: the tomo estimators cannot be read at all without an
    assignment matrix, whereas this stage's primary estimator is a raw probability
    that needs no matrix, so the calibration here is a decomposition bonus rather
    than a prerequisite and the decisive pair goes first.

    CAL-FIRST WAS REJECTED, THEN THE GROUNDS FOR REJECTING IT WERE RETRACTED.
    The obvious worry is that a broken readout channel should be caught BEFORE
    paying for the four 206-instruction science circuits.  Two measurements from
    the 2026-10-12 run were originally read as closing that idea:

      * The vendor's own ChipInfo cannot screen.  This one STANDS.  It reported
        every one of the nine block qubits at readout_fidelity 0.90-0.97,
        including physical 85 at 0.9612, while the day-of cal9 measured one
        channel at 0.4218 with contrast -0.156; ChipInfo's worst qubit (physical
        66 at 0.9014) measured PERFECT at 1.0000.  ChipInfo and day-of readout
        are essentially uncorrelated on this block, so there is no free pre-spend
        gate and the screening must cost something.
      * cal9 cannot screen the science circuits either, because its channel labels
        DO NOT TRANSFER.  This one WAS WRONG and is retracted.  The apparent
        violations (2 of 36 comparisons outside cal9's own hard P(read 0) bound,
        read as evidence that the vendor re-routes logical qubits per circuit)
        were an artefact of comparing cal9's channel against the wrong column of
        the science counts: the position lookup used the CLASSICAL BIT of a logical
        qubit as its KEY POSITION, putting logical q3 at 7 instead of 3.  On the
        corrected lookup all 36 comparisons lie inside their bounds -- logical q3
        allows P(read 0) in [0.432, 0.589] and the four science circuits read
        0.459, 0.504, 0.523, 0.523.

    So cal9 DOES characterise the block the science circuits ran on, and CAL-FIRST
    IS VIABLE after all: two circuits of 9 and 18 instructions, about two seconds
    of machine time, would have flagged the dead data channel before the 824
    instructions of science were paid for.  The chunk order above is left as it
    stands only because changing it now would make the 2026-10-12 artifact
    unreproducible; a re-run should put cal9 first and gate on
    `screen.block_usable`.

    The earlier claim that the warm-minus-cold difference is "common-mode in
    readout error" and so largely cancels was also TOO STRONG, and the corrected
    analysis shows precisely how.  Common-mode cancellation needs every channel to
    transmit with POSITIVE contrast; a channel with 1-e01-e10 < 0 contributes to
    the difference with its sign flipped, so one dead channel can manufacture a
    gap of the size and shape a real effect would have.  On this run that is not
    hypothetical: the five-channel null reads +2.84 sigma and the four-channel
    version that drops the dead channel reads -0.10 sigma.  An earlier version of
    this note made the same point from the warm arm returning P(data=00000) below
    the 1/32 uniform baseline; that argument is WITHDRAWN, because the number it
    quoted (0.0091) came from the complement bit window.  On the correct window
    the warm arm reads 0.0407 (signal) and 0.0305 (null), both at the baseline
    within error.  Readout is a confounder to be measured, not a factor to be
    assumed away.

    Every circuit is late-measure.  On the 2026-09 feasibility set the mid-circuit
    variant of the same branch returned P(expected key) = 0.000 and n_sel = 0 out
    of 369, while the late variant of that branch returned 0.119 and F_s = 0.80;
    mid-circuit measurement is not usable on this platform and is not risked."""
    out, meta = [], []
    # The frame and the branch index are the SAME fact stated twice, so they are
    # checked against the syndrome table rather than trusted: a circuit built for
    # frame F but post-selected on the wrong s measures nothing.
    for _s, _f in ((WARMCOLD_SIGNAL_S, WARMCOLD_SIGNAL_FRAME),
                   (WARMCOLD_NULL_S, WARMCOLD_NULL_FRAME)):
        assert warmcold_frame_of(_s) == _f, (_s, _f, warmcold_frame_of(_s))
    for s, frame, role in ((WARMCOLD_SIGNAL_S, WARMCOLD_SIGNAL_FRAME, 'signal'),
                           (WARMCOLD_NULL_S, WARMCOLD_NULL_FRAME, 'null')):
        for arm in ('warm', 'cold'):
            phi = phi_warm if arm == 'warm' else phi_cold
            out.append(q.build_branch_circuit(q.ENC5, WARMCOLD_PSI, frame, s, phi,
                                              late_measure=True))
            meta.append(dict(kind='warmcold', role=role, arm=arm,
                             psi=WARMCOLD_PSI, p=0.0, frame=frame, s=s,
                             measure='late', angles=WARMCOLD_ANGLES[arm]))
    for kind in ('zero', 'one'):
        out.append(build_cal9(kind))
        meta.append(dict(kind='cal9', cal=kind, rep=0))
    return out, meta


# ---------------------------------------------------------------------
# The supplementary stage: the psi='0' grid that reproduces the 2026-09
# feasibility set bit-exactly, on a chip and at a shot count that can
# settle whether its device point still holds.
# ---------------------------------------------------------------------
# WHY THIS STAGE EXISTS.  Every hardware number the manuscript quotes from the
# 2026-09 feasibility run -- F_12 = 0.80 [0.68, 0.88], the +6.8 sigma syndrome
# excess, the 1.75 relaxation satellite, eps_instr = 1.04e-2 and its sibling
# 1.81e-2, "0 hits in 735 shots", "0 of 369" -- re-derives EXACTLY from the
# archived raw counts in raw_verify_options.json (checked: 44/55, 9/25, 0/44,
# 0/369, satellite 96/55 = 1.7455, z = +6.82, rank 2 of 16).  The arithmetic is
# not in doubt.  What is in doubt is what those numbers are numbers ABOUT, and two
# facts found after the run force the question.
#
#   1. THAT RUN WAS psi='0', AND NOTHING SINCE HAS BEEN.  Inverting the recorded
#      ideal references identifies the input state bit-exactly:
#      ideal_data_zero_prob('0','IIIII',0,cold) = 0.9999997714222 against the
#      artifact's 0.9999997714222045 (diff 5.4e-15), and ('0','IIXII',12) gives
#      0.9999957389415 against 0.9999957389414843 (diff 4.9e-15).  No other psi
#      comes within 1e-7.  Every 2026-10 circuit -- both probes, the INVALID A/B
#      and the FAILED A/B -- ran at psi='+', whose ideal for the same branch is
#      0.9999967267587.  So the manuscript's device point and the 2026-10
#      measurements are not the same experiment, and they differ in the one
#      respect that matters most: this file's own design note for the warmcold
#      stage says the cold recovery's failure is a LOGICAL Z, "invisible in the Z
#      basis and maximal in the X/Y bases".  psi='0' IS the Z basis.  The device
#      point was measured in the basis that hides the failure mode the paper
#      itself identifies as the dominant one.  At psi='+' the same nominal branch
#      (late, IIXII, s=12, cold angles) returns F_s = 137/1133 = 0.121
#      [0.102, 0.140] and eps_instr = 2.39e-2, against psi='0' 0.80 [0.68, 0.88]
#      and 1.04e-2.
#
#   2. IT IS NOT CHIP DRIFT.  The 2026-10-15 probe on WK_C180 -- the SAME chip
#      that produced 0.80 -- and the 2026-10-21 signal|warm circuit on WK_C180_2
#      are bit-identical (both ideal 0.9999999999999947, matching to 1e-16) and
#      agree to 1.23x: p_hw = 2/985 = 0.002030 against 50/19994 = 0.002501.  The
#      two chips agree with each other in October; neither is near the September
#      psi='0' figure.  So the discrepancy tracks the input state, not the chip.
#
# The open question is therefore narrow and cheap: AT ITS OWN INPUT STATE, does
# the September branch still return F_s ~ 0.80?  This stage asks it on both chips
# at 54x the original shots, with each circuit verified bit-exact against the
# archived ideal reference BEFORE anything is bought (the assert in build_suppl).
# Either answer is publishable and they mean opposite things:
#   F_s ~ 0.80 at psi='0'  -> the September device point is real but BASIS-BOUND.
#       Layer 4 must then say so and quote the psi='+' value beside it, because
#       the envelope prices a certified non-Pauli headroom the Z basis cannot see,
#       and a device point measured where the dominant failure is invisible is not
#       the device point the envelope should be drawn against.
#   F_s ~ 0.12 at psi='0'  -> the September number was a 371-shot artefact or the
#       block drifted, the device point moves to 2.4e-2, and the envelope verdict
#       goes from 1.35x outside its loose end to 3.1x outside it.
# Both strengthen the paper's negative hardware conclusion; neither leaves the
# abstract's "1.04% effective per-instruction error" quotable as it stands.
#
# THE MID-CIRCUIT PAIR RIDES ALONG.  "dead mid-circuit measurement" is an ABSTRACT
# claim resting on 735 shots from this same run (A: 0 of 369 syndromes, a 5.0
# sigma deficit; B: 44 syndromes but 0 data hits).  Re-running both at 20 000
# shots on two chips detects a hit rate down to 5e-5.  It is re-tested rather than
# assumed because a platform capability can be repaired in a month, and if it was,
# the late-measurement workaround this whole hardware protocol is built on loses
# its stated reason.  Note what is and is not being claimed: the 2026-09-10
# raw_trial_batch.json (31 755 shots, also mid-measure) has a pooled
# P(data=00000) = 0.1219, 92.8 sigma over the 1/32 floor -- but that is the
# UNCONDITIONED data marginal, whereas "0 hits in 735" is the POST-SELECTED one.
# A structured data marginal beside a dead post-selected fidelity is exactly the
# pattern the manuscript describes, so there is no contradiction to resolve, only
# a statistic to keep straight.
#
# cal9 IS DELIBERATELY NOT IN THIS STAGE.  The 2026-10-21 A/B spent 40 000 of its
# 120 000 shots on two calibration circuits, because every circuit in one
# submission gets the same shot count.  Here the screen is a separate
# --cal9-screen submission at 1 000 shots each -- what its own docstring
# recommends -- so it costs 0.6 s instead of 12.4 s and none of the science budget
# goes on it.
SUPPL_GRID = (
    # label, measure, frame, s   -- label is the 2026-09 circuit it reproduces
    ('C', 'late', 'IIIII', 0),
    ('D', 'late', 'IIXII', 12),
    ('A', 'mid', 'IIIII', 0),
    ('B', 'mid', 'IIXII', 12),
)
SUPPL_PSI = '0'
# The 2026-09 values this stage reproduces or refutes.  A/B/C/D are the labels
# hw_error_budget.json already uses, so a reader can go straight to the source.
SUPPL_TARGETS = {
    'A': dict(shots_used=369, n_sel=0, n_hit=0, F_s=None,
              ideal=0.9999997714222045, implied_eps=None),
    'B': dict(shots_used=366, n_sel=44, n_hit=0, F_s=0.0,
              ideal=0.9999957389414843, implied_eps=None),
    'C': dict(shots_used=370, n_sel=25, n_hit=9, F_s=0.36,
              ideal=0.9999997714222045, implied_eps=0.018140238890985816),
    'D': dict(shots_used=371, n_sel=55, n_hit=44, F_s=0.80,
              ideal=0.9999957389414843, implied_eps=0.010396618939971347),
}
SUPPL_IDEAL_TOL = 1e-12
# Degeneracy guard, scoped to the suppl stage.  A healthy suppl circuit spreads its
# shots over many distinct bitstrings; a failed one collapses onto a handful.  The
# statistic is the Shannon entropy of the FULL 9-bit outcome distribution NORMALISED
# by log2(min(n, 512)), the most entropy n shots over a 9-qubit register can carry.
# Normalising is not cosmetic: raw entropy tracks the shot count, so the archived
# 2026-09 run measures 5.58-6.05 bits at n~369 while the 2026-10-24 runs measure
# 7.03-7.45 bits at n=20000, and an absolute bit threshold tuned on the latter sits
# only 0.58 bits above the former.  The ratio removes that dependence:
#   degenerate 2026-10-24T141813Z   0.275-0.292
#   archived 2026-09 (n~369)        0.654-0.709
#   healthy 2026-10-24 (n=20000)    0.781-0.827
# 0.45 sits in the empty gap, 0.158 above the degenerate maximum and 0.204 below the
# archived minimum, so it fires on the failure and cannot fire on either valid run.
# WHY IT MUST EXIST: the degenerate submission returned ~50 distinct bitstrings with
# ~50% of all shots in a SINGLE outcome on all four circuits, and near-identical
# z ~= -36.5 across two frames and two depths -- impossible for genuinely different
# circuits.  It still yielded F_s = 0.0000 and therefore the SAME "NOT REPRODUCED"
# string as the two valid runs, so an unguarded pipeline would have accepted it as a
# third independent confirmation and as the complete 4/4 September record.  Bad data
# that reproduces the expected verdict string is the failure mode this closes.
# CAUSE, MEASURED 2026-10-25 -- not inferred from the symptom.  The degenerate
# submission was an APPARATUS FAULT on the September block.  A cal9screen bought for
# 0.6252 s (2 x 1000 shots, WK_C180 block [147,155,157,164,165,166,174,175,176],
# 2026-10-25T02:26:33Z) reproduces it independently: data2 e01=0.117 e10=0.827
# contrast=+0.056, data3 e01=0.086 e10=0.844 contrast=+0.070, anc0 e01=0.037
# e10=0.820 contrast=+0.143, anc3 e01=0.099 e10=0.859 contrast=+0.042 -- 4 of 9
# channels BROKEN, VERDICT "block NOT USABLE, primary estimator INVALID,
# post-selected F_s uninterpretable".  Two are DATA channels, so by the rule quoted
# in selftest (m) any science run there is INVALID however many shots it buys.
# They are the same channels the degenerate counts implicate: re-read against
# exact_probs, that dump gives TV distance 1.0000 and ZERO overlap with the ideal
# support on all four circuits, with q2/q3 pinned at p1=0.998-1.000 and q8 at
# 0.997-1.000 where the ideal is exactly 0.000.  So the collapse is ~5 of 9 readout
# channels reporting a constant, NOT extra noise: entropy FELL 7.2 -> 2.6 bits and
# distinct outcomes 502 -> 50 while the answer went wrong, which is the opposite of
# a device getting noisier.  Neither re-routing (adds depth, raises entropy) nor a
# bit permutation (preserves entropy) can produce it, and gcd(counts)=1 excludes a
# rescaled histogram.
# The "near-identical z ~= -36.5 across two frames and two depths" above is likewise
# explained rather than mysterious: z = (p_s - n/16)/sd at n=20000 gives exp=1250 and
# sd=34.23, so an EMPTY expected-syndrome bin yields exactly -36.52, the statistic's
# floor.  All four circuits sit at the floor because dead data channels make the
# correct syndrome unattainable, not because the circuits are alike.
# Onset is bracketed, not guessed: the same block returned healthy suppl data at
# 2026-10-24T12:03Z (ratio 0.793/0.802, 502 distinct, ideal overlap 0.021) and
# degenerate data at 14:18Z, and was still NOT USABLE 12 h later -- a sustained
# fault, unlike the 2026-10-12 dead q3, which had recovered by the 2026-10-13
# screen.  The 14:18 submission carried NO prior screen on that block, while the
# valid 12:09 run on [52,61,62,70,71,79,80,88,89] was preceded by a PASSING screen
# on that same block at 12:05:12Z.  That is the whole difference, and it is why
# launch_mixedness.sh's suppl branch says to run `screen` first: an explicit --block
# bypasses resolve_block entirely, so nothing else validates it before the job does.
# Not applied across stages: cal9screen circuits are meant to be near-deterministic
# (entropy 0.238-1.042 bits) and warmcold legitimately reaches down to 1.464, so a
# uniform threshold would misfire on valid data.
SUPPL_MIN_OUTCOME_ENTROPY_RATIO = 0.45


def suppl_gate_inventory():
    """The 2026-09 gate inventory, read from the artifact that records it.

    Deliberately NOT restated as literals beside SUPPL_TARGETS: a circuit whose
    size is checked against a copy of the number it is supposed to reproduce can
    drift from the real one without anything looking wrong, which is the same
    failure mode warmcold_frame_of avoids by reaching into SYND_TABLE."""
    path = os.path.join(ROOT, 'hw_error_budget.json')
    with open(path) as f:
        return json.load(f)['gate_inventory']


def build_suppl(phi_cold):
    """The four 2026-09 feasibility circuits, rebuilt from the same builder.

    Cold angles only, and psi is fixed to SUPPL_PSI='0' rather than taken from
    WARMCOLD_PSI: the point of this stage is to reproduce a specific archived run,
    and that run's input state was identified from its own ideal references, not
    guessed.  Passing WARMCOLD_PSI here would silently rebuild a DIFFERENT
    experiment and compare it to the September numbers anyway -- exactly the psi
    confound this stage exists to remove.  The ideal reference is checked against
    the archived value at build time, so a circuit that is not bit-exact cannot be
    submitted at all."""
    out, meta = [], []
    inv = suppl_gate_inventory()
    for label, measure, frame, s in SUPPL_GRID:
        assert warmcold_frame_of(s) == frame, (label, s, frame, warmcold_frame_of(s))
        gates = q.build_branch_circuit(q.ENC5, SUPPL_PSI, frame, s, phi_cold,
                                       late_measure=(measure == 'late'))
        ideal = ideal_data_zero_prob(SUPPL_PSI, frame, s, phi_cold)
        tgt = SUPPL_TARGETS[label]['ideal']
        assert abs(ideal - tgt) < SUPPL_IDEAL_TOL, (
            '%s: rebuilt ideal %.16f does not match the 2026-09 artifact\'s '
            '%.16f (diff %.2e >= %.0e), so this is NOT the same circuit and '
            'comparing it to the September numbers would be meaningless'
            % (label, ideal, tgt, abs(ideal - tgt), SUPPL_IDEAL_TOL))
        # Size and entangling content, against the archived inventory.  The ideal
        # reference alone would not catch a rebuild that reached the same number
        # by a different route -- a different decomposition of R_zz, or a vendor
        # option that inserts gates -- and the whole point is that this is the
        # SAME circuit at the same depth, so that any difference in the outcome
        # is in the device or the month, not in the compilation.
        ref = inv[label]
        n_cnot = sum(1 for x in gates if x[0] == 'CNOT')
        assert len(gates) == ref['instructions'], (
            '%s: rebuilt at %d instructions, the 2026-09 inventory says %d'
            % (label, len(gates), ref['instructions']))
        assert n_cnot == ref['two_qubit_gates'], (
            '%s: rebuilt with %d CNOT, the 2026-09 inventory says %d two-qubit '
            'gates' % (label, n_cnot, ref['two_qubit_gates']))
        out.append(gates)
        meta.append(dict(kind='suppl', label=label, reproduces_2026_09=label,
                         psi=SUPPL_PSI, p=0.0, frame=frame, s=s, measure=measure,
                         angles=WARMCOLD_ANGLES['cold'],
                         n_instructions=len(gates), n_cnot=n_cnot,
                         matches_2026_09_inventory=True,
                         ideal_P_data_zero=ideal,
                         ideal_matches_2026_09_to=abs(ideal - tgt)))
    return out, meta


def stage_circuits(stage, phi=None, phi_cold=None, cal_first=False):
    """(gate_lists, meta) for one cloud job.  meta entries are plain dicts so they
    survive the JSON dump and can be rebuilt identically in --mode analyze."""
    if stage == 'probe':
        out = [build_cal('zero'), build_cal('one'), build_cal('zero')]
        meta = [dict(kind='cal', cal='zero', rep=0), dict(kind='cal', cal='one', rep=0),
                dict(kind='cal', cal='zero', rep=1)]
        for b in range(NQ5):
            out.append(build_fingerprint(b))
            meta.append(dict(kind='fingerprint', qubit=b))
        return out, meta
    if stage == 'tomo':
        assert sorted(TOMO_STATE_PRIORITY) == sorted(TOMO_STATES), \
            'TOMO_STATE_PRIORITY must be a reordering of TOMO_STATES, not a subset'
        # ORDER IS A BUDGET ARTEFACT, not cosmetics.  Under a hard real-time
        # allocation the job can be killed mid-run; submit_cloud dumps the raw
        # counts after every chunk and STAGE_CHUNK['tomo'] = 3 puts the chunk
        # boundaries at 3/6/9/12/15/18/19, so each completed chunk is a
        # scientifically complete prefix (see the STAGE_CHUNK comment).  The
        # calibration therefore comes FIRST -- a truncated dump holding data but no
        # calibration is the worst possible outcome, because then even the circuits
        # that did return cannot be readout-mitigated -- then the depth-1 grid
        # grouped by input state, then the depth-2 scan last as a pure bonus.  The
        # grouping is per STATE rather than per basis because analyze_tomo needs all
        # three bases of a state for that state's radius, and all four states for
        # M2: a basis-major order would leave a half-measured state at every
        # boundary and make every truncation unanalysable.
        out = [build_cal('zero'), build_cal('one'), build_cal('zero')]
        meta = [dict(kind='cal', cal='zero', rep=0),
                dict(kind='cal', cal='one', rep=0),
                dict(kind='cal', cal='zero', rep=1)]
        for psi in TOMO_STATE_PRIORITY:
            for basis in BASES:
                out.append(build_tomo(psi, basis, rounds=1))
                meta.append(dict(kind='tomo', psi=psi, basis=basis, rounds=1))
        for psi, basis in DEPTH_SCAN:
            out.append(build_tomo(psi, basis, rounds=2))
            meta.append(dict(kind='tomo', psi=psi, basis=basis, rounds=2))
        # The tomography job carries its OWN three calibration circuits.  A
        # submission has one block, and the block is whatever the platform ranks
        # best at submission time, so a job that depends on a calibration measured
        # in an earlier job silently assumes the two landed on the same physical
        # qubits.  Carrying them costs three 5-10 instruction circuits and makes
        # the headline M1/M2/M3 numbers self-contained on whichever chip ran them
        # -- which is also what lets the whole ~300 s budget go to this one stage
        # instead of first spending 34 s on the probe canary.
        return out, meta
    if stage == 'ladder':
        # The DEPTH_SCAN grid at deeper rounds: for psi=|0> the Bloch vector lies
        # along z and for psi=|+> along x, so (0,Z) and (+,X) are the radius at any
        # depth, while (0,X) and (+,Z) remain null controls at any depth.  Same
        # meta kind as the tomo stage, so --stage all merges them into one depth
        # curve and analyze_tomo's loops pick up every rounds value present.
        out, meta = [], []
        for r in LADDER_ROUNDS:
            for psi, basis in DEPTH_SCAN:
                out.append(build_tomo(psi, basis, rounds=r))
                meta.append(dict(kind='tomo', psi=psi, basis=basis, rounds=r))
        return out, meta
    if stage == 'bench':
        assert phi is not None, 'the bench stage needs the trained angles'
        g1, m1 = build_bench_ab(phi)
        g2, m2 = build_bench(phi)
        return g1 + g2, m1 + m2
    if stage == 'warmcold':
        assert phi is not None and phi_cold is not None, (
            'the warmcold stage needs BOTH angle sets: the warm arm is the file '
            'the manuscript reports (%s) and the cold arm is the from-zero '
            'control (%s)' % (WARMCOLD_ANGLES['warm'], WARMCOLD_ANGLES['cold']))
        gates, meta = build_warmcold(phi, phi_cold)
        if not cal_first:
            return gates, meta
        # CAL-FIRST ORDER.  Puts the two nine-qubit readout-calibration circuits
        # ahead of the four science circuits, which with chunk = WARMCOLD_N_CAL
        # makes the calibration exactly submission 1.
        #
        # Why this is now available and was not before: cal9's channel labels were
        # believed not to transfer to the science circuits, so screening on cal9
        # would have said nothing about the block the science circuits land on.
        # That belief is retracted -- the apparent violations were an artefact of a
        # wrong key-position lookup, and all 36 comparisons lie inside their bounds
        # (see check_cal9_transferability).  cal9 therefore DOES characterise the
        # science block, and the 2026-10-12 run's dead data channel (logical q3,
        # contrast -0.156) would have been visible before any science circuit was
        # paid for.
        #
        # What cal-first does NOT do is make the calibration cheap.  Billing is per
        # SHOT, not per instruction, so at a common shot count the two 9- and
        # 18-instruction calibration circuits cost exactly what two 206-instruction
        # science circuits cost.  Their advantage is that they are interpretable on
        # their own and decide whether the block is usable at all.  A genuinely
        # cheap screen needs cal9 submitted as its own job at reduced shots
        # (1000/circuit is ample for e01/e10), which is a separate submission
        # rather than a reordering of this one.
        #
        # The default stays science-first so the 2026-10-12 artifact remains
        # reproducible from the same command line that produced it.
        order = sorted(range(len(meta)),
                       key=lambda i: (0 if meta[i].get('kind') == 'cal9' else 1, i))
        return ([gates[i] for i in order], [meta[i] for i in order])
    if stage == 'suppl':
        assert phi_cold is not None, (
            'the suppl stage reproduces the 2026-09 feasibility set, which ran the '
            'COLD angle set (%s) only; it needs --angles or the default cold file'
            % WARMCOLD_ANGLES['cold'])
        return build_suppl(phi_cold)
    raise ValueError(stage)


# ---------------------------------------------------------------------
# 2) exact ideal references (no account, no sampling)
# ---------------------------------------------------------------------
def canonical_key(idx, n=NQ5):
    """Basis index -> bitstring with char j = qubit j.  In qcloud_vscr_new's kron
    order qubit 0 is the MOST significant bit, so qubit j sits at n-1-j."""
    return ''.join(str((idx >> (n - 1 - j)) & 1) for j in range(n))


def unitary_of(gates, n=NQ5):
    """Exact unitary of a gate list, measurements dropped."""
    u = [g for g in gates if g[0] != 'MEAS']
    return q.gates_to_unitary(u, n=n)


def exact_probs(gates, n=NQ5):
    """Ideal output distribution of `gates` applied to |0...0>, as a dict keyed by
    the CANONICAL bitstring (char j = qubit j)."""
    U = unitary_of(gates, n)
    p = np.abs(U[:, 0]) ** 2
    return {canonical_key(i, n): float(p[i]) for i in range(2 ** n)}


def ideal_bloch_of(gates, n=NQ5):
    """(code-space weight, Bloch vector) of the exact output state, read the way
    the hardware estimator reads it: project on q0..q3 = 0000, renormalise, then
    take the three Pauli expectations of q4.  Used to check that the estimator and
    the state-vector agree before either is trusted with real counts."""
    U = unitary_of(gates, n)
    psi = U[:, 0]
    P = np.abs(psi) ** 2
    cs = [i for i in range(2 ** n) if (i >> 1) & 15 == 0]     # q0..q3 == 0
    w = float(sum(P[i] for i in cs))
    rl = np.array([[psi[cs[a]] * np.conj(psi[cs[b]]) for b in (0, 1)]
                   for a in (0, 1)])
    rl = rl / w
    nvec = np.array([float(np.trace(Pm @ rl).real) for Pm in PAULI5])
    return w, nvec



# ---------------------------------------------------------------------
# 3) bitstring layout, detected by maximum likelihood against the exact ideal
#    distributions.  The old qcloud_vscr_new.detect_layout scored four hardcoded
#    hypotheses by data-hit count, which is enough for a 9-cbit circuit whose
#    ideal outcome is a single key.  A 5-cbit tomography circuit has a richer
#    ideal distribution, so the layout can be pinned over ALL 5! = 120
#    assignments of observed character -> qubit, with a likelihood margin that
#    goes into the artifact the way ED Table 4's evidence table does.
# ---------------------------------------------------------------------
def str_to_idx(key):
    return int(key, 2)


def idx_to_str(i, n=NQ5):
    return format(i, '0%db' % n)


def _remap(key, sigma):
    """Observed string -> canonical string (char b = qubit b) under the hypothesis
    that observed character j came from qubit sigma[j]."""
    out = ['0'] * len(key)
    for j, ch in enumerate(key):
        out[sigma[j]] = ch
    return ''.join(out)


def layout_hypotheses(n=NQ5):
    return [tuple(s) for s in itertools.permutations(range(n))]


def detect_layout_ml(results, ideals, n=NQ5, top=5, tie_tol=1e-6, verbose=True):
    """Score every permutation by the multinomial log-likelihood of the observed
    counts under the exact ideal distributions.

    `ideals[i]` is exact_probs() of circuit i, keyed by the CANONICAL string.
    Returns (sigma, evidence).

    Ties are reported, not silently broken.  On the tomography circuits alone the
    ideal distributions cannot separate the four code-space positions (see
    build_fingerprint), so the tie set is expected to have 4! = 24 members there;
    what MUST be determined is the position of the logical qubit, because that is
    the only thing any estimator in this file actually reads.  `evidence` records
    whether the tie set determines it, and the function raises if it does not, so
    an ambiguous layout can never reach the artifact looking like a resolved one.
    The representative returned is the lexicographically smallest member of the tie
    set, which makes the choice reproducible rather than arbitrary."""
    scores = []
    for sigma in layout_hypotheses(n):
        ll = 0.0
        for counts, ideal in zip(results, ideals):
            for key, cnt in counts.items():
                if cnt <= 0 or len(key) != n:
                    continue
                ll += cnt * math.log(max(ideal.get(_remap(key, sigma), 0.0),
                                         LAYOUT_FLOOR))
        scores.append((ll, sigma))
    scores.sort(key=lambda t: (-t[0], t[1]))
    best_ll, _ = scores[0]
    tie = [sg for ll, sg in scores if best_ll - ll <= tie_tol]
    rest = [t for t in scores if t[1] not in tie]
    runner_ll, runner = rest[0] if rest else (float('-inf'), None)
    pos_of = {sg: [j for j in range(n) if sg[j] == n - 1][0] for sg in tie}
    determines_logical = len(set(pos_of.values())) == 1
    determines_codespace = len({frozenset(j for j in range(n)
                                          if sg[j] != n - 1) for sg in tie}) == 1
    if not determines_logical:
        raise RuntimeError(
            'the ideal distributions do not determine which observed position '
            'carries the logical qubit: %d tied hypotheses disagree on it (%s). '
            'Add build_fingerprint() circuits to this job.'
            % (len(tie), sorted(set(pos_of.values()))))
    best = min(tie)
    ev = dict(n_hypotheses=len(scores), best_loglik=best_ll,
              best_sigma=list(best), logical_position=pos_of[best],
              n_tied=len(tie), tie_set=[list(s) for s in sorted(tie)],
              tie_determines_logical=bool(determines_logical),
              tie_determines_codespace=bool(determines_codespace),
              tie_tol=tie_tol,
              runner_up_loglik=runner_ll,
              runner_up_sigma=(list(runner) if runner is not None else None),
              margin=(best_ll - runner_ll) if rest else None,
              top=[dict(loglik=s, sigma=list(sg)) for s, sg in scores[:top]])
    if verbose:
        print('[layout] sigma=%s loglik=%.1f  %d tied hypotheses (all agree the '
              'logical qubit is at observed position %d)  margin to the rest=%s'
              % (list(best), best_ll, len(tie), pos_of[best],
                 ('%.1f' % ev['margin']) if rest else 'n/a'), flush=True)
    return best, ev


# ---------------------------------------------------------------------
# 4) readout assignment matrix, measured by this run, and its mitigation
# ---------------------------------------------------------------------
def assignment_from_cal(counts_zero, counts_one, n=NQ5):
    """Per-OBSERVED-POSITION flip rates: e01[j] = P(char j reads 1 | true 0),
    e10[j] = P(char j reads 0 | true 1).  Position-based on purpose: mitigation
    needs no layout, only the later interpretation of positions does."""
    def marg(counts, val):
        tot = float(sum(counts.values()))
        return [sum(c for key, c in counts.items()
                    if len(key) == n and key[j] == val) / tot if tot > 0
                else float('nan') for j in range(n)]
    return dict(e01=marg(counts_zero, '1'), e10=marg(counts_one, '0'),
                n_zero=int(sum(counts_zero.values())),
                n_one=int(sum(counts_one.values())))


def assignment_matrix(e01, e10, n=NQ5):
    """The 2^n x 2^n independent-bit assignment matrix A with p_obs = A p_true.
    Kronecker order matches int(key, 2): position j is bit n-1-j."""
    A = np.array([[1.0]])
    for j in range(n):
        A = np.kron(A, np.array([[1.0 - e01[j], e10[j]],
                                 [e01[j], 1.0 - e10[j]]]))
    return A


def mitigate(probs_vec, e01, e10, n=NQ5, clip=True):
    """Invert the assignment matrix on a probability vector indexed by int(key, 2).

    Returns (p_true, diagnostics).  Linear inversion of a measured assignment
    matrix can leave the simplex; that is reported rather than hidden, and the
    projection `clip` applies is named in the diagnostics.  Kept for the
    distribution-level diagnostic -- every ESTIMATOR in this file goes through
    `observable_weights` instead, because clipping a noisy inversion is biased and
    the bias shows up as a spurious several-sigma violation of M2."""
    A = assignment_matrix(e01, e10, n)
    b = np.asarray(probs_vec, dtype=float)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    diag = dict(negativity_before_clip=float(-min(0.0, x.min())),
                max_refit_residual=float(np.abs(A @ x - b).max()),
                clipped=bool(clip), sum_before_clip=float(x.sum()))
    if clip:
        x = np.clip(x, 0.0, None)
        s = x.sum()
        if s > 0:
            x = x / s
    return x, diag


def observable_weights(O_true, e01, e10, n=NQ5):
    """The observed-side weight vector of a diagonal observable: O~ = A^{-T} O.

    Then <O>_true = sum_j O~[j] p_obs[j] = the sample mean of O~ over the shots,
    which is EXACTLY unbiased for any per-bit readout channel, needs no inversion of
    a distribution, and never needs clipping.  Its sampling variance follows from
    the same weights, so the mitigated numbers get proper standard errors instead of
    borrowing the raw ones."""
    A = assignment_matrix(e01, e10, n)
    return np.linalg.solve(A.T, np.asarray(O_true, dtype=float))


def true_bit_masks(sigma, n=NQ5):
    """For each qubit b, the bitmask of ITS TRUE value inside a position-space index
    (the index int(key, 2) of a string whose character j came from qubit sigma[j])."""
    return [1 << (n - 1 - j) for j, b in sorted(
        [(j, b) for j, b in enumerate(sigma)], key=lambda t: t[1])]


def observables_for(sigma, n=NQ5):
    """The four true-side diagonal observables this experiment needs, as vectors over
    position-space indices: the code-space indicator, the logical Z, their product
    (the numerator of the conditioned component), and the all-ones trace."""
    masks = true_bit_masks(sigma, n)
    cs_mask = 0
    for b in range(n - 1):
        cs_mask |= masks[b]
    idx = np.arange(2 ** n)
    o_cs = ((idx & cs_mask) == 0).astype(float)
    o_z = np.where((idx & masks[n - 1]) == 0, 1.0, -1.0)
    return dict(cs=o_cs, z=o_z, num=o_z * o_cs, one=np.ones(2 ** n))


def expectation(counts_vec, weights):
    """(mean, standard error) of a mitigated observable from a count vector.

    The standard error is the exact i.i.d. sampling error of the sample mean of the
    weights, sqrt((<O~^2> - <O~>^2)/N) -- no binomial approximation, and valid for
    the conditioned ratio's numerator and denominator alike."""
    N = float(counts_vec.sum())
    if N <= 0:
        return float('nan'), float('nan')
    p = counts_vec / N
    m = float(weights @ p)
    var = float((weights ** 2) @ p) - m * m
    return m, math.sqrt(max(var, 0.0) / N)


def wilson(k, n, z=1.96):
    """Wilson score interval -- the same estimator ED Table 3 and the abstract use
    for the hardware branch fidelity."""
    if n <= 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0))
    return ((c - h) / d, (c + h) / d)


# ---------------------------------------------------------------------
# 5) estimators: code-space weight, Bloch components, radius
# ---------------------------------------------------------------------
def qubit_masks(sigma, n=NQ5):
    """For a layout hypothesis sigma (observed char j came from qubit sigma[j]),
    the bitmask of each qubit inside an index parsed by int(key, 2)."""
    inv = {b: j for j, b in enumerate(sigma)}
    return [1 << (n - 1 - inv[b]) for b in range(n)]


def cs_split(sigma, n=NQ5):
    """(code-space index list, q4 bitmask).  The code space after decoding is
    'q0..q3 all zero', which IS the projector V V^dag that device_prep applies."""
    masks = qubit_masks(sigma, n)
    cs_mask = 0
    for b in range(n - 1):
        cs_mask |= masks[b]
    cs = [i for i in range(2 ** n) if (i & cs_mask) == 0]
    return cs, masks[n - 1]


def counts_to_vec(counts, n=NQ5):
    v = np.zeros(2 ** n)
    for key, c in counts.items():
        if len(key) == n:
            v[str_to_idx(key)] += float(c)
    return v


def _weights(O, e01, e10, n, mitigated):
    if not mitigated:
        return {k: O[k] for k in O}
    return {k: observable_weights(O[k], e01, e10, n) for k in O}


def _ratio_stats(v, W):
    """(w, num, sigma_cs) with exact sampling errors, including the correlation
    between the numerator and the post-selection denominator -- they come from the
    same shots, and dropping that covariance understates the error on the ratio."""
    N = float(v.sum())
    p = v / N if N > 0 else v
    w = float(W['cs'] @ p)
    num = float(W['num'] @ p)
    var_w = max(float((W['cs'] ** 2) @ p) - w * w, 0.0) / max(N, 1)
    var_n = max(float((W['num'] ** 2) @ p) - num * num, 0.0) / max(N, 1)
    cov = (float((W['num'] * W['cs']) @ p) - num * w) / max(N, 1)
    if w > 1e-12:
        s = num / w
        var_s = (var_n / w ** 2 - 2 * num * cov / w ** 3
                 + num ** 2 * var_w / w ** 4)
        se = math.sqrt(max(var_s, 0.0))
    else:
        s, se = float('nan'), float('nan')
    z = float(W['z'] @ p)
    se_z = math.sqrt(max(float((W['z'] ** 2) @ p) - z * z, 0.0) / max(N, 1))
    return dict(w=w, w_stderr=math.sqrt(var_w), num=num, sigma_cs=s,
                sigma_cs_stderr=se, sigma_uncond=z, sigma_uncond_stderr=se_z,
                norm_check=float(W['one'] @ p))


def read_tomo(counts, sigma, n=NQ5, e01=None, e10=None):
    """Everything one tomography circuit yields, raw and readout-mitigated.

    Both variants are sample means of a diagonal observable, so both are unbiased
    and both carry their own exact sampling error; the mitigated one uses
    O~ = A^{-T} O rather than an inverted-and-clipped distribution (see
    observable_weights).  `sigma_cs` is the conditioned Bloch component that M1
    reads; `sigma_uncond` is the UNCONDITIONAL logical <Z_L> that M2 needs, which
    must not be post-selected because post-selection is not a linear functional of
    the input state."""
    O = observables_for(sigma, n)
    v = counts_to_vec(counts, n)
    out = dict(n_used=int(v.sum()), n_requested=None)
    mit = e01 is not None and e10 is not None
    out['raw'] = _ratio_stats(v, _weights(O, e01, e10, n, False))
    k = int(round(out['raw']['w'] * out['n_used']))
    out['raw']['w_wilson95'] = list(wilson(k, out['n_used']))
    out['raw']['w_ci95'] = [out['raw']['w'] - 1.96 * out['raw']['w_stderr'],
                            out['raw']['w'] + 1.96 * out['raw']['w_stderr']]
    if mit:
        out['mitigated'] = _ratio_stats(v, _weights(O, e01, e10, n, True))
        out['mitigated']['w_wilson95'] = None
        out['mitigated']['w_ci95'] = [
            out['mitigated']['w'] - 1.96 * out['mitigated']['w_stderr'],
            out['mitigated']['w'] + 1.96 * out['mitigated']['w_stderr']]
        p = v / v.sum() if v.sum() > 0 else v
        _pm, diag = mitigate(p, e01, e10, n)
        out['mitigation_diagnostics'] = diag
    return out


def bootstrap_radius(counts_by_basis, sigma, n=NQ5, e01=None, e10=None,
                     n_boot=2000, seed=20260921, variant='mitigated'):
    """Monte-Carlo CI on the Bloch radius r = |(<s_X>,<s_Y>,<s_Z>)| and on the mean
    code-space weight.  Resamples each basis circuit multinomially from its own
    observed counts and reruns the estimator, so the ratio's correlation, the
    mitigation and the norm of the three components all propagate together."""
    rng = np.random.default_rng(seed)
    O = observables_for(sigma, n)
    W = _weights(O, e01, e10, n, variant == 'mitigated')
    obs = {}
    for B, counts in counts_by_basis.items():
        v = counts_to_vec(counts, n)
        obs[B] = (v, max(v.sum(), 1.0))
    rs, ws = [], []
    for _ in range(n_boot):
        comps, wsum = [], []
        for B in BASES:
            v, tot = obs[B]
            s = rng.multinomial(int(tot), v / tot).astype(float)
            st = _ratio_stats(s, W)
            comps.append(st['sigma_cs'] if st['sigma_cs'] == st['sigma_cs'] else 0.0)
            wsum.append(st['w'])
        rs.append(float(np.linalg.norm(comps)))
        ws.append(float(np.mean(wsum)))
    rs, ws = np.array(rs), np.array(ws)

    def pct(a, f):
        return float(np.percentile(a, f))
    return dict(radius=pct(rs, 50), radius_ci95=[pct(rs, 2.5), pct(rs, 97.5)],
                radius_mean=float(rs.mean()), radius_std=float(rs.std(ddof=1)),
                w_mean=pct(ws, 50), w_ci95=[pct(ws, 2.5), pct(ws, 97.5)],
                w_std=float(ws.std(ddof=1)),
                n_boot=n_boot, seed=seed, variant=variant)


# ---------------------------------------------------------------------
# 6) the isotropic-depolarizing model at the HARDWARE's own depth, and the
#    one-parameter fit that turns M1 into a test of the surrogate (M3)
# ---------------------------------------------------------------------
def _kraus(p):
    I2 = np.eye(2, dtype=complex)
    return [math.sqrt(1.0 - p) * I2] + [math.sqrt(p / 3.0) * s for s in PAULI5]


def codespace_projector():
    """32x2 orthonormal basis of the [[5,1,3]] code space, from ssvr_qec's own
    logical vectors.  Only its column SPACE matters: w = Tr(V^dag rho V) and
    r^2 = 2 Tr(rho_L^2) - 1 are both invariant under a change of basis inside the
    code space, which is what makes the anchor below a statement about physics
    rather than about a convention."""
    import ssvr_qec as m
    V = np.stack([m.ZERO_L.numpy(), m.ONE_L.numpy()], axis=1)
    G = V.conj().T @ V
    assert np.abs(G - np.eye(2)).max() < 1e-12, G
    return V


def _propagate(gates, p_instr=None, p1=None, p2=None, psi_vec=None, n=NQ5):
    """rho after `gates`, with a depolarizing channel after every instruction on
    every qubit it touches.  Shared by model_mixedness and model_output_probs so the
    two can never drift apart."""
    if psi_vec is None:
        psi_vec = np.zeros(2 ** n, dtype=complex)
        psi_vec[0] = 1.0
    rho = np.outer(psi_vec, np.asarray(psi_vec, dtype=complex).conj())
    for g in gates:
        if g[0] == 'MEAS':
            continue
        if g[0] == 'CNOT':
            U = q.embed2(q.CNOT4, g[1], g[2], n=n)
            qs, p = [g[1], g[2]], (p2 if p_instr is None else p_instr)
        else:
            U = q.embed1(q._gate_matrix(g), g[1], n=n)
            qs, p = [g[1]], (p1 if p_instr is None else p_instr)
        rho = U @ rho @ U.conj().T
        if p > 0.0:
            # one qubit's channel at a time, exactly as device_prep does it.
            # Applying the union of the Kraus sets as a single flat list would
            # treat "depolarise a" and "depolarise b" as ALTERNATIVES instead of
            # composing them, and would multiply Tr(rho) by 2 per two-qubit gate.
            for qb in qs:
                K = [q.embed1(k, qb, n=n) for k in _kraus(p)]
                rho = sum(k @ rho @ k.conj().T for k in K)
            tr = float(np.trace(rho).real)
            assert abs(tr - 1.0) < 1e-9, tr
    return rho


def model_output_probs(gates, p1, p2, psi_vec=None, n=NQ5):
    """The modelled output distribution over canonical bitstrings -- what a device
    with these fidelities would hand back, before readout error."""
    rho = _propagate(gates, p1=p1, p2=p2, psi_vec=psi_vec, n=n)
    d = np.diag(rho).real
    return {canonical_key(i, n): float(d[i]) for i in range(2 ** n)}


def model_mixedness(gates, p_instr=None, p1=None, p2=None, psi_vec=None,
                    proj='decode', n=NQ5):
    """Propagate rho through `gates` with a depolarizing channel after every
    instruction, on every qubit that instruction touches, then project.

    This is `mixed_input_boundary.device_prep`'s model, re-implemented on an
    ARBITRARY compiled gate list and as an explicit 32x32 density matrix, so it
    can be run at the depth the hardware circuit actually has and so the audit can
    re-derive it independently.  Rates: a single `p_instr` (device_prep's choice,
    and what the M3 fit produces), or per-type `p1`/`p2` from vendor single- and
    two-qubit fidelities.  proj='decode' projects onto q0..q3=0 after the gate
    list has decoded; proj='iso' applies V V^dag, which is what device_prep does
    for an encoder-only list."""
    assert (p_instr is not None) != (p1 is not None and p2 is not None), \
        'give either p_instr or both of p1 and p2'
    rho = _propagate(gates, p_instr=p_instr, p1=p1, p2=p2, psi_vec=psi_vec, n=n)
    if proj == 'decode':
        cs = [0, 1]                       # q0..q3 == 0, q4 free (q4 is the LSB)
        w = float(np.trace(rho[np.ix_(cs, cs)]).real)
        rl = rho[np.ix_(cs, cs)]
    elif proj == 'iso':
        V = codespace_projector()
        rl = V.conj().T @ rho @ V
        w = float(np.trace(rl).real)
    else:
        raise ValueError(proj)
    assert w > 0.0, 'the model predicts zero code-space weight'
    rl = rl / w
    nvec = np.array([float(np.trace(Pm @ rl).real) for Pm in PAULI5])
    r = float(np.linalg.norm(nvec))
    purity = float(np.trace(rl @ rl).real)
    # the UNCONDITIONAL logical <Z_L>, taken before any projection: this is the
    # linear functional M2 uses, and it is linear in rho_in by construction.
    Z4 = q.embed1(PAULI5[2], n - 1, n=n)
    sz = float(np.trace(Z4 @ rho).real)
    return dict(radius=r, codespace_weight=w, purity=purity, leakage=1.0 - w,
                bloch=nvec.tolist(), n_instructions=len(gates), proj=proj,
                p_instr=p_instr, p1=p1, p2=p2, sigma_z_uncond=sz)


def prep_vec(psi, n=NQ5):
    """The physical |0...0>-based state vector that prep(psi) on q4 produces."""
    U = q.gates_to_unitary(list(q.TEST_STATES[psi][0]), n=n)
    return U[:, 0]


def fit_p_instr(gates, w_target, psi_vec, proj='decode', n=NQ5,
                lo=0.0, hi=0.25, iters=80):
    """The single p_instr whose modelled code-space weight equals the measured one.

    Bisection: w(p) is continuous and strictly decreasing on [0, 0.25] (asserted
    at the endpoints), so the solution is unique.  This is the same inversion
    hw_error_budget.py performs to get the implied mean instruction error, only
    here the measured quantity is a code-space weight rather than a syndrome
    marginal, and the fitted rate is then used to PREDICT r -- one fitted
    parameter, one independent prediction."""
    w_lo = model_mixedness(gates, p_instr=lo, psi_vec=psi_vec, proj=proj, n=n)
    w_hi = model_mixedness(gates, p_instr=hi, psi_vec=psi_vec, proj=proj, n=n)
    # w(lo) is exactly 1 for a noiseless gate list, and a device good enough to
    # land there -- or a noiseless dry run -- must return p_instr = lo rather than
    # fail an assertion, so the endpoint is handled explicitly.
    if w_target >= w_lo['codespace_weight'] - 1e-12:
        return lo
    assert w_lo['codespace_weight'] > w_target > w_hi['codespace_weight'], \
        (w_lo['codespace_weight'], w_target, w_hi['codespace_weight'])
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        wm = model_mixedness(gates, p_instr=mid, psi_vec=psi_vec, proj=proj,
                             n=n)['codespace_weight']
        if wm > w_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------
# 7) self-tests -- every estimator this module uses, anchored before it touches
#    a single real count.  No account, no QPU time, no artifact required.
# ---------------------------------------------------------------------
def _counts_from_probs(probs, N):
    """Deterministic integer counts from an exact distribution (rounding error
    ~1/N, which is what lets test (b) use a tight tolerance)."""
    out = {}
    for key, p in probs.items():
        c = int(round(p * N))
        if c > 0:
            out[key] = c
    return out


def _tomo_pairs():
    return ([(psi, B, 1) for psi in TOMO_STATES for B in BASES]
            + [(psi, B, 2) for psi, B in DEPTH_SCAN])


def selftests(verbose=True):
    res = {}

    def say(msg):
        if verbose:
            print('    ' + msg, flush=True)

    # (a) every tomography circuit is IDEALLY a perfect readout of the input
    #     Bloch vector, at both depths -- including the two null controls, whose
    #     ideal component is exactly zero.  The circuit ends with R_B, so the
    #     component that carries the signal is the Z component of the ROTATED
    #     output state, and the two transverse ones must vanish; that is exactly
    #     what the hardware estimator reads as <sigma_B>.  Only the Z component of
    #     the rotated state is the measured quantity; the two transverse ones carry
    #     the input's other Bloch components, so the invariants to assert are that
    #     the Z component is exact, the state is still pure (|n| = 1) and the
    #     code-space weight is exactly 1.
    worst_ideal = 0.0
    for psi, B, rounds in _tomo_pairs():
        gates = build_tomo(psi, B, rounds)
        w, nvec = ideal_bloch_of(gates)
        want = IDEAL_BLOCH[psi][BASIS_AXIS[B]]
        worst_ideal = max(worst_ideal, abs(w - 1.0), abs(nvec[2] - want),
                          abs(np.linalg.norm(nvec) - 1.0))
    assert worst_ideal < 1e-12, worst_ideal
    res['ideal_bloch_residual'] = worst_ideal
    say('(a) %d tomography circuits: exact code-space weight 1 and exact ideal '
        'Bloch component (2 of them null), worst residual %.2e'
        % (len(_tomo_pairs()), worst_ideal))

    # (b) the COUNT-based estimator agrees with the state-vector estimator on the
    #     same ideal distribution, so a hardware number is not an artifact of how
    #     the counts are sliced.
    sigma_id = tuple(range(NQ5))
    worst_est = 0.0
    for psi, B, rounds in _tomo_pairs():
        gates = build_tomo(psi, B, rounds)
        counts = _counts_from_probs(exact_probs(gates), 10 ** 7)
        rec = read_tomo(counts, sigma_id)['raw']
        worst_est = max(worst_est, abs(rec['w'] - 1.0))
        want = IDEAL_BLOCH[psi][BASIS_AXIS[B]]
        worst_est = max(worst_est, abs(rec['sigma_cs'] - want),
                        abs(rec['sigma_uncond'] - want))
    assert worst_est < 1e-6, worst_est
    res['estimator_vs_statevector'] = worst_est
    say('(b) count estimator vs exact state vector over all %d circuits: worst '
        'residual %.2e' % (len(_tomo_pairs()), worst_est))

    # (c) ANCHOR: the re-implemented model reproduces the AUDITED device_prep
    #     numbers in mixed_input_boundary.json at the recorded p_instr, on the
    #     encoder-only gate list with the V V^dag projection device_prep uses.
    mb = json.load(open(os.path.join(ROOT, 'mixed_input_boundary.json')))
    dp = mb['device_prep']
    enc = q.ENC5
    assert len(enc) == dp['n_encoder_instructions'], len(enc)
    worst_anchor = 0.0
    for b, label in ((0, '0_L'), (1, '1_L')):
        pv = np.zeros(DIM5, dtype=complex)
        pv[b] = 1.0
        got = model_mixedness(enc, p_instr=dp['p_instr'], psi_vec=pv, proj='iso')
        ref = dp[label]
        for key in ('radius', 'codespace_weight', 'purity', 'leakage'):
            worst_anchor = max(worst_anchor, abs(got[key] - ref[key]))
    assert worst_anchor < 1e-9, worst_anchor
    res['anchor_device_prep'] = worst_anchor
    say('(c) model re-implementation vs the AUDITED device_prep block at '
        'p_instr=%.12f: worst |diff| on r, w, purity, leakage = %.2e'
        % (dp['p_instr'], worst_anchor))

    # (d) readout mitigation is an exact inverse of the assignment matrix it is
    #     given, and it recovers (w, <sigma>) of a distribution that was pushed
    #     through a synthetic readout channel.
    rng = np.random.default_rng(4242)
    worst_mit = 0.0
    for _ in range(6):
        e01 = list(rng.uniform(0.01, 0.09, NQ5))
        e10 = list(rng.uniform(0.01, 0.09, NQ5))
        gates = build_tomo('0', 'Z', 1)
        ptrue = np.zeros(DIM5)
        for key, pr in exact_probs(gates).items():
            ptrue[str_to_idx(key)] = pr
        pobs = assignment_matrix(e01, e10) @ ptrue
        back, diag = mitigate(pobs, e01, e10)
        worst_mit = max(worst_mit, float(np.abs(back - ptrue).max()),
                        diag['negativity_before_clip'])
        cs, _m4 = cs_split(tuple(range(NQ5)))
        wt = float(sum(ptrue[i] for i in cs))
        wb = float(sum(back[i] for i in cs))
        worst_mit = max(worst_mit, abs(wt - wb))
        # the estimator this file actually uses: O~ = A^{-T} O, which must be
        # EXACTLY unbiased on the noisy distribution -- no inversion of p, no clip,
        # hence no bias, which is what the clip-and-renormalise version had
        O = observables_for(tuple(range(NQ5)))
        W = _weights(O, e01, e10, NQ5, True)
        for key in ('cs', 'z', 'num', 'one'):
            worst_mit = max(worst_mit,
                            abs(float(W[key] @ pobs) - float(O[key] @ ptrue)))
        # and in the SYMMETRIC special case e01 = e10 = e it must reduce to the
        # familiar contrast division by (1 - 2e).  It does NOT reduce to a division
        # by (1 - e01 - e10) when the readout is asymmetric -- that was wrong on
        # first writing this test, and the A^{-T} O machinery is what makes the
        # asymmetric case exact instead of approximate.
        m4 = cs_split(tuple(range(NQ5)))[1]
        e_sym = 0.05
        A_sym = assignment_matrix([e_sym] * NQ5, [e_sym] * NQ5)
        po_sym = A_sym @ ptrue
        p0 = float(sum(po_sym[i] for i in range(DIM5) if (i & m4) == 0))
        p1 = float(sum(po_sym[i] for i in range(DIM5) if (i & m4) != 0))
        analytic = (p0 - p1) / (1.0 - 2.0 * e_sym)
        W_sym = _weights(O, [e_sym] * NQ5, [e_sym] * NQ5, NQ5, True)
        worst_mit = max(worst_mit, abs(analytic - float(O['z'] @ ptrue)),
                        abs(analytic - float(W_sym['z'] @ po_sym)),
                        abs(float(W_sym['z'] @ po_sym) - float(W['z'] @ pobs)))
    assert worst_mit < 1e-12, worst_mit
    res['mitigation_roundtrip'] = worst_mit
    say('(d) readout mitigation: the distribution inversion is exact, and the '
        'O~ = A^-T O estimator actually used here is exactly unbiased on a noisy '
        'distribution and reduces to the analytic contrast division for one bit -- '
        'worst residual %.2e' % worst_mit)

    # (e) M2's mathematical content, tested on the model at a NONZERO noise rate:
    #     the unconditional <Z_L> is a linear functional of rho_in, so the two pure
    #     ensembles of I/2 must give the same average exactly, while the individual
    #     terms are O(1).  If this ever fails the hardware claim is meaningless.
    worst_lin, min_span = 0.0, None
    for p in (0.0, 0.005, 0.02, 0.05):
        terms = {}
        # the prep is carried by psi_vec here, so the gate list must NOT also
        # contain it -- build_tomo() folds prep into the list and is used with
        # psi_vec=None instead.  Doing both applies the prep twice.
        g = enc_dec_gates(1) + [('MEAS', i, i) for i in range(NQ5)]
        for psi in ('0', '1', '+', '-'):
            terms[psi] = model_mixedness(g, p_instr=p, proj='decode',
                                         psi_vec=prep_vec(psi))['sigma_z_uncond']
        Lz = 0.5 * (terms['0'] + terms['1'])
        Lx = 0.5 * (terms['+'] + terms['-'])
        worst_lin = max(worst_lin, abs(Lz - Lx))
        # non-vacuity is only claimed where the terms are still O(1): at p=0.05
        # the unconditional <Z_L> has depolarised to ~0.01 for EVERY input, so two
        # zeros agreeing would prove nothing.  The same guard is applied to the
        # hardware numbers, where the span is recorded in the artifact.
        if p <= 0.005:
            span = max(terms.values()) - min(terms.values())
            min_span = span if min_span is None else min(min_span, span)
    assert worst_lin < 1e-12, worst_lin
    assert min_span > 0.5, min_span
    res['linear_ensemble_residual'] = worst_lin
    res['linear_term_span_min'] = min_span
    say('(e) linear score over the two pure ensembles of I/2 at p_instr up to 0.05: '
        'worst |L_Z - L_X| = %.2e while the individual <Z_L> terms span >= %.3f'
        % (worst_lin, min_span))

    # (f) layout detection: (f1) on the tomography circuits ALONE the ideal
    #     distributions leave the four code-space positions undetermined -- 4! tied
    #     hypotheses -- but they DO pin the logical position, and every estimator
    #     in this file must be invariant across the tie set.  (f2) adding the five
    #     fingerprint circuits removes the tie outright.
    rng2 = np.random.default_rng(99)

    def synth(circuits, sigma_true, shots=4000):
        ideals, results = [], []
        for g in circuits:
            idl = exact_probs(g)
            ideals.append(idl)
            v = np.zeros(DIM5)
            for key, pr in idl.items():
                obs = ['0'] * NQ5
                for j in range(NQ5):
                    obs[j] = key[sigma_true[j]]
                v[str_to_idx(''.join(obs))] += pr
            results.append(dict(zip((idx_to_str(i) for i in range(DIM5)),
                                    rng2.multinomial(shots, v))))
        return ideals, results

    tomo_gates = [build_tomo(psi, B, r) for psi, B, r in _tomo_pairs()]
    fp_gates = [build_fingerprint(b) for b in range(NQ5)]
    smallest_margin, worst_invariance = None, 0.0
    for sigma_true in [(0, 1, 2, 3, 4), (4, 3, 2, 1, 0), (2, 0, 4, 1, 3)]:
        ideals, results = synth(tomo_gates, sigma_true)
        sigma_hat, ev = detect_layout_ml(results, ideals, verbose=False)
        assert ev['n_tied'] == 24, ev['n_tied']
        assert ev['tie_determines_logical'] and ev['tie_determines_codespace']
        want_pos = [j for j in range(NQ5) if sigma_true[j] == NQ5 - 1][0]
        assert ev['logical_position'] == want_pos, (ev, want_pos)
        # every estimator must agree across the whole tie set
        ref = read_tomo(results[0], tuple(sigma_hat))
        for sg in ev['tie_set']:
            got = read_tomo(results[0], tuple(sg))
            for var in ('raw',):
                for key in ('w', 'sigma_cs', 'sigma_uncond'):
                    worst_invariance = max(
                        worst_invariance,
                        abs(got[var][key] - ref[var][key]))
        ideals2, results2 = synth(tomo_gates + fp_gates, sigma_true)
        sigma_hat2, ev2 = detect_layout_ml(results2, ideals2, verbose=False)
        assert tuple(sigma_hat2) == tuple(sigma_true), (sigma_hat2, sigma_true)
        assert ev2['n_tied'] == 1, ev2['n_tied']
        smallest_margin = ev2['margin'] if smallest_margin is None else min(
            smallest_margin, ev2['margin'])
    assert worst_invariance < 1e-12, worst_invariance
    assert smallest_margin > 100.0, smallest_margin
    res['layout_tie_invariance'] = worst_invariance
    res['layout_margin_min'] = smallest_margin
    say('(f) layout: tomo circuits alone leave exactly 24 tied hypotheses (the four '
        'code-space positions) but pin the logical one, and w / <sigma_cs> / '
        '<sigma_uncond> are identical across all 24 to %.2e; adding the 5 '
        'fingerprint circuits breaks the tie uniquely in all 3 planted orders, '
        'smallest margin %.0f' % (worst_invariance, smallest_margin))

    # (g) the M3 inversion: a planted p_instr is recovered from the model's own
    #     code-space weight, which is exactly what the fit does with a measured one
    worst_fit = 0.0
    for p_true in (0.005, 0.02, 0.06):
        g = enc_dec_gates(1)
        w = model_mixedness(g, p_instr=p_true, psi_vec=prep_vec('0'),
                            proj='decode')['codespace_weight']
        worst_fit = max(worst_fit,
                        abs(fit_p_instr(g, w, prep_vec('0')) - p_true))
    assert worst_fit < 1e-6, worst_fit
    res['fit_p_instr_residual'] = worst_fit
    say('(g) one-parameter fit of p_instr from the code-space weight recovers a '
        'planted rate to %.2e' % worst_fit)

    # (h) the five cloud stages have the size and architecture the budget assumes
    _z = np.zeros((16, 60))
    sizes = {s: len(stage_circuits(s, phi=_z, phi_cold=_z)[0])
             for s in ('probe', 'tomo', 'bench', 'ladder', 'warmcold')}
    # suppl is sized with the REAL cold angles, not the placeholder zeros every
    # other stage accepts.  It rebuilds the archived 2026-09 circuits and asserts
    # each one's ideal reference against hw_error_budget.json, so with phi=0 the
    # recovery is not the cold recovery and build_suppl (correctly) refuses to call
    # the result the same circuit.  Sizing it with zeros would therefore raise --
    # which is the assertion doing its job, not a bug to route around.
    sizes['suppl'] = len(stage_circuits(
        'suppl', phi=_z, phi_cold=load_angle_set('cold'))[0])
    assert sizes == {'probe': 8, 'tomo': 19, 'bench': 34,
                     'ladder': 4 * len(LADDER_ROUNDS),
                     'warmcold': WARMCOLD_N_SCIENCE + WARMCOLD_N_CAL,
                     'suppl': len(SUPPL_GRID)}, sizes
    res['stage_sizes'] = sizes
    res['n_circuits_total'] = sum(sizes.values())
    res['shots_total'] = sum(sizes[s] * STAGE_SHOTS[s] for s in sizes)
    say('(h) stage circuit counts %s = %d circuits, %d shots at the default '
        'per-stage budgets' % (sizes, res['n_circuits_total'], res['shots_total']))

    # (i) the tomo job carries its own calibration, so it never has to assume that
    #     the canary's physical block is still this job's block (QCLOUD_SETUP.md 8.2)
    _g, tm = stage_circuits('tomo')
    kinds = [m['kind'] for m in tm]
    assert kinds.count('cal') == 3 and kinds.count('tomo') == 16, kinds
    assert [m['cal'] for m in tm if m['kind'] == 'cal'] == ['zero', 'one', 'zero']
    # and the ladder is the SAME grid at strictly deeper rounds, with the same meta
    # kind, which is what lets --stage all merge the two into one depth curve
    _gl, ml = stage_circuits('ladder')
    assert all(m['kind'] == 'tomo' for m in ml), ml
    assert sorted({m['rounds'] for m in ml}) == sorted(LADDER_ROUNDS)
    assert min(m['rounds'] for m in ml) > 2
    assert {(m['psi'], m['basis']) for m in ml} == set(DEPTH_SCAN)
    merged = sorted({m['rounds'] for m in tm + ml if m['kind'] == 'tomo'})
    assert merged == [1, 2] + sorted(LADDER_ROUNDS), merged
    # the two signal circuits the radius is read from must be present at every depth,
    # otherwise analyze_tomo's M3 fit skips that depth silently
    tomo_keys = {(m['psi'], m['basis'], m['rounds'])
                 for m in tm + ml if m['kind'] == 'tomo'}
    for r in merged:
        assert ('0', 'Z', r) in tomo_keys, r
    res['ladder_rounds'] = list(LADDER_ROUNDS)
    res['merged_depths'] = merged
    say('(i) the tomo job carries its own 3 calibration circuits; the ladder adds '
        'rounds %s over the %s grid, so the merged depth curve is %s and (|0>,Z) '
        'is present at every depth' % (list(LADDER_ROUNDS),
                                       [list(p) for p in DEPTH_SCAN], merged))
    # (j) the tomo ORDER is load-bearing, not cosmetic.  Under the confirmed hard
    #     real-time budget the job can be cut off mid-run; submit_cloud dumps after
    #     every chunk, so what survives is a prefix, and STAGE_CHUNK['tomo'] has to
    #     put every chunk boundary on a prefix that is analysable on its own.  A
    #     future reordering could silently destroy that salvage path, so assert it.
    ck = STAGE_CHUNK['tomo']
    bounds = list(range(ck, len(tm), ck)) + [len(tm)]
    assert kinds[:3] == ['cal'] * 3, 'the calibration must be the first chunk'
    salv = []
    for b in bounds:
        pre = tm[:b]
        cal_ok = ([m['cal'] for m in pre if m['kind'] == 'cal']
                  == ['zero', 'one', 'zero'])
        assert cal_ok, 'calibration split across a chunk boundary at %d' % b
        have = {(m['psi'], m['basis']) for m in pre
                if m['kind'] == 'tomo' and m['rounds'] == 1}
        d1 = sorted(p for p in TOMO_STATES
                    if all((p, B) in have for B in BASES))
        # a radius is the norm of THREE Bloch components, so a state missing even
        # one basis has no radius at all: a boundary must never leave a half-
        # measured input behind, or the shots spent on it are worth nothing
        partial = [p for p in TOMO_STATES
                   if 0 < len([B for B in BASES if (p, B) in have]) < len(BASES)]
        assert not partial, ('chunk boundary %d leaves a half-measured input %s'
                             % (b, partial))
        salv.append((b, d1))
    assert salv[-1][1] == sorted(TOMO_STATES), salv
    # and depth 1 must be COMPLETE before any depth-2 circuit is submitted, because
    # M3 and M5 anchor every deeper point on the depth-1 measurement
    first_d2 = min(i for i, m in enumerate(tm)
                   if m['kind'] == 'tomo' and m['rounds'] == 2)
    last_d1 = max(i for i, m in enumerate(tm)
                  if m['kind'] == 'tomo' and m['rounds'] == 1)
    assert last_d1 < first_d2, (last_d1, first_d2)
    res['tomo_chunk'] = ck
    res['tomo_chunk_boundaries'] = bounds
    res['tomo_salvage_ladder'] = [dict(circuits=b, inputs_complete=i)
                                  for b, i in salv]
    say('(j) tomo is cal-first, then depth-1 grouped by input, then depth-2; '
        'chunk=%d puts all %d submission boundaries on an analysable prefix and no '
        'boundary leaves a half-measured input: %s'
        % (ck, len(bounds),
           ' -> '.join('%dc/%dinp' % (b, len(i)) for b, i in salv)))
    # (k) block choice and the significance of r<1 are ANTI-correlated, and the code
    #     must keep taking the block the vendor's optimiser picks rather than the one
    #     that makes the hypothesis test pass.  The 2026-10-11 preflight measured
    #     WK_C180_2's best_qubit_blocks(5) at f2q=0.99275, BETTER than the quoted
    #     table's 0.9831; better gates give a purer state, so the measured
    #     calibration predicts a LARGER radius and a WEAKER r<1 than every power
    #     estimate in QCLOUD_SETUP.md sec.7-8 assumed.  Ranking the chip's 96
    #     connected 5-qubit blocks by that significance puts the vendor's pick LAST
    #     and [40,49,50,58,68] first at 6.3 sigma, purely because those are the worst
    #     qubits on the chip.  Both facts are asserted so neither can be quietly
    #     edited away: the direction of the correction, and the rejection of
    #     significance-maximising block choice.
    S = STAGE_SHOTS['tomo']
    n_eff_frac = MEASURED_FREAD ** 4
    ci_half = CI_HALF_WIDTH_AT_1000 * (1000.0 / (S * n_eff_frac)) ** 0.5
    r_meas = vendor_prediction(MEASURED_F1Q, MEASURED_F2Q, MEASURED_FREAD)['radius']
    r_quot = vendor_prediction(QUOTED_F1Q, QUOTED_F2Q, QUOTED_FREAD)['radius']
    z_meas = (1.0 - r_meas) / ci_half
    z_quot = (1.0 - r_quot) / ci_half
    assert abs(r_meas - 0.9478) < 5e-4, r_meas
    assert abs(r_quot - 0.8550) < 5e-4, r_quot
    assert r_meas > r_quot, (r_meas, r_quot)   # the device beats its own datasheet
    assert z_meas < z_quot, (z_meas, z_quot)   # so the headline claim is weaker
    assert z_meas < 1.0, z_meas                # and not significant at this budget
    assert abs(z_meas - 0.60) < 0.05, z_meas
    # the trap, in numbers: the significance-maximising block is the WORST device
    TRAP_BLOCK = [40, 49, 50, 58, 68]
    r_trap = vendor_prediction(0.99924, 0.9579, 0.9489)['radius']
    z_trap = (1.0 - r_trap) / (CI_HALF_WIDTH_AT_1000
                               * (1000.0 / (S * 0.9489 ** 4)) ** 0.5)
    assert r_trap < r_meas and z_trap > 5.0, (r_trap, z_trap)
    assert 'best_qubit_blocks' in BLOCK_SELECTION_RULE, BLOCK_SELECTION_RULE
    assert 'never' in BLOCK_SELECTION_RULE, BLOCK_SELECTION_RULE
    # 5 sigma on the vendor's block is not a budget question -- it is out of reach by
    # a factor large enough that no plausible top-up changes the answer
    s5 = (1000.0 * (CI_HALF_WIDTH_AT_1000 * 5.0 / (1.0 - r_meas)) ** 2
          / n_eff_frac)
    # `tm` is the 19-entry tomo meta list bound in (j); stage_circuits returns a
    # (gates, meta) PAIR, so len(stage_circuits(...)) is 2 and not the circuit count
    wall5 = len(tm) * s5 * MS_PER_SHOT / 1000.0
    assert len(tm) == 19, len(tm)
    assert wall5 > 50 * BUDGET_S, wall5
    res['measured_calibration'] = dict(
        block=VENDOR_BLOCK_WK_C180_2, as_of=MEASURED_CALIB_UTC,
        f1q=MEASURED_F1Q, f2q=MEASURED_F2Q, fread=MEASURED_FREAD,
        radius=r_meas, sigma_r_below_1=z_meas, ci_half_width=ci_half,
        quoted_radius=r_quot, quoted_sigma=z_quot,
        shots_for_5_sigma_per_circuit=s5, wall_s_for_5_sigma=wall5,
        significance_maximising_block=TRAP_BLOCK,
        significance_maximising_radius=r_trap,
        significance_maximising_sigma=z_trap, rule=BLOCK_SELECTION_RULE)
    say('(k) the vendor block %s measures f2q=%.5f, BETTER than the quoted %.4f, so '
        'the predicted radius is %.4f not %.4f and r<1 is %.2f sigma, not the %.2f '
        'the quoted table implied.  The significance-maximising block %s would give '
        '%.2f sigma precisely BECAUSE it is the worst device on the chip (r=%.3f), '
        'and is rejected: %s.  5 sigma here needs %.0f s = %.0fx the budget.'
        % (VENDOR_BLOCK_WK_C180_2, MEASURED_F2Q, QUOTED_F2Q, r_meas, r_quot,
           z_meas, z_quot, TRAP_BLOCK, z_trap, r_trap, BLOCK_SELECTION_RULE,
           wall5, wall5 / BUDGET_S))
    # (l) the warm-vs-cold recovery A/B.  This stage spends the ENTIRE remaining
    #     allocation on one claim, so every part of that claim's pre-registration
    #     is locked here instead of left in prose: which branch, which frame,
    #     which input state, how large the effect is, what the matched null is,
    #     what the job costs, and why the primary estimator is unconditioned
    #     rather than post-selected.  Everything is recomputed from the builders
    #     and the exact 512-dim simulator, so none of it can drift from the code
    #     it describes -- and none of it was chosen after seeing hardware data.
    wc_warm, wc_cold = load_warmcold_angles()
    assert wc_warm.shape == (16, 60), wc_warm.shape
    assert wc_cold.shape == (16, 60), wc_cold.shape
    assert not np.allclose(wc_warm, wc_cold), 'the two arms must differ'
    # the branch index and the frame are the same fact stated twice, so they are
    # checked against the syndrome table in both directions
    assert warmcold_frame_of(WARMCOLD_SIGNAL_S) == WARMCOLD_SIGNAL_FRAME
    assert warmcold_frame_of(WARMCOLD_NULL_S) == WARMCOLD_NULL_FRAME
    assert warmcold_s_of(WARMCOLD_SIGNAL_FRAME) == WARMCOLD_SIGNAL_S
    assert warmcold_s_of(WARMCOLD_NULL_FRAME) == WARMCOLD_NULL_S

    def _wc_gap(psi, s, aw=wc_warm, ac=wc_cold):
        fr = warmcold_frame_of(s)
        return (ideal_data_zero_prob(psi, fr, s, aw)
                - ideal_data_zero_prob(psi, fr, s, ac))

    gap_sig = _wc_gap(WARMCOLD_PSI, WARMCOLD_SIGNAL_S)
    gap_null = _wc_gap(WARMCOLD_PSI, WARMCOLD_NULL_S)
    assert abs(gap_sig - WARMCOLD_GAP_IDEAL_SIGNAL) < 1e-9, gap_sig
    assert abs(gap_null) < WARMCOLD_GAP_TOL, gap_null
    # s=5 is the ONLY one of the sixteen branches carrying the effect, so the
    # branch was not picked after seeing which one worked
    wc_big = [s for s in range(16)
              if abs(_wc_gap(WARMCOLD_PSI, s)) > WARMCOLD_GAP_TOL]
    assert wc_big == [WARMCOLD_SIGNAL_S], wc_big
    # psi='+' IS LOAD-BEARING: the cold arm's failure on branch 5 is a logical Z,
    # which is invisible in the Z basis and maximal in the X/Y bases.  Re-running
    # this A/B at psi='0' would return a null BY CONSTRUCTION and look like a
    # failed experiment, so the state dependence is asserted in both directions.
    wc_z_basis = max(abs(_wc_gap(p, WARMCOLD_SIGNAL_S)) for p in ('0', '1'))
    wc_xy_basis = min(abs(_wc_gap(p, WARMCOLD_SIGNAL_S))
                      for p in ('+', '-', '+i', '-i'))
    assert wc_z_basis < WARMCOLD_GAP_TOL, wc_z_basis
    assert wc_xy_basis > 0.49, wc_xy_basis
    # at p=0 the syndrome is deterministic, so the unconditioned primary
    # estimator and the post-selected secondary one have the SAME ideal value --
    # which is what makes reporting both a consistency check rather than a choice
    wc_p_s = []
    for _s in (WARMCOLD_SIGNAL_S, WARMCOLD_NULL_S):
        _fr = warmcold_frame_of(_s)
        for _arm in (wc_warm, wc_cold):
            _g = q.build_branch_circuit(q.ENC5, WARMCOLD_PSI, _fr, _s, _arm,
                                        late_measure=True)
            _pv, _ps = q.branch_amplitudes(_g, _s)
            wc_p_s.append(_ps)
            assert abs(float(abs(_pv[_s]) ** 2)
                       - ideal_data_zero_prob(WARMCOLD_PSI, _fr, _s, _arm)) < 1e-9
    assert min(wc_p_s) > 1 - 1e-9, wc_p_s

    # the job must FIT the allocation with margin: a job that cannot overrun is
    # cut off, and a cut-off job returns nothing at all
    wc_g, wc_m = stage_circuits('warmcold', wc_warm, wc_cold)
    n_wc = WARMCOLD_N_SCIENCE + WARMCOLD_N_CAL
    assert len(wc_g) == len(wc_m) == n_wc, (len(wc_g), len(wc_m), n_wc)
    wc_shots = n_wc * STAGE_SHOTS['warmcold']
    wc_cost = wc_shots * MS_PER_SHOT / 1000.0
    assert wc_cost <= BUDGET_S, (wc_cost, BUDGET_S)
    assert BUDGET_S - wc_cost >= 30.0, ('no margin against a cut-off',
                                        BUDGET_S - wc_cost)
    # chunk=2 puts the SIGNAL PAIR first and complete, so a cut-off after chunk 1
    # still yields the headline; the null and the calibration are the bonus.  This
    # is deliberately cal-LAST, the opposite of tomo's cal-first order, because
    # this stage's primary estimator is a raw probability whose warm-minus-cold
    # difference is common-mode in readout error and needs no assignment matrix.
    assert STAGE_CHUNK['warmcold'] == 2, STAGE_CHUNK['warmcold']
    _ck = STAGE_CHUNK['warmcold']
    wc_bounds = [min(len(wc_m), i * _ck)
                 for i in range(1, -(-len(wc_m) // _ck) + 1)]
    assert wc_bounds == [2, 4, 6], wc_bounds
    assert [m['kind'] for m in wc_m] == ['warmcold'] * 4 + ['cal9'] * 2, wc_m
    assert {(m['role'], m['arm']) for m in wc_m[:2]} == {
        ('signal', 'warm'), ('signal', 'cold')}, wc_m[:2]
    assert {(m['role'], m['arm']) for m in wc_m[2:4]} == {
        ('null', 'warm'), ('null', 'cold')}, wc_m[2:4]
    # every science circuit is late-measure and nine qubits wide: the mid-circuit
    # variant of this same branch returned n_sel = 0 out of 369 in 2026-09
    for _g, _m in zip(wc_g, wc_m):
        if _m['kind'] != 'warmcold':
            assert len(_g) in (q.NQ, 2 * q.NQ), _g
            continue
        assert _m['measure'] == 'late', _m
        assert _m['p'] == 0.0 and _m['psi'] == WARMCOLD_PSI, _m
        assert sum(1 for x in _g if x[0] == 'MEAS') == q.NQ, _g
        assert max(x[1] for x in _g) < q.NQ, _g
    # power: at the data survival measured in 2026-09 the effect is comfortably
    # resolvable, and it is still resolvable at HALF that survival
    wc_pw = warmcold_power(shots=STAGE_SHOTS['warmcold'])
    wc_pp = warmcold_power(w=WARMCOLD_W_PESSIMISTIC,
                           shots=STAGE_SHOTS['warmcold'])
    assert wc_pw['sigma_z'] >= 5.0, wc_pw
    assert wc_pp['sigma_z'] >= 3.0, wc_pp
    assert abs(wc_pw['observable_gap']
               - WARMCOLD_W_MEASURED_2026_09 * WARMCOLD_GAP_IDEAL_SIGNAL) < 1e-12
    # P and w are different numbers and mixing them up was a real error here, so the
    # conversion is asserted rather than trusted
    assert abs(w_from_p_data_zero(WARMCOLD_P_DATA_ZERO_2026_09)
               - WARMCOLD_W_MEASURED_2026_09) < 1e-12
    assert abs(w_from_p_data_zero(WARMCOLD_P_DATA_ZERO_2026_10_12)
               - WARMCOLD_W_MEASURED_2026_10_12) < 1e-12
    assert 0 < WARMCOLD_W_MEASURED_2026_09 < WARMCOLD_P_DATA_ZERO_2026_09, (
        'subtracting the uniform floor must lower the measured P, not raise it')
    assert 0 < WARMCOLD_W_MEASURED_2026_10_12 < WARMCOLD_P_DATA_ZERO_2026_10_12
    # THE THIRD ROW IS THE ONE THAT DECIDES WHAT THE ALLOCATION BUYS.  At the only
    # WITH-OPTIONS survival ever measured, the same 3000 shots are worth ~1 sigma:
    # the design is NOT robust across the observed range of w, which is what an
    # earlier version of this file claimed on the strength of a constant that had P
    # and w mixed up.  Asserting the row is small is what keeps that claim from
    # coming back, and it is why --wc-probe exists: 1000 shots (14 s) distinguish a
    # 6 sigma world from a 1 sigma world before six circuits are paid for.
    wc_p10 = warmcold_power(w=WARMCOLD_W_MEASURED_2026_10_12,
                            shots=STAGE_SHOTS['warmcold'])
    assert wc_p10['sigma_z'] < 1.5, wc_p10
    assert 5 * WARMCOLD_W_MEASURED_2026_10_12 < WARMCOLD_W_MEASURED_2026_09, (
        'the two committed survivals are no longer an order of magnitude apart, so '
        'the arithmetic that justifies spending 14 s on --wc-probe before the A/B '
        'has changed and this assert must be re-derived, not deleted',
        WARMCOLD_W_MEASURED_2026_10_12, WARMCOLD_W_MEASURED_2026_09)
    assert WC_PROBE_MIN_SIGMA <= wc_pw['sigma_z'], (
        'the probe gate and the stage\'s own detection bar must be the same number')

    # THE ESTIMATOR CHANGE IS JUSTIFIED BY DATA ALREADY ON DISK, not by taste, so
    # the justification is re-derived here whenever that file is present.  If the
    # ancilla marginal ever stops being near-uniform this assert fires, and the
    # post-selected F_s becomes the right primary estimator again.
    wc_data = None
    _raw = os.path.join(ROOT, 'raw_trial_batch.json')
    if os.path.exists(_raw):
        with open(_raw) as f:
            _r16 = json.load(f)
        _H, _ws = [], []
        for _r in _r16:
            _n = sum(_r.values())
            _pat = {}
            for _k, _v in _r.items():
                _a = _k[::-1][5:9]
                _pat[_a] = _pat.get(_a, 0) + _v
            _p = np.array(sorted(_pat.values(), reverse=True), float) / _n
            _H.append(float(-(_p * np.log2(_p)).sum()))
            _ws.append(sum(v for k, v in _r.items() if k[4:9] == '00000') / _n)
        wc_data = dict(n_circuits=len(_r16),
                       ancilla_entropy_bits_min=min(_H),
                       ancilla_entropy_bits_max=max(_H),
                       ancilla_entropy_uniform_bits=4.0,
                       # named for what they ARE: the measured probability, not the
                       # survival w.  The keys used to be called data_survival_*,
                       # which is how the P/w mix-up in the power constants happened.
                       p_data_zero_mean=float(np.mean(_ws)),
                       p_data_zero_min=min(_ws),
                       p_data_zero_max=max(_ws),
                       w_implied_mean=w_from_p_data_zero(float(np.mean(_ws))),
                       w_implied_min=w_from_p_data_zero(min(_ws)),
                       w_implied_max=w_from_p_data_zero(max(_ws)))
        assert min(_H) > 3.0, (
            'the 2026-09 ancilla marginal is NOT near-uniform, so the case for '
            'the unconditioned primary estimator is gone and the post-selected '
            'F_s should be primary again', min(_H))
        assert abs(np.mean(_ws) - WARMCOLD_P_DATA_ZERO_2026_09) < 5e-3, _ws
        assert np.mean(_ws) > 2 * WARMCOLD_W_FLOOR_UNIFORM, _ws
    res['warmcold'] = dict(
        psi=WARMCOLD_PSI, signal_s=WARMCOLD_SIGNAL_S,
        signal_frame=WARMCOLD_SIGNAL_FRAME, null_s=WARMCOLD_NULL_S,
        null_frame=WARMCOLD_NULL_FRAME, gap_ideal_signal=gap_sig,
        gap_ideal_null=gap_null, branches_with_effect=wc_big,
        z_basis_gap=wc_z_basis, xy_basis_gap_min=wc_xy_basis,
        angles=WARMCOLD_ANGLES, n_circuits=n_wc,
        shots_per_circuit=STAGE_SHOTS['warmcold'], shots_total=wc_shots,
        cost_s=wc_cost, margin_s=BUDGET_S - wc_cost,
        chunk=STAGE_CHUNK['warmcold'], chunk_boundaries=wc_bounds,
        primary_estimator=WARMCOLD_PRIMARY,
        secondary_estimator=WARMCOLD_SECONDARY,
        power_at_measured_w=wc_pw, power_at_pessimistic_w=wc_pp,
        power_at_with_options_w=wc_p10,
        w_measured_2026_09=WARMCOLD_W_MEASURED_2026_09,
        w_measured_2026_10_12=WARMCOLD_W_MEASURED_2026_10_12,
        p_data_zero_2026_09=WARMCOLD_P_DATA_ZERO_2026_09,
        p_data_zero_2026_10_12=WARMCOLD_P_DATA_ZERO_2026_10_12,
        probe_gate=dict(shots=WC_PROBE_SHOTS, cost_s=cost_s(WC_PROBE_SHOTS),
                        min_sigma=WC_PROBE_MIN_SIGMA,
                        ladder=WC_PROBE_SHOT_LADDER,
                        why='the two committed survivals differ 10x and straddle '
                            'the detection bar, so w is measured before the A/B is '
                            'bought'),
        measured_2026_09=wc_data)
    say('(l) warm-vs-cold A/B: psi=%s, branch s=%d (%s) is the ONLY one of 16 '
        'carrying an effect, ideal gap %+.6f; s=%d (%s) is the matched null at '
        '%+.6f.  The failure is a logical Z, so the effect vanishes in the Z '
        'basis (%.2e) and psi=%s is load-bearing.  %d circuits x %d shots = %d '
        'shots = %.0f s of the %.0f s budget (%.0f s margin), chunk=%d so the '
        'signal pair alone survives a cut-off.  Primary estimator %s, because '
        'the 2026-09 ancilla marginal is near-uniform (H=%.2f-%.2f of 4.00 bits) '
        'and its measured P(data=00000)=%.4f beat the %.4f floor by only ~4x.  '
        'POWER IS A RANGE, NOT A NUMBER: that P implies w=%.4f, worth %.1f sigma '
        'at %d shots (%.1f at half that w), while the only WITH-OPTIONS survival '
        'on record (2026-10-12, P=%.4f -> w=%.4f) is worth %.1f sigma -- so the '
        'same allocation buys a detection or a bound depending on a fact that '
        'cannot be computed offline.  --wc-probe measures it for %d shots '
        '(%.0f s) before the six circuits are bought.'
        % (WARMCOLD_PSI, WARMCOLD_SIGNAL_S, WARMCOLD_SIGNAL_FRAME, gap_sig,
           WARMCOLD_NULL_S, WARMCOLD_NULL_FRAME, gap_null, wc_z_basis,
           WARMCOLD_PSI, n_wc, STAGE_SHOTS['warmcold'], wc_shots, wc_cost,
           BUDGET_S, BUDGET_S - wc_cost, STAGE_CHUNK['warmcold'],
           WARMCOLD_PRIMARY,
           wc_data['ancilla_entropy_bits_min'] if wc_data else float('nan'),
           wc_data['ancilla_entropy_bits_max'] if wc_data else float('nan'),
           WARMCOLD_P_DATA_ZERO_2026_09, WARMCOLD_W_FLOOR_UNIFORM,
           WARMCOLD_W_MEASURED_2026_09, wc_pw['sigma_z'],
           STAGE_SHOTS['warmcold'], wc_pp['sigma_z'],
           WARMCOLD_P_DATA_ZERO_2026_10_12, WARMCOLD_W_MEASURED_2026_10_12,
           wc_p10['sigma_z'], WC_PROBE_SHOTS, cost_s(WC_PROBE_SHOTS)))
    # (l2) --cal-first reordering.  cal9's labels DO transfer to the science
    # circuits (see check_cal9_transferability), so putting the calibration pair
    # first is what lets a hard-budget kill still leave the block's readout screen
    # on disk.  The reordering must be a PERMUTATION, not a rebuild: same circuits,
    # same meta, only the order differs.  If it ever stopped being a permutation the
    # cal-first job would silently measure something other than the pre-registered
    # stage, and the artifact would not be comparable with the science-first one.
    _cf_g, _cf_m = stage_circuits('warmcold', wc_warm, wc_cold, cal_first=True)
    assert [x['kind'] for x in _cf_m[:WARMCOLD_N_CAL]] == ['cal9'] * WARMCOLD_N_CAL
    assert [x['kind'] for x in wc_m[:WARMCOLD_N_SCIENCE]] == \
        ['warmcold'] * WARMCOLD_N_SCIENCE, 'the default must stay science-first so ' \
        'the 2026-10-12 artifact remains reproducible from its own command line'
    assert sorted(str(x) for x in _cf_m) == sorted(str(x) for x in wc_m)
    assert sorted(len(g) for g in _cf_g) == sorted(len(g) for g in wc_g)
    assert [len(g) for g in _cf_g[:WARMCOLD_N_CAL]] == [9, 18]
    # every science circuit must survive the reordering unchanged, not merely in
    # count: match them by (kind, arm, s) so a swap between the two arms is caught
    _key = lambda m: (m['kind'], m.get('arm'), m.get('s'), m.get('cal'))
    assert sorted(map(_key, _cf_m)) == sorted(map(_key, wc_m))
    for _k in set(_key(x) for x in wc_m):
        _a = [g for g, m in zip(wc_g, wc_m) if _key(m) == _k]
        _b = [g for g, m in zip(_cf_g, _cf_m) if _key(m) == _k]
        assert _a == _b, ('cal-first must not alter any circuit', _k)



    # (m) the layout must be DERIVED from the circuit plus the platform's measured
    # key order, never inferred from the outcomes it is meant to explain; and a
    # simulator cross-check must normalise the simulator's key order first.  Three
    # traps fired on the 2026-10-12 run:
    #   * detect_layout reported the COMPLEMENT of the correct window (anc=[0,4]
    #     rev=True instead of anc=[5,9] rev=True) by 62 data-hits to 2, because its
    #     criterion is scored on the very outcomes it selects;
    #   * the noiseless cross-check then "confirmed" that wrong window as unique --
    #     CPUQVM's native key order is the global reverse of the vendor's, so the
    #     check validated the simulator's convention against itself.  cpuqvm_sample
    #     now converts to the vendor order, which is what makes this test mean
    #     anything, and it is asserted below by requiring the derived window to be
    #     the UNIQUE one that verifies;
    #   * the cal9 breakage was first attributed to 'ancilla q8' because the
    #     science circuit's MEAS map was applied to cal9's keys.  cal9 uses
    #     MEAS(i,i); the science circuit uses q5->c0..q8->c3, q0->c4..q4->c8.
    _sci = [(g, m) for g, m in zip(wc_g, wc_m) if m['kind'] == 'warmcold']
    _sg = [g for g, _ in _sci]
    _sm = [m for _, m in _sci]
    _anc, _rev = layout_from_meas(_sg[0])
    assert (_anc, _rev) == (slice(q.NQ - q.NQ_ANC, q.NQ), True), (_anc, _rev)
    # all four science circuits share one MEAS layer, so they share one layout
    for _g in _sg[1:]:
        assert layout_from_meas(_g) == (_anc, _rev)
    # cbit_of_position returns the LOGICAL QUBIT at an analysed position, and
    # under the vendor key order (raw position p <-> qubit NQ-1-p) reversing the
    # key makes that correspondence the identity.  Three properties are locked:
    # bijectivity for each rev, involution in pos, and agreement between the two
    # revs (reversing a key moves position p to NQ-1-p).  The logical-qubit
    # lookup in _analyze_cal9 and the inverse lookup in _warmcold_healthy_subset
    # both rely on them, and getting the direction backwards is exactly the bug
    # that mislabelled the 2026-10-12 broken channel as an ancilla.
    for _r in (True, False):
        _cb = [cbit_of_position(_p, _r) for _p in range(q.NQ)]
        assert sorted(_cb) == list(range(q.NQ)), (_r, _cb)
        for _p in range(q.NQ):
            assert cbit_of_position(cbit_of_position(_p, _r), _r) == _p, (_p, _r)
            assert cbit_of_position(_p, False) == \
                cbit_of_position(q.NQ - 1 - _p, True), _p
    # analysed position j IS logical qubit j under rev=True
    assert [cbit_of_position(_p, True) for _p in range(q.NQ)] == \
        list(range(q.NQ))
    _c9g = build_cal9('zero')
    assert [(t[1], t[2]) for t in _c9g if t[0] == 'MEAS'] == \
        [(i, i) for i in range(q.NQ)], 'cal9 must keep the identity MEAS map'
    _scimeas = [(t[1], t[2]) for t in _sg[0] if t[0] == 'MEAS']
    assert _scimeas == [(5, 0), (6, 1), (7, 2), (8, 3),
                        (0, 4), (1, 5), (2, 6), (3, 7), (4, 8)], _scimeas
    # the noiseless simulator must verify the derived layout, and it must be the
    # UNIQUE candidate that verifies -- a check that passes for a wrong window too
    # would not be a check
    _mm = [dict(m, phi=(wc_warm if m['arm'] == 'warm' else wc_cold))
           for m in _sm]
    _cands = [(slice(a, a + q.NQ_ANC), r)
              for a in (0, q.NQ_DATA) for r in (True, False)]
    _verified = []
    for _a, _r2 in _cands:
        _w = verify_layout_noiseless(_sg, _mm, shots=600, anc=_a, rev=_r2)
        if _w['layout_verified']:
            _verified.append((_a, _r2))
        if (_a, _r2) == (_anc, _rev):
            assert _w['min_acceptance_rate'] >= 1.0, _w
            assert _w['max_dev_in_sigma'] < 5.0, _w
            res['layout_noiseless'] = _w
    assert _verified == [(_anc, _rev)], (
        'the noiseless cross-check must single out exactly the MEAS-derived '
        'layout; it accepted %r of %r' % (_verified, _cands))

    # (m2) cal9 role labelling, on synthetic counts so it needs no QPU time.  A
    # broken channel must be attributed to the register the SCIENCE circuit reads
    # it in, not to the window its position happens to fall in.
    def _synth_cal9(bad_logical=None, frac=0.6, n=1000):
        """cal9 counts in which LOGICAL qubit `bad_logical` reports the wrong bit
        `frac` of the time.  frac=0.6 gives e01=e10=0.6 and contrast 1-1.2 = -0.2,
        broken with the same sign as the real run's -0.156.

        The bit is flipped at the RAW key position, because _analyze_cal9 reverses
        keys when rev=True: cal9's MEAS map is the identity, so analysed position
        equals the logical index and raw position equals NQ-1-logical.  Flipping
        the analysed position directly would silently break a different qubit."""
        zk, ok = '0' * q.NQ, '1' * q.NQ
        zero, one = {zk: n}, {ok: n}
        if bad_logical is not None:
            raw = q.NQ - 1 - bad_logical
            k = int(round(n * frac))
            zero = {zk: n - k, zk[:raw] + '1' + zk[raw + 1:]: k}
            one = {ok: n - k, ok[:raw] + '0' + ok[raw + 1:]: k}
        return [(zero, dict(kind='cal9', cal='zero', rep=0)),
                (one, dict(kind='cal9', cal='one', rep=0))]

    _clean = _analyze_cal9(_synth_cal9(), _anc, _rev)
    assert _clean['n_broken'] == 0, _clean['broken_qubits']
    assert _clean['screen']['primary_estimator_valid'] is True
    assert _clean['screen']['secondary_estimator_valid'] is True
    assert _clean['all_zero_key_fraction_data'] == 1.0
    assert _clean['all_zero_key_fraction_ancilla'] == 1.0
    assert _clean['broken_contrast_threshold'] == CAL9_BROKEN_CONTRAST == 0.5
    # cal9 key position p carries logical q_p (identity MEAS map under rev=True),
    # and logical q0..q4 ARE the data register -- so cal9's data positions are
    # 0..4 and its ancilla positions 5..8.  Under the vendor key order the science
    # circuit's windows are data=[0:5] and ancilla=[5:9], i.e. the SAME positions:
    # analysed position j is logical qubit j for both circuits.  An earlier comment
    # called them mirrored, which was an artefact of the wrong key-order convention
    # that also put the science data window at [4:9].
    assert _clean['cal9_data_positions'] == [0, 1, 2, 3, 4], _clean
    assert _clean['cal9_anc_positions'] == [5, 6, 7, 8], _clean
    assert _clean['data_slice'] == [0, 5] and _clean['anc_slice'] == [5, 9], \
        (_clean['data_slice'], _clean['anc_slice'])
    # the two registers must tile the key exactly once
    assert sorted(_clean['cal9_data_positions'] + _clean['cal9_anc_positions']) \
        == list(range(q.NQ))
    assert sorted(_clean['per_qubit']) == sorted(
        ['data%d' % i for i in range(q.NQ_DATA)]
        + ['anc%d' % i for i in range(q.NQ_ANC)]), sorted(_clean['per_qubit'])
    for _p in range(q.NQ):
        _lab = ('data%d' % _p) if _p < q.NQ_DATA else ('anc%d' % (_p - q.NQ_DATA))
        assert _clean['per_qubit'][_lab]['logical_qubit'] == _p, (_p, _lab)
        assert _clean['per_qubit'][_lab]['key_pos'] == _p, (_p, _lab)
    # logical q3 is a DATA qubit and the science circuit reads it at analysed key
    # position 3, inside the primary estimator's window [0:5].  Computing that
    # position from the classical-bit map instead gave 7, which is outside the
    # window and contradicted the very statement it was printed to support.
    assert _clean['per_qubit']['data3']['science_key_pos'] == 3
    assert 3 in range(*_clean['data_slice']), _clean['data_slice']
    assert _clean['per_qubit']['anc3']['logical_qubit'] == 8
    # an ancilla must land in the ancilla window [5:9] and NOT in the data one
    assert _clean['per_qubit']['anc3']['science_key_pos'] == 8
    assert 8 not in range(*_clean['data_slice']), _clean['data_slice']

    # break logical q3 -> a broken DATA channel, which invalidates the PRIMARY
    # estimator and must make the run INVALID, not FAILED
    _b3 = _analyze_cal9(_synth_cal9(3), _anc, _rev)
    assert _b3['n_broken'] == 1 and _b3['broken_qubits'][0]['label'] == 'data3'
    assert _b3['broken_qubits'][0]['logical_qubit'] == 3
    assert _b3['broken_qubits'][0]['science_key_pos'] == 3
    assert _b3['screen']['primary_estimator_valid'] is False, _b3['screen']
    assert _b3['screen']['secondary_estimator_valid'] is True
    assert _b3['screen']['block_usable'] is False
    assert abs(_b3['per_qubit']['data3']['contrast'] + 0.2) < 1e-9
    # break logical q5 -> an ANCILLA: the primary estimator stays valid and only
    # the post-selected F_s is lost
    _b5 = _analyze_cal9(_synth_cal9(5), _anc, _rev)
    assert _b5['n_broken'] == 1 and _b5['broken_qubits'][0]['label'] == 'anc0'
    assert _b5['broken_qubits'][0]['logical_qubit'] == 5
    assert _b5['screen']['primary_estimator_valid'] is True
    assert _b5['screen']['secondary_estimator_valid'] is False
    assert _b5['screen']['block_usable'] is True
    # the separated all-zero fractions must move with whichever register broke;
    # a single joint 9-bit number cannot tell the two cases apart
    assert _b3['all_zero_key_fraction_data'] < 0.5 \
        < _b3['all_zero_key_fraction_ancilla'], _b3
    assert _b5['all_zero_key_fraction_ancilla'] < 0.5 \
        < _b5['all_zero_key_fraction_data'], _b5
    assert _b3['mean_readout_fidelity_healthy'] == 1.0

    # (m3) transferability.  A channel with errors (e01, e10) can only return '0'
    # with probability in [min(1-e01, e10), max(1-e01, e10)] whatever circuit ran
    # before the measurement, so a science marginal outside that range PROVES the
    # two circuits are not reading the same physical channel.  This is how the
    # vendor's per-circuit re-routing is detected without vendor cooperation.
    _tr_bad = _analyze_cal9(_synth_cal9(3), _anc, _rev)
    assert abs(_tr_bad['per_qubit']['data3']['e01'] - 0.6) < 1e-9
    assert abs(_tr_bad['per_qubit']['data3']['e10'] - 0.6) < 1e-9

    def _sci_raw(p_zero, raw_pos, n=20000):
        """Science-circuit raw counts whose analysed position reads '0' with
        probability p_zero.  Under rev=True analysed position p is raw NQ-1-p."""
        k = int(round(n * p_zero))
        base = '0' * q.NQ
        flip = base[:raw_pos] + '1' + base[raw_pos + 1:]
        return {base: k, flip: n - k}

    # logical q3 sits at science analysed position 3, i.e. raw position NQ-1-3 = 5
    _raw_q3 = q.NQ - 1 - _clean['per_qubit']['data3']['science_key_pos']
    assert _raw_q3 == 5, _raw_q3
    _v = check_cal9_transferability({'signal|cold': _sci_raw(0.80, _raw_q3)},
                                    _tr_bad, rev=_rev, phi=wc_warm)
    assert _v['available'] and _v['n_comparisons'] == q.NQ, _v
    assert _v['cal9_transferable'] is False, _v
    assert _v['vendor_rerouting_detected'] is True
    assert _v['n_violations'] == 1, _v['violations']
    assert _v['violations'][0]['channel'] == 'data3', _v['violations']
    _lo, _hi = _v['violations'][0]['allowed_P0_range']
    assert abs(_lo - 0.4) < 1e-9 and abs(_hi - 0.6) < 1e-9, (_lo, _hi)
    # inside the range -> no violation, so the check is a real test and not just a
    # restatement of "some channel is broken"
    _ok = check_cal9_transferability({'signal|cold': _sci_raw(0.50, _raw_q3)},
                                     _tr_bad, rev=_rev, phi=wc_warm)
    assert _ok['cal9_transferable'] is True and _ok['n_violations'] == 0, _ok
    # a CLEAN cal9 bounds nothing (e01=e10=0 gives the range [0,1]), so the check
    # only has power once a channel is actually broken -- which is the correct
    # behaviour, not a gap: with no broken channel there is nothing to transfer
    _okc = check_cal9_transferability({'signal|cold': _sci_raw(0.97, _raw_q3)},
                                      _clean, rev=_rev, phi=wc_warm)
    assert _okc['cal9_transferable'] is True and _okc['n_violations'] == 0, _okc

    # (m4) the post-hoc healthy subset must be taken over the SCIENCE circuit's
    # data window.  Under the vendor key order both cal9 and the science circuit
    # put logical q_j at analysed position j, so for logical q3 the two positions
    # coincide at 3 -- which is convenient but means this test no longer
    # discriminates the mirror-image mistake it was written to catch.  What it
    # still pins is that the exclusion is expressed in the data window [0:5] and
    # that the kept set is its complement, not the old [4:9] window.
    _hs = _warmcold_healthy_subset({'signal|warm': _sci_raw(0.9, _raw_q3)},
                                   _tr_bad, rev=_rev, phi=wc_warm)
    assert _hs['excluded_positions'] == [3], _hs
    assert _hs['kept_positions'] == [0, 1, 2, 4], _hs
    assert _hs['post_hoc'] is True and _hs['identical_to_primary'] is False
    # with nothing broken the subset must collapse onto the pre-registered
    # estimator rather than quietly becoming a different test
    _hs0 = _warmcold_healthy_subset({'signal|warm': _sci_raw(0.9, _raw_q3)},
                                    _clean, rev=_rev, phi=wc_warm)
    assert _hs0['identical_to_primary'] is True, _hs0
    assert _hs0['kept_positions'] == [0, 1, 2, 3, 4], _hs0

    _lv = res['layout_noiseless']
    say('(m) layout is DERIVED from the MEAS(q->cbit) map PLUS the platform\'s '
        'measured key order, not inferred from the outcomes: anc=key[%d:%d] '
        'rev=%s, i.e. data=[%d:%d], and on the noiseless simulator it is the '
        'UNIQUE one of %d candidate windows that verifies (acceptance %.4f, '
        'worst |P(data=00000) - ideal| = %.1f sigma over %d shots/circuit).  '
        'That cross-check is only independent because cpuqvm_sample converts '
        'simulator keys into the vendor convention: CPUQVM\'s native order is the '
        'GLOBAL REVERSE of the vendor\'s, so an unnormalised check verifies the '
        'complement window -- which is how an earlier version of this selftest '
        '"confirmed" anc=[0:4] and the whole warmcold analysis ran on four '
        'ancillas plus one data qubit, reporting a -7.8 sigma signal that does '
        'not exist (correct window: -0.7 sigma).  The derived window also '
        'reproduces the 2026-09 feasibility numbers exactly (44/55 at F_12=0.80, '
        '0 of 369 mid-measure syndromes, 2.4%% identity mode), which is the '
        'hardware anchor.  cal9 uses the identity MEAS map while the science '
        'circuit uses q5->c0..q8->c3, q0->c4..q4->c8, but under the vendor key '
        'order analysed position j is logical q_j for BOTH, so cal9\'s labels '
        'transfer directly: the 2026-10-12 broken channel is logical q3, a DATA '
        'qubit at science key position %d, NOT the ancilla q8 that reading cal9 '
        'through the science map produced.  A broken data channel makes the run '
        'INVALID (apparatus fault); a broken ancilla only costs the post-selected '
        'F_s.  The transferability bound [min(1-e01,e10), max(1-e01,e10)] on '
        'P(read 0) CAN detect per-circuit vendor re-routing from data the job '
        'already returned -- (m3) plants a violation and catches it -- but on the '
        'real 2026-10-12 counts all 36 comparisons lie inside their bounds, so '
        'the re-routing previously reported there is RETRACTED: it was the wrong '
        'column of the science counts being compared.'
        % (_anc.start, _anc.stop, _rev, _lv['data_slice'][0],
           _lv['data_slice'][1], len(_cands),
           _lv['min_acceptance_rate'], _lv['max_dev_in_sigma'], _lv['shots'],
           _clean['per_qubit']['data3']['science_key_pos']))

    # (n) THE ESTIMATOR FAMILY.  The primary estimator is P(data=00000) over all
    #     five data channels, and that choice is defensible only if the alternatives
    #     are visibly worse where they can be checked for free: the NULL branch,
    #     whose ideal gap is 3e-6, so any |z| above ~3 there is a false signal by
    #     construction.  This re-derives the whole family from the frozen
    #     2026-10-12 counts -- the same counts the manuscript's INVALID paragraph
    #     quotes -- so four facts are locked in code instead of living in prose that
    #     could drift: what the primary reads, that its null is contaminated, that
    #     the contamination is entirely the one channel cal9 flagged, and how large
    #     a false signal post-hoc estimator selection would have manufactured.
    _fam_path = os.path.join(ROOT, STAGE_RAW['warmcold'])
    if os.path.exists(_fam_path) and os.path.exists(_meta_sidecar_path(_fam_path)):
        _fr = _load_dump('warmcold')
        _fm = read_meta_sidecar(_fam_path)['meta']
        assert len(_fr) == len(_fm) == len(wc_m), (len(_fr), len(_fm), len(wc_m))
        _cbn, _gbn = {}, {}
        for _i, _mi in enumerate(_fm):
            assert _mi == wc_m[_i], ('the frozen dump\'s submission-time meta has '
                                     'drifted from the current builders', _i)
            if _mi.get('kind') != 'warmcold':
                continue
            _nm = '%s|%s' % (_mi['role'], _mi['arm'])
            _cbn[_nm], _gbn[_nm] = _fr[_i], wc_g[_i]
        assert sorted(_cbn) == ['null|cold', 'null|warm', 'signal|cold',
                                'signal|warm'], sorted(_cbn)
        _c9f = _analyze_cal9([(r, m) for r, m in zip(_fr, _fm)
                              if m.get('kind') == 'cal9'], _anc, _rev)
        _sub = _warmcold_healthy_subset(_cbn, _c9f, rev=_rev)
        _fam = warmcold_estimator_family(_cbn, cal9_out=_sub, gates_by_name=_gbn,
                                        rev=_rev, anc=_anc)
        _pr = _fam['primary_member']
        _subk = [k for k, v in _fam['members'].items() if v['kind'] == 'subset']
        assert len(_subk) == 1, _subk
        _sk = _subk[0]

        def _z(k, b):
            return _fam['members'][k]['pairs'][b]['z']

        # 1. the probability the power constants quote IS this dump's signal|warm
        #    arm, on the MEAS-derived window -- not a number retyped from a log
        _psw = _fam['members'][_pr]['per_circuit']['signal|warm']['P']
        assert abs(_psw - WARMCOLD_P_DATA_ZERO_2026_10_12) < 5e-4, (_psw,)
        # 2. the primary's NULL is not clean on this run, which is precisely why the
        #    run is an apparatus fault (INVALID) and not a null result
        assert _z(_pr, 'null') > 2.5, _z(_pr, 'null')
        assert abs(_z(_pr, 'signal') + 0.73) < 0.05, _z(_pr, 'signal')
        # 3. dropping the ONE channel cal9 flagged cleans the null and leaves the
        #    signal at zero with it: the bias belongs to that channel, in both arms,
        #    rather than migrating between estimators
        assert abs(_z(_sk, 'null')) < 0.5, _z(_sk, 'null')
        assert abs(_z(_sk, 'signal')) < 0.5, _z(_sk, 'signal')
        assert _c9f['n_broken'] == 1 and _sub['n_kept'] == q.NQ_DATA - 1, _sub
        # 4. post-hoc selection over the family would have manufactured a huge false
        #    signal on the null branch, and the most signal-significant member is
        #    neither the primary nor the cal9-cleared subset
        assert _fam['max_abs_null_z'] > 5.0, _fam['max_abs_null_z']
        assert _fam['members'][_fam['max_abs_null_member']]['kind'] == 'single_bit'
        assert _fam['max_abs_signal_member'] not in (_pr, _sk), _fam
        # THE LOCK THAT MATTERS: no member of the family is simultaneously >=3 sigma
        # in the pre-registered direction and clean on the matched null.  If one ever
        # were, the 2026-10-12 run would be a result rather than an apparatus fault
        # and everything the manuscript says about it would have to be rewritten --
        # so that possibility fails here, loudly, instead of being overlooked.
        _clean_sig = [k for k, v in _fam['members'].items()
                      if v['pairs'].get('signal', {}).get('complete')
                      and v['pairs']['signal']['z'] >= WC_PROBE_MIN_SIGMA
                      and v['pairs'].get('null', {}).get('complete')
                      and abs(v['pairs']['null']['z']) < 1.0]
        assert _clean_sig == [], _clean_sig
        res['estimator_family'] = dict(
            source_dump=os.path.basename(_fam_path), n_members=_fam['n_members'],
            primary_member=_pr, cal9_cleared_member=_sk,
            kinds={k: v['kind'] for k, v in _fam['members'].items()},
            z_signal=_fam['signal_z_by_member'], z_null=_fam['null_z_by_member'],
            max_abs_null_z=_fam['max_abs_null_z'],
            max_abs_null_member=_fam['max_abs_null_member'],
            max_abs_signal_member=_fam['max_abs_signal_member'],
            P_signal_warm_primary=_psw,
            w_implied=w_from_p_data_zero(_psw),
            members_both_clean_and_significant=_clean_sig,
            data_slice=_fam['data_slice'], rev=_fam['rev'])
        say('(n) estimator family on the frozen 2026-10-12 counts (%d members, data '
            'window [%d:%d] rev=%s).  The primary %s reads P(signal|warm)=%.4f, '
            'i.e. w=%.4f: signal %+.2f sigma, null %+.2f sigma -- the NULL IS '
            'CONTAMINATED, which is why that run is an apparatus fault and not a '
            'null result.  Dropping the single channel cal9 flagged moves the null '
            'to %+.2f sigma and the signal to %+.2f sigma, so the bias is that '
            'channel\'s alone and not the estimator\'s.  Ranking the family by the '
            'SIGNAL branch instead would have picked %s at %+.2f sigma, whose own '
            'null reads %+.2f sigma; the largest null deviation anywhere in the '
            'family is %+.1f sigma (%s) on a branch whose ideal gap is %.1e.  NO '
            'member is both >=%.0f sigma in the pre-registered direction and clean '
            '(<1 sigma) on the null, which is what makes the primary a pre-registered '
            'choice rather than the winner of a search.'
            % (_fam['n_members'], _fam['data_slice'][0], _fam['data_slice'][1],
               _fam['rev'], _pr, _psw, w_from_p_data_zero(_psw),
               _z(_pr, 'signal'), _z(_pr, 'null'), _z(_sk, 'null'),
               _z(_sk, 'signal'), _fam['max_abs_signal_member'],
               _fam['signal_z_by_member'][_fam['max_abs_signal_member']],
               _fam['null_z_by_member'].get(_fam['max_abs_signal_member'],
                                            float('nan')),
               _fam['null_z_by_member'][_fam['max_abs_null_member']],
               _fam['max_abs_null_member'], WARMCOLD_GAP_IDEAL_NULL,
               WC_PROBE_MIN_SIGMA))

    # (o) --wc-probe's analysis chain, exercised end to end on the NOISELESS
    #     simulator so the gate that decides whether to spend 168 s is itself tested
    #     before it is trusted with money.  Three things are pinned: the expected key
    #     derived analytically equals the mode of an independent noiseless sample; a
    #     perfect device reads w = 1 and clears every pipeline gate; and a corrupted
    #     return (keys narrower than the block, or a short count) FAILS the pipeline
    #     gate rather than being quietly analysed.
    _pi = [i for i, m in enumerate(wc_m)
           if m.get('kind') == 'warmcold' and m.get('role') == 'signal'
           and m.get('arm') == 'warm']
    assert len(_pi) == 1, _pi
    _pg, _pmet = wc_g[_pi[0]], wc_m[_pi[0]]
    _ek, _ekp = expected_vendor_key(_pg)
    assert _ekp > 1 - 1e-9, ('at p=0 the ideal output is a delta, so the expected '
                             'key must carry all of it', _ekp)
    _pc = q.cpuqvm_sample(_pg, 1000)
    assert max(_pc, key=_pc.get) == _ek, (
        'the analytically derived expected key and the noiseless sampler\'s mode '
        'disagree, so one of the two key-order conventions is wrong',
        max(_pc, key=_pc.get), _ek)
    _blk = list(range(q.NQ))
    _po = analyze_wc_probe(_pc, _pmet, _pg, wc_warm, shots_requested=1000,
                           chip_id='NOISELESS', block=_blk, ms_per_shot=None,
                           budget_remaining_after=BUDGET_S, verbose=False)
    assert _po['pipeline']['ok'], _po['pipeline']['problems']
    _tol = 4.0 / math.sqrt(1000)
    assert abs(_po['primary']['p'] - 1.0) < _tol, _po['primary']
    assert abs(_po['w']['point'] - 1.0) < _tol, _po['w']
    assert _po['primary']['structure_detected'] is True
    assert _po['noiseless_check']['min_acceptance_rate'] >= 1.0
    assert _po['verdict'] == 'DETECTION AFFORDABLE' and _po['exit_code'] == 0, _po
    # the ladder must price 6 circuits, be monotone in cost, and tighten the bound a
    # null would report -- otherwise the shot count it recommends is not the one the
    # budget arithmetic priced
    _lad = _po['power']['ladder']
    assert [r['n_shots_total'] for r in _lad] == \
        [(WARMCOLD_N_SCIENCE + WARMCOLD_N_CAL) * r['shots_per_circuit'] for r in _lad]
    assert all(_lad[i]['cost_s'] < _lad[i + 1]['cost_s']
               for i in range(len(_lad) - 1))
    assert all(_lad[i]['w_bound95_if_null'] > _lad[i + 1]['w_bound95_if_null']
               for i in range(len(_lad) - 1))
    assert _po['power']['s_star'] == max(
        r['shots_per_circuit'] for r in _lad if r['fits_remaining_budget'])
    # a key narrower than the block is the failure the 5-qubit probe stage exists to
    # avoid, and it must FAIL rather than be analysed on the wrong window
    _narrow = {k[q.NQ - NQ5:]: v for k, v in _pc.items()}
    _pn = analyze_wc_probe(_narrow, _pmet, _pg, wc_warm, shots_requested=1000,
                           chip_id='NOISELESS', block=_blk, verbose=False)
    assert _pn['pipeline']['ok'] is False and _pn['exit_code'] == 4, _pn['pipeline']
    assert _pn['verdict'] == 'PIPELINE FAILED'
    # a short return is the other silent-invalidating failure: the platform bills the
    # request but delivers fewer shots, so the power table would price the wrong N
    _short = {k: max(1, v // 2) for k, v in list(_pc.items())[:5]}
    _ps = analyze_wc_probe(_short, _pmet, _pg, wc_warm, shots_requested=1000,
                           chip_id='NOISELESS', block=_blk, verbose=False)
    assert _ps['pipeline']['ok'] is False, _ps['pipeline']
    assert any('returned' in s for s in _ps['pipeline']['problems'])
    res['wcprobe_chain'] = dict(
        expected_key=_ek, expected_key_ideal_prob=_ekp,
        sampler_mode_agrees=True, noiseless_p=_po['primary']['p'],
        noiseless_w=_po['w']['point'], s_star_at_budget=_po['power']['s_star'],
        verdict=_po['verdict'], n_ladder=len(_lad),
        narrow_key_rejected=True, short_return_rejected=True)
    say('(o) --wc-probe\'s chain is tested before it is trusted with money: the '
        'expected key derived analytically (%s, ideal probability %.6f) equals the '
        'mode of an independent noiseless sample, so the canonical->native->vendor '
        'key-order conversions check each other; on that noiseless device the probe '
        'reads P(data=00000)=%.4f -> w=%.4f, clears every pipeline gate, and prices '
        'S*=%d shots/circuit against a %.0f s budget as %s.  A return whose keys are '
        'narrower than the block, and one that delivers fewer than %.0f%% of the '
        'requested shots, are both REJECTED as pipeline failures (exit 4) rather '
        'than analysed on the wrong window -- those are the two ways a probe could '
        'otherwise report a confident w for data that is not the circuit it claims.'
        % (_ek, _ekp, _po['primary']['p'], _po['w']['point'],
           _po['power']['s_star'], BUDGET_S, _po['verdict'],
           100.0 * WC_PROBE_MIN_RETURN_FRACTION))

    # (p) PLAN A'S CHIP RULE (QCLOUD_SETUP.md 9.7).  Two probes, one per chip, and a
    #     decision that has to be arithmetic rather than a judgement call at the
    #     terminal -- the alternative is picking where to spend the last ~200 s AFTER
    #     seeing which chip looked better, with nothing on disk to show the rule was
    #     fixed beforehand.  Locked here: the detection threshold is an OUTPUT of the
    #     power model selftest (l) already pins, not a number retyped into prose; S*
    #     follows the budget passed in rather than the stale remainder each probe
    #     carries; a pipeline-failed probe is not a candidate; and when nothing clears
    #     the bar the A/B goes to the chip the 2026-10-12 INVALID was measured on --
    #     NOT to the chip with the higher w, which is what an outcome-chasing rule
    #     would do.
    def _fake_probe(chip, w, block, ok=True, priced_at=272.0):
        pr = {k: v for k, v in _po.items() if k != 'gates'}
        pr['chip_id'], pr['block'] = chip, list(block)
        pr['dump_raw'] = '/nonexistent/%s.json' % chip
        p = w + (1.0 - w) / 32.0        # the exact inverse of w_from_p_data_zero
        assert abs(w_from_p_data_zero(p) - w) < 1e-12, (p, w)
        pr['primary'] = dict(pr['primary'], p=p)
        pr['w'] = dict(pr['w'], point=w, lo95=max(0.0, w - 0.0126), hi95=w + 0.0126)
        pr['power'] = wc_probe_power(w, pr['w']['lo95'], None, priced_at)
        if not ok:
            pr['pipeline'] = dict(ok=False, problems=['keys came back 5 bits wide'])
            pr['verdict'], pr['exit_code'] = 'PIPELINE FAILED', 4
        else:
            pr['pipeline'] = dict(_po['pipeline'])
            pr['verdict'] = ('DETECTION AFFORDABLE'
                             if pr['power']['detection_affordable']
                             else 'BOUND EXPERIMENT')
            pr['exit_code'] = 0 if pr['power']['detection_affordable'] else 5
        return pr

    _bA = [52, 61, 62, 70, 71, 79, 80, 88, 89]
    _bB = [39, 48, 49, 57, 58, 66, 67, 75, 85]
    # The threshold must come FROM the power model and BE the threshold: z(w_min) is at
    # the bar and z just below it is not.  sigma_z monotone in w is what makes the
    # bisection inside min_w_for_detection valid, so it is checked, not assumed.
    _zs = [warmcold_power(w=x / 2000.0, shots=2000)['sigma_z'] for x in range(2001)]
    assert all(_zs[i] <= _zs[i + 1] + 1e-12 for i in range(len(_zs) - 1)), 'not monotone'
    _wmin = min_w_for_detection(2000)
    assert warmcold_power(w=_wmin, shots=2000)['sigma_z'] >= WC_PROBE_MIN_SIGMA
    assert warmcold_power(w=_wmin - 1e-5, shots=2000)['sigma_z'] < WC_PROBE_MIN_SIGMA
    assert abs(_wmin - 0.0467) < 2e-3, ('QCLOUD_SETUP.md 9.7 quotes 0.0467 at S=2000',
                                        _wmin)
    assert min_w_for_detection(2500) < _wmin, 'more shots must lower the bar'
    # z(w=1, S) = 0.8165*sqrt(S), so the bar is unreachable only below S~13.5: at
    # S=10 even a perfect device misses 3 sigma and no w can rescue the A/B.
    assert min_w_for_detection(10) is None, 'a perfect device must miss the bar at S=10'
    assert min_w_for_detection(50) is not None, 'at S=50 a perfect device clears it'
    # DETECTION: the 2026-09 world.  S* must be the one the ARGUMENT buys (200 s ->
    # 2000 shots/circuit), not the one the probes were priced at (272 s -> 2500).
    _hi = [_fake_probe('WK_C180', 0.0936, _bA),
           _fake_probe('WK_C180_2', 0.0098, _bB)]
    _d = pick_wc_chip(_hi, 200.0, verbose=False)
    assert (_d['mode'], _d['exit_code']) == ('DETECTION', 0), _d['reason']
    assert _d['chip_id'] == 'WK_C180' and _d['block'] == _bA
    assert _d['s_star'] == 2000 and _d['w_min_for_detection'] == _wmin
    assert _d['power']['budget_remaining_after'] == 200.0
    assert 'A/B counts' in _d['winners_curse'], 'the disclosure must ride along'
    # BOUND: both w low.  The rule sends the A/B to the 2026-10-12 chip even though
    # the OTHER chip measured a higher w -- that is the whole point of declaring the
    # rule before buying the probes.
    _lo = [_fake_probe('WK_C180', 0.0120, _bA),
           _fake_probe('WK_C180_2', 0.0098, _bB)]
    _b = pick_wc_chip(_lo, 200.0, verbose=False)
    assert (_b['mode'], _b['exit_code']) == ('BOUND', 5), _b['reason']
    assert _b['chip_id'] == WC_PICK_BOUND_CHIP and _b['block'] == _bB
    assert _b['best_w_chip'] == 'WK_C180', _b['best_w_chip']
    # a refused bound chip falls back to the surviving one, and says so in the reason
    _f = pick_wc_chip([_fake_probe('WK_C180', 0.0120, _bA),
                       _fake_probe('WK_C180_2', 0.0098, _bB, ok=False)],
                      200.0, verbose=False)
    assert (_f['mode'], _f['chip_id']) == ('BOUND', 'WK_C180')
    assert len(_f['refused']) == 1 and 'NOT' in _f['reason'], _f['reason']
    # nothing usable, and a budget no rung of the ladder fits, both STOP (exit 4)
    # rather than buy the A/B; omitting the budget is refused (exit 3 at the CLI)
    _s = pick_wc_chip([_fake_probe('WK_C180_2', 0.0098, _bB, ok=False)], 200.0,
                      verbose=False)
    assert (_s['mode'], _s['exit_code'], _s['chip_id']) == ('STOP', 4, None)
    _t = pick_wc_chip(_hi, 40.0, verbose=False)
    assert (_t['mode'], _t['exit_code'], _t['s_star']) == ('STOP', 4, None)
    try:
        pick_wc_chip(_hi, None, verbose=False)
        raise AssertionError('wcpick without --budget-remaining must be refused')
    except BudgetError:
        pass
    res['wcpick_rule'] = dict(
        w_min_at_2000=_wmin, w_min_at_2500=min_w_for_detection(2500),
        sigma_z_monotone_in_w=True, detection_chip=_d['chip_id'],
        bound_chip=_b['chip_id'], bound_chip_declared=WC_PICK_BOUND_CHIP,
        s_star_follows_argument=_d['s_star'], s_star_in_probe_was=2500,
        refused_probe_excluded=True, fallback_disclosed=True,
        stop_when_none_usable=True, stop_when_no_rung_fits=True,
        budget_required=True)
    say('(p) plan A\'s chip rule is arithmetic, not a terminal judgement.  The '
        'detection threshold is computed by inverting the power model selftest (l) '
        'pins, so w >= %.4f at S=2000 (%.4f at S=2500) is where the %.0f sigma bar '
        'sits, with sigma_z verified monotone in w so the bisection is valid.  On two '
        'synthetic probes S* follows the budget ARGUMENT (%d shots/circuit at 200 s) '
        'rather than the stale 272 s the probes were priced at; a probe that failed its '
        'pipeline gate is excluded, and when it was the declared bound chip the '
        'fallback is disclosed in the reason.  When both w are low the A/B goes to %s '
        '-- the chip the 2026-10-12 INVALID was measured on -- even though the OTHER '
        'chip measured the higher w (0.0120 vs 0.0098), which is what a rule declared '
        'before the data was bought has to do.  No usable probe, or a budget no rung of '
        'the ladder fits, returns STOP (exit 4) instead of buying the A/B, and omitting '
        '--budget-remaining raises BudgetError (exit 3 at the CLI).'
        % (_wmin, min_w_for_detection(2500), WC_PROBE_MIN_SIGMA, _d['s_star'],
           WC_PICK_BOUND_CHIP))

    # (q) THE SUPPL STAGE IS CHECKED AGAINST THE DATA IT CLAIMS TO REPRODUCE.
    # build_suppl asserts bit-exactness at build time, but an assertion that the
    # right circuit was BUILT says nothing about whether the analysis reads it back
    # correctly.  So the archived 2026-09 counts are fed through analyze_suppl and
    # every number is required to equal hw_error_budget.json's own, exactly -- the
    # comparison is against the artifact, not a restatement of it, so the two cannot
    # agree by having been copied from a common source.  The verdict rule is then
    # tested on synthetic F_s values placed on both sides of each threshold, a route
    # sharing nothing with the count-level check.
    _sup_raw_path = os.path.join(ROOT, 'raw_verify_options.json')
    assert os.path.exists(_sup_raw_path), (
        'raw_verify_options.json is the archived raw counts of the 2026-09 '
        'feasibility run and is tracked in git; selftest (q) cannot verify the '
        'suppl analysis without it')
    with open(_sup_raw_path) as _f:
        _sup_raw = json.load(_f)
    _cold_ang = load_angle_set('cold')
    # NOTE: (n) rebinds `_z` to a nested helper, so this block uses its own name.
    _zq = np.zeros((16, 60))
    _sg, _sm = stage_circuits('suppl', phi=_zq, phi_cold=_cold_ang)
    _eb = json.load(open(os.path.join(ROOT, 'hw_error_budget.json')))
    _ORD = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    _sup = analyze_suppl([_sup_raw[_ORD[m['label']]] for m in _sm], _sm,
                         _cold_ang, shots_requested=400, verbose=False)
    _exact = {}
    for _lab in ('A', 'B', 'C', 'D'):
        _c, _r = _sup['per_circuit'][_lab], _eb['circuits'][_lab]
        _nsel_ref = int(round(_r['syndrome_channel']['joint_all_correct']
                              * _r['shots_used']))
        assert _c['n_selected'] == _nsel_ref, (_lab, _c['n_selected'], _nsel_ref)
        assert _c['n_total'] == _r['shots_used'], (_lab, _c['n_total'])
        _nh_ref = _r['data_channel'].get('n_hit')
        # n_sel = 0 makes F_s undefined, which the artifact records as n_hit=None;
        # the analysis still knows zero shots hit, so the two agree on the physics
        # and differ only in whether an undefined conditional is None or 0.
        assert _nh_ref is None or _c['n_selected_hit'] == _nh_ref, (_lab, _nh_ref)
        if _nh_ref is not None:
            assert abs(_c['F_s'] - _r['data_channel']['F_s']) < 1e-12, _lab
        _eps_ref = _r['budget']['implied_mean_instruction_error']
        if _eps_ref is not None:
            assert abs(_c['implied_eps_instr'] - _eps_ref) < 1e-12, (
                _lab, _c['implied_eps_instr'], _eps_ref)
        _exact[_lab] = dict(n_sel=_c['n_selected'], n_hit=_c['n_selected_hit'],
                            F_s=_c['F_s'], eps=_c['implied_eps_instr'],
                            rank=_c['syndrome_rank'],
                            z=_c['syndrome_z_vs_code_blind'],
                            satellite=_c['relaxation_satellite_ratio'])
    # The numbers main.tex quotes from this run, re-derived here rather than cited.
    assert _exact['D']['F_s'] == 0.8 and _exact['D']['n_hit'] == 44, _exact['D']
    assert _exact['D']['rank'] == 2, _exact['D']
    assert abs(_exact['D']['z'] - 6.82) < 0.01, _exact['D']['z']
    assert abs(_exact['D']['satellite'] - 1.7455) < 1e-3, _exact['D']['satellite']
    assert _exact['A']['n_sel'] == 0 and abs(_exact['A']['z'] + 4.96) < 0.01
    assert _exact['B']['n_hit'] == 0 and abs(_exact['B']['z'] - 4.56) < 0.01
    assert _sup['comparison']['verdict'] == 'REPRODUCED', _sup['comparison']

    # The verdict rule, on synthetic values either side of each threshold.
    def _mk(fs):
        return {'D': {'F_s': fs, 'implied_eps_instr': 0.0239}}
    assert suppl_verdict(_mk(0.80))['verdict'] == 'REPRODUCED'
    assert suppl_verdict(_mk(0.40))['verdict'] == 'REPRODUCED'      # at target/2
    assert suppl_verdict(_mk(0.39))['verdict'] == 'INCONCLUSIVE'    # just below
    assert suppl_verdict(_mk(0.19))['verdict'] == 'INCONCLUSIVE'
    assert suppl_verdict(_mk(0.18))['verdict'] == 'NOT REPRODUCED'  # at 1.5x psi='+'
    assert suppl_verdict(_mk(0.12))['verdict'] == 'NOT REPRODUCED'
    assert suppl_verdict({})['verdict'] == 'NOT ANALYSABLE'
    # The degeneracy guard and the nan guard beside it.  Both exist because the
    # 2026-10-24T141813Z submission returned counts collapsed onto ~50 bitstrings
    # with ~50% of all shots in a single outcome, which yields F_s = 0.0000 and so
    # the SAME "NOT REPRODUCED" string a valid run produces.  Unguarded, it was
    # indistinguishable from a third independent confirmation of the verdict, and it
    # was a COMPLETE 4/4 record on the September block -- strictly more attractive
    # to a reader than the valid 2/4 one it sat beside.
    _deg = suppl_verdict(_mk(0.0), degenerate=True, entropy_ratio_max=0.292)
    assert _deg['verdict'] == 'NOT ANALYSABLE', _deg
    assert _deg.get('degenerate') is True and '0.292' in _deg['reason']
    # the guard must not override a genuine verdict when the data is NOT degenerate
    assert suppl_verdict(_mk(0.80), degenerate=False)['verdict'] == 'REPRODUCED'
    # an uncomputable implied error rate cannot yield a verdict quoting nan: the
    # degenerate run printed "moves to eps=nan ... to nanx outside it" into a
    # sentence the manuscript would have reproduced verbatim.
    _nan = suppl_verdict({'D': {'F_s': 0.0, 'implied_eps_instr': float('nan')}})
    assert _nan['verdict'] == 'NOT ANALYSABLE', _nan
    assert 'nan' not in _nan['reason'].lower(), _nan['reason']
    # The archived 2026-09 counts are VALID and must clear the guard -- at n~369
    # shots, low enough that a raw-bit threshold tuned on the 20 000-shot runs
    # would have rejected them.  This is the assertion that caught the first
    # implementation of the guard, which thresholded bits instead of the ratio.
    assert _sup['degenerate'] is False, _sup['entropy_ratio_min']
    assert _sup['entropy_ratio_min'] > SUPPL_MIN_OUTCOME_ENTROPY_RATIO
    assert 0.60 < _sup['entropy_ratio_min'] and _sup['entropy_ratio_max'] < 0.75, (
        _sup['entropy_ratio_min'], _sup['entropy_ratio_max'])
    assert suppl_verdict({'D': {'F_s': None}})['verdict'] == 'NOT ANALYSABLE'
    # build_suppl must REFUSE angles that do not reproduce the archived ideal.
    try:
        build_suppl(_zq)
        raise AssertionError('build_suppl accepted placeholder zero angles, so the '
                             'bit-exactness gate is not actually gating')
    except AssertionError as _exc:
        assert 'does not match the 2026-09' in str(_exc), str(_exc)
    res['suppl_reproduction'] = dict(
        psi=SUPPL_PSI, grid=[list(x) for x in SUPPL_GRID],
        exact_on_archived_counts=_exact,
        verdict_on_own_source_data=_sup['comparison']['verdict'],
        layouts_all_agree=bool(len({tuple(_sup['per_circuit'][l]['anc_slice'])
                                    + (_sup['per_circuit'][l]['rev'],)
                                    for l in 'ABCD'}) == 1),
        verdict_rule_thresholds=dict(
            reproduced_at_or_above=SUPPL_TARGETS['D']['F_s'] / 2.0,
            not_reproduced_at_or_below=0.12092 * 1.5),
        refuses_non_reproducing_angles=True)
    say('(q) the suppl stage reproduces the 2026-09 feasibility run EXACTLY when fed '
        'that run\'s own archived counts: D returns 44/55 at F_s=%.4f, eps=%.10f '
        'against the artifact\'s %.10f, syndrome rank %d at z=%+.2f and satellite '
        '%.4f; C returns 9/25 at %.4f; A returns n_sel=0 at z=%.2f and B n_hit=0 at '
        'z=%+.2f -- Layer 4\'s "0 of 369" and "4.6 sigma", re-derived rather than '
        'restated.  All four derive the same window (anc=[5,9] rev=True), so mid and '
        'late share a classical-bit order here and the per-circuit derivation is '
        'defensive rather than load-bearing.  The circuits are bit-exact at BUILD '
        'time as well: ideal references match to %.1e and %.1e and the instruction '
        'and CNOT counts are checked against the archived inventory, and build_suppl '
        'REFUSES placeholder angles (diff 2.3e-07) instead of submitting a circuit '
        'that is not the one it claims.  The verdict rule is pre-declared in code and '
        'tested on both sides of each threshold: >=%.2f REPRODUCED, <=%.5f NOT '
        'REPRODUCED, between them INCONCLUSIVE, and no post-selected shot NOT '
        'ANALYSABLE.'
        % (_exact['D']['F_s'], _exact['D']['eps'],
           _eb['circuits']['D']['budget']['implied_mean_instruction_error'],
           _exact['D']['rank'], _exact['D']['z'], _exact['D']['satellite'],
           _exact['C']['F_s'], _exact['A']['z'], _exact['B']['z'],
           abs(_sm[0]['ideal_P_data_zero'] - SUPPL_TARGETS['C']['ideal']),
           abs(_sm[1]['ideal_P_data_zero'] - SUPPL_TARGETS['D']['ideal']),
           SUPPL_TARGETS['D']['F_s'] / 2.0, 0.12092 * 1.5))

    return res


# ---------------------------------------------------------------------
# 8) free pre-flight queries, and the three cloud stages
# ---------------------------------------------------------------------
def _now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _backend(chip_id, token):
    svc = q._make_qcloud_service(token)
    try:
        return svc, svc.backend(chip_id)
    except TypeError:
        return svc, svc.backend(chip_id=chip_id)


def _block_connected(block, edges):
    """True if `block` is connected using only the calibrated edges among it.

    A submission gets one block and the encoder needs every qubit reachable, so
    "the vendor ranked it best" is not sufficient on its own -- the calibration
    response has to contain the edges that make it a single piece."""
    S = set(int(b) for b in block)
    if not S:
        return False
    adj = {u: set() for u in S}
    for (a, b) in edges:
        if a in S and b in S:
            adj[a].add(b)
            adj[b].add(a)
    start = next(iter(S))
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        for v in adj[u] - seen:
            seen.add(v)
            stack.append(v)
    return len(seen) == len(S)


def measured_block_calibration(chip_id, block, token, shots=None):
    """FREE: the vendor's own per-qubit figures of merit for `block`, aggregated
    and pushed through the same prediction the quoted table goes through.

    ChipInfo was 401 Unauthorized from 2026-09-21 until 2026-10-11, so every error
    budget in this repo was built on the quoted aggregate table rather than on
    measurements.  It answers now, and the two disagree in the direction that
    decides the experiment: on WK_C180_2's best 5-qubit block the measured
    two-qubit fidelity is 0.99275 against a quoted 0.9831.  Better gates mean a
    purer state, so the headline signal 1-r shrinks and the significance of r<1
    shrinks with it.  This recomputes that comparison from the live device at
    preflight time instead of trusting the module constants, so a recalibration
    shows up as a changed number rather than a stale prediction.

    Readout enters twice and in opposite directions, which is why both radii are
    reported.  `radius` carries gate errors only, because the tomo stage measures
    its own assignment matrix and mitigates readout.  `codespace_survival` is the
    fraction of shots the post-selection keeps, which readout degrades and
    mitigation cannot give back -- it can only widen the error bar, and that
    widening is charged to `sigma_r_below_1` through n_effective_shots."""
    shots = shots or STAGE_SHOTS['tomo']
    blk = [int(b) for b in block]
    out = dict(chip_id=chip_id, block=blk, shots=shots, attempted_utc=_now(),
               available=False)
    try:
        _svc, be = _backend(chip_id, token)
        ci = be.chip_info()
        sq = {int(x.get_qubit_id()): x for x in ci.single_qubit_info()}
        dq = {}
        for e in ci.double_qubits_info():
            t = tuple(sorted(int(u) for u in e.get_qubits()))
            if len(t) == 2 and e.get_fidelity() is not None:
                dq[t] = max(dq.get(t, 0.0), float(e.get_fidelity()))
        out['chip_info'] = dict(qubits_num=ci.qubits_num(),
                                single_gate_timing_ns=ci.get_single_gate_timing(),
                                double_gate_timing_ns=ci.get_double_gate_timing(),
                                basic_gates=list(ci.get_basic_gates()),
                                n_qubits_calibrated=len(sq),
                                n_edges_calibrated=len(dq))
    except Exception as exc:
        out['error'] = repr(exc)[:400]
        return out
    missing = [b for b in blk if b not in sq]
    if missing:
        out['error'] = ('no single-qubit calibration for %s; ChipInfo on %s covers '
                        'qubits %s' % (missing, chip_id, sorted(sq)))
        return out
    out['per_qubit'] = {str(b): dict(
        readout_fidelity=sq[b].get_readout_fidelity(),
        single_gate_fidelity=sq[b].get_single_gate_fidelity(),
        t1_us=sq[b].get_t1(), t2_us=sq[b].get_t2(),
        frequency_hz=sq[b].get_frequency()) for b in blk}
    edges = {('%d-%d' % t): dq[t] for t in sorted(dq)
             if t[0] in blk and t[1] in blk}
    out['edges'] = edges
    out['connected'] = _block_connected(blk, dq)
    if not out['connected']:
        out['error'] = ('block %s is NOT connected by its calibrated edges %s, so a '
                        '5-qubit submission on it cannot be routed' % (blk, edges))
        return out
    f1 = sum(sq[b].get_single_gate_fidelity() for b in blk) / float(len(blk))
    fr = sum(sq[b].get_readout_fidelity() for b in blk) / float(len(blk))
    if not edges:
        out.update(available=False, f1q=f1, fread=fr,
                   error='block has no calibrated two-qubit edge; not usable')
        return out
    f2 = sum(edges.values()) / float(len(edges))
    out.update(available=True, n_edges=len(edges), f1q=f1, f2q=f2, fread=fr)
    pm = vendor_prediction(f1, f2, fr)
    pq = vendor_prediction(QUOTED_F1Q, QUOTED_F2Q, QUOTED_FREAD)
    n_eff = shots * fr ** 4
    ci_half = CI_HALF_WIDTH_AT_1000 * (1000.0 / max(n_eff, 1.0)) ** 0.5
    out.update(measured_prediction=pm, quoted_prediction=pq,
               codespace_survival=fr ** 4, n_effective_shots=n_eff,
               ci_half_width=ci_half)
    if ci_half > 0:
        out['sigma_r_below_1'] = (1.0 - pm['radius']) / ci_half
        out['sigma_if_quoted_table_were_true'] = (1.0 - pq['radius']) / ci_half
    return out



def preflight(chip_ids, token=None, verbose=True):
    """FREE cloud queries only -- submits nothing and spends no QPU time.

    Three things must be settled before any shots are bought: which chips are
    online and how long their queues are, which physical block each stage should
    be pinned to (5 qubits for probe/tomo, 9 for bench -- qcloud_vscr_new's own
    auto-query hardcodes 9, so a 5-qubit stage must pin its block explicitly), and
    whether the vendor calibration metadata is still behind the 401 that
    hw_error_budget.py records.  Every failure is captured verbatim rather than
    raised, so the artifact always says which of the two happened."""
    out = dict(attempted_utc=_now(), chip_ids=list(chip_ids),
               spends_qpu_time=False, backends=None, blocks={}, calibration={},
               measured_block={}, errors=[])
    try:
        import hw_error_budget as heb
        tok = token or heb.resolve_token()
    except Exception as exc:
        out['errors'].append('token: %r' % (exc,))
        return out
    try:
        svc, _ = _backend(chip_ids[0], tok)
        bl = svc.backends()
        out['backends'] = [b if isinstance(b, dict) else str(b)
                           for b in (bl if isinstance(bl, (list, tuple)) else [bl])]
    except Exception as exc:
        out['errors'].append('backends(): %r' % (exc,))
    for chip in chip_ids:
        for nq in (NQ5, q.NQ):
            try:
                _svc, be = _backend(chip, tok)
                bb = be.best_qubit_blocks(nq)
                blocks = list(bb.qubit_blocks) if bb.qubit_blocks else []
                out['blocks']['%s|%d' % (chip, nq)] = (
                    [[int(x) for x in b] for b in blocks[:3]])
            except Exception as exc:
                out['errors'].append('best_qubit_blocks(%d) on %s: %r'
                                     % (nq, chip, exc))
    # The vendor's per-qubit calibration, queried for the block THIS chip actually
    # offers.  hw_error_budget.BLOCK is the 9-qubit block the 2026-09 WK_C180 run
    # was pinned to (q147-176) and WK_C180_2 only publishes q39-86, so calling it
    # without a block returns a record whose every figure of merit is null -- which
    # reads as "the vendor publishes nothing" when the truth is "we asked about the
    # wrong qubits".  Pass the live block, and say which was used.
    tok2 = None
    try:
        import hw_error_budget as heb
        tok2 = heb.resolve_token()
        for chip in chip_ids:
            live = out['blocks'].get('%s|%d' % (chip, q.NQ))
            live5 = out['blocks'].get('%s|%d' % (chip, NQ5))
            blk = (live[0] if live else None)
            out['calibration'][chip] = heb.try_vendor_calibration(
                chip, block=blk,
                block_source=('live best_qubit_blocks(%d)' % q.NQ) if blk
                else 'hw_error_budget.BLOCK (the 2026-09 WK_C180 run)')
            if live5:
                mb = measured_block_calibration(chip, live5[0], tok2)
                out['measured_block'][chip] = mb
                if mb.get('error'):
                    out['errors'].append('measured_block_calibration(%s): %s'
                                         % (chip, mb['error']))
    except Exception as exc:
        out['errors'].append('vendor calibration: %r' % (exc,))
    if verbose:
        print('[preflight] blocks: %s' % json.dumps(out['blocks']), flush=True)
        for chip, mb in out['measured_block'].items():
            if not mb.get('available'):
                print('[preflight] %s measured calibration UNAVAILABLE: %s'
                      % (chip, mb.get('error')), flush=True)
                continue
            pm, pq = mb['measured_prediction'], mb['quoted_prediction']
            print('[preflight] %s block %s connected=%s  MEASURED f1q=%.5f '
                  'f2q=%.5f fread=%.5f' % (chip, mb['block'], mb['connected'],
                                           mb['f1q'], mb['f2q'], mb['fread']),
                  flush=True)
            print('[preflight]   predicted radius  measured calib %.4f  vs quoted '
                  'table %.4f   -> signal 1-r %.4f vs %.4f'
                  % (pm['radius'], pq['radius'], 1 - pm['radius'],
                     1 - pq['radius']), flush=True)
            if mb.get('sigma_r_below_1') is not None:
                print('[preflight]   at %d shots/circuit (n_eff %.0f after '
                      'code-space post-selection, CI +-%.4f): r<1 is %.2f sigma '
                      'on the MEASURED calibration, and would have been %.2f sigma '
                      'had the quoted table been right'
                      % (mb['shots'], mb['n_effective_shots'], mb['ci_half_width'],
                         mb['sigma_r_below_1'],
                         mb['sigma_if_quoted_table_were_true']), flush=True)
                print('[preflight]   => the quoted table UNDERSTATES this device; '
                      'the mixedness headline is weaker than sec.7-8 predicted.  '
                      'Block choice stays %s' % BLOCK_SELECTION_RULE, flush=True)
        for e in out['errors']:
            print('[preflight] %s' % e, flush=True)
    return out


def query_best_block(chip_id, nq, token):
    """The top-ranked physical block of size nq on `chip_id` (free query)."""
    _svc, be = _backend(chip_id, token)
    bb = be.best_qubit_blocks(nq)
    blocks = list(bb.qubit_blocks) if bb.qubit_blocks else []
    return [int(x) for x in blocks[0]] if blocks else None


def query_blocks(chip_id, nq, token):
    """ALL ranked physical blocks of size nq on `chip_id`, best first (free query).

    `query_best_block` returns only the top one, which stops being enough once a
    block has to be REJECTED on measured grounds: the vendor's ranking knows
    nothing about the day-of readout fault that disqualified
    WARMCOLD_BLOCK_REJECTED_2026_10_12, and may hand it back unchanged."""
    _svc, be = _backend(chip_id, token)
    bb = be.best_qubit_blocks(nq)
    blocks = list(bb.qubit_blocks) if bb.qubit_blocks else []
    return [[int(x) for x in b] for b in blocks]


def science_meas_layer():
    """The science circuit's MEAS layer on its own, with no angles and no noise.

    `layout_from_meas` inspects only the MEAS(q -> cbit) tuples, so the bit window
    can be derived without building -- or loading the trained angles for -- a whole
    206-instruction circuit.  Kept as a function so the screen and the full
    analysis cannot drift onto different reference maps."""
    return list(q.extraction_measures()) + list(q.data_measures())

def run_cal9_screen(token, chip_id, shots=CAL9_SCREEN_SHOTS, block=None,
                    exclude=None, dump_raw=None, job_timeout=3600,
                    use_options=True, mapping=True, verbose=True,
                    budget_remaining=None, ms_per_shot=None):
    """Pay for the two calibration circuits ONLY and report whether the block is
    usable -- before any science circuit is bought.

    WHY THIS IS A SEPARATE JOB AND NOT `--cal-first`.  cal-first reorders a single
    submission, so calibration and science are paid for together: under a hard
    budget that saves the DATA if the job is killed, but not the MONEY, because the
    science circuits are already in the queue.  Only a separate submission has a
    real decision point between the screen and the spend.  Both exist because they
    buy different things.

    COST.  2 x `shots` = 2000 shots at the default, i.e. 28 s at the pessimistic
    14 ms/shot.  That is deliberately small because the remaining allocation
    CANNOT be read from the API: QCloudService exposes
    backend/backends/best_qubit_blocks/setup_logging and ChipInfo only topology and
    calibration, with no balance, quota or usage query anywhere in the SDK.  So
    `budget_remaining` -- the seconds the account holder reads off the dashboard --
    is a REQUIRED argument and check_budget refuses the submission without it.
    This screen is also what turns the billing rate from a hypothesis into a
    measurement: read the dashboard again afterwards and (before - after) / 2000 is
    the billed ms/shot, which is the only such measurement that exists.

    BLOCK CHOICE.  `block` if given, else the best-ranked block that is not in
    `exclude` (default: the blocks rejected on measured grounds ON THIS CHIP,
    rejected_blocks_for).  The exclusion is the whole point: the vendor's own top
    pick on WK_C180_2 is the block whose DATA qubit q3 measured contrast -0.156,
    nothing free predicts that (ChipInfo reported the channel that measured 0.4218
    at 0.9612, and its own worst qubit at 0.9014 measured perfect), so a rerun on
    the vendor's pick reproduces the fault.  If every ranked block is excluded this
    raises rather than silently falling back to a known-bad one.

    Returns the screen dict and what was submitted, including the `billing` ledger
    entry.  Submits NO science circuit and makes no claim about warm versus cold."""
    if exclude is None:
        exclude = rejected_blocks_for(chip_id)
    dump_raw = dump_raw or unique_dump_path('cal9screen', chip_id)
    bill = check_budget('cal9screen', 2 * shots, budget_remaining, ms_per_shot,
                        verbose=verbose)
    nq = STAGE_NQ['warmcold']
    if block is None:
        if not use_options:
            raise ValueError('cal9 screen needs a pinned block when vendor options '
                             'are off: with no mapping and no block nothing chooses '
                             'the qubits')
        ranked = query_blocks(chip_id, nq, token)
        banned = {tuple(sorted(b)) for b in exclude}
        cand = [b for b in ranked if tuple(sorted(b)) not in banned]
        if verbose:
            print('[cal9screen] best_qubit_blocks(%d) returned %d block(s); %d '
                  'excluded on measured grounds'
                  % (nq, len(ranked), len(ranked) - len(cand)), flush=True)
            for b in ranked[:5]:
                print('[cal9screen]   %s%s'
                      % (b, '   <-- REJECTED (2026-10-12 dead data channel q3)'
                         if tuple(sorted(b)) in banned else ''), flush=True)
        if not cand:
            raise NoEligibleBlock(
                'every ranked %d-qubit block is on the exclusion list %r, so there '
                'is no block to screen; pass --block explicitly rather than falling '
                'back to a block already known to have a dead data channel'
                % (nq, sorted(banned)))
        block = cand[0]
    if any(sorted(block) == sorted(b) for b in REJECTED_BLOCKS.get(chip_id, ())):
        # Printed regardless of `exclude`, so that allowing a rejected block is
        # always visible in the log and can never happen silently.
        print('[cal9screen] WARNING: block %s on %s is one this file REJECTS on '
              'measured grounds (2026-10-12: logical q3 / physical 85 measured '
              'contrast -0.156, a dead DATA channel).  Screening it answers '
              'whether that fault is persistent, NOT which block to use -- '
              'best_qubit_blocks(9) offers no alternative and its record carries '
              'no per-qubit quality data.' % (block, chip_id), flush=True)
    gates = [build_cal9('zero'), build_cal9('one')]
    meta = [dict(kind='cal9', cal='zero', rep=0),
            dict(kind='cal9', cal='one', rep=0)]
    # written BEFORE submission so a cut-off job still leaves the meta describing
    # whatever prefix was dumped -- same reasoning as run_stage
    write_meta_sidecar(dump_raw, meta, gates, 'cal9screen', chip_id, block, shots,
                       mapping=mapping, use_options=use_options)
    if verbose:
        print('[cal9screen] chip=%s block=%s shots/circuit=%d -> %d shots = %.1f s '
              'at %.3f ms/shot; mapping=%s'
              % (chip_id, block, shots, len(gates) * shots, bill['cost_s'],
                 bill['ms_per_shot'], mapping), flush=True)
    return _finish_cal9_screen(token, chip_id, shots, block, gates, meta, dump_raw,
                               job_timeout, use_options, mapping, verbose, bill)


def _finish_cal9_screen(token, chip_id, shots, block, gates, meta, dump_raw,
                        job_timeout, use_options, mapping, verbose, bill=None):
    """Submit the screened calibration pair and report the block verdict.

    Split out of run_cal9_screen only to keep that function readable; it is not
    meant to be called on its own."""
    t0 = time.time()
    results = q.submit_cloud(gates, token, chip_id=chip_id, shots=shots, batch=True,
                             dump_raw=dump_raw, chunk=len(gates),
                             job_timeout=job_timeout, use_options=use_options,
                             block=block, mapping=mapping)
    wall = time.time() - t0
    anc, rev = layout_from_meas(science_meas_layer())
    got = [(r, m) for r, m in zip(results, meta) if r]
    if len(got) < 2:
        print('[cal9screen] INCONCLUSIVE: only %d of 2 calibration circuits '
              'returned, so e01/e10 cannot be formed -- both a |0^9> and a |1^9> '
              'preparation are needed.  The block is NOT cleared; an absent screen '
              'is not a passing one.' % len(got), flush=True)
        return dict(available=False, block=block, chip_id=chip_id, shots=shots,
                    wall_s=wall, dump_raw=dump_raw, n_returned=len(got),
                    cal9=None, screen=None, block_usable=False, billing=bill,
                    note='truncated return; no screen, no clearance')
    c9 = _analyze_cal9(got, anc, rev)
    s = c9['screen']
    if verbose:
        print('[cal9screen] %d/%d and %d/%d shots; mean readout fidelity %.4f '
              '(%.4f over healthy channels)'
              % (int(sum(got[0][0].values())), shots,
                 int(sum(got[1][0].values())), shots,
                 c9['mean_readout_fidelity'],
                 c9['mean_readout_fidelity_healthy']), flush=True)
        for lab in (['data%d' % i for i in range(q.NQ_DATA)]
                    + ['anc%d' % i for i in range(q.NQ_ANC)]):
            d = c9['per_qubit'][lab]
            print('[cal9screen]   %-6s logical q%-2d e01=%.4f e10=%.4f fid=%.4f '
                  'contrast=%+.4f  %s'
                  % (lab, d['logical_qubit'], d['e01'], d['e10'],
                     d['readout_fidelity'], d['contrast'],
                     'BROKEN' if d['broken'] else 'ok'), flush=True)
        print('[cal9screen] SCREEN: %s' % s['note'], flush=True)
        print('[cal9screen] VERDICT: block %s is %s -- primary estimator %s, '
              'post-selected F_s %s'
              % (block, 'USABLE' if s['block_usable'] else 'NOT USABLE',
                 'valid' if s['primary_estimator_valid'] else 'INVALID',
                 'valid' if s['secondary_estimator_valid'] else 'uninterpretable'),
              flush=True)
        n_all = WARMCOLD_N_SCIENCE + WARMCOLD_N_CAL
        if s['block_usable']:
            print('[cal9screen] -> science circuits MAY be bought on this block, '
                  'but the NEXT gate is --wc-probe, not the A/B: 1 circuit x %d '
                  'shots = %.1f s, and it measures the data-register survival w '
                  'that sets the A/B\'s power.  The full warmcold stage at '
                  'STAGE_SHOTS=%d is %d circuits x %d shots = %.0f s at the '
                  'pessimistic %.0f ms/shot, so buying it before w is known risks '
                  'spending the allocation on a bound.  ALSO: read the dashboard '
                  'again now -- (before - after) / %d shots is the billed ms/shot, '
                  'the only measurement of it that exists.'
                  % (shots, cost_s(shots, bill['ms_per_shot'] if bill else None),
                     STAGE_SHOTS['warmcold'], n_all, STAGE_SHOTS['warmcold'],
                     cost_s(n_all * STAGE_SHOTS['warmcold']), MS_PER_SHOT,
                     2 * shots), flush=True)
        else:
            print('[cal9screen] -> DO NOT buy science circuits on this block.  '
                  'Re-screen another ranked block (%d shots = %.1f s each).  A dead '
                  'DATA channel makes the primary estimator uninterpretable '
                  'however many shots are spent on it; a dead ANCILLA would only '
                  'cost the post-selected F_s.'
                  % (len(gates) * shots,
                     cost_s(len(gates) * shots,
                            bill['ms_per_shot'] if bill else None)), flush=True)
    return dict(available=True, block=block, chip_id=chip_id, shots=shots,
                wall_s=wall, dump_raw=dump_raw, n_returned=len(got),
                layout=dict(anc_slice=[anc.start, anc.stop], rev=rev),
                cal9=c9, screen=s, block_usable=bool(s['block_usable']),
                billing=bill)


# ---------------------------------------------------------------------
# the warm/cold PROBE: buy ONE science circuit and measure w before buying six
# ---------------------------------------------------------------------
# WHY A PROBE EXISTS.  The A/B's observable gap is w * 0.500, where w is the
# data-register survival P(data=00000) rescaled above the 1/32 uniform floor, and w
# is the ONE quantity that decides whether six circuits buy a detection or a bound.
# Two committed measurements of it disagree by 8x and they differ in two variables
# at once -- chip and compiler options:
#     2026-09  WK_C180    no QCloudOptions  w = 0.0936 (16 branches, ~2000 shots)
#     2026-10  WK_C180_2  with options      w = 0.0111 (4 circuits, 3000 shots,
#                                           and one data channel was dead)
# At w = 0.0936 and S = 2000 the A/B is a 5.0 sigma detection; at w = 0.0111 it is
# 0.6 sigma, i.e. a bound.  Buying the six circuits before w is known therefore
# risks spending the whole remaining allocation on a number whose size was already
# predictable.  One circuit at 1000 shots costs 14 s at the pessimistic rate and
# measures w to +-0.011, which is enough to tell those two worlds apart.
#
# The probe is the SIGNAL|WARM entry of the pre-registered stage, byte-identical to
# it (asserted, not assumed), so it is not a new experiment and needs no new
# pre-registration: it is one sixth of the job, bought first.
WC_PROBE_SHOTS = 1000
WC_PROBE_NOISELESS_SHOTS = 1000
WC_PROBE_SHOT_LADDER = (1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000,
                        10000, 12000, 15000, 20000)
WC_PROBE_MIN_SIGMA = 3.0          # the A/B's own pre-registered detection bar
WC_PROBE_MIN_STRUCTURE_Z = 3.0    # P(data=00000) above the 1/32 uniform floor
WC_PROBE_MIN_RETURN_FRACTION = 0.9


def resolve_block(chip_id, nq, token, block=None, exclude=None, tag='block',
                  verbose=True):
    """The physical block a submission lands on, chosen by the pre-registered rule
    and never by an outcome: `block` if the caller pinned one, else the top-ranked
    best_qubit_blocks(nq) entry that is not rejected on measured grounds for THIS
    chip.

    Same rule as run_cal9_screen's, factored out so the screen, the probe and the
    stage cannot drift onto different blocks -- which would silently invalidate the
    screen's clearance of the block the science circuits actually use.  Raises
    rather than falling back if every ranked block is rejected."""
    rejected = {tuple(sorted(b)) for b in REJECTED_BLOCKS.get(chip_id, ())}
    if block is not None:
        block = [int(x) for x in block]
        if tuple(sorted(block)) in rejected and verbose:
            print('[%s] WARNING: block %s on %s is one this file rejects on measured '
                  'grounds; it is being used because it was pinned explicitly'
                  % (tag, block, chip_id), flush=True)
        return block
    ranked = query_blocks(chip_id, nq, token)
    banned = ({tuple(sorted(b)) for b in exclude} if exclude is not None
              else rejected)
    cand = [b for b in ranked if tuple(sorted(b)) not in banned]
    if verbose:
        print('[%s] best_qubit_blocks(%d) on %s returned %d block(s); %d excluded '
              'on measured grounds' % (tag, nq, chip_id, len(ranked),
                                       len(ranked) - len(cand)), flush=True)
        for b in ranked[:5]:
            print('[%s]   %s%s' % (tag, b,
                                   '   <-- REJECTED on measured grounds'
                                   if tuple(sorted(b)) in rejected else ''),
                  flush=True)
    if not cand:
        raise NoEligibleBlock(
            'every ranked %d-qubit block on %s is on the exclusion list %r, so '
            'there is no block to run on; pass --block explicitly rather than '
            'falling back to a block already known to have a dead data channel'
            % (nq, chip_id, sorted(banned)))
    return [int(x) for x in cand[0]]


def canonical_to_native_key(canon, gates, nq=None):
    """hw_mixedness's canonical key -> the LOCAL SIMULATOR's native key.

    THE TRAP, caught by selftest (o) rather than by the hardware run.  These two are
    NOT related by a global reversal, and assuming they are produces a key whose
    ancilla nibble is backwards:
      * canonical (exact_probs, canonical_key): character j is LOGICAL QUBIT j;
      * CPUQVM native: key position p holds CLASSICAL BIT nq-1-p, and the classical
        bit a logical qubit lands in comes from the circuit's own MEAS map, which for
        the science circuit is q5->c0..q8->c3, q0->c4..q4->c8.
    So the conversion has to go through cbit_of_qubit.  Worked example from the
    signal|warm branch, whose ideal outcome is q6=q8=1: canonical '000000101' ->
    native '000001010' -> vendor '101000000', and the last of those is what the QPU
    and cpuqvm_sample both return."""
    nq = nq or q.NQ
    assert len(canon) == nq, (len(canon), nq)
    native = ['0'] * nq
    for qb in range(nq):
        native[nq - 1 - cbit_of_qubit(qb, gates, nq=nq)] = canon[qb]
    return ''.join(native)


def expected_vendor_key(gates, n=None):
    """The ONE key the ideal circuit puts all its weight on, in the VENDOR's order.

    Derived, not sampled and not read off the hardware: exact_probs gives the ideal
    512-dim distribution in the canonical order, its argmax is the ideal outcome,
    canonical_to_native_key puts that in the simulator's native order and
    sim_key_to_vendor_key in the order the QPU returns.  At p=0 with a deterministic
    syndrome the ideal distribution is a delta, so the argmax is unambiguous -- and
    selftest (o) asserts the result equals the mode of an independent noiseless
    sample, which is what makes the two key-order conversions check each other."""
    n = n or q.NQ
    pv = exact_probs(gates, n=n)
    best = max(pv.items(), key=lambda kv: kv[1])
    native = canonical_to_native_key(best[0], gates, nq=n)
    return q.sim_key_to_vendor_key(native, nq=n), float(best[1])


def wc_probe_power(w_point, w_lo95, ms_per_shot=None, budget_remaining=None,
                   ladder=WC_PROBE_SHOT_LADDER):
    """What each rung of the shot ladder would buy, and which rung fits.

    One row per candidate per-circuit shot count S: the A/B's total shots (6S), its
    cost at the operative rate and at the pessimistic one, whether it fits the
    remaining budget with BUDGET_MARGIN_S held back, the A/B's z at the measured w
    and at w's lower 95% bound, and -- the number that makes a null worth buying --
    the 95% CL upper bound on w that a null at that S would report.  Pure arithmetic
    on warmcold_power, so the ladder cannot disagree with selftest (l)."""
    w_pt, w_lo = max(0.0, w_point), max(0.0, w_lo95)
    n_ab = WARMCOLD_N_SCIENCE + WARMCOLD_N_CAL
    rows = []
    for S in ladder:
        pw = warmcold_power(w=w_pt, shots=S)
        pl = warmcold_power(w=w_lo, shots=S)
        c = cost_s(n_ab * S, ms_per_shot)
        rows.append(dict(
            shots_per_circuit=S, n_shots_total=n_ab * S, cost_s=c,
            cost_s_pessimistic=cost_s(n_ab * S),
            fits_remaining_budget=bool(
                budget_remaining is not None
                and c <= float(budget_remaining) - BUDGET_MARGIN_S),
            sigma_z_at_w_point=pw['sigma_z'], sigma_z_at_w_lo95=pl['sigma_z'],
            sigma_pair=pw['sigma_pair'],
            w_bound95_if_null=(1.96 * pw['sigma_pair']
                               / WARMCOLD_GAP_IDEAL_SIGNAL)))
    aff = [r for r in rows if r['fits_remaining_budget']]
    row = aff[-1] if aff else None
    return dict(ladder=rows, n_circuits_in_ab=n_ab,
                ms_per_shot=(MS_PER_SHOT if ms_per_shot is None
                             else float(ms_per_shot)),
                ms_per_shot_source=ms_per_shot_source_label(ms_per_shot),
                budget_remaining_after=budget_remaining,
                margin_s=BUDGET_MARGIN_S,
                s_star=row['shots_per_circuit'] if row else None,
                sigma_z_at_s_star=row['sigma_z_at_w_point'] if row else None,
                w_bound95_at_s_star=row['w_bound95_if_null'] if row else None,
                min_sigma=WC_PROBE_MIN_SIGMA,
                detection_affordable=bool(
                    row and row['sigma_z_at_w_point'] >= WC_PROBE_MIN_SIGMA))


def analyze_wc_probe(counts, meta, gates, phi_warm, shots_requested=None,
                     chip_id=None, block=None, billing=None, ms_per_shot=None,
                     budget_remaining_after=None, verbose=True):
    """Read the probe's counts and answer the two questions that gate the A/B.

    QUESTION 1 (pipeline): did the platform return what was asked for -- one circuit,
    keys as wide as the block, the window layout_from_meas derives from the MEAS map,
    and (on the noiseless simulator) total acceptance?  A 'no' is exit 4 and stops the
    run: no shot count repairs a key that came back five bits wide.

    QUESTION 2 (power): how large is w, the data-register survival, and therefore how
    large would the A/B's observable gap w*0.500 be?  wc_probe_power turns that into a
    ladder of affordable shot counts and says whether the largest one clears the stage's
    own pre-registered 3 sigma bar.  Below it the six circuits buy a 95% CL upper bound
    on w instead of a detection, and the ladder states what that bound would be at every
    rung, so the choice is arithmetic rather than hope.

    This is NOT a warm-versus-cold claim: one arm cannot form a gap.  It measures the
    scale a gap would need to have to be visible, on the day, on this block."""
    shots_requested = shots_requested or WC_PROBE_SHOTS
    anc, rev = layout_from_meas(science_meas_layer())
    ds = _data_slice_for(anc)
    nq_block = len(block) if block else q.NQ
    widths = sorted({len(k) for k in counts})
    n_tot = int(sum(counts.values()))
    ek, ek_ideal = expected_vendor_key(gates)
    out = dict(kind='wcprobe', chip_id=chip_id, block=block, meta=meta,
               n_instructions=len(gates), shots_requested=shots_requested,
               shots_returned=n_tot,
               return_fraction=n_tot / float(max(shots_requested, 1)),
               primary_estimator=WARMCOLD_PRIMARY,
               layout=dict(anc_slice=[anc.start, anc.stop], rev=rev,
                           data_slice=[ds.start, ds.stop],
                           source='layout_from_meas(science_meas_layer()), never '
                                  'detect_layout -- selftest (m)'),
               key_widths=widths, nq_block=nq_block,
               expected_key=ek, expected_key_ideal_prob=ek_ideal,
               billing=billing)
    # ---- question 1: the pipeline, checked BEFORE any physics is quoted
    problems = []
    if widths != [nq_block]:
        problems.append(
            'returned keys are %s bits wide but the block has %d qubits, so the '
            'window layout_from_meas derived does not apply to this data'
            % (widths, nq_block))
    if n_tot < WC_PROBE_MIN_RETURN_FRACTION * shots_requested:
        problems.append(
            'only %d of %d requested shots returned (%.0f%%), below the %.0f%% '
            'floor: the platform shorted the job, so the power table would be '
            'built on a shot count that was not the one bought'
            % (n_tot, shots_requested, 100.0 * n_tot / max(shots_requested, 1),
               100.0 * WC_PROBE_MIN_RETURN_FRACTION))
    mv = dict(meta)
    mv['phi'] = phi_warm     # verify_layout_noiseless computes its own ideal ref
    nl = verify_layout_noiseless([gates], [mv], shots=WC_PROBE_NOISELESS_SHOTS,
                                 anc=anc, rev=rev)
    out['noiseless_check'] = nl
    if not nl['layout_verified']:
        problems.append(
            'the noiseless cross-check did NOT verify this window (acceptance '
            '%.3f, max deviation %.1f sigma), so the window is wrong or a builder '
            'changed' % (nl['min_acceptance_rate'], nl['max_dev_in_sigma']))
    n_ek = int(sum(v for k, v in counts.items() if k == ek))
    p_ek = n_ek / float(max(n_tot, 1))
    out['expected_key_stats'] = dict(
        n_hits=n_ek, p_hw=p_ek, p_ideal=ek_ideal,
        eps_per_instruction=(1.0 - p_ek ** (1.0 / len(gates)) if p_ek > 0 else None),
        note='this circuit is %d instructions; the manuscript\'s device point '
             '(1.04e-2) is per instruction of the 204-instruction inventory of '
             'circuit D, so the two are comparable per instruction, not as raw '
             'probabilities' % len(gates))
    out['pipeline'] = dict(ok=not problems, problems=problems)

    # ---- question 2: w, and what the remaining budget can buy with it
    n_hit, n_dz = data_zero_counts(counts, anc, rev)
    p = n_hit / float(max(n_dz, 1))
    lo, hi = wilson(n_hit, n_dz)
    floor = WARMCOLD_W_FLOOR_UNIFORM
    scale = 1.0 / (1.0 - floor)
    sig_p = math.sqrt(max(p * (1.0 - p), 1.0 / max(n_dz, 1)) / max(n_dz, 1))
    z_floor = (p - floor) / sig_p if sig_p > 0 else 0.0
    out['primary'] = dict(
        n_hit=n_hit, n_total=n_dz, p=p, wilson95=[lo, hi], floor_uniform=floor,
        z_vs_floor=z_floor,
        structure_detected=bool(z_floor >= WC_PROBE_MIN_STRUCTURE_Z))
    out['w'] = dict(point=(p - floor) * scale, lo95=(lo - floor) * scale,
                    hi95=(hi - floor) * scale,
                    formula='(P(data=00000) - 1/32) / (1 - 1/32)')
    # per-bit polarisation: a dead data channel shows up as one bit pinned near zero
    # while the others carry the structure.  A diagnostic, not a gate -- the cal9
    # screen is the gate -- but it says WHICH channel to look at.
    per_bit = []
    for qb in range(q.NQ_DATA):
        cb = cbit_of_qubit(qb, gates, nq=q.NQ)
        pos = len(next(iter(counts))) - 1 - cb if counts else 0
        n0 = int(sum(v for k, v in counts.items()
                     if len(k) > cb and k[len(k) - 1 - cb] == '0'))
        p0 = n0 / float(max(n_tot, 1))
        per_bit.append(dict(logical_qubit=qb, cbit=cb, key_position=pos, p0=p0,
                            polarisation=2.0 * p0 - 1.0))
    out['per_bit'] = per_bit
    out['power'] = wc_probe_power(out['w']['point'], out['w']['lo95'],
                                  ms_per_shot, budget_remaining_after)
    if not out['pipeline']['ok']:
        out['verdict'], out['exit_code'] = 'PIPELINE FAILED', 4
    elif out['power']['detection_affordable']:
        out['verdict'], out['exit_code'] = 'DETECTION AFFORDABLE', 0
    else:
        out['verdict'], out['exit_code'] = 'BOUND EXPERIMENT', 5
    if verbose:
        _wc_probe_report(out)
    return out


def _wc_probe_report(out):
    """Print the probe verdict in the order the decision needs it: pipeline first,
    then w, then what the remaining budget can buy with that w."""
    pl, pr, w, pw = out['pipeline'], out['primary'], out['w'], out['power']
    print('[wcprobe] chip=%s block=%s  %d/%d shots returned (%.1f%%), key widths '
          '%s, %d-instruction circuit'
          % (out['chip_id'], out['block'], out['shots_returned'],
             out['shots_requested'], 100.0 * out['return_fraction'],
             out['key_widths'], out['n_instructions']), flush=True)
    for s in pl['problems']:
        print('[wcprobe]   PROBLEM: %s' % s, flush=True)
    print('[wcprobe] PIPELINE: %s (noiseless acceptance %.3f, deviation %.1f sigma; '
          'expected key %s at P_ideal=%.4f, measured P=%.4f)'
          % ('OK' if pl['ok'] else 'FAILED',
             out['noiseless_check']['min_acceptance_rate'],
             out['noiseless_check']['max_dev_in_sigma'], out['expected_key'],
             out['expected_key_ideal_prob'], out['expected_key_stats']['p_hw']),
          flush=True)
    print('[wcprobe] PRIMARY %s = %.4f  [%.4f, %.4f] Wilson 95%%, uniform floor '
          '%.4f -> z_vs_floor = %+.2f  %s'
          % (out['primary_estimator'], pr['p'], pr['wilson95'][0],
             pr['wilson95'][1], pr['floor_uniform'], pr['z_vs_floor'],
             'STRUCTURE DETECTED' if pr['structure_detected']
             else 'NO structure above the floor'), flush=True)
    print('[wcprobe] implied data-register survival w = %+.4f  [%+.4f, %+.4f]  (%s)'
          % (w['point'], w['lo95'], w['hi95'], w['formula']), flush=True)
    print('[wcprobe] per-bit polarisation P(0)-P(1): %s'
          % '  '.join('q%d=%+.3f' % (d['logical_qubit'], d['polarisation'])
                      for d in out['per_bit']), flush=True)
    print('[wcprobe] the A/B is %d circuits; rate %.3f ms/shot (%s); %s s remain '
          'after the probe, %.0f s held back as cut-off margin'
          % (pw['n_circuits_in_ab'], pw['ms_per_shot'], pw['ms_per_shot_source'],
             ('%.1f' % pw['budget_remaining_after']
              if pw['budget_remaining_after'] is not None else 'UNREAD'),
             pw['margin_s']), flush=True)
    print('[wcprobe]   %8s %11s %8s %6s %8s %10s %13s'
          % ('S/circ', 'total shots', 'cost s', 'fits', 'z(w)', 'z(w_lo95)',
             'w<95% if null'), flush=True)
    for r in pw['ladder']:
        print('[wcprobe]   %8d %11d %8.1f %6s %8.2f %10.2f %13.4f%s'
              % (r['shots_per_circuit'], r['n_shots_total'], r['cost_s'],
                 'yes' if r['fits_remaining_budget'] else 'no',
                 r['sigma_z_at_w_point'], r['sigma_z_at_w_lo95'],
                 r['w_bound95_if_null'],
                 '   <-- S*' if r['shots_per_circuit'] == pw['s_star'] else ''),
              flush=True)
    if pw['s_star'] is None:
        print('[wcprobe] VERDICT: NO RUNG OF THE LADDER FITS the remaining budget, '
              'so do not submit the A/B.  Top up the allocation or report the bound '
              'the probe itself already sets: w < %.4f (95%% CL).'
              % w['hi95'], flush=True)
    else:
        cost = [r['cost_s'] for r in pw['ladder']
                if r['shots_per_circuit'] == pw['s_star']][0]
        print('[wcprobe] VERDICT: %s at S*=%d shots/circuit (%d circuits = %.1f s, '
              'z=%.2f at the measured w).'
              % (out['verdict'], pw['s_star'], pw['n_circuits_in_ab'], cost,
                 pw['sigma_z_at_s_star']), flush=True)
        if out['verdict'] == 'DETECTION AFFORDABLE':
            print('[wcprobe] -> the A/B is a DETECTION experiment at S*: submit it '
                  'with --cal-first so a cut-off still leaves the block screen.',
                  flush=True)
        else:
            print('[wcprobe] -> the A/B would NOT reach %.0f sigma, so at S* it '
                  'reports w < %.4f (95%% CL) rather than a detection.  That is a '
                  'characterised measurement and it does close the 2026-10-12 '
                  'INVALID, but whether the remaining allocation should buy a bound '
                  'is a human decision, not this script\'s.'
                  % (WC_PROBE_MIN_SIGMA, pw['w_bound95_at_s_star']), flush=True)
    return out


def run_wc_probe(token, chip_id, phi_warm, phi_cold, shots=WC_PROBE_SHOTS,
                 block=None, exclude=None, dump_raw=None, job_timeout=3600,
                 use_options=True, mapping=True, verbose=True,
                 budget_remaining=None, ms_per_shot=None):
    """Buy ONE science circuit -- the stage's signal|warm entry -- and analyse it.

    This is the second of the three gates in front of the warm-vs-cold A/B (cal9
    screen, probe, A/B) and the only one that measures the effect SIZE rather than
    the apparatus.  It submits a single circuit, so it costs `shots` shots: 1000 is
    14 s at the pessimistic rate, against 168 s for the six-circuit A/B at 2000.

    The circuit is not a new one.  It is selected out of stage_circuits('warmcold')
    by (role, arm) and asserted byte-identical to a fresh build of the pre-registered
    signal branch, so the probe measures the same physical circuit the A/B will and
    cannot drift from the manuscript's pre-registration.  Its meta is passed through
    unchanged, which is what lets the sidecar and --mode analyze verify it later.

    Returns the probe dict (see analyze_wc_probe) including `exit_code`: 0 detection
    affordable, 5 the A/B would only set a bound, 4 pipeline failure.  Submits no
    other circuit and makes no warm-versus-cold claim."""
    gates_all, meta_all = stage_circuits('warmcold', phi_warm, phi_cold)
    sel = [i for i, m in enumerate(meta_all)
           if m.get('kind') == 'warmcold' and m.get('role') == 'signal'
           and m.get('arm') == 'warm']
    assert len(sel) == 1, ('the pre-registered stage must contain exactly one '
                           'signal|warm science circuit', sel)
    gates, meta = [gates_all[sel[0]]], [meta_all[sel[0]]]
    m0 = meta[0]
    assert (m0['psi'], m0['s'], m0['frame'], m0['measure'], m0['p']) == \
        (WARMCOLD_PSI, WARMCOLD_SIGNAL_S, WARMCOLD_SIGNAL_FRAME, 'late', 0.0), m0
    assert m0['angles'] == WARMCOLD_ANGLES['warm'], m0
    assert gates[0] == q.build_branch_circuit(
        q.ENC5, WARMCOLD_PSI, WARMCOLD_SIGNAL_FRAME, WARMCOLD_SIGNAL_S, phi_warm,
        late_measure=True), 'the probe circuit is not the pre-registered one'
    # Budget BEFORE the block query: the cost does not depend on which block is
    # chosen, and a refusal to spend must be diagnosable without a working token.
    bill = check_budget('wcprobe', shots, budget_remaining, ms_per_shot,
                        verbose=verbose)
    nq = STAGE_NQ['warmcold']
    block = resolve_block(chip_id, nq, token, block, exclude, tag='wcprobe',
                          verbose=verbose)
    dump_raw = dump_raw or unique_dump_path('wcprobe', chip_id)
    write_meta_sidecar(dump_raw, meta, gates, 'wcprobe', chip_id, block, shots,
                       mapping=mapping, use_options=use_options)
    if verbose:
        print('[wcprobe] chip=%s block=%s 1 circuit x %d shots = %.1f s at %.3f '
              'ms/shot; mapping=%s; this measures w, it does NOT test warm vs cold'
              % (chip_id, block, shots, bill['cost_s'], bill['ms_per_shot'],
                 mapping), flush=True)
    t0 = time.time()
    results = q.submit_cloud(gates, token, chip_id=chip_id, shots=shots, batch=True,
                             dump_raw=dump_raw, chunk=1, job_timeout=job_timeout,
                             use_options=use_options, block=block, mapping=mapping)
    wall = time.time() - t0
    if not results or not results[0]:
        print('[wcprobe] INCONCLUSIVE: no counts returned, so w is unmeasured and '
              'the A/B is NOT cleared.  An absent probe is not a passing one.',
              flush=True)
        return _save_probe_sidecar(dict(
            kind='wcprobe', chip_id=chip_id, block=block, shots_requested=shots,
            shots_returned=0, dump_raw=dump_raw, wall_s=wall, billing=bill,
            pipeline=dict(ok=False, problems=['no counts returned']),
            verdict='INCONCLUSIVE', exit_code=4), verbose=verbose)
    # ms_per_shot here is what the CALLER supplied -- None unless --ms-per-shot was
    # given -- and NOT bill['ms_per_shot'].  check_budget resolves None to the
    # pessimistic constant MS_PER_SHOT, and handing that already-resolved number to
    # analyze_wc_probe made wc_probe_power stamp 'dashboard-measured' onto a rate
    # nobody had measured: probe A of 2026-10-15 recorded 14.000 ms/shot as
    # dashboard-measured when it was the assumed constant (corrected in its
    # hw_evidence log).  wc_probe_power resolves None itself, so every number is
    # unchanged and only the provenance label is fixed; `bill` keeps its own true
    # label, and the two now agree instead of contradicting each other.
    out = analyze_wc_probe(
        results[0], m0, gates[0], phi_warm, shots_requested=shots, chip_id=chip_id,
        block=block, billing=bill, ms_per_shot=ms_per_shot,
        budget_remaining_after=bill['budget_remaining_after'], verbose=verbose)
    out['dump_raw'] = dump_raw
    out['wall_s'] = wall
    out['shots_used'] = [int(sum(results[0].values()))]
    out['gates'] = gates
    return _save_probe_sidecar(out, verbose=verbose)


def probe_sidecar_path(dump_raw):
    """Where a probe's analysed verdict is stored, next to its raw counts.

    The artifact keeps ONE art['wcprobe'] key, so plan A's two probes -- one per
    chip -- would overwrite each other there and the chip decision would be made
    from a file that no longer says which chip it came from.  The sidecar is named
    after the (already unique, timestamped) raw dump, so N probes leave N verdicts on
    disk and --mode wcpick can read all of them."""
    return str(dump_raw) + '.probe.json'


def _save_probe_sidecar(out, verbose=True):
    """Write the probe verdict to its sidecar and record the path on the dict.

    The circuit gates stay out of the sidecar: they are already in the .meta.json
    next to the dump, and the A/B's own submission re-derives them."""
    path = probe_sidecar_path(out['dump_raw'])
    with open(path, 'w') as f:
        json.dump({k: v for k, v in out.items() if k != 'gates'}, f, indent=1,
                  default=str, ensure_ascii=False)
    out['probe_json'] = path
    if verbose:
        print('[wcprobe] verdict sidecar %s  (feed it to --mode wcpick '
              '--probe-json; the artifact keeps only the LAST probe, so two probes '
              'need two sidecars)' % os.path.basename(path), flush=True)
    return out


# Plan A's chip rule, declared BEFORE any of today's paid data was seen
# (QCLOUD_SETUP.md 9.7): the two offline priors contradict each other -- the vendor
# calibration says WK_C180_2's two-qubit gate error is 3.7x lower, the one measured w
# says WK_C180's survival is 10x higher -- so both chips are probed and the tie is
# broken by measurement.  When NEITHER w reaches the detection bar the A/B goes here,
# on the chip whose block the 2026-10-12 INVALID was measured on, so the rerun closes
# that record on the same hardware instead of on a different one.
WC_PICK_BOUND_CHIP = 'WK_C180_2'


def min_w_for_detection(shots, min_sigma=WC_PROBE_MIN_SIGMA,
                        gap_ideal=WARMCOLD_GAP_IDEAL_SIGNAL, tol=1e-9):
    """The smallest data-register survival w whose A/B reaches `min_sigma` at `shots`.

    Inverts warmcold_power's own z by bisection instead of restating a threshold in
    prose: the 0.045 quoted in QCLOUD_SETUP.md 9.7 is an OUTPUT of this function at
    S=2000, so the chip rule cannot drift away from the power model selftest (l)
    locks.  Returns None if even a perfect device (w=1) misses the bar at that shot
    count, which is the case where the A/B is unbuyable at any w.

    sigma_z is monotone non-decreasing in w -- the gap grows as w while the pooled
    binomial noise grows only as sqrt(pbar(1-pbar)) -- so bisection is exact to `tol`.
    selftest (p) checks that monotonicity on a grid rather than trusting this
    comment."""
    hi_z = warmcold_power(w=1.0, shots=shots, gap_ideal=gap_ideal)['sigma_z']
    if not hi_z or hi_z < min_sigma:
        return None
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        z = warmcold_power(w=mid, shots=shots, gap_ideal=gap_ideal)['sigma_z'] or 0.0
        if z >= min_sigma:
            hi = mid
        else:
            lo = mid
    return hi


def _wc_pick_decision(usable, ref, ms_per_shot, budget_remaining, refused=()):
    """Steps 2-4 of pick_wc_chip's rule, given at least one usable probe and an S*."""
    s_star = ref['s_star']
    w_min = min_w_for_detection(s_star)
    best = max(usable, key=lambda p: float(p['w']['point']))
    w_best = float(best['w']['point'])
    if w_min is not None and w_best >= w_min:
        chosen, mode = best, 'DETECTION'
        reason = ('best usable w=%.4f on %s clears the %.0f sigma bar at S*=%d, which '
                  'needs w>=%.4f, so the A/B is a DETECTION experiment there'
                  % (w_best, best['chip_id'], WC_PROBE_MIN_SIGMA, s_star, w_min))
    else:
        pref = [p for p in usable if p['chip_id'] == WC_PICK_BOUND_CHIP]
        chosen, mode = (pref[0] if pref else best), 'BOUND'
        reason = ('no usable w reached the %s that the %.0f sigma bar needs at S*=%d '
                  '(best was %.4f on %s), so the A/B reports a 95%% CL upper bound on '
                  'w and runs on %s'
                  % ('%.4f' % w_min if w_min is not None else 'UNREACHABLE w',
                     WC_PROBE_MIN_SIGMA, s_star, w_best, best['chip_id'],
                     chosen['chip_id']))
        if not pref:
            # Two DIFFERENT situations used to share one sentence, and that sentence
            # asserted the stronger of them: that the bound chip's probe "was refused",
            # i.e. that it ran and failed its pipeline gate.  When no probe for that
            # chip is among the inputs at all -- never bought, or still sitting in the
            # vendor's queue, as WK_C180_2's did for 77 minutes on 2026-10-15 --
            # `refused` is EMPTY, and the record nonetheless told a reader that a
            # measurement had rejected that chip.  It had not.  A missing probe and a
            # failed probe are different claims about the world, and this string is
            # evidence rather than a comment on evidence, so they are now worded
            # differently and each says which of the two it is.
            if any(r.get('chip_id') == WC_PICK_BOUND_CHIP for r in refused):
                reason += (", which is NOT %s because that chip's probe was REFUSED by "
                           "its pipeline gate and is therefore excluded (see `refused`)"
                           % WC_PICK_BOUND_CHIP)
            else:
                reason += (", which is NOT %s because NO probe for that chip is among "
                           "the %d input(s): it was not run or did not return.  That is "
                           "NOT a pipeline refusal -- `refused` is empty -- and no "
                           "measurement of that chip is implied by this choice"
                           % (WC_PICK_BOUND_CHIP, len(usable) + len(refused)))
        elif w_min is None:
            reason += ('; note the bar is unreachable at S*=%d even for a perfect '
                       'device' % s_star)
    pw = wc_probe_power(float(chosen['w']['point']), float(chosen['w']['lo95']),
                        ms_per_shot, budget_remaining)
    assert pw['s_star'] == s_star, ('S* must depend on the budget and the rate only, '
                                    'not on w', pw['s_star'], s_star)
    return dict(mode=mode, exit_code=0 if mode == 'DETECTION' else 5,
                chip_id=chosen['chip_id'], block=chosen['block'], s_star=s_star,
                w_min_for_detection=w_min, w_best_usable=w_best,
                best_w_chip=best['chip_id'], reason=reason, power=pw,
                chosen_w=float(chosen['w']['point']),
                chosen_w_ci=[float(chosen['w']['lo95']), float(chosen['w']['hi95'])],
                chosen_probe_shots=chosen.get('shots_returned'),
                chosen_dump_raw=chosen.get('dump_raw'))


def pick_wc_chip(probes, budget_remaining, ms_per_shot=None, verbose=True):
    """Apply the pre-registered chip rule to the probes that came back.  FREE.

    Submits nothing and reads no token: it is arithmetic on probe sidecars, so where
    the last ~200 s goes is reproducible from the evidence on disk instead of being a
    judgement call made at the terminal.

    THE RULE (QCLOUD_SETUP.md 9.7, written before today's paid data):
      1. a probe that failed its pipeline gate (exit 4) is NOT a candidate -- an
         unusable block or a truncated return says nothing about w;
      2. S* is the largest ladder rung that fits `budget_remaining` at the operative
         rate.  It depends on the budget and the rate ONLY, not on w, which is what
         makes step 3 a threshold rather than a trade-off;
      3. if the best usable w >= min_w_for_detection(S*), the A/B is a DETECTION
         experiment and goes on that chip;
      4. otherwise the A/B reports a 95% CL upper bound on w wherever it runs, so it
         goes on WC_PICK_BOUND_CHIP -- the 2026-10-12 chip and block -- falling back
         to the best-w chip only if that one was refused (disclosed in `reason`).

    WHY THIS DOES NOT INFLATE THE A/B'S TYPE-I ERROR: the probes' shots are disjoint
    from the A/B's, and the null branch's ideal gap is 3e-6 on any chip, so choosing a
    chip on an independent measurement cannot manufacture a signal.  What it DOES cost
    is a winner's curse on w -- the probe resolves w to about +-0.0064 at 1000 shots --
    so w's interval must come from the A/B's own counts, never from the probe that
    selected the chip.  Both facts ride along in the returned dict, so they land in
    the evidence JSON next to the decision they qualify."""
    if budget_remaining is None:
        raise BudgetError(
            'wcpick: --budget-remaining is REQUIRED.  S* is the largest rung that fits '
            'the CURRENT dashboard reading, and each probe priced its ladder against '
            'its own post-probe remainder, which is already stale.')
    usable, refused = [], []
    for p in probes:
        cid = p.get('chip_id')
        if not cid or 'w' not in p or 'exit_code' not in p:
            raise ValueError(
                'wcpick: the input is not a --wc-probe verdict sidecar (needs chip_id, '
                'w and exit_code); pass the .probe.json files the probes wrote: %r'
                % (sorted(p)[:6] if isinstance(p, dict) else p))
        pl = p.get('pipeline') or {}
        if p.get('exit_code') == 4 or not pl.get('ok'):
            refused.append(dict(chip_id=cid, block=p.get('block'),
                                verdict=p.get('verdict'),
                                problems=list(pl.get('problems') or [])))
            continue
        usable.append(p)
    ref = wc_probe_power(0.0, 0.0, ms_per_shot, budget_remaining)
    base = dict(kind='wcpick', budget_remaining=float(budget_remaining),
                ms_per_shot=ref['ms_per_shot'],
                ms_per_shot_source=ref['ms_per_shot_source'],
                margin_s=ref['margin_s'], n_probes=len(probes), refused=refused,
                bound_chip=WC_PICK_BOUND_CHIP,
                rule=('QCLOUD_SETUP.md 9.7: probe both chips; detect on the higher w '
                      'if it clears the %.0f sigma bar at S*, else bound on %s'
                      % (WC_PROBE_MIN_SIGMA, WC_PICK_BOUND_CHIP)),
                winners_curse=('w must be reported from the A/B counts, not from the '
                               'probe that selected the chip'),
                usable=[dict(chip_id=p['chip_id'], block=p.get('block'),
                             w=float(p['w']['point']),
                             w_ci=[float(p['w']['lo95']), float(p['w']['hi95'])],
                             shots_returned=p.get('shots_returned'),
                             verdict=p.get('verdict')) for p in usable])
    if not usable:
        base.update(mode='STOP', exit_code=4, chip_id=None, block=None, s_star=None,
                    reason='every probe failed its pipeline gate, so no w was measured '
                           'and the A/B is NOT cleared')
    elif ref['s_star'] is None:
        base.update(mode='STOP', exit_code=4, chip_id=None, block=None, s_star=None,
                    reason='no rung of the shot ladder fits %.1f s minus the %.0f s '
                           'cut-off margin, so the A/B cannot be bought at any w'
                           % (float(budget_remaining), ref['margin_s']))
    else:
        base.update(_wc_pick_decision(usable, ref, ms_per_shot, budget_remaining,
                                      refused))
    if verbose:
        _wcpick_report(base)
    return base


def _wcpick_report(out):
    """Print the chip decision as a record: the readings it was applied to, the shot
    count it produced, and the exact command that spends the money."""
    print('[wcpick] %d probe(s) read; budget %.1f s at %.3f ms/shot (%s); %.0f s held '
          'back as cut-off margin'
          % (out['n_probes'], out['budget_remaining'], out['ms_per_shot'],
             out['ms_per_shot_source'], out['margin_s']), flush=True)
    for u in out.get('usable') or []:
        print('[wcpick]   usable  %-11s block=%s  w=%+.4f [%+.4f, %+.4f] from %s shots'
              '  %s' % (u['chip_id'], u['block'], u['w'], u['w_ci'][0], u['w_ci'][1],
                        u['shots_returned'], u['verdict']), flush=True)
    for r in out.get('refused') or []:
        print('[wcpick]   REFUSED %-11s block=%s  %s: %s'
              % (r['chip_id'], r['block'], r['verdict'],
                 '; '.join(r['problems']) or 'pipeline gate failed'), flush=True)
    if out['mode'] == 'STOP':
        print('[wcpick] VERDICT: STOP -- %s.  Do NOT submit the A/B; no larger shot '
              'count repairs this.' % out['reason'], flush=True)
        return out
    pw, s = out['power'], out['s_star']
    cost = [r['cost_s'] for r in pw['ladder'] if r['shots_per_circuit'] == s][0]
    print('[wcpick] detection needs w >= %.4f at S*=%d for the %.0f sigma bar; %s'
          % (out['w_min_for_detection'], s, WC_PROBE_MIN_SIGMA, out['reason']),
          flush=True)
    print('[wcpick] VERDICT: %s on chip=%s block=%s at S*=%d shots/circuit '
          '(%d circuits = %.1f s, %.1f s left after, z=%.2f at the probe\'s w, '
          'w < %.4f at 95%% CL if it comes back null)'
          % (out['mode'], out['chip_id'], out['block'], s, pw['n_circuits_in_ab'],
             cost, out['budget_remaining'] - cost, pw['sigma_z_at_s_star'],
             pw['w_bound95_at_s_star']), flush=True)
    print('[wcpick]   %s' % out['winners_curse'], flush=True)
    print('[wcpick] next: screen THIS chip first (%d circuits x %d shots = %.1f s), '
          'then submit the A/B with\n'
          '  CHIP=%s BUDGET=<post-screen reading> MSPS=%.3f SHOTS=%d%s \\\n'
          '      bash launch_mixedness.sh warmcold'
          % (WARMCOLD_N_CAL, CAL9_SCREEN_SHOTS,
             cost_s(WARMCOLD_N_CAL * CAL9_SCREEN_SHOTS, out['ms_per_shot']),
             out['chip_id'], out['ms_per_shot'], s,
             (' BLOCK=%s' % ','.join(str(b) for b in out['block']))
             if out['mode'] == 'BOUND' else ''), flush=True)
    return out


def run_stage(stage, token, chip_id, shots=None, block=None, chunk=None,
              job_timeout=21600, use_options=True, dump_raw=None, phi=None,
              verbose=True, phi_cold=None, mapping=True, cal_first=False,
              budget_remaining=None, ms_per_shot=None, exclude=None):
    """Submit one stage, chunked, dumping the raw counts after every chunk.

    The chunking is not a convenience: the account is billed in real time and a job
    CANNOT overrun its allocation (confirmed by the account holder, 2026-10), so a
    job that runs past the remaining budget is killed and returns nothing.  With
    chunk = len(gates) -- the old default -- that loses the entire stage AND the
    time it had already billed.  With STAGE_CHUNK[stage] the boundaries land on
    scientifically complete prefixes, so a kill costs only the chunk in flight.
    A chunk is still one submission and therefore one queue wait; STAGE_CHUNK is
    the trade-off between those two.

    `mapping=False` (with an explicit `block`) disables the vendor's server-side
    re-routing, so the logical-to-physical assignment is identical by construction
    for every circuit in the job.  It is a CONTROL, not a demonstrated necessity:
    the 2026-10-12 warmcold run had mapping ENABLED and its in-job calibration
    still transferred to the science circuits (all 36 channel comparisons inside
    their hard bounds), so there is no evidence that this platform re-routes
    logical qubits per circuit.  An earlier version of this docstring claimed
    otherwise; that claim rested on a bit-window error and is retracted.  Use
    mapping=False when you want the assignment guaranteed rather than measured --
    it costs nothing but does require an explicit `block`, since with mapping off
    nothing else will choose the qubits.  See qcloud_vscr_new._build_options.

    `cal_first=True` (warmcold only) reorders the stage so the two calibration
    circuits are submission 1 and forces chunk = WARMCOLD_N_CAL.  It exists
    because cal9's labels DO transfer to the science circuits -- the contrary
    finding is retracted in check_cal9_transferability -- so the calibration can
    decide whether the block is usable before the science circuits are bought.
    It does not make the calibration cheaper: billing is per shot, not per
    instruction, so at a common shot count cal9 costs what any other circuit
    costs.  The default is False so the 2026-10-12 artifact stays reproducible
    from the command line that produced it.

    `dump_raw` defaults to a NEW timestamped file, never to STAGE_RAW[stage]:
    those names are frozen evidence (FROZEN_EVIDENCE) holding the counts the
    manuscript's 2026-09/10-12/10-13 numbers are computed from, and submit_cloud
    opens its dump with 'w' after every chunk.  `budget_remaining` is the seconds
    left on the dashboard and is REQUIRED -- check_budget refuses the submission
    without it, because a job that overruns is killed and returns nothing."""
    shots = shots or STAGE_SHOTS[stage]
    dump_raw = dump_raw or unique_dump_path(stage, chip_id)
    gates, meta = stage_circuits(stage, phi, phi_cold, cal_first=cal_first)
    bill = check_budget('stage %s' % stage, len(gates) * shots, budget_remaining,
                        ms_per_shot, verbose=verbose)
    nq = STAGE_NQ[stage]
    if block is None and use_options:
        # The SAME rule the screen and the probe use (resolve_block): the vendor's
        # top-ranked block minus the ones rejected on measured grounds for this chip.
        # The query_best_block this replaces had no exclusion, so on WK_C180_2 --
        # whose best_qubit_blocks(9) returns exactly ONE block -- a rerun without
        # --block would have landed on the very block whose dead data channel the
        # screen exists to avoid.  A failed QUERY is tolerable (the server then
        # chooses qubits); "every ranked block is rejected" is not, and propagates.
        try:
            block = resolve_block(chip_id, nq, token, None, exclude,
                                  tag='stage %s' % stage, verbose=verbose)
        except NoEligibleBlock:
            raise
        except Exception as exc:
            print('[stage %s] best_qubit_blocks(%d) unavailable (%s); the server '
                  'will choose qubits' % (stage, nq, exc), flush=True)
    if verbose:
        inv = {}
        for g in gates:
            inv[len(g)] = inv.get(len(g), 0) + 1
        print('[stage %s] %d circuits on %d qubits, %d shots each, block=%s, '
              'instruction counts %s' % (stage, len(gates), nq, shots, block, inv),
              flush=True)
    t0 = time.time()
    if cal_first and chunk is None and stage == 'warmcold':
        # Make the calibration pair EXACTLY submission 1.  Under a hard real-time
        # budget a kill costs the chunk in flight, so with cal9 first a cut-off job
        # still leaves the block's readout screen on disk -- which is the one piece
        # of information that decides whether the science circuits were worth
        # buying, and the piece the 2026-10-12 run put last.
        n_chunk = WARMCOLD_N_CAL
        if verbose:
            print('[stage %s] cal-first: submission 1 is the %d calibration '
                  'circuit(s), so a budget kill still leaves the block screen'
                  % (stage, n_chunk), flush=True)
    else:
        n_chunk = chunk or STAGE_CHUNK.get(stage) or len(gates)
    # written BEFORE submission so a cut-off job still leaves the meta that
    # describes the prefix submit_cloud managed to dump -- see write_meta_sidecar
    write_meta_sidecar(dump_raw, meta, gates, stage, chip_id, block, shots,
                       mapping=mapping, use_options=use_options)
    if verbose and use_options and not mapping:
        print('[stage %s] vendor MAPPING DISABLED: every circuit is pinned to '
              'block %s, so this job\'s own calibration describes its science '
              'circuits' % (stage, block), flush=True)
    if verbose:
        print('[stage %s] %d circuits in %d submission(s) of <=%d, dumping to %s '
              'after each so a mid-run kill keeps every completed chunk'
              % (stage, len(gates), -(-len(gates) // n_chunk), n_chunk,
                 os.path.basename(dump_raw)), flush=True)
    results = q.submit_cloud(gates, token, chip_id=chip_id, shots=shots,
                             batch=True, dump_raw=dump_raw,
                             chunk=n_chunk, job_timeout=job_timeout,
                             use_options=use_options, block=block,
                             mapping=mapping)
    wall = time.time() - t0
    assert len(results) == len(gates), (len(results), len(gates))
    used = [int(sum(r.values())) for r in results]
    if verbose:
        print('[stage %s] received %d circuits in %.0f s wall; shots used %d/%d '
              'per circuit (min %d, max %d)'
              % (stage, len(results), wall, shots * len(gates), sum(used),
                 min(used), max(used)), flush=True)
    return dict(results=results, meta=meta, gates=gates, shots_requested=shots,
                shots_used=used, chip_id=chip_id, block=block, wall_s=wall,
                dump_raw=dump_raw, stage=stage, billing=bill)


# ---------------------------------------------------------------------
# 9) analysis: raw counts -> the five measured statements
# ---------------------------------------------------------------------
def analyze_probe(results, meta, shots_requested=None, n=NQ5, verbose=True):
    """Readout assignment matrix (M4), its drift, and the bit order (from the five
    fingerprint circuits, which determine it uniquely)."""
    zeros = [r for r, m in zip(results, meta) if m.get('cal') == 'zero']
    ones = [r for r, m in zip(results, meta) if m.get('cal') == 'one']
    fps = [(r, m['qubit']) for r, m in zip(results, meta)
           if m.get('kind') == 'fingerprint']
    assert len(zeros) == 2 and len(ones) == 1 and len(fps) == n, \
        (len(zeros), len(ones), len(fps))
    cal = assignment_from_cal(zeros[0], ones[0], n)
    cal2 = assignment_from_cal(zeros[1], ones[0], n)
    drift = [abs(a - b) for a, b in zip(cal['e01'], cal2['e01'])]
    fp_results = [r for r, _ in fps]
    fp_ideals = [exact_probs(build_fingerprint(b)) for _, b in fps]
    sigma, ev = detect_layout_ml(fp_results, fp_ideals, n, verbose=verbose)
    out = dict(assignment=cal, assignment_repeat_e01=cal2['e01'],
               e01_drift=drift, e01_drift_max=max(drift),
               n_used=[int(sum(r.values())) for r in results],
               shots_requested=shots_requested,
               layout_sigma=list(sigma), layout_evidence=ev,
               # the readout error of the logical qubit, in the position the
               # layout puts it -- the single number that sets the contrast loss
               logical_position=ev['logical_position'],
               logical_e01=cal['e01'][ev['logical_position']],
               logical_e10=cal['e10'][ev['logical_position']],
               mean_readout_error=0.5 * (sum(cal['e01']) / n
                                         + sum(cal['e10']) / n))
    if verbose:
        print('[probe] e01=%s\n[probe] e10=%s\n[probe] drift max=%.4f  readout '
              'fidelity=%.4f' % ([round(x, 4) for x in cal['e01']],
                                 [round(x, 4) for x in cal['e10']],
                                 max(drift), 1 - out['mean_readout_error']),
              flush=True)
    return sigma, out


def analyze_tomo(results, meta, sigma, cal, shots_requested=None, n=NQ5,
                 n_boot=2000, seed=20260921, vendor=None, verbose=True):
    """M1 mixedness, M2 linear exactness, M3 the surrogate test, M5 the depth scan.

    `vendor`, if given as (f1q, f2q, fread), adds the prediction that the vendor's
    own aggregate fidelities give through the same model -- the arithmetic behind a
    backend choice, recorded in the artifact rather than in a chat message."""
    e01, e10 = cal['e01'], cal['e10']
    circuits, by_key = {}, {}
    for rec, m in zip(results, meta):
        if m.get('kind') != 'tomo':
            continue      # the stage's own calibration circuits, handled separately
        key = '%s|%s|r%d' % (m['psi'], m['basis'], m['rounds'])
        out = read_tomo(rec, tuple(sigma), n, e01, e10)
        out['n_requested'] = shots_requested
        out['ideal'] = dict(zip(
            ('w', 'sigma'),
            (1.0, IDEAL_BLOCH[m['psi']][BASIS_AXIS[m['basis']]])))
        circuits[key] = out
        by_key[(m['psi'], m['basis'], m['rounds'])] = (rec, out, m)
    # every depth the submitted circuits actually reached, so the ladder stage
    # extends the curve instead of being analysed as if it were depth 1 and 2
    rounds_present = sorted({k[2] for k in by_key})
    assert 1 in rounds_present, (
        'the ladder stage is the SAME estimator at deeper rounds and must be '
        'analysed together with the depth-1/2 tomography, not on its own '
        '(rounds present = %s)' % (rounds_present,))

    # WHICH INPUTS ARE ACTUALLY MEASURABLE.  Under the confirmed hard real-time
    # budget the job can be cut off mid-run, so the dump can be a strict prefix of
    # the stage and whole input states can be missing.  A radius is the norm of a
    # THREE-component Bloch vector, so a state with fewer than three bases has no
    # radius at all -- reporting one would be a different quantity wearing the same
    # name.  Compute over the complete states and record the rest explicitly.
    states_ok = [p for p in TOMO_STATES
                 if all((p, B, 1) in by_key for B in BASES)]
    states_incomplete = [p for p in TOMO_STATES if p not in states_ok]
    if states_incomplete:
        print('[tomo] INCOMPLETE inputs (no radius reported for these): %s'
              % {p: [B for B in BASES if (p, B, 1) in by_key]
                 for p in states_incomplete}, flush=True)
    assert states_ok, ('not one input state has all three bases at rounds=1, so '
                       'there is no radius to report at all; circuits present = %s'
                       % (sorted(by_key),))

    per_input = {}
    for psi in states_ok:
        counts_by_basis = {B: by_key[(psi, B, 1)][0] for B in BASES}
        comps = {}
        for var in ('raw', 'mitigated'):
            comps[var] = [by_key[(psi, B, 1)][1][var]['sigma_cs'] for B in BASES]
        boot = {var: bootstrap_radius(counts_by_basis, tuple(sigma), n,
                                      e01 if var == 'mitigated' else None,
                                      e10 if var == 'mitigated' else None,
                                      n_boot=n_boot, seed=seed, variant=var)
                for var in ('raw', 'mitigated')}
        w = {var: float(np.mean([by_key[(psi, B, 1)][1][var]['w']
                                 for B in BASES])) for var in ('raw', 'mitigated')}
        rec = dict(bloch={v: comps[v] for v in comps}, w=w,
                   purity={v: (1 + comps[v][0] ** 2 + comps[v][1] ** 2
                               + comps[v][2] ** 2) / 2 for v in comps},
                   bootstrap=boot)
        for v in comps:
            rec['radius_' + v] = float(np.linalg.norm(comps[v]))
        per_input[psi] = rec

    # M2: the two pure ensembles of rho_in = I/2 must give the same LINEAR score.
    # This needs the Z circuit of ALL FOUR inputs, because the statement is that
    # two ensembles are equal; a one-sided version over a subset would still print
    # a small "diff" and read as a pass, which is worse than no number.
    m2_have = [p for p in TOMO_STATES if (p, 'Z', 1) in by_key]
    lin = {}
    if len(m2_have) == len(TOMO_STATES):
        terms, ses = {}, {}
        for psi in TOMO_STATES:
            zc = by_key[(psi, 'Z', 1)][1]
            terms[psi] = dict(raw=zc['raw']['sigma_uncond'],
                              mitigated=zc['mitigated']['sigma_uncond'])
            ses[psi] = dict(raw=zc['raw']['sigma_uncond_stderr'],
                            mitigated=zc['mitigated']['sigma_uncond_stderr'])
        for var in ('raw', 'mitigated'):
            Lz = 0.5 * (terms['0'][var] + terms['1'][var])
            Lx = 0.5 * (terms['+'][var] + terms['-'][var])
            se = 0.5 * math.sqrt(sum(ses[p][var] ** 2 for p in TOMO_STATES))
            vals = [terms[p][var] for p in TOMO_STATES]
            lin[var] = dict(terms={p: terms[p][var] for p in TOMO_STATES},
                            L_Z=Lz, L_X=Lx, diff=Lz - Lx, stderr=se,
                            ci95=[Lz - Lx - 1.96 * se, Lz - Lx + 1.96 * se],
                            term_span=max(vals) - min(vals),
                            ci_over_span=(3.92 * se) / max(max(vals) - min(vals),
                                                           1e-12),
                            non_vacuous=bool(max(vals) - min(vals) > 0.5))
        lin['available'] = True
    else:
        lin['available'] = False
        lin['inputs_present'] = m2_have
        lin['inputs_missing'] = [p for p in TOMO_STATES if p not in m2_have]
        lin['note'] = ('M2 compares the two pure ensembles of I/2 and needs the Z '
                       'circuit of all four inputs; %d of 4 returned.  Deliberately '
                       'NOT reported over the subset: a one-sided L_Z would print a '
                       'number that looks like the test passed.' % len(m2_have))

    # M1: the headline device mixedness, as the range over the inputs that
    #     completed.  The list of inputs is recorded next to the numbers, because
    #     "radius_min/max over 2 inputs" and "over 4 inputs" are not the same claim
    #     and a truncated run must not inherit the wording of a complete one.
    radii = {v: [per_input[p]['radius_' + v] for p in states_ok]
             for v in ('raw', 'mitigated')}
    ws = {v: [per_input[p]['w'][v] for p in states_ok] for v in ('raw', 'mitigated')}
    boot_mit = [per_input[p]['bootstrap']['mitigated'] for p in states_ok]
    mixedness = dict(
        inputs=list(states_ok), inputs_missing=list(states_incomplete),
        n_inputs=len(states_ok),
        radius_raw=radii['raw'], radius_mitigated=radii['mitigated'],
        radius_min=min(radii['mitigated']), radius_max=max(radii['mitigated']),
        radius_ci95_widest=[min(b['radius_ci95'][0] for b in boot_mit),
                            max(b['radius_ci95'][1] for b in boot_mit)],
        radius_stderr_max=max(b['radius_std'] for b in boot_mit),
        w_raw=ws['raw'], w_mitigated=ws['mitigated'],
        w_mean_mitigated=float(np.mean(ws['mitigated'])),
        leakage_mean_mitigated=1.0 - float(np.mean(ws['mitigated'])),
        purity_mitigated=[per_input[p]['purity']['mitigated']
                          for p in states_ok],
        purity_consistency_max=float(np.max(np.abs(
            [per_input[p]['purity']['mitigated']
             - (1 + per_input[p]['radius_mitigated'] ** 2) / 2
             for p in states_ok]))),
        r_below_one=bool(max(radii['mitigated']) < 1.0),
        ci_upper_below_one=bool(max(b['radius_ci95'][1] for b in boot_mit) < 1.0))

    # M3: fit ONE p_instr to the measured code-space weight, then let the model
    #     predict r with nothing left to choose.  Done at both depths: if a single
    #     rate explains both, the loss is per-gate; if not, time is contributing.
    #     At rounds=2 only the (psi=0, Z) and (psi=+, X) circuits are signal -- for
    #     those inputs the Bloch vector lies along the measured axis, so that single
    #     component IS the radius.
    model = {}
    for rounds in rounds_present:
        if ('0', 'Z', rounds) not in by_key:
            continue      # a depth without the (|0>, Z) signal circuit cannot be fitted
        gates = enc_dec_gates(rounds)
        w_meas = float(by_key[('0', 'Z', rounds)][1]['mitigated']['w'])
        if rounds == 1:
            if '0' not in per_input:
                # (|0>, Z) returned but not all three bases of |0>, so there is a
                # code-space weight to fit and no radius to compare it against.
                model['rounds1'] = dict(
                    w_measured=w_meas, r_measured=None, fit=None,
                    note='the |0>_L input is missing one or both of its X/Y circuits, '
                         'so its radius is not defined and the surrogate cannot be '
                         'tested at depth 1')
                continue
            r_meas = per_input['0']['radius_mitigated']
            se = per_input['0']['bootstrap']['mitigated']['radius_std']
        else:
            # beyond depth 1 only the DEPTH_SCAN grid is submitted, and for |0>_L the
            # Bloch vector lies along z, so that one component IS the radius and its
            # own standard error is the right one to scale the residual by
            r_meas = abs(float(by_key[('0', 'Z', rounds)][1]['mitigated']['sigma_cs']))
            se = float(by_key[('0', 'Z', rounds)][1]['raw']['sigma_cs_stderr'])
        try:
            p_fit = fit_p_instr(gates, w_meas, prep_vec('0'))
        except AssertionError as exc:
            model['rounds%d' % rounds] = dict(
                w_measured=w_meas, r_measured=r_meas, fit=None,
                note='the measured code-space weight is outside the range the '
                     'isotropic model can produce on [0, 0.25]: %s' % (exc,))
            continue
        pred = model_mixedness(gates, p_instr=p_fit, psi_vec=prep_vec('0'))
        model['rounds%d' % rounds] = dict(
            w_measured=w_meas, p_instr_fitted=p_fit,
            predicted=pred, r_measured=r_meas, r_predicted=pred['radius'],
            residual=pred['radius'] - r_meas,
            residual_in_sigma=((pred['radius'] - r_meas) / se) if se > 0 else None,
            # with se = 0 (a noiseless dry run) fall back to an absolute tolerance
            # rather than reporting a disagreement that is exactly zero
            agrees_within_2sigma=bool(
                (abs(pred['radius'] - r_meas) <= 2 * se) if se > 0
                else (abs(pred['radius'] - r_meas) < 1e-9)),
            n_instructions=len(gates))
    if model.get('rounds1', {}).get('p_instr_fitted') is not None and \
            model.get('rounds2', {}).get('p_instr_fitted') is not None:
        p1_, p2_ = model['rounds1']['p_instr_fitted'], model['rounds2']['p_instr_fitted']
        model['per_gate_vs_per_time'] = dict(
            p_instr_depth1=p1_, p_instr_depth2=p2_, ratio=p2_ / p1_ if p1_ else None,
            consistent_with_per_gate=bool(abs(p2_ - p1_) <= 0.2 * max(p1_, 1e-12)),
            note='a ratio near 1 means one rate per instruction explains both '
                 'depths (gate-like loss); a ratio well above 1 means the deeper '
                 'circuit loses more per instruction, i.e. time-like decoherence '
                 'is contributing')
    if vendor:
        model['vendor_prediction'] = vendor_prediction(*vendor)

    # M5: the depth scan itself, including the two null controls and how many
    #     standard errors each sits from zero.  Written over every depth actually
    #     submitted (the ladder stage adds rounds=4 and optionally 8), with the
    #     r1/r2 keys kept because the report, the figure and selftest (g) read them.
    scan = {}
    for psi, B in DEPTH_SCAN:
        depths = [r for r in rounds_present if (psi, B, r) in by_key]
        if 1 not in depths:
            # every ratio and every "did it move with depth" statement is anchored
            # on depth 1, so a deeper-only pair is recorded as unavailable rather
            # than printed as a curve with one end missing
            if depths:
                # same key shape as a complete entry, with None where a quantity is
                # undefined, so that every consumer (the report, the figure,
                # selftest (g)) can read the keys without special-casing truncation
                scan['%s|%s' % (psi, B)] = dict(
                    null_control=IDEAL_BLOCH[psi][BASIS_AXIS[B]] == 0.0,
                    depths=depths, by_rounds={
                        r: dict(sigma_cs_mitigated=by_key[(psi, B, r)][1]
                                ['mitigated']['sigma_cs'],
                                sigma_cs_raw=by_key[(psi, B, r)][1]['raw']['sigma_cs'],
                                stderr=by_key[(psi, B, r)][1]['raw']
                                ['sigma_cs_stderr'],
                                w=by_key[(psi, B, r)][1]['mitigated']['w'],
                                null_sigma=None) for r in depths},
                    sigma_cs_r1_mitigated=None, sigma_cs_r2_mitigated=None,
                    ratio_r2_over_r1=None, sigma_cs_r1_raw=None,
                    sigma_cs_r2_raw=None, stderr_r1=None, stderr_r2=None,
                    null_sigma_r1=None, null_sigma_r2=None, w_r1=None, w_r2=None,
                    note='depth 1 did not return for this pair, so nothing is '
                         'anchored and no ratio to depth 1 is defined')
            continue
        d1 = by_key[(psi, B, 1)][1]
        null = IDEAL_BLOCH[psi][BASIS_AXIS[B]] == 0.0
        se1 = d1['raw']['sigma_cs_stderr']
        per_depth = {}
        for r in depths:
            dd = by_key[(psi, B, r)][1]
            per_depth[r] = dict(
                sigma_cs_mitigated=dd['mitigated']['sigma_cs'],
                sigma_cs_raw=dd['raw']['sigma_cs'],
                stderr=dd['raw']['sigma_cs_stderr'], w=dd['mitigated']['w'],
                null_sigma=(dd['raw']['sigma_cs'] / dd['raw']['sigma_cs_stderr']
                            if (null and dd['raw']['sigma_cs_stderr'] > 0) else None))
        d2 = by_key[(psi, B, 2)][1] if (psi, B, 2) in by_key else None
        se2 = d2['raw']['sigma_cs_stderr'] if d2 else None
        scan['%s|%s' % (psi, B)] = dict(
            null_control=null, depths=depths, by_rounds=per_depth,
            sigma_cs_r1_mitigated=d1['mitigated']['sigma_cs'],
            sigma_cs_r2_mitigated=(d2['mitigated']['sigma_cs'] if d2 else None),
            ratio_r2_over_r1=((d2['mitigated']['sigma_cs'] / d1['mitigated']['sigma_cs'])
                              if (d2 and abs(d1['mitigated']['sigma_cs']) > 1e-12)
                              else None),
            sigma_cs_r1_raw=d1['raw']['sigma_cs'],
            sigma_cs_r2_raw=(d2['raw']['sigma_cs'] if d2 else None),
            stderr_r1=se1, stderr_r2=se2,
            null_sigma_r1=(d1['raw']['sigma_cs'] / se1) if (null and se1 > 0) else None,
            null_sigma_r2=((d2['raw']['sigma_cs'] / se2)
                           if (null and d2 and se2 > 0) else None),
            w_r1=d1['mitigated']['w'],
            w_r2=(d2['mitigated']['w'] if d2 else None))
    nulls = [v for v in scan.values() if v['null_control']]
    all_null_sigmas = [abs(p['null_sigma']) for v in nulls
                       for p in v['by_rounds'].values() if p['null_sigma'] is not None]
    scan_summary = dict(
        n_null_controls=len(nulls),
        depths=sorted({r for v in scan.values() for r in v['depths']}),
        max_null_sigma=max(all_null_sigmas) if all_null_sigmas else 0.0,
        nulls_within_3sigma=bool(all(s < 3.0 for s in all_null_sigmas)),
        note='the null controls measure a Bloch component that is exactly zero '
             'ideally; a value beyond a few standard errors is a phase or '
             'calibration problem, not evidence of mixedness')
    out = dict(circuits=circuits, per_input=per_input, mixedness=mixedness,
               linear_ensemble=lin, model=model, depth_scan=scan,
               depth_scan_summary=scan_summary,
               layout_sigma=list(sigma), assignment=cal,
               shots_requested=shots_requested,
               # what the truncated run could and could not support, at the top
               # level so a reader of the artifact sees it before any number
               inputs_complete=list(states_ok),
               inputs_incomplete=list(states_incomplete),
               n_tomo_circuits=len(by_key),
               depths_analysed=list(rounds_present))
    return out, _tomo_report(out)


def _tomo_report(out):
    """The printable summary; returned separately so --mode analyze and the
    offline audit both print exactly what the artifact claims."""
    lines = []
    per_input, lin, model = out['per_input'], out['linear_ensemble'], out['model']
    done = out.get('inputs_complete') or list(TOMO_STATES)
    missing = out.get('inputs_incomplete') or []
    lines.append('[tomo] M1 device mixedness (readout-mitigated), over %d of %d '
                 'inputs%s:' % (len(done), len(TOMO_STATES),
                                '' if not missing else ' (MISSING: %s)'
                                % ','.join('|%s>_L' % m for m in missing)))
    for p_ in done:
        b = per_input[p_]['bootstrap']['mitigated']
        lines.append('   |%s>_L  r = %.4f [%.4f, %.4f]  purity = %.4f  w = %.4f'
                     % (p_, per_input[p_]['radius_mitigated'], b['radius_ci95'][0],
                         b['radius_ci95'][1], per_input[p_]['purity']['mitigated'],
                         per_input[p_]['w']['mitigated']))
    if not lin.get('available', True):
        lines.append('[tomo] M2 UNAVAILABLE: %s' % lin['note'])
    else:
        for var in ('raw', 'mitigated'):
            L = lin[var]
            lines.append('[tomo] M2 (%s): L_Z = %+.5f  L_X = %+.5f  diff = %+.5f '
                         '+-%.5f  CI95 [%+.4f, %+.4f]  term span %.3f  '
                         'non-vacuous=%s'
                         % (var, L['L_Z'], L['L_X'], L['diff'], L['stderr'],
                            L['ci95'][0], L['ci95'][1], L['term_span'],
                            L['non_vacuous']))
    for k in ('rounds1', 'rounds2'):
        M = model.get(k, {})
        if M.get('p_instr_fitted') is None:
            lines.append('[tomo] M3 %s: %s' % (k, M.get('note', 'missing')))
            continue
        sig = M['residual_in_sigma']
        lines.append('[tomo] M3 %s: w_meas %.4f -> p_instr %.5f -> r_pred %.4f vs '
                     'r_meas %.4f (%s sigma, agrees=%s)'
                     % (k, M['w_measured'], M['p_instr_fitted'], M['r_predicted'],
                        M['r_measured'],
                        ('%.1f' % sig) if sig is not None and sig == sig else 'n/a',
                        M['agrees_within_2sigma']))
    if 'per_gate_vs_per_time' in model:
        t = model['per_gate_vs_per_time']
        rat = t['ratio']
        lines.append('[tomo] M5 per-gate vs per-time: p_instr(depth2)/p_instr(depth1)'
                     ' = %s -> per-gate consistent = %s'
                     % (('%0.3f' % rat) if rat is not None
                        else 'undefined (both depths fitted to zero noise)',
                        t['consistent_with_per_gate']))
    s = out['depth_scan_summary']
    mns = s['max_null_sigma']
    # which depths actually returned, because "the radius does not move with depth"
    # is a claim about a curve and a truncated run may only have one point on it
    lines.append('[tomo] M5 null controls: max %s sigma, all within 3 sigma = %s '
                 '(depths measured: %s, %d null pair(s))'
                 % (('%0.1f' % mns) if mns == mns else 'n/a',
                    s['nulls_within_3sigma'], s['depths'], s['n_null_controls']))
    if len(s['depths']) < 2:
        lines.append('[tomo] M5 DEPTH CURVE UNAVAILABLE: only depth %s returned, so '
                     'there is no trend to report -- the depth-2 scan is the last '
                     'chunk of the stage and is what a budget cut-off removes first'
                     % (s['depths'],))
    return lines


def vendor_prediction(f1q, f2q, fread, rounds=1):
    """What a vendor's aggregate fidelities predict for THIS circuit, through the
    same model -- the arithmetic behind a backend choice, recorded in the artifact
    rather than left in a chat message.

    Single-qubit error on 1Q instructions, two-qubit error on CNOTs, then the readout
    error as a contrast factor on the logical bit and as a survival factor on the
    four code-space bits the post-selection requires."""
    gates = enc_dec_gates(rounds)
    d = model_mixedness(gates, p1=1.0 - f1q, p2=1.0 - f2q, psi_vec=prep_vec('0'))
    n_cnot = sum(1 for g in gates if g[0] == 'CNOT')
    contrast = 1.0 - 2.0 * (1.0 - fread)
    return dict(f1q=f1q, f2q=f2q, fread=fread, rounds=rounds,
                n_instructions=len(gates), n_cnot=n_cnot,
                two_qubit_survival=f2q ** n_cnot,
                codespace_weight=d['codespace_weight'], radius=d['radius'],
                radius_as_measured=d['radius'] * contrast,
                logical_contrast=contrast,
                codespace_bits_all_read_correct=fread ** 4,
                combined_signal=d['radius'] * contrast * fread ** 4)


def analyze_bench(results, meta, phi, shots_requested=None, verbose=True):
    """The 9-qubit stage: the mid/late A-B, then the full 16-branch sum at p=0.

    Reuses qcloud_vscr_new's own layout detection and branch statistics so the new
    full-branch point and the existing 4-circuit feasibility point are computed by
    literally the same code -- a difference between them is then a difference in the
    device or the data, not in the analysis."""
    ab = [(r, m) for r, m in zip(results, meta) if m.get('kind') == 'ab']
    bm = [(r, m) for r, m in zip(results, meta) if m.get('kind') == 'bench']
    ab_out = {}
    for r, m in ab:
        n_tot = int(sum(r.values()))
        tup = [(m['psi'], m['p'], m['frame'], m['s'])]
        anc, rev = q.detect_layout(tup, [r])
        n_sel, n_hit = q.counts_to_branch_stats(r, m['s'], anc, rev)
        lo, hi = wilson(n_hit, n_sel)
        ab_out[m['measure']] = dict(
            n_total=n_tot, n_selected=n_sel, n_hit=n_hit,
            selected_fraction=n_sel / n_tot if n_tot else None,
            F_s=(n_hit / n_sel) if n_sel else None, F_s_wilson95=[lo, hi],
            anc_slice=[anc.start, anc.stop], rev=rev,
            top3=sorted(r.items(), key=lambda kv: -kv[1])[:3])
    if len(ab_out) == 2:
        ab_out['verdict'] = dict(
            mid_selected_fraction=ab_out['mid']['selected_fraction'],
            late_selected_fraction=ab_out['late']['selected_fraction'],
            mid_measure_works=bool(
                (ab_out['mid']['selected_fraction'] or 0)
                > 0.5 * (ab_out['late']['selected_fraction'] or 0)),
            note='mid-circuit measurement returned 0 selected shots out of 735 on '
                 'WK_C180; this is the same circuit re-run on this chip, and the '
                 'verdict decides which variant a future full sweep should use')
    tuples = [(m['psi'], m['p'], m['frame'], m['s']) for r, m in bm]
    counts = [r for r, m in bm]
    anc, rev = q.detect_layout(tuples, counts)
    per_branch, per_state = {}, {}
    for (psi, p, fr, s), r in zip(tuples, counts):
        n_tot = int(sum(r.values()))
        n_sel, n_hit = q.counts_to_branch_stats(r, s, anc, rev)
        lo, hi = wilson(n_hit, n_sel)
        per_branch['%s|s%d' % (psi, s)] = dict(
            psi=psi, s=s, n_total=n_tot, n_selected=n_sel, n_hit=n_hit,
            P_s=n_sel / n_tot if n_tot else None,
            F_s=(n_hit / n_sel) if n_sel else None, F_s_wilson95=[lo, hi])
        d = per_state.setdefault(psi, dict(hits=0, selected=0, total=0))
        d['hits'] += n_hit
        d['selected'] += n_sel
        d['total'] += n_tot
    bench = {}
    for psi, d in per_state.items():
        N = d['total'] / 16.0
        F = d['hits'] / N if N > 0 else None
        lo, hi = wilson(d['hits'], int(round(N)))
        F_id = q.simulator_fidelity_frame(psi, BENCH_FRAME, phi)
        bench[psi] = dict(shots_total=d['total'], n_selected=d['selected'],
                          n_hit=d['hits'], effective_N=N, F_hw=F,
                          F_hw_wilson95=[lo, hi], F_ideal_same_frame=F_id,
                          gap=F - F_id if F is not None else None,
                          selected_fraction=d['selected'] / d['total']
                          if d['total'] else None)
    out = dict(ab_test=ab_out, per_branch=per_branch, per_state=bench,
               anc_slice=[anc.start, anc.stop], rev=rev,
               shots_requested=shots_requested)
    if verbose:
        for k, v in ab_out.items():
            if isinstance(v, dict) and 'selected_fraction' in v:
                print('[bench] A/B %s: selected %d/%d (%.3f), F_s=%s %s'
                      % (k, v['n_selected'], v['n_total'], v['selected_fraction'],
                         ('%.3f' % v['F_s']) if v['F_s'] is not None else 'n/a',
                         ['%.3f' % x for x in v['F_s_wilson95']]), flush=True)
        for psi, v in bench.items():
            print('[bench] psi=%s p=0: F_hw = %.4f [%.4f, %.4f] vs ideal %.6f '
                  '(gap %+.4f) from %d hits / %d shots'
                  % (psi, v['F_hw'], v['F_hw_wilson95'][0], v['F_hw_wilson95'][1],
                     v['F_ideal_same_frame'], v['gap'], v['n_hit'],
                     v['shots_total']), flush=True)
    return out


def ideal_data_zero_prob(psi, frame, s, phi):
    """Exact, UNCONDITIONED P(the five data qubits read 00000) for one branch
    circuit, from the full 512-dim output state.

    Deliberately not the post-selected F_s: the primary hardware estimator is
    unconditioned (see WARMCOLD_PRIMARY), so its ideal reference has to be
    unconditioned too or the comparison is not like for like.  The two coincide
    at p=0 because the syndrome is deterministic there -- asserted, not assumed,
    in selftest (l).  In `apply_gates`' index convention qubit 0 is the MOST
    significant bit, so data q0..q4 are idx>>4 and ancillas a0..a3 are idx&15."""
    gates = q.build_branch_circuit(q.ENC5, psi, frame, s, phi, late_measure=True)
    pv = np.zeros(q.DIM9, dtype=complex)
    pv[0] = 1.0
    for kind, gs in q.split_unitary_sections(gates):
        if kind == 'U':
            pv = q.apply_gates(pv, gs)
    # reshape(32, 16) indexes [data, ancilla]: row 0 is exactly the five data
    # qubits all reading zero, summed over every ancilla outcome, which is what
    # "unconditioned" means here
    return float((np.abs(pv) ** 2).reshape(32, 16)[0].sum())


def layout_from_meas(gates, nq=None):
    """(anc_slice, rev) for HARDWARE keys, from the circuit's MEAS map plus the
    platform's key bit-order.

    Two facts are combined, and only the first is derivable from the circuit:

      * WHICH classical bits carry the ancilla.  The stabiliser measurements are
        emitted first, so the ancillas are exactly the qubits measured into
        classical bits 0..NQ_ANC-1.  That is a property of the builder and is
        asserted below, not assumed.

      * WHERE a classical bit lands in the returned key.  This is a PLATFORM
        property that no circuit can tell you; see q.sim_key_to_vendor_key for
        the measurement that established it.  The vendor writes each register in
        DESCENDING classical-bit order at the key positions of its own bits, so
        the raw key is [c3 c2 c1 c0][c8 c7 c6 c5 c4] = [a3a2a1a0][d4d3d2d1d0].
        Reversing it (rev=True) puts the ancilla at positions NQ-NQ_ANC..NQ-1,
        i.e. [5:9], in ASCENDING classical-bit order -- the order
        counts_to_branch_stats compares against format(s, '04b') -- and the data
        register at [0:5].

    So the answer for the nine-qubit block is (slice(5, 9), True): the hypothesis
    ED Table 4 selected on data-hit evidence (44 hits against 0) and the one that
    reproduces the reported 44/55 post-selected fidelity, the 0-of-369 mid-measure
    syndrome deficit and the 2.4% identity-branch mode.

    HOW MUCH OF THIS IS MEASURED.  The s=12 branch pins the ANCILLA group's
    internal order directly (its ideal outcome has c0=c1=1, c2=c3=0 and the
    observed mode puts those ones at key positions 3 and 2).  The DATA group's
    internal order is an extrapolation of the same rule: every informative
    circuit has an all-zero data register, so no key distinguishes d0 from d4.
    That is harmless for every estimator actually reported -- the data test is
    `data == '00000'`, which is invariant under permuting the data positions, and
    WHICH positions are data follows from the ancilla occupying [0:4].  It is not
    harmless for naming an individual data qubit, so per-data-qubit labels carry
    that caveat wherever they are printed.

    WHY THIS FUNCTION WAS WRONG ONCE, AND WHY A SELF-TEST DID NOT CATCH IT.  An
    earlier version returned (slice(0, 4), True) and a noiseless cross-check
    "confirmed" it as the unique valid window.  Both statements were true of the
    LOCAL SIMULATOR and false of the hardware: CPUQVM returns the globally
    reversed key order, whose correct window is the exact complement of the
    vendor's.  Because verify_layout_noiseless sampled through cpuqvm_sample, the
    cross-check validated the simulator's convention against itself -- a
    circularity of the same species as detect_layout's, one level up, and
    invisible for the same reason.  cpuqvm_sample now normalises at the boundary,
    so the cross-check compares like with like and has independent content.
    """
    nq = nq or q.NQ
    meas = [(t[1], t[2]) for t in gates if t[0] == 'MEAS']
    assert len(meas) == nq, ('expected one MEAS per qubit', len(meas), nq)
    assert len(set(meas)) == nq and len(set(c for _, c in meas)) == nq, meas
    # ancilla = measured into the first NQ_ANC classical bits, by construction.
    # This is only meaningful for a circuit that HAS an ancilla/data split; cal9
    # uses MEAS(i,i) and must not be passed here.
    anc_cbits = set(cb for _qb, cb in meas if cb < q.NQ_ANC)
    assert len(anc_cbits) == q.NQ_ANC, anc_cbits
    anc_qb = sorted(qb for qb, cb in meas if cb < q.NQ_ANC)
    assert anc_qb == list(range(q.NQ_DATA, nq)), (
        'ancilla cbits 0..NQ_ANC-1 must be read off qubits NQ_DATA..NQ-1; got %r. '
        'Passing an identity-MEAS circuit (e.g. cal9) here would silently label '
        'the DATA qubits as ancillas.' % (anc_qb,))
    data_cbits = sorted(cb for _qb, cb in meas if cb >= q.NQ_ANC)
    assert data_cbits == list(range(q.NQ_ANC, nq)), data_cbits
    # vendor key order + reversal -> ancilla occupies the TOP NQ_ANC positions
    return slice(nq - q.NQ_ANC, nq), True


def cbit_of_position(pos, rev, nq=None):
    """The LOGICAL QUBIT an analysed-key position refers to, given `rev`.

    Named for the classical bit because that is what it returns for the
    identity-MEAS calibration circuits, where qubit index and classical bit
    coincide.  What it actually computes is the logical qubit, and the two are
    the same number only for those circuits -- for a science circuit, whose MEAS
    map is q5->c0..q8->c3 and q0->c4..q4->c8, the classical bit of qubit j is
    (j + NQ_ANC) mod NQ.  Callers that need the classical bit must apply that
    map themselves; callers that need to NAME a channel (which is what the cal9
    table and the healthy-subset diagnostic do) want the qubit, which is this.

    The rule behind it is the platform's key order, established in
    q.sim_key_to_vendor_key: raw key position p carries logical qubit NQ-1-p.
    Reversing the key therefore puts logical qubit j at analysed position j --
    a direct correspondence, which is why this is an involution in `pos` for
    fixed `rev` and why selftest (m) asserts that.

    Getting this backwards is not cosmetic.  It is what labelled the 2026-10-12
    broken channel 'ancilla q8' instead of a data qubit, and the two readings
    imply opposite verdicts: a broken ancilla costs only the post-selected F_s,
    a broken data qubit makes the primary estimator uninterpretable."""
    nq = nq or q.NQ
    return pos if rev else (nq - 1 - pos)


def cbit_of_qubit(qb, gates=None, nq=None):
    """The classical bit a LOGICAL QUBIT is measured into, from a MEAS map.

    Defaults to the science circuit's map (q5->c0..q8->c3, q0->c4..q4->c8), i.e.
    cbit = (qubit + NQ_ANC) mod NQ.  Pass `gates` for any other circuit -- cal9
    uses MEAS(i,i), where the classical bit is the qubit index -- because
    applying the science map to cal9's keys is exactly the mistake that produced
    the 'ancilla q8' mislabel."""
    nq = nq or q.NQ
    if gates is None:
        return (qb + q.NQ_ANC) % nq
    for t in gates:
        if t[0] == 'MEAS' and t[1] == qb:
            return t[2]
    raise KeyError('qubit %r is not measured in the given gates' % (qb,))


def verify_layout_noiseless(gates_list, metas, shots=1000, anc=None, rev=None):
    """Cross-check a layout against the NOISELESS simulator.

    This is the independent evidence that settles `layout_from_meas`: with zero
    noise the p=0 syndrome is deterministic, so a correct layout accepts every
    shot (P_s = 1.000) and reproduces the ideal P(data=00000), while a wrong
    window accepts none (P_s = 0.000).  The separation is total, not marginal --
    unlike detect_layout's 3365-vs-3267 outcome-scored coin flip.

    Each meta must carry 'psi', 'frame', 's' and 'phi'; 'phi' is required rather
    than defaulted because a silently-zero angle set would compute a wrong ideal
    reference and make the cross-check pass for the wrong reason."""
    if anc is None or rev is None:
        anc, rev = layout_from_meas(gates_list[0])
    ds = _data_slice_for(anc)
    per, worst_ps, worst_sigma = {}, 1.0, 0.0
    for g, m in zip(gates_list, metas):
        # cpuqvm_sample returns a counts dict (bitstring -> count), not a key list
        counts = q.cpuqvm_sample(g, shots)
        n_shot = int(sum(counts.values()))
        n_sel, _n_hit = q.counts_to_branch_stats(counts, m['s'], anc, rev)
        ps = n_sel / n_shot if n_shot else 0.0
        n_dz, n_tot = data_zero_counts(counts, anc, rev)
        pz = n_dz / n_tot if n_tot else 0.0
        ideal = ideal_data_zero_prob(m['psi'], m['frame'], m['s'], m['phi'])
        # binomial sigma of the SAMPLED probability; an ideal value of exactly 0 or
        # 1 would give sigma 0, so floor it at the smallest resolvable step
        sg = math.sqrt(max(pz * (1.0 - pz), 1.0 / max(n_tot, 1)) / max(n_tot, 1))
        dev_sigma = abs(pz - ideal) / sg if sg > 0 else (
            0.0 if pz == ideal else float('inf'))
        worst_ps = min(worst_ps, ps)
        worst_sigma = max(worst_sigma, dev_sigma)
        per['%s|%s|s%d' % (m.get('role', m['psi']), m.get('arm', 'x'), m['s'])] = \
            dict(P_s=ps, P_data_zero=pz, ideal=ideal, n_total=n_tot,
                 dev_in_sigma=dev_sigma)
    # acceptance is the SHARP discriminator and needs no tolerance: at p=0 with no
    # noise the syndrome is deterministic, so the correct window accepts every
    # single shot and any other window accepts a different fraction.  The ideal
    # probability is a secondary check and does need one, because it is estimated
    # from a finite sample.
    return dict(anc_slice=[anc.start, anc.stop], rev=rev,
                data_slice=[ds.start, ds.stop], per_circuit=per,
                min_acceptance_rate=worst_ps, max_dev_in_sigma=worst_sigma,
                shots=shots,
                layout_verified=bool(worst_ps >= 1.0 and worst_sigma < 5.0))


def _data_slice_for(anc_slice):
    """The data-bit window, given the ancilla window `detect_layout` chose.
    Mirrors q.counts_to_branch_stats so the two never disagree."""
    return slice(4, 9) if anc_slice.start == 0 else slice(0, 5)


def data_zero_counts(counts, anc_slice, rev=False):
    """(n_data_zero, n_total): the primary estimator's numerator and denominator.

    The window MUST be taken on the same possibly-reversed key that
    q.counts_to_branch_stats uses, and this was a real bug the noiseless dry run
    caught rather than a style point.  '00000' is a palindrome, so reversing the
    CONTENT of the window cannot change the test -- but reversing the KEY moves
    WHICH positions the window covers: for a nine-bit key, key[::-1][4:9] is
    key[0:5], not key[4:9].  Slicing the unreversed key therefore reads the
    ancilla register as if it were the data register and returns 0 hits on a
    perfect device, while the post-selected F_s computed by the vendor-path helper
    stays correct -- which is exactly the asymmetry that showed up in the dry run
    and is why both estimators are carried."""
    ds = _data_slice_for(anc_slice)
    n_tot = int(sum(counts.values()))
    n_hit = 0
    for k, v in counts.items():
        kk = k[::-1] if rev else k
        if kk[ds] == '00000':
            n_hit += v
    return n_hit, n_tot


def warmcold_power(w=WARMCOLD_W_MEASURED_2026_09, shots=None,
                   gap_ideal=WARMCOLD_GAP_IDEAL_SIGNAL):
    """What a warm/cold gap of `gap_ideal` is worth on a device that transmits
    only a fraction `w` of the data register.

    A device that randomises the data register cannot transmit WHICH recovery
    ran, so the observable gap is w * gap_ideal and the noise floor contributes
    (1-w)/32 to both arms.  This is the stage's whole power estimate in one
    function, so --mode plan can print it and selftest (l) can assert it instead
    of either restating it in prose that could drift."""
    shots = shots or STAGE_SHOTS['warmcold']
    gap = w * gap_ideal
    p_warm = w + (1 - w) / 32.0
    p_cold = 0.5 * w + (1 - w) / 32.0
    pbar = 0.5 * (p_warm + p_cold)
    sigma = math.sqrt(2.0 * pbar * (1.0 - pbar) / shots)
    return dict(w=w, shots=shots, observable_gap=gap, sigma_pair=sigma,
                sigma_z=gap / sigma if sigma else None,
                p_warm=p_warm, p_cold=p_cold)


# A readout channel is BROKEN, not merely noisy, when its contrast collapses.
# contrast = 1 - (e01 + e10) is 1.0 for a perfect channel and 0.5 at
# e01=e10=0.25; healthy qubits on this platform sit at 0.88-1.00.  The 2026-10-12
# run had one channel at -0.156 -- it reported '1' MORE often from |0> than from
# |1>, i.e. the measurement was anti-correlated with the state prepared.  No
# threshold between 0.5 and 0.88 would miss that, so 0.5 is a deliberately loose
# and unambiguous line rather than a tuned one.
CAL9_BROKEN_CONTRAST = 0.5


def _analyze_cal9(cal, anc, rev=False, meas_map=None):
    """Per-qubit readout error of the block, from the job's own two 9-qubit
    calibration circuits.

    It IS a prerequisite for the primary estimator, and the earlier version of
    this docstring said otherwise.  The claim was that the warm-minus-cold
    difference of P(data=00000) is common-mode in readout error and therefore
    needs no calibration.  That holds only while every data channel is HEALTHY:
    a confusion matrix with e01 ~ e10 acts on the difference as an approximately
    uniform attenuation, common to both arms.  It does not hold when a data
    channel is broken, because then the attenuation factor for that channel is
    1-e01-e10 = contrast, which is NEGATIVE -- the broken channel contributes
    with its sign flipped, so a warm-minus-cold difference can be produced by
    readout alone with no physics behind it.  On the 2026-10-12 run the broken
    channel was logical q3 with e01=0.5679, e10=0.5885, contrast=-0.1564, i.e.
    exactly that case.

    Reports ALL NINE channels, not just the five data ones.  The 2026-10-12 run
    had a broken DATA channel (logical q3, contrast -0.156, read by the science
    circuit at analysed key position 7, inside the primary estimator's data
    window [4,9]) and the data-only table was silent about it, so the artifact
    contained no hint that a channel was dead; the breakage surfaced only as a
    degraded F_s reported as though it meant something.  Every channel now
    carries e01, e10, fidelity and contrast, and any contrast below
    CAL9_BROKEN_CONTRAST is flagged BROKEN with its role, so `screen` states
    explicitly whether the PRIMARY estimator (data register) and the SECONDARY
    one (post-selected F_s, which needs healthy ancillas) are still
    interpretable.

    A broken ANCILLA would have been survivable -- it degrades F_s but leaves the
    data window intact -- and that is what this was first read as, before the
    per-channel table was labelled with logical qubit indices.  See
    check_cal9_transferability for the second, worse problem: on that run cal9
    did not even describe the science circuits."""
    if len(cal) < 2:
        return dict(available=False,
                    note='both cal9 circuits must return; got %d (a truncated '
                         'dump loses the readout decomposition but NOT the A/B)'
                         % len(cal))
    d = {m['cal']: r for r, m in cal}
    ds = _data_slice_for(anc)
    # the SAME reversal the branch statistics apply, so the window covers the data
    # register rather than the ancillas -- see data_zero_counts for why this is not
    # a no-op even though both reference patterns are palindromes
    dz = {(k[::-1] if rev else k): v for k, v in d['zero'].items()}
    do = {(k[::-1] if rev else k): v for k, v in d['one'].items()}
    n0 = int(sum(dz.values()))
    n1 = int(sum(do.values()))
    # logical qubit index per analysed key position, from the calibration
    # circuit's own MEAS map, so a broken channel is reported as 'logical q8'
    # rather than as an anonymous raw key position
    if meas_map is None:
        meas_map = dict((cb, qb) for qb, cb in
                        ((t[1], t[2]) for t in build_cal9('zero')
                         if t[0] == 'MEAS'))
    # cal9 and the science circuits do NOT share a MEAS map: build_cal9 ends with
    # MEAS(i,i) (the identity) whereas build_branch_circuit measures the stabiliser
    # ancillas first (q5->c0 .. q8->c3, then q0->c4 .. q4->c8).  A cal9 key
    # position therefore identifies a LOGICAL qubit through cal9's OWN map, and
    # that logical qubit's ROLE is fixed by the construction (q0..NQ_DATA-1 are the
    # data register).  Labelling cal9 positions with the science circuit's window
    # semantics -- which this function originally did -- reported the 2026-10-12
    # broken channel as an ancilla when it is logical q3, a DATA qubit, and so
    # wrongly declared the primary estimator valid.
    def _logical(pos):
        return meas_map.get(cbit_of_position(pos, rev))

    def _role(pos):
        lg = _logical(pos)
        if lg is None:
            return 'unk'
        return 'data' if lg < q.NQ_DATA else 'anc'

    # positions in cal9's analysed key that carry each register.  With the
    # identity cal9 map and rev=True these are 0..4 (data) and 5..8 (ancilla).
    # They are NOT a mirror of the science circuit's windows: under the vendor
    # key order analysed position j is logical qubit j for BOTH circuits, so the
    # registers occupy 0..4 / 5..8 in cal9's key and the science circuit simply
    # reads the same logical qubits at the same analysed positions, grouped as
    # data=[0:5] and ancilla=[5:9].  An earlier comment described them as
    # mirrored, which was an artefact of a wrong key-order convention.
    data_pos = [p for p in range(q.NQ) if _role(p) == 'data']
    anc_pos = [p for p in range(q.NQ) if _role(p) == 'anc']
    per_q, broken = {}, []
    for pos in range(q.NQ):
        lg = _logical(pos)
        role = _role(pos)
        idx = (lg if role == 'data'
               else (lg - q.NQ_DATA if lg is not None else pos))
        e01 = int(sum(v for k, v in dz.items() if k[pos] == '1'))
        e10 = int(sum(v for k, v in do.items() if k[pos] == '0'))
        f01 = e01 / n0 if n0 else None
        f10 = e10 / n1 if n1 else None
        both = f01 is not None and f10 is not None
        fid = (1.0 - 0.5 * (f01 + f10)) if both else None
        con = (1.0 - (f01 + f10)) if both else None
        is_broken = bool(con is not None and con < CAL9_BROKEN_CONTRAST)
        lab = '%s%d' % (role, idx)
        per_q[lab] = dict(
            role=role, index=idx, key_pos=pos, logical_qubit=lg,
            # The science circuit's analysed key position for this logical qubit.
            # Under the vendor key order (raw position p <-> qubit NQ-1-p) the
            # REVERSED key puts logical qubit j at analysed position j, so this is
            # just `lg` when rev -- not the classical bit (lg+NQ_ANC) mod NQ that
            # an earlier version used.  Conflating the two put logical q3 at
            # position 7 instead of 3, which is outside the data window [0:5] and
            # made the "broken channel is inside the primary estimator" statement
            # self-contradictory.
            science_key_pos=(lg if rev else q.NQ - 1 - lg)
            if lg is not None else None,
            n_zero=n0, n_one=n1, n_e01=e01, n_e10=e10,
            e01=f01, e10=f10, readout_fidelity=fid, contrast=con,
            broken=is_broken)
        if is_broken:
            broken.append(dict(label=lab, role=role, key_pos=pos,
                               logical_qubit=lg,
                               science_key_pos=per_q[lab]['science_key_pos'],
                               contrast=con, e01=f01, e10=f10,
                               readout_fidelity=fid))
    rf = [v['readout_fidelity'] for v in per_q.values()
          if v['readout_fidelity'] is not None]
    healthy = [v['readout_fidelity'] for v in per_q.values()
               if v['readout_fidelity'] is not None and not v['broken']]

    def _frac(dd, n, positions, bit='0'):
        """Fraction of shots whose given key positions all read `bit`."""
        if not n or not positions:
            return None
        return int(sum(v for k, v in dd.items()
                       if all(k[p] == bit for p in positions))) / n

    n_bd = sum(1 for b in broken if b['role'] == 'data')
    n_ba = sum(1 for b in broken if b['role'] == 'anc')
    return dict(available=True, n_zero=n0, n_one=n1, rev=rev,
                # the SCIENCE circuit's windows, in its own analysed-key space
                data_slice=[ds.start, ds.stop],
                anc_slice=[anc.start, anc.stop],
                # and cal9's, which are the mirror image because cal9 uses the
                # identity MEAS map while the science circuit measures ancillas
                # first -- conflating these two was the labelling bug
                cal9_data_positions=data_pos,
                cal9_anc_positions=anc_pos,
                per_qubit=per_q,
                per_data_qubit=dict((k, v) for k, v in per_q.items()
                                    if v['role'] == 'data'),
                broken_contrast_threshold=CAL9_BROKEN_CONTRAST,
                broken_qubits=broken, n_broken=len(broken),
                mean_readout_fidelity=(sum(rf) / len(rf)) if rf else None,
                mean_readout_fidelity_healthy=(
                    sum(healthy) / len(healthy)) if healthy else None,
                # separate all-zero fractions for the two registers, over the
                # positions that actually carry them.  Mixing registers was
                # actively misleading on 2026-10-12: the joint 9-bit fraction
                # (0.337) is dragged down by one broken channel, and the window
                # that looked like 'the data register' was in fact logical q4..q8.
                all_zero_key_fraction_data=_frac(dz, n0, data_pos),
                all_zero_key_fraction_ancilla=_frac(dz, n0, anc_pos),
                all_one_key_fraction_data=_frac(do, n1, data_pos, '1'),
                all_one_key_fraction_ancilla=_frac(do, n1, anc_pos, '1'),
                # the joint 9-bit fraction: a 9-qubit |0...0> product state should
                # return exactly one key, so this is the block's joint readout
                # fidelity.  Kept for continuity with the 2026-09 artifact, but it
                # is NOT the data register's survival -- see the two above.
                all_zero_key_fraction_zero_circuit=(
                    dz.get('0' * q.NQ, 0) / n0) if n0 else None,
                all_one_key_fraction_one_circuit=(
                    do.get('1' * q.NQ, 0) / n1) if n1 else None,
                screen=dict(
                    n_broken_data=n_bd, n_broken_ancilla=n_ba,
                    # a broken DATA channel invalidates the primary estimator: it
                    # enters P(data=00000) directly, and an asymmetric breakage is
                    # NOT common-mode between the two arms
                    primary_estimator_valid=bool(n_bd == 0),
                    # a broken ANCILLA invalidates the post-selected F_s but
                    # leaves the unconditioned primary estimator intact -- exactly
                    # the 2026-10-12 situation, and exactly why its FAILED verdict
                    # could still be trusted as a physical result
                    secondary_estimator_valid=bool(n_ba == 0),
                    block_usable=bool(n_bd == 0),
                    note='all %d channels screened; %d broken (%d data, %d '
                         'ancilla)' % (q.NQ, len(broken), n_bd, n_ba)))


def check_cal9_transferability(counts_by_name, cal9_out, rev=True,
                               tol_sigma=5.0, phi=None):
    """Can cal9's readout characterisation be transferred to the science circuits?

    A readout channel with errors (e01, e10) can only ever return '0' with
    probability in [min(1-e01, e10), max(1-e01, e10)] -- the endpoints being a
    qubit prepared deterministically in |1> or |0>.  That range is a HARD bound
    independent of what circuit ran before the measurement.  So if a science
    circuit's observed marginal at some channel falls outside the range cal9
    measured for that channel, the two circuits are NOT reading the same physical
    channel, and the only mechanism that can do that is the vendor's transpiler
    re-routing logical qubits differently per circuit.

    This matters because it detects vendor re-routing WITHOUT vendor cooperation,
    using data the job already returned.

    THE 2026-10-12 "IT FIRES" RESULT IS RETRACTED, and this docstring used to
    assert it in the present tense.  It read: cal9 measured contrast -0.156
    (e01=0.568, e10=0.589) at logical q3, bounding P(read 0) to [0.432, 0.589],
    yet the science circuits read 0.708 and 0.665 there -- above the bound -- so
    cal9 characterised a DIFFERENT physical channel and could not validate their
    estimator.  Those comparisons were made at the WRONG key position: the lookup
    used the classical bit the MEAS map assigns rather than the analysed position
    (see the comment below), so it compared cal9's channel against a different
    science channel and reported the mismatch as transpiler re-routing.  With the
    position fixed, all 36 comparisons on that same run lie inside their bounds:
    hw_mixedness.json records warmcold.cal9_transferability as n_comparisons=36,
    n_violations=0, cal9_transferable=true, and warmcold.verdict.cal9_transferable
    true.  selftest (m3) locks both directions -- a marginal of 0.80 against that
    channel's bound is flagged, 0.50 and 0.97 are not.

    cal9 therefore DOES characterise the science block, which is what makes a paid
    cal9 screen a valid gate for it (stage_circuits' CAL-FIRST comment relies on
    this), and what made the 2026-10-19 screen of WK_C180_2's block a real
    clearance of the block the A/B would use rather than a statement about some
    other qubits.  The dead channel itself was never in doubt -- it is measured,
    not inferred -- only the claim that cal9 could not see the science circuits.
    Leaving the retracted version in place was not harmless: it contradicted the
    artifact in the same repository, and read as a reason to distrust a screen
    that had just cost 2 hours of queue."""
    if not cal9_out.get('available') or not counts_by_name:
        return dict(available=False,
                    note='needs both cal9 circuits and at least one science '
                         'circuit to compare')
    pq = cal9_out['per_qubit']
    # Analysed key position of each logical qubit in the SCIENCE circuits.  Under
    # the vendor key order the reversed key puts logical qubit j at analysed
    # position j, so this is `j` itself -- NOT the classical bit the MEAS map
    # assigns it.  An earlier version used the classical bit as the position,
    # which put logical q3 at position 7 (its classical bit) instead of 3 and so
    # compared cal9's channel against the wrong column of the science counts.
    # The MEAS map is still read, but as a cross-check on the two independent
    # routes to the same answer rather than as the source of the position.
    sci_meas = dict((cb, qb) for qb, cb in
                    ((t[1], t[2]) for t in _science_meas_reference(phi)
                     if t[0] == 'MEAS'))
    pos_of_logical = dict((qb, qb if rev else q.NQ - 1 - qb)
                          for qb in sci_meas.values())
    for lab, d in pq.items():
        lg = d.get('logical_qubit')
        if lg is None or d.get('science_key_pos') is None:
            continue
        assert d['science_key_pos'] == pos_of_logical[lg], (lab, lg, d, rev)
    rows, violations = [], []
    for lab, d in sorted(pq.items()):
        lo = min(1.0 - d['e01'], d['e10'])
        hi = max(1.0 - d['e01'], d['e10'])
        lg = d['logical_qubit']
        if lg is None or lg not in pos_of_logical:
            continue
        sp = cbit_of_position(pos_of_logical[lg], rev)
        for name, r in sorted(counts_by_name.items()):
            if not r:
                continue
            keys = {(k[::-1] if rev else k): v for k, v in r.items()}
            n = int(sum(keys.values()))
            if not n:
                continue
            hit = int(sum(v for k, v in keys.items() if k[sp] == '0'))
            obs = hit / n
            sig = math.sqrt(max(obs * (1.0 - obs), 1e-12) / n)
            outside = (obs < lo - tol_sigma * sig) or (obs > hi + tol_sigma * sig)
            rec = dict(channel=lab, logical_qubit=d['logical_qubit'],
                       science_key_pos=sp, cal9_e01=d['e01'], cal9_e10=d['e10'],
                       cal9_contrast=d['contrast'],
                       allowed_P0_range=[lo, hi], circuit=name,
                       n=n, n_zero=hit, observed_P0=obs, sigma=sig,
                       outside_allowed_range=bool(outside))
            rows.append(rec)
            if outside:
                violations.append(rec)
    if not rows:
        return dict(available=False,
                    note='no per-circuit counts attached; nothing to compare')
    return dict(
        available=True, tol_sigma=tol_sigma, n_comparisons=len(rows),
        n_violations=len(violations), violations=violations, per_comparison=rows,
        cal9_transferable=bool(not violations),
        # a violation means cal9 and the science circuits disagree about which
        # physical channel a logical index refers to, i.e. the vendor re-routed
        vendor_rerouting_detected=bool(bool(violations)),
        note=('cal9 is consistent with the science circuits on all %d '
              'comparisons' % len(rows)) if not violations else
             ('%d of %d comparisons fall OUTSIDE the hard P(read 0) range that '
              'cal9 measured for that channel, so cal9 and the science circuits '
              'are reading DIFFERENT physical channels: the vendor transpiler '
              're-routes logical qubits per circuit, and cal9 cannot validate or '
              'invalidate the science estimator'
              % (len(violations), len(rows))))


def _science_meas_reference(phi=None):
    """The MEAS map of a science circuit, rebuilt from the committed angle file.

    Angles do not affect the measurement layer, so any arm's angles give the same
    map; rebuilding it keeps this module honest about the science circuit's
    q5->c0..q8->c3, q0->c4..q4->c8 ordering instead of hard-coding it."""
    if phi is None:
        phi = load_angle_set('warm')
    return q.build_branch_circuit(q.ENC5, WARMCOLD_PSI, WARMCOLD_SIGNAL_FRAME,
                                  WARMCOLD_SIGNAL_S, phi, late_measure=True)
def _warmcold_healthy_subset(counts_by_name, cal9_out, rev=True, phi=None):
    """POST-HOC estimator over the data channels cal9 did NOT flag as broken.

    This is not the pre-registered test and must never be reported as one: the
    subset is chosen after seeing which channel failed, so it carries a selection
    effect.  Its value is diagnostic, and on the 2026-10-12 run it is decisive --
    but in the opposite direction to what an earlier version of this docstring
    claimed, because that version was computed on the complement bit window (see
    layout_from_meas).  On the correct window:

      primary,  5 data channels : signal -0.00382 +- 0.00520 (-0.73 sigma)
                                  null   +0.01142 +- 0.00402 (+2.84 sigma)
      subset,   q3 excluded     : signal -0.00031 +- 0.00688 (-0.05 sigma)
                                  null   -0.00061 +- 0.00595 (-0.10 sigma)

    So the null control's 2.84 sigma deviation is attributable ENTIRELY to the
    one broken data channel: dropping it moves the null to -0.10 sigma, and moves
    the signal to -0.05 sigma with it.  That is what a genuine single-channel
    artefact looks like -- the bias disappears with the channel, in both arms,
    rather than migrating.  It is also the opposite of the earlier reading, which
    had the five-channel null clean at +0.3 sigma and the four-channel null at
    -5.2 sigma, i.e. exclusion making things worse; that pattern was an artefact
    of measuring the ancilla register plus one data qubit and calling it the data
    register.

    The substantive conclusion is unchanged by which of the two is right: there is
    NO detectable warm-vs-cold effect.  The signal is -0.73 sigma on the
    pre-registered estimator and -0.05 sigma on the subset, and the implied
    transmission factor is w = -0.008, i.e. zero.

    It is only meaningful if cal9's channel labels transfer to the science
    circuits -- see check_cal9_transferability, which on this run says they do
    not (2 of 36 comparisons outside the hard bound, both at q3).  So the
    exclusion is doubly caveated: post-hoc in its choice of channel, and resting
    on a label the transferability check disputes.  Both numbers are reported
    with that attached rather than either one being promoted."""
    if not cal9_out.get('available'):
        return dict(available=False, note='cal9 did not return; cannot identify '
                                          'which channels to exclude')
    ds = slice(*cal9_out['data_slice'])
    all_pos = list(range(ds.start, ds.stop))
    # Map a broken cal9 channel's logical qubit to its science-circuit analysed
    # key position.  Taken from cal9's own table rather than recomputed here, so
    # the two can never disagree about the convention -- an earlier version fed a
    # CLASSICAL BIT into cbit_of_position (which expects a key POSITION), turning
    # logical q3 into position 7 via its classical bit 7.  7 is outside the data
    # window [0:5], so the broken channel was silently not excluded and this
    # function reported "cal9 flagged no channel inside the data register" while
    # the screen one line above was calling the primary estimator INVALID.
    pos_of_logical = dict(
        (v['logical_qubit'], v['science_key_pos'])
        for v in cal9_out.get('per_qubit', {}).values()
        if v.get('logical_qubit') is not None
        and v.get('science_key_pos') is not None)
    # independent restatement of the same convention, so a change to either side
    # is caught: under the vendor key order the reversed key puts logical qubit j
    # at analysed position j
    for _qb, _pos in pos_of_logical.items():
        assert _pos == (_qb if rev else q.NQ - 1 - _qb), (_qb, _pos, rev)
    excluded = sorted(pos_of_logical[b['logical_qubit']]
                      for b in cal9_out['broken_qubits']
                      if b['logical_qubit'] in pos_of_logical
                      and pos_of_logical[b['logical_qubit']] in all_pos)
    keep = [p for p in all_pos if p not in excluded]
    if not excluded:
        return dict(available=True, excluded_positions=[], kept_positions=keep,
                    identical_to_primary=True,
                    note='cal9 flagged no channel inside the data register, so '
                         'this subset IS the pre-registered estimator')

    def _p(counts, positions):
        keys = {(k[::-1] if rev else k): v for k, v in counts.items()}
        n = int(sum(keys.values()))
        hit = int(sum(v for k, v in keys.items()
                      if all(k[p] == '0' for p in positions)))
        return hit, n, (hit / n if n else None)

    per, gaps = {}, {}
    for name, r in sorted(counts_by_name.items()):
        hit, n, p = _p(r, keep)
        lo, hi = wilson(hit, n)
        _eh, _en, ep = _p(r, excluded)
        per[name] = dict(n=n, n_subset_zero=hit, P_subset_zero=p,
                         P_subset_zero_wilson95=[lo, hi],
                         n_kept_positions=len(keep),
                         n_excluded_positions=len(excluded),
                         P_excluded_zero=ep)
    for role, ideal in (('signal', WARMCOLD_GAP_IDEAL_SIGNAL),
                        ('null', WARMCOLD_GAP_IDEAL_NULL)):
        w, c = per.get('%s|warm' % role), per.get('%s|cold' % role)
        if not (w and c) or w['P_subset_zero'] is None or c['P_subset_zero'] is None:
            gaps[role] = dict(complete=False)
            continue
        gap = w['P_subset_zero'] - c['P_subset_zero']
        sg = math.sqrt(w['P_subset_zero'] * (1 - w['P_subset_zero']) / w['n']
                       + c['P_subset_zero'] * (1 - c['P_subset_zero']) / c['n'])
        gaps[role] = dict(complete=True, gap=gap, sigma=sg,
                          z=(gap / sg) if sg > 0 else 0.0,
                          P_warm=w['P_subset_zero'], P_cold=c['P_subset_zero'],
                          gap_ideal=ideal,
                          # the subset's ideal reference is NOT the primary one: it
                          # marginalises over the excluded qubits, so the ideal gap
                          # has to be recomputed on the same subset or the
                          # transmission factor is meaningless.  Both arms'
                          # excluded qubits are marginalised identically, and at
                          # p=0 the null branch is ~1 for both, so the null's ideal
                          # gap stays ~0 while the signal's does not.
                          null_ideal_still_zero=(role == 'null'))
    return dict(available=True, excluded_positions=excluded, kept_positions=keep,
                n_kept=len(keep), n_excluded_channels=len(excluded),
                identical_to_primary=False, per_circuit=per, pairs=gaps,
                post_hoc=True,
                selection_caveat='the excluded set was chosen AFTER seeing which '
                                 'channel cal9 flagged, so this is a subset '
                                 'analysis, not the pre-registered test',
                note='%d of %d data channels kept; %d excluded as broken'
                     % (len(keep), len(all_pos), len(excluded)))


def _gap_z(p_warm, n_warm, p_cold, n_cold):
    """(gap, sigma, z) for a warm-minus-cold difference of two independent
    proportions -- the same unpaired binomial error bar _warmcold_pair uses, so the
    family table and the headline verdict cannot disagree about what sigma means."""
    gap = p_warm - p_cold
    var = 0.0
    for p, n in ((p_warm, n_warm), (p_cold, n_cold)):
        n = max(int(n), 1)
        var += p * (1.0 - p) / n
    sg = math.sqrt(var)
    if sg <= 0:
        return gap, 0.0, (0.0 if gap == 0 else float('inf'))
    return gap, sg, gap / sg


def warmcold_estimator_family(counts_by_name, cal9_out=None, gates_by_name=None,
                              rev=True, anc=None):
    """Every plausible A/B estimator on ONE set of counts, scored on the signal
    branch AND on the matched null.

    WHY THIS IS A FUNCTION AND NOT A PARAGRAPH.  The primary estimator is
    P(data=00000) over all five data channels, and that choice is defensible only if
    the alternatives are visibly worse on the NULL branch, where the ideal gap is
    3e-6 and so any |z| above ~3 is a false signal by construction.  Ranking
    estimators by how significant the SIGNAL branch looks is exactly how a dead
    readout channel gets promoted into a discovery: on the frozen 2026-10-12 counts
    the single-bit marginals and the full-key estimator show null-branch deviations
    far larger than the primary's, while the primary restricted to the four channels
    cal9 cleared sits at -0.10 sigma.  This table is the evidence that the estimator
    was not chosen by which one worked, and selftest (n) re-derives it from the
    frozen dump and asserts the ordering.

    Members: the primary (all five data channels zero), the cal9-cleared subset, the
    five single-bit marginals, the full expected key, and the post-selected F_s the
    2026-09 feasibility run reported.  `gates_by_name` is needed for the
    expected-key member only; without it that member is skipped rather than guessed.
    """
    if anc is None:
        anc, rev = layout_from_meas(science_meas_layer())
    ds = _data_slice_for(anc)
    all_pos = list(range(ds.start, ds.stop))

    def _rev(counts):
        return {(k[::-1] if rev else k): v for k, v in counts.items()}

    def _p(counts, pred):
        keys = _rev(counts)
        n = int(sum(keys.values()))
        hit = int(sum(v for k, v in keys.items() if pred(k)))
        return (hit / n if n else None), n, hit

    kept = all_pos
    if cal9_out and cal9_out.get('available') and cal9_out.get('kept_positions'):
        kept = list(cal9_out['kept_positions'])
    members = {}
    members['P_data_zero_all%d' % len(all_pos)] = dict(
        kind='primary', detail='all %d data channels read 0' % len(all_pos),
        pred=lambda k: all(k[p] == '0' for p in all_pos))
    if kept != all_pos:
        members['P_data_zero_cal9_cleared%d' % len(kept)] = dict(
            kind='subset', detail='positions %s: the ones cal9 did not flag' % kept,
            pred=lambda k: all(k[p] == '0' for p in kept))
    for p in all_pos:
        members['P_bit%d_zero' % p] = dict(
            kind='single_bit', detail='data position %d alone' % p,
            pred=lambda k, _p=p: k[_p] == '0')
    if gates_by_name:
        ek_by_role = {}
        for role in ('signal', 'null'):
            g = gates_by_name.get('%s|warm' % role)
            if g is None:
                continue
            ek, _ = expected_vendor_key(g)
            ek_by_role[role] = ek[::-1] if rev else ek
        if len(ek_by_role) == 2:
            # ONE member, evaluated against each branch's OWN pre-registered key.  A
            # statistic that changes with the branch is still null-controllable --
            # that is what makes the pair (signal z, null z) meaningful -- whereas
            # applying the signal branch's key to the null branch would be zero by
            # construction and would fake a clean null.
            members['P_expected_key'] = dict(
                kind='full_key',
                detail='the full %d-bit key each branch pre-registers' % q.NQ,
                pred_by_role={r: (lambda k, _e=e: k == _e)
                              for r, e in ek_by_role.items()},
                pred=None, keys=ek_by_role)
    rows = {}
    for name, spec in members.items():
        pbr = spec.get('pred_by_role') or {}
        per, pairs = {}, {}
        for cname, counts in sorted(counts_by_name.items()):
            r = cname.split('|')[0]
            pred = pbr.get(r, spec['pred'])
            if pred is None:
                continue
            p, n, hit = _p(counts, pred)
            per[cname] = dict(P=p, n=n, n_hit=hit)
        for r, ideal in (('signal', WARMCOLD_GAP_IDEAL_SIGNAL),
                         ('null', WARMCOLD_GAP_IDEAL_NULL)):
            w, c = per.get('%s|warm' % r), per.get('%s|cold' % r)
            if not w or not c or w['P'] is None or c['P'] is None:
                pairs[r] = dict(complete=False)
                continue
            gap, sg, z = _gap_z(w['P'], w['n'], c['P'], c['n'])
            pairs[r] = dict(complete=True, gap=gap, sigma=sg, z=z, gap_ideal=ideal,
                            P_warm=w['P'], P_cold=c['P'])
        rows[name] = dict(kind=spec['kind'], detail=spec['detail'],
                          per_circuit=per, pairs=pairs)
    # the post-selected secondary, scored as a member of the SAME family rather than
    # as a separate analysis: same counts, same arms, same error bar, and its own
    # branch's syndrome as the selection (which is what makes it null-controllable)
    fs_per, fs_pairs = {}, {}
    for role, s in (('signal', WARMCOLD_SIGNAL_S), ('null', WARMCOLD_NULL_S)):
        for arm in ('warm', 'cold'):
            counts = counts_by_name.get('%s|%s' % (role, arm))
            if not counts:
                continue
            n_sel, n_hit = q.counts_to_branch_stats(counts, s, anc, rev)
            n = int(sum(counts.values()))
            fs_per['%s|%s' % (role, arm)] = dict(
                P=(n_hit / n_sel if n_sel else None), n=n_sel, n_hit=n_hit,
                n_total=n, selected_fraction=(n_sel / n if n else None))
        w, c = fs_per.get('%s|warm' % role), fs_per.get('%s|cold' % role)
        ideal = (WARMCOLD_GAP_IDEAL_SIGNAL if role == 'signal'
                 else WARMCOLD_GAP_IDEAL_NULL)
        if w and c and w['P'] is not None and c['P'] is not None:
            gap, sg, z = _gap_z(w['P'], w['n'], c['P'], c['n'])
            fs_pairs[role] = dict(complete=True, gap=gap, sigma=sg, z=z,
                                  gap_ideal=ideal, P_warm=w['P'], P_cold=c['P'])
        else:
            fs_pairs[role] = dict(complete=False)
    if fs_per:
        rows['F_s_postselected'] = dict(
            kind='postselected', per_circuit=fs_per, pairs=fs_pairs,
            detail='P(data=00000 | this branch\'s syndrome), the 2026-09 '
                   'feasibility estimator; it conditions on the ancilla and so '
                   'discards most shots')
    nulls = {k: v['pairs']['null']['z'] for k, v in rows.items()
             if v['pairs'].get('null', {}).get('complete')}
    sigs = {k: v['pairs']['signal']['z'] for k, v in rows.items()
            if v['pairs'].get('signal', {}).get('complete')}
    return dict(rev=rev, anc_slice=[anc.start, anc.stop],
                data_slice=[ds.start, ds.stop], members=rows, n_members=len(rows),
                null_z_by_member=nulls, signal_z_by_member=sigs,
                max_abs_null_z=(max(abs(z) for z in nulls.values())
                                if nulls else None),
                max_abs_null_member=(max(nulls, key=lambda k: abs(nulls[k]))
                                     if nulls else None),
                max_abs_signal_member=(max(sigs, key=lambda k: abs(sigs[k]))
                                       if sigs else None),
                primary_member='P_data_zero_all%d' % len(all_pos))


def analyze_suppl(results, meta, phi_cold, shots_requested=None, verbose=True):
    """Read the supplementary psi='0' grid against the 2026-09 values it rebuilds.

    Every quantity here has a 2026-09 counterpart in hw_error_budget.json and the
    point of the stage is the comparison, so the output is a table of pairs rather
    than a fresh estimate: F_s, the end-to-end expected-key rate, the implied
    per-instruction error, the syndrome's rank and significance against code-blind,
    and the a1 relaxation satellite.

    The layout is derived PER CIRCUIT from that circuit's own MEAS map rather than
    once for the job.  On this grid the four circuits happen to agree -- both
    variants emit the four ancilla measurements before the five data measurements,
    so mid and late land on the same classical-bit order, and feeding the archived
    2026-09 counts back through this function confirms it (all four derive
    anc=[5,9] rev=True).  The per-circuit derivation is therefore defensive rather
    than load-bearing here: it is what stops a future grid that DOES reorder the
    measurements from being silently read on one shared window, which is the class
    of error that produced the retracted -7.8 sigma signal."""
    sci = [(r, m) for r, m in zip(results, meta) if m.get('kind') == 'suppl']
    out = dict(psi=SUPPL_PSI, p=0.0, shots_requested=shots_requested,
               reproduces='2026-09 feasibility set (raw_verify_options.json)',
               per_circuit={})
    if not sci:
        out['not_analysable'] = ('this job returned no suppl circuit, so there is '
                                 'nothing to compare against the 2026-09 set')
        print('[suppl] NOT ANALYSABLE: %s' % out['not_analysable'], flush=True)
        return out
    inv = suppl_gate_inventory()
    eb = json.load(open(os.path.join(ROOT, 'hw_error_budget.json')))
    for r, m in sci:
        lab, late = m['label'], (m['measure'] == 'late')
        # Re-verify at ANALYSIS time as well as build time: the build assert proves
        # the right circuit was submitted, this proves what came back is still being
        # read as that circuit.
        gates = q.build_branch_circuit(q.ENC5, m['psi'], m['frame'], m['s'],
                                       phi_cold, late_measure=late)
        assert len(gates) == inv[lab]['instructions'], (lab, len(gates))
        anc, rev = layout_from_meas(gates)
        n_sel, n_sel_hit = q.counts_to_branch_stats(r, m['s'], anc, rev)
        n_zero, n_tot = data_zero_counts(r, anc, rev)
        s_ia = format(m['s'], '04b')
        syn = {}
        for k, v in _counts_items(r):
            rk = k[::-1] if rev else k
            a = rk[anc.start:anc.stop]
            syn[a] = syn.get(a, 0) + v
        p_s = syn.get(s_ia, 0)
        ordered = sorted(syn.values(), reverse=True)
        pos = (ordered.index(p_s) + 1) if p_s in ordered else None
        exp = n_tot / 16.0
        sd = math.sqrt(n_tot * (1 / 16.) * (15 / 16.)) if n_tot else 0.0
        z = ((p_s - exp) / sd) if sd else None
        sb = list(s_ia)
        sat = None
        if sb[1] == '1':
            sb[1] = '0'
            sat = ''.join(sb)
        e2e = (n_sel_hit / float(n_tot)) if n_tot else None
        # Shannon entropy of the FULL raw outcome distribution, plus the same
        # quantity normalised by the most entropy these shots could carry.  Must be
        # measured over the complete bitstring, NOT over `syn`: syn is keyed by the
        # 4-bit ancilla syndrome and is capped at 4 bits, so thresholding it on a
        # 9-bit scale flags every healthy run.  (That mistake was made once here and
        # caught by the archived 2026-09 counts in selftest (q), which sit at only
        # 5.58-6.05 bits because they carry ~369 shots rather than 20 000 -- the
        # reason the guard thresholds the RATIO and not the raw bit count.)
        _items = _counts_items(r)
        _nc = float(sum(v for _k, v in _items))
        _p = [v / _nc for _k, v in _items if v > 0] if _nc else []
        outcome_H = -sum(x * math.log2(x) for x in _p) if _p else 0.0
        _hcap = math.log2(min(_nc, 2.0 ** 9)) if _nc > 1 else 0.0
        outcome_Hr = (outcome_H / _hcap) if _hcap > 0 else 0.0
        ref = eb['circuits'][lab]
        out['per_circuit'][lab] = dict(
            label=lab, measure=m['measure'], frame=m['frame'], s=m['s'],
            n_total=n_tot, n_selected=n_sel, n_selected_hit=n_sel_hit,
            n_data_zero=n_zero, outcome_entropy_bits=outcome_H,
            outcome_entropy_ratio=outcome_Hr,
            F_s=(n_sel_hit / float(n_sel)) if n_sel else None,
            F_s_wilson95=(list(wilson(n_sel_hit, n_sel)) if n_sel else None),
            P_expected_key=e2e,
            implied_eps_instr=((1.0 - e2e ** (1.0 / inv[lab]['instructions']))
                               if e2e else None),
            n_instructions=inv[lab]['instructions'],
            p_correct_syndrome=(p_s / float(n_tot)) if n_tot else None,
            syndrome_rank=pos, syndrome_z_vs_code_blind=z,
            relaxation_satellite_s=(int(sat, 2) if sat else None),
            relaxation_satellite_ratio=(
                (syn.get(sat, 0) / float(p_s)) if (sat and p_s) else None),
            anc_slice=[anc.start, anc.stop], rev=rev,
            ideal_P_data_zero=m.get('ideal_P_data_zero'),
            target_2026_09=dict(
                shots_used=ref['shots_used'],
                n_sel=int(round(ref['syndrome_channel']['joint_all_correct']
                                * ref['shots_used'])),
                n_hit=ref['data_channel'].get('n_hit'),
                F_s=ref['data_channel'].get('F_s'),
                P_expected_key=ref['budget']['end_to_end_expected_key'],
                implied_eps_instr=ref['budget']['implied_mean_instruction_error']))
        if verbose:
            c = out['per_circuit'][lab]
            # An explicit None test, not `x or nan`: a noiseless circuit has
            # implied_eps == 0.0, which is falsy, and printing that as nan would
            # hide the one value that says the chain is exact.
            _e = c['implied_eps_instr']
            _er = ref['budget']['implied_mean_instruction_error']
            print('[suppl] %s (%s, %s, s=%d) n=%d  F_s=%s vs 2026-09 %s  '
                  'eps=%.4e vs %.4e  syndrome rank %s z=%+.1f  satellite=%s'
                  % (lab, m['measure'], m['frame'], m['s'], n_tot,
                     ('%.4f' % c['F_s']) if c['F_s'] is not None else 'n/a',
                     ref['data_channel'].get('F_s'),
                     _e if _e is not None else float('nan'),
                     _er if _er is not None else float('nan'),
                     pos, z if z is not None else float('nan'),
                     ('%.3f' % c['relaxation_satellite_ratio'])
                     if c['relaxation_satellite_ratio'] is not None else 'n/a'),
                  flush=True)
    _srt = sorted(out['per_circuit'].items())
    out['outcome_entropy_bits'] = dict(
        (k, v.get('outcome_entropy_bits')) for k, v in _srt)
    out['outcome_entropy_ratio'] = dict(
        (k, v.get('outcome_entropy_ratio')) for k, v in _srt)
    _ents = [v.get('outcome_entropy_bits') for _k, v in _srt
             if v.get('outcome_entropy_bits') is not None]
    _rat = [v.get('outcome_entropy_ratio') for _k, v in _srt
            if v.get('outcome_entropy_ratio') is not None]
    out['entropy_min'] = min(_ents) if _ents else None
    out['entropy_max'] = max(_ents) if _ents else None
    out['entropy_ratio_min'] = min(_rat) if _rat else None
    out['entropy_ratio_max'] = max(_rat) if _rat else None
    # Degenerate when EVEN THE BEST returned circuit is below the guard.  Valid runs
    # measure 0.654-0.827 (archived 2026-09 and both 2026-10-24 submissions), the
    # failed one 0.275-0.292, so 0.45 cannot fire on good data.
    out['degenerate'] = bool(_rat) and max(_rat) < SUPPL_MIN_OUTCOME_ENTROPY_RATIO
    out['comparison'] = suppl_verdict(out['per_circuit'],
                                      degenerate=out['degenerate'],
                                      entropy_ratio_max=out['entropy_ratio_max'])
    if verbose:
        print('[suppl] VERDICT: %s' % out['comparison']['verdict'], flush=True)
        print('[suppl]   %s' % out['comparison']['reason'], flush=True)
    return out


def suppl_verdict(per, degenerate=False, entropy_ratio_max=None):
    """The pre-declared reading of the psi='0' reproduction.

    Declared here, in code, before the data is bought -- the same discipline the
    warmcold A/B's decision rule was held to, and the reason that A/B could be
    called FAILED rather than reinterpreted.  The bar is a factor, not a vibe:
    F_s within 2x of 0.80 reproduces the September device point, and anything at
    or below the psi='+' value does not.  In between is reported as inconclusive
    rather than rounded into whichever answer is convenient.

    A run whose outcome distribution is degenerate is refused before any of that
    logic runs, because such a run also produces F_s = 0.0000 and so would
    otherwise be indistinguishable from a genuine NOT REPRODUCED."""
    if degenerate:
        return dict(verdict='NOT ANALYSABLE', degenerate=True,
                    entropy_ratio_max=entropy_ratio_max,
                    reason=('outcome distribution is degenerate: even the best '
                            'returned circuit carries a normalised outcome entropy '
                            'of only %.3f, below the %.2f suppl guard.  Valid runs on '
                            'these same circuits measure 0.654-0.827; this one '
                            'collapsed onto ~50 distinct bitstrings instead of ~500, '
                            'with ~50%% of all shots in a single outcome, and returned '
                            'near-identical z ~= -36.5 across two frames and two '
                            'measurement depths, which genuinely different circuits '
                            'cannot do.  It is rejected as a device or compilation '
                            'failure, not read as evidence about the September device '
                            'point -- noting that its F_s of 0.0000 would otherwise '
                            'have produced the same verdict string as a real one.'
                            % (entropy_ratio_max if entropy_ratio_max is not None
                               else -1.0, SUPPL_MIN_OUTCOME_ENTROPY_RATIO)))
    d = per.get('D')
    if not d or d.get('F_s') is None:
        return dict(verdict='NOT ANALYSABLE',
                    reason='circuit D returned no post-selected shot')
    fs, tgt = d['F_s'], SUPPL_TARGETS['D']['F_s']
    fs_plus = 0.12092          # the psi='+' value for the same nominal branch
    eps = d.get('implied_eps_instr')
    # An uncomputable implied error rate means the device point cannot be moved and
    # the envelope ratio cannot be formed.  Emitting a verdict anyway prints nan
    # into a sentence the manuscript quotes verbatim -- the degenerate 2026-10-24
    # submission produced "moves to eps=nan ... to nanx outside it" and still read
    # NOT REPRODUCED, so the guard is on the number, not on the string.
    if eps is None or eps != eps:
        return dict(verdict='NOT ANALYSABLE',
                    reason=('circuit D has F_s = %.4f but no finite implied error '
                            'rate, so the device point cannot be moved and the '
                            'envelope ratio cannot be formed; no reproduction '
                            'verdict is supportable from this run' % fs),
                    F_s_D=fs, target_2026_09=tgt, psi_plus_value=fs_plus,
                    implied_eps_D=eps)
    if fs >= tgt / 2.0:
        v = 'REPRODUCED'
        why = ('F_s = %.4f is within 2x of the 2026-09 %.2f at psi=\'0\', so the '
               'September device point is real but BASIS-BOUND: it was measured '
               'where the dominant logical-Z failure is invisible, and Layer 4 must '
               'quote the psi=\'+\' value %.4f beside it.' % (fs, tgt, fs_plus))
    elif fs <= fs_plus * 1.5:
        v = 'NOT REPRODUCED'
        # eps is guaranteed finite by the guard above
        _e = eps
        why = ('F_s = %.4f at psi=\'0\' is at or below the psi=\'+\' value %.4f, so '
               'the 2026-09 %.2f does not survive 54x the shots on its own input '
               'state; the device point moves to eps=%.4e and the envelope verdict '
               'goes from 1.35x outside its loose end to %.2fx outside it.'
               % (fs, fs_plus, tgt, _e, _e / 7.70e-3))
    else:
        v = 'INCONCLUSIVE'
        why = ('F_s = %.4f sits between the 2026-09 %.2f and the psi=\'+\' %.4f; '
               'neither reading is supported, and the shot count must go up rather '
               'than the interpretation bend.' % (fs, tgt, fs_plus))
    return dict(verdict=v, reason=why, F_s_D=fs, target_2026_09=tgt,
                psi_plus_value=fs_plus, implied_eps_D=eps,
                rule=("REPRODUCED if F_s >= target/2; NOT REPRODUCED if "
                      "F_s <= 1.5 x the psi='+' value; else INCONCLUSIVE.  "
                      "Declared in code before purchase."))


def _counts_items(r):
    """Iterate (key, count) over whatever shape the platform returned."""
    return list(r.items()) if isinstance(r, dict) else list(r)


def analyze_warmcold(results, meta, phi_warm, phi_cold, shots_requested=None,
                     verbose=True):
    """The warm-vs-cold A/B: two arms x (one signal branch + one null branch).

    Reuses qcloud_vscr_new's `counts_to_branch_stats`, so the secondary
    post-selected number is computed by literally the same code the 2026-09
    feasibility run used and any difference is in the device, not in the analysis.
    The LAYOUT, by contrast, is derived from the circuits' MEAS maps
    (`layout_from_meas`) rather than from `detect_layout`, whose outcome-scored
    criterion is circular; detect_layout is still computed and any disagreement is
    recorded in `out['layout']`.

    Tolerates a truncated dump: with only chunk 1 present whatever pairs returned
    are still analysed, and the missing ones are reported as missing rather than
    silently dropped.  The verdict distinguishes FAILED (the pre-registered
    criteria were not met) from INVALID (the apparatus is not characterised well
    enough for the criteria to mean anything)."""
    sci = [(r, m) for r, m in zip(results, meta) if m.get('kind') == 'warmcold']
    cal = [(r, m) for r, m in zip(results, meta) if m.get('kind') == 'cal9']
    per = {}
    counts_by_name = {}
    out = dict(primary_estimator=WARMCOLD_PRIMARY,
               secondary_estimator=WARMCOLD_SECONDARY,
               psi=WARMCOLD_PSI, p=0.0, shots_requested=shots_requested)
    if not sci:
        msg = ('this job returned no warmcold circuit at all, so there is no '
               'warm-vs-cold comparison to report')
        print('[warmcold] NOT ANALYSABLE: %s' % msg, flush=True)
        out['not_analysable'] = msg
        return out
    # ONE layout for the whole job: all four science circuits share the block and
    # the submission, so they share the bit order.
    #
    # The layout is DERIVED from the circuit's MEAS map plus the platform's
    # measured key order, not inferred from the outcomes.  `q.detect_layout`
    # scores candidate windows against the counts, but "is this key a syndrome
    # match" depends on the window being scored, so the score can favour the
    # wrong window by any margin: on the 2026-10-12 run it reported
    # anc=[0,4]/rev=True, the exact COMPLEMENT of the correct anc=[5,9]/rev=True,
    # by 62 data-hits to 2, and the artifact then labelled the ancilla register
    # as the data register.  Both are computed here and any disagreement is
    # recorded rather than hidden.
    ref_gates = q.build_branch_circuit(q.ENC5, sci[0][1]['psi'],
                                       sci[0][1]['frame'], sci[0][1]['s'],
                                       phi_warm, late_measure=True)
    anc, rev = layout_from_meas(ref_gates)
    det_anc, det_rev = q.detect_layout(
        [(m['psi'], m['p'], m['frame'], m['s']) for r, m in sci],
        [r for r, m in sci])
    agree = (det_anc == anc and det_rev == rev)
    # A disagreement DOES invalidate the primary number, and an earlier comment
    # here claimed otherwise on the strength of "'00000' is a palindrome".  That
    # reasoning confused reversing the CONTENT of the window with moving the
    # window: the two candidate windows cover disjoint qubit sets.  Under the
    # vendor key order, anc=[5,9]/rev=True tests logical qubits 0-4 (the data
    # register) while anc=[0,4]/rev=True tests logical qubits 4-8 -- one data
    # qubit and all four ANCILLAS.  They are different observables, and on the
    # 2026-10-12 counts they differ by a factor of several sigma in the reported
    # gap.  The alternative window is still computed, but as evidence about how
    # badly the choice matters, not as a reassurance that it does not.
    alt_hit = [data_zero_counts(r, det_anc, det_rev)[0] for r, m in sci]
    out['anc_slice'] = [anc.start, anc.stop]
    out['rev'] = rev
    ds_ref = _data_slice_for(anc)
    out['layout'] = dict(
        source='derived from the circuit MEAS(q->cbit) map',
        anc_slice=[anc.start, anc.stop], rev=rev,
        data_slice=[ds_ref.start, ds_ref.stop],
        detect_layout_anc_slice=[det_anc.start, det_anc.stop],
        detect_layout_rev=det_rev,
        detect_layout_agrees=agree,
        primary_estimator_identical_under_both=bool(
            all(a == b for a, b in zip(alt_hit,
                                       [data_zero_counts(r, anc, rev)[0]
                                        for r, m in sci]))),
        note=('agrees with detect_layout' if agree else
              'DISAGREES with detect_layout, which scores candidate windows '
              'against the measured outcomes and is therefore circular; the '
              'MEAS-derived layout is authoritative and selftest (m) verifies it '
              'on the noiseless simulator'))
    for r, m in sci:
        phi = phi_warm if m['arm'] == 'warm' else phi_cold
        counts_by_name['%s|%s' % (m['role'], m['arm'])] = r
        n_hit, n_tot = data_zero_counts(r, anc, rev)
        n_sel, n_sel_hit = q.counts_to_branch_stats(r, m['s'], anc, rev)
        lo, hi = wilson(n_hit, n_tot)
        if n_sel:
            slo, shi = wilson(n_sel_hit, n_sel)
        else:
            slo = shi = None
        per['%s|%s' % (m['role'], m['arm'])] = dict(
            role=m['role'], arm=m['arm'], s=m['s'], frame=m['frame'],
            angles=m['angles'], n_total=n_tot, n_data_zero=n_hit,
            P_data_zero=n_hit / n_tot if n_tot else None,
            P_data_zero_wilson95=[lo, hi],
            ideal_P_data_zero=ideal_data_zero_prob(m['psi'], m['frame'], m['s'],
                                                   phi),
            n_selected=n_sel, n_selected_hit=n_sel_hit,
            selected_fraction=n_sel / n_tot if n_tot else None,
            F_s=(n_sel_hit / n_sel) if n_sel else None,
            F_s_wilson95=[slo, shi],
            top3=sorted(r.items(), key=lambda kv: -kv[1])[:3])
    out['per_circuit'] = per
    out['pairs'] = {}
    for role, gap_ideal in (('signal', WARMCOLD_GAP_IDEAL_SIGNAL),
                            ('null', WARMCOLD_GAP_IDEAL_NULL)):
        w_, c_ = per.get('%s|warm' % role), per.get('%s|cold' % role)
        if not (w_ and c_):
            out['pairs'][role] = dict(
                complete=False,
                missing=[a for a, d in (('warm', w_), ('cold', c_)) if not d],
                note='truncated dump: this pair did not both return')
            continue
        out['pairs'][role] = _warmcold_pair(w_, c_, gap_ideal)
    out['cal9'] = _analyze_cal9(cal, anc, rev)
    # two diagnostics the 2026-10-12 run showed were missing.  The first asks
    # whether cal9's channel labels even transfer to the science circuits (they do
    # not, if the vendor re-routes per circuit); the second reports what the A/B
    # looks like with the suspect channel removed.  Neither is the pre-registered
    # test, and both are recorded precisely so the pre-registered one cannot be
    # quietly replaced by whichever of the three is most flattering.
    out['cal9_transferability'] = check_cal9_transferability(
        counts_by_name, out['cal9'], rev, phi=phi_warm)
    out['posthoc_healthy_subset'] = _warmcold_healthy_subset(
        counts_by_name, out['cal9'], rev, phi=phi_warm)
    # PROVENANCE, spelled out because the key name invites a misreading that the
    # 2026-10-21 WK_C180_2 run actually produced: `measured_w` is NOT measured by
    # THIS run.  It is a FORWARD projection at warmcold_power's default w, i.e. the
    # frozen 2026-09 constant WARMCOLD_W_MEASURED_2026_09 = +0.0936.  This run's
    # own transmission estimate is verdict.signal_transmission_w, which came out
    # -0.0937 -- the SAME magnitude with the OPPOSITE sign, because the pre-
    # registered prediction failed on direction.  Two numbers that nearly cancel
    # sitting in one file under names that both suggest "measured" is exactly how a
    # manuscript ends up quoting the historical constant as if it were the result.
    # Read the actual measurement from out['verdict'].
    out['power_prediction'] = dict(
        measured_w=warmcold_power(shots=shots_requested),
        pessimistic_w=warmcold_power(w=WARMCOLD_W_PESSIMISTIC,
                                     shots=shots_requested),
        provenance=(
            'FORWARD PROJECTION ONLY, not this run\'s measurement.  measured_w uses '
            'warmcold_power\'s default w = WARMCOLD_W_MEASURED_2026_09 (%+.5f), the '
            'frozen 2026-09 constant; pessimistic_w uses WARMCOLD_W_PESSIMISTIC '
            '(half of it).  This run\'s own transmission estimate is '
            'verdict.signal_transmission_w, and the model\'s shared-w assumption is '
            'what that number tests -- invert p = w + (1-w)/32 (warm) or '
            'p = 0.5w + (1-w)/32 (cold) on per_circuit to see that the two arms do '
            'not share one w.'
            % WARMCOLD_W_MEASURED_2026_09))
    out['verdict'] = _warmcold_verdict(out)
    if verbose:
        for ln in _warmcold_report(out):
            print(ln, flush=True)
    return out


def _warmcold_pair(w_, c_, gap_ideal):
    """One arm-pair comparison: the primary gap, its sigma, and what the device
    transmitted.  The error bar is the unpaired binomial one (both arms are
    independent shots on the same block), not a paired-subtraction one -- there
    is nothing to pair at the shot level."""
    gap = w_['P_data_zero'] - c_['P_data_zero']
    var = 0.0
    for d in (w_, c_):
        n = max(d['n_total'], 1)
        var += d['P_data_zero'] * (1.0 - d['P_data_zero']) / n
    sig = math.sqrt(var)
    if sig > 0:
        z = gap / sig
    else:
        # Zero variance in BOTH arms happens on a noiseless simulator and on a
        # device that saturates.  The gap is then either exactly zero -- nothing to
        # resolve, z = 0 -- or a deterministic contradiction that no finite z can
        # describe.  Returning None here would make the verdict crash instead of
        # reporting, which is the worse failure.
        z = 0.0 if gap == 0.0 else float('inf')
    fs = (w_['F_s'] - c_['F_s']
          if (w_['F_s'] is not None and c_['F_s'] is not None) else None)
    # the transmission factor is only meaningful where there IS an ideal effect to
    # transmit; dividing the null branch's measured gap by its 3e-6 ideal gap would
    # produce a huge number that means nothing
    trans = (gap / gap_ideal) if abs(gap_ideal) > WARMCOLD_GAP_TOL else None
    return dict(complete=True, s=w_['s'], frame=w_['frame'],
                gap_measured=gap, sigma_gap=sig, z=z,
                gap_ideal=gap_ideal,
                ideal_gap_recomputed=(w_['ideal_P_data_zero']
                                      - c_['ideal_P_data_zero']),
                transmission_w=trans,
                P_data_zero_warm=w_['P_data_zero'],
                P_data_zero_cold=c_['P_data_zero'],
                F_s_warm=w_['F_s'], F_s_cold=c_['F_s'], F_s_gap=fs,
                selected_fraction_warm=w_['selected_fraction'],
                selected_fraction_cold=c_['selected_fraction'])


def _warmcold_verdict(out):
    """The two-part pass/fail, fixed before the run: the signal branch must move
    in the predicted direction, and the matched null branch must not move.  A
    signal with a dirty null is not a result -- it is a systematic."""
    sig = out['pairs'].get('signal', {})
    nul = out['pairs'].get('null', {})
    v = dict(signal_complete=bool(sig.get('complete')),
             null_complete=bool(nul.get('complete')),
             z_threshold=3.0)
    if sig.get('complete'):
        v['signal_z'] = sig['z']
        v['signal_gap'] = sig['gap_measured']
        v['signal_detected'] = bool(abs(sig['z']) >= v['z_threshold'])
        v['signal_direction_correct'] = bool(sig['gap_measured'] > 0)
        v['signal_transmission_w'] = sig['transmission_w']
    if nul.get('complete'):
        v['null_z'] = nul['z']
        v['null_gap'] = nul['gap_measured']
        v['null_consistent_with_zero'] = bool(abs(nul['z']) < v['z_threshold'])
    # The cal9 screen decides whether a non-pass is a RESULT or an APPARATUS
    # FAULT.  A broken data channel invalidates the primary estimator outright; a
    # broken ancilla invalidates only the post-selected F_s.  Conflating the two
    # is how a device failure gets written up as physics.
    cal = out.get('cal9') or {}
    scr = cal.get('screen') or {}
    lay = out.get('layout') or {}
    tr = out.get('cal9_transferability') or {}
    v['cal9_screen'] = dict(
        available=bool(cal.get('available')),
        n_broken=cal.get('n_broken'),
        broken_qubits=[b['label'] for b in (cal.get('broken_qubits') or [])],
        broken_logical_qubits=[b['logical_qubit']
                               for b in (cal.get('broken_qubits') or [])],
        primary_estimator_valid=scr.get('primary_estimator_valid'),
        secondary_estimator_valid=scr.get('secondary_estimator_valid'))
    v['cal9_transferable'] = tr.get('cal9_transferable')
    v['vendor_rerouting_detected'] = tr.get('vendor_rerouting_detected')
    v['layout_derived'] = bool(lay)
    v['layout_agrees_with_detect_layout'] = lay.get('detect_layout_agrees')
    # Validity needs BOTH a clean screen AND evidence that the screen describes the
    # channels the science circuits actually used.  If cal9's labels do not
    # transfer, the screen cannot clear the run either -- an unbroken cal9 would
    # say nothing about a re-routed science circuit.
    rerouted = bool(tr.get('vendor_rerouting_detected'))
    primary_valid = bool(scr.get('primary_estimator_valid', True)) and not rerouted
    v['primary_estimator_valid'] = primary_valid
    if v['signal_complete'] and v['null_complete']:
        v['ab_test_passed'] = bool(v.get('signal_detected')
                                   and v.get('signal_direction_correct')
                                   and v.get('null_consistent_with_zero'))
        if rerouted:
            v['ab_test_result'] = 'INVALID'
            v['note'] = (
                'NO CONCLUSION: %d of %d channel comparisons violate the hard '
                'P(read 0) bound that cal9 measured, so cal9 and the science '
                'circuits are reading DIFFERENT physical channels -- the vendor '
                'transpiler re-routes per circuit.  The block\'s readout is '
                'therefore uncharacterised for the circuits that matter, the '
                'primary estimator cannot be validated, and the pre-registered '
                'criteria (%s) cannot be interpreted as physics.  This is an '
                'apparatus/compilation fault, not a warm-vs-cold result'
                % (tr.get('n_violations', 0), tr.get('n_comparisons', 0),
                   'signal %+.1f sigma, null %+.1f sigma'
                   % (v.get('signal_z', float('nan')),
                      v.get('null_z', float('nan')))))
        elif not scr.get('primary_estimator_valid', True):
            v['ab_test_result'] = 'INVALID'
            v['note'] = (
                'NO CONCLUSION: the cal9 screen found a BROKEN channel in the '
                'DATA register (%s), so the primary estimator P(data=00000) is '
                'not interpretable.  Whether or not the A/B criteria were met is '
                'irrelevant -- this is an apparatus fault, not a physical '
                'negative result, and the block must be re-picked and re-run'
                % ','.join(v['cal9_screen']['broken_qubits']))
        elif v['ab_test_passed']:
            v['ab_test_result'] = 'PASSED'
            v['note'] = ('warm-started recovery beats cold-started on branch s=%d '
                         'by %+.4f +- %.4f (%.1f sigma), and the matched null '
                         'branch s=%d moves by %+.4f +- %.4f (%.1f sigma)'
                         % (sig['s'], sig['gap_measured'], sig['sigma_gap'],
                            sig['z'], nul['s'], nul['gap_measured'],
                            nul['sigma_gap'], nul['z']))
        else:
            v['ab_test_result'] = 'FAILED'
            # The pre-registered criterion is directional, so a large gap the
            # WRONG way is a failure and must never be reported as a detection.
            # Saying so plainly is the whole point of having pre-registered it.
            why = []
            if not v.get('signal_direction_correct'):
                why.append('the signal gap is NEGATIVE (%+.4f): warm performed '
                           'significantly WORSE than cold, the opposite of the '
                           'pre-registered prediction' % sig['gap_measured'])
            if not v.get('signal_detected'):
                why.append('the signal gap is not resolved (%.1f sigma < %.1f)'
                           % (sig['z'], v['z_threshold']))
            if not v.get('null_consistent_with_zero'):
                why.append('the null control moved (%.1f sigma), so the '
                           'apparatus is not clean' % nul['z'])
            v['note'] = ('A/B FAILED: %s. Null branch s=%d moved by %+.4f +- '
                         '%.4f (%.1f sigma)%s'
                         % ('; '.join(why), nul['s'], nul['gap_measured'],
                            nul['sigma_gap'], nul['z'],
                            '' if v['cal9_screen']['secondary_estimator_valid']
                            else ', and a broken ANCILLA (%s) means the '
                                 'post-selected F_s is uninterpretable while '
                                 'the unconditioned primary estimator above '
                                 'still stands'
                            % ','.join(v['cal9_screen']['broken_qubits'])))
    elif v['signal_complete']:
        v['ab_test_result'] = 'INCOMPLETE'
        v['note'] = ('signal pair complete (%.1f sigma) but the null control did '
                     'not return, so the result is UNCONTROLLED and must not be '
                     'reported as a pass' % sig['z'])
    else:
        v['ab_test_result'] = 'INCOMPLETE'
        v['note'] = 'the signal pair did not both return; nothing to conclude'
    return v


def _warmcold_report(out):
    """Human-readable summary.  States the estimator choice and its measured
    justification every time, so the number cannot be quoted without it."""
    lines = ['[warmcold] primary estimator: %s' % out['primary_estimator'],
             '[warmcold]   secondary %s is reported for continuity with the '
             '2026-09 run but is NOT primary: that run\'s ancilla marginal is '
             'near-uniform (H = 3.67-3.97 of 4.00 bits), so post-selecting '
             'conditions on an uninformative bit and discards ~94%% of shots.'
             % out['secondary_estimator']]
    for k in ('signal|warm', 'signal|cold', 'null|warm', 'null|cold'):
        v = out.get('per_circuit', {}).get(k)
        if not v:
            continue
        lines.append('[warmcold] %-12s s=%-3d %-4s P(data=00000) = %.4f '
                     '[%.4f, %.4f]  ideal %.6f  |  F_s = %s from %d/%d selected'
                     % (k, v['s'], v['arm'], v['P_data_zero'],
                        v['P_data_zero_wilson95'][0],
                        v['P_data_zero_wilson95'][1], v['ideal_P_data_zero'],
                        ('%.3f' % v['F_s']) if v['F_s'] is not None else 'n/a',
                        v['n_selected'], v['n_total']))
    for role in ('signal', 'null'):
        p = out['pairs'].get(role, {})
        if not p.get('complete'):
            lines.append('[warmcold] %-6s INCOMPLETE: missing %s'
                         % (role, p.get('missing')))
            continue
        lines.append('[warmcold] %-6s s=%-3d gap = %+.4f +- %.4f -> %.1f sigma  '
                     '(ideal gap %+.6f; the device transmits w = %s)'
                     % (role, p['s'], p['gap_measured'], p['sigma_gap'], p['z'],
                        p['gap_ideal'],
                        ('%.3f' % p['transmission_w'])
                        if p['transmission_w'] is not None else 'n/a (null)'))
    lay = out.get('layout', {})
    if lay:
        lines.append('[warmcold] layout: anc=%s rev=%s data=%s, DERIVED from the '
                     'MEAS(q->cbit) map (detect_layout said anc=%s rev=%s -> %s)'
                     % (lay['anc_slice'], lay['rev'], lay['data_slice'],
                        lay['detect_layout_anc_slice'], lay['detect_layout_rev'],
                        'agrees' if lay['detect_layout_agrees']
                        else 'DISAGREES; detect_layout scores candidate windows '
                             'against the outcomes it is trying to explain'))
    c9 = out.get('cal9', {})
    if c9.get('available'):
        # ALL nine channels, with the contrast that exposes a dead one.  Reporting
        # only the data five is what let a broken ancilla pass unnoticed on
        # 2026-10-12.
        lines.append('[warmcold] cal9: %d/%d shots, mean readout fidelity %.4f '
                     '(%.4f over healthy channels)'
                     % (c9['n_zero'], c9['n_one'],
                        c9['mean_readout_fidelity'],
                        c9['mean_readout_fidelity_healthy']))
        for lab in sorted(c9['per_qubit'],
                          key=lambda k: (c9['per_qubit'][k]['role'],
                                         c9['per_qubit'][k]['index'])):
            d = c9['per_qubit'][lab]
            lines.append('[warmcold]   %-6s logical q%-2s  e01=%.4f e10=%.4f  '
                         'fid=%.4f  contrast=%+.4f  %s'
                         % (lab, d['logical_qubit'], d['e01'], d['e10'],
                            d['readout_fidelity'], d['contrast'],
                            'BROKEN' if d['broken'] else 'ok'))
        lines.append('[warmcold] cal9 all-zero fractions, SEPARATED by register '
                     '(cal9 key positions data=%s anc=%s -- the SAME positions the '
                     'science circuit uses, because under the vendor key order '
                     'analysed position j is logical qubit j for both): data %.4f, '
                     'ancilla %.4f, joint 9-bit %.4f  (|1...1> circuit: data '
                     '%.4f, ancilla %.4f)'
                     % (c9['cal9_data_positions'], c9['cal9_anc_positions'],
                        c9['all_zero_key_fraction_data'],
                        c9['all_zero_key_fraction_ancilla'],
                        c9['all_zero_key_fraction_zero_circuit'],
                        c9['all_one_key_fraction_data'],
                        c9['all_one_key_fraction_ancilla']))
        s = c9['screen']
        lines.append('[warmcold] cal9 SCREEN: %s -> primary estimator %s, '
                     'post-selected F_s %s'
                     % (s['note'],
                        'VALID' if s['primary_estimator_valid'] else 'INVALID',
                        'valid' if s['secondary_estimator_valid']
                        else 'UNINTERPRETABLE (broken ancilla)'))
        for b in c9['broken_qubits']:
            lines.append('[warmcold]   BROKEN %s = logical q%s (contrast %+.4f, '
                         'e01=%.4f, e10=%.4f): %s'
                         % (b['label'], b['logical_qubit'], b['contrast'],
                            b['e01'], b['e10'],
                            'it is read by the science circuit at analysed key '
                            'position %s, INSIDE the primary estimator\'s data '
                            'window %s, so P(data=00000) is not interpretable'
                            % (b['science_key_pos'], c9['data_slice'])
                            if b['role'] == 'data' else
                            'it enters only the post-selected F_s, so the '
                            'unconditioned primary estimator still stands'))
    elif 'note' in c9:
        lines.append('[warmcold] cal9 UNAVAILABLE: %s' % c9['note'])
    tr = out.get('cal9_transferability', {})
    if tr.get('available'):
        lines.append('[warmcold] cal9 TRANSFERABILITY (%d comparisons, %d sigma '
                     'tolerance): %s'
                     % (tr['n_comparisons'], tr['tol_sigma'],
                        'consistent -- cal9 does characterise the science '
                        'circuits\' channels' if tr['cal9_transferable']
                        else 'VIOLATED %d time(s) -- VENDOR RE-ROUTING DETECTED'
                             % tr['n_violations']))
        for b in tr.get('violations', []):
            lines.append('[warmcold]   %s in %s: observed P(read 0) = %.4f '
                         '(n=%d) lies OUTSIDE the hard range [%.4f, %.4f] that '
                         'cal9 measured for logical q%s'
                         % (b['channel'], b['circuit'], b['observed_P0'], b['n'],
                            b['allowed_P0_range'][0], b['allowed_P0_range'][1],
                            b['logical_qubit']))
        if not tr['cal9_transferable']:
            lines.append('[warmcold]   -> cal9 and the science circuits are NOT '
                         'reading the same physical channels, so cal9 can neither '
                         'validate nor invalidate the primary estimator, and the '
                         'block cannot be screened before spending QPU time')
    hs = out.get('posthoc_healthy_subset', {})
    if hs.get('available') and not hs.get('identical_to_primary'):
        lines.append('[warmcold] POST-HOC subset (NOT pre-registered; excludes '
                     'science key positions %s, keeps %s):'
                     % (hs['excluded_positions'], hs['kept_positions']))
        for role in ('signal', 'null'):
            g = hs['pairs'].get(role, {})
            if not g.get('complete'):
                lines.append('[warmcold]   %-6s INCOMPLETE' % role)
                continue
            lines.append('[warmcold]   %-6s gap = %+.4f +- %.4f -> %+.1f sigma  '
                         '(warm %.4f, cold %.4f)'
                         % (role, g['gap'], g['sigma'], g['z'],
                            g['P_warm'], g['P_cold']))
        lines.append('[warmcold]   %s' % hs['selection_caveat'])
    v = out.get('verdict', {})
    if 'ab_test_result' in v:
        res = v['ab_test_result']
        if res == 'INVALID':
            # 'detected' and 'clean' are conclusions; an INVALID run earned neither,
            # so the summary states the numbers without the verdict words
            lines.append('[warmcold] VERDICT: A/B INVALID -- pre-registered '
                         'estimator gave signal %+.4f (%.1f sigma) and null %+.4f '
                         '(%.1f sigma), but the run does not support either being '
                         'read as physics (see below)'
                         % (v.get('signal_gap', float('nan')),
                            v.get('signal_z', float('nan')),
                            v.get('null_gap', float('nan')),
                            v.get('null_z', float('nan'))))
        else:
            lines.append('[warmcold] VERDICT: A/B %s -- signal %s at %.1f sigma '
                         '(gap %+.4f), null %s at %.1f sigma'
                         % (res,
                            'detected' if v.get('signal_detected')
                            else 'NOT detected',
                            v.get('signal_z', float('nan')),
                            v.get('signal_gap', float('nan')),
                            'clean' if v.get('null_consistent_with_zero')
                            else 'CONTAMINATED', v.get('null_z', float('nan'))))
        lines.append('[warmcold]   %s' % v.get('note'))
        # the null control's own robustness: a null that only passes with all five
        # channels included was never clean, and saying so belongs next to the
        # verdict rather than buried in the subset block above
        hs = out.get('posthoc_healthy_subset', {})
        if hs.get('pairs', {}).get('null', {}).get('complete'):
            nz = hs['pairs']['null']['z']
            if v.get('null_consistent_with_zero') and abs(nz) >= v.get(
                    'z_threshold', 3.0):
                lines.append('[warmcold]   WARNING: the null control is NOT '
                             'ROBUST -- it reads %.1f sigma with all five data '
                             'channels but %+.1f sigma once the suspect channel '
                             'is dropped, so its apparent cleanliness depended on '
                             'a cancellation and cannot be claimed'
                             % (v.get('null_z', float('nan')), nz))
    else:
        lines.append('[warmcold] VERDICT: %s' % v.get('note'))
    return lines


# ---------------------------------------------------------------------
# 10) artifact, figure, and the offline dry run
# ---------------------------------------------------------------------
def load_artifact(path=OUT_JSON):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_artifact(art, path=OUT_JSON):
    art['updated_utc'] = _now()
    with open(path, 'w') as f:
        json.dump(art, f, indent=1, default=str, ensure_ascii=False)
    print('[out] %s' % path, flush=True)
    return path


def stage_record(run):
    """The bookkeeping half of a stage: what was submitted, where, and how many shots
    actually came back -- the last of which was ~92% of the request on WK_C180, and
    is the denominator every statistic in this file uses."""
    return dict(stage=run['stage'], chip_id=run['chip_id'], block=run['block'],
                shots_requested=run['shots_requested'],
                shots_used=run['shots_used'],
                shots_used_total=int(sum(run['shots_used'])),
                shots_expected_total=run['shots_requested'] * len(run['results']),
                shots_return_fraction=(sum(run['shots_used'])
                                       / max(run['shots_requested']
                                             * len(run['results']), 1)),
                wall_s=run['wall_s'], dump_raw=run['dump_raw'],
                n_circuits=len(run['results']),
                instruction_counts=sorted({len(g) for g in run['gates']}),
                attempts=run.get('attempts', []),
                billing=run.get('billing'),
                meta=run['meta'])


ARCHIVE_KEY = 'hardware_run_archive'


def archive_previous(art, stage, new_dump=None, verbose=True):
    """Move a PREVIOUS hardware run's record for `stage` into art[ARCHIVE_KEY]
    instead of overwriting it.

    WHY.  art['warmcold'] currently holds the 2026-10-12 analysis whose verdict is
    INVALID, and main.tex quotes its numbers (signal_z=-0.7314, null_z=+2.8379,
    data3 contrast -0.156).  A new run writes the same key, so without this the
    manuscript would cite an artifact that no longer contains what it cites -- the
    same class of silent evidence loss as overwriting the raw dump, one level up.

    Idempotent by construction: the archive label is the previous record's own dump
    basename, and re-analysing THAT dump is not a new run, so it neither archives
    nor duplicates.  Records with no identifiable dump are left alone rather than
    archived under a made-up label."""
    prev_a, prev_r = art.get(stage), (art.get('runs') or {}).get(stage)
    src = None
    for rec in (prev_a, prev_r):
        if isinstance(rec, dict):
            src = src or rec.get('source_dump') or rec.get('dump_raw')
    if not src:
        return None
    base = os.path.basename(str(src))
    if new_dump and os.path.basename(str(new_dump)) == base:
        return None
    label = '%s@%s' % (stage, base)
    arch = art.setdefault(ARCHIVE_KEY, {})
    if label not in arch:
        arch[label] = dict(archived_utc=_now(), source_dump=str(src),
                           analysis=prev_a, run=prev_r)
        if verbose:
            print('[archive] the previous %s record came from %s and a manuscript '
                  'number may cite it, so it is preserved under %s[%r] rather than '
                  'overwritten' % (stage, base, ARCHIVE_KEY, label), flush=True)
    art.pop(stage, None)
    if isinstance(art.get('runs'), dict):
        art['runs'].pop(stage, None)
    return label


def dry_run(shots=4000, n_boot=200, seed=7, vendor=None, verbose=True):
    """Sample every probe and tomo circuit on the local CPUQVM and push the counts
    through the ENTIRE analysis chain.  Spends no QPU time and needs no account, and
    it is the only way to learn before buying shots whether the artifact assembles.

    The sampled device is noiseless, so every answer must come out at its ideal value
    within Monte-Carlo error -- w = 1, r = 1, L_Z = L_X = 0, each null control at
    zero -- and a deviation beyond 4/sqrt(shots) is a bug in the chain, not a
    property of a device."""
    pg, pm = stage_circuits('probe')
    pres = [q.cpuqvm_sample(g, shots) for g in pg]
    sigma, probe_out = analyze_probe(pres, pm, shots, verbose=verbose)
    tg, tm = stage_circuits('tomo')
    tres = [q.cpuqvm_sample(g, shots) for g in tg]
    tomo_out, lines = analyze_tomo(tres, tm, sigma, probe_out['assignment'], shots,
                                   n_boot=n_boot, seed=seed, vendor=vendor,
                                   verbose=False)
    if verbose:
        for ln in lines:
            print(ln, flush=True)
    tol = 4.0 / math.sqrt(shots)
    chk = {}
    for psi in TOMO_STATES:
        rec = tomo_out['per_input'][psi]
        chk['w_%s' % psi] = abs(rec['w']['mitigated'] - 1.0)
        chk['r_%s' % psi] = abs(rec['radius_mitigated'] - 1.0)
    L = tomo_out['linear_ensemble']['mitigated']
    chk['L_Z'], chk['L_X'], chk['diff'] = abs(L['L_Z']), abs(L['L_X']), abs(L['diff'])
    for k, v in tomo_out['depth_scan'].items():
        if v['null_control']:
            chk['null_%s_r1' % k] = abs(v['sigma_cs_r1_mitigated'])
            chk['null_%s_r2' % k] = abs(v['sigma_cs_r2_mitigated'])
    worst = max(chk.values())
    if verbose:
        if tomo_out['model'].get('vendor_prediction'):
            print('[dry-run] what the quoted vendor fidelities predict for this '
                  'circuit: %s' % json.dumps(tomo_out['model']['vendor_prediction']),
                  flush=True)
        print('[dry-run] noiseless CPUQVM, %d shots/circuit: worst deviation from '
              'the ideal values = %.4f (tolerance %.4f)' % (shots, worst, tol),
              flush=True)
        print('[dry-run] per-quantity: %s'
              % json.dumps({k: round(v, 5) for k, v in chk.items()}), flush=True)
    assert worst < tol, (worst, tol, chk)
    return dict(probe=probe_out, tomo=tomo_out, worst_deviation=worst,
                tolerance=tol, shots=shots, checks=chk)


def warmcold_dry_run(shots=4000, verbose=True):
    """Sample the six warmcold circuits on the local CPUQVM and push the counts
    through the ENTIRE warmcold analysis chain.  Spends no QPU time and needs no
    account, and it is the only way to learn before buying shots whether the A/B
    artifact actually assembles.

    The sampled device is noiseless, so every answer is known in advance and
    exactly: P(data=00000) equals each circuit's own ideal value, the gap appears
    on branch s=5 and nowhere else, and the null branch does not move.  That makes
    this a much sharper check than a tolerance test -- a wrong bit order or a wrong
    data window does not show up as a small numerical error, it shows up as the gap
    on the WRONG branch, because only one of the sixteen has a gap at all.
    `detect_layout` is exercised on real nine-bit keys for the same reason."""
    phi_w, phi_c = load_warmcold_angles()
    gates, meta = stage_circuits('warmcold', phi_w, phi_c)
    results = [q.cpuqvm_sample(g, shots) for g in gates]
    out = analyze_warmcold(results, meta, phi_w, phi_c, shots, verbose=verbose)
    tol = 4.0 / math.sqrt(shots)
    chk = {k: abs(v['P_data_zero'] - v['ideal_P_data_zero'])
           for k, v in out['per_circuit'].items()}
    worst = max(chk.values())
    assert worst < tol, (chk, tol)
    g_sig = out['pairs']['signal']['gap_measured']
    g_nul = out['pairs']['null']['gap_measured']
    assert abs(g_sig - WARMCOLD_GAP_IDEAL_SIGNAL) < tol, (g_sig, tol)
    assert abs(g_nul) < tol, (g_nul, tol)
    # a noiseless local simulator returns every shot; the platform returned ~92% of
    # what was asked for in 2026-09, so this equality is the simulator's ideal and
    # deliberately not something to expect from the device
    for k, v in out['per_circuit'].items():
        assert v['n_total'] == shots, (k, v['n_total'], shots)
    c9 = out['cal9']
    assert c9['available'], c9
    assert abs(c9['mean_readout_fidelity'] - 1.0) < tol, c9
    assert abs(c9['all_zero_key_fraction_zero_circuit'] - 1.0) < tol, c9
    # on ideal data the pre-registered verdict must fire as a PASS: signal found in
    # the predicted direction, null clean.  If this fails, the verdict logic and
    # not the device is what is broken.
    assert out['verdict']['ab_test_passed'] is True, out['verdict']
    if verbose:
        print('[warmcold-dry-run] noiseless chain OK: worst |P(data=00000) - '
              'ideal| = %.2e over %d circuits (tol %.1e), signal gap %+.6f, null '
              'gap %+.6f, verdict PASSED'
              % (worst, len(chk), tol, g_sig, g_nul), flush=True)
    return out


def suppl_dry_run(shots=2000, verbose=True):
    """Sample the suppl grid on the local CPUQVM and push it through analyze_suppl.

    Spends no QPU time and needs no account, and it is the last check before money
    is committed: it proves the artifact assembles, that the layout each circuit
    derives reads its own counts back correctly, and that the pre-declared verdict
    rule fires on data whose answer is known in advance.

    On a noiseless device the two LATE circuits are deterministic -- every shot
    returns the correct syndrome, so n_sel == n_tot and F_s equals 1 within
    sampling error -- which is asserted strictly.  The two MID circuits are sampled
    and RECORDED but not asserted, and that asymmetry is deliberate rather than an
    omission: whether mid-circuit measurement returns anything at all on this
    platform is precisely what S2 exists to measure, the 2026-09 answer was "0 of
    369", and a local simulator that models mid-circuit measurement ideally cannot
    predict a vendor implementation that does not.  Asserting a number there would
    be asserting the simulator's opinion about the device."""
    phi_cold = load_angle_set('cold')
    gates, meta = stage_circuits('suppl', phi_cold=phi_cold)
    results = [q.cpuqvm_sample(g, shots) for g in gates]
    out = analyze_suppl(results, meta, phi_cold, shots, verbose=verbose)
    tol = 4.0 / math.sqrt(shots)
    for lab, m in zip(('C', 'D', 'A', 'B'), meta):
        c = out['per_circuit'][lab]
        if m['measure'] != 'late':
            if verbose:
                print('[suppl-dry-run] %s (mid) sampled, NOT asserted: n_sel=%d/%d '
                      'F_s=%s -- the platform\'s mid-circuit behaviour is what S2 '
                      'measures and is not predictable offline'
                      % (lab, c['n_selected'], c['n_total'], c['F_s']), flush=True)
            continue
        assert c['n_total'] == shots, (lab, c['n_total'], shots)
        assert c['n_selected'] == shots, (
            '%s: a noiseless circuit must return the correct syndrome on every '
            'shot, got %d/%d -- so the window this circuit derived does not read '
            'its own counts' % (lab, c['n_selected'], shots))
        assert abs(c['F_s'] - 1.0) < tol, (lab, c['F_s'], tol)
        assert abs(c['P_expected_key'] - c['ideal_P_data_zero']) < tol, lab
    # On ideal data F_s = 1.0, so the pre-declared rule must return REPRODUCED.
    # If this fails the rule is broken, not the device.
    assert out['comparison']['verdict'] == 'REPRODUCED', out['comparison']
    if verbose:
        print('[suppl-dry-run] noiseless chain OK: both late circuits return all '
              '%d shots on the correct syndrome at F_s=1.000, the verdict rule '
              'fires REPRODUCED, and all four derive anc=[5,9] rev=True'
              % shots, flush=True)
    return out


def warmcold_power_study(shots=None, w=WARMCOLD_W_MEASURED_2026_09,
                         n_draws=400, seeds=None, verbose=True):
    """Repeat binomial draws of the A/B at the OUTCOME level and report the
    distribution of the z-score the real analysis would produce.

    This checks the error propagation and the power claim in `warmcold_power`; it
    is NOT a device simulation.  It draws n_data_zero from Binomial(shots, p) at
    the p that model predicts and pushes each pair through the SAME
    `_warmcold_pair` the hardware counts go through, so the sigma formula is
    validated against the sampling distribution it claims to describe, and the null
    branch's false-positive rate is measured rather than quoted.  Spends no QPU
    time.  What it cannot tell you is whether the real device's w is 0.12 -- only
    the hardware run can, which is why the verdict is a pass/fail on the MEASURED z
    and never on agreement with this study."""
    shots = shots or STAGE_SHOTS['warmcold']
    seeds = seeds or (SEED_BANK[0],)
    pred = warmcold_power(w=w, shots=shots)
    zs, gaps, null_zs = [], [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        for _ in range(n_draws):
            per = {}
            for role, gap_ideal in (('signal', WARMCOLD_GAP_IDEAL_SIGNAL),
                                    ('null', WARMCOLD_GAP_IDEAL_NULL)):
                for arm in ('warm', 'cold'):
                    if role == 'signal':
                        p = (pred['p_warm'] if arm == 'warm' else pred['p_cold'])
                        ideal = 1.0 if arm == 'warm' else 1.0 - gap_ideal
                    else:
                        p = pred['p_warm']      # both arms are warm-ideal here
                        ideal = 1.0
                    k = int(rng.binomial(shots, p))
                    per['%s|%s' % (role, arm)] = dict(
                        s=(WARMCOLD_SIGNAL_S if role == 'signal'
                           else WARMCOLD_NULL_S),
                        frame=(WARMCOLD_SIGNAL_FRAME if role == 'signal'
                               else WARMCOLD_NULL_FRAME),
                        P_data_zero=k / shots, n_total=shots, F_s=None,
                        selected_fraction=None, ideal_P_data_zero=ideal)
                pr = _warmcold_pair(per['%s|warm' % role],
                                    per['%s|cold' % role], gap_ideal)
                (zs if role == 'signal' else null_zs).append(pr['z'])
                if role == 'signal':
                    gaps.append(pr['gap_measured'])
    zs, gaps = np.array(zs, float), np.array(gaps, float)
    null_zs = np.array(null_zs, float)
    out = dict(shots=shots, w=w, n_draws=int(n_draws * len(seeds)),
               seeds=list(seeds), prediction=pred,
               z_mean=float(zs.mean()), z_sd=float(zs.std()),
               z_p05=float(np.percentile(zs, 5)),
               fraction_z_above_3=float((zs >= 3.0).mean()),
               fraction_z_above_5=float((zs >= 5.0).mean()),
               gap_mean=float(gaps.mean()), gap_sd=float(gaps.std()),
               null_z_sd=float(null_zs.std()),
               null_fraction_abs_z_above_3=float((np.abs(null_zs) >= 3.0).mean()))
    if verbose:
        print('[warmcold-power] %d draws at %d shots, w=%.4f: predicted gap '
              '%.4f / sigma_pair %.4f / %.1f sigma'
              % (out['n_draws'], shots, w, pred['observable_gap'],
                 pred['sigma_pair'], pred['sigma_z']), flush=True)
        print('[warmcold-power]   sampled z = %.1f +- %.1f (5th pct %.1f); '
              'P(z>=3) = %.3f, P(z>=5) = %.3f'
              % (out['z_mean'], out['z_sd'], out['z_p05'],
                 out['fraction_z_above_3'], out['fraction_z_above_5']), flush=True)
        print('[warmcold-power]   sampled gap = %.4f +- %.4f against the predicted '
              '%.4f; the NULL branch exceeds 3 sigma on %.4f of draws (a '
              'well-behaved statistic gives ~0.0027)'
              % (out['gap_mean'], out['gap_sd'], pred['observable_gap'],
                 out['null_fraction_abs_z_above_3']), flush=True)
    return out


def _to_observed(probs_canonical, sigma):
    """Re-key a canonical distribution (char b = qubit b) into the observed order
    under the layout hypothesis sigma (observed char j came from qubit sigma[j])."""
    out = {}
    for key, p in probs_canonical.items():
        obs = ''.join(key[sigma[j]] for j in range(len(key)))
        out[obs] = out.get(obs, 0.0) + p
    return out


def _sample(probs, shots, rng, e01=None, e10=None):
    n = len(next(iter(probs)))
    v = np.zeros(2 ** n)
    for key, p in probs.items():
        v[str_to_idx(key)] = p
    if e01 is not None:
        v = assignment_matrix(e01, e10, n) @ v
    v = np.clip(v, 0, None)
    v = v / v.sum()
    return dict(zip((idx_to_str(i, n) for i in range(2 ** n)),
                    (int(c) for c in rng.multinomial(shots, v))))


def simulate_device(vendor, shots=4000, seed=20260921, n_boot=400,
                    sigma_true=(4, 3, 2, 1, 0), verbose=True):
    """POWER STUDY: what this design would return on a device with the quoted vendor
    fidelities at this shot count -- computed BEFORE any shots are bought.

    Ground truth is the model's own (w, r) at the vendor rates; the synthetic counts
    are that model's output distribution pushed through a readout channel and
    sampled multinomially; the recovered numbers then go through exactly the analysis
    chain the real data will use, including the layout detection, the assignment
    matrix built from the synthetic probe, and the bootstrap.

    What this buys is not a prediction of the device but a prediction of the ERROR
    BARS, which is the only part of the plan the budget can actually settle: it says
    whether the shots on offer resolve r well enough to price the over-read, and it
    says so in the same units the artifact will report."""
    f1, f2, fro = vendor
    e_ro = 1.0 - fro
    e01 = [e_ro] * NQ5
    e10 = [e_ro] * NQ5
    rng = np.random.default_rng(seed)
    truth = {}
    pg, pm = stage_circuits('probe')
    pres = []
    for g, m in zip(pg, pm):
        pres.append(_sample(_to_observed(exact_probs(g), sigma_true), shots, rng,
                            e01, e10))
    sigma, probe_out = analyze_probe(pres, pm, shots, verbose=False)
    assert tuple(sigma) == tuple(sigma_true), (sigma, sigma_true)
    tg, tm = stage_circuits('tomo')
    tres = []
    for g, m in zip(tg, tm):
        # `g` already contains prep(psi), so psi_vec must stay at |00000>: passing
        # prep_vec() here as well applies the prep twice, which silently turns |+>
        # into |0> and every ensemble average into nonsense.  The self-tests caught
        # exactly this in test (e); the same trap is guarded here by construction.
        probs = model_output_probs(g, 1.0 - f1, 1.0 - f2)
        tres.append(_sample(_to_observed(probs, sigma_true), shots, rng, e01, e10))
        if m.get('kind') != 'tomo':
            continue          # the job's own calibration circuits, no truth to record
        d = model_mixedness(g, p1=1.0 - f1, p2=1.0 - f2)
        truth['%s|%s|r%d' % (m['psi'], m['basis'], m['rounds'])] = dict(
            w=d['codespace_weight'], r=d['radius'],
            sigma_cs=d['bloch'][2], sigma_z_uncond=d['sigma_z_uncond'])
    # Use the calibration the tomo job CARRIES, not the probe's, so the power study
    # exercises the same path analyze_stage takes on real counts (QCLOUD_SETUP.md 8.2:
    # a submission has one block, so a job must not assume the canary's block is
    # still its own).
    zs = [r for r, m in zip(tres, tm)
          if m.get('kind') == 'cal' and m.get('cal') == 'zero']
    os_ = [r for r, m in zip(tres, tm)
           if m.get('kind') == 'cal' and m.get('cal') == 'one']
    assert len(zs) == 2 and len(os_) == 1, (len(zs), len(os_))
    tomo_cal = assignment_from_cal(zs[0], os_[0], NQ5)
    out, lines = analyze_tomo(tres, tm, sigma, tomo_cal, shots,
                              n_boot=n_boot, seed=seed, vendor=vendor, verbose=False)
    if verbose:
        print('[simulate] device model: f1q=%.4f f2q=%.4f fread=%.4f, %d shots per '
              'circuit, layout %s recovered exactly' % (f1, f2, fro, shots,
                                                        list(sigma)), flush=True)
        for ln in lines:
            print(ln, flush=True)
        print('[simulate] ground truth vs recovered (mitigated):')
        for psi in TOMO_STATES:
            t = truth['%s|Z|r1' % psi]
            rec = out['per_input'][psi]
            b = rec['bootstrap']['mitigated']
            print('   |%s>_L  w: true %.4f  recovered %.4f   r: true %.4f  '
                  'recovered %.4f [%.4f, %.4f]  raw %.4f'
                  % (psi, t['w'], rec['w']['mitigated'], t['r'],
                     rec['radius_mitigated'], b['radius_ci95'][0],
                     b['radius_ci95'][1], rec['radius_raw']), flush=True)
        d0 = out['per_input']['0']['bootstrap']['mitigated']
        half = 0.5 * (d0['radius_ci95'][1] - d0['radius_ci95'][0])
        print('[simulate] resolution: 95%% CI half-width on r = %.4f at %d shots; '
              '%d shots would be needed for +-0.02'
              % (half, shots,
                 int(shots * (half / 0.02) ** 2) if half > 0 else shots), flush=True)
        L = out['linear_ensemble']['mitigated']
        print('[simulate] M2 achievable: |L_Z - L_X| bounded by +-%.4f (95%%) while '
              'the terms span %.3f, i.e. %.1f%% of the span'
              % (1.96 * L['stderr'], L['term_span'],
                 100 * L['ci_over_span']), flush=True)
    return dict(vendor=list(vendor), shots=shots, seed=seed, truth=truth,
                tomo=out, probe=probe_out, sigma=list(sigma))


def seed_scan(vendor, shots=8000, seeds=(20260921, 11, 202, 3003, 40004, 777777),
              verbose=True):
    """Repeat the power study over several sampling seeds and report the DISTRIBUTION
    of the two test statistics, not one draw of them.

    This exists because M2's null hypothesis is "L_Z - L_X = 0", and a single draw
    lands beyond 2 sigma about 5% of the time.  Without the scan, one unlucky draw
    would look like a systematic violation of an identity that selftest (e) proves
    to 5e-17; with it, the reported z-scores can be seen to have mean ~0 and rms ~1,
    which is what "consistent with exact linearity, limited by shots" means.  The
    same scan gives the seed-to-seed scatter of r, an independent check that the
    bootstrap CI is the right width."""
    zs, rs, ws = [], [], []
    for sd in seeds:
        s = simulate_device(vendor, shots=shots, seed=sd, n_boot=1, verbose=False)
        L = s['tomo']['linear_ensemble']['mitigated']
        zs.append(L['diff'] / L['stderr'] if L['stderr'] > 0 else float('nan'))
        rs.append(s['tomo']['per_input']['0']['radius_mitigated'])
        ws.append(s['tomo']['per_input']['0']['w']['mitigated'])
        if verbose:
            print('  seed %-9d L_Z=%+.5f L_X=%+.5f diff=%+.5f +-%.5f = %+5.2f '
                  'sigma   r(|0>)=%.4f  w=%.4f'
                  % (sd, L['L_Z'], L['L_X'], L['diff'], L['stderr'], zs[-1],
                     rs[-1], ws[-1]), flush=True)
    zs = np.array(zs, dtype=float)
    out = dict(vendor=list(vendor), shots=shots, seeds=list(seeds),
               m2_zscores=[float(z) for z in zs],
               m2_z_mean=float(np.mean(zs)), m2_z_rms=float(np.sqrt(np.mean(zs ** 2))),
               m2_z_max_abs=float(np.max(np.abs(zs))),
               r_scatter_ddof1=float(np.std(rs, ddof=1)),
               r_values=[float(x) for x in rs],
               w_scatter_ddof1=float(np.std(ws, ddof=1)),
               interpretation=(
                   'mean ~0 and rms ~1 means the residual L_Z - L_X is sampling '
                   'noise on an identity that holds exactly, not a systematic; the '
                   'r scatter is an independent estimate of the same error the '
                   'bootstrap reports, so the two agreeing is what licenses the '
                   'quoted CI'))
    if verbose:
        print('[seed-scan] M2 z-score over %d seeds: mean %+.2f  rms %.2f  '
              'max|z| %.2f   (pure sampling noise => mean ~0, rms ~1)'
              % (len(seeds), out['m2_z_mean'], out['m2_z_rms'],
                 out['m2_z_max_abs']), flush=True)
        print('[seed-scan] r(|0>) scatter %.4f, w scatter %.4f over the same seeds'
              % (out['r_scatter_ddof1'], out['w_scatter_ddof1']), flush=True)
    return out


def make_figure(art, tag):
    """Four panels: the measured radius against the model-implied one, the code-space
    weight at both depths, the M2 ensemble equality, and the branch-summed point."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(ROOT, 'figures'), exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    tomo = art.get('tomo')
    # a job cut off inside its first chunk records calibration only and has no
    # per_input at all (analyze_stage); there is then nothing to draw in three of
    # the four panels, which is not the same as the stage being absent
    if tomo and tomo.get('per_input'):
        pi = tomo['per_input']
        # A truncated run returns fewer inputs than the grid defines.  Draw what was
        # actually measured and say so in the title; never draw four bars where two
        # exist, and never crash -- the figure is produced from the same partial
        # artifact the numbers come from.
        states = tomo.get('inputs_complete') or [p for p in TOMO_STATES if p in pi]
        n_miss = len(TOMO_STATES) - len(states)
        suffix = ('' if not n_miss else ' (%d of %d inputs missing)'
                  % (n_miss, len(TOMO_STATES)))
        xs = np.arange(len(states))
        a = ax[0][0]
        for i, var in enumerate(('raw', 'mitigated')):
            vals = [pi[p]['radius_' + var] for p in states]
            ci = [pi[p]['bootstrap'][var]['radius_ci95'] for p in states]
            # clamp: a bootstrap CI endpoint can land a few ulps on the wrong side
            # of the point estimate when the CI is degenerate (a noiseless dry run,
            # or a circuit whose counts are deterministic), and matplotlib rejects a
            # negative yerr.  A negative error bar is float noise, not information.
            err = np.array([[max(0.0, v - lo) for v, (lo, _h) in zip(vals, ci)],
                            [max(0.0, hi - v) for v, (_lo, hi) in zip(vals, ci)]])
            a.errorbar(xs + 0.18 * (i * 2 - 1), vals, yerr=err, fmt='os'[i],
                       capsize=3, label='r %s' % var)
        try:
            mb = json.load(open(os.path.join(ROOT, 'mixed_input_boundary.json')))
            a.axhline(mb['device_prep']['radius_max'], ls='--', color='gray',
                      label='model-implied r, 36 gates (ED 10d)')
        except Exception:
            pass
        if tomo['model'].get('rounds1', {}).get('p_instr_fitted') is not None:
            a.axhline(tomo['model']['rounds1']['r_predicted'], ls=':', color='crimson',
                      label='model at the fitted rate, matched depth')
        a.set_xticks(xs)
        a.set_xticklabels(['|%s>L' % p for p in states])
        a.set_ylim(0, 1.05)
        a.set_ylabel('Bloch radius r')
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
        a.set_title('M1 measured device mixedness' + suffix)
        b = ax[0][1]
        b.bar(xs - 0.18, [pi[p]['w']['mitigated'] for p in states], 0.34,
              label='w, depth 1 (78 instr)')
        ds = tomo['depth_scan']
        # the depth-2 circuits are the LAST chunk of the stage, so they are exactly
        # what a budget cut-off removes; draw depth 1 alone instead of crashing
        w2 = [(k.split('|')[0], ds[k].get('w_r2')) for k in ('0|Z', '+|X')
              if k in ds and ds[k].get('w_r2') is not None]
        w2 = [(p, v) for p, v in w2 if p in states]
        if w2:
            b.bar(xs[[states.index(p) for p, _ in w2]] + 0.18,
                  [v for _, v in w2], 0.34, label='w, depth 2 (150 instr)')
        b.set_xticks(xs)
        b.set_xticklabels(['|%s>L' % p for p in states])
        b.set_ylabel('code-space weight w')
        b.legend(fontsize=7)
        b.grid(alpha=0.3, axis='y')
        b.set_title('M5 code-space weight vs depth' + suffix)
        c = ax[1][0]
        lin = tomo['linear_ensemble']
        if lin.get('available', True):
            for i, var in enumerate(('raw', 'mitigated')):
                L = lin[var]
                c.bar(np.arange(4) + 0.18 * (i * 2 - 1),
                      [L['terms'][p] for p in TOMO_STATES], 0.34,
                      label='terms %s' % var)
            L = lin['mitigated']
            c.axhline(L['L_Z'], color='navy', ls='-', label='L_Z = %.4f' % L['L_Z'])
            c.axhline(L['L_X'], color='crimson', ls='--',
                      label='L_X = %.4f' % L['L_X'])
            c.set_xticks(np.arange(4))
            c.set_xticklabels(['<Z_L> |%s>' % p for p in TOMO_STATES])
            c.set_ylabel('unconditional <Z_L>')
            c.legend(fontsize=7)
            c.grid(alpha=0.3, axis='y')
            c.set_title('M2 two pure ensembles of I/2: L_Z must equal L_X')
        else:
            c.text(0.5, 0.5, 'M2 UNAVAILABLE\n\n%s' % lin.get('note', ''),
                   ha='center', va='center', transform=c.transAxes, fontsize=8)
            c.set_title('M2 linear exactness')
            c.set_axis_off()
    d = ax[1][1]
    bench = (art.get('bench') or {}).get('per_state')
    if bench:
        ks = sorted(bench)
        d.bar(np.arange(len(ks)) - 0.18, [bench[k]['F_ideal_same_frame'] for k in ks],
              0.34, label='ideal protocol (same frame)')
        err = np.array([[bench[k]['F_hw'] - bench[k]['F_hw_wilson95'][0] for k in ks],
                        [bench[k]['F_hw_wilson95'][1] - bench[k]['F_hw'] for k in ks]])
        d.bar(np.arange(len(ks)) + 0.18, [bench[k]['F_hw'] for k in ks], 0.34,
              yerr=err, capsize=3, label='hardware, 16 branches summed')
        d.set_xticks(np.arange(len(ks)))
        d.set_xticklabels(['|%s> p=0' % k for k in ks])
        d.set_ylabel('recovery fidelity F')
        d.legend(fontsize=7)
        d.grid(alpha=0.3, axis='y')
        d.set_title('full-branch benchmark point')
    else:
        d.text(0.5, 0.5, 'bench stage not run', ha='center', va='center',
               transform=d.transAxes)
        d.set_axis_off()
    fig.tight_layout()
    path = os.path.join(ROOT, 'figures', 'fig_hw_mixedness_%s.png' % tag)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('[out] %s' % path, flush=True)
    return path


# ---------------------------------------------------------------------
# 11) stage plumbing shared by --mode cloud and --mode analyze, and the CLI
# ---------------------------------------------------------------------
def analyze_stage(stage, results, meta, shots_requested, art, phi=None,
                  vendor=None, n_boot=2000, verbose=True, phi_cold=None,
                  source_dump=None):
    """Push one stage's counts through the analysis and merge the result into `art`.

    The tomo stage depends on the probe stage for the bit order and the assignment
    matrix.  If no probe is available the layout is recovered from the tomography
    circuits themselves -- which determines the logical position and leaves the four
    code-space positions tied, provably harmlessly (selftest (f)) -- and mitigation
    is skipped, with the fallback recorded in the artifact rather than hidden.

    `source_dump` is the file these counts came from.  It is recorded on the result
    and used by archive_previous to tell a NEW run from a re-analysis of the same
    run, so re-running --mode analyze is idempotent while a genuinely new dump
    preserves the record a manuscript number may already cite."""
    archive_previous(art, 'tomo' if stage == 'ladder' else stage,
                     new_dump=source_dump, verbose=verbose)
    if stage == 'probe':
        sigma, out = analyze_probe(results, meta, shots_requested, verbose=verbose)
        art['probe'] = dict(analysis=out, shots_requested=shots_requested,
                            n_circuits=len(results), source_dump=source_dump)
        art.setdefault('meta', {})['layout_sigma'] = list(sigma)
        return out
    if stage in ('tomo', 'ladder'):
        pr = art.get('probe', {}).get('analysis')
        # the bit order needs the fingerprint circuits, which only the probe stage
        # submits; the assignment matrix does not, and the tomo stage carries its
        # own three calibration circuits so that it does not have to assume the
        # probe's block is still this job's block.
        own = [m.get('cal') for m in meta if m.get('kind') == 'cal']
        own_cal = None
        if own.count('zero') >= 2 and own.count('one') >= 1:
            zs = [r for r, m in zip(results, meta)
                  if m.get('kind') == 'cal' and m.get('cal') == 'zero']
            os_ = [r for r, m in zip(results, meta)
                   if m.get('kind') == 'cal' and m.get('cal') == 'one']
            own_cal = assignment_from_cal(zs[0], os_[0], NQ5)
        if not any(m.get('kind') == 'tomo' for m in meta):
            # The allocation ran out inside the FIRST chunk, so what came back is
            # calibration and nothing else.  That is still a real measurement --
            # the readout assignment matrix of the block the platform chose.  It is
            # NOT made redundant by the vendor's ChipInfo figures of merit even
            # though that endpoint has answered since 2026-10-11 (QCLOUD_SETUP.md
            # 8.8.1): the self-measured matrix carries this block's drift and the
            # platform's shots shortfall, neither of which is in the published
            # calibration.  So record it and stop with a message that says what is
            # missing.  Going on would ask detect_layout_ml to infer a bit order
            # from ZERO tomography circuits and surface its "120 tied hypotheses"
            # error, which reads like a builder bug rather than what it is: an
            # empty measurement.
            msg = ('this job returned only its %d calibration circuit(s) and no '
                   'tomography, so there is no radius, no M2 and no depth curve to '
                   'report -- the allocation ran out inside the first chunk'
                   % len(results))
            print('[%s] NOT ANALYSABLE: %s' % (stage, msg), flush=True)
            art['tomo'] = dict(assignment=own_cal, n_circuits=len(results),
                               n_tomo_circuits=0, inputs_complete=[],
                               inputs_incomplete=list(TOMO_STATES),
                               source_dump=source_dump, not_analysable=msg)
            if own_cal is not None:
                print('[%s] readout assignment matrix recovered anyway: e01=%s '
                      'e10=%s' % (stage, [round(x, 4) for x in own_cal['e01']],
                                  [round(x, 4) for x in own_cal['e10']]), flush=True)
            return art['tomo']

        if pr:
            sigma = tuple(pr['layout_sigma'])
            lay_src = 'probe stage'
        else:
            tm = [m for m in meta if m.get('kind') == 'tomo']
            tr = [r for r, m in zip(results, meta) if m.get('kind') == 'tomo']
            ideals = [exact_probs(build_tomo(m['psi'], m['basis'], m['rounds']))
                      for m in tm]
            sigma, ev = detect_layout_ml(tr, ideals, verbose=verbose)
            lay_src = ('the tomography circuits alone (the four code-space positions '
                       'are tied and readout mitigation is applied per observed '
                       'position, which selftest (f) proves is harmless)')
            art.setdefault('meta', {})['layout_evidence'] = ev
        if own_cal is not None:
            cal, source = own_cal, 'this job\'s own calibration circuits'
        elif pr:
            cal, source = pr['assignment'], 'probe stage'
        else:
            cal = dict(e01=[0.0] * NQ5, e10=[0.0] * NQ5, n_zero=0, n_one=0)
            source = ('none available: readout mitigation is skipped, so every '
                      'mitigated number equals its raw value')
        if pr and own_cal is not None:
            d = max(abs(a - b) for a, b in zip(pr['assignment']['e01'], own_cal['e01']))
            source += ' (max |e01 - probe e01| = %.4f)' % d
        source += '; layout from ' + lay_src
        out, lines = analyze_tomo(results, meta, sigma, cal, shots_requested,
                                  n_boot=n_boot, vendor=vendor, verbose=False)
        out['calibration_source'] = source
        out['source_dump'] = source_dump
        art['tomo'] = out
        art.setdefault('meta', {})['layout_sigma'] = list(sigma)
        if verbose:
            for ln in lines:
                print(ln, flush=True)
        return out
    if stage == 'bench':
        assert phi is not None
        out = analyze_bench(results, meta, phi, shots_requested, verbose=verbose)
        out['source_dump'] = source_dump
        art['bench'] = out
        return out
    if stage == 'warmcold':
        assert phi is not None and phi_cold is not None, (
            'the warmcold stage needs BOTH angle sets to compare; refusing to '
            'analyse an A/B with only one arm defined')
        out = analyze_warmcold(results, meta, phi, phi_cold, shots_requested,
                               verbose=verbose)
        out['source_dump'] = source_dump
        art['warmcold'] = out
        return out
    if stage == 'suppl':
        assert phi_cold is not None, (
            'the suppl stage reproduces the 2026-09 cold-angle feasibility set; '
            'refusing to analyse it without the cold angle file it claims to use')
        out = analyze_suppl(results, meta, phi_cold, shots_requested,
                            verbose=verbose)
        out['source_dump'] = source_dump
        # WHICH BLOCK this was measured on, taken from the submission-time sidecar
        # rather than from the environment: two suppl results can differ by nothing
        # but the block (S5 is a block contrast), and without this the live entry
        # and its archived twin are distinguishable only by dump filename.  Also
        # records the truncation here rather than only in truncated_stages, so an
        # entry read on its own says how many of its own circuits came back.
        side = read_meta_sidecar(source_dump) if source_dump else None
        if side:
            n_sub = int(side.get('n_circuits') or 0)
            out['chip_id'] = side.get('chip_id')
            out['block'] = side.get('block')
            out['submitted_utc'] = side.get('written_utc')
            out['n_circuits_submitted'] = n_sub
            out['n_circuits_received'] = len(results)
            out['truncated'] = bool(n_sub and len(results) < n_sub)
        entries = art.setdefault('suppl', [])
        # suppl is the one LIST-valued stage, so archive_previous's "same dump means
        # no-op" rule is not enough here: appending unconditionally made a second
        # --mode analyze of the SAME dump leave two identical entries, contradicting
        # the idempotence this function's docstring promises.  Replace in place when
        # the dump basename matches, append when it is genuinely a new dump -- which
        # is what lets two blocks (S5's contrast) coexist without duplicating one.
        base = os.path.basename(str(source_dump)) if source_dump else None
        if base:
            for i, e in enumerate(entries):
                if os.path.basename(str(e.get('source_dump') or '')) == base:
                    entries[i] = out
                    return out
        entries.append(out)
        return out
    raise ValueError(stage)


def _load_dump(stage, path=None):
    """Counts from a dump file: the stage's frozen legacy dump, or `path`."""
    path = path or os.path.join(ROOT, STAGE_RAW[stage])
    with open(path) as f:
        raw = json.load(f)
    return [{str(k): int(v) for k, v in d.items()} for d in raw]


def _meta_sidecar_path(dump_path):
    return dump_path + '.meta.json'


def write_meta_sidecar(dump_path, meta, gates, stage, chip_id=None, block=None,
                       shots=None, mapping=True, use_options=True):
    """Record WHICH meta a dump's counts belong to, next to the dump.

    `--mode analyze` rebuilds meta from the CURRENT builders.  The existing guard
    only compares LENGTHS, so a builder change that reorders circuits without
    changing their number -- moving the calibration pair from last to first, say
    -- would silently re-label every circuit in an older dump, and an analysis
    would then compare the wrong arms or read a calibration as a science circuit.
    The sidecar makes the pairing explicit and checkable.

    It is written BEFORE submission, not after, so that a mid-job cut-off still
    leaves meta on disk describing the prefix that submit_cloud dumped.

    It is also the single choke point that refuses to clobber frozen evidence:
    every paid submission calls this before it calls submit_cloud, and submit_cloud
    is what opens `dump_path` with 'w'."""
    assert_writable_dump(dump_path)
    path = _meta_sidecar_path(dump_path)
    with open(path, 'w') as f:
        json.dump(dict(
            stage=stage, chip_id=chip_id, block=block, shots=shots,
            # the submission configuration is part of what the data MEANS: a run
            # with vendor mapping on cannot be compared to one with it off, and
            # the 2026-10-12 artifact would have been interpretable at a glance if
            # it had carried this field
            mapping=mapping, use_options=use_options,
            written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            n_circuits=len(meta),
            instruction_counts=[len(g) for g in gates],
            kinds=[m.get('kind') for m in meta],
            meta=meta), f, indent=1, sort_keys=True)
    return path


def read_meta_sidecar(dump_path):
    """The sidecar written at submission time, or None for a legacy dump."""
    path = _meta_sidecar_path(dump_path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def print_plan(phi, budget_s=BUDGET_S, vendor=None):
    """Print the executable run plan of QCLOUD_SETUP.md 8, with the circuit counts
    and instruction totals computed from the builders rather than typed in, so the
    table cannot drift away from the code it describes.  Touches nothing."""
    ms_per_shot = MS_PER_SHOT   # CONFIRMED: see the MS_PER_SHOT comment
    rows, tot_shots, tot_cost = [], 0, 0.0
    shots_budget = int(budget_s * 1000 / ms_per_shot)
    print('=' * 78)
    print('RUN PLAN -- OriginQ Wukong WK_C180_2, one physical block per job')
    print('COST MODEL CONFIRMED by the account holder (2026-10), so the old ~50x')
    print('two-model hedge is closed and no canary job is needed to settle it:')
    print('  * QPU time is billed in REAL TIME, while the job runs')
    print('  * exhausting it immediately disables the account')
    print('  * a job CANNOT overrun its allocation -- it is cut off')
    print('  => metered per shot: %.0f ms/shot x %.0f s = ~%d shots for EVERYTHING'
          % (ms_per_shot, budget_s, shots_budget))
    print('  %.0f ms/shot over a 204-instruction circuit is ~68 us per instruction,'
          % ms_per_shot)
    print('  ~2000x a superconducting gate time, so the bill is per-shot overhead')
    print('  (readout, reset, control dead time) and FLAT in circuit length:')
    print('  shortening the circuits would not buy shots.  Only shots is a knob.')
    print('  "Cannot overrun" makes a MID-JOB CUT-OFF the expected outcome rather')
    print('  than a rare one, so the stage is chunked and ordered to be salvaged.')
    print('=' * 78)
    for stage in ('probe', 'tomo', 'bench', 'ladder', 'warmcold'):
        if stage == 'bench':
            gates, meta = stage_circuits(stage, phi)
        elif stage == 'warmcold':
            _w, _c = load_warmcold_angles()
            gates, meta = stage_circuits(stage, _w, _c)
        else:
            gates, meta = stage_circuits(stage)
        instr = [len(g) for g in gates]
        cnot = [sum(1 for x in g if x[0] == 'CNOT') for g in gates]
        shots = STAGE_SHOTS[stage]
        n_shots = shots * len(gates)
        cost = n_shots * ms_per_shot / 1000.0
        tot_shots += n_shots
        tot_cost += cost
        depths = sorted({m['rounds'] for m in meta if 'rounds' in m})
        rows.append((STAGE_JOB[stage], stage, STAGE_NQ[stage], len(gates),
                     min(instr), max(instr), sum(cnot), shots, n_shots, cost,
                     depths))
    hdr = ('job stage   nq  circ  instr/circuit   CNOT   shots/circ  total shots  '
           'billed cost     depths')
    print(hdr)
    print('-' * len(hdr))
    for (job, stage, nq, nc, imin, imax, ct, sh, tot, cost, depths) in rows:
        print('%3d  %-7s %2d  %4d  %5d-%-5d %7d %10d %12d %11.0f s   %s'
              % (job, stage, nq, nc, imin, imax, ct, sh, tot, cost,
                 depths or '-'))
    print('-' * len(hdr))
    print('    %-7s      %4d                            %12d %11.0f s'
          % ('TOTAL', sum(r[3] for r in rows), tot_shots, tot_cost))
    print()
    print('THE SHOTS / COST / PRECISION TRADE-OFF FOR THE tomo STAGE (19 circuits).')
    print('Kept because tomo stays implemented, but note what the right-hand')
    print('column says: on this block\'s MEASURED calibration the best sigma(r<1)')
    print('that fits the budget is ~1.8, so this knob has no good setting.  The')
    print('recommended warmcold job has its own power table further down.')
    print('"sigma(r<1)" assumes the vendor-')
    print('predicted r~0.86 and CI = 0.028 x sqrt(8000/S), the scaling measured by')
    print('the power study; the cost is S x 19 x %.0f ms.' % ms_per_shot)
    print('  shots/circ  total shots          cost  CI half-width  sigma(r<1)')
    for s_ in (8000, 4000, 2000, 1500, 1000, 750, 500):
        n_ = s_ * 19
        ci = 0.028 * math.sqrt(8000.0 / s_)
        c_ = n_ * ms_per_shot / 1000.0
        if s_ == STAGE_SHOTS['tomo']:
            mark = '   <- DEFAULT, %.0f s spare' % (budget_s - c_)
        elif c_ > budget_s:
            mark = '   (over: cut off)'
        else:
            mark = ''
        print('  %10d  %11d  %10.0f s  %13.3f  %10.1f%s'
              % (s_, n_, c_, ci, 0.14 / ci, mark))
    tomo_cost = [r[9] for r in rows if r[1] == 'tomo'][0]
    print()
    print('WHAT FITS IN THE ~%.0f s THAT IS LEFT -- each line is that stage ALONE'
          % budget_s)
    print('measured against the WHOLE allocation.  They are mutually exclusive,')
    print('not cumulative: tomo by itself consumes %.0f of the ~%.0f s, so'
          % (tomo_cost, budget_s))
    print('whichever stage runs, it is the only one.')
    for (_j, stage, _nq, nc, _im, _ix, _ct, sh, tot, cost, _d) in rows:
        if cost <= budget_s:
            v = 'fits ALONE, %.0f s spare' % (budget_s - cost)
        else:
            v = 'DOES NOT FIT (%.1fx over)' % (cost / budget_s)
        print('  %-7s %2d circ x %5d shots = %7d shots = %6.0f s   %s'
              % (stage, nc, sh, tot, cost, v))
    wc_row = [r for r in rows if r[1] == 'warmcold'][0]
    wc_cost = wc_row[9]
    wc_shots_tot = wc_row[8]
    wc_pow = warmcold_power()
    wc_pow_p = warmcold_power(w=WARMCOLD_W_PESSIMISTIC)
    wc_pow_10 = warmcold_power(w=WARMCOLD_W_MEASURED_2026_10_12)
    # recomputed rather than quoted: the Z-basis vanishing of the effect is what
    # makes psi='+' load-bearing, and a number this load-bearing should not be a
    # literal in a print statement
    _wcw, _wcc = load_warmcold_angles()
    wc_z_gap = max(abs(ideal_data_zero_prob(p, WARMCOLD_SIGNAL_FRAME,
                                            WARMCOLD_SIGNAL_S, _wcw)
                       - ideal_data_zero_prob(p, WARMCOLD_SIGNAL_FRAME,
                                              WARMCOLD_SIGNAL_S, _wcc))
                   for p in ('0', '1'))
    ci_tomo = CI_HALF_WIDTH_AT_1000 * (
        1000.0 / (STAGE_SHOTS['tomo'] * MEASURED_FREAD ** 4)) ** 0.5
    r_tomo = vendor_prediction(MEASURED_F1Q, MEASURED_F2Q,
                               MEASURED_FREAD)['radius']
    z_tomo = (1.0 - r_tomo) / ci_tomo
    # same arithmetic as selftest (k), which asserts wall5 = 18651 s = 62x budget;
    # recomputing it here rather than quoting 62 keeps the two from drifting
    s5_tomo = (1000.0 * (CI_HALF_WIDTH_AT_1000 * 5.0 / (1.0 - r_tomo)) ** 2
               / MEASURED_FREAD ** 4)
    wall5_tomo = (len(stage_circuits('tomo')[1]) * s5_tomo
                  * MS_PER_SHOT / 1000.0)
    print('  => warmcold IS THE CHOICE, and tomo is not, for a reason that is')
    print('     arithmetic rather than preference.  BOTH fit the budget.  But')
    print('     tomo\'s headline r<1 is worth only %.2f sigma on this block\'s'
          % z_tomo)
    print('     MEASURED calibration (f2q=%.5f, better than the quoted %.4f):'
          % (MEASURED_F2Q, QUOTED_F2Q))
    print('     the device beats its own datasheet, so %.0f s buys a NULL'
          % tomo_cost)
    print('     result, and 5 sigma would need %.0f s = %.0fx the allocation.'
          % (wall5_tomo, wall5_tomo / budget_s))
    print('     warmcold spends %.0f s on a warm-minus-cold gap that is exactly'
          % wc_cost)
    print('     %.6f in the noiseless limit on branch s=%d, and that survives'
          % (WARMCOLD_GAP_IDEAL_SIGNAL, WARMCOLD_SIGNAL_S))
    print('     the data-register survival w -- BUT w IS NOT ONE NUMBER, and an')
    print('     earlier version of this plan quoted a single sigma as if it were.')
    print('     Two committed measurements exist, on different chips AND with')
    print('     different compiler options, and they straddle the detection bar:')
    for _lab, _pz, _w in (
            ('2026-09 WK_C180   no QCloudOptions', WARMCOLD_P_DATA_ZERO_2026_09,
             WARMCOLD_W_MEASURED_2026_09),
            ('2026-10 WK_C180_2 with options    ',
             WARMCOLD_P_DATA_ZERO_2026_10_12, WARMCOLD_W_MEASURED_2026_10_12)):
        _pw = warmcold_power(w=_w, shots=STAGE_SHOTS['warmcold'])
        print('       %s  P(data=00000)=%.4f -> w=%.4f  gap %.4f +- %.4f = %5.1f'
              ' sigma' % (_lab, _pz, _w, _pw['observable_gap'], _pw['sigma_pair'],
                          _pw['sigma_z']))
    print('     Which world a rerun lands in is not computable offline, and the')
    print('     difference is a DETECTION versus a BOUND.  So the plan is three')
    print('     gated submissions rather than one:')
    print('       1. --cal9-screen %6d shots %6.1f s  is the block usable at all?'
          % (2 * CAL9_SCREEN_SHOTS, cost_s(2 * CAL9_SCREEN_SHOTS)))
    print('       2. --wc-probe    %6d shots %6.1f s  how large is w, and what does'
          % (WC_PROBE_SHOTS, cost_s(WC_PROBE_SHOTS)))
    print('          the remaining budget buy with it?  It prints the shot ladder,')
    print('          the z each rung is worth, the w bound a null would report, and')
    print('          the largest rung S* that fits --budget-remaining.')
    print('       3. --stage warmcold --cal-first --shots S*.')
    print('     Every step requires --budget-remaining (the dashboard figure; the')
    print('     API cannot answer it) and refuses to overrun it, and the dashboard')
    print('     delta across a step is the BILLED ms/shot, which is what should then')
    print('     replace the pessimistic %.0f every cost above is priced at.  At'
          % MS_PER_SHOT)
    print('     equal cost a claim beats a null, and a characterised bound beats an')
    print('     uncharacterised one either way.  tomo stays implemented for a top-up.')
    print()
    print('WARMCOLD SALVAGE LADDER -- chunk=%d, so a cut-off after any boundary'
          % STAGE_CHUNK['warmcold'])
    print('below still leaves a reportable result, and the decisive pair is')
    print('submitted FIRST because the primary estimator needs no calibration:')
    _gw, mw = stage_circuits('warmcold', _wcw, _wcc)
    for b in list(range(STAGE_CHUNK['warmcold'], len(mw),
                        STAGE_CHUNK['warmcold'])) + [len(mw)]:
        pre = mw[:b]
        roles = {(m['role'], m['arm']) for m in pre
                 if m.get('kind') == 'warmcold'}
        have_sig = {('signal', 'warm'), ('signal', 'cold')} <= roles
        have_nul = {('null', 'warm'), ('null', 'cold')} <= roles
        cal_ok = len([m for m in pre if m.get('kind') == 'cal9']) == 2
        got = []
        if have_sig:
            got.append('the SIGNAL A/B (headline, %.1f sigma at w=%.4f but only '
                       '%.1f at w=%.4f -- see the two-world table above)'
                       % (wc_pow['sigma_z'], WARMCOLD_W_MEASURED_2026_09,
                          wc_pow_10['sigma_z'], WARMCOLD_W_MEASURED_2026_10_12))
        if have_nul:
            got.append('the NULL control (without it the result is UNCONTROLLED)')
        if cal_ok:
            got.append('the readout decomposition of w')
        print('  after %d circuits (%5d shots, %3.0f s): %s'
              % (b, b * STAGE_SHOTS['warmcold'],
                 b * STAGE_SHOTS['warmcold'] * MS_PER_SHOT / 1000.0,
                 ' + '.join(got) if got else 'nothing reportable'))
    print()
    print('SALVAGE LADDER (8.3) -- the platform enforces the ceiling itself, so')
    print('there are no wall-clock abort rules left to guess at.  What matters is')
    print('that a cut-off is not a total loss: tomo is submitted in chunks of')
    print('%d circuits, submit_cloud writes the raw counts to disk after every'
          % STAGE_CHUNK['tomo'])
    print('chunk, and the circuits are ordered cal-first, then depth-1 grouped by')
    print('input state, then depth-2.  So every boundary below is analysable on')
    print('its own, and each one adds a complete scientific increment:')
    ck = STAGE_CHUNK['tomo']
    _gt, mt = stage_circuits('tomo')
    sh_t = STAGE_SHOTS['tomo']
    for b in list(range(ck, len(mt), ck)) + [len(mt)]:
        pre = mt[:b]
        have = {(m['psi'], m['basis']) for m in pre
                if m['kind'] == 'tomo' and m['rounds'] == 1}
        d1 = [p for p in TOMO_STATES if all((p, B) in have for B in BASES)]
        got = []
        if [m['cal'] for m in pre if m['kind'] == 'cal'] == ['zero', 'one', 'zero']:
            got.append('assignment matrix')
        if d1:
            got.append('M1 for |%s>_L' % '>,|'.join(d1))
        if '0' in d1:
            got.append('M3 at depth 1')
        if len(d1) == len(TOMO_STATES):
            got.append('M2')
        if any(m['kind'] == 'tomo' and m['rounds'] == 2 for m in pre):
            got.append('M3 at depth 2 / M5')
        print('  %2d/%d circuits (%3.0f s billed)  inputs %-16s -> %s'
              % (b, len(mt), b * sh_t * ms_per_shot / 1000.0,
                 ('|%s>' % '>,|'.join(d1)) if d1 else '-',
                 '; '.join(got) or 'nothing analysable yet'))
    print('  A cut-off therefore costs at most the chunk in flight (%d circuits ='
          % ck)
    print('  %.0f s), and the dump already on disk still analyses.'
          % (ck * sh_t * ms_per_shot / 1000.0))
    print('  --mode analyze accepts that truncated prefix, reports exactly which')
    print('  inputs and depths are missing, and refuses to present a subset as')
    print('  if it were the complete claim.')
    print()
    # The significance of r<1 on the MEASURED calibration of the vendor's own best
    # block, recomputed here rather than quoted, so the plan and selftest (k) cannot
    # drift apart.  The quoted table said 0.9831 for f2q; the device measures
    # 0.99275, which makes the state purer and the headline claim weaker.
    n_eff_frac = MEASURED_FREAD ** 4
    ci_tomo = CI_HALF_WIDTH_AT_1000 * (
        1000.0 / (STAGE_SHOTS['tomo'] * n_eff_frac)) ** 0.5
    r_meas = vendor_prediction(MEASURED_F1Q, MEASURED_F2Q, MEASURED_FREAD)['radius']
    r_quot = vendor_prediction(QUOTED_F1Q, QUOTED_F2Q, QUOTED_FREAD)['radius']
    z_meas = (1.0 - r_meas) / ci_tomo
    z_quot = (1.0 - r_quot) / ci_tomo
    s5 = (1000.0 * (CI_HALF_WIDTH_AT_1000 * 5.0 / (1.0 - r_meas)) ** 2
          / n_eff_frac)
    wall5 = len(stage_circuits('tomo')[1]) * s5 * MS_PER_SHOT / 1000.0
    print('WHAT THE RECOMMENDED JOB BUYS -- warmcold, %d circuits x %d shots,'
          % (WARMCOLD_N_SCIENCE + WARMCOLD_N_CAL, STAGE_SHOTS['warmcold']))
    print('%d shots = %.0f s of the ~%.0f s allocation (%.0f s margin):'
          % (wc_shots_tot, wc_cost, budget_s, budget_s - wc_cost))
    print('  W1  THE HEADLINE.  Warm-started vs cold-started variational recovery')
    print('      on the one branch where they differ, s=%d (frame %s, psi=%s).'
          % (WARMCOLD_SIGNAL_S, WARMCOLD_SIGNAL_FRAME, WARMCOLD_PSI))
    print('      The noiseless gap is exactly %.6f and it is a LOGICAL Z, so it'
          % WARMCOLD_GAP_IDEAL_SIGNAL)
    print('      is invisible in the Z basis (gap %.1e) and maximal in X/Y.'
          % wc_z_gap)
    print('      Through the data survival w the gap is w*%.3f, and w is the ONE'
          % WARMCOLD_GAP_IDEAL_SIGNAL)
    print('      input this plan cannot compute offline.  The two committed')
    print('      measurements bracket it: w=%.4f (2026-09 WK_C180, no vendor'
          % WARMCOLD_W_MEASURED_2026_09)
    print('      options) gives %.4f +- %.4f = %.1f sigma, while w=%.4f'
          % (wc_pow['observable_gap'], wc_pow['sigma_pair'], wc_pow['sigma_z'],
             WARMCOLD_W_MEASURED_2026_10_12))
    print('      (2026-10-12 WK_C180_2, WITH options) gives %.4f +- %.4f = %.1f'
          % (wc_pow_10['observable_gap'], wc_pow_10['sigma_pair'],
             wc_pow_10['sigma_z']))
    print('      sigma; at half the larger w it is still %.1f sigma.  --wc-probe'
          % wc_pow_p['sigma_z'])
    print('      settles which world today is for %.0f s, before six circuits are'
          % cost_s(WC_PROBE_SHOTS))
    print('      bought.')
    print('  W2  A MATCHED NULL.  s=%d (frame %s) has an ideal warm/cold gap of'
          % (WARMCOLD_NULL_S, WARMCOLD_NULL_FRAME))
    print('      %.1e, so it must come out at zero.  A signal with a dirty null'
          % WARMCOLD_GAP_IDEAL_NULL)
    print('      is a systematic, not a result; the verdict requires both.')
    print('  W3  The transmission factor w = gap_hw / gap_ideal, i.e. what')
    print('      FRACTION of the logical advantage this device preserves.  This')
    print('      is the number that connects the simulation to the hardware and')
    print('      it is measurable even though the absolute fidelity is not.')
    print('  W4  Free inside the same job: the block\'s own 9-qubit readout')
    print('      assignment (2 cal circuits), which decomposes w into a readout')
    print('      part and a 206-instruction circuit part; the platform\'s shots')
    print('      shortfall; and the bit order recovered from the data.')
    print()
    print('WHAT tomo WOULD BUY INSTEAD (8.4), at %d shots/circuit -- kept for'
          % STAGE_SHOTS['tomo'])
    print('comparison, and NOT recommended: its headline is a %.2f sigma null.'
          % z_meas)
    print('  M1  the headline: measured w, Bloch radius r and purity for four')
    print('      logical inputs, with a bootstrap CI of +-%.4f at this shot count'
          % ci_tomo)
    print('      (n_eff %.0f after code-space post-selection keeps fread^4=%.3f).'
          % (STAGE_SHOTS['tomo'] * n_eff_frac, n_eff_frac))
    print('      The 2026-10-11 preflight MEASURED the vendor block %s at'
          % VENDOR_BLOCK_WK_C180_2)
    print('      f2q=%.5f, better than the quoted %.4f, so the predicted radius'
          % (MEASURED_F2Q, QUOTED_F2Q))
    print('      is %.4f not %.4f and r<1 is %.2f sigma, NOT the %.2f sigma the'
          % (r_meas, r_quot, z_meas, z_quot))
    print('      quoted table implied.  5 sigma would need %.0f shots/circuit ='
          % s5)
    print('      %.0f s = %.0fx this %.0f s budget, so on this block the mixedness'
          % (wall5, wall5 / budget_s, budget_s))
    print('      headline is NOT purchasable: the device is better than its')
    print('      datasheet, and that -- not the budget -- is the binding')
    print('      constraint.  Do NOT "fix" this by ranking blocks on significance:')
    print('      the worst block on the chip gives 6.3 sigma (see 8.8, selftest k).')
    print('      What it still buys is replacing the paper\'s 4-circuit x 400-shot')
    print('      feasibility set with a real 19-circuit tomography.')
    print('  M2  the linear-fraction ensemble equality L_Z = L_X, to +-0.060')
    print('      (0.023 at 8000 shots) against a term span of ~1.43.  This is a')
    print('      NULL test: its precision depends on shots alone, so unlike M1 and')
    print('      M3 it is not weakened by the device being good.')
    print('  M3  the isotropic depolarizing surrogate fitted at the hardware')
    print('      depth and then asked to predict r with nothing left to choose.')
    print('      The power study, built on the QUOTED vendor table, predicts it')
    print('      FAILS here at -3.4 sigma (depth 1) and -3.9 sigma (depth 2); at')
    print('      1000 shots that becomes roughly -1.2 and -1.4 sigma.  On the')
    print('      MEASURED calibration of 8.8.3 this device is closer to isotropic')
    print('      depolarizing than the table said, so M3 may lose its')
    print('      discriminating power altogether.  Report whichever the data gives,')
    print('      not whichever the prediction promised.')
    print('  M5  two null controls plus the depth-1 -> depth-2 radius ratio.')
    print('  Also, for free inside the same job: the 5x5 readout assignment matrix')
    print('      of the block the platform chose.  ChipInfo has answered since')
    print('      2026-10-11 and publishes per-qubit readout fidelities, but this')
    print('      matrix is NOT redundant with them -- it carries this block\'s drift')
    print('      and the platform\'s shots shortfall, which the published')
    print('      calibration does not.  Also free: the platform\'s shots shortfall')
    print('      and the bit order recovered from the data.')
    print()
    print('DEFERRED, NOT ABANDONED -- these stay implemented for a top-up:')
    print('  tomo    266 s, and it FITS, but its headline r<1 is a %.2f sigma'
          % z_meas)
    print('          null on this block\'s measured calibration and 5 sigma needs')
    print('          %.0fx the whole allocation.  Run it only if the point is to'
          % (wall5 / budget_s))
    print('          report the device\'s mixedness as consistent with unity.')
    print('  probe   redundant: tomo self-calibrates and self-detects its layout.')
    print('  bench   952 s, upgrades the feasibility set to a 16-branch protocol')
    print('          benchmark and re-tests the documented mid/late A/B anomaly.')
    print('  ladder  224 s, rounds=%s -> a radius-vs-depth curve, i.e. a'
          % (LADDER_ROUNDS,))
    print('          per-round logical error rate against the threshold behaviour.')
    print()
    print('NOT PURSUED AT ANY BUDGET: the over-read delta^T C delta (~5e7 shots)')
    print('and any significance claim for "4.4% over identity" (~2 sigma at best).')
    if vendor:
        vp = vendor_prediction(*vendor)
        print()
        print('what the quoted vendor fidelities predict for the depth-1 circuit:')
        for k in ('two_qubit_survival', 'codespace_weight', 'radius',
                  'radius_as_measured'):
            print('  %-24s %.6f' % (k, vp[k]))
    print('=' * 78)
    return dict(rows=[dict(zip(('job', 'stage', 'nq', 'circuits', 'instr_min',
                                'instr_max', 'cnot_total', 'shots_per_circuit',
                                'shots_total', 'pessimistic_cost_s', 'depths'), r))
                      for r in rows],
                shots_total=tot_shots, pessimistic_cost_s=tot_cost,
                budget_s=budget_s, ms_per_shot_assumed=ms_per_shot)


def main(argv=None):
    # declared first: LADDER_ROUNDS is read by the --rounds help string below and
    # rebound from it, and Python requires the global declaration to precede any
    # use of the name in the function body
    global LADDER_ROUNDS
    ap = argparse.ArgumentParser(
        description='measured device mixedness for the mixed-input boundary',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', default='verify',
                    choices=['verify', 'sample', 'selftest', 'simulate', 'preflight',
                             'plan', 'wcpick', 'cloud', 'analyze'])
    ap.add_argument('--stage', default='all',
                    choices=['probe', 'tomo', 'bench', 'ladder', 'warmcold',
                             'suppl', 'all'])
    ap.add_argument('--rounds', default=None,
                    help='comma-separated extra depths for the ladder stage '
                         '(default %s; 8 costs 581 instructions/288 CNOT per circuit)'
                         % (LADDER_ROUNDS,))
    ap.add_argument('--chip-id', default='WK_C180_2')
    ap.add_argument('--chip-ids', default=None,
                    help='comma-separated chips for --mode preflight '
                         '(default: --chip-id)')
    ap.add_argument('--token', default=os.environ.get('ORIGINQ_TOKEN', ''))
    ap.add_argument('--shots', type=int, default=None,
                    help='override the per-stage default %s' % STAGE_SHOTS)
    ap.add_argument('--block', default=None,
                    help='comma-separated physical qubit block (default: the '
                         'top-ranked best_qubit_blocks(n) on the chip, free query)')
    ap.add_argument('--chunk', type=int, default=None,
                    help='circuits per job (default: the whole stage in ONE job)')
    ap.add_argument('--job-timeout', type=int, default=21600)
    ap.add_argument('--no-options', action='store_true')
    ap.add_argument('--no-mapping', action='store_true',
                    help='disable the vendor\'s server-side RE-ROUTING while '
                         'keeping the block pinned, so the logical-to-physical '
                         'assignment is identical by construction for every '
                         'circuit of the job.  This is a CONTROL, not a '
                         'demonstrated necessity: the 2026-10-12 warmcold run had '
                         'mapping ON and its in-job cal9 still transferred to the '
                         'science circuits (all 36 channel comparisons inside '
                         'their hard P(read 0) bounds), so this platform shows no '
                         'evidence of per-circuit re-routing.  Use it when you '
                         'want the assignment guaranteed rather than measured.  '
                         'Requires --block; without a pinned block the server '
                         'still chooses qubits and disabling mapping buys '
                         'nothing.')
    ap.add_argument('--cal-first', action='store_true',
                    help='warmcold only: submit the two nine-qubit readout '
                         'calibration circuits FIRST and force chunk=%d, so '
                         'submission 1 is exactly the calibration.  cal9\'s '
                         'channel labels DO transfer to the science circuits (the '
                         'contrary finding is retracted in '
                         'check_cal9_transferability), so this is what lets a '
                         'cut-off job still leave the block\'s readout screen on '
                         'disk -- the 2026-10-12 run put calibration last and a '
                         'dead data channel went unnoticed until after all four '
                         'science circuits were paid for.  NOTE it does not make '
                         'the calibration cheap: billing is per shot, not per '
                         'instruction.  A cheap screen is cal9 alone as its own '
                         'job at ~1000 shots/circuit.' % WARMCOLD_N_CAL)
    ap.add_argument('--cal9-screen', action='store_true',
                    help='SCREEN ONLY: submit just the two nine-qubit readout '
                         'calibration circuits (%d shots/circuit by default = %d '
                         'shots = %.0f s at the confirmed %.0f ms/shot) and report '
                         'whether the block is usable, then STOP.  No science '
                         'circuit is submitted, so this is the only path with a '
                         'real decision point between the screen and the spend.  '
                         'Picks the best ranked block that is not the one rejected '
                         'on 2026-10-12 (dead data channel q3, contrast -0.156) '
                         'unless --block is given.  Exit code 0 if the block is '
                         'usable, 4 if not, so it can gate a script.  Safe at both '
                         'ends of the unverified billing rate; a full warmcold '
                         'stage is not (84 s against a possibly ~48 s remainder).'
                         % (CAL9_SCREEN_SHOTS, 2 * CAL9_SCREEN_SHOTS,
                            2 * CAL9_SCREEN_SHOTS * MS_PER_SHOT / 1000.0,
                            MS_PER_SHOT))
    ap.add_argument('--allow-rejected-block', action='store_true',
                    help='with --cal9-screen: screen the block the 2026-10-12 run '
                         'was rejected on instead of refusing it.  Needed because '
                         'best_qubit_blocks(9) on WK_C180_2 returns exactly ONE '
                         'block, [39,48,49,57,58,66,67,75,85], and its record '
                         'carries no per-qubit quality data, so there is neither an '
                         'alternative block nor any free way to pre-screen one.  '
                         'The screen then answers a DIFFERENT question from block '
                         'selection: is the dead data channel (logical q3, physical '
                         '85, contrast -0.156) persistent, or was it transient?  '
                         'Both answers are useful -- healthy means the A/B can be '
                         'rerun on this block, still dead means a 9-qubit '
                         'warm-vs-cold run is not purchasable on this chip at any '
                         'shot count.  Deliberately not the default: it spends real '
                         'money on a block this file declares unusable.')
    ap.add_argument('--wc-probe', action='store_true',
                    help='buy ONE science circuit -- the stage\'s signal|warm entry, '
                         'byte-identical to the pre-registered one -- at %d shots '
                         '(%.1f s at the pessimistic %.0f ms/shot) and measure the '
                         'data-register survival w that sets the A/B\'s power, '
                         'BEFORE the six-circuit stage is bought.  Reports the '
                         'pipeline verdict, w with its 95%% interval, a ladder of '
                         'shot counts with the z each would buy and the w bound a '
                         'null would report, and the largest rung that fits '
                         '--budget-remaining.  Exit 0 if a detection is affordable, '
                         '5 if the A/B would only set a bound, 4 if the pipeline '
                         'failed.  Makes no warm-versus-cold claim: one arm cannot '
                         'form a gap.'
                         % (WC_PROBE_SHOTS, cost_s(WC_PROBE_SHOTS), MS_PER_SHOT))
    ap.add_argument('--budget-remaining', type=float, default=None,
                    help='seconds of QPU time the account dashboard says are LEFT.  '
                         'REQUIRED for every paid submission (--cal9-screen, '
                         '--wc-probe, --mode cloud): the remaining allocation '
                         'cannot be queried from the API, so this is the only way '
                         'the hard budget gate in check_budget can work, and a job '
                         'that overruns is killed and returns nothing.  Also read '
                         'the dashboard again AFTER each submission -- the delta '
                         'over a known shot count is the billed ms/shot, which is '
                         'what --ms-per-shot should then be set to.')
    ap.add_argument('--ms-per-shot', type=float, default=None,
                    help='billed milliseconds per shot, measured from a dashboard '
                         'delta across a known shot count.  Default: the pessimistic '
                         'constant MS_PER_SHOT=%.0f, which is NOT reproducible from '
                         'anything on disk (the 2026-09 wall clock implies ~1.0).  '
                         'Passing a measured value lets the probe\'s ladder price the '
                         'A/B at the real rate instead of the worst case.'
                         % MS_PER_SHOT)
    ap.add_argument('--probe-json', action='append', default=None,
                    help='a --wc-probe verdict sidecar (the .probe.json written next '
                         'to that probe\'s raw counts).  Repeat it once per probe.  '
                         'Used by --mode wcpick, which applies the pre-registered '
                         'chip rule of QCLOUD_SETUP.md 9.7 to the probes that came '
                         'back and prints the chip, the block, the S* the current '
                         '--budget-remaining buys and the command that spends it.  '
                         'Free: it submits nothing and needs no token.')
    ap.add_argument('--dump-raw', default=None,
                    help='raw-counts file.  For a PAID mode this is where the counts '
                         'are WRITTEN (default: a new timestamped file, because the '
                         'legacy names in STAGE_RAW are frozen evidence holding the '
                         'counts the manuscript\'s 2026-09/10-12/10-13 numbers come '
                         'from).  For --mode analyze it is the dump to READ, which is '
                         'how a new run is analysed without touching the frozen one; '
                         'it requires a single --stage, not all.')
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--seeds', type=int, default=1,
                    help='for --mode simulate: repeat the power study over this '
                         'many sampling seeds and report the distribution of the '
                         'M2 z-score and of r, so one unlucky draw is not mistaken '
                         'for a systematic')
    ap.add_argument('--noise', default='depolarizing')
    ap.add_argument('--vendor', default=None,
                    help='f1q,f2q,fread -- record what these aggregate vendor '
                         'fidelities predict for this circuit through the same model')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--json', default=OUT_JSON)
    ap.add_argument('--no-figure', action='store_true')
    ap.add_argument('--figure', action='store_true',
                    help='for --mode simulate only: also write the figure. Off by '
                         'default so a power-study plot cannot be mistaken for '
                         'hardware data sitting in the same figures/ directory')
    a = ap.parse_args(argv)

    setup()
    vendor = tuple(float(x) for x in a.vendor.split(',')) if a.vendor else None
    if a.rounds:
        LADDER_ROUNDS = tuple(int(x) for x in a.rounds.split(',') if x.strip())
    stages = (['probe', 'tomo', 'bench', 'ladder', 'warmcold']
              if a.stage == 'all' else [a.stage])

    if a.mode == 'selftest':
        st = selftests()
        print('[selftest] PASS %s' % json.dumps(st, default=str), flush=True)
        return 0

    if a.mode == 'verify':
        st = selftests()
        print('\n=== exact ideal reference of every submitted circuit ===')
        for stage in stages:
            if stage == 'warmcold':
                phi, phi_c = load_warmcold_angles()
            else:
                phi = q.load_angles(a.noise) if stage == 'bench' else None
                phi_c = None
            gates, meta = stage_circuits(stage, phi, phi_c)
            inv = {}
            for g in gates:
                key = (len(g), sum(1 for x in g if x[0] == 'CNOT'),
                       sum(1 for x in g if x[0] == 'MEAS'))
                inv[key] = inv.get(key, 0) + 1
            print('  stage %-6s %2d circuits, (instructions, CNOT, MEAS) '
                  'histogram %s' % (stage, len(gates), dict(inv)))
            if stage in ('tomo', 'ladder'):
                for g, m in zip(gates, meta):
                    if m.get('kind') != 'tomo':
                        print('    cal|%s rep%d  ideal: a product of |0> (or |1> '
                              'on every qubit), used only for the assignment matrix'
                              % (m.get('cal'), m.get('rep')))
                        continue
                    w, nvec = ideal_bloch_of(g)
                    print('    %s|%s|r%d  ideal w=%.12f  <Z>rot=%+.12f (want %+.0f)'
                          % (m['psi'], m['basis'], m['rounds'], w, nvec[2],
                             IDEAL_BLOCH[m['psi']][BASIS_AXIS[m['basis']]]))
            if stage == 'warmcold':
                print('    (the A/B is exact and pre-registered: s=%d/%s is the '
                      'ONLY branch where the two angle sets differ, s=%d/%s is '
                      'the matched null)'
                      % (WARMCOLD_SIGNAL_S, WARMCOLD_SIGNAL_FRAME,
                         WARMCOLD_NULL_S, WARMCOLD_NULL_FRAME))
                for _g, m in zip(gates, meta):
                    if m.get('kind') != 'warmcold':
                        print('    cal9|%-5s ideal: all nine bits read %s (the '
                              "block's own readout assignment)"
                              % (m.get('cal'),
                                 '1' if m.get('cal') == 'one' else '0'))
                        continue
                    print('    %-6s|%-4s s=%-3d frame=%s  ideal '
                          'P(data=00000)=%.9f'
                          % (m['role'], m['arm'], m['s'], m['frame'],
                             ideal_data_zero_prob(
                                 m['psi'], m['frame'], m['s'],
                                 phi if m['arm'] == 'warm' else phi_c)))

        print('[verify] self-tests %s' % json.dumps(st, default=str), flush=True)
        return 0

    if a.mode == 'sample':
        if not q.HAS_PYQPANDA3:
            print('ERROR: pyqpanda3 required: %s' % q._PYQPANDA_IMPORT_ERR)
            return 3
        if a.stage == 'warmcold':
            wc = warmcold_dry_run(shots=a.shots or 4000)
            art0 = load_artifact(a.json)
            art0['warmcold_dry_run'] = dict(
                shots=a.shots or 4000, spends_qpu_time=False,
                note='noiseless local CPUQVM through the full analysis chain; '
                     'every value here is an ideal reference, NOT hardware',
                per_circuit={k: {kk: vv for kk, vv in v.items()
                                 if kk != 'top3'}
                             for k, v in wc['per_circuit'].items()},
                pairs=wc['pairs'], cal9=wc['cal9'], verdict=wc['verdict'])
            if a.seeds > 0:
                # always at the stage's real shot count, so the study answers
                # "what will the job that costs 252 s actually achieve" and not
                # "what would a differently-sized job achieve"
                art0['warmcold_power_study'] = warmcold_power_study(
                    shots=STAGE_SHOTS['warmcold'], n_draws=200,
                    seeds=tuple(SEED_BANK[:min(a.seeds, len(SEED_BANK))]))
                art0['warmcold_power_study_pessimistic'] = warmcold_power_study(
                    shots=STAGE_SHOTS['warmcold'],
                    w=WARMCOLD_W_PESSIMISTIC, n_draws=200,
                    seeds=tuple(SEED_BANK[:min(a.seeds, len(SEED_BANK))]))
            save_artifact(art0, a.json)
            return 0
        if a.stage == 'suppl':
            sp = suppl_dry_run(shots=a.shots or 2000)
            art0 = load_artifact(a.json)
            art0['suppl_dry_run'] = dict(
                shots=a.shots or 2000, spends_qpu_time=False,
                note='noiseless local CPUQVM through the full suppl analysis chain; '
                     'every value here is an ideal reference, NOT hardware, and the '
                     'mid-circuit rows are recorded rather than asserted because the '
                     'platform\'s mid behaviour is what S2 measures',
                per_circuit=sp['per_circuit'], comparison=sp['comparison'])
            save_artifact(art0, a.json)
            return 0
        dry_run(shots=a.shots or 4000, vendor=vendor)
        return 0

    if a.mode == 'simulate':
        if vendor is None:
            print('ERROR: --mode simulate needs --vendor f1q,f2q,fread')
            return 2
        sim = simulate_device(vendor, shots=a.shots or STAGE_SHOTS['tomo'],
                              n_boot=a.n_boot)
        art = load_artifact(a.json)
        art['power_study'] = sim
        if a.seeds > 1:
            art['power_study']['seed_scan'] = seed_scan(
                vendor, shots=a.shots or STAGE_SHOTS['tomo'],
                seeds=tuple(SEED_BANK[:a.seeds]))
        art.setdefault('meta', {}).update(dict(
            task='power study for the measured-device-mixedness design',
            mode=a.mode, vendor=vendor,
            shots=a.shots or STAGE_SHOTS['tomo'], spends_qpu_time=False))
        if a.figure:
            # off by default: a power-study figure sitting in figures/ beside the
            # real hardware ones is exactly the kind of thing that gets mistaken
            # for data in a manuscript, so it has to be asked for explicitly
            make_figure(dict(tomo=sim['tomo']),
                        a.tag or 'power_study_%s' % a.chip_id)
        save_artifact(art, a.json)
        return 0

    if a.mode == 'preflight':
        chips = ([c.strip() for c in a.chip_ids.split(',') if c.strip()]
                 if a.chip_ids else [a.chip_id])
        out = preflight(chips, token=a.token or None)
        if vendor:
            out['vendor_prediction'] = vendor_prediction(*vendor)
            print('[preflight] what the quoted vendor fidelities predict: %s'
                  % json.dumps(out['vendor_prediction']), flush=True)
        # A preflight snapshot is evidence about a DAY: QCLOUD_SETUP.md 8.8 quotes
        # the measured calibration of B5 out of the 2026-10-11 file, so today's free
        # queries must land in their own timestamped file rather than overwrite it.
        path = os.path.join(ROOT, 'hw_mixedness_preflight_%s.json' % _utc_stamp())
        assert_writable_dump(path)
        with open(path, 'w') as f:
            json.dump(out, f, indent=1, default=str, ensure_ascii=False)
        print('[out] %s  (the legacy hw_mixedness_preflight.json is FROZEN: it is '
              'the 2026-10-11 snapshot QCLOUD_SETUP.md 8.8 cites)' % path, flush=True)
        return 0

    if a.mode == 'wcpick':
        # FREE, and deliberately outside the --mode cloud branch: it submits nothing
        # and needs no token, so where the last ~200 s goes can be re-derived from
        # the probe sidecars on disk at any time -- including by a reader who does
        # not trust this transcript.  Exit code carries the verdict like the probe's:
        # 0 detection affordable, 5 bound only (a human decision to spend), 4 stop.
        if not a.probe_json:
            print('ERROR: --mode wcpick needs at least one --probe-json, the '
                  '.probe.json sidecar each --wc-probe run wrote next to its raw '
                  'counts (printed as "[wcprobe] verdict sidecar ...").')
            return 2
        probes = []
        for pj in a.probe_json:
            p = pj if os.path.isabs(pj) else os.path.join(ROOT, pj)
            if not os.path.exists(p):
                print('ERROR: no such probe sidecar: %s' % p, flush=True)
                return 2
            with open(p) as f:
                probes.append(json.load(f))
        try:
            pick = pick_wc_chip(probes, a.budget_remaining, a.ms_per_shot)
        except BudgetError as exc:
            print('ERROR: %s' % exc, flush=True)
            return 3
        except ValueError as exc:
            print('ERROR: %s' % exc, flush=True)
            return 2
        if not os.path.isdir(RUN_LOG_DIR):
            os.makedirs(RUN_LOG_DIR)
        # The decision record must never be silently overwritten.  Two wcpick calls
        # inside the same second used to build the SAME filename and the second
        # clobbered the first -- observed on 2026-10-15, where a DETECTION verdict
        # priced at the measured 1.0 ms/shot was replaced on disk by a BOUND verdict
        # priced at the pessimistic 14.0, leaving an evidence file that contradicted
        # the terminal which had just printed it.  So uniqueness is asserted by probing
        # the filesystem rather than assumed from a clock with 1-second resolution --
        # the discipline unique_dump_path already applies to raw counts -- and
        # assert_writable_dump's refusal to land on committed evidence applies here too.
        base_name = '%s_wcpick.json' % _utc_stamp()
        out_path, n = os.path.join(RUN_LOG_DIR, base_name), 1
        while os.path.exists(out_path):
            out_path = os.path.join(RUN_LOG_DIR,
                                    '%s_%d.json' % (base_name[:-5], n))
            n += 1
        out_path = assert_writable_dump(out_path)
        with open(out_path, 'w') as f:
            json.dump(pick, f, indent=1, default=str, ensure_ascii=False)
        print('[out] %s  (the decision record.  It is NOT written into %s: the '
              'artifact holds the measurements the manuscript quotes, not the '
              'routing decision made between two of them.)'
              % (out_path, os.path.basename(a.json)), flush=True)
        return int(pick['exit_code'])

    # ---- cloud / analyze ----
    phi = q.load_angles(a.noise)
    q.PHI_GLOBAL = phi
    # The warmcold A/B compares TWO different committed trainings, so its cold arm
    # comes from its own file and not from --noise.  --noise still selects the
    # warm arm for every other stage; load_warmcold_angles() is what pins the A/B
    # to the two files the manuscript is built from.
    phi_cold = load_angle_set('cold')
    # AND THE TRAP THAT GOES WITH IT: q.load_angles('depolarizing') resolves to
    # vscr_angles_dep.npz, which IS the cold arm.  Passing that `phi` into the
    # warmcold stage would compare cold against cold and return a flawless null
    # that looks exactly like a failed experiment.  The warm arm is therefore
    # loaded separately and by name, and the two are never interchangeable.
    phi_warm = load_angle_set('warm')
    assert not np.allclose(phi_warm, phi_cold), (
        'the warm and cold angle files are identical, so there is no A/B to run')

    if a.mode == 'plan':
        plan = print_plan(phi, vendor=vendor)
        art0 = load_artifact(a.json)
        art0.setdefault('meta', {})['plan'] = plan
        save_artifact(art0, a.json)
        return 0

    art = load_artifact(a.json)
    art.setdefault('meta', {}).update(dict(
        task='measured device mixedness for the mixed-input boundary',
        chip_id=a.chip_id, mode=a.mode, stages=stages, vendor=vendor,
        angles=(os.environ.get('VSCR_ANGLES_FILE')
                or 'vscr_angles_%s.npz'
                % ('dep' if a.noise == 'depolarizing' else a.noise)),
        python=sys.version.split()[0], numpy=np.__version__))

    if a.mode == 'analyze':
        # --dump-raw names ONE file, so it cannot mean "analyse every stage"
        if a.dump_raw and a.stage == 'all':
            print('ERROR: --dump-raw names a single file, so --mode analyze needs a '
                  'single --stage with it.  Without --dump-raw the frozen legacy '
                  'dumps in STAGE_RAW are read and --stage all is fine.', flush=True)
            return 2
        dump_arg = None
        if a.dump_raw:
            dump_arg = (a.dump_raw if os.path.isabs(a.dump_raw)
                        else os.path.join(ROOT, a.dump_raw))
            side0 = read_meta_sidecar(dump_arg)
            if side0 and side0.get('stage') == 'wcprobe':
                # A probe dump is ONE circuit, not a stage prefix, so it has its own
                # analysis path; re-running it is what makes the quoted w
                # reproducible from the file rather than from the run's stdout.
                gates_p, meta_p = stage_circuits('warmcold', phi_warm, phi_cold)
                sel = [i for i, m in enumerate(meta_p)
                       if m.get('kind') == 'warmcold' and m.get('role') == 'signal'
                       and m.get('arm') == 'warm']
                raw = _load_dump('warmcold', dump_arg)
                out = analyze_wc_probe(
                    raw[0], meta_p[sel[0]], gates_p[sel[0]], phi_warm,
                    shots_requested=(side0.get('shots') or WC_PROBE_SHOTS),
                    chip_id=side0.get('chip_id'), block=side0.get('block'),
                    ms_per_shot=a.ms_per_shot,
                    budget_remaining_after=a.budget_remaining, verbose=True)
                out['dump_raw'] = dump_arg
                art['wcprobe'] = out
                save_artifact(art, a.json)
                return out['exit_code']
        loaded = {}
        for stage in stages:
            path = dump_arg or os.path.join(ROOT, STAGE_RAW[stage])
            if not os.path.exists(path):
                print('[analyze] %s: no dump at %s -- skipped (that job was '
                      'dropped by the abort rules, or has not run yet)'
                      % (stage, os.path.basename(path)), flush=True)
                continue
            results = _load_dump(stage, path)
            if stage == 'bench':
                _g, meta = stage_circuits(stage, phi)
            elif stage == 'warmcold':
                # --cal-first REORDERS this stage (the two cal9 circuits move to the
                # front), so the rebuild must be told about it, or it disagrees with
                # the submission-time sidecar and a perfectly good dump is refused.
                # That is exactly what happened to the 2026-10-21 WK_C180_2 run: it
                # was submitted --cal-first, and the recovery command the launcher
                # printed at submission time omitted the flag, so re-analysis died on
                # a meta mismatch.  Honouring the flag does NOT weaken the check
                # below -- a genuinely reordered dump still fails it, and a cal-first
                # dump analysed without the flag still fails it with a message that
                # names the fix.
                _g, meta = stage_circuits(stage, phi_warm, phi_cold,
                                          cal_first=a.cal_first)
            else:
                # phi/phi_cold are passed for every stage, not just warmcold:
                # bench needs phi and suppl needs phi_cold, and a stage that needs
                # neither ignores them.  Calling stage_circuits(stage) bare here
                # made --mode analyze --stage suppl die on its own angle-set assert
                # -- the stage could be bought but its data could not be read back.
                _g, meta = stage_circuits(stage, phi_warm, phi_cold)
            # The dump stores counts only, so its meta has to come from somewhere.
            # Prefer the sidecar written at submission time and REFUSE a mismatch:
            # reconstructing meta from the current builders is only safe while no
            # builder has reordered circuits, and the length check below cannot
            # detect a reordering.
            side = read_meta_sidecar(path)
            if side is None:
                print('[analyze] %s: no meta sidecar -- this dump predates '
                      'sidecar writing, so its meta is RECONSTRUCTED from the '
                      'current builders.  That is only correct if no builder has '
                      'since reordered circuits without changing their number.'
                      % stage, flush=True)
            else:
                sm = side.get('meta') or []
                n_cmp = min(len(sm), len(meta))
                diff = [i for i in range(n_cmp) if sm[i] != meta[i]]
                if diff:
                    i = diff[0]
                    # Name the fix when the mismatch is the cal9/warmcold swap that
                    # --cal-first produces.  "the meta does not match" alone gives the
                    # reader nothing to do about it, and the flag is easy to omit --
                    # the launcher's own printed recovery command once did, on the run
                    # it had just submitted with it.
                    hint = ''
                    if {sm[i].get('kind'), meta[i].get('kind')} == {'cal9',
                                                                   'warmcold'}:
                        hint = ('\n  This is the --cal-first reordering: the dump was '
                                'SUBMITTED with the two cal9 circuits first but is '
                                'being REBUILT without them.  Re-run with --cal-first. '
                                'The sidecar records the order that was actually '
                                'bought, so the rebuild has to match it.')
                    raise AssertionError(
                        '%s: the meta recorded at submission time (%s) does not '
                        'match what the current builders produce, so this dump '
                        'cannot be labelled reliably.  First difference at '
                        'circuit %d:\n  submitted: %r\n  rebuilt:   %r%s'
                        % (stage, side.get('written_utc'), i, sm[i], meta[i], hint))
                meta = sm
                print('[analyze] %s: meta VERIFIED against the submission-time '
                      'sidecar (%s, %d circuits, block %s)'
                      % (stage, side.get('written_utc'), side.get('n_circuits'),
                         side.get('block')), flush=True)
            # A TRUNCATED DUMP IS THE EXPECTED OUTCOME, not an error.  QPU time is
            # billed in real time and a job cannot overrun its allocation, so the
            # job is cut off when the budget runs out; submit_cloud has by then
            # written every chunk that completed, and chunks are submitted in
            # order, so what is on disk is a PREFIX of the stage.  That is exactly
            # why stage_circuits orders tomo so that every chunk boundary is a
            # complete scientific increment.  Accept the prefix, say loudly what is
            # missing, record it in the artifact, and never pad it.
            if len(results) > len(meta):
                raise AssertionError(
                    '%s: the dump holds %d circuits but this version of the '
                    'builders defines %d, so the dump cannot have come from these '
                    'builders -- do not analyse it against the wrong meta'
                    % (stage, len(results), len(meta)))
            if len(results) < len(meta):
                gone = meta[len(results):]

                def _lab(m):
                    # One label per meta schema, not one assumed schema.  This ran
                    # with warmcold's psi/basis/rounds hardcoded and so raised
                    # KeyError: 'basis' the first time a TRUNCATED suppl dump was
                    # analysed -- i.e. the reporting path for "your job was cut off,
                    # here is what is missing" was itself broken for the stage most
                    # likely to be cut off.  Label alone is not enough for suppl:
                    # the same letter appears at two measurement depths.
                    if m.get('kind') == 'cal':
                        return m['cal']
                    if m.get('kind') == 'suppl':
                        return '%s|%s|%s|s%d' % (m['label'], m['measure'],
                                                 m['frame'], m['s'])
                    return '%s|%s|r%d' % (m['psi'], m['basis'], m['rounds'])
                art.setdefault('truncated_stages', {})[stage] = dict(
                    received=len(results), defined=len(meta),
                    missing=[_lab(m) for m in gone])
                print('[analyze] %s: TRUNCATED -- %d of %d circuits returned (the '
                      'job was cut off when the allocation ran out).  Missing, in '
                      'submission order: %s'
                      % (stage, len(results), len(meta), [_lab(m) for m in gone]),
                      flush=True)
                meta = meta[:len(results)]
            print('[analyze] %s: %d circuits from %s'
                  % (stage, len(results), STAGE_RAW[stage]), flush=True)
            loaded[stage] = (results, meta)
        # the ladder job is the SAME estimator at deeper rounds, so its counts are
        # appended to the tomography's and analysed once; analysed alone it would
        # have no depth-1 circuit and analyze_tomo would (correctly) refuse it
        if 'tomo' in loaded and 'ladder' in loaded:
            (rt, mt), (rl, ml) = loaded['tomo'], loaded['ladder']
            loaded['tomo'] = (rt + rl, mt + ml)
            print('[analyze] ladder merged into tomo: %d + %d circuits, depths %s'
                  % (len(rt), len(rl),
                     sorted({m['rounds'] for m in mt + ml})), flush=True)
            loaded.pop('ladder')
        for stage in [s for s in stages if s in loaded]:
            results, meta = loaded[stage]
            # read back what was ACTUALLY requested for this stage rather than
            # assuming the default: the abort rules mean the tomo job may have run
            # at 8000, 3000 or 1500 shots, and the ladder at something else again
            sr = a.shots or art.get('runs', {}).get(stage, {}).get(
                'shots_requested') or STAGE_SHOTS[stage]
            analyze_stage(stage, results, meta, sr, art,
                          phi=phi_warm if stage == 'warmcold' else phi,
                          vendor=vendor, n_boot=a.n_boot, phi_cold=phi_cold,
                          source_dump=path)
    else:
        # --cal9-screen and --wc-probe submit no stage at all (they short-circuit
        # below, before the stage loop), so the one-stage-only rule that protects the
        # allocation from a four-stage sweep does not apply to them and must not
        # swallow them.
        if a.stage == 'all' and not a.cal9_screen and not a.wc_probe:
            print('ERROR: --mode cloud refuses --stage all.  QPU time is billed in '
                  'real time, the allocation is ~%.0f s offline (pass '
                  '--budget-remaining with the dashboard figure: the live one is '
                  'not queryable), and ONE stage consumes it (QCLOUD_SETUP.md 8).  '
                  'Submitting all of them would bill the first, be cut off part way '
                  'through, and leave the account disabled with nothing to show for '
                  'the rest.  Submit ONE stage:\n'
                  '  bash launch_mixedness.sh warmcold  # THE PLAN: %d circuits at '
                  '%d shots = %.0f s -- but the warm/cold gap is worth %.1f sigma '
                  'at the 2026-09 no-options survival w=%.4f and only %.1f sigma at '
                  'the 2026-10 with-options w=%.4f, so run --wc-probe FIRST to find '
                  'out which world today is; chunked so the signal pair alone '
                  'survives a cut-off\n'
                  'only if the allocation is ever topped up:\n'
                  '  bash launch_mixedness.sh tomo      # 266 s, but r<1 is only '
                  '%.2f sigma on this block\'s measured calibration -- it buys a '
                  'null\n'
                  '  bash launch_mixedness.sh probe     # redundant: tomo carries '
                  'its own calibration and recovers its own bit order\n'
                  '  bash launch_mixedness.sh bench     # 952 s at its default '
                  'shots\n'
                  '  bash launch_mixedness.sh ladder    # 224 s at its default '
                  'shots'
                  % (BUDGET_S,
                     WARMCOLD_N_SCIENCE + WARMCOLD_N_CAL,
                     STAGE_SHOTS['warmcold'],
                     (WARMCOLD_N_SCIENCE + WARMCOLD_N_CAL)
                     * STAGE_SHOTS['warmcold'] * MS_PER_SHOT / 1000.0,
                     warmcold_power(w=WARMCOLD_W_MEASURED_2026_09)['sigma_z'],
                     WARMCOLD_W_MEASURED_2026_09,
                     warmcold_power(w=WARMCOLD_W_MEASURED_2026_10_12)['sigma_z'],
                     WARMCOLD_W_MEASURED_2026_10_12,
                     (1.0 - vendor_prediction(MEASURED_F1Q, MEASURED_F2Q,
                                              MEASURED_FREAD)['radius'])
                     / (CI_HALF_WIDTH_AT_1000 * (
                         1000.0 / (STAGE_SHOTS['tomo']
                                   * MEASURED_FREAD ** 4)) ** 0.5)))
            return 2
        tok = a.token
        if not tok:
            import hw_error_budget as heb
            tok = heb.resolve_token()
        if not q.HAS_PYQPANDA3:
            print('ERROR: pyqpanda3 required: %s' % q._PYQPANDA_IMPORT_ERR)
            return 3
        block = ([int(x) for x in a.block.split(',') if x.strip()]
                 if a.block else None)
        excl = () if a.allow_rejected_block else None    # None -> this chip's list
        if a.dump_raw:
            # Refuse a clobbering path BEFORE anything is queried or spent: the
            # legacy dump names are frozen evidence, and finding that out after a
            # block query (or worse, after a submission) is finding it too late.
            try:
                assert_writable_dump(a.dump_raw if os.path.isabs(a.dump_raw)
                                     else os.path.join(ROOT, a.dump_raw))
            except RuntimeError as exc:
                print('ERROR: %s' % exc, flush=True)
                return 3
        if a.cal9_screen:
            # SCREEN ONLY.  Deliberately short-circuits the stage loop: the whole
            # point is that no science circuit is submitted until a human has read
            # this verdict, so --cal9-screen must not be combinable with a stage run
            # in the same process.  Cost is 2 x shots (28 s at the pessimistic rate
            # for the default 1000).
            try:
                scr = run_cal9_screen(tok, a.chip_id,
                                      shots=a.shots or CAL9_SCREEN_SHOTS,
                                      block=block, exclude=excl,
                                      dump_raw=a.dump_raw,
                                      job_timeout=a.job_timeout,
                                      use_options=not a.no_options,
                                      mapping=not a.no_mapping, verbose=True,
                                      budget_remaining=a.budget_remaining,
                                      ms_per_shot=a.ms_per_shot)
            except BudgetError as exc:
                print('ERROR: %s' % exc, flush=True)
                return 3
            record_billing(art, scr.get('billing'))
            art['cal9_screen'] = {k: v for k, v in scr.items()
                                  if k not in ('cal9',)}
            art['cal9_screen']['cal9_per_qubit'] = (
                (scr.get('cal9') or {}).get('per_qubit'))
            save_artifact(art, a.json)
            print('[cal9screen] dump %s (screen only -- no warm-vs-cold claim is '
                  'made or implied).  Read the dashboard again: the delta over %d '
                  'shots is the billed ms/shot to pass as --ms-per-shot next.'
                  % (os.path.basename(scr['dump_raw']),
                     2 * (a.shots or CAL9_SCREEN_SHOTS)), flush=True)
            return 0 if scr.get('block_usable') else 4
        if a.wc_probe:
            # PROBE ONLY, gate 2 of 3.  Short-circuits for the same reason the screen
            # does: the human reads w and the shot ladder before the six-circuit A/B
            # is bought.  Exit code carries the verdict so a launcher can gate on it.
            try:
                pr = run_wc_probe(tok, a.chip_id, phi_warm, phi_cold,
                                  shots=a.shots or WC_PROBE_SHOTS, block=block,
                                  exclude=excl, dump_raw=a.dump_raw,
                                  job_timeout=a.job_timeout,
                                  use_options=not a.no_options,
                                  mapping=not a.no_mapping, verbose=True,
                                  budget_remaining=a.budget_remaining,
                                  ms_per_shot=a.ms_per_shot)
            except BudgetError as exc:
                print('ERROR: %s' % exc, flush=True)
                return 3
            record_billing(art, pr.get('billing'))
            art['wcprobe'] = {k: v for k, v in pr.items() if k != 'gates'}
            save_artifact(art, a.json)
            print('[wcprobe] dump %s (one arm only -- this is a measurement of w, '
                  'not a warm-versus-cold result)'
                  % os.path.basename(pr.get('dump_raw') or ''), flush=True)
            return pr.get('exit_code', 4)
        for stage in stages:
            shots = a.shots or STAGE_SHOTS[stage]
            attempts, run = [], None
            while True:
                try:
                    run = run_stage(stage, tok, a.chip_id, shots=shots, block=block,
                                    chunk=a.chunk, job_timeout=a.job_timeout,
                                    use_options=not a.no_options,
                                    mapping=not a.no_mapping,
                                    cal_first=a.cal_first,
                                    phi=phi_warm if stage == 'warmcold' else phi,
                                    phi_cold=phi_cold, dump_raw=a.dump_raw,
                                    budget_remaining=a.budget_remaining,
                                    ms_per_shot=a.ms_per_shot, exclude=excl)
                    break
                except (BudgetError, NoEligibleBlock) as exc:
                    # A refusal to SPEND is not a refusal BY the platform.  Halving
                    # --shots would silently substitute a different stage for the
                    # pre-registered one, and re-picking a block would defeat the
                    # exclusion, so neither retry applies: stop, record, and say so.
                    print('ERROR: %s' % exc, flush=True)
                    attempts.append(dict(stage=stage, shots=shots,
                                         error=repr(exc)[:400]))
                    art.setdefault('refused_attempts', []).extend(attempts)
                    save_artifact(art, a.json)
                    return 3
                except Exception as exc:
                    attempts.append(dict(stage=stage, shots=shots,
                                         error=repr(exc)[:400]))
                    msg = str(exc).lower()
                    if (not any(k in msg for k in QPU_REFUSAL_MARKERS)
                            or shots <= SHOTS_FLOOR):
                        art.setdefault('refused_attempts', []).extend(attempts)
                        save_artifact(art, a.json)
                        raise
                    nxt = max(SHOTS_FLOOR, shots // 2)
                    print('[cloud] %s refused at %d shots (%s); retrying at %d'
                          % (stage, shots, repr(exc)[:160], nxt), flush=True)
                    shots = nxt
            run['attempts'] = attempts
            record_billing(art, run.get('billing'))
            # analyze_stage ARCHIVES the previous record for this stage (see
            # archive_previous), so it must run before the new run record is written
            # into art['runs'] -- the other way round it would archive the run it
            # was given and leave the artifact with no current record at all.
            analyze_stage(stage, run['results'], run['meta'],
                          run['shots_requested'], art,
                          phi=phi_warm if stage == 'warmcold' else phi,
                          vendor=vendor, n_boot=a.n_boot, phi_cold=phi_cold,
                          source_dump=run['dump_raw'])
            art.setdefault('runs', {})[stage] = stage_record(run)
            # save after EVERY stage: a quota refusal on a later one then costs
            # nothing that was already measured
            save_artifact(art, a.json)
            block = None

    tag = a.tag or '%s_%s' % (a.chip_id, datetime.now().strftime('%Y%m%d_%H%M%S'))
    if not a.no_figure and 'tomo' in art:
        make_figure(art, tag)
    save_artifact(art, a.json)
    return 0


if __name__ == '__main__':
    sys.exit(main())

