#!/bin/bash
# launch_mixedness.sh -- the OriginQ real-QPU run that MEASURES the device
# mixedness the mixed-input boundary prices (ED Table 10d), instead of inferring it.
#
# ONE backend (WK_C180_2) and ONE affordable job (warmcold).  The account holder
# confirmed that QPU time is billed in real time, that exhausting it immediately
# disables the account, and that a job cannot overrun its allocation; with ~300 s
# left and 14 ms/shot that is ~21 400 shots for everything.  The full arithmetic,
# the shot/precision trade-off and both salvage ladders are QCLOUD_SETUP.md 8 and
# `--mode plan`, and selftests (k) and (l) lock the two decisive facts.
#
#   warmcold (THE PLAN)      6 circuits x 3000 shots, 9 qubits (18 000 shots, ~252 s)
#   tomo     (fits, but a   19 circuits x 1000 shots, 5 qubits (19 000 shots, ~266 s)
#            NULL result)
#   probe    (redundant)     8 circuits x  300 shots, 5 qubits ( 2 400 shots, ~ 34 s)
#   ladder   (fits, top-up)  4 circuits x 4000 shots, 5 qubits (16 000 shots, ~224 s)
#   bench    (unaffordable) 34 circuits x 2000 shots, 9 qubits (68 000 shots, ~952 s)
#
# WHY warmcold AND NOT tomo -- both fit, so this is arithmetic, not preference.
# tomo's headline is r<1, and selftest (k) measures this block at f2q=0.99275,
# BETTER than the quoted 0.9831, which puts r<1 at 0.60 sigma: 266 s buys a null,
# and 5 sigma needs 18 651 s = 62x the allocation.  warmcold instead measures the
# gap between the paper's warm-started recovery and a cold-started control on the
# ONE branch where they differ, s=5 / frame IZIII / psi=+, where the noiseless gap
# is exactly 0.500000.  Through the data-register survival w the observable gap is
# w*0.5 -- and w IS NOT ONE NUMBER.  The 2026-09 run (submitted with NO vendor
# options) measured P(data=00000)=0.1219, i.e. w=0.0936, worth 0.0468 +- 0.0077 =
# 6.1 sigma at 3000 shots/circuit; the 2026-10-12 run (WITH options) measured
# 0.0407, i.e. w=0.0098, worth 1.0 sigma.  An earlier version of this file quoted
# 7.3 sigma, which came from feeding the measured PROBABILITY 0.1219 to the power
# model as if it were w -- a 30% overstatement of the gap on top of the platform
# difference.  selftest (l) locks the branch, the state, the null, the cost and all
# three power rows, and --mode sample --stage warmcold runs the whole chain on the
# local CPUQVM for zero QPU time and must report VERDICT: A/B PASSED before
# anything is bought.  Which of the two worlds today is in cannot be computed
# offline: that is what gate 2 below measures for 14 s, before gate 3 spends the
# allocation on it.
#
# probe is deferred as redundant, not merely unaffordable: tomo carries its own
# three calibration circuits, and analyze_stage recovers the bit order from the
# tomography circuits themselves, which selftest (f) proves is sufficient.
#
# warmcold is submitted in chunks of 2 circuits, raw counts written to disk after
# every chunk, and the SIGNAL PAIR IS FIRST: 2 circuits buy the headline A/B on
# their own, 4 add the null control without which the result is uncontrolled, 6 add
# the block's own 9-qubit readout assignment.  That is deliberately the opposite of
# tomo's calibration-first order, because this stage's primary estimator is a raw
# probability whose warm-minus-cold difference is common-mode in readout error and
# needs no assignment matrix -- so the decisive pair goes first and a cut-off still
# yields the claim.
#
# tomo is submitted in chunks of 3 circuits and the raw counts are written to disk
# after every chunk, because a job that runs past the allocation is CUT OFF and a
# cut-off job returns nothing.  The circuits are ordered calibration-first, then
# depth-1 grouped by input state, then depth-2, so each chunk boundary is an
# analysable prefix: 3 circuits buy the assignment matrix, 6 buy M1 for |0>_L and
# M3 at depth 1, 15 buy M1 and M2 in full, 19 add M5.  A cut-off is therefore a
# partial result, not a lost one.
#
# Usage -- PLAN A (QCLOUD_SETUP.md 9.7): probe BOTH chips, let a pre-registered rule
# pick the one the A/B runs on, screen it, then buy the A/B.  Every step is bracketed
# by a dashboard reading.  BUDGET is MANDATORY: it is the number of seconds the OriginQ
# dashboard says are LEFT, the API cannot answer that question, and hw_mixedness.py now
# refuses to submit without it.  MSPS is the billed ms/shot measured from the dashboard
# delta across the PREVIOUS step; omit it and every cost is priced at the pessimistic
# 14 ms/shot instead.
#
#   BUDGET=300 CHIP=WK_C180 bash launch_mixedness.sh wcprobe   # probe 1: 1000 shots
#       # how large is w on THIS chip?  Prints P(data=00000), w and its 95% interval,
#       # the shot ladder, and its verdict sidecar path (pathA).  Exit 0 detection
#       # affordable, 5 bound only, 4 pipeline fault.  Then read the dashboard again:
#       # 1000*(300 - reading)/1000 shots is the billed ms/shot to pass as MSPS next.
#   BUDGET=286 CHIP=WK_C180_2 MSPS=14 BLOCK=39,48,49,57,58,66,67,75,85 \
#       bash launch_mixedness.sh wcprobe                       # probe 2: 1000 shots
#       # the same circuit on the other chip (-> pathB).  BLOCK is MANDATORY here:
#       # WK_C180_2's only 9-qubit block is on the rejection list, so resolve_block
#       # raises NoEligibleBlock rather than silently landing on a dead data channel.
#   PROBE_JSONS="pathA pathB" BUDGET=272 MSPS=14 \
#       bash launch_mixedness.sh wcpick                        # FREE, no token
#       # applies the rule declared BEFORE the probes were bought: detect on the
#       # higher-w chip if w >= min_w_for_detection(S*), else bound on WK_C180_2 (the
#       # 2026-10-12 chip and block).  Prints the chip, the block, S*, the z that w is
#       # worth, the bound a null would report, the gate-4 command, and writes the
#       # decision record to hw_evidence/<UTC>_wcpick.json.  Exit 0 / 5 / 4 (STOP).
#   BUDGET=272 MSPS=14 [CHIP=<chosen>] [BLOCK=<chosen>] \
#       bash launch_mixedness.sh screen                        # 2000 shots, <=28 s
#       # is the CHOSEN block usable at all?  Exit 0 usable, 4 not -- and a 4 here
#       # stops the run, because no shot count repairs a dead data channel.
#   BUDGET=244 MSPS=2.0 SHOTS=2000 [CHIP=<chosen>] [BLOCK=<chosen>] \
#       bash launch_mixedness.sh warmcold                      # the A/B: 6 x S*
#       # CAL_FIRST=1 by default: submission 1 is the block's readout screen, so a
#       # cut-off still leaves the data that decides whether the run is readable.
#       # A post-screen is worth buying afterwards if >=28 s remain.
#
# The remaining stages are unchanged and still deferred:
#   bash launch_mixedness.sh tomo                            # a 0.60 sigma null
#   SHOTS=750 bash launch_mixedness.sh tomo                  # more margin, ~200 s
#   BLOCK=39,48,49,58,67 bash launch_mixedness.sh tomo       # pin the 5-qubit block
#   ROUNDS=4,8 bash launch_mixedness.sh ladder               # extra depths
#
# NOTE THE BLOCK WIDTH: warmcold and bench run on NINE qubits, so their default
# block is best_qubit_blocks(9) = [39,48,49,57,58,66,67,75,85], NOT the five-qubit
# [39,48,49,58,67] that tomo/probe/ladder use.  Passing a 5-qubit BLOCK to
# warmcold would submit 9-qubit circuits onto 5 qubits.  A submission has one
# block, so the two stage families cannot share one job.
#
# Before spending anything, these cost zero QPU time and must all pass.  The first
# four are the ones that gate the warmcold job; the last two gate tomo.
#   ./qenv/bin/python hw_mixedness.py --mode plan              # the budget table
#   ./qenv/bin/python hw_mixedness.py --mode verify --stage warmcold
#   ./qenv/bin/python run_selftests.py hwmixedness             # selftests (a)-(p)
#   ./qenv/bin/python hw_mixedness.py --mode sample --stage warmcold --shots 4000
#       # samples all 6 circuits on the local CPUQVM, pushes them through the full
#       # analysis chain, and prints the power study at the real 3000-shot budget.
#       # MUST end in "[warmcold] VERDICT: A/B PASSED" and
#       # "[warmcold-dry-run] noiseless chain OK" before any QPU time is bought.
#       # This is what caught the reversed-key data-window bug, so do not skip it.
#   ./qenv/bin/python hw_mixedness.py --mode sample            # the tomo chain
#   ./qenv/bin/python hw_mixedness.py --mode simulate --seeds 6 --shots 1000 \
#       --vendor 0.9993,0.9831,0.9612 --json /tmp/power.json   # at the real budget
#   ./qenv/bin/python hw_mixedness.py --mode preflight \
#       --chip-ids WK_C180_2 --vendor 0.9993,0.9831,0.9612      # FREE cloud queries
#
# Only WK_C180_2 is in scope.  WK_C180_3 is absent from this script because it is
# OUT OF SCOPE, not because it does not exist: devs.txt is a 2026-09-08 snapshot of
# service.backends() listing only WK_C180, WK_C180_2 and the photonic PQPUMESH8, but
# the live query of 2026-10-11 returns WK_C180_3: true -- that chip came up after the
# snapshot, and HanYuan_01 is the offline one.  WK_C180 stays in the vendor table
# only because its quoted fidelities are the comparison that justified choosing
# WK_C180_2, not because anything is submitted to it.
#
# MEASURED CALIBRATION, 2026-10-11 (QCLOUD_SETUP.md 8.8) -- read this before running.
# best_qubit_blocks(5) on WK_C180_2 is [39,48,49,58,67], and ChipInfo (no longer 401)
# publishes f2q=0.99275 for it, BETTER than the 0.9831 this script passes as --vendor.
# Better gates mean a purer state, so the predicted radius is 0.9478 not 0.8550 and
# "r<1" is a 0.60 sigma statement at SHOTS=1000, not the ~1.8 sigma the quoted table
# implied; 5 sigma needs ~18 700 s of QPU, 62x the budget.  The binding constraint on
# that headline is the device being better than its datasheet, not the ~300 s.
# The block must stay the vendor's pick.  Ranking the chip's 96 connected 5-qubit
# blocks by the significance of r<1 puts this one LAST, 96/96, and [40,49,50,58,68]
# first at 6.3 sigma -- purely because those are the worst qubits on the chip.
# Choosing the apparatus so the hypothesis test passes is not a measurement, so
# selftest (k) asserts both the direction of the correction and the rejection.
cd /home/yqc/github/QEC
CHIP=${CHIP:-WK_C180_2}
STAGE=${1:?"usage: bash launch_mixedness.sh {screen|wcprobe|wcpick|warmcold|suppl|tomo|probe|bench|ladder}"}
case "$STAGE" in
  screen|wcprobe|wcpick|warmcold|suppl|probe|tomo|bench|ladder) ;;
  *) echo "unknown stage '$STAGE' (there is deliberately no 'all': see 8.3)"; exit 2 ;;
esac
# THE BUDGET IS AN INPUT, NOT A GUESS.  QPU time is billed in real time, a job
# cannot overrun its allocation, and the remainder cannot be queried from the API,
# so the account holder reads it off the dashboard and passes it in.  Without it
# hw_mixedness.py refuses to submit, which is the point: the 2026-10-12 run spent
# 18 000 shots against a remainder nobody had read.
BUDGET=${BUDGET:?"BUDGET=<seconds left on the OriginQ dashboard> is required -- it cannot be queried from the API"}
# A 5-qubit BLOCK on a 9-qubit stage -- or the reverse -- puts circuits wider than
# the block they land on.  The widths are STAGE_NQ in hw_mixedness.py.  Guard it
# here so the mistake is caught before it costs QPU time instead of after.
NQ_STAGE=5
case "$STAGE" in screen|wcprobe|warmcold|bench|suppl) NQ_STAGE=9 ;; esac
if [ -n "$BLOCK" ]; then
  NQ_BLOCK=$(echo "$BLOCK" | tr ',' '\n' | grep -c .)
  if [ "$NQ_BLOCK" -ne "$NQ_STAGE" ]; then
    echo "ERROR: '$STAGE' runs on $NQ_STAGE qubits but BLOCK has $NQ_BLOCK entries."
    echo "       A block is a list of PHYSICAL qubits on ONE chip, so it must come"
    echo "       from this chip's own ranked list:"
    echo "         ./qenv/bin/python hw_mixedness.py --mode preflight --chip-ids $CHIP"
    echo "       On WK_C180_2 the 9-qubit list has exactly one entry and this file"
    echo "       rejects it on measured grounds (REJECTED_BLOCKS), so a 9-qubit run"
    echo "       there needs --allow-rejected-block or a different chip."
    exit 2
  fi
fi
# the paper's warm-start angles, not the v1 snapshot the feasibility run used
export VSCR_ANGLES_FILE=${VSCR_ANGLES_FILE:-$PWD/vscr_angles_paper_dep.npz}
# vendor aggregate fidelities quoted for the chosen backend, recorded in the
# artifact so the backend choice is reproducible rather than asserted
case "$CHIP" in
  WK_C180_2) VEND=0.9993,0.9831,0.9612 ;;
  WK_C180)   VEND=0.9988,0.9790,0.9260 ;;   # comparison only, nothing runs here
  *)         VEND= ;;
esac
VENDOR_ARG=""
[ -n "$VEND" ] && VENDOR_ARG="--vendor $VEND"
# pin the block across CHUNKS so the calibration measured in chunk 1 applies to the
# tomography in the later ones; empty means "ask best_qubit_blocks(n) again", which
# may return a different block and silently invalidate that assumption
BLOCK_ARG=""
[ -n "$BLOCK" ] && BLOCK_ARG="--block $BLOCK"
ROUNDS_ARG=""
[ -n "$ROUNDS" ] && ROUNDS_ARG="--rounds $ROUNDS"
SHOTS_ARG=""
[ -n "$SHOTS" ] && SHOTS_ARG="--shots $SHOTS"
DUMP_ARG=""
[ -n "$DUMP" ] && DUMP_ARG="--dump-raw $DUMP"
# A log name that CANNOT collide.  hw_evidence/ is TRACKED and these logs are the only
# record of a paid run's wall time, per-chunk returns and dashboard readings -- yet the
# name was built from a clock with 1-second resolution and then opened with '>', so two
# calls in the same second silently replaced the first one's evidence with the second
# one's.  Observed 2026-10-15 with two wcpick calls: the file that survived on disk held
# a DIFFERENT verdict (BOUND at 14.0 ms/shot) from the one the terminal had just printed
# (DETECTION at 1.0).  Uniqueness is therefore asserted by probing the filesystem rather
# than assumed from the clock -- the same discipline unique_dump_path and
# assert_writable_dump apply to raw counts inside hw_mixedness.py.
unique_log() {  # $1 = desired path; echoes a path that does not exist yet
  local want="$1" base="${1%.log}" n=1
  while [ -e "$want" ]; do want="${base}_${n}.log"; n=$((n + 1)); done
  printf '%s\n' "$want"
}
# gate-specific flags.  'screen' and 'wcprobe' are not stages: they submit one and
# two circuits respectively and short-circuit before the stage loop, so the stage
# passed underneath them is the nine-qubit one they belong to.
if [ "$STAGE" = "wcpick" ]; then
  # FREE, and the step plan A exists for: it applies the chip rule QCLOUD_SETUP.md 9.7
  # declared BEFORE any of today's money was spent, to the probe sidecars PROBE_JSONS
  # names.  Reading no token and submitting nothing is the point -- where the last
  # ~200 s goes becomes reproducible from files on disk, so nobody has to take it on
  # trust that the rule was fixed in advance.  Placed above the ARGS block because that
  # one cats .originq_token unconditionally and passes --mode cloud.
  #   PROBE_JSONS="<chip A sidecar> <chip B sidecar>" BUDGET=<R2> MSPS=<measured> \
  #       bash launch_mixedness.sh wcpick
  # Each --wc-probe run prints its sidecar as "[wcprobe] verdict sidecar <path>".
  # BUDGET here is the CURRENT dashboard reading (R2, taken after BOTH probes): S* must
  # be priced against what is actually left, not against what a probe saw when it ran.
  # Exit code IS the verdict: 0 detection affordable | 4 STOP, do not buy the A/B |
  # 5 bound only, which is a human decision to spend.
  if [ -z "$PROBE_JSONS" ]; then
    echo "ERROR: STAGE=wcpick needs PROBE_JSONS='<probe1.probe.json> <probe2.probe.json>'"
    echo "       (the paths the --wc-probe runs printed as 'verdict sidecar ...')"
    exit 2
  fi
  mkdir -p hw_evidence
  TS=$(date -u +%Y%m%dT%H%M%SZ)
  LOG=$(unique_log "hw_evidence/${TS}_wcpick.log")
  WARGS=(--mode wcpick --budget-remaining "$BUDGET")
  for f in $PROBE_JSONS; do WARGS+=(--probe-json "$f"); done
  [ -n "$MSPS" ] && WARGS+=(--ms-per-shot "$MSPS")
  {
    echo "# launch_mixedness.sh wcpick  (FREE: no submission, no token read)"
    echo "# utc=$TS budget_after_probes=${BUDGET}s ms_per_shot=${MSPS:-unset (priced at the pessimistic 14)}"
    echo "# probes=$PROBE_JSONS"
    echo "# rule=QCLOUD_SETUP.md 9.7 (declared before the probes were bought)"
  } > "$LOG"
  ./qenv/bin/python -u hw_mixedness.py "${WARGS[@]}" 2>&1 | tee -a "$LOG"
  RC=${PIPESTATUS[0]}
  echo "exit=$RC  (0 detection affordable | 4 STOP | 5 bound only)"
  echo "# the decision record is in $LOG plus the hw_evidence/*_wcpick.json it names"
  exit "$RC"
fi
GATE_ARG=""
PY_STAGE="$STAGE"
case "$STAGE" in
  screen)
    GATE_ARG="--cal9-screen"
    [ -n "$ALLOW_REJECTED" ] && GATE_ARG="$GATE_ARG --allow-rejected-block"
    PY_STAGE=warmcold ;;
  wcprobe)
    GATE_ARG="--wc-probe"
    PY_STAGE=warmcold ;;
  warmcold)
    # cal9's channel labels DO transfer to the science circuits (the contrary
    # finding is retracted in check_cal9_transferability), so putting the
    # calibration pair first is what lets a budget cut-off still leave the block's
    # readout screen on disk.  CAL_FIRST=0 reproduces the 2026-10-12 order.
    [ "${CAL_FIRST:-1}" = "1" ] && GATE_ARG="--cal-first" ;;
  suppl)
    # The psi='0' reproduction of the 2026-09 feasibility set (see the SUPPL_GRID
    # comment).  It carries NO gate argument and deliberately no in-job cal9:
    # billing is per SHOT rather than per instruction, so two calibration circuits
    # inside a 20 000-shot submission cost 40 000 shots -- which is exactly what
    # the 2026-10-21 A/B spent a third of its budget on.  Run `screen` first as its
    # own 1 000-shot submission (0.6 s at the measured rate) and only buy this if
    # it passes:
    #   CHIP=<chip> BUDGET=<s> MSPS=0.309 bash launch_mixedness.sh screen
    #   CHIP=<chip> BUDGET=<s> MSPS=0.309 bash launch_mixedness.sh suppl
    # MSPS is REQUIRED here.  At the falsified 14 ms/shot default this stage prices
    # at 1 120 s and the budget gate refuses it; at the measured 0.3090 ms/shot it
    # is 24.7 s.  selftest (q) and `--mode sample --stage suppl` must both pass
    # before this is bought.
    #
    # BLOCK IS LOAD-BEARING FOR THIS STAGE, unlike every other one.  best_qubit_blocks(9)
    # returns exactly ONE block per chip (all four preflight records agree:
    # WK_C180 -> [52,61,62,70,71,79,80,88,89], WK_C180_2 -> [39,48,49,57,58,66,67,75,85]),
    # but the 2026-09 feasibility run whose numbers this stage reproduces ran on
    # WK_C180 block [147,155,157,164,165,166,174,175,176] -- a DIFFERENT physical
    # block on the same chip.  Running only on the ranked block would control psi
    # and leave block confounded, so the reproduction is bought on the September
    # block explicitly and the ranked block is run beside it as a within-chip
    # contrast.  Run `screen` on that block FIRST: it is 0.62 s, and it is the only
    # offline-unanswerable question here -- whether the platform still accepts a
    # block its own ranking no longer returns.  An explicit --block bypasses
    # resolve_block entirely, so nothing validates it before the job does.
    ;;
esac
# Logs are EVIDENCE and hw_evidence/ is TRACKED (see .gitignore): the wall time, the
# per-chunk returns, the dashboard readings that bracket the spend and the verdict
# printed at the time exist nowhere else, and the manuscript quotes them.
mkdir -p hw_evidence
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$(unique_log "hw_evidence/${TS}_${CHIP}_${STAGE}.log")
{
  echo "# launch_mixedness.sh $STAGE"
  echo "# utc=$TS chip=$CHIP stage=$PY_STAGE $GATE_ARG"
  echo "# dashboard_before=${BUDGET}s ms_per_shot=${MSPS:-unset (priced at the pessimistic 14)}"
  echo "# block=${BLOCK:-auto} shots=${SHOTS:-default} cal_first=${CAL_FIRST:-1}"
  echo "# angles=$VSCR_ANGLES_FILE vendor=${VEND:-none}"
  echo "# WHEN THE JOB LANDS: read the dashboard again and append the reading, so the"
  echo "# billed ms/shot stops being a hypothesis:"
  echo "#   echo 'dashboard_after=<seconds>' >> $LOG"
} > "$LOG"
ARGS=(--mode cloud --stage "$PY_STAGE" --chip-id "$CHIP"
      --token "$(cat .originq_token)" --budget-remaining "$BUDGET"
      --n-boot 2000 --tag "${CHIP}_${TS}")
[ -n "$MSPS" ] && ARGS+=(--ms-per-shot "$MSPS")
# shellcheck disable=SC2206
[ -n "$GATE_ARG" ] && ARGS+=($GATE_ARG)
[ -n "$VENDOR_ARG" ] && ARGS+=($VENDOR_ARG)
[ -n "$BLOCK_ARG" ] && ARGS+=($BLOCK_ARG)
[ -n "$ROUNDS_ARG" ] && ARGS+=($ROUNDS_ARG)
[ -n "$SHOTS_ARG" ] && ARGS+=($SHOTS_ARG)
[ -n "$DUMP_ARG" ] && ARGS+=($DUMP_ARG)
echo "chip=$CHIP stage=$STAGE budget=${BUDGET}s ms/shot=${MSPS:-14 (pessimistic)} \
shots=${SHOTS:-default} block=${BLOCK:-auto} angles=$VSCR_ANGLES_FILE \
vendor=${VEND:-none} log=$LOG"
case "$STAGE" in
  screen|wcprobe)
    # Gates 1 and 2 are short (<=28 s of QPU time) and their verdict decides whether
    # gate 3 is worth buying, so they run in the FOREGROUND and the exit code IS the
    # verdict: 0 pass / 3 refused by the budget gate / 4 not usable / 5 bound only.
    ./qenv/bin/python -u hw_mixedness.py "${ARGS[@]}" 2>&1 | tee -a "$LOG"
    RC=${PIPESTATUS[0]}
    echo "exit=$RC  (0 pass | 3 budget refusal | 4 not usable/pipeline fault | 5 bound only)"
    echo "# now read the dashboard and record it:  echo 'dashboard_after=<s>' >> $LOG"
    exit "$RC"
    ;;
esac
nohup ./qenv/bin/python -u hw_mixedness.py "${ARGS[@]}" >> "$LOG" 2>&1 &
echo "MIXEDNESS_PID=$!"
echo "tail -f $LOG"
echo "# when it lands -- or is cut off, which the confirmed budget makes likely --"
echo "# inspect what came back:"
echo "#   grep -E 'wall|shots used|refus|TRUNCAT|batch|budget' $LOG"
echo "# then analyse it.  --mode analyze accepts a complete OR a truncated dump,"
echo "# reports exactly which inputs and depths are missing, and reads the NEW"
echo "# timestamped dump named in the log (the legacy names are frozen evidence)."
# The re-analysis has to be told about the SAME reordering this run was submitted
# with: --cal-first moves the two cal9 circuits to the front, and analyze rebuilds
# the expected meta to compare against the submission-time sidecar.  Omitting the
# flag makes a perfectly good dump die on a meta mismatch -- which is what the
# 2026-10-21 WK_C180_2 run's own printed instructions did, telling the reader to
# re-analyse it in the one way that could not work.
CF_HINT=""
[ "${CAL_FIRST:-1}" = "1" ] && [ "$STAGE" = "warmcold" ] && CF_HINT=" --cal-first"
echo "#   ./qenv/bin/python hw_mixedness.py --mode analyze --stage ${PY_STAGE}${CF_HINT} \\"
echo "#       --dump-raw <the qcloud_raw_${PY_STAGE}_${CHIP}_*.json this run wrote>"
# An if-block, not `[ ... ] && echo`, because this is the LAST command in the
# script: a short-circuited && would make the launcher exit 1 whenever CAL_FIRST=0,
# and a caller that checks $? would read a successful submission as a failure.
if [ -n "$CF_HINT" ]; then
  echo "#   (--cal-first is REQUIRED here: this run was submitted with it, so the"
  echo "#    dump's circuit order has cal9 first, and the sidecar check refuses a"
  echo "#    rebuild that does not match that order)"
fi
exit 0
