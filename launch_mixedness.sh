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
# is exactly 0.500000.  Through the data-register survival w=0.1219 measured by
# the 2026-09 run that is 0.0610 +- 0.0084 = 7.3 sigma, or 4.4 sigma at half that
# survival.  selftest (l) locks the branch, the state, the null, the cost and the
# power; --mode sample --stage warmcold runs the whole chain on the local CPUQVM
# for zero QPU time and must report VERDICT: A/B PASSED before anything is bought.
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
# Usage -- the stage is MANDATORY, there is no "all":
#   bash launch_mixedness.sh warmcold                        # THE PLAN, ~252 s
#   SHOTS=2000 bash launch_mixedness.sh warmcold             # more margin, ~168 s
#   BLOCK=39,48,49,57,58,66,67,75,85 \
#       bash launch_mixedness.sh warmcold                    # pin the 9-qubit block
#   bash launch_mixedness.sh tomo                            # deferred: a 0.60 sigma null
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
#   ./qenv/bin/python run_selftests.py hwmixedness             # selftests (a)-(l)
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
STAGE=${1:?"usage: bash launch_mixedness.sh {warmcold|tomo|probe|bench|ladder}"}
case "$STAGE" in
  warmcold|probe|tomo|bench|ladder) ;;
  *) echo "unknown stage '$STAGE' (there is deliberately no 'all': see 8.3)"; exit 2 ;;
esac
# A 5-qubit BLOCK on a 9-qubit stage -- or the reverse -- puts circuits wider than
# the block they land on.  The widths are STAGE_NQ in hw_mixedness.py.  Guard it
# here so the mistake is caught before it costs QPU time instead of after.
NQ_STAGE=5
case "$STAGE" in warmcold|bench) NQ_STAGE=9 ;; esac
if [ -n "$BLOCK" ]; then
  NQ_BLOCK=$(echo "$BLOCK" | tr ',' '\n' | grep -c .)
  if [ "$NQ_BLOCK" -ne "$NQ_STAGE" ]; then
    echo "ERROR: stage '$STAGE' runs on $NQ_STAGE qubits but BLOCK has $NQ_BLOCK entries."
    echo "       warmcold/bench need best_qubit_blocks(9) = 39,48,49,57,58,66,67,75,85;"
    echo "       tomo/probe/ladder need best_qubit_blocks(5) = 39,48,49,58,67."
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
LOG=qcloud_mixedness_${STAGE}.log
echo "chip=$CHIP stage=$STAGE shots=${SHOTS:-default} block=${BLOCK:-auto} \
angles=$VSCR_ANGLES_FILE vendor=${VEND:-none} log=$LOG"
nohup ./qenv/bin/python -u hw_mixedness.py --mode cloud \
    --stage "$STAGE" \
    --chip-id "$CHIP" \
    --token "$(cat .originq_token)" \
    $VENDOR_ARG $BLOCK_ARG $ROUNDS_ARG $SHOTS_ARG \
    --n-boot 2000 \
    --tag "${CHIP}_$(date +%Y%m%d_%H%M%S)" \
    > "$LOG" 2>&1 &
echo "MIXEDNESS_PID=$!"
echo "tail -f $LOG"
echo "# when it lands -- or is cut off, which the confirmed budget makes likely --"
echo "# inspect what came back:"
echo "#   grep -E 'wall|shots used|refus|TRUNCAT|batch' $LOG"
echo "# then analyse it; --mode analyze accepts a complete OR a truncated dump and"
echo "# reports exactly which inputs and depths are missing:"
echo "#   ./qenv/bin/python hw_mixedness.py --mode analyze --stage all"
