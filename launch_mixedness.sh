#!/bin/bash
# launch_mixedness.sh -- the OriginQ real-QPU run that MEASURES the device
# mixedness the mixed-input boundary prices (ED Table 10d), instead of inferring it.
#
# ONE backend (WK_C180_2) and ONE affordable job (tomo).  The account holder
# confirmed that QPU time is billed in real time, that exhausting it immediately
# disables the account, and that a job cannot overrun its allocation; with ~300 s
# left and 14 ms/shot that is ~21 400 shots for everything, and the tomo stage at
# 1000 shots is 19 000 of them (~266 s).  The full arithmetic, the shot/precision
# trade-off and the salvage ladder are QCLOUD_SETUP.md 8 and `--mode plan`.
#
#   tomo     (THE PLAN)     19 circuits x 1000 shots, 5 qubits (19 000 shots, ~266 s)
#   probe    (deferred)      8 circuits x  300 shots, 5 qubits ( 2 400 shots, ~ 34 s)
#   bench    (unaffordable) 34 circuits x 2000 shots, 9 qubits (68 000 shots, ~952 s)
#   ladder   (unaffordable)  4 circuits x 4000 shots, 5 qubits (16 000 shots, ~224 s)
#
# probe is deferred as redundant, not merely unaffordable: tomo carries its own
# three calibration circuits, and analyze_stage recovers the bit order from the
# tomography circuits themselves, which selftest (f) proves is sufficient.
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
#   bash launch_mixedness.sh tomo                            # the plan
#   SHOTS=750 bash launch_mixedness.sh tomo                  # more margin, ~200 s
#   BLOCK=147,155,157,164,165 bash launch_mixedness.sh tomo  # pin the block
#   ROUNDS=4,8 bash launch_mixedness.sh ladder               # extra depths
#
# Before spending anything, these cost zero QPU time and must all pass:
#   ./qenv/bin/python hw_mixedness.py --mode plan              # the budget table
#   ./qenv/bin/python hw_mixedness.py --mode verify
#   ./qenv/bin/python run_selftests.py hwmixedness
#   ./qenv/bin/python hw_mixedness.py --mode sample
#   ./qenv/bin/python hw_mixedness.py --mode simulate --seeds 6 --shots 1000 \
#       --vendor 0.9993,0.9831,0.9612 --json /tmp/power.json   # at the real budget
#   ./qenv/bin/python hw_mixedness.py --mode preflight \
#       --chip-ids WK_C180_2 --vendor 0.9993,0.9831,0.9612      # FREE cloud queries
#
# Only WK_C180_2 is in scope.  WK_C180_3 is NOT in devs.txt (the recorded
# service.backends() output lists only WK_C180, WK_C180_2 and the photonic
# PQPUMESH8), so it does not exist and has been removed from this script; WK_C180
# stays in the vendor table only because its quoted fidelities are the comparison
# that justifies choosing WK_C180_2, not because anything is submitted to it.
cd /home/yqc/github/QEC
CHIP=${CHIP:-WK_C180_2}
STAGE=${1:?"usage: bash launch_mixedness.sh {tomo|probe|bench|ladder}"}
case "$STAGE" in
  probe|tomo|bench|ladder) ;;
  *) echo "unknown stage '$STAGE' (there is deliberately no 'all': see 8.3)"; exit 2 ;;
esac
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
