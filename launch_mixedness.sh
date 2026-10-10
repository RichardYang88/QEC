#!/bin/bash
# launch_mixedness.sh -- the OriginQ real-QPU run that MEASURES the device
# mixedness the mixed-input boundary prices (ED Table 10d), instead of inferring it.
#
# FOUR jobs, in strict priority order, submitted ONE AT A TIME.  The run plan, the
# cost model and the abort rules are QCLOUD_SETUP.md 8 and `--mode plan`; the short
# version is that the platform's metering unit is undocumented and the two candidate
# models differ ~50x in what 300 s buys, so Job 1 exists to decide between them and
# every later job is a human decision taken from Job 1's measured wall clock.
#
#   probe  (Job 1)   8 circuits x   300 shots, 5 qubits  (  2 400 shots)
#   tomo   (Job 2)  19 circuits x  8000 shots, 5 qubits  (152 000 shots)
#   bench  (Job 3)  34 circuits x  2000 shots, 9 qubits  ( 68 000 shots)
#   ladder (Job 4)   4 circuits x  4000 shots, 5 qubits  ( 16 000 shots)
#
# Usage -- the stage is MANDATORY, there is no "all":
#   bash launch_mixedness.sh probe          # Job 1, then read wall_s from the log
#   CHIP=WK_C180 bash launch_mixedness.sh tomo
#   BLOCK=147,155,157,164,165 bash launch_mixedness.sh tomo   # pin the block
#   ROUNDS=4,8 bash launch_mixedness.sh ladder                # extra depths
#
# Before spending anything, these cost zero and must all pass:
#   ./qenv/bin/python hw_mixedness.py --mode plan              # the budget table
#   ./qenv/bin/python hw_mixedness.py --mode verify
#   ./qenv/bin/python run_selftests.py hwmixedness
#   ./qenv/bin/python hw_mixedness.py --mode sample
#   ./qenv/bin/python hw_mixedness.py --mode simulate --seeds 6 \
#       --vendor 0.9993,0.9831,0.9612 --json /tmp/power.json   # power study
#   ./qenv/bin/python hw_mixedness.py --mode preflight \
#       --chip-ids WK_C180_2,WK_C180                           # FREE cloud queries
#
# WK_C180_3 is NOT in devs.txt (the recorded service.backends() output lists only
# WK_C180, WK_C180_2 and the photonic PQPUMESH8), so its vendor row below is
# unverified and is kept only so that a preflight which does find the chip can be
# analysed with the quoted figures.  Do not submit to it before preflight confirms it.
cd /home/yqc/github/QEC
CHIP=${CHIP:-WK_C180_2}
STAGE=${1:?"usage: bash launch_mixedness.sh {probe|tomo|bench|ladder}"}
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
  WK_C180)   VEND=0.9988,0.9790,0.9260 ;;
  WK_C180_3) VEND=0.9990,0.9706,0.9492 ;;   # unverified, see the note above
  *)         VEND= ;;
esac
VENDOR_ARG=""
[ -n "$VEND" ] && VENDOR_ARG="--vendor $VEND"
# pin the block across jobs so the canary's assignment matrix applies to the
# tomography; empty means "ask best_qubit_blocks(n) again", which may return a
# different block and silently invalidate that assumption
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
echo "# when it lands:  grep -E 'wall_s|shots used|refus' $LOG"
