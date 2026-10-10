#!/bin/bash
# launch_mixedness.sh -- the OriginQ real-QPU run that MEASURES the device
# mixedness the mixed-input boundary prices (ED Table 10d), instead of inferring it.
#
# Three jobs, in priority order, each dumped incrementally and analysed as soon as
# it lands, so a quota refusal on a later job loses nothing already measured:
#   probe :  8 circuits x 1500 shots, 5 qubits  (12 000 shots)
#   tomo  : 16 circuits x 8000 shots, 5 qubits  (128 000 shots)
#   bench : 34 circuits x 1000 shots, 9 qubits  ( 34 000 shots)
# Total 58 circuits, 174 000 shots, ~4 s of pure execution at 15-40 us/shot.
#
# Usage:
#   bash launch_mixedness.sh                 # all three stages on WK_C180_2
#   bash launch_mixedness.sh probe           # one stage only (recommended first:
#                                            # it is what tells us whether QPU time
#                                            # is metered per shot or per job)
#   CHIP=WK_C180 bash launch_mixedness.sh    # a different backend
#
# Before spending anything, these cost zero and must all pass:
#   ./qenv/bin/python hw_mixedness.py --mode verify
#   ./qenv/bin/python run_selftests.py hwmixedness
#   ./qenv/bin/python hw_mixedness.py --mode sample
#   ./qenv/bin/python hw_mixedness.py --mode simulate --seeds 6 \
#       --vendor 0.9993,0.9831,0.9612 --json /tmp/power.json   # power study
#   ./qenv/bin/python hw_mixedness.py --mode preflight \
#       --chip-ids WK_C180_2,WK_C180,WK_C180_3                 # FREE cloud queries
cd /home/yqc/github/QEC
CHIP=${CHIP:-WK_C180_2}
STAGE=${1:-all}
# the paper's warm-start angles, not the v1 snapshot the feasibility run used
export VSCR_ANGLES_FILE=${VSCR_ANGLES_FILE:-$PWD/vscr_angles_paper_dep.npz}
# vendor aggregate fidelities quoted for the chosen backend, recorded in the
# artifact so the backend choice is reproducible rather than asserted
case "$CHIP" in
  WK_C180_2) VEND=0.9993,0.9831,0.9612 ;;
  WK_C180)   VEND=0.9988,0.9790,0.9260 ;;
  WK_C180_3) VEND=0.9990,0.9706,0.9492 ;;
  *)         VEND= ;;
esac
VENDOR_ARG=""
[ -n "$VEND" ] && VENDOR_ARG="--vendor $VEND"
LOG=qcloud_mixedness_${STAGE}.log
echo "chip=$CHIP stage=$STAGE angles=$VSCR_ANGLES_FILE vendor=${VEND:-none} log=$LOG"
nohup ./qenv/bin/python -u hw_mixedness.py --mode cloud \
    --stage "$STAGE" \
    --chip-id "$CHIP" \
    --token "$(cat .originq_token)" \
    $VENDOR_ARG \
    --n-boot 2000 \
    --tag "${CHIP}_$(date +%Y%m%d_%H%M%S)" \
    > "$LOG" 2>&1 &
echo "MIXEDNESS_PID=$!"
echo "tail -f $LOG"
