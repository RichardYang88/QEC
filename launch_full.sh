#!/bin/bash
# Launch the full OriginQ real-QPU experiment (late-measure protocol).
# 128 circuits x 1000 shots, 4 batch tasks of 32 (= 2 complete data points each).
cd /home/yqc/github/QEC
TOK=$(cat .originq_token)
nohup ./qenv/bin/python -u qcloud_vscr_new.py --mode cloud \
    --chip-id WK_C180 \
    --token "$TOK" \
    --shots 1000 \
    --p-values 0.02,0.08 \
    --frames 2 \
    --states 0,+ \
    --late-measure \
    --chunk 32 \
    --job-timeout 21600 \
    --dump-raw qcloud_raw_full.json \
    --tag full_late_1000 \
    > qcloud_full.log 2>&1 &
echo "FULL_PID=$!"
