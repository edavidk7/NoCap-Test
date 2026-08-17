#!/bin/bash
# Queue runner for warmdown length experiments
# All use: GA 0:2,0.57:4, warmup=507, LR=0.0018, lambda=0.025, ema 0.99-0.95, seed 42
# Warmdown start ~6000 in all cases, varying total length
set -e
cd /home/ubuntu/NoCap-Test

echo "=== WARMDOWN QUEUE START: $(date) ==="
echo ""

echo "=== RUN 1/3: 10000 iters, warmdown=4000 (mild compression) ==="
echo "Start: $(date)"
bash run_warmdown_1.sh
echo "End: $(date)"
echo ""

echo "=== RUN 2/3: 9500 iters, warmdown=3500 (moderate compression) ==="
echo "Start: $(date)"
bash run_warmdown_2.sh
echo "End: $(date)"
echo ""

echo "=== RUN 3/3: 9000 iters, warmdown=3000 (aggressive compression) ==="
echo "Start: $(date)"
bash run_warmdown_3.sh
echo "End: $(date)"
echo ""

echo "=== ALL 3 WARMDOWN RUNS COMPLETE: $(date) ==="
