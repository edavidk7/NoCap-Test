#!/bin/bash
# Queue runner for overnight GA schedule experiments
# All 4 runs use: v1 spans, lambda=0.025, ema 0.98/0.97/0.96/0.95, seed 42
set -e
cd /home/ubuntu/NoCap-Test

echo "=== OVERNIGHT QUEUE START: $(date) ==="
echo ""

echo "=== RUN 1/4: GA 0:2,0.4:3,0.6:4 (earlier ga=4 from user's schedule) ==="
echo "Start: $(date)"
bash run_overnight_1.sh
echo "End: $(date)"
echo ""

echo "=== RUN 2/4: GA 0:2,0.35:3,0.57:4 (ga=3 before optimal ga=4 point) ==="
echo "Start: $(date)"
bash run_overnight_2.sh
echo "End: $(date)"
echo ""

echo "=== RUN 3/4: GA 0:2,0.55:4 (simple 2-stage, earlier ga=4) ==="
echo "Start: $(date)"
bash run_overnight_3.sh
echo "End: $(date)"
echo ""

echo "=== RUN 4/4: GA 0:2,0.4:3,0.55:4 (ga=3 + aggressive ga=4) ==="
echo "Start: $(date)"
bash run_overnight_4.sh
echo "End: $(date)"
echo ""

echo "=== ALL 4 RUNS COMPLETE: $(date) ==="
