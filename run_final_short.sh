#!/bin/bash
# The headline run: v1's span allocation + the tile fix + margin spent on fewer tokens.
#
# v1 (ramp + 4x64 2x128 6x1024) gave 3.365592 in 129.7 min -- the best loss measured, with
# 0.0165 of unspent margin above the 3.3821 target. v3 then showed --attn_kernel_block 64
# is worth 3.2 min for free (126.5 vs 129.7 min on the same work). Nothing has yet combined
# v1's spans with the tile fix, and nothing has spent the margin.
#
# This does both. Iterations cut 11575 -> 10566 (-8.7% tokens), aiming at ~3.3800 and
# keeping 0.0021 of margin -- roughly twice the 0.0009 fixed-seed noise floor. The ga
# switch still lands exactly on the warmdown boundary (step 6023).
#
# Projected: ~115.5 min, -22.8% against the 149.6 min baseline.
#
# Short EMAs ride along (0.98/0.97/0.96 = 50/33/25 step horizons). They are passive and
# cannot affect the raw trajectory, but they matter for the metric that actually counts:
# the earliest wall clock at which ANY valid model crosses 3.3821. On v4, ema0.98 crossed
# at step 11008 while raw did not clear until 11136 -- 1.9 min earlier. Final loss is the
# wrong scoreboard; first crossing is the right one.
#
# The token-scaling law dL/dln(tokens) = -0.158 has predicted five shortenings to within
# 0.005 and three within 0.001, and has erred conservatively every time -- runs landed
# below their aim, never meaningfully above.
#
# Standings, all B=64, same seed:
#   causal                                  3.379164  127.1 m
#   ramp only                               3.377058  127.5 m
#   v1  ramp + 4x64 2x128 6x1024            3.365592  129.7 m   <- spans from here
#   v3  ramp + 6x64 2x128 4x1024, tile 64   3.368408  126.5 m   <- tile from here
cd /home/ubuntu/NoCap-Test || exit 1
mkdir -p /home/ubuntu/NoCap-Test/logs
LOG=/home/ubuntu/NoCap-Test/logs/run_final_short.log

echo ">>> START shortened v1-spans + tile64  $(date -u '+%H:%M UTC')"
uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M --model d12 \
  --batch_size 64 --sequence_length 1024 \
  --grad_accum_schedule "0:2,0.57:4" \
  --num_iterations 10566 --warmup_iters 571 --warmdown_iters 4543 \
  --val_loss_every 128 --val_batch_size 16 \
  --weight_decay 0.1 --learning_rate 0.0018 \
  --attn_head_windows "0:64,1:64,2:64,3:64,4:128,5:128" \
  --attn_mask_impl flex --attn_kernel_block 64 \
  --ema_decay 0.98 0.97 0.96 \
  --seed 42 --log_wandb 2>&1 | tee "$LOG"
echo ">>> DONE rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
