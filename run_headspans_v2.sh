#!/bin/bash
# Hand-picked span ladder with a 256 tier inserted, via flex block masks.
#
# Spans: 2x64, 2x128, 2x256, 2x512, 4x1024. The two 256 heads come out of the global
# group, so this is the config that worked with one rung added to the ladder.
#
# Why not the "data-driven" allocation: spreading tiers to match the copy-distance curve
# (4x128, 4x256, 2x512, 2x1024) left only 2 global heads and gave up almost the whole
# advantage -- about -0.002 vs causal at mid-run against the hand-picked config's -0.009.
# The corpus statistics say what a head of a given width can reach; they say nothing about
# how many heads must stay global, and that looks like the variable that matters. This
# keeps 4 global heads rather than 2.
#
# Results being compared against, all at B=64/ga=4, 8276 it, same schedule and seed:
#   plain causal                             3.379164   127.1 min
#   2x64 2x128 2x512 6x1024 (dense mask)     3.373024   158.2 min  (-0.0061)
#   4x128 4x256 2x512 2x1024 (flex)          killed at step 4204, tracking ~parity
#
# flex costs +2.0% (930 vs 911.5 ms/step) against dense's +25.9%, so the -0.006 class of
# gain is worth ~3.9% of wall clock and would net out positive here.
#
# Note the two 64 heads are kept only to stay comparable with the config that worked --
# flex block granularity is 128, so they cost exactly what 128 costs while seeing less.
cd /home/ubuntu/NoCap-Test || exit 1
mkdir -p /home/ubuntu/NoCap-Test/logs
LOG=/home/ubuntu/NoCap-Test/logs/run_headspans_v2.log

echo ">>> START spans 2x64 2x128 2x256 2x512 4x1024 (flex)  $(date -u '+%H:%M UTC')"
uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M --model d12 \
  --batch_size 64 --sequence_length 1024 --grad_accumulation_steps 4 \
  --num_iterations 8276 --warmup_iters 444 --warmdown_iters 3555 \
  --val_loss_every 128 --val_batch_size 16 \
  --weight_decay 0.1 --learning_rate 0.0018 \
  --attn_head_windows "0:64,1:64,2:128,3:128,4:256,5:256,6:512,7:512" \
  --attn_mask_impl flex \
  --seed 42 --log_wandb 2>&1 | tee "$LOG"
echo ">>> DONE rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
