#!/bin/bash
# Batch ramp-up: small effective batch early, the measured optimum late.
#
# Rationale: the critical batch size grows through training, so a batch that is
# token-efficient early is too noisy late. Every previous schedule in this project either
# ran the wrong direction (batch *cuts*, which inject noise -- the +0.145 / +0.026 / +0.020
# spikes all came from cuts) or was confounded (the 0:2..0.35:32 ramp used 1.90B tokens
# instead of 2.17B and spent 85.6% of them at the worst setting). Ramping *up* is the
# benign direction and has never been tested cleanly.
#
# Phase 1: 6598 steps at ga=2  -> 131,072 tokens/update
# Phase 2: 4977 steps at ga=4  -> 262,144 tokens/update, the measured optimum
#
# The transition is aligned exactly to the start of the warmdown, so the whole annealing
# phase runs at the optimal batch -- the earlier finding was that the warmdown drop is a
# property of the batch size being used *during* it, so mixing regimes inside the anneal
# would confound the result.
#
# Micro-batches: 6598*2 + 4977*4 = 33,104, exactly the best config's 8276*4. Same tokens
# (2.169B), same wall clock (~127 min). Only the batch schedule differs.
#
# --ema_decay 0.998 is a passive mid-run predictor: the EMA curve estimates where
# annealing will land, which the raw curve does not. Judge this run on the EMA line at the
# two-thirds mark before waiting for the end.
#
# Reference: 3.379164 in 127.1 min at a constant 262,144 tokens/update.
cd /home/ubuntu/NoCap-Test || exit 1
mkdir -p /home/ubuntu/NoCap-Test/logs
LOG=/home/ubuntu/NoCap-Test/logs/run_batch_ramp.log

echo ">>> START batch ramp ga2 -> ga4  $(date -u '+%H:%M UTC')"
uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M --model d12 \
  --batch_size 64 --sequence_length 1024 \
  --grad_accum_schedule "0:2,0.57:4" \
  --num_iterations 11575 --warmup_iters 625 --warmdown_iters 4977 \
  --val_loss_every 128 --val_batch_size 16 \
  --weight_decay 0.1 --learning_rate 0.0018 \
  --ema_decay 0.998 \
  --seed 42 --log_wandb 2>&1 | tee "$LOG"
echo ">>> DONE rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
