#!/bin/bash
# Batch ramp + per-head spans: the two changes that worked, combined.
#
# They are orthogonal -- the ramp alters the optimization schedule and leaves attention
# alone, the spans alter the attention mask and leave the schedule alone -- so if the
# effects are even partly additive this should land below either on its own.
#
# Spans: 4x64, 2x128, 6x1024.
# Schedule: 6598 steps at ga=2 (131,072 tokens/update), then 4977 at ga=4 (262,144),
# switching exactly at the warmdown boundary. 33,104 micro-batches, 2.170B tokens --
# identical budget to every config below.
#
# Ingredients, all at B=64 and the same seed:
#   causal, no ramp                       3.379164   127.1 m
#   batch ramp alone                      3.377058   127.5 m   -0.0021
#   A  4x64 2x128 2x512 4x1024, no ramp   3.373514   129.3 m   -0.0056
#   C  2x64 2x128 8x1024,       no ramp   3.372455   130.8 m   -0.0067
#   D  4x128 2x512 6x1024,      no ramp   3.377452   130.5 m   -0.0017  <- no head below
#                                                                          128; the 64s
#                                                                          are what matter
cd /home/ubuntu/NoCap-Test || exit 1
mkdir -p /home/ubuntu/NoCap-Test/logs
LOG=/home/ubuntu/NoCap-Test/logs/run_ramp_spans.log

echo ">>> START ramp + spans 4x64 2x128 6x1024  $(date -u '+%H:%M UTC')"
uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M --model d12 \
  --batch_size 64 --sequence_length 1024 \
  --grad_accum_schedule "0:2,0.57:4" \
  --num_iterations 11575 --warmup_iters 625 --warmdown_iters 4977 \
  --val_loss_every 128 --val_batch_size 16 \
  --weight_decay 0.1 --learning_rate 0.0018 \
  --attn_head_windows "0:64,1:64,2:64,3:64,4:128,5:128" \
  --attn_mask_impl flex \
  --seed 42 --log_wandb 2>&1 | tee "$LOG"
echo ">>> DONE rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
