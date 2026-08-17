#!/bin/bash
# Ramp + spans: six narrow heads while keeping four global, at SDPA speed.
#
# Separates the two things v2 changed at once. v2 went 4->6 narrow heads AND 6->2 global,
# and lost half the benefit (-0.0047 over ramp vs v1's -0.0097). The 2-global config is the
# suspect: it is the one setting that has failed every time it was tried, while 4, 6 and 8
# global all landed in a tight band. This keeps four global and drops the 512 tier, which
# has never earned its place in any run.
#
# Spans: 6x64, 2x128, 4x1024.
#
# --attn_kernel_block 64 sets the flex Triton tile to 64 while the mask block stays at the
# default 128. Benchmarked at 910.4 ms/step against plain causal SDPA's 910.4 and the
# default tile's 927.3 -- it erases the flex overhead entirely, worth ~2.3 min over the run.
# Same mask, so the arithmetic is unchanged; only kernel tiling and FP order differ.
#
# Standings, all B=64, same token budget and seed:
#   causal                                     3.379164   127.1 m
#   batch ramp alone                           3.377058   127.5 m   -0.0021
#   ramp + 4x64 2x128 6x1024        (v1)       3.365592   129.7 m   -0.0136  <- best
#   ramp + 6x64 2x128 2x512 2x1024  (v2)       ~3.372 projected      -0.007
cd /home/ubuntu/NoCap-Test || exit 1
mkdir -p /home/ubuntu/NoCap-Test/logs
LOG=/home/ubuntu/NoCap-Test/logs/run_ramp_spans_v3.log

echo ">>> START ramp + spans 6x64 2x128 4x1024, tile 64  $(date -u '+%H:%M UTC')"
uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M --model d12 \
  --batch_size 64 --sequence_length 1024 \
  --grad_accum_schedule "0:2,0.57:4" \
  --num_iterations 11575 --warmup_iters 625 --warmdown_iters 4977 \
  --val_loss_every 128 --val_batch_size 16 \
  --weight_decay 0.1 --learning_rate 0.0018 \
  --attn_head_windows "0:64,1:64,2:64,3:64,4:64,5:64,6:128,7:128" \
  --attn_mask_impl flex --attn_kernel_block 64 \
  --seed 42 --log_wandb 2>&1 | tee "$LOG"
echo ">>> DONE rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
