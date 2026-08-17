#!/bin/bash
# MSWA: Multi-Scale Windowed Attention — per-block head spans.
# 4 tiers of 3 blocks, gradually opening from mostly-local to the current config.
#
#   blocks 0-2:  8×64  2×128  2×256               (heavy local)
#   blocks 3-5:  6×64  2×128  2×256  2×512         (adding medium range)
#   blocks 6-8:  4×64  2×128  2×256  2×512  2×1024 (mixed)
#   blocks 9-11: 4×64  2×128  6×1024               (current run.sh config)
#
# Fastest schedule: B=64, ga ramp 2->4 @0.57, 10566 it, v1 kernel block 64.

T1="0:64,1:64,2:64,3:64,4:64,5:64,6:64,7:64,8:128,9:128,10:256,11:256"
T2="0:64,1:64,2:64,3:64,4:64,5:64,6:128,7:128,8:256,9:256,10:512,11:512"
T3="0:64,1:64,2:64,3:64,4:128,5:128,6:256,7:256,8:512,9:512,10:1024,11:1024"
T4="0:64,1:64,2:64,3:64,4:128,5:128"

uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M \
  --model d12 \
  --batch_size 64 \
  --sequence_length 1024 \
  --grad_accum_schedule "0:2,0.57:4" \
  --num_iterations 10566 \
  --warmup_iters 571 \
  --warmdown_iters 4543 \
  --learning_rate 0.0018 \
  --weight_decay 0.1 \
  --attn_head_windows \
    "$T1" "$T1" "$T1" \
    "$T2" "$T2" "$T2" \
    "$T3" "$T3" "$T4" \
    "$T4" \
  --attn_kernel_block 64 \
  --ema_decay 0.98 0.97 0.96 \
  --val_loss_every 128 \
  --val_batch_size 16 \
  --log_wandb \
  --seed 42 2>&1 | tee logs/run_mswa.log
