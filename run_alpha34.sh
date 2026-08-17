#!/bin/bash
# Alternating MLP expansion: alpha 3 on even blocks, 4 on odd blocks.
# Average expansion 3.5x vs stock 4x — ~12.5% fewer MLP FLOPs.
# d_ff values: 3*768=2304 and 4*768=3072, both 64-aligned.
# Fastest schedule: B=64, ga ramp 2->4 @0.57, 10566 it, v1 spans.
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
  --mlp_alpha 3 4 3 4 3 4 3 4 3 4 3 4 \
  --attn_head_windows "0:64,1:64,2:64,3:64,4:128,5:128" \
  --attn_kernel_block 64 \
  --ema_decay 0.98 0.97 0.96 \
  --val_loss_every 128 \
  --val_batch_size 16 \
  --log_wandb \
  --seed 42 2>&1 | tee logs/run_alpha34.log
