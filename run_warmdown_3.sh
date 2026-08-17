#!/bin/bash
# Warmdown test 3: 9000 iters, warmdown=3000 (starts ~6000, 1566 fewer steps than baseline)
uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M \
  --model d12 \
  --batch_size 64 \
  --sequence_length 1024 \
  --grad_accum_schedule "0:2,0.57:4" \
  --num_iterations 9000 \
  --warmup_iters 507 \
  --warmdown_iters 3000 \
  --learning_rate 0.0018 \
  --weight_decay 0.1 \
  --attn_head_windows "0:64,1:64,2:64,3:64,4:128,5:128" \
  --attn_kernel_block 64 \
  --ema_decay 0.99 0.98 0.97 0.96 0.95 \
  --emb_aux_lambda 0.1 \
  --val_loss_every 64 \
  --val_batch_size 16 \
  --log_wandb \
  --seed 42 2>&1 | tee logs/run_warmdown_3.log
