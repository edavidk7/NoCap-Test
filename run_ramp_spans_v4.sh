#!/bin/bash
# Ramp + spans: 6x64, 3x128, 3x1024, at SDPA speed.
#
# One head moved from global to 128 relative to v3 (6x64, 2x128, 4x1024), so this probes
# just below the apparent global-head floor. Four, six and eight global all landed in a
# tight band; two collapsed. Three is the untested point between them.
cd /home/ubuntu/NoCap-Test || exit 1
mkdir -p /home/ubuntu/NoCap-Test/logs
LOG=/home/ubuntu/NoCap-Test/logs/run_ramp_spans_v4.log

echo ">>> START ramp + spans 6x64 3x128 3x1024, tile 64  $(date -u '+%H:%M UTC')"
uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M --model d12 \
  --batch_size 64 --sequence_length 1024 \
  --grad_accum_schedule "0:2,0.57:4" \
  --num_iterations 11575 --warmup_iters 625 --warmdown_iters 4977 \
  --val_loss_every 128 --val_batch_size 16 \
  --weight_decay 0.1 --learning_rate 0.0018 \
  --attn_head_windows "0:64,1:64,2:64,3:64,4:64,5:64,6:128,7:128,8:128" \
  --attn_mask_impl flex --attn_kernel_block 64 \
  --ema_decay 0.98 0.999 \
  --seed 42 --log_wandb 2>&1 | tee "$LOG"
echo ">>> DONE rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
