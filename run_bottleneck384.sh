#!/bin/bash
# Best config + factorised tied embedding through 384.
#
# Everything else is the reference recipe -- B=64, ramp ga2->ga4 at the warmdown boundary,
# v1 spans (4x64, 2x128, 6x1024), tile 64, gelu, full 11575 iterations -- so the head
# factorisation is the only variable and the loss compares directly to:
#     v1  3.365592  129.7 min
#     E   3.365252  128.2 min
#
# head_gate is deliberately not included.
#
# The head becomes: table 50304x384 -> in-proj 384->768 -> trunk -> rmsnorm ->
# GeGLU 768->2x384 -> tied 384xV. Params 123.57M -> 105.14M.
#
# Expected -13.9% forward FLOPs but less in wall clock: the V-sized logits tensor (6.6 GB
# at B=64) and its softmax are untouched, so this halves the matmul, not the bandwidth.
# A 5-10% saving would be the realistic read; anything near 14% would be a surprise.
#
# For scale on what it has to beat: dropping 4 of 12 FFNs was also ~14% of FLOPs and cost
# 0.043 of loss, which was fatal. This targets the head instead of the trunk, and the
# GeGLU is there to blunt the softmax bottleneck a rank-384 output projection imposes.
#
# A 10-step compile check runs first -- the bottleneck path is new and has only been
# verified on CPU, and a compile failure at 08:00 would otherwise idle the GPU for hours.
cd /home/ubuntu/NoCap-Test || exit 1
mkdir -p logs
LOG=logs/run_bottleneck384.log

ARGS=(--input_bin "data/fineweb10B/fineweb_train_*.bin"
      --input_val_bin "data/fineweb10B/fineweb_val_*.bin"
      --output_dir pylog124M --model d12
      --batch_size 64 --sequence_length 1024
      --grad_accum_schedule "0:2,0.57:4"
      --val_batch_size 16
      --weight_decay 0.1 --learning_rate 0.0018
      --attn_head_windows "0:64,1:64,2:64,3:64,4:128,5:128"
      --attn_kernel_block 64
      --lm_head_bottleneck 384
      --seed 42)

echo ">>> compile check  $(date -u '+%H:%M UTC')"
if ! timeout 600 uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py "${ARGS[@]}" \
      --num_iterations 10 --warmup_iters 2 --warmdown_iters 2 --val_loss_every 10 2>&1 \
      | tee logs/bottleneck384_smoke.log | grep -q "step:10/10 | val loss"; then
  echo ">>> COMPILE CHECK FAILED -- not starting the full run  $(date -u '+%H:%M UTC')"
  grep -oE "(AttributeError|ValueError|RuntimeError|TypeError|InductorError): .{0,120}" logs/bottleneck384_smoke.log | head -3
  exit 1
fi
echo ">>> compile check passed  $(date -u '+%H:%M UTC')"

echo ">>> START bottleneck 384, full length  $(date -u '+%H:%M UTC')"
uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py "${ARGS[@]}" \
  --num_iterations 11575 --warmup_iters 625 --warmdown_iters 4977 \
  --val_loss_every 128 --ema_decay 0.98 0.97 0.96 --log_wandb 2>&1 | tee "$LOG"
echo ">>> DONE rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
