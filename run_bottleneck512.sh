#!/bin/bash
# 576 head bottleneck, plain linear output projection, on the fastest known schedule.
#
# Schedule is the reference recipe: B=64, ga ramp 2->4 at the warmdown boundary, v1 spans,
# tile 64, 11575 iterations (625 warmup / 4977 warmdown). That is the configuration every
# other number in EXPERIMENTS.md is measured against:
#     v1 (no bottleneck)   3.365592   129.7 min
#
# Two things differ from the earlier linear-512 run, which came in ~0.016 behind v1 when
# compared at equal wall clock:
#   - d_b 512 -> 576, so the head keeps more rank. Break-even headroom drops with it,
#     from 0.0198 to 0.0141, because the speedup falls from 11.8% to ~8.6%.
#   - out_proj is initialised (std = 1/sqrt(n_embd)) instead of falling back to PyTorch's
#     default, which had been scaling the head output to 0.578x and squashing the logits
#     toward uniform at init.
cd /home/ubuntu/NoCap-Test || exit 1
mkdir -p logs
# distinct log name: reusing a LOG path already cost us the linear-512 run's log once
LOG=logs/run_bneck576_linear.log

ARGS=(--input_bin "data/fineweb10B/fineweb_train_*.bin"
      --input_val_bin "data/fineweb10B/fineweb_val_*.bin"
      --output_dir pylog124M --model d12
      --batch_size 64 --sequence_length 1024
      --grad_accum_schedule "0:2,0.57:4"
      --val_batch_size 16
      --weight_decay 0.1 --learning_rate 0.0018
      --attn_head_windows "0:64,1:64,2:64,3:64,4:128,5:128"
      --attn_kernel_block 64
      --lm_head_bottleneck 576
      --lm_bottleneck_act linear
      --seed 42)

echo ">>> compile check  $(date -u '+%H:%M UTC')"
if ! timeout 600 uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py "${ARGS[@]}" \
      --num_iterations 10 --warmup_iters 2 --warmdown_iters 2 --val_loss_every 10 2>&1 \
      | tee logs/bneck576_smoke.log | grep -q "step:10/10 | val loss"; then
  echo ">>> COMPILE CHECK FAILED -- not starting the full run  $(date -u '+%H:%M UTC')"
  grep -oE "(AttributeError|ValueError|RuntimeError|TypeError|InductorError): .{0,120}" logs/bneck576_smoke.log | head -3
  exit 1
fi
echo ">>> compile check passed  $(date -u '+%H:%M UTC')"

echo ">>> START bottleneck 576 linear, ramp schedule, full length  $(date -u '+%H:%M UTC')"
uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py "${ARGS[@]}" \
  --num_iterations 11575 --warmup_iters 625 --warmdown_iters 4977 \
  --val_loss_every 128 --ema_decay 0.98 0.97 0.96 --log_wandb 2>&1 | tee "$LOG"
echo ">>> DONE rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
