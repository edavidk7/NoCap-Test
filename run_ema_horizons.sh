#!/bin/bash
# EMA horizon sweep on the fastest vanilla config (8276 it -> 3.379210 in 131.3 min).
#
# EMA never feeds back into training -- swap_in/swap_out restores the parameters exactly --
# so four decays can be evaluated in one run. Only the extra validation passes cost
# anything, and validation is outside the timed region.
#
#   0.998  ~500 steps    already tested: ended 0.0009 *behind* the raw weights
#   0.999  ~1000 steps   12% of the run
#   0.9995 ~2000 steps   24%, comparable to the 3555-step warmdown
#   0.9998 ~5000 steps   60%, reaches back past the start of the warmdown
#
# Expectation is that longer is worse at the end: with LR annealed to ~0, a longer horizon
# averages in more weights from earlier, higher-LR steps. This measures how much.
cd /home/ubuntu/NoCap-Test || exit 1
mkdir -p /home/ubuntu/NoCap-Test/logs
LOG=/home/ubuntu/NoCap-Test/logs/run_ema_horizons.log

echo ">>> START ema horizon sweep  $(date -u '+%H:%M UTC')"
uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M --model d12 \
  --batch_size 16 --sequence_length 1024 --grad_accumulation_steps 16 \
  --num_iterations 8276 --warmup_iters 444 --warmdown_iters 3555 \
  --val_loss_every 128 --val_batch_size 16 \
  --weight_decay 0.1 --learning_rate 0.0018 \
  --ema_decay 0.998 0.999 0.9995 0.9998 \
  --seed 42 --log_wandb 2>&1 | tee "$LOG"
echo ">>> DONE rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
