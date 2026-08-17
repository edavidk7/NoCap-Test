#!/bin/bash
# Higher LR + LR floor + weight averaging, on the fastest vanilla config.
#
# The premise: every EMA horizon finished behind the raw weights because the trapezoid
# anneals LR to exactly zero, leaving the average nothing to cancel. Holding LR at 10% of
# peak keeps the iterate bouncing in a noise ball, which is the regime where averaging is
# supposed to pay -- and it is what torch's own SWALR does. The higher LR is there because
# averaging is what makes a higher LR survivable.
#
# Three estimators are logged at once, all passive observers costing only extra eval passes:
#   ema 0.998 / 0.999 / 0.9995  exponential, 500 / 1000 / 2000 step horizons
#   swa from 90% of the run     equal-weight mean of the last 828 steps, the classic SWA
#                               estimator, which suits a stationary noise ball better than
#                               an exponential one
# The raw weights are logged too, as the built-in control.
#
# Reference to beat: 3.379210 in 131.3 min (lr 0.0018, anneal to zero, no averaging).
# Note this varies three things at once against that reference, so a win needs follow-up
# to attribute; a loss closes the whole direction.
cd /home/ubuntu/NoCap-Test || exit 1
mkdir -p /home/ubuntu/NoCap-Test/logs
LOG=/home/ubuntu/NoCap-Test/logs/run_lrfloor_swa.log

echo ">>> START lr 0.0024 / floor 0.10 / ema+swa  $(date -u '+%H:%M UTC')"
uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M --model d12 \
  --batch_size 16 --sequence_length 1024 --grad_accumulation_steps 16 \
  --num_iterations 8276 --warmup_iters 444 --warmdown_iters 3555 \
  --val_loss_every 128 --val_batch_size 16 \
  --weight_decay 0.1 --learning_rate 0.0024 --lr_final_frac 0.10 \
  --ema_decay 0.998 0.999 0.9995 --swa_start_frac 0.9 \
  --seed 42 --log_wandb 2>&1 | tee "$LOG"
echo ">>> DONE rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
