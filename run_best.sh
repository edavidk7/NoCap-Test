#!/bin/bash
# Shortened variant of the current best config.
#
# Best MEASURED result:  8549 it, warmup 459, warmdown 3672 -> 3.373880 in 134.4 min (-10.2%)
# This run spends the remaining margin (that run landed 0.005 under its aim):
#                        8276 it, warmup 444, warmdown 3555 -> aims 3.3790, projected 130.2 min (-13.0%)
#
# Two changes from the stock baseline, both found by sweeping existing knobs:
#   grad_accumulation_steps 32 -> 16   (-0.0082)
#   warmdown 21.5% -> 43% of tokens    (-0.0099)
cd /home/ubuntu/NoCap-Test || exit 1
LOG=/home/ubuntu/NoCap-Test/logs/run_short5.log
mkdir -p /home/ubuntu/NoCap-Test/logs

ITERS=${1:-8276}
WARMUP=${2:-444}
WARMDOWN=${3:-3555}

echo ">>> START ${ITERS} it / warmup ${WARMUP} / warmdown ${WARMDOWN}  $(date -u '+%H:%M UTC')"
uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M --model d12 \
  --batch_size 16 --sequence_length 1024 \
  --grad_accumulation_steps 16 --num_iterations "$ITERS" \
  --warmup_iters "$WARMUP" --warmdown_iters "$WARMDOWN" \
  --val_loss_every 128 --val_batch_size 16 \
  --weight_decay 0.1 --learning_rate 0.0018 \
  --seed ${4:-42} --log_wandb 2>&1 | tee "$LOG"
echo ">>> DONE rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
