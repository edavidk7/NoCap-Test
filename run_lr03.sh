#!/bin/bash
# Peak-LR probe: identical to the glu/silu/drop_n2/alpha6 run at lr 0.0022, with the
# learning rate raised to 0.03 (13.6x). Single variable changed.
cd /home/ubuntu/NoCap-Test || exit 1
mkdir -p /home/ubuntu/NoCap-Test/logs
LOG=/home/ubuntu/NoCap-Test/logs/run_lr03.log

LR=${1:-0.03}

echo ">>> START glu/silu lr ${LR}  $(date -u '+%H:%M UTC')"
uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M --model d12 \
  --batch_size 16 --grad_accumulation_steps 16 --sequence_length 1024 \
  --val_loss_every 128 --val_batch_size 16 \
  --num_iterations 8276 --weight_decay 0.1 \
  --learning_rate "$LR" \
  --warmup_iters 444 --warmdown_iters 3555 \
  --log_wandb --ff_kind glu --seed 42 \
  --mlp_drop_n 2 --mlp_alpha 6.0 --mlp_act silu --ema_decay 0.996 2>&1 | tee "$LOG"
echo ">>> DONE rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
