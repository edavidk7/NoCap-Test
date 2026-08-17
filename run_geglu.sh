#!/bin/bash
# GEGLU (glu + gelu, rmsnorm on the gated state removed) dropped into the current best
# time-to-target config: ga=16, 8969 it, warmup 482, warmdown 1927, lr 0.0018, seed 42.
# The MLP version of this exact config gave 3.380843 in 140.9 min (-5.8% vs baseline).
# Parameter-matched: GLU alpha=4 -> d_ff 2048 == MLP alpha=4 -> d_ff 3072.
cd /home/ubuntu/NoCap-Test || exit 1
LOG=/tmp/claude-1000/-home-ubuntu-NoCap-Test/cc2fbcec-5291-4690-b18e-6818029acfb7/scratchpad/run_geglu_8969.log
uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M --model d12 \
  --batch_size 16 --sequence_length 1024 \
  --grad_accumulation_steps 16 --num_iterations 8969 \
  --warmup_iters 482 --warmdown_iters 1927 \
  --val_loss_every 128 --val_batch_size 16 \
  --weight_decay 0.1 --learning_rate 0.0018 \
  --ff_kind glu --mlp_act gelu --mlp_alpha 4.0 --mlp_drop_n 1 \
  --seed 42 --log_wandb 2>&1 | tee "$LOG"
echo ">>> GEGLU DONE rc=${PIPESTATUS[0]} $(date -u '+%H:%M UTC')"
