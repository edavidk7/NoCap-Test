#!/bin/bash
# Sliding-window attention, quality test.
#
# B16 x T2048 x ga16 = 524,288 tokens/update, 4138 iterations = 2.17B tokens, so the same
# token budget as the best config reached in half the steps. Warmup/warmdown are scaled to
# keep the same fractions of the run (5.4% / 43%).
#
# Two things differ from the best config, not one: the attention span, and the update
# frequency. 524,288 tokens/update is the point the ga sweep measured at 3.379890 against
# 262,144's 3.371701, so ~0.008 of the result is already spoken for.
#
# Wall clock is set by micro-batch work, not step count: 4138 x 16 = 66,208 micro-batches
# of 32,768 tokens, and a T=2048 micro-batch costs 2.12x a T=1024 one. That is ~144 min
# against the best config's 130, not the 2x speedup halving the steps suggests.
#
# Comparisons, all at the same 2.17B token budget:
#   T=1024, full causal, ga16 -> 3.379210  (131 min)  every sequence starts cold
#   T=2048, full causal, ga8  -> 3.385583  (145 min)  later tokens get up to 2048 context
#   T=2048, window 1024, ga16 -> this run  (~144 min) later tokens get exactly 1024
cd /home/ubuntu/NoCap-Test || exit 1
mkdir -p /home/ubuntu/NoCap-Test/logs
LOG=/home/ubuntu/NoCap-Test/logs/run_flexwin.log

echo ">>> START flex window 1024 @ T=2048  $(date -u '+%H:%M UTC')"
uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M --model d12 \
  --batch_size 16 --sequence_length 2048 --grad_accumulation_steps 16 \
  --num_iterations 4138 --warmup_iters 222 --warmdown_iters 1778 \
  --val_loss_every 128 --val_batch_size 16 \
  --weight_decay 0.1 --learning_rate 0.0018 \
  --attn_window 1024 \
  --seed 42 --log_wandb 2>&1 | tee "$LOG"
echo ">>> DONE rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
