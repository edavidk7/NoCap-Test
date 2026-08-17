#!/bin/bash
# Confirmation run for the micro-batch throughput finding.
#
# Identical to the best config in every way that affects the optimization: B x T x ga is
# still 262,144 tokens per update, the same 8276 iterations, the same schedule and seed.
# Only the split between batch size and accumulation changes, 16x16 -> 64x4, which is a
# pure throughput change that exact arithmetic would leave untouched.
#
# Benchmarked at constant tokens/update:
#   B=16  ga=16   940.4 ms/step   10.6 GB
#   B=32  ga=8    948.5 ms/step   18.0 GB
#   B=64  ga=4    911.5 ms/step   32.3 GB   <- this run, -3.1%
#   B=128 ga=2    935.0 ms/step   62.3 GB
#
# Expect ~126 min against the reference's 131.3, and a val loss within the 0.0009
# fixed-seed noise floor of 3.379210. Anything further from that means the accumulation
# reduction order matters more than exact arithmetic suggests, which is the reason this
# run exists rather than just claiming the benchmark.
cd /home/ubuntu/NoCap-Test || exit 1
mkdir -p /home/ubuntu/NoCap-Test/logs
LOG=/home/ubuntu/NoCap-Test/logs/run_bs64.log

echo ">>> START B=64 ga=4  $(date -u '+%H:%M UTC')"
uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M --model d12 \
  --batch_size 64 --sequence_length 1024 --grad_accumulation_steps 4 \
  --num_iterations 8276 --warmup_iters 444 --warmdown_iters 3555 \
  --val_loss_every 128 --val_batch_size 16 \
  --weight_decay 0.1 --learning_rate 0.0018 \
  --seed 42 --log_wandb 2>&1 | tee "$LOG"
echo ">>> DONE rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
