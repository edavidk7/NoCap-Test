#!/bin/bash
# Data-driven per-head attention spans, via flex_attention block masks.
#
# The first spans run (2x64, 2x128, 2x512, 6x1024, dense mask) gave 3.373024 against
# causal's 3.379164 -- a real -0.0061 that held through the entire warmdown. This tests a
# span allocation chosen from the corpus statistics instead of by hand, on the faster
# implementation.
#
# Allocation: 4x128, 4x256, 2x512, 2x1024. Reasoning from analyze_context.py (2M tokens):
#   - 128 covers 99.6% of sentences, so syntax-local heads need nothing more. It is also
#     the flex block floor, so spans below 128 cost the same while seeing less -- the
#     previous run's two heads at 64 were giving up coverage for free.
#   - rare-token copy distance has no saturation point: coverage is 57.6 / 70.7 / 81.2 /
#     88.9% at 128 / 256 / 512 / 1024, so each widening still buys 8-13 points. Tiers are
#     therefore spread rather than clustered low.
#   - two heads stay global for the ~11% of rare copies beyond 512.
#
# flex rather than dense: a dense SDPA mask costs +24.7% (it cannot skip masked work),
# while flex skips blocks at a fixed ~43 ms/step overhead. With only 4 heads at the
# narrowest tier this is likely still slightly slower than plain causal, but far cheaper
# than dense -- and the -0.0061 loss gain is worth ~4% of wall clock if it survives.
#
# Reference: 3.379164 in 127.1 min, identical config with plain causal attention.
cd /home/ubuntu/NoCap-Test || exit 1
mkdir -p /home/ubuntu/NoCap-Test/logs
LOG=/home/ubuntu/NoCap-Test/logs/run_headspans_dd.log

echo ">>> START data-driven spans 4x128 4x256 2x512 2x1024 (flex)  $(date -u '+%H:%M UTC')"
uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M --model d12 \
  --batch_size 64 --sequence_length 1024 --grad_accumulation_steps 4 \
  --num_iterations 8276 --warmup_iters 444 --warmdown_iters 3555 \
  --val_loss_every 128 --val_batch_size 16 \
  --weight_decay 0.1 --learning_rate 0.0018 \
  --attn_head_windows "0:128,1:128,2:128,3:128,4:256,5:256,6:256,7:256,8:512,9:512" \
  --attn_mask_impl flex \
  --seed 42 --log_wandb 2>&1 | tee "$LOG"
echo ">>> DONE rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
