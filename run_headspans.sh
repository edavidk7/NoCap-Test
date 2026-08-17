#!/bin/bash
# Per-head attention spans, as a dense SDPA mask, on the best config.
#
# Spans: heads 0-1 see 64 tokens, 2-3 see 128, 4-5 see 512, 6-11 keep the full 1024.
#
# This is a QUALITY probe, not a speed attempt. A dense attn_mask knocks SDPA off the
# fused causal kernel and costs +24.7% (1136.5 vs 911.5 ms/step), so the run takes ~157
# min against the best config's 127.1. flex_attention is the faster vehicle for masked
# attention (-1.0% at 12 heads @128) and is where this would be re-implemented if the
# spans turn out to help the loss.
#
# Reference: identical config with plain causal attention -> 3.379164 in 127.1 min.
# Only the mask differs, so any change in final loss is attributable to the spans.
cd /home/ubuntu/NoCap-Test || exit 1
mkdir -p /home/ubuntu/NoCap-Test/logs
LOG=/home/ubuntu/NoCap-Test/logs/run_headspans.log

echo ">>> START per-head spans 2x64 2x128 2x512 6x1024  $(date -u '+%H:%M UTC')"
uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M --model d12 \
  --batch_size 64 --sequence_length 1024 --grad_accumulation_steps 4 \
  --num_iterations 8276 --warmup_iters 444 --warmdown_iters 3555 \
  --val_loss_every 128 --val_batch_size 16 \
  --weight_decay 0.1 --learning_rate 0.0018 \
  --attn_head_windows "0:64,1:64,2:128,3:128,4:512,5:512" \
  --seed 42 --log_wandb 2>&1 | tee "$LOG"
echo ">>> DONE rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
