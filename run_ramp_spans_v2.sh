#!/bin/bash
# Ramp + spans, with more narrow heads: 6x64, 2x128, 2x512, 2x1024.
#
# Identical to the run that produced the best result so far (3.365592 in 129.7 min) except
# for the span allocation, including the default flex kernel tile -- so the comparison is
# clean. That run used 4x64, 2x128, 6x1024.
#
# Rationale for pushing on the 64s: D_128floor showed they carry the effect (widening the
# two 64-heads to 128 cost 0.0034, the largest single effect in the sweep). This asks
# whether six of them beat four, and reintroduces a 512 tier at the cost of four global
# heads.
#
# Standings, all B=64, same token budget, same seed:
#   causal                                        3.379164   127.1 m
#   batch ramp alone                              3.377058   127.5 m   -0.0021
#   C  2x64 2x128 8x1024, no ramp                 3.372455   130.8 m   -0.0067
#   A  4x64 2x128 2x512 4x1024, no ramp           3.373514   129.3 m   -0.0056
#   ramp + 4x64 2x128 6x1024                      3.365592   129.7 m   -0.0136  <- best
cd /home/ubuntu/NoCap-Test || exit 1
mkdir -p /home/ubuntu/NoCap-Test/logs
LOG=/home/ubuntu/NoCap-Test/logs/run_ramp_spans_v2.log

echo ">>> START ramp + spans 6x64 2x128 2x512 2x1024  $(date -u '+%H:%M UTC')"
uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
  --input_bin "data/fineweb10B/fineweb_train_*.bin" \
  --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
  --output_dir pylog124M --model d12 \
  --batch_size 64 --sequence_length 1024 \
  --grad_accum_schedule "0:2,0.57:4" \
  --num_iterations 11575 --warmup_iters 625 --warmdown_iters 4977 \
  --val_loss_every 128 --val_batch_size 16 \
  --weight_decay 0.1 --learning_rate 0.0018 \
  --attn_head_windows "0:64,1:64,2:64,3:64,4:64,5:64,6:128,7:128,8:512,9:512" \
  --attn_mask_impl flex \
  --seed 42 --log_wandb 2>&1 | tee "$LOG"
echo ">>> DONE rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
