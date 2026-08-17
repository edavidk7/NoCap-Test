#!/bin/bash
# Overnight ablations on the MLP path, which has only ever been run with gelu.
#
# Every activation experiment so far was inside GLU, and GLU itself costs ~0.030 at equal
# FLOPs, so none of it transfers to the plain MLP. head_gate has never been run True in any
# logged run.
#
# All three are full length (11575 it) on the current best recipe -- B=64, ramp ga2->ga4 at
# the warmdown boundary, v1 spans, tile 64 -- so only the one flag varies and the loss is
# directly comparable to:
#     v1  4x64 2x128 6x1024   3.365592
#     E   6x64      6x1024    3.365252
#
# Not the shortened 10566 budget: that holds only 0.0024 of margin, which would turn any
# regression into a bare pass/fail with no magnitude.
#
# relu2 gets a probe first. Squared ReLU has a larger output scale and can want a lower LR;
# the one glu+relu2 run on record sat at 4.5 but was killed early, so it says nothing. The
# probe runs 60 steps at 0.0018 and at 0.0012 and takes whichever is healthier, which is a
# divergence check rather than a fine comparison -- early loss has been a poor predictor of
# final loss throughout this project, so it is only used to reject a blown-up run.
cd /home/ubuntu/NoCap-Test || exit 1
mkdir -p logs
OUT=/tmp/claude-1000/-home-ubuntu-NoCap-Test/30d907c8-e4b4-45b9-8d1e-687bea7eb6e6/scratchpad

COMMON=(--input_bin "data/fineweb10B/fineweb_train_*.bin"
        --input_val_bin "data/fineweb10B/fineweb_val_*.bin"
        --output_dir pylog124M --model d12
        --batch_size 64 --sequence_length 1024
        --grad_accum_schedule "0:2,0.57:4"
        --num_iterations 11575 --warmup_iters 625 --warmdown_iters 4977
        --val_loss_every 128 --val_batch_size 16
        --weight_decay 0.1
        --attn_head_windows "0:64,1:64,2:64,3:64,4:128,5:128"
        --attn_kernel_block 64
        --ema_decay 0.98 0.97 0.96
        --seed 42 --log_wandb)

full () {
  local name=$1; shift
  echo "=============================================================="
  echo ">>> START $name  $(date -u '+%H:%M UTC') / $(TZ=Europe/Zurich date '+%H:%M %Z')"
  echo "=============================================================="
  uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py "${COMMON[@]}" "$@" \
    2>&1 | tee "logs/night_$name.log"
  echo ">>> DONE $name rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
  sleep 20
}

# probe: 60 steps, report the loss at step 55; empty if it crashed or went non-finite
probe () {
  local lr=$1
  timeout 400 uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
    --input_bin "data/fineweb10B/fineweb_train_*.bin" \
    --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
    --output_dir "$OUT/probe" --model d12 \
    --batch_size 64 --sequence_length 1024 --grad_accumulation_steps 4 \
    --num_iterations 60 --warmup_iters 10 --warmdown_iters 10 \
    --val_loss_every 1000 --val_batch_size 16 \
    --weight_decay 0.1 --learning_rate "$lr" --seed 42 \
    --attn_head_windows "0:64,1:64,2:64,3:64,4:128,5:128" --attn_kernel_block 64 \
    --mlp_act relu2 2>&1 | sed -n 's/^step:55\/60 | loss \([0-9.]*\) .*/\1/p'
}

full silu     --learning_rate 0.0018 --mlp_act silu
full headgate --learning_rate 0.0018 --head_gate

echo "=== relu2 LR probe  $(date -u '+%H:%M UTC') ==="
L18=$(probe 0.0018); L12=$(probe 0.0012)
echo "relu2 probe @55 steps:  lr0.0018 -> '${L18:-crashed}'   lr0.0012 -> '${L12:-crashed}'"
LR=$(python3 -c "
a='${L18}'; b='${L12}'
fa=float(a) if a else 1e9
fb=float(b) if b else 1e9
import math
if not math.isfinite(fa): fa=1e9
# only prefer the lower LR if 0.0018 is clearly unhealthy, not for small differences
print('0.0012' if fa > fb + 0.30 else '0.0018')
")
echo "relu2 will use lr $LR"
full relu2 --learning_rate "$LR" --mlp_act relu2

echo ">>> NIGHT ABLATIONS COMPLETE $(date -u '+%H:%M UTC')"
