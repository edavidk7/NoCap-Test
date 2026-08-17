#!/bin/bash
# The two untested endpoints of "how narrow should the narrow heads be", at full length.
#
# All three of these cost exactly 306 block-units -- 128 and 64 occupy the same two key
# blocks per query row at 128 granularity -- so this is a pure quality question with no
# speed component. v1 sits between them and is measured; the endpoints are not.
#
#   6x64  6x1024   all narrow heads at the block floor      untested
#   4x64  2x128  6x1024   (v1)                              3.365592
#   6x128 6x1024   all narrow heads at 128                  untested
#
# D_128floor is the only prior evidence: widening two heads from 64 to 128, everything
# else held, cost 0.0034. That points toward 64 -- but it was measured without the ramp
# and is a single data point, which is why both endpoints get run rather than just one.
#
# Full length deliberately. The shortened schedule is sized against v1's specific margin;
# a different span config needs its own measurement before its budget can be set.
#
# Everything else is the v1/v3 recipe: B=64, ramp ga2->ga4 at the warmdown boundary,
# 11575 it, tile 64, seed 42.
cd /home/ubuntu/NoCap-Test || exit 1
mkdir -p /home/ubuntu/NoCap-Test/logs

COMMON=(--input_bin "data/fineweb10B/fineweb_train_*.bin"
        --input_val_bin "data/fineweb10B/fineweb_val_*.bin"
        --output_dir pylog124M --model d12
        --batch_size 64 --sequence_length 1024
        --grad_accum_schedule "0:2,0.57:4"
        --num_iterations 11575 --warmup_iters 625 --warmdown_iters 4977
        --val_loss_every 128 --val_batch_size 16
        --weight_decay 0.1 --learning_rate 0.0018
        --attn_mask_impl flex --attn_kernel_block 64
        --ema_decay 0.98 0.97 0.96
        --seed 42 --log_wandb)

run () {
  local name=$1 spec=$2
  echo "=============================================================="
  echo ">>> START $name  spans '$spec'  $(date -u '+%H:%M UTC') / $(TZ=Europe/Zurich date '+%H:%M %Z')"
  echo "=============================================================="
  uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py "${COMMON[@]}" \
    --attn_head_windows "$spec" 2>&1 | tee "/home/ubuntu/NoCap-Test/logs/span_$name.log"
  echo ">>> DONE $name rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
  sleep 20
}

run E_6x64  "0:64,1:64,2:64,3:64,4:64,5:64"
run F_6x128 "0:128,1:128,2:128,3:128,4:128,5:128"

echo ">>> SPAN ENDPOINTS COMPLETE $(date -u '+%H:%M UTC')"
