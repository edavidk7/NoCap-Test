#!/bin/bash
# Can a sub-128 flex mask block be made to work, and is it worth anything?
#
# BLOCK_SIZE=64 failed earlier even with the forward tile set to 64, because flex's
# backward pass uses its own tiles (BLOCK_M1/N1/M2/N2) which were left at 128 -- a mask
# block must divide all of them. This sets both.
#
# The prize is capped: attention is ~5.7% of step time and --attn_kernel_block 64 already
# matched plain SDPA at 910.4 ms/step, so at most ~1.9% remains. Worth closing properly
# rather than leaving the earlier claim half-verified.
cd /home/ubuntu/NoCap-Test || exit 1
SPANS="0:64,1:64,2:64,3:64,4:64,5:64,6:128,7:128"
OUT=/tmp/claude-1000/-home-ubuntu-NoCap-Test/30d907c8-e4b4-45b9-8d1e-687bea7eb6e6/scratchpad

bench () {
  local name=$1; shift
  local out a b
  out=$(timeout 550 uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
    --input_bin "data/fineweb10B/fineweb_train_*.bin" \
    --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
    --output_dir "$OUT/bwd_$name" --model d12 \
    --batch_size 64 --sequence_length 1024 --grad_accumulation_steps 4 \
    --num_iterations 40 --warmup_iters 5 --warmdown_iters 5 \
    --val_loss_every 100 --val_batch_size 16 \
    --weight_decay 0.1 --learning_rate 0.0018 --seed 42 "$@" 2>&1)
  a=$(echo "$out" | sed -n 's/^step:13\/40 .*train_time:\([0-9.]*\)s.*/\1/p')
  b=$(echo "$out" | sed -n 's/^step:39\/40 .*train_time:\([0-9.]*\)s.*/\1/p')
  if [ -z "$b" ]; then
    echo "$name: FAILED -- $(echo "$out" | grep -oE 'ValueError: [^\"]*' | head -1)"
  else
    python3 -c "print(f'{\"$name\":40} {($b-$a)/26*1000:7.1f} ms/step')"
  fi
}

echo "=== backward-tile sweep  $(date -u '+%H:%M UTC') ==="
bench "causal SDPA (control)"
bench "mask 128, fwd tile 64 (current best)" --attn_head_windows "$SPANS" --attn_kernel_block 64
bench "mask 64,  fwd 64 + bwd 64"            --attn_head_windows "$SPANS" --attn_block_size 64 --attn_kernel_block 64 --attn_kernel_block_bwd 64
bench "mask 64,  bwd 64 only"                --attn_head_windows "$SPANS" --attn_block_size 64 --attn_kernel_block_bwd 64
bench "mask 32,  fwd 32 + bwd 32"            --attn_head_windows "$SPANS" --attn_block_size 32 --attn_kernel_block 32 --attn_kernel_block_bwd 32
echo "=== done $(date -u '+%H:%M UTC') ==="
