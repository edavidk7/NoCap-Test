#!/bin/bash
# Does a finer flex mask block actually pay for narrow heads?
#
# The mask block must divide the Triton kernel tile, and the tile defaults to 128 -- which
# is why BLOCK_SIZE=64 failed out of the box. It is settable via kernel_options, so the
# question is open: a 64-wide block halves the computed work for a 64-token span (two
# 64-blocks instead of two 128-blocks) while making each tile less efficient on tensor
# cores. Which dominates is a measurement.
#
# This matters because D_128floor showed the 64-token heads carry the whole quality effect
# (-0.0034 when widened to 128, the largest single effect in the sweep) -- and right now
# those heads pay 128-block cost for a 64-token span.
#
# Reference at B=64/ga=4: plain causal SDPA 911.5 ms/step, spans at default 128 block ~930.
#
# Each config runs 40 steps; the marginal rate is taken over steps 13-39 to exclude compile.
cd /home/ubuntu/NoCap-Test || exit 1
SPANS="0:64,1:64,2:64,3:64,4:128,5:128"   # the config currently training
OUT=/tmp/claude-1000/-home-ubuntu-NoCap-Test/30d907c8-e4b4-45b9-8d1e-687bea7eb6e6/scratchpad

bench () {
  local name=$1; shift
  local out
  out=$(timeout 550 uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
    --input_bin "data/fineweb10B/fineweb_train_*.bin" \
    --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
    --output_dir "$OUT/tile_$name" --model d12 \
    --batch_size 64 --sequence_length 1024 --grad_accumulation_steps 4 \
    --num_iterations 40 --warmup_iters 5 --warmdown_iters 5 \
    --val_loss_every 100 --val_batch_size 16 \
    --weight_decay 0.1 --learning_rate 0.0018 --seed 42 "$@" 2>&1)
  local a b
  a=$(echo "$out" | sed -n 's/^step:13\/40 .*train_time:\([0-9.]*\)s.*/\1/p')
  b=$(echo "$out" | sed -n 's/^step:39\/40 .*train_time:\([0-9.]*\)s.*/\1/p')
  if [ -z "$b" ]; then
    echo "$name: FAILED -- $(echo "$out" | grep -oE 'ValueError: [^\"]*|Error: [^\"]*' | head -1)"
  else
    python3 -c "print(f'{\"$name\":34} {($b-$a)/26*1000:7.1f} ms/step')"
  fi
}

echo "=== flex mask block / kernel tile sweep  $(date -u '+%H:%M UTC') ==="
bench "causal SDPA (control)"
bench "spans, block 128 (default)"  --attn_head_windows "$SPANS"
bench "spans, block 64  tile 64"    --attn_head_windows "$SPANS" --attn_block_size 64  --attn_kernel_block 64
bench "spans, block 128 tile 64"    --attn_head_windows "$SPANS" --attn_block_size 128 --attn_kernel_block 64
bench "spans, block 32  tile 32"    --attn_head_windows "$SPANS" --attn_block_size 32  --attn_kernel_block 32
bench "spans, block 64  tile 128"   --attn_head_windows "$SPANS" --attn_block_size 64
echo "=== done $(date -u '+%H:%M UTC') ==="
