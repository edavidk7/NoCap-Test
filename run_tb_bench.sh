#!/bin/bash
# Throughput over physical sequence length T and batch size B, at fixed tokens/update.
#
#   usage: bash run_tb_bench.sh ["<span spec>"]
#
# Every config holds B x T x ga = 262,144 tokens per optimizer update, so ms/step is
# directly comparable and the only thing changing is how those tokens are shaped.
#
# Short runs: 24 iterations, marginal rate taken over steps 13-23. That is 10 steps, so
# expect ~1% scatter -- enough to rank configs, not to separate ones within a few percent.
# Most of the wall time here is torch.compile, once per (T, B) shape.
#
# Each shape is run twice, plain causal and with the span mask, because the mask's value
# should shrink as T falls: a windowed head is already linear in T, so shortening T mostly
# removes work from the *global* heads. At T=256 there are only two 128-blocks per row and
# almost nothing left for the mask to skip, so expect the two to converge there for
# mechanical reasons rather than modelling ones.
cd /home/ubuntu/NoCap-Test || exit 1
SPANS=${1:-"0:64,1:64,2:64,3:64,4:128,5:128"}
OUT=/tmp/claude-1000/-home-ubuntu-NoCap-Test/30d907c8-e4b4-45b9-8d1e-687bea7eb6e6/scratchpad

bench () {
  local name=$1 T=$2 B=$3 GA=$4; shift 4
  local out a b m
  out=$(timeout 550 uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py \
    --input_bin "data/fineweb10B/fineweb_train_*.bin" \
    --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
    --output_dir "$OUT/tb" --model d12 \
    --batch_size "$B" --sequence_length "$T" --grad_accumulation_steps "$GA" \
    --num_iterations 24 --warmup_iters 3 --warmdown_iters 3 \
    --val_loss_every 100 --val_batch_size 16 \
    --weight_decay 0.1 --learning_rate 0.0018 --seed 42 "$@" 2>&1)
  a=$(echo "$out" | sed -n 's/^step:13\/24 .*train_time:\([0-9.]*\)s.*/\1/p')
  b=$(echo "$out" | sed -n 's/^step:23\/24 .*train_time:\([0-9.]*\)s.*/\1/p')
  m=$(echo "$out" | sed -n 's/^peak memory consumption: \(.*\)/\1/p')
  if [ -z "$b" ]; then
    echo "$name  T=$T B=$B ga=$GA : FAILED -- $(echo "$out" | grep -oE '(ValueError|RuntimeError|AttributeError|TypeError|error): [^\"]{0,90}' | head -1)"
  else
    python3 -c "print(f'{\"$name\":8} T={$T:<5} B={$B:<4} ga={$GA:<2} {($b-$a)/10*1000:8.1f} ms/step   peak $m')"
  fi
}

echo "=== T/B throughput, spans '$SPANS'  $(date -u '+%H:%M UTC') ==="
for cfg in "2048 32 4" "1024 64 4" "512 128 4" "512 64 8" "256 256 4" "256 128 8"; do
  set -- $cfg
  bench "causal" "$1" "$2" "$3"
  bench "spans"  "$1" "$2" "$3" --attn_head_windows "$SPANS" --attn_kernel_block 64
done
echo "=== done $(date -u '+%H:%M UTC') ==="
