#!/bin/bash
# FFN architecture ablation. Every hyperparameter is held at the values the already-run
# variants used, so all five are directly comparable:
#   lr 0.0018, wd 0.1, 8276 it, warmup 444, warmdown 3555, B16 x ga16 x T1024,
#   seed 42, ema 0.998, val every 128.
# Only ff_kind / mlp_act / mlp_alpha / mlp_drop_n vary.
#
# Already measured, not re-run here:
#   Your proposal  glu+silu a6 n2 -> ~3.4447 (raw, at step 8192)
#   SwiGLU matched glu+silu a4 n1 -> running when this queue was written
#   Baseline       mlp      a4 n1 -> 3.380107 raw / 3.380984 ema  (queued last anyway,
#                                    as a duplicate you can kill)
#
# Note on widths: alpha is interpreted per-family. mlp a4 -> hidden 3072;
# glu a4 -> hidden 2048 (the 2/3 GLU convention). So "SwiGLU matched" matches alpha,
# not parameter count -- it is the *smaller* FFN.
cd /home/ubuntu/NoCap-Test || exit 1
mkdir -p /home/ubuntu/NoCap-Test/logs

COMMON=(--input_bin "data/fineweb10B/fineweb_train_*.bin"
        --input_val_bin "data/fineweb10B/fineweb_val_*.bin"
        --output_dir pylog124M --model d12
        --batch_size 16 --grad_accumulation_steps 16 --sequence_length 1024
        --val_loss_every 128 --val_batch_size 16
        --num_iterations 8276 --weight_decay 0.1 --learning_rate 0.0018
        --warmup_iters 444 --warmdown_iters 3555
        --seed 42 --ema_decay 0.998 --log_wandb)

run () {
  local name=$1; shift
  echo "=============================================================="
  echo ">>> START $name  $(date -u '+%H:%M UTC') / $(TZ=Europe/Zurich date '+%H:%M %Z')"
  echo "=============================================================="
  uv run torchrun --standalone --nproc_per_node=1 train_gpt2.py "${COMMON[@]}" "$@" \
    2>&1 | tee "/home/ubuntu/NoCap-Test/logs/ffn_$name.log"
  echo ">>> DONE $name rc=${PIPESTATUS[0]}  $(date -u '+%H:%M UTC')"
  sleep 20
}

# Drop-only control: isolates removing every other FFN, with no gating change.
run drop_only    --ff_kind mlp --mlp_drop_n 2

# Similar FFN budget: gated and sparse, at the narrower GLU width.
run glu_a4_n2    --ff_kind glu --mlp_act silu --mlp_drop_n 2

echo ">>> ABLATION QUEUE COMPLETE $(date -u '+%H:%M UTC')"

# --------------------------------------------------------------------------------
# Hyperparameter probes for the alpha=6 / drop_n=2 / silu architecture.
#
# That config finished at 3.442892 in 118.3 min. Time-matched against the baseline
# (which is 9.4% slower per step: 946.3 vs 857.7 ms) the gap is ~0.048, so these are
# looking for a much larger effect than any knob has produced so far. Both probes hold
# the micro-batch count at 132,416 -- the same 118.3 min -- so they are comparable to
# the run above and to each other.
A6=(--ff_kind glu --mlp_act silu --mlp_alpha 6.0 --mlp_drop_n 2)

# P1: LR is the one knob never swept for this architecture. Fewer FFN params (~75% of
# baseline) plausibly wants a higher rate.
run a6n2_lr0028  "${A6[@]}" --learning_rate 0.0028

# P2: fewer params -> smaller critical batch. ga 12 with 11,035 iterations is the same
# 132,416 micro-batches, with warmup/warmdown held at the same fractions (5.4% / 43%).
run a6n2_ga12    "${A6[@]}" --grad_accumulation_steps 12 --num_iterations 11035 \
                            --warmup_iters 592 --warmdown_iters 4741

echo ">>> A6N2 PROBES COMPLETE $(date -u '+%H:%M UTC')"
