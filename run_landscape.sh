#!/bin/bash

RUN_SH="${1:?usage: run_landscape.sh <run.sh> [checkpoint.pt] [extra loss_landscape.py flags...]}"
CKPT="${2:-}"

if [ ! -f "$RUN_SH" ]; then
  echo "error: $RUN_SH not found" >&2
  exit 1
fi

shift
[ $# -gt 0 ] && shift
EXTRA_ARGS=("$@")

ARGS=$(sed -n '/train_gpt2\.py/,$p' "$RUN_SH" | sed '1s/^.*train_gpt2\.py//')

if [ -z "$ARGS" ]; then
  echo "error: no 'train_gpt2.py' invocation found in $RUN_SH" >&2
  exit 1
fi

CMD="uv run loss_landscape.py${ARGS}"
if [ -n "$CKPT" ]; then
  CMD="$CMD --checkpoint $(printf '%q' "$CKPT")"
fi
for extra in "${EXTRA_ARGS[@]:-}"; do
  [ -n "$extra" ] && CMD="$CMD $(printf '%q' "$extra")"
done

echo "+ $CMD"
eval "$CMD"
