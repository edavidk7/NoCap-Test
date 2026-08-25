import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_pylog(path):
    """Parse pylog124M format: 's:STEP val:VALUE' lines."""
    data = {}
    with open(path) as f:
        for line in f:
            m = re.match(r's:(\d+) val:([\d.]+)', line)
            if m:
                data[int(m.group(1))] = {'raw': float(m.group(2))}
            m = re.match(r's:(\d+) ema0\.(\d+)_val:([\d.]+)', line)
            if m:
                step = int(m.group(1))
                key = f"ema0.{m.group(2)}"
                if step not in data:
                    data[step] = {}
                data[step][key] = float(m.group(3))
    return data


def parse_train_log(path):
    """Parse train_gpt2.py tee'd log: 'step:N/T | val loss V | ema0.96 V ...'"""
    data = {}
    with open(path) as f:
        for line in f:
            m = re.match(r'step:(\d+)/\d+ \| val loss ([\d.]+)', line)
            if m:
                step = int(m.group(1))
                data[step] = {'raw': float(m.group(2))}
                for em in re.finditer(r'ema(0\.\d+) ([\d.]+)', line):
                    data[step][em.group(1)] = float(em.group(2))
    return data


def parse_wandb(run_path):
    """Fetch val loss history from wandb API."""
    try:
        import wandb
    except ImportError:
        print("wandb not installed, cannot fetch remote runs", file=sys.stderr)
        sys.exit(1)
    api = wandb.Api()
    run = api.run(run_path)
    data = {}
    for row in run.scan_history(keys=["step", "val/loss", "val/ema0.98_loss", "val/ema0.97_loss", "val/ema0.96_loss"]):
        step = row.get("step")
        val = row.get("val/loss")
        if step is not None and val is not None:
            data[step] = {'raw': val}
            for k in ["val/ema0.98_loss", "val/ema0.97_loss", "val/ema0.96_loss"]:
                v = row.get(k)
                if v is not None:
                    ema_key = k.replace("val/", "").replace("_loss", "")
                    data[step][ema_key] = v
    return data


def load_run(source):
    """Auto-detect format and load."""
    if source.startswith("wandb:"):
        return parse_wandb(source[len("wandb:"):])
    path = Path(source)
    if not path.exists():
        print(f"File not found: {source}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        first_lines = [f.readline() for _ in range(50)]
    for line in first_lines:
        if re.match(r's:\d+ ', line):
            return parse_pylog(path)
    return parse_train_log(path)


def step_to_tokens(s, total_steps=10566, B=64, T=1024, ga_schedule="0:2,0.57:4"):
    """Convert step to cumulative tokens using ga schedule."""
    stages = []
    for part in ga_schedule.split(","):
        frac, ga = part.split(":")
        stages.append((float(frac), int(ga)))
    stages.sort()

    boundaries = []
    for i, (frac, ga) in enumerate(stages):
        start = int(frac * total_steps)
        end = int(stages[i + 1][0] * total_steps) if i + 1 < len(stages) else total_steps
        boundaries.append((start, end, ga))

    tokens = 0
    for start, end, ga in boundaries:
        if s <= start:
            break
        steps_in = min(s, end) - start
        tokens += steps_in * B * T * ga
    return tokens


def main():
    parser = argparse.ArgumentParser(description="Plot val loss deltas vs baseline")
    parser.add_argument("--baseline", required=True, help="Baseline run (log path or wandb:path)")
    parser.add_argument("--runs", nargs="+", required=True, help="Run logs or wandb:paths to compare")
    parser.add_argument("--labels", nargs="+", default=None, help="Labels for each run (default: filenames)")
    parser.add_argument("--metric", default="raw", choices=["raw", "ema0.98", "ema0.97", "ema0.96"],
                        help="Which val metric to plot (default: raw)")
    parser.add_argument("--out", default="lambda_deltas.png", help="Output path (default: lambda_deltas.png)")
    parser.add_argument("--total_steps", type=int, default=10566)
    parser.add_argument("--ga_schedule", default="0:2,0.57:4")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.runs):
        print("Number of labels must match number of runs", file=sys.stderr)
        sys.exit(1)

    baseline = load_run(args.baseline)
    run_data = [(load_run(r), args.labels[i] if args.labels else Path(r).stem) for i, r in enumerate(args.runs)]

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4', '#FF5722', '#795548']

    fig, ax = plt.subplots(figsize=(14, 7))

    for idx, (data, label) in enumerate(run_data):
        common = sorted(set(data.keys()) & set(baseline.keys()))
        common = [s for s in common if args.metric in data[s] and args.metric in baseline[s]]
        if not common:
            print(f"Warning: no common steps with metric '{args.metric}' for {label}", file=sys.stderr)
            continue
        tokens = [step_to_tokens(s, args.total_steps, ga_schedule=args.ga_schedule) / 1e9 for s in common]
        deltas = [data[s][args.metric] - baseline[s][args.metric] for s in common]
        ax.plot(tokens, deltas, color=colors[idx % len(colors)], linewidth=1.8, label=label, alpha=0.9)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Tokens (billions)', fontsize=13)
    ax.set_ylabel(f'Val loss delta ({args.metric}) vs baseline', fontsize=13)
    ax.set_title(f'Val Loss Delta vs Baseline ({args.metric})', fontsize=14)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
