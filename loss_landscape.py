import json
import os
import subprocess
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import train_gpt2 as tg

# -----------------------------------------------------------------------------
# MPS-compatible attention: replace flex_attention with SDPA + explicit mask

def _sdpa_forward(self, x, cos, sin, attn_mask=None):
    B, T, C = x.size()
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, self.head_dim)
    q = q.view(B, T, self.n_head, self.head_dim)
    v = v.view(B, T, self.n_head, self.head_dim)
    q = tg.apply_rotary_emb(q, cos, sin)
    k = tg.apply_rotary_emb(k, cos, sin)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if attn_mask is not None:
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    else:
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return self.c_proj(y)


tg.CausalSelfAttention.forward = _sdpa_forward


def build_head_window_mask(windows, T, device):
    q_idx = torch.arange(T, device=device).view(T, 1)
    k_idx = torch.arange(T, device=device).view(1, T)
    causal = k_idx <= q_idx
    masks = [causal & (q_idx - k_idx < w) for w in windows]
    return torch.stack(masks, dim=0).unsqueeze(0)


def build_model_masks(model, T, device):
    cfg = model.config
    specs = cfg.attn_head_windows
    if not specs:
        model.block_masks = {}
        return
    if isinstance(specs, str):
        specs = [specs]
    specs = tg.per_block_specs(specs, cfg.n_layer)
    for spec in dict.fromkeys(specs):
        blocks = [i for i, s in enumerate(specs) if s == spec]
        windows = tg.parse_head_windows(spec, cfg.n_head, cfg.max_seq_len)
        print(f"attention spans, blocks {blocks}: {windows}")
    cache = {}
    for spec in dict.fromkeys(specs):
        windows = tg.parse_head_windows(spec, cfg.n_head, cfg.max_seq_len)
        cache[spec] = build_head_window_mask(windows, T, device)
    model.block_masks = {T: [cache[spec] for spec in specs]}


# -----------------------------------------------------------------------------
# Model construction: identical to train_gpt2.py's own, given the same args

def build_model(args, device):
    model_config = tg.build_model_config(args)
    seed = args.seed if args.seed is not None else 0
    torch.manual_seed(seed)
    model = tg.GPT(model_config)
    model = model.to(device=device, dtype=torch.float32)

    if args.checkpoint:
        print(f"loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        # torch.compile wraps the model in an OptimizedModule, whose state_dict
        # keys are prefixed "_orig_mod." -- strip it so keys match our plain model
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    model.eval()
    build_model_masks(model, args.sequence_length, device)
    return model


# -----------------------------------------------------------------------------
# Real-data batch from the FineWeb val shard

def load_val_batch(B, T, device):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fineweb10B", "fineweb_val_000000.bin")
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "bad magic in data shard"
        ntok = header[2]
        tokens = np.frombuffer(f.read(ntok * 2), dtype=np.uint16)
    need = B * T + 1
    tokens = tokens[:need].astype(np.int64)
    buf = torch.from_numpy(tokens)
    x = buf[:-1].view(B, T).to(device)
    y = buf[1:].view(B, T).to(device)
    return x, y


# -----------------------------------------------------------------------------
# Weight-group addressing (Q/K/V share c_attn.weight, split by row)

WEIGHT_GROUPS = {
    "Q": ("attn.c_attn.weight", (0, 1)),
    "K": ("attn.c_attn.weight", (1, 2)),
    "V": ("attn.c_attn.weight", (2, 3)),
    "MLP": ("mlp.c_fc.weight", None),
}


def get_full_weight(model, block_idx, group_name):
    attr_path, row_frac = WEIGHT_GROUPS[group_name]
    block = model.transformer.h[block_idx]
    obj = block
    for part in attr_path.split(".")[:-1]:
        obj = getattr(obj, part)
    weight = getattr(obj, attr_path.split(".")[-1])
    return weight, row_frac


def filter_normalized_direction(theta: torch.Tensor, seed: int) -> torch.Tensor:
    """Li et al. (2018), "Visualizing the Loss Landscape of Neural Nets":
    a random Gaussian direction, filter-normalized so each row (each output
    neuron's weight vector -- a Linear layer's "filter") is rescaled to the
    same norm as the corresponding row of theta:

        d_j <- (d_j / ||d_j||) * ||theta_j||

    This is what makes loss-landscape sharpness/flatness comparable across
    weights that live at different scales (an FC layer's rows are its
    filters, per the paper's own note that this generalizes straight from
    Conv layers) -- without it, a large-weight layer looks artificially flat
    and a small-weight layer looks artificially sharp, purely from ReLU/
    RMSNorm-style scale invariance, not from anything about the loss itself.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = torch.randn(theta.shape, generator=g).to(theta.device, theta.dtype)
    d_norms = d.norm(dim=1, keepdim=True).clamp_min(1e-12)
    theta_norms = theta.norm(dim=1, keepdim=True)
    return d * (theta_norms / d_norms)


@torch.no_grad()
def landscape(model, x, y, block_idx, group_name, grid, alpha_max, device, seed_a, seed_b):
    full_weight, row_frac = get_full_weight(model, block_idx, group_name)
    n_embd = model.config.n_embd
    if row_frac is None:
        lo, hi = 0, full_weight.shape[0]
    else:
        lo, hi = row_frac[0] * n_embd, row_frac[1] * n_embd
    w_init = full_weight[lo:hi].clone()

    A = filter_normalized_direction(w_init, seed_a)
    B = filter_normalized_direction(w_init, seed_b)

    axis0 = np.linspace(0.0, alpha_max, grid)  # alpha, coefficient on A
    axis1 = np.linspace(0.0, alpha_max, grid)  # beta, coefficient on B

    ce_grid = np.zeros((grid, grid), dtype=np.float32)
    aux_grid = np.zeros((grid, grid), dtype=np.float32)

    aux_lambda = model.config.emb_aux_lambda

    # mininterval throttles refresh rate regardless of how fast points
    # complete -- when stdout isn't a real tty (e.g. piped to a log file by
    # a background-task runner), tqdm can't overwrite in place and instead
    # emits a new line per refresh, so an unthrottled per-point bar at
    # ~2 pt/s over 225 points floods the log; capped at 1 refresh/5s here
    pbar = tqdm(total=grid * grid, desc=f"block{block_idx} {group_name}", leave=False,
                unit="pt", mininterval=5.0)
    for i, alpha in enumerate(axis0):
        for j, beta in enumerate(axis1):
            full_weight[lo:hi] = w_init + alpha * A + beta * B
            _, ce, aux = model(x, y, return_logits=False)
            ce_grid[i, j] = ce.item()
            aux_grid[i, j] = ce.item() + aux_lambda * aux.item() if aux is not None else ce.item()
            pbar.update(1)
    pbar.close()
    full_weight[lo:hi] = w_init
    return axis0, axis1, ce_grid, aux_grid


RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loss_landscape_results")


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode().strip()
    except Exception:
        return None


def build_landscape_arg_parser():
    parser = tg.build_arg_parser()
    parser.add_argument("--grid", type=int, default=15)
    parser.add_argument("--alpha-max", type=float, default=4.0)
    parser.add_argument(
        "--landscape-batch", type=int, default=4,
        help="batch size for the landscape's own loss eval -- independent of --batch_size, "
        "which is a training-loop arg unused here",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="label appended to the timestamped run directory, e.g. --tag conic",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="path to a train_gpt2.py checkpoint (.pt, as saved by torch.save(dict(model=state_dict, ...))) "
        "to use as W_init instead of a fresh random init",
    )
    return parser


def main():
    args = build_landscape_arg_parser().parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{stamp}_{args.tag}" if args.tag else stamp
    out_dir = os.path.join(RESULTS_ROOT, run_dir_name, "data")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    model = build_model(args, device)
    x, y = load_val_batch(args.landscape_batch, args.sequence_length, device)

    blocks = [0, 5, 11]
    groups = ["Q", "K", "V", "MLP"]

    seed = args.seed if args.seed is not None else 0

    config = {
        "grid": args.grid,
        "alpha_max": args.alpha_max,
        "landscape_batch": args.landscape_batch,
        "seqlen": args.sequence_length,
        "blocks": blocks,
        "groups": groups,
        "model": args.model,
        "attn_head_windows": args.attn_head_windows,
        "ff_kind": args.ff_kind,
        "mlp_act": args.mlp_act,
        "emb_aux_lambda": args.emb_aux_lambda[0],
        "seed": seed,
        "checkpoint": args.checkpoint,
        "git_commit": git_commit(),
        "started_at": stamp,
        "device": str(device),
        "combination": "W = W_init + alpha*A + beta*B (conic, alpha,beta >= 0)",
        "direction_normalization": "filter-wise (Li et al. 2018): each row of A/B rescaled to the matching row-norm of W_init",
    }
    with open(os.path.join(RESULTS_ROOT, run_dir_name, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    t_start = time.time()
    total = len(blocks) * len(groups)
    pairs = [(b, gi, g) for b in blocks for gi, g in enumerate(groups)]
    outer = tqdm(pairs, total=total, desc="landscape sweep", unit="group", mininterval=1.0)
    for done, (b, gi, g) in enumerate(outer, start=1):
        outer.set_postfix_str(f"block {b} {g}")
        t0 = time.time()
        # every A/B matrix is seeded off --seed, offset per (block, group)
        # so they differ from each other but the whole sweep reproduces
        # from one seed
        seed_a = seed * 1_000_000 + 1000 * b + 10 * gi + 1
        seed_b = seed * 1_000_000 + 1000 * b + 10 * gi + 2
        axis0, axis1, ce_grid, aux_grid = landscape(
            model, x, y, b, g, args.grid, args.alpha_max, device, seed_a, seed_b
        )
        dt = time.time() - t0
        fname = os.path.join(out_dir, f"block{b}_{g}.npz")
        np.savez(fname, axis0=axis0, axis1=axis1, ce=ce_grid, aux=aux_grid)
        tqdm.write(f"[{done}/{total}] block {b:>2} {g:<3} done in {dt:5.1f}s -> {fname}")
    outer.close()
    total_s = time.time() - t_start
    print(f"total wall clock: {total_s:.1f}s")

    config["total_wall_clock_s"] = total_s
    with open(os.path.join(RESULTS_ROOT, run_dir_name, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"run directory: {os.path.join(RESULTS_ROOT, run_dir_name)}")


if __name__ == "__main__":
    main()
