"""Measure the attention span each (layer, head) of a trained checkpoint actually uses.

    uv run --with matplotlib python measure_attn_span.py --ckpt logs/<uuid>/final.pt

Rather than guessing a span schedule from corpus statistics or from priors about depth,
this recomputes the real attention weights on validation data and asks, per head: how far
back does its probability mass actually go?

Reported per head is the distance within which a given fraction of attention mass falls,
averaged over query positions (early positions are excluded, since a token at position 20
cannot attend further than 20 and would bias every head short).

Runs on CPU by default so it does not contend with training.
"""

import argparse
import glob

import numpy as np
import torch
import torch.nn.functional as F


def load_model(ckpt_path, device):
    import train_gpt2 as T

    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = blob["args"]
    vocab = blob["model"]["_orig_mod.transformer.wte.weight"].shape[0]
    cfg = T.GPTConfig(
        vocab_size=vocab, n_layer=12, n_head=12, n_embd=768,
        mlp_alpha=args.get("mlp_alpha", [4.0])[0] if isinstance(args.get("mlp_alpha"), list) else 4.0,
        ff_kind=args.get("ff_kind", "mlp"), mlp_act=args.get("mlp_act", "gelu"),
        mlp_drop_n=args.get("mlp_drop_n", 1), max_seq_len=1024,
    )
    model = T.GPT(cfg)
    # checkpoints are saved from the torch.compile wrapper, which prefixes every key
    sd = {k.removeprefix("_orig_mod."): v for k, v in blob["model"].items()}
    model.load_state_dict(sd)
    return model.to(device).eval(), args


def val_batch(batch, seq_len, device):
    path = sorted(glob.glob("data/fineweb10B/fineweb_val_*.bin"))[0]
    with open(path, "rb") as fh:
        fh.read(256 * 4)
        tok = np.frombuffer(fh.read(batch * seq_len * 2), dtype=np.uint16).astype(np.int64)
    return torch.tensor(tok[: batch * seq_len], device=device).view(batch, seq_len)


@torch.no_grad()
def attention_distances(model, idx, quantiles, min_query_pos):
    """For each (layer, head), the distance containing each quantile of attention mass."""
    import train_gpt2 as T

    B, TT = idx.shape
    x = model.transformer.wte(idx)
    cos, sin = model.rotary(x)
    offs = torch.arange(TT).view(-1, 1) - torch.arange(TT).view(1, -1)  # query - key
    out = []

    for block in model.transformer.h:
        attn = block.attn
        h = T.rmsnorm(x)
        qkv = attn.c_attn(h)
        q, k, v = qkv.split(attn.n_embd, dim=2)
        q = T.apply_rotary_emb(q.view(B, TT, attn.n_head, attn.head_dim), cos, sin).transpose(1, 2)
        k = T.apply_rotary_emb(k.view(B, TT, attn.n_head, attn.head_dim), cos, sin).transpose(1, 2)
        v = v.view(B, TT, attn.n_head, attn.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (attn.head_dim ** 0.5)
        scores = scores.masked_fill(offs < 0, float("-inf"))
        w = scores.softmax(-1)[:, :, min_query_pos:, :]  # (B, H, Q, K)

        # cumulative mass as a function of lookback distance, per query
        d = offs[min_query_pos:, :].clamp(min=0)
        order = d.argsort(dim=-1)                                  # nearest key first
        wsort = torch.gather(w, -1, order.expand(B, attn.n_head, -1, -1))
        dsort = torch.gather(d, -1, order)
        cum = wsort.cumsum(-1)

        per_head = []
        for qi in quantiles:
            hit = (cum >= qi).float().argmax(-1)                   # first index reaching the quantile
            dist = torch.gather(dsort.expand(B, attn.n_head, -1, -1), -1, hit.unsqueeze(-1)).squeeze(-1)
            per_head.append(dist.float().mean(dim=(0, 2)))         # average over batch and query
        out.append(torch.stack(per_head, 0))                       # (Q, H)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).contiguous().view(B, TT, -1)
        x = x + block.attn_scale * attn.c_proj(y)
        if not block.no_ffn:
            x = x + block.mlp(T.rmsnorm(x))
        else:
            x = x + T.rmsnorm(x)
    return torch.stack(out, 0).numpy()                             # (L, Q, H)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quantiles", type=float, nargs="+", default=[0.5, 0.9])
    ap.add_argument("--min_query_pos", type=int, default=512,
                    help="ignore early query positions, which cannot attend far and bias every head short")
    ap.add_argument("--out", default="attn_span.png")
    args = ap.parse_args()

    model, cfg = load_model(args.ckpt, args.device)
    idx = val_batch(args.batch, args.seq_len, args.device)
    print(f"checkpoint spans arg: {cfg.get('attn_head_windows', '(none - plain causal)')}")
    dist = attention_distances(model, idx, args.quantiles, args.min_query_pos)

    for qi, q in enumerate(args.quantiles):
        print(f"\n=== distance containing {q:.0%} of attention mass, per layer x head ===")
        print("layer " + " ".join(f"{h:>5}" for h in range(dist.shape[2])) + "    mean  median")
        for l in range(dist.shape[0]):
            row = dist[l, qi]
            print(f"{l:5d} " + " ".join(f"{v:5.0f}" for v in row) + f"  {row.mean():6.0f}  {np.median(row):6.0f}")
        print("layer means by depth third: "
              f"early {dist[:4, qi].mean():.0f}   mid {dist[4:8, qi].mean():.0f}   late {dist[8:, qi].mean():.0f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(args.quantiles), figsize=(6 * len(args.quantiles), 4.5), squeeze=False)
    for qi, q in enumerate(args.quantiles):
        ax = axes[0][qi]
        im = ax.imshow(dist[:, qi], aspect="auto", cmap="viridis", origin="lower")
        ax.set_xlabel("head"); ax.set_ylabel("layer"); ax.set_title(f"distance holding {q:.0%} of attention mass")
        fig.colorbar(im, ax=ax, label="tokens")
    fig.suptitle(f"Effective attention span per layer x head  ({args.ckpt})")
    fig.tight_layout(); fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
