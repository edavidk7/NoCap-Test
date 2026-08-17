"""Plot the token copy-distance distribution for train and val, side by side.

    uv run --with matplotlib python plot_copy_distance.py --tokens 2000000
    uv run --with matplotlib python plot_copy_distance.py --tokens 10000000 --out cd10M.png

Copy distance is, for each token, the gap back to its previous occurrence. It is the
signal induction/copying heads exploit, so the cumulative curve says what fraction of that
signal a head with a given attention span can reach.

"Rare" tokens are those occurring fewer than --rare_below times in the sample. Common
tokens repeat every few positions and drag the distribution down while carrying almost no
information, so the rare-token curve is the decision-relevant one.

Train and val are drawn on the same axes limits so the panels can be compared directly --
a mismatch between them would mean span choices tuned on train do not transfer to what is
actually scored.
"""

import argparse
import glob

import numpy as np


def load_tokens(split, n_tokens):
    """Read the first n_tokens from the first shard of a split."""
    paths = sorted(glob.glob(f"data/fineweb10B/fineweb_{split}_*.bin"))
    assert paths, f"no shards found for split '{split}'"
    out, want = [], n_tokens
    for path in paths:  # val is a single shard; train needs several for large N
        with open(path, "rb") as fh:
            fh.read(256 * 4)  # header
            chunk = np.frombuffer(fh.read(want * 2), dtype=np.uint16).astype(np.int64)
        out.append(chunk)
        want -= len(chunk)
        if want <= 0:
            break
    return np.concatenate(out)


def copy_distances(tok, max_dist, rare_below):
    """Gap to each token's previous occurrence, split into all tokens and rare tokens."""
    freq = np.bincount(tok, minlength=50257)
    last = {}
    every, rare = [], []
    for i, t in enumerate(tok):
        t = int(t)
        prev = last.get(t)
        if prev is not None:
            d = i - prev
            if d <= max_dist:
                every.append(d)
                if freq[t] < rare_below:
                    rare.append(d)
        last[t] = i
    return np.array(every), np.array(rare)


def panel(ax, every, rare, title, spans, max_dist):
    bins = np.logspace(0, np.log10(max_dist), 60)
    ax.hist(every, bins=bins, alpha=0.55, label=f"all tokens (n={len(every):,})", color="tab:blue")
    ax.hist(rare, bins=bins, alpha=0.55, label=f"rare tokens (n={len(rare):,})", color="tab:orange")
    ax.set_xscale("log")
    ax.set_xlabel("distance to previous occurrence (tokens)")
    ax.set_title(title)
    for s in spans:
        if s <= max_dist:
            ax.axvline(s, color="grey", ls="--", lw=0.8)
            frac = 100 * (rare <= s).mean() if len(rare) else 0.0
            ax.text(s, ax.get_ylim()[1] * 0.97, f" {s}\n {frac:.0f}%", fontsize=7,
                    va="top", ha="left", color="dimgrey")
    ax.legend(fontsize=8, loc="upper right")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=2_000_000, help="tokens to sample from each split")
    ap.add_argument("--max_dist", type=int, default=1024, help="cap on distance; set to the training context length")
    ap.add_argument("--rare_below", type=int, default=40, help="a token is rare if it occurs fewer than this many times")
    ap.add_argument("--spans", type=int, nargs="+", default=[64, 128, 256, 512, 1024], help="candidate spans to mark")
    ap.add_argument("--out", default="copy_distance.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    stats = {}
    for ax, split in zip(axes, ("train", "val")):
        tok = load_tokens(split, args.tokens)
        every, rare = copy_distances(tok, args.max_dist, args.rare_below)
        panel(ax, every, rare, f"{split}  ({len(tok):,} tokens sampled)", args.spans, args.max_dist)
        stats[split] = (every, rare)
        print(f"\n{split}: {len(tok):,} tokens")
        for name, d in (("all ", every), ("rare", rare)):
            cov = "  ".join(f"<={s}: {100 * (d <= s).mean():.1f}%" for s in args.spans if s <= args.max_dist)
            print(f"  {name} median {np.median(d):6.0f}   {cov}")

    axes[0].set_ylabel("token count")
    fig.suptitle(
        f"Copy distance, capped at {args.max_dist} (rare = freq < {args.rare_below}); "
        "dashed lines are candidate attention spans, % = rare-token coverage"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"\nwrote {args.out}")

    tr, va = stats["train"][1], stats["val"][1]
    drift = max(abs((tr <= s).mean() - (va <= s).mean()) for s in args.spans if s <= args.max_dist)
    print(f"largest train/val gap in rare-token coverage: {100 * drift:.1f} pts "
          f"({'distributions agree, spans tuned on train should transfer' if drift < 0.03 else 'they differ -- tune on val'})")


if __name__ == "__main__":
    main()
