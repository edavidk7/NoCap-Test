"""Measure how far back attention actually needs to look, to size per-head spans.

Run with:
    uv run --with tiktoken python analyze_context.py
    uv run --with tiktoken python analyze_context.py --tokens 2000000 --split train

tiktoken is only needed for the sentence statistics; the copy-distance analysis, which is
the more decision-relevant one, runs without it.

Three things are measured:

  SENTENCE LENGTH   how much text a head needs to see one full clause. Bounds the useful
                    span for syntax-local heads.
  DOCUMENT LENGTH   beyond this, attention is looking at unrelated text.
  COPY DISTANCE     for each token, how far back its previous occurrence is. This is what
                    induction/copying heads exploit, and it is the statistic that says how
                    much *predictive* information a given span captures. Reported for all
                    tokens and, separately, for rare tokens -- repeats of "the" carry
                    little information, repeats of a proper noun carry a lot.

Note on acting on the numbers: with flex_attention the mask block granularity is 128 (the
kernel tile), so any span of 128 or less costs exactly the same. A wider span sees strictly
more for the same price, so there is no reason to choose a span below 128 unless narrowness
is wanted as a regulariser. Spans worth distinguishing are multiples of 128.
"""

import argparse
import glob

import numpy as np


def load_tokens(split, n_tokens):
    pattern = f"data/fineweb10B/fineweb_{split}_*.bin"
    paths = sorted(glob.glob(pattern))
    assert paths, f"no shards matched {pattern}"
    with open(paths[0], "rb") as fh:
        fh.read(256 * 4)  # header
        return np.frombuffer(fh.read(n_tokens * 2), dtype=np.uint16).astype(np.int64)


def sentence_lengths(tok, eot):
    """Tokens per sentence, splitting on tokens that render as sentence-final punctuation."""
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    finals = {".", "!", "?", '."', '!"', '?"', ".)", ".”", "!”", "?”"}
    enders = {t for t in range(50257) if enc.decode([t]).strip() in finals}
    lens, cur = [], 0
    for t in tok:
        cur += 1
        if int(t) in enders or t == eot:
            lens.append(cur)
            cur = 0
    return np.array([n for n in lens if n > 1])


def copy_distances(tok, freq_below, max_dist, freq):
    """Distance from each token to its previous occurrence, capped at max_dist."""
    last, near, far = {}, [], []
    for i, t in enumerate(tok):
        t = int(t)
        if t in last:
            d = i - last[t]
            if d <= max_dist:
                near.append(d)
                if freq[t] < freq_below:
                    far.append(d)
        last[t] = i
    return np.array(near), np.array(far)


def report(name, d, spans):
    print(f"\n{name} (n={len(d):,})")
    if not len(d):
        return
    print("  percentiles: " + "  ".join(f"p{p}={np.percentile(d, p):.0f}" for p in (25, 50, 75, 90, 95, 99)))
    print("  coverage:    " + "  ".join(f"<={s}: {100 * (d <= s).mean():.1f}%" for s in spans))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--tokens", type=int, default=400_000, help="how many tokens to sample from the first shard")
    ap.add_argument("--max_dist", type=int, default=1024, help="cap on copy distance; set to the training context length")
    ap.add_argument("--rare_below", type=int, default=40, help="a token is 'rare' if it occurs fewer than this many times in the sample")
    ap.add_argument("--spans", type=int, nargs="+", default=[64, 128, 256, 512, 1024], help="candidate attention spans to report coverage for")
    args = ap.parse_args()

    tok = load_tokens(args.split, args.tokens)
    eot = 50256
    freq = np.bincount(tok, minlength=50257)
    print(f"{args.split} split, {len(tok):,} tokens, {int((tok == eot).sum())} document boundaries")

    docs = np.diff(np.flatnonzero(tok == eot))
    report("DOCUMENT LENGTH", docs, args.spans)

    try:
        report("SENTENCE LENGTH", sentence_lengths(tok, eot), args.spans)
    except ImportError:
        print("\nSENTENCE LENGTH  skipped (needs tiktoken: rerun with `uv run --with tiktoken`)")

    near, rare = copy_distances(tok, args.rare_below, args.max_dist, freq)
    report(f"COPY DISTANCE, all tokens (capped at {args.max_dist})", near, args.spans)
    report(f"COPY DISTANCE, rare tokens only (freq < {args.rare_below})", rare, args.spans)

    print(
        "\nReading it: coverage on the rare-token line is the fraction of copyable"
        "\ninformation a head with that span can reach. The gap between two spans is what"
        "\nwidening a head buys. Spans of 128 or less are indistinguishable in cost."
    )


if __name__ == "__main__":
    main()
