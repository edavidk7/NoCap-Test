"""
Per-head attention mask visualization for the technical report.
T=1024. Three masks: local window w=64, local window w=128, full causal.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

T = 1024

def window_mask(T, span):
    q = np.arange(T)[:, None]
    k = np.arange(T)[None, :]
    causal = k <= q
    if span is None:
        return causal
    return causal & (k > q - span)

PANELS = [
    ("local window, $w=64$", 64),
    ("local window, $w=128$", 128),
    ("full causal", None),
]

ATTEND = "#1f77b4"  # matplotlib default blue (C0)
MASK = "#f2efe8"
BG = "#ffffff"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
})

fig, axes = plt.subplots(1, 3, figsize=(11, 4.3), dpi=200)
cmap = ListedColormap([MASK, ATTEND])

for ax, (title, span) in zip(axes, PANELS):
    m = window_mask(T, span)
    ax.imshow(m, cmap=cmap, origin="upper", interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=11.5, pad=10)
    ax.set_xticks([0, 512, 1023]); ax.set_yticks([0, 512, 1023])
    ax.tick_params(labelsize=8)
    ax.set_xlabel("key position", fontsize=9)
    for s in ax.spines.values():
        s.set_edgecolor("#333333"); s.set_linewidth(0.6)

axes[0].set_ylabel("query position", fontsize=9)

fig.suptitle("Attention masks, $T=1024$", fontsize=14, fontweight="bold", y=1.01)

legend_elems = [Patch(facecolor=ATTEND, edgecolor="#333333", label="attendable"),
                Patch(facecolor=MASK, edgecolor="#333333", label="masked")]
fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.05), fontsize=9)

plt.tight_layout()

out = "/Users/davidkorcak/Documents/NoCap-Test/attn_masks_3configs.png"
fig.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight")
print("saved", out)
