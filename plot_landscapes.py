import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loss_landscape_results")
CMAP = "viridis"


def latest_run():
    runs = [d for d in glob.glob(os.path.join(RESULTS_ROOT, "*")) if os.path.isdir(d)]
    if not runs:
        raise SystemExit(f"no runs found under {RESULTS_ROOT}")
    return os.path.basename(max(runs, key=os.path.getmtime))


def plain_decimal(axis):
    """No scientific notation, no offset text (e.g. "+1.1e1"), on this axis."""
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    axis.set_major_formatter(fmt)


ap = argparse.ArgumentParser()
ap.add_argument("--run", default=None, help="run directory name under loss_landscape_results/ (default: most recent)")
args = ap.parse_args()

run_name = args.run or latest_run()
run_dir = os.path.join(RESULTS_ROOT, run_name)
data_dir = os.path.join(run_dir, "data")
out_dir = os.path.join(run_dir, "figures")
os.makedirs(out_dir, exist_ok=True)
print(f"reading {data_dir}, writing {out_dir}")

config = {}
config_path = os.path.join(run_dir, "run_config.json")
if os.path.exists(config_path):
    config = json.load(open(config_path))

# auto-discover (block, group) pairs from filenames instead of hardcoding
pat = re.compile(r"block(\d+)_(\w+)\.npz$")
found = {}
for f in glob.glob(os.path.join(data_dir, "block*_*.npz")):
    m = pat.search(f)
    if m:
        found.setdefault(int(m.group(1)), []).append(m.group(2))
blocks = sorted(found)
group_order = ["Q", "K", "V", "MLP"]
groups = [g for g in group_order if all(g in found[b] for b in blocks)]
if not blocks or not groups:
    raise SystemExit(f"no block*_*.npz files found in {data_dir}")

aux_lambda = config.get("emb_aux_lambda", 0.1)

for b in blocks:
    for g in groups:
        data = np.load(os.path.join(data_dir, f"block{b}_{g}.npz"))
        axis0, axis1 = data["axis0"], data["axis1"]
        ce, aux = data["ce"], data["aux"]
        A0, A1 = np.meshgrid(axis0, axis1, indexing="ij")

        ce_f = np.nan_to_num(ce, nan=np.nanmax(ce), posinf=np.nanmax(ce[np.isfinite(ce)]) if np.isfinite(ce).any() else 20.0)
        aux_f = np.nan_to_num(aux, nan=np.nanmax(aux), posinf=np.nanmax(aux[np.isfinite(aux)]) if np.isfinite(aux).any() else 20.0)

        vmin = min(ce_f.min(), aux_f.min())
        vmax = max(np.percentile(ce_f, 95), np.percentile(aux_f, 95))
        norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-9), clip=True)
        sm = ScalarMappable(norm=norm, cmap=CMAP)

        fig = plt.figure(figsize=(12.5, 4.5), dpi=200, constrained_layout=True)
        row_axes = []
        for col, (grid, title) in enumerate([(ce_f, "CE only"), (aux_f, rf"CE $+ {aux_lambda:g} \times$ aux")]):
            ax = fig.add_subplot(1, 2, col + 1, projection="3d")
            ax.plot_surface(A0, A1, grid, facecolors=sm.to_rgba(grid), rstride=1, cstride=1,
                              linewidth=0, antialiased=True, shade=False)
            ax.set_title(f"block {b} {g} -- {title}", fontsize=9)
            ax.set_xlabel(r"$\alpha_1$", fontsize=9)
            ax.set_ylabel(r"$\alpha_2$", fontsize=9)
            ax.set_zlabel("loss", fontsize=7)
            ax.tick_params(labelsize=6)
            plain_decimal(ax.xaxis)
            plain_decimal(ax.yaxis)
            plain_decimal(ax.zaxis)
            ax.view_init(elev=25, azim=-55)
            row_axes.append(ax)

        cbar = fig.colorbar(sm, ax=row_axes, shrink=0.7, aspect=30, fraction=0.025, pad=0.12, location="right")
        cbar.ax.tick_params(labelsize=7)
        plain_decimal(cbar.ax.yaxis)
        cbar.set_label(f"block {b} {g} loss", fontsize=8)

        out_path = os.path.join(out_dir, f"block{b}_{g}_landscape3d.png")
        fig.savefig(out_path)
        plt.close(fig)
        print("saved", out_path)
