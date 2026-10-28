"""Builds figures/11_lifecycle_timeline.png from the expert-lifecycle result
files (one row per region: reactivated archived expert vs spawned fresh).

Usage: python thesis/scripts/make_lifecycle_figure.py code/results thesis/figures
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

res, out = Path(sys.argv[1]), Path(sys.argv[2])
regions = (22, 12, 29)
fig, axes = plt.subplots(len(regions), 1, figsize=(11, 4.2), sharex=True)
for ax, r in zip(axes, regions):
    d = json.loads((res / f"step6_lifecycle_place{r}_direct.json").read_text())
    for e in d["events"]:
        t = pd.Timestamp(e["date"])
        ax.axvline(t, color="#2e9e5b" if e["reactivated"] else "#e8743b", lw=1.6)
    ax.set_yticks([])
    ax.set_ylabel(f"region {r}", rotation=0, ha="right", va="center", fontsize=10)
    ax.set_title(f"{d['n_reactivated']} reactivated, {d['n_fresh']} spawned fresh "
                 f"({d['n_reactivated'] + d['n_fresh']} regime changes)", fontsize=9, loc="right")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
handles = [plt.Line2D([], [], color="#2e9e5b", lw=2), plt.Line2D([], [], color="#e8743b", lw=2)]
fig.legend(handles, ["reactivated archived expert", "spawned fresh expert"], loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(out / "11_lifecycle_timeline.png", dpi=200)
