"""Builds figures/10_stage_c_lead1.png from the saved Stage C result files.

Usage (from the repository root, on the phase1-causal-graph branch):
    python thesis/scripts/make_stage_c_figures.py code/results/phase1 thesis/figures
Pure plotting of existing results; no training.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

res, out = Path(sys.argv[1]), Path(sys.argv[2])


def load(name):
    d = json.loads((res / name).read_text())
    return {p["place"]: p for p in d["places"]}


none = load("stage_c_lead1/full_none.json")
pc = load("stage_c_lead1/full_pcmci_restrict.json")
ids = sorted(none)
sk0 = np.array([none[i]["skill_vs_persistence"] for i in ids])
sk1 = np.array([pc[i]["skill_vs_persistence"] for i in ids])
r0 = np.array([none[i]["r2_vs_climatology"] for i in ids])
r1 = np.array([pc[i]["r2_vs_climatology"] for i in ids])
npri = np.array([len(pc[i]["prior_regions"]) for i in ids])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
cmap = plt.get_cmap("viridis")
for a, x, y, lab in ((ax[0], sk0, sk1, "skill vs persistence"), (ax[1], r0, r1, "R$^2$ vs climatology")):
    sc = a.scatter(x, y, c=np.minimum(npri, 3), cmap=cmap, vmin=0, vmax=3, s=28, edgecolor="k", linewidth=0.3)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.05 * (hi - lo)
    a.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8)
    a.set_xlabel(f"no prior: {lab}")
    a.set_ylabel(f"PCMCI prior (restrict): {lab}")
    better = int((y > x).sum())
    a.set_title(f"{lab}: {better} of {len(x)} regions above the diagonal", fontsize=10)
    a.grid(alpha=0.3)
cb = fig.colorbar(sc, ax=ax, ticks=[0, 1, 2, 3], fraction=0.03)
cb.set_label("outside regions in prior (3 = 3 or more)")
fig.savefig(out / "10_stage_c_lead1.png", dpi=200, bbox_inches="tight")
print("median skill change", float(np.median(sk1 - sk0)), "median R2 change", float(np.median(r1 - r0)))
