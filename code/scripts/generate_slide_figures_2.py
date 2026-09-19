"""Second batch of slide figures: density-cliff, PCMCI table, CIA-vs-GSINA,
test-suite growth. Companion to generate_slide_figures.py.

Numbers for the density-cliff plot are taken directly from
Causal_MoE_Architecture.md SS10 steps 2/3 (no saved JSON exists for those
older runs) -- hardcoded here with an explicit source comment, not
invented. Everything else reads real result JSON files in results/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "slide_figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load(name):
    return json.loads((RESULTS / name).read_text())


# ---------------------------------------------------------------------
# Fig 9: density-cliff plot, steps 2/3/4 combined
# Source of the step2/3 numbers: Causal_MoE_Architecture.md SS10.
#   Step 2 (CausalDynamics, n=100 candidates incl. self-loops):
#     NONE graph: AUC 0.694 (sparse, r=0.04)
#     AO graph:   AUC 0.586 (dense, r=0.49, best result w/ entropy pressure)
#   Step 3 (semi-synthetic, real BSISO features + injected rule):
#     5 clusters  (25 candidates):  precision/recall 0.333
#     15 clusters (225 candidates): precision/recall 0.062 (w/ entropy pressure)
# Step 4 (real data, no ground truth) uses score_std as the proxy metric
# instead of precision, read directly from step4_place22.json.
# ---------------------------------------------------------------------
def fig_density_cliff():
    step4 = load("step4_place22.json")
    step4_by_variant = {v["variant"]: v for v in step4["variants"]}

    # Panel A: step 3 (real climate features, known injected rule) -- the
    # cleanest apples-to-apples precision comparison.
    step3_sizes = [25, 225]
    step3_precision = [0.333, 0.062]

    # Panel B: step 4 (real data, no ground truth) -- score_std as the
    # discrimination proxy, across the three real candidate-set sizes.
    step4_sizes = [step4_by_variant[v]["n_candidate_sources"] ** 2 for v in ["direct", "2hop", "full"]]
    step4_std = [step4_by_variant[v]["score_std"] for v in ["direct", "2hop", "full"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(step3_sizes, step3_precision, "o-", color="#C44E52", linewidth=2, markersize=10)
    for x, y in zip(step3_sizes, step3_precision):
        ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(10, 8), fontsize=10)
    ax1.set_xscale("log")
    ax1.set_xlabel("Candidate edge-set size (log scale)")
    ax1.set_ylabel("Precision/Recall vs known rule")
    ax1.set_title("Step 3: Semi-Synthetic Check\n(real features, known injected rule)")
    ax1.set_ylim(-0.02, 0.4)

    ax2.plot(step4_sizes, step4_std, "o-", color="#4C72B0", linewidth=2, markersize=10)
    for x, y, v in zip(step4_sizes, step4_std, ["direct (n=8)", "2hop (n=19)", "full (n=50)"]):
        ax2.annotate(v, (x, y), textcoords="offset points", xytext=(10, 8), fontsize=9)
    ax2.set_xscale("log")
    ax2.set_xlabel("Candidate edge-set size (n^2, log scale)")
    ax2.set_ylabel("Edge-score std (discrimination proxy)")
    ax2.set_title("Step 4: Real BSISO Data\n(no ground truth available)")

    fig.suptitle("The Density Cliff: Causal Recovery Degrades as Candidate Set Grows", fontsize=13)
    fig.text(0.5, -0.02,
              "Same pattern across synthetic (step 2), semi-synthetic (step 3), and real data (step 4).",
              ha="center", fontsize=9, style="italic", color="#444")
    fig.tight_layout()
    fig.savefig(OUT / "06_1_density_cliff.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Fig 10: PCMCI cross-check table, clean image
# ---------------------------------------------------------------------
def fig_pcmci_table():
    pcmci = load("pcmci_place22_direct.json")
    step4 = load("step4_place22.json")
    direct = next(v for v in step4["variants"] if v["variant"] == "direct")
    selected = set(direct["selected_sources"])

    scores = pcmci["pcmci_scores"]
    items = sorted(scores.items(), key=lambda kv: -kv[1]["strength"])

    rows = []
    for src, info in items:
        src_i = int(src)
        label = f"c{src_i} (self)" if src_i == pcmci["target"] else f"c{src_i}"
        rows.append([
            label,
            f"{info['strength']:.3f}",
            f"{info['lag']}",
            f"{info['p_value']:.2e}",
            "Yes" if src_i in selected else "No",
        ])

    fig, ax = plt.subplots(figsize=(7.5, 0.5 + 0.45 * len(rows)))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["Source", "Link strength", "Lag (days)", "p-value", "Selected by splitter?"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for i, row in enumerate(rows, start=1):
        if row[-1] == "Yes":
            for j in range(5):
                tbl[i, j].set_facecolor("#FBE3E3")
    ax.set_title("PCMCI Independent Cross-Check — Region 22", pad=14)
    fig.tight_layout()
    fig.savefig(OUT / "08_4_pcmci_table.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Fig 11: CIA vs GSINA vs original splitter, combined comparison
# ---------------------------------------------------------------------
def fig_fix_comparison():
    baselines = load("step8_baselines_place22_direct.json")["results_mse"]

    cia_files = sorted(RESULTS.glob("step_cia_place22_direct_w*.json"))
    cia_best = min((json.loads(f.read_text())["expert_mse"] for f in cia_files), default=None)

    gsina_files = [f for f in RESULTS.glob("step_gsina_place22_direct_seed*.json")]
    gsina_mse = [json.loads(f.read_text())["expert_mse"] for f in gsina_files]
    gsina_best = min(gsina_mse) if gsina_mse else None

    labels = ["Persistence", "Random\nSubset", "Original\nSplitter", "Plain GNN\n(all edges)"]
    values = [baselines["persistence"], baselines["random_subset"],
              baselines["full_causal_splitter"], baselines["plain_gnn_all_edges"]]
    colors = ["#8C8C8C", "#DD8452", "#C44E52", "#55A868"]

    if cia_best is not None:
        labels.append("CIA Fix\n(best)")
        values.append(cia_best)
        colors.append("#4C72B0")
    if gsina_best is not None:
        labels.append("GSINA Fix\n(best)")
        values.append(gsina_best)
        colors.append("#8172B2")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0002, f"{v:.5f}", ha="center", fontsize=9)
    ax.set_ylabel("Forecast MSE (lower is better)")
    ax.set_title("Fix Attempts: Do CIA / GSINA Close the Gap?  (Region 22)")
    ax.set_ylim(0, max(values) * 1.15)

    note = "GSINA is the only method that selected cluster 38, the strongest real driver." if gsina_best else ""
    if note:
        fig.text(0.5, -0.02, note, ha="center", fontsize=9, style="italic", color="#444")
    fig.tight_layout()
    fig.savefig(OUT / "08_5_fix_comparison.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Fig 12: test suite growth over the project
# Source: Causal_MoE_Architecture.md SS10 progress log (cumulative counts
# stated at each step's completion).
# ---------------------------------------------------------------------
def fig_test_growth():
    steps = ["Step 1\n(mesh)", "Step 2\n(splitter)", "Step 3\n(semi-synth)",
             "Step 4\n(expert)", "CIA fix"]
    counts = [25, 51, 67, 96, 110]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, counts, "o-", color="#55A868", linewidth=2.5, markersize=9)
    for x, y in zip(steps, counts):
        ax.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Cumulative automated tests passing")
    ax.set_title("Test Coverage Grew With Every Stage")
    ax.set_ylim(0, 125)
    fig.tight_layout()
    fig.savefig(OUT / "05_1_test_growth.png")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating fig 9: density cliff...")
    fig_density_cliff()
    print("Generating fig 10: PCMCI table...")
    fig_pcmci_table()
    print("Generating fig 11: CIA vs GSINA comparison...")
    fig_fix_comparison()
    print("Generating fig 12: test suite growth...")
    fig_test_growth()
    print(f"\nAll figures written to {OUT}")
