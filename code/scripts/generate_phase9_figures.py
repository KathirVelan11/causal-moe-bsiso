"""Phase 9.9: figures for the final writeup, from already-computed result
JSONs only -- no new training. Companion to generate_slide_figures.py
(which covers the earlier PPT slides); this one covers the Phase 9
mesh-wide sweeps, variant comparison, and the lifecycle/forgetting result.

Usage: python scripts/generate_phase9_figures.py
Output: results/phase9_figures/*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "phase9_figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE = "#4C72B0"
RED = "#C44E52"
GREEN = "#55A868"
ORANGE = "#DD8452"
GATE = 0.02


def load(name):
    return json.loads((RESULTS / name).read_text())


def fig_skill_per_place():
    d = load("step7_all_places_direct_lead7.json")
    places = sorted(d["places"], key=lambda p: p["skill_vs_persistence"])
    x = np.arange(len(places))
    skills = [p["skill_vs_persistence"] for p in places]
    colors = [GREEN if s >= 0 else RED for s in skills]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x, skills, color=colors, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Place (sorted by skill)")
    ax.set_ylabel("Skill vs persistence (1 - MSE/MSE_persist)")
    ax.set_title(f"Lead-7 skill vs persistence, all 50 places "
                 f"({d['n_beating_persistence']}/50 beat persistence, "
                 f"mean skill={d['skill_mean']:+.3f})")
    fig.tight_layout()
    fig.savefig(OUT / "01_skill_per_place_lead7.png")
    plt.close(fig)


def fig_variant_comparison():
    variants = ["direct", "2hop", "full"]
    files = {
        "direct": "step7_all_places_direct_lead7.json",
        "2hop": "step7_all_places_2hop.json",
        "full": "step7_all_places_full.json",
    }
    skill_means, std_means, gate_pass, n_cand = [], [], [], []
    for v in variants:
        d = load(files[v])
        places = d["places"]
        skill_means.append(np.mean([p["skill_vs_persistence"] for p in places]))
        stds = [p["score_std"] for p in places]
        std_means.append(np.mean(stds))
        gate_pass.append(sum(1 for s in stds if s >= GATE))
        n_cand.append(np.mean([p["n_candidate_sources"] for p in places]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(variants))

    ax1.bar(x, skill_means, width=0.5, color=BLUE)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{v}\n(~{n:.0f} candidates)" for v, n in zip(variants, n_cand)])
    ax1.set_ylabel("Mean skill vs persistence")
    ax1.set_title("Accuracy: flat across candidate-pool size")
    for i, s in enumerate(skill_means):
        ax1.text(i, s + 0.01, f"{s:+.3f}", ha="center")

    ax2.bar(x, gate_pass, width=0.5, color=ORANGE)
    ax2.set_xticks(x)
    ax2.set_xticklabels(variants)
    ax2.set_ylabel("Places clearing discrimination gate (std>=0.02) / 50")
    ax2.set_title("Discrimination: collapses as pool grows")
    ax2.set_ylim(0, 50)
    for i, g in enumerate(gate_pass):
        ax2.text(i, g + 1, f"{g}/50", ha="center")

    fig.suptitle("Accuracy is robust to candidate-pool size; edge discrimination is not",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "02_variant_comparison.png", bbox_inches="tight")
    plt.close(fig)


def fig_score_std_histogram():
    d5 = load("step7_all_places_direct_lead7.json")
    d2 = load("step7_all_places_direct_epochs2.json.bak")
    stds5 = [p["score_std"] for p in d5["places"]]
    stds2 = [p["score_std"] for p in d2["places"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, max(max(stds5), max(stds2)) * 1.05, 20)
    ax.hist(stds2, bins=bins, alpha=0.55, label=f"epochs=2 ({sum(1 for s in stds2 if s>=GATE)}/50 clear gate)", color=RED)
    ax.hist(stds5, bins=bins, alpha=0.55, label=f"epochs=5 ({sum(1 for s in stds5 if s>=GATE)}/50 clear gate)", color=GREEN)
    ax.axvline(GATE, color="black", linestyle="--", linewidth=1.5, label="discrimination gate (0.02)")
    ax.set_xlabel("Edge-score std (per place)")
    ax.set_ylabel("Number of places")
    ax.set_title("Undertraining explains most of the discrimination gap\n(epochs=2 vs epochs=5, direct variant, lead-7)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "03_score_std_histogram_epochs2_vs_5.png")
    plt.close(fig)


def fig_lead1_vs_lead7():
    d7 = load("step7_all_places_direct_lead7.json")
    d1 = load("step7_all_places_direct_lead1.json")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, d, title in zip(axes, [d1, d7], ["Lead-1 (hardest: self-persistence ~93% of variance)",
                                              "Lead-7 (established result)"]):
        places = sorted(d["places"], key=lambda p: p["skill_vs_persistence"])
        skills = [p["skill_vs_persistence"] for p in places]
        colors = [GREEN if s >= 0 else RED for s in skills]
        ax.bar(np.arange(len(places)), skills, color=colors, width=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        beat = sum(1 for p in places if p["beats_persistence"])
        ax.set_title(f"{title}\n{beat}/50 beat persistence, mean skill={d['skill_mean']:+.3f}")
        ax.set_xlabel("Place (sorted)")
    axes[0].set_ylabel("Skill vs persistence")
    fig.suptitle("Persistence is a much harder baseline at lead-1 than lead-7\n"
                 "(though both beat climatology)",
                 fontsize=11, y=1.06)
    fig.tight_layout()
    fig.savefig(OUT / "04_lead1_vs_lead7.png", bbox_inches="tight")
    plt.close(fig)


def fig_variant_ab_forgetting():
    places = ["place22", "place12", "place29"]
    files = {
        "place22": "step6_lifecycle_place22_direct.json",
        "place12": "step6_lifecycle_place12_direct.json",
        "place29": "step6_lifecycle_place29_direct.json",
    }
    a_mse, b_mse, a_forget, b_forget = [], [], [], []
    for p in places:
        d = load(files[p])
        cmp = d["variant_a_vs_variant_b"]
        a_mse.append(cmp["mean_variant_a_mse_k1"])
        b_mse.append(cmp["mean_variant_b_mse_k"])
        a_forget.append(cmp["forgetting_variant_a_k1"]["average_forgetting"])
        b_forget.append(cmp["forgetting_variant_b_k"]["average_forgetting"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(places))
    w = 0.35
    ax1.bar(x - w/2, a_mse, w, label="Variant A (k=1)", color=BLUE)
    ax1.bar(x + w/2, b_mse, w, label="Variant B (k=3)", color=GREEN)
    ax1.set_xticks(x); ax1.set_xticklabels(places)
    ax1.set_ylabel("Held-out MSE (lower better)")
    ax1.set_title("MoE pool blend (B) vs single-expert (A): MSE")
    ax1.legend()

    ax2.bar(x - w/2, a_forget, w, label="Variant A (k=1)", color=BLUE)
    ax2.bar(x + w/2, b_forget, w, label="Variant B (k=3)", color=GREEN)
    ax2.set_xticks(x); ax2.set_xticklabels(places)
    ax2.set_ylabel("Avg forgetting (lower better)")
    ax2.set_title("MoE pool blend (B) vs single-expert (A): forgetting")
    ax2.legend()

    fig.suptitle("Variant B beats Variant A on both MSE and forgetting at all 3 places tested",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "05_variant_ab_forgetting.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_skill_per_place()
    fig_variant_comparison()
    fig_score_std_histogram()
    fig_lead1_vs_lead7()
    fig_variant_ab_forgetting()
    print(f"wrote 5 figures to {OUT}")
