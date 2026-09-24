"""Generates PNG figures for the review PPT slides 07/08/09, from the
already-computed result JSON/NPZ files in results/ -- no new training,
pure plotting, except for figure 8 which needs one short inference pass
to get real predicted-vs-actual OLR values (train_step4_single_place.py
doesn't currently save raw predictions, only the summary MSE).

Usage: python scripts/generate_slide_figures.py
Output: results/slide_figures/*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
# Fig 1 (slide 07): MSE by candidate-set variant vs persistence (step 4)
# ---------------------------------------------------------------------
def fig_variant_mse():
    d = load("step4_place22.json")
    variants = [v["variant"] for v in d["variants"]]
    mse = [v["expert_mse"] for v in d["variants"]]
    pers = d["variants"][0]["persistence_mse"]

    fig, ax = plt.subplots(figsize=(6, 4.3))
    x = np.arange(len(variants))
    bars = ax.bar(x, mse, width=0.5, color="#4C72B0", label="Expert MSE")
    ax.axhline(pers, color="#C44E52", linestyle="--", linewidth=2, label=f"Persistence baseline ({pers:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}\n(n={d['variants'][i]['n_candidate_sources']})" for i, v in enumerate(variants)])
    ax.set_ylabel("Forecast MSE (lower is better)")
    ax.set_title("Forecast Error by Candidate Edge Set — Region 22")
    ax.set_ylim(0, pers * 1.22)
    for b, m in zip(bars, mse):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.0004, f"{m:.4f}", ha="center", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "07_1_variant_mse.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Fig 2 (slide 07): skill vs persistence across all 50 regions
# ---------------------------------------------------------------------
def fig_all_places_skill():
    d = load("step7_all_places_direct.json")
    places = sorted(d["places"], key=lambda p: p["skill_vs_persistence"])
    skills = [p["skill_vs_persistence"] * 100 for p in places]
    colors = ["#C44E52" if s < 0 else "#55A868" for s in skills]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(places)), skills, color=colors, width=0.9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Region (sorted by skill)")
    ax.set_ylabel("Skill vs Persistence (%)")
    ax.set_title(f"Forecast Skill Across All 50 Regions  "
                 f"({d['n_beating_persistence']}/{d['n_places_done']} beat baseline, "
                 f"mean {d['skill_mean']*100:.1f}%)")
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(OUT / "07_2_all_places_skill.png")
    plt.close(fig)


# ---------------------------------------------------------------------
# Fig 3 (slide 07): drift detector latency / false-alarm table as image
# ---------------------------------------------------------------------
def fig_drift_table():
    d = load("step5_drift_place22_direct.json")
    w = d["windows"]

    detector_labels = [
        ("channel1_error", "Channel 1 (error)"),
        ("channel2_causal", "Channel 2 (causal)"),
        ("fused_OR", "Fused OR"),
        ("fused_AND", "Fused AND"),
    ]
    rows = []
    for wname, label in [("abrupt_el_nino_1997_98", "Abrupt\n(El Nino 97-98)"),
                          ("gradual_drift_2013_2022", "Gradual\n(2013-2022)")]:
        if wname not in w:
            continue
        for key, det_label in detector_labels:
            c = w[wname][key]
            rows.append([label, det_label,
                         "Detected" if c["detected_in_window"] else "Missed",
                         f"{c['latency_days']}d" if c["latency_days"] is not None else "-",
                         f"{c['false_alarm_rate_per_year']:.2f}/yr"])

    fig, ax = plt.subplots(figsize=(8.5, 0.6 + 0.5 * len(rows)))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["Window", "Detector", "Result", "Latency", "False-alarm rate"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    for i, row in enumerate(rows, start=1):
        if row[2] == "Missed":
            for j in range(5):
                tbl[i, j].set_facecolor("#FBE3E3")
    ax.set_title("Drift Detection Performance — Region 22", pad=20)
    fig.tight_layout()
    fig.savefig(OUT / "07_3_drift_table.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Fig 4 (slide 08): ablation comparison bar chart -- THE headline chart
# ---------------------------------------------------------------------
def fig_ablation():
    d = load("step8_baselines_place22_direct.json")
    order = ["persistence", "random_subset", "full_causal_splitter", "plain_gnn_all_edges"]
    labels = {
        "persistence": "Persistence\n(naive baseline)",
        "random_subset": "Random\nEdge Subset",
        "full_causal_splitter": "Full Causal\nSplitter (ours)",
        "plain_gnn_all_edges": "Plain GNN\n(all edges)",
    }
    mse = [d["results_mse"][k] for k in order]
    colors = ["#8C8C8C", "#DD8452", "#C44E52", "#55A868"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([labels[k] for k in order], mse, color=colors, width=0.6)
    for b, m in zip(bars, mse):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.0002, f"{m:.5f}", ha="center", fontsize=9)
    ax.set_ylabel("Forecast MSE (lower is better)")
    ax.set_title("Ablation: Does Causal Selection Help?  (Region 22)")
    ax.set_ylim(0, max(mse) * 1.15)
    fig.text(0.5, -0.02,
              "Causal splitter performs like random selection, and worse than using all edges.",
              ha="center", fontsize=9, style="italic", color="#444")
    fig.tight_layout()
    fig.savefig(OUT / "08_1_ablation.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Fig 5 (slide 08): PCMCI link strength vs what the splitter selected
# ---------------------------------------------------------------------
def fig_pcmci_vs_selection():
    pcmci = load("pcmci_place22_direct.json")
    step4 = load("step4_place22.json")
    direct = next(v for v in step4["variants"] if v["variant"] == "direct")
    selected = set(direct["selected_sources"])

    scores = pcmci["pcmci_scores"]
    items = sorted(((int(k), v["strength"]) for k, v in scores.items() if int(k) != pcmci["target"]),
                   key=lambda kv: -kv[1])
    srcs = [k for k, _ in items]
    strengths = [v for _, v in items]
    colors = ["#C44E52" if s in selected else "#8C8C8C" for s in srcs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([f"c{s}" for s in srcs], strengths, color=colors, width=0.6)
    ax.set_ylabel("PCMCI Link Strength (independent statistical test)")
    ax.set_xlabel("Candidate source region")
    ax.set_title("PCMCI-Verified Drivers vs Splitter's Actual Selection")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#C44E52", label="Selected by splitter"),
                        Patch(color="#8C8C8C", label="Not selected")], loc="upper right")
    fig.text(0.5, -0.02,
              "The splitter missed the two strongest real drivers and picked three weaker ones.",
              ha="center", fontsize=9, style="italic", color="#444")
    fig.tight_layout()
    fig.savefig(OUT / "08_2_pcmci_vs_selection.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Fig 6 (slide 08): CIA fix attempt -- before vs after
# ---------------------------------------------------------------------
def fig_cia_attempt():
    baseline = load("step8_baselines_place22_direct.json")["results_mse"]["full_causal_splitter"]
    weights, mses = [], []
    for f in sorted(RESULTS.glob("step_cia_place22_direct_w*.json")):
        d = json.loads(f.read_text())
        weights.append(d["cia_weight"])
        mses.append(d["expert_mse"])
    order = np.argsort(weights)
    weights = [weights[i] for i in order]
    mses = [mses[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(weights, mses, "o-", color="#4C72B0", linewidth=2, markersize=8, label="CIA-regularized splitter")
    ax.axhline(baseline, color="#C44E52", linestyle="--", linewidth=2, label=f"Original splitter ({baseline:.5f})")
    ax.set_xlabel("CIA alignment weight")
    ax.set_ylabel("Forecast MSE")
    ax.set_title("CIA Fix Attempt: MSE vs Alignment Strength — Region 22")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "08_3_cia_attempt.png")
    plt.close(fig)


# ---------------------------------------------------------------------
# Fig 7 (slide 09): hibernate/reactivate timeline over 1979-2022
# ---------------------------------------------------------------------
def fig_lifecycle_timeline():
    d = load("step6_lifecycle_place22_direct.json")
    events = d["events"]
    dates = [np.datetime64(e["date"]) for e in events]
    reactivated = [e["reactivated"] for e in events]

    fig, ax = plt.subplots(figsize=(11, 3.2))
    for date, react in zip(dates, reactivated):
        ax.axvline(date, color="#55A868" if react else "#DD8452", linewidth=1.5, alpha=0.85)
    ax.set_yticks([])
    ax.set_xlabel("Date")
    ax.set_title(f"Expert Lifecycle Events, 1979-2022 — Region 22  "
                 f"({d['n_reactivated']} reactivated, {d['n_fresh']} spawned fresh)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color="#55A868", lw=2, label="Reactivated archived expert"),
                        Line2D([0], [0], color="#DD8452", lw=2, label="Spawned fresh expert")],
               loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=9, frameon=False)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(OUT / "09_1_lifecycle_timeline.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Fig 8 (slide 09): forecast vs actual OLR -- needs one short inference
# pass since train_step4_single_place.py doesn't save raw predictions.
# ---------------------------------------------------------------------
def fig_forecast_vs_actual():
    import torch
    from causal_moe.data.candidate_edges import build_candidate_source_set, expand_source_clusters_to_edge_index
    from causal_moe.experts.expert import PlaceExpert
    from causal_moe.splitter.dirgnn import DIRGNNSplitter
    from train_step4_single_place import load_cache, build_generator_windows, CACHE_PATH

    target = 22
    cache = load_cache(CACHE_PATH)
    candidate_set = build_candidate_source_set(cache["edge_index"], target, "direct", n_clusters=cache["n_clusters"])
    sources = candidate_set.source_clusters
    causal_edge_index_full = torch.from_numpy(expand_source_clusters_to_edge_index(sources, target))

    n_samples_cap = 4000
    features = cache["features"][:n_samples_cap]
    targets = cache["targets"][:n_samples_cap]
    dates = cache["sample_dates"][:n_samples_cap]
    n_features = features.shape[-1]

    olr_lag0_all = cache["features"][:, :, cache["olr_lag0_channel"]]
    gen_windows_full = build_generator_windows(olr_lag0_all, cache["sample_time_index"], 10)
    gen_windows = torch.from_numpy(gen_windows_full[:n_samples_cap])

    x_all = torch.from_numpy(features)
    y_all = torch.from_numpy(targets)
    y_target = y_all[:, target].numpy()

    n_causal_candidates = causal_edge_index_full.shape[1]
    r = min(0.5, 3.0 / n_causal_candidates)

    torch.manual_seed(0)
    splitter = DIRGNNSplitter(in_channels=n_features, hidden_channels=16, r=r, generator_window_len=10, generator_in_channels=1, use_self_features=False)
    expert = PlaceExpert(in_channels=n_features, hidden_channels=16)
    optimizer = torch.optim.Adam(list(splitter.parameters()) + list(expert.parameters()), lr=1e-3)

    from causal_moe.experts.expert import compute_expert_loss
    from causal_moe.splitter.dirgnn import LambdaWarmupSchedule
    rng = np.random.default_rng(0)
    n_samples = x_all.shape[0]
    epochs = 2
    schedule = LambdaWarmupSchedule(lambda_max=1.0, total_steps=max(epochs * n_samples, 1))
    step = 0
    target_mask = torch.zeros(cache["n_clusters"], dtype=torch.bool)
    target_mask[target] = True

    print("Quick retrain for prediction-array export (matches step4 settings, seed=0)...")
    for epoch in range(epochs):
        order = rng.permutation(n_samples)
        for start in range(0, n_samples - 16 + 1, 16):
            idx_batch = order[start:start + 16]
            optimizer.zero_grad()
            batch_loss = torch.zeros(())
            for idx in idx_batch:
                candidates = np.delete(np.arange(n_samples), idx)
                env_idx = rng.choice(candidates, size=min(6, candidates.shape[0]), replace=False)
                env_t = x_all[env_idx]
                lam = schedule.value(step)
                splitter_out = splitter.compute_loss(
                    x_all[idx], y_all[idx], env_t, causal_edge_index_full, lam,
                    x_generator_window=gen_windows[idx], target_node_mask=target_mask,
                )
                split = splitter.split(gen_windows[idx], causal_edge_index_full)
                expert_out = compute_expert_loss(expert, x_all[idx], target, y_target[idx],
                                                  split.causal_edge_index, split.causal_edge_weight)
                batch_loss = batch_loss + splitter_out.total_loss + expert_out.total_loss
                step += 1
            (batch_loss / len(idx_batch)).backward()
            optimizer.step()
        print(f"  epoch {epoch+1}/{epochs} done")

    splitter.eval()
    expert.eval()
    preds = []
    with torch.no_grad():
        for idx in range(n_samples):
            split = splitter.split(gen_windows[idx], causal_edge_index_full)
            pred = expert(x_all[idx], target, split.causal_edge_index, split.causal_edge_weight)
            preds.append(pred.item())
    preds = np.array(preds)

    window = slice(0, 400)
    dates_plot = dates[window]
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(dates_plot, y_target[window], color="#333333", linewidth=1.3, label="Actual OLR")
    ax.plot(dates_plot, preds[window], color="#C44E52", linewidth=1.1, alpha=0.85, label="Forecast (1-day ahead)")
    ax.set_ylabel("OLR anomaly (normalized)")
    ax.set_title("Forecast vs Actual OLR — Region 22")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "09_2_forecast_vs_actual.png", bbox_inches="tight")
    plt.close(fig)
    np.savez(RESULTS / "forecast_vs_actual_place22.npz", dates=dates, y_true=y_target, y_pred=preds)


if __name__ == "__main__":
    print("Generating fig 1: variant MSE...")
    fig_variant_mse()
    print("Generating fig 2: all-places skill...")
    fig_all_places_skill()
    print("Generating fig 3: drift table...")
    fig_drift_table()
    print("Generating fig 4: ablation (headline)...")
    fig_ablation()
    print("Generating fig 5: PCMCI vs selection...")
    fig_pcmci_vs_selection()
    print("Generating fig 6: CIA attempt...")
    fig_cia_attempt()
    print("Generating fig 7: lifecycle timeline...")
    fig_lifecycle_timeline()
    print("Generating fig 8: forecast vs actual (short retrain needed)...")
    fig_forecast_vs_actual()
    print(f"\nAll figures written to {OUT}")
