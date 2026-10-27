"""Phase 1 figures (README SS13), plotted from saved results only -- no
training, no PCMCI.

Reads results/phase1/*.json|npz, cache/windowed_clustered50_lead7.npz
(region labels) and the dataset's grid/ocean-mask metadata. Writes
results/phase1/figures/*.png:

  01_bsiso_season.png        BSISO amplitude by day of year + season per year
  02_graph_map.png           validated outside drivers on a map
  03_links_through_checks.png  links left after each check; prior size
  04_b1_decoys.png           real link strengths vs the decoy pass
  05_b2_speed.png            implied speed vs distance, per source field
  06_b3_k.png                validation R^2 vs k
  07_b4_persistence.png      persistence per target and per field
  08_stage_d_drift.png       causal-structure distance by year vs ENSO
  09_stage_c_pcmci_vs_none.png  PCMCI prior (full, restrict) vs no prior
                             (only if stage_c/full_pcmci_restrict.json exists)

Usage:
    python scripts/generate_phase1_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.collections
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal_moe.causal_graph import validate as V
from causal_moe.data.raw import default_dataset_root

CODE = Path(__file__).resolve().parents[1]
RES = CODE / "results" / "phase1"
FIG = RES / "figures"

# Reference palette (dataviz skill), light mode.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CRITICAL = "#d03b3b"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURFACE, NEUTRAL = "#e5e4e0", "#fcfcfb", "#f0efec"
BLUE_LIGHT, RED_LIGHT = "#cde2fb", "#f8d4d3"
LAND, BOUNDARY = "#e2e0da", "#bdbbb4"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "font.size": 10, "legend.frameon": False, "lines.linewidth": 2,
})


def load(name):
    return json.loads((RES / name).read_text())


def save(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / name}")


def note(ax, text, loc=(0.0, -0.16), fig_y=-0.01):
    """Footnote under the whole figure (figure-level, so it never squeezes
    the axes layout); `loc` is kept for the map, which places it per axis."""
    if ax.figure.axes[0] is ax and loc[1] < -0.3:
        ax.text(*loc, text, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top")
    else:
        ax.figure.text(0.01, fig_y, text, fontsize=8.5, color=MUTED, va="top")


# ------------------------------------------------------------------ 01 ----
def fig_season():
    d = np.load(RES / "bsiso_season.npz")
    time = pd.DatetimeIndex(d["time"])
    amp = pd.Series(d["amplitude"], index=time)
    in_season = d["in_season"]
    doy = time.dayofyear

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [1, 1.6]})
    by_doy = amp.groupby(doy)
    q25, q50, q75 = by_doy.quantile(0.25), by_doy.median(), by_doy.quantile(0.75)
    ax1.fill_between(q50.index, q25, q75, color=BLUE_LIGHT, lw=0, label="25th–75th percentile")
    ax1.plot(q50.index, q50, color=BLUE, label="median")
    ax1.axhline(1.0, color=INK2, lw=1, ls="--")
    ax1.text(5, 1.03, "activity threshold = 1", color=INK2, fontsize=8.5, va="bottom")
    ax1.set_title("BSISO amplitude by day of year, 1979–2022")
    ax1.set_ylabel("amplitude (normalized PCs)")
    ax1.set_xlim(1, 366)
    ax1.legend(loc="upper left", fontsize=8.5)

    years = sorted(set(time.year))
    for y in years:
        m = (time.year == y) & in_season
        if not m.any():
            continue
        color = ORANGE if y >= 2013 else BLUE
        ax2.plot([doy[m].min(), doy[m].max()], [y, y], color=color, lw=4, solid_capstyle="round")
    ax2.plot([], [], color=BLUE, lw=4, label="discovery / training years (1979–2012)")
    ax2.plot([], [], color=ORANGE, lw=4, label="held-out test years (2013–2022)")
    ax2.set_title("Active BSISO season per year (target days)")
    ax2.set_xlim(1, 366)
    ax2.invert_yaxis()
    ax2.set_xlabel("day of year")
    months = pd.date_range("2001-01-01", periods=12, freq="MS")
    for a in (ax1, ax2):
        a.set_xticks(months.dayofyear, months.strftime("%b"))
    ax2.legend(loc="lower left", fontsize=8.5, ncol=2, bbox_to_anchor=(0, -0.2))
    fig.tight_layout()
    save(fig, "01_bsiso_season.png")


# ------------------------------------------------------------------ 02 ----
def region_centres(labels, lat, lon):
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    lat_c, lon_c = lat_g.reshape(-1), np.deg2rad(lon_g.reshape(-1))
    out = {}
    for r in range(int(labels.max()) + 1):
        m = labels == r
        ang = np.arctan2(np.sin(lon_c[m]).mean(), np.cos(lon_c[m]).mean())
        out[r] = (float(np.rad2deg(ang) % 360), float(lat_c[m].mean()))
    return out


def fig_map(graph):
    archive = default_dataset_root() / "rt_preprocessed_archive"
    grid = np.load(archive / "olr_rt_1979_2022.npz")
    lat, lon = grid["lat"], grid["lon"]
    ocean = np.load(archive / "parameters" / "sst_ocean_mask.npz")["ocean_mask"]
    labels = np.load(CODE / "cache" / "windowed_clustered50_lead7.npz")["cluster_labels"]
    centres = region_centres(labels, lat, lon)

    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.pcolormesh(np.append(lon, 360) - 1.25, np.append(lat, lat[-1] - 2.5) + 1.25,
                  np.where(ocean, np.nan, 1.0), cmap=matplotlib.colors.ListedColormap([LAND]), shading="flat")
    # Region boundaries: cell edges where neighbouring labels differ.
    lab2d = labels.reshape(lat.size, lon.size)
    segs = []
    for i in range(lat.size):
        for j in range(lon.size):
            x, y = lon[j], lat[i]
            if lab2d[i, j] != lab2d[i, (j + 1) % lon.size]:
                segs.append([(x + 1.25, y - 1.25), (x + 1.25, y + 1.25)])
            if i + 1 < lat.size and lab2d[i, j] != lab2d[i + 1, j]:
                segs.append([(x - 1.25, y - 1.25), (x + 1.25, y - 1.25)])
    ax.add_collection(matplotlib.collections.LineCollection(segs, colors=BOUNDARY, linewidths=0.6, zorder=1))
    ax.grid(False)

    max_val = max(abs(e["val"]) for v in graph["targets"].values() for e in v["drivers"])
    for t, v in graph["targets"].items():
        t = int(t)
        # one arrow per (source region -> target), strongest link
        best = {}
        for e in v["drivers"]:
            if abs(e["val"]) > abs(best.get(e["src_region"], {"val": 0})["val"]):
                best[e["src_region"]] = e
        for s, e in best.items():
            (x0, y0), (x1, y1) = centres[s], centres[t]
            if x1 - x0 > 180:
                x1 -= 360
            elif x0 - x1 > 180:
                x1 += 360
            color = BLUE if e.get("b4_persists") else ORANGE
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8 + 3.2 * abs(e["val"]) / max_val,
                                        shrinkA=3, shrinkB=3, mutation_scale=9, alpha=0.9))
    for r, (x, y) in centres.items():
        ax.text(x, y, str(r), fontsize=6.5, color=INK2, ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.15", fc=SURFACE, ec=MUTED, lw=0.5))
    ax.plot([], [], color=BLUE, lw=2, label="link reappears at 100 regions (B4)")
    ax.plot([], [], color=ORANGE, lw=2, label="link flagged by B4 (not found at 100 regions)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.3), ncol=2, fontsize=9)
    ax.set_xlim(0, 360)
    ax.set_ylim(-31.25, 31.25)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, 361, 30), [f"{x}°E" if x <= 180 else f"{360 - x}°W" for x in range(0, 361, 30)])
    ax.set_yticks([-30, -15, 0, 15, 30], ["30°S", "15°S", "0°", "15°N", "30°N"])
    ax.set_title("Validated outside drivers: source region → target region (arrow width = |MCI strength|)")
    note(ax, "Grey shading = land; thin lines = the 50 region boundaries; circles = region ids at their centres. "
             "One arrow per source→target pair (strongest link).", loc=(0, -0.36))
    save(fig, "02_graph_map.png")


# ------------------------------------------------------------------ 03 ----
def fig_links(graph, a50):
    T = graph["targets"]
    own_a = sum(1 for t, g in a50["graph"].items() for e in g["edges"] if e["src_region"] == int(t))
    out_a = sum(1 for t, g in a50["graph"].items() for e in g["edges"] if e["src_region"] != int(t))
    own_b2 = sum(1 for t, v in T.items() for e in v["candidates_after_b2"] if e["src_region"] == int(t))
    out_b2 = sum(1 for t, v in T.items() for e in v["candidates_after_b2"] if e["src_region"] != int(t))
    out_b3 = sum(len(v["drivers"]) for v in T.values())
    out_b4 = sum(1 for v in T.values() for e in v["drivers"] if e["b4_persists"])
    own_b1 = sum(v["n_after_b1"] for v in T.values()) - out_a  # B1 removes nothing (threshold 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    stages = ["PCMCI\n(FDR 0.05)", "after B1\ndecoys", "after B2\nlag mask", "after B3\ntop-k", "B4: persist at\n100 regions"]
    own = [own_a, own_b1, own_b2, own_b2, None]
    out = [out_a, out_a, out_b2, out_b3, out_b4]
    x = np.arange(len(stages))
    w = 0.38
    own_x = x[:4] - w / 2 - 0.01
    b1 = ax1.bar(own_x, own[:4], w, color=BLUE, label="links within the target's own region")
    b2 = ax1.bar(x + w / 2 + 0.01, out, w, color=ORANGE, label="links from other regions (the prior)")
    for bars in (b1, b2):
        for b in bars:
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 6, f"{int(b.get_height())}",
                     ha="center", fontsize=8.5, color=INK)
    ax1.set_xticks(x, stages)
    ax1.set_ylabel("links (all 50 targets)")
    ax1.set_title("Links left after each check")
    ax1.legend(loc="lower left", bbox_to_anchor=(0, -0.28), ncol=2, fontsize=8.5)
    ax1.set_ylim(0, 420)
    ax1.grid(axis="x", visible=False)
    note(ax1, "Own-region links are always candidates for the splitter, so B3's budget only counts outside links; "
              "B4 flags, it does not delete.", loc=(0, -0.2))

    n_regions = pd.Series([len(v["prior"]) for v in T.values()]).value_counts().sort_index()
    ax2.bar(n_regions.index, n_regions.values, 0.7, color=BLUE)
    for k, n in n_regions.items():
        ax2.text(k, n + 0.4, str(n), ha="center", fontsize=8.5)
    ax2.set_xlabel("prior regions per target")
    ax2.set_ylabel("targets")
    ax2.set_title(f"Size of each target's prior (mean {np.mean([len(v['prior']) for v in T.values()]):.2f})")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "03_links_through_checks.png")


# ------------------------------------------------------------------ 04 ----
def fig_decoys(a50, b1):
    clean = [abs(e["val"]) for g in a50["graph"].values() for e in g["edges"] if e["src_region"] != -1]
    dec = [abs(e["val"]) for g in b1["graph"].values() for e in g["edges"] if e["src_region"] != -1]
    rep = b1["report"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bins = np.linspace(0, max(clean + dec) * 1.02, 40)
    ax.hist(clean, bins=bins, histtype="step", lw=2, color=BLUE, label=f"real links, clean pass ({len(clean)})")
    ax.hist(dec, bins=bins, histtype="step", lw=2, color=ORANGE, ls="--",
            label=f"real links, pass with decoys in the pool ({len(dec)})")
    ax.set_yscale("log")
    ax.set_xlabel("|MCI partial correlation|")
    ax.set_ylabel("links (log scale)")
    ax.set_title("B1: phase-randomized decoys in the pool")
    ax.text(0.98, 0.95, f"{rep['n_decoys_picked']} of {rep['n_decoys']} decoys selected as a parent\n"
                        f"decoy rate {rep['decoy_rate']:.0%}  (gate < {rep['gate']:.0%}: "
                        f"{'PASS' if rep['passes_gate'] else 'FAIL'})",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color=INK,
            bbox=dict(fc=SURFACE, ec=GRID))
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, fontsize=8.5)
    note(ax, "Decoys keep each source series' spectrum and autocorrelation but have no real link to anything. "
             "Adding them barely changes the real graph.", fig_y=-0.14)
    save(fig, "04_b1_decoys.png")


# ------------------------------------------------------------------ 05 ----
def fig_speed(graph):
    T = graph["targets"]
    kept = [e for t, v in T.items() for e in v["candidates_after_b2"] if e["src_region"] != int(t)]
    # candidates_after_b2 drops the computed distance; recover it from drivers/b2_failed
    info = {(int(t), e["src_region"], e["field"], e["lag"]): e for t, v in T.items() for e in v["drivers"] + v["b2_failed"]}
    pts = []
    for t, v in T.items():
        for e in v["candidates_after_b2"] + v["b2_failed"]:
            if e["src_region"] == int(t):
                continue
            full = info.get((int(t), e["src_region"], e["field"], e["lag"]))
            if full is None:
                continue
            pts.append((e["field"], full["distance_km"], full["implied_speed_ms"], full["b2_pass"]))
    fields = ["sst", "olr", "pw", "h850", "u850", "u200"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharex=True, sharey=True)
    for ax, f in zip(axes.ravel(), fields):
        p = [x for x in pts if x[0] == f]
        ok = [x for x in p if x[3]]
        bad = [x for x in p if not x[3]]
        ax.scatter([x[1] for x in ok], [x[2] for x in ok], s=26, color=BLUE, edgecolor=SURFACE, lw=0.8, zorder=3)
        ax.scatter([x[1] for x in bad], [x[2] for x in bad], s=40, color=CRITICAL, marker="X",
                   edgecolor=SURFACE, lw=0.8, zorder=4)
        ax.axhline(V.SPEED_LIMIT_MS[f], color=INK2, ls="--", lw=1)
        ax.text(0, V.SPEED_LIMIT_MS[f] * 1.12, f"limit {V.SPEED_LIMIT_MS[f]:g} m/s", ha="left",
                fontsize=8, color=INK2)
        ax.set_title(f"{f}  ({len(p)} outside links{', ' + str(len(bad)) + ' rejected' if bad else ''})",
                     fontsize=10.5)
        ax.set_yscale("log")
        ax.set_xlim(-150, 6000)
        ax.set_ylim(0.3, 150)
        if not p:
            ax.text(0.5, 0.5, "no outside links", transform=ax.transAxes, ha="center", color=MUTED)
    for ax in axes[1]:
        ax.set_xlabel("closest distance between regions (km)")
    for ax in axes[:, 0]:
        ax.set_ylabel("implied speed (m/s)")
    axes[0, 0].scatter([], [], s=26, color=BLUE, label="kept")
    axes[0, 0].scatter([], [], s=40, color=CRITICAL, marker="X", label="rejected (faster than the limit)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower left", ncol=2, bbox_to_anchor=(0.01, -0.03))
    fig.suptitle("B2: implied speed = distance / lag, checked against a generous limit per source field",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "05_b2_speed.png")
    return len(kept)


# ------------------------------------------------------------------ 06 ----
def fig_k(graph):
    s = graph["summary"]
    curve = {int(k): v for k, v in s["k_validation_r2"].items()}
    ks = sorted(curve)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.plot(ks, [curve[k] for k in ks], color=BLUE, marker="o", ms=6, mec=SURFACE)
    ax1.scatter([s["k"]], [curve[s["k"]]], s=120, facecolor="none", edgecolor=INK, lw=1.5, zorder=4)
    ax1.annotate(f"chosen k = {s['k']}", (s["k"], curve[s["k"]]), xytext=(-70, -30),
                 textcoords="offset points", fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax1.set_xlabel("k = outside links kept per target")
    ax1.set_ylabel("mean validation R²")
    ax1.set_title("B3: R² on 2005–2012 season days (fit 1979–2004)")
    gains = [(curve[k] - curve[k - 1]) * 1e4 for k in ks[1:]]
    ax2.bar(ks[1:], gains, 0.7, color=BLUE)
    ax2.set_xlabel("k")
    ax2.set_ylabel("R² gain from the k-th link (×10⁻⁴)")
    ax2.set_title("Each extra outside link adds a little, then nothing")
    ax2.grid(axis="x", visible=False)
    note(ax1, "Own-region links are always in the model, so R² starts high (0.962). "
              "k = 10 equals the most outside links any target has: B3 cuts nothing.", loc=(0, -0.2))
    fig.tight_layout()
    save(fig, "06_b3_k.png")


# ------------------------------------------------------------------ 07 ----
def fig_persistence(graph):
    T = graph["targets"]
    scores = sorted(((int(t), v["persistence_score"]) for t, v in T.items() if v["persistence_score"] is not None),
                    key=lambda x: x[1])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [2.4, 1]})
    colors = [BLUE if s >= V.PERSISTENCE_PASS else ORANGE for _, s in scores]
    ax1.bar(range(len(scores)), [s for _, s in scores], 0.75, color=colors)
    ax1.axhline(V.PERSISTENCE_PASS, color=INK2, ls="--", lw=1)
    ax1.set_xticks(range(len(scores)), [str(t) for t, _ in scores], fontsize=7)
    ax1.set_xlabel("target region (sorted by score)")
    ax1.set_ylabel("share of drivers that persist")
    n_pass = sum(1 for _, s in scores if s >= V.PERSISTENCE_PASS)
    ax1.set_title(f"B4: persistence at 100 regions — {n_pass} of {len(scores)} targets pass (≥ 0.5)")
    ax1.legend(handles=[matplotlib.patches.Patch(color=BLUE, label="pass"),
                        matplotlib.patches.Patch(color=ORANGE, label="below 0.5 (flagged)")],
               loc="upper left", fontsize=8.5)
    ax1.grid(axis="x", visible=False)
    note(ax1, "Targets 13 and 42 have no outside driver and are not scored.", loc=(0, -0.2))

    drivers = [e for v in T.values() for e in v["drivers"]]
    fields = ["olr", "u200", "u850", "h850", "pw"]
    share = [np.mean([e["b4_persists"] for e in drivers if e["field"] == f]) for f in fields]
    counts = [sum(e["field"] == f for e in drivers) for f in fields]
    bars = ax2.barh(fields[::-1], share[::-1], 0.6, color=BLUE)
    for b, f, n in zip(bars, fields[::-1], counts[::-1]):
        k = sum(e["b4_persists"] for e in drivers if e["field"] == f)
        ax2.text(b.get_width() + 0.02, b.get_y() + b.get_height() / 2, f"{k} / {n}", va="center", fontsize=9)
    ax2.set_xlim(0, 1.1)
    ax2.set_xlabel("share of links that persist")
    ax2.set_title("By source field")
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "07_b4_persistence.png")


# ------------------------------------------------------------------ 08 ----
def fig_drift():
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, w in zip(axes, (1, 3)):
        s = load(f"stage_d_drift_w{w}.json")["summary"]
        years, d = s["years"], s["mean_distance"]
        for y, ph in s["enso_phase"].items():
            y = int(y)
            if ph != "neutral":
                ax.axvspan(y - 0.5, y + 0.5, color=RED_LIGHT if ph == "nino" else BLUE_LIGHT, lw=0, zorder=0)
        ax.plot(years, d, color=INK, lw=2, marker="o", ms=4, zorder=3)
        for y in s["aggregate_alarm_years"]:
            ax.axvline(y, color=ORANGE, lw=2, ls="--", zorder=2)
        ax.set_ylabel("mean 1 − Jaccard\n(50 targets)")
        ax.set_title(f"{w}-season windows: change in parent sets vs the previous window "
                     f"(phase change vs none: {s['mean_distance_phase_change']:.3f} vs "
                     f"{s['mean_distance_no_change']:.3f}, p = {s['mannwhitney_p']:.2f})", fontsize=10.5)
        ax.grid(axis="x", visible=False)
    axes[1].set_xlabel("year (end of window)")
    axes[1].set_xlim(1979.5, 2022.5)
    handles = [matplotlib.patches.Patch(color=RED_LIGHT, label="El Niño season (Niño-3.4 > +0.5 SD)"),
               matplotlib.patches.Patch(color=BLUE_LIGHT, label="La Niña season (< −0.5 SD)"),
               matplotlib.lines.Line2D([], [], color=ORANGE, ls="--", lw=2, label="STARS regime-shift alarm")]
    fig.legend(handles=handles, loc="lower left", ncol=3, bbox_to_anchor=(0.01, -0.02), fontsize=9)
    fig.suptitle("Stage D: does the causal structure change with ENSO?", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "08_stage_d_drift.png")


# ------------------------------------------------------------------ 09 ----
def fig_stage_c():
    none, pcmci = load("stage_c/full_none.json"), load("stage_c/full_pcmci_restrict.json")
    configs = [("no prior", none, BLUE), ("PCMCI prior (restrict)", pcmci, ORANGE)]
    d_r2 = pcmci["r2_vs_climatology_mean"] - none["r2_vs_climatology_mean"]
    d_skill = pcmci["skill_mean"] - none["skill_mean"]
    if abs(d_r2) < 0.005 and abs(d_skill) < 0.005:
        verdict = "no meaningful difference from no prior"
    elif d_r2 > 0 and d_skill > 0:
        verdict = "PCMCI prior beats no prior on both mean scores"
    elif d_r2 < 0 and d_skill < 0:
        verdict = "PCMCI prior is worse than no prior on both mean scores"
    else:
        verdict = "mixed: PCMCI prior wins on one mean score, loses on the other"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    w = 0.38
    for ax, keys, labels, fmt in (
        (ax1, ["n_beating_climatology", "n_beating_persistence"], ["beat climatology", "beat persistence"], "{:.0f}"),
        (ax2, ["r2_vs_climatology_mean", "skill_mean"], ["R² vs climatology", "skill vs persistence"], "{:+.3f}"),
    ):
        x = np.arange(len(keys))
        for i, (name, s, color) in enumerate(configs):
            vals = [s[k] for k in keys]
            bars = ax.bar(x + (i - 0.5) * (w + 0.02), vals, w, color=color, label=name)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + (1 if fmt == "{:.0f}" else 0.01),
                        fmt.format(v), ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(x, labels)
        ax.grid(axis="x", visible=False)
        ax.axhline(0, color=INK2, lw=0.8)
    ax1.set_ylim(0, 55)
    ax1.set_ylabel(f"regions (of {none['n_places_done']})")
    ax1.set_title("Regions that beat each baseline")
    ax2.set_ylabel("mean over regions")
    ax2.set_title("Mean scores on 2013–2022 season days")
    ax1.legend(loc="upper left", fontsize=8.5)
    fig.suptitle(f"Stage C: does the causal prior help? (full pool, restrict mode) — {verdict}",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    note(ax1, f"Change with the prior: R² {d_r2:+.4f}, skill {d_skill:+.4f}. Compared only against no prior: "
              "the random and lagged-correlation controls were not run in this pass, "
              "so a gain cannot yet be credited to the causal graph itself.")
    save(fig, "09_stage_c_pcmci_vs_none.png")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    graph = load("causal_graph_50.json")
    a50 = load("stage_a_50.json")
    b1 = load("b1_decoy_pass.json")
    fig_season()
    fig_map(graph)
    fig_links(graph, a50)
    fig_decoys(a50, b1)
    fig_speed(graph)
    fig_k(graph)
    fig_persistence(graph)
    fig_drift()
    if (RES / "stage_c" / "full_pcmci_restrict.json").exists():
        fig_stage_c()


if __name__ == "__main__":
    main()
