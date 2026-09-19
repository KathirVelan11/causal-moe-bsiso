"""SS9 step 5: drift detection (SS4.4) on real single-place BSISO data,
using the splitter + expert trained in step 4.

What this produces (SS6 "Detector reliability -- latency + false-alarm rate,
per channel AND for the fused rule (both OR and AND, reported)"):
  - Channel 1 (ADWIN over the expert's rolling forecast error),
  - Channel 2 (STARS/Rodionov over the causal-signature distance series),
  - both fusion rules (OR and AND),
  each evaluated against BOTH deliberate-shift windows SS8 defines:
    1. abrupt  -- 1997-03-01 .. 1998-07-31 (El Nino 1997-98),
    2. gradual -- 2013-01-01 .. 2022-12-31 (Arora et al. 2026 MISO drift).

SS8's stated expectation for the GRADUAL window is the interesting one:
"Channel 2 (causal-set similarity) shows accumulating drift across those
years while Channel 1 (error) stays comparatively flat, since that contrast
is the whole reason two channels exist." That contrast is reported
explicitly below.

Usage:
    python scripts/run_step5_drift.py [--target 22] [--variant direct]
        [--epochs 5] [--checkpoint PATH]
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print = functools.partial(print, flush=True)  # noqa: A001

import numpy as np
import torch

from causal_moe.data.candidate_edges import (
    build_candidate_source_set,
    expand_source_clusters_to_edge_index,
)
from causal_moe.data.splits import EL_NINO_1997_98, GRADUAL_DRIFT_WINDOW, deliberate_shift_mask
from causal_moe.drift.channels import (
    causal_signature_distance_series,
    evaluate_detector,
    fuse_channels,
    run_channel1_error_adwin,
    run_channel2_causal_stars,
)
from causal_moe.experts.expert import PlaceExpert
from causal_moe.splitter.dirgnn import DIRGNNSplitter
from scripts.train_step4_single_place import (
    CACHE_PATH,
    OLR_LAG0_CHANNEL,
    build_generator_windows,
    load_cache,
    train_place,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=22)
    parser.add_argument("--variant", type=str, default="direct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n-samples-cap", type=int, default=0)
    parser.add_argument("--r", type=float, default=None)
    parser.add_argument("--n-swaps", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generator-window", type=int, default=10)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--adwin-delta", type=float, default=0.002)
    parser.add_argument("--stars-cutoff", type=int, default=120)
    parser.add_argument("--stars-p", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-years", type=int, default=15,
        help="train the step-4 pair on only the FIRST N years, then run drift "
             "detection forward over the whole record (see note below)",
    )
    args = parser.parse_args()

    # Drift detection is a TIME-SERIES analysis: both channels (ADWIN,
    # STARS) assume consecutive samples are consecutive days. A random
    # subsample would silently destroy that, so any cap here is applied as
    # a chronological prefix instead.
    args.cap_contiguous = True

    RESULTS_DIR.mkdir(exist_ok=True)

    cache = load_cache()
    features_all = cache["features"]
    targets_all = cache["targets"]
    sample_dates = cache["sample_dates"]
    n_clusters = cache["n_clusters"]
    target = args.target

    # --- Baseline-only training (SS4.4's premise) -------------------------
    # The expert/splitter must be trained on a PRE-SHIFT baseline period
    # only. Training across the whole record (including the deliberate-shift
    # windows SS8 defines) would let the model adapt to those very shifts,
    # muting the error rise and causal-set movement the detector is supposed
    # to catch -- the detector would then be tested on data its own model
    # had already absorbed. Training on the first `--train-years` years and
    # then running detection FORWARD over the full record keeps the shift
    # windows genuinely unseen.
    days_per_year = 365.25
    n_train = int(args.train_years * days_per_year)
    n_train = min(n_train, features_all.shape[0])
    train_args = argparse.Namespace(**vars(args))
    train_args.n_samples_cap = n_train
    train_args.cap_contiguous = True

    print(f"training step-4 splitter+expert for place {target}, variant={args.variant} "
          f"on the first {args.train_years} years ({n_train} samples, "
          f"{str(sample_dates[0])[:10]} .. {str(sample_dates[n_train-1])[:10]}); "
          f"drift detection then runs forward over all {features_all.shape[0]} samples")
    trained = train_place(
        variant=args.variant,
        target=target,
        features_all=features_all,
        targets_all=targets_all,
        sample_time_index=cache["sample_time_index"],
        real_edge_index=cache["edge_index"],
        n_clusters=n_clusters,
        args=train_args,
    )
    splitter: DIRGNNSplitter = trained["splitter"]
    expert: PlaceExpert = trained["expert"]
    # Detection runs over the ENTIRE record, not just the training slice.
    keep_idx = np.arange(features_all.shape[0])

    # --- Per-timestep signals over the WHOLE kept record ------------------
    candidate_set = build_candidate_source_set(cache["edge_index"], target, args.variant, n_clusters=n_clusters)
    sources = candidate_set.source_clusters
    edge_index = torch.from_numpy(expand_source_clusters_to_edge_index(sources, target))

    olr_lag0_all = features_all[:, :, OLR_LAG0_CHANNEL]
    gen_windows_all = build_generator_windows(olr_lag0_all, cache["sample_time_index"], args.generator_window)
    x_all = torch.from_numpy(features_all[keep_idx])
    gen_windows = torch.from_numpy(gen_windows_all[keep_idx])
    y_target = targets_all[keep_idx][:, target]
    dates = sample_dates[keep_idx]

    print(f"scoring {x_all.shape[0]} timesteps for signature + error series...")
    t0 = time.time()
    splitter.eval()
    expert.eval()
    signatures = np.empty((x_all.shape[0], edge_index.shape[1]), dtype=np.float64)
    errors = np.empty(x_all.shape[0], dtype=np.float64)
    with torch.no_grad():
        for i in range(x_all.shape[0]):
            split = splitter.split(gen_windows[i], edge_index)
            signatures[i] = split.edge_scores.numpy()
            pred = expert(x_all[i], target, split.causal_edge_index, split.causal_edge_weight)
            errors[i] = abs(float(pred) - float(y_target[i]))
    print(f"  scored in {time.time()-t0:.1f}s")

    signature_distance = causal_signature_distance_series(signatures, baseline_window=365)

    # --- Channels ---------------------------------------------------------
    ch1 = run_channel1_error_adwin(errors, delta=args.adwin_delta)
    ch2 = run_channel2_causal_stars(signature_distance, cut_off_length=args.stars_cutoff, p=args.stars_p)
    print(f"\nChannel 1 (ADWIN on error): {len(ch1.alarm_indices)} alarms")
    print(f"Channel 2 (STARS on causal signature): {len(ch2.alarm_indices)} alarms")

    T = errors.shape[0]
    fused_or = fuse_channels(ch1, ch2, T, rule="OR")
    fused_and = fuse_channels(ch1, ch2, T, rule="AND")
    print(f"Fused OR: {len(fused_or.alarm_indices)} alarms")
    print(f"Fused AND: {len(fused_and.alarm_indices)} alarms")

    # --- Evaluate against BOTH deliberate-shift windows (SS8) -------------
    windows = {
        "abrupt_el_nino_1997_98": deliberate_shift_mask(dates, EL_NINO_1997_98),
        "gradual_drift_2013_2022": deliberate_shift_mask(dates, GRADUAL_DRIFT_WINDOW),
    }

    report: dict = {
        "target": target,
        "variant": args.variant,
        "n_timesteps": int(T),
        "n_candidate_sources": int(sources.shape[0]),
        "train_years": args.train_years,
        "n_train_samples": int(n_train),
        "train_end_date": str(sample_dates[n_train - 1])[:10],
        "detection_span": [str(sample_dates[0])[:10], str(sample_dates[-1])[:10]],
        "expert_mse_train_period": trained["expert_mse"],
        "persistence_mse_train_period": trained["persistence_mse"],
        "windows": {},
    }

    for window_name, mask in windows.items():
        print(f"\n=== deliberate-shift window: {window_name} "
              f"({int(mask.sum())} of {T} samples) ===")
        entries = {}
        for label, alarms in [
            ("channel1_error", ch1.alarm_indices),
            ("channel2_causal", ch2.alarm_indices),
            ("fused_OR", fused_or.alarm_indices),
            ("fused_AND", fused_and.alarm_indices),
        ]:
            ev = evaluate_detector(alarms, mask, rule=label)
            entries[label] = {
                "n_alarms": ev.n_alarms,
                "detected_in_window": ev.detected,
                "latency_days": ev.latency_days,
                "false_alarms_outside_window": ev.false_alarms_outside_window,
                "false_alarm_rate_per_year": round(ev.false_alarm_rate_per_year, 4),
            }
            print(f"  {label:>16}: detected={ev.detected}  latency={ev.latency_days}  "
                  f"outside={ev.false_alarms_outside_window}  "
                  f"FAR/yr={ev.false_alarm_rate_per_year:.3f}")
        report["windows"][window_name] = entries

        # SS8's specific expectation for the gradual window: Channel 2
        # accumulates while Channel 1 stays flat. Quantify that contrast.
        in_w = mask
        out_w = ~mask
        if in_w.any() and out_w.any():
            ch2_contrast = float(signature_distance[in_w].mean() - signature_distance[out_w].mean())
            ch1_contrast = float(errors[in_w].mean() - errors[out_w].mean())
            report["windows"][window_name]["signal_contrast"] = {
                "causal_distance_in_minus_out": round(ch2_contrast, 5),
                "error_in_minus_out": round(ch1_contrast, 5),
            }
            print(f"  contrast: causal-distance(in-out)={ch2_contrast:+.4f}  "
                  f"error(in-out)={ch1_contrast:+.4f}")

    out_path = RESULTS_DIR / f"step5_drift_place{target}_{args.variant}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")

    np.savez(
        RESULTS_DIR / f"step5_signals_place{target}_{args.variant}.npz",
        errors=errors,
        signature_distance=signature_distance,
        signatures=signatures,
        dates=dates,
        ch1_alarms=np.array(ch1.alarm_indices, dtype=np.int64),
        ch2_alarms=np.array(ch2.alarm_indices, dtype=np.int64),
    )
    print(f"wrote signal arrays for step 6 reuse")


if __name__ == "__main__":
    main()
