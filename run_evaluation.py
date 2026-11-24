"""Leak-free recession forecasts at several horizons, plus the original
(leaky, 1-month) protocol for comparison.

    uv run python run_evaluation.py

Writes results/metrics.json, results/predictions.csv and results/walk_forward.png.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from recession.data import DATA_DIR, load_raw
from recession.evaluate import leak_free_single_split, leak_free_walk_forward, original_leaky
from recession.models import MODEL_NAMES

OUT = Path(__file__).resolve().parent / "results"
HORIZONS = (1, 3, 6, 12)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    raw = load_raw()
    leaky = original_leaky(pd.read_csv(DATA_DIR / "Dataset_Cleaned.csv"))
    results: dict = {"original_leaky_1m_single_split": {k: asdict(v) for k, v in leaky.items()}}

    frames = []
    for h in HORIZONS:
        results[f"{h}m"] = {"single_split": {}, "walk_forward": {}}
        preds = {}
        for name in MODEL_NAMES:
            scores, _ = leak_free_single_split(name, raw, h)
            results[f"{h}m"]["single_split"][name] = asdict(scores)
            scores, p = leak_free_walk_forward(name, raw, h)
            results[f"{h}m"]["walk_forward"][name] = asdict(scores)
            preds[name] = p
        frame = pd.DataFrame(preds)
        frame["recession_actual"] = raw["recession"].shift(-h).loc[frame.index]
        frame.insert(0, "horizon_months", h)
        frames.append(frame)
        print(f"done: {h}-month horizon", flush=True)

    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.concat(frames).to_csv(OUT / "predictions.csv", index_label="forecast_origin")
    plot(frames)
    report(results)


def report(results: dict) -> None:
    print("\noriginal leaky protocol (1-month horizon, single split)")
    for name, s in results["original_leaky_1m_single_split"].items():
        print(f"  {name:18s} ROC AUC {s['roc_auc']:.3f}  AP {s['avg_precision']:.3f}")
    for h in HORIZONS:
        print(f"\n{h}-month horizon (leak-free)")
        for protocol in ("single_split", "walk_forward"):
            for name, s in results[f"{h}m"][protocol].items():
                print(
                    f"  {protocol:12s} {name:18s} ROC AUC {s['roc_auc']:.3f}  "
                    f"AP {s['avg_precision']:.3f}  Brier {s['brier']:.3f}  "
                    f"({s['n_recession_months']}/{s['n_months']} recession)"
                )


def plot(frames: list[pd.DataFrame]) -> None:
    """One panel per horizon, plotted by the month being forecast (origin + h)."""
    fig, axes = plt.subplots(len(frames), 1, figsize=(11, 2.6 * len(frames)), sharex=True)
    for ax, frame in zip(axes, frames, strict=True):
        h = int(frame["horizon_months"].iloc[0])
        target_month = frame.index + pd.DateOffset(months=h)
        ax.fill_between(
            target_month,
            0,
            frame["recession_actual"].values,
            step="pre",
            color="0.85",
            label="NBER recession",
        )
        for name in MODEL_NAMES:
            ax.plot(target_month, frame[name], label=name, linewidth=1.1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(recession)")
        ax.set_title(f"Forecast made {h} month{'s' if h > 1 else ''} ahead", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Month being forecast")
    fig.suptitle("Walk-forward out-of-sample forecasts, refit each January (2002-2024)")
    fig.tight_layout()
    fig.savefig(OUT / "walk_forward.png", dpi=150)


if __name__ == "__main__":
    main()
