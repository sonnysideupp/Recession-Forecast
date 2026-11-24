"""Compare the original (leaky) evaluation with leak-free ones.

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
from recession.evaluate import (
    leak_free_single_split,
    leak_free_walk_forward,
    original_leaky,
)
from recession.models import MODEL_NAMES

OUT = Path(__file__).resolve().parent / "results"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    raw = load_raw()
    results: dict[str, dict] = {"original_leaky_single_split": {}}

    for name, scores in original_leaky(pd.read_csv(DATA_DIR / "Dataset_Cleaned.csv")).items():
        results["original_leaky_single_split"][name] = asdict(scores)

    results["leak_free_single_split"] = {}
    results["leak_free_walk_forward"] = {}
    predictions = {}
    for name in MODEL_NAMES:
        scores, _ = leak_free_single_split(name, raw)
        results["leak_free_single_split"][name] = asdict(scores)
        scores, preds = leak_free_walk_forward(name, raw)
        results["leak_free_walk_forward"][name] = asdict(scores)
        predictions[name] = preds
        print(f"done: {name}", flush=True)

    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    pred_frame = pd.DataFrame(predictions)
    pred_frame["recession_next_month"] = raw["recession"].shift(-1).loc[pred_frame.index]
    pred_frame.to_csv(OUT / "predictions.csv", index_label="Date")
    plot(pred_frame)

    for protocol, by_model in results.items():
        print(f"\n{protocol}")
        for name, s in by_model.items():
            print(
                f"  {name:18s} ROC AUC {s['roc_auc']:.3f}  AP {s['avg_precision']:.3f}  "
                f"Brier {s['brier']:.3f}  ({s['n_recession_months']}/{s['n_months']} recession)"
            )


def plot(pred_frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    rec = pred_frame["recession_next_month"]
    ax.fill_between(rec.index, 0, rec.values, step="pre", color="0.85", label="NBER recession")
    for name in MODEL_NAMES:
        ax.plot(pred_frame.index, pred_frame[name], label=name, linewidth=1.2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(recession next month)")
    ax.set_title("Walk-forward out-of-sample forecasts, 2002-2024 (refit each January)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "walk_forward.png", dpi=150)


if __name__ == "__main__":
    main()
