#!/usr/bin/env python3
"""Calibration uncertainty audit for the IJIS major revision.

Consumes the locked matched-tuning official-test scores.  Reports the declared
10-bin equal-width ECE, Brier score, percentile bootstrap intervals, and bin
occupancy.  This audit quantifies uncertainty rather than interpreting small
four-decimal ECE differences as precise model rankings.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

SEED = 42
N_BOOT = 10_000
N_BINS = 10


def ece_equal_width(y: np.ndarray, scores: np.ndarray, n_bins: int = N_BINS):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        count = int(mask.sum())
        if count:
            mean_score = float(scores[mask].mean())
            empirical_rate = float(y[mask].mean())
            gap = abs(empirical_rate - mean_score)
            ece += count / len(y) * gap
        else:
            mean_score = np.nan
            empirical_rate = np.nan
            gap = np.nan
        rows.append(
            {
                "bin": i + 1,
                "lower": lo,
                "upper": hi,
                "count": count,
                "mean_score": mean_score,
                "empirical_positive_rate": empirical_rate,
                "absolute_gap": gap,
            }
        )
    return float(ece), rows


def bootstrap(y: np.ndarray, scores: np.ndarray):
    rng = np.random.default_rng(SEED)
    eces = []
    briers = []
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        ss = scores[idx]
        ece, _ = ece_equal_width(yy, ss)
        eces.append(ece)
        briers.append(float(brier_score_loss(yy, ss)))
    return {
        "ECE_CI_low": float(np.percentile(eces, 2.5)),
        "ECE_CI_high": float(np.percentile(eces, 97.5)),
        "Brier_CI_low": float(np.percentile(briers, 2.5)),
        "Brier_CI_high": float(np.percentile(briers, 97.5)),
    }


def main() -> None:
    output_dir = Path(os.environ.get("FAIR_BETH_REVISION_OUTPUT", "results/revision_audits")).resolve()
    scores_path = output_dir / "matched_tuning_test_scores.csv"
    if not scores_path.exists():
        raise SystemExit(f"Missing locked matched-tuning scores: {scores_path}")
    data = pd.read_csv(scores_path)
    y = data.pop("y_true").to_numpy(dtype=int)

    summary_rows = []
    bin_rows = []
    for model in data.columns:
        scores = data[model].to_numpy(dtype=float)
        ece, bins = ece_equal_width(y, scores)
        brier = float(brier_score_loss(y, scores))
        ci = bootstrap(y, scores)
        summary_rows.append(
            {
                "Model": model,
                "N": len(y),
                "Positives": int(y.sum()),
                "Prevalence": float(y.mean()),
                "ECE": ece,
                "Brier": brier,
                "Binning": "10 equal-width bins on [0,1]",
                **ci,
            }
        )
        for row in bins:
            bin_rows.append({"Model": model, **row})

    summary = pd.DataFrame(summary_rows).sort_values("Model")
    detail = pd.DataFrame(bin_rows)
    summary.to_csv(output_dir / "calibration_uncertainty.csv", index=False)
    detail.to_csv(output_dir / "calibration_bin_occupancy.csv", index=False)

    print("=== CALIBRATION WITH UNCERTAINTY ===")
    print(
        summary[
            [
                "Model",
                "ECE",
                "ECE_CI_low",
                "ECE_CI_high",
                "Brier",
                "Brier_CI_low",
                "Brier_CI_high",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
