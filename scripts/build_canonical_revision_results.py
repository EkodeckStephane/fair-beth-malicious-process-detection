#!/usr/bin/env python3
"""Build canonical, internally consistent result tables for the IJIS major revision.

Inputs are reviewer-driven audit outputs already frozen under
``results/revision_audits``.  This script does not retrain models.  It selects
one declared operating policy (development-only FPR budget), computes paired
comparisons on the common official BETH test set, applies Holm correction to
McNemar tests, and writes the small set of tables that the paper and supplement
should cite.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results" / "revision_audits"
OUT = ROOT / "results" / "canonical"
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY_POLICY = "val_fpr_budget"
REFERENCE_MODEL = "TabRF"
BOOTSTRAP_REPS = 20_000
SEED = 20260826


def f1_from_arrays(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    den = 2 * tp + fp + fn
    return 0.0 if den == 0 else (2.0 * tp) / den


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value via Binomial(n=b+c, p=.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Sum the lower binomial tail exactly enough for the tiny paired test set.
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * float(p_values[idx])
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def bootstrap_delta_f1(
    y: np.ndarray,
    pred_ref: np.ndarray,
    pred_other: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, float, float]:
    yb = y[indices]
    pr = pred_ref[indices]
    po = pred_other[indices]

    def batch_f1(p: np.ndarray) -> np.ndarray:
        tp = np.sum((yb == 1) & (p == 1), axis=1)
        fp = np.sum((yb == 0) & (p == 1), axis=1)
        fn = np.sum((yb == 1) & (p == 0), axis=1)
        den = 2 * tp + fp + fn
        return np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0)

    delta = batch_f1(pr) - batch_f1(po)
    observed = f1_from_arrays(y, pred_ref) - f1_from_arrays(y, pred_other)
    lo, hi = np.quantile(delta, [0.025, 0.975])
    return float(observed), float(lo), float(hi)


def main() -> None:
    test = pd.read_csv(AUDIT / "matched_tuning_test_results.csv")
    scores = pd.read_csv(AUDIT / "matched_tuning_test_scores.csv")

    primary = (
        test.loc[test["ThresholdPolicy"].eq(PRIMARY_POLICY)]
        .copy()
        .sort_values(["F1", "AP", "AUC"], ascending=False)
        .reset_index(drop=True)
    )
    if primary.empty or REFERENCE_MODEL not in set(primary["Model"]):
        raise RuntimeError("Primary matched-tuning results or TabRF reference missing")

    # Canonical main table: only the predeclared development-only FPR-budget policy.
    primary.to_csv(OUT / "main_results.csv", index=False)

    y = scores["y_true"].to_numpy(dtype=int)
    if len(y) != 198 or int(y.sum()) != 121:
        raise RuntimeError(f"Unexpected official BETH test composition: n={len(y)}, positives={int(y.sum())}")

    thresholds = primary.set_index("Model")["Threshold"].to_dict()
    models = [m for m in primary["Model"].tolist() if m in scores.columns]
    pred = {m: (scores[m].to_numpy(dtype=float) >= float(thresholds[m])).astype(int) for m in models}

    ref_pred = pred[REFERENCE_MODEL]
    ref_correct = ref_pred == y
    rng = np.random.default_rng(SEED)
    boot_idx = rng.integers(0, len(y), size=(BOOTSTRAP_REPS, len(y)), dtype=np.int16)

    paired_rows = []
    raw_p = []
    for model in models:
        if model == REFERENCE_MODEL:
            continue
        other = pred[model]
        other_correct = other == y
        b = int(np.sum(ref_correct & ~other_correct))
        c = int(np.sum(~ref_correct & other_correct))
        p = exact_mcnemar_p(b, c)
        delta, lo, hi = bootstrap_delta_f1(y, ref_pred, other, boot_idx)
        raw_p.append(p)
        paired_rows.append(
            {
                "ReferenceModel": REFERENCE_MODEL,
                "Comparator": model,
                "ThresholdPolicy": PRIMARY_POLICY,
                "N": len(y),
                "DeltaF1_ReferenceMinusComparator": delta,
                "DeltaF1_CI_low": lo,
                "DeltaF1_CI_high": hi,
                "McNemar_b_ref_correct_only": b,
                "McNemar_c_comparator_correct_only": c,
                "McNemar_p_exact": p,
            }
        )

    holm = holm_adjust(raw_p)
    for row, p_adj in zip(paired_rows, holm):
        row["McNemar_p_Holm"] = p_adj
        row["Holm_significant_0_05"] = bool(p_adj < 0.05)
    paired = pd.DataFrame(paired_rows).sort_values("DeltaF1_ReferenceMinusComparator", ascending=False)
    paired.to_csv(OUT / "paired_comparisons.csv", index=False)

    # Diagnostic gaps used in the FAIR-X interpretation.
    by_model = primary.set_index("Model")
    diag = []
    if "MetaRF" in by_model.index:
        diag.append({
            "Metric": "TDG_F1",
            "ModelA": "TabRF",
            "ModelB": "MetaRF",
            "ValueA": float(by_model.loc["TabRF", "F1"]),
            "ValueB": float(by_model.loc["MetaRF", "F1"]),
            "Gap": float(by_model.loc["TabRF", "F1"] - by_model.loc["MetaRF", "F1"]),
            "Interpretation": "tabular-vs-meta decision gap under the same transferred FPR-budget policy",
        })
    if "ScoreRF" in by_model.index:
        diag.append({
            "Metric": "SCG_F1",
            "ModelA": "TabRF",
            "ModelB": "ScoreRF",
            "ValueA": float(by_model.loc["TabRF", "F1"]),
            "ValueB": float(by_model.loc["ScoreRF", "F1"]),
            "Gap": float(by_model.loc["TabRF", "F1"] - by_model.loc["ScoreRF", "F1"]),
            "Interpretation": "tabular-vs-score decision gap under the same transferred FPR-budget policy",
        })

    ap_winner = primary.sort_values("AP", ascending=False).iloc[0]
    f1_winner = primary.sort_values("F1", ascending=False).iloc[0]
    diag.append({
        "Metric": "RankingDecisionSeparation",
        "ModelA": str(ap_winner["Model"]),
        "ModelB": str(f1_winner["Model"]),
        "ValueA": float(ap_winner["AP"]),
        "ValueB": float(f1_winner["F1"]),
        "Gap": np.nan,
        "Interpretation": "AP winner need not be the fixed-threshold F1 winner under transfer",
    })

    fpr_rows = test.loc[test["ThresholdPolicy"].eq("val_fpr_budget")].set_index("Model")
    maxf_rows = test.loc[test["ThresholdPolicy"].eq("val_max_f1")].set_index("Model")
    for model in sorted(set(fpr_rows.index) & set(maxf_rows.index)):
        diag.append({
            "Metric": "ThresholdPolicyDeltaF1",
            "ModelA": model,
            "ModelB": "same model / val_max_f1",
            "ValueA": float(fpr_rows.loc[model, "F1"]),
            "ValueB": float(maxf_rows.loc[model, "F1"]),
            "Gap": float(fpr_rows.loc[model, "F1"] - maxf_rows.loc[model, "F1"]),
            "Interpretation": "F1(val_fpr_budget) - F1(val_max_f1) on the untouched official test set",
        })
    pd.DataFrame(diag).to_csv(OUT / "diagnostic_gaps.csv", index=False)

    # Small auditable copies of reviewer-specific evidence.  Keep original audit
    # files untouched; canonical copies are the paper-facing data sources.
    copy_map = {
        AUDIT / "calibration_uncertainty.csv": OUT / "calibration_uncertainty.csv",
        AUDIT / "dataset_inventory.csv": OUT / "dataset_inventory.csv",
        AUDIT / "host_overlap.csv": OUT / "host_overlap.csv",
        AUDIT / "dns_schema_comparison.csv": OUT / "dns_schema_comparison.csv",
        AUDIT / "supplementary_source_sensitivity.csv": OUT / "supplementary_sensitivity.csv",
        AUDIT / "sequence_length_matched" / "sequence_length_matched_ablation.csv": OUT / "sequence_ablation.csv",
    }
    for src, dst in copy_map.items():
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copyfile(src, dst)

    readme = f"""# Canonical IJIS major-revision results\n\nThese files are generated by `scripts/build_canonical_revision_results.py` from the frozen reviewer-driven audit outputs in `results/revision_audits/`.\n\nThe primary operating policy is `{PRIMARY_POLICY}`: model hyperparameters are selected with the matched development-only search budget, thresholds are selected before official-test labels are loaded, and the locked thresholds are transferred unchanged to the common BETH test set. `main_results.csv` is therefore the single source for revised matched-tabular performance claims.\n\n`paired_comparisons.csv` compares TabRF with every other matched model on the same 198 official-test processes (121 malicious), with a {BOOTSTRAP_REPS:,}-resample paired bootstrap confidence interval for Delta-F1, an exact McNemar test, and Holm correction across the planned TabRF-vs-baseline family.\n\nReviewer-specific robustness evidence is exposed separately for calibration uncertainty, raw dataset identity/schema/host checks, leave-one-supplementary-source-out sensitivity, and the length-matched 1,152-token recurrent ablation.\n\nDo not hand-edit generated CSVs. Regenerate them from the script after any upstream audit change.\n"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    print(f"Wrote canonical revision results to {OUT}")
    print(primary[["Model", "F1", "FPR", "AP", "AUC"]].to_string(index=False))
    print("\nPaired comparisons (Holm-adjusted):")
    print(paired[["Comparator", "DeltaF1_ReferenceMinusComparator", "DeltaF1_CI_low", "DeltaF1_CI_high", "McNemar_p_Holm"]].to_string(index=False))


if __name__ == "__main__":
    main()
