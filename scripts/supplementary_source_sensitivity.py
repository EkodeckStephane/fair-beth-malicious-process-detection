#!/usr/bin/env python3
"""Leave-one-supplementary-file-out sensitivity for the BETH development pool.

The audit addresses whether the threshold-transfer result depends on a particular
2021may process-event file.  It uses the matched-tuning TabRF configuration
locked by ``matched_hyperparameter_tuning.py``.  For each omitted supplementary
file, the event vocabulary is rebuilt from the remaining development records,
50 development-only threshold estimates are obtained when the class counts
permit it, their median is locked, and the final model is evaluated once on the
official test split.

No DNS file is silently mixed into this experiment: DNS telemetry has a
different schema and does not provide the (hostName, processId) experimental
unit used by FAIR-BETH.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from beth_limit_lifting_analyses import aggregate_records, load_csvs, records_to_features

SEED = 42
REPEATS = 10
MAX_FOLDS = 5
FPR_BUDGET = 0.05
OFFICIAL_TRAIN = "labelled_training_data.csv"
OFFICIAL_TEST = "labelled_testing_data.csv"


def choose_threshold_by_fpr(y_val: np.ndarray, scores: np.ndarray) -> float:
    fpr, _, thresholds = roc_curve(y_val, scores)
    valid = np.where(fpr <= FPR_BUDGET)[0]
    return float(thresholds[valid[-1]]) if len(valid) else float(np.max(scores) + 1e-12)


def make_model(params: dict, seed: int) -> RandomForestClassifier:
    allowed = {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}
    clean = {key: value for key, value in params.items() if key in allowed}
    return RandomForestClassifier(
        **clean,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def test_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "AUC": float(roc_auc_score(y, scores)),
        "AP": float(average_precision_score(y, scores)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "FPR": float(fp / max(tn + fp, 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("results/revision_audits"), type=Path)
    parser.add_argument(
        "--tuning-lock",
        default=Path("results/revision_audits/matched_tuning_lock.json"),
        type=Path,
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.tuning_lock.exists():
        raise SystemExit(
            "Matched-tuning lock is required before source sensitivity; refusing an ad-hoc RF configuration."
        )
    with args.tuning_lock.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    tabrf = lock.get("best", {}).get("TabRF")
    if not tabrf:
        raise SystemExit("TabRF configuration is absent from matched_tuning_lock.json")
    params = tabrf["params"]

    supplementary = sorted(
        path.name
        for path in args.data_dir.glob("labelled_2021may*.csv")
        if not path.name.endswith("-dns.csv")
    )
    if not supplementary:
        raise SystemExit(f"No non-DNS 2021may files found in {args.data_dir}")

    # Aggregate each source independently so source membership is explicit.
    records_by_file = {}
    for filename in [OFFICIAL_TRAIN, *supplementary, OFFICIAL_TEST]:
        frame = load_csvs([filename])
        records_by_file[filename] = aggregate_records(frame)
        print(
            f"[source] {filename}: processes={len(records_by_file[filename])} "
            f"positive={sum(int(r['y']) for r in records_by_file[filename])}"
        )

    # The test records are represented here, but labels are not consumed until
    # every source-specific threshold has been fixed.
    test_records = records_by_file[OFFICIAL_TEST]

    summary_rows = []
    threshold_rows = []
    locked_experiments = []

    for omitted in supplementary:
        included_names = [OFFICIAL_TRAIN] + [name for name in supplementary if name != omitted]
        dev_records = [record for name in included_names for record in records_by_file[name]]
        y_dev = np.asarray([record["y"] for record in dev_records], dtype=int)
        n_pos = int(y_dev.sum())
        n_neg = int(len(y_dev) - n_pos)

        if n_pos < 2 or n_neg < 2:
            summary_rows.append(
                {
                    "ExcludedFile": omitted,
                    "Status": "insufficient_two_class_development_data",
                    "DevelopmentProcesses": len(y_dev),
                    "DevelopmentPositives": n_pos,
                    "DevelopmentNegatives": n_neg,
                }
            )
            continue

        vocab = sorted({int(eid) for record in dev_records for eid in record["eventId"]})
        X_dev, y_dev_check, _ = records_to_features(dev_records, vocab, prefix="full")
        if not np.array_equal(y_dev, y_dev_check):
            raise RuntimeError("Label mismatch while rebuilding development features")

        n_splits = min(MAX_FOLDS, n_pos, n_neg)
        thresholds = []
        valid_fold_counter = 0
        for repeat in range(REPEATS):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED + repeat)
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev), start=1):
                if len(np.unique(y_dev[train_idx])) < 2 or len(np.unique(y_dev[val_idx])) < 2:
                    continue
                model = make_model(params, seed=SEED + repeat * 100 + fold)
                model.fit(X_dev[train_idx], y_dev[train_idx])
                val_scores = model.predict_proba(X_dev[val_idx])[:, 1]
                threshold = choose_threshold_by_fpr(y_dev[val_idx], val_scores)
                thresholds.append(float(threshold))
                valid_fold_counter += 1
                threshold_rows.append(
                    {
                        "ExcludedFile": omitted,
                        "Repeat": repeat + 1,
                        "Fold": fold,
                        "TrainN": len(train_idx),
                        "TrainPos": int(y_dev[train_idx].sum()),
                        "ValN": len(val_idx),
                        "ValPos": int(y_dev[val_idx].sum()),
                        "Threshold": float(threshold),
                    }
                )

        if not thresholds:
            summary_rows.append(
                {
                    "ExcludedFile": omitted,
                    "Status": "no_valid_threshold_fold",
                    "DevelopmentProcesses": len(y_dev),
                    "DevelopmentPositives": n_pos,
                    "DevelopmentNegatives": n_neg,
                    "NFolds": n_splits,
                }
            )
            continue

        threshold_array = np.asarray(thresholds, dtype=float)
        locked_threshold = float(np.median(threshold_array))
        final_model = make_model(params, seed=SEED)
        final_model.fit(X_dev, y_dev)

        # Store everything needed for evaluation without touching test labels.
        locked_experiments.append(
            {
                "omitted": omitted,
                "included_names": included_names,
                "n_dev": len(y_dev),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "vocab": vocab,
                "model": final_model,
                "threshold": locked_threshold,
                "thresholds": threshold_array,
                "valid_fold_counter": valid_fold_counter,
                "n_splits": n_splits,
            }
        )

    # All configurations and thresholds are now fixed. Evaluate each on the
    # isolated official test split using a vocabulary fit only on its own
    # remaining development pool.
    y_test = np.asarray([record["y"] for record in test_records], dtype=int)
    for exp in locked_experiments:
        X_test, y_test_check, _ = records_to_features(test_records, exp["vocab"], prefix="full")
        if not np.array_equal(y_test, y_test_check):
            raise RuntimeError("Official-test label order changed during feature reconstruction")
        test_scores = exp["model"].predict_proba(X_test)[:, 1]
        metrics = test_metrics(y_test, test_scores, exp["threshold"])
        q1, q3 = np.percentile(exp["thresholds"], [25, 75])
        summary_rows.append(
            {
                "ExcludedFile": exp["omitted"],
                "Status": "ok",
                "DevelopmentProcesses": exp["n_dev"],
                "DevelopmentPositives": exp["n_pos"],
                "DevelopmentNegatives": exp["n_neg"],
                "NFolds": exp["n_splits"],
                "ValidThresholdFits": exp["valid_fold_counter"],
                "ThresholdMedian": exp["threshold"],
                "ThresholdQ1": float(q1),
                "ThresholdQ3": float(q3),
                "ThresholdMin": float(np.min(exp["thresholds"])),
                "ThresholdMax": float(np.max(exp["thresholds"])),
                **metrics,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("ExcludedFile")
    details = pd.DataFrame(threshold_rows).sort_values(["ExcludedFile", "Repeat", "Fold"])
    summary.to_csv(args.output_dir / "supplementary_source_sensitivity.csv", index=False)
    details.to_csv(args.output_dir / "supplementary_source_thresholds.csv", index=False)

    metadata = {
        "model": "matched-tuned TabRF",
        "matched_tuning_lock_sha256": lock.get("lock_sha256"),
        "params": params,
        "threshold_policy": "median of repeated development-only validation-FPR thresholds",
        "fpr_budget": FPR_BUDGET,
        "repeats": REPEATS,
        "max_folds": MAX_FOLDS,
        "supplementary_files": supplementary,
    }
    with (args.output_dir / "supplementary_source_sensitivity_protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print("\n=== LEAVE-ONE-SUPPLEMENTARY-FILE-OUT SENSITIVITY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
