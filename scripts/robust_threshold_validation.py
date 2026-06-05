#!/usr/bin/env python3
"""Repeated cross-fitted threshold selection for the BETH process detector."""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold


SEED = 42
FPR_BUDGET = 0.05
N_SPLITS = 5
N_REPEATS = 10
N_ESTIMATORS = 500

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PIPELINE_OUTPUT = Path(
    os.environ.get("FAIR_BETH_PIPELINE_OUTPUT", SCRIPT_DIR / "pipeline_output")
).resolve()
OUTPUT_DIR = Path(
    os.environ.get(
        "FAIR_BETH_ROBUST_THRESHOLD_OUTPUT",
        REPO_ROOT / "results" / "robust_threshold_validation",
    )
).resolve()


def load_pickle(name):
    with open(PIPELINE_OUTPUT / name, "rb") as handle:
        return pickle.load(handle)


def choose_threshold_by_fpr(y_true, scores, budget=FPR_BUDGET):
    fpr, _, thresholds = roc_curve(y_true, scores)
    finite = np.isfinite(thresholds)
    valid = np.where((fpr <= budget) & finite)[0]
    if not len(valid):
        return float(np.nextafter(np.max(scores), np.inf))
    return float(thresholds[valid[-1]])


def threshold_metrics(y_true, scores, threshold):
    pred = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "ap": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "observed_fpr": float(fp / max(fp + tn, 1)),
    }


def make_model(seed):
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    development_parts = [
        load_pickle("train_data.pkl"),
        load_pickle("cal_data.pkl"),
        load_pickle("val_strat_data.pkl"),
    ]
    test = load_pickle("test_data.pkl")
    X_dev = np.concatenate([part["X_tab"] for part in development_parts], axis=0)
    y_dev = np.concatenate([part["y"] for part in development_parts], axis=0)
    X_test = test["X_tab"]
    y_test = test["y"]

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=SEED,
    )
    rows = []
    thresholds = []
    for fold_id, (train_idx, validation_idx) in enumerate(
        splitter.split(X_dev, y_dev), start=1
    ):
        model = make_model(SEED + fold_id)
        model.fit(X_dev[train_idx], y_dev[train_idx])
        validation_scores = model.predict_proba(X_dev[validation_idx])[:, 1]
        threshold = choose_threshold_by_fpr(y_dev[validation_idx], validation_scores)
        fold_metrics = threshold_metrics(
            y_dev[validation_idx], validation_scores, threshold
        )
        rows.append(
            {
                "fold": fold_id,
                "repeat": (fold_id - 1) // N_SPLITS + 1,
                "split": (fold_id - 1) % N_SPLITS + 1,
                "train_n": int(len(train_idx)),
                "train_positives": int(y_dev[train_idx].sum()),
                "validation_n": int(len(validation_idx)),
                "validation_positives": int(y_dev[validation_idx].sum()),
                "threshold": threshold,
                **fold_metrics,
            }
        )
        thresholds.append(threshold)
        print(
            f"[{fold_id:02d}/{N_SPLITS * N_REPEATS}] "
            f"positives={int(y_dev[validation_idx].sum())} "
            f"threshold={threshold:.6f} f1={fold_metrics['f1']:.4f}"
        )

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(OUTPUT_DIR / "cross_fitted_thresholds.csv", index=False)

    thresholds = np.asarray(thresholds, dtype=float)
    aggregate_threshold = float(np.median(thresholds))
    final_model = make_model(SEED)
    final_model.fit(X_dev, y_dev)
    test_scores = final_model.predict_proba(X_test)[:, 1]
    aggregate_test = threshold_metrics(y_test, test_scores, aggregate_threshold)

    sensitivity_rows = []
    for fold_id, threshold in enumerate(thresholds, start=1):
        sensitivity_rows.append(
            {
                "source_fold": fold_id,
                "threshold": float(threshold),
                **threshold_metrics(y_test, test_scores, float(threshold)),
            }
        )
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(
        OUTPUT_DIR / "locked_threshold_test_sensitivity.csv", index=False
    )

    summary = {
        "protocol": {
            "splits": N_SPLITS,
            "repeats": N_REPEATS,
            "threshold_count": int(len(thresholds)),
            "fpr_budget": FPR_BUDGET,
            "estimators": N_ESTIMATORS,
            "seed": SEED,
            "test_used_for_selection": False,
        },
        "development": {
            "n": int(len(y_dev)),
            "positives": int(y_dev.sum()),
            "negatives": int((y_dev == 0).sum()),
            "validation_positives_per_fold": sorted(
                fold_df["validation_positives"].unique().astype(int).tolist()
            ),
        },
        "threshold_distribution": {
            "median": aggregate_threshold,
            "q1": float(np.quantile(thresholds, 0.25)),
            "q3": float(np.quantile(thresholds, 0.75)),
            "min": float(np.min(thresholds)),
            "max": float(np.max(thresholds)),
        },
        "aggregate_locked_threshold_test": aggregate_test,
        "test_sensitivity_across_preselected_thresholds": {
            metric: {
                "median": float(sensitivity_df[metric].median()),
                "q1": float(sensitivity_df[metric].quantile(0.25)),
                "q3": float(sensitivity_df[metric].quantile(0.75)),
                "min": float(sensitivity_df[metric].min()),
                "max": float(sensitivity_df[metric].max()),
            }
            for metric in ["precision", "recall", "f1", "observed_fpr", "fp", "fn"]
        },
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(OUTPUT_DIR / "test_scores.pkl", "wb") as handle:
        pickle.dump(
            {
                "scores": test_scores,
                "labels": y_test,
                "aggregate_threshold": aggregate_threshold,
            },
            handle,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
