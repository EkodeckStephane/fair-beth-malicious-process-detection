#!/usr/bin/env python3
"""Additional tabular baselines and calibration audit for FAIR-BETH.

The script reuses the frozen FAIR-BETH data artifacts. It does not refit
preprocessing, does not read final-test labels before training, and applies the
same validation-derived threshold policies used in the main paper.
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None


SEED = 42
FPR_BUDGET = 0.05
BOOTSTRAP_N = 1000
N_BINS = 10


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def choose_threshold_by_fpr(y_val, val_scores, fpr_budget=FPR_BUDGET):
    fpr, _, thresholds = roc_curve(y_val, val_scores)
    valid = np.where(fpr <= fpr_budget)[0]
    # sklearn returns thresholds in decreasing order, so valid[-1] is the
    # most permissive (lowest) threshold still satisfying the FPR budget.
    return float(thresholds[valid[-1]]) if len(valid) else 0.5


def choose_threshold_by_f1(y_val, val_scores):
    candidates = np.unique(val_scores)
    if len(candidates) > 1000:
        candidates = np.quantile(val_scores, np.linspace(0, 1, 1000))
    best_thr, best_f1 = 0.5, -1.0
    for thr in candidates:
        value = f1_score(y_val, val_scores >= thr, zero_division=0)
        if value > best_f1:
            best_thr, best_f1 = float(thr), float(value)
    return best_thr


def metrics_at_threshold(y_true, scores, threshold):
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "auc": roc_auc_score(y_true, scores),
        "ap": average_precision_score(y_true, scores),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_ci(y_true, scores, threshold):
    rng = np.random.default_rng(SEED)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    values = {k: [] for k in ["auc", "ap", "precision", "recall", "f1"]}
    n = len(y_true)
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        m = metrics_at_threshold(y_b, scores[idx], threshold)
        for key in values:
            values[key].append(m[key])
    return {
        key: f"[{np.percentile(vals, 2.5):.4f}, {np.percentile(vals, 97.5):.4f}]"
        for key, vals in values.items() if vals
    }


def expected_calibration_error(y_true, scores, n_bins=N_BINS):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        count = int(mask.sum())
        if count == 0:
            rows.append((lo, hi, count, np.nan, np.nan))
            continue
        conf = float(scores[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (count / len(y_true)) * abs(acc - conf)
        rows.append((lo, hi, count, conf, acc))
    return float(ece), rows


def build_models(pos_weight):
    models = {
        "LogReg": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED),
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            l2_regularization=0.1,
            random_state=SEED,
        ),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(48, 24),
                alpha=1e-3,
                early_stopping=True,
                max_iter=300,
                random_state=SEED,
            ),
        ),
        "RF-500": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=pos_weight,
            random_state=SEED,
            n_jobs=-1,
        )
    return models


def save_reliability_plot(calibration_rows, out_path):
    plot_models = [row for row in calibration_rows if row["Split"] == "test"]
    n = len(plot_models)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, item in zip(axes.ravel(), plot_models):
        ax.axis("on")
        curve = item["Curve"]
        xs = [c for _, _, count, c, _ in curve if count > 0]
        ys = [a for _, _, count, _, a in curve if count > 0]
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(xs, ys, marker="o")
        ax.set_title(f"{item['Model']} (ECE={item['ECE']:.3f})")
        ax.set_xlabel("Mean score")
        ax.set_ylabel("Empirical positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    output_dir = os.environ.get("FAIR_BETH_PIPELINE_OUTPUT", "./pipeline_output")
    out_dir = os.environ.get(
        "FAIR_BETH_AUDIT_OUTPUT",
        os.path.join(output_dir, "additional_audits"),
    )
    os.makedirs(out_dir, exist_ok=True)

    train = load_pickle(os.path.join(output_dir, "train_data.pkl"))
    cal = load_pickle(os.path.join(output_dir, "cal_data.pkl"))
    val = load_pickle(os.path.join(output_dir, "val_strat_data.pkl"))
    test = load_pickle(os.path.join(output_dir, "test_data.pkl"))
    fusion = load_pickle(os.path.join(output_dir, "fusion_results.pkl"))

    X_fit = np.vstack([train["X_tab"], cal["X_tab"]])
    y_fit = np.concatenate([train["y"], cal["y"]])
    X_val, y_val = val["X_tab"], val["y"]
    X_test, y_test = test["X_tab"], test["y"]

    n_pos = int(y_fit.sum())
    n_neg = int(len(y_fit) - n_pos)
    pos_weight = n_neg / max(n_pos, 1)

    score_sets = {}
    rows = []
    for name, model in build_models(pos_weight).items():
        model.fit(X_fit, y_fit)
        val_scores = model.predict_proba(X_val)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]
        score_sets[name] = {"val": val_scores, "test": test_scores}

    for name, model_out in fusion["rf_models"].items():
        score_sets[name] = {"val": model_out["val"], "test": model_out["test"]}

    for name, scores in score_sets.items():
        thresholds = {
            "val_fpr_budget": choose_threshold_by_fpr(y_val, scores["val"]),
            "val_max_f1": choose_threshold_by_f1(y_val, scores["val"]),
        }
        for policy, threshold in thresholds.items():
            m = metrics_at_threshold(y_test, scores["test"], threshold)
            ci = bootstrap_ci(y_test, scores["test"], threshold)
            rows.append({
                "Model": name,
                "ThresholdPolicy": policy,
                "Threshold": f"{threshold:.6f}",
                "AUC": f"{m['auc']:.4f}",
                "AUC_95CI": ci["auc"],
                "AP": f"{m['ap']:.4f}",
                "AP_95CI": ci["ap"],
                "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "F1": f"{m['f1']:.4f}",
                "F1_95CI": ci["f1"],
                "TN": m["tn"],
                "FP": m["fp"],
                "FN": m["fn"],
                "TP": m["tp"],
            })

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "tabular_sota_comparison.csv"), index=False)

    calibration_rows = []
    for name, scores in score_sets.items():
        for split, y_true, split_scores in [
            ("validation", y_val, scores["val"]),
            ("test", y_test, scores["test"]),
        ]:
            ece, curve = expected_calibration_error(y_true, split_scores)
            calibration_rows.append({
                "Model": name,
                "Split": split,
                "ECE": ece,
                "Brier": brier_score_loss(y_true, split_scores),
                "Curve": curve,
            })
    pd.DataFrame([
        {k: v for k, v in row.items() if k != "Curve"} for row in calibration_rows
    ]).to_csv(os.path.join(out_dir, "calibration_audit.csv"), index=False)
    save_reliability_plot(calibration_rows, os.path.join(out_dir, "reliability_diagrams.png"))

    print(f"[+] Wrote {os.path.join(out_dir, 'tabular_sota_comparison.csv')}")
    print(f"[+] Wrote {os.path.join(out_dir, 'calibration_audit.csv')}")
    print(f"[+] Wrote {os.path.join(out_dir, 'reliability_diagrams.png')}")


if __name__ == "__main__":
    main()
