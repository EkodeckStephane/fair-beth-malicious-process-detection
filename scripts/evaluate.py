#!/usr/bin/env python3
"""Fair evaluation, ablations, confidence intervals, and figures."""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import BOOTSTRAP_N, FPR_BUDGET, OUTPUT_DIR, SEED
from utils import ensure_dir


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
    best_thr = 0.5
    best_f1 = -1.0
    for thr in candidates:
        f1 = f1_score(y_val, (val_scores >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return float(best_thr)


def metrics_at_threshold(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, scores),
        "ap": average_precision_score(y_true, scores),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "threshold": float(threshold),
        "tn": int(confusion_matrix(y_true, preds).ravel()[0]),
        "fp": int(confusion_matrix(y_true, preds).ravel()[1]),
        "fn": int(confusion_matrix(y_true, preds).ravel()[2]),
        "tp": int(confusion_matrix(y_true, preds).ravel()[3]),
    }


def bootstrap_ci(y_true, scores, threshold, n_boot=BOOTSTRAP_N, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    values = {k: [] for k in ["auc", "ap", "precision", "recall", "f1"]}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        s_b = scores[idx]
        m = metrics_at_threshold(y_b, s_b, threshold)
        for k in values:
            values[k].append(m[k])
    return {
        k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        for k, v in values.items() if v
    }


def fmt_ci(metric, ci):
    lo, hi = ci[metric]
    return f"[{lo:.4f}, {hi:.4f}]"


def plot_cm(y_true, scores, threshold, title, path):
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_roc(y_true, score_dict, path):
    plt.figure(figsize=(6, 5))
    for name, scores in score_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_pr(y_true, score_dict, path):
    plt.figure(figsize=(6, 5))
    for name, scores in score_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    with open(os.path.join(OUTPUT_DIR, "test_data.pkl"), "rb") as f:
        test_data = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "val_strat_data.pkl"), "rb") as f:
        val_data = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "fusion_results.pkl"), "rb") as f:
        fusion = pickle.load(f)

    y_test = test_data["y"]
    y_val = val_data["y"]

    candidates = {
        "IF-Platt": {
            "val": fusion["detector_scores"]["tab"]["val"],
            "test": fusion["detector_scores"]["tab"]["test"],
            "kind": "detector",
        },
        "GRUAE-Platt": {
            "val": fusion["detector_scores"]["seq"]["val"],
            "test": fusion["detector_scores"]["seq"]["test"],
            "kind": "detector",
        },
        "SimpleAvg": {
            "val": fusion["detector_scores"]["simple"]["val"],
            "test": fusion["detector_scores"]["simple"]["test"],
            "kind": "fusion",
        },
    }

    for name, model_out in fusion["rf_models"].items():
        candidates[name] = {
            "val": model_out["val"],
            "test": model_out["test"],
            "kind": "supervised",
        }

    rows = []
    for name, item in candidates.items():
        thr_fpr = choose_threshold_by_fpr(y_val, item["val"], FPR_BUDGET)
        thr_f1 = choose_threshold_by_f1(y_val, item["val"])
        for policy, threshold in [("val_fpr_budget", thr_fpr), ("val_max_f1", thr_f1)]:
            m = metrics_at_threshold(y_test, item["test"], threshold)
            ci = bootstrap_ci(y_test, item["test"], threshold)
            rows.append({
                "Model": name,
                "Kind": item["kind"],
                "ThresholdPolicy": policy,
                "Threshold": f"{m['threshold']:.6f}",
                "AUC": f"{m['auc']:.4f}",
                "AUC_95CI": fmt_ci("auc", ci),
                "AP": f"{m['ap']:.4f}",
                "AP_95CI": fmt_ci("ap", ci),
                "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "F1": f"{m['f1']:.4f}",
                "F1_95CI": fmt_ci("f1", ci),
                "TN": m["tn"],
                "FP": m["fp"],
                "FN": m["fn"],
                "TP": m["tp"],
            })
            print(
                f"{name:12s} {policy:14s} "
                f"AUC={m['auc']:.4f} AP={m['ap']:.4f} "
                f"Pr={m['precision']:.4f} Re={m['recall']:.4f} F1={m['f1']:.4f} "
                f"thr={m['threshold']:.6f} cm=[[{m['tn']},{m['fp']}],[{m['fn']},{m['tp']}]]"
            )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "results_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Results table saved: {csv_path}")

    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    ensure_dir(fig_dir)

    score_dict = {name: item["test"] for name, item in candidates.items()}
    plot_roc(y_test, score_dict, os.path.join(fig_dir, "roc_curves.png"))
    plot_pr(y_test, score_dict, os.path.join(fig_dir, "pr_curves.png"))

    for name in ["TabRF", "MetaRF", "ScoreRF"]:
        if name in candidates:
            threshold = choose_threshold_by_fpr(y_val, candidates[name]["val"], FPR_BUDGET)
            plot_cm(
                y_test,
                candidates[name]["test"],
                threshold,
                f"{name} - validation FPR budget",
                os.path.join(fig_dir, f"cm_{name.lower()}_budget.png"),
            )


if __name__ == "__main__":
    main()
