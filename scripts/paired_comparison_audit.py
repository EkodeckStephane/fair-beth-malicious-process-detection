#!/usr/bin/env python3
"""Paired decision-comparison audit for the main FAIR-BETH tabular claims."""
import json
import math
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


SEED = 42
FPR_BUDGET = 0.05
BOOTSTRAP_N = 10000


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def choose_threshold_by_fpr(y_val, scores, budget=FPR_BUDGET):
    fpr, _, thresholds = roc_curve(y_val, scores)
    finite = np.isfinite(thresholds)
    thresholds = thresholds[finite]
    fpr = fpr[finite]
    valid = np.where(fpr <= budget)[0]
    return float(thresholds[valid[-1]]) if len(valid) else 0.5


def exact_mcnemar_p(b, c):
    n = int(b + c)
    if n == 0:
        return 1.0
    k = int(min(b, c))
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * cdf))


def f1_at(y, scores, threshold):
    return float(f1_score(y, scores >= threshold, zero_division=0))


def paired_bootstrap(y, a_scores, b_scores, a_thr, b_thr):
    rng = np.random.default_rng(SEED)
    y = np.asarray(y)
    a_scores = np.asarray(a_scores)
    b_scores = np.asarray(b_scores)
    n = len(y)
    values = {"delta_f1": [], "delta_auc": [], "delta_ap": []}
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        sa = a_scores[idx]
        sb = b_scores[idx]
        values["delta_f1"].append(f1_at(yb, sa, a_thr) - f1_at(yb, sb, b_thr))
        values["delta_auc"].append(roc_auc_score(yb, sa) - roc_auc_score(yb, sb))
        values["delta_ap"].append(average_precision_score(yb, sa) - average_precision_score(yb, sb))
    out = {}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        out[key] = {
            "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
            "p_sign_two_sided": float(min(1.0, 2.0 * min(np.mean(arr <= 0), np.mean(arr >= 0)))),
        }
    return out


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

    rf500 = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    rf500.fit(X_fit, y_fit)

    score_sets = {
        "TabRF": {
            "val": fusion["rf_models"]["TabRF"]["val"],
            "test": fusion["rf_models"]["TabRF"]["test"],
        },
        "MetaRF": {
            "val": fusion["rf_models"]["MetaRF"]["val"],
            "test": fusion["rf_models"]["MetaRF"]["test"],
        },
        "RF-500": {
            "val": rf500.predict_proba(X_val)[:, 1],
            "test": rf500.predict_proba(X_test)[:, 1],
        },
    }

    for item in score_sets.values():
        item["threshold"] = choose_threshold_by_fpr(y_val, item["val"])
        item["pred"] = item["test"] >= item["threshold"]

    rows = []
    comparisons = [("TabRF", "MetaRF"), ("TabRF", "RF-500"), ("RF-500", "MetaRF")]
    for a, b in comparisons:
        a_item = score_sets[a]
        b_item = score_sets[b]
        boot = paired_bootstrap(
            y_test,
            a_item["test"],
            b_item["test"],
            a_item["threshold"],
            b_item["threshold"],
        )
        a_correct = a_item["pred"] == y_test
        b_correct = b_item["pred"] == y_test
        b_count = int(np.sum(a_correct & ~b_correct))
        c_count = int(np.sum(~a_correct & b_correct))
        rows.append({
            "comparison": f"{a} - {b}",
            "delta_f1": f1_at(y_test, a_item["test"], a_item["threshold"])
            - f1_at(y_test, b_item["test"], b_item["threshold"]),
            "delta_f1_ci95_low": boot["delta_f1"]["ci95"][0],
            "delta_f1_ci95_high": boot["delta_f1"]["ci95"][1],
            "delta_f1_bootstrap_p_two_sided": boot["delta_f1"]["p_sign_two_sided"],
            "delta_auc": roc_auc_score(y_test, a_item["test"]) - roc_auc_score(y_test, b_item["test"]),
            "delta_auc_ci95_low": boot["delta_auc"]["ci95"][0],
            "delta_auc_ci95_high": boot["delta_auc"]["ci95"][1],
            "delta_ap": average_precision_score(y_test, a_item["test"])
            - average_precision_score(y_test, b_item["test"]),
            "delta_ap_ci95_low": boot["delta_ap"]["ci95"][0],
            "delta_ap_ci95_high": boot["delta_ap"]["ci95"][1],
            "mcnemar_b_a_correct_b_wrong": b_count,
            "mcnemar_c_a_wrong_b_correct": c_count,
            "mcnemar_exact_p_two_sided": exact_mcnemar_p(b_count, c_count),
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "paired_comparison_audit.csv")
    json_path = os.path.join(out_dir, "paired_comparison_audit.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": {
                    "bootstrap_resamples": BOOTSTRAP_N,
                    "fpr_budget": FPR_BUDGET,
                    "seed": SEED,
                    "threshold_policy": "validation-FPR budget",
                },
                "thresholds": {
                    key: float(value["threshold"])
                    for key, value in score_sets.items()
                },
                "rows": df.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )
    print(f"[+] Wrote {csv_path}")
    print(f"[+] Wrote {json_path}")


if __name__ == "__main__":
    main()
