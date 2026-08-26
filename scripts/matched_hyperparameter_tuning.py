#!/usr/bin/env python3
"""Matched development-only hyperparameter tuning for the IJIS major revision.

The scientific purpose is to remove configuration-budget confounding from the
baseline comparison.  Every candidate family receives the same budget of 12
configurations and the same 4-fold stratified development-only CV budget
(48 fits per family).  Average precision is the selection metric so that
hyperparameter selection is separated from the validation-derived operating
threshold.  The official test labels are loaded only after model configurations
and validation thresholds have been written to a lock file.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

SEED = 42
N_CONFIGS = 12
N_FOLDS = 4
FPR_BUDGET = 0.05
BOOTSTRAP_N = 10_000

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def select_budget(grid: dict[str, list[Any]], n: int, seed: int) -> list[dict[str, Any]]:
    candidates = list(ParameterGrid(grid))
    if len(candidates) < n:
        raise ValueError(f"Grid has only {len(candidates)} candidates; matched budget requires {n}")
    if len(candidates) == n:
        return candidates
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(candidates), size=n, replace=False))
    return [candidates[int(i)] for i in idx]


def random_forest(params: dict[str, Any], y: np.ndarray) -> BaseEstimator:
    return RandomForestClassifier(
        **params,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def extra_trees(params: dict[str, Any], y: np.ndarray) -> BaseEstimator:
    return ExtraTreesClassifier(
        **params,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def grad_boost(params: dict[str, Any], y: np.ndarray) -> BaseEstimator:
    return GradientBoostingClassifier(**params, random_state=SEED)


def hist_grad_boost(params: dict[str, Any], y: np.ndarray) -> BaseEstimator:
    return HistGradientBoostingClassifier(**params, random_state=SEED)


def logistic(params: dict[str, Any], y: np.ndarray) -> BaseEstimator:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    **params,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def mlp(params: dict[str, Any], y: np.ndarray) -> BaseEstimator:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    **params,
                    max_iter=400,
                    early_stopping=False,
                    random_state=SEED,
                ),
            ),
        ]
    )


def xgboost(params: dict[str, Any], y: np.ndarray) -> BaseEstimator:
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed")
    n_pos = max(int(np.sum(y == 1)), 1)
    n_neg = int(np.sum(y == 0))
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=n_neg / n_pos,
        random_state=SEED,
        n_jobs=-1,
    )


def fit_model(model: BaseEstimator, family: str, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    if family == "MLP":
        weights = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(X, y, model__sample_weight=weights)
    else:
        model.fit(X, y)
    return model


def predict_scores(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(model.decision_function(X), dtype=float)


def choose_threshold_by_fpr(y_val: np.ndarray, scores: np.ndarray) -> float:
    fpr, _, thresholds = roc_curve(y_val, scores)
    valid = np.where(fpr <= FPR_BUDGET)[0]
    return float(thresholds[valid[-1]]) if len(valid) else float(np.max(scores) + 1e-12)


def choose_threshold_by_f1(y_val: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(scores)
    best = (float(candidates[0]), -1.0)
    for threshold in candidates:
        value = f1_score(y_val, scores >= threshold, zero_division=0)
        if value > best[1]:
            best = (float(threshold), float(value))
    return best[0]


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
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


def bootstrap_f1_ci(y: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    n = len(y)
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        values.append(float(f1_score(yy, scores[idx] >= threshold, zero_division=0)))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main() -> None:
    pipeline_dir = Path(os.environ.get("FAIR_BETH_PIPELINE_OUTPUT", "./pipeline_output")).resolve()
    output_dir = Path(os.environ.get("FAIR_BETH_REVISION_OUTPUT", "./results/revision_audits")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Development artifacts only at tuning time.
    train = load_pickle(pipeline_dir / "train_data.pkl")
    cal = load_pickle(pipeline_dir / "cal_data.pkl")
    val = load_pickle(pipeline_dir / "val_strat_data.pkl")
    fusion = load_pickle(pipeline_dir / "fusion_results.pkl")

    y_train = np.asarray(train["y"], dtype=int)
    y_cal = np.asarray(cal["y"], dtype=int)
    y_val = np.asarray(val["y"], dtype=int)

    score_train = np.column_stack(
        [
            fusion["detector_scores"]["tab"]["train"],
            fusion["detector_scores"]["seq"]["train"],
            fusion["detector_scores"]["simple"]["train"],
        ]
    )
    score_cal = np.column_stack(
        [
            fusion["detector_scores"]["tab"]["cal"],
            fusion["detector_scores"]["seq"]["cal"],
            fusion["detector_scores"]["simple"]["cal"],
        ]
    )
    score_val = np.column_stack(
        [
            fusion["detector_scores"]["tab"]["val"],
            fusion["detector_scores"]["seq"]["val"],
            fusion["detector_scores"]["simple"]["val"],
        ]
    )

    X_train_tab = np.asarray(train["X_tab"], dtype=np.float32)
    X_cal_tab = np.asarray(cal["X_tab"], dtype=np.float32)
    X_val_tab = np.asarray(val["X_tab"], dtype=np.float32)

    representations = {
        "TabRF": (X_train_tab, X_cal_tab, X_val_tab, "RF"),
        "ScoreRF": (score_train, score_cal, score_val, "RF"),
        "MetaRF": (
            np.hstack([X_train_tab, score_train]),
            np.hstack([X_cal_tab, score_cal]),
            np.hstack([X_val_tab, score_val]),
            "RF",
        ),
        "ExtraTrees": (X_train_tab, X_cal_tab, X_val_tab, "ExtraTrees"),
        "GradientBoosting": (X_train_tab, X_cal_tab, X_val_tab, "GradientBoosting"),
        "HistGradientBoosting": (X_train_tab, X_cal_tab, X_val_tab, "HistGradientBoosting"),
        "MLP": (X_train_tab, X_cal_tab, X_val_tab, "MLP"),
        "LogReg": (X_train_tab, X_cal_tab, X_val_tab, "LogReg"),
    }
    if XGBClassifier is not None:
        representations["XGBoost"] = (X_train_tab, X_cal_tab, X_val_tab, "XGBoost")

    rf_grid = {
        "n_estimators": [200, 350, 500],
        "max_depth": [None, 8, 16],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.5, None],
    }
    grids = {
        "RF": rf_grid,
        "ExtraTrees": rf_grid,
        "GradientBoosting": {
            "n_estimators": [100, 200, 350],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [1, 2, 3],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.8, 1.0],
        },
        "HistGradientBoosting": {
            "learning_rate": [0.03, 0.05, 0.1],
            "max_iter": [100, 200, 350],
            "max_leaf_nodes": [7, 15, 31],
            "min_samples_leaf": [5, 10, 20],
            "l2_regularization": [0.0, 0.1, 1.0],
        },
        "MLP": {
            "hidden_layer_sizes": [(32,), (64,), (48, 24), (64, 32), (96, 48)],
            "alpha": [1e-4, 1e-3, 1e-2],
            "learning_rate_init": [3e-4, 1e-3, 3e-3],
            "activation": ["relu", "tanh"],
        },
        "LogReg": {
            "C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            "penalty": ["l1", "l2"],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 350],
            "max_depth": [2, 3, 5],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_lambda": [1.0, 5.0],
        },
    }
    builders: dict[str, Callable[[dict[str, Any], np.ndarray], BaseEstimator]] = {
        "RF": random_forest,
        "ExtraTrees": extra_trees,
        "GradientBoosting": grad_boost,
        "HistGradientBoosting": hist_grad_boost,
        "MLP": mlp,
        "LogReg": logistic,
        "XGBoost": xgboost,
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_rows: list[dict[str, Any]] = []
    best: dict[str, dict[str, Any]] = {}
    fitted: dict[str, BaseEstimator] = {}
    thresholds: dict[str, dict[str, float]] = {}

    for model_name, (X_train, X_cal, X_val, family) in representations.items():
        candidates = select_budget(grids[family], N_CONFIGS, seed=SEED + sum(map(ord, model_name)))
        candidate_results = []
        for candidate_id, params in enumerate(candidates, start=1):
            fold_scores: list[float] = []
            for fold_id, (tr, va) in enumerate(cv.split(X_train, y_train), start=1):
                model = builders[family](params, y_train[tr])
                model = fit_model(model, family, X_train[tr], y_train[tr])
                scores = predict_scores(model, X_train[va])
                ap = float(average_precision_score(y_train[va], scores))
                fold_scores.append(ap)
                cv_rows.append(
                    {
                        "Model": model_name,
                        "Family": family,
                        "Candidate": candidate_id,
                        "Fold": fold_id,
                        "AP": ap,
                        "Params": stable_json(params),
                    }
                )
            candidate_results.append(
                {
                    "candidate": candidate_id,
                    "params": params,
                    "mean_ap": float(np.mean(fold_scores)),
                    "std_ap": float(np.std(fold_scores, ddof=1)),
                }
            )

        selected = sorted(
            candidate_results,
            key=lambda row: (-row["mean_ap"], row["std_ap"], stable_json(row["params"])),
        )[0]
        best[model_name] = {
            "family": family,
            "candidate": selected["candidate"],
            "params": selected["params"],
            "cv_mean_ap": selected["mean_ap"],
            "cv_std_ap": selected["std_ap"],
        }

        X_fit = np.vstack([X_train, X_cal])
        y_fit = np.concatenate([y_train, y_cal])
        model = builders[family](selected["params"], y_fit)
        model = fit_model(model, family, X_fit, y_fit)
        val_scores = predict_scores(model, X_val)
        thresholds[model_name] = {
            "val_fpr_budget": choose_threshold_by_fpr(y_val, val_scores),
            "val_max_f1": choose_threshold_by_f1(y_val, val_scores),
        }
        fitted[model_name] = model
        print(
            f"[lock] {model_name}: AP={selected['mean_ap']:.4f}±{selected['std_ap']:.4f} "
            f"thrFPR={thresholds[model_name]['val_fpr_budget']:.6g}"
        )

    protocol = {
        "seed": SEED,
        "selection_metric": "average_precision",
        "tuning_subset": "train_data only",
        "refit_subset": "train_data + cal_data",
        "threshold_subset": "val_strat_data only",
        "test_access": "after configuration and threshold lock",
        "n_candidate_configurations_per_model": N_CONFIGS,
        "cv_folds": N_FOLDS,
        "fits_per_model_family": N_CONFIGS * N_FOLDS,
        "fpr_budget": FPR_BUDGET,
        "best": best,
        "thresholds": thresholds,
    }
    protocol_text = json.dumps(protocol, indent=2, sort_keys=True, default=str)
    protocol["lock_sha256"] = hashlib.sha256(protocol_text.encode("utf-8")).hexdigest()
    lock_path = output_dir / "matched_tuning_lock.json"
    with lock_path.open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True, default=str)

    pd.DataFrame(cv_rows).to_csv(output_dir / "matched_tuning_cv.csv", index=False)
    pd.DataFrame(
        [
            {
                "Model": name,
                "Family": item["family"],
                "Candidate": item["candidate"],
                "CV_mean_AP": item["cv_mean_ap"],
                "CV_std_AP": item["cv_std_ap"],
                "Params": stable_json(item["params"]),
                "Threshold_valFPR": thresholds[name]["val_fpr_budget"],
                "Threshold_valMaxF1": thresholds[name]["val_max_f1"],
            }
            for name, item in best.items()
        ]
    ).to_csv(output_dir / "matched_tuning_summary.csv", index=False)

    # Only now consume the isolated official-test labels and test representations.
    test = load_pickle(pipeline_dir / "test_data.pkl")
    y_test = np.asarray(test["y"], dtype=int)
    X_test_tab = np.asarray(test["X_tab"], dtype=np.float32)
    score_test = np.column_stack(
        [
            fusion["detector_scores"]["tab"]["test"],
            fusion["detector_scores"]["seq"]["test"],
            fusion["detector_scores"]["simple"]["test"],
        ]
    )
    test_representations = {
        "TabRF": X_test_tab,
        "ScoreRF": score_test,
        "MetaRF": np.hstack([X_test_tab, score_test]),
        "ExtraTrees": X_test_tab,
        "GradientBoosting": X_test_tab,
        "HistGradientBoosting": X_test_tab,
        "MLP": X_test_tab,
        "LogReg": X_test_tab,
    }
    if "XGBoost" in fitted:
        test_representations["XGBoost"] = X_test_tab

    rows: list[dict[str, Any]] = []
    score_frame: dict[str, Any] = {"y_true": y_test}
    for model_name, model in fitted.items():
        test_scores = predict_scores(model, test_representations[model_name])
        score_frame[model_name] = test_scores
        for policy_name, threshold in thresholds[model_name].items():
            m = metrics(y_test, test_scores, threshold)
            lo, hi = bootstrap_f1_ci(y_test, test_scores, threshold)
            rows.append(
                {
                    "Model": model_name,
                    "ThresholdPolicy": policy_name,
                    "Threshold": threshold,
                    **m,
                    "F1_CI_low": lo,
                    "F1_CI_high": hi,
                }
            )

    pd.DataFrame(rows).to_csv(output_dir / "matched_tuning_test_results.csv", index=False)
    pd.DataFrame(score_frame).to_csv(output_dir / "matched_tuning_test_scores.csv", index=False)
    print("\n=== MATCHED-TUNING TEST RESULTS (validation-FPR policy) ===")
    print(
        pd.DataFrame(rows)
        .query("ThresholdPolicy == 'val_fpr_budget'")
        [["Model", "AUC", "AP", "Precision", "Recall", "F1", "FPR", "FP", "FN"]]
        .sort_values("F1", ascending=False)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
