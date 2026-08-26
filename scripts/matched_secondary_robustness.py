#!/usr/bin/env python3
"""Matched-configuration secondary robustness analyses for FAIR-X / FAIR-BETH.

Keep the TabRF hyperparameters frozen at the configuration selected by the
matched development-only search and rerun the secondary analyses that most
directly condition the manuscript's interpretation: repeated threshold
estimation, host-disjoint robustness, chronological robustness, and early-prefix
evaluation. Official BETH test labels are consumed only after each model and
threshold rule is fixed.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

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
from sklearn.model_selection import GroupShuffleSplit, RepeatedStratifiedKFold, train_test_split

SEED = 42
FPR_BUDGET = 0.05
N_SPLITS = 5
N_REPEATS = 10
PREFIXES: list[int | str] = [10, 25, 50, 100, 250, "full"]
GROUP_COLS = ["hostName", "processId"]
BASE_KEYS = [
    "n_events", "n_unique_eventIds", "entropy_eventIds",
    "mean_argsNum", "std_argsNum", "max_argsNum",
    "mean_returnValue", "std_returnValue",
    "duration", "event_rate",
    "n_unique_args", "mean_args_len", "n_unique_parents",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--lock", type=Path, default=Path("results/revision_audits/matched_tuning_lock.json"))
    p.add_argument("--canonical-main", type=Path, default=Path("results/canonical/main_results.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results/revision_audits/matched_secondary_robustness"))
    p.add_argument("--canonical-dir", type=Path, default=Path("results/canonical"))
    return p.parse_args()


def extra_development_files(data_dir: Path) -> list[str]:
    return sorted(
        p.name for p in data_dir.glob("labelled_2021may*.csv")
        if not p.name.endswith("-dns.csv")
    )


def load_csvs(data_dir: Path, names: list[str]) -> pd.DataFrame:
    frames = []
    for name in names:
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, low_memory=False)
        df["source_file"] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def aggregate_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for (host, pid), g in df.groupby(GROUP_COLS):
        g = g.sort_values("timestamp")
        if g.empty:
            continue
        records.append(
            {
                "hostName": str(host),
                "processId": int(pid),
                "source_file": str(g["source_file"].iloc[0]),
                "first_timestamp": float(g["timestamp"].iloc[0]),
                "y": int(g["evil"].max()),
                "eventId": g["eventId"].astype(int).to_numpy(),
                "argsNum": g["argsNum"].astype(float).to_numpy(),
                "returnValue": g["returnValue"].astype(float).to_numpy(),
                "timestamp": g["timestamp"].astype(float).to_numpy(),
                "args": g["args"].astype(str).to_numpy(),
                "parentProcessId": g["parentProcessId"].astype(str).to_numpy(),
            }
        )
    return records


def build_vocab(records: list[dict[str, Any]]) -> list[int]:
    values: set[int] = set()
    for r in records:
        values.update(map(int, r["eventId"]))
    return sorted(values)


def entropy_from_counts(counter: Counter[int], n: int) -> float:
    if n <= 0:
        return 0.0
    probs = np.asarray(list(counter.values()), dtype=float) / n
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


def records_to_features(
    records: list[dict[str, Any]], vocab_event_ids: list[int], prefix: int | str = "full"
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    eid_to_col = {int(e): i for i, e in enumerate(vocab_event_ids)}
    X_base = np.zeros((len(records), len(BASE_KEYS)), dtype=np.float32)
    X_hist = np.zeros((len(records), len(vocab_event_ids)), dtype=np.float32)
    y = np.zeros(len(records), dtype=np.int64)
    meta: list[dict[str, Any]] = []

    for i, r in enumerate(records):
        n_full = len(r["eventId"])
        k = n_full if prefix == "full" else min(int(prefix), n_full)
        eids = r["eventId"][:k]
        args_num = r["argsNum"][:k]
        ret = r["returnValue"][:k]
        ts = r["timestamp"][:k]
        args = r["args"][:k]
        parents = r["parentProcessId"][:k]
        counter: Counter[int] = Counter(map(int, eids.tolist()))
        duration = float(ts[-1] - ts[0]) if k > 1 else 0.0
        event_rate = float(k / (duration + 1e-6))
        values = {
            "n_events": k,
            "n_unique_eventIds": len(counter),
            "entropy_eventIds": entropy_from_counts(counter, k),
            "mean_argsNum": float(args_num.mean()),
            "std_argsNum": float(args_num.std(ddof=0)) if k > 1 else 0.0,
            "max_argsNum": int(args_num.max()),
            "mean_returnValue": float(ret.mean()),
            "std_returnValue": float(ret.std(ddof=0)) if k > 1 else 0.0,
            "duration": duration,
            "event_rate": event_rate,
            "n_unique_args": len(set(args.tolist())),
            "mean_args_len": float(np.mean([len(a) for a in args])),
            "n_unique_parents": len(set(parents.tolist())),
        }
        for j, key in enumerate(BASE_KEYS):
            X_base[i, j] = values[key]
        for eid, cnt in counter.items():
            col = eid_to_col.get(eid)
            if col is not None:
                X_hist[i, col] = cnt / max(k, 1)
        y[i] = r["y"]
        meta.append(
            {
                "hostName": r["hostName"],
                "processId": r["processId"],
                "source_file": r["source_file"],
                "first_timestamp": r["first_timestamp"],
            }
        )
    return np.concatenate([X_base, X_hist], axis=1), y, pd.DataFrame(meta)


def choose_threshold_by_fpr(y: np.ndarray, scores: np.ndarray) -> float:
    fpr, _, thresholds = roc_curve(y, scores)
    finite = np.isfinite(thresholds)
    valid = np.where((fpr <= FPR_BUDGET) & finite)[0]
    if not len(valid):
        return float(np.nextafter(np.max(scores), np.inf))
    return float(thresholds[valid[-1]])


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result = {
        "AUC": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else math.nan,
        "AP": float(average_precision_score(y, scores)) if len(np.unique(y)) == 2 else math.nan,
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    result["FPR"] = float(fp / max(tn + fp, 1))
    return result


def make_model(params: dict[str, Any], seed: int = SEED) -> RandomForestClassifier:
    return RandomForestClassifier(
        **params,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def primary_split_indices(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_idx, temp_idx = train_test_split(
        np.arange(len(y)), test_size=0.35, stratify=y, random_state=42
    )
    cal_idx, val_idx = train_test_split(
        temp_idx, test_size=0.429, stratify=y[temp_idx], random_state=43
    )
    return train_idx, cal_idx, val_idx


def assert_primary_parity(
    dev_records: list[dict[str, Any]], test_records: list[dict[str, Any]], params: dict[str, Any],
    lock: dict[str, Any], canonical_main: Path,
) -> dict[str, Any]:
    """Rebuild the matched TabRF full-trace row from raw CSVs and require parity."""
    vocab = build_vocab(dev_records)
    X_dev, y_dev, _ = records_to_features(dev_records, vocab, prefix="full")
    X_test, y_test, _ = records_to_features(test_records, vocab, prefix="full")
    tr, cal, val = primary_split_indices(y_dev)
    fit_idx = np.concatenate([tr, cal])
    model = make_model(params, SEED)
    model.fit(X_dev[fit_idx], y_dev[fit_idx])
    val_scores = model.predict_proba(X_dev[val])[:, 1]
    threshold = choose_threshold_by_fpr(y_dev[val], val_scores)
    test_scores = model.predict_proba(X_test)[:, 1]
    m = metrics(y_test, test_scores, threshold)

    locked_thr = float(lock["thresholds"]["TabRF"]["val_fpr_budget"])
    if not np.isclose(threshold, locked_thr, rtol=0, atol=1e-12):
        raise RuntimeError(f"Raw rebuild threshold mismatch: {threshold} != {locked_thr}")

    canonical = pd.read_csv(canonical_main)
    row = canonical[(canonical["Model"] == "TabRF") & (canonical["ThresholdPolicy"] == "val_fpr_budget")].iloc[0]
    for col in ["AUC", "AP", "Precision", "Recall", "F1", "FPR"]:
        if not np.isclose(float(m[col]), float(row[col]), rtol=0, atol=1e-12):
            raise RuntimeError(f"Raw rebuild {col} mismatch: {m[col]} != {row[col]}")
    for col in ["TN", "FP", "FN", "TP"]:
        if int(m[col]) != int(row[col]):
            raise RuntimeError(f"Raw rebuild {col} mismatch: {m[col]} != {row[col]}")
    return {"Threshold": threshold, **m, "vocab_size": len(vocab)}


def repeated_threshold_analysis(
    dev_records: list[dict[str, Any]], test_records: list[dict[str, Any]], params: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    vocab = build_vocab(dev_records)
    X_dev, y_dev, _ = records_to_features(dev_records, vocab, prefix="full")
    X_test, y_test, _ = records_to_features(test_records, vocab, prefix="full")
    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    rows: list[dict[str, Any]] = []
    thresholds: list[float] = []
    for fold_id, (tr, va) in enumerate(splitter.split(X_dev, y_dev), start=1):
        model = make_model(params, SEED + fold_id)
        model.fit(X_dev[tr], y_dev[tr])
        val_scores = model.predict_proba(X_dev[va])[:, 1]
        threshold = choose_threshold_by_fpr(y_dev[va], val_scores)
        rows.append(
            {
                "fold": fold_id,
                "repeat": (fold_id - 1) // N_SPLITS + 1,
                "split": (fold_id - 1) % N_SPLITS + 1,
                "train_n": len(tr),
                "train_positives": int(y_dev[tr].sum()),
                "validation_n": len(va),
                "validation_positives": int(y_dev[va].sum()),
                "threshold": threshold,
                **metrics(y_dev[va], val_scores, threshold),
            }
        )
        thresholds.append(threshold)

    threshold_arr = np.asarray(thresholds)
    aggregate_threshold = float(np.median(threshold_arr))
    final_model = make_model(params, SEED)
    final_model.fit(X_dev, y_dev)
    test_scores = final_model.predict_proba(X_test)[:, 1]
    test_metrics = metrics(y_test, test_scores, aggregate_threshold)
    summary = {
        "development_n": int(len(y_dev)),
        "development_positives": int(y_dev.sum()),
        "folds": N_SPLITS * N_REPEATS,
        "validation_positives_per_fold": sorted(set(int(r["validation_positives"]) for r in rows)),
        "threshold_median": aggregate_threshold,
        "threshold_q1": float(np.quantile(threshold_arr, 0.25)),
        "threshold_q3": float(np.quantile(threshold_arr, 0.75)),
        "threshold_min": float(np.min(threshold_arr)),
        "threshold_max": float(np.max(threshold_arr)),
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    return pd.DataFrame(rows), summary


def split_robustness_analysis(
    dev_records: list[dict[str, Any]], test_records: list[dict[str, Any]], params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_dev = np.asarray([r["y"] for r in dev_records], dtype=int)
    meta = pd.DataFrame(
        {
            "hostName": [r["hostName"] for r in dev_records],
            "first_timestamp": [r["first_timestamp"] for r in dev_records],
        }
    )
    rows: list[dict[str, Any]] = []

    def run_split(name: str, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
        if len(np.unique(y_dev[train_idx])) < 2 or len(np.unique(y_dev[val_idx])) < 2:
            rows.append(
                {
                    "split": name,
                    "status": "skipped_single_class_fold",
                    "train_n": len(train_idx),
                    "train_pos": int(y_dev[train_idx].sum()),
                    "val_n": len(val_idx),
                    "val_pos": int(y_dev[val_idx].sum()),
                }
            )
            return
        train_records = [dev_records[i] for i in train_idx]
        val_records = [dev_records[i] for i in val_idx]
        vocab = build_vocab(train_records)
        Xtr, ytr, _ = records_to_features(train_records, vocab, "full")
        Xv, yv, _ = records_to_features(val_records, vocab, "full")
        Xt, yt, _ = records_to_features(test_records, vocab, "full")
        model = make_model(params, SEED)
        model.fit(Xtr, ytr)
        sv = model.predict_proba(Xv)[:, 1]
        threshold = choose_threshold_by_fpr(yv, sv)
        st = model.predict_proba(Xt)[:, 1]
        rows.append(
            {
                "split": name,
                "status": "ok",
                "train_n": len(train_idx),
                "train_pos": int(ytr.sum()),
                "val_n": len(val_idx),
                "val_pos": int(yv.sum()),
                "vocab_size": len(vocab),
                "threshold": threshold,
                **metrics(yt, st, threshold),
            }
        )

    groups = meta["hostName"].astype(str).to_numpy()
    gss = GroupShuffleSplit(n_splits=10, test_size=0.25, random_state=SEED)
    for i, (tr, va) in enumerate(gss.split(np.zeros(len(y_dev)), y_dev, groups=groups), start=1):
        run_split(f"host_disjoint_{i}", tr, va)

    order = np.argsort(meta["first_timestamp"].to_numpy())
    n = len(order)
    temporal_train = order[: int(0.70 * n)]
    temporal_val = order[int(0.70 * n): int(0.85 * n)]
    run_split("temporal_train70_val15_to_official_test", temporal_train, temporal_val)

    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    host = detail[(detail["status"] == "ok") & detail["split"].str.startswith("host_disjoint_")]
    if not host.empty:
        for stat_name, fn in [
            ("host_disjoint_median", lambda x: x.median()),
            ("host_disjoint_min", lambda x: x.min()),
            ("host_disjoint_max", lambda x: x.max()),
        ]:
            summary_rows.append(
                {
                    "analysis": stat_name,
                    "valid_runs": int(len(host)),
                    "skipped_runs": int(10 - len(host)),
                    **{col: float(fn(host[col])) for col in ["AUC", "AP", "Precision", "Recall", "F1", "FPR"]},
                }
            )
    temporal = detail[(detail["status"] == "ok") & (detail["split"] == "temporal_train70_val15_to_official_test")]
    if not temporal.empty:
        r = temporal.iloc[0]
        summary_rows.append(
            {
                "analysis": "temporal_train70_val15_to_official_test",
                "valid_runs": 1,
                "skipped_runs": 0,
                **{col: float(r[col]) for col in ["AUC", "AP", "Precision", "Recall", "F1", "FPR"]},
            }
        )
    return detail, pd.DataFrame(summary_rows)


def prefix_analysis(
    dev_records: list[dict[str, Any]], test_records: list[dict[str, Any]], params: dict[str, Any]
) -> pd.DataFrame:
    y_dev = np.asarray([r["y"] for r in dev_records], dtype=int)
    tr, cal, val = primary_split_indices(y_dev)
    fit_idx = np.concatenate([tr, cal])
    vocab = build_vocab(dev_records)
    rows: list[dict[str, Any]] = []
    for prefix in PREFIXES:
        X_dev, yy, _ = records_to_features(dev_records, vocab, prefix)
        X_test, y_test, _ = records_to_features(test_records, vocab, prefix)
        model = make_model(params, SEED)
        model.fit(X_dev[fit_idx], yy[fit_idx])
        val_scores = model.predict_proba(X_dev[val])[:, 1]
        threshold = choose_threshold_by_fpr(yy[val], val_scores)
        test_scores = model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "prefix": str(prefix),
                "training_regime": "prefix-specific features; matched TabRF hyperparameters fixed",
                "threshold": threshold,
                **metrics(y_test, test_scores, threshold),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.canonical_dir.mkdir(parents=True, exist_ok=True)

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    params = dict(lock["best"]["TabRF"]["params"])

    dev_names = ["labelled_training_data.csv", *extra_development_files(args.data_dir)]
    dev_records = aggregate_records(load_csvs(args.data_dir, dev_names))
    test_records = aggregate_records(load_csvs(args.data_dir, ["labelled_testing_data.csv"]))

    inventory = {
        "tabrf_params": params,
        "matched_lock_sha256": lock.get("lock_sha256"),
        "development_files": dev_names,
        "development_processes": len(dev_records),
        "development_positives": int(sum(r["y"] for r in dev_records)),
        "official_test_processes": len(test_records),
        "official_test_positives": int(sum(r["y"] for r in test_records)),
    }

    print("[1/4] Rebuilding primary matched TabRF from raw CSVs...")
    parity = assert_primary_parity(dev_records, test_records, params, lock, args.canonical_main)
    inventory["primary_raw_rebuild"] = parity

    print("[2/4] Repeated threshold estimation with matched TabRF configuration...")
    repeat_detail, repeat_summary = repeated_threshold_analysis(dev_records, test_records, params)
    repeat_detail.to_csv(args.output_dir / "matched_repeated_thresholds.csv", index=False)
    (args.output_dir / "matched_repeated_threshold_summary.json").write_text(
        json.dumps(repeat_summary, indent=2), encoding="utf-8"
    )

    print("[3/4] Host-disjoint and chronological robustness...")
    split_detail, split_summary = split_robustness_analysis(dev_records, test_records, params)
    split_detail.to_csv(args.output_dir / "matched_split_robustness_runs.csv", index=False)
    split_summary.to_csv(args.output_dir / "matched_split_robustness_summary.csv", index=False)

    print("[4/4] Prefix-specific matched-configuration evaluation...")
    prefix = prefix_analysis(dev_records, test_records, params)
    prefix.to_csv(args.output_dir / "matched_prefix_results.csv", index=False)

    canonical_summary_rows: list[dict[str, Any]] = [
        {
            "analysis": "primary_matched_process_split",
            "detail": "matched TabRF; train+cal fit; val-FPR threshold; official test",
            "AUC": parity["AUC"], "AP": parity["AP"], "Precision": parity["Precision"],
            "Recall": parity["Recall"], "F1": parity["F1"], "FPR": parity["FPR"],
            "valid_runs": 1, "skipped_runs": 0,
        },
        {
            "analysis": "repeated_threshold_median",
            "detail": "50 development-only threshold folds; median threshold; all-development refit; official test",
            "AUC": repeat_summary["test_AUC"], "AP": repeat_summary["test_AP"],
            "Precision": repeat_summary["test_Precision"], "Recall": repeat_summary["test_Recall"],
            "F1": repeat_summary["test_F1"], "FPR": repeat_summary["test_FPR"],
            "valid_runs": 50, "skipped_runs": 0,
        },
    ]
    for _, r in split_summary.iterrows():
        canonical_summary_rows.append(
            {
                "analysis": r["analysis"],
                "detail": "matched TabRF hyperparameters; split-specific train vocabulary and validation threshold; official test",
                "AUC": r["AUC"], "AP": r["AP"], "Precision": r["Precision"],
                "Recall": r["Recall"], "F1": r["F1"], "FPR": r["FPR"],
                "valid_runs": int(r["valid_runs"]), "skipped_runs": int(r["skipped_runs"]),
            }
        )
    canonical_summary = pd.DataFrame(canonical_summary_rows)
    canonical_summary.to_csv(args.canonical_dir / "matched_secondary_robustness.csv", index=False)
    prefix.to_csv(args.canonical_dir / "matched_prefix_results.csv", index=False)

    (args.output_dir / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print("\n=== MATCHED SECONDARY ROBUSTNESS ===")
    print(canonical_summary.to_string(index=False))
    print("\n=== MATCHED PREFIX RESULTS ===")
    print(prefix.to_string(index=False))


if __name__ == "__main__":
    main()
