#!/usr/bin/env python3
"""MalBehavD-V1 temporal-shift audit on a creation-date-validated corpus.

The audit mirrors the pilot design: a fixed chronologically isolated final test
set (most recent 20% by PE creation timestamp), an earlier 80% development pool
used for repeated stratified 5-fold x 10-repeat threshold selection, a locked
median threshold, one final evaluation on the isolated test set, and bootstrap
confidence intervals. Feature construction mirrors external_malbehavd_validation.py.

Required inputs:
  - MALBEHAVD_CSV, or external_malbehavd/MalBehavD-V1-dataset.csv
  - MALBEHAVD_VT_METADATA_DIR, or external_malbehavd/vt_metadata/

The metadata directory should contain JSONL files named vt_full_creation*.jsonl
or vt_pilot_creation*.jsonl with sha256, label, status, first_submission_date,
and creation_date fields.
"""
import datetime
import json
import math
import os
from collections import Counter
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
REPO_ROOT = Path(__file__).resolve().parents[1]
VT_METADATA_DIR = Path(
    os.environ.get("MALBEHAVD_VT_METADATA_DIR", REPO_ROOT / "external_malbehavd" / "vt_metadata")
)
CSV_PATH = Path(
    os.environ.get("MALBEHAVD_CSV", REPO_ROOT / "external_malbehavd" / "MalBehavD-V1-dataset.csv")
)
OUT_DIR = Path(
    os.environ.get("MALBEHAVD_TEMPORAL_OUTPUT", REPO_ROOT / "results" / "external_malbehavd_temporal")
)
SENTINEL_CUTOFF = datetime.datetime(1990, 1, 1).timestamp()
TOP_K = 128
FPR_BUDGET = 0.05
TEST_FRAC = 0.20
N_SPLITS = 5
N_REPEATS = 10
FAMILIES = ["Nt", "Reg", "Ldr", "Get", "Set", "Create", "Write", "Read", "Open", "Close", "Query", "Other"]


def api_family(api):
    for fam in FAMILIES[:-1]:
        if api.startswith(fam):
            return fam
    return "Other"


def entropy(tokens):
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def build_vocab(seqs, top_k=TOP_K):
    counts = Counter()
    for seq in seqs:
        counts.update(seq)
    return [api for api, _ in counts.most_common(top_k)]


def featurize(seqs, vocab):
    rows = []
    for seq in seqs:
        length = len(seq)
        counts = Counter(seq)
        row = {
            "api_count": length,
            "api_unique": len(counts),
            "api_entropy": entropy(seq),
            "repeat_ratio": 0.0 if length == 0 else 1.0 - (len(counts) / length),
        }
        for fam in FAMILIES:
            row[f"fam_{fam}"] = 0
        for api in seq:
            row[f"fam_{api_family(api)}"] += 1
        denom = max(length, 1)
        for fam in FAMILIES:
            row[f"fam_{fam}"] /= denom
        for api in vocab:
            row[f"api_{api}"] = counts.get(api, 0) / denom
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


def choose_threshold_by_fpr(y_val, scores, budget=FPR_BUDGET):
    fpr, _, thresholds = roc_curve(y_val, scores)
    finite = np.isfinite(thresholds)
    valid = np.where((fpr <= budget) & finite)[0]
    if not len(valid):
        return float(np.nextafter(np.max(scores), np.inf))
    return float(thresholds[valid[-1]])


def metrics_at(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, scores)) if len(set(y_true)) == 2 else float("nan"),
        "ap": float(average_precision_score(y_true, scores)) if len(set(y_true)) == 2 else float("nan"),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "observed_fpr": float(fp / max(fp + tn, 1)),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) load ALL collected records (pilot + full), dedup by sha256
    records_by_hash = {}
    metadata_files = sorted(VT_METADATA_DIR.glob("vt_full_creation*.jsonl")) + sorted(
        VT_METADATA_DIR.glob("vt_pilot_creation*.jsonl")
    )
    if not metadata_files:
        raise FileNotFoundError(f"No VirusTotal metadata JSONL files found in {VT_METADATA_DIR}")
    for fn in metadata_files:
        with fn.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    records_by_hash[r["sha256"]] = r
    print(f"Total unique hashes collected: {len(records_by_hash)} / 2543")

    clean = []
    excluded = {"missing": 0, "pre_1990": 0, "forged": 0, "not_found": 0}
    excluded_by_label = {0: Counter(), 1: Counter()}
    kept_by_label = Counter()
    for r in records_by_hash.values():
        if r.get("status") != "ok":
            excluded["not_found"] += 1
            excluded_by_label[r["label"]]["not_found"] += 1
            continue
        cd = r.get("creation_date")
        fsd = r.get("first_submission_date")
        if cd is None:
            excluded["missing"] += 1
            excluded_by_label[r["label"]]["missing"] += 1
            continue
        if cd < SENTINEL_CUTOFF:
            excluded["pre_1990"] += 1
            excluded_by_label[r["label"]]["pre_1990"] += 1
            continue
        if fsd is not None and cd > fsd:
            excluded["forged"] += 1
            excluded_by_label[r["label"]]["forged"] += 1
            continue
        clean.append((cd, r["label"], r["sha256"]))
        kept_by_label[r["label"]] += 1

    clean.sort()
    n = len(clean)
    print(f"Clean records: {n} / {len(records_by_hash)} ({100*n/len(records_by_hash):.1f}%)")
    print(f"Kept: benign={kept_by_label[0]} malicious={kept_by_label[1]}")
    print(f"Excluded overall: {excluded}")
    for lbl in [0, 1]:
        print(f"  Excluded label={lbl}: {dict(excluded_by_label[lbl])}")

    # 2) load API sequences for these hashes from the original CSV
    df = pd.read_csv(CSV_PATH)
    api_cols = [c for c in df.columns if c not in ("sha256", "labels")]
    df = df.drop_duplicates(subset="sha256").set_index("sha256")

    def get_seq(h):
        row = df.loc[h]
        vals = []
        for c in api_cols:
            v = row[c]
            if isinstance(v, str) and v:
                vals.append(v)
        return vals

    seqs_by_hash = {h: get_seq(h) for _, _, h in clean}

    # 3) fixed chronological test = most recent 20%; dev pool = earlier 80%
    k_test = int(TEST_FRAC * n)
    dev = clean[: n - k_test]
    test = clean[n - k_test :]
    y_dev = np.array([r[1] for r in dev])
    y_test = np.array([r[1] for r in test])
    print(f"\nDev pool: {len(dev)} (malicious={y_dev.sum()}, {100*y_dev.mean():.1f}%); "
          f"Test (chronologically isolated): {len(test)} (malicious={y_test.sum()}, {100*y_test.mean():.1f}%)")
    dev_date_range = [
        datetime.datetime.utcfromtimestamp(dev[0][0]).isoformat(sep=" "),
        datetime.datetime.utcfromtimestamp(dev[-1][0]).isoformat(sep=" "),
    ]
    test_date_range = [
        datetime.datetime.utcfromtimestamp(test[0][0]).isoformat(sep=" "),
        datetime.datetime.utcfromtimestamp(test[-1][0]).isoformat(sep=" "),
    ]
    print(f"Dev date range: [{dev_date_range[0]} .. {dev_date_range[1]}]")
    print(f"Test date range: [{test_date_range[0]} .. {test_date_range[1]}]")

    dev_hashes = [r[2] for r in dev]
    test_hashes = [r[2] for r in test]
    dev_seqs = [seqs_by_hash[h] for h in dev_hashes]
    test_seqs = [seqs_by_hash[h] for h in test_hashes]

    # 4) repeated stratified 5x10 CV within dev pool to select 50 thresholds
    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    fold_rows = []
    thresholds = []
    dev_idx = np.arange(len(dev))
    for fold_id, (tr_idx, va_idx) in enumerate(splitter.split(dev_idx, y_dev), start=1):
        vocab = build_vocab([dev_seqs[i] for i in tr_idx])
        X_tr = featurize([dev_seqs[i] for i in tr_idx], vocab)
        X_va = featurize([dev_seqs[i] for i in va_idx], vocab)
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2,
                                      random_state=SEED + fold_id, n_jobs=-1)
        clf.fit(X_tr, y_dev[tr_idx])
        va_scores = clf.predict_proba(X_va)[:, 1]
        thr = choose_threshold_by_fpr(y_dev[va_idx], va_scores)
        m = metrics_at(y_dev[va_idx], va_scores, thr)
        fold_rows.append({"fold": fold_id, "threshold": thr, "val_n": len(va_idx),
                           "val_positives": int(y_dev[va_idx].sum()), **m})
        thresholds.append(thr)
        if fold_id % 10 == 0:
            print(f"  fold {fold_id}/50 threshold={thr:.4f}")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT_DIR / "cross_fitted_thresholds.csv", index=False)
    thresholds = np.array(thresholds)
    median_thr = float(np.median(thresholds))
    print(f"\nThreshold distribution: median={median_thr:.4f} "
          f"IQR=[{np.quantile(thresholds,0.25):.4f}, {np.quantile(thresholds,0.75):.4f}] "
          f"min={thresholds.min():.4f} max={thresholds.max():.4f}")

    # 5) train final model on full dev pool, evaluate once on isolated test
    vocab_final = build_vocab(dev_seqs)
    X_dev_final = featurize(dev_seqs, vocab_final)
    X_test_final = featurize(test_seqs, vocab_final)
    final_clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2,
                                        random_state=SEED, n_jobs=-1)
    final_clf.fit(X_dev_final, y_dev)
    test_scores = final_clf.predict_proba(X_test_final)[:, 1]

    locked = metrics_at(y_test, test_scores, median_thr)
    print(f"\nLocked-threshold (median={median_thr:.4f}) result on isolated future test:")
    print(json.dumps(locked, indent=2))

    # 6) sensitivity across all 50 preselected thresholds on the same fixed test
    sens_rows = []
    for fold_id, thr in enumerate(thresholds, start=1):
        m = metrics_at(y_test, test_scores, float(thr))
        sens_rows.append({"source_fold": fold_id, "threshold": float(thr), **m})
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(OUT_DIR / "locked_threshold_sensitivity.csv", index=False)
    print("\nSensitivity across 50 preselected thresholds on fixed test:")
    for col in ["precision", "recall", "f1", "observed_fpr"]:
        print(f"  {col}: median={sens_df[col].median():.4f} "
              f"min={sens_df[col].min():.4f} max={sens_df[col].max():.4f}")

    # oracle test threshold (non-deployable, upper bound)
    oracle_thr, oracle_f1 = 0.5, -1.0
    for thr in np.unique(test_scores):
        f1 = f1_score(y_test, (test_scores >= thr).astype(int), zero_division=0)
        if f1 > oracle_f1:
            oracle_thr, oracle_f1 = thr, f1
    oracle_metrics = metrics_at(y_test, test_scores, oracle_thr)
    print(f"\nOracle test-threshold (non-deployable, upper bound): {json.dumps(oracle_metrics, indent=2)}")

    # 7) bootstrap CI on the locked-threshold test result (1000 resamples)
    rng = np.random.default_rng(SEED)
    boot_f1, boot_auc, boot_prec, boot_rec = [], [], [], []
    for _ in range(1000):
        idx = rng.integers(0, len(y_test), len(y_test))
        yb = y_test[idx]
        if len(set(yb)) < 2:
            continue
        sb = test_scores[idx]
        m = metrics_at(yb, sb, median_thr)
        boot_f1.append(m["f1"]); boot_prec.append(m["precision"]); boot_rec.append(m["recall"])
        boot_auc.append(roc_auc_score(yb, sb))

    def ci(vals):
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return [float(lo), float(hi)]

    print("\nBootstrap 95% CI (n=1000, locked threshold):")
    print(f"  F1 CI: {ci(boot_f1)}")
    print(f"  Precision CI: {ci(boot_prec)}")
    print(f"  Recall CI: {ci(boot_rec)}")
    print(f"  AUC CI: {ci(boot_auc)}")

    summary = {
        "n_total_collected": len(records_by_hash),
        "n_clean": n,
        "excluded_overall": excluded,
        "excluded_by_label": {str(k): dict(v) for k, v in excluded_by_label.items()},
        "dev_n": len(dev), "dev_malicious": int(y_dev.sum()),
        "test_n": len(test), "test_malicious": int(y_test.sum()),
        "dev_date_range_utc": dev_date_range,
        "test_date_range_utc": test_date_range,
        "threshold_median": median_thr,
        "threshold_iqr": [float(np.quantile(thresholds, 0.25)), float(np.quantile(thresholds, 0.75))],
        "threshold_min": float(thresholds.min()),
        "threshold_max": float(thresholds.max()),
        "locked_test_result": locked,
        "locked_threshold_sensitivity": {
            "precision": {
                "median": float(sens_df["precision"].median()),
                "min": float(sens_df["precision"].min()),
                "max": float(sens_df["precision"].max()),
            },
            "recall": {
                "median": float(sens_df["recall"].median()),
                "min": float(sens_df["recall"].min()),
                "max": float(sens_df["recall"].max()),
            },
            "f1": {
                "median": float(sens_df["f1"].median()),
                "min": float(sens_df["f1"].min()),
                "max": float(sens_df["f1"].max()),
            },
            "observed_fpr": {
                "median": float(sens_df["observed_fpr"].median()),
                "min": float(sens_df["observed_fpr"].min()),
                "max": float(sens_df["observed_fpr"].max()),
            },
        },
        "oracle_test_result": oracle_metrics,
        "bootstrap_ci": {"f1": ci(boot_f1), "precision": ci(boot_prec), "recall": ci(boot_rec), "auc": ci(boot_auc)},
    }
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
