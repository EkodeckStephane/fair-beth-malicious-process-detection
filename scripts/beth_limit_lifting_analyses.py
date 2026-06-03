#!/usr/bin/env python3
"""Additional BETH analyses for attribution, streaming, cost, and split robustness."""
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 42
FPR_BUDGET = 0.05
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = Path(os.environ.get("FAIR_BETH_PIPELINE_DIR", SCRIPT_DIR)).resolve()
DATA_DIR = Path(os.environ.get("BETH_DATA_DIR", PIPELINE_DIR / "BETH_Dataset")).resolve()
PIPELINE_OUTPUT = Path(os.environ.get("FAIR_BETH_PIPELINE_OUTPUT", PIPELINE_DIR / "pipeline_output")).resolve()
OUTPUT_DIR = Path(os.environ.get("BETH_AUDIT_OUTPUT_DIR", REPO_ROOT / "results" / "beth_limit_lifting_generated")).resolve()
FIG_DIR = OUTPUT_DIR / "figures"

TRAIN_FILES = ["labelled_training_data.csv"]
VAL_FILES = ["labelled_validation_data.csv"]
TEST_FILES = ["labelled_testing_data.csv"]
GROUP_COLS = ["hostName", "processId"]
BASE_KEYS = [
    "n_events", "n_unique_eventIds", "entropy_eventIds",
    "mean_argsNum", "std_argsNum", "max_argsNum",
    "mean_returnValue", "std_returnValue",
    "duration", "event_rate",
    "n_unique_args", "mean_args_len", "n_unique_parents",
]


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_pickle(name):
    with open(PIPELINE_OUTPUT / name, "rb") as f:
        return pickle.load(f)


def load_csvs(file_names):
    frames = []
    for fn in file_names:
        path = DATA_DIR / fn
        df = pd.read_csv(path, low_memory=False)
        df["source_file"] = fn
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def extra_development_files():
    return sorted(
        f for f in os.listdir(DATA_DIR)
        if f.startswith("labelled_2021may") and f.endswith(".csv") and not f.endswith("-dns.csv")
    )


def aggregate_records(df):
    records = []
    for (host, pid), g in df.groupby(GROUP_COLS):
        g = g.sort_values("timestamp")
        if len(g) == 0:
            continue
        records.append({
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
        })
    return records


def prefix_slice(record, prefix):
    n = len(record["eventId"])
    if prefix == "full":
        k = n
    else:
        k = min(int(prefix), n)
    return k


def entropy_from_counts(counter, n):
    if n <= 0:
        return 0.0
    probs = np.array(list(counter.values()), dtype=float) / n
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


def records_to_features(records, vocab_event_ids, prefix="full"):
    eid_to_col = {int(e): i for i, e in enumerate(vocab_event_ids)}
    X_base = np.zeros((len(records), len(BASE_KEYS)), dtype=np.float32)
    X_hist = np.zeros((len(records), len(vocab_event_ids)), dtype=np.float32)
    y = np.zeros(len(records), dtype=np.int64)
    meta = []

    for i, r in enumerate(records):
        k = prefix_slice(r, prefix)
        eids = r["eventId"][:k]
        args_num = r["argsNum"][:k]
        ret = r["returnValue"][:k]
        ts = r["timestamp"][:k]
        args = r["args"][:k]
        parents = r["parentProcessId"][:k]
        counter = Counter(eids.tolist())
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
            col = eid_to_col.get(int(eid))
            if col is not None:
                X_hist[i, col] = cnt / k
        y[i] = r["y"]
        meta.append({
            "hostName": r["hostName"],
            "source_file": r["source_file"],
            "first_timestamp": r["first_timestamp"],
            "processId": r["processId"],
        })
    return np.concatenate([X_base, X_hist], axis=1), y, pd.DataFrame(meta)


def choose_threshold_by_fpr(y_val, scores, fpr_budget=FPR_BUDGET):
    fpr, _, thresholds = roc_curve(y_val, scores)
    valid = np.where(fpr <= fpr_budget)[0]
    return float(thresholds[valid[-1]]) if len(valid) else float(np.max(scores) + 1e-9)


def metrics(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "AUC": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else np.nan,
        "AP": float(average_precision_score(y, scores)) if len(np.unique(y)) == 2 else np.nan,
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def score_rf(model_out, X):
    Xs = model_out["scaler"].transform(X)
    return model_out["model"].predict_proba(Xs)[:, 1]


def feature_names(vocab_event_ids):
    return BASE_KEYS + [f"eventId_{int(e)}" for e in vocab_event_ids]


def permutation_importance_tab_rf(test_data, tabrf, threshold, names):
    rng = np.random.default_rng(SEED)
    X = np.array(test_data["X_tab"], copy=True)
    y = test_data["y"]
    base_scores = score_rf(tabrf, X)
    base_f1 = f1_score(y, (base_scores >= threshold).astype(int), zero_division=0)
    base_auc = roc_auc_score(y, base_scores)
    rows = []
    repeats = 100
    for j, name in enumerate(names):
        drops_f1 = []
        drops_auc = []
        for _ in range(repeats):
            Xp = np.array(X, copy=True)
            Xp[:, j] = rng.permutation(Xp[:, j])
            s = score_rf(tabrf, Xp)
            drops_f1.append(base_f1 - f1_score(y, (s >= threshold).astype(int), zero_division=0))
            drops_auc.append(base_auc - roc_auc_score(y, s))
        rows.append({
            "feature": name,
            "mean_f1_drop": float(np.mean(drops_f1)),
            "std_f1_drop": float(np.std(drops_f1)),
            "mean_auc_drop": float(np.mean(drops_auc)),
            "std_auc_drop": float(np.std(drops_auc)),
        })
    df = pd.DataFrame(rows).sort_values("mean_f1_drop", ascending=False)
    df.to_csv(OUTPUT_DIR / "beth_tabrf_permutation_importance.csv", index=False)
    top = df.head(15).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top["feature"], top["mean_f1_drop"])
    plt.xlabel("Mean F1 drop after permutation")
    plt.title("BETH TabRF held-out permutation importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "beth_tabrf_permutation_importance.png", dpi=300)
    plt.close()
    return df


def prefix_evaluation(dev_records, test_records, train_data, val_strat_idx, tabrf):
    vocab = train_data["vocab_event_ids"]
    rows = []
    for prefix in [10, 25, 50, 100, 250, "full"]:
        X_dev, y_dev, _ = records_to_features(dev_records, vocab, prefix=prefix)
        X_val = X_dev[val_strat_idx]
        y_val = y_dev[val_strat_idx]
        X_test, y_test, _ = records_to_features(test_records, vocab, prefix=prefix)
        s_val = score_rf(tabrf, X_val)
        s_test = score_rf(tabrf, X_test)
        thr = choose_threshold_by_fpr(y_val, s_val)
        row = {"prefix": str(prefix), "threshold": thr}
        row.update(metrics(y_test, s_test, thr))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "beth_prefix_results.csv", index=False)
    plt.figure(figsize=(6, 4))
    plt.plot(df["prefix"], df["F1"], marker="o", label="F1")
    plt.plot(df["prefix"], df["Recall"], marker="s", label="Recall")
    plt.plot(df["prefix"], df["Precision"], marker="^", label="Precision")
    plt.xlabel("Observed process events")
    plt.ylabel("Metric")
    plt.ylim(0, 1)
    plt.title("BETH TabRF prefix evaluation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "beth_prefix_curve.png", dpi=300)
    plt.close()
    return df


def cost_evaluation(y_test, scores):
    thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, 201)))
    rows = []
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        for fn_cost, fp_cost in [(1, 1), (5, 1), (10, 1), (25, 1), (50, 1)]:
            rows.append({
                "threshold": float(thr),
                "fn_cost": fn_cost,
                "fp_cost": fp_cost,
                "expected_cost": int(fn_cost) * int(fn) + int(fp_cost) * int(fp),
                "fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
            })
    df = pd.DataFrame(rows)
    best = df.loc[df.groupby(["fn_cost", "fp_cost"])["expected_cost"].idxmin()].copy()
    df.to_csv(OUTPUT_DIR / "beth_cost_curve_all_thresholds.csv", index=False)
    best.to_csv(OUTPUT_DIR / "beth_cost_optimal_thresholds.csv", index=False)
    plt.figure(figsize=(6, 4))
    for fn_cost in [1, 5, 10, 25, 50]:
        sub = df[df["fn_cost"] == fn_cost]
        plt.plot(sub["threshold"], sub["expected_cost"], label=f"FN:FP={fn_cost}:1")
    plt.xlabel("TabRF threshold")
    plt.ylabel("Expected test cost")
    plt.title("BETH TabRF operational cost sensitivity")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "beth_cost_sensitivity.png", dpi=300)
    plt.close()
    return best


def evasion_stress_tests(test_data, tabrf, threshold):
    rng = np.random.default_rng(SEED)
    X = np.array(test_data["X_tab"], copy=True)
    y = test_data["y"]
    rows = []

    def add_result(name, Xp):
        s = score_rf(tabrf, Xp)
        row = {"stress": name}
        row.update(metrics(y, s, threshold))
        rows.append(row)

    add_result("none", X)

    for frac in [0.10, 0.25, 0.50]:
        Xp = np.array(X, copy=True)
        event_part = Xp[:, len(BASE_KEYS):]
        mask = rng.random(event_part.shape) < frac
        event_part[mask] = 0.0
        row_sums = event_part.sum(axis=1, keepdims=True)
        event_part[:] = np.divide(event_part, row_sums, out=np.zeros_like(event_part), where=row_sums > 0)
        Xp[:, len(BASE_KEYS):] = event_part
        add_result(f"event_hist_dropout_{frac:.2f}", Xp)

    benign_mean = X[y == 0, len(BASE_KEYS):].mean(axis=0)
    for alpha in [0.10, 0.25, 0.50]:
        Xp = np.array(X, copy=True)
        event_part = Xp[:, len(BASE_KEYS):]
        event_part[y == 1] = (1 - alpha) * event_part[y == 1] + alpha * benign_mean
        row_sums = event_part.sum(axis=1, keepdims=True)
        event_part[:] = np.divide(event_part, row_sums, out=np.zeros_like(event_part), where=row_sums > 0)
        Xp[:, len(BASE_KEYS):] = event_part
        add_result(f"malicious_hist_benign_mixing_{alpha:.2f}", Xp)

    event_rate_idx = BASE_KEYS.index("event_rate")
    duration_idx = BASE_KEYS.index("duration")
    for scale in [0.5, 0.25, 0.10]:
        Xp = np.array(X, copy=True)
        Xp[y == 1, event_rate_idx] *= scale
        Xp[y == 1, duration_idx] /= max(scale, 1e-6)
        add_result(f"malicious_slowdown_{scale:.2f}", Xp)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "beth_evasion_stress_tests.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.barh(df["stress"].iloc[::-1], df["F1"].iloc[::-1])
    plt.xlabel("F1 at original validation-FPR threshold")
    plt.title("BETH TabRF feature-space stress tests")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "beth_evasion_stress_tests.png", dpi=300)
    plt.close()
    return df


def train_eval_group_temporal(dev_records, test_records, vocab):
    X_dev, y_dev, meta_dev = records_to_features(dev_records, vocab, prefix="full")
    X_test, y_test, _ = records_to_features(test_records, vocab, prefix="full")
    rows = []

    def run_split(name, train_idx, val_idx):
        if len(np.unique(y_dev[train_idx])) < 2 or len(np.unique(y_dev[val_idx])) < 2:
            rows.append({"split": name, "status": "skipped_single_class_fold"})
            return
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_dev[train_idx])
        Xv = scaler.transform(X_dev[val_idx])
        Xt = scaler.transform(X_test)
        model = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
        model.fit(Xtr, y_dev[train_idx])
        sv = model.predict_proba(Xv)[:, 1]
        st = model.predict_proba(Xt)[:, 1]
        thr = choose_threshold_by_fpr(y_dev[val_idx], sv)
        row = {
            "split": name,
            "status": "ok",
            "train_n": int(len(train_idx)),
            "train_pos": int(y_dev[train_idx].sum()),
            "val_n": int(len(val_idx)),
            "val_pos": int(y_dev[val_idx].sum()),
            "threshold": float(thr),
        }
        row.update(metrics(y_test, st, thr))
        rows.append(row)

    gss = GroupShuffleSplit(n_splits=10, test_size=0.25, random_state=SEED)
    groups = meta_dev["hostName"].astype(str).to_numpy()
    for i, (tr, va) in enumerate(gss.split(X_dev, y_dev, groups=groups), start=1):
        run_split(f"host_disjoint_{i}", tr, va)

    order = np.argsort(meta_dev["first_timestamp"].to_numpy())
    n = len(order)
    train_idx = order[: int(0.70 * n)]
    val_idx = order[int(0.70 * n): int(0.85 * n)]
    run_split("temporal_70_15_15", train_idx, val_idx)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "beth_group_temporal_robustness.csv", index=False)
    ok = df[df["status"] == "ok"].copy()
    if not ok.empty:
        summary = ok[["AUC", "AP", "Precision", "Recall", "F1", "FP", "FN"]].describe()
        summary.to_csv(OUTPUT_DIR / "beth_group_temporal_robustness_summary.csv")
    return df


def main():
    ensure_dirs()
    train_data = load_pickle("train_data.pkl")
    test_data = load_pickle("test_data.pkl")
    fusion = load_pickle("fusion_results.pkl")
    tabrf = fusion["rf_models"]["TabRF"]
    threshold = fusion["thresholds"]["TabRF"]["budget_fpr"]
    names = feature_names(train_data["vocab_event_ids"])

    print("[*] Loading and aggregating raw BETH CSVs...")
    dev_df = load_csvs(TRAIN_FILES + extra_development_files())
    test_df = load_csvs(TEST_FILES)
    dev_records = aggregate_records(dev_df)
    test_records = aggregate_records(test_df)
    y_dev_full = np.array([r["y"] for r in dev_records])
    train_idx, temp_idx = train_test_split(
        np.arange(len(y_dev_full)), test_size=0.35, stratify=y_dev_full, random_state=42
    )
    _, val_strat_idx = train_test_split(
        temp_idx, test_size=0.429, stratify=y_dev_full[temp_idx], random_state=43
    )

    inventory = {
        "development_processes": len(dev_records),
        "development_malicious": int(y_dev_full.sum()),
        "official_test_processes": len(test_records),
        "official_test_malicious": int(sum(r["y"] for r in test_records)),
        "validation_indices_reproduced": int(len(val_strat_idx)),
    }
    with open(OUTPUT_DIR / "inventory.json", "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    print("[*] BETH TabRF permutation importance...")
    permutation_importance_tab_rf(test_data, tabrf, threshold, names)

    print("[*] BETH prefix evaluation...")
    prefix_evaluation(dev_records, test_records, train_data, val_strat_idx, tabrf)

    print("[*] BETH operational cost evaluation...")
    y_test = test_data["y"]
    scores = tabrf["test"]
    cost_evaluation(y_test, scores)

    print("[*] BETH feature-space evasion stress tests...")
    evasion_stress_tests(test_data, tabrf, threshold)

    print("[*] BETH host-disjoint and temporal-disjoint robustness...")
    train_eval_group_temporal(dev_records, test_records, train_data["vocab_event_ids"])

    print(f"[+] Done. Outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
