import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


SEED = 42
DATA_PATH = Path("external_malbehavd/MalBehavD-V1-dataset.csv")
OUT_DIR = Path("external_malbehavd/output")
PREFIXES = [10, 25, 50, 100, "full"]
TOP_K = 128
REPEATS = 30
FPR_BUDGET = 0.05


def row_sequence(row, api_cols):
    vals = []
    for c in api_cols:
        v = row[c]
        if isinstance(v, str) and v:
            vals.append(v)
    return vals


def entropy(tokens):
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def api_family(api):
    families = ["Nt", "Reg", "Ldr", "Get", "Set", "Create", "Write", "Read", "Open", "Close", "Query"]
    for fam in families:
        if api.startswith(fam):
            return fam
    return "Other"


def build_vocab(seqs, top_k=TOP_K):
    counts = Counter()
    for seq in seqs:
        counts.update(seq)
    return [api for api, _ in counts.most_common(top_k)]


def featurize(seqs, vocab, prefix):
    rows = []
    for seq in seqs:
        s = seq if prefix == "full" else seq[: int(prefix)]
        length = len(s)
        counts = Counter(s)
        row = {
            "api_count": length,
            "api_unique": len(counts),
            "api_entropy": entropy(s),
            "repeat_ratio": 0.0 if length == 0 else 1.0 - (len(counts) / length),
        }
        for fam in ["Nt", "Reg", "Ldr", "Get", "Set", "Create", "Write", "Read", "Open", "Close", "Query", "Other"]:
            row[f"fam_{fam}"] = 0
        for api in s:
            row[f"fam_{api_family(api)}"] += 1
        denom = max(length, 1)
        for fam in ["Nt", "Reg", "Ldr", "Get", "Set", "Create", "Write", "Read", "Open", "Close", "Query", "Other"]:
            row[f"fam_{fam}"] /= denom
        for api in vocab:
            row[f"api_{api}"] = counts.get(api, 0) / denom
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


def choose_threshold(y, scores, policy):
    y = np.asarray(y)
    scores = np.asarray(scores)
    thresholds = np.unique(scores)
    if policy == "val_fpr_budget":
        # Most-permissive threshold within the FPR budget (matches BETH policy).
        benign_scores = scores[y == 0]
        best = float("inf")
        for thr in thresholds:
            fpr = np.mean(benign_scores >= thr) if len(benign_scores) else 0.0
            if fpr <= FPR_BUDGET:
                best = min(best, thr)
                break
        return float(best) if best != float("inf") else float(np.nextafter(thresholds.min(), -np.inf))
    if policy == "val_max_f1":
        best_thr, best_f1 = thresholds[0], -1.0
        for thr in thresholds:
            pred = (scores >= thr).astype(int)
            f1 = f1_score(y, pred, zero_division=0)
            if f1 > best_f1:
                best_thr, best_f1 = thr, f1
        return float(best_thr)
    if policy == "oracle_test_max_f1":
        best_thr, best_f1 = thresholds[0], -1.0
        for thr in thresholds:
            pred = (scores >= thr).astype(int)
            f1 = f1_score(y, pred, zero_division=0)
            if f1 > best_f1:
                best_thr, best_f1 = thr, f1
        return float(best_thr)
    raise ValueError(policy)


def metrics(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y, scores),
        "AP": average_precision_score(y, scores),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def bootstrap_ci(y, scores, threshold, n=1000):
    rng = np.random.default_rng(SEED)
    vals = []
    y = np.asarray(y)
    scores = np.asarray(scores)
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metrics(y[idx], scores[idx], threshold)["F1"])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return f"[{lo:.4f}, {hi:.4f}]"


def fit_eval_split(seqs, y, prefix="full", seed=SEED):
    idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(idx, test_size=0.40, stratify=y, random_state=seed)
    cal_idx, temp_idx = train_test_split(temp_idx, test_size=0.625, stratify=y[temp_idx], random_state=seed + 1)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.60, stratify=y[temp_idx], random_state=seed + 2)

    vocab = build_vocab([seqs[i] for i in train_idx])
    X_train = featurize([seqs[i] for i in train_idx], vocab, prefix)
    X_val = featurize([seqs[i] for i in val_idx], vocab, prefix)
    X_test = featurize([seqs[i] for i in test_idx], vocab, prefix)

    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y[train_idx])
    val_scores = clf.predict_proba(X_val)[:, 1]
    test_scores = clf.predict_proba(X_test)[:, 1]

    rows = []
    for policy in ["val_fpr_budget", "val_max_f1"]:
        thr = choose_threshold(y[val_idx], val_scores, policy)
        row = {
            "Dataset": "MalBehavD-V1",
            "Prefix": str(prefix),
            "Policy": policy,
            "Threshold": thr,
            **metrics(y[test_idx], test_scores, thr),
        }
        row["F1_95CI"] = bootstrap_ci(y[test_idx], test_scores, thr)
        rows.append(row)

    oracle_thr = choose_threshold(y[test_idx], test_scores, "oracle_test_max_f1")
    rows.append(
        {
            "Dataset": "MalBehavD-V1",
            "Prefix": str(prefix),
            "Policy": "oracle_test_max_f1_non_deployable",
            "Threshold": oracle_thr,
            **metrics(y[test_idx], test_scores, oracle_thr),
            "F1_95CI": bootstrap_ci(y[test_idx], test_scores, oracle_thr),
        }
    )
    return rows, clf, X_test, y[test_idx], test_scores


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    api_cols = [c for c in df.columns if c not in ("sha256", "labels")]
    seqs = [row_sequence(row, api_cols) for _, row in df.iterrows()]
    y = df["labels"].astype(int).to_numpy()

    inventory = {
        "samples": int(len(y)),
        "benign": int((y == 0).sum()),
        "malicious": int((y == 1).sum()),
        "min_sequence_length": int(min(map(len, seqs))),
        "median_sequence_length": float(np.median([len(s) for s in seqs])),
        "max_sequence_length": int(max(map(len, seqs))),
        "unique_apis": int(len(set(api for seq in seqs for api in seq))),
    }
    (OUT_DIR / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    main_rows, clf, X_test, y_test, test_scores = fit_eval_split(seqs, y, prefix="full", seed=SEED)
    pd.DataFrame(main_rows).to_csv(OUT_DIR / "external_results_table.csv", index=False)

    repeated = []
    splitter = StratifiedShuffleSplit(n_splits=REPEATS, test_size=0.25, random_state=SEED)
    for rep, (dev_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        dev_y = y[dev_idx]
        train_idx, val_idx = train_test_split(dev_idx, test_size=0.25, stratify=dev_y, random_state=SEED + rep)
        vocab = build_vocab([seqs[i] for i in train_idx])
        X_train = featurize([seqs[i] for i in train_idx], vocab, "full")
        X_val = featurize([seqs[i] for i in val_idx], vocab, "full")
        X_test_rep = featurize([seqs[i] for i in test_idx], vocab, "full")
        clf_rep = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=SEED + rep,
            n_jobs=-1,
        )
        clf_rep.fit(X_train, y[train_idx])
        val_scores = clf_rep.predict_proba(X_val)[:, 1]
        scores = clf_rep.predict_proba(X_test_rep)[:, 1]
        thr = choose_threshold(y[val_idx], val_scores, "val_fpr_budget")
        repeated.append({"repeat": rep, **metrics(y[test_idx], scores, thr)})
    rep_df = pd.DataFrame(repeated)
    rep_df.to_csv(OUT_DIR / "repeated_split_results.csv", index=False)
    rep_df.describe(percentiles=[0.25, 0.5, 0.75]).to_csv(OUT_DIR / "repeated_split_summary.csv")

    prefix_rows = []
    for prefix in PREFIXES:
        rows, *_ = fit_eval_split(seqs, y, prefix=prefix, seed=SEED)
        prefix_rows.extend(rows)
    prefix_df = pd.DataFrame(prefix_rows)
    prefix_df.to_csv(OUT_DIR / "prefix_results.csv", index=False)

    fig_rows = prefix_df[prefix_df["Policy"] == "val_fpr_budget"].copy()
    fig_rows["prefix_order"] = fig_rows["Prefix"].map({"10": 10, "25": 25, "50": 50, "100": 100, "full": 175})
    fig_rows = fig_rows.sort_values("prefix_order")
    plt.figure(figsize=(6.5, 4.0))
    plt.plot(fig_rows["Prefix"], fig_rows["F1"], marker="o", label="F1")
    plt.plot(fig_rows["Prefix"], fig_rows["Recall"], marker="s", label="Recall")
    plt.plot(fig_rows["Prefix"], fig_rows["Precision"], marker="^", label="Precision")
    plt.xlabel("API-call prefix length")
    plt.ylabel("Metric")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "malbehavd_prefix_curve.png", dpi=220)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, test_scores)
    plt.figure(figsize=(5.0, 4.0))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("MalBehavD-V1 PR curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "malbehavd_pr_curve.png", dpi=220)
    plt.close()

    perm = permutation_importance(clf, X_test, y_test, scoring="f1", n_repeats=20, random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    top = imp.head(15).iloc[::-1]
    plt.figure(figsize=(7.0, 4.8))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.xlabel("Permutation importance on F1")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "malbehavd_feature_importance.png", dpi=220)
    plt.close()

    print("Inventory:", inventory)
    print(pd.DataFrame(main_rows).to_string(index=False))
    print(rep_df[["AUC", "AP", "Precision", "Recall", "F1"]].describe(percentiles=[0.25, 0.5, 0.75]).to_string())
    print(prefix_df.to_string(index=False))


if __name__ == "__main__":
    main()
