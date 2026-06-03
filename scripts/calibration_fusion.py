#!/usr/bin/env python3
"""
Defensible calibration and fusion.

Main protocol:
  - no heuristic MITRE features unless explicitly enabled in config;
  - Platt scaling is used only to normalize detector scores;
  - every supervised model is evaluated later with the same validation-derived
    threshold policy;
  - ablations are saved so the paper can report what actually adds signal.
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_curve

from config import OUTPUT_DIR, N_BINS_ECE, FPR_BUDGET
from utils import compute_ece, platt_scaling


def score_tabular(model_dict, X_tab):
    scaler = model_dict["scaler"]
    model = model_dict["model"]
    return -model.decision_function(scaler.transform(X_tab))


def score_sequence(model_dict, X_seq, device, batch_size=512):
    from detector_sequence import GRUAutoEncoder

    cfg = model_dict["config"]
    model = GRUAutoEncoder(
        cfg["vocab_size"], cfg["embed_dim"], cfg["hidden_dim"], cfg["num_layers"]
    )
    model.load_state_dict(model_dict["model_state"])
    model.to(device)
    model.eval()

    loader = DataLoader(TensorDataset(torch.from_numpy(X_seq)),
                        batch_size=batch_size, shuffle=False)
    errors = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            log_probs = torch.log_softmax(logits, dim=-1)
            target = batch_x.unsqueeze(-1)
            nll = -log_probs.gather(dim=-1, index=target).squeeze(-1)
            mask = (batch_x != 0).float()
            seq_err = (nll * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            errors.extend(seq_err.cpu().numpy())
    return np.asarray(errors)


def choose_threshold_by_fpr(y_val, val_scores, fpr_budget=FPR_BUDGET):
    fpr, _, thresholds = roc_curve(y_val, val_scores)
    valid = np.where(fpr <= fpr_budget)[0]
    return float(thresholds[valid[-1]]) if len(valid) else 0.5


def choose_threshold_by_f1(y_val, val_scores):
    candidates = np.unique(val_scores)
    if len(candidates) > 1000:
        candidates = np.quantile(val_scores, np.linspace(0, 1, 1000))
    best_thr = 0.5
    best_f1 = -1.0
    for thr in candidates:
        preds = (val_scores >= thr).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return float(best_thr), float(best_f1)


def fit_rf(train_X, train_y, cal_X, cal_y, val_X, test_X, n_estimators=300):
    scaler = StandardScaler()
    X_fit = np.vstack([train_X, cal_X])
    y_fit = np.concatenate([train_y, cal_y])
    X_fit_s = scaler.fit_transform(X_fit)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_fit_s, y_fit)

    return {
        "model": rf,
        "scaler": scaler,
        "train": rf.predict_proba(scaler.transform(train_X))[:, 1],
        "cal": rf.predict_proba(scaler.transform(cal_X))[:, 1],
        "val": rf.predict_proba(scaler.transform(val_X))[:, 1],
        "test": rf.predict_proba(scaler.transform(test_X))[:, 1],
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device : {device}")

    with open(os.path.join(OUTPUT_DIR, "train_data.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "cal_data.pkl"), "rb") as f:
        cal_data = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "val_strat_data.pkl"), "rb") as f:
        val_data = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "test_data.pkl"), "rb") as f:
        test_data = pickle.load(f)

    y_train = train_data["y"]
    y_cal = cal_data["y"]
    y_val = val_data["y"]

    with open(os.path.join(OUTPUT_DIR, "detector_tabular.pkl"), "rb") as f:
        tab_model = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "detector_sequence.pkl"), "rb") as f:
        seq_model = pickle.load(f)

    print("[*] Scoring detector A (Isolation Forest)...")
    s_tab_train = score_tabular(tab_model, train_data["X_tab"])
    s_tab_cal = score_tabular(tab_model, cal_data["X_tab"])
    s_tab_val = score_tabular(tab_model, val_data["X_tab"])
    s_tab_test = score_tabular(tab_model, test_data["X_tab"])

    print("[*] Scoring detector B (GRU-AE)...")
    s_seq_train = score_sequence(seq_model, train_data["X_seq"], device)
    s_seq_cal = score_sequence(seq_model, cal_data["X_seq"], device)
    s_seq_val = score_sequence(seq_model, val_data["X_seq"], device)
    s_seq_test = score_sequence(seq_model, test_data["X_seq"], device)

    print("[*] Platt scaling on calibration set...")
    platt_tab = platt_scaling(s_tab_cal, y_cal)
    platt_seq = platt_scaling(s_seq_cal, y_cal)

    def calibrate(model, scores):
        return model.predict_proba(scores.reshape(-1, 1))[:, 1]

    p_tab = {
        "train": calibrate(platt_tab, s_tab_train),
        "cal": calibrate(platt_tab, s_tab_cal),
        "val": calibrate(platt_tab, s_tab_val),
        "test": calibrate(platt_tab, s_tab_test),
    }
    p_seq = {
        "train": calibrate(platt_seq, s_seq_train),
        "cal": calibrate(platt_seq, s_seq_cal),
        "val": calibrate(platt_seq, s_seq_val),
        "test": calibrate(platt_seq, s_seq_test),
    }

    ece = {
        "tab": compute_ece(y_cal, p_tab["cal"], N_BINS_ECE),
        "seq": compute_ece(y_cal, p_seq["cal"], N_BINS_ECE),
    }
    print(f"    ECE tab={ece['tab']:.6f} seq={ece['seq']:.6f}")

    simple = {k: (p_tab[k] + p_seq[k]) / 2.0 for k in p_tab}

    score_features = {
        split: np.column_stack([p_tab[split], p_seq[split], simple[split]])
        for split in ["train", "cal", "val", "test"]
    }
    tab_features = {
        "train": train_data["X_tab"],
        "cal": cal_data["X_tab"],
        "val": val_data["X_tab"],
        "test": test_data["X_tab"],
    }
    full_features = {
        split: np.hstack([score_features[split], tab_features[split]])
        for split in score_features
    }

    print("[*] Training ablation RF models...")
    models = {
        "ScoreRF": fit_rf(score_features["train"], y_train, score_features["cal"], y_cal,
                          score_features["val"], score_features["test"], n_estimators=200),
        "TabRF": fit_rf(tab_features["train"], y_train, tab_features["cal"], y_cal,
                        tab_features["val"], tab_features["test"], n_estimators=200),
        "MetaRF": fit_rf(full_features["train"], y_train, full_features["cal"], y_cal,
                         full_features["val"], full_features["test"], n_estimators=300),
    }

    thresholds = {}
    for name, result in models.items():
        budget = choose_threshold_by_fpr(y_val, result["val"], FPR_BUDGET)
        optimal, f1 = choose_threshold_by_f1(y_val, result["val"])
        thresholds[name] = {
            "budget_fpr": budget,
            "optimal_f1": optimal,
            "val_optimal_f1": f1,
        }
        print(f"    {name}: threshold_fpr={budget:.6f}, threshold_f1={optimal:.6f}")

    out = {
        "platt_models": {"tab": platt_tab, "seq": platt_seq},
        "ece": ece,
        "detector_scores": {"tab": p_tab, "seq": p_seq, "simple": simple},
        "rf_models": models,
        "thresholds": thresholds,
        # Backward-compatible aliases for older plotting/article scripts.
        "simple_scores": simple,
        "meta_scores": {k: models["MetaRF"][k] for k in ["train", "cal", "val", "test"]},
        "threshold_budget": thresholds["MetaRF"]["budget_fpr"],
        "threshold_optimal": thresholds["MetaRF"]["optimal_f1"],
    }
    path = os.path.join(OUTPUT_DIR, "fusion_results.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"[+] Fusion results saved: {path}")


if __name__ == "__main__":
    main()
