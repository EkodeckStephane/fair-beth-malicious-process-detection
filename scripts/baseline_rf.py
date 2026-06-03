#!/usr/bin/env python3
"""
Baseline : Random Forest simple sur les features tabulaires agrégées.
Sert de référence honnête pour évaluer l'apport du multi-détecteur.
"""
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

from config import OUTPUT_DIR
from utils import set_seed

set_seed()

def main():
    print("[*] Chargement des données...")
    with open(os.path.join(OUTPUT_DIR, "train_data.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "test_data.pkl"), "rb") as f:
        test_data = pickle.load(f)

    X_train = train_data["X_tab"]
    y_train = train_data["y"]
    X_test = test_data["X_tab"]
    y_test = test_data["y"]

    print(f"[*] Features tabulaires : {X_train.shape[1]} dimensions")
    print(f"[*] Train : {len(y_train):,} samples (evil={y_train.sum():,})")
    print(f"[*] Test  : {len(y_test):,} samples (evil={y_test.sum():,})")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    print("[*] Entraînement du Random Forest baseline...")
    rf.fit(X_train_s, y_train)

    probs = rf.predict_proba(X_test_s)[:, 1]
    preds = rf.predict(X_test_s)

    auc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)
    f1 = f1_score(y_test, preds, zero_division=0)
    pr = precision_score(y_test, preds, zero_division=0)
    re = recall_score(y_test, preds, zero_division=0)

    print(f"\n=== Baseline RF (features tabulaires seules) ===")
    print(f"AUC={auc:.4f} | AP={ap:.4f} | Pr={pr:.4f} | Re={re:.4f} | F1={f1:.4f}")

    out = {
        "model": rf,
        "scaler": scaler,
        "test_probs": probs,
        "test_preds": preds,
        "metrics": {"auc": auc, "ap": ap, "precision": pr, "recall": re, "f1": f1}
    }
    path = os.path.join(OUTPUT_DIR, "baseline_rf.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"[+] Baseline sauvegardé : {path}")

if __name__ == "__main__":
    main()
