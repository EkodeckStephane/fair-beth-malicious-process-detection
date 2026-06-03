#!/usr/bin/env python3
"""
Détecteur A : Isolation Forest sur les features tabulaires agrégées par processus.
Entraînement non supervisé (les labels ne servent que pour le réglage de contamination).
"""
import os
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import OUTPUT_DIR, IF_N_ESTIMATORS
from utils import ensure_dir

class TabularDetector:
    def __init__(self, contamination='auto', random_state=42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            max_samples='auto',
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
            bootstrap=False
        )
        self.fitted = False

    def fit(self, X):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        self.fitted = True
        return self

    def score(self, X):
        if not self.fitted:
            raise RuntimeError("Le modèle n'a pas été entraîné. Appelez fit() d'abord.")
        Xs = self.scaler.transform(X)
        return -self.model.decision_function(Xs)

def main():
    print("[*] Chargement des données pré-traitées...")
    with open(os.path.join(OUTPUT_DIR, "train_data.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "val_data.pkl"), "rb") as f:
        val_data = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "test_data.pkl"), "rb") as f:
        test_data = pickle.load(f)

    X_train = train_data["X_tab"]
    y_train = train_data["y"]

    contamination = float(np.mean(y_train)) if np.mean(y_train) > 0 else 'auto'
    print(f"[*] Contamination utilisée : {contamination}")

    det = TabularDetector(contamination=contamination)
    det.fit(X_train)

    s_train = det.score(X_train)
    s_val   = det.score(val_data["X_tab"])
    s_test  = det.score(test_data["X_tab"])

    out = {
        "train_scores": s_train,
        "val_scores": s_val,
        "test_scores": s_test,
        "scaler": det.scaler,
        "model": det.model,
        "contamination": contamination,
    }
    path = os.path.join(OUTPUT_DIR, "detector_tabular.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"[+] Détecteur tabulaire sauvegardé : {path}")
    print(f"    Train scores : [{s_train.min():.4f}, {s_train.max():.4f}]")

if __name__ == "__main__":
    main()
