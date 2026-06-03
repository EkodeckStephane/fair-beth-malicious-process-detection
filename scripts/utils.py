#!/usr/bin/env python3
"""
Utilitaires communs : reproductibilité, ECE, Platt scaling, I/O JSON.
"""
import os
import json
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Fixe les graines pour la reproductibilité totale."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_ece(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error (Guo et al., 2017).
    y_true : array binaire {0,1}
    y_prob : array de probabilités prédites [0,1]
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        cnt = mask.sum()
        if cnt == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += (cnt / N) * abs(avg_conf - avg_acc)
    return float(ece)

def platt_scaling(scores, y_true):
    """
    Apprend une régression logistique (Platt 1999) sur un set de calibration.
    Retourne le modèle sklearn entraîné.
    """
    from sklearn.linear_model import LogisticRegression
    scores = np.asarray(scores).reshape(-1, 1)
    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr.fit(scores, y_true)
    return lr
