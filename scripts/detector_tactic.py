#!/usr/bin/env python3
"""
Détecteur C : GRU bidirectionnel prédicteur de techniques MITRE ATT&CK.
Sortie multi-label (présence/absence de techniques par processus).
Les probabilités prédites servent de features pour la fusion finale.
NOUVEAU : calcule des statistiques agrégées (max, mean, std, entropy)
sur le vecteur de probabilités MITRE pour un signal plus discriminant.
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import f1_score

from config import (
    OUTPUT_DIR, DEVICE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, PATIENCE, DROPOUT
)
from utils import set_seed

set_seed()

class GRUMitrePredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, n_techniques, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_techniques)
        )

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        out = out.permute(0, 2, 1)
        pooled = torch.max(out, dim=2)[0]
        logits = self.classifier(pooled)
        return logits

def train_tactic(model, train_loader, val_loader, device, epochs, patience, pos_weight):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_f1 = -1.0
    wait = 0
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        all_probs, all_true = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                probs = torch.sigmoid(model(batch_x)).cpu().numpy()
                all_probs.append(probs)
                all_true.append(batch_y.numpy())
        all_probs = np.concatenate(all_probs)
        all_true = np.concatenate(all_true)
        preds = (all_probs > 0.5).astype(int)
        val_f1 = f1_score(all_true, preds, average='micro', zero_division=0)
        print(f"Epoch {epoch+1:02d} — train_loss={np.mean(losses):.4f} | val_micro_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "gru_tactic_best.pt"))
        else:
            wait += 1
            if wait >= patience:
                print("[*] Early stopping.")
                break
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "gru_tactic_best.pt")))
    return model

def extract_probs_and_stats(model, X_seq, device, batch_size=512):
    """
    Extrait les probabilités MITRE et calcule des statistiques agrégées
    par processus : max, mean, std, entropie de Shannon.
    """
    model.eval()
    ds = TensorDataset(torch.from_numpy(X_seq))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for (batch_x,) in dl:
            batch_x = batch_x.to(device)
            all_probs.append(torch.sigmoid(model(batch_x)).cpu().numpy())
    probs = np.concatenate(all_probs)  # (n_samples, n_techniques)

    # Statistiques par sample
    p_max = probs.max(axis=1)
    p_mean = probs.mean(axis=1)
    p_std = probs.std(axis=1, ddof=0)
    # Entropie de Shannon sur la distribution des probas MITRE
    eps = 1e-10
    p_norm = probs / (probs.sum(axis=1, keepdims=True) + eps)
    p_entropy = -np.sum(p_norm * np.log2(p_norm + eps), axis=1)

    return probs, p_max, p_mean, p_std, p_entropy

def main():
    print(f"[*] Device : {DEVICE}")
    with open(os.path.join(OUTPUT_DIR, "train_data.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "val_data.pkl"), "rb") as f:
        val_data = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "test_data.pkl"), "rb") as f:
        test_data = pickle.load(f)

    X_train = train_data["X_seq"]
    Y_train = train_data["Y_mitre"]
    X_val   = val_data["X_seq"]
    Y_val   = val_data["Y_mitre"]
    X_test  = test_data["X_seq"]
    Y_test  = test_data["Y_mitre"]

    vocab_size = train_data["vocab_size"]
    n_techniques = len(train_data["vocab_techniques"])
    seq_len = train_data["seq_len"]
    print(f"[*] vocab_size={vocab_size}, n_techniques={n_techniques}, seq_len={seq_len}")

    pos_counts = Y_train.sum(axis=0)
    neg_counts = len(Y_train) - pos_counts
    pos_weight_np = np.where(pos_counts > 0, neg_counts / (pos_counts + 1e-6), 1.0)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32).to(DEVICE)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(Y_val))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(Y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model = GRUMitrePredictor(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, n_techniques, DROPOUT)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Paramètres du GRU-MITRE : {n_params:,}")

    model = train_tactic(model, train_loader, val_loader, DEVICE, EPOCHS, PATIENCE, pos_weight)

    # Extraction des probas et statistiques sur tous les jeux
    P_train, max_train, mean_train, std_train, ent_train = extract_probs_and_stats(model, X_train, DEVICE)
    P_val,   max_val,   mean_val,   std_val,   ent_val   = extract_probs_and_stats(model, X_val,   DEVICE)
    P_test,  max_test,  mean_test,  std_test,  ent_test  = extract_probs_and_stats(model, X_test,  DEVICE)

    out = {
        "train_probs": P_train, "train_max": max_train, "train_mean": mean_train,
        "train_std": std_train, "train_entropy": ent_train,
        "val_probs": P_val,     "val_max": max_val,     "val_mean": mean_val,
        "val_std": std_val,     "val_entropy": ent_val,
        "test_probs": P_test,   "test_max": max_test,   "test_mean": mean_test,
        "test_std": std_test,   "test_entropy": ent_test,
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "n_techniques": n_techniques,
            "seq_len": seq_len,
        }
    }
    path = os.path.join(OUTPUT_DIR, "detector_tactic.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"[+] Détecteur tactique sauvegardé : {path}")
    print(f"    Probas test : max=[{max_test.min():.4f}, {max_test.max():.4f}] | entropy=[{ent_test.min():.4f}, {ent_test.max():.4f}]")

if __name__ == "__main__":
    main()
