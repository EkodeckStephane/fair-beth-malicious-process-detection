#!/usr/bin/env python3
"""
Détecteur B : GRU Autoencoder sur les séquences d'eventIds.
Entraîné uniquement sur les processus bénins (apprentissage non supervisé).
Anomalie = erreur de reconstruction moyenne (NLL par token).
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import (
    OUTPUT_DIR, DEVICE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, PATIENCE, DROPOUT
)
from utils import set_seed

set_seed()

class GRUAutoEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.enc_gru = nn.GRU(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dec_gru = nn.GRU(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.out_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.enc_gru(emb)
        dec_out, _ = self.dec_gru(emb, h)
        logits = self.out_proj(dec_out)
        return logits

def reconstruction_errors(model, dataloader, device):
    model.eval()
    all_errors = []
    with torch.no_grad():
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            log_probs = torch.log_softmax(logits, dim=-1)
            target = batch_x.unsqueeze(-1)
            nll = -log_probs.gather(dim=-1, index=target).squeeze(-1)
            mask = (batch_x != 0).float()
            seq_err = (nll * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            all_errors.extend(seq_err.cpu().numpy())
    return np.array(all_errors)

def train_model(model, train_loader, val_loader, device, epochs, patience):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    best_val = float('inf')
    wait = 0
    for epoch in range(epochs):
        model.train()
        losses = []
        for (batch_x,) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss_mat = criterion(logits.view(-1, logits.size(-1)), batch_x.view(-1))
            loss_mat = loss_mat.view(batch_x.size())
            mask = (batch_x != 0).float()
            loss = (loss_mat * mask).sum() / mask.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        val_err = reconstruction_errors(model, val_loader, device).mean()
        print(f"Epoch {epoch+1:02d} — train_loss={np.mean(losses):.4f} | val_recon={val_err:.4f}")
        if val_err < best_val:
            best_val = val_err
            wait = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "gru_ae_best.pt"))
        else:
            wait += 1
            if wait >= patience:
                print("[*] Early stopping.")
                break
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "gru_ae_best.pt")))
    return model

def main():
    print(f"[*] Device : {DEVICE}")
    with open(os.path.join(OUTPUT_DIR, "train_data.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "val_data.pkl"), "rb") as f:
        val_data = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "test_data.pkl"), "rb") as f:
        test_data = pickle.load(f)

    X_train_seq = train_data["X_seq"]
    y_train = train_data["y"]
    X_val_seq = val_data["X_seq"]
    X_test_seq = test_data["X_seq"]
    vocab_size = train_data["vocab_size"]
    seq_len = train_data["seq_len"]

    X_train_benign = X_train_seq[y_train == 0]
    print(f"[*] Séquences bénignes pour entraînement : {len(X_train_benign):,}")

    train_ds = TensorDataset(torch.from_numpy(X_train_benign))
    val_ds   = TensorDataset(torch.from_numpy(X_val_seq))
    test_ds  = TensorDataset(torch.from_numpy(X_test_seq))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model = GRUAutoEncoder(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Paramètres du GRU-AE : {n_params:,}")

    model = train_model(model, train_loader, val_loader, DEVICE, EPOCHS, PATIENCE)

    train_loader_all = DataLoader(TensorDataset(torch.from_numpy(X_train_seq)),
                                  batch_size=BATCH_SIZE, shuffle=False)
    s_train = reconstruction_errors(model, train_loader_all, DEVICE)
    s_val   = reconstruction_errors(model, val_loader, DEVICE)
    s_test  = reconstruction_errors(model, test_loader, DEVICE)

    out = {
        "train_scores": s_train,
        "val_scores": s_val,
        "test_scores": s_test,
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "seq_len": seq_len,
        }
    }
    path = os.path.join(OUTPUT_DIR, "detector_sequence.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"[+] Détecteur séquentiel sauvegardé : {path}")
    print(f"    Train recon : [{s_train.min():.4f}, {s_train.max():.4f}]")

if __name__ == "__main__":
    main()
