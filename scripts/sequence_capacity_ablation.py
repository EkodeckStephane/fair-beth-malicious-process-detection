#!/usr/bin/env python3
"""Capacity and recurrent-cell ablation without target-copy reconstruction."""

import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
MAX_TOKENS = 512
BATCH_SIZE = 128
MAX_EPOCHS = 8
PATIENCE = 2
LEARNING_RATE = 1e-3
FPR_BUDGET = 0.05

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PIPELINE_OUTPUT = Path(
    os.environ.get("FAIR_BETH_PIPELINE_OUTPUT", SCRIPT_DIR / "pipeline_output")
).resolve()
OUTPUT_DIR = Path(
    os.environ.get(
        "FAIR_BETH_SEQUENCE_ABLATION_OUTPUT",
        REPO_ROOT / "results" / "sequence_capacity_ablation",
    )
).resolve()

CONFIGS = [
    {"name": "GRU-64x1", "cell": "GRU", "embed": 32, "hidden": 64, "layers": 1},
    {"name": "GRU-128x2", "cell": "GRU", "embed": 64, "hidden": 128, "layers": 2},
    {"name": "LSTM-128x2", "cell": "LSTM", "embed": 64, "hidden": 128, "layers": 2},
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_pickle(name):
    with open(PIPELINE_OUTPUT / name, "rb") as handle:
        return pickle.load(handle)


def normalize_sequences(values):
    values = np.asarray(values, dtype=np.int64)
    result = np.zeros((len(values), MAX_TOKENS), dtype=np.int64)
    width = min(values.shape[1], MAX_TOKENS)
    result[:, :width] = values[:, :width]
    return result


class NextEventModel(nn.Module):
    def __init__(self, vocab_size, cell, embed, hidden, layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed, padding_idx=0)
        recurrent = nn.GRU if cell == "GRU" else nn.LSTM
        self.recurrent = recurrent(
            embed,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.2 if layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden, vocab_size)

    def forward(self, tokens):
        encoded, _ = self.recurrent(self.embedding(tokens))
        return self.output(encoded)


def sequence_nll(model, loader, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(inputs)
            losses = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,
                reduction="none",
            ).reshape_as(targets)
            mask = (targets != 0).float()
            nll = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            scores.extend(nll.cpu().numpy().tolist())
    return np.asarray(scores, dtype=float)


def train_model(config, train_loader, validation_loader, vocab_size, device):
    set_seed(SEED)
    model = NextEventModel(
        vocab_size,
        config["cell"],
        config["embed"],
        config["hidden"],
        config["layers"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = np.inf
    best_state = None
    wait = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            optimizer.zero_grad()
            logits = model(inputs)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        validation_loss = float(sequence_nll(model, validation_loader, device).mean())
        training_loss = float(np.mean(epoch_losses))
        history.append(
            {
                "epoch": epoch,
                "training_nll": training_loss,
                "validation_nll": validation_loss,
            }
        )
        print(
            f"{config['name']} epoch={epoch} "
            f"train_nll={training_loss:.4f} val_nll={validation_loss:.4f}"
        )
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, history


def threshold_metrics(y_true, scores, threshold):
    prediction = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "ap": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "observed_fpr": float(fp / max(fp + tn, 1)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_pickle("train_data.pkl")
    validation = load_pickle("val_data.pkl")
    test = load_pickle("test_data.pkl")

    train_sequences = normalize_sequences(train["X_seq"])
    validation_sequences = normalize_sequences(validation["X_seq"])
    test_sequences = normalize_sequences(test["X_seq"])
    benign_train = train_sequences[train["y"] == 0]

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(benign_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation_sequences)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_sequences)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    results = []
    histories = {}
    for config in CONFIGS:
        model, history = train_model(
            config, train_loader, validation_loader, train["vocab_size"], device
        )
        validation_scores = sequence_nll(model, validation_loader, device)
        test_scores = sequence_nll(model, test_loader, device)
        threshold = float(
            np.quantile(validation_scores, 1.0 - FPR_BUDGET, method="higher")
        )
        row = {
            **config,
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "max_tokens": MAX_TOKENS,
            "epochs_run": len(history),
            "threshold": threshold,
            "validation_empirical_fpr": float(
                np.mean(validation_scores >= threshold)
            ),
            **threshold_metrics(test["y"], test_scores, threshold),
        }
        results.append(row)
        histories[config["name"]] = history
        with open(OUTPUT_DIR / f"{config['name']}_scores.pkl", "wb") as handle:
            pickle.dump(
                {
                    "validation_scores": validation_scores,
                    "test_scores": test_scores,
                    "test_labels": test["y"],
                    "threshold": threshold,
                    "config": config,
                },
                handle,
            )
        print(json.dumps(row, indent=2))

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "sequence_capacity_ablation.csv", index=False)
    with open(OUTPUT_DIR / "training_history.json", "w", encoding="utf-8") as handle:
        json.dump(histories, handle, indent=2)
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "device": str(device),
                "training_samples": int(len(benign_train)),
                "validation_samples": int(len(validation_sequences)),
                "test_samples": int(len(test_sequences)),
                "target_copy_path": False,
                "score_orientation_fixed_before_test": "higher next-event NLL is anomalous",
                "results": results,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
