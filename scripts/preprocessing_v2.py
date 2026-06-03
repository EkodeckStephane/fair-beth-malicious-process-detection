#!/usr/bin/env python3
"""
Prétraitement BETH : ingestion, feature engineering comportemental,
agrégation par processus, construction des séquences, mapping MITRE.

CORRECTIONS v3 :
  1. Les fichiers 2021may sont réintégrés au train pour enrichissement.
  2. Les colonnes 'sus' et 'evil' sont EXCLUES des features tabulaires.
  3. Temporal sampling uniforme pour les séquences dépassant le 95e percentile.
  4. Triple split stratifié depuis le train enrichi :
       train_data  (65%)  → entraînement des détecteurs
       cal_data    (20%)  → Platt scaling
       val_strat_data (15%) → sélection de seuil (biclasse garanti)
  5. Le val officiel (monoclasse) est conservé comme val_data pour
     l'early stopping des modèles séquentiels (loss de reconstruction / F1 micro MITRE).
"""
import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR, OUTPUT_DIR, TRAIN_FILES, VAL_FILES, TEST_FILES,
    GROUP_COLS, SEQ_LEN_PERCENTILE, MITRE_MAP_PATH,
    USE_2021MAY_AS_DEVELOPMENT, USE_HEURISTIC_MITRE_FEATURES
)
from utils import set_seed, ensure_dir, save_json

set_seed()

def save_pickle_atomic(obj, path):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)

def load_csvs(file_names):
    dfs = []
    for fn in file_names:
        path = os.path.join(DATA_DIR, fn)
        if not os.path.exists(path):
            print(f"[!] Fichier manquant : {path} — ignoré.")
            continue
        df = pd.read_csv(path, low_memory=False)
        df["source_file"] = fn
        dfs.append(df)
        print(f"[+] {fn} : {len(df):,} lignes, {len(df.columns)} colonnes")
    if not dfs:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {DATA_DIR}")
    return pd.concat(dfs, ignore_index=True)

def attach_mitre_to_events(df, mitre_map):
    if not USE_HEURISTIC_MITRE_FEATURES:
        df["mitre_techniques"] = [[] for _ in range(len(df))]
        return df

    def map_event(eid):
        key = str(int(eid))
        return mitre_map.get(key, ["T1499"])
    df["mitre_techniques"] = df["eventId"].apply(map_event)
    return df

def aggregate_process(group_df):
    group_df = group_df.sort_values("timestamp")
    n_events = len(group_df)
    if n_events == 0:
        return None

    event_ids = group_df["eventId"].values.astype(int)
    args_nums = group_df["argsNum"].values.astype(float)
    ret_vals = group_df["returnValue"].values.astype(float)
    timestamps = group_df["timestamp"].values.astype(float)

    evt_counter = Counter(event_ids)
    tech_list = []
    for techs in group_df["mitre_techniques"]:
        if isinstance(techs, list):
            tech_list.extend(techs)
    tech_counter = Counter(tech_list)

    duration = float(timestamps[-1] - timestamps[0]) if n_events > 1 else 0.0
    event_rate = n_events / (duration + 1e-6)
    probs = np.array(list(evt_counter.values()), dtype=float) / n_events
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    args_texts = group_df["args"].astype(str).values
    n_unique_args = len(set(args_texts))
    mean_args_len = np.mean([len(a) for a in args_texts])
    n_unique_parents = group_df["parentProcessId"].nunique()
    sequence = event_ids.tolist()
    label = int(group_df["evil"].max())

    return {
        "n_events": n_events,
        "n_unique_eventIds": len(evt_counter),
        "entropy_eventIds": float(entropy),
        "mean_argsNum": float(args_nums.mean()),
        "std_argsNum": float(args_nums.std(ddof=0)) if n_events > 1 else 0.0,
        "max_argsNum": int(args_nums.max()),
        "mean_returnValue": float(ret_vals.mean()),
        "std_returnValue": float(ret_vals.std(ddof=0)) if n_events > 1 else 0.0,
        "duration": duration,
        "event_rate": float(event_rate),
        "n_unique_args": n_unique_args,
        "mean_args_len": float(mean_args_len),
        "n_unique_parents": n_unique_parents,
        "event_counter": evt_counter,
        "tech_counter": tech_counter,
        "sequence": sequence,
        "label": label,
    }

def build_features(dfs, mitre_map, fit_vocab=True,
                   vocab_event_ids=None, vocab_techniques=None):
    all_proc_feats = []
    for df in tqdm(dfs, desc="Fichiers"):
        if df.empty:
            continue
        df = attach_mitre_to_events(df, mitre_map)
        grouped = df.groupby(GROUP_COLS)
        for _, g in tqdm(grouped, desc="Processus", leave=False):
            feat = aggregate_process(g)
            if feat is not None:
                all_proc_feats.append(feat)

    print(f"[+] Processus agrégés : {len(all_proc_feats):,}")

    if fit_vocab:
        all_eids = set()
        all_techs = set()
        for feat in all_proc_feats:
            all_eids.update(feat["event_counter"].keys())
            if USE_HEURISTIC_MITRE_FEATURES:
                all_techs.update(feat["tech_counter"].keys())
        vocab_event_ids = sorted(all_eids)
        vocab_techniques = sorted(all_techs)
        print(f"[+] Vocab eventIds : {len(vocab_event_ids)}")
        print(f"[+] Vocab MITRE    : {len(vocab_techniques)}")

    n_event_types = len(vocab_event_ids)
    n_techniques = len(vocab_techniques)
    eid2idx = {e: i + 2 for i, e in enumerate(vocab_event_ids)}
    unk_idx = 1
    tech2idx = {t: i for i, t in enumerate(vocab_techniques)}

    lengths = [len(f["sequence"]) for f in all_proc_feats]
    seq_len = int(np.percentile(lengths, SEQ_LEN_PERCENTILE))
    max_len = max(lengths)
    print(f"[+] Longueur séquences : min={min(lengths)}, max={max_len}, 95th={seq_len}")

    n_samples = len(all_proc_feats)
    base_keys = [
        "n_events", "n_unique_eventIds", "entropy_eventIds",
        "mean_argsNum", "std_argsNum", "max_argsNum",
        "mean_returnValue", "std_returnValue",
        "duration", "event_rate",
        "n_unique_args", "mean_args_len", "n_unique_parents"
    ]
    X_base = np.zeros((n_samples, len(base_keys)), dtype=np.float32)
    X_evt_hist = np.zeros((n_samples, n_event_types), dtype=np.float32)
    X_tech_hist = np.zeros((n_samples, n_techniques), dtype=np.float32)
    X_seq = np.zeros((n_samples, seq_len), dtype=np.int64)
    Y_mitre = np.zeros((n_samples, n_techniques), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)

    for i, feat in enumerate(tqdm(all_proc_feats, desc="Arrays")):
        for j, k in enumerate(base_keys):
            X_base[i, j] = feat[k]
        ne = feat["n_events"]
        for eid, cnt in feat["event_counter"].items():
            idx = eid2idx.get(eid, unk_idx)
            if idx != unk_idx:
                X_evt_hist[i, idx - 2] = cnt / ne
        for tech, cnt in feat["tech_counter"].items():
            idx = tech2idx.get(tech)
            if idx is not None:
                X_tech_hist[i, idx] = cnt / ne
                Y_mitre[i, idx] = 1.0
        raw_seq = feat["sequence"]
        if len(raw_seq) > seq_len:
            indices = np.linspace(0, len(raw_seq) - 1, seq_len, dtype=int)
            sampled_seq = [raw_seq[idx] for idx in indices]
        else:
            sampled_seq = raw_seq
        sampled_seq = sampled_seq[:seq_len]
        X_seq[i, :len(sampled_seq)] = [eid2idx.get(e, unk_idx) for e in sampled_seq]
        y[i] = feat["label"]

    tab_parts = [X_base, X_evt_hist]
    if USE_HEURISTIC_MITRE_FEATURES:
        tab_parts.append(X_tech_hist)
    X_tab = np.concatenate(tab_parts, axis=1)
    return {
        "X_tab": X_tab, "X_seq": X_seq, "Y_mitre": Y_mitre, "y": y,
        "vocab_event_ids": vocab_event_ids, "vocab_techniques": vocab_techniques,
        "vocab_size": n_event_types + 2, "seq_len": seq_len,
        "base_keys": base_keys, "eid2idx": eid2idx, "tech2idx": tech2idx,
    }

def extract_subset(data, idx):
    """Extrait un sous-ensemble indexé en préservant les métadonnées."""
    subset = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.shape[0] == len(data["y"]):
            subset[k] = v[idx]
        else:
            subset[k] = v  # métadonnées partagées (vocab, mappings, etc.)
    return subset

def main():
    ensure_dir(OUTPUT_DIR)
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and not f.endswith("-dns.csv")] if os.path.isdir(DATA_DIR) else []
    print(f"[*] Fichiers CSV comportementaux détectés : {len(all_files)}")
    for f in all_files:
        print(f"    - {f}")

    print("\n[*] Chargement TRAIN (split officiel)...")
    df_train = load_csvs(TRAIN_FILES)

    # Réintégration des fichiers 2021may au train (hors DNS)
    extra_files = [f for f in all_files if f.startswith("labelled_2021may") and not f.endswith("-dns.csv")]
    if USE_2021MAY_AS_DEVELOPMENT and extra_files:
        print(f"[*] Chargement EXTRA ({len(extra_files)} fichiers 2021may) pour enrichissement...")
        df_extra = load_csvs(extra_files)
        df_train = pd.concat([df_train, df_extra], ignore_index=True)
        print(f"[+] Train enrichi : {len(df_train):,} lignes")
    elif extra_files:
        print("[*] EXTRA 2021may detecte mais non utilise (USE_2021MAY_AS_DEVELOPMENT=False).")

    print("[*] Chargement VAL (officiel)...")
    df_val = load_csvs(VAL_FILES)
    print("[*] Chargement TEST (officiel)...")
    df_test = load_csvs(TEST_FILES)

    all_eids = set()
    for df in (df_train, df_val, df_test):
        if not df.empty and "eventId" in df.columns:
            all_eids.update(df["eventId"].dropna().unique().astype(int))
    print(f"[*] EventIds uniques observés : {len(all_eids)}")

    if USE_HEURISTIC_MITRE_FEATURES:
        from generate_mitre_map import generate_mitre_map
        mitre_map = generate_mitre_map(all_eids)
        save_json(mitre_map, MITRE_MAP_PATH)
        print(f"[+] Mapping MITRE sauvegarde ({len(mitre_map)} eventIds mappes).")
    else:
        mitre_map = {}
        print("[*] Features MITRE heuristiques desactivees pour les resultats principaux.")

    # Construction du train enrichi complet
    print("\n[*] Construction TRAIN enrichi complet...")
    full_train_data = build_features([df_train], mitre_map, fit_vocab=True)
    y_full = full_train_data["y"]

    # Triple split stratifié : 65% train / 20% cal / 15% val_strat
    print("[*] Triple split stratifié depuis le train enrichi...")
    train_idx, temp_idx = train_test_split(
        np.arange(len(y_full)), test_size=0.35, stratify=y_full, random_state=42
    )
    cal_idx, val_strat_idx = train_test_split(
        temp_idx, test_size=0.429, stratify=y_full[temp_idx], random_state=43
    )
    # 0.429 * 0.35 ≈ 0.15 ; 0.571 * 0.35 ≈ 0.20

    train_data = extract_subset(full_train_data, train_idx)
    cal_data   = extract_subset(full_train_data, cal_idx)
    val_strat_data = extract_subset(full_train_data, val_strat_idx)

    print(f"[+] train_data   : n={len(train_data['y']):,} (evil={train_data['y'].sum():,}, ratio={train_data['y'].mean():.4f})")
    print(f"[+] cal_data     : n={len(cal_data['y']):,} (evil={cal_data['y'].sum():,}, ratio={cal_data['y'].mean():.4f})")
    print(f"[+] val_strat_data : n={len(val_strat_data['y']):,} (evil={val_strat_data['y'].sum():,}, ratio={val_strat_data['y'].mean():.4f})")

    print("[*] Construction VAL (officiel)...")
    val_data = build_features([df_val], mitre_map, fit_vocab=False,
                              vocab_event_ids=train_data["vocab_event_ids"],
                              vocab_techniques=train_data["vocab_techniques"])
    print("[*] Construction TEST (officiel)...")
    test_data = build_features([df_test], mitre_map, fit_vocab=False,
                               vocab_event_ids=train_data["vocab_event_ids"],
                               vocab_techniques=train_data["vocab_techniques"])

    datasets = [
        ("train", train_data),
        ("cal", cal_data),
        ("val_strat", val_strat_data),
        ("val", val_data),
        ("test", test_data)
    ]

    for name, data in datasets:
        path = os.path.join(OUTPUT_DIR, f"{name}_data.pkl")
        save_pickle_atomic(data, path)
        evil_rate = data["y"].mean()
        print(f"[+] {name:10s} -> {path}  (n={len(data['y']):,}, evil={data['y'].sum():,}, ratio={evil_rate:.4f})")

    print("\n=== STATISTIQUES ===")
    for name, data in datasets:
        y = data["y"]
        print(f"{name:10s} : total={len(y):,} | bénin={np.sum(y==0):,} | malveillant={np.sum(y==1):,} | taux={np.mean(y):.4f}")

if __name__ == "__main__":
    main()
