"""
Configuration globale du pipeline LockBit 2026 – BETH Dataset.
Toutes les constantes sont soit des paramètres utilisateur ajustables,
soit des placeholders pour valeurs calculées à l'exécution.
"""
import os
import torch

# Reproductibilité
SEED = 42

# Device : CPU par défaut pour portabilité ; GPU utilisé si disponible
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Chemins
DATA_DIR = os.environ.get("BETH_DATA_DIR", "./BETH_Dataset")          # L'utilisateur place ici les CSV BETH
OUTPUT_DIR = os.environ.get("FAIR_BETH_PIPELINE_OUTPUT", "./pipeline_output")     # Résultats intermédiaires et finaux
MITRE_MAP_PATH = os.path.join(OUTPUT_DIR, "mitre_map.json")

# Fichiers officiels (splits) — on utilise STRICTEMENT ces fichiers,
# sans concaténer les fichiers 2021may qui introduisent un dataset shift.
TRAIN_FILES = ["labelled_training_data.csv"]
VAL_FILES   = ["labelled_validation_data.csv"]
TEST_FILES  = ["labelled_testing_data.csv"]

# Defensible experimental protocol.
# The official train/validation splits contain no evil process after process
# aggregation. The 2021may files are therefore used only to build an enriched
# development set; the official test split remains isolated.
USE_2021MAY_AS_DEVELOPMENT = True

# EventId-to-MITRE mappings in BETH are heuristic. Keep them out of the main
# results unless explicitly enabled for exploratory analysis.
USE_HEURISTIC_MITRE_FEATURES = False
RUN_HEURISTIC_TACTIC_DETECTOR = False

# Agrégation : un "processus" est défini par hostName + processId
GROUP_COLS = ["hostName", "processId"]

# Séquences : longueur fixée au 95e percentile des longueurs observées sur le train
SEQ_LEN_PERCENTILE = 95

# Hyperparamètres des modèles légers (adaptés à un Quadro P2000 4GB ou CPU)
EMBED_DIM = 32
HIDDEN_DIM = 64
NUM_LAYERS = 1
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
EPOCHS = 30
PATIENCE = 5          # Early stopping
DROPOUT = 0.2

# Fusion & Calibration
ETA = 1e-6            # Lissage ECE pour éviter division par zéro
N_BINS_ECE = 10
FPR_BUDGET = 0.05     # Budget faux-positifs pour le seuil adaptatif
BOOTSTRAP_N = 1000
CALIBRATION_SPLIT = 0.20  # Fraction du train utilisée pour le calibration set (stratifié)

# Isolation Forest
IF_N_ESTIMATORS = 200
