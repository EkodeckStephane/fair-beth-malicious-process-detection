"""
Global configuration for the FAIR-BETH behavioral-malware evaluation pipeline.

Constants below are either user-adjustable execution parameters or values
consumed by the reproducible experimental scripts. Reviewer-driven matched
model selection and uncertainty analyses are implemented in their dedicated
revision scripts rather than encoded as hidden configuration choices here.
"""
import os
import torch

# Reproducibility
SEED = 42

# CPU by default for portability; CUDA is used when available.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
DATA_DIR = os.environ.get("BETH_DATA_DIR", "./BETH_Dataset")
OUTPUT_DIR = os.environ.get("FAIR_BETH_PIPELINE_OUTPUT", "./pipeline_output")
MITRE_MAP_PATH = os.path.join(OUTPUT_DIR, "mitre_map.json")

# Official BETH splits. The official test split is kept isolated. Supplementary
# 2021may process files are used only for the enriched development construction
# documented by the revision audits.
TRAIN_FILES = ["labelled_training_data.csv"]
VAL_FILES   = ["labelled_validation_data.csv"]
TEST_FILES  = ["labelled_testing_data.csv"]
USE_2021MAY_AS_DEVELOPMENT = True

# EventId-to-MITRE mappings in BETH are heuristic. Keep them out of the main
# results unless explicitly enabled for exploratory analysis.
USE_HEURISTIC_MITRE_FEATURES = False
RUN_HEURISTIC_TACTIC_DETECTOR = False

# Process-level experimental unit.
GROUP_COLS = ["hostName", "processId"]

# Sequence width is derived from the training distribution only.
SEQ_LEN_PERCENTILE = 95

# Lightweight detector parameters used by the base pipeline.
EMBED_DIM = 32
HIDDEN_DIM = 64
NUM_LAYERS = 1
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
EPOCHS = 30
PATIENCE = 5
DROPOUT = 0.2

# Fusion, calibration, and thresholding.
ETA = 1e-6
N_BINS_ECE = 10
FPR_BUDGET = 0.05
BOOTSTRAP_N = 1000
CALIBRATION_SPLIT = 0.20

# Isolation Forest
IF_N_ESTIMATORS = 200
