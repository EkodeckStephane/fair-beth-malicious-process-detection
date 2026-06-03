#!/usr/bin/env python3
"""Run the full experimental pipeline."""
import os
import subprocess
import sys

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    RUN_HEURISTIC_TACTIC_DETECTOR,
    USE_HEURISTIC_MITRE_FEATURES,
)
from utils import ensure_dir


def check_data():
    if not os.path.isdir(DATA_DIR):
        print(f"[!] Data directory not found: {DATA_DIR}")
        sys.exit(1)
    csvs = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csvs:
        print(f"[!] No CSV files found in {DATA_DIR}")
        sys.exit(1)
    print(f"[+] {len(csvs)} CSV file(s) detected in {DATA_DIR}")


def run_step(name, script):
    print(f"\n{'=' * 60}")
    print(f"STEP: {name}")
    print(f"{'=' * 60}")
    if not os.path.exists(script):
        print(f"[!] Missing script: {script}")
        sys.exit(1)
    ret = subprocess.run([sys.executable, script])
    if ret.returncode != 0:
        print(f"[!] Step failed: {name} (exit code {ret.returncode})")
        sys.exit(ret.returncode)


def main():
    ensure_dir(OUTPUT_DIR)
    check_data()
    run_step("System information", "system_info.py")

    steps = []
    if USE_HEURISTIC_MITRE_FEATURES:
        steps.append(("MITRE mapping generation", "generate_mitre_map.py"))

    steps.extend([
        ("Preprocessing and feature engineering", "preprocessing_v2.py"),
        ("Detector A: Isolation Forest", "detector_tabular.py"),
        ("Detector B: GRU autoencoder", "detector_sequence.py"),
    ])

    if RUN_HEURISTIC_TACTIC_DETECTOR:
        steps.append(("Exploratory detector C: GRU-MITRE", "detector_tactic.py"))

    steps.extend([
        ("Calibration and ablation fusion", "calibration_fusion.py"),
        ("Legacy baseline RF export", "baseline_rf.py"),
        ("Evaluation and figures", "evaluate.py"),
    ])

    for name, script in steps:
        run_step(name, script)

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETED")
    print(f"{'=' * 60}")
    print(f"Outputs: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
