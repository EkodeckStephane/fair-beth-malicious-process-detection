# FAIR-X / FAIR-BETH: Reproducibility Package

[![GitHub repository](https://img.shields.io/badge/repo-EkodeckStephane/fair--beth--malicious--process--detection-blue)](https://github.com/EkodeckStephane/fair-beth-malicious-process-detection)

This repository contains the reproducibility package for the IEEE TDSC submission:

**FAIR-X: A Threshold-Transfer Evaluation Contract for Behavioral Malware Detection under Distribution Shift**  
(Previous working title: *FAIR-BETH: A Fair Thresholding and Ablation Protocol for Behavioral Malicious-Process Detection under Distribution Shift*)

The contribution is a reusable, executable evaluation contract (FAIR-X) for behavioral security datasets under distribution shift. FAIR-BETH is the instantiation of FAIR-X on the public BETH dataset. The repository evaluates BETH process-level malicious-process detection, performs external protocol validation on MalBehavD-V1, and provides additional audits for feature attribution, prefix detection, cost sensitivity, feature-space stress testing, and split robustness.

---

## Repository URL

**https://github.com/EkodeckStephane/fair-beth-malicious-process-detection**

---

## What is new in this revision

- **General evaluation contract (FAIR-X).** The manuscript now introduces FAIR-X as a general methodology before instantiating it as FAIR-BETH on BETH and MalBehavD-V1.
- **Corrected RF-500 baseline.** The `RF-500` model now actually uses 500 trees. All tables, prose, and CSV artifacts have been regenerated and synchronized.
- **Aligned threshold policy.** Every BETH and MalBehavD-V1 script now selects the same most-permissive threshold satisfying a 5% validation-FPR budget, with clarifying comments in the code.
- **Expanded audits.** Repeated cross-fitted thresholds, target-copy-free GRU/LSTM ablations, calibration audit, prefix evaluation, cost sensitivity, feature-space stress tests, and host/temporal split robustness.
- **Explicit claim boundaries.** The work is framed as an evaluation protocol; it does not claim ransomware-family attribution, calibrated posterior risk, direct cross-dataset model transfer, or autonomous containment readiness.

---

## Key results (after revision)

On the official BETH process-level test split under the common validation-FPR policy:

| Model | AUC | AP | Precision | Recall | F1 | FP | FN |
|-------|-----|----|-----------|--------|----|----|----|
| TabRF | 0.7360 | 0.8270 | 0.7500 | 0.6942 | **0.7210** | 28 | 37 |
| RF-500 | 0.7490 | 0.8352 | 0.7431 | 0.6694 | 0.7043 | 28 | 40 |
| MetaRF | 0.7113 | 0.8105 | 0.7308 | 0.6281 | 0.6756 | 28 | 45 |

- TabRF has the highest F1 point estimate, but its confidence interval overlaps RF-500's interval, and MetaRF's interval overlaps TabRF's.
- Repeated cross-fitted threshold audit: median locked threshold 0.006 yields F1 0.6964 but expands the 5% development-FPR target to 32.47% test FPR.
- MalBehavD-V1 external validation: repeated-split median F1 0.9533 under the same FAIR-X contract.

All numbers are reproduced from the CSV/JSON artifacts in `results/`.

---

## Repository contents

- `scripts/` — preprocessing, model training, evaluation, external validation, and audit scripts.
  - `scripts/run_pipeline.py`
  - `scripts/evaluate.py`
  - `scripts/external_malbehavd_validation.py`
  - `scripts/beth_limit_lifting_analyses.py`
  - `scripts/tabular_sota_and_calibration_audit.py`
  - `scripts/robust_threshold_validation.py`
  - `scripts/sequence_capacity_ablation.py`
  - ... and supporting modules.
- `results/beth_limit_lifting/` — derived BETH audit CSV tables (permutation importance, prefix, cost, stress, group/temporal robustness).
- `results/beth_additional_audits/` — tabular baseline comparison and calibration audit CSVs.
- `results/external_malbehavd/` — derived MalBehavD-V1 validation CSV tables.
- `results/robust_threshold_validation/` — cross-fitted thresholds and locked-threshold sensitivity CSV.
- `results/sequence_capacity_ablation/` — target-copy-free recurrent ablation CSV.
- `figures/` — generated figures used by the paper and supplement.
- `paper/` — LaTeX source, compiled PDF snapshots, bibliography, and figures for the main manuscript, supplement, and cover letter.

---

## Data

Raw datasets are **not** redistributed in this repository.

- **BETH** must be obtained from its original public source. Place it under `scripts/BETH_Dataset/` or set the environment variable `BETH_DATA_DIR`.
- **MalBehavD-V1** must be obtained from its public repository: https://github.com/mpasco/MalbehavD-V1. Set `MALBEHAVD_CSV` to point to the downloaded CSV.

The repository includes only derived metrics, plots, and reproducibility scripts.

---

## Environment

The experiments were run on **Windows 11** with:

- Python 3.13.5
- CPU-only PyTorch 2.8.0
- scikit-learn 1.7.2
- pandas 2.2.3
- NumPy 2.2.6
- matplotlib 3.10.5
- XGBoost 3.2.0
- and the packages listed in `requirements.txt`.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

The code is written for cross-platform execution; paths are handled with `pathlib` or environment variables, and scripts can be run from either `bash` (Git Bash / WSL) or PowerShell.

---

## Quick start

### 1. Prepare BETH

Place the BETH CSV files in `scripts/BETH_Dataset/` or set:

```powershell
$env:BETH_DATA_DIR="C:\path\to\BETH_Dataset"
```

### 2. Run the main BETH pipeline

```powershell
cd scripts
python run_pipeline.py
```

This creates `pipeline_output/` with model artifacts, scores, and the main results table.

### 3. Run the additional BETH audits

```powershell
cd ..
$env:BETH_DATA_DIR="C:\path\to\BETH_Dataset"
$env:FAIR_BETH_PIPELINE_OUTPUT="C:\path\to\pipeline_output"
python scripts/beth_limit_lifting_analyses.py
```

### 4. Run the tabular baseline and calibration audit

```powershell
$env:FAIR_BETH_PIPELINE_OUTPUT="C:\path\to\pipeline_output"
$env:FAIR_BETH_AUDIT_OUTPUT="C:\path\to\additional_audits"
python scripts/tabular_sota_and_calibration_audit.py
```

### 5. Run the repeated cross-fitted threshold audit

```powershell
$env:FAIR_BETH_PIPELINE_OUTPUT="C:\path\to\pipeline_output"
python scripts/robust_threshold_validation.py
```

### 6. Run the target-copy-free sequence capacity audit

```powershell
$env:FAIR_BETH_PIPELINE_OUTPUT="C:\path\to\pipeline_output"
python scripts/sequence_capacity_ablation.py
```

### 7. Run the external MalBehavD-V1 validation

```powershell
$env:MALBEHAVD_CSV="C:\path\to\MalBehavD-V1-dataset.csv"
python scripts/external_malbehavd_validation.py
```

---

## Building the paper

The LaTeX sources are in `paper/`:

```powershell
cd paper
pdflatex fair_beth_tdsc_v2.tex
bibtex fair_beth_tdsc_v2
pdflatex fair_beth_tdsc_v2.tex
pdflatex fair_beth_tdsc_v2.tex
pdflatex fair_beth_tdsc_v2_supplement.tex
pdflatex cover_letter_fair_beth_tdsc_v2.tex
```

The `paper/` directory already contains compiled PDF snapshots.

---

## Authoritative numerical sources

The generated CSV files are the authoritative numerical sources for the paper:

- `results/beth_additional_audits/tabular_sota_comparison.csv`
- `results/beth_additional_audits/calibration_audit.csv`
- `results/beth_limit_lifting/*.csv`
- `results/external_malbehavd/*.csv`
- `results/robust_threshold_validation/*.csv`
- `results/sequence_capacity_ablation/*.csv`

These are the files used to produce the tables and figures in the revised manuscript.

---

## Citation

If you use this repository, please cite the accompanying paper and the original BETH and MalBehavD-V1 datasets.

---

## License

Code in this repository is released under the MIT License. Dataset licenses and access terms remain those of the original dataset providers.
