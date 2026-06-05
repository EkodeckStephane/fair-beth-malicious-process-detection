# FAIR-BETH: Fair Thresholding and Ablation Protocol for Behavioral Malicious-Process Detection

This repository contains the reproducibility package for:

**FAIR-BETH: A Fair Thresholding and Ablation Protocol for Behavioral Malicious-Process Detection under Distribution Shift**

The contribution is a reusable evaluation contract for behavioral security datasets under distribution shift. The code evaluates BETH process-level malicious-process detection, external protocol validation on MalBehavD-V1, and additional audits for feature attribution, prefix detection, cost sensitivity, feature-space stress testing, and split robustness.

## Repository Contents

- `scripts/`: preprocessing, model training, evaluation, external validation, and audit scripts.
- `results/beth_limit_lifting/`: derived BETH audit CSV tables.
- `results/external_malbehavd/`: derived MalBehavD-V1 validation CSV tables.
- `figures/`: generated figures used by the paper.
- `paper/`: LaTeX source, bibliography, figures, and compiled PDF snapshot.

## Data

Raw datasets are not redistributed in this repository.

- BETH must be obtained from its original public source and placed under `scripts/BETH_Dataset/` or the path configured in `scripts/config.py`.
- MalBehavD-V1 must be obtained from its public GitHub repository: `https://github.com/mpasco/MalbehavD-V1`.

The repository includes only derived metrics, plots, and reproducibility scripts.

## Main Scripts

- `scripts/run_pipeline.py`: runs the main BETH pipeline.
- `scripts/evaluate.py`: computes BETH metrics, thresholds, confidence intervals, and figures.
- `scripts/external_malbehavd_validation.py`: runs external MalBehavD-V1 validation.
- `scripts/beth_limit_lifting_analyses.py`: runs attribution, prefix, cost, stress, and split-robustness audits on BETH.
- `scripts/tabular_sota_and_calibration_audit.py`: runs additional tabular baselines, including XGBoost when available, and writes calibration audit outputs.
- `scripts/robust_threshold_validation.py`: selects 50 cross-fitted RF-500 thresholds over all BETH development positives and performs one locked-threshold test evaluation.
- `scripts/sequence_capacity_ablation.py`: compares target-copy-free GRU and LSTM next-event predictors across recurrent capacities.

## Reproducibility Notes

The experiments were run on Windows 11 with Python 3.13.5, CPU-only PyTorch, scikit-learn, pandas, NumPy, matplotlib, seaborn, tqdm, and psutil. Exact package versions used in the paper are reported in the manuscript.

Typical setup:

```powershell
python -m pip install -r requirements.txt
```

Run the main BETH pipeline from the `scripts/` directory after placing BETH CSV files in `scripts/BETH_Dataset/`, or set `BETH_DATA_DIR` to the dataset location:

```powershell
cd scripts
$env:BETH_DATA_DIR="C:\path\to\BETH_Dataset"
python run_pipeline.py
```

Run the additional BETH audits from the repository root after the main pipeline has produced `pipeline_output/`, or point the script to an existing output directory:

```powershell
$env:BETH_DATA_DIR="C:\path\to\BETH_Dataset"
$env:FAIR_BETH_PIPELINE_OUTPUT="C:\path\to\pipeline_output"
python scripts/beth_limit_lifting_analyses.py
```

Run the additional tabular baseline and calibration audit:

```powershell
$env:FAIR_BETH_PIPELINE_OUTPUT="C:\path\to\pipeline_output"
$env:FAIR_BETH_AUDIT_OUTPUT="C:\path\to\additional_audits"
python scripts/tabular_sota_and_calibration_audit.py
```

Run the repeated cross-fitted threshold audit:

```powershell
$env:FAIR_BETH_PIPELINE_OUTPUT="C:\path\to\pipeline_output"
python scripts/robust_threshold_validation.py
```

Run the target-copy-free sequence capacity audit:

```powershell
$env:FAIR_BETH_PIPELINE_OUTPUT="C:\path\to\pipeline_output"
python scripts/sequence_capacity_ablation.py
```

Run the external MalBehavD-V1 validation after downloading the public CSV and adjusting the path if needed:

```powershell
$env:MALBEHAVD_CSV="C:\path\to\MalBehavD-V1-dataset.csv"
python scripts/external_malbehavd_validation.py
```

## Citation

If you use this repository, cite the accompanying paper and the original datasets.

## License

Code in this repository is released under the MIT License. Dataset licenses and access terms remain those of the original dataset providers.
