# FAIR-X / FAIR-BETH Reproducibility Package

[![GitHub repository](https://img.shields.io/badge/repo-EkodeckStephane/fair--beth--malicious--process--detection-blue)](https://github.com/EkodeckStephane/fair-beth-malicious-process-detection)

This repository contains the code, derived results, figures, manuscript sources, and integrity manifest for the FAIR-X / FAIR-BETH behavioral malware evaluation study.

Paper title:

**FAIR-X: A Threshold-Transfer Evaluation Contract for Behavioral Malware Detection under Distribution Shift**

FAIR-X is an evaluation contract for behavioral security datasets under distribution shift. FAIR-BETH is the process-level BETH instantiation. The repository evaluates fixed alerting decisions, ranking metrics, threshold transfer, calibration, baseline parity, sequence modeling, external protocol portability on MalBehavD-V1, and BETH deployment-boundary audits.

Repository URL:

`https://github.com/EkodeckStephane/fair-beth-malicious-process-detection`

## What Was Done

The study converts behavioral malware evaluation into a threshold-transfer problem. A model produces a score; a development procedure fixes an operating threshold; the final claim concerns the locked decision rule on an isolated test distribution.

The repository implements:

- FAIR-X contract checks: label exclusion, test-label isolation, development-only feature and vocabulary fitting, common threshold policies, baseline parity, uncertainty reporting, and claim-boundary audits.
- BETH process aggregation by `(hostName, processId)` with process labels derived from `evil`.
- Main BETH models: Isolation Forest with Platt scaling, GRU autoencoder with Platt scaling, ScoreRF, TabRF, RF-500, and MetaRF.
- Common BETH threshold policies: validation-FPR budget at 5 percent and validation-max-F1.
- Strong tabular baseline audit: ExtraTrees, RF-500, XGBoost, gradient boosting, histogram gradient boosting, logistic regression, MLP, and ScoreRF under the same validation-FPR policy.
- Paired uncertainty audit: paired bootstrap deltas and exact McNemar tests for TabRF, RF-500, and MetaRF.
- Repeated threshold-transfer audit: 50 development-only RF-500 thresholds from all 29 BETH development positives.
- Target-copy-free recurrent audit: GRU/LSTM next-event prediction with larger recurrent capacity.
- Calibration audit: ECE, Brier score, and reliability diagrams.
- BETH limit-lifting audits: permutation attribution, prefix detection, cost sensitivity, feature-space stress tests, host-disjoint validation, and chronological split robustness.
- External MalBehavD-V1 validation: stratified i.i.d. protocol-portability test plus metadata-dependent PE-creation-date chronological audit.

## Main Claim Boundaries

The strongest supported claim is methodological: threshold policy, baseline strength, and diagnostic gaps determine whether a behavioral malware detector supports a deployable decision claim.

The BETH empirical claim is narrower: under the common validation-FPR policy, direct tabular learning has the strongest BETH thresholded operating point among the evaluated systems. MetaRF remains below TabRF and RF-500, and repeated thresholding confirms that the development FPR objective expands under the official BETH shift.

Family-specific ransomware attribution, direct BETH-to-MalBehavD model transfer, calibrated posterior-risk interpretation, real-time containment, and enterprise deployment require separate evidence. The repository includes audits that state those boundaries quantitatively.

## Key BETH Results

Official BETH process-level test split, validation-FPR budget policy:

| Model | AUC | AP | Precision | Recall | F1 | F1 95% CI | FP | FN |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| IF-Platt | 0.3496 | 0.5195 | 0.2500 | 0.0083 | 0.0160 | [0.0000, 0.0476] | 3 | 120 |
| GRUAE-Platt | 0.3914 | 0.5347 | 0.5000 | 0.0413 | 0.0763 | [0.0161, 0.1452] | 5 | 116 |
| SimpleAvg | 0.3403 | 0.5089 | 0.2778 | 0.0413 | 0.0719 | [0.0150, 0.1379] | 13 | 116 |
| ScoreRF | 0.5931 | 0.6843 | 0.6769 | 0.3636 | 0.4731 | [0.3825, 0.5650] | 21 | 77 |
| TabRF | 0.7360 | 0.8270 | 0.7500 | 0.6942 | 0.7210 | [0.6484, 0.7849] | 28 | 37 |
| RF-500 | 0.7490 | 0.8352 | 0.7431 | 0.6694 | 0.7043 | [0.6372, 0.7687] | 28 | 40 |
| MetaRF | 0.7113 | 0.8105 | 0.7308 | 0.6281 | 0.6756 | [0.5990, 0.7456] | 28 | 45 |

Validation-max-F1 sensitivity on the same official BETH test split:

| Model | Precision | Recall | F1 | F1 95% CI |
|---|---:|---:|---:|---|
| IF-Platt | 0.4590 | 0.2314 | 0.3077 | [0.2128, 0.3895] |
| GRUAE-Platt | 0.5638 | 0.4380 | 0.4930 | [0.4095, 0.5702] |
| SimpleAvg | 0.5060 | 0.3471 | 0.4118 | [0.3179, 0.5022] |
| ScoreRF | 0.6852 | 0.3058 | 0.4229 | [0.3250, 0.5165] |
| TabRF | 0.9583 | 0.3802 | 0.5444 | [0.4512, 0.6304] |
| MetaRF | 0.9474 | 0.2975 | 0.4528 | [0.3544, 0.5476] |

## Stronger Tabular Baseline Audit

Official BETH test split, common validation-FPR policy:

| Model | AUC | AP | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabRF | 0.7360 | 0.8270 | 0.7500 | 0.6942 | 0.7210 | 28 | 37 |
| RF-500 | 0.7490 | 0.8352 | 0.7431 | 0.6694 | 0.7043 | 28 | 40 |
| MetaRF | 0.7113 | 0.8105 | 0.7308 | 0.6281 | 0.6756 | 28 | 45 |
| ExtraTrees | 0.7973 | 0.8572 | 0.9649 | 0.4545 | 0.6180 | 2 | 66 |
| HistGradientBoosting | 0.7106 | 0.8090 | 0.7033 | 0.5289 | 0.6038 | 27 | 57 |
| XGBoost | 0.6717 | 0.7841 | 0.7176 | 0.5041 | 0.5922 | 24 | 60 |
| MLP | 0.7865 | 0.8339 | 0.9231 | 0.3967 | 0.5549 | 4 | 73 |
| GradientBoosting | 0.6153 | 0.7552 | 0.6571 | 0.3802 | 0.4817 | 24 | 75 |
| ScoreRF | 0.5931 | 0.6843 | 0.6769 | 0.3636 | 0.4731 | 21 | 77 |
| LogReg | 0.7698 | 0.8024 | 0.7059 | 0.0992 | 0.1739 | 5 | 109 |

Paired comparison audit:

| Comparison | Delta F1 [95% CI] | Delta AUC [95% CI] | Delta AP [95% CI] | McNemar p |
|---|---:|---:|---:|---:|
| TabRF - MetaRF | +0.0455 [0.0027, 0.0908] | +0.0246 [-0.0045, 0.0546] | +0.0166 [0.0024, 0.0318] | 0.1516 |
| TabRF - RF-500 | +0.0167 [-0.0138, 0.0484] | -0.0130 [-0.0317, 0.0051] | -0.0081 [-0.0174, 0.0004] | 0.5488 |
| RF-500 - MetaRF | +0.0288 [-0.0092, 0.0680] | +0.0376 [0.0127, 0.0635] | +0.0247 [0.0120, 0.0390] | 0.3593 |

## Threshold Transfer, Calibration, and Sequence Audits

Repeated BETH threshold-transfer audit:

| Quantity | Result |
|---|---:|
| Development processes | 5,819 |
| Development positives | 29 |
| Repeated folds | 50 |
| Positives per held-out fold | 5-6 |
| Threshold median (IQR) | 0.006 (0.004-0.006) |
| Locked-test precision | 0.7573 |
| Locked-test recall | 0.6446 |
| Locked-test F1 | 0.6964 |
| Locked-test FPR | 0.3247 |

Target-copy-free sequence capacity audit:

| Model | Parameters | AUC | AP | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GRU-64x1 | 23,666 | 0.4238 | 0.5928 | 0.6667 | 0.0826 | 0.1471 | 5 | 111 |
| GRU-128x2 | 183,218 | 0.4987 | 0.6409 | 0.8182 | 0.0744 | 0.1364 | 2 | 112 |
| LSTM-128x2 | 241,074 | 0.4775 | 0.6323 | 0.7500 | 0.0744 | 0.1353 | 3 | 112 |

Calibration audit highlights:

| Model | ECE | Brier |
|---|---:|---:|
| LogReg | 0.2905 | 0.2734 |
| ExtraTrees | 0.3932 | 0.3498 |
| RF-500 | 0.4994 | 0.4481 |
| TabRF | 0.4984 | 0.4506 |
| MetaRF | 0.5007 | 0.4621 |

## BETH Limit-Lifting Audits

Prefix evaluation under validation-FPR thresholding:

| Prefix | AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 10 events | 0.4443 | 0.6250 | 0.0413 | 0.0775 |
| 25 events | 0.5553 | 0.9000 | 0.2231 | 0.3576 |
| 50 events | 0.6678 | 0.8500 | 0.2810 | 0.4224 |
| 100 events | 0.6414 | 0.7857 | 0.3636 | 0.4972 |
| 250 events | 0.6572 | 0.9020 | 0.3802 | 0.5349 |
| Full trace | 0.7360 | 0.7500 | 0.6942 | 0.7210 |

Split-robustness audit:

| Split audit | AUC | Recall | F1 |
|---|---:|---:|---:|
| Main process split | 0.7360 | 0.6942 | 0.7210 |
| Host-disjoint median | 0.5041 | 0.1860 | 0.2739 |
| Host-disjoint range | 0.4095-0.8300 | 0.0992-0.5702 | 0.1678-0.7188 |
| Temporal 70/15/15 | 0.6865 | 0.2893 | 0.4294 |

Cost-sensitivity audit:

| FN:FP cost ratio | Oracle threshold | FP | FN | F1 |
|---|---:|---:|---:|---:|
| 1:1 | 0.0274 | 21 | 42 | 0.7149 |
| 5:1 | 0.0000 | 77 | 0 | 0.7586 |
| 10:1 | 0.0000 | 77 | 0 | 0.7586 |
| 25:1 | 0.0000 | 77 | 0 | 0.7586 |
| 50:1 | 0.0000 | 77 | 0 | 0.7586 |

These oracle cost rows expose operating-point sensitivity. Deployable response policies require validation-time cost selection, wall-clock latency evidence, rollback safety, and analyst-workload evaluation.

## External MalBehavD-V1 Results

Stratified i.i.d. fixed split:

| Threshold policy | AUC | AP | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation-FPR budget | 0.9902 | 0.9922 | 0.9439 | 0.9585 | 0.9512 | 11 | 8 |
| Validation-max-F1 | 0.9902 | 0.9922 | 0.9735 | 0.9534 | 0.9634 | 5 | 9 |
| Oracle test max-F1 | 0.9902 | 0.9922 | 0.9737 | 0.9585 | 0.9661 | 5 | 8 |

Thirty repeated stratified splits under the validation-FPR policy:

| Metric | Mean | Median | IQR |
|---|---:|---:|---:|
| F1 | 0.9530 | 0.9533 | [0.9429, 0.9634] |
| AUC | 0.9869 | - | - |
| AP | 0.9902 | - | - |

Creation-date chronological audit over 2,273 timestamp-valid samples:

| Threshold policy | AUC | AP | Precision | Recall | F1 | FP | FN | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Locked median threshold | 0.9406 | 0.9384 | 0.9510 | 0.7391 | 0.8318 | 7 | 48 | 0.0259 |
| Oracle test max-F1 | 0.9406 | 0.9384 | 0.9085 | 0.8098 | 0.8563 | 15 | 35 | 0.0556 |

MalBehavD-V1 prefix evaluation:

| API prefix | AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 10 calls | 0.9672 | 0.9494 | 0.8756 | 0.9111 |
| 25 calls | 0.9813 | 0.9775 | 0.9016 | 0.9380 |
| 50 calls | 0.9877 | 0.9780 | 0.9223 | 0.9493 |
| 100 calls | 0.9905 | 0.9538 | 0.9637 | 0.9588 |
| Full sequence | 0.9902 | 0.9439 | 0.9585 | 0.9512 |

The MalBehavD-V1 audit tests FAIR-X protocol execution on a second behavioral schema. The creation-date split is metadata-dependent because PE creation timestamps can reflect build time and can be forged; family-disjoint validation and enterprise telemetry validation are separate evidence targets.

## Repository Layout

```text
.
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- ARTIFACT_MANIFEST.md
|-- scripts/
|   |-- config.py
|   |-- preprocessing_v2.py
|   |-- run_pipeline.py
|   |-- detector_tabular.py
|   |-- detector_sequence.py
|   |-- calibration_fusion.py
|   |-- evaluate.py
|   |-- tabular_sota_and_calibration_audit.py
|   |-- paired_comparison_audit.py
|   |-- robust_threshold_validation.py
|   |-- sequence_capacity_ablation.py
|   |-- beth_limit_lifting_analyses.py
|   |-- external_malbehavd_validation.py
|   `-- malbehavd_temporal_audit_full.py
|-- results/
|   |-- beth_additional_audits/
|   |-- beth_limit_lifting/
|   |-- external_malbehavd/
|   |-- external_malbehavd_temporal/
|   |-- robust_threshold_validation/
|   `-- sequence_capacity_ablation/
|-- figures/
`-- paper/
```

## Authoritative Artifact Files

Main BETH pipeline output is generated by `scripts/run_pipeline.py` as `scripts/pipeline_output/` by default, or at `FAIR_BETH_PIPELINE_OUTPUT` when that environment variable is set. The revised manuscript tables were regenerated from the local pipeline output and the derived audit files listed below.

Included derived numerical sources:

- `results/beth_additional_audits/tabular_sota_comparison.csv`
- `results/beth_additional_audits/calibration_audit.csv`
- `results/beth_additional_audits/paired_comparison_audit.csv`
- `results/beth_additional_audits/paired_comparison_audit.json`
- `results/beth_limit_lifting/beth_tabrf_permutation_importance.csv`
- `results/beth_limit_lifting/beth_prefix_results.csv`
- `results/beth_limit_lifting/beth_cost_optimal_thresholds.csv`
- `results/beth_limit_lifting/beth_cost_curve_all_thresholds.csv`
- `results/beth_limit_lifting/beth_evasion_stress_tests.csv`
- `results/beth_limit_lifting/beth_group_temporal_robustness.csv`
- `results/beth_limit_lifting/beth_host_disjoint_robustness_summary.csv`
- `results/beth_limit_lifting/beth_temporal_70_15_15_result.csv`
- `results/external_malbehavd/external_results_table.csv`
- `results/external_malbehavd/repeated_split_results.csv`
- `results/external_malbehavd/repeated_split_summary.csv`
- `results/external_malbehavd/prefix_results.csv`
- `results/external_malbehavd/feature_importance.csv`
- `results/external_malbehavd_temporal/summary.json`
- `results/external_malbehavd_temporal/cross_fitted_thresholds.csv`
- `results/external_malbehavd_temporal/locked_threshold_sensitivity.csv`
- `results/robust_threshold_validation/summary.json`
- `results/robust_threshold_validation/cross_fitted_thresholds.csv`
- `results/robust_threshold_validation/locked_threshold_test_sensitivity.csv`
- `results/sequence_capacity_ablation/sequence_capacity_ablation.csv`
- `results/sequence_capacity_ablation/summary.json`

Included generated figures:

- `figures/beth_tabrf_permutation_importance.png`
- `figures/beth_prefix_curve.png`
- `figures/beth_cost_sensitivity.png`
- `figures/beth_evasion_stress_tests.png`
- `figures/malbehavd_feature_importance.png`
- `figures/malbehavd_prefix_curve.png`
- `figures/malbehavd_pr_curve.png`
- `figures/reliability_diagrams.png`

Use `ARTIFACT_MANIFEST.md` to verify SHA-256 hashes for the submitted snapshot.

## Data Requirements

Raw datasets and raw VirusTotal metadata remain with their original providers or access channels.

BETH:

- Obtain the public BETH CSV files from the original dataset source.
- Place them under `scripts/BETH_Dataset/`, or set `BETH_DATA_DIR`.
- Expected official files:
  - `labelled_training_data.csv`
  - `labelled_validation_data.csv`
  - `labelled_testing_data.csv`
- Supplementary development positives are loaded from files matching `labelled_2021may*.csv`, excluding `*-dns.csv`.

MalBehavD-V1:

- Obtain the CSV from `https://github.com/mpasco/MalbehavD-V1`.
- Set `MALBEHAVD_CSV` to the downloaded CSV path.

MalBehavD-V1 temporal metadata:

- Set `MALBEHAVD_VT_METADATA_DIR` to a directory containing local JSONL metadata.
- Each record should include `sha256`, `label`, `status`, `first_submission_date`, and `creation_date`.
- The temporal audit used 2,543 resolved unique hashes and 2,273 timestamp-valid samples after quality filtering.

## Environment

The regenerated experiments used:

- Windows 11
- Python 3.13.5
- CPU-only PyTorch 2.8.0
- scikit-learn 1.7.2
- pandas 2.2.3
- NumPy 2.2.6
- matplotlib 3.10.5
- seaborn 0.13.2
- tqdm 4.67.1
- psutil 7.0.0
- XGBoost 3.2.0

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Reproduction Workflow

The commands below use PowerShell syntax. Equivalent shell exports work on Linux, macOS, Git Bash, and WSL.

### 1. Prepare BETH

```powershell
$env:BETH_DATA_DIR="C:\path\to\BETH_Dataset"
```

### 2. Run the main BETH pipeline

```powershell
cd scripts
python run_pipeline.py
cd ..
```

Default output: `scripts/pipeline_output/`

Optional custom output:

```powershell
$env:FAIR_BETH_PIPELINE_OUTPUT="C:\path\to\pipeline_output"
```

### 3. Run BETH tabular, calibration, and paired audits

```powershell
$env:FAIR_BETH_PIPELINE_OUTPUT="C:\path\to\pipeline_output"
$env:FAIR_BETH_AUDIT_OUTPUT="C:\path\to\results\beth_additional_audits"
python scripts/tabular_sota_and_calibration_audit.py
python scripts/paired_comparison_audit.py
```

### 4. Run repeated threshold-transfer audit

```powershell
$env:FAIR_BETH_PIPELINE_OUTPUT="C:\path\to\pipeline_output"
$env:FAIR_BETH_ROBUST_THRESHOLD_OUTPUT="C:\path\to\results\robust_threshold_validation"
python scripts/robust_threshold_validation.py
```

### 5. Run target-copy-free sequence capacity audit

```powershell
$env:FAIR_BETH_PIPELINE_OUTPUT="C:\path\to\pipeline_output"
$env:FAIR_BETH_SEQUENCE_ABLATION_OUTPUT="C:\path\to\results\sequence_capacity_ablation"
python scripts/sequence_capacity_ablation.py
```

### 6. Run BETH limit-lifting audits

```powershell
$env:BETH_DATA_DIR="C:\path\to\BETH_Dataset"
$env:FAIR_BETH_PIPELINE_OUTPUT="C:\path\to\pipeline_output"
python scripts/beth_limit_lifting_analyses.py
```

This script produces attribution, prefix, cost, feature-space stress, host-disjoint, and chronological robustness outputs.

### 7. Run MalBehavD-V1 stratified validation

```powershell
$env:MALBEHAVD_CSV="C:\path\to\MalBehavD-V1-dataset.csv"
python scripts/external_malbehavd_validation.py
```

### 8. Run MalBehavD-V1 creation-date temporal audit

```powershell
$env:MALBEHAVD_CSV="C:\path\to\MalBehavD-V1-dataset.csv"
$env:MALBEHAVD_VT_METADATA_DIR="C:\path\to\vt_metadata"
$env:MALBEHAVD_TEMPORAL_OUTPUT="C:\path\to\results\external_malbehavd_temporal"
python scripts/malbehavd_temporal_audit_full.py
```

## Paper Build

The manuscript source snapshots are in `paper/`.

```powershell
cd paper
pdflatex fair_x_tdsc_v2.tex
bibtex fair_x_tdsc_v2
pdflatex fair_x_tdsc_v2.tex
pdflatex fair_x_tdsc_v2.tex
pdflatex fair_x_tdsc_v2_supplement.tex
pdflatex cover_letter_fair_x_tdsc_v2.tex
```

The Springer submission source used for the current submission is maintained in the companion submission directory outside this repository snapshot, with the same experimental results and synchronized references.

## Interpretation Guide

Use the results as follows:

- AUC and AP measure ranking.
- Precision, recall, F1, FP, and FN measure a locked thresholded decision.
- Validation-FPR thresholding is the main deployable-style policy.
- Validation-max-F1 is a sensitivity policy.
- Oracle thresholds are upper-bound diagnostics.
- Calibration metrics measure probability reliability, while the paper treats model outputs as scores for thresholding.
- BETH is the severe endpoint-distribution-shift case study.
- MalBehavD-V1 is protocol portability evidence under stratified i.i.d. splits plus a bounded creation-date chronological audit.
- Host-disjoint, prefix, cost, feature-stress, and temporal BETH audits define deployment-boundary evidence.

## Citation

Please cite the accompanying FAIR-X / FAIR-BETH paper and the original BETH and MalBehavD-V1 datasets when using this repository.

## License

Code in this repository is released under the MIT License. Dataset licenses and access terms remain those of the original dataset providers.
