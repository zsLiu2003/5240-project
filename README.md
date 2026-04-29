# ChemBERTa Binding Energy Prediction Project

This project formulates molecular binding energy prediction as a **supervised regression** task. The model takes a molecular SMILES string as input and predicts a continuous binding energy label (`Energy_min`). The main method uses ChemBERTa with SMILES augmentation and two-stage supervised fine-tuning.

## Project Structure

```text
5240-project/
├── code/                  # Training, evaluation, splitting, and analysis scripts
├── data/                  # Cleaned dataset and train/validation/test splits
├── results/               # Experiment outputs, summaries, reports, figures, and selected checkpoints
├── presentation_plan.md   # Suggested four-person presentation plan
└── README.md              # This file
```

## Top-Level Files

| Path | Purpose |
|---|---|
| `README.md` | Project-level documentation and file guide. |
| `presentation_plan.md` | Presentation plan for four group members, including slide structure and key talking points. |
| `.gitignore` | Git ignore rules. |

## `code/` Directory

| File | Purpose |
|---|---|
| `code/README.md` | Earlier code-level README with quick-start commands and experiment descriptions. |
| `code/config.py` | Central configuration for paths, model name, hyperparameters, augmentation settings, seeds, and device. |
| `code/dataset.py` | Dataset class and dataloader creation. Handles SMILES tokenization, label normalization, train/val/test splits, and CV dataloaders. |
| `code/model.py` | ChemBERTa regression model definition: pretrained ChemBERTa encoder plus a feedforward regression head. |
| `code/train.py` | Main training entry point. Supports main method, ablations, augmentation, cross-validation, single-seed runs, and single-fold runs. |
| `code/evaluate.py` | Evaluation utilities for RMSE, MAE, R2, prediction collection, and metric printing. |
| `code/smiles_augmentation.py` | RDKit-based randomized SMILES augmentation utilities. |
| `code/train_ecfp_ridge.py` | Traditional baseline using ECFP fingerprints with Ridge regression. |
| `code/analyze_results.py` | Earlier result aggregation and plotting script for experiment summaries. |
| `code/deep_analysis.py` | Main deep analysis pipeline. Produces core mechanism analysis, error-tail analysis, SMILES consistency probe, special case analysis, and attention visualizations. |
| `code/attention_case_study.py` | Standalone attention rollout visualization script for selected SMILES cases and checkpoint. |
| `code/smiles_robustness_probe.py` | Supplemental experiment comparing no-augmentation and augmentation checkpoints under randomized SMILES variants. |
| `code/random_split.py` | Script to create random train/validation/test splits. |
| `code/scaffold_split.py` | Script to create scaffold-based splits. Included for completeness, although random split is more suitable for this inorganic/ionic dataset. |
| `code/test_local.py` | Lightweight local smoke test using dummy data. |
| `code/script.sh` | Shell script placeholder or helper script for running experiments. |

## `data/` Directory

| File | Purpose |
|---|---|
| `data/clean_dataset.csv` | Cleaned dataset used for modeling. Contains molecule metadata, SMILES, and energy labels. |
| `data/energy_data.csv` | Source/intermediate energy data. |
| `data/property.csv` | Source/intermediate molecular property data. |
| `data/split_seed42.csv` | Train/validation/test split for seed 42. |
| `data/split_seed123.csv` | Train/validation/test split for seed 123. |
| `data/split_seed456.csv` | Train/validation/test split for seed 456. |

Important columns:

| Column | Meaning |
|---|---|
| `CID` | Molecule identifier. |
| `Formula` | Molecular formula. |
| `SMILES` | Molecular SMILES string used as model input. |
| `Energy_min` | Target label for supervised regression. |
| `split` | Train/validation/test split indicator. Present in split files. |

## `results/` Directory

The `results/` directory contains experiment outputs, summary tables, reports, figures, and selected checkpoints.

### Main Summary Files

| File | Purpose |
|---|---|
| `results/main_method_summary.csv` | CV summary for No Aug + Two-stage. |
| `results/main_method_aug0.5_summary.csv` | CV summary for Aug + Two-stage, the final method. |
| `results/ablation_stage1_only_aug0.5_summary.csv` | CV summary for Aug + Stage1-only. |
| `results/ablation_stage2_only_aug0.5_summary.csv` | CV summary for Aug + Stage2-only. |
| `results/baseline_results.csv` | ECFP + Ridge baseline summary. |
| `results/ablation1_full_finetune_results.csv` | Earlier non-CV full fine-tuning ablation summary. |
| `results/ablation2_from_scratch_results.csv` | Earlier non-CV from-scratch ablation summary. |
| `results/ablation_stage1_only_results.csv` | Earlier non-CV Stage1-only summary. |
| `results/ablation_stage2_only_results.csv` | Earlier non-CV Stage2-only summary. |

### Report Files

| File | Purpose |
|---|---|
| `results/experiment_analysis.md` | Initial experiment summary and interpretation. |
| `results/four_cv_results_report.md` | Report focused on the four final CV experiments. |
| `results/four_experiment_summary.md` | Short summary table and key findings for the four final experiments. |

### Figure Files

| File | Purpose |
|---|---|
| `results/ablation_comparison.png` | Visualization comparing ablation methods. |
| `results/augmentation_comparison.png` | Visualization comparing augmentation effects. |
| `results/complete_comparison.png` | Earlier complete comparison figure. |

## Main Experiment Output Directories

Each experiment run directory usually contains:

| File | Purpose |
|---|---|
| `results.json` | Configuration, training history, label statistics, best validation RMSE, and final test metrics. |
| `predictions.csv` | Test-set true values, predictions, and errors. |
| `best_model.pt` | Saved model checkpoint, only kept for selected important runs. |

### Four Final CV Experiments

These are the main experiment groups used in the final comparison:

| Directory pattern | Meaning |
|---|---|
| `results/main_method_seed*_fold*/` | No Aug + Two-stage CV runs. |
| `results/main_method_aug0.5_seed*_fold*/` | Aug + Two-stage CV runs. This is the final method. |
| `results/ablation_stage1_only_aug0.5_seed*_fold*/` | Aug + Stage1-only CV runs. |
| `results/ablation_stage2_only_aug0.5_seed*_fold*/` | Aug + Stage2-only CV runs. |

The final comparison uses 3 seeds x 5 folds = 15 runs per method.

### Baseline Output Directories

| Directory pattern | Meaning |
|---|---|
| `results/baseline_ecfp_ridge_seed*/` | ECFP + Ridge baseline predictions for each seed. |

### Selected Checkpoints

| File | Purpose |
|---|---|
| `results/main_method_aug0.5_seed42_fold0/best_model.pt` | Final Aug + Two-stage checkpoint used for deep analysis and attention visualization. |
| `results/main_method_seed42_fold0/best_model.pt` | No Aug + Two-stage checkpoint used for the SMILES robustness probe. |
| `results/main_method_aug0.5_seed42_fold0_v2/` | Additional saved version of the Aug + Two-stage fold 0 run. |
| `results/main_method_seed42_fold0_v2/` | Additional saved version of the No Aug + Two-stage fold 0 run. |

## `results/deep_analysis/`

This directory contains the main deep analysis outputs.

| File | Purpose |
|---|---|
| `method_overview.csv` | Summary metrics for the four final methods. |
| `core_effect_summary.csv` | Decomposes the final method into augmentation effect, encoder adaptation effect, and Stage 1 warm-up effect. |
| `error_tail_summary.csv` | Error distribution analysis: mean/median/Q75/Q90/max absolute error and high-error rate. |
| `special_case_summary.csv` | Case-level error summary for the final method. |
| `selected_attention_cases.csv` | Selected high-error and low-error representative cases for attention visualization. |
| `smiles_consistency_probe.csv` | Prediction consistency of selected molecules under randomized equivalent SMILES strings. |
| `attention_top_tokens.csv` | Top-attended tokens from attention rollout for selected cases. |
| `attention_figures.csv` | Mapping from selected case IDs to generated attention figure paths. |
| `core_effects.png` | Bar plot of the three core effects. |
| `results_level_deep_analysis.md` | Detailed A1/A2/A3 results-level analysis. |
| `special_case_analysis.md` | Special case analysis and motivation. |
| `attention_interpretation.md` | Attention visualization explanation and top-token tables. |
| `deep_analysis_report.md` | Integrated deep analysis report. |

### `results/deep_analysis/attention_figures/`

Contains attention rollout PNG figures for selected special cases. These are qualitative visualizations and should not be interpreted as causal proof.

## `results/smiles_robustness_probe/`

This directory contains the supplemental no-augmentation vs augmentation robustness experiment.

| File | Purpose |
|---|---|
| `smiles_robustness_summary.csv` | Summary comparing prediction stability under randomized SMILES variants. |
| `smiles_variant_predictions.csv` | All predictions for original and randomized SMILES variants. |
| `per_case_robustness_delta.csv` | Per-molecule comparison of robustness gap between no-aug and aug models. |
| `smiles_robustness_report.md` | Markdown report interpreting the robustness probe. |

Main finding:

| Quantity | No Aug + Two-stage | Aug + Two-stage |
|---|---:|---:|
| Mean prediction std across randomized SMILES | 0.0805 | 0.0628 |
| Randomized RMSE degradation | 0.0262 | 0.0231 |
| All-variant RMSE | 0.1662 | 0.1493 |

This supports the claim that SMILES augmentation improves representation robustness.

## Main Commands

Run the final Aug + Two-stage CV experiment:

```bash
python code/train.py \
  --experiment main \
  --augment \
  --aug-prob 0.5 \
  --cv \
  --data-dir data \
  --results-dir results
```

Run one CV fold:

```bash
python code/train.py \
  --experiment main \
  --augment \
  --aug-prob 0.5 \
  --cv \
  --seed 42 \
  --fold 0 \
  --data-dir data \
  --results-dir results
```

Run the deep analysis pipeline:

```bash
python code/deep_analysis.py
```

Run the SMILES robustness probe:

```bash
python code/smiles_robustness_probe.py \
  --no-aug-checkpoint results/main_method_seed42_fold0/best_model.pt \
  --aug-checkpoint results/main_method_aug0.5_seed42_fold0/best_model.pt \
  --seed 42 \
  --n-variants 8 \
  --output-dir results/smiles_robustness_probe
```

Run the ECFP + Ridge baseline:

```bash
python code/train_ecfp_ridge.py --data-dir data --results-dir results
```

## Final Presentation Materials

Use the following files when preparing slides:

| File | Purpose |
|---|---|
| `presentation_plan.md` | Four-person presentation structure and slide-by-slide plan. |
| `results/deep_analysis/deep_analysis_report.md` | Main interpretation and deep analysis. |
| `results/smiles_robustness_probe/smiles_robustness_report.md` | Supplemental robustness evidence for augmentation. |
| `results/deep_analysis/attention_figures/` | Attention visualization figures for special cases. |

## Notes and Assumptions

- The task is supervised regression: SMILES input, `Energy_min` target.
- The final comparison is based on four CV methods, each with 15 runs.
- Case-level analysis assumes that each `predictions.csv` preserves the same row order as the corresponding test split in `data/split_seed*.csv`.
- Future runs should save `CID` and `SMILES` directly in `predictions.csv` to avoid relying on row-order matching.
