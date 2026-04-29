# Two-stage ChemBERTa Experiment Analysis

## Method

The main method uses supervised regression from SMILES to binding energy with a pretrained ChemBERTa encoder and a small regression head. Training is split into two stages: Stage 1 freezes the encoder and learns only the regression head, so the model first maps pretrained molecular representations to the target energy scale. Stage 2 unfreezes the encoder with a smaller backbone learning rate and a larger head learning rate, allowing task-specific adaptation without immediately destroying pretrained chemical features.

SMILES augmentation is applied only to the training split. Each molecule keeps the same label, but its SMILES string can be replaced by a randomized equivalent representation. This tests whether the model learns molecule-level structure instead of overfitting to one textual serialization.

## Results

| Method | RMSE | MAE | R2 | Runs | Source |
|---|---:|---:|---:|---:|---|
| ECFP + Ridge | 0.1181 +/- 0.0130 | 0.0887 +/- 0.0159 | 0.6474 +/- 0.0438 | 3 | baseline_results.csv |
| Full Finetune | 0.1332 +/- 0.0010 | 0.0947 +/- 0.0060 | 0.5391 +/- 0.1187 | 3 | ablation1_full_finetune_results.csv |
| From Scratch | 0.0985 +/- 0.0025 | 0.0742 +/- 0.0026 | 0.7457 +/- 0.0748 | 3 | ablation2_from_scratch_results.csv |
| Stage1-only | 0.1769 +/- 0.0257 | 0.1384 +/- 0.0196 | 0.2004 +/- 0.2187 | 3 | ablation_stage1_only_results.csv |
| Stage2-only | 0.1483 +/- 0.0230 | 0.1132 +/- 0.0140 | 0.4211 +/- 0.2436 | 3 | ablation_stage2_only_results.csv |
| Main Method (CV) | 0.1507 +/- 0.0169 | 0.1182 +/- 0.0137 | 0.4020 +/- 0.1972 | 15 | main_method_summary.csv |
| Main Method | 0.1608 +/- 0.0178 | 0.1231 +/- 0.0159 | 0.3382 +/- 0.1578 | 3 | main_method_seed*/results.json |
| Main + Aug (0.5) | 0.1340 +/- 0.0165 | 0.1052 +/- 0.0119 | 0.5163 +/- 0.2126 | 3 | main_method_aug0.5_summary.csv |

## Ablation Interpretation

- Compared with Stage1-only, the full two-stage method changes RMSE by +9.98%, isolating the value of supervised encoder adaptation after head warm-up.
- Compared with Stage2-only, the full two-stage method changes RMSE by -7.78%, isolating the value of learning a stable regression head before unfreezing the backbone.

## Augmentation Interpretation

- Best augmentation setting: Main + Aug (0.5), RMSE 0.1340.
- Relative to the non-augmented main method, this is +16.66% RMSE change.
