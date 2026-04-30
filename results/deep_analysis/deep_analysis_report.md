# Deep Analysis Report

## Main Message

The four-experiment comparison suggests that the final method succeeds through a combination of representation robustness and task adaptation. SMILES augmentation makes the model less tied to one string form of a molecule; Stage 2 fine-tuning is necessary to adapt ChemBERTa to binding-energy regression; Stage 1 warm-up provides a smaller stabilization effect.

## Method Overview

The main quantitative results are visualized in `results/deep_analysis/figures/`.

| figure | description |
| --- | --- |
| /hdd2/zesen/daily/5240/5240-project/results/deep_analysis/figures/core_effects.png | Relative RMSE improvement for the three core effects. |

| method | runs | RMSE_mean | RMSE_std | MAE_mean | MAE_std | R2_mean | R2_std | worst_fold_RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Aug + Two-stage | 15 | 0.1507 | 0.0169 | 0.1182 | 0.0137 | 0.4020 | 0.1972 | 0.1930 |
| Aug + Two-stage | 15 | 0.1312 | 0.0160 | 0.1042 | 0.0132 | 0.5345 | 0.1869 | 0.1746 |
| Aug + Stage1-only | 15 | 0.1783 | 0.0115 | 0.1441 | 0.0106 | 0.1802 | 0.1717 | 0.1972 |
| Aug + Stage2-only | 15 | 0.1414 | 0.0212 | 0.1111 | 0.0150 | 0.4479 | 0.2487 | 0.1824 |

## Core Effects

| effect | baseline | candidate | relative_improvement_pct | folds_improved | folds_worse |
| --- | --- | --- | --- | --- | --- |
| SMILES augmentation | No Aug + Two-stage | Aug + Two-stage | 12.9189 | 15 | 0 |
| Encoder adaptation | Aug + Stage1-only | Aug + Two-stage | 26.3858 | 15 | 0 |
| Stage 1 warm-up | Aug + Stage2-only | Aug + Two-stage | 7.1719 | 12 | 3 |

## Error Tail

| method | n_predictions | mean_abs_error | median_abs_error | q75_abs_error | q90_abs_error | max_abs_error | pct_abs_error_gt_0.20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No Aug + Two-stage | 480 | 0.1182 | 0.0967 | 0.1756 | 0.2476 | 0.4751 | 18.9583 |
| Aug + Two-stage | 480 | 0.1042 | 0.0873 | 0.1492 | 0.2210 | 0.4153 | 14.7917 |
| Aug + Stage1-only | 480 | 0.1441 | 0.1226 | 0.2121 | 0.3127 | 0.4325 | 28.3333 |
| Aug + Stage2-only | 480 | 0.1111 | 0.0872 | 0.1644 | 0.2383 | 0.5048 | 17.7083 |

## Selected Special Cases

| case_id | case_type | Formula | SMILES | true | pred_mean | abs_error_mean |
| --- | --- | --- | --- | --- | --- | --- |
| cid_23718351_1 | largest-error failure | CH3BNaO | [B-]OC.[Na+] | -2.8372 | -2.4792 | 0.3580 |
| cid_140455213_2 | halogen failure | ILiO | [Li+].[O-]I | -2.3537 | -2.6489 | 0.2952 |
| cid_159330689_3 | additional high-error failure | BH3INa | [BH3-]I.[Na+] | -2.5873 | -2.8792 | 0.2919 |
| cid_58676901_4 | additional high-error failure | CHBNNa | [B][C-]=N.[Na+] | -2.6942 | -2.9597 | 0.2655 |
| cid_22960788_5 | additional high-error failure | CH3NaO2 | CO[O-].[Na+] | -2.8261 | -2.5635 | 0.2626 |
| cid_58627710_6 | additional high-error failure | BKN | [B]=[N-].[K+] | -2.4554 | -2.7030 | 0.2476 |
| cid_58604523_7 | additional high-error failure | C2H6NRb | C[CH-]N.[Rb+] | -3.1476 | -2.9173 | 0.2304 |
| cid_129740401_8 | low-error success contrast | Cl2IK | Cl[I-]Cl.[K+] | -2.3512 | -2.3306 | 0.0217 |

## SMILES Consistency

| case_id | case_type | n_variants | prediction_std | prediction_range |
| --- | --- | --- | --- | --- |
| cid_129740401_8 | low-error success contrast | 2 | 0.0097 | 0.0194 |
| cid_140455213_2 | halogen failure | 2 | 0.0311 | 0.0621 |
| cid_159330689_3 | additional high-error failure | 2 | 0.0268 | 0.0536 |
| cid_22960788_5 | additional high-error failure | 4 | 0.1889 | 0.4649 |
| cid_23718351_1 | largest-error failure | 4 | 0.1940 | 0.4921 |
| cid_58604523_7 | additional high-error failure | 4 | 0.0695 | 0.1743 |
| cid_58627710_6 | additional high-error failure | 2 | 0.0363 | 0.0726 |
| cid_58676901_4 | additional high-error failure | 4 | 0.0572 | 0.1403 |

## Attention Evidence

Attention figures are saved in `results/deep_analysis/attention_figures/`, and top-attended tokens are saved in `attention_top_tokens.csv`. These visualizations are qualitative support for the case analysis rather than proof of causal model reasoning.

## Assumption

The case-level mapping assumes each `predictions.csv` preserves the same row order as the corresponding `test` split in `data/split_seed*.csv`. Future runs should save `CID` and `SMILES` directly in `predictions.csv` to remove this assumption.
