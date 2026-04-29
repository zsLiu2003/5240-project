# A. Mechanistic Results Analysis

## Core Insight

The final method works because it combines two complementary mechanisms: SMILES augmentation improves robustness to molecular string representation, while Stage 2 fine-tuning adapts ChemBERTa features to binding-energy regression. Stage 1 warm-up adds a smaller stabilization effect before encoder updates.

## A1. Core Result Interpretation

| effect | baseline_RMSE | candidate_RMSE | relative_improvement_pct | folds_improved | folds_worse |
| --- | --- | --- | --- | --- | --- |
| SMILES augmentation | 0.1507 | 0.1312 | 12.9189 | 15 | 0 |
| Encoder adaptation | 0.1783 | 0.1312 | 26.3858 | 15 | 0 |
| Stage 1 warm-up | 0.1414 | 0.1312 | 7.1719 | 12 | 3 |

## A2. Error-tail Experiment

This post-hoc experiment checks whether the final gain comes from reducing difficult large-error predictions rather than only shifting the average.

| method | n_predictions | mean_abs_error | median_abs_error | q75_abs_error | q90_abs_error | max_abs_error | pct_abs_error_gt_0.20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No Aug + Two-stage | 480 | 0.1182 | 0.0967 | 0.1756 | 0.2476 | 0.4751 | 18.9583 |
| Aug + Two-stage | 480 | 0.1042 | 0.0873 | 0.1492 | 0.2210 | 0.4153 | 14.7917 |
| Aug + Stage1-only | 480 | 0.1441 | 0.1226 | 0.2121 | 0.3127 | 0.4325 | 28.3333 |
| Aug + Stage2-only | 480 | 0.1111 | 0.0872 | 0.1644 | 0.2383 | 0.5048 | 17.7083 |

For the final method, Q90 absolute error is 0.2210 eV and 14.8% of predictions exceed 0.20 eV absolute error.

## A3. SMILES Consistency Probe

This probe evaluates the same molecule under randomized equivalent SMILES strings. Lower prediction variance supports the intended augmentation mechanism: the model should be less dependent on a single SMILES serialization.

| case_id | case_type | CID | n_variants | prediction_std | prediction_range | mean_abs_error |
| --- | --- | --- | --- | --- | --- | --- |
| cid_129740401_8 | low-error success contrast | 129740401 | 2 | 0.0097 | 0.0194 | 0.0098 |
| cid_140455213_2 | halogen failure | 140455213 | 2 | 0.0311 | 0.0621 | 0.3353 |
| cid_159330689_3 | additional high-error failure | 159330689 | 2 | 0.0268 | 0.0536 | 0.2375 |
| cid_22960788_5 | additional high-error failure | 22960788 | 4 | 0.1889 | 0.4649 | 0.1741 |
| cid_23718351_1 | largest-error failure | 23718351 | 4 | 0.1940 | 0.4921 | 0.1760 |
| cid_58604523_7 | additional high-error failure | 58604523 | 4 | 0.0695 | 0.1743 | 0.1886 |
| cid_58627710_6 | additional high-error failure | 58627710 | 2 | 0.0363 | 0.0726 | 0.1656 |
| cid_58676901_4 | additional high-error failure | 58676901 | 4 | 0.0572 | 0.1403 | 0.0484 |
