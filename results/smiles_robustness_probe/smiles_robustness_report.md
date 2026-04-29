# SMILES Robustness Probe

## Motivation

This supplemental experiment directly tests the mechanism behind SMILES augmentation. If augmentation improves representation robustness, the augmented model should produce more stable predictions for equivalent randomized SMILES strings of the same molecule.

## Setup

- Split seed: `42`
- Randomized variants per molecule: `8`
- Max cases: `all test cases`
- No-aug checkpoint: `/hdd2/zesen/daily/5240/5240-project/results/main_method_seed42_fold0/best_model.pt`
- Aug checkpoint: `/hdd2/zesen/daily/5240/5240-project/results/main_method_aug0.5_seed42_fold0/best_model.pt`

## Summary

| model | n_cases | n_variant_rows | original_RMSE | original_MAE | randomized_RMSE | randomized_MAE | all_variants_RMSE | all_variants_MAE | RMSE_degradation_randomized_minus_original | mean_prediction_std_across_variants | median_prediction_std_across_variants | q90_prediction_std_across_variants | mean_prediction_range_across_variants | q90_prediction_range_across_variants | mean_abs_variant_delta | q90_max_abs_variant_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Aug + Two-stage | 32 | 114 | 0.1469 | 0.1087 | 0.1731 | 0.1398 | 0.1662 | 0.1311 | 0.0262 | 0.0805 | 0.0783 | 0.1456 | 0.2036 | 0.3464 | 0.0788 | 0.2900 |
| Aug + Two-stage | 32 | 114 | 0.1323 | 0.1013 | 0.1554 | 0.1234 | 0.1493 | 0.1172 | 0.0231 | 0.0628 | 0.0617 | 0.1122 | 0.1600 | 0.2826 | 0.0590 | 0.2320 |

## Interpretation

- Augmentation reduces mean prediction std across randomized SMILES (0.0805 -> 0.0628), supporting the SMILES robustness claim.
- Augmentation also reduces RMSE degradation under randomized SMILES (0.0262 -> 0.0231).

## Largest No-Aug vs Aug Robustness Gaps

| model_no_aug | CID | Formula | original_SMILES | prediction_std_no_aug | prediction_range_no_aug | mean_abs_variant_delta_no_aug | max_abs_variant_delta_no_aug | model_aug | prediction_std_aug | prediction_range_aug | mean_abs_variant_delta_aug | max_abs_variant_delta_aug | std_reduction_aug_vs_no_aug | range_reduction_aug_vs_no_aug |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Aug + Two-stage | 59091021 | BH2O2Rb | [B-](O)O.[Rb+] | 0.1467 | 0.2934 | 0.1467 | 0.2934 | Aug + Two-stage | 0.0280 | 0.0560 | 0.0280 | 0.0560 | 0.1187 | 0.2374 |
| No Aug + Two-stage | 59673869 | C2H4NRb | C[C-]=N.[Rb+] | 0.1881 | 0.4778 | 0.1628 | 0.4369 | Aug + Two-stage | 0.1245 | 0.3260 | 0.1137 | 0.2037 | 0.0636 | 0.1517 |
| No Aug + Two-stage | 153488305 | BFNa | [B-]F.[Na+] | 0.1758 | 0.3517 | 0.1758 | 0.3517 | Aug + Two-stage | 0.1168 | 0.2336 | 0.1168 | 0.2336 | 0.0590 | 0.1181 |
| No Aug + Two-stage | 161111143 | BClH3Na | [BH3-]Cl.[Na+] | 0.0982 | 0.1964 | 0.0982 | 0.1964 | Aug + Two-stage | 0.0515 | 0.1030 | 0.0515 | 0.1030 | 0.0467 | 0.0934 |
| No Aug + Two-stage | 58604523 | C2H6NRb | C[CH-]N.[Rb+] | 0.1129 | 0.2984 | 0.0798 | 0.2387 | Aug + Two-stage | 0.0695 | 0.1743 | 0.0944 | 0.1743 | 0.0434 | 0.1241 |
| No Aug + Two-stage | 140455213 | ILiO | [Li+].[O-]I | 0.0706 | 0.1413 | 0.0706 | 0.1413 | Aug + Two-stage | 0.0311 | 0.0621 | 0.0311 | 0.0621 | 0.0396 | 0.0791 |
| No Aug + Two-stage | 158336150 | C2H4IK | C[CH-]I.[K+] | 0.1851 | 0.5184 | 0.1336 | 0.3176 | Aug + Two-stage | 0.1486 | 0.3630 | 0.1425 | 0.3170 | 0.0365 | 0.1554 |
| No Aug + Two-stage | 19607366 | C2BrLi | [Li+].[C-]#CBr | 0.1354 | 0.3589 | 0.1376 | 0.2291 | Aug + Two-stage | 0.1028 | 0.3219 | 0.0732 | 0.1684 | 0.0326 | 0.0370 |
| No Aug + Two-stage | 129646904 | CH3N2Na | [C-](=N)N.[Na+] | 0.0812 | 0.2262 | 0.0596 | 0.1452 | Aug + Two-stage | 0.0499 | 0.1381 | 0.0385 | 0.0788 | 0.0312 | 0.0882 |
| No Aug + Two-stage | 129740401 | Cl2IK | Cl[I-]Cl.[K+] | 0.0378 | 0.0755 | 0.0378 | 0.0755 | Aug + Two-stage | 0.0097 | 0.0194 | 0.0097 | 0.0194 | 0.0281 | 0.0561 |
| No Aug + Two-stage | 157489353 | C2HLiO | [Li+].C#C[O-] | 0.0670 | 0.2037 | 0.0821 | 0.2037 | Aug + Two-stage | 0.0390 | 0.1154 | 0.0285 | 0.0770 | 0.0280 | 0.0883 |
| No Aug + Two-stage | 161449774 | HN2NaO | [NH-]N=O.[Na+] | 0.0885 | 0.2339 | 0.1075 | 0.2339 | Aug + Two-stage | 0.0607 | 0.1502 | 0.1028 | 0.1502 | 0.0277 | 0.0836 |

## Caution

This probe uses matched checkpoints and randomized SMILES variants. It supports or weakens the augmentation mechanism claim, but it is still a post-hoc inference experiment rather than a new cross-validated training result.
