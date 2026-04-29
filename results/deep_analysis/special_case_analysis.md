# B. Special Case Analysis

## Motivation

After the final method improves average performance, the next question is where it still fails. The selected cases test whether remaining errors are random or concentrated in chemically unusual small ionic molecules such as charged metal salts, halogen-containing species, and polar fragments.

| case_id | case_type | motivation | Formula | SMILES | true | pred_mean | abs_error_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cid_23718351_1 | largest-error failure | Shows the strongest remaining limitation of the final method. | CH3BNaO | [B-]OC.[Na+] | -2.8372 | -2.4792 | 0.3580 |
| cid_140455213_2 | halogen failure | Tests whether heavy halogen-containing molecules remain difficult. | ILiO | [Li+].[O-]I | -2.3537 | -2.6489 | 0.2952 |
| cid_159330689_3 | additional high-error failure | Adds coverage of the high-error tail. | BH3INa | [BH3-]I.[Na+] | -2.5873 | -2.8792 | 0.2919 |
| cid_58676901_4 | additional high-error failure | Adds coverage of the high-error tail. | CHBNNa | [B][C-]=N.[Na+] | -2.6942 | -2.9597 | 0.2655 |
| cid_22960788_5 | additional high-error failure | Adds coverage of the high-error tail. | CH3NaO2 | CO[O-].[Na+] | -2.8261 | -2.5635 | 0.2626 |
| cid_58627710_6 | additional high-error failure | Adds coverage of the high-error tail. | BKN | [B]=[N-].[K+] | -2.4554 | -2.7030 | 0.2476 |
| cid_58604523_7 | additional high-error failure | Adds coverage of the high-error tail. | C2H6NRb | C[CH-]N.[Rb+] | -3.1476 | -2.9173 | 0.2304 |
| cid_129740401_8 | low-error success contrast | Provides a contrast case showing that not all ionic or unusual molecules fail. | Cl2IK | Cl[I-]Cl.[K+] | -2.3512 | -2.3306 | 0.0217 |

## Interpretation

The high-error cases are dominated by formally charged, metal-containing, or heavy-atom species. The low-error contrast case is included to avoid overclaiming: the model can handle some ionic molecules, but its remaining failures are concentrated in uncommon local chemical environments.
