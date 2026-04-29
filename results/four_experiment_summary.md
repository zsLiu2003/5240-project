# Four CV Experiment Summary

| Method | Runs | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|
| No Aug + Two-stage | 15 | 0.1507 +/- 0.0169 | 0.1182 +/- 0.0137 | 0.4020 +/- 0.1972 |
| Aug + Two-stage | 15 | 0.1312 +/- 0.0160 | 0.1042 +/- 0.0132 | 0.5345 +/- 0.1869 |
| Aug + Stage1-only | 15 | 0.1783 +/- 0.0115 | 0.1441 +/- 0.0106 | 0.1802 +/- 0.1717 |
| Aug + Stage2-only | 15 | 0.1414 +/- 0.0212 | 0.1111 +/- 0.0150 | 0.4479 +/- 0.2487 |

## Key Findings

- SMILES augmentation improves two-stage CV RMSE by 12.92% (0.1507 -> 0.1312).
- Within augmented training, two-stage improves over Stage1-only by 26.39% RMSE.
- Within augmented training, two-stage changes RMSE relative to Stage2-only by 7.17%.
