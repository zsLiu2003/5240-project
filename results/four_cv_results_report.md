# Four Cross-validation Experiment Results

This report summarizes only the four experiments used for the final comparison. All results are evaluated with the same cross-validation protocol: 3 random seeds and 5 folds per seed, for 15 runs per method.

## Results

| Method | Runs | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|
| No Aug + Two-stage | 15 | 0.1507 +/- 0.0169 | 0.1182 +/- 0.0137 | 0.4020 +/- 0.1972 |
| Aug + Two-stage | 15 | 0.1312 +/- 0.0160 | 0.1042 +/- 0.0132 | 0.5345 +/- 0.1869 |
| Aug + Stage1-only | 15 | 0.1783 +/- 0.0115 | 0.1441 +/- 0.0106 | 0.1802 +/- 0.1717 |
| Aug + Stage2-only | 15 | 0.1414 +/- 0.0212 | 0.1111 +/- 0.0150 | 0.4479 +/- 0.2487 |

## Key Comparisons

### Effect of SMILES Augmentation

Comparing the two-stage method with and without SMILES augmentation:

| Comparison | RMSE Change | Relative RMSE Improvement | Fold-level Result |
|---|---:|---:|---:|
| No Aug + Two-stage -> Aug + Two-stage | 0.1507 -> 0.1312 | +12.92% | 15/15 folds improved |

SMILES augmentation consistently improves performance across all folds. This suggests that randomized equivalent SMILES representations help the model generalize beyond a single string serialization of each molecule.

### Effect of Stage 2 after Stage 1

Comparing augmented Stage1-only with the full augmented two-stage method:

| Comparison | RMSE Change | Relative RMSE Improvement | Fold-level Result |
|---|---:|---:|---:|
| Aug + Stage1-only -> Aug + Two-stage | 0.1783 -> 0.1312 | +26.40% | 15/15 folds improved |

Stage1-only performs worst among the four methods. Freezing the ChemBERTa encoder and training only the regression head is not sufficient for this supervised binding-energy prediction task. Stage 2, which unfreezes and fine-tunes the encoder, is necessary for task-specific adaptation.

### Effect of Stage 1 Warm-up before Stage 2

Comparing augmented Stage2-only with the full augmented two-stage method:

| Comparison | RMSE Change | Relative RMSE Improvement | Fold-level Result |
|---|---:|---:|---:|
| Aug + Stage2-only -> Aug + Two-stage | 0.1414 -> 0.1312 | +7.20% | 12/15 folds improved |

Stage2-only is much stronger than Stage1-only, showing that encoder fine-tuning is important. However, the full two-stage method still performs better on average. The Stage 1 warm-up gives the regression head a stable supervised mapping before the encoder is updated, making later fine-tuning more effective.

## Interpretation

The best-performing method is Aug + Two-stage, with the lowest RMSE and MAE and the highest R2. The results support two main conclusions:

1. SMILES augmentation improves generalization by training the model to treat different valid SMILES strings for the same molecule as equivalent supervised examples.
2. Two-stage supervised training is more effective than either Stage1-only or Stage2-only. Stage 1 learns the regression head on top of pretrained molecular representations, while Stage 2 adapts the encoder to the binding-energy prediction task.

Overall, the full method combines representation-level robustness from SMILES augmentation with stable task adaptation from two-stage training.
