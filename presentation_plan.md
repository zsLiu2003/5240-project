# CENG5240 Project Presentation Plan

## Overall Story

Our project formulates binding energy prediction as a **supervised molecular regression** problem:

> Given a molecule represented by a SMILES string, predict its binding energy.

The main message of the presentation should be:

> SMILES augmentation plus two-stage ChemBERTa fine-tuning improves generalization by combining representation robustness and task-specific encoder adaptation.

The presentation should not only report model performance. According to the Week 8 project framework, we should demonstrate that we understand:

- the bigger chemical/biological engineering problem,
- how the task is formulated as supervised learning,
- what data is needed and how it is processed,
- why the model and training strategy are reasonable,
- how model performance is evaluated,
- why the model performs well or fails in certain cases.

## Recommended Four-Person Structure

Suggested total length: 12-16 minutes.

- 16 min version: about 4 min per person.
- 12 min version: about 3 min per person, with fewer slides.

## Speaker 1: Background and Data Processing

This person must cover the background and data processing.

### Slide 1: Big Picture and Motivation

Key points:

- Binding energy prediction is important in chemical and biological engineering.
- Experimental or high-level computational evaluation can be expensive.
- A machine learning model can provide faster screening or prediction.
- Our task is not molecule generation; it is property prediction.

Supervised learning point:

> Each molecule has an input representation and a known target energy label, so this is a supervised learning problem.

### Slide 2: Supervised Learning Formulation

Key points:

- Input `X`: molecular SMILES string.
- Label `y`: binding energy, using `Energy_min`.
- Model objective: learn `f(SMILES) -> binding energy`.
- Problem type: regression, because the output is continuous.
- Loss: MSE on normalized labels.
- Evaluation: RMSE, MAE, and R2 on held-out test molecules.

Suggested wording:

> The key supervised learning assumption is that the model learns a mapping from labeled examples and is evaluated on held-out molecules, not on the training labels.

### Slide 3: Dataset and Label Choice

Key points:

- Dataset size: 207 molecules.
- Available information includes formula, SMILES, molecular weight, HBA/HBD, TPSA, complexity, and energy values.
- We use `Energy_min` as the target label because it represents the strongest binding energy signal.
- The dataset contains many small charged or ionic metal-containing molecules, so it is different from typical drug-like organic molecule datasets.

### Slide 4: Data Cleaning and Splitting

Key points:

- Cleaned invalid or missing records.
- Generated train/validation/test splits for seeds 42, 123, and 456.
- Final comparison uses 3 seeds x 5 folds = 15 runs per method.
- Labels are normalized using training-set statistics.
- SMILES strings are tokenized for ChemBERTa.

Important point about data leakage:

> All preprocessing that depends on labels, such as normalization, uses training statistics only. Validation and test sets are held out for model selection and final evaluation.

## Speaker 2: Model and Training Strategy

### Slide 5: Why ChemBERTa?

Key points:

- SMILES is a molecular string representation.
- ChemBERTa is pretrained on molecular SMILES and can provide useful molecular representations.
- We add a regression head on top of the ChemBERTa `[CLS]` embedding.

Supervised learning point:

> Pretraining gives general molecular representations, but our supervised labels teach the model the task-specific mapping to binding energy.

### Slide 6: Two-Stage Supervised Fine-Tuning

Key points:

- Stage 1: freeze the ChemBERTa encoder and train only the regression head.
- Stage 2: unfreeze the encoder and fine-tune it with a smaller learning rate.
- The head uses a larger learning rate than the backbone.
- This is useful for a small dataset because full fine-tuning from the beginning can be unstable.

Suggested wording:

> Stage 1 learns the supervised energy scale; Stage 2 adapts the molecular representation to the supervised objective.

### Slide 7: SMILES Augmentation

Key points:

- The same molecule can have multiple valid SMILES strings.
- During training, a molecule can be represented by randomized equivalent SMILES.
- The label does not change, because the molecule is the same.
- This encourages the model to learn molecule-level structure rather than memorizing one SMILES serialization.

Important wording:

> Augmentation does not create new labels. It creates alternative input views for the same supervised target.

### Slide 8: Experimental Design

Main four experiments:

| Method | Purpose |
|---|---|
| No Aug + Two-stage | Baseline two-stage method without SMILES augmentation |
| Aug + Two-stage | Final method |
| Aug + Stage1-only | Tests whether frozen ChemBERTa embeddings are enough |
| Aug + Stage2-only | Tests whether Stage 1 warm-up is useful |

Interpretation:

- No Aug vs Aug tests the effect of SMILES augmentation.
- Stage1-only vs Two-stage tests the need for encoder adaptation.
- Stage2-only vs Two-stage tests the role of Stage 1 warm-up.

## Speaker 3: Main Results and Mechanistic Analysis

### Slide 9: Four Main Experiment Results

Use the method overview from:

`results/deep_analysis/deep_analysis_report.md`

Key numbers:

| Method | RMSE | MAE | R2 |
|---|---:|---:|---:|
| No Aug + Two-stage | 0.1507 +/- 0.0169 | 0.1182 +/- 0.0137 | 0.4020 +/- 0.1972 |
| Aug + Two-stage | 0.1312 +/- 0.0160 | 0.1042 +/- 0.0132 | 0.5345 +/- 0.1869 |
| Aug + Stage1-only | 0.1783 +/- 0.0115 | 0.1441 +/- 0.0106 | 0.1802 +/- 0.1717 |
| Aug + Stage2-only | 0.1414 +/- 0.0212 | 0.1111 +/- 0.0150 | 0.4479 +/- 0.2487 |

Main message:

> The final method is not just the best average result. It improves both accuracy and robustness across folds.

### Slide 10: What Each Component Contributes

Use the core effect summary:

| Effect | Comparison | Relative RMSE improvement | Fold-level result |
|---|---|---:|---:|
| SMILES augmentation | No Aug + Two-stage -> Aug + Two-stage | 12.92% | 15/15 folds improved |
| Encoder adaptation | Aug + Stage1-only -> Aug + Two-stage | 26.39% | 15/15 folds improved |
| Stage 1 warm-up | Aug + Stage2-only -> Aug + Two-stage | 7.17% | 12/15 folds improved |

Core insight:

> The largest gain comes from encoder adaptation, meaning frozen pretrained embeddings are not enough. Augmentation consistently helps by improving robustness. Warm-up helps, but is secondary.

### Slide 11: Error Tail Analysis

Use `results/deep_analysis/error_tail_summary.csv`.

Key numbers:

| Method | Q90 absolute error | Max absolute error | % abs error > 0.20 eV |
|---|---:|---:|---:|
| No Aug + Two-stage | 0.2476 | 0.4751 | 18.96% |
| Aug + Two-stage | 0.2210 | 0.4153 | 14.79% |
| Aug + Stage1-only | 0.3127 | 0.4325 | 28.33% |
| Aug + Stage2-only | 0.2383 | 0.5048 | 17.71% |

Interpretation:

> The final method does not only improve the average. It also reduces high-error tail cases, which is important for reliability.

### Slide 12: SMILES Robustness Probe

Use:

`results/smiles_robustness_probe/smiles_robustness_report.md`

Key numbers:

| Model | Prediction std across randomized SMILES | Randomized RMSE degradation | All-variant RMSE |
|---|---:|---:|---:|
| No Aug + Two-stage | 0.0805 | 0.0262 | 0.1662 |
| Aug + Two-stage | 0.0628 | 0.0231 | 0.1493 |

Interpretation:

> This directly supports the augmentation mechanism: the augmented model is less sensitive to equivalent SMILES representations.

## Speaker 4: Special Cases, Attention, and Limitations

### Slide 13: Special Case Analysis

Use:

`results/deep_analysis/special_case_analysis.md`

Representative cases:

| Case type | Formula | SMILES | Mean abs error |
|---|---|---|---:|
| Largest-error failure | CH3BNaO | `[B-]OC.[Na+]` | 0.3580 |
| Halogen failure | ILiO | `[Li+].[O-]I` | 0.2952 |
| Additional high-error failure | BH3INa | `[BH3-]I.[Na+]` | 0.2919 |
| Low-error success contrast | Cl2IK | `Cl[I-]Cl.[K+]` | 0.0217 |

Interpretation:

> Remaining errors are not random. They concentrate in chemically unusual ionic species, metal-containing salts, halogen species, and rare local bonding patterns.

### Slide 14: Attention Visualization

Use attention figures from:

`results/deep_analysis/attention_figures/`

Recommended figures:

- `cid_23718351_1.png`: largest-error failure, `[B-]OC.[Na+]`
- `cid_140455213_2.png`: halogen or ionic failure, `[Li+].[O-]I`
- `cid_129740401_8.png`: low-error success contrast, `Cl[I-]Cl.[K+]`

Important caution:

> Attention is qualitative evidence, not causal proof. It shows whether the model attends to chemically meaningful tokens such as charge, metal, halogen, oxygen, boron, and nitrogen.

Suggested interpretation:

- In failure cases, the model may attend to important tokens but still produce inaccurate predictions.
- This suggests the issue is not only token recognition, but how well ChemBERTa represents unusual ionic chemistry.
- The success contrast shows that the model can handle some ionic structures.

### Slide 15: Limitations and Future Improvements

Limitations:

- Dataset is small: 207 molecules.
- Many molecules are unusual charged or ionic species.
- SMILES is a text representation and may not fully capture ionic interactions or 3D geometry.
- Attention visualization is qualitative and not a causal explanation.
- Current prediction files do not directly store `CID` and `SMILES`; case-level mapping assumes prediction order matches the test split.

Future improvements:

- Use graph neural networks.
- Add 3D or geometric molecular features.
- Collect a larger and more diverse dataset.
- Save molecule metadata directly in prediction outputs.
- Add explicit consistency training for randomized SMILES variants.

### Slide 16: Final Takeaways

Suggested final points:

1. We formulated binding energy prediction as supervised molecular regression.
2. Augmentation plus two-stage fine-tuning improves generalization by combining SMILES robustness and encoder adaptation.
3. Remaining failures reveal a meaningful limitation: unusual charged metal-containing molecules are still difficult for SMILES-based ChemBERTa.

## Timing Plan

### 16-Minute Version

| Speaker | Topic | Time |
|---|---|---:|
| Speaker 1 | Background and data processing | 4 min |
| Speaker 2 | Model and training strategy | 4 min |
| Speaker 3 | Results and mechanistic analysis | 4.5 min |
| Speaker 4 | Special cases, attention, limitations | 3.5 min |

### 12-Minute Version

| Speaker | Topic | Time |
|---|---|---:|
| Speaker 1 | Background and data processing | 3 min |
| Speaker 2 | Model and training strategy | 3 min |
| Speaker 3 | Results and analysis | 3 min |
| Speaker 4 | Special cases and limitations | 3 min |

For the shorter version, reduce to about 3 slides per speaker and combine attention with limitations.

## Key Supervised Learning Points to Emphasize

Everyone should repeatedly connect their section to supervised learning:

- We have labeled examples: molecule input and energy target.
- The target is continuous, so this is regression.
- Train, validation, and test sets are separated.
- The model minimizes training loss but is judged by held-out metrics.
- Ablation experiments test which supervised training component matters.
- Deep analysis checks not only whether the model works, but why and where it fails.

## Suggested Closing Statement

> Our results show that ChemBERTa can be adapted to supervised binding-energy prediction, but the adaptation strategy matters. SMILES augmentation improves representation robustness, Stage 2 fine-tuning is necessary for task-specific learning, and remaining errors point to the challenge of representing unusual charged ionic molecules using SMILES-only models.
