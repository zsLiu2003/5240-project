# Review: Training Process, Main Results, Deep Analysis

This note only covers the last three presentation sections:

1. Training process
2. Main results analysis
3. Deep analysis

The main problem is that the current PPT and `presentation_plan.md` use different model stories.

- Current PPT: Ridge / MLP / Random Forest / XGBoost using molecular descriptors.
- Project plan and final result files: ChemBERTa from SMILES, SMILES augmentation, two-stage fine-tuning.

For the later three sections, the ChemBERTa story is stronger because the repository has complete CV ablations, robustness probes, and case analysis for it. The descriptor/XGBoost slides should either be removed or clearly presented as an earlier baseline, not the main method.

## Part 2: Training Process

### Current Issue

Current Slides 7-8 describe:

- Linear Regression
- MLP
- Random Forest
- XGBoost

This does not match the planned "training process" section, which should explain:

- ChemBERTa
- regression head
- two-stage fine-tuning
- SMILES augmentation
- ablation design

### Recommended Slide Structure

#### Slide: Why ChemBERTa?

Key content:

- Input is molecular SMILES.
- ChemBERTa is pretrained on molecular SMILES.
- Add a regression head to predict continuous binding energy.
- Pretraining gives molecular language representations, but supervised labels teach the task-specific mapping.

Suggested speaking:

> We use ChemBERTa because our input is a SMILES string. ChemBERTa has already learned general molecular syntax from large SMILES corpora. We add a regression head and fine-tune it using our binding-energy labels, so the task remains supervised regression.

#### Slide: Two-Stage Fine-Tuning

Key content:

- Stage 1: freeze ChemBERTa encoder, train regression head.
- Stage 2: unfreeze encoder, fine-tune with smaller backbone learning rate.
- Head uses larger learning rate than backbone.
- Motivation: small dataset, avoid unstable full fine-tuning from the first epoch.

Suggested speaking:

> In Stage 1, the model first learns the energy scale using fixed pretrained features. In Stage 2, we let the encoder adapt to the binding-energy task. This matters because our dataset is small and chemically unusual, so directly updating the whole model from the beginning can be unstable.

#### Slide: SMILES Augmentation

Key content:

- Same molecule can have multiple valid SMILES strings.
- Randomized SMILES are used only during training.
- Label stays unchanged.
- Goal: reduce dependence on one arbitrary string serialization.

Suggested speaking:

> SMILES augmentation does not create new molecules or new labels. It creates different valid string views of the same molecule. This encourages the model to learn molecule-level structure instead of memorizing one canonical SMILES order.

#### Slide: Experimental Design

Use this table:

| Method | Purpose |
|---|---|
| No Aug + Two-stage | Baseline two-stage training without SMILES augmentation |
| Aug + Two-stage | Final method |
| Aug + Stage1-only | Tests whether frozen ChemBERTa features are enough |
| Aug + Stage2-only | Tests whether Stage 1 warm-up helps |

Suggested speaking:

> These four settings isolate the effect of each training component. No Aug versus Aug tests SMILES augmentation. Stage1-only versus two-stage tests whether the encoder must adapt. Stage2-only versus two-stage tests whether the frozen-head warm-up helps.

## Part 3: Main Results Analysis

### Current Issue

Current Slides 8-10 say XGBoost is best and then interpret descriptor feature importance. This is not compatible with the ChemBERTa result story.

If you keep these slides, the audience will think the final model is XGBoost. Then the later attention and SMILES robustness analysis will feel disconnected.

### Recommended Slide Structure

#### Slide: Four Main Experiment Results

Use:

| Method | RMSE | MAE | R2 |
|---|---:|---:|---:|
| No Aug + Two-stage | 0.1507 +/- 0.0169 | 0.1182 +/- 0.0137 | 0.4020 +/- 0.1972 |
| Aug + Two-stage | 0.1312 +/- 0.0160 | 0.1042 +/- 0.0132 | 0.5345 +/- 0.1869 |
| Aug + Stage1-only | 0.1783 +/- 0.0115 | 0.1441 +/- 0.0106 | 0.1802 +/- 0.1717 |
| Aug + Stage2-only | 0.1414 +/- 0.0212 | 0.1111 +/- 0.0150 | 0.4479 +/- 0.2487 |

Main message:

> Aug + Two-stage is the best overall method across 15 CV runs.

Suggested speaking:

> The final method gives the lowest RMSE and MAE. Compared with the no-augmentation two-stage model, RMSE decreases from 0.1507 to 0.1312. Compared with Stage1-only, the improvement is much larger, showing that frozen pretrained embeddings are not enough for this binding-energy task.

#### Slide: What Each Component Contributes

Use:

| Effect | Comparison | Relative RMSE Improvement | Fold-Level Result |
|---|---|---:|---:|
| SMILES augmentation | No Aug + Two-stage -> Aug + Two-stage | 12.92% | 15/15 folds improved |
| Encoder adaptation | Aug + Stage1-only -> Aug + Two-stage | 26.39% | 15/15 folds improved |
| Stage 1 warm-up | Aug + Stage2-only -> Aug + Two-stage | 7.17% | 12/15 folds improved |

Main message:

> Encoder adaptation is the largest effect; augmentation is the most consistent effect; warm-up is useful but secondary.

Suggested speaking:

> The largest gain comes from Stage 2 encoder adaptation. This means ChemBERTa's pretrained representation is useful, but it still needs supervised adaptation to this dataset. SMILES augmentation improves every fold, so it is a very consistent robustness improvement. Stage 1 warm-up helps in 12 out of 15 folds, but the effect is smaller.

#### Slide: Error Tail Analysis

Use:

| Method | Q90 Abs Error | Max Abs Error | Abs Error > 0.20 eV |
|---|---:|---:|---:|
| No Aug + Two-stage | 0.2476 | 0.4751 | 18.96% |
| Aug + Two-stage | 0.2210 | 0.4153 | 14.79% |
| Aug + Stage1-only | 0.3127 | 0.4325 | 28.33% |
| Aug + Stage2-only | 0.2383 | 0.5048 | 17.71% |

Main message:

> The final method improves average accuracy and reduces large-error cases.

Suggested speaking:

> For screening, we care not only about average error but also about the high-error tail. The final method lowers the fraction of predictions above 0.20 eV absolute error from 18.96% to 14.79%. This suggests better reliability, although high-error cases still remain.

### What to Avoid in Part 3

- Do not say XGBoost is the final recommended model if the following slides discuss ChemBERTa.
- Do not claim descriptor feature importance proves chemical causality.
- Do not say "95% variance explained" unless the exact result is shown and it belongs to the same final experiment protocol.
- Do not mix single-split results with 15-run CV results without explaining the difference.

## Part 4: Deep Analysis

### Current Issue

The current PPT has a "Molecule Selection" slide and a mostly empty "Device Performance" slide. These do not yet explain why the model succeeds or fails.

The deep analysis should answer:

- Where does the model still fail?
- Does SMILES augmentation really improve robustness?
- What can attention visualization show, and what can it not prove?
- What are the limitations?

### Recommended Slide Structure

#### Slide: Special Case Analysis

Use:

| Case Type | Formula | SMILES | Mean Abs Error |
|---|---|---|---:|
| Largest-error failure | CH3BNaO | `[B-]OC.[Na+]` | 0.3580 |
| Halogen failure | ILiO | `[Li+].[O-]I` | 0.2952 |
| Additional high-error failure | BH3INa | `[BH3-]I.[Na+]` | 0.2919 |
| Low-error success contrast | Cl2IK | `Cl[I-]Cl.[K+]` | 0.0217 |

Main message:

> The remaining errors are chemically structured, not random.

Suggested speaking:

> The high-error cases are mostly charged, metal-containing, or heavy-halogen species. This suggests that the remaining model failures are related to unusual local chemistry. But we also include a low-error ionic contrast case, so the conclusion is not that all ionic molecules fail. The limitation is more specific: some rare ionic bonding patterns are hard for this SMILES-only model.

#### Slide: SMILES Robustness Probe

Use:

| Model | Prediction Std Across Randomized SMILES | Randomized RMSE Degradation | All-Variant RMSE |
|---|---:|---:|---:|
| No Aug + Two-stage | 0.0805 | 0.0262 | 0.1662 |
| Aug + Two-stage | 0.0628 | 0.0231 | 0.1493 |

Main message:

> This supports the mechanism behind augmentation.

Suggested speaking:

> We test the same molecule under randomized equivalent SMILES strings. The augmented model has lower prediction variation, from 0.0805 to 0.0628. This supports our claim that augmentation makes the model less sensitive to arbitrary SMILES ordering.

#### Slide: Attention Visualization

Main message:

> Attention is useful qualitative evidence, not causal proof.

Recommended wording:

> Attention maps show that the model often focuses on chemically meaningful tokens such as charge symbols, metal ions, halogens, boron, oxygen, and nitrogen. However, in failure cases the model may still attend to the right tokens but predict the wrong energy. So the issue is not just token recognition; it is whether the learned representation captures unusual ionic chemistry and binding physics.

Do not say:

> The attention map proves the model understands the chemistry.

Say instead:

> The attention map is consistent with chemically meaningful token usage, but it does not prove causal reasoning.

#### Slide: Limitations and Future Work

Use:

- Dataset is small.
- Many molecules are unusual charged or ionic species.
- SMILES does not explicitly encode 3D geometry or surface interaction.
- Attention visualization is qualitative.
- Some case-level mapping assumes prediction order matches the test split because prediction files do not directly save CID and SMILES.

Future work:

- Add graph or 3D molecular features.
- Add surface-specific descriptors.
- Collect more diverse training data.
- Save molecule metadata directly in prediction outputs.
- Add explicit consistency training for randomized SMILES.

Suggested speaking:

> The main limitation is that this is a small-data SMILES-only model for a chemically unusual dataset. The model improves robustness and accuracy, but it still lacks explicit 3D and surface-interaction information. Future work should combine SMILES or graph learning with geometry and surface-specific descriptors.

## Suggested Timing

For three speakers after intro/data:

| Speaker | Section | Slides | Time |
|---|---|---:|---:|
| Speaker 2 | Training process | 4 | 3-4 min |
| Speaker 3 | Main results analysis | 3 | 3-4 min |
| Speaker 4 | Deep analysis | 4 | 3-4 min |

If time is tight:

- Combine "Why ChemBERTa" and "Two-stage fine-tuning".
- Keep only two result slides: main result table and component contribution.
- Combine attention and limitations into one slide.

## Most Important Fixes

1. Replace descriptor-model training slides with ChemBERTa training slides.
2. Replace XGBoost result discussion with the four ChemBERTa CV experiments.
3. Replace descriptor feature contribution with component contribution analysis.
4. Replace molecule selection/device-performance slides with special cases, SMILES robustness, attention caution, and limitations.
5. Keep the logic chain clear:

> Training design -> CV result improvement -> component contribution -> error-tail reliability -> special-case failure mechanism -> limitations.
