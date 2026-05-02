# PPT Content Review for Group 5

## 1. Biggest Issue

The current PPT and `presentation_plan.md` are not telling the same technical story.

- Current PPT: descriptor-based ML pipeline, using Ridge / MLP / Random Forest / XGBoost.
- `presentation_plan.md` and project results: SMILES-based ChemBERTa regression, SMILES augmentation, and two-stage fine-tuning.

For the final presentation, choose one story. Based on the repository README, results, and plan, the more defensible final story is:

> Supervised binding-energy regression from SMILES using ChemBERTa, with SMILES augmentation and two-stage fine-tuning.

If you keep the current PPT, the later analysis around ChemBERTa, attention, and SMILES robustness cannot be used. If you follow the plan, Slides 7-10 in the current PPT should be replaced rather than lightly edited.

## 2. Slide-Level Problems in the Current PPT

### Slides 1-4: Intro and Data Screening

These slides are mostly usable, but they need to connect more clearly to the supervised learning task.

Suggested correction:

- Say that the screening workflow produces candidate molecules.
- The ML task is not device prediction directly.
- The supervised target is molecule-surface binding energy.
- Final cleaned modeling dataset has 207 molecules, not 414.

Important data wording:

> We started from 414 energy records, but after pairing and cleaning the final supervised dataset contains 207 molecules with SMILES and `Energy_min` labels.

### Slide 5: Dataset Overview

Current issue:

- Speaker note says 414 candidate molecules, each labelled with binding energy.
- Repository data shows `clean_dataset.csv` has 207 rows.
- `energy_data.csv` and `property.csv` have 414 rows, but these are source/intermediate files.

Recommended fix:

> The cleaned dataset used for supervised learning contains 207 molecules. Each molecule has a SMILES representation, molecular descriptors, and energy labels. We use `Energy_min` as the regression target because it represents the strongest binding case.

### Slide 6: Feature Distribution

Current issue:

- This slide is still descriptor-focused. It is fine for data characterization, but should not become the main model story if the final method is ChemBERTa.

Recommended use:

- Keep it as data overview only.
- Do not overclaim that descriptors are the main model input.

Better interpretation:

> These distributions show that our molecules are small, chemically unusual, and often ionic or polar. This helps explain why a pretrained molecular language model may need task-specific fine-tuning.

### Slides 7-9: Training and Model Results

Current issue:

- These slides discuss Ridge, MLP, Random Forest, and XGBoost.
- They conflict with the planned four-person structure, where Speaker 2 presents ChemBERTa training and Speaker 3 presents ChemBERTa ablations.
- The statement "XGBoost captures 95% of variance" is not supported by the final results in the repository.

Recommended action:

- Replace Slides 7-9 with ChemBERTa training and final experiment results.

Suggested replacement sequence:

1. Why ChemBERTa?
2. Two-stage fine-tuning and SMILES augmentation.
3. Four experiment comparison:

| Method | RMSE | MAE | R2 |
|---|---:|---:|---:|
| No Aug + Two-stage | 0.1507 +/- 0.0169 | 0.1182 +/- 0.0137 | 0.4020 +/- 0.1972 |
| Aug + Two-stage | 0.1312 +/- 0.0160 | 0.1042 +/- 0.0132 | 0.5345 +/- 0.1869 |
| Aug + Stage1-only | 0.1783 +/- 0.0115 | 0.1441 +/- 0.0106 | 0.1802 +/- 0.1717 |
| Aug + Stage2-only | 0.1414 +/- 0.0212 | 0.1111 +/- 0.0150 | 0.4479 +/- 0.2487 |

Main wording:

> The final model, Aug + Two-stage, gives the best average RMSE and MAE. More importantly, the ablations show why: augmentation improves representation robustness, and Stage 2 encoder fine-tuning adapts ChemBERTa to the binding-energy target.

### Slide 10: Molecule Feature Contribution

Current issue:

- The slide claims "molecular size and complexity" and "H-bond acceptors" are the strongest positive effects.
- This is a descriptor-model interpretation, not ChemBERTa analysis.
- It also risks overclaiming causality: model feature importance or coefficients are not proof that larger or acceptor-rich molecules truly cause stronger binding.

Recommended replacement:

Use component contribution analysis:

| Effect | Comparison | Relative RMSE Improvement | Fold Result |
|---|---|---:|---:|
| SMILES augmentation | No Aug + Two-stage -> Aug + Two-stage | 12.92% | 15/15 improved |
| Encoder adaptation | Aug + Stage1-only -> Aug + Two-stage | 26.39% | 15/15 improved |
| Stage 1 warm-up | Aug + Stage2-only -> Aug + Two-stage | 7.17% | 12/15 improved |

Main interpretation:

> The largest gain comes from encoder adaptation. This means frozen pretrained ChemBERTa features are not enough for this small ionic binding-energy dataset. SMILES augmentation gives consistent improvement across all folds, and Stage 1 warm-up gives a smaller but usually positive stabilization effect.

### Slide 11: Molecule Selection

Current issue:

- The selected molecules are presented as if they are final recommended additives.
- There is no clear link to test-set prediction error, model uncertainty, or experimental validation.
- This can sound like the model has already proven these molecules will improve devices.

Recommended fix:

Either remove this slide or turn it into "Representative high-predicted-binding candidates" with cautious wording.

Safer wording:

> These examples show molecules with strong predicted binding energies. They should be treated as screening candidates, not confirmed device additives. Further DFT and experimental validation are needed.

If following the ChemBERTa plan, replace Slide 11 with error-tail analysis:

| Method | Q90 Abs Error | Max Abs Error | Abs Error > 0.20 eV |
|---|---:|---:|---:|
| No Aug + Two-stage | 0.2476 | 0.4751 | 18.96% |
| Aug + Two-stage | 0.2210 | 0.4153 | 14.79% |
| Aug + Stage1-only | 0.3127 | 0.4325 | 28.33% |
| Aug + Stage2-only | 0.2383 | 0.5048 | 17.71% |

Interpretation:

> The final method improves not only average error but also the high-error tail, which matters for screening reliability.

### Slide 12: Device Performance

Current issue:

- The slide title is present, but extracted text has no clear content.
- If you include device performance, make sure it is separated from ML prediction results.

Recommended fix:

Use this as a transition:

> ML provides fast binding-energy screening. Device performance still requires experimental validation, because binding energy is only one factor affecting PCE, VOC, JSC, and FF.

If you do not have strong device data, remove this slide from the ML presentation.

## 3. Recommended Third Part: Main Results Analysis

Speaker 3 should not focus on XGBoost or descriptor feature importance if the final method is ChemBERTa. A stronger 3-4 minute structure is:

### Slide A: Main Experiment Results

Message:

> Augmentation plus two-stage fine-tuning is the best overall method.

Use the four-method table above.

Talk track:

> We compare four controlled settings. The final method achieves RMSE 0.1312 and MAE 0.1042, better than no augmentation, Stage1-only, and Stage2-only. Because each method is evaluated over 15 runs, the comparison is more reliable than a single split.

### Slide B: What Each Component Adds

Message:

> The improvement is not a black box; each component has a measurable role.

Use the core effect table.

Talk track:

> The largest effect is encoder adaptation, with a 26.39% RMSE improvement over frozen Stage1-only training. This tells us that pretrained ChemBERTa embeddings alone do not fit this binding-energy task. Augmentation improves RMSE by 12.92% and improves every fold, which supports the idea that randomized SMILES helps the model learn molecule-level information instead of one fixed string.

### Slide C: Error Tail and Reliability

Message:

> The final method reduces large-error cases, but does not eliminate them.

Use the error-tail table.

Talk track:

> For screening, average RMSE is not enough. A model can have acceptable mean error but still fail badly on important candidates. The augmented two-stage model reduces the percentage of predictions with absolute error above 0.20 eV from 18.96% to 14.79%. However, the remaining high-error tail motivates deeper case analysis.

## 4. Recommended Fourth Part: Deep Analysis

Speaker 4 should explain where the model still fails and what can be learned from it. Avoid presenting attention as proof.

### Slide D: Special Failure Cases

Use:

| Case Type | Formula | SMILES | Mean Abs Error |
|---|---|---|---:|
| Largest-error failure | CH3BNaO | `[B-]OC.[Na+]` | 0.3580 |
| Halogen failure | ILiO | `[Li+].[O-]I` | 0.2952 |
| Additional high-error failure | BH3INa | `[BH3-]I.[Na+]` | 0.2919 |
| Low-error contrast | Cl2IK | `Cl[I-]Cl.[K+]` | 0.0217 |

Better interpretation:

> The remaining failures are concentrated in chemically unusual ionic species, metal-containing salts, heavy halogen species, and rare local bonding patterns. The low-error contrast case is important because it prevents overclaiming: ionic molecules are not always wrong, but some ionic chemistries remain difficult.

### Slide E: SMILES Robustness Probe

Use:

| Model | Prediction Std Across Randomized SMILES | Randomized RMSE Degradation | All-Variant RMSE |
|---|---:|---:|---:|
| No Aug + Two-stage | 0.0805 | 0.0262 | 0.1662 |
| Aug + Two-stage | 0.0628 | 0.0231 | 0.1493 |

Interpretation:

> This directly supports the augmentation mechanism. The augmented model is less sensitive to equivalent SMILES variants, so its predictions depend less on arbitrary string ordering.

### Slide F: Attention and Limitations

Recommended wording:

> Attention visualization is qualitative evidence, not causal proof. It shows that the model often attends to charge, metal, halogen, oxygen, boron, and nitrogen tokens. But in failure cases, attending to the right tokens is not enough. The likely limitation is how well a SMILES-only pretrained model represents unusual ionic bonding and surface-binding physics.

Limitations to state:

- Dataset is small: 207 molecules.
- Molecules are chemically unusual and often charged or ionic.
- SMILES does not explicitly encode 3D geometry or surface interaction.
- Attention is qualitative, not causal.
- Current prediction files do not directly store CID and SMILES, so some case-level mapping depends on row-order assumptions.

Future improvements:

- Add 3D/geometric descriptors or graph neural networks.
- Use larger and more diverse training data.
- Save metadata directly in prediction outputs.
- Add explicit consistency training across randomized SMILES.

## 5. Suggested Four-Person Final Structure

### Speaker 1: Intro + Data

- Perovskite interface problem.
- Why molecular additive screening matters.
- Supervised regression formulation.
- Dataset: 207 cleaned molecules, SMILES input, `Energy_min` target.

### Speaker 2: Training Process

- ChemBERTa input and regression head.
- Label normalization.
- Two-stage fine-tuning.
- SMILES augmentation.
- Four controlled experiments.

### Speaker 3: Main Results Analysis

- Four experiment table.
- Component contribution analysis.
- Error-tail reliability analysis.

### Speaker 4: Deep Analysis

- Special high-error cases.
- SMILES robustness probe.
- Attention visualization with caution.
- Limitations and future work.

## 6. High-Priority Edits Before Final Presentation

1. Decide whether final story is descriptor models or ChemBERTa. Do not mix them as if they are one pipeline.
2. Change dataset wording from 414 final labeled molecules to 207 cleaned supervised samples.
3. Remove or demote XGBoost/Ridge slides if using ChemBERTa as final method.
4. Replace feature-contribution claims with ablation/component-contribution analysis.
5. Replace molecule-selection claims with either cautious candidate screening or special-case error analysis.
6. Add limitations explicitly, especially small data size and SMILES-only representation.
