# Speaker 2 Notes: Model and Training Strategy

This section should explain how we train ChemBERTa for supervised binding-energy regression, and how each training component is later tested by ablation experiments.

Recommended time: 3-4 minutes.

## Overall Message

> We use ChemBERTa to learn from molecular SMILES strings, then adapt it to binding-energy prediction using supervised two-stage fine-tuning and SMILES augmentation.

This part is about method design, not final results. The main goal is to explain:

- why ChemBERTa is used,
- why two-stage fine-tuning is used,
- why SMILES augmentation is used,
- how the ablation experiments test each design choice.

## Slide 1: ChemBERTa for Supervised Molecular Regression

### Suggested Title

ChemBERTa for SMILES-Based Binding Energy Regression

### Slide Content

```text
Input: SMILES string
Target: Energy_min binding energy
Model: ChemBERTa encoder + regression head
Task type: supervised regression
```

### Suggested Visual

```text
SMILES
  ↓
Tokenizer
  ↓
ChemBERTa Encoder
  ↓
[CLS] Molecular Representation
  ↓
Regression Head
  ↓
Predicted Binding Energy
```

### Speaker Notes

Our input is a molecular SMILES string, and our target is the continuous binding energy label, Energy_min. We use ChemBERTa because it is pretrained on molecular SMILES and can provide useful molecular string representations. On top of the ChemBERTa encoder, we add a regression head to predict binding energy.

So the task is supervised molecular regression: pretrained representations provide the starting point, and our labeled energy data teaches the task-specific mapping.

### Important Wording

Use:

> ChemBERTa provides pretrained molecular SMILES representations.

Avoid:

> ChemBERTa directly understands physical binding mechanisms.

## Slide 2: Two-Stage Supervised Fine-Tuning

### Suggested Title

Two-Stage Supervised Fine-Tuning

### Slide Content

```text
Stage 1:
Freeze ChemBERTa encoder
Train only the regression head

Stage 2:
Unfreeze ChemBERTa encoder
Fine-tune encoder + regression head

Learning rates:
smaller LR for ChemBERTa backbone
larger LR for regression head
```

### Suggested Visual

```text
Stage 1
Locked ChemBERTa Encoder + Trainable Regression Head
        ↓
Stage 2
Unlocked ChemBERTa Encoder + Trainable Regression Head
```

You can use lock/unlock icons to show frozen and unfrozen encoder states.

### Speaker Notes

Because our dataset is small, we avoid updating the whole pretrained model from the beginning. In Stage 1, the encoder is frozen, so the regression head first learns the supervised energy scale from fixed pretrained features.

In Stage 2, we unfreeze the encoder and fine-tune the full model, using a smaller learning rate for ChemBERTa and a larger learning rate for the regression head. This balances training stability with task-specific adaptation.

### Important Wording

Use:

> Stage 1 learns the supervised energy scale; Stage 2 adapts the molecular representation to the binding-energy task.

Avoid saying:

> Two-stage fine-tuning is always better.

The later results show that it is helpful for this project.

## Slide 3: SMILES Augmentation for Robustness

### Suggested Title

SMILES Augmentation for Representation Robustness

### Slide Content

```text
One molecule can have multiple valid SMILES strings.

During training:
canonical SMILES → randomized equivalent SMILES

The binding-energy label stays unchanged.

Goal:
reduce sensitivity to arbitrary SMILES ordering
```

### Suggested Visual

```text
SMILES variant 1  ┐
SMILES variant 2  ├→ same molecule → same Energy_min label
SMILES variant 3  ┘
```

### Speaker Notes

The same molecule can be written as multiple valid SMILES strings. During training, we randomly replace the canonical SMILES with equivalent randomized SMILES.

Importantly, this does not create new molecules or new labels. It creates alternative input views for the same supervised target. The goal is to encourage the model to learn molecule-level structure rather than memorize one specific SMILES order.

### Important Wording

Use:

> Augmentation changes the input view, not the supervised target.

Avoid:

> SMILES augmentation creates new molecules.

## Slide 4: Controlled Experiments and Ablations

### Suggested Title

Controlled Experiments and Ablations

### Slide Content

| Method | Purpose |
|---|---|
| No Aug + Two-stage | Baseline without SMILES augmentation |
| Aug + Two-stage | Final method |
| Aug + Stage1-only | Tests whether frozen ChemBERTa features are sufficient |
| Aug + Stage2-only | Tests whether Stage 1 warm-up is useful |

Additional bullets:

```text
3 seeds × 5 folds = 15 runs per method

No Aug vs Aug:
effect of SMILES augmentation

Stage1-only vs Two-stage:
need for encoder adaptation

Stage2-only vs Two-stage:
role of Stage 1 warm-up
```

### Speaker Notes

These four experiments are designed to isolate the role of each training component. No Aug versus Aug tests whether randomized SMILES improves robustness. Stage1-only versus two-stage tests whether frozen pretrained ChemBERTa features are enough. Stage2-only versus two-stage tests whether the Stage 1 warm-up helps before full fine-tuning.

The final comparison uses 3 seeds and 5 folds, so each method is evaluated over 15 runs.

### Transition to Speaker 3

> With this design, the next section can evaluate not only which model performs best, but also why each component matters.

## Recommended Timing

| Slide | Time |
|---|---:|
| ChemBERTa model | 45-55 sec |
| Two-stage fine-tuning | 55-65 sec |
| SMILES augmentation | 45-55 sec |
| Experimental design | 45-55 sec |

Total: about 3.2-3.8 minutes.

## What to Remove From the Current PPT

For this section, remove or avoid using the current slides that focus on:

- Linear Regression,
- MLP,
- Random Forest,
- XGBoost,
- descriptor feature contribution,
- molecule recommendation as final model output,
- device performance as direct model prediction.

These topics do not match the current final project story. The final project story is ChemBERTa-based supervised binding-energy regression.

## Things to Avoid During Presentation

- Do not present XGBoost, Random Forest, Ridge, or MLP as the main method.
- Do not say SMILES augmentation creates new molecules.
- Do not say ChemBERTa proves or directly understands physical binding mechanisms.
- Do not overclaim that the model explains device performance.
- Do not present descriptor feature importance as the main interpretation for the ChemBERTa method.

## Strong Closing Sentence

> In summary, our training strategy combines pretrained SMILES representations, supervised two-stage fine-tuning, and SMILES augmentation. The following results section tests whether each of these components actually improves binding-energy prediction.
