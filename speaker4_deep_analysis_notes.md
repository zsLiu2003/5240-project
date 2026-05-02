# Speaker 4 Notes: Deep Analysis, Attention, and Limitations

This section should explain what we learn after the main results. It should not simply repeat the performance table. The goal is to answer:

- Where does the final model still fail?
- Does SMILES augmentation actually improve robustness?
- What does the attention visualization show?
- What are the limitations of the current model?

Recommended time: 3-4 minutes.

## Overall Message

> The final model improves average prediction performance, but deep analysis shows that remaining errors are concentrated in unusual charged ionic chemistry. SMILES augmentation improves robustness to equivalent SMILES views, while attention visualization provides qualitative support but not causal proof.

This section is mainly about the model's boundary conditions: where it works, where it fails, and what should be improved next.

## Recommended Slide Structure

Use 3 slides:

1. Special case analysis: where does the model still fail?
2. SMILES robustness probe: why does augmentation help?
3. Attention visualization and limitations: what does the model attend to, and what can we conclude?

## Slide 1: Special Case Analysis

### Suggested Title

Where Does the Model Still Fail?

### Slide Content

Use a compact table:

| Case type | Formula | SMILES | Mean abs error |
|---|---|---|---:|
| Largest-error failure | CH3BNaO | `[B-]OC.[Na+]` | 0.3580 |
| Halogen failure | ILiO | `[Li+].[O-]I` | 0.2952 |
| High-error failure | BH3INa | `[BH3-]I.[Na+]` | 0.2919 |
| Low-error contrast | Cl2IK | `Cl[I-]Cl.[K+]` | 0.0217 |

Optional short takeaway on the slide:

```text
Remaining errors are chemically structured, not random.
```

### Speaker Notes

After the final model improves average RMSE, we still need to ask where it fails. The largest-error cases are mostly small charged ionic molecules, metal-containing salts, or halogen-containing species.

For example, `[B-]OC.[Na+]` is the largest-error failure, with mean absolute error around 0.358 eV. `[Li+].[O-]I` and `[BH3-]I.[Na+]` are also high-error cases, and they contain formal charge, metal ions, and heavy halogen or boron-containing local structures.

This suggests that the remaining errors are not random. They are concentrated in chemically unusual ionic or charged molecules.

However, we also include the low-error contrast case `Cl[I-]Cl.[K+]`, with mean absolute error around 0.0217 eV. This is important because it prevents overclaiming. We should not say all ionic molecules fail. A more careful conclusion is that some rare charged bonding patterns remain difficult for this SMILES-only model.

### Main Interpretation

Use:

> The remaining failures are chemically structured. They concentrate in unusual ionic, charged, metal-containing, or halogen-containing molecules.

Avoid:

> The model cannot handle ionic molecules.

Better wording:

> The model can handle some ionic molecules, but certain rare charged local environments remain difficult.

### Why This Slide Matters

This slide connects model error to chemistry. It shows that error analysis is not just numerical: it reveals what type of molecules are still difficult for the model.

## Slide 2: SMILES Robustness Probe

### Suggested Title

Does Augmentation Improve SMILES Robustness?

### Slide Content

Use this table:

| Model | Prediction std across randomized SMILES | Randomized RMSE degradation | All-variant RMSE |
|---|---:|---:|---:|
| No Aug + Two-stage | 0.0805 | 0.0262 | 0.1662 |
| Aug + Two-stage | 0.0628 | 0.0231 | 0.1493 |

Optional short takeaway on the slide:

```text
Augmentation reduces sensitivity to equivalent SMILES ordering.
```

### Speaker Notes

This probe directly tests the mechanism behind SMILES augmentation. For the same molecule, we generate equivalent randomized SMILES strings and check whether the model prediction changes.

If augmentation improves representation robustness, then the augmented model should produce more stable predictions for different valid SMILES views of the same molecule.

The result supports this mechanism. The mean prediction standard deviation across randomized SMILES decreases from 0.0805 for the no-augmentation model to 0.0628 for the augmented model. The all-variant RMSE also decreases from 0.1662 to 0.1493.

So augmentation does not only improve the final test metric. It also makes predictions less sensitive to arbitrary SMILES serialization.

### Main Interpretation

Use:

> SMILES augmentation improves robustness to equivalent input views.

Avoid:

> SMILES augmentation creates new molecules.

Better wording:

> Augmentation creates alternative string views for the same molecule and the same supervised label.

### Important Caution

This robustness probe supports the augmentation mechanism, but it is still a post-hoc analysis on matched checkpoints. It should be presented as supporting evidence, not as a separate full cross-validated benchmark.

## Slide 3: Attention Visualization and Limitations

### Suggested Title

Attention Visualization: Qualitative Evidence, Not Causal Proof

### Figure to Use

Recommended figure:

```text
results/deep_analysis/attention_figures/attention_overview_all_cases_numbered_redblue_saved.png
```

This is the current combined attention overview figure.

### How to Explain the Figure

Your current figure is a large overview heatmap. It contains:

- rows: selected molecules or SMILES variants,
- row labels: case IDs and variant type,
- `O`: original SMILES,
- `R`: randomized SMILES variant,
- columns: token positions in the SMILES string,
- red cells: relatively higher attention within that row,
- blue cells: relatively lower attention within that row,
- right side: top-attended tokens for each row,
- colorbar: row-wise attention z-score.

Important explanation:

> This is a row-wise attention z-score. It is useful for identifying which token positions receive relatively high attention within each case. It should not be interpreted as prediction error, binding energy, or a causal chemical explanation.

### Suggested Slide Annotation

Because the figure is information-dense, do not expect the audience to read every row. Add annotations directly on the slide:

```text
O = original SMILES
R = randomized SMILES
Red = higher relative attention
Blue = lower relative attention
```

Add 2-3 callout boxes or arrows for selected cases:

```text
[B-]OC.[Na+]      largest-error failure
[Li+].[O-]I       halogen / ionic failure
Cl[I-]Cl.[K+]     low-error ionic contrast
```

Add one small text box:

```text
Common high-attention tokens:
charge tokens: -], +]
ionic separator: .
hetero / halogen atoms: B, O, N, I, Cl
```

### Speaker Notes

This attention map is used as qualitative evidence for how the final model processes selected success and failure cases.

Each row is one molecule or one randomized SMILES variant. Each column is a token position in the SMILES string. Red means the token receives relatively higher attention within that row, while blue means lower attention. The right side lists top-attended tokens.

The important pattern is that high-attention tokens often correspond to chemically meaningful parts of these SMILES strings, such as formal charge tokens, brackets, dot separators between ionic fragments, metal or halogen-related tokens, boron, oxygen, and nitrogen.

For example, in the high-error ionic cases, the model often attends to charge-related tokens such as `-]` and `+]`, as well as separators like `.` and atoms such as `B`, `O`, `N`, or `I`.

However, this does not prove that the model causally understands the chemistry. In some failure cases, the model attends to chemically important tokens but still predicts the wrong binding energy. Therefore, the issue is not simply whether the model can recognize tokens. The deeper limitation is whether a SMILES-only pretrained model can represent unusual ionic bonding and surface-binding physics.

### Main Interpretation

Use:

> Attention maps are consistent with the model using chemically meaningful tokens, but they are qualitative evidence, not causal proof.

Avoid:

> Attention proves the model understands the chemistry.

Better wording:

> The model may attend to chemically meaningful tokens, but attending to them is not the same as correctly modeling binding energy.

## How to Present the Current Attention Figure

The current attention overview figure is large and dense. It should be used as an overview figure, not as something the audience must read row by row.

### Recommended Presentation Strategy

1. First, explain the axes and colors.
2. Then, point out only 2-3 selected cases.
3. Next, summarize the common high-attention token types.
4. Finally, state the caution: attention is qualitative, not causal.

### Suggested Spoken Walkthrough

Use this order:

1. "Each row is one selected molecule or SMILES variant."
2. "O means original SMILES and R means randomized SMILES."
3. "The x-axis is token position."
4. "Red means relatively higher attention within that molecule."
5. "The top-attended tokens often include charge tokens, ionic separators, and heteroatom or halogen tokens."
6. "This suggests the model is looking at chemically meaningful parts of the SMILES."
7. "But attention is not causal proof. Some failure cases still have high attention on important tokens, so the remaining limitation is representation of unusual ionic chemistry, not just token detection."

### What to Emphasize

The strongest point is:

> The model may look at chemically meaningful tokens, but looking at them is not the same as correctly modeling binding energy.

This sentence helps you avoid overclaiming and makes the analysis more rigorous.

## Suggested Limitations Slide Content

If you have a separate limitations slide, use:

```text
Limitations
- Small dataset: 207 molecules
- Many molecules are charged, ionic, or metal-containing
- SMILES does not explicitly encode 3D geometry or surface interaction
- Attention visualization is qualitative, not causal
- Some case-level mapping assumes prediction order matches the test split

Future Work
- Add graph or 3D molecular features
- Include surface-specific descriptors
- Collect larger and more diverse training data
- Save CID and SMILES directly in prediction outputs
- Add explicit consistency training across randomized SMILES
```

### Speaker Notes for Limitations

The main limitation is that this is a small-data, SMILES-only model for a chemically unusual binding-energy dataset. The model improves robustness and accuracy, but SMILES alone does not explicitly represent 3D geometry, ionic interactions, or surface-binding physics.

Future work should combine SMILES-based learning with graph or 3D molecular features, include surface-specific descriptors, and collect more diverse training data. It would also be useful to save CID and SMILES directly in prediction outputs, so future case-level analysis does not depend on row-order assumptions.

## Recommended Timing

| Slide | Time |
|---|---:|
| Special case analysis | 60-75 sec |
| SMILES robustness probe | 50-60 sec |
| Attention visualization and limitations | 80-100 sec |

Total: about 3.2-3.9 minutes.

If the presentation must be closer to 3 minutes, reduce the attention explanation and combine limitations into the final 20 seconds.

## Full 3-4 Minute Script

Here is a complete possible script:

> After showing the main results, we further analyze where the final model still fails and why augmentation helps.
>
> First, we look at special cases. The largest-error molecules are mostly small charged ionic species, metal-containing salts, or halogen-containing molecules. For example, `[B-]OC.[Na+]` has the largest mean absolute error, around 0.358 eV. `[Li+].[O-]I` and `[BH3-]I.[Na+]` are also high-error cases. This suggests that the remaining errors are chemically structured, not random. However, the low-error contrast case `Cl[I-]Cl.[K+]` shows that not all ionic molecules fail. The more careful conclusion is that some rare charged local environments are difficult for this SMILES-only model.
>
> Next, we test the mechanism behind SMILES augmentation. For the same molecule, we generate equivalent randomized SMILES strings and check how much the prediction changes. The augmented model has lower prediction variation across variants: the mean prediction standard deviation decreases from 0.0805 to 0.0628. The all-variant RMSE also decreases from 0.1662 to 0.1493. This supports the claim that augmentation improves robustness to arbitrary SMILES ordering.
>
> Finally, we use attention visualization as qualitative evidence. In this heatmap, each row is one molecule or one SMILES variant, and each column is a token position. Red indicates relatively higher attention within that row, while blue indicates lower attention. The top-attended tokens often include charge tokens, dot separators between ionic fragments, and atoms such as boron, oxygen, nitrogen, iodine, or chlorine. This is consistent with the model focusing on chemically meaningful parts of the SMILES.
>
> But attention is not causal proof. In some failure cases, the model attends to important charge or halogen tokens but still predicts the wrong binding energy. So the limitation is not only token recognition. The deeper issue is whether a SMILES-only pretrained model can represent unusual ionic bonding and surface-binding physics.
>
> Overall, the deep analysis shows that augmentation and fine-tuning improve ChemBERTa, but unusual charged ionic chemistry remains the main limitation of the current SMILES-only binding-energy model.

## Strong Closing Sentence

> The key takeaway is that augmentation and fine-tuning improve ChemBERTa, but unusual charged ionic chemistry remains the main limitation of a SMILES-only binding-energy model.

