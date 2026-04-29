"""
Attention case study for the final Aug + Two-stage model.

The script loads a saved checkpoint and visualizes CLS attention rollout for
selected SMILES and randomized equivalent variants.
"""

import argparse
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from config import Config
from model import create_model
from smiles_augmentation import randomize_smiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CASES = {
    'charged_oxygen_metal': 'C(=O)[O-].[Na+]',
    'halogen_ionic': '[Li+].[CH2-]Br',
    'heavy_atom_metal': '[K+].I[I-]I',
}


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def attention_rollout(attentions, attention_mask):
    """Average heads and multiply layers with residual attention."""
    keep_len = int(attention_mask.sum().item())
    rollout = None

    for layer_attention in attentions:
        attn = layer_attention[0, :, :keep_len, :keep_len].mean(dim=0).detach().cpu().numpy()
        attn = attn + np.eye(attn.shape[0])
        attn = attn / attn.sum(axis=-1, keepdims=True)
        rollout = attn if rollout is None else attn @ rollout

    return rollout


def tokenize_and_score(model, tokenizer, smiles, device, max_length):
    encoding = tokenizer(
        smiles,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )

    rollout = attention_rollout(outputs.attentions, attention_mask[0])
    keep_len = int(attention_mask[0].sum().item())
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0, :keep_len].detach().cpu().tolist())

    # CLS row after rollout. Remove special tokens for a cleaner plot.
    scores = rollout[0]
    visible = [
        (token, float(score))
        for token, score in zip(tokens, scores)
        if token not in {'<s>', '</s>', '<pad>'}
    ]
    return visible


def plot_case(case_name, variant_rows, output_path):
    max_len = max(len(row['tokens']) for row in variant_rows)
    fig, axes = plt.subplots(
        len(variant_rows),
        1,
        figsize=(max(9, max_len * 0.42), 2.6 * len(variant_rows)),
        squeeze=False,
    )

    for ax, row in zip(axes[:, 0], variant_rows):
        tokens = row['tokens']
        scores = row['scores']
        x = np.arange(len(tokens))
        colors = [
            '#d95f02' if any(mark in token for mark in ['+', '-', 'Br', 'Cl', 'I', 'Na', 'Li', 'K', 'Rb', 'Cs', 'O'])
            else '#1f77b4'
            for token in tokens
        ]
        ax.bar(x, scores, color=colors, edgecolor='black', linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('CLS rollout')
        ax.set_title(f"{case_name} | {row['variant_label']}: {row['smiles']}", fontsize=10)
        ax.grid(axis='y', alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def load_cases(cases_csv=None):
    """Load attention cases from CSV or fall back to built-in examples."""
    if cases_csv is None:
        return CASES

    path = project_path(cases_csv)
    df = pd.read_csv(path)
    if 'SMILES' not in df.columns:
        raise ValueError(f'{path} must contain a SMILES column')

    if 'case' in df.columns:
        names = df['case'].astype(str)
    elif 'CID' in df.columns:
        names = 'cid_' + df['CID'].astype(str)
    else:
        names = [f'case_{idx + 1}' for idx in range(len(df))]

    return dict(zip(names, df['SMILES'].astype(str)))


def markdown_table(df, floatfmt='.6f'):
    """Render a dataframe as a GitHub-style markdown table without tabulate."""
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: format(x, floatfmt) if pd.notna(x) else '')
        else:
            display[col] = display[col].map(lambda x: '' if pd.isna(x) else str(x))

    headers = [str(col) for col in display.columns]
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in display.values.tolist():
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to best_model.pt')
    parser.add_argument('--output-dir', default='results/attention_case_study')
    parser.add_argument('--cases-csv', default=None,
                        help='Optional CSV with columns case and SMILES, e.g. deep analysis candidates.')
    parser.add_argument('--variants', type=int, default=2,
                        help='Number of randomized variants per case, excluding the original.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    checkpoint = project_path(args.checkpoint)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config()
    config.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.ATTN_IMPLEMENTATION = 'eager'
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = create_model(config).to(config.DEVICE)
    state_dict = torch.load(checkpoint, map_location=config.DEVICE)
    model.load_state_dict(state_dict)

    rng = random.Random(args.seed)
    summary_rows = []
    cases = load_cases(args.cases_csv)

    for case_name, smiles in cases.items():
        randomized = randomize_smiles(smiles, n_variants=args.variants, rng=rng)
        variants = [('original', smiles)] + [
            (f'randomized_{idx}', variant)
            for idx, variant in enumerate(randomized, start=1)
        ]

        variant_rows = []
        for variant_label, variant_smiles in variants:
            visible = tokenize_and_score(
                model, tokenizer, variant_smiles, config.DEVICE, config.MAX_LENGTH
            )
            tokens = [token for token, _ in visible]
            scores = [score for _, score in visible]
            top_tokens = sorted(visible, key=lambda x: x[1], reverse=True)[:5]

            variant_rows.append({
                'variant_label': variant_label,
                'smiles': variant_smiles,
                'tokens': tokens,
                'scores': scores,
            })
            for rank, (token, score) in enumerate(top_tokens, start=1):
                summary_rows.append({
                    'case': case_name,
                    'variant': variant_label,
                    'smiles': variant_smiles,
                    'rank': rank,
                    'token': token,
                    'attention_rollout': score,
                })

        plot_path = output_dir / f'{case_name}.png'
        plot_case(case_name, variant_rows, plot_path)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / 'attention_top_tokens.csv', index=False)

    report = [
        '# Attention Case Study',
        '',
        f'Checkpoint: `{checkpoint}`',
        '',
        'Orange bars mark tokens that are likely chemically important in these cases, such as charged atoms, metals, halogens, and oxygen-containing fragments.',
        '',
        '## Generated Figures',
        '',
    ]
    for case_name in cases:
        report.append(f'- `{case_name}`: `{output_dir / f"{case_name}.png"}`')
    report.extend([
        '',
        '## Top-attended Tokens',
        '',
        markdown_table(summary),
        '',
    ])
    (output_dir / 'attention_case_study.md').write_text('\n'.join(report), encoding='utf-8')

    print(f'Wrote figures and report to {output_dir}')


if __name__ == '__main__':
    main()
