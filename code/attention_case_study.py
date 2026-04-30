"""
Attention case study for the final Aug + Two-stage model.

The script loads a saved checkpoint and visualizes CLS attention rollout for
selected SMILES and randomized equivalent variants.
"""

import argparse
from pathlib import Path
import random
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
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


def _display_token(token):
    return token.replace('Ġ', '').replace('▁', '')


def _is_chemically_salient(token):
    markers = ['+', '-', 'Br', 'Cl', 'I', 'Na', 'Li', 'K', 'Rb', 'Cs', 'O', 'B', 'N']
    return any(mark in token for mark in markers)


def _attention_cmap():
    return LinearSegmentedColormap.from_list(
        'muted_attention',
        ['#f7f7f2', '#d8ddd8', '#9fb3b2', '#607f8e', '#2f4b5c'],
    )


def plot_case(case_name, variant_rows, output_path):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.titlesize': 11,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
    })
    max_len = max(len(row['tokens']) for row in variant_rows)
    fig, axes = plt.subplots(
        len(variant_rows),
        1,
        figsize=(max(9.5, max_len * 0.42), 1.55 * len(variant_rows) + 1.35),
        squeeze=False,
        constrained_layout=True,
    )
    fig.patch.set_facecolor('#fbfbf8')
    fig.suptitle(
        f'Attention rollout for {case_name}',
        fontsize=12,
        fontweight='bold',
        color='#252a31',
    )
    cmap = _attention_cmap()
    max_score = max(float(np.max(row['scores'])) for row in variant_rows)
    norm = Normalize(vmin=0.0, vmax=max_score)

    for ax, row in zip(axes[:, 0], variant_rows):
        tokens = row['tokens']
        scores = np.asarray(row['scores'])
        salient = [_is_chemically_salient(token) for token in tokens]
        top_idx = set(np.argsort(scores)[-3:])

        ax.imshow(scores[np.newaxis, :], cmap=cmap, norm=norm, aspect='auto', extent=(-0.5, len(tokens) - 0.5, 0, 1))
        for idx, is_salient in enumerate(salient):
            if is_salient:
                ax.add_patch(Rectangle(
                    (idx - 0.5, 0),
                    1,
                    1,
                    fill=False,
                    edgecolor='#8a6f4d',
                    linewidth=1.1,
                ))
        for idx in top_idx:
            ax.plot(idx, 1.12, marker='v', markersize=4.5, color='#5b6470', clip_on=False)
            ax.text(
                idx,
                1.24,
                f'{scores[idx]:.3f}',
                ha='center',
                va='bottom',
                fontsize=7,
                color='#4b5563',
            )

        ax.set_xticks(np.arange(len(tokens)))
        ax.set_xticklabels(
            [_display_token(token) for token in tokens],
            rotation=45,
            ha='right',
            rotation_mode='anchor',
        )
        for label, is_salient in zip(ax.get_xticklabels(), salient):
            if is_salient:
                label.set_color('#6f5d49')
                label.set_fontweight('bold')
        ax.set_yticks([])
        ax.set_ylabel(
            row['variant_label'].replace('_', ' ').title(),
            color='#252a31',
            fontweight='bold',
            fontsize=9,
            labelpad=8,
        )
        ax.set_title(
            textwrap.fill(row['smiles'], width=120),
            loc='left',
            fontsize=8,
            fontweight='normal',
            color='#667085',
            pad=6,
        )
        ax.set_xlim(-0.5, len(tokens) - 0.5)
        ax.set_ylim(-0.08, 1.36)
        ax.set_facecolor('#fbfbf8')
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.tick_params(axis='x', length=0, pad=4, labelsize=8)

    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes[:, 0],
        location='right',
        shrink=0.62,
        pad=0.015,
    )
    cbar.set_label('CLS rollout', fontsize=8, color='#4b5563')
    cbar.ax.tick_params(labelsize=7, length=2, colors='#6b7280')
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
