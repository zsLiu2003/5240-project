"""
Focused deep analysis for the four final CV experiments.

The analysis is organized around one question:
Why does Aug + Two-stage outperform the alternatives?

Outputs are written to results/deep_analysis/.
"""

import os
from pathlib import Path
import random
import textwrap

os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-ceng5240')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from config import Config
from model import create_model
from smiles_augmentation import randomize_smiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / 'results'
OUTPUT_DIR = RESULTS_DIR / 'deep_analysis'
FIGURES_DIR = OUTPUT_DIR / 'figures'

COLORS = {
    'ink': '#26313B',
    'axis': '#3E4650',
    'grid': '#E8ECF0',
    'neutral': '#BFC7CF',
    'neutral_light': '#E9EDF2',
    'violet': '#8E8BFE',
    'coral': '#E88482',
    'heatmap_low': '#8EC9F2',
    'heatmap_high': '#F2A0A1',
}


def _apply_paper_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'axes.titleweight': 'normal',
        'axes.labelweight': 'normal',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.edgecolor': COLORS['axis'],
        'text.color': COLORS['ink'],
        'axes.labelcolor': COLORS['axis'],
        'xtick.color': COLORS['axis'],
        'ytick.color': COLORS['axis'],
    })


METHODS = {
    'No Aug + Two-stage': {
        'summary': 'main_method_summary.csv',
        'dir_prefix': 'main_method',
    },
    'Aug + Two-stage': {
        'summary': 'main_method_aug0.5_summary.csv',
        'dir_prefix': 'main_method_aug0.5',
    },
    'Aug + Stage1-only': {
        'summary': 'ablation_stage1_only_aug0.5_summary.csv',
        'dir_prefix': 'ablation_stage1_only_aug0.5',
    },
    'Aug + Stage2-only': {
        'summary': 'ablation_stage2_only_aug0.5_summary.csv',
        'dir_prefix': 'ablation_stage2_only_aug0.5',
    },
}


CORE_EFFECTS = [
    {
        'effect': 'SMILES augmentation',
        'baseline': 'No Aug + Two-stage',
        'candidate': 'Aug + Two-stage',
        'interpretation': (
            'Tests whether randomized equivalent SMILES improve robustness to '
            'molecular string representation.'
        ),
    },
    {
        'effect': 'Encoder adaptation',
        'baseline': 'Aug + Stage1-only',
        'candidate': 'Aug + Two-stage',
        'interpretation': (
            'Tests whether frozen ChemBERTa embeddings are sufficient, or whether '
            'the encoder must adapt to binding-energy regression.'
        ),
    },
    {
        'effect': 'Stage 1 warm-up',
        'baseline': 'Aug + Stage2-only',
        'candidate': 'Aug + Two-stage',
        'interpretation': (
            'Tests whether fitting the regression head before encoder fine-tuning '
            'stabilizes task adaptation.'
        ),
    },
]


FINAL_METHOD = 'Aug + Two-stage'
FINAL_CHECKPOINT = RESULTS_DIR / 'main_method_aug0.5_seed42_fold0' / 'best_model.pt'
FINAL_RESULT_JSON = RESULTS_DIR / 'main_method_aug0.5_seed42_fold0' / 'results.json'


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def markdown_table(df, floatfmt='.4f'):
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


def load_summary(method_name):
    path = RESULTS_DIR / METHODS[method_name]['summary']
    if not path.exists():
        raise FileNotFoundError(f'Missing summary for {method_name}: {path}')
    df = pd.read_csv(path)
    required = {'seed', 'fold', 'RMSE', 'MAE', 'R2'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path} missing columns: {sorted(missing)}')
    df = df.copy()
    df['method'] = method_name
    return df


def load_all_summaries():
    return pd.concat([load_summary(name) for name in METHODS], ignore_index=True)


def method_overview(all_runs):
    rows = []
    for method, df in all_runs.groupby('method', sort=False):
        rows.append({
            'method': method,
            'runs': len(df),
            'RMSE_mean': df['RMSE'].mean(),
            'RMSE_std': df['RMSE'].std(ddof=1),
            'MAE_mean': df['MAE'].mean(),
            'MAE_std': df['MAE'].std(ddof=1),
            'R2_mean': df['R2'].mean(),
            'R2_std': df['R2'].std(ddof=1),
            'worst_fold_RMSE': df['RMSE'].max(),
        })
    return pd.DataFrame(rows)


def core_effect_summary(all_runs):
    rows = []
    for spec in CORE_EFFECTS:
        base = all_runs[all_runs['method'] == spec['baseline']]
        cand = all_runs[all_runs['method'] == spec['candidate']]
        merged = base.merge(cand, on=['seed', 'fold'], suffixes=('_base', '_candidate'))
        merged['rmse_improvement'] = merged['RMSE_base'] - merged['RMSE_candidate']
        merged['relative_improvement_pct'] = merged['rmse_improvement'] / merged['RMSE_base'] * 100.0

        rows.append({
            'effect': spec['effect'],
            'baseline': spec['baseline'],
            'candidate': spec['candidate'],
            'baseline_RMSE': merged['RMSE_base'].mean(),
            'candidate_RMSE': merged['RMSE_candidate'].mean(),
            'RMSE_improvement': merged['rmse_improvement'].mean(),
            'relative_improvement_pct': (
                (merged['RMSE_base'].mean() - merged['RMSE_candidate'].mean())
                / merged['RMSE_base'].mean()
                * 100.0
            ),
            'folds_improved': int((merged['rmse_improvement'] > 0).sum()),
            'folds_worse': int((merged['rmse_improvement'] < 0).sum()),
            'best_fold_gain_pct': merged['relative_improvement_pct'].max(),
            'worst_fold_gain_pct': merged['relative_improvement_pct'].min(),
            'interpretation': spec['interpretation'],
        })
    return pd.DataFrame(rows)


def test_rows_for_seed(seed):
    split_path = PROJECT_ROOT / 'data' / f'split_seed{seed}.csv'
    df = pd.read_csv(split_path)
    return df[df['split'] == 'test'].reset_index(drop=True)


def load_predictions(method_name):
    spec = METHODS[method_name]
    rows = []
    for _, run in load_summary(method_name).iterrows():
        seed = int(run['seed'])
        fold = int(run['fold'])
        pred_path = RESULTS_DIR / f"{spec['dir_prefix']}_seed{seed}_fold{fold}" / 'predictions.csv'
        if not pred_path.exists():
            raise FileNotFoundError(f'Missing predictions: {pred_path}')
        pred_df = pd.read_csv(pred_path).reset_index(drop=True)
        meta_df = test_rows_for_seed(seed)
        if len(pred_df) != len(meta_df):
            raise ValueError(
                f'Prediction/test mismatch for {pred_path}: '
                f'{len(pred_df)} predictions vs {len(meta_df)} test rows'
            )
        joined = pd.concat([meta_df, pred_df], axis=1)
        joined['method'] = method_name
        joined['seed'] = seed
        joined['fold'] = fold
        joined['abs_error'] = joined['error'].abs()
        rows.append(joined)
    return pd.concat(rows, ignore_index=True)


def load_all_predictions():
    return pd.concat([load_predictions(method) for method in METHODS], ignore_index=True)


def error_tail_summary(predictions):
    rows = []
    for method, df in predictions.groupby('method', sort=False):
        abs_error = df['abs_error']
        rows.append({
            'method': method,
            'n_predictions': len(df),
            'mean_abs_error': abs_error.mean(),
            'median_abs_error': abs_error.median(),
            'q75_abs_error': abs_error.quantile(0.75),
            'q90_abs_error': abs_error.quantile(0.90),
            'max_abs_error': abs_error.max(),
            'pct_abs_error_gt_0.20': (abs_error > 0.20).mean() * 100.0,
        })
    return pd.DataFrame(rows)


def add_case_flags(df):
    out = df.copy()
    smiles = out['SMILES'].astype(str)
    formula = out['Formula'].astype(str)
    out['has_metal'] = smiles.str.contains(r'Li|Na|K|Rb|Cs|Mg|Ca|Zn|Cu|Fe', regex=True)
    out['has_halogen'] = smiles.str.contains(r'Cl|Br|I|F', regex=True) | formula.str.contains(r'Cl|Br|I|F', regex=True)
    out['has_formal_charge'] = smiles.str.contains(r'\+|-', regex=True)
    out['has_oxygen'] = smiles.str.contains('O', regex=False) | formula.str.contains('O', regex=False)
    return out


def summarize_final_cases(final_predictions):
    group_cols = [
        'seed', 'CID', 'Formula', 'SMILES', 'MW', 'HBA', 'HBD', 'TPSA',
        'Heavy_Atoms', 'Complexity', 'Energy_min',
    ]
    case_summary = (
        final_predictions
        .groupby(group_cols, dropna=False)
        .agg(
            true=('true', 'mean'),
            pred_mean=('pred', 'mean'),
            pred_std=('pred', 'std'),
            error_mean=('error', 'mean'),
            abs_error_mean=('abs_error', 'mean'),
            abs_error_max=('abs_error', 'max'),
            folds=('fold', 'count'),
        )
        .reset_index()
    )
    return add_case_flags(case_summary).sort_values('abs_error_mean', ascending=False)


def select_cases(case_summary, max_cases=8):
    selected = []

    def add(row, case_type, motivation):
        if any(item['CID'] == row['CID'] for item in selected):
            return
        selected.append({
            'case_id': f"cid_{int(row['CID'])}_{len(selected) + 1}",
            'case_type': case_type,
            'motivation': motivation,
            'seed': int(row['seed']),
            'CID': row['CID'],
            'Formula': row['Formula'],
            'SMILES': row['SMILES'],
            'true': row['true'],
            'pred_mean': row['pred_mean'],
            'pred_std': row['pred_std'],
            'abs_error_mean': row['abs_error_mean'],
            'has_metal': row['has_metal'],
            'has_halogen': row['has_halogen'],
            'has_formal_charge': row['has_formal_charge'],
            'has_oxygen': row['has_oxygen'],
        })

    add(
        case_summary.iloc[0],
        'largest-error failure',
        'Shows the strongest remaining limitation of the final method.',
    )

    category_specs = [
        ('charged metal failure', case_summary['has_metal'] & case_summary['has_formal_charge'],
         'Tests whether residual failures concentrate in ionic metal-containing species.'),
        ('halogen failure', case_summary['has_halogen'],
         'Tests whether heavy halogen-containing molecules remain difficult.'),
        ('oxygen-containing failure', case_summary['has_oxygen'],
         'Tests whether charged/polar oxygen fragments are a recurring failure mode.'),
    ]
    for case_type, mask, motivation in category_specs:
        subset = case_summary[mask]
        if not subset.empty:
            add(subset.iloc[0], case_type, motivation)

    for _, row in case_summary.head(12).iterrows():
        if len(selected) >= max_cases - 1:
            break
        add(row, 'additional high-error failure', 'Adds coverage of the high-error tail.')

    low_error_pool = case_summary[
        case_summary['has_formal_charge'] | case_summary['has_metal'] | case_summary['has_halogen']
    ].sort_values('abs_error_mean', ascending=True)
    if low_error_pool.empty:
        low_error_pool = case_summary.sort_values('abs_error_mean', ascending=True)
    add(
        low_error_pool.iloc[0],
        'low-error success contrast',
        'Provides a contrast case showing that not all ionic or unusual molecules fail.',
    )

    return pd.DataFrame(selected).head(max_cases)


def load_final_model():
    if not FINAL_CHECKPOINT.exists():
        raise FileNotFoundError(f'Missing final checkpoint: {FINAL_CHECKPOINT}')

    config = Config()
    config.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.ATTN_IMPLEMENTATION = 'eager'
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = create_model(config).to(config.DEVICE)
    state_dict = torch.load(FINAL_CHECKPOINT, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    result = pd.read_json(FINAL_RESULT_JSON) if FINAL_RESULT_JSON.exists() else None
    if FINAL_RESULT_JSON.exists():
        import json
        with open(FINAL_RESULT_JSON) as f:
            loaded = json.load(f)
        label_stats = loaded['label_stats']
    else:
        split = pd.read_csv(PROJECT_ROOT / 'data' / 'split_seed42.csv')
        train = split[split['split'] == 'train']
        label_stats = {'mean': train['Energy_min'].mean(), 'std': train['Energy_min'].std()}

    return model, tokenizer, config, label_stats


def predict_smiles(model, tokenizer, config, label_stats, smiles):
    encoding = tokenizer(
        smiles,
        max_length=config.MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(config.DEVICE)
    attention_mask = encoding['attention_mask'].to(config.DEVICE)
    with torch.no_grad():
        pred_norm = model(input_ids, attention_mask).detach().cpu().numpy()[0]
    return float(pred_norm * label_stats['std'] + label_stats['mean'])


def smiles_consistency_probe(selected_cases, model, tokenizer, config, label_stats, n_variants=6):
    rng = random.Random(5240)
    rows = []
    for _, case in selected_cases.iterrows():
        smiles = case['SMILES']
        variants = [smiles] + randomize_smiles(smiles, n_variants=n_variants, rng=rng)
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)

        predictions = [
            predict_smiles(model, tokenizer, config, label_stats, variant)
            for variant in unique_variants
        ]
        for idx, (variant, pred) in enumerate(zip(unique_variants, predictions)):
            rows.append({
                'case_id': case['case_id'],
                'case_type': case['case_type'],
                'CID': case['CID'],
                'variant_idx': idx,
                'SMILES_variant': variant,
                'prediction': pred,
                'true': case['true'],
                'abs_error': abs(pred - case['true']),
                'n_unique_variants': len(unique_variants),
                'variant_prediction_std': float(np.std(predictions, ddof=0)),
                'variant_prediction_range': float(np.max(predictions) - np.min(predictions)),
            })
    return pd.DataFrame(rows)


def attention_rollout(attentions, attention_mask):
    keep_len = int(attention_mask.sum().item())
    rollout = None
    for layer_attention in attentions:
        attn = layer_attention[0, :, :keep_len, :keep_len].mean(dim=0).detach().cpu().numpy()
        attn = attn + np.eye(attn.shape[0])
        attn = attn / attn.sum(axis=-1, keepdims=True)
        rollout = attn if rollout is None else attn @ rollout
    return rollout


def attention_scores(model, tokenizer, config, smiles):
    encoding = tokenizer(
        smiles,
        max_length=config.MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(config.DEVICE)
    attention_mask = encoding['attention_mask'].to(config.DEVICE)
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
    scores = rollout[0]
    return [
        (token, float(score))
        for token, score in zip(tokens, scores)
        if token not in {'<s>', '</s>', '<pad>'}
    ]


def _display_token(token):
    return token.replace('Ġ', '').replace('▁', '')


def _is_chemically_salient(token):
    markers = ['+', '-', 'Br', 'Cl', 'I', 'Na', 'Li', 'K', 'Rb', 'Cs', 'O', 'B', 'N']
    return any(mark in token for mark in markers)


def _attention_cmap():
    cmap = LinearSegmentedColormap.from_list(
        'light_sky_white_red',
        [(0.0, COLORS['heatmap_low']), (0.5, '#FFFFFF'), (1.0, COLORS['heatmap_high'])],
    )
    cmap.set_bad('#FAFBFC')
    return cmap


def _zscore_norm(values):
    vmax = max(1.0, np.nanpercentile(np.abs(values), 96))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    return norm, vmax


def _row_zscore(values):
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(values)
    return (values - mean) / std


def plot_attention_case(case_id, rows, output_path):
    _apply_paper_style()
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.titlesize': 11,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
    })
    max_len = max(len(row['tokens']) for row in rows)
    fig, axes = plt.subplots(
        len(rows),
        1,
        figsize=(max(9.5, max_len * 0.42), 1.55 * len(rows) + 1.35),
        squeeze=False,
        constrained_layout=True,
    )
    fig.patch.set_facecolor('white')
    fig.suptitle(
        f'Attention rollout for {case_id}',
        fontsize=9,
        fontweight='normal',
        color=COLORS['ink'],
    )
    cmap = _attention_cmap()
    z_rows = [_row_zscore(row['scores']) for row in rows]
    norm, vmax = _zscore_norm(np.concatenate(z_rows))

    for ax, row, z_scores in zip(axes[:, 0], rows, z_rows):
        tokens = row['tokens']
        scores = np.asarray(row['scores'])
        salient = [_is_chemically_salient(token) for token in tokens]
        top_idx = set(np.argsort(scores)[-3:])

        ax.imshow(z_scores[np.newaxis, :], cmap=cmap, norm=norm, aspect='auto', extent=(-0.5, len(tokens) - 0.5, 0, 1))
        for idx, is_salient in enumerate(salient):
            if is_salient:
                ax.add_patch(Rectangle(
                    (idx - 0.5, 0),
                    1,
                    1,
                    fill=False,
                    edgecolor=COLORS['coral'],
                    linewidth=0.8,
                ))
        for idx in top_idx:
            ax.plot(idx, 1.12, marker='v', markersize=3.8, color=COLORS['violet'], clip_on=False)
            ax.text(
                idx,
                1.24,
                f'{scores[idx]:.3f}',
                ha='center',
                va='bottom',
                fontsize=6.5,
                color=COLORS['axis'],
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
                label.set_color(COLORS['coral'])
                label.set_fontweight('bold')
        ax.set_yticks([])
        ax.set_ylabel(
            row['variant_label'].replace('_', ' ').title(),
            color=COLORS['ink'],
            fontweight='bold',
            fontsize=9,
            labelpad=8,
        )
        ax.set_title(
            textwrap.fill(row['smiles'], width=120),
            loc='left',
            fontsize=8,
            fontweight='normal',
            color=COLORS['axis'],
            pad=6,
        )
        ax.set_xlim(-0.5, len(tokens) - 0.5)
        ax.set_ylim(-0.08, 1.36)
        ax.set_facecolor('white')
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.tick_params(axis='x', length=0, pad=4, labelsize=8)

    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes[:, 0],
        location='right',
        shrink=0.62,
        pad=0.015,
    )
    cbar.set_label('Row z-score', fontsize=7, color=COLORS['axis'])
    cbar.set_ticks([-vmax, 0, vmax])
    cbar.set_ticklabels([f'{-vmax:.1f}', '0', f'{vmax:.1f}'])
    cbar.ax.tick_params(labelsize=7, length=2, colors=COLORS['axis'])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_attention_overview(rows, output_path):
    _apply_paper_style()
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 7,
    })
    max_len = max(len(row['tokens']) for row in rows)
    matrix = np.full((len(rows), max_len), np.nan)
    for row_idx, row in enumerate(rows):
        scores = np.asarray(row['scores'])
        matrix[row_idx, :len(scores)] = scores

    cmap = _attention_cmap().copy()
    z_matrix = np.vstack([_row_zscore(row) for row in matrix])
    filled = np.nan_to_num(z_matrix, nan=0.0)
    if len(rows) > 1:
        row_order = dendrogram(linkage(filled, method='average', metric='euclidean'), no_plot=True)['leaves']
    else:
        row_order = [0]
    ordered_rows = [rows[idx] for idx in row_order]
    ordered_z = z_matrix[row_order]
    norm, vmax = _zscore_norm(ordered_z)

    fig = plt.figure(
        figsize=(max(9.2, max_len * 0.34 + 3.6), max(5.2, len(rows) * 0.40 + 1.3)),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(1, 3, width_ratios=[max_len * 0.34, 1.55, 0.08], wspace=0.035)
    ax = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1], sharey=ax)
    cax = fig.add_subplot(gs[0, 2])
    fig.patch.set_facecolor('white')

    ax.set_facecolor('white')
    ax.imshow(
        ordered_z,
        cmap=cmap,
        norm=norm,
        aspect='auto',
        interpolation='nearest',
    )

    ylabels = []
    top_text_rows = []
    variant_colors = []
    for row_idx, row in enumerate(ordered_rows):
        scores = np.asarray(row['scores'])
        tokens = row['tokens']
        top_idx = list(np.argsort(scores)[-2:][::-1])
        for token_idx in top_idx:
            z_value = ordered_z[row_idx, token_idx]
            ax.add_patch(Rectangle(
                (token_idx - 0.5, row_idx - 0.5),
                1,
                1,
                fill=False,
                edgecolor=COLORS['ink'],
                linewidth=0.45,
                alpha=0.55,
            ))
            ax.text(
                token_idx,
                row_idx,
                f'z={z_value:+.1f}',
                ha='center',
                va='center',
                fontsize=5.0,
                color=COLORS['ink'],
                alpha=0.88,
            )
        top_text_rows.append(
            ', '.join(f'{_display_token(tokens[idx])} {scores[idx]:.3f}' for idx in top_idx)
        )
        short_id = row['case_id'].replace('cid_', '')
        if row['variant_label'] == 'original':
            ylabels.append(f'{short_id}  O')
            variant_colors.append(COLORS['violet'])
        else:
            ylabels.append(f'{short_id}  R')
            variant_colors.append(COLORS['coral'])

    ax.set_yticks(np.arange(len(ordered_rows)))
    ax.set_yticklabels(ylabels)
    for tick_label, color in zip(ax.get_yticklabels(), variant_colors):
        tick_label.set_color(color)
    ax.set_xticks(np.arange(0, max_len, 2))
    ax.set_xlabel('Token position')
    ax.set_title('Attention z-score by case and variant', loc='left', color=COLORS['ink'])
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.tick_params(axis='both', length=0, colors=COLORS['axis'])
    for tick_label, color in zip(ax.get_yticklabels(), variant_colors):
        tick_label.set_color(color)

    ax_text.set_facecolor('white')
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(ax.get_ylim())
    ax_text.axis('off')
    ax_text.text(
        0.0,
        -0.72,
        'Top attention tokens (raw score)',
        ha='left',
        va='center',
        fontsize=7.5,
        color=COLORS['ink'],
    )
    for row_idx, top_text in enumerate(top_text_rows):
        ax_text.text(0.0, row_idx, top_text, ha='left', va='center', fontsize=6.5, color=COLORS['axis'])

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label('Row z-score', fontsize=7, color=COLORS['axis'])
    cbar.set_ticks([-vmax, 0, vmax])
    cbar.set_ticklabels([f'{-vmax:.1f}', '0', f'{vmax:.1f}'])
    cbar.ax.tick_params(labelsize=7, length=2, colors=COLORS['axis'])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def attention_visualization(selected_cases, model, tokenizer, config):
    figures_dir = OUTPUT_DIR / 'attention_figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(5240)
    top_rows = []
    figure_rows = []
    overview_rows = []

    for _, case in selected_cases.iterrows():
        variants = [('original', case['SMILES'])]
        randomized = randomize_smiles(case['SMILES'], n_variants=1, rng=rng)
        if randomized and randomized[0] != case['SMILES']:
            variants.append(('randomized_1', randomized[0]))

        plot_rows = []
        for variant_label, smiles in variants:
            visible = attention_scores(model, tokenizer, config, smiles)
            tokens = [token for token, _ in visible]
            scores = [score for _, score in visible]
            plot_rows.append({
                'case_id': case['case_id'],
                'case_type': case['case_type'],
                'variant_label': variant_label,
                'smiles': smiles,
                'tokens': tokens,
                'scores': scores,
            })
            for rank, (token, score) in enumerate(sorted(visible, key=lambda x: x[1], reverse=True)[:5], start=1):
                top_rows.append({
                    'case_id': case['case_id'],
                    'case_type': case['case_type'],
                    'variant': variant_label,
                    'SMILES': smiles,
                    'rank': rank,
                    'token': token,
                    'attention_rollout': score,
                })

        overview_rows.extend(plot_rows)

    overview_path = figures_dir / 'attention_overview_all_cases.png'
    plot_attention_overview(overview_rows, overview_path)
    figure_rows.append({
        'case_id': 'all_selected_cases',
        'case_type': 'combined overview',
        'figure': str(overview_path),
    })

    return pd.DataFrame(top_rows), pd.DataFrame(figure_rows)


def save_core_effect_plot(core_df):
    _apply_paper_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = core_df.sort_values('relative_improvement_pct', ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.8, 3.0),
        gridspec_kw={'width_ratios': [1.05, 1.45]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor('white')

    ax_rmse, ax_gain = axes
    for ax in axes:
        ax.set_facecolor('white')
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(0.6)
        ax.tick_params(length=2.5, width=0.6, colors=COLORS['axis'])

    for idx, row in plot_df.iterrows():
        ax_rmse.hlines(
            y=idx,
            xmin=row['candidate_RMSE'],
            xmax=row['baseline_RMSE'],
            color=COLORS['neutral'],
            linewidth=1.0,
            zorder=1,
        )
    ax_rmse.scatter(plot_df['baseline_RMSE'], y, s=30, color=COLORS['coral'], label='Baseline', zorder=3)
    ax_rmse.scatter(plot_df['candidate_RMSE'], y, s=30, color=COLORS['violet'], label='Aug + Two-stage', zorder=3)
    ax_rmse.set_yticks(y)
    ax_rmse.set_yticklabels(plot_df['effect'])
    ax_rmse.set_xlabel('RMSE')
    ax_rmse.set_title('Paired RMSE', loc='left')
    ax_rmse.grid(axis='x', color=COLORS['grid'], linewidth=0.6)
    ax_rmse.legend(frameon=False, fontsize=7, loc='lower right')

    bar_colors = [COLORS['violet'] if idx % 2 == 0 else COLORS['coral'] for idx in range(len(plot_df))]
    ax_gain.barh(
        y,
        plot_df['relative_improvement_pct'],
        height=0.22,
        color=bar_colors,
        edgecolor='white',
        linewidth=0.8,
        zorder=2,
    )
    ax_gain.scatter(plot_df['relative_improvement_pct'], y, s=22, color=COLORS['ink'], zorder=4)
    ax_gain.axvline(0, color=COLORS['axis'], linewidth=0.8)
    ax_gain.set_yticks(y)
    ax_gain.set_yticklabels([])
    ax_gain.set_xlabel('Relative RMSE improvement (%)')
    ax_gain.set_title('Effect size', loc='left')
    ax_gain.grid(axis='x', color=COLORS['grid'], linewidth=0.6, zorder=0)
    xmax = max(plot_df['relative_improvement_pct'].max() * 1.16, 5)
    ax_gain.set_xlim(0, xmax)
    for idx, value in enumerate(plot_df['relative_improvement_pct']):
        ax_gain.text(value + xmax * 0.018, idx, f'{value:.1f}%', va='center', fontsize=7, color=COLORS['axis'])

    path = FIGURES_DIR / 'core_effects.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def _method_colors(methods):
    palette = [COLORS['violet'], COLORS['coral']]
    return {method: palette[idx % len(palette)] for idx, method in enumerate(methods)}


def save_method_performance_plot(all_runs, method_df):
    _apply_paper_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    methods = method_df['method'].tolist()
    colors = _method_colors(methods)
    metrics = [('RMSE', 'Lower is better'), ('MAE', 'Lower is better'), ('R2', 'Higher is better')]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), constrained_layout=True)
    fig.patch.set_facecolor('white')

    rng = np.random.default_rng(5240)
    for ax, (metric, subtitle) in zip(axes, metrics):
        ax.set_facecolor('white')
        for idx, method in enumerate(methods):
            values = all_runs.loc[all_runs['method'] == method, metric].to_numpy()
            jitter = rng.normal(0, 0.045, size=len(values))
            ax.scatter(
                np.full(len(values), idx) + jitter,
                values,
                s=16,
                color=colors[method],
                alpha=0.38,
                edgecolor='none',
            )
            mean = method_df.loc[method_df['method'] == method, f'{metric}_mean'].iloc[0]
            std = method_df.loc[method_df['method'] == method, f'{metric}_std'].iloc[0]
            ax.errorbar(
                idx,
                mean,
                yerr=std,
                fmt='D',
                markersize=5,
                color=COLORS['ink'],
                ecolor=COLORS['ink'],
                elinewidth=1.0,
                capsize=3,
                zorder=4,
            )
        ax.set_title(f'{metric}\n{subtitle}', loc='left', fontsize=8.5, fontweight='normal', color=COLORS['ink'])
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=35, ha='right')
        ax.grid(axis='y', color=COLORS['grid'], linewidth=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(0.6)
        ax.tick_params(axis='y', length=2.5, width=0.6, colors=COLORS['axis'])
        ax.tick_params(axis='x', length=2.5, width=0.6, colors=COLORS['axis'])
    path = FIGURES_DIR / 'method_performance.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def save_error_tail_plot(tail_df):
    _apply_paper_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    methods = tail_df['method'].tolist()
    colors = _method_colors(methods)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6), constrained_layout=True)
    fig.patch.set_facecolor('white')

    quantile_cols = [
        ('median_abs_error', 'Median'),
        ('q75_abs_error', 'Q75'),
        ('q90_abs_error', 'Q90'),
        ('max_abs_error', 'Max'),
    ]
    x = np.arange(len(quantile_cols))
    for _, row in tail_df.iterrows():
        axes[0].plot(
            x,
            [row[col] for col, _ in quantile_cols],
            marker='o',
            linewidth=1.6,
            markersize=4,
            color=colors[row['method']],
            label=row['method'],
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([label for _, label in quantile_cols])
    axes[0].set_ylabel('Absolute error (eV)')
    axes[0].set_title('Error distribution tail', loc='left')

    y = np.arange(len(methods))
    axes[1].barh(
        y,
        tail_df['pct_abs_error_gt_0.20'],
        color=[colors[method] for method in methods],
        edgecolor='#ffffff',
    )
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(methods)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Predictions with abs error > 0.20 eV (%)')
    axes[1].set_title('Large-error rate', loc='left')
    for idx, value in enumerate(tail_df['pct_abs_error_gt_0.20']):
        axes[1].text(value + 0.4, idx, f'{value:.1f}%', va='center', fontsize=7, color=COLORS['axis'])

    for ax in axes:
        ax.set_facecolor('white')
        ax.grid(axis='x' if ax is axes[1] else 'y', color=COLORS['grid'], linewidth=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(0.6)
        ax.tick_params(length=2.5, width=0.6, colors=COLORS['axis'])
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    path = FIGURES_DIR / 'error_tail_summary.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def save_smiles_consistency_plot(consistency_df):
    _apply_paper_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary = (
        consistency_df
        .groupby(['case_id', 'case_type', 'CID'], as_index=False)
        .agg(
            true=('true', 'first'),
            pred_min=('prediction', 'min'),
            pred_max=('prediction', 'max'),
            prediction_std=('variant_prediction_std', 'max'),
            prediction_range=('variant_prediction_range', 'max'),
        )
        .sort_values('prediction_range', ascending=True)
    )
    y_positions = {case_id: idx for idx, case_id in enumerate(summary['case_id'])}
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for _, row in summary.iterrows():
        y = y_positions[row['case_id']]
        ax.hlines(y, row['pred_min'], row['pred_max'], color=COLORS['violet'], linewidth=2.0, alpha=0.72)
        ax.plot(row['true'], y, marker='|', markersize=13, color=COLORS['coral'], markeredgewidth=1.6)
        variants = consistency_df[consistency_df['case_id'] == row['case_id']]
        ax.scatter(variants['prediction'], np.full(len(variants), y), s=18, color=COLORS['violet'], alpha=0.9, zorder=3)
        ax.text(row['pred_max'] + 0.025, y, f'range {row["prediction_range"]:.3f}', va='center', fontsize=6.5, color=COLORS['axis'])

    ax.set_yticks(np.arange(len(summary)))
    ax.set_yticklabels([case_id.replace('cid_', '') for case_id in summary['case_id']])
    ax.set_xlabel('Predicted / true energy (eV)')
    ax.set_title('SMILES consistency across randomized equivalent strings', loc='left')
    ax.grid(axis='x', color=COLORS['grid'], linewidth=0.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.6)
    ax.tick_params(length=2.5, width=0.6, colors=COLORS['axis'])
    ax.text(0.01, 0.02, 'Dots: predictions for SMILES variants; vertical ticks: true labels.',
            transform=ax.transAxes, fontsize=7, color=COLORS['axis'])
    path = FIGURES_DIR / 'smiles_consistency.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def save_special_case_plot(selected_cases):
    _apply_paper_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cases = selected_cases.sort_values('abs_error_mean', ascending=True)
    y = np.arange(len(cases))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3), gridspec_kw={'width_ratios': [1.4, 1.0]}, constrained_layout=True)
    fig.patch.set_facecolor('white')

    axes[0].hlines(y, cases['true'], cases['pred_mean'], color=COLORS['violet'], linewidth=1.8, alpha=0.72)
    axes[0].scatter(cases['true'], y, marker='|', s=110, color=COLORS['coral'], linewidth=1.8, label='True')
    axes[0].scatter(cases['pred_mean'], y, s=20, color=COLORS['violet'], alpha=0.95, label='Pred mean')
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([case_id.replace('cid_', '') for case_id in cases['case_id']])
    axes[0].set_xlabel('Energy (eV)')
    axes[0].set_title('True vs predicted mean', loc='left')
    axes[0].legend(frameon=False, fontsize=8, loc='lower right')

    axes[1].barh(y, cases['abs_error_mean'], color=COLORS['coral'], edgecolor='white')
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel('Mean absolute error (eV)')
    axes[1].set_title('Selected-case error', loc='left')
    for idx, value in enumerate(cases['abs_error_mean']):
        axes[1].text(value + 0.006, idx, f'{value:.3f}', va='center', fontsize=7, color=COLORS['axis'])

    for ax in axes:
        ax.set_facecolor('white')
        ax.grid(axis='x', color=COLORS['grid'], linewidth=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(0.6)
        ax.tick_params(length=2.5, width=0.6, colors=COLORS['axis'])
    path = FIGURES_DIR / 'special_case_errors.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def save_visual_summaries(all_runs, method_df, tail_df, selected_cases, consistency_df, core_df):
    figure_rows = [
        {'figure': str(save_core_effect_plot(core_df)), 'description': 'Relative RMSE improvement for the three core effects.'},
    ]
    return pd.DataFrame(figure_rows)


def write_markdown_outputs(method_df, core_df, tail_df, selected_cases,
                           consistency_df, attention_top_df, attention_figures,
                           visual_figures):
    final_tail = tail_df[tail_df['method'] == FINAL_METHOD].iloc[0]

    a_report = [
        '# A. Mechanistic Results Analysis',
        '',
        '## Core Insight',
        '',
        'The final method works because it combines two complementary mechanisms: '
        'SMILES augmentation improves robustness to molecular string representation, '
        'while Stage 2 fine-tuning adapts ChemBERTa features to binding-energy regression. '
        'Stage 1 warm-up adds a smaller stabilization effect before encoder updates.',
        '',
        '## A1. Core Result Interpretation',
        '',
        markdown_table(visual_figures),
        '',
        markdown_table(core_df[
            ['effect', 'baseline_RMSE', 'candidate_RMSE', 'relative_improvement_pct',
             'folds_improved', 'folds_worse']
        ]),
        '',
        '## A2. Error-tail Experiment',
        '',
        'This post-hoc experiment checks whether the final gain comes from reducing difficult '
        'large-error predictions rather than only shifting the average.',
        '',
        markdown_table(tail_df),
        '',
        f"For the final method, Q90 absolute error is {final_tail['q90_abs_error']:.4f} eV "
        f"and {final_tail['pct_abs_error_gt_0.20']:.1f}% of predictions exceed 0.20 eV absolute error.",
        '',
        '## A3. SMILES Consistency Probe',
        '',
        'This probe evaluates the same molecule under randomized equivalent SMILES strings. '
        'Lower prediction variance supports the intended augmentation mechanism: the model '
        'should be less dependent on a single SMILES serialization.',
        '',
        markdown_table(
            consistency_df.groupby(['case_id', 'case_type', 'CID'], as_index=False)
            .agg(
                n_variants=('SMILES_variant', 'count'),
                prediction_std=('variant_prediction_std', 'max'),
                prediction_range=('variant_prediction_range', 'max'),
                mean_abs_error=('abs_error', 'mean'),
            )
        ),
        '',
    ]
    (OUTPUT_DIR / 'results_level_deep_analysis.md').write_text('\n'.join(a_report), encoding='utf-8')

    b_report = [
        '# B. Special Case Analysis',
        '',
        '## Motivation',
        '',
        'After the final method improves average performance, the next question is where it still fails. '
        'The selected cases test whether remaining errors are random or concentrated in chemically unusual '
        'small ionic molecules such as charged metal salts, halogen-containing species, and polar fragments.',
        '',
        markdown_table(selected_cases[
            ['case_id', 'case_type', 'motivation', 'Formula', 'SMILES', 'true',
             'pred_mean', 'abs_error_mean']
        ]),
        '',
        '## Interpretation',
        '',
        'The high-error cases are dominated by formally charged, metal-containing, or heavy-atom species. '
        'The low-error contrast case is included to avoid overclaiming: the model can handle some ionic '
        'molecules, but its remaining failures are concentrated in uncommon local chemical environments.',
        '',
    ]
    (OUTPUT_DIR / 'special_case_analysis.md').write_text('\n'.join(b_report), encoding='utf-8')

    c_report = [
        '# C. Attention Visualization',
        '',
        '## Motivation',
        '',
        'Attention rollout is used as qualitative evidence for how the final model processes selected '
        'success and failure cases. It should not be interpreted as a causal explanation, but it can show '
        'whether chemically important tokens such as charges, metals, halogens, oxygen, boron, or nitrogen '
        'receive high attention.',
        '',
        '## Figures',
        '',
        markdown_table(attention_figures),
        '',
        '## Top-attended Tokens',
        '',
        markdown_table(attention_top_df),
        '',
    ]
    (OUTPUT_DIR / 'attention_interpretation.md').write_text('\n'.join(c_report), encoding='utf-8')

    final_report = [
        '# Deep Analysis Report',
        '',
        '## Main Message',
        '',
        'The four-experiment comparison suggests that the final method succeeds through a combination of '
        'representation robustness and task adaptation. SMILES augmentation makes the model less tied to one '
        'string form of a molecule; Stage 2 fine-tuning is necessary to adapt ChemBERTa to binding-energy '
        'regression; Stage 1 warm-up provides a smaller stabilization effect.',
        '',
        '## Method Overview',
        '',
        'The main quantitative results are visualized in `results/deep_analysis/figures/`.',
        '',
        markdown_table(visual_figures),
        '',
        markdown_table(method_df),
        '',
        '## Core Effects',
        '',
        markdown_table(core_df[
            ['effect', 'baseline', 'candidate', 'relative_improvement_pct',
             'folds_improved', 'folds_worse']
        ]),
        '',
        '## Error Tail',
        '',
        markdown_table(tail_df),
        '',
        '## Selected Special Cases',
        '',
        markdown_table(selected_cases[
            ['case_id', 'case_type', 'Formula', 'SMILES', 'true', 'pred_mean', 'abs_error_mean']
        ]),
        '',
        '## SMILES Consistency',
        '',
        markdown_table(
            consistency_df.groupby(['case_id', 'case_type'], as_index=False)
            .agg(
                n_variants=('SMILES_variant', 'count'),
                prediction_std=('variant_prediction_std', 'max'),
                prediction_range=('variant_prediction_range', 'max'),
            )
        ),
        '',
        '## Attention Evidence',
        '',
        'Attention figures are saved in `results/deep_analysis/attention_figures/`, and top-attended tokens '
        'are saved in `attention_top_tokens.csv`. These visualizations are qualitative support for the case '
        'analysis rather than proof of causal model reasoning.',
        '',
        '## Assumption',
        '',
        'The case-level mapping assumes each `predictions.csv` preserves the same row order as the corresponding '
        '`test` split in `data/split_seed*.csv`. Future runs should save `CID` and `SMILES` directly in '
        '`predictions.csv` to remove this assumption.',
        '',
    ]
    (OUTPUT_DIR / 'deep_analysis_report.md').write_text('\n'.join(final_report), encoding='utf-8')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_runs = load_all_summaries()
    method_df = method_overview(all_runs)
    core_df = core_effect_summary(all_runs)
    all_predictions = load_all_predictions()
    tail_df = error_tail_summary(all_predictions)
    final_predictions = all_predictions[all_predictions['method'] == FINAL_METHOD]
    case_summary = summarize_final_cases(final_predictions)
    selected_cases = select_cases(case_summary)

    model, tokenizer, config, label_stats = load_final_model()
    consistency_df = smiles_consistency_probe(selected_cases, model, tokenizer, config, label_stats)
    attention_top_df, attention_figures = attention_visualization(selected_cases, model, tokenizer, config)
    visual_figures = save_visual_summaries(all_runs, method_df, tail_df, selected_cases, consistency_df, core_df)

    method_df.to_csv(OUTPUT_DIR / 'method_overview.csv', index=False)
    core_df.to_csv(OUTPUT_DIR / 'core_effect_summary.csv', index=False)
    tail_df.to_csv(OUTPUT_DIR / 'error_tail_summary.csv', index=False)
    case_summary.to_csv(OUTPUT_DIR / 'special_case_summary.csv', index=False)
    selected_cases.to_csv(OUTPUT_DIR / 'selected_attention_cases.csv', index=False)
    consistency_df.to_csv(OUTPUT_DIR / 'smiles_consistency_probe.csv', index=False)
    attention_top_df.to_csv(OUTPUT_DIR / 'attention_top_tokens.csv', index=False)
    attention_figures.to_csv(OUTPUT_DIR / 'attention_figures.csv', index=False)
    visual_figures.to_csv(OUTPUT_DIR / 'visual_figures.csv', index=False)

    write_markdown_outputs(
        method_df, core_df, tail_df, selected_cases,
        consistency_df, attention_top_df, attention_figures,
        visual_figures,
    )

    print(f'Wrote deep analysis outputs to {OUTPUT_DIR}')
    print(f'Visual figures: {FIGURES_DIR}')


if __name__ == '__main__':
    main()
