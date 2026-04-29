"""
Focused deep analysis for the four final CV experiments.

The analysis is organized around one question:
Why does Aug + Two-stage outperform the alternatives?

Outputs are written to results/deep_analysis/.
"""

import os
from pathlib import Path
import random

os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-ceng5240')

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
RESULTS_DIR = PROJECT_ROOT / 'results'
OUTPUT_DIR = RESULTS_DIR / 'deep_analysis'


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


def plot_attention_case(case_id, rows, output_path):
    fig, axes = plt.subplots(
        len(rows),
        1,
        figsize=(max(8, max(len(row['tokens']) for row in rows) * 0.45), 2.5 * len(rows)),
        squeeze=False,
    )
    markers = ['+', '-', 'Br', 'Cl', 'I', 'Na', 'Li', 'K', 'Rb', 'Cs', 'O', 'B', 'N']
    for ax, row in zip(axes[:, 0], rows):
        x = np.arange(len(row['tokens']))
        colors = [
            '#d95f02' if any(mark in token for mark in markers) else '#1f77b4'
            for token in row['tokens']
        ]
        ax.bar(x, row['scores'], color=colors, edgecolor='black', linewidth=0.35)
        ax.set_xticks(x)
        ax.set_xticklabels(row['tokens'], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('CLS rollout')
        ax.set_title(f"{case_id} | {row['variant_label']}: {row['smiles']}", fontsize=10)
        ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def attention_visualization(selected_cases, model, tokenizer, config):
    figures_dir = OUTPUT_DIR / 'attention_figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(5240)
    top_rows = []
    figure_rows = []

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

        figure_path = figures_dir / f"{case['case_id']}.png"
        plot_attention_case(case['case_id'], plot_rows, figure_path)
        figure_rows.append({
            'case_id': case['case_id'],
            'case_type': case['case_type'],
            'figure': str(figure_path),
        })

    return pd.DataFrame(top_rows), pd.DataFrame(figure_rows)


def save_core_effect_plot(core_df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(core_df))
    ax.bar(x, core_df['relative_improvement_pct'], color=['#4c78a8', '#f58518', '#54a24b'])
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(core_df['effect'], rotation=12, ha='right')
    ax.set_ylabel('Relative RMSE improvement (%)')
    ax.set_title('Core Effects Behind Aug + Two-stage')
    for idx, value in enumerate(core_df['relative_improvement_pct']):
        ax.text(idx, value + 0.6, f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    path = OUTPUT_DIR / 'core_effects.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def write_markdown_outputs(method_df, core_df, tail_df, selected_cases,
                           consistency_df, attention_top_df, attention_figures):
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
    core_plot_path = save_core_effect_plot(core_df)

    method_df.to_csv(OUTPUT_DIR / 'method_overview.csv', index=False)
    core_df.to_csv(OUTPUT_DIR / 'core_effect_summary.csv', index=False)
    tail_df.to_csv(OUTPUT_DIR / 'error_tail_summary.csv', index=False)
    case_summary.to_csv(OUTPUT_DIR / 'special_case_summary.csv', index=False)
    selected_cases.to_csv(OUTPUT_DIR / 'selected_attention_cases.csv', index=False)
    consistency_df.to_csv(OUTPUT_DIR / 'smiles_consistency_probe.csv', index=False)
    attention_top_df.to_csv(OUTPUT_DIR / 'attention_top_tokens.csv', index=False)
    attention_figures.to_csv(OUTPUT_DIR / 'attention_figures.csv', index=False)

    write_markdown_outputs(
        method_df, core_df, tail_df, selected_cases,
        consistency_df, attention_top_df, attention_figures,
    )

    print(f'Wrote deep analysis outputs to {OUTPUT_DIR}')
    print(f'Core effects plot: {core_plot_path}')


if __name__ == '__main__':
    main()
