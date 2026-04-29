"""
SMILES robustness probe for comparing no-augmentation and augmentation models.

This is a post-hoc analysis experiment. It does not train a model.

Question:
Does SMILES augmentation make the model less sensitive to equivalent randomized
SMILES strings for the same molecule?

Expected matched inputs:
- No Aug + Two-stage checkpoint, e.g. results/main_method_seed42_fold0/best_model.pt
- Aug + Two-stage checkpoint, e.g. results/main_method_aug0.5_seed42_fold0/best_model.pt
- Their corresponding results.json files for label normalization stats
"""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import AutoTokenizer

from config import Config
from model import create_model
from smiles_augmentation import randomize_smiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_label_stats(results_json):
    results_json = project_path(results_json)
    if not results_json.exists():
        raise FileNotFoundError(f'Missing results.json with label stats: {results_json}')
    with open(results_json) as f:
        result = json.load(f)
    return {
        'mean': float(result['label_stats']['mean']),
        'std': float(result['label_stats']['std']),
    }


def load_model(checkpoint_path, device):
    checkpoint_path = project_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Missing checkpoint: {checkpoint_path}')

    config = Config()
    config.DEVICE = device
    model = create_model(config).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


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


def load_test_set(data_dir, seed):
    split_path = project_path(data_dir) / f'split_seed{seed}.csv'
    if not split_path.exists():
        raise FileNotFoundError(f'Missing split file: {split_path}')
    df = pd.read_csv(split_path)
    return df[df['split'] == 'test'].reset_index(drop=True)


def sample_cases(test_df, max_cases, seed):
    if max_cases is None or max_cases >= len(test_df):
        return test_df.copy()
    return test_df.sample(n=max_cases, random_state=seed).reset_index(drop=True)


def unique_variants(smiles, n_variants, rng):
    variants = [smiles] + randomize_smiles(smiles, n_variants=n_variants, rng=rng)
    seen = set()
    out = []
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            out.append(variant)
    return out


def evaluate_model_on_variants(model_name, model, tokenizer, config, label_stats,
                               case_df, n_variants, rng):
    rows = []
    for _, row in case_df.iterrows():
        smiles = row[config.SMILES_COL]
        target = float(row[config.LABEL_COL])
        variants = unique_variants(smiles, n_variants=n_variants, rng=rng)
        predictions = [
            predict_smiles(model, tokenizer, config, label_stats, variant)
            for variant in variants
        ]

        pred_std = float(np.std(predictions, ddof=0))
        pred_range = float(np.max(predictions) - np.min(predictions))
        original_pred = predictions[0]

        for idx, (variant, pred) in enumerate(zip(variants, predictions)):
            rows.append({
                'model': model_name,
                'CID': row.get('CID', np.nan),
                'Formula': row.get('Formula', ''),
                'original_SMILES': smiles,
                'variant_idx': idx,
                'is_original': idx == 0,
                'SMILES_variant': variant,
                'true': target,
                'pred': pred,
                'error': pred - target,
                'abs_error': abs(pred - target),
                'original_pred': original_pred,
                'variant_delta_from_original': pred - original_pred,
                'n_unique_variants': len(variants),
                'prediction_std_across_variants': pred_std,
                'prediction_range_across_variants': pred_range,
            })
    return pd.DataFrame(rows)


def compute_metrics(targets, preds):
    return {
        'RMSE': float(mean_squared_error(targets, preds) ** 0.5),
        'MAE': float(mean_absolute_error(targets, preds)),
        'R2': float(r2_score(targets, preds)),
    }


def summarize_model(df):
    original_df = df[df['is_original']]
    randomized_df = df[~df['is_original']]
    all_metrics = compute_metrics(df['true'], df['pred'])
    original_metrics = compute_metrics(original_df['true'], original_df['pred'])
    randomized_metrics = (
        compute_metrics(randomized_df['true'], randomized_df['pred'])
        if not randomized_df.empty else {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}
    )

    per_case = (
        df.groupby(['model', 'CID', 'original_SMILES'], dropna=False)
        .agg(
            prediction_std=('prediction_std_across_variants', 'max'),
            prediction_range=('prediction_range_across_variants', 'max'),
            mean_abs_variant_delta=('variant_delta_from_original', lambda x: float(np.mean(np.abs(x)))),
            max_abs_variant_delta=('variant_delta_from_original', lambda x: float(np.max(np.abs(x)))),
            n_unique_variants=('SMILES_variant', 'count'),
        )
        .reset_index()
    )

    return {
        'model': df['model'].iloc[0],
        'n_cases': int(original_df.shape[0]),
        'n_variant_rows': int(df.shape[0]),
        'original_RMSE': original_metrics['RMSE'],
        'original_MAE': original_metrics['MAE'],
        'randomized_RMSE': randomized_metrics['RMSE'],
        'randomized_MAE': randomized_metrics['MAE'],
        'all_variants_RMSE': all_metrics['RMSE'],
        'all_variants_MAE': all_metrics['MAE'],
        'RMSE_degradation_randomized_minus_original': randomized_metrics['RMSE'] - original_metrics['RMSE'],
        'mean_prediction_std_across_variants': per_case['prediction_std'].mean(),
        'median_prediction_std_across_variants': per_case['prediction_std'].median(),
        'q90_prediction_std_across_variants': per_case['prediction_std'].quantile(0.90),
        'mean_prediction_range_across_variants': per_case['prediction_range'].mean(),
        'q90_prediction_range_across_variants': per_case['prediction_range'].quantile(0.90),
        'mean_abs_variant_delta': per_case['mean_abs_variant_delta'].mean(),
        'q90_max_abs_variant_delta': per_case['max_abs_variant_delta'].quantile(0.90),
    }


def markdown_table(df, floatfmt='.4f'):
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: format(x, floatfmt) if pd.notna(x) else '')
        else:
            display[col] = display[col].map(lambda x: '' if pd.isna(x) else str(x))

    lines = [
        '| ' + ' | '.join(display.columns.astype(str)) + ' |',
        '| ' + ' | '.join(['---'] * len(display.columns)) + ' |',
    ]
    for row in display.values.tolist():
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def write_report(summary_df, per_case_delta_df, output_dir, args):
    aug_row = summary_df[summary_df['model'] == 'Aug + Two-stage']
    no_aug_row = summary_df[summary_df['model'] == 'No Aug + Two-stage']

    interpretation = []
    if not aug_row.empty and not no_aug_row.empty:
        aug_std = aug_row.iloc[0]['mean_prediction_std_across_variants']
        no_aug_std = no_aug_row.iloc[0]['mean_prediction_std_across_variants']
        aug_degradation = aug_row.iloc[0]['RMSE_degradation_randomized_minus_original']
        no_aug_degradation = no_aug_row.iloc[0]['RMSE_degradation_randomized_minus_original']

        if aug_std < no_aug_std:
            interpretation.append(
                f'Augmentation reduces mean prediction std across randomized SMILES '
                f'({no_aug_std:.4f} -> {aug_std:.4f}), supporting the SMILES robustness claim.'
            )
        else:
            interpretation.append(
                f'Augmentation does not reduce mean prediction std in this probe '
                f'({no_aug_std:.4f} -> {aug_std:.4f}); the robustness claim should be stated cautiously.'
            )

        if aug_degradation < no_aug_degradation:
            interpretation.append(
                f'Augmentation also reduces RMSE degradation under randomized SMILES '
                f'({no_aug_degradation:.4f} -> {aug_degradation:.4f}).'
            )
        else:
            interpretation.append(
                f'RMSE degradation under randomized SMILES is not lower for the augmented model '
                f'({no_aug_degradation:.4f} -> {aug_degradation:.4f}).'
            )

    report = [
        '# SMILES Robustness Probe',
        '',
        '## Motivation',
        '',
        'This supplemental experiment directly tests the mechanism behind SMILES augmentation. '
        'If augmentation improves representation robustness, the augmented model should produce '
        'more stable predictions for equivalent randomized SMILES strings of the same molecule.',
        '',
        '## Setup',
        '',
        f'- Split seed: `{args.seed}`',
        f'- Randomized variants per molecule: `{args.n_variants}`',
        f'- Max cases: `{args.max_cases if args.max_cases is not None else "all test cases"}`',
        f'- No-aug checkpoint: `{project_path(args.no_aug_checkpoint)}`',
        f'- Aug checkpoint: `{project_path(args.aug_checkpoint)}`',
        '',
        '## Summary',
        '',
        markdown_table(summary_df),
        '',
        '## Interpretation',
        '',
    ]
    report.extend([f'- {line}' for line in interpretation])
    report.extend([
        '',
        '## Largest No-Aug vs Aug Robustness Gaps',
        '',
        markdown_table(per_case_delta_df.head(12)),
        '',
        '## Caution',
        '',
        'This probe uses matched checkpoints and randomized SMILES variants. It supports or weakens '
        'the augmentation mechanism claim, but it is still a post-hoc inference experiment rather '
        'than a new cross-validated training result.',
        '',
    ])
    (output_dir / 'smiles_robustness_report.md').write_text('\n'.join(report), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Compare SMILES robustness for no-aug vs aug checkpoints.')
    parser.add_argument('--no-aug-checkpoint', required=True)
    parser.add_argument('--aug-checkpoint', required=True)
    parser.add_argument('--no-aug-results-json', default=None)
    parser.add_argument('--aug-results-json', default=None)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-variants', type=int, default=8)
    parser.add_argument('--max-cases', type=int, default=None,
                        help='Limit number of test molecules for a quicker probe. Default: all test molecules.')
    parser.add_argument('--output-dir', default='results/smiles_robustness_probe')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--random-seed', type=int, default=5240)
    args = parser.parse_args()

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    no_aug_results_json = args.no_aug_results_json or str(project_path(args.no_aug_checkpoint).with_name('results.json'))
    aug_results_json = args.aug_results_json or str(project_path(args.aug_checkpoint).with_name('results.json'))

    no_aug_label_stats = load_label_stats(no_aug_results_json)
    aug_label_stats = load_label_stats(aug_results_json)

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    no_aug_model, no_aug_config = load_model(args.no_aug_checkpoint, args.device)
    aug_model, aug_config = load_model(args.aug_checkpoint, args.device)

    test_df = load_test_set(args.data_dir, args.seed)
    case_df = sample_cases(test_df, args.max_cases, args.random_seed)
    rng = random.Random(args.random_seed)

    no_aug_df = evaluate_model_on_variants(
        'No Aug + Two-stage',
        no_aug_model,
        tokenizer,
        no_aug_config,
        no_aug_label_stats,
        case_df,
        args.n_variants,
        rng,
    )
    aug_df = evaluate_model_on_variants(
        'Aug + Two-stage',
        aug_model,
        tokenizer,
        aug_config,
        aug_label_stats,
        case_df,
        args.n_variants,
        rng,
    )
    variant_predictions = pd.concat([no_aug_df, aug_df], ignore_index=True)

    summary_df = pd.DataFrame([
        summarize_model(no_aug_df),
        summarize_model(aug_df),
    ])

    per_case = (
        variant_predictions
        .groupby(['model', 'CID', 'Formula', 'original_SMILES'], dropna=False)
        .agg(
            prediction_std=('prediction_std_across_variants', 'max'),
            prediction_range=('prediction_range_across_variants', 'max'),
            mean_abs_variant_delta=('variant_delta_from_original', lambda x: float(np.mean(np.abs(x)))),
            max_abs_variant_delta=('variant_delta_from_original', lambda x: float(np.max(np.abs(x)))),
        )
        .reset_index()
    )
    no_aug_case = per_case[per_case['model'] == 'No Aug + Two-stage']
    aug_case = per_case[per_case['model'] == 'Aug + Two-stage']
    per_case_delta = no_aug_case.merge(
        aug_case,
        on=['CID', 'Formula', 'original_SMILES'],
        suffixes=('_no_aug', '_aug'),
    )
    per_case_delta['std_reduction_aug_vs_no_aug'] = (
        per_case_delta['prediction_std_no_aug'] - per_case_delta['prediction_std_aug']
    )
    per_case_delta['range_reduction_aug_vs_no_aug'] = (
        per_case_delta['prediction_range_no_aug'] - per_case_delta['prediction_range_aug']
    )
    per_case_delta = per_case_delta.sort_values('std_reduction_aug_vs_no_aug', ascending=False)

    variant_predictions.to_csv(output_dir / 'smiles_variant_predictions.csv', index=False)
    summary_df.to_csv(output_dir / 'smiles_robustness_summary.csv', index=False)
    per_case_delta.to_csv(output_dir / 'per_case_robustness_delta.csv', index=False)
    write_report(summary_df, per_case_delta, output_dir, args)

    print(f'Wrote SMILES robustness probe outputs to {output_dir}')


if __name__ == '__main__':
    main()
