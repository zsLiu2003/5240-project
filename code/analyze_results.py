"""
结果分析脚本

对比所有实验结果，生成清晰的对比分析和可视化。
重点展示：
1. 消融实验：验证两阶段训练的必要性
2. 数据增强效果：对比增强前后的性能
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _style_axis(ax, grid_axis='y'):
    ax.set_facecolor('#fbfbf8')
    ax.grid(axis=grid_axis, color='#d6d3cb', linewidth=0.7, alpha=0.7)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(length=0, colors='#667085')


def _method_palette(methods):
    colors = ['#607d8b', '#8d7b68', '#7a8f75', '#8b8996', '#6f8f9f', '#9a8f6f', '#778899']
    return {method: colors[idx % len(colors)] for idx, method in enumerate(methods)}


def _metric_dataframe(data):
    rows = []
    for method, metrics in data.items():
        rows.append({
            'method': method,
            'RMSE_mean': metrics['rmse_mean'],
            'RMSE_std': 0.0 if np.isnan(metrics['rmse_std']) else metrics['rmse_std'],
            'MAE_mean': metrics['mae_mean'],
            'MAE_std': 0.0 if np.isnan(metrics['mae_std']) else metrics['mae_std'],
            'R2_mean': metrics['r2_mean'],
            'R2_std': 0.0 if np.isnan(metrics['r2_std']) else metrics['r2_std'],
            'n_runs': metrics['n_runs'],
            'source': metrics.get('source', ''),
        })
    return pd.DataFrame(rows)


def _summarize_metrics(df, source):
    return {
        'rmse_mean': df['RMSE'].mean(),
        'rmse_std': df['RMSE'].std(),
        'mae_mean': df['MAE'].mean(),
        'mae_std': df['MAE'].std(),
        'r2_mean': df['R2'].mean(),
        'r2_std': df['R2'].std(),
        'n_runs': len(df),
        'source': source,
    }


def _load_seed_result_dirs(results_dir, experiment_name):
    """Load non-CV seed result dirs, excluding *_fold* directories."""
    rows = []
    for result_path in sorted(results_dir.glob(f'{experiment_name}_seed*')):
        if '_fold' in result_path.name:
            continue
        json_path = result_path / 'results.json'
        if not json_path.exists():
            continue
        with open(json_path) as f:
            result = json.load(f)
        rows.append({
            'seed': result.get('seed'),
            'RMSE': result['test_metrics']['RMSE'],
            'MAE': result['test_metrics']['MAE'],
            'R2': result['test_metrics']['R2'],
        })
    return pd.DataFrame(rows)


def load_experiment_results(results_dir=None):
    """加载所有实验结果"""
    results_dir = Path(results_dir) if results_dir is not None else PROJECT_ROOT / 'results'

    experiments = {
        'ECFP + Ridge': 'baseline_results.csv',
        'Full Finetune': 'ablation1_full_finetune_results.csv',
        'From Scratch': 'ablation2_from_scratch_results.csv',
        'Stage1-only': 'ablation_stage1_only_results.csv',
        'Stage2-only': 'ablation_stage2_only_results.csv',
        'Main Method (CV)': 'main_method_summary.csv',
    }

    data = {}
    for name, filename in experiments.items():
        filepath = results_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            data[name] = _summarize_metrics(df, filename)
        else:
            print(f'Warning: {filename} not found, skipping {name}')

    main_seed_df = _load_seed_result_dirs(results_dir, 'main_method')
    if not main_seed_df.empty:
        data['Main Method'] = _summarize_metrics(main_seed_df, 'main_method_seed*/results.json')

    aug_seed_df = _load_seed_result_dirs(results_dir, 'main_method_aug0.5')
    if not aug_seed_df.empty:
        data['Main + Aug (0.5)'] = _summarize_metrics(
            aug_seed_df, 'main_method_aug0.5_seed*/results.json'
        )

    # 自动加载增强实验结果（例如 main_method_aug0.5_summary.csv）
    for filepath in sorted(results_dir.glob('*_aug*_summary.csv')):
        df = pd.read_csv(filepath)
        if {'RMSE', 'MAE', 'R2'}.issubset(df.columns):
            stem = filepath.stem.replace('_summary', '')
            if stem.startswith('main_method_aug'):
                prob = stem.replace('main_method_aug', '')
                name = f'Main + Aug ({prob}, CV)' if 'fold' in df.columns else f'Main + Aug ({prob})'
            else:
                name = stem.replace('_', ' ').title()
            data[name] = _summarize_metrics(df, filepath.name)

    return data


def print_comparison_table(data):
    """打印对比表格"""
    print('\n' + '='*90)
    print('实验结果对比')
    print('='*90)
    print(f'{"方法":<25} {"RMSE":<20} {"MAE":<20} {"R²":<20} {"运行次数":<10}')
    print('-'*90)

    for name, metrics in data.items():
        rmse_str = f"{metrics['rmse_mean']:.4f} ± {metrics['rmse_std'] if not np.isnan(metrics['rmse_std']) else 0.0:.4f}"
        mae_str = f"{metrics['mae_mean']:.4f} ± {metrics['mae_std'] if not np.isnan(metrics['mae_std']) else 0.0:.4f}"
        r2_str = f"{metrics['r2_mean']:.4f} ± {metrics['r2_std'] if not np.isnan(metrics['r2_std']) else 0.0:.4f}"
        print(f'{name:<25} {rmse_str:<20} {mae_str:<20} {r2_str:<20} {metrics["n_runs"]:<10}')
    print('='*90)


def analyze_ablation(data):
    """分析消融实验"""
    print('\n' + '='*90)
    print('消融实验分析')
    print('='*90)

    if 'Main Method' in data and 'Full Finetune' in data:
        main_rmse = data['Main Method']['rmse_mean']
        full_rmse = data['Full Finetune']['rmse_mean']
        diff = ((main_rmse - full_rmse) / full_rmse) * 100
        print(f'\n1. Main Method vs Full Finetune:')
        print(f'   Main Method RMSE: {main_rmse:.4f}')
        print(f'   Full Finetune RMSE: {full_rmse:.4f}')
        print(f'   差异: {diff:+.2f}%')
        if diff > 0:
            print(f'   → Main Method表现较差，可能需要优化两阶段策略')
        else:
            print(f'   → Main Method表现更好，验证了两阶段训练的有效性')

    if 'Main Method' in data and 'Stage1-only' in data:
        main_rmse = data['Main Method']['rmse_mean']
        stage1_rmse = data['Stage1-only']['rmse_mean']
        diff = ((stage1_rmse - main_rmse) / main_rmse) * 100
        print(f'\n2. Stage1-only vs Main Method:')
        print(f'   Stage1-only RMSE: {stage1_rmse:.4f}')
        print(f'   Main Method RMSE: {main_rmse:.4f}')
        print(f'   改进: {diff:+.2f}%')
        if diff > 0:
            print(f'   → Stage2微调带来了{diff:.2f}%的性能提升')
        else:
            print(f'   → Stage2微调未带来改进，需要检查配置')

    if 'Main Method' in data and 'Stage2-only' in data:
        main_rmse = data['Main Method']['rmse_mean']
        stage2_rmse = data['Stage2-only']['rmse_mean']
        diff = ((stage2_rmse - main_rmse) / main_rmse) * 100
        print(f'\n3. Stage2-only vs Main Method:')
        print(f'   Stage2-only RMSE: {stage2_rmse:.4f}')
        print(f'   Main Method RMSE: {main_rmse:.4f}')
        print(f'   改进: {diff:+.2f}%')
        if diff > 0:
            print(f'   → Stage1预热带来了{diff:.2f}%的性能提升')
        else:
            print(f'   → Stage1预热未带来改进，需要检查配置')


def analyze_augmentation(data):
    """分析数据增强效果"""
    print('\n' + '='*90)
    print('数据增强效果分析')
    print('='*90)

    comparisons = [
        ('Main Method', 'Main + Aug (0.5)', '普通3-seed split'),
        ('Main Method (CV)', 'Main + Aug (0.5, CV)', '5-fold CV'),
    ]
    found = False
    for baseline_name, aug_name, protocol in comparisons:
        if baseline_name in data and aug_name in data:
            found = True
            baseline_rmse = data[baseline_name]['rmse_mean']
            aug_rmse = data[aug_name]['rmse_mean']
            improvement = ((baseline_rmse - aug_rmse) / baseline_rmse) * 100
            print(f'\n{protocol}:')
            print(f'  Baseline: {baseline_name} RMSE = {baseline_rmse:.4f}')
            print(f'  Augmented: {aug_name} RMSE = {aug_rmse:.4f}')
            print(f'  RMSE改进: {improvement:+.2f}%')

    if not found:
        print('\n未找到可比的增强实验结果')


def create_visualizations(data, output_dir=None):
    """生成可视化"""
    output_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 消融实验对比
    ablation_methods = ['ECFP + Ridge', 'Full Finetune', 'From Scratch',
                        'Stage1-only', 'Stage2-only', 'Main Method', 'Main Method (CV)']
    ablation_data = {k: v for k, v in data.items() if k in ablation_methods}

    if ablation_data:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        methods = list(ablation_data.keys())
        metrics = [
            ('rmse_mean', 'rmse_std', 'RMSE (eV)', False),
            ('mae_mean', 'mae_std', 'MAE (eV)', False),
            ('r2_mean', 'r2_std', 'R²', True)
        ]

        for idx, (mean_key, std_key, title, higher_better) in enumerate(metrics):
            ax = axes[idx]
            means = [ablation_data[m][mean_key] for m in methods]
            stds = [0.0 if np.isnan(ablation_data[m][std_key]) else ablation_data[m][std_key] for m in methods]

            x = np.arange(len(methods))
            bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                         color=['#6c757d', '#3498db', '#9b59b6', '#e74c3c',
                                '#f39c12', '#2ecc71', '#1abc9c'][:len(methods)],
                         edgecolor='black', linewidth=1.5)

            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
            ax.set_ylabel(title, fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            for i, (mean, std) in enumerate(zip(means, stds)):
                ax.text(i, mean + std + (max(means) * 0.02), f'{mean:.4f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.suptitle('消融实验对比', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'ablation_comparison.png', dpi=300, bbox_inches='tight')
        print(f'\n可视化已保存: {output_dir / "ablation_comparison.png"}')
        plt.close()

    # 2. 数据增强效果对比
    aug_methods = ['Main Method', 'Main + Aug (0.5)',
                   'Main Method (CV)', 'Main + Aug (0.5, CV)']
    aug_data = {k: v for k, v in data.items() if k in aug_methods}

    if len(aug_data) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = list(aug_data.keys())
        rmse_means = [aug_data[m]['rmse_mean'] for m in methods]
        rmse_stds = [0.0 if np.isnan(aug_data[m]['rmse_std']) else aug_data[m]['rmse_std'] for m in methods]

        x = np.arange(len(methods))
        colors = ['#2ecc71'] + ['#3498db'] * (len(methods) - 1)
        bars = ax.bar(x, rmse_means, yerr=rmse_stds, capsize=5, alpha=0.8,
                     color=colors, edgecolor='black', linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.set_ylabel('RMSE (eV)', fontsize=13, fontweight='bold')
        ax.set_title('数据增强效果对比', fontsize=15, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for i, (mean, std) in enumerate(zip(rmse_means, rmse_stds)):
            ax.text(i, mean + std + 0.005, f'{mean:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / 'augmentation_comparison.png', dpi=300, bbox_inches='tight')
        print(f'可视化已保存: {output_dir / "augmentation_comparison.png"}')
        plt.close()


def create_research_visualizations(data, output_dir=None):
    """Generate publication-style figures for pre-deep-analysis experiment results."""
    results_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / 'results'
    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_rows = []
    metrics_df = _metric_dataframe(data)
    if not metrics_df.empty:
        figure_rows.append({
            'figure': str(_plot_experiment_metric_overview(metrics_df, figures_dir)),
            'description': 'All pre-deep-analysis methods summarized by RMSE, MAE, and R2.',
        })

    cv_df = _load_four_cv_summaries(results_dir)
    if not cv_df.empty:
        figure_rows.append({
            'figure': str(_plot_cv_rmse_heatmap(cv_df, figures_dir)),
            'description': 'RMSE heatmap across seeds and folds for the four final CV experiments.',
        })
        figure_rows.append({
            'figure': str(_plot_cv_metric_ranking(cv_df, figures_dir)),
            'description': 'Distribution of CV fold metrics before deep-analysis interpretation.',
        })

    predictions = _load_representative_predictions(results_dir)
    if not predictions.empty:
        figure_rows.append({
            'figure': str(_plot_prediction_scatter(predictions, figures_dir)),
            'description': 'True-vs-predicted scatter for representative pre-deep-analysis methods.',
        })
        figure_rows.append({
            'figure': str(_plot_residual_distribution(predictions, figures_dir)),
            'description': 'Residual and absolute-error distributions for representative methods.',
        })

    robustness_path = results_dir / 'smiles_robustness_probe' / 'smiles_robustness_summary.csv'
    if robustness_path.exists():
        robustness_df = pd.read_csv(robustness_path)
        figure_rows.append({
            'figure': str(_plot_smiles_robustness_summary(robustness_df, figures_dir)),
            'description': 'SMILES robustness probe before deep-analysis case selection.',
        })

    index = pd.DataFrame(figure_rows)
    index.to_csv(figures_dir / 'pre_deep_visual_figures.csv', index=False)
    print(f'科研风格可视化已保存: {figures_dir}')
    return index


def _plot_experiment_metric_overview(metrics_df, figures_dir):
    display_order = [
        'ECFP + Ridge', 'Full Finetune', 'From Scratch', 'Stage1-only', 'Stage2-only',
        'Main Method', 'Main Method (CV)', 'Main + Aug (0.5)', 'Main + Aug (0.5, CV)',
        'Ablation Stage1 Only Aug0.5', 'Ablation Stage2 Only Aug0.5',
    ]
    metrics_df = metrics_df.copy()
    metrics_df['order'] = metrics_df['method'].map({name: idx for idx, name in enumerate(display_order)}).fillna(999)
    metrics_df = metrics_df.sort_values(['order', 'method'])
    methods = metrics_df['method'].tolist()
    colors = _method_palette(methods)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, max(4.2, len(methods) * 0.36)), constrained_layout=True)
    fig.patch.set_facecolor('#fbfbf8')
    for ax, metric, title in zip(
        axes,
        ['RMSE', 'MAE', 'R2'],
        ['RMSE (lower is better)', 'MAE (lower is better)', 'R2 (higher is better)'],
    ):
        y = np.arange(len(methods))
        means = metrics_df[f'{metric}_mean'].to_numpy()
        stds = metrics_df[f'{metric}_std'].to_numpy()
        ax.errorbar(
            means,
            y,
            xerr=stds,
            fmt='o',
            markersize=4.5,
            color='#252a31',
            ecolor='#87949a',
            elinewidth=1.0,
            capsize=2.5,
            zorder=3,
        )
        ax.scatter(means, y, s=32, color=[colors[m] for m in methods], zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(methods if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_title(title, loc='left', fontweight='bold')
        _style_axis(ax, grid_axis='x')
    path = figures_dir / 'experiment_metric_overview.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def _load_four_cv_summaries(results_dir):
    specs = {
        'No Aug + Two-stage': 'main_method_summary.csv',
        'Aug + Two-stage': 'main_method_aug0.5_summary.csv',
        'Aug + Stage1-only': 'ablation_stage1_only_aug0.5_summary.csv',
        'Aug + Stage2-only': 'ablation_stage2_only_aug0.5_summary.csv',
    }
    rows = []
    for method, filename in specs.items():
        path = results_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df['method'] = method
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _plot_cv_rmse_heatmap(cv_df, figures_dir):
    cv_df = cv_df.copy()
    cv_df['run'] = cv_df['seed'].astype(str) + '-f' + cv_df['fold'].astype(str)
    run_order = sorted(cv_df['run'].unique(), key=lambda item: (int(item.split('-f')[0]), int(item.split('-f')[1])))
    method_order = ['No Aug + Two-stage', 'Aug + Two-stage', 'Aug + Stage1-only', 'Aug + Stage2-only']
    matrix = cv_df.pivot(index='method', columns='run', values='RMSE').reindex(index=method_order, columns=run_order)

    cmap = LinearSegmentedColormap.from_list('rmse_muted', ['#f2f0e8', '#9fb3b2', '#2f4b5c'])
    fig, ax = plt.subplots(figsize=(12.5, 3.8), constrained_layout=True)
    fig.patch.set_facecolor('#fbfbf8')
    im = ax.imshow(matrix.to_numpy(), cmap=cmap, aspect='auto')
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha='right')
    ax.set_title('Cross-validation RMSE by seed and fold', loc='left', fontweight='bold')
    ax.tick_params(length=0, colors='#667085')
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix.iloc[row_idx, col_idx]
            ax.text(col_idx, row_idx, f'{value:.3f}', ha='center', va='center', fontsize=6.5, color='#1f2933')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cbar.set_label('RMSE', fontsize=8, color='#4b5563')
    cbar.ax.tick_params(labelsize=7, colors='#667085', length=2)
    path = figures_dir / 'cv_rmse_heatmap.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def _plot_cv_metric_ranking(cv_df, figures_dir):
    methods = ['No Aug + Two-stage', 'Aug + Two-stage', 'Aug + Stage1-only', 'Aug + Stage2-only']
    colors = _method_palette(methods)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    fig.patch.set_facecolor('#fbfbf8')
    rng = np.random.default_rng(5240)
    for ax, metric, title in zip(axes, ['RMSE', 'MAE', 'R2'], ['RMSE', 'MAE', 'R2']):
        for idx, method in enumerate(methods):
            values = cv_df.loc[cv_df['method'] == method, metric].to_numpy()
            jitter = rng.normal(0, 0.045, size=len(values))
            ax.scatter(np.full(len(values), idx) + jitter, values, s=16, alpha=0.45, color=colors[method], edgecolor='none')
            ax.plot(idx, values.mean(), marker='D', color='#252a31', markersize=5)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=35, ha='right')
        ax.set_title(title, loc='left', fontweight='bold')
        _style_axis(ax)
    path = figures_dir / 'cv_metric_distribution.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def _load_representative_predictions(results_dir):
    specs = {
        'ECFP + Ridge': results_dir / 'baseline_ecfp_ridge_seed42' / 'predictions.csv',
        'No Aug + Two-stage': results_dir / 'main_method_seed42_fold0' / 'predictions.csv',
        'Aug + Two-stage': results_dir / 'main_method_aug0.5_seed42_fold0' / 'predictions.csv',
        'Aug + Stage1-only': results_dir / 'ablation_stage1_only_aug0.5_seed42_fold0' / 'predictions.csv',
        'Aug + Stage2-only': results_dir / 'ablation_stage2_only_aug0.5_seed42_fold0' / 'predictions.csv',
    }
    rows = []
    for method, path in specs.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if {'true', 'pred', 'error'}.issubset(df.columns):
            df = df.copy()
            df['method'] = method
            df['abs_error'] = df['error'].abs()
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _plot_prediction_scatter(predictions, figures_dir):
    methods = predictions['method'].drop_duplicates().tolist()
    colors = _method_palette(methods)
    cols = min(3, len(methods))
    rows = int(np.ceil(len(methods) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), squeeze=False, constrained_layout=True)
    fig.patch.set_facecolor('#fbfbf8')
    vmin = min(predictions['true'].min(), predictions['pred'].min())
    vmax = max(predictions['true'].max(), predictions['pred'].max())
    for ax, method in zip(axes.ravel(), methods):
        df = predictions[predictions['method'] == method]
        ax.scatter(df['true'], df['pred'], s=14, alpha=0.45, color=colors[method], edgecolor='none')
        ax.plot([vmin, vmax], [vmin, vmax], color='#3f3a35', linewidth=1.0)
        ax.set_title(method, loc='left', fontsize=10, fontweight='bold')
        ax.set_xlabel('True energy')
        ax.set_ylabel('Predicted energy')
        _style_axis(ax)
    for ax in axes.ravel()[len(methods):]:
        ax.axis('off')
    path = figures_dir / 'prediction_scatter_representative.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def _plot_residual_distribution(predictions, figures_dir):
    methods = predictions['method'].drop_duplicates().tolist()
    colors = _method_palette(methods)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)
    fig.patch.set_facecolor('#fbfbf8')
    bins = np.linspace(predictions['error'].min(), predictions['error'].max(), 30)
    for method in methods:
        df = predictions[predictions['method'] == method]
        axes[0].hist(df['error'], bins=bins, histtype='step', linewidth=1.6, color=colors[method], label=method)
    axes[0].axvline(0, color='#3f3a35', linewidth=0.9)
    axes[0].set_title('Residual distribution', loc='left', fontweight='bold')
    axes[0].set_xlabel('Prediction error')
    axes[0].set_ylabel('Count')
    axes[0].legend(frameon=False, fontsize=7)

    data = [predictions.loc[predictions['method'] == method, 'abs_error'].to_numpy() for method in methods]
    parts = axes[1].violinplot(data, showmeans=False, showmedians=True, widths=0.75)
    for body, method in zip(parts['bodies'], methods):
        body.set_facecolor(colors[method])
        body.set_edgecolor('none')
        body.set_alpha(0.42)
    for key in ['cmedians', 'cbars', 'cmins', 'cmaxes']:
        parts[key].set_color('#252a31')
        parts[key].set_linewidth(1.0)
    axes[1].set_xticks(np.arange(1, len(methods) + 1))
    axes[1].set_xticklabels(methods, rotation=35, ha='right')
    axes[1].set_title('Absolute-error distribution', loc='left', fontweight='bold')
    axes[1].set_ylabel('Absolute error')
    for ax in axes:
        _style_axis(ax)
    path = figures_dir / 'prediction_error_distribution.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def _plot_smiles_robustness_summary(robustness_df, figures_dir):
    methods = robustness_df['model'].tolist()
    colors = _method_palette(methods)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    fig.patch.set_facecolor('#fbfbf8')
    y = np.arange(len(methods))

    axes[0].barh(y - 0.17, robustness_df['original_RMSE'], height=0.32, color='#9fb3b2', label='Original')
    axes[0].barh(y + 0.17, robustness_df['randomized_RMSE'], height=0.32, color='#607d8b', label='Randomized')
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(methods)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('RMSE')
    axes[0].set_title('Original vs randomized SMILES RMSE', loc='left', fontweight='bold')
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].barh(y, robustness_df['mean_prediction_std_across_variants'], color=[colors[m] for m in methods])
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Mean prediction std across variants')
    axes[1].set_title('Representation sensitivity', loc='left', fontweight='bold')
    for idx, value in enumerate(robustness_df['mean_prediction_std_across_variants']):
        axes[1].text(value + 0.003, idx, f'{value:.3f}', va='center', fontsize=8, color='#4b5563')

    for ax in axes:
        _style_axis(ax, grid_axis='x')
    path = figures_dir / 'smiles_robustness_summary.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def write_markdown_report(data, output_dir=None):
    """生成简明Markdown报告，方便写入实验文档。"""
    output_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'experiment_analysis.md'

    lines = [
        '# Two-stage ChemBERTa Experiment Analysis',
        '',
        '## Method',
        '',
        'The main method uses supervised regression from SMILES to binding energy with a pretrained ChemBERTa encoder and a small regression head. Training is split into two stages: Stage 1 freezes the encoder and learns only the regression head, so the model first maps pretrained molecular representations to the target energy scale. Stage 2 unfreezes the encoder with a smaller backbone learning rate and a larger head learning rate, allowing task-specific adaptation without immediately destroying pretrained chemical features.',
        '',
        'SMILES augmentation is applied only to the training split. Each molecule keeps the same label, but its SMILES string can be replaced by a randomized equivalent representation. This tests whether the model learns molecule-level structure instead of overfitting to one textual serialization.',
        '',
        '## Results',
        '',
        '| Method | RMSE | MAE | R2 | Runs | Source |',
        '|---|---:|---:|---:|---:|---|',
    ]

    for name, metrics in data.items():
        rmse_std = metrics['rmse_std'] if not np.isnan(metrics['rmse_std']) else 0.0
        mae_std = metrics['mae_std'] if not np.isnan(metrics['mae_std']) else 0.0
        r2_std = metrics['r2_std'] if not np.isnan(metrics['r2_std']) else 0.0
        lines.append(
            f"| {name} | {metrics['rmse_mean']:.4f} +/- {rmse_std:.4f} | "
            f"{metrics['mae_mean']:.4f} +/- {mae_std:.4f} | "
            f"{metrics['r2_mean']:.4f} +/- {r2_std:.4f} | "
            f"{metrics['n_runs']} | {metrics.get('source', '')} |"
        )

    if 'Main Method' in data and 'Stage1-only' in data and 'Stage2-only' in data:
        main = data['Main Method']['rmse_mean']
        stage1 = data['Stage1-only']['rmse_mean']
        stage2 = data['Stage2-only']['rmse_mean']
        lines.extend([
            '',
            '## Ablation Interpretation',
            '',
            f'- Compared with Stage1-only, the full two-stage method changes RMSE by {(stage1 - main) / main * 100:+.2f}%, isolating the value of supervised encoder adaptation after head warm-up.',
            f'- Compared with Stage2-only, the full two-stage method changes RMSE by {(stage2 - main) / main * 100:+.2f}%, isolating the value of learning a stable regression head before unfreezing the backbone.',
        ])

    aug_lines = []
    for baseline_name, aug_name, protocol in [
        ('Main Method', 'Main + Aug (0.5)', 'standard 3-seed split'),
        ('Main Method (CV)', 'Main + Aug (0.5, CV)', '5-fold CV'),
    ]:
        if baseline_name in data and aug_name in data:
            baseline = data[baseline_name]['rmse_mean']
            aug_rmse = data[aug_name]['rmse_mean']
            aug_lines.append(
                f'- Under the {protocol} protocol, {aug_name} changes RMSE by '
                f'{(baseline - aug_rmse) / baseline * 100:+.2f}% '
                f'({baseline:.4f} -> {aug_rmse:.4f}).'
            )

    if aug_lines:
        lines.extend(['', '## Augmentation Interpretation', ''] + aug_lines)

    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'分析报告已保存: {report_path}')


def main():
    """主函数"""
    print('='*90)
    print('ChemBERTa实验结果分析')
    print('='*90)

    # 加载结果
    data = load_experiment_results()

    if not data:
        print('未找到任何实验结果')
        return

    # 打印对比表格
    print_comparison_table(data)

    # 消融实验分析
    analyze_ablation(data)

    # 数据增强分析
    analyze_augmentation(data)

    # 生成可视化
    create_visualizations(data)
    create_research_visualizations(data)
    write_markdown_report(data)

    print('\n' + '='*90)
    print('分析完成')
    print('='*90)


if __name__ == '__main__':
    main()
