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
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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

    # 自动加载增强实验结果（例如 main_method_aug0.5_summary.csv）
    for filepath in sorted(results_dir.glob('*_aug*_summary.csv')):
        df = pd.read_csv(filepath)
        if {'RMSE', 'MAE', 'R2'}.issubset(df.columns):
            stem = filepath.stem.replace('_summary', '')
            if stem.startswith('main_method_aug'):
                prob = stem.replace('main_method_aug', '')
                name = f'Main + Aug ({prob})'
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

    if 'Main Method' not in data:
        print('Main Method结果未找到，无法分析增强效果')
        return

    baseline_rmse = data['Main Method']['rmse_mean']
    print(f'\nBaseline (无增强): RMSE = {baseline_rmse:.4f}')

    aug_results = []
    for name in data.keys():
        if 'Aug' in name:
            aug_rmse = data[name]['rmse_mean']
            improvement = ((baseline_rmse - aug_rmse) / baseline_rmse) * 100
            aug_results.append((name, aug_rmse, improvement))
            print(f'{name}: RMSE = {aug_rmse:.4f}, 改进 = {improvement:+.2f}%')

    if aug_results:
        best = max(aug_results, key=lambda x: x[2])
        print(f'\n最佳增强配置: {best[0]}')
        print(f'  RMSE: {best[1]:.4f}')
        print(f'  相对改进: {best[2]:+.2f}%')
    else:
        print('\n未找到增强实验结果')


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
    aug_methods = ['Main Method'] + [k for k in data.keys() if 'Aug' in k]
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

    aug_keys = [k for k in data if 'Aug' in k]
    if 'Main Method' in data and aug_keys:
        baseline = data['Main Method']['rmse_mean']
        best_key = min(aug_keys, key=lambda k: data[k]['rmse_mean'])
        best_rmse = data[best_key]['rmse_mean']
        lines.extend([
            '',
            '## Augmentation Interpretation',
            '',
            f'- Best augmentation setting: {best_key}, RMSE {best_rmse:.4f}.',
            f'- Relative to the non-augmented main method, this is {(baseline - best_rmse) / baseline * 100:+.2f}% RMSE change.',
        ])

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
    write_markdown_report(data)

    print('\n' + '='*90)
    print('分析完成')
    print('='*90)


if __name__ == '__main__':
    main()
