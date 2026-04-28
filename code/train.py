"""
Training script for ChemBERTa binding energy prediction
"""

import argparse
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time

from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from config import Config
from dataset import create_dataloaders, create_cv_dataloaders
from model import create_model
from evaluate import evaluate, print_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, config):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)  # Normalized labels

        # Forward pass
        predictions = model(input_ids, attention_mask)

        # Compute loss
        loss = criterion(predictions, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def set_encoder_trainable(model, unfreeze_layers=None):
    """Set encoder requires_grad. None = full unfreeze; int N = last N transformer blocks."""
    if unfreeze_layers is None:
        for p in model.encoder.parameters():
            p.requires_grad = True
        return

    for p in model.encoder.parameters():
        p.requires_grad = False
    layers = model.encoder.encoder.layer  # RoBERTa-style stack
    n = len(layers)
    keep_from = max(0, n - unfreeze_layers)
    for layer in layers[keep_from:]:
        for p in layer.parameters():
            p.requires_grad = True
    print(f'Unfroze last {min(unfreeze_layers, n)}/{n} encoder layers')


def _run_loop(model, train_loader, val_loader, label_stats, criterion,
              optimizer, scheduler, max_epochs, patience, history,
              config, stage_label):
    """Inner training loop for a single stage. Mutates history in-place.

    Returns (best_val_rmse, best_model_state, best_epoch).
    """
    early_stopping = EarlyStopping(patience=patience, verbose=config.VERBOSE)
    best_val_rmse = float('inf')
    best_model_state = None
    best_epoch = 0

    for epoch in range(max_epochs):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, config.DEVICE, config
        )
        val_metrics, _, _ = evaluate(model, val_loader, label_stats, config.DEVICE)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(float(train_loss))
        history['val_rmse'].append(float(val_metrics['RMSE']))
        history['val_mae'].append(float(val_metrics['MAE']))
        history['val_r2'].append(float(val_metrics['R2']))
        history['lr'].append(float(current_lr))
        history['stage'].append(stage_label)

        epoch_time = time.time() - epoch_start
        print(f'[{stage_label}] Epoch {epoch+1}/{max_epochs} ({epoch_time:.1f}s) '
              f'lr={current_lr:.2e} train={train_loss:.4f} '
              f'val_rmse={val_metrics["RMSE"]:.4f} val_r2={val_metrics["R2"]:.4f}')

        if val_metrics['RMSE'] < best_val_rmse:
            best_val_rmse = val_metrics['RMSE']
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            print(f'  -> new best (val RMSE {best_val_rmse:.4f})')

        early_stopping(val_metrics['RMSE'], epoch)
        if early_stopping.early_stop:
            print(f'[{stage_label}] Early stopping at epoch {epoch+1}')
            break

    return best_val_rmse, best_model_state, best_epoch


def train(config, seed=42, fold_idx=None, dataloaders=None):
    """
    Main training function

    Args:
        config: Config object
        seed: Random seed
        fold_idx: If set, marks this as a CV fold (affects output dir name)
        dataloaders: Optional (train_loader, val_loader, test_loader, label_stats).
                     If None, loads from split_seed{seed}.csv.

    Returns:
        results: dict with training history and final metrics
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f'\n{"="*60}')
    if fold_idx is not None:
        print(f'Training seed {seed} fold {fold_idx}')
    else:
        print(f'Training with seed {seed}')
    print(f'{"="*60}\n')

    # Create output directory
    if fold_idx is not None:
        out_name = f'{config.EXPERIMENT_NAME}_seed{seed}_fold{fold_idx}'
    else:
        out_name = f'{config.EXPERIMENT_NAME}_seed{seed}'
    output_dir = project_path(config.RESULTS_DIR) / out_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data (or use provided)
    if dataloaders is None:
        csv_path = project_path(config.DATA_DIR) / f'split_seed{seed}.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f'Split file not found: {csv_path}')
        train_loader, val_loader, test_loader, label_stats = create_dataloaders(
            csv_path, config, seed
        )
    else:
        train_loader, val_loader, test_loader, label_stats = dataloaders

    # Create model
    model = create_model(config)
    model = model.to(config.DEVICE)

    criterion = nn.MSELoss()

    # Training history (extended with lr/stage tracking)
    history = {
        'train_loss': [], 'val_rmse': [], 'val_mae': [], 'val_r2': [],
        'lr': [], 'stage': [],
    }

    print(f'\nStarting training...')
    start_time = time.time()

    if config.TWO_STAGE:
        # ---- Stage 1: encoder frozen, head only ----
        if config.STAGE1_MAX_EPOCHS > 0:
            for p in model.encoder.parameters():
                p.requires_grad = False
            head_params = [p for p in model.head.parameters() if p.requires_grad]
            optimizer1 = AdamW(head_params, lr=config.STAGE1_LR_HEAD,
                               weight_decay=config.WEIGHT_DECAY)
            steps1 = max(1, len(train_loader) * config.STAGE1_MAX_EPOCHS)
            scheduler1 = get_cosine_schedule_with_warmup(
                optimizer1,
                num_warmup_steps=int(config.WARMUP_RATIO * steps1),
                num_training_steps=steps1,
            )
            print(f'\n[Stage1] head-only, lr={config.STAGE1_LR_HEAD}, '
                  f'max_epochs={config.STAGE1_MAX_EPOCHS}, patience={config.STAGE1_PATIENCE}')
            best_rmse_1, best_state_1, best_epoch_1 = _run_loop(
                model, train_loader, val_loader, label_stats, criterion,
                optimizer1, scheduler1, config.STAGE1_MAX_EPOCHS, config.STAGE1_PATIENCE,
                history, config, 'stage1',
            )
            if best_state_1 is not None:
                model.load_state_dict(best_state_1)
        else:
            print('\n[Stage1] Skipped (max_epochs=0)')
            best_rmse_1 = float('inf')
            best_state_1 = None
            best_epoch_1 = 0

        # ---- Stage 2: unfreeze (full or last N), differential LR ----
        if config.STAGE2_MAX_EPOCHS > 0:
            set_encoder_trainable(model, config.STAGE2_UNFREEZE_LAYERS)
            backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
            head_params = [p for p in model.head.parameters() if p.requires_grad]
            optimizer2 = AdamW(
                [
                    {'params': head_params, 'lr': config.STAGE2_LR_HEAD},
                    {'params': backbone_params, 'lr': config.STAGE2_LR_BACKBONE},
                ],
                weight_decay=config.WEIGHT_DECAY,
            )
            steps2 = max(1, len(train_loader) * config.STAGE2_MAX_EPOCHS)
            scheduler2 = get_cosine_schedule_with_warmup(
                optimizer2,
                num_warmup_steps=int(config.WARMUP_RATIO * steps2),
                num_training_steps=steps2,
            )
            print(f'\n[Stage2] head_lr={config.STAGE2_LR_HEAD} backbone_lr={config.STAGE2_LR_BACKBONE} '
                  f'max_epochs={config.STAGE2_MAX_EPOCHS} patience={config.STAGE2_PATIENCE}')
            best_rmse_2, best_state_2, best_epoch_2 = _run_loop(
                model, train_loader, val_loader, label_stats, criterion,
                optimizer2, scheduler2, config.STAGE2_MAX_EPOCHS, config.STAGE2_PATIENCE,
                history, config, 'stage2',
            )
        else:
            print('\n[Stage2] Skipped (max_epochs=0)')
            best_rmse_2 = float('inf')
            best_state_2 = None
            best_epoch_2 = 0

        # Pick the overall best across both stages
        if best_state_2 is not None and best_rmse_2 <= best_rmse_1:
            best_model_state = best_state_2
            best_val_rmse = best_rmse_2
            stage1_count = sum(1 for s in history['stage'] if s == 'stage1')
            best_epoch = stage1_count + best_epoch_2
        elif best_state_1 is not None:
            best_model_state = best_state_1
            best_val_rmse = best_rmse_1
            best_epoch = best_epoch_1
        else:
            # Both stages skipped or failed
            best_model_state = None
            best_val_rmse = float('inf')
            best_epoch = 0

    else:
        # ---- Single-stage path (ablations + legacy main) ----
        optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        steps = max(1, len(train_loader) * config.MAX_EPOCHS)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(config.WARMUP_RATIO * steps),
            num_training_steps=steps,
        )
        best_val_rmse, best_model_state, best_epoch = _run_loop(
            model, train_loader, val_loader, label_stats, criterion,
            optimizer, scheduler, config.MAX_EPOCHS, config.EARLY_STOPPING_PATIENCE,
            history, config, 'single',
        )

    training_time = time.time() - start_time
    print(f'\nTraining completed in {training_time:.1f}s (best val RMSE: {best_val_rmse:.4f})')

    # Save best model checkpoint
    if best_model_state is not None and config.SAVE_MODEL:
        torch.save(best_model_state, output_dir / 'best_model.pt')

    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation on test set
    print(f'\n{"="*60}')
    print('Final Evaluation on Test Set')
    print(f'{"="*60}\n')

    test_metrics, test_preds, test_targets = evaluate(
        model, test_loader, label_stats, config.DEVICE
    )
    print_metrics(test_metrics)

    # Save results
    results = {
        'seed': int(seed),
        'fold_idx': int(fold_idx) if fold_idx is not None else None,
        'config': {
            'model': config.MODEL_NAME,
            'freeze_encoder': bool(config.FREEZE_ENCODER),
            'from_scratch': bool(config.FROM_SCRATCH),
            'two_stage': bool(config.TWO_STAGE),
            'experiment_name': config.EXPERIMENT_NAME,
            'batch_size': int(config.BATCH_SIZE),
            'learning_rate': float(config.LEARNING_RATE),
            'max_epochs': int(config.MAX_EPOCHS),
            'stage1_lr_head': float(config.STAGE1_LR_HEAD) if config.TWO_STAGE else None,
            'stage2_lr_head': float(config.STAGE2_LR_HEAD) if config.TWO_STAGE else None,
            'stage2_lr_backbone': float(config.STAGE2_LR_BACKBONE) if config.TWO_STAGE else None,
            'stage2_unfreeze_layers': config.STAGE2_UNFREEZE_LAYERS if config.TWO_STAGE else None,
        },
        'label_stats': {
            'mean': float(label_stats['mean']),
            'std': float(label_stats['std']),
        },
        'history': history,
        'test_metrics': {
            'RMSE': float(test_metrics['RMSE']),
            'MAE': float(test_metrics['MAE']),
            'R2': float(test_metrics['R2']),
        },
        'training_time': float(training_time),
        'best_epoch': int(best_epoch),
        'best_val_rmse': float(best_val_rmse),
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    pred_df = pd.DataFrame({
        'true': test_targets,
        'pred': test_preds,
        'error': test_preds - test_targets,
    })
    pred_df.to_csv(output_dir / 'predictions.csv', index=False)

    print(f'\nResults saved to {output_dir}')

    return results


def train_all_seeds(config):
    """Train on all seeds and aggregate results"""
    all_results = []

    for seed in config.SEEDS:
        results = train(config, seed)
        all_results.append(results)

    # Aggregate results
    test_rmse = [r['test_metrics']['RMSE'] for r in all_results]
    test_mae = [r['test_metrics']['MAE'] for r in all_results]
    test_r2 = [r['test_metrics']['R2'] for r in all_results]

    print(f'\n{"="*60}')
    print('Aggregated Results Across All Seeds')
    print(f'{"="*60}\n')
    print(f'RMSE: {np.mean(test_rmse):.4f} ± {np.std(test_rmse):.4f} eV')
    print(f'MAE:  {np.mean(test_mae):.4f} ± {np.std(test_mae):.4f} eV')
    print(f'R²:   {np.mean(test_r2):.4f} ± {np.std(test_r2):.4f}')

    # Save aggregated results
    summary = pd.DataFrame({
        'seed': config.SEEDS,
        'RMSE': test_rmse,
        'MAE': test_mae,
        'R2': test_r2
    })
    summary_path = project_path(config.RESULTS_DIR) / config.SUMMARY_FILENAME
    summary.to_csv(summary_path, index=False)
    print(f'\nSummary saved to {summary_path}')

    return all_results


def train_cv(config, n_folds=5):
    """Train with K-fold cross-validation on train+val, test fixed.

    For each seed, runs n_folds CV folds. Each fold trains on a different
    train/val split and evaluates on the same test set.

    Returns aggregated results across all seeds and folds.
    """
    all_results = []

    for seed in config.SEEDS:
        csv_path = project_path(config.DATA_DIR) / f'split_seed{seed}.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f'Split file not found: {csv_path}')

        print(f'\n{"#"*60}')
        print(f'Seed {seed}: {n_folds}-fold CV')
        print(f'{"#"*60}')

        for fold_idx in range(n_folds):
            dataloaders = create_cv_dataloaders(csv_path, config, seed, fold_idx, n_folds)
            results = train(config, seed, fold_idx=fold_idx, dataloaders=dataloaders)
            all_results.append(results)

    # Aggregate across all seeds and folds
    test_rmse = [r['test_metrics']['RMSE'] for r in all_results]
    test_mae = [r['test_metrics']['MAE'] for r in all_results]
    test_r2 = [r['test_metrics']['R2'] for r in all_results]

    print(f'\n{"="*60}')
    print(f'Aggregated Results Across {len(config.SEEDS)} Seeds × {n_folds} Folds')
    print(f'{"="*60}\n')
    print(f'RMSE: {np.mean(test_rmse):.4f} ± {np.std(test_rmse):.4f} eV')
    print(f'MAE:  {np.mean(test_mae):.4f} ± {np.std(test_mae):.4f} eV')
    print(f'R²:   {np.mean(test_r2):.4f} ± {np.std(test_r2):.4f}')

    # Save detailed results (per fold)
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            'seed': r['seed'],
            'fold': r.get('fold_idx', -1),
            'RMSE': r['test_metrics']['RMSE'],
            'MAE': r['test_metrics']['MAE'],
            'R2': r['test_metrics']['R2'],
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = project_path(config.RESULTS_DIR) / config.SUMMARY_FILENAME
    summary.to_csv(summary_path, index=False)
    print(f'\nDetailed summary saved to {summary_path}')

    return all_results


def build_config_from_args():
    parser = argparse.ArgumentParser(description='Train ChemBERTa regression experiments.')
    parser.add_argument(
        '--experiment',
        choices=['main', 'stage1_only', 'stage2_only', 'full_finetune', 'from_scratch'],
        default='main',
        help='Experiment variant to run.'
    )
    parser.add_argument('--data-dir', default=None, help='Directory containing split_seed*.csv files.')
    parser.add_argument('--results-dir', default=None, help='Directory for output results.')
    parser.add_argument('--max-epochs', type=int, default=None, help='Override max epochs.')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size.')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate.')
    parser.add_argument('--unfreeze-layers', type=int, default=None,
                        help='Stage 2: only unfreeze last N transformer layers (default: all).')
    parser.add_argument('--no-save-model', action='store_true', help='Do not save best_model.pt.')
    parser.add_argument('--cv', action='store_true', help='Use K-fold cross-validation.')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds (default 5).')
    args = parser.parse_args()

    config = Config()
    if args.data_dir is not None:
        config.DATA_DIR = args.data_dir
    if args.results_dir is not None:
        config.RESULTS_DIR = args.results_dir
    if args.max_epochs is not None:
        config.MAX_EPOCHS = args.max_epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.no_save_model:
        config.SAVE_MODEL = False
    if args.unfreeze_layers is not None:
        config.STAGE2_UNFREEZE_LAYERS = args.unfreeze_layers
    config.USE_CV = args.cv
    config.CV_FOLDS = args.cv_folds

    if args.experiment == 'main':
        config.EXPERIMENT_NAME = 'main_method'
        config.SUMMARY_FILENAME = 'main_method_summary.csv'
        config.FREEZE_ENCODER = True
        config.FROM_SCRATCH = False
        config.TWO_STAGE = True
        if args.lr is not None:
            config.STAGE1_LR_HEAD = args.lr
    elif args.experiment == 'stage1_only':
        config.EXPERIMENT_NAME = 'ablation_stage1_only'
        config.SUMMARY_FILENAME = 'ablation_stage1_only_results.csv'
        config.FREEZE_ENCODER = True
        config.FROM_SCRATCH = False
        config.TWO_STAGE = True
        config.STAGE2_MAX_EPOCHS = 0  # Skip stage 2
        if args.lr is not None:
            config.STAGE1_LR_HEAD = args.lr
    elif args.experiment == 'stage2_only':
        config.EXPERIMENT_NAME = 'ablation_stage2_only'
        config.SUMMARY_FILENAME = 'ablation_stage2_only_results.csv'
        config.FREEZE_ENCODER = True  # Will be unfrozen in stage 2
        config.FROM_SCRATCH = False
        config.TWO_STAGE = True
        config.STAGE1_MAX_EPOCHS = 0  # Skip stage 1
        if args.lr is not None:
            config.STAGE2_LR_BACKBONE = args.lr
    elif args.experiment == 'full_finetune':
        config.EXPERIMENT_NAME = 'ablation1_full_finetune'
        config.SUMMARY_FILENAME = 'ablation1_full_finetune_results.csv'
        config.FREEZE_ENCODER = False
        config.FROM_SCRATCH = False
        config.TWO_STAGE = False
        config.LEARNING_RATE = args.lr if args.lr is not None else 2e-5
    elif args.experiment == 'from_scratch':
        config.EXPERIMENT_NAME = 'ablation2_from_scratch'
        config.SUMMARY_FILENAME = 'ablation2_from_scratch_results.csv'
        config.FREEZE_ENCODER = False
        config.FROM_SCRATCH = True
        config.TWO_STAGE = False
        config.LEARNING_RATE = args.lr if args.lr is not None else 1e-4

    return config


if __name__ == '__main__':
    config = build_config_from_args()
    print('Configuration:')
    print(config)

    # Train with CV or standard seeds
    if config.USE_CV:
        results = train_cv(config, n_folds=config.CV_FOLDS)
    else:
        results = train_all_seeds(config)
