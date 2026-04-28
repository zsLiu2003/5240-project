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

from config import Config
from dataset import create_dataloaders
from model import create_model
from evaluate import evaluate, print_metrics


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


def train_one_epoch(model, train_loader, optimizer, criterion, device, config):
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

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def train(config, seed=42):
    """
    Main training function

    Args:
        config: Config object
        seed: Random seed

    Returns:
        results: dict with training history and final metrics
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f'\n{"="*60}')
    print(f'Training with seed {seed}')
    print(f'{"="*60}\n')

    # Create output directory
    output_dir = Path(config.RESULTS_DIR) / f'{config.EXPERIMENT_NAME}_seed{seed}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    csv_path = Path(config.DATA_DIR) / f'split_seed{seed}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'Split file not found: {csv_path}')

    train_loader, val_loader, test_loader, label_stats = create_dataloaders(
        csv_path, config, seed
    )

    # Create model
    model = create_model(config)
    model = model.to(config.DEVICE)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        verbose=config.VERBOSE
    )

    # Training history
    history = {
        'train_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'val_r2': []
    }

    best_val_rmse = float('inf')
    best_model_state = None

    # Training loop
    print(f'\nStarting training...')
    start_time = time.time()

    for epoch in range(config.MAX_EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, config.DEVICE, config
        )

        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, label_stats, config.DEVICE)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_rmse'].append(val_metrics['RMSE'])
        history['val_mae'].append(val_metrics['MAE'])
        history['val_r2'].append(val_metrics['R2'])

        # Print progress
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch+1}/{config.MAX_EPOCHS} ({epoch_time:.1f}s)')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val RMSE: {val_metrics["RMSE"]:.4f} eV')
        print(f'  Val MAE:  {val_metrics["MAE"]:.4f} eV')
        print(f'  Val R²:   {val_metrics["R2"]:.4f}')

        # Save best model
        if val_metrics['RMSE'] < best_val_rmse:
            best_val_rmse = val_metrics['RMSE']
            best_model_state = copy.deepcopy(model.state_dict())
            if config.SAVE_MODEL:
                torch.save(best_model_state, output_dir / 'best_model.pt')
            print(f'  → New best model (RMSE: {best_val_rmse:.4f})')

        # Early stopping
        early_stopping(val_metrics['RMSE'], epoch)
        if early_stopping.early_stop:
            print(f'\nEarly stopping triggered at epoch {epoch+1}')
            print(f'Best epoch was {early_stopping.best_epoch+1}')
            break

    training_time = time.time() - start_time
    print(f'\nTraining completed in {training_time:.1f}s')

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

    # Save results (convert numpy types to Python types for JSON serialization)
    results = {
        'seed': int(seed),
        'config': {
            'model': config.MODEL_NAME,
            'freeze_encoder': bool(config.FREEZE_ENCODER),
            'from_scratch': bool(config.FROM_SCRATCH),
            'experiment_name': config.EXPERIMENT_NAME,
            'batch_size': int(config.BATCH_SIZE),
            'learning_rate': float(config.LEARNING_RATE),
            'max_epochs': int(config.MAX_EPOCHS)
        },
        'label_stats': {
            'mean': float(label_stats['mean']),
            'std': float(label_stats['std'])
        },
        'history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_rmse': [float(x) for x in history['val_rmse']],
            'val_mae': [float(x) for x in history['val_mae']],
            'val_r2': [float(x) for x in history['val_r2']]
        },
        'test_metrics': {
            'RMSE': float(test_metrics['RMSE']),
            'MAE': float(test_metrics['MAE']),
            'R2': float(test_metrics['R2'])
        },
        'training_time': float(training_time),
        'best_epoch': int(early_stopping.best_epoch)
    }

    # Save results as JSON
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame({
        'true': test_targets,
        'pred': test_preds,
        'error': test_preds - test_targets
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
    summary_path = Path(config.RESULTS_DIR) / config.SUMMARY_FILENAME
    summary.to_csv(summary_path, index=False)
    print(f'\nSummary saved to {summary_path}')

    return all_results


def build_config_from_args():
    parser = argparse.ArgumentParser(description='Train ChemBERTa regression experiments.')
    parser.add_argument(
        '--experiment',
        choices=['main', 'full_finetune', 'from_scratch'],
        default='main',
        help='Experiment variant to run.'
    )
    parser.add_argument('--data-dir', default=None, help='Directory containing split_seed*.csv files.')
    parser.add_argument('--results-dir', default=None, help='Directory for output results.')
    parser.add_argument('--max-epochs', type=int, default=None, help='Override max epochs.')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size.')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate.')
    parser.add_argument('--no-save-model', action='store_true', help='Do not save best_model.pt.')
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

    if args.experiment == 'main':
        config.EXPERIMENT_NAME = 'main_method'
        config.SUMMARY_FILENAME = 'main_method_summary.csv'
        config.FREEZE_ENCODER = True
        config.FROM_SCRATCH = False
        if args.lr is not None:
            config.LEARNING_RATE = args.lr
    elif args.experiment == 'full_finetune':
        config.EXPERIMENT_NAME = 'ablation1_full_finetune'
        config.SUMMARY_FILENAME = 'ablation1_full_finetune_results.csv'
        config.FREEZE_ENCODER = False
        config.FROM_SCRATCH = False
        config.LEARNING_RATE = args.lr if args.lr is not None else 2e-5
    elif args.experiment == 'from_scratch':
        config.EXPERIMENT_NAME = 'ablation2_from_scratch'
        config.SUMMARY_FILENAME = 'ablation2_from_scratch_results.csv'
        config.FREEZE_ENCODER = False
        config.FROM_SCRATCH = True
        config.LEARNING_RATE = args.lr if args.lr is not None else 1e-4

    return config


if __name__ == '__main__':
    config = build_config_from_args()
    print('Configuration:')
    print(config)

    # Train on all seeds
    results = train_all_seeds(config)
