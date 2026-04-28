"""
Evaluation functions
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate(model, dataloader, label_stats, device):
    """
    Evaluate model on a dataset

    Args:
        model: ChemBERTaRegressor
        dataloader: DataLoader
        label_stats: dict with 'mean' and 'std' for denormalization
        device: torch device

    Returns:
        metrics: dict with RMSE, MAE, R2
        predictions: numpy array of predictions
        targets: numpy array of ground truth
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_raw = batch['label_raw'].numpy()  # Already in original scale

            # Forward pass
            preds_normalized = model(input_ids, attention_mask)

            # Denormalize predictions
            preds = preds_normalized.cpu().numpy() * label_stats['std'] + label_stats['mean']

            all_preds.append(preds)
            all_targets.append(labels_raw)

    # Concatenate all batches
    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

    return metrics, predictions, targets


def print_metrics(metrics, prefix=''):
    """Pretty print metrics"""
    print(f'{prefix}RMSE: {metrics["RMSE"]:.4f} eV')
    print(f'{prefix}MAE:  {metrics["MAE"]:.4f} eV')
    print(f'{prefix}R²:   {metrics["R2"]:.4f}')
