"""
Local testing script - verify code works without GPU/full data
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Import our modules
from config import Config
from dataset import BindingEnergyDataset, create_dataloaders
from model import create_model
from evaluate import evaluate


def create_dummy_data(output_path='../data/split_seed42.csv', n_samples=30):
    """
    Create dummy data for testing

    Args:
        output_path: Where to save dummy CSV
        n_samples: Number of samples to generate
    """
    print('Creating dummy data for testing...')

    # Generate dummy SMILES (simple molecules)
    dummy_smiles = [
        'CC', 'CCC', 'CCCC', 'C=C', 'C#C',
        'CCO', 'CCCO', 'CC(C)C', 'C1CC1', 'c1ccccc1'
    ] * (n_samples // 10 + 1)
    dummy_smiles = dummy_smiles[:n_samples]

    # Generate dummy binding energies
    np.random.seed(42)
    dummy_energies = np.random.uniform(-3.0, -2.5, n_samples)

    # Assign splits
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    splits = ['train'] * n_train + ['val'] * n_val + ['test'] * (n_samples - n_train - n_val)

    # Create DataFrame
    df = pd.DataFrame({
        'SMILES': dummy_smiles,
        'Energy_min': dummy_energies,
        'split': splits
    })

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f'Dummy data saved to {output_path}')
    print(f'  Total: {len(df)} samples')
    print(f'  Train: {(df["split"]=="train").sum()}')
    print(f'  Val: {(df["split"]=="val").sum()}')
    print(f'  Test: {(df["split"]=="test").sum()}')

    return output_path


def test_dataset():
    """Test dataset loading"""
    print('\n' + '='*60)
    print('TEST 1: Dataset Loading')
    print('='*60)

    from transformers import AutoTokenizer

    config = Config()
    config.BATCH_SIZE = 4  # Small batch for testing

    # Create dummy data if needed
    csv_path = Path(config.DATA_DIR) / 'split_seed42.csv'
    if not csv_path.exists():
        csv_path = create_dummy_data(csv_path)

    # Load tokenizer
    print('\nLoading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # Create dataset
    print('\nCreating dataset...')
    dataset = BindingEnergyDataset(csv_path, tokenizer, config, split='train')

    # Test __getitem__
    print('\nTesting __getitem__...')
    sample = dataset[0]
    print(f'  input_ids shape: {sample["input_ids"].shape}')
    print(f'  attention_mask shape: {sample["attention_mask"].shape}')
    print(f'  label: {sample["label"].item():.4f}')
    print(f'  label_raw: {sample["label_raw"].item():.4f}')

    print('\n✓ Dataset test passed')


def test_model():
    """Test model creation and forward pass"""
    print('\n' + '='*60)
    print('TEST 2: Model Creation and Forward Pass')
    print('='*60)

    config = Config()
    config.DEVICE = 'cpu'  # Force CPU for testing

    # Create model
    print('\nCreating model...')
    model = create_model(config)

    # Create dummy input
    batch_size = 4
    seq_len = 128
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len)

    # Forward pass
    print('\nTesting forward pass...')
    model.eval()
    with torch.no_grad():
        output = model(dummy_input_ids, dummy_attention_mask)

    print(f'  Input shape: {dummy_input_ids.shape}')
    print(f'  Output shape: {output.shape}')
    print(f'  Output values: {output}')

    assert output.shape == (batch_size,), f'Expected shape ({batch_size},), got {output.shape}'

    print('\n✓ Model test passed')


def test_dataloader():
    """Test dataloader"""
    print('\n' + '='*60)
    print('TEST 3: DataLoader')
    print('='*60)

    config = Config()
    config.BATCH_SIZE = 4
    config.DEVICE = 'cpu'

    # Create dummy data if needed
    csv_path = Path(config.DATA_DIR) / 'split_seed42.csv'
    if not csv_path.exists():
        csv_path = create_dummy_data(csv_path)

    # Create dataloaders
    print('\nCreating dataloaders...')
    train_loader, val_loader, test_loader, label_stats = create_dataloaders(
        csv_path, config, seed=42
    )

    print(f'\nLabel stats:')
    print(f'  Mean: {label_stats["mean"]:.4f}')
    print(f'  Std: {label_stats["std"]:.4f}')

    # Test iteration
    print('\nTesting train_loader iteration...')
    batch = next(iter(train_loader))
    print(f'  Batch keys: {batch.keys()}')
    print(f'  input_ids shape: {batch["input_ids"].shape}')
    print(f'  attention_mask shape: {batch["attention_mask"].shape}')
    print(f'  label shape: {batch["label"].shape}')

    print('\n✓ DataLoader test passed')


def test_training_step():
    """Test one training step"""
    print('\n' + '='*60)
    print('TEST 4: Training Step')
    print('='*60)

    config = Config()
    config.BATCH_SIZE = 4
    config.DEVICE = 'cpu'

    # Create dummy data if needed
    csv_path = Path(config.DATA_DIR) / 'split_seed42.csv'
    if not csv_path.exists():
        csv_path = create_dummy_data(csv_path)

    # Create dataloader
    train_loader, _, _, _ = create_dataloaders(csv_path, config, seed=42)

    # Create model
    model = create_model(config)
    model = model.to(config.DEVICE)

    # Create optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    # Get one batch
    batch = next(iter(train_loader))
    input_ids = batch['input_ids'].to(config.DEVICE)
    attention_mask = batch['attention_mask'].to(config.DEVICE)
    labels = batch['label'].to(config.DEVICE)

    # Training step
    print('\nPerforming one training step...')
    model.train()

    # Forward
    predictions = model(input_ids, attention_mask)
    loss = criterion(predictions, labels)

    print(f'  Predictions: {predictions}')
    print(f'  Labels: {labels}')
    print(f'  Loss: {loss.item():.4f}')

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('\n✓ Training step test passed')


def test_evaluation():
    """Test evaluation function"""
    print('\n' + '='*60)
    print('TEST 5: Evaluation')
    print('='*60)

    config = Config()
    config.BATCH_SIZE = 4
    config.DEVICE = 'cpu'

    # Create dummy data if needed
    csv_path = Path(config.DATA_DIR) / 'split_seed42.csv'
    if not csv_path.exists():
        csv_path = create_dummy_data(csv_path)

    # Create dataloader
    _, _, test_loader, label_stats = create_dataloaders(csv_path, config, seed=42)

    # Create model
    model = create_model(config)
    model = model.to(config.DEVICE)

    # Evaluate
    print('\nEvaluating model...')
    metrics, predictions, targets = evaluate(model, test_loader, label_stats, config.DEVICE)

    print(f'\nMetrics:')
    print(f'  RMSE: {metrics["RMSE"]:.4f}')
    print(f'  MAE: {metrics["MAE"]:.4f}')
    print(f'  R²: {metrics["R2"]:.4f}')

    print(f'\nPredictions shape: {predictions.shape}')
    print(f'Targets shape: {targets.shape}')

    print('\n✓ Evaluation test passed')


def run_all_tests():
    """Run all tests"""
    print('\n' + '='*60)
    print('RUNNING ALL TESTS')
    print('='*60)

    try:
        test_dataset()
        test_model()
        test_dataloader()
        test_training_step()
        test_evaluation()

        print('\n' + '='*60)
        print('ALL TESTS PASSED ✓')
        print('='*60)
        print('\nYou can now run the full training on GPU with:')
        print('  python train.py')

    except Exception as e:
        print(f'\n✗ Test failed with error:')
        print(f'  {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_tests()
