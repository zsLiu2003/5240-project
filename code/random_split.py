"""
Random split script for molecular dataset
(Scaffold split not applicable for inorganic salts)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def random_split(df, frac_train=0.7, frac_val=0.15, frac_test=0.15, seed=42):
    """
    Random split dataset

    Args:
        df: DataFrame
        frac_train: Fraction for training
        frac_val: Fraction for validation
        frac_test: Fraction for test
        seed: Random seed

    Returns:
        df: DataFrame with 'split' column
    """
    np.random.seed(seed)

    n_total = len(df)
    n_train = int(frac_train * n_total)
    n_val = int(frac_val * n_total)

    # Shuffle indices
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    # Assign splits
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]

    # Create split column
    df['split'] = 'test'
    df.loc[train_indices, 'split'] = 'train'
    df.loc[val_indices, 'split'] = 'val'

    print(f'Split statistics:')
    print(f'  Train: {len(train_indices)} ({len(train_indices)/n_total*100:.1f}%)')
    print(f'  Val:   {len(val_indices)} ({len(val_indices)/n_total*100:.1f}%)')
    print(f'  Test:  {len(test_indices)} ({len(test_indices)/n_total*100:.1f}%)')

    return df


def create_random_splits(input_csv, output_dir, seeds=[42, 123, 456]):
    """
    Create random splits for multiple seeds

    Args:
        input_csv: Path to clean_dataset.csv
        output_dir: Directory to save split files
        seeds: List of random seeds
    """
    # Load data
    print(f'Loading data from {input_csv}...')
    df = pd.read_csv(input_csv)

    print(f'Dataset shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')

    # Check required columns
    required_cols = ['SMILES', 'Energy_min']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Required column "{col}" not found in dataset')

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate splits for each seed
    for seed in seeds:
        print(f'\n{"="*60}')
        print(f'Generating random split for seed {seed}')
        print(f'{"="*60}')

        # Create split
        df_split = random_split(df.copy(), seed=seed)

        # Save
        output_path = output_dir / f'split_seed{seed}.csv'
        df_split.to_csv(output_path, index=False)
        print(f'\nSaved to {output_path}')

        # Print label statistics per split
        for split_name in ['train', 'val', 'test']:
            split_data = df_split[df_split['split'] == split_name]
            if len(split_data) > 0:
                print(f'\n{split_name.upper()} set label stats:')
                print(f'  Mean: {split_data["Energy_min"].mean():.4f} eV')
                print(f'  Std:  {split_data["Energy_min"].std():.4f} eV')
                print(f'  Min:  {split_data["Energy_min"].min():.4f} eV')
                print(f'  Max:  {split_data["Energy_min"].max():.4f} eV')


if __name__ == '__main__':
    # Paths
    input_csv = '../data/clean_dataset.csv'
    output_dir = '../data'

    # Check if input exists
    if not Path(input_csv).exists():
        print(f'Error: Input file not found: {input_csv}')
        print('Please make sure you have run the data preparation step first.')
        exit(1)

    # Create splits
    create_random_splits(input_csv, output_dir, seeds=[42, 123, 456])

    print(f'\n{"="*60}')
    print('Random split completed!')
    print(f'{"="*60}')
    print('\nGenerated files:')
    print('  - split_seed42.csv')
    print('  - split_seed123.csv')
    print('  - split_seed456.csv')
    print('\nNote: Using random split instead of scaffold split')
    print('because most molecules are inorganic salts without scaffolds.')
    print('\nYou can now run training with:')
    print('  python train.py')
