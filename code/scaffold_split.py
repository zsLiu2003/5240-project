"""
Scaffold split script - generate train/val/test splits
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict


def generate_scaffold(smiles):
    """
    Generate Bemis-Murcko scaffold for a SMILES string

    Args:
        smiles: SMILES string

    Returns:
        scaffold: Scaffold SMILES (or None if invalid)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold
    except:
        return None


def scaffold_split(df, smiles_col='SMILES', frac_train=0.7, frac_val=0.15, frac_test=0.15, seed=42):
    """
    Split dataset by molecular scaffolds

    Args:
        df: DataFrame with SMILES column
        smiles_col: Name of SMILES column
        frac_train: Fraction for training set
        frac_val: Fraction for validation set
        frac_test: Fraction for test set
        seed: Random seed

    Returns:
        df: DataFrame with added 'split' column
    """
    np.random.seed(seed)

    # Generate scaffolds for all molecules
    print('Generating scaffolds...')
    scaffolds = {}
    for idx, smiles in enumerate(df[smiles_col]):
        scaffold = generate_scaffold(smiles)
        if scaffold is None:
            print(f'Warning: Could not generate scaffold for SMILES at index {idx}: {smiles}')
            scaffold = smiles  # Use original SMILES as fallback
        scaffolds[idx] = scaffold

    # Group molecules by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, scaffold in scaffolds.items():
        scaffold_to_indices[scaffold].append(idx)

    # Convert to list and shuffle
    scaffold_sets = list(scaffold_to_indices.values())
    np.random.shuffle(scaffold_sets)

    print(f'Total molecules: {len(df)}')
    print(f'Unique scaffolds: {len(scaffold_sets)}')

    # Assign scaffolds to splits
    n_total = len(df)
    n_train = int(frac_train * n_total)
    n_val = int(frac_val * n_total)

    train_indices = []
    val_indices = []
    test_indices = []

    for scaffold_set in scaffold_sets:
        if len(train_indices) < n_train:
            train_indices.extend(scaffold_set)
        elif len(val_indices) < n_val:
            val_indices.extend(scaffold_set)
        else:
            test_indices.extend(scaffold_set)

    # Create split column
    df['split'] = 'test'  # Default
    df.loc[train_indices, 'split'] = 'train'
    df.loc[val_indices, 'split'] = 'val'

    # Print statistics
    print(f'\nSplit statistics:')
    print(f'  Train: {len(train_indices)} ({len(train_indices)/n_total*100:.1f}%)')
    print(f'  Val:   {len(val_indices)} ({len(val_indices)/n_total*100:.1f}%)')
    print(f'  Test:  {len(test_indices)} ({len(test_indices)/n_total*100:.1f}%)')

    return df


def create_scaffold_splits(input_csv, output_dir, seeds=[42, 123, 456]):
    """
    Create scaffold splits for multiple seeds

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
        print(f'Generating scaffold split for seed {seed}')
        print(f'{"="*60}')

        # Create split
        df_split = scaffold_split(df.copy(), smiles_col='SMILES', seed=seed)

        # Save
        output_path = output_dir / f'split_seed{seed}.csv'
        df_split.to_csv(output_path, index=False)
        print(f'\nSaved to {output_path}')

        # Verify scaffold separation
        train_scaffolds = set()
        test_scaffolds = set()

        for idx, row in df_split.iterrows():
            scaffold = generate_scaffold(row['SMILES'])
            if row['split'] == 'train':
                train_scaffolds.add(scaffold)
            elif row['split'] == 'test':
                test_scaffolds.add(scaffold)

        overlap = train_scaffolds & test_scaffolds
        print(f'\nScaffold verification:')
        print(f'  Train scaffolds: {len(train_scaffolds)}')
        print(f'  Test scaffolds: {len(test_scaffolds)}')
        print(f'  Overlap: {len(overlap)}')

        if len(overlap) > 0:
            print(f'  Warning: {len(overlap)} scaffolds appear in both train and test!')
        else:
            print(f'  ✓ No scaffold leakage between train and test')


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
    create_scaffold_splits(input_csv, output_dir, seeds=[42, 123, 456])

    print(f'\n{"="*60}')
    print('Scaffold split completed!')
    print(f'{"="*60}')
    print('\nGenerated files:')
    print('  - split_seed42.csv')
    print('  - split_seed123.csv')
    print('  - split_seed456.csv')
    print('\nYou can now run training with:')
    print('  python train.py')
