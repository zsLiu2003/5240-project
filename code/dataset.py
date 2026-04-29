"""
Dataset class for ChemBERTa binding energy prediction
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from transformers import AutoTokenizer
import numpy as np
import random
from sklearn.model_selection import KFold


class BindingEnergyDataset(Dataset):
    """
    Dataset for SMILES -> Binding Energy regression
    """

    def __init__(self, csv_path, tokenizer, config, split='train', seed=42):
        """
        Args:
            csv_path: Path to CSV file with columns [SMILES, Energy_min, split]
            tokenizer: HuggingFace tokenizer
            config: Config object
            split: 'train', 'val', or 'test'
        """
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.augmentation_rng = random.Random(
            int(getattr(config, 'AUGMENTATION_SEED', seed)) + {'train': 0, 'val': 1, 'test': 2}.get(split, 0)
        )

        # Load data
        df = pd.read_csv(csv_path)

        # Filter by split
        self.data = df[df['split'] == split].reset_index(drop=True)

        # Extract SMILES and labels
        self.smiles = self.data[config.SMILES_COL].values
        self.labels = self.data[config.LABEL_COL].values.astype(np.float32)

        # Compute label statistics (for normalization)
        if split == 'train':
            self.label_mean = self.labels.mean()
            self.label_std = self.labels.std()
        else:
            # Will be set externally from training set stats
            self.label_mean = None
            self.label_std = None

        print(f'{split.upper()} set: {len(self)} samples')
        print(f'  Label range: [{self.labels.min():.3f}, {self.labels.max():.3f}]')
        if split == 'train':
            print(f'  Label mean: {self.label_mean:.3f}, std: {self.label_std:.3f}')
            if getattr(self.config, 'AUGMENT_SMILES', False):
                print(f'  SMILES augmentation: on-the-fly randomization '
                      f'(p={getattr(self.config, "AUGMENTATION_PROB", 0.5):.2f})')

    def set_label_stats(self, mean, std):
        """Set label normalization stats from training set"""
        self.label_mean = mean
        self.label_std = std

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        """
        Returns:
            input_ids: Tokenized SMILES
            attention_mask: Attention mask
            label: Normalized binding energy
        """
        smiles = self.smiles[idx]
        label = self.labels[idx]

        # Data augmentation (only for training set)
        if self.split == 'train' and getattr(self.config, 'AUGMENT_SMILES', False):
            from smiles_augmentation import augment_smiles
            smiles = augment_smiles(
                smiles,
                augmentation_prob=getattr(self.config, 'AUGMENTATION_PROB', 0.5),
                rng=self.augmentation_rng,
            )

        # Tokenize SMILES
        encoding = self.tokenizer(
            smiles,
            max_length=self.config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Normalize label
        if self.label_mean is not None and self.label_std is not None:
            label_normalized = (label - self.label_mean) / self.label_std
        else:
            label_normalized = label

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label_normalized, dtype=torch.float32),
            'label_raw': torch.tensor(label, dtype=torch.float32)  # For evaluation
        }


def create_dataloaders(csv_path, config, seed=42):
    """
    Create train/val/test dataloaders

    Args:
        csv_path: Path to split CSV file
        config: Config object
        seed: Random seed

    Returns:
        train_loader, val_loader, test_loader, label_stats
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # Create datasets
    train_dataset = BindingEnergyDataset(csv_path, tokenizer, config, split='train', seed=seed)
    val_dataset = BindingEnergyDataset(csv_path, tokenizer, config, split='val', seed=seed)
    test_dataset = BindingEnergyDataset(csv_path, tokenizer, config, split='test', seed=seed)

    # Set label stats for val/test from training set
    label_stats = {
        'mean': train_dataset.label_mean,
        'std': train_dataset.label_std
    }
    val_dataset.set_label_stats(label_stats['mean'], label_stats['std'])
    test_dataset.set_label_stats(label_stats['mean'], label_stats['std'])

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging, increase on GPU
        pin_memory=True if config.DEVICE == 'cuda' else False
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )

    return train_loader, val_loader, test_loader, label_stats


def create_cv_dataloaders(csv_path, config, seed, fold_idx, n_folds=5):
    """
    Create train/val/test dataloaders for K-fold CV.

    CV is done on train+val combined; test set remains fixed.

    Args:
        csv_path: Path to split CSV file
        config: Config object
        seed: Random seed for KFold
        fold_idx: Current fold index (0 to n_folds-1)
        n_folds: Number of folds

    Returns:
        train_loader, val_loader, test_loader, label_stats
    """
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    df = pd.read_csv(csv_path)

    # Separate test set (fixed)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    # Combine train+val for CV
    dev_df = df[df['split'].isin(['train', 'val'])].reset_index(drop=True)

    # K-fold split on dev_df
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(kf.split(dev_df))
    train_idx, val_idx = splits[fold_idx]

    # Create temporary CSV-like dataframes with 'split' column
    train_fold_df = dev_df.iloc[train_idx].copy()
    train_fold_df['split'] = 'train'
    val_fold_df = dev_df.iloc[val_idx].copy()
    val_fold_df['split'] = 'val'
    test_df_copy = test_df.copy()
    test_df_copy['split'] = 'test'

    # Combine and save to temp CSV
    import tempfile
    temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    combined = pd.concat([train_fold_df, val_fold_df, test_df_copy], ignore_index=True)
    combined.to_csv(temp_csv.name, index=False)
    temp_csv.close()

    # Use existing create_dataloaders
    train_loader, val_loader, test_loader, label_stats = create_dataloaders(
        temp_csv.name, config, seed
    )

    # Clean up temp file
    import os
    os.unlink(temp_csv.name)

    print(f'  CV Fold {fold_idx+1}/{n_folds}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_df)}')

    return train_loader, val_loader, test_loader, label_stats
