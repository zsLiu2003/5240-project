"""
ECFP + Ridge regression baseline for binding energy prediction.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from config import Config


ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol

    # Some inorganic salts in this dataset violate RDKit's default valence
    # sanitization rules. Keep graph connectivity for fingerprinting.
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def smiles_to_ecfp(smiles_list, radius=2, n_bits=2048):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprints = []

    for smiles in smiles_list:
        mol = mol_from_smiles(smiles)
        if mol is None:
            raise ValueError(f'Invalid SMILES: {smiles}')
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.float32)
        # ConvertToNumpyArray is intentionally avoided for compatibility
        # across RDKit builds.
        on_bits = list(fp.GetOnBits())
        arr[on_bits] = 1.0
        fingerprints.append(arr)

    return np.vstack(fingerprints)


def compute_metrics(targets, predictions):
    return {
        'RMSE': float(mean_squared_error(targets, predictions) ** 0.5),
        'MAE': float(mean_absolute_error(targets, predictions)),
        'R2': float(r2_score(targets, predictions)),
    }


def train_one_seed(config, seed, output_dir):
    csv_path = project_path(config.DATA_DIR) / f'split_seed{seed}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'Split file not found: {csv_path}')

    df = pd.read_csv(csv_path)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    x_train = smiles_to_ecfp(train_df[config.SMILES_COL])
    x_val = smiles_to_ecfp(val_df[config.SMILES_COL])
    x_test = smiles_to_ecfp(test_df[config.SMILES_COL])

    y_train = train_df[config.LABEL_COL].to_numpy(dtype=np.float32)
    y_val = val_df[config.LABEL_COL].to_numpy(dtype=np.float32)
    y_test = test_df[config.LABEL_COL].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    best_alpha = None
    best_val_rmse = float('inf')
    best_model = None

    for alpha in ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        val_rmse = mean_squared_error(y_val, val_pred) ** 0.5
        if val_rmse < best_val_rmse:
            best_alpha = alpha
            best_val_rmse = val_rmse
            best_model = model

    test_pred = best_model.predict(x_test)
    metrics = compute_metrics(y_test, test_pred)

    seed_dir = output_dir / f'baseline_ecfp_ridge_seed{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        'true': y_test,
        'pred': test_pred,
        'error': test_pred - y_test,
    }).to_csv(seed_dir / 'predictions.csv', index=False)

    return {
        'seed': seed,
        'best_alpha': best_alpha,
        'val_rmse': float(best_val_rmse),
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description='Train ECFP + Ridge baseline.')
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--results-dir', default=None)
    args = parser.parse_args()

    config = Config()
    if args.data_dir is not None:
        config.DATA_DIR = args.data_dir
    if args.results_dir is not None:
        config.RESULTS_DIR = args.results_dir

    output_dir = project_path(config.RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [train_one_seed(config, seed, output_dir) for seed in config.SEEDS]
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / 'baseline_results.csv', index=False)

    print(summary.to_string(index=False))
    print('\nAggregated results:')
    for metric in ['RMSE', 'MAE', 'R2']:
        print(f'{metric}: {summary[metric].mean():.4f} ± {summary[metric].std(ddof=0):.4f}')
    print(f'\nSummary saved to {output_dir / "baseline_results.csv"}')


if __name__ == '__main__':
    main()
