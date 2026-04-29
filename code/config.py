"""
Configuration file for ChemBERTa binding energy prediction
"""

class Config:
    # Data paths
    DATA_DIR = 'data'
    RESULTS_DIR = 'results'

    # Model settings
    MODEL_NAME = 'seyonec/ChemBERTa-zinc-base-v1'  # HuggingFace model
    FREEZE_ENCODER = True  # Main method: freeze encoder
    FROM_SCRATCH = False  # Ablation 2: random initialized encoder
    HIDDEN_SIZE = 768  # ChemBERTa hidden size
    EXPERIMENT_NAME = 'main_method'
    SUMMARY_FILENAME = 'main_method_summary.csv'

    # Regression head architecture
    HEAD_HIDDEN_DIM = 256
    HEAD_DROPOUT = 0.1

    # Training hyperparameters
    BATCH_SIZE = 16
    MAX_EPOCHS = 100
    LEARNING_RATE = 1e-4  # Initial peak LR; cosine schedule decays from here
    WEIGHT_DECAY = 0.01
    EARLY_STOPPING_PATIENCE = 10
    WARMUP_RATIO = 0.1  # Fraction of total steps for linear warmup

    # Two-stage training (main method only)
    TWO_STAGE = False
    STAGE1_MAX_EPOCHS = 10
    STAGE1_PATIENCE = 3
    STAGE1_LR_HEAD = 5e-4
    STAGE2_MAX_EPOCHS = 20
    STAGE2_PATIENCE = 5
    STAGE2_LR_HEAD = 1e-4
    STAGE2_LR_BACKBONE = 1e-5
    STAGE2_UNFREEZE_LAYERS = None  # None = full unfreeze; int N = last N transformer layers

    # Data settings
    MAX_LENGTH = 128  # Max SMILES token length
    SEEDS = [42, 123, 456]  # Random seeds for robustness
    USE_CV = False  # Use K-fold cross-validation
    CV_FOLDS = 5  # Number of CV folds

    # Data augmentation settings
    AUGMENT_SMILES = False  # Enable SMILES augmentation
    AUGMENTATION_PROB = 0.5  # Augmentation probability (0.0-1.0)
    AUGMENTATION_SEED = 42  # Seed for on-the-fly randomized SMILES

    # Label settings
    LABEL_COL = 'Energy_min'  # Use strongest binding energy
    SMILES_COL = 'SMILES'

    # Device
    DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

    # Logging
    VERBOSE = True
    SAVE_MODEL = True

    def __repr__(self):
        return '\n'.join([f'{k}: {v}' for k, v in self.__dict__.items() if not k.startswith('_')])
