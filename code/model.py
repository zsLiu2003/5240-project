"""
Model definition: ChemBERTa + Regression Head
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class RegressionHead(nn.Module):
    """
    2-layer feedforward network for regression
    """

    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] - [CLS] embeddings

        Returns:
            [batch_size] - predicted binding energies
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


class ChemBERTaRegressor(nn.Module):
    """
    ChemBERTa encoder + regression head for binding energy prediction
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Load encoder. From-scratch ablation keeps the same architecture but
        # intentionally discards pretrained weights.
        model_kwargs = {}
        if getattr(config, 'ATTN_IMPLEMENTATION', None) is not None:
            model_kwargs['attn_implementation'] = config.ATTN_IMPLEMENTATION

        if getattr(config, 'FROM_SCRATCH', False):
            encoder_config = AutoConfig.from_pretrained(config.MODEL_NAME, **model_kwargs)
            self.encoder = AutoModel.from_config(encoder_config)
            print('Encoder randomly initialized (from-scratch mode)')
        else:
            self.encoder = AutoModel.from_pretrained(config.MODEL_NAME, **model_kwargs)

        # Freeze encoder if specified
        if config.FREEZE_ENCODER:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print('Encoder frozen (linear probing mode)')
        else:
            print('Encoder trainable (full fine-tuning mode)')

        # Regression head
        self.head = RegressionHead(
            input_dim=config.HIDDEN_SIZE,
            hidden_dim=config.HEAD_HIDDEN_DIM,
            dropout=config.HEAD_DROPOUT
        )

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            predictions: [batch_size] - predicted binding energies
        """
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Extract [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Pass through regression head
        predictions = self.head(cls_embedding)

        return predictions

    def get_trainable_params(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters())


def create_model(config):
    """
    Create model and print parameter info

    Args:
        config: Config object

    Returns:
        model: ChemBERTaRegressor
    """
    model = ChemBERTaRegressor(config)

    total_params = model.get_total_params()
    trainable_params = model.get_trainable_params()

    print(f'\nModel created:')
    print(f'  Total parameters: {total_params:,}')
    print(f'  Trainable parameters: {trainable_params:,}')
    print(f'  Frozen parameters: {total_params - trainable_params:,}')
    print(f'  Trainable ratio: {trainable_params / total_params * 100:.2f}%')

    return model
