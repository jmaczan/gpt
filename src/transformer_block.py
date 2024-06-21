import torch
import torch.nn as nn

from src.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, embeddings_dim, heads_count):
        super().__init__()

        self.heads_count = heads_count
        self.embeddings_dim = embeddings_dim

        self.model = [
            nn.LayerNorm(),
            nn.Linear(),
            MultiHeadAttention(embeddings_dim=embeddings_dim, heads_count=heads_count),
            nn.Linear(),
            nn.Dropout(),
            nn.LayerNorm(),
            nn.Linear(),
            nn.GELU(),
            nn.Linear(),
            nn.Dropout(),
        ]

    def forward(self, x):
        return self.model(x)
