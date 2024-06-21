import torch
import torch.nn as nn

from src.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, embeddings_dim, heads_count):
        super().__init__()

        self.heads_count = heads_count
        self.embeddings_dim = embeddings_dim

        self.layer_norm1 = nn.LayerNorm(embeddings_dim)
        self.multi_head_attention = MultiHeadAttention(
            embeddings_dim=embeddings_dim, heads_count=heads_count
        )
        self.dropout1 = nn.Dropout(p=0.1)
        self.layer_norm2 = nn.LayerNorm(embeddings_dim)
        self.linear1 = nn.Linear(
            in_featsures=embeddings_dim, out_features=embeddings_dim * 4
        )
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(
            in_features=embeddings_dim * 4, out_features=embeddings_dim
        )
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = self.multi_head_attention(x)
        x = self.dropout1(x)
        x = residual + x

        residual = x
        x = self.layer_norm2(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear(x)
        x = self.dropout2(x)
        x = residual + x

        return x
