import torch
import torch.nn as nn

from attention_head import AttentionHead


class MultiHeadAttention(nn.Module):
    def __init__(self, embeddings_dim, heads_count):
        super().__init__()

        self.heads_count = heads_count
        self.embeddings_dim = embeddings_dim
        self.single_head_size = embeddings_dim // heads_count
        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    embedding_dim=self.single_head_size,
                )
                for _ in range(self.heads_count)
            ]
        )

        self.W_O = nn.Parameter(torch.empty((embeddings_dim, embeddings_dim)))

        nn.init.xavier_uniform_(self.W_O)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]

        concatenated_outputs = torch.cat(head_outputs, dim=2)

        output = self.W_O @ concatenated_outputs

        return output
