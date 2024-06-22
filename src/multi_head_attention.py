import torch
import torch.nn as nn

from attention_head import AttentionHead


class MultiHeadAttention(nn.Module):
    def __init__(self, embeddings_dim, heads_count):
        super().__init__()

        self.heads_count = heads_count
        self.embeddings_dim = embeddings_dim
        assert (
            embeddings_dim % heads_count == 0
        ), "embedding_dim must be divisible by heads_count"
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
        batch_size, sequence_length, embedding_dim = x.size()

        x = x.view(batch_size, sequence_length, self.heads_count, self.single_head_size)

        x = x.transpose(1, 2)

        head_outputs = [head(x[:, i, :, :]) for i, head in enumerate(self.heads)]

        concatenated_outputs = torch.cat(head_outputs, dim=-1)

        concatenated_outputs = (
            concatenated_outputs.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, embedding_dim)
        )
        # output = self.W_O @ concatenated_outputs

        output = torch.matmul(concatenated_outputs, self.W_O)

        return output
