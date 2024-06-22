import torch
import torch.nn as nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length, embedding_dim, n=10_000):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.positional_encodings = self.precompute_encodings(
            sequence_length, embedding_dim, n
        ).to(device)

    def forward(self, embeddings):
        """
        embeddings = (batch, sequence_length (column), embedding_dim (row))
        """
        return embeddings + self.positional_encodings[:embeddings.size(1), :]

    def precompute_encodings(self, sequence_length, embedding_dim, n=10_000):

        sequence_indices = torch.arange(sequence_length).view(sequence_length, 1)
        embedding_indices = torch.arange(embedding_dim).view(1, embedding_dim)

        even_index_mask = embedding_indices % 2 == 0
        odd_index_mask = ~even_index_mask

        positional_encodings = torch.zeros(
            (sequence_length, embedding_dim), dtype=torch.float32
        )

        positional_encodings[:, even_index_mask[0]] = torch.sin(
            sequence_indices
            * torch.div(
                1,
                torch.pow(
                    n,
                    (2 * torch.div(embedding_indices // 2, embedding_dim)),
                )[0],
            )
        )[:, even_index_mask[0]]

        positional_encodings[:, odd_index_mask[0]] = torch.cos(
            torch.div(
                sequence_indices,
                torch.pow(
                    n,
                    (2 * torch.div(embedding_indices // 2, embedding_dim)),
                ),
            )
        )[:, odd_index_mask[0]]

        return positional_encodings
