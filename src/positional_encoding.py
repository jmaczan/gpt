import torch
import torch.nn as nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, embeddings, context_window=20):
        super().__init__()

        _, _, embedding_dim = embeddings.size()
        self.embedding_dim = embedding_dim
        self.max_len = context_window
        self.positional_encodings = self.precompute_encodings(self, embeddings)

    def forward(self, embeddings):
        """
        embeddings = (batch, sequence_length (column), embedding_dim (row))
        """

        return embeddings + self.positional_encodings

    def precompute_encodings(self, embeddings):
        _, sequence_length, embedding_dim = embeddings.size()

        sequence_indices = torch.arange(sequence_length).view(sequence_length, 1)
        embedding_indices = torch.arange(embedding_dim).view(1, embedding_dim)

        even_index_mask = embedding_indices % 2 == 0
        odd_index_mask = ~even_index_mask

        positional_encodings = torch.zeros(
            (sequence_length, embedding_dim), dtype=torch.float
        )

        positional_encodings[even_index_mask] = torch.sin(
            torch.div(
                sequence_indices,
                torch.pow(
                    10000, (float(2 * torch.div(embedding_indices, embedding_dim)))
                ),
            )
        )[even_index_mask]

        positional_encodings[odd_index_mask] = torch.cos(
            torch.div(
                sequence_indices,
                torch.pow(
                    10000, (float(2 * torch.div(embedding_indices, embedding_dim)))
                ),
            )
        )[odd_index_mask]

        return positional_encodings
