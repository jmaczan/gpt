import torch
import torch.nn as nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length, embedding_dim):
        super().__init__()

        self.positional_encodings = self.precompute_encodings(
            sequence_length, embedding_dim
        )

    def forward(self, embeddings):
        """
        embeddings = (batch, sequence_length (column), embedding_dim (row))
        """

        return embeddings + self.positional_encodings

    def precompute_encodings(self, sequence_length, embedding_dim):

        sequence_indices = torch.arange(sequence_length).view(sequence_length, 1)
        embedding_indices = torch.arange(embedding_dim).view(1, embedding_dim)

        even_index_mask = embedding_indices % 2 == 0
        odd_index_mask = ~even_index_mask

        positional_encodings = torch.zeros(
            (sequence_length, embedding_dim), dtype=torch.float32
        )

        positional_encodings[:, even_index_mask[0]] = torch.sin(
            torch.div(
                sequence_indices,
                torch.pow(
                    10000,
                    (2 * torch.div(embedding_indices[even_index_mask], embedding_dim)),
                ),
            )
        )

        positional_encodings[:, odd_index_mask[0]] = torch.cos(
            torch.div(
                sequence_indices,
                torch.pow(
                    10000,
                    (2 * torch.div(embedding_indices[odd_index_mask], embedding_dim)),
                ),
            )
        )

        return positional_encodings
