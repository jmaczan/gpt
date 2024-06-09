import torch
import torch.nn as nn
from torch.nn import functional as F

from positional_encoding import PositionalEncoding

torch.manual_seed(1995)


class AttentionHead(nn.Module):

    def __init__(self, embeddings):
        super().__init__()

        batch, sequence_length, embedding_dim = embeddings.shape

        self.W_Q = torch.empty((sequence_length, embedding_dim))
        self.W_K = torch.empty((sequence_length, embedding_dim))
        self.W_V = torch.empty((sequence_length, embedding_dim))

        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)

    def forward(self, embeddings):
        batch, sequence_length, embeddings_dim = embeddings.shape

        embeddings = PositionalEncoding(
            sequence_length=sequence_length, embedding_dim=embeddings_dim
        ).forward(embeddings)
