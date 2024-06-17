import torch
import torch.nn as nn
from torch.nn import functional as F

from src.positional_encoding import PositionalEncoding

torch.manual_seed(1995)


class AttentionHead(nn.Module):

    def __init__(self, embedding_dim, positional_encoding):
        super().__init__()

        self.positional_encoding = positional_encoding

        self.W_Q = nn.Parameter(torch.empty((embedding_dim, embedding_dim)))
        self.W_K = nn.Parameter(torch.empty((embedding_dim, embedding_dim)))
        self.W_V = nn.Parameter(torch.empty((embedding_dim, embedding_dim)))

        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)

    def forward(self, embeddings):
        _, sequence_length, embeddings_dim = (
            embeddings.shape
        )  # it doesn't take into account splitting embedding_dim per attention head (d_head = d_model / h), where h is number of attention heads

        embeddings = self.positional_encoding.forward(embeddings)

        # compute Q, K and V for each token in each of embeddings
        Q = embeddings @ self.W_Q
        K = embeddings @ self.W_K
        V = embeddings @ self.W_V

        attention_scores = (Q @ K.transpose(1, 2)) / torch.sqrt(
            torch.tensor(embeddings_dim, dtype=torch.float32)
        )

        # build -inf upper triangle mask
        mask = torch.triu(torch.ones((1, sequence_length, sequence_length)), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))

        masked_scores = attention_scores + mask

        # softmax
        softmax = torch.nn.Softmax(dim=1)
        probabilities = softmax(masked_scores)

        output = probabilities @ V

        return output
