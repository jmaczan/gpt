import torch
import torch.nn as nn
from torch.nn import functional as F

from positional_encoding import PositionalEncoding

torch.manual_seed(1995)


class AttentionHead(nn.Module):

    def __init__(self, embeddings):
        super().__init__()

        _, _, embedding_dim = embeddings.shape

        self.W_Q = torch.empty((embedding_dim, embedding_dim))
        self.W_K = torch.empty((embedding_dim, embedding_dim))
        self.W_V = torch.empty((embedding_dim, embedding_dim))

        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)

    def forward(self, embeddings):
        _, sequence_length, embeddings_dim = (
            embeddings.shape
        )  # it doesn't take into account splitting embedding_dim per attention head (d_head = d_model / h), where h is number of attention heads

        embeddings = PositionalEncoding(  # I think it can be pulled out to __init__ and reused across batches (good)
            # or pulled to gpt.py and reused across attention heads (better)
            sequence_length=sequence_length,
            embedding_dim=embeddings_dim,
        ).forward(
            embeddings
        )

        # compute Q, K and V for each token in each of embeddings
        Q = embeddings @ self.W_Q
        K = embeddings @ self.W_K
        V = embeddings @ self.W_V

        attention_scores = Q @ K.transpose(1, 2)

        # build -inf upper triangle mask
        mask = torch.triu(attention_scores, diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))

        masked_scores = attention_scores + mask

        # softmax
        probabilities = torch.nn.Softmax(masked_scores)

        output = probabilities @ V

        return output
