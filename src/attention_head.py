import torch
import torch.nn as nn


class AttentionHead(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.W_Q = nn.Parameter(torch.empty((embedding_dim, embedding_dim)))
        self.W_K = nn.Parameter(torch.empty((embedding_dim, embedding_dim)))
        self.W_V = nn.Parameter(torch.empty((embedding_dim, embedding_dim)))

        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)

    def forward(self, embeddings):
        batch_size, sequence_length, embeddings_dim = embeddings.shape

        # compute Q, K and V for each token in each of embeddings
        Q = embeddings @ self.W_Q
        K = embeddings @ self.W_K
        V = embeddings @ self.W_V

        attention_scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(embeddings_dim, dtype=torch.float32)
        )

        # build -inf upper triangle mask
        mask = torch.triu(
            torch.full((sequence_length, sequence_length), float("-inf")), diagonal=1
        )
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1).to(embeddings.device)

        masked_scores = attention_scores + mask

        # softmax
        probabilities = torch.nn.Softmax(dim=-1)(masked_scores)

        output = probabilities @ V

        return output
