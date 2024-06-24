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

        Q = embeddings @ self.W_Q
        K = embeddings @ self.W_K
        V = embeddings @ self.W_V

        attention_scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(embeddings_dim, dtype=torch.float32, device=embeddings.device)
        )

        mask = torch.triu(
            torch.full(
                (sequence_length, sequence_length),
                float("-inf"),
                device=embeddings.device,
            ),
            diagonal=1,
        )
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        masked_scores = attention_scores + mask
        probabilities = torch.nn.Softmax(dim=-1)(masked_scores)

        output = probabilities @ V

        return output
