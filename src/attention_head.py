import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_size):
        super().__init__()

        self.head_size = head_size

        self.W_Q = nn.Linear(embedding_dim, head_size, bias=False)
        self.W_K = nn.Linear(embedding_dim, head_size, bias=False)
        self.W_V = nn.Linear(embedding_dim, head_size, bias=False)

        self.dropout = nn.Dropout(0.1)

        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)

    def forward(self, embeddings):
        batch_size, sequence_length, embeddings_dim = embeddings.shape

        Q = self.W_Q(embeddings)
        K = self.W_K(embeddings)

        attention_scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_size, dtype=torch.float32, device=embeddings.device)
        )

        mask = torch.tril(
            torch.ones(sequence_length, sequence_length, device=embeddings.device)
        )

        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
        attention_scores = torch.nn.Softmax(dim=-1)(attention_scores)
        attention_scores = self.dropout(attention_scores)

        output = self.W_V(attention_scores)

        return output
