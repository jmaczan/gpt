import torch
import torch.nn as nn
from torch.nn import functional as F

from src.multi_head_attention import MultiHeadAttention
from src.positional_encoding import PositionalEncoding

torch.manual_seed(1995)

default_context_window = 20
default_embedding_dimension = 8
default_vocabulary_size = 300
default_attention_heads_count = 8


class GPT(nn.Module):
    """
    Works well with github.com/jmaczan/bpe-tokenizer

    About this model:
    - I learn details while I build, so expect a messy code, at least until it's declared as finished and polished
    - Decoder-only
    - Might not match all the details of original GPT-2/3, but in general follows the same implementation rules
    """

    def __init__(
        self,
        vocabulary_size=default_vocabulary_size,
        embedding_dimension=default_embedding_dimension,
        context_window=default_context_window,
        heads_count=default_attention_heads_count,
    ):
        super().__init__()

        self.embeddings = nn.Embedding(vocabulary_size, embedding_dimension)
        self.positional_encoding = PositionalEncoding(
            sequence_length=context_window,
            embedding_dim=embedding_dimension,
        )
        self.model = nn.Sequential(
            [
                nn.ModuleList(
                    [
                        MultiHeadAttention(
                            embeddings_dim=embedding_dimension, heads_count=heads_count
                        )
                    ]
                ),
            ]
        )
        self.context_window = context_window

    def forward(self, x):
        return self.model(x)
