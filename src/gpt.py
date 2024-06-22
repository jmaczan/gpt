import torch
import torch.nn as nn

from positional_encoding import PositionalEncoding
from transformer_block import TransformerBlock

torch.manual_seed(1995)

default_context_window = 20
default_embedding_dimension = 8
default_vocabulary_size = 300
default_attention_heads_count = 4
default_transformer_blocks_count = 4
default_batch_size = 32


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
        blocks_count=default_transformer_blocks_count,
    ):
        super().__init__()

        self.blocks_count = blocks_count
        self.context_window = context_window

        self.embeddings = nn.Embedding(vocabulary_size, embedding_dimension)
        self.positional_encoding = PositionalEncoding(
            sequence_length=context_window,
            embedding_dim=embedding_dimension,
        )
        self.dropout = nn.Dropout(p=0.1)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embeddings_dim=embedding_dimension, heads_count=heads_count
                )
                for _ in range(self.blocks_count)
            ]
        )
        self.layer_norm = nn.LayerNorm(embedding_dimension)
        self.linear = nn.Linear(embedding_dimension, vocabulary_size)

    def forward(self, x):
        x = self.embeddings(x)
        x = x + self.positional_encoding(x)
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.layer_norm(x)
        x = self.linear(x)

        return x
