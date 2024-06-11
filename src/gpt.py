import torch
import torch.nn as nn
from torch.nn import functional as F

from attention_head import AttentionHead

torch.manual_seed(1995)

default_context_window = 20
default_embedding_dimension = 8
default_vocabulary_size = 300


class GPT(nn.Module):
    """
    Works well with github.com/jmaczan/bpe-tokenizer

    About this model:
    - I learn details while I build, so except messy code, at least until it's declared as finished and polished
    - Decoder-only
    - Might not match all the details of original GPT-2/3, but in general follows the same implementation rules
    """

    def __init__(
        self,
        vocabulary_size=default_vocabulary_size,
        embedding_dimension=default_embedding_dimension,
        context_window=default_context_window,
    ):
        super().__init__()

        self.embeddings = nn.Embedding(vocabulary_size, embedding_dimension)
        self.model = nn.Sequential([AttentionHead])
        self.context_window = context_window

    def forward(self, x):
        return self.model(x)
