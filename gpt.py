import math
import torch
import torch.nn as nn
from torch.nn import functional as F

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
        self.model = nn.Sequential([])


class PositionalEncoding(nn.Module):
    def __init__(self, embeddings, context_window=20):
        super().__init__()
        # I assume it has 3 dimensions - batch, columns and rows, but I might be completely off here

        batch, d_model = embeddings.size()
        self.d_model = d_model
        self.max_len = context_window

    def forward(self, embeddings):
        """
        embeddings = (batch, number_of_embeddings (column), embedding_dimension (row))
        """
        batch, embeddings = embeddings.size()

        mapped_embeddings = embeddings

        for batch_index, batch_item in enumerate(batch):
            for embedding_index, embedding in enumerate(batch_item):
                for index, position in enumerate(embedding):
                    mapped_embeddings[batch_index][embedding_index][index] = (
                        position
                        + (
                            self.pe_even(embedding_index, index, self.d_model)
                            if index % 2 == 0
                            else self.pe_odd(embedding_index, index, self.d_model)
                        )
                    )

        return mapped_embeddings

    def pe_even(self, index_of_embedding, i, d_model):
        """
        index_of_embedding - an index number of currently processed embedding. in other words, it's embedding's position in embeddings matrix, take value from [0, number_of_embeddings]
        i - position inside an embedding
        d_model - embedding_dimension, in other words amount of positions inside an embedding
        """
        return math.sin(index_of_embedding / math.pow(10000, (float(2 * i / d_model))))

    def pe_odd(self, index_of_embedding, i, d_model):
        return math.cos(index_of_embedding / math.pow(10000, (float(2 * i / d_model))))
