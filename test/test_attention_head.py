import unittest
import torch

from src.attention_head import AttentionHead

random_seed = 42


class TestAttentionHead(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(random_seed)

        self.embeddings = torch.rand((3, 4, 5))
        self.batch, self.sequence_length, self.embedding_dim = self.embeddings.shape
        self.head = AttentionHead(embeddings=self.embeddings)

    def test_matrices_shapes(self):
        torch.manual_seed(random_seed)

        self.assertEqual(
            self.head.W_Q.shape,
            (self.embedding_dim, self.embedding_dim),
        )
        self.assertEqual(
            self.head.W_K.shape,
            (self.embedding_dim, self.embedding_dim),
        )
        self.assertEqual(
            self.head.W_V.shape,
            (self.embedding_dim, self.embedding_dim),
        )

    def test_forward(self):
        self.head.forward(embeddings=self.embeddings)


if __name__ == "__main__":
    unittest.main()
