import unittest
import torch

from src.positional_encoding import PositionalEncoding

random_seed = 42


class TestPositionalEncoding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(random_seed)
        self.embeddings = torch.rand((3, 4, 5))
        self.batch, self.sequence_length, self.embedding_dim = self.embeddings.shape
        self.pe = PositionalEncoding(
            sequence_length=self.sequence_length, embedding_dim=self.embedding_dim
        )

    def test_output_shape(self):
        torch.manual_seed(random_seed)
        self.assertEqual(
            self.pe.positional_encodings.shape,
            (self.sequence_length, self.embedding_dim),
        )

    def test_precomputed_positional_encodings_are_not_modified_during_forward(self):
        original_encodings = self.pe.positional_encodings.clone()
        self.pe.forward(torch.rand(self.embeddings.shape))
        self.assertTrue(torch.equal(original_encodings, self.pe.positional_encodings))

    def test_correct_encodings(self):
        expected_encodings = torch.tensor(
            [
                [0.00, 1.00, 0.00, 1.00],
                [0.84, 0.54, 0.09, 0.99],
                [0.90, -0.41, 0.19, 0.98],
                [0.14, -0.99, 0.29, 0.96],
            ]
        )
        self.pe = PositionalEncoding(sequence_length=4, embedding_dim=4, n=100)
        computed_encodings = self.pe.positional_encodings
        self.assertTrue(
            torch.allclose(computed_encodings, expected_encodings, atol=1e-2)
        )


if __name__ == "__main__":
    unittest.main()
