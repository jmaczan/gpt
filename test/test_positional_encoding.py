import unittest
import torch

from src.positional_encoding import PositionalEncoding


class TestPositionalEncoding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        tensor = torch.rand((3, 4, 5))
        self.pe = PositionalEncoding(embeddings=tensor)

    def tearDown(self) -> None:
        self.pe = None

    def test_pe(self):
        torch.manual_seed(42)
        tensor = torch.rand((3, 4, 5))
        print(tensor)
        self.pe.forward()
