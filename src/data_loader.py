from transformers import GPT2Tokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from gpt import default_context_window, default_batch_size


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, context_window):
        self.tokens = tokenizer.encode(text)
        self.context_window = context_window

    def __len__(self):
        return len(self.tokens) - self.context_window

    def __getitem__(self, index):
        x = self.tokens[index : index + self.context_window]
        y = self.tokens[index + 1 : index + self.context_window + 1]
        return torch.tensor(x), torch.tensor(y)


def get_tokenizer(model="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_data_loader(
    tokenizer, batch_size=default_batch_size, data_path="data/dataset.txt"
):

    with open(data_path, "r") as file:
        text = file.read()

    dataset = TextDataset(
        text=text, tokenizer=tokenizer, context_window=default_context_window
    )

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
