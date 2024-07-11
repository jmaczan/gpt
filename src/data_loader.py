from transformers import GPT2Tokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from gpt import default_context_window, default_batch_size


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, context_window):
        self.tokenizer = tokenizer
        self.tokens = tokenizer.encode(text)
        self.context_window = context_window

    def __len__(self):
        return max(0, len(self.tokens) - self.context_window)

    def __getitem__(self, index):
        x = self.tokens[index : index + self.context_window]
        y = self.tokens[index + 1 : index + self.context_window + 1]
        
        if len(x) < self.context_window:
            x = x + [self.tokenizer.pad_token_id] * (self.context_window - len(x))
            y = y + [self.tokenizer.pad_token_id] * (self.context_window - len(y))
        
        return torch.tensor(x), torch.tensor(y)

def verify_tokenizer(tokenizer, text="Rick:\nWhat's up?"):
    tokens = tokenizer.encode(text)
    decoded_text = tokenizer.decode(tokens)
    assert text == decoded_text, "Tokenization disrepancy detected"


def get_tokenizer(model="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token

    verify_tokenizer(tokenizer)

    return tokenizer


def get_data_loader(
    tokenizer,
    batch_size=default_batch_size,
    data_path="data/dataset.txt",
    shuffle=True,
):

    with open(data_path, "r") as file:
        text = file.read()

    text = " ".join(text.split())

  

    dataset = TextDataset(
        text=text, tokenizer=tokenizer, context_window=default_context_window
    )
    print(f"Total tokens in dataset: {len(dataset.tokens)}")
    print(f"Number of samples: {len(dataset)}")

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
