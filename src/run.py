import torch
from data_loader import get_tokenizer
from train import load_checkpoint
from gpt import (
    GPT
)

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    config = checkpoint['config']

    model = GPT(
        vocabulary_size=config['vocabulary_size']
    )

def run(model_path='checkpoints/best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = get_tokenizer()
    
    model=load_model(model_path)

    model = load_checkpoint(model_path, device)

if __name__ == "__main__":
    run()