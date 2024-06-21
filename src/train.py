import torch
import torch.nn as nn

from src.gpt import GPT


default_num_epochs = 10
default_learning_rate = 0.001


def train(num_epochs=default_num_epochs, lr=default_learning_rate):
    model = GPT()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_loader = None

    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocabulary_size), targets.view(-1))
            loss.backward()
            optimizer.step()

    return None


if __name__ == "__main__":
    train()
