import torch
import torch.nn as nn

from data_loader import get_data_loader, get_tokenizer
from gpt import (
    GPT,
    default_context_window,
    default_attention_heads_count,
    default_embedding_dimension,
    default_transformer_blocks_count,
    default_vocabulary_size,
)


default_num_epochs = 10
default_learning_rate = 0.001


def train(
    num_epochs=default_num_epochs,
    lr=default_learning_rate,
    embedding_dimension=default_embedding_dimension,
    context_window=default_context_window,
    heads_count=default_attention_heads_count,
    blocks_count=default_transformer_blocks_count,
):
    tokenizer = get_tokenizer()

    vocabulary_size = len(tokenizer)

    model = GPT(
        vocabulary_size=vocabulary_size,
        embedding_dimension=embedding_dimension,
        context_window=context_window,
        heads_count=heads_count,
        blocks_count=blocks_count,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_loader = get_data_loader(tokenizer=tokenizer)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocabulary_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{num_epochs}. Total loss: {total_loss}. Loss: {total_loss/len(data_loader)}"
        )


if __name__ == "__main__":
    train()
