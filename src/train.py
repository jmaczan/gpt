import torch
import torch.nn as nn
import os
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

checkpoints_directory = "checkpoints"
os.makedirs(checkpoints_directory, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, loss, path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, path)

    print(f"Checkpoitn saved at {path}")


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    print(f"Loaded from checkpoint {path} at epoch {epoch} with loss {loss}")

    return epoch, loss


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

        average_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}. Average loss: {average_loss}")

        checkpoint_path = os.path.join(
            checkpoints_directory, f"epoch_{epoch + 1}_loss_{average_loss}.pth"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            loss=average_loss,
            path=os.path.join(checkpoint_path),
        )


if __name__ == "__main__":
    train()
