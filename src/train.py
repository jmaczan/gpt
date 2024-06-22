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
    default_batch_size
)


default_num_epochs = 50
default_learning_rate = 0.0001

checkpoints_directory = "checkpoints"
os.makedirs(checkpoints_directory, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, loss, path, config):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config
    }

    torch.save(checkpoint, path)

    print(f"Checkpoint saved at {path}")


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    config = checkpoint["config"]

    print(f"Loaded from checkpoint {path} at epoch {epoch} with loss {loss}")

    return epoch, loss, config


def train(
    num_epochs=default_num_epochs,
    lr=default_learning_rate,
    embedding_dimension=default_embedding_dimension,
    context_window=default_context_window,
    heads_count=default_attention_heads_count,
    blocks_count=default_transformer_blocks_count,
    batch_size=default_batch_size,
    checkpoint_path=None
):
    tokenizer = get_tokenizer()

    vocabulary_size = len(tokenizer)

    config = {
        "vocabulary_size": vocabulary_size,
        "embedding_dimension": embedding_dimension,
        "context_window": context_window,
        "heads_count": heads_count,
        "blocks_count": blocks_count,
        "batch_size": batch_size
    }

    model = GPT(
        vocabulary_size=vocabulary_size,
        embedding_dimension=embedding_dimension,
        context_window=context_window,
        heads_count=heads_count,
        blocks_count=blocks_count,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_loader = get_data_loader(tokenizer=tokenizer)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, _, loaded_config = load_checkpoint(path=checkpoint_path, model=model, optimizer=optimizer)
        config.update(loaded_config)

    for epoch in range(start_epoch, num_epochs):
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
            path=checkpoint_path,
            config=config
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-checkpoint", type=str)

    parser.add_argument("--num-epochs", type=int, default=default_num_epochs)

    parser.add_argument("--lr", type=int, default=default_learning_rate)

    parser.add_argument("--batch_size", type=int, default=default_batch_size)

    args = parser.parse_args()
    train(checkpoint_path=args.from_checkpoint, num_epochs=args.num_epochs, lr=args.lr, batch_size=args.batch_size)

