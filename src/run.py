import torch
from torch.nn import functional as F
from data_loader import get_tokenizer
from train import load_checkpoint
from gpt import GPT

default_max_output = 100
default_temperature = 1.0
default_top_k = 10


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    config = checkpoint["config"]

    model = GPT(
        vocabulary_size=config["vocabulary_size"],
        embedding_dimension=config["embedding_dimension"],
        context_window=config["context_window"],
        heads_count=config["heads_count"],
        blocks_count=config["blocks_count"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

    model.eval()

    return model, config


def prepare_context(text, tokenizer, context_window):
    tokens = tokenizer.encode(text)

    if len(tokens) > context_window:
        tokens = tokens[-context_window:]
    else:
        tokens = [tokenizer.pad_token_id] * (context_window - len(tokens)) + tokens

    return torch.tensor(tokens).unsqueeze(0)


def inference(
    prompt,
    model,
    tokenizer,
    context_window,
    max_output=default_max_output,
    temperature=default_temperature,
    top_k=default_top_k,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    output = tokenizer.encode(prompt)
    context = prepare_context(
        text=prompt, tokenizer=tokenizer, context_window=context_window
    ).to(device)

    with torch.no_grad():
        for _ in range(max_output):
            outputs = model(context)
            next_token = outputs[0, -1, :] / temperature

            top_k_logits, top_k_indices = torch.topk(next_token, top_k)
            top_k_probs = F.softmax(top_k_logits, dim=-1)
            next_token = torch.multinomial(top_k_probs, 1).item()

            output.append(top_k_indices[next_token].item())

            print(tokenizer.decode(output, skip_special_tokens=True))
            print("--------------------")
            context = torch.cat(
                [context, torch.tensor([[next_token]], device=device)], dim=1
            )
            context = context[:, -context_window:]

    return tokenizer.decode(output, skip_special_tokens=True)


def run(model_path="checkpoints/best_model.pth", prompt=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = get_tokenizer()

    model, config = load_model(model_path, device)

    generated_text = inference(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        context_window=config["context_window"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-checkpoint", type=str)

    args = parser.parse_args()
    run(model_path=args.from_checkpoint)
