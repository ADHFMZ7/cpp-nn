import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn
import torch.nn.functional as F


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, model_width: int, feed_forward_width: int, max_sequence_length: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, model_width)
        self.position_embedding = nn.Parameter(torch.zeros(max_sequence_length, model_width))
        self.attention_norm = nn.LayerNorm(model_width)
        self.query = nn.Linear(model_width, model_width)
        self.key = nn.Linear(model_width, model_width)
        self.value = nn.Linear(model_width, model_width)
        self.attention_output = nn.Linear(model_width, model_width)
        self.feed_forward_norm = nn.LayerNorm(model_width)
        self.feed_forward_in = nn.Linear(model_width, feed_forward_width)
        self.feed_forward_out = nn.Linear(feed_forward_width, model_width)
        self.language_model_head = nn.Linear(model_width, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[0]
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding[positions]

        normalized = self.attention_norm(x)
        query = self.query(normalized)
        key = self.key(normalized)
        value = self.value(normalized)

        scores = query @ key.T / (query.shape[-1] ** 0.5)
        future_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        attention = F.softmax(scores.masked_fill(future_mask, float("-inf")), dim=-1)
        x = x + self.attention_output(attention @ value)

        normalized = self.feed_forward_norm(x)
        hidden = F.gelu(self.feed_forward_in(normalized))
        x = x + self.feed_forward_out(hidden)
        return self.language_model_head(x)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train and export a tiny causal transformer.")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "checkpoints" / "tiny_transformer.safetensors",
        help="Checkpoint output path (default: checkpoints/tiny_transformer.safetensors)",
    )
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be positive")

    torch.manual_seed(7)
    torch.set_num_threads(1)

    config = {
        "architecture": "tiny_causal_transformer",
        "vocab_size": 8,
        "model_width": 8,
        "feed_forward_width": 16,
        "max_sequence_length": 8,
        "num_layers": 1,
        "num_attention_heads": 1,
        "activation": "gelu",
        "normalization": "layer_norm",
        "dtype": "float64",
        "training_task": "next_token_increment_mod_vocab",
    }

    model = TinyTransformer(
        vocab_size=config["vocab_size"],
        model_width=config["model_width"],
        feed_forward_width=config["feed_forward_width"],
        max_sequence_length=config["max_sequence_length"],
    )
    tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    targets = torch.tensor([2, 3, 4, 5, 6, 7, 0], dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)

    model.train()
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_loss = F.cross_entropy(model(tokens), targets).item()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(
        tensors,
        str(args.output),
        metadata={"config": json.dumps(config, sort_keys=True)},
    )

    for name, tensor in tensors.items():
        print(f"{name}\n {tensor}\n\n")

    print(f"Final training loss: {final_loss:.4f}")
    print(f"Saved {len(tensors)} tensors to {args.output}")
    print("Linear weights retain PyTorch's [out_features, in_features] layout.")


if __name__ == "__main__":
    main()