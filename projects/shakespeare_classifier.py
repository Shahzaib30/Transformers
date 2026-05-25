"""Character-level Shakespeare speaker classifier.

This is a cleaned-up, runnable version of the earlier experiment. It reads a
simple text file with lines labeled `ROMEO:` or `JULIET:` and trains a tiny
attention-based classifier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ShakespeareClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, num_classes: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = nn.MultiheadAttention(embed_size, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding(x)
        attn_out, _ = self.attention(embedding, embedding, embedding)
        pooled = self.norm(attn_out).mean(dim=1)
        return self.head(pooled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Shakespeare speaker classifier.")
    parser.add_argument("--input-file", default="input.txt")
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--embed-size", type=int, default=128)
    return parser.parse_args()


def load_dataset(file_path: Path):
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    blocks = text.split("\n\n")
    samples = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        label = lines[0]
        content = " ".join(lines[1:]).lower()
        if label == "ROMEO:":
            samples.append((content, 0))
        elif label == "JULIET:":
            samples.append((content, 1))
    return samples


def encode(text: str, char_to_idx: dict[str, int], seq_length: int) -> list[int]:
    encoded = [char_to_idx.get(char, 0) for char in text]
    if len(encoded) < seq_length:
        encoded.extend([0] * (seq_length - len(encoded)))
    return encoded[:seq_length]


def make_batch(samples, char_to_idx, seq_length: int, batch_size: int):
    indices = np.random.randint(0, len(samples), size=batch_size)
    inputs = []
    labels = []
    for index in indices:
        sample, label = samples[index]
        inputs.append(encode(sample, char_to_idx, seq_length))
        labels.append(label)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def main() -> None:
    args = parse_args()
    file_path = Path(args.input_file)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    samples = load_dataset(file_path)
    if not samples:
        raise ValueError("No labeled ROMEO/JULIET samples were found in the input file.")

    all_text = " ".join(sample[0] for sample in samples)
    chars = sorted(set(all_text))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShakespeareClassifier(len(chars), args.embed_size, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step in range(args.steps):
        xb, yb = make_batch(samples, char_to_idx, args.seq_length, args.batch_size)
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{args.steps} | loss={loss.item():.4f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
