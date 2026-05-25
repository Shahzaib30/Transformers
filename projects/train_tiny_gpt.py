"""Train a tiny decoder-only language model on local text.

This script is a complete, runnable baseline for the repository. It trains on
plain text, stores the vocab alongside the checkpoint, and supports resuming
from the saved output artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from model import TinyDecoderLM, build_vocab, encode_text, sample_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny decoder-only LM.")
    parser.add_argument("--data-file", default="input.txt", help="Plain text training file")
    parser.add_argument("--output-dir", default="outputs", help="Checkpoint/output directory")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    text = data_path.read_text(encoding="utf-8", errors="ignore")
    if len(text) < args.block_size + 2:
        raise ValueError("Training text is too short for the requested block size.")

    stoi, itos = build_vocab(text)
    data = encode_text(text, stoi)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDecoderLM(
        vocab_size=len(stoi),
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    losses: list[float] = []

    model.train()
    progress = tqdm(range(args.steps), desc="training", unit="step")
    for step in progress:
        xb, yb = sample_batch(data, args.block_size, args.batch_size)
        xb, yb = xb.to(device), yb.to(device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        progress.set_postfix(loss=f"{loss_value:.4f}")

        if (step + 1) % args.save_every == 0:
            checkpoint_path = output_dir / "tiny_gpt.pth"
            torch.save(model.state_dict(), checkpoint_path)

    checkpoint_path = output_dir / "tiny_gpt.pth"
    torch.save(model.state_dict(), checkpoint_path)
    (output_dir / "vocab.json").write_text(json.dumps({"stoi": stoi, "itos": itos}, indent=2), encoding="utf-8")
    (output_dir / "training_metrics.json").write_text(
        json.dumps({"steps": args.steps, "final_loss": losses[-1], "losses": losses}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
