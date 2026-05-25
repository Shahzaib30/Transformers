"""Generate text from a trained tiny language model checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import TinyDecoderLM, decode_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    parser.add_argument("--checkpoint", default="outputs/tiny_gpt.pth")
    parser.add_argument("--vocab-file", default="outputs/vocab.json")
    parser.add_argument("--prompt", default="One day, a small bird")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    vocab_file = Path(args.vocab_file)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_file}")

    vocab_data = json.loads(vocab_file.read_text(encoding="utf-8"))
    stoi = {str(key): int(value) for key, value in vocab_data["stoi"].items()}
    itos = {int(key): str(value) for key, value in vocab_data["itos"].items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDecoderLM(
        vocab_size=len(stoi),
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    encoded_prompt = [stoi.get(ch, 0) for ch in args.prompt]
    tokens = torch.tensor([encoded_prompt], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            tokens_cond = tokens[:, -args.block_size :]
            logits, _ = model(tokens_cond)
            logits = logits[:, -1, :] / max(args.temperature, 1e-6)
            if args.top_k > 0:
                top_values, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                logits[logits < top_values[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

    print(decode_tokens(tokens[0], itos))


if __name__ == "__main__":
    main()
