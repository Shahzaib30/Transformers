"""Reusable transformer language model components.

This module keeps the core model code in one place so training and generation
scripts can stay small and focused.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CausalSelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class TinyDecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(
            *[CausalSelfAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        batch_size, seq_len = idx.size()
        positions = torch.arange(seq_len, device=idx.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            vocab_size = logits.size(-1)
            loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss


def sample_batch(data: torch.Tensor, block_size: int, batch_size: int):
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[start : start + block_size] for start in starts])
    y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return x, y


def build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {ch: idx for idx, ch in enumerate(chars)}
    itos = {idx: ch for ch, idx in stoi.items()}
    return stoi, itos


def encode_text(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def decode_tokens(tokens: list[int] | torch.Tensor, itos: dict[int, str]) -> str:
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    return "".join(itos[token] for token in tokens)
