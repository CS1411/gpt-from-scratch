"""The GPT: embeddings, a stack of transformer blocks, a final layer norm and a projection to logits."""
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson6"))
from blocks import Block  # noqa: E402


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 32   # context length
    n_embd: int = 64       # embedding size
    n_heads: int = 4       # attention heads
    n_layers: int = 3      # transformer blocks


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg.n_embd, cfg.n_heads, cfg.block_size) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T))
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)  # (B, T, vocab): a score for every possible next character
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def save_gpt(model, path):
    torch.save({"config": asdict(model.cfg), "state": model.state_dict()}, path)


def load_gpt(path):
    ckpt = torch.load(path)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state"])
    return model.eval()
