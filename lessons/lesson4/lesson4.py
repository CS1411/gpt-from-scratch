"""Lesson 4: scaled dot-product self-attention, one head, from scratch (no causal mask yet)."""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson2"))
from data import CharTokenizer, get_batch, load_text  # noqa: E402


def embedded_batch(block_size=8, batch_size=4, n_embd=32):
    "The embedded batch from Lessons 2-3: (B, T, C)."
    text = load_text()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    xb, _ = get_batch(data, block_size, batch_size)
    return nn.Embedding(tok.vocab_size, n_embd)(xb) + nn.Embedding(block_size, n_embd)(torch.arange(block_size))


if __name__ == "__main__":
    torch.manual_seed(1337)
    x = embedded_batch()
    B, T, C = x.shape

    head_size = 16
    key = nn.Linear(C, head_size, bias=False)    # what do I offer?
    query = nn.Linear(C, head_size, bias=False)  # what am I looking for?
    value = nn.Linear(C, head_size, bias=False)  # what do I pass on if attended to?

    k, q, v = key(x), query(x), value(x)                 # each (B, T, head_size)
    wei = q @ k.transpose(-2, -1) * head_size ** -0.5    # (B, T, T) scaled scores
    wei = F.softmax(wei, dim=-1)                         # each row sums to 1
    out = wei @ v                                        # (B, T, head_size)

    rows_sum_to_one = bool(torch.allclose(wei.sum(-1), torch.ones(B, T)))
    print("q, k, v :", tuple(q.shape))
    print("attention weights:", tuple(wei.shape), "(query token x key token)")
    print("every row sums to 1:", rows_sum_to_one)
    print("head output :", tuple(out.shape))
    print("\nattention matrix for one sequence (row i = how token i spreads its attention):")
    print(wei[0].round(decimals=2))
    assert rows_sum_to_one and tuple(out.shape) == (4, 8, 16)
