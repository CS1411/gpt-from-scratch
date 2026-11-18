"""Lesson 5: the causal mask. Set the future to -inf before the softmax so a token only looks back."""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson4"))
from lesson4 import embedded_batch  # noqa: E402

if __name__ == "__main__":
    torch.manual_seed(1337)
    x = embedded_batch()
    B, T, C = x.shape

    head_size = 16
    key, query, value = (nn.Linear(C, head_size, bias=False) for _ in range(3))
    k, q, v = key(x), query(x), value(x)
    wei = q @ k.transpose(-2, -1) * head_size ** -0.5

    print("BEFORE masking: a token attends everywhere (note non-zero above the diagonal):")
    print(F.softmax(wei, dim=-1)[0].round(decimals=2))

    tril = torch.tril(torch.ones(T, T))              # 1s on and below the diagonal
    wei = wei.masked_fill(tril == 0, float("-inf"))  # future -> -inf, so softmax gives it weight 0
    wei = F.softmax(wei, dim=-1)

    print("\nAFTER the causal mask: lower-triangular, each token only looks back:")
    print(wei[0].round(decimals=2))
    out = wei @ v
    print("\nrow 0 attends only to token 0:", wei[0, 0].round(decimals=2).tolist())

    assert torch.equal(wei, torch.tril(wei)) and bool(torch.allclose(wei.sum(-1), torch.ones(B, T)))
