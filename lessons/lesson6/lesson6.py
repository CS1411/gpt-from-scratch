"""Lesson 6: the transformer block. Input shape equals output shape, so blocks stack."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson4"))
from lesson4 import embedded_batch  # noqa: E402
from blocks import Block  # noqa: E402

if __name__ == "__main__":
    torch.manual_seed(1337)
    x = embedded_batch()
    block = Block(n_embd=32, n_heads=4, block_size=8)
    out = block(x)
    n_params = sum(p.numel() for p in block.parameters())
    print("block input :", tuple(x.shape))
    print("block output:", tuple(out.shape), " <- same shape in and out, so blocks stack")
    print("parameters in one block:", n_params)
    assert out.shape == x.shape and n_params == 12608
