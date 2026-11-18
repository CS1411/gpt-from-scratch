"""The transformer block: masked multi-head attention and a feed-forward layer, each with a residual
connection and a layer norm in front (pre-norm). Written by hand, not with nn.MultiheadAttention."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    "One causal self-attention head."

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ v


class MultiHeadAttention(nn.Module):
    "Several heads in parallel, concatenated and projected back to n_embd."

    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()
        head_size = n_embd // n_heads
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class FeedForward(nn.Module):
    "A per-token MLP: expand 4x, ReLU, project back."

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    "x = x + attention(norm(x)); x = x + feedforward(norm(x))."

    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_heads, block_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
