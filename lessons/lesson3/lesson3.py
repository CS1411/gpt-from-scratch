"""Lesson 3: tokens to vectors. Token embeddings plus positional embeddings."""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson2"))
from data import CharTokenizer, get_batch, load_text  # noqa: E402

if __name__ == "__main__":
    torch.manual_seed(1337)
    text = load_text()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    block_size = 8   # how many tokens of context the model sees at once
    batch_size = 4   # sequences per batch
    n_embd = 32      # size of each embedding vector

    xb, yb = get_batch(data, block_size, batch_size)
    print("input batch x:", tuple(xb.shape), "(batch, time)")
    print("target batch y:", tuple(yb.shape), "-> y is x shifted one step (the 'next token' at each position)")

    token_embedding = nn.Embedding(tok.vocab_size, n_embd)  # what the token is
    position_embedding = nn.Embedding(block_size, n_embd)   # where it sits

    tok_emb = token_embedding(xb)                                # (B, T, C)
    pos_emb = position_embedding(torch.arange(block_size))      # (T, C)
    x = tok_emb + pos_emb                                        # broadcast -> (B, T, C)

    print("token embeddings    :", tuple(tok_emb.shape))
    print("position embeddings :", tuple(pos_emb.shape))
    print("model input (tok+pos):", tuple(x.shape), " <- one 32-d vector per token, position baked in")

    assert tuple(x.shape) == (4, 8, 32) and torch.equal(xb[:, 1:], yb[:, :-1])
