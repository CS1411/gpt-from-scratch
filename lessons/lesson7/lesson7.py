"""Lesson 7: assemble the GPT and train it with next-token cross-entropy. A few minutes on a CPU."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson2"))
from data import CharTokenizer, get_batch, load_text, split_data  # noqa: E402
from gpt import GPT, GPTConfig, save_gpt  # noqa: E402

WEIGHTS = Path(__file__).resolve().parent / "gpt.pt"  # git-ignored

if __name__ == "__main__":
    torch.manual_seed(1337)
    text = load_text()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    train_data, val_data = split_data(data)

    cfg = GPTConfig(vocab_size=tok.vocab_size)
    batch_size = 16
    model = GPT(cfg)
    print("total parameters:", sum(p.numel() for p in model.parameters()))

    # one forward pass before any training: the loss should sit near log(vocab_size)
    xb = torch.stack([train_data[i:i + cfg.block_size]
                      for i in torch.randint(len(train_data) - cfg.block_size, (batch_size,))])
    yb = torch.stack([train_data[i + 1:i + cfg.block_size + 1]
                      for i in torch.randint(len(train_data) - cfg.block_size, (batch_size,))])
    _, loss = model(xb, yb)
    print(f"untrained loss: {loss.item():.3f} (random guessing = log({tok.vocab_size}) = "
          f"{math.log(tok.vocab_size):.3f})\n")

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split, src in (("train", train_data), ("val", val_data)):
            losses = torch.zeros(10)
            for k in range(10):
                _, l = model(*get_batch(src, cfg.block_size, batch_size))
                losses[k] = l.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # the loop every neural net uses: batch, loss, backward, step
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    max_iters, eval_interval = 400, 100
    first = None
    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            l = estimate_loss()
            first = first or l["val"]
            print(f"step {it:4d}: train loss {l['train']:.3f}  val loss {l['val']:.3f}")
        _, loss = model(*get_batch(train_data, cfg.block_size, batch_size))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    save_gpt(model, WEIGHTS)
    print(f"\nsaved {WEIGHTS.name}")
    context = torch.zeros((1, 1), dtype=torch.long)  # a single start token
    print(tok.decode(model.generate(context, max_new_tokens=400)[0].tolist()))
    assert l["val"] < first - 1.0  # the held-out loss fell well below where it started
