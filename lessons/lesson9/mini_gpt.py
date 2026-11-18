"""MiniGPT: load a trained GPT and complete a prompt. The same shape as calling any LLM."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
for lesson in ("lesson2", "lesson7"):
    sys.path.insert(0, str(HERE.parent / lesson))
from data import CharTokenizer, load_text  # noqa: E402
from gpt import load_gpt  # noqa: E402


class MiniGPT:
    "Load a trained GPT and generate text from a prompt."

    def __init__(self, weights=HERE.parent / "lesson7" / "gpt.pt"):
        self.tok = CharTokenizer(load_text())
        self.model = load_gpt(weights)

    @torch.no_grad()
    def complete(self, prompt="\n", max_new_tokens=300, temperature=0.8, top_k=20):
        block_size = self.model.cfg.block_size
        idx = torch.tensor([self.tok.encode(prompt)], dtype=torch.long)
        for _ in range(max_new_tokens):
            logits, _ = self.model(idx[:, -block_size:])
            logits = logits[:, -1, :] / temperature
            kth = torch.topk(logits, top_k)[0][:, [-1]]
            logits = logits.masked_fill(logits < kth, float("-inf"))
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return self.tok.decode(idx[0].tolist())
