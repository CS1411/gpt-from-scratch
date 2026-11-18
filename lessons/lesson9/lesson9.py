"""Lesson 9: ship your mini-GPT behind one clean interface. Needs gpt.pt from Lesson 7."""
import torch

from mini_gpt import MiniGPT

if __name__ == "__main__":
    torch.manual_seed(1337)
    gpt = MiniGPT()
    out = gpt.complete(prompt="ROMEO:", max_new_tokens=200)
    print(out)
    assert out.startswith("ROMEO:") and len(out) == len("ROMEO:") + 200
