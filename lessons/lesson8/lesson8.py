"""Lesson 8: make it talk. Same model, different sampling. Needs gpt.pt from Lesson 7."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for lesson in ("lesson2", "lesson7"):
    sys.path.insert(0, str(HERE.parent / lesson))
from data import CharTokenizer, load_text  # noqa: E402
from gpt import load_gpt  # noqa: E402
from sampling import sample  # noqa: E402

if __name__ == "__main__":
    tok = CharTokenizer(load_text())
    model = load_gpt(HERE.parent / "lesson7" / "gpt.pt")

    print("=== temperature 0.5 (conservative) ===\n", sample(model, tok, temperature=0.5))
    print("\n=== temperature 1.4 (wild) ===\n", sample(model, tok, temperature=1.4))
    print("\n=== top-k = 10 ===\n", sample(model, tok, top_k=10))
    print("\n=== top-p = 0.9 (nucleus) ===\n", sample(model, tok, top_p=0.9))

    assert sample(model, tok, 50, seed=3) == sample(model, tok, 50, seed=3)  # seeded sampling is repeatable
