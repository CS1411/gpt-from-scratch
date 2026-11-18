"""Corpus, character tokenizer and batching, shared by every lesson."""
import urllib.request
from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parents[2] / "data"  # git-ignored
CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text(path=DATA_DIR / "shakespeare.txt"):
    "The tiny-shakespeare corpus (about 1 MB). Downloaded once, then cached in data/."
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CORPUS_URL, path)
    return path.read_text()


class CharTokenizer:
    "One integer id per character: the simplest tokenizer."

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def split_data(data, train_frac=0.9):
    "Hold out the last 10% for validation."
    n = int(train_frac * len(data))
    return data[:n], data[n:]


def get_batch(src, block, bs):
    "Random (input, target) windows; the target is the input shifted by one token."
    ix = torch.randint(len(src) - block, (bs,))
    x = torch.stack([src[i:i + block] for i in ix])
    y = torch.stack([src[i + 1:i + block + 1] for i in ix])
    return x, y
