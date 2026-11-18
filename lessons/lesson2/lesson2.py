"""Lesson 2: text to tokens. A character tokenizer, and the corpus split into tensors."""
import torch

from data import CharTokenizer, load_text, split_data

if __name__ == "__main__":
    text = load_text()
    print("corpus length:", len(text), "characters\n")
    print(text[:180])

    tok = CharTokenizer(text)
    print("\nvocabulary:", tok.vocab_size, "unique characters")
    print("".join(tok.chars))

    print("\nencode('Hello'):", tok.encode("Hello"))
    print("decode back    :", tok.decode(tok.encode("Hello")))

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    train_data, val_data = split_data(data)
    print("\ndata tensor:", tuple(data.shape), "| train:", len(train_data), "| val:", len(val_data))

    assert tok.decode(tok.encode("Hello")) == "Hello"
    assert (len(text), tok.vocab_size, len(train_data), len(val_data)) == (1115394, 65, 1003854, 111540)
