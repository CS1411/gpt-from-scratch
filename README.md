# GPT from Scratch

A decoder-only transformer written in PyTorch from the tokenizer up to text generation, built while working through the askThili drill *Build GPT from Scratch*. Self-attention and the causal mask are written by hand (no `nn.MultiheadAttention`), and the model trains on a CPU in a few minutes.

Papers: [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) and [Language Models are Unsupervised Multitask Learners (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

## Progress

- [x] Lesson 1 - What a language model is
- [x] Lesson 2 - Text to tokens
- [x] Lesson 3 - Tokens to vectors
- [x] Lesson 4 - Self-attention
- [x] Lesson 5 - Look back, never forward
- [x] Lesson 6 - The transformer block
- [x] Lesson 7 - Assemble and train the GPT
- [x] Lesson 8 - Make it talk (decoding strategies)
- [x] Lesson 9 - Ship your mini-GPT
- [ ] Completion test
- [ ] Project

## What it does

```text
raw text -> characters -> token + position embeddings -> 3 transformer blocks -> next-character logits -> sample
```

```text
$ python lesson9.py
ROMEO:
Ence:
VIf hord and, the my thent in lepon,
My nove ald mongor te dove sheaid, ...
```

A model this small learns the form of the text (spaces, line breaks, speaker tags with colons, word-shaped chunks) rather than its sense. Closing that gap is what scale does.

## Results

Each lesson's run output is saved next to it as `output.txt`.

| Lesson | Result |
|---|---|
| 1 | Cross-entropy: 1.609 for a clueless model (log 5), 0.182 when leaning correct, 5.027 when confidently wrong |
| 2 | Tiny-shakespeare: 1,115,394 characters, vocabulary of 65, split into 1,003,854 train and 111,540 validation tokens |
| 3 | Token plus position embeddings turn a (4, 8) batch into (4, 8, 32) vectors |
| 4 | One scaled dot-product attention head, weights (4, 8, 8), every row summing to 1 |
| 5 | The causal mask makes the weights lower-triangular, so a token only looks back |
| 6 | One transformer block keeps the shape (4, 8, 32) and has 12,608 parameters |
| 7 | **159,937-parameter GPT.** Loss 4.450 untrained (random guessing is 4.174); held-out loss falls from 4.41 to about 2.18 in 400 steps, with train and validation tracking each other |
| 8 | Temperature, top-k and top-p sampling change the output from repetitive to wild to fluent |
| 9 | `MiniGPT.complete(prompt)` behind one interface |

Exact figures can differ in the last digit between library versions; the numbers above are from the run saved in `output.txt`.

## Run it

Requires Python 3.12 and git. Lesson 2 downloads the tiny-shakespeare corpus (about 1.1 MB) from the `char-rnn` repository into `data/`, which is git-ignored.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cd lessons/lesson2 && python lesson2.py    # downloads and tokenizes the corpus
cd ../lesson3 && python lesson3.py
cd ../lesson4 && python lesson4.py
cd ../lesson5 && python lesson5.py
cd ../lesson6 && python lesson6.py
cd ../lesson7 && python lesson7.py         # trains the GPT, writes gpt.pt (about 5 minutes on CPU)
cd ../lesson8 && python lesson8.py         # sampling strategies
cd ../lesson9 && python lesson9.py         # MiniGPT.complete
```

Lessons 8 and 9 read `lessons/lesson7/gpt.pt`, so run Lesson 7 first. Weight files are git-ignored.

## How the code fits together

```text
lessons/
  lesson1  cross-entropy on toy logits
  lesson2  data.py: corpus loading, CharTokenizer, train/val split, batching
  lesson3  token and positional embeddings
  lesson4  one attention head from scratch
  lesson5  the causal mask
  lesson6  blocks.py: Head, MultiHeadAttention, FeedForward, Block
  lesson7  gpt.py: the GPT model; lesson7.py: the training loop
  lesson8  sampling.py: temperature, top-k and top-p
  lesson9  mini_gpt.py: MiniGPT, load and complete
notes/     short notes for every lesson
```
