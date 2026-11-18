# Lesson 9 - Ship your mini-GPT

- `MiniGPT` wraps everything behind one interface: load the trained weights, then `complete(prompt, ...)`. That is the same shape as calling any LLM.
- The checkpoint stores the model config alongside the weights, so the model can be rebuilt from the file alone.
- Prompting with `ROMEO:` gives speaker-tag-and-verse shaped text, still made of invented words at this scale.
- Portfolio project brief (a separate repository): an input pipeline (tokenizer, embeddings, batching); self-attention written by hand; a trained transformer with the loss shown falling; generation with at least one control; and your own touch, such as your own corpus, a write-up and incremental commits. Stretch: a bigger model, byte-pair tokens, a longer run.
- Summary: text to tokens to vectors, hand-written attention and causal mask, blocks stacked into a GPT, trained until the loss fell, decoded with the knobs a chat API exposes. The architecture is the same as GPT, Gemini, Claude and Llama, differing only in scale.
- Code: `lessons/lesson9/mini_gpt.py` and `lesson9.py`.
