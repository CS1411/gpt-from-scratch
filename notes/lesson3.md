# Lesson 3 - Tokens to vectors

- A token id carries no meaning (42 is not "bigger" than 41), so the first layer is an embedding table: `nn.Embedding(vocab_size, n_embd)` maps each id to a learned vector.
- Attention has no sense of order, so a second table gives every position 0, 1, 2, ... its own vector. The model input is the sum: what the token is plus where it sits.
- Batches are random windows of the data. The target is the input shifted by one token, so every position has a "next token" to predict.
- Shapes: input batch (4, 8) as (batch, time); token embeddings (4, 8, 32); position embeddings (8, 32); model input (4, 8, 32).
- GPT uses learned positions. The original transformer used fixed sinusoidal encodings, which need no parameters and extrapolate to longer sequences.
- Code: `lessons/lesson3/lesson3.py`.
