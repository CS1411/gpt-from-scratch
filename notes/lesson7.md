# Lesson 7 - Assemble and train the GPT

- `GPT.forward` is every lesson in order: token plus position embeddings, a stack of blocks, a final layer norm, and `lm_head` projecting each vector to one logit per character. Given targets it returns the cross-entropy loss from Lesson 1.
- Config: context 32, embedding 64, 4 heads, 3 blocks, batch size 16. That is 159,937 parameters.
- Before training, the loss is 4.450, close to random guessing at log(65) = 4.174.
- The loop is the usual four steps: batch, loss, backward, optimiser step (AdamW, learning rate 3e-3), 400 steps on a CPU.
- Held-out loss falls from 4.41 to about 2.18, and train and validation loss track each other, so it is learning rather than memorising.
- Samples are not Shakespeare, but the model has learned the form of the text: spaces, line breaks, capitalised line starts, speaker tags with colons and word-shaped chunks. Closing the gap between form and sense is what scale does.
- Code: `lessons/lesson7/gpt.py` and `lesson7.py`. Weights are saved to `gpt.pt`, which is git-ignored.
