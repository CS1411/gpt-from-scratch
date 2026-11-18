# Lesson 5 - Look back, never forward

- Next-token prediction only works if position t cannot see positions after t, or the model cheats by reading the token it should guess.
- The causal mask: before the softmax, set every future score to -inf. exp(-inf) is 0, so those positions get exactly zero weight and each row renormalises over only the past and present.
- A lower-triangular matrix of ones (`torch.tril`) marks what is allowed; `masked_fill(tril == 0, -inf)` applies it.
- Result: the weight matrix is lower-triangular. Token 0 attends only to itself, token 1 to {0, 1}, and so on, and every row still sums to 1.
- This masked self-attention is the core of every GPT decoder.
- Code: `lessons/lesson5/lesson5.py`.
