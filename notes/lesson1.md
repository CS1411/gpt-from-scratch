# Lesson 1 - What a language model is

- A GPT is a next-token machine: at every position it outputs a score (logit) for every token in the vocabulary, and softmax turns those into probabilities.
- Training has one goal: make the probability of the true next token high. The loss is cross-entropy, L = -log P(true next token).
- Three reference points for a 5-token vocabulary: a clueless model (uniform) has loss log(5) = 1.609; leaning correct (P = 0.834) has 0.182; confidently wrong (P = 0.0066) has 5.027.
- So loss sits at log(V) with no knowledge, falls towards 0 as the model gets the right token confident, and explodes when it is confidently wrong.
- Code: `lessons/lesson1/lesson1.py`.
