# Lesson 8 - Make it talk (decoding strategies)

- The same trained model sounds different depending on how the next token is picked from its distribution.
- Temperature divides the logits before the softmax. Below 1 it sharpens (safer, more repetitive); above 1 it flattens (more surprising, more errors).
- Top-k keeps only the k most likely tokens and renormalises, blocking the long tail of nonsense.
- Top-p (nucleus) keeps the smallest set of tokens whose probabilities sum to p, so the number kept adapts to how confident the model is. The token that crosses the threshold is kept, and the most likely token is always kept.
- Observed: temperature 0.5 loops on common words; temperature 1.4 invents unlikely spellings and breaks down; top-k 10 and top-p 0.9 stay fluent while still varying.
- Real chat APIs expose exactly these knobs (temperature and top_p).
- Code: `lessons/lesson8/sampling.py` and `lesson8.py` (needs `gpt.pt` from Lesson 7).
