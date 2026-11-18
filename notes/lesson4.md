# Lesson 4 - Self-attention

- Each token's vector gets three projections: a query (what am I looking for?), a key (what do I offer?) and a value (what do I pass on if attended to?).
- Token i compares its query with every key by dot product. The scores are divided by sqrt(head_size) so large dot products do not saturate the softmax, then softmax makes each row sum to 1.
- The output is the weighted sum of values: attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.
- One head with head_size 16 on the (4, 8, 32) batch gives weights of shape (4, 8, 8) (query token by key token) and an output of (4, 8, 16). Every row sums to 1.
- Problem: every token attends to all eight positions, including ones after it. For next-token prediction that lets a token see the answer, which Lesson 5 fixes.
- Code: `lessons/lesson4/lesson4.py`.
