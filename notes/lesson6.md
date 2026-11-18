# Lesson 6 - The transformer block

- Four ingredients: multi-head attention (several heads in parallel, concatenated and projected), a feed-forward layer (expand 4x, ReLU, project back), residual connections, and layer norm.
- A block is x = x + attention(norm(x)), then x = x + feedforward(norm(x)). Normalising before each sub-layer is "pre-norm".
- Residual connections let gradients flow straight through a deep stack, so each layer only has to learn a refinement. Layer norm keeps activations well scaled.
- Input shape equals output shape, (4, 8, 32) in and out, which is why blocks stack.
- One block with n_embd 32 and 4 heads has 12,608 parameters.
- Code: `lessons/lesson6/blocks.py` (Head, MultiHeadAttention, FeedForward, Block, written by hand rather than with `nn.MultiheadAttention`) and `lesson6.py`.
