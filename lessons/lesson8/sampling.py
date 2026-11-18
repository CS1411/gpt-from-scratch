"""Decoding strategies: temperature, top-k and top-p (nucleus) sampling on a trained GPT."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def sample(model, tok, max_new_tokens=300, temperature=1.0, top_k=None, top_p=None, seed=0, prompt=None):
    """Generate text one token at a time.

    temperature: divide the logits before the softmax (below 1 sharpens, above 1 flattens).
    top_k: keep only the k most likely tokens.
    top_p: keep the smallest set of tokens whose probabilities sum to p.
    """
    torch.manual_seed(seed)
    block_size = model.cfg.block_size
    idx = (torch.tensor([tok.encode(prompt)], dtype=torch.long) if prompt
           else torch.zeros((1, 1), dtype=torch.long))
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block_size:])
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            kth = torch.topk(logits, top_k)[0][:, [-1]]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p is not None:
            s_logits, s_idx = torch.sort(logits, descending=True)
            cum = F.softmax(s_logits, dim=-1).cumsum(-1)
            drop = cum > top_p
            drop[:, 1:] = drop[:, :-1].clone()  # keep the token that crosses the threshold
            drop[:, 0] = False                  # always keep the most likely token
            s_logits = s_logits.masked_fill(drop, float("-inf"))
            logits = torch.empty_like(logits).scatter_(-1, s_idx, s_logits)
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return tok.decode(idx[0].tolist())
