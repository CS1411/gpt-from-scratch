"""Lesson 1: a language model predicts the next token; training minimises cross-entropy."""
import torch
import torch.nn.functional as F


def describe(name, logits, target):
    probs = F.softmax(logits, dim=1)
    true_prob = probs[0, target.item()]
    loss = F.cross_entropy(logits, target)
    print(f"\n{name}")
    print("-" * len(name))
    print("Logits :", logits.tolist()[0])
    print("Probabilities:", [round(p, 4) for p in probs[0].tolist()])
    print("P(correct) :", round(true_prob.item(), 4))
    print("CrossEntropy = -log(P(correct))")
    print(f"             = -log({true_prob:.4f})")
    print(f"             = {loss.item():.4f}")
    return loss.item()


if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size = 5
    target = torch.tensor([2])  # the correct token is index 2
    examples = {
        "No clue": torch.zeros(1, vocab_size),
        "Leans correct": torch.tensor([[0., 0., 3., 0., 0.]]),
        "Confidently wrong": torch.tensor([[5., 0., 0., 0., 0.]]),
    }
    losses = {name: describe(name, logits, target) for name, logits in examples.items()}
    # A clueless model sits at log(V); being right pulls the loss to 0; being confidently wrong is punished most.
    assert abs(losses["No clue"] - torch.log(torch.tensor(5.0)).item()) < 1e-6
    assert losses["Leans correct"] < losses["No clue"] < losses["Confidently wrong"]
