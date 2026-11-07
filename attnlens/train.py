"""Seeded SGD training loop with fresh deterministic samples each epoch."""

from __future__ import annotations

import random

from .data import MARKER, make_dataset
from .model import Cache, Params, forward, loss_and_grads, predict, sgd_step


def train(dim: int = 16, heads: int = 4, layers: int = 2, epochs: int = 200,
          lr: float = 0.05, seed: int = 7, per_epoch: int = 64,
          n_eval: int = 128, length: int = 8) -> dict:
    if lr <= 0:
        raise ValueError("lr must be > 0")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    rng = random.Random(seed)
    params = Params(vocab=12, dim=dim, heads=heads, layers=layers, rng=rng)

    losses: list[float] = []
    for ep in range(epochs):
        batch = make_dataset(per_epoch, seed=seed + 1 + ep, length=length)
        epoch_loss = 0.0
        for tokens, target in batch:
            loss, grads = loss_and_grads(tokens, target, params)
            sgd_step(params, grads, lr)
            epoch_loss += loss
        losses.append(epoch_loss / len(batch))

    eval_set = make_dataset(n_eval, seed=seed + 100000, length=length)
    correct = 0
    recall_attn: list[float] = []
    for tokens, target in eval_set:
        cache = Cache()
        logits = forward(tokens, params, cache)
        if predict(logits[-1]) == target:
            correct += 1
        key_pos = tokens.index(MARKER) + 1
        top = cache.layer_caches[-1]["attn"]
        recall_attn.append(sum(top[h][-1][key_pos] for h in range(heads)) / heads)
    return {
        "params": params,
        "losses": losses,
        "final_loss": round(losses[-1], 4),
        "accuracy": round(correct / len(eval_set), 4),
        "mean_recall_attention": round(sum(recall_attn) / len(recall_attn), 4),
        "config": {"dim": dim, "heads": heads, "layers": layers, "epochs": epochs,
                   "lr": lr, "seed": seed, "per_epoch": per_epoch,
                   "n_eval": n_eval, "length": length},
    }
