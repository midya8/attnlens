"""Two-layer causal multi-head self-attention, trained from scratch, pure Python.

Token embeddings plus fixed sinusoidal positions feed two residual attention
blocks (multi-head, causal) and an output projection with cross-entropy loss.
Two layers are the interesting case: the first can mark the token after a
marker, the second can use that to recall it - the induction-head pattern.
Gradients are derived by hand (no autograd, no dependencies) and every
parameter group's gradient is verified against finite differences in the test
suite. One seeded RNG drives initialization and data generation, so a config
and seed always reproduce the same run.
"""

from __future__ import annotations

import math
import random

Matrix = list[list[float]]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    bt = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def transpose(a: Matrix) -> Matrix:
    return [list(col) for col in zip(*a)]


def softmax(row: list[float]) -> list[float]:
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    total = sum(exps)
    return [e / total for e in exps]


def positional_encoding(length: int, dim: int) -> Matrix:
    return [
        [math.sin(p / 10000.0 ** (2 * (i // 2) / dim)) if i % 2 == 0
         else math.cos(p / 10000.0 ** (2 * (i // 2) / dim))
         for i in range(dim)]
        for p in range(length)
    ]


class AttentionLayer:
    """One causal multi-head self-attention block with its own parameters."""

    def __init__(self, dim: int, heads: int, rng: random.Random) -> None:
        if heads < 1 or dim % heads != 0:
            raise ValueError("dim must be divisible by heads, heads >= 1")
        self.dim, self.heads = dim, heads
        self.head_dim = dim // heads
        scale = 1.0 / math.sqrt(dim)
        init = lambda r, c: [[(rng.random() - 0.5) * 2 * scale for _ in range(c)] for _ in range(r)]
        self.wq = [init(dim, self.head_dim) for _ in range(heads)]
        self.wk = [init(dim, self.head_dim) for _ in range(heads)]
        self.wv = [init(dim, self.head_dim) for _ in range(heads)]
        self.wo = init(dim, dim)

    def forward(self, x: Matrix) -> tuple[Matrix, dict]:
        length = len(x)
        scale = 1.0 / math.sqrt(self.head_dim)
        head_out, qs, ks, vs, attns = [], [], [], [], []
        for h in range(self.heads):
            q = matmul(x, self.wq[h])
            k = matmul(x, self.wk[h])
            v = matmul(x, self.wv[h])
            attn: Matrix = []
            for i in range(length):
                scores = [sum(q[i][d] * k[j][d] for d in range(self.head_dim)) * scale
                          if j <= i else -1e30 for j in range(length)]
                attn.append(softmax(scores))
            head_out.append(matmul(attn, v))
            qs.append(q); ks.append(k); vs.append(v); attns.append(attn)
        concat = [[head_out[h][r][c] for h in range(self.heads) for c in range(self.head_dim)]
                  for r in range(length)]
        out = matmul(concat, self.wo)
        cache = {"x": x, "q": qs, "k": ks, "v": vs, "attn": attns, "concat": concat}
        return out, cache

    def backward(self, cache: dict, dout: Matrix) -> tuple[Matrix, dict[str, list]]:
        length = len(cache["x"])
        scale = 1.0 / math.sqrt(self.head_dim)
        grads = {"wo": matmul(transpose(cache["concat"]), dout)}
        dconcat = matmul(dout, transpose(self.wo))
        dx = [[0.0] * self.dim for _ in range(length)]
        for h in range(self.heads):
            dhead = [[dconcat[r][h * self.head_dim + c] for c in range(self.head_dim)]
                     for r in range(length)]
            q, k, v, attn = cache["q"][h], cache["k"][h], cache["v"][h], cache["attn"][h]
            dattn = matmul(dhead, transpose(v))
            dv = matmul(transpose(attn), dhead)
            dscores: Matrix = []
            for i in range(length):
                row_dot = sum(dattn[i][j] * attn[i][j] for j in range(length))
                dscores.append([attn[i][j] * (dattn[i][j] - row_dot) for j in range(length)])
            dq = [[sum(dscores[i][j] * k[j][d] for j in range(length)) * scale
                   for d in range(self.head_dim)] for i in range(length)]
            dk = [[sum(dscores[i][j] * q[i][d] for i in range(length)) * scale
                   for d in range(self.head_dim)] for j in range(length)]
            grads[f"wq{h}"] = matmul(transpose(cache["x"]), dq)
            grads[f"wk{h}"] = matmul(transpose(cache["x"]), dk)
            grads[f"wv{h}"] = matmul(transpose(cache["x"]), dv)
            for r in range(length):
                for d in range(self.dim):
                    dx[r][d] += (
                        sum(dq[r][c] * self.wq[h][d][c] for c in range(self.head_dim))
                        + sum(dk[r][c] * self.wk[h][d][c] for c in range(self.head_dim))
                        + sum(dv[r][c] * self.wv[h][d][c] for c in range(self.head_dim))
                    )
        return dx, grads


class Params:
    """All trainable parameters: embeddings, attention layers, output head."""

    def __init__(self, vocab: int, dim: int, heads: int, layers: int,
                 rng: random.Random) -> None:
        if layers < 1:
            raise ValueError("layers must be >= 1")
        self.vocab, self.dim, self.heads, self.layers_count = vocab, dim, heads, layers
        scale = 1.0 / math.sqrt(dim)
        self.embed = [[(rng.random() - 0.5) * 2 * scale for _ in range(dim)]
                      for _ in range(vocab)]
        self.layers = [AttentionLayer(dim, heads, rng) for _ in range(layers)]
        self.wout = [[(rng.random() - 0.5) * 2 * scale for _ in range(vocab)]
                     for _ in range(dim)]
        self.bout = [0.0] * vocab

    def groups(self) -> dict[str, list]:
        groups: dict[str, list] = {"embed": self.embed, "wout": self.wout,
                                   "bout": [self.bout]}
        for li, layer in enumerate(self.layers):
            groups[f"l{li}_wo"] = layer.wo
            for h in range(layer.heads):
                groups[f"l{li}_wq{h}"] = layer.wq[h]
                groups[f"l{li}_wk{h}"] = layer.wk[h]
                groups[f"l{li}_wv{h}"] = layer.wv[h]
        return groups


class Cache:
    def __init__(self) -> None:
        self.layer_caches: list[dict] = []
        self.layer_inputs: list[Matrix] = []
        self.final: Matrix = []
        self.logits: Matrix = []


def forward(tokens: list[int], p: Params, cache: Cache | None = None) -> Matrix:
    """Residual stack forward; returns logits (T x V). Layer attention caches
    are kept when a Cache is supplied."""
    pos = positional_encoding(len(tokens), p.dim)
    x = [[p.embed[t][i] + pos[r][i] for i in range(p.dim)] for r, t in enumerate(tokens)]
    layer_caches, layer_inputs = [], []
    for layer in p.layers:
        layer_inputs.append(x)
        out, lcache = layer.forward(x)
        layer_caches.append(lcache)
        x = [[x[r][d] + out[r][d] for d in range(p.dim)] for r in range(len(x))]
    logits = [[sum(x[r][d] * p.wout[d][c] for d in range(p.dim)) + p.bout[c]
               for c in range(p.vocab)] for r in range(len(tokens))]
    if cache is not None:
        cache.layer_caches, cache.layer_inputs = layer_caches, layer_inputs
        cache.final, cache.logits = x, logits
    return logits


def cross_entropy(logits_row: list[float], target: int) -> float:
    # Stable log-sum-exp, without a probability floor that disagrees with gradients.
    maximum = max(logits_row)
    return (maximum - logits_row[target]) + math.log(
        sum(math.exp(x - maximum) for x in logits_row))


def loss_and_grads(tokens: list[int], target: int, p: Params) -> tuple[float, dict[str, list]]:
    """Cross-entropy at the final position, with hand-derived gradients."""
    cache = Cache()
    logits = forward(tokens, p, cache)
    last = len(tokens) - 1
    probs = softmax(logits[last])
    loss = cross_entropy(logits[last], target)

    grads: dict[str, list] = {}
    dlogits = probs
    dlogits[target] -= 1.0

    grads["wout"] = [[cache.final[last][d] * dlogits[c] for c in range(p.vocab)]
                     for d in range(p.dim)]
    grads["bout"] = [list(dlogits)]
    dx = [[0.0] * p.dim for _ in tokens]
    for d in range(p.dim):
        dx[last][d] = sum(dlogits[c] * p.wout[d][c] for c in range(p.vocab))

    for li in reversed(range(p.layers_count)):
        layer = p.layers[li]
        din, lgrads = layer.backward(cache.layer_caches[li], dx)
        for name, g in lgrads.items():
            grads[f"l{li}_{name}"] = g
        dx = [[dx[r][d] + din[r][d] for d in range(p.dim)] for r in range(len(tokens))]

    dembed = [[0.0] * p.dim for _ in range(p.vocab)]
    for r, t in enumerate(tokens):
        for d in range(p.dim):
            dembed[t][d] += dx[r][d]
    grads["embed"] = dembed
    return loss, grads


def sgd_step(p: Params, grads: dict[str, list], lr: float, max_norm: float = 1.0) -> None:
    """SGD with global-norm gradient clipping (the usual stabilizer)."""
    total = math.sqrt(sum(x * x for group in grads.values() for row in group for x in row))
    scale = min(1.0, max_norm / total) if total > 0 else 1.0
    for name, group in p.groups().items():
        g = grads[name]
        for i in range(len(group)):
            for j in range(len(group[i])):
                group[i][j] -= lr * scale * g[i][j]


def predict(logits_row: list[float]) -> int:
    return max(range(len(logits_row)), key=lambda c: logits_row[c])
