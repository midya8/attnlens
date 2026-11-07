"""Deterministic marker-recall task: attend back to the token after a marker.

Each sequence embeds one KEY directly after a MARKER among random filler
tokens, and ends with QUERY. The target is the key token. Answering requires
attending from the final position back to the marker's neighbor - position
alone cannot solve it, because the marker's position varies.
"""

from __future__ import annotations

import random

KEYS = list(range(10))   # tokens 0..9 are recallable keys and filler
MARKER = 10
QUERY = 11
VOCAB = 12


def make_sample(rng: random.Random, length: int = 8) -> tuple[list[int], int]:
    if length < 4:
        raise ValueError("length must be >= 4")
    key = rng.choice(KEYS)
    marker_pos = rng.randint(0, length - 4)  # key at marker_pos+1, QUERY at end
    tokens = [rng.choice(KEYS) for _ in range(length)]
    tokens[marker_pos] = MARKER
    tokens[marker_pos + 1] = key
    tokens[-1] = QUERY
    return tokens, key


def make_dataset(n: int, seed: int, length: int = 8) -> list[tuple[list[int], int]]:
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = random.Random(seed)
    return [make_sample(rng, length) for _ in range(n)]
