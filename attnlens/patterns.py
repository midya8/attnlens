"""Attention-pattern analysis: what did each head actually learn to do?"""

from __future__ import annotations

import math

from .data import MARKER
from .model import Cache, Params, forward


def head_stats(attn: list[list[list[float]]], tokens: list[int]) -> list[dict]:
    """Per-head statistics for one sequence: mean row entropy, diagonal mass
    (attention to the immediately previous token), and - when the sequence
    carries a marker - the final position's attention on the recalled key."""
    marker_positions = [i for i, t in enumerate(tokens) if t == MARKER]
    key_pos = marker_positions[0] + 1 if marker_positions else None
    stats = []
    for h, a in enumerate(attn):
        length = len(a)
        entropy = 0.0
        for row in a:
            entropy += -sum(x * math.log(x) for x in row if x > 0)
        entropy /= length
        diagonal = sum(a[i][i - 1] for i in range(1, length)) / (length - 1)
        stat = {
            "head": h,
            "mean_entropy": round(entropy, 4),
            "prev_token_mass": round(diagonal, 4),
        }
        if key_pos is not None:
            stat["recall_attention"] = round(a[-1][key_pos], 4)
        stats.append(stat)
    return stats


def analyze(tokens: list[int], p: Params, layers: int) -> dict:
    cache = Cache()
    forward(tokens, p, cache)
    return {
        "tokens": tokens,
        "layers": [
            head_stats(cache.layer_caches[li]["attn"], tokens)
            for li in range(layers)
        ],
    }


def heatmap(attn_row: list[float], width: int = 8) -> str:
    """One attention row as a text bar chart."""
    bars = []
    for w in attn_row:
        filled = int(round(w * width))
        bars.append("█" * filled + "·" * (width - filled))
    return " ".join(f"{w:.2f} {b}" for w, b in zip(attn_row, bars))
