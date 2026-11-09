"""Fixed-weight head-output interventions, not attention-as-explanation.

Ablations zero selected head result vectors BEFORE the shared output projection,
at every position, and recompute all downstream activations. Baseline is the
unchanged original forward pass. No attention patterns are frozen.
"""
from __future__ import annotations

import json
import math
import random
from .data import MARKER, QUERY
from .model import (Cache, Params, cross_entropy, forward, matmul,
                    positional_encoding, predict, softmax)


def integer(value, lower, upper, name):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f'{name} must be an integer in [{lower}, {upper}]')
    return value


def dump_model(p: Params) -> dict:
    return {'schema': 1, 'vocab': p.vocab, 'dim': p.dim, 'heads': p.heads,
            'layers': p.layers_count,
            'groups': {k: [row[:] for row in v] for k, v in p.groups().items()}}


def load_model(data: dict) -> Params:
    if not isinstance(data, dict) or set(data) != {'schema', 'vocab', 'dim', 'heads', 'layers', 'groups'}:
        raise ValueError('invalid model fields')
    if type(data['schema']) is not int or data['schema'] != 1:
        raise ValueError('unsupported schema')
    vocab = integer(data['vocab'], 2, 32, 'vocab')
    dim = integer(data['dim'], 2, 16, 'dim')
    heads = integer(data['heads'], 1, 4, 'heads')
    layers = integer(data['layers'], 1, 3, 'layers')
    if dim % heads:
        raise ValueError('dim must be divisible by heads')
    p = Params(vocab, dim, heads, layers, random.Random(0))
    groups = data['groups']
    if not isinstance(groups, dict) or groups.keys() != p.groups().keys():
        raise ValueError('invalid parameter groups')
    for name, expected in p.groups().items():
        raw = groups[name]
        if not isinstance(raw, list) or len(raw) != len(expected):
            raise ValueError('invalid parameter rows')
        for dst, src in zip(expected, raw):
            if not isinstance(src, list) or len(src) != len(dst):
                raise ValueError('invalid parameter columns')
            if any(type(x) not in (int, float) or abs(x) > 1e4 or not math.isfinite(x) for x in src):
                raise ValueError('invalid parameter value')
            dst[:] = src
    return p


def validate(tokens, target, p, muted):
    if not isinstance(tokens, list) or not 1 <= len(tokens) <= 24:
        raise ValueError('tokens must contain 1..24 integers')
    for t in tokens:
        integer(t, 0, p.vocab - 1, 'token')
    integer(target, 0, p.vocab - 1, 'target')
    if not isinstance(muted, (list, tuple)):
        raise ValueError('muted must be a list of [layer, head] pairs')
    pairs = []
    for pair in muted:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError('invalid muted pair')
        li = integer(pair[0], 0, p.layers_count - 1, 'layer')
        h = integer(pair[1], 0, p.heads - 1, 'head')
        if (li, h) in pairs:
            raise ValueError('duplicate muted head')
        pairs.append((li, h))
    return set(pairs)


def run(tokens: list[int], target: int, p: Params, muted=()) -> dict:
    """Return full tensors and a prediction under this precise intervention."""
    blocked = validate(tokens, target, p, muted)
    pos = positional_encoding(len(tokens), p.dim)
    x = [[p.embed[t][d] + pos[r][d] for d in range(p.dim)] for r, t in enumerate(tokens)]
    embedded = [row[:] for row in x]
    trace = []
    for li, layer in enumerate(p.layers):
        _, cache = layer.forward(x)
        before = [row[:] for row in cache['concat']]
        effective = [row[:] for row in before]
        for h in range(p.heads):
            if (li, h) in blocked:
                for row in effective:
                    row[h * layer.head_dim:(h + 1) * layer.head_dim] = [0.0] * layer.head_dim
        projected = matmul(effective, layer.wo)
        after = [[x[r][d] + projected[r][d] for d in range(p.dim)] for r in range(len(tokens))]
        trace.append({'input': x, 'q': cache['q'], 'k': cache['k'], 'v': cache['v'],
                      'attention': cache['attn'], 'concat': before, 'effective': effective,
                      'projected': projected, 'residual': after})
        x = after
    logits = [[sum(row[d] * p.wout[d][c] for d in range(p.dim)) + p.bout[c]
               for c in range(p.vocab)] for row in x]
    probabilities = softmax(logits[-1])
    prediction = predict(logits[-1])
    return {'tokens': tokens[:], 'target': target, 'muted': [list(x) for x in sorted(blocked)],
            'embedded': embedded, 'layers': trace, 'logits': logits,
            'probabilities': probabilities, 'prediction': prediction,
            'correct': prediction == target, 'loss': cross_entropy(logits[-1], target),
            'target_probability': probabilities[target],
            'margin': logits[-1][target] - max(x for i, x in enumerate(logits[-1]) if i != target)}


def summary(report):
    return {k: report[k] for k in ('prediction', 'correct', 'loss', 'target_probability', 'margin')}


def sweep(tokens, target, p):
    """All 2^(L*H) subsets for small workbench models, no selected weak controls."""
    if p.layers_count * p.heads > 8:
        raise ValueError('subset sweep is limited to 8 heads')
    heads = [(li, h) for li in range(p.layers_count) for h in range(p.heads)]
    baseline = run(tokens, target, p)
    rows = []
    for bits in range(1 << len(heads)):
        muted = [list(pair) for i, pair in enumerate(heads) if bits & (1 << i)]
        result = summary(run(tokens, target, p, muted))
        result.update({'mask': bits, 'muted': muted,
                       'loss_delta': result['loss'] - baseline['loss'],
                       'probability_delta': result['target_probability'] - baseline['target_probability']})
        rows.append(result)
    return rows


def counterfactuals(tokens):
    """Change every possible recalled key, plus each single filler position.

    Marker and query are unchanged; only the recalled-key variants change the
    task target. These are input sensitivity tests, not activation patches.
    """
    if (not isinstance(tokens, list) or len(tokens) < 4 or tokens.count(MARKER) != 1
            or tokens[-1] != QUERY or any(type(t) is not int or not 0 <= t < 12 for t in tokens)):
        raise ValueError('expected one marker and a final query in a valid recall sequence')
    marker = tokens.index(MARKER)
    if marker >= len(tokens) - 2 or any(t == QUERY for t in tokens[:-1]):
        raise ValueError('marker must precede a key before the final query')
    key = marker + 1
    variants = []
    for token in range(10):
        changed = tokens[:]
        changed[key] = token
        variants.append({'kind': 'key', 'position': key, 'tokens': changed, 'target': token})
    for position in range(len(tokens) - 1):
        if position in (marker, key):
            continue
        changed = tokens[:]
        changed[position] = (changed[position] + 1) % 10
        variants.append({'kind': 'filler', 'position': position, 'tokens': changed, 'target': tokens[key]})
    return variants


def inspect(tokens, target, p, muted=()):
    baseline = run(tokens, target, p)
    intervention = run(tokens, target, p, muted)
    return {'baseline': baseline, 'intervention': intervention, 'subsets': sweep(tokens, target, p),
            'counterfactuals': [{**v, **summary(run(v['tokens'], v['target'], p))}
                                for v in counterfactuals(tokens)]}


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Inspect a fixed-weight recall checkpoint with head-output interventions.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokens', required=True, help='comma-separated integer tokens')
    parser.add_argument('--mute', default='', help='comma-separated layer:head indices, zero based')
    args = parser.parse_args(argv)
    try:
        def unique(pairs):
            obj = {}
            for k, v in pairs:
                if k in obj:
                    raise ValueError('duplicate JSON key')
                obj[k] = v
            return obj
        with open(args.model, encoding='utf8') as f:
            text = f.read(100001)
        if len(text.encode('utf8')) > 100000:
            raise ValueError('model too large')
        model = load_model(json.loads(text, object_pairs_hook=unique,
                                     parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON'))))
        tokens = [int(x) for x in args.tokens.split(',')]
        counterfactuals(tokens)
        muted = [[int(x) for x in pair.split(':')] for pair in args.mute.split(',') if pair]
        output = inspect(tokens, tokens[tokens.index(MARKER)+1], model, muted)
        print(json.dumps(output, allow_nan=False, sort_keys=True))
    except (ValueError, TypeError, OSError, RecursionError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
