"""Opt-in binary64 causal inference via the standard online softmax recurrence.

Not a GPU kernel, KV cache, training backend, or speed benchmark. Core attention
never materializes T x T scores/probabilities. Trace mode and the separate dense
audit deliberately do; their storage is NOT included in the core buffer counts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .intervene import dump_model, integer, load_model, run as legacy_run, validate
from .model import Params, cross_entropy, matmul, positional_encoding, predict, softmax

SCHEMA = 'attnlens-online/v1'


def _matrix(x, name, rows=None, cols=None):
    if not isinstance(x, list) or not 1 <= len(x) <= 24 or (rows is not None and len(x) != rows):
        raise ValueError(f'{name}: expected 1..24 rows with matching sequence length')
    width = len(x[0]) if isinstance(x[0], list) else 0
    if not 1 <= width <= 16 or (cols is not None and width != cols):
        raise ValueError(f'{name}: expected 1..16 matching columns')
    for row in x:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f'{name}: ragged matrix')
        if any(type(v) not in (int, float) or abs(v) > 1e100 or not math.isfinite(v) for v in row):
            raise ValueError(f'{name}: finite numeric values with magnitude <= 1e100 required')
    return width


def attention(q, k, v, block_size=4, order='forward', *, trace=False):
    """Causal scaled dot-product attention, self included, no arbitrary mask.

    One score list of at most block_size entries is allocated at a time. Q/K/V,
    returned output, numerator vector and scalar temporaries remain live. Optional
    rows store a complete audit and are not a bounded-memory execution mode.
    """
    width = _matrix(q, 'q')
    _matrix(k, 'k', len(q), width)
    value_width = _matrix(v, 'v', len(q))
    integer(block_size, 1, 24, 'block_size')
    if order not in ('forward', 'reverse'):
        raise ValueError('order must be forward or reverse')
    if type(trace) is not bool:
        raise ValueError('trace must be boolean')
    output, rows = [], []
    stats = {'score_evaluations': 0, 'blocks': 0, 'peak_score_buffer': 0,
             'score_matrix_slots': 0, 'probability_matrix_slots': 0,
             'numerator_slots': value_width}
    scale = 1 / math.sqrt(width)
    for i, query in enumerate(q):
        maximum, normalizer, numerator = None, 0.0, [0.0] * value_width
        starts = range(0, i + 1, block_size)
        if order == 'reverse':
            starts = reversed(starts)
        steps = []
        for start in starts:
            # Delete this list before allocating the next block; the observed
            # buffer count is its actual length, not an estimated memory size.
            scores = [sum(query[d] * k[j][d] for d in range(width)) * scale
                      for j in range(start, min(start + block_size, i + 1))]
            stats['peak_score_buffer'] = max(stats['peak_score_buffer'], len(scores))
            stats['score_evaluations'] += len(scores)
            stats['blocks'] += 1
            next_max = max(scores) if maximum is None else max(maximum, max(scores))
            alpha = 0.0 if maximum is None else math.exp(maximum - next_max)
            if trace:
                previous = {'previous_max': maximum, 'previous_normalizer': normalizer,
                            'previous_numerator': numerator[:]}
                weights, weighted = [], [0.0] * value_width
            normalizer *= alpha
            for d in range(value_width):
                numerator[d] *= alpha
            block_sum = 0.0
            for offset, score in enumerate(scores):
                weight = math.exp(score - next_max)
                block_sum += weight
                for d in range(value_width):
                    term = weight * v[start + offset][d]
                    numerator[d] += term
                    if trace:
                        weighted[d] += term
                if trace:
                    weights.append(weight)
            normalizer += block_sum
            maximum = next_max
            if trace:
                steps.append({**previous, 'keys': list(range(start, start + len(scores))),
                              'scores': scores[:], 'weights': weights,
                              'block_sum': block_sum, 'block_weighted': weighted,
                              'rescale': alpha, 'maximum': maximum,
                              'normalizer': normalizer, 'numerator': numerator[:],
                              'output': [x / normalizer for x in numerator]})
            del scores
        result = [x / normalizer for x in numerator]
        output.append(result)
        if trace:
            rows.append({'query': i, 'steps': steps, 'output': result[:]})
    return {'output': output, 'stats': stats, 'rows': rows}


def _stack(tokens, p, block_size, order, trace, dense=False):
    pos = positional_encoding(len(tokens), p.dim)
    x = [[p.embed[t][d] + pos[r][d] for d in range(p.dim)] for r, t in enumerate(tokens)]
    embedded, layers = [row[:] for row in x], []
    totals = {'score_evaluations': 0, 'blocks': 0, 'peak_score_buffer': 0,
              'score_matrix_slots': 0, 'probability_matrix_slots': 0,
              'numerator_slots': p.dim // p.heads}
    for layer in p.layers:
        head_outputs, heads = [], []
        for h in range(p.heads):
            q, k, v = [matmul(x, weights[h]) for weights in (layer.wq, layer.wk, layer.wv)]
            if dense:
                # Independent dense control. Full causal probability matrix is
                # allocated here only, not in the inference backend.
                weights = [softmax([sum(a * b for a, b in zip(query, key)) / math.sqrt(layer.head_dim)
                                    for key in k[:i + 1]]) + [0.0] * (len(x) - i - 1)
                           for i, query in enumerate(q)]
                result = {'output': matmul(weights, v), 'attention': weights}
            else:
                result = attention(q, k, v, block_size, order, trace=trace)
                for key in ('score_evaluations', 'blocks'):
                    totals[key] += result['stats'][key]
                totals['peak_score_buffer'] = max(totals['peak_score_buffer'], result['stats']['peak_score_buffer'])
            head_outputs.append(result['output'])
            if trace:
                heads.append({'q': q, 'k': k, 'v': v, **result})
        concat = [[head_outputs[h][r][d] for h in range(p.heads) for d in range(layer.head_dim)]
                  for r in range(len(tokens))]
        projected = matmul(concat, layer.wo)
        residual = [[x[r][d] + projected[r][d] for d in range(p.dim)] for r in range(len(tokens))]
        if trace:
            layers.append({'input': x, 'heads': heads, 'concat': concat,
                           'projected': projected, 'residual': residual})
        x = residual
    logits = [[sum(row[d] * p.wout[d][c] for d in range(p.dim)) + p.bout[c]
               for c in range(p.vocab)] for row in x]
    return {'embedded': embedded if trace else [], 'layers': layers, 'logits': logits,
            **({'stats': totals} if not dense else {})}


def forward_online(tokens, p: Params, block_size=4, order='forward'):
    """Actual inference-only residual stack, no dense audit and no saved trace.

    Validates the serialized checkpoint bounds without mutating the caller's
    parameters. Training/backprop continue to use the unchanged model.forward.
    """
    checked = load_model(dump_model(p))
    validate(tokens, 0, checked, ())
    integer(block_size, 1, 24, 'block_size')
    if order not in ('forward', 'reverse'):
        raise ValueError('order must be forward or reverse')
    return _stack(tokens, checked, block_size, order, False)


def _summary(logits, target):
    probabilities = softmax(logits[-1])
    prediction = predict(logits[-1])
    return {'prediction': prediction, 'correct': prediction == target,
            'loss': cross_entropy(logits[-1], target), 'probabilities': probabilities,
            'target_probability': probabilities[target]}


def audit(spec):
    """Strict bounded JSON-shaped full audit. Debug and dense storage quadratic."""
    if not isinstance(spec, dict) or set(spec) != {'schema', 'model', 'tokens', 'target', 'block_size', 'order'}:
        raise ValueError('invalid online spec fields')
    if spec['schema'] != SCHEMA:
        raise ValueError('unsupported online schema')
    p = load_model(spec['model'])
    validate(spec['tokens'], spec['target'], p, ())
    integer(spec['block_size'], 1, 24, 'block_size')
    if spec['order'] not in ('forward', 'reverse'):
        raise ValueError('order must be forward or reverse')
    online = _stack(spec['tokens'], p, spec['block_size'], spec['order'], True)
    dense = _stack(spec['tokens'], p, 1, 'forward', True, dense=True)
    legacy = legacy_run(spec['tokens'], spec['target'], p)
    errors = [abs(a - b) for ra, rb in zip(online['logits'], dense['logits']) for a, b in zip(ra, rb)]
    relative = [abs(a - b) / max(1, abs(a), abs(b)) for ra, rb in zip(online['logits'], dense['logits']) for a, b in zip(ra, rb)]
    legacy_error = max(abs(a - b) / max(1, abs(a), abs(b)) for ra, rb in zip(legacy['logits'], dense['logits']) for a, b in zip(ra, rb))
    return {'schema': SCHEMA, 'tokens': spec['tokens'][:], 'target': spec['target'],
            'block_size': spec['block_size'], 'order': spec['order'],
            'online': online, 'dense': dense, 'online_summary': _summary(online['logits'], spec['target']),
            'dense_summary': _summary(dense['logits'], spec['target']),
            'max_absolute_logit_error': max(errors), 'max_scaled_logit_error': max(relative),
            'legacy_max_scaled_logit_error': legacy_error,
            'legacy_prediction': legacy['prediction']}


def parse(text):
    if len(text.encode('utf8')) > 131072:
        raise ValueError('spec exceeds 128 KiB')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('nonfinite JSON')
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Audit actual online causal inference against dense prefix attention; no GPU or speed claim.')
    parser.add_argument('spec', help='bounded attnlens-online/v1 JSON input')
    parser.add_argument('--output', help='write only after full input validation and successful audit')
    args = parser.parse_args(argv)
    try:
        with open(args.spec, encoding='utf8') as file:
            spec = parse(file.read(131073))
        report = audit(spec)
        text = json.dumps(report, allow_nan=False, sort_keys=True) + '\n'
        if args.output:
            Path(args.output).write_text(text, encoding='utf8')
        else:
            print(text, end='')
    except (ValueError, TypeError, OSError, RecursionError, OverflowError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
