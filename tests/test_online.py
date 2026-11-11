"""Independent dense Decimal oracle and actual residual-stack controls."""
import copy
from decimal import Decimal, localcontext
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from attnlens.intervene import dump_model, load_model
from attnlens.model import Params, forward
from attnlens.online import SCHEMA, attention, audit, forward_online, parse

DATA = json.loads((Path(__file__).resolve().parents[1] / 'docs/data.json').read_text())


def close(a, b, tolerance=1e-10):
    if isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            close(x, y, tolerance)
    else:
        assert math.isfinite(a) and math.isfinite(b)
        assert abs(a - b) <= tolerance * max(1, abs(a), abs(b)), (a, b)


def decimal_dense(q, k, v):
    # No use of production softmax, scale, recurrence, or matrix multiplication.
    with localcontext() as context:
        context.prec = 80
        d = Decimal(len(q[0])).sqrt()
        result = []
        for i, query in enumerate(q):
            scores = [sum((Decimal.from_float(float(a)) * Decimal.from_float(float(b))
                           for a, b in zip(query, key)), Decimal(0)) / d for key in k[:i+1]]
            maximum = max(scores)
            exp = [(x - maximum).exp() for x in scores]
            total = sum(exp)
            result.append([float(sum((w * Decimal.from_float(float(v[j][c]))
                                      for j, w in enumerate(exp)), Decimal(0)) / total)
                           for c in range(len(v[0]))])
        return result


def spec(model=0, case=0, block=3, order='forward'):
    c = DATA['cases'][case]
    return {'schema': SCHEMA, 'model': copy.deepcopy(DATA['models'][model]['model']),
            'tokens': c['tokens'][:], 'target': c['target'], 'block_size': block, 'order': order}


class OnlineTests(unittest.TestCase):
    def test_exhaustive_ternary_scores_and_signed_values(self):
        count = 0
        for scores in itertools.product((-2., 0., 2.), repeat=4):
            q, k, v = [[1.]] * 4, [[x] for x in scores], [[-3., 2.], [0., -7.], [4., 5.], [-2., -1.]]
            expected = decimal_dense(q, k, v)
            for block in range(1, 5):
                for order in ('forward', 'reverse'):
                    close(attention(q, k, v, block, order)['output'], expected, 2e-14)
                    count += 1
        self.assertEqual(count, 648)

    def test_seeded_decimal_dense_all_dimensions_and_blocks(self):
        rng = random.Random(917)
        for seed in range(80):
            n, d, dv = rng.randint(1, 12), rng.randint(1, 8), rng.randint(1, 8)
            q = [[rng.uniform(-4, 4) for _ in range(d)] for _ in range(n)]
            k = [[rng.uniform(-4, 4) for _ in range(d)] for _ in range(n)]
            v = [[rng.uniform(-9, 9) for _ in range(dv)] for _ in range(n)]
            reference = decimal_dense(q, k, v)
            for block in (1, 2, 3, n, 24):
                for order in ('forward', 'reverse'):
                    close(attention(q, k, v, block, order)['output'], reference, 2e-12)

    def test_overflow_avoided_and_underflow_explicit(self):
        q, k, v = [[1.]] * 4, [[10000.], [-10000.], [9999.], [10001.]], [[2., -9.], [100., 100.], [-3., 8.], [7., -4.]]
        self.assertRaises(OverflowError, math.exp, 10000)
        for b in range(1, 5):
            for order in ('forward', 'reverse'):
                report = attention(q, k, v, b, order, trace=True)
                close(report['output'], decimal_dense(q, k, v))
                self.assertTrue(any(0. in row['steps'][-1]['weights'] for row in report['rows'][1:]) or b == 1)
        extreme = attention([[1e100], [1e100]], [[-1e100], [1e100]], [[-1e100], [1e100]], 1, 'reverse')
        self.assertEqual(extreme['output'], [[-1e100], [1e100]])

    def test_singleton_and_equal_scores_are_prefix_means(self):
        for n in range(1, 25):
            q, k, v = [[0., 0.]] * n, [[2., -3.]] * n, [[(-1)**i * i, i+1.] for i in range(n)]
            expected = [[sum(v[j][d] for j in range(i+1))/(i+1) for d in range(2)] for i in range(n)]
            for b in (1, 3, 24):
                close(attention(q, k, v, b)['output'], expected, 1e-14)

    def test_every_trace_transition_and_prefix_reconstruction(self):
        rng = random.Random(11)
        for n in range(1, 25):
            q, k, v = [[[rng.uniform(-3, 3) for _ in range(3)] for _ in range(n)] for _ in range(3)]
            for order in ('forward', 'reverse'):
                a = attention(q, k, v, 3, order, trace=True)
                for row in a['rows']:
                    seen, maximum, norm, numerator = [], None, 0., [0.] * 3
                    for step in row['steps']:
                        self.assertEqual(step['previous_max'], maximum)
                        self.assertEqual(step['previous_normalizer'], norm)
                        self.assertEqual(step['previous_numerator'], numerator)
                        self.assertTrue(all(j <= row['query'] for j in step['keys']))
                        seen += step['keys']
                        self.assertEqual(len(seen), len(set(seen)))
                        maximum = step['maximum']
                        expected_scores = [sum(q[row['query']][d] * k[j][d] for d in range(3)) / math.sqrt(3) for j in seen]
                        weights = [math.exp(x - max(expected_scores)) for x in expected_scores]
                        close(maximum, max(expected_scores))
                        close(step['normalizer'], sum(weights))
                        expected_num = [sum(w * v[j][d] for j, w in zip(seen, weights)) for d in range(3)]
                        close(step['numerator'], expected_num)
                        close(step['output'], [x / sum(weights) for x in expected_num])
                        norm, numerator = step['normalizer'], step['numerator']
                    self.assertEqual(sorted(seen), list(range(row['query']+1)))
                    self.assertEqual(row['output'], step['output'])

    def test_observed_buffers_and_exact_causal_work_counts(self):
        for n in range(1, 25):
            for b in range(1, 25):
                for order in ('forward', 'reverse'):
                    r = attention([[1., 2.]] * n, [[2., 3.]] * n, [[1., 0., -1.]] * n, b, order)
                    self.assertEqual(r['rows'], [])
                    self.assertEqual(r['stats'], {'score_evaluations': n*(n+1)//2,
                        'blocks': sum((i+b)//b for i in range(n)), 'peak_score_buffer': min(n, b),
                        'score_matrix_slots': 0, 'probability_matrix_slots': 0, 'numerator_slots': 3})

    def test_future_suffix_cannot_change_earlier_outputs(self):
        rng = random.Random(43)
        for n in range(2, 25):
            q, k, v = [[[rng.uniform(-4, 4) for _ in range(4)] for _ in range(n)] for _ in range(3)]
            for order in ('forward', 'reverse'):
                full = attention(q, k, v, 3, order)['output']
                for end in range(1, n):
                    self.assertEqual(full[:end], attention(q[:end], k[:end], v[:end], 3, order)['output'])

    def test_inputs_not_mutated(self):
        inputs = [[[1., 2.], [3., 4.]]] * 3
        before = copy.deepcopy(inputs)
        a = attention(*inputs, trace=True)
        self.assertEqual(inputs, before)
        a['rows'][0]['steps'][0]['numerator'][0] = -99
        self.assertEqual(inputs, before)
        s = spec()
        original = copy.deepcopy(s)
        audit(s)
        self.assertEqual(s, original)

    def test_original_trained_init_all_cases_all_blocks_orders(self):
        count = 0
        for model in range(2):
            p = load_model(DATA['models'][model]['model'])
            for case, c in enumerate(DATA['cases']):
                original = forward(c['tokens'], p)
                for b in range(1, 25):
                    for order in ('forward', 'reverse'):
                        r = forward_online(c['tokens'], p, b, order)
                        close(r['logits'], original)
                        self.assertEqual(r['layers'], [])
                        self.assertEqual(r['stats']['score_evaluations'], len(c['tokens'])*(len(c['tokens'])+1)//2*p.heads*p.layers_count)
                        count += 1
        self.assertEqual(count, 1056)

    def test_full_audit_actual_checkpoints(self):
        for model in range(2):
            for case in range(11):
                for order in ('forward', 'reverse'):
                    a = audit(spec(model, case, 3, order))
                    self.assertLess(a['max_scaled_logit_error'], 1e-10)
                    self.assertLess(a['legacy_max_scaled_logit_error'], 1e-10)
                    self.assertEqual(a['online_summary']['prediction'], a['dense_summary']['prediction'])
                    self.assertEqual(len(a['online']['layers']), 2)
                    for lo, ld in zip(a['online']['layers'], a['dense']['layers']):
                        for key in ('input', 'concat', 'projected', 'residual'):
                            close(lo[key], ld[key])
                        for ho, hd in zip(lo['heads'], ld['heads']):
                            for key in ('q', 'k', 'v', 'output'):
                                close(ho[key], hd[key])

    def test_seeded_full_models_varying_shapes(self):
        rng = random.Random(338)
        for case in range(100):
            heads = rng.randint(1, 4)
            dim = heads * rng.randint(1, 16//heads)
            if dim < 2:
                dim = 2
            p = Params(rng.randint(2, 32), dim, heads, rng.randint(1, 3), rng)
            tokens = [rng.randrange(p.vocab) for _ in range(rng.randint(1, 24))]
            before = dump_model(p)
            for order in ('forward', 'reverse'):
                close(forward_online(tokens, p, rng.randint(1, 24), order)['logits'], forward(tokens, p))
            self.assertEqual(dump_model(p), before)

    def test_full_model_prefix_invariance(self):
        for model in DATA['models']:
            p = load_model(model['model'])
            tokens = DATA['cases'][0]['tokens']
            for order in ('forward', 'reverse'):
                full = forward_online(tokens, p, 3, order)['logits']
                for end in range(1, len(tokens)):
                    self.assertEqual(full[:end], forward_online(tokens[:end], p, 3, order)['logits'])

    def test_legacy_finite_sentinel_difference_is_reported(self):
        # Legitimately bounded custom checkpoint, not a change to old training.
        p = Params(2, 2, 1, 3, random.Random(0))
        for name, group in p.groups().items():
            for row in group:
                row[:] = [10000.] * len(row)
            if '_wk' in name:
                for row in group:
                    row[:] = [-10000.] * len(row)
        p.embed[1] = [0., 0.]
        a = audit({'schema': SCHEMA, 'model': dump_model(p), 'tokens': [0, 1],
                   'target': 0, 'block_size': 1, 'order': 'reverse'})
        self.assertGreater(a['legacy_max_scaled_logit_error'], .01)
        self.assertLess(a['max_scaled_logit_error'], 1e-10)

    def test_invalid_matrices(self):
        valid = [[1., 2.], [3., 4.]]
        for bad in ([], [1], [[]], [[1.], [2., 3.]], [[True]], [[float('nan')]], [[float('inf')]], [[10**1000]], [[1.]]*25, [[1.]*17]):
            for index in range(3):
                args = [valid, valid, valid]
                args[index] = bad
                with self.assertRaises(ValueError):
                    attention(*args)
        for kwargs in ({'block_size': True}, {'block_size': 0}, {'block_size': 25}, {'block_size': 1.5}, {'order': None}, {'trace': 1}):
            with self.assertRaises(ValueError):
                attention(valid, valid, valid, **kwargs)

    def test_strict_specs(self):
        for update in ({'schema': 1}, {'target': True}, {'tokens': []}, {'tokens': [True]}, {'tokens': [12]}, {'tokens': [1.5]}, {'block_size': False}, {'block_size': 25}, {'order': 'shuffle'}, {'extra': 1}, {'model': {}}):
            s = spec()
            s.update(update)
            with self.assertRaises(ValueError):
                audit(s)
        for key in spec():
            s = spec()
            del s[key]
            with self.assertRaises(ValueError):
                audit(s)

    def test_strict_json(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', ' '*131073, '['*2000):
            with self.assertRaises((ValueError, RecursionError)):
                parse(text)

    def test_cli_success_and_invalid_output_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp)/'spec.json', Path(tmp)/'audit.json'
            source.write_text(json.dumps(spec()))
            proc = subprocess.run([sys.executable, '-m', 'attnlens.online', str(source), '--output', str(dest)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(dest.read_text()), audit(spec()))
            before = dest.read_bytes()
            for text in ('{"x":1,"x":2}', '{}', ' '*131073):
                source.write_text(text)
                proc = subprocess.run([sys.executable, '-m', 'attnlens.online', str(source), '--output', str(dest)], capture_output=True, text=True)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(dest.read_bytes(), before)
            proc = subprocess.run([sys.executable, '-m', 'attnlens.online', str(Path(tmp)/'missing')], capture_output=True)
            self.assertEqual(proc.returncode, 2)


if __name__ == '__main__':
    unittest.main()
