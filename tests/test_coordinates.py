"""Finite signed-coordinate/head relabeling controls, not a correctness oracle."""
import copy
import json
import math
from pathlib import Path
import random
import unittest

from attnlens.intervene import load_model, run
from attnlens.model import Params, forward, loss_and_grads, sgd_step
from attnlens.online import forward_online

DATA = json.loads((Path(__file__).resolve().parents[1] / 'docs/data.json').read_text())


def layout(p, seed):
    rng = random.Random(seed)
    result = []
    for layer in p.layers:
        heads = list(range(p.heads))
        rng.shuffle(heads)
        entries = []
        for old in heads:
            qk = list(range(layer.head_dim))
            value = qk[:]
            rng.shuffle(qk)
            rng.shuffle(value)
            entries.append((old, qk, [rng.choice((-1, 1)) for _ in qk],
                            value, [rng.choice((-1, 1)) for _ in value]))
        result.append(entries)
    return result


def relabel_groups(groups, p, mapping):
    # New head h reads old head old. Q and K use the same signed permutation;
    # V columns and the corresponding output-projection rows move together.
    out = copy.deepcopy(groups)
    for li, entries in enumerate(mapping):
        d = p.layers[li].head_dim
        for h, (old, qk, signs, value, vsigns) in enumerate(entries):
            for family in ('wq', 'wk', 'wv'):
                perm, sign = (value, vsigns) if family == 'wv' else (qk, signs)
                src = groups[f'l{li}_{family}{old}']
                out[f'l{li}_{family}{h}'] = [[row[j] * sign[c] for c, j in enumerate(perm)] for row in src]
            for c, oldc in enumerate(value):
                out[f'l{li}_wo'][h*d+c] = [vsigns[c] * x for x in groups[f'l{li}_wo'][old*d+oldc]]
    return out


def transformed(p, mapping):
    q = copy.deepcopy(p)
    groups = relabel_groups(p.groups(), p, mapping)
    for name, dst in q.groups().items():
        for row, src in zip(dst, groups[name]):
            row[:] = src
    return q


class CoordinateTests(unittest.TestCase):
    def close(self, a, b, tol=2e-10):
        if isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.close(a[key], b[key], tol)
        elif isinstance(a, list):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.close(x, y, tol)
        else:
            self.assertTrue(math.isfinite(a) and math.isfinite(b))
            self.assertLessEqual(abs(a-b), tol * max(1, abs(a), abs(b)), (a, b))

    def test_saved_models_dense_online_and_mapped_interventions(self):
        count = 0
        for record in DATA['models']:
            p = load_model(record['model'])
            for seed in (61, 97):
                mapping = layout(p, seed)
                q = transformed(p, mapping)
                for case in DATA['cases']:
                    tokens, target = case['tokens'], case['target']
                    self.close(forward(tokens, p), forward(tokens, q))
                    for block, order in ((1, 'forward'), (3, 'reverse')):
                        self.close(forward_online(tokens, p, block, order)['logits'],
                                   forward_online(tokens, q, block, order)['logits'])
                    # Exhaust all 16 subsets for the two archived 2x2 models.
                    for bits in range(1 << (p.heads * p.layers_count)):
                        new = [(li, h) for li in range(p.layers_count) for h in range(p.heads)
                               if bits & (1 << (li*p.heads+h))]
                        old = [(li, mapping[li][h][0]) for li, h in new]
                        self.close(run(tokens, target, p, old)['logits'],
                                   run(tokens, target, q, new)['logits'])
                    count += 1
        self.assertEqual(count, 44)

    def test_generated_losses_and_every_parameter_gradient(self):
        rng = random.Random(991)
        for case in range(36):
            dim, heads, layers = ((2, 1, 1), (4, 2, 2), (6, 3, 3), (8, 4, 2))[case % 4]
            p = Params(7, dim, heads, layers, rng)
            mapping = layout(p, case + 80)
            q = transformed(p, mapping)
            tokens = [rng.randrange(7) for _ in range(1 + case % 8)]
            target = rng.randrange(7)
            lp, gp = loss_and_grads(tokens, target, p)
            lq, gq = loss_and_grads(tokens, target, q)
            self.close(lp, lq)
            self.close(relabel_groups(gp, p, mapping), gq)

    def test_clipped_and_unclipped_online_sgd_paths(self):
        rng = random.Random(280)
        for max_norm in (1e-5, 1e6):
            for dim, heads in ((2, 1), (4, 2), (6, 3), (8, 4)):
                p = Params(7, dim, heads, 2, rng)
                mapping = layout(p, dim + 900)
                q = transformed(p, mapping)
                clipped = []
                for step in range(6):
                    tokens = [rng.randrange(7) for _ in range(1 + step)]
                    target = rng.randrange(7)
                    lp, gp = loss_and_grads(tokens, target, p)
                    lq, gq = loss_and_grads(tokens, target, q)
                    self.close(lp, lq)
                    self.close(relabel_groups(gp, p, mapping), gq)
                    norm = math.sqrt(sum(x*x for group in gp.values() for row in group for x in row))
                    clipped.append(norm > max_norm)
                    sgd_step(p, gp, .03, max_norm)
                    sgd_step(q, gq, .03, max_norm)
                    self.close(relabel_groups(p.groups(), p, mapping), q.groups())
                self.assertEqual(clipped, [max_norm == 1e-5] * 6)

    def test_one_sided_value_change_is_not_a_symmetry(self):
        p = load_model(DATA['models'][0]['model'])
        q = copy.deepcopy(p)
        # No compensating output-projection row sign: this must change logits.
        for row in q.layers[0].wv[0]:
            row[0] *= -1
        difference = max(abs(x-y) for case in DATA['cases']
                         for a, b in zip(forward(case['tokens'], p), forward(case['tokens'], q))
                         for x, y in zip(a, b))
        self.assertGreater(difference, 1e-4)


if __name__ == '__main__':
    unittest.main()
