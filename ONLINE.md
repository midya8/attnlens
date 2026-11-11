# Online causal inference, with every rescaling step exposed

AttnLens 0.3 adds an **opt-in inference backend** using standard blocked online
softmax on the actual trained and initialized checkpoints. It recomputes Q/K/V,
head results, output projections, residuals and all vocabulary logits. The
original training, gradients, head-output interventions and checkpoint bytes
are preserved. The original workbench gains a link, not a replacement.

Open `docs/online.html` through a local HTTP server. No API keys or dependencies.
The browser loads source weights from the same `data.json` as the intervention
workbench, then computes every intermediate in independent JavaScript. It does
not load Python-generated score, accumulator or output tensors.

## Run native inference or an audit

```sh
python -m pip install .
python -m attnlens.online docs/online-example.json --output audit.json
attnlens-online docs/online-example.json --output audit-entry.json
python -m http.server 8000 --directory docs
# Open http://localhost:8000/online.html
```

The checked-in input is complete, including the exact trained checkpoint. In
the browser, edit tokens and target, block size or block order, then download
both input and audit. The downloaded input replays with the same commands.

```python
from attnlens.intervene import load_model
from attnlens.online import attention, audit, forward_online, parse

spec = parse(open('docs/online-example.json', encoding='utf8').read())
p = load_model(spec['model'])
# Actual inference, without dense matrices or saved accumulator trace:
result = forward_online(spec['tokens'], p, block_size=3, order='reverse')
print(result['logits'], result['stats'])
# Separate, intentionally storage-heavy debug mode:
report = audit(spec)
print(report['max_scaled_logit_error'])
```

`forward_online` returns all logits and measured structural buffer/work counts.
Its `embedded` and `layers` fields are empty: it does not retain debug traces.
It validates a copy of the serialized checkpoint, without mutating the caller's
parameters. It is not a backward/training backend or a drop-in gradient cache.

## Recurrence and exact causal contract

For one query, admit keys `0..query`, **including self**. Partition that prefix
into consecutive blocks of at most `B` keys. Forward consumes them left to right;
reverse consumes the same blocks right to left, keeping keys within each block
in increasing order. A partial final block remains partial. Future keys are
never scored, even with reverse order. There is no arbitrary mask, empty row,
dropout, padding mask or attention bias in this contract.

Start with no maximum, `l=0`, and value-width numerator `u=0`. For each block:

```text
s_j = dot(Q_query, K_j) / sqrt(head_dimension)
m_new = max(previous maximum, max block scores)
alpha = 0 for the first block; otherwise exp(m_old - m_new)
w_j = exp(s_j - m_new)
l_new = alpha*l_old + sum(w_j)
u_new = alpha*u_old + sum(w_j * V_j)
head_result = u_new / l_new
```

The actual core scales its live numerator in place, adds weighted values, and
normalizes only the final result. Trace mode records the previous maximum,
normalizer and numerator; key indices and scores; rescaling factor; each
shifted exponential; block mass and weighted values; new state and intermediate
normalized result. No softmax approximation, sparse selection, learned change,
linear-attention substitution, or discarded admitted key is involved.

Mathematical equivalence does **not** mean bitwise binary64 identity. Block size
and order alter floating-point summation; CPU/browser math libraries can differ.
Max-shifting avoids exponential overflow, but very small terms can underflow to
zero. Near-tied predictions need not remain identical for every custom model.
No universal error bound is inferred from the fixture tests.

## What the counters measure, and what they do not

For sequence length `T`, `H` heads and `L` layers:

- Score evaluations: `L*H*T*(T+1)/2`, independent of block size/order.
- Block visits: `L*H*sum(ceil((i+1)/B), i=0..T-1)`.
- Peak live score-buffer entries: actual maximum `len(scores)` observed during
  execution, at most `min(B,T)`. The old list is deleted before the next block.
- Full score/probability-matrix slots: zero **in the inference core**.
- Numerator entries: one active head's value dimension.

These are structural counts, **not** process memory in bytes, peak RSS,
hardware IO, allocation counts, speed or a benchmark. Q/K/V projections, model
weights, output vectors, concatenated heads, projection/residual arrays, input
validation copies, logits, scalars and Python object overhead remain. Work is
still quadratic. There is no KV cache or fused GPU kernel.

The optional trace copies every block state. `audit` additionally computes and
retains an independent dense causal probability matrix for each head, copied
Q/K/V and complete residual pathways. JSON and DOM storage add more. These modes
are deliberately quadratic and are **not** bounded-total-memory demonstrations.

## Dense control and the legacy finite-sentinel edge case

The control independently computes each complete causal prefix softmax and
multiplies dense probabilities by values. It skips future keys exactly. Audit
reports both full pathways and maximum absolute/scaled logit errors over every
position and vocabulary output. Scaled error is
`abs(a-b)/max(1,abs(a),abs(b))`, not relative error near zero.

The unchanged original `model.forward` uses `-1e30` for future scores. On the
supplied checkpoints, that agrees with causal skipping within tolerance.
However, a valid extreme custom checkpoint can produce permitted scores below
`-1e30`, making the legacy sentinel incorrectly competitive with admitted keys.
The new backend does not copy that behavior. `legacy_max_scaled_logit_error` and
`legacy_prediction` report the legacy comparison separately. A tested, bounded
custom checkpoint exhibits the discrepancy. The old training/backward engine
is intentionally not silently changed in this extension.

## Input and failure behavior

`attnlens-online/v1` has exactly `schema`, `model`, `tokens`, `target`,
`block_size`, and `order`. The existing strict checkpoint loader is reused:
vocabulary 2-32, dimension 2-16 divisible by heads 1-4, layers 1-3, exact parameter
shapes, finite values of magnitude at most 10,000. Tokens contain 1-24 integer
IDs in that vocabulary; target is one ID; block size is an integer 1-24; order
is `forward` or `reverse`. Booleans are not integers in this contract.

The standalone `attention` API accepts matching Q/K sequence lengths, identical
query/key widths 1-16 and value width 1-16; 1-24 rows; finite numeric magnitude
at most `1e100`. The bound keeps dot products and weighted numerators within
binary64 range. All rows are nonempty causal prefixes.

The parser rejects duplicate object keys, nonfinite JSON and input larger than
128 KiB in UTF-8. Invalid schema, dimensions, values, options or missing files
return CLI exit 2. The output destination is not opened until the full input
and audit succeed; invalid input does not truncate an existing report. CLI
output files are ordinary writes, not transactional replacements on disk failure.
The browser hides stale results and disables exports as soon as editable input
changes, including invalid input.

## Reproduction and test evidence

```sh
python -m unittest discover -s tests -v
node --test --test-concurrency=1 tests/*.cjs
PYTHONPATH=. python scripts/verify_regeneration.py
bash examples/verify_wheel.sh
```

The current suite has 76 Python tests, retaining the original 59 byte-for-byte:

- 648 exhaustive signed-value/ternary-score cases against independent 80-digit
  Decimal dense softmax; 80 seeded Q/K/V sets × ten block/order controls.
- Every score-buffer bound and work count for lengths 1-24, blocks 1-24 and both
  orders; causal prefix invariance; singleton/equal-score means; overflow and
  underflow cases; state-by-state prefix reconstruction.
- 1,056 actual trained/initialized original-forward comparisons across all 11
  fixtures, every block size 1-24 and both orders; 44 complete dense model audits;
  100 seeded model shapes × two orders; whole-stack prefix invariance; legacy
  finite-sentinel counterexample; strict validation and invalid-output retention.

Independent JavaScript compares **106 full model reports and 80 standalone
reports**, 5,585 query rows, 11,254 accumulator states and 775,344 numeric fields
with native Python at scaled tolerance `1e-10`. It also checks every length,
block size/order buffer count and causal prefixes.

Actual Chrome checks 66 complete native model reports and all 2,112 head/query
views, every displayed accumulator and Q/K/V/residual table, all logits and
metrics. Whole reports use a floating-point tolerance; exact displayed strings
are checked against the actual browser values, including small rounding deltas.
It checks 198 layouts across 320/390/1280px widths, stale-result invalidation,
trusted keyboard edits and two actual downloaded files replayed by native Python.
Saved screenshots are inspected separately; a passing DOM test is not a visual
inspection. The original intervention browser suite still runs independently.

The isolated wheel check exercises both new CLI paths and native API outside
the checkout, alongside the original training/intervention reproduction.

## Sources and attribution

This is an educational implementation by Midya Rahmani, not a new algorithm.

- Milakov and Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/html/1805.02867v2), 2018. Running maximum, normalizer and parallel merge recurrence.
- Dao et al., [FlashAttention](https://arxiv.org/html/2205.14135v2), 2022. Tiled exact attention and IO-aware GPU implementation. Our Python code is not that GPU kernel; their hardware performance does not transfer here.
- UW CSE599M, [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf), 2023 course notes.
- PyTorch, [Numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html). Nonassociativity and backend-dependent rounding.
- Higham, [Accurately Computing the Softmax Function](https://nhigham.com/2020/02/03/accurately-computing-the-softmax-function/), 2020. Max-shifting and numerical caveats.
- PyTorch [historical all-masked-row issue #103963](https://github.com/pytorch/pytorch/issues/103963), 2023. A motivating edge case, not a claim about current PyTorch. This contract always includes self.
