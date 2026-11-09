# AttnLens

Causal multi-head self-attention trained from scratch in pure Python, with hand-derived gradients and a fixed-weight head-intervention workbench.

**[Open the interactive workbench](https://midya8.github.io/attnlens/)** · [Methods, exact results and attribution](INTERVENTIONS.md)

The native trainer implements token embeddings, sinusoidal positions, two residual causal attention blocks, output logits, cross-entropy and clipped SGD with no dependencies or autograd. A tiny marker-recall task asks for the token after a varying-position marker. The new workbench loads complete trained weights and independently computes inference in JavaScript. It zeroes selected head result vectors before the shared output projection at every position, then recomputes all downstream activations. It does not freeze attention patterns.

## Inspect the computation

- Trained and same-seed initialized checkpoints; 11 sequences including three declared edge structures, never selected by quality.
- All 16 head subsets on each input, including no-op and all-off residual controls. Every head is available, not a selected weak baseline.
- All 128 original held-out examples evaluated under every subset, recomputed in the browser and checked against Python. This is one fixed seed/task, not a general benchmark.
- All ten recalled-key replacements, with the original as a no-op, plus one changed value at every filler position. The task target changes only with a recalled-key replacement.
- Full baseline/intervention logits, probabilities, attention, Q/K/V, pre/post-ablation head results, projected outputs, residual streams and every source weight.
- Editable valid sequences, complete checkpoint/audit JSON exports, keyboard controls, invalid-input clearing and phone layouts.

Attention weights are a descriptive pattern. Ablation measures the output effect of this exact replacement, which can create out-of-distribution activations. It is not a universal importance score, natural-language explanation, activation-patching analysis or identified induction circuit. Head effects can interact and need not add.

## Reproduce

```bash
python -m unittest discover -s tests -v
PYTHONPATH=. python scripts/verify_regeneration.py
node --test tests/*.cjs                       # Node 22 + actual Chrome
bash examples/demo.sh                        # unchanged original gate checks
bash examples/verify_wheel.sh                # clean environment outside checkout
python -m http.server 8080 --directory docs  # http://localhost:8080
```

Rebuild all weights and data with `PYTHONPATH=. python scripts/build_workbench.py`. `verify_regeneration.py` instead compares the entire rebuilt payload at normalized tolerance 1e-12 without writing the file. The browser never claims to reproduce Python's RNG or training.

```bash
# Original CLI and gates are retained.
python -m attnlens train --dim 8 --heads 2 --layers 2 --epochs 100 --seed 7 \
  --json report.json --markdown report.md \
  --max-final-loss 0.05 --min-accuracy 0.95 --min-recall-attention 0.7
python -m attnlens heatmap --dim 8 --heads 2 --layers 2 --epochs 100 --seed 7

# Download a checkpoint from the workbench, then independently inspect it.
python -m attnlens.intervene --model attnlens-checkpoint.json \
  --tokens 6,10,2,8,2,7,5,11 --mute 0:1 > audit.json
```

For the workbench checkpoint (seed 7, 100 epochs, 64 examples per epoch, dim 8, 2 heads, 2 layers, SGD 0.1), all heads kept give 128/128 held-out correct and mean held-out cross-entropy 0.000482558 nats. Muting all four heads leaves the embedding/position/output residual path: 19/128 correct, mean loss 2.900797842. All 16 outcomes, including failures and non-monotone effects, are available. These are descriptive fixed-weight intervention results, not a decomposition of learning gains or an architecture comparison.

## Verification and numerical repair

The 46 original tests, original dataset/trainer settings and unchanged demo are retained. Thirteen added Python tests cover complete no-op equivalence, all-off residual identity, exact head-slice removal, downstream recomputation, nonmutation, strict checkpoint validation, causality, full export, exhaustive subsets, matched edits and extreme-loss finite differences. CI runs all 59 on Python 3.10-3.13, plus browser and installed-wheel jobs.

The old cross-entropy floored target probabilities at 1e-12 but still returned the unclipped softmax gradient. For a very unlikely target the reported loss was constant while gradients were nonzero. Stable log-sum-exp now computes the actual loss (e.g. logits `[1000, -1000]`, target 1 yield 2000 nats) without changing forward weights, optimizer or original positive gates. Training traces are recomputed rather than silently keeping stale numbers.

Independent Node and actual Chrome compare 352 full tensor reports, 22 complete audits and every displayed tensor/weight cell over 704 layer/head/row views. Complete held-out subset comparisons cover 4096 predictions across two checkpoints, and full JSON exports are tested. Normalized native/JS tolerance is 1e-10, not a cross-runtime bitwise claim.

## Limits and attribution

This is an educational attention-only model on synthetic recall, not useful-scale language modeling. The model has no MLP, normalization or dropout. One trained seed and initialization do not establish a one-layer difficulty gap, a classic induction mechanism or seed robustness. High recall attention is an observation, not proof of function. The UI permits fixed-checkpoint input tests, not a configurable architecture or retraining system.

The architecture, induction-head analyses and intervention concepts come from prior work linked in [INTERVENTIONS.md](INTERVENTIONS.md). This repository implements a small, independently inspectable reproduction and adds tests, numerical repair and an intervention workbench. No novelty or research-paper result is claimed.

MIT.
