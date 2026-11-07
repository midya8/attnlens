# AttnLens

Causal multi-head self-attention trained from scratch in pure Python, with gradient verification and attention-pattern analysis.

AttnLens implements two residual blocks of causal multi-head self-attention - Q/K/V projections, scaled dot-product attention with masking, sinusoidal positions, cross-entropy loss - with hand-derived backpropagation and no dependencies. It trains on a marker-recall task (recall the token that followed a marker, position varies) that one layer cannot solve cleanly: the key token's own content does not say it followed a marker, so the first layer must move that information before the second can recall it. After training, the top head attends to the key token with ~0.9 weight - the induction-head pattern, measured rather than asserted.

## Why

Attention is the mechanism under every modern language model, and "attention weights" get quoted constantly without anyone checking what a head actually does. AttnLens makes the whole loop concrete:

- **real training, no autograd** - every gradient (embeddings, per-head Q/K/V, output projections, both layers through the residuals) is derived by hand and verified against finite differences in the test suite;
- **deterministic end to end** - one seeded RNG drives initialization and the fresh training samples generated each epoch, so a config and seed reproduce the run, and quality numbers are pinnable in tests;
- **stable optimization** - global-norm gradient clipping; without it the loss diverges (the demo's bounds and tests pin the stable configuration);
- **pattern analysis** - per-head mean entropy, previous-token mass, and recall attention (the query position's weight on the key token), with text heatmaps and terminal/JSON/Markdown reports;
- **CI gates** - `--max-final-loss`, `--min-accuracy` and `--min-recall-attention` fail the pipeline when training regresses.

## Usage

```bash
python3 -m attnlens train --dim 8 --heads 2 --layers 2 --epochs 100 --seed 7 \
  --json report.json --markdown report.md \
  --max-final-loss 0.05 --min-accuracy 0.95 --min-recall-attention 0.7
python3 -m attnlens heatmap --dim 8 --heads 2 --layers 2 --epochs 100 --seed 7
```

With seed 7 the model reaches 100% held-out accuracy and ~0.87 mean recall attention; the heatmap command prints the top head's query row attending to the key position.

## Verification

- 46 unit tests cover softmax stability, positional encodings, causal masking (changing a future token leaves earlier logits untouched), attention row normalization, residual structure, finite-difference gradient checks for every parameter group across both layers, SGD loss decrease, gradient clipping behavior, dataset structure and determinism, head statistics (entropy, prev-token mass, recall attention) including the no-marker case, all three report formats, the CLI end to end including both gate sides, and pinned trained quality: 100 seeded epochs reach >= 95% accuracy and >= 0.7 recall attention.
- `bash examples/demo.sh` runs fully offline: trains, verifies the induction pattern in the top head, asserts report contents, and confirms an impossible gate fails.
- CI runs the same suite across Python 3.10-3.13.

Bounds worth stating: this is a two-layer model on a synthetic recall task - it demonstrates the attention mechanism, manual backpropagation and pattern analysis, not language modeling at any useful scale. The marker-recall task is designed so one layer struggles and two layers suffice; that gap is the point, not a limitation discovered later. Attention weights show what the heads learned to do on this task; they are not explanations of decisions in larger models.

## License

MIT
