#!/usr/bin/env bash
# Fully offline AttnLens demo: train two-layer causal multi-head attention
# from scratch on the marker-recall task, verify the induction pattern in the
# top head, and exercise both gate sides.
set -u
cd "$(dirname "$0")/.."

OUT="$(mktemp -d /tmp/attnlens-demo-XXXX)"
trap 'echo "demo artifacts: $OUT"' EXIT

echo "== train (expect pass) =="
python3 -m attnlens train --dim 8 --heads 2 --layers 2 --epochs 100 \
  --per-epoch 64 --n-eval 64 --seed 7 --lr 0.1 \
  --json "$OUT/report.json" --markdown "$OUT/report.md" \
  --max-final-loss 0.05 --min-accuracy 0.95 --min-recall-attention 0.7 \
  || { echo "FAIL quality gate that should pass"; exit 1; }

python3 - "$OUT/report.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["accuracy"] >= 0.95
assert report["final_loss"] < 0.05
assert report["mean_recall_attention"] >= 0.7
top = max(report["head_analysis"]["layers"][-1], key=lambda s: s.get("recall_attention", 0))
assert top["recall_attention"] >= 0.7
print("assertions ok: accuracy %s, top-head recall attention %s" % (
    report["accuracy"], top["recall_attention"]))
PY

echo "== heatmap =="
python3 -m attnlens heatmap --dim 8 --heads 2 --layers 2 --epochs 100 --seed 7 \
  || { echo "FAIL heatmap"; exit 1; }

echo "== gate (expect fail) =="
if python3 -m attnlens train --dim 8 --heads 2 --layers 2 --epochs 100 \
  --per-epoch 64 --n-eval 64 --seed 7 --lr 0.1 \
  --min-accuracy 1.01 >/dev/null 2>&1; then
  echo "FAIL impossible gate passed"; exit 1
fi
echo "impossible accuracy gate rejected as expected"

echo "DEMO OK"
