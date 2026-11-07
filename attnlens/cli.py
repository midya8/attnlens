"""AttnLens command line: train the tiny attention model, inspect its heads."""

from __future__ import annotations

import argparse
import sys

from .data import MARKER, QUERY, make_dataset
from .patterns import analyze, heatmap
from .report import build_report, to_json, to_markdown, to_terminal
from .train import train


def cmd_train(args: argparse.Namespace) -> int:
    try:
        result = train(dim=args.dim, heads=args.heads, layers=args.layers,
                       epochs=args.epochs, lr=args.lr, seed=args.seed,
                       per_epoch=args.per_epoch, n_eval=args.n_eval)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    sample, _ = make_dataset(1, seed=args.seed + 999)[0]
    report = build_report(result, analyze(sample, result["params"], args.layers))
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            fh.write(to_json(report))
    if args.markdown_path:
        with open(args.markdown_path, "w", encoding="utf-8") as fh:
            fh.write(to_markdown(report))
    print(to_terminal(report))

    failures: list[str] = []
    if args.max_final_loss is not None and result["final_loss"] > args.max_final_loss:
        failures.append(f"final_loss {result['final_loss']} > {args.max_final_loss}")
    if args.min_accuracy is not None and result["accuracy"] < args.min_accuracy:
        failures.append(f"accuracy {result['accuracy']} < {args.min_accuracy}")
    if (args.min_recall_attention is not None
            and result["mean_recall_attention"] < args.min_recall_attention):
        failures.append(
            f"recall_attention {result['mean_recall_attention']} < {args.min_recall_attention}")
    for failure in failures:
        print(f"GATE FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


def cmd_heatmap(args: argparse.Namespace) -> int:
    try:
        result = train(dim=args.dim, heads=args.heads, layers=args.layers,
                       epochs=args.epochs, lr=args.lr, seed=args.seed,
                       per_epoch=args.per_epoch, n_eval=8)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    sample, target = make_dataset(1, seed=args.seed + 999)[0]
    analysis = analyze(sample, result["params"], args.layers)
    key_pos = sample.index(MARKER) + 1
    print(f"sequence: {sample}  (MARKER={MARKER} at {sample.index(MARKER)}, "
          f"key={target} at {key_pos}, QUERY={QUERY} last)")
    for li, layer in enumerate(analysis["layers"]):
        for s in layer:
            print(f"layer{li} head{s['head']}: query row recall={s.get('recall_attention', 0):.3f}")
    from .model import Cache, forward
    cache = Cache()
    forward(sample, result["params"], cache)
    top_row = cache.layer_caches[-1]["attn"][0][-1]
    print("top head query row:", heatmap(top_row))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="attnlens",
        description="Train causal multi-head self-attention from scratch and analyze what its heads learn.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", help="train and evaluate with quality gates")
    p.add_argument("--dim", type=int, default=8)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--per-epoch", type=int, default=64)
    p.add_argument("--n-eval", type=int, default=128)
    p.add_argument("--json", dest="json_path", default=None)
    p.add_argument("--markdown", dest="markdown_path", default=None)
    p.add_argument("--max-final-loss", type=float, default=None)
    p.add_argument("--min-accuracy", type=float, default=None)
    p.add_argument("--min-recall-attention", type=float, default=None)
    h = sub.add_parser("heatmap", help="train and print the query row's attention pattern")
    for name, default in (("dim", 8), ("heads", 2), ("layers", 2), ("epochs", 100)):
        h.add_argument(f"--{name}", type=int, default=default)
    h.add_argument("--lr", type=float, default=0.1)
    h.add_argument("--seed", type=int, default=7)
    h.add_argument("--per-epoch", type=int, default=64)
    args = parser.parse_args(argv)
    return cmd_train(args) if args.command == "train" else cmd_heatmap(args)


if __name__ == "__main__":
    sys.exit(main())
