"""Terminal, JSON and Markdown reports for training runs and head analysis."""

from __future__ import annotations

import json


def build_report(result: dict, analysis: dict) -> dict:
    return {
        "config": result["config"],
        "final_loss": result["final_loss"],
        "accuracy": result["accuracy"],
        "mean_recall_attention": result["mean_recall_attention"],
        "loss_curve": [round(x, 4) for x in result["losses"][:: max(1, len(result["losses"]) // 10)]],
        "head_analysis": analysis,
    }


def to_json(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def to_markdown(report: dict) -> str:
    lines = [
        "# AttnLens report",
        "",
        f"- Final loss: {report['final_loss']}",
        f"- Held-out accuracy: {report['accuracy']}",
        f"- Mean recall attention (top layer, query->key): {report['mean_recall_attention']}",
        "",
        "| Layer | Head | Mean entropy | Prev-token mass | Recall attention |",
        "|---|---|---|---|---|",
    ]
    for li, layer in enumerate(report["head_analysis"]["layers"]):
        for s in layer:
            lines.append(
                f"| {li} | {s['head']} | {s['mean_entropy']:.4f} "
                f"| {s['prev_token_mass']:.4f} | {s.get('recall_attention', 0):.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def to_terminal(report: dict) -> str:
    lines = [
        f"final_loss={report['final_loss']} accuracy={report['accuracy']} "
        f"recall_attention={report['mean_recall_attention']}",
        f"loss_curve={report['loss_curve']}",
    ]
    for li, layer in enumerate(report["head_analysis"]["layers"]):
        for s in layer:
            lines.append(
                f"  layer{li} head{s['head']}: entropy={s['mean_entropy']:.3f} "
                f"prev={s['prev_token_mass']:.3f} recall={s.get('recall_attention', 0):.3f}"
            )
    return "\n".join(lines)
