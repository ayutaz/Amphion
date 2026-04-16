"""Generate final evaluation report (Table II/III format).

Aggregates all metrics (MCD, F0-RMSE, SingMOS, SECS, SVT-F1) into
JSON + Markdown report matching paper Table II (objective evaluation)
and Table III (melody accuracy comparison).

Ticket: M4-13
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Metric JSON files and their keys
METRIC_FILES = {
    "MCD": ("mcd", "mean_mcd", "std_mcd"),
    "F0-RMSE": ("f0_rmse", "mean_f0_rmse", "std_f0_rmse"),
    "SingMOS": ("singmos", "mean_singmos", "std_singmos"),
    "SECS": ("secs", "mean_secs", "std_secs"),
    "SVT-F1": ("svt_f1", "mean_f1", "std_f1"),
}

# Paper target values for comparison
PAPER_TARGETS = {
    "MCD": {"seen": 4.17, "unseen": None},
    "F0-RMSE": {"seen": 0.042, "unseen": None},
    "SingMOS": {"seen": 4.32, "unseen": None},
    "SECS": {"seen": 0.912, "unseen": None},
    "SVT-F1": {"seen": 0.711, "unseen": None},
}


def _load_metric_json(json_path: Path) -> dict | None:
    """Safely load a metric JSON file.

    Args:
        json_path: Path to JSON file.

    Returns:
        Parsed dict, or None if file not found or invalid.
    """
    try:
        with open(json_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("Metric file not found: %s", json_path)
        return None
    except json.JSONDecodeError as e:
        logger.warning("Invalid JSON in %s: %s", json_path, e)
        return None


def generate_report(
    eval_dir: str | Path,
    output_path: str | Path = "results.json",
) -> dict:
    """Aggregate all metrics into JSON + Markdown report.

    Expected directory structure:
        eval_dir/
            mcd_seen.json, mcd_unseen.json
            f0_rmse_seen.json, f0_rmse_unseen.json
            singmos_seen.json, singmos_unseen.json
            secs_seen.json, secs_unseen.json
            svt_f1_seen.json, svt_f1_unseen.json

    Args:
        eval_dir: Directory containing metric JSON files.
        output_path: Path to write the aggregated report JSON.

    Returns:
        Report dict with all metrics organized by Seen/Unseen.
    """
    eval_dir = Path(eval_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {}

    for metric_name, (file_prefix, mean_key, std_key) in METRIC_FILES.items():
        metric_report = {}

        for split in ("seen", "unseen"):
            json_path = eval_dir / f"{file_prefix}_{split}.json"
            data = _load_metric_json(json_path)

            if data is not None:
                metric_report[split] = {
                    "mean": data.get(mean_key),
                    "std": data.get(std_key),
                    "n_samples": data.get("n_samples"),
                }
            else:
                metric_report[split] = {"mean": None, "std": None, "n_samples": None}

        # Add paper target for comparison
        metric_report["paper_target"] = PAPER_TARGETS.get(metric_name, {})

        # Compute delta from paper target
        for split in ("seen", "unseen"):
            target = metric_report["paper_target"].get(split)
            actual = metric_report[split]["mean"]
            if target is not None and actual is not None:
                metric_report[split]["delta"] = actual - target
            else:
                metric_report[split]["delta"] = None

        report[metric_name] = metric_report

    # Save JSON report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("JSON report saved to %s", output_path)

    # Generate and save Markdown report
    md_path = output_path.with_suffix(".md")
    md_content = _generate_markdown(report)
    with open(md_path, "w") as f:
        f.write(md_content)
    logger.info("Markdown report saved to %s", md_path)

    return report


def _generate_markdown(report: dict) -> str:
    """Generate Markdown tables from the report dict.

    Produces Table II (objective evaluation) and Table III (melody accuracy) format.

    Args:
        report: Aggregated report dict.

    Returns:
        Markdown string.
    """
    lines = []

    # Table II: Objective Evaluation
    lines.append("## Table II: Objective Evaluation Results")
    lines.append("")
    lines.append("| Metric | Seen | Unseen | Paper (Seen) | Delta |")
    lines.append("|--------|------|--------|--------------|-------|")

    for metric_name in METRIC_FILES:
        data = report.get(metric_name, {})
        seen_mean = data.get("seen", {}).get("mean")
        unseen_mean = data.get("unseen", {}).get("mean")
        paper_target = data.get("paper_target", {}).get("seen")
        delta = data.get("seen", {}).get("delta")

        seen_str = _format_value(metric_name, seen_mean)
        unseen_str = _format_value(metric_name, unseen_mean)
        paper_str = _format_value(metric_name, paper_target)
        delta_str = _format_delta(metric_name, delta)

        lines.append(
            f"| {metric_name} | {seen_str} | {unseen_str} | {paper_str} | {delta_str} |"
        )

    lines.append("")

    # Table III: Melody Accuracy (F0-RMSE and SVT-F1 focused)
    lines.append("## Table III: Melody Accuracy Comparison")
    lines.append("")
    lines.append("| Metric | Seen | Unseen | Paper (Seen) |")
    lines.append("|--------|------|--------|--------------|")

    melody_metrics = ["F0-RMSE", "SVT-F1"]
    for metric_name in melody_metrics:
        data = report.get(metric_name, {})
        seen_mean = data.get("seen", {}).get("mean")
        unseen_mean = data.get("unseen", {}).get("mean")
        paper_target = data.get("paper_target", {}).get("seen")

        seen_str = _format_value(metric_name, seen_mean)
        unseen_str = _format_value(metric_name, unseen_mean)
        paper_str = _format_value(metric_name, paper_target)

        lines.append(
            f"| {metric_name} | {seen_str} | {unseen_str} | {paper_str} |"
        )

    lines.append("")

    # Summary with std
    lines.append("## Detailed Results (Mean +/- Std)")
    lines.append("")
    lines.append("| Metric | Seen | Unseen |")
    lines.append("|--------|------|--------|")

    for metric_name in METRIC_FILES:
        data = report.get(metric_name, {})
        seen_mean = data.get("seen", {}).get("mean")
        seen_std = data.get("seen", {}).get("std")
        unseen_mean = data.get("unseen", {}).get("mean")
        unseen_std = data.get("unseen", {}).get("std")

        seen_str = _format_mean_std(metric_name, seen_mean, seen_std)
        unseen_str = _format_mean_std(metric_name, unseen_mean, unseen_std)

        lines.append(f"| {metric_name} | {seen_str} | {unseen_str} |")

    lines.append("")
    return "\n".join(lines)


def _format_value(metric_name: str, value) -> str:
    """Format a metric value for display."""
    if value is None:
        return "N/A"
    if metric_name == "MCD":
        return f"{value:.2f}"
    elif metric_name == "F0-RMSE":
        return f"{value:.4f}"
    elif metric_name == "SingMOS":
        return f"{value:.2f}"
    elif metric_name == "SECS":
        return f"{value:.3f}"
    elif metric_name == "SVT-F1":
        return f"{value:.3f}"
    return f"{value:.4f}"


def _format_delta(metric_name: str, delta) -> str:
    """Format a delta value (difference from paper target)."""
    if delta is None:
        return "N/A"
    sign = "+" if delta >= 0 else ""
    if metric_name == "MCD":
        return f"{sign}{delta:.2f}"
    elif metric_name == "F0-RMSE":
        return f"{sign}{delta:.4f}"
    elif metric_name == "SingMOS":
        return f"{sign}{delta:.2f}"
    elif metric_name == "SECS":
        return f"{sign}{delta:.3f}"
    elif metric_name == "SVT-F1":
        return f"{sign}{delta:.3f}"
    return f"{sign}{delta:.4f}"


def _format_mean_std(metric_name: str, mean, std) -> str:
    """Format mean +/- std for display."""
    if mean is None:
        return "N/A"
    mean_str = _format_value(metric_name, mean)
    if std is not None:
        std_str = _format_value(metric_name, std)
        return f"{mean_str} +/- {std_str}"
    return mean_str


def main():
    parser = argparse.ArgumentParser(
        description="Generate final evaluation report (Table II/III format)"
    )
    parser.add_argument(
        "--eval_dir", type=str, required=True,
        help="Directory containing metric JSON files",
    )
    parser.add_argument(
        "--output", type=str, default="results/eval_report.json",
        help="Output JSON path (Markdown saved alongside with .md extension)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    report = generate_report(args.eval_dir, args.output)

    # Print summary to stdout
    print("\n=== Evaluation Report Summary ===")
    for metric_name in METRIC_FILES:
        data = report.get(metric_name, {})
        seen = data.get("seen", {}).get("mean")
        paper = data.get("paper_target", {}).get("seen")
        if seen is not None and paper is not None:
            delta = seen - paper
            print(f"  {metric_name}: {seen:.4f} (paper: {paper:.4f}, delta: {delta:+.4f})")
        elif seen is not None:
            print(f"  {metric_name}: {seen:.4f}")
        else:
            print(f"  {metric_name}: N/A")


if __name__ == "__main__":
    main()
