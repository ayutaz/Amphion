"""Aggregate ablation results into Table V format.

Reads per-condition JSON results from M4-05 to M4-09 and generates
CSV + Markdown summary tables matching paper Table V layout.
6 conditions x 5 metrics (MCD, F0-RMSE, SingMOS, SECS, SVT-F1).

Ticket: M4-11
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CONDITIONS = ["full", "wo_cl", "wo_scl", "wo_fcl", "wo_svt", "wo_cl_svt"]

# Metric name -> (JSON filename pattern, JSON key for mean value)
METRICS = {
    "MCD": ("mcd", "mean_mcd"),
    "F0-RMSE": ("f0_rmse", "mean_f0_rmse"),
    "SingMOS": ("singmos", "mean_singmos"),
    "SECS": ("secs", "mean_secs"),
    "SVT-F1": ("svt_f1", "mean_f1"),
}

# Paper target values (Seen set)
PAPER_TARGETS = {
    "MCD": 4.17,
    "F0-RMSE": 0.042,
    "SingMOS": 4.32,
    "SECS": 0.912,
    "SVT-F1": 0.711,
}


def load_results(results_dir: str | Path, split: str = "seen") -> list[dict]:
    """Load evaluation results for all conditions and metrics.

    Expected directory structure:
        results_dir/
            {condition}/
                {metric}_{split}.json

    Args:
        results_dir: Root directory containing per-condition result subdirs.
        split: "seen" or "unseen".

    Returns:
        List of dicts, one per condition, with metric values.
    """
    results_dir = Path(results_dir)
    rows = []

    for cond in CONDITIONS:
        row = {"condition": cond}
        for metric_name, (metric_file, metric_key) in METRICS.items():
            json_path = results_dir / cond / f"{metric_file}_{split}.json"
            try:
                with open(json_path) as f:
                    data = json.load(f)
                row[metric_name] = data.get(metric_key)
            except FileNotFoundError:
                logger.warning("Missing: %s", json_path)
                row[metric_name] = None
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Error reading %s: %s", json_path, e)
                row[metric_name] = None
        rows.append(row)

    return rows


def generate_summary(results_dir: str | Path, split: str = "seen") -> str:
    """Read per-condition JSON results and generate Markdown table.

    Returns Table V format markdown string:
    | Condition | MCD | F0-RMSE | SingMOS | SECS | SVT-F1 |

    Args:
        results_dir: Root directory with per-condition results.
        split: "seen" or "unseen".

    Returns:
        Markdown table string.
    """
    rows = load_results(results_dir, split)
    metric_names = list(METRICS.keys())

    # Build header
    lines = []
    lines.append(f"## Ablation Study Results ({split.capitalize()} Set) - Table V")
    lines.append("")
    header = "| Condition | " + " | ".join(metric_names) + " |"
    separator = "|" + "|".join(["---"] * (len(metric_names) + 1)) + "|"
    lines.append(header)
    lines.append(separator)

    # Data rows
    for row in rows:
        cells = [row["condition"]]
        for m in metric_names:
            val = row.get(m)
            if val is None:
                cells.append("N/A")
            elif isinstance(val, float):
                # Format based on metric scale
                if m in ("MCD",):
                    cells.append(f"{val:.2f}")
                elif m in ("F0-RMSE",):
                    cells.append(f"{val:.4f}")
                elif m in ("SingMOS",):
                    cells.append(f"{val:.2f}")
                elif m in ("SECS",):
                    cells.append(f"{val:.3f}")
                elif m in ("SVT-F1",):
                    cells.append(f"{val:.3f}")
                else:
                    cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    # Paper target row
    target_cells = ["Paper (full)"]
    for m in metric_names:
        target = PAPER_TARGETS.get(m)
        if target is not None:
            if m in ("MCD",):
                target_cells.append(f"{target:.2f}")
            elif m in ("F0-RMSE",):
                target_cells.append(f"{target:.4f}")
            elif m in ("SingMOS",):
                target_cells.append(f"{target:.2f}")
            elif m in ("SECS",):
                target_cells.append(f"{target:.3f}")
            elif m in ("SVT-F1",):
                target_cells.append(f"{target:.3f}")
            else:
                target_cells.append(str(target))
        else:
            target_cells.append("N/A")
    lines.append("| " + " | ".join(target_cells) + " |")

    return "\n".join(lines)


def save_summary(
    results_dir: str | Path,
    output_dir: str | Path,
    split: str = "seen",
) -> None:
    """Save ablation summary as CSV and Markdown.

    Args:
        results_dir: Root directory with per-condition results.
        output_dir: Directory to write output files.
        split: "seen" or "unseen".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_results(results_dir, split)
    metric_names = list(METRICS.keys())

    # Save CSV
    csv_path = output_dir / f"ablation_table_v_{split}.csv"
    header = "condition," + ",".join(metric_names)
    csv_lines = [header]
    for row in rows:
        cells = [row["condition"]]
        for m in metric_names:
            val = row.get(m)
            cells.append(str(val) if val is not None else "")
        csv_lines.append(",".join(cells))
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    logger.info("CSV saved to %s", csv_path)

    # Save Markdown
    md_path = output_dir / f"ablation_table_v_{split}.md"
    md_content = generate_summary(results_dir, split)
    with open(md_path, "w") as f:
        f.write(md_content + "\n")
    logger.info("Markdown saved to %s", md_path)

    # Save JSON
    json_path = output_dir / f"ablation_table_v_{split}.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    logger.info("JSON saved to %s", json_path)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate ablation results into Table V format"
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Root directory with per-condition result subdirs",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Output directory for summary files",
    )
    parser.add_argument(
        "--split", type=str, default="seen", choices=["seen", "unseen"],
        help="Evaluation split",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    save_summary(args.results_dir, args.output_dir, args.split)

    # Print markdown to stdout
    md = generate_summary(args.results_dir, args.split)
    print(md)


if __name__ == "__main__":
    main()
