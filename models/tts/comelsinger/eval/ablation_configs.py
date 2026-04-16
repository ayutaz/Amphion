"""Generate ablation experiment configurations (Table V).

6 conditions: full / wo_cl / wo_scl / wo_fcl / wo_svt / wo_cl_svt
Each condition overrides loss weights from the base configuration.
Paper Section IV-B confirmed loss weights for full model:
    lambda_cl=0.5, lambda_scl=1.0, lambda_fcl=0.1, lambda_svt=0.5, lambda_mask=0.3

Ticket: M4-10
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Full model loss weights (paper Section IV-B)
FULL_WEIGHTS = {
    "lambda_cl": 0.5,
    "lambda_scl": 1.0,
    "lambda_fcl": 0.1,
    "lambda_svt": 0.5,
    "lambda_mask": 0.3,
}

# Ablation configurations: each overrides specific weights to 0
ABLATION_CONFIGS: dict[str, dict[str, float]] = {
    "full": {
        "lambda_cl": 0.5,
        "lambda_scl": 1.0,
        "lambda_fcl": 0.1,
        "lambda_svt": 0.5,
        "lambda_mask": 0.3,
    },
    "wo_cl": {
        "lambda_cl": 0.0,
        "lambda_scl": 0.0,
        "lambda_fcl": 0.0,
        "lambda_svt": 0.5,
        "lambda_mask": 0.3,
    },
    "wo_scl": {
        "lambda_cl": 0.5,
        "lambda_scl": 0.0,
        "lambda_fcl": 0.1,
        "lambda_svt": 0.5,
        "lambda_mask": 0.3,
    },
    "wo_fcl": {
        "lambda_cl": 0.5,
        "lambda_scl": 1.0,
        "lambda_fcl": 0.0,
        "lambda_svt": 0.5,
        "lambda_mask": 0.3,
    },
    "wo_svt": {
        "lambda_cl": 0.5,
        "lambda_scl": 1.0,
        "lambda_fcl": 0.1,
        "lambda_svt": 0.0,
        "lambda_mask": 0.3,
    },
    "wo_cl_svt": {
        "lambda_cl": 0.0,
        "lambda_scl": 0.0,
        "lambda_fcl": 0.0,
        "lambda_svt": 0.0,
        "lambda_mask": 0.3,
    },
}

# Human-readable descriptions for each condition
ABLATION_DESCRIPTIONS: dict[str, str] = {
    "full": "Full model (all components enabled)",
    "wo_cl": "Without contrastive learning (SCL + FCL disabled)",
    "wo_scl": "Without sequence-level contrastive learning",
    "wo_fcl": "Without frame-level contrastive learning",
    "wo_svt": "Without SVT pitch supervision",
    "wo_cl_svt": "Without contrastive learning and SVT (baseline)",
}


def get_config(condition: str) -> dict[str, float]:
    """Get loss weight configuration for a given ablation condition.

    Args:
        condition: One of "full", "wo_cl", "wo_scl", "wo_fcl", "wo_svt", "wo_cl_svt".

    Returns:
        dict of loss weight parameters.

    Raises:
        ValueError: If condition is not recognized.
    """
    if condition not in ABLATION_CONFIGS:
        raise ValueError(
            f"Unknown ablation condition: {condition}. "
            f"Must be one of: {list(ABLATION_CONFIGS.keys())}"
        )
    return copy.deepcopy(ABLATION_CONFIGS[condition])


def generate_ablation_yamls(
    base_config_path: str | Path,
    output_dir: str | Path,
) -> list[str]:
    """Generate per-condition YAML files by overriding loss weights.

    Reads the base YAML, overrides loss weights for each ablation condition,
    and saves to output_dir/{condition}.yaml.

    Args:
        base_config_path: Path to base configuration YAML.
        output_dir: Directory to write ablation YAML files.

    Returns:
        List of generated YAML file paths.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required to generate YAML configs. Install with: uv add pyyaml")

    base_path = Path(base_config_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(base_path) as f:
        base_config = yaml.safe_load(f)

    generated = []
    for condition, weights in ABLATION_CONFIGS.items():
        config = copy.deepcopy(base_config)

        # Ensure nested structure exists
        config.setdefault("model", {})
        config["model"].setdefault("s2a", {})
        config["model"]["s2a"].setdefault("loss", {})

        # Override loss weights
        config["model"]["s2a"]["loss"].update(weights)

        # Set condition-specific checkpoint paths
        config["model"]["s2a"]["lora_ckpt_dir"] = f"checkpoints/ablation/{condition}/lora"
        config["model"]["s2a"]["pitch_emb_path"] = f"checkpoints/ablation/{condition}/pitch_emb.pt"

        # Add metadata
        config["ablation_condition"] = condition
        config["ablation_description"] = ABLATION_DESCRIPTIONS[condition]

        out_path = out_dir / f"{condition}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        generated.append(str(out_path))
        logger.info("Generated %s", out_path)

    return generated


def generate_ablation_json(output_path: str | Path) -> None:
    """Export all ablation configs as a single JSON file for programmatic use.

    Args:
        output_path: Path to write JSON output.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "conditions": ABLATION_CONFIGS,
        "descriptions": ABLATION_DESCRIPTIONS,
        "full_weights": FULL_WEIGHTS,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Ablation configs saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate ablation study configurations (Table V)")
    parser.add_argument("--base_config", type=str, help="Base YAML config path (optional)")
    parser.add_argument("--output_dir", type=str, default="configs/ablation", help="Output directory for YAMLs")
    parser.add_argument("--json_output", type=str, default=None, help="Also export as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.base_config:
        paths = generate_ablation_yamls(args.base_config, args.output_dir)
        logger.info("Generated %d ablation YAML files.", len(paths))
    else:
        logger.info("No base config provided. Printing ablation configs:")
        for condition, weights in ABLATION_CONFIGS.items():
            logger.info("  %s: %s", condition, weights)

    if args.json_output:
        generate_ablation_json(args.json_output)


if __name__ == "__main__":
    main()
