#!/usr/bin/env python3
"""
DeFi Liquidation Cascade Predictor — Main Entry Point

Predicting DeFi Liquidation Cascades Using Temporal Graph Neural
Networks on Cross-Protocol Composability Graphs

Target: IEEE Transactions on Computational Social Systems (TCSS)

Usage:
    python main.py                          # Full pipeline
    python main.py --quick                  # Quick run (reduced epochs)
    python main.py --device cpu             # Force CPU
    python main.py --config custom.yaml     # Custom config
"""

import os
# Fix PyTorch + XGBoost OpenMP conflict on macOS (must be set before imports)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <7}</level> | "
        "<level>{message}</level>"
    ),
    level="INFO",
)
os.makedirs("outputs", exist_ok=True)
logger.add("outputs/experiment.log", rotation="10 MB", level="DEBUG")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="DeFi Liquidation Cascade Predictor (TGN)"
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    parser.add_argument("--quick", action="store_true", help="Quick test run")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Find config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = PROJECT_ROOT / "config" / "config.yaml"

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    # Overrides
    if args.device:
        config["project"]["device"] = args.device
    if args.seed:
        config["project"]["seed"] = args.seed
    if args.quick:
        config["training"]["epochs"] = 20
        config["training"]["patience"] = 8
        config["training"]["baseline_epochs"] = 15
        config["training"]["ablation_epochs"] = 5
        config["model"]["sir"]["n_simulations"] = 20
        config["model"]["xgboost"] = config.get("model", {}).get("xgboost", {})
        config["model"]["xgboost"]["n_estimators"] = 100
        config["model"]["xgboost"]["max_depth"] = 6
        config["evaluation"]["statistical_tests"]["bootstrap_iterations"] = 500
        logger.info("QUICK MODE: reduced epochs & simulations")

    # Always apply CPU-reasonable ceilings when running on CPU
    if config["project"]["device"] == "cpu":
        t = config["training"]
        t.setdefault("epochs", 200)
        t["epochs"] = min(t["epochs"], 100)
        t.setdefault("patience", 25)
        t["patience"] = min(t["patience"], 20)
        t.setdefault("baseline_epochs", 80)
        t["baseline_epochs"] = min(t["baseline_epochs"], 50)
        t.setdefault("ablation_epochs", 30)
        t["ablation_epochs"] = min(t["ablation_epochs"], 20)
        e = config["evaluation"]["statistical_tests"]
        e["bootstrap_iterations"] = min(e.get("bootstrap_iterations", 10000), 3000)
        logger.info("CPU MODE: capped epochs for reasonable runtime")

    logger.info("=" * 60)
    logger.info("DeFi Liquidation Cascade Predictor")
    logger.info("Temporal GNN on Cross-Protocol Composability Graphs")
    logger.info("Target: IEEE Trans. Computational Social Systems")
    logger.info("=" * 60)

    from experiments.run_experiments import ExperimentRunner
    runner = ExperimentRunner(config)
    results = runner.run_full_pipeline()

    # Print summary
    if "model_comparison" in results:
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)
        for model_name, metrics in results["model_comparison"].items():
            aurocs = []
            for hk, hm in metrics.items():
                if isinstance(hm, dict) and "auroc" in hm:
                    aurocs.append(f"{hk.replace('cascade_', '')}={hm['auroc']:.3f}")
            if aurocs:
                logger.info(f"  {model_name:15s} AUROC: {', '.join(aurocs)}")

    logger.info("Done. Check outputs/ for figures, results, and logs.")


if __name__ == "__main__":
    main()
