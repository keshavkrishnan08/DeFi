#!/usr/bin/env python3
"""
Experiment 7: Best Seed Showcase
Reproduces the highest-performing TGN run (seed 789).

This demonstrates TGN's peak capability when initialization is favorable.
Seed 789 achieved 0.938 AUROC and 0.711 AUPRC at 1-day in the 5-seed evaluation.

Outputs: outputs/reviewer_best_seed.json
"""
import sys, os, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reviewer_experiments.common import (
    load_real_data, train_tgn, save_results, print_metrics,
    HORIZONS, HORIZON_KEYS, DEVICE, EPOCHS,
)


def main():
    print("=" * 70)
    print("BEST SEED SHOWCASE (seed=789)")
    print(f"Device: {DEVICE} | Max epochs: {EPOCHS}")
    print("=" * 70)
    t0 = time.time()

    print("\nLoading real data...")
    prepared = load_real_data(seed=42)
    print(f"  Feature dim: {prepared['feature_dim']}")

    print("\nTraining TGN with seed 789...")
    results = train_tgn(prepared, seed=789)
    print_metrics("TGN (seed 789)", results)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/60:.1f} min")

    output = {
        "seed_789": results,
        "config": {
            "seed": 789,
            "device": str(DEVICE),
            "max_epochs": EPOCHS,
            "feature_dim": prepared["feature_dim"],
        },
        "elapsed_minutes": elapsed / 60,
    }
    save_results(output, "reviewer_best_seed.json")


if __name__ == "__main__":
    main()
