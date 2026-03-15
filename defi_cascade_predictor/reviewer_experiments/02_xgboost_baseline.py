#!/usr/bin/env python3
"""
Experiment 2: XGBoost Baseline
Reproduces Table III, XGBoost row.

Trains XGBoost on the same features and splits as TGN.
Also runs multi-seed validation to confirm low variance.
Outputs: outputs/reviewer_xgboost.json
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reviewer_experiments.common import (
    load_real_data, compute_metrics, save_results, print_metrics,
    HORIZONS, HORIZON_KEYS,
)
from models.baselines.xgboost_model import XGBoostCascadePredictor

SEEDS = [42, 123, 456, 789, 1337]


def train_xgboost(prepared, seed=42):
    """Train XGBoost and return test metrics."""
    np.random.seed(seed)
    nf_np = prepared["node_features_np"]
    la = prepared["label_arrays"]
    train_sl = prepared["splits"]["train"]
    test_sl = prepared["splits"]["test"]

    xgb_model = XGBoostCascadePredictor(prediction_horizons=HORIZONS)
    window = 30
    X_all = xgb_model.prepare_features(list(nf_np), window=window)
    y_all = {k: la[k].numpy()[window : window + len(X_all)] for k in HORIZON_KEYS}
    min_l = min(len(X_all), min(len(v) for v in y_all.values()))
    X_all = X_all[:min_l]
    y_all = {k: v[:min_l] for k, v in y_all.items()}

    n_train = max(train_sl.stop - window, 1)
    xgb_model.fit(X_all[:n_train], {k: v[:n_train] for k, v in y_all.items()})

    n_test = max(test_sl.start - window, 0)
    preds = xgb_model.predict(X_all[n_test:])
    targets = {k: y_all[k][n_test:] for k in HORIZON_KEYS}
    for k in HORIZON_KEYS:
        ml = min(len(preds[k]), len(targets[k]))
        preds[k] = preds[k][:ml]
        targets[k] = targets[k][:ml]

    return compute_metrics(preds, targets)


def main():
    print("=" * 70)
    print("XGBOOST BASELINE")
    print(f"Device: CPU (XGBoost is CPU-only)")
    print("=" * 70)
    t0 = time.time()

    print("\nLoading real data...")
    prepared = load_real_data(seed=42)
    print(f"  Feature dim: {prepared['feature_dim']}")

    # Primary run
    print("\nTraining XGBoost (seed 42)...")
    primary = train_xgboost(prepared, seed=42)
    print_metrics("XGBoost (primary)", primary)

    # Multi-seed validation
    print("\nMulti-seed validation...")
    seed_results = {}
    for seed in SEEDS:
        metrics = train_xgboost(prepared, seed=seed)
        seed_results[str(seed)] = metrics

    # Variance analysis
    for k, label in zip(HORIZON_KEYS, ["1d", "3d", "7d", "30d"]):
        aurocs = [seed_results[s][k]["auroc"] for s in seed_results]
        auprcs = [seed_results[s][k]["auprc"] for s in seed_results]
        print(f"  {label}: AUROC std={np.std(aurocs, ddof=1):.4f}, AUPRC std={np.std(auprcs, ddof=1):.4f}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/60:.1f} min")

    results = {
        "primary": primary,
        "multiseed": seed_results,
        "elapsed_minutes": elapsed / 60,
    }
    save_results(results, "reviewer_xgboost.json")


if __name__ == "__main__":
    main()
