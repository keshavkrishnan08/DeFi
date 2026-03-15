#!/usr/bin/env python3
"""
Experiment 4: TVL-Only Ablation
Addresses feature circularity concern.

Trains TGN and XGBoost on only 8 raw TVL features vs all 46.
If TGN retains its advantage on TVL-only, the result isn't driven by
derived features that might leak cascade information.

Outputs: outputs/reviewer_tvl_ablation.json
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reviewer_experiments.common import (
    load_real_data, train_tgn, compute_metrics, save_results, print_metrics,
    HORIZONS, HORIZON_KEYS,
)
from models.baselines.xgboost_model import XGBoostCascadePredictor


def train_xgboost(prepared):
    """Train XGBoost on prepared data."""
    nf_np = prepared["node_features_np"]
    la = prepared["label_arrays"]
    train_sl = prepared["splits"]["train"]
    test_sl = prepared["splits"]["test"]

    xgb = XGBoostCascadePredictor(prediction_horizons=HORIZONS)
    window = 30
    X_all = xgb.prepare_features(list(nf_np), window=window)
    y_all = {k: la[k].numpy()[window : window + len(X_all)] for k in HORIZON_KEYS}
    min_l = min(len(X_all), min(len(v) for v in y_all.values()))
    X_all, y_all = X_all[:min_l], {k: v[:min_l] for k, v in y_all.items()}

    n_train = max(train_sl.stop - window, 1)
    xgb.fit(X_all[:n_train], {k: v[:n_train] for k, v in y_all.items()})

    n_test = max(test_sl.start - window, 0)
    preds = xgb.predict(X_all[n_test:])
    targets = {k: y_all[k][n_test:] for k in HORIZON_KEYS}
    for k in HORIZON_KEYS:
        ml = min(len(preds[k]), len(targets[k]))
        preds[k], targets[k] = preds[k][:ml], targets[k][:ml]
    return compute_metrics(preds, targets)


def main():
    print("=" * 70)
    print("TVL-ONLY ABLATION")
    print("=" * 70)
    t0 = time.time()

    results = {}

    # Full features (46 base -> 322 augmented)
    print("\n[1/4] Loading full features...")
    prep_full = load_real_data(tvl_only=False)
    print(f"  Feature dim: {prep_full['feature_dim']}")

    print("\n[2/4] TGN (all 46 features)...")
    results["tgn_full"] = train_tgn(prep_full)
    print_metrics("TGN (46 features)", results["tgn_full"])

    print("\n  XGBoost (all 46 features)...")
    results["xgb_full"] = train_xgboost(prep_full)
    print_metrics("XGBoost (46 features)", results["xgb_full"])

    # TVL-only (8 base -> 56 augmented)
    print("\n[3/4] Loading TVL-only features...")
    prep_tvl = load_real_data(tvl_only=True)
    print(f"  Feature dim: {prep_tvl['feature_dim']}")

    print("\n[4/4] TGN (TVL-only 8 features)...")
    results["tgn_tvl"] = train_tgn(prep_tvl)
    print_metrics("TGN (TVL-only)", results["tgn_tvl"])

    print("\n  XGBoost (TVL-only 8 features)...")
    results["xgb_tvl"] = train_xgboost(prep_tvl)
    print_metrics("XGBoost (TVL-only)", results["xgb_tvl"])

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Completed in {elapsed/60:.1f} min")
    print(f"\nKEY FINDING:")
    for k, label in zip(HORIZON_KEYS, ["1d", "3d", "7d", "30d"]):
        tgn_a = results["tgn_tvl"][k]["auprc"]
        xgb_a = results["xgb_tvl"][k]["auprc"]
        winner = "TGN" if tgn_a > xgb_a else "XGBoost"
        print(f"  {label}: TGN TVL-only={tgn_a:.3f} vs XGBoost TVL-only={xgb_a:.3f} ({winner} wins)")

    results["elapsed_minutes"] = elapsed / 60
    save_results(results, "reviewer_tvl_ablation.json")


if __name__ == "__main__":
    main()
