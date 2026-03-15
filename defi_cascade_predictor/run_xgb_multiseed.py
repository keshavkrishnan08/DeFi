"""
Run XGBoost with 5 seeds to produce multi-seed comparison for the paper.
Quick experiment: ~1 minute on CPU.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# Reuse the real data pipeline
from experiments.run_experiments import RealDataPipeline

SEEDS = [42, 123, 456, 789, 1337]
HORIZONS = [24, 72, 168, 720]
HORIZON_KEYS = [f"cascade_{h}h" for h in HORIZONS]
WINDOW = 30


def prepare_xgb_features(node_features_sequence, window=30):
    """Same as XGBoostCascadePredictor.prepare_features but standalone."""
    all_features = []
    seq_len = len(node_features_sequence)
    for t in range(window, seq_len):
        features = []
        current = node_features_sequence[t].flatten()
        features.extend(current)
        current_2d = node_features_sequence[t]
        features.extend(current_2d.mean(axis=0))
        features.extend(current_2d.std(axis=0))
        features.extend(current_2d.min(axis=0))
        features.extend(current_2d.max(axis=0))
        window_data = np.array(node_features_sequence[max(0, t - window):t])
        if window_data.shape[0] > 1:
            changes = np.diff(window_data, axis=0)
            features.extend(changes.mean(axis=(0, 1)))
            features.extend(changes.std(axis=(0, 1)))
            trend = window_data[-1].mean(axis=0) - window_data[0].mean(axis=0)
            features.extend(trend)
        else:
            n_feat = current_2d.shape[1]
            features.extend(np.zeros(n_feat * 3))
        all_features.append(features)
    return np.array(all_features, dtype=np.float32)


def build_node_features(data):
    """Build [T, N, F] feature array from pipeline data, matching run_experiments.py."""
    tvl = data["tvl"]
    prices = data["prices"]
    macro = data["macro"]
    lending = data["lending"]
    dates = data["dates"]
    n_days, n_protocols = tvl.shape

    # --- TVL features (8) ---
    log_tvl = np.log1p(tvl)
    tvl_change_1d = np.zeros_like(tvl)
    tvl_change_7d = np.zeros_like(tvl)
    tvl_change_30d = np.zeros_like(tvl)
    for t in range(1, n_days):
        tvl_change_1d[t] = (tvl[t] - tvl[t-1]) / (tvl[t-1] + 1e-10)
    for t in range(7, n_days):
        tvl_change_7d[t] = (tvl[t] - tvl[t-7]) / (tvl[t-7] + 1e-10)
    for t in range(30, n_days):
        tvl_change_30d[t] = (tvl[t] - tvl[t-30]) / (tvl[t-30] + 1e-10)

    running_max = np.maximum.accumulate(tvl, axis=0)
    tvl_drawdown = (tvl - running_max) / (running_max + 1e-10)

    tvl_rank = np.zeros_like(tvl)
    for t in range(n_days):
        order = np.argsort(np.argsort(-tvl[t]))
        tvl_rank[t] = order / max(n_protocols - 1, 1)

    tvl_zscore = np.zeros_like(tvl)
    for t in range(90, n_days):
        window_data = tvl[t-90:t]
        mu = window_data.mean(axis=0)
        sigma = window_data.std(axis=0) + 1e-10
        tvl_zscore[t] = (tvl[t] - mu) / sigma

    ma30 = np.zeros_like(tvl)
    for t in range(30, n_days):
        ma30[t] = tvl[t-30:t].mean(axis=0)
    tvl_ma_ratio = tvl / (ma30 + 1e-10)
    tvl_ma_ratio[:30] = 1.0

    tvl_feats = np.stack([log_tvl, tvl_change_1d, tvl_change_7d, tvl_change_30d,
                          tvl_drawdown, tvl_rank, tvl_zscore, tvl_ma_ratio], axis=-1)

    # --- Price features (9) ---
    log_price = np.log1p(prices)
    price_ret_1d = np.zeros_like(prices)
    price_ret_7d = np.zeros_like(prices)
    price_ret_30d = np.zeros_like(prices)
    for t in range(1, n_days):
        price_ret_1d[t] = (prices[t] - prices[t-1]) / (prices[t-1] + 1e-10)
    for t in range(7, n_days):
        price_ret_7d[t] = (prices[t] - prices[t-7]) / (prices[t-7] + 1e-10)
    for t in range(30, n_days):
        price_ret_30d[t] = (prices[t] - prices[t-30]) / (prices[t-30] + 1e-10)

    vol_7d = np.zeros_like(prices)
    vol_30d = np.zeros_like(prices)
    log_returns = np.zeros_like(prices)
    for t in range(1, n_days):
        log_returns[t] = np.log((prices[t] + 1e-10) / (prices[t-1] + 1e-10))
    for t in range(7, n_days):
        vol_7d[t] = log_returns[t-7:t].std(axis=0)
    for t in range(30, n_days):
        vol_30d[t] = log_returns[t-30:t].std(axis=0)

    volume_proxy = np.log1p(prices * tvl * 0.01)
    volume_ratio = np.ones_like(prices)

    price_running_max = np.maximum.accumulate(prices, axis=0)
    price_drawdown = (prices - price_running_max) / (price_running_max + 1e-10)

    price_feats = np.stack([log_price, price_ret_1d, price_ret_7d, price_ret_30d,
                            vol_7d, vol_30d, volume_proxy, volume_ratio, price_drawdown], axis=-1)

    # --- Lending features (8) ---
    util = lending[:, :, 0] if lending.ndim == 3 else np.full((n_days, n_protocols), 0.5)
    borrow_rate = lending[:, :, 1] if lending.ndim == 3 else np.full((n_days, n_protocols), 0.05)
    supply_rate = lending[:, :, 2] if lending.ndim == 3 else np.full((n_days, n_protocols), 0.03)
    rate_spread = lending[:, :, 3] if lending.ndim == 3 else np.full((n_days, n_protocols), 0.02)

    total_supply = np.log1p(tvl * 0.6)
    total_borrow = np.log1p(tvl * util)
    borrow_supply = util.copy()
    rate_change = np.zeros_like(util)

    lending_feats = np.stack([util, borrow_rate, supply_rate, total_supply,
                              total_borrow, borrow_supply, rate_spread, rate_change], axis=-1)

    # --- Network features (6) - constant ---
    net_feats = np.zeros((n_days, n_protocols, 6))
    # Simple degree-based features (constant across time)
    degrees = np.array([8, 7, 6, 5, 7, 6, 4, 7, 5, 5, 4, 3, 5, 5, 3], dtype=float)
    degrees_norm = degrees / degrees.max()
    for t in range(n_days):
        net_feats[t, :, 0] = degrees_norm
        net_feats[t, :, 1] = degrees_norm * 0.8  # betweenness proxy
        net_feats[t, :, 2] = degrees_norm * 0.9  # eigenvector proxy
        net_feats[t, :, 3] = degrees_norm * 0.7  # clustering proxy
        net_feats[t, :, 4] = degrees_norm * 0.85  # pagerank proxy
        net_feats[t, :, 5] = degrees / 10.0  # shared collateral count

    # --- Macro features (9) - same across protocols ---
    macro_feats = np.zeros((n_days, n_protocols, 9))
    for j in range(n_protocols):
        macro_feats[:, j, :] = macro[:n_days, :9] if macro.shape[1] >= 9 else np.zeros((n_days, 9))

    # --- Temporal features (6) ---
    temp_feats = np.zeros((n_days, n_protocols, 6))
    import pandas as pd
    for t in range(n_days):
        d = dates[t]
        dow = d.dayofweek if hasattr(d, 'dayofweek') else 0
        month = d.month if hasattr(d, 'month') else 1
        temp_feats[t, :, 0] = np.sin(2 * np.pi * dow / 7)
        temp_feats[t, :, 1] = np.cos(2 * np.pi * dow / 7)
        temp_feats[t, :, 2] = np.sin(2 * np.pi * month / 12)
        temp_feats[t, :, 3] = np.cos(2 * np.pi * month / 12)
        temp_feats[t, :, 4] = min(t / 365.0, 1.0)
        temp_feats[t, :, 5] = 0.0

    # Concatenate: [T, N, 46]
    all_feats = np.concatenate([tvl_feats, price_feats, lending_feats,
                                net_feats, macro_feats, temp_feats], axis=-1)

    # Normalize using training stats only
    train_end = 518  # ~2022-10-31
    train_data = all_feats[:train_end].reshape(-1, all_feats.shape[-1])
    mu = train_data.mean(axis=0)
    sigma = train_data.std(axis=0) + 1e-10
    all_feats = (all_feats - mu) / sigma
    all_feats = np.clip(all_feats, -10, 10)

    return all_feats


def main():
    import xgboost as xgb

    print("=" * 60)
    print("XGBoost Multi-Seed Experiment (5 seeds)")
    print("=" * 60)

    # Load real data
    print("Loading real data...")
    pipeline = RealDataPipeline(seed=42)
    data = pipeline.load_all()

    # Build features
    print("Building features...")
    node_features = build_node_features(data)
    T, N, F = node_features.shape
    print(f"  Feature matrix: ({T}, {N}, {F})")

    # Prepare XGBoost tabular features
    nf_list = [node_features[t] for t in range(T)]
    X_all = prepare_xgb_features(nf_list, window=WINDOW)
    print(f"  XGBoost features: {X_all.shape}")

    # Prepare labels
    labels = data["labels"]
    y_all = {}
    for key in HORIZON_KEYS:
        y = labels[key].values if hasattr(labels[key], 'values') else np.array(labels[key])
        y_all[key] = y[WINDOW:WINDOW + len(X_all)].astype(float)

    # Date-based splits matching the paper
    train_end_idx = 518 - WINDOW  # ~2022-10-31
    test_start_idx = 669 - WINDOW  # ~2023-04-01

    X_train = X_all[:train_end_idx]
    X_test = X_all[test_start_idx:]
    y_train = {k: v[:train_end_idx] for k, v in y_all.items()}
    y_test = {k: v[test_start_idx:] for k, v in y_all.items()}

    min_test = min(len(X_test), min(len(v) for v in y_test.values()))
    X_test = X_test[:min_test]
    y_test = {k: v[:min_test] for k, v in y_test.items()}

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    for k in HORIZON_KEYS:
        print(f"  {k} test positive rate: {y_test[k].mean():.4f}")

    # Run 5 seeds
    results = {seed: {} for seed in SEEDS}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        for hkey in HORIZON_KEYS:
            y_tr = y_train[hkey]
            y_te = y_test[hkey]

            pos_count = y_tr.sum()
            neg_count = len(y_tr) - pos_count
            scale_pos = neg_count / max(pos_count, 1)

            model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="auc",
                random_state=seed,
                n_jobs=1,
                scale_pos_weight=scale_pos,
                verbosity=0,
            )

            model.fit(X_train, y_tr, verbose=False)
            y_prob = model.predict_proba(X_test)[:, 1]

            auroc = roc_auc_score(y_te, y_prob)
            auprc = average_precision_score(y_te, y_prob)

            results[seed][hkey] = {"auroc": auroc, "auprc": auprc}
            print(f"  {hkey}: AUROC={auroc:.3f}, AUPRC={auprc:.3f}")

    # Summary table
    print("\n" + "=" * 80)
    print("MULTI-SEED XGBOOST RESULTS")
    print("=" * 80)

    header = f"{'Seed':>6}"
    for h in HORIZONS:
        header += f" | {'AUROC_'+str(h):>10} {'AUPRC_'+str(h):>10}"
    print(header)
    print("-" * len(header))

    all_aurocs = {k: [] for k in HORIZON_KEYS}
    all_auprcs = {k: [] for k in HORIZON_KEYS}

    for seed in SEEDS:
        row = f"{seed:>6}"
        for hkey in HORIZON_KEYS:
            r = results[seed][hkey]
            row += f" | {r['auroc']:>10.3f} {r['auprc']:>10.3f}"
            all_aurocs[hkey].append(r["auroc"])
            all_auprcs[hkey].append(r["auprc"])
        print(row)

    print("-" * len(header))
    row = f"{'Mean':>6}"
    for hkey in HORIZON_KEYS:
        mu_auroc = np.mean(all_aurocs[hkey])
        mu_auprc = np.mean(all_auprcs[hkey])
        row += f" | {mu_auroc:>10.3f} {mu_auprc:>10.3f}"
    print(row)

    row = f"{'Std':>6}"
    for hkey in HORIZON_KEYS:
        std_auroc = np.std(all_aurocs[hkey])
        std_auprc = np.std(all_auprcs[hkey])
        row += f" | {std_auroc:>10.3f} {std_auprc:>10.3f}"
    print(row)

    print("\n--- LaTeX-ready for paper ---")
    for hkey in HORIZON_KEYS:
        mu_a = np.mean(all_aurocs[hkey])
        std_a = np.std(all_aurocs[hkey])
        mu_p = np.mean(all_auprcs[hkey])
        std_p = np.std(all_auprcs[hkey])
        print(f"{hkey}: AUROC={mu_a:.3f}±{std_a:.3f}, AUPRC={mu_p:.3f}±{std_p:.3f}")


if __name__ == "__main__":
    main()
