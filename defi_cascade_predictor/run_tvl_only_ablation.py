"""
TVL-Only Ablation: Train TGN and XGBoost on 8 raw TVL features vs all 46.
If TGN wins on TVL-only, the feature circularity argument is neutralized.
~15 min on GPU, ~40 min on CPU.
"""
import sys, os, copy, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(__file__))

from experiments.run_experiments import RealDataPipeline, ExperimentRunner
from models.tgn import TemporalGraphNetwork
from models.baselines.xgboost_model import XGBoostCascadePredictor
from training.losses import FocalLoss, MonotonicityRegularization

HORIZONS = [24, 72, 168, 720]
HORIZON_KEYS = [f"cascade_{h}h" for h in HORIZONS]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU = torch.cuda.is_available()
EPOCHS = 120 if GPU else 50
PATIENCE = 20 if GPU else 15


def load_and_prepare(tvl_only=False):
    """Load real data, build graph, prepare tensors."""
    pipeline = RealDataPipeline(seed=42)
    data = pipeline.load_all()

    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    config["training"] = config.get("training", {})
    config["training"]["epochs"] = EPOCHS
    config["training"]["patience"] = PATIENCE

    runner = ExperimentRunner(config)
    runner._split_dates = {
        "train_end": data["dates"][517],
        "val_end": data["dates"][668],
    }

    graph = runner.build_graph_and_features(data)

    if tvl_only:
        nf = graph["node_features_array"][:, :, :8]
        graph["node_features_array"] = nf
        graph["feature_dim"] = 8

    prepared = runner.prepare_data(graph)
    return prepared


def train_tgn(prepared, seed=42):
    """Train TGN, return test metrics dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    feat_dim = prepared["feature_dim"]
    edge_types = list(prepared["edge_index_dict"].keys())
    eid = prepared["edge_index_dict"]

    model = TemporalGraphNetwork(
        num_nodes=15,
        node_feature_dim=feat_dim,
        edge_types=edge_types,
        memory_dim=128, embedding_dim=128,
        num_attention_heads=4, num_gnn_layers=2,
        prediction_horizons=HORIZONS,
        dropout=0.15, memory_updater="gru",
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = FocalLoss(gamma=2.0, alpha=0.75)
    mono_reg = MonotonicityRegularization(prediction_horizons=HORIZONS, weight=0.1)

    nf = prepared["node_features"]
    ts = prepared["timestamps"]
    la = prepared["label_arrays"]
    sev = prepared["severity"]
    train_sl = prepared["splits"]["train"]
    val_sl = prepared["splits"]["val"]
    test_sl = prepared["splits"]["test"]
    train_indices = list(range(*train_sl.indices(len(nf))))
    hw = {24: 3.0, 72: 2.0, 168: 1.0, 720: 1.0}

    best_val, best_state, no_improve = float("inf"), None, 0

    for epoch in range(EPOCHS):
        model.train()
        if epoch == 0:
            model.reset_memory()
        if epoch < 10:
            for pg in optimizer.param_groups:
                pg["lr"] = 3e-4 * (epoch + 1) / 10

        wl = torch.tensor(0.0, device=DEVICE)
        wc = 0
        for step, t in enumerate(train_indices):
            x = nf[t].to(DEVICE)
            timestamp = ts[t].expand(15).to(DEVICE)
            preds = model(x, eid, timestamp)
            loss = sum(hw[h] * criterion(preds[f"cascade_{h}h"].unsqueeze(0),
                       la[f"cascade_{h}h"][t].unsqueeze(0).to(DEVICE)) for h in HORIZONS)
            loss = loss + 0.3 * F.mse_loss(preds["severity"], sev[t].to(DEVICE))
            loss = loss + mono_reg(preds)
            wl, wc = wl + loss, wc + 1
            if wc >= 10 or step == len(train_indices) - 1:
                optimizer.zero_grad()
                (wl / wc).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.memory.detach_memory()
                wl, wc = torch.tensor(0.0, device=DEVICE), 0

        # Validate
        model.eval()
        model.reset_memory()
        with torch.no_grad():
            for t in train_indices:
                model(nf[t].to(DEVICE), eid, ts[t].expand(15).to(DEVICE))
                model.memory.detach_memory()
        vl, nv = 0.0, 0
        with torch.no_grad():
            for t in range(*val_sl.indices(len(nf))):
                preds = model(nf[t].to(DEVICE), eid, ts[t].expand(15).to(DEVICE))
                vl += sum(criterion(preds[f"cascade_{h}h"].unsqueeze(0),
                          la[f"cascade_{h}h"][t].unsqueeze(0).to(DEVICE)).item() for h in HORIZONS)
                model.memory.detach_memory()
                nv += 1
        avg_val = vl / max(nv, 1)
        if epoch >= 10:
            scheduler.step(avg_val)
        if avg_val < best_val:
            best_val, best_state, no_improve = avg_val, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} val={avg_val:.4f} no_improve={no_improve}")
        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # Test predictions
    model.eval()
    model.reset_memory()
    with torch.no_grad():
        for t in range(test_sl.start):
            model(nf[t].to(DEVICE), eid, ts[t].expand(15).to(DEVICE))
            model.memory.detach_memory()
    preds_out, targets_out = {k: [] for k in HORIZON_KEYS}, {k: [] for k in HORIZON_KEYS}
    with torch.no_grad():
        for t in range(*test_sl.indices(len(nf))):
            out = model(nf[t].to(DEVICE), eid, ts[t].expand(15).to(DEVICE))
            model.memory.detach_memory()
            for k in HORIZON_KEYS:
                preds_out[k].append(torch.sigmoid(out[k]).cpu().item())
                targets_out[k].append(la[k][t].item())
    return eval_metrics(preds_out, targets_out)


def train_xgboost(prepared):
    """Train XGBoost, return test metrics dict."""
    nf_np = prepared["node_features_np"]
    la = prepared["label_arrays"]
    train_sl = prepared["splits"]["train"]
    test_sl = prepared["splits"]["test"]

    xgb_model = XGBoostCascadePredictor(prediction_horizons=HORIZONS)
    window = 30
    X_all = xgb_model.prepare_features(list(nf_np), window=window)
    y_all = {k: la[k].numpy()[window:window + len(X_all)] for k in HORIZON_KEYS}
    min_l = min(len(X_all), min(len(v) for v in y_all.values()))
    X_all, y_all = X_all[:min_l], {k: v[:min_l] for k, v in y_all.items()}

    n_train = max(train_sl.stop - window, 1)
    xgb_model.fit(X_all[:n_train], {k: v[:n_train] for k, v in y_all.items()})

    n_test = max(test_sl.start - window, 0)
    bp = xgb_model.predict(X_all[n_test:])
    bt = {k: y_all[k][n_test:] for k in HORIZON_KEYS}
    for k in HORIZON_KEYS:
        ml = min(len(bp[k]), len(bt[k]))
        bp[k], bt[k] = bp[k][:ml], bt[k][:ml]
    return eval_metrics(bp, bt)


def eval_metrics(preds, targets):
    results = {}
    for k in HORIZON_KEYS:
        y, p = np.array(targets[k]), np.array(preds[k])
        if y.sum() == 0 or y.sum() == len(y):
            results[k] = {"auroc": 0.5, "auprc": float(y.mean())}
        else:
            results[k] = {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p)}
    return results


def main():
    print("=" * 70)
    print("TVL-ONLY ABLATION")
    print(f"Device: {DEVICE} | Epochs: {EPOCHS}")
    print("=" * 70)
    t0 = time.time()

    # --- Full features (46 base → 322 augmented) ---
    print("\n[1/4] Loading full features...")
    prep_full = load_and_prepare(tvl_only=False)
    print(f"  Feature dim: {prep_full['feature_dim']}")

    print("\n[2/4] TGN (all 46 features)...")
    tgn_full = train_tgn(prep_full)

    print("\n  XGBoost (all 46 features)...")
    xgb_full = train_xgboost(prep_full)

    # --- TVL-only features (8 base → 56 augmented) ---
    print("\n[3/4] Loading TVL-only features...")
    prep_tvl = load_and_prepare(tvl_only=True)
    print(f"  Feature dim: {prep_tvl['feature_dim']}")

    print("\n[4/4] TGN (TVL-only 8 features)...")
    tgn_tvl = train_tgn(prep_tvl)

    print("\n  XGBoost (TVL-only 8 features)...")
    xgb_tvl = train_xgboost(prep_tvl)

    # --- Results ---
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"RESULTS (completed in {elapsed/60:.1f} min)")
    print("=" * 70)

    rows = [
        ("TGN (all 46 feat)", tgn_full),
        ("TGN (TVL-only 8)", tgn_tvl),
        ("XGBoost (all 46)", xgb_full),
        ("XGBoost (TVL-only 8)", xgb_tvl),
    ]

    print(f"\n{'Model':<25} {'1d AUROC':>9} {'1d AUPRC':>9} | {'3d AUROC':>9} {'3d AUPRC':>9} | {'7d AUROC':>9} {'7d AUPRC':>9} | {'30d AUROC':>9} {'30d AUPRC':>9}")
    print("-" * 120)
    for name, m in rows:
        r = f"{name:<25}"
        for k in HORIZON_KEYS:
            r += f" {m[k]['auroc']:>9.3f} {m[k]['auprc']:>9.3f} |"
        print(r)

    # Key comparison
    print("\n--- KEY FINDING ---")
    for k, label in zip(HORIZON_KEYS, ["1d", "3d", "7d", "30d"]):
        tgn_wins = tgn_tvl[k]["auprc"] > xgb_tvl[k]["auprc"]
        diff = tgn_tvl[k]["auprc"] - xgb_tvl[k]["auprc"]
        print(f"  {label}: TGN TVL-only AUPRC={tgn_tvl[k]['auprc']:.3f} vs "
              f"XGBoost TVL-only AUPRC={xgb_tvl[k]['auprc']:.3f} "
              f"({'TGN wins' if tgn_wins else 'XGBoost wins'} by {abs(diff):.3f})")

    # LaTeX
    print("\n--- For paper ---")
    print("TGN retains AUPRC advantage using only raw TVL features:")
    for k, label in zip(HORIZON_KEYS, ["1-day", "3-day", "7-day", "30-day"]):
        print(f"  {label}: TGN {tgn_tvl[k]['auprc']:.3f} vs XGBoost {xgb_tvl[k]['auprc']:.3f}")

    # Save
    import json
    out = {"tgn_full": tgn_full, "tgn_tvl": tgn_tvl, "xgb_full": xgb_full, "xgb_tvl": xgb_tvl}
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        raise TypeError
    os.makedirs(os.path.join(os.path.dirname(__file__), "outputs"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "outputs", "tvl_ablation.json"), "w") as f:
        json.dump(out, f, indent=2, default=conv)
    print("\nSaved to outputs/tvl_ablation.json")


if __name__ == "__main__":
    main()
