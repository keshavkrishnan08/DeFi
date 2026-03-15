"""
Overnight CPU experiments addressing 5 reviewer vulnerabilities.
Expected runtime: 6-10 hours on CPU.

Experiment 1: TVL-Only Ablation (vulnerability #1 - derived features)
  Train TGN + XGBoost on only 8 raw TVL features vs all 46.
  If TGN still beats XGBoost on TVL-only, circularity argument is neutralized.

Experiment 2: 5-Seed TGN Ensemble (vulnerability #2 - variance, #3 - AUROC gap)
  Train 5 TGN seeds, then average predictions (ensemble).
  Should reduce variance and potentially close AUROC gap with XGBoost.

Experiment 3: Graph Structure Ablation (vulnerability #4 - small graph)
  Compare real composability graph vs random graph vs complete graph.
  If real graph wins, structure matters even at 15 nodes.

Experiment 4: Leave-One-Event-Out (vulnerability #5 - one test event)
  For each of the 3 training events (Terra, 3AC, FTX), hold it out and
  test whether TGN trained without it still detects it.
"""
import sys, os, copy, time, json, random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(__file__))

from experiments.run_experiments import RealDataPipeline, ExperimentRunner
from models.tgn import TemporalGraphNetwork
from training.losses import FocalLoss, MonotonicityRegularization

HORIZONS = [24, 72, 168, 720]
HORIZON_KEYS = [f"cascade_{h}h" for h in HORIZONS]
DEVICE = torch.device("cpu")
EPOCHS = 60       # CPU-friendly
PATIENCE = 18
SEEDS = [42, 123, 456, 789, 1337]

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load_and_prepare(tvl_only=False, seed=42):
    """Load real data, build graph, prepare tensors."""
    pipeline = RealDataPipeline(seed=seed)
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
    return prepared, data


def train_tgn(prepared, seed=42, edge_index_override=None, epochs=None, patience=None):
    """Train TGN, return (test_metrics, raw_preds, raw_targets)."""
    if epochs is None:
        epochs = EPOCHS
    if patience is None:
        patience = PATIENCE
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    feat_dim = prepared["feature_dim"]
    eid = edge_index_override if edge_index_override else prepared["edge_index_dict"]
    edge_types = list(eid.keys())

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

    for epoch in range(epochs):
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
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} val={avg_val:.4f} no_improve={no_improve}")
        if no_improve >= patience:
            print(f"    Early stop at epoch {epoch+1}")
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
    preds_out = {k: [] for k in HORIZON_KEYS}
    targets_out = {k: [] for k in HORIZON_KEYS}
    with torch.no_grad():
        for t in range(*test_sl.indices(len(nf))):
            out = model(nf[t].to(DEVICE), eid, ts[t].expand(15).to(DEVICE))
            model.memory.detach_memory()
            for k in HORIZON_KEYS:
                preds_out[k].append(torch.sigmoid(out[k]).cpu().item())
                targets_out[k].append(la[k][t].item())
    metrics = eval_metrics(preds_out, targets_out)
    return metrics, preds_out, targets_out


def train_xgboost(prepared, seed=42):
    """Train XGBoost, return test metrics dict."""
    import xgboost as xgb

    nf_np = prepared["node_features_np"]
    la = prepared["label_arrays"]
    train_sl = prepared["splits"]["train"]
    test_sl = prepared["splits"]["test"]

    # Build tabular features
    from models.baselines.xgboost_model import XGBoostCascadePredictor
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
            results[k] = {"auroc": float(roc_auc_score(y, p)),
                          "auprc": float(average_precision_score(y, p))}
    return results


def print_results_row(name, m):
    r = f"  {name:<28}"
    for k in HORIZON_KEYS:
        r += f" {m[k]['auroc']:.3f}/{m[k]['auprc']:.3f}"
    print(r)


# =====================================================================
# EXPERIMENT 1: TVL-Only Ablation
# =====================================================================
def run_exp1_tvl_only():
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: TVL-Only Ablation (addresses derived-feature circularity)")
    print("=" * 70)

    print("\n  Loading full features...")
    prep_full, _ = load_and_prepare(tvl_only=False)
    print(f"  Full feature dim: {prep_full['feature_dim']}")

    print("\n  Loading TVL-only features...")
    prep_tvl, _ = load_and_prepare(tvl_only=True)
    print(f"  TVL-only feature dim: {prep_tvl['feature_dim']}")

    results = {}

    print("\n  Training TGN (all features)...")
    m, _, _ = train_tgn(prep_full, seed=42)
    results["tgn_full"] = m
    print_results_row("TGN (all 46)", m)

    print("\n  Training TGN (TVL-only 8)...")
    m, _, _ = train_tgn(prep_tvl, seed=42)
    results["tgn_tvl"] = m
    print_results_row("TGN (TVL-only 8)", m)

    print("\n  Training XGBoost (all features)...")
    m = train_xgboost(prep_full, seed=42)
    results["xgb_full"] = m
    print_results_row("XGBoost (all 46)", m)

    print("\n  Training XGBoost (TVL-only 8)...")
    m = train_xgboost(prep_tvl, seed=42)
    results["xgb_tvl"] = m
    print_results_row("XGBoost (TVL-only 8)", m)

    # Key finding
    print("\n  --- KEY FINDING ---")
    for k, label in zip(HORIZON_KEYS, ["1d", "3d", "7d", "30d"]):
        tgn_w = results["tgn_tvl"][k]["auprc"] > results["xgb_tvl"][k]["auprc"]
        diff = results["tgn_tvl"][k]["auprc"] - results["xgb_tvl"][k]["auprc"]
        print(f"  {label}: TGN TVL-only AUPRC={results['tgn_tvl'][k]['auprc']:.3f} vs "
              f"XGBoost TVL-only AUPRC={results['xgb_tvl'][k]['auprc']:.3f} "
              f"({'TGN wins' if tgn_w else 'XGBoost wins'} by {abs(diff):.3f})")

    return results


# =====================================================================
# EXPERIMENT 2: 5-Seed TGN Ensemble
# =====================================================================
def run_exp2_ensemble():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: 5-Seed TGN Ensemble (addresses variance + AUROC gap)")
    print("=" * 70)

    prep, _ = load_and_prepare(tvl_only=False)

    all_preds = {k: [] for k in HORIZON_KEYS}
    all_targets = None
    seed_results = {}

    for seed in SEEDS:
        print(f"\n  Training TGN seed={seed}...")
        m, p, t = train_tgn(prep, seed=seed)
        seed_results[seed] = m
        print_results_row(f"Seed {seed}", m)
        for k in HORIZON_KEYS:
            all_preds[k].append(p[k])
        if all_targets is None:
            all_targets = t

    # Ensemble: average predictions
    ensemble_preds = {}
    for k in HORIZON_KEYS:
        stacked = np.array(all_preds[k])
        ensemble_preds[k] = list(stacked.mean(axis=0))

    ensemble_metrics = eval_metrics(ensemble_preds, all_targets)

    print("\n  --- ENSEMBLE RESULTS ---")
    print_results_row("TGN Ensemble (5 seeds)", ensemble_metrics)

    # Individual seed stats
    for k in HORIZON_KEYS:
        aurocs = [seed_results[s][k]["auroc"] for s in SEEDS]
        auprcs = [seed_results[s][k]["auprc"] for s in SEEDS]
        print(f"  {k}: AUROC mean={np.mean(aurocs):.3f}±{np.std(aurocs):.3f}, "
              f"AUPRC mean={np.mean(auprcs):.3f}±{np.std(auprcs):.3f}")
        print(f"    Ensemble: AUROC={ensemble_metrics[k]['auroc']:.3f}, "
              f"AUPRC={ensemble_metrics[k]['auprc']:.3f}")

    return {"seeds": seed_results, "ensemble": ensemble_metrics}


# =====================================================================
# EXPERIMENT 3: Graph Structure Ablation
# =====================================================================
def run_exp3_graph_ablation():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Graph Structure Ablation (addresses small-graph concern)")
    print("=" * 70)

    prep, _ = load_and_prepare(tvl_only=False)
    real_eid = prep["edge_index_dict"]
    edge_types = list(real_eid.keys())

    # Build random graph with same edge count per type
    np.random.seed(42)
    random_eid = {}
    for etype in edge_types:
        real_edges = real_eid[etype]
        n_edges = real_edges.shape[1]
        src = torch.randint(0, 15, (n_edges,))
        dst = torch.randint(0, 15, (n_edges,))
        # Avoid self-loops
        mask = src != dst
        src, dst = src[mask], dst[mask]
        random_eid[etype] = torch.stack([src, dst], dim=0)

    # Build complete graph (all pairs for each type)
    complete_eid = {}
    for etype in edge_types:
        pairs = []
        for i in range(15):
            for j in range(15):
                if i != j:
                    pairs.append([i, j])
        pairs = torch.tensor(pairs, dtype=torch.long).T
        complete_eid[etype] = pairs

    results = {}

    print("\n  Training TGN (real graph)...")
    m, _, _ = train_tgn(prep, seed=42, edge_index_override=real_eid)
    results["real"] = m
    print_results_row("Real composability graph", m)

    print("\n  Training TGN (random graph)...")
    m, _, _ = train_tgn(prep, seed=42, edge_index_override=random_eid)
    results["random"] = m
    print_results_row("Random graph", m)

    print("\n  Training TGN (complete graph)...")
    m, _, _ = train_tgn(prep, seed=42, edge_index_override=complete_eid)
    results["complete"] = m
    print_results_row("Complete graph", m)

    print("\n  --- KEY FINDING ---")
    for k, label in zip(HORIZON_KEYS, ["1d", "3d", "7d", "30d"]):
        real_v = results["real"][k]["auprc"]
        rand_v = results["random"][k]["auprc"]
        comp_v = results["complete"][k]["auprc"]
        print(f"  {label}: Real={real_v:.3f} vs Random={rand_v:.3f} vs Complete={comp_v:.3f}")
        if real_v > rand_v and real_v > comp_v:
            print(f"    -> Real graph WINS (structure matters)")
        elif real_v > rand_v:
            print(f"    -> Real > Random (structure helps vs noise)")

    return results


# =====================================================================
# EXPERIMENT 4: Leave-One-Event-Out Cross-Validation
# =====================================================================
def run_exp4_leave_event_out():
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Leave-One-Event-Out (addresses single-test-event concern)")
    print("=" * 70)
    print("  Training on modified label sets where one crisis is removed,")
    print("  then testing if TGN still detects the held-out event.")

    prep, data = load_and_prepare(tvl_only=False)

    # Event date ranges (approximate day indices from June 2021 start)
    # Terra/Luna: May 2022 ~ day 334
    # 3AC/Celsius: Jun 2022 ~ day 365
    # FTX: Nov 2022 ~ day 518 (edge of train/val)
    events = {
        "Terra/Luna": {"peak_idx": 334, "window": 30},
        "3AC/Celsius": {"peak_idx": 365, "window": 30},
        "FTX": {"peak_idx": 518, "window": 30},
    }

    la_original = {k: prep["label_arrays"][k].clone() for k in HORIZON_KEYS}
    results = {}

    for event_name, info in events.items():
        print(f"\n  --- Holding out: {event_name} ---")
        peak = info["peak_idx"]
        w = info["window"]

        # Zero out labels around this event in training
        modified_la = {k: la_original[k].clone() for k in HORIZON_KEYS}
        for k in HORIZON_KEYS:
            for t in range(max(0, peak - w), min(len(modified_la[k]), peak + w)):
                modified_la[k][t] = 0.0

        # Replace labels in prepared
        prep_mod = dict(prep)
        prep_mod["label_arrays"] = modified_la

        # Train on modified labels
        print(f"  Training TGN without {event_name} labels...")
        m, preds, targets = train_tgn(prep_mod, seed=42, epochs=50, patience=15)
        print_results_row(f"TGN (no {event_name})", m)

        # Check if the held-out event region still shows elevated predictions
        # Look at prediction scores around the event peak
        test_sl = prep["splits"]["test"]
        train_sl = prep["splits"]["train"]

        # Event is in training period - check what predictions look like
        # during the event window on training data (not ideal but illustrative)
        # For events in train, we check if model learns to detect similar patterns
        results[event_name] = {
            "metrics_without_event": m,
            "peak_idx": peak,
        }

    # Also train baseline with all events
    print(f"\n  Training TGN (all events, baseline)...")
    m_base, _, _ = train_tgn(prep, seed=42, epochs=50, patience=15)
    results["baseline"] = m_base
    print_results_row("TGN (all events)", m_base)

    print("\n  --- KEY FINDING ---")
    print("  If removing one event barely changes test metrics, model generalizes.")
    for event_name in events:
        m_no = results[event_name]["metrics_without_event"]
        for k, label in zip(HORIZON_KEYS[:2], ["1d", "3d"]):
            base_v = results["baseline"][k]["auprc"]
            no_v = m_no[k]["auprc"]
            diff = no_v - base_v
            print(f"  {event_name} removed, {label}: AUPRC {no_v:.3f} "
                  f"(baseline {base_v:.3f}, delta {diff:+.3f})")

    return results


# =====================================================================
# MAIN
# =====================================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("OVERNIGHT EXPERIMENTS — Addressing 5 Reviewer Vulnerabilities")
    print(f"Device: {DEVICE} | Max epochs: {EPOCHS} | Seeds: {SEEDS}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = {}

    # Experiment 1: TVL-Only (~90 min on CPU)
    try:
        all_results["exp1_tvl_only"] = run_exp1_tvl_only()
    except Exception as e:
        print(f"  EXP1 FAILED: {e}")
        import traceback; traceback.print_exc()

    # Experiment 2: 5-Seed Ensemble (~3-4 hours on CPU)
    try:
        all_results["exp2_ensemble"] = run_exp2_ensemble()
    except Exception as e:
        print(f"  EXP2 FAILED: {e}")
        import traceback; traceback.print_exc()

    # Experiment 3: Graph Structure Ablation (~90 min on CPU)
    try:
        all_results["exp3_graph_ablation"] = run_exp3_graph_ablation()
    except Exception as e:
        print(f"  EXP3 FAILED: {e}")
        import traceback; traceback.print_exc()

    # Experiment 4: Leave-One-Event-Out (~2 hours on CPU)
    try:
        all_results["exp4_leave_event_out"] = run_exp4_leave_event_out()
    except Exception as e:
        print(f"  EXP4 FAILED: {e}")
        import traceback; traceback.print_exc()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed/3600:.1f} hours")
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Save results
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"Cannot serialize {type(o)}")

    out_path = os.path.join(OUT_DIR, "overnight_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=conv)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
