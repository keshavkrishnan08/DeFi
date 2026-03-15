"""
Ablation experiments for IEEE TCSS paper revision.
Addresses reviewer concerns:
  1. TVL-only ablation (R2, R3, R7): feature circularity
  2. Platt scaling calibration (R3, R7): poor ECE
  3. TGN ensemble (R1, R9): high seed variance
  4. Edge-type ablation (R4, R6): graph construction

Run on Colab GPU for ~45 min total, or CPU for experiments 1-3 only (~90 min).
"""
import sys, os, copy, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

sys.path.insert(0, os.path.dirname(__file__))

from experiments.run_experiments import RealDataPipeline, ExperimentRunner
from models.tgn import TemporalGraphNetwork
from models.baselines.xgboost_model import XGBoostCascadePredictor
from training.losses import FocalLoss, MonotonicityRegularization

HORIZONS = [24, 72, 168, 720]
HORIZON_KEYS = [f"cascade_{h}h" for h in HORIZONS]
SEEDS = [42, 123, 456, 789, 1337]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU = torch.cuda.is_available()
# Epochs: more on GPU, fewer on CPU
ABLATION_EPOCHS = 120 if GPU else 50
ENSEMBLE_EPOCHS = 80 if GPU else 40
PATIENCE = 20 if GPU else 15


def load_and_prepare(tvl_only=False):
    """Load real data, build graph, prepare tensors. If tvl_only, keep only 8 TVL features."""
    pipeline = RealDataPipeline(seed=42)
    data = pipeline.load_all()

    # Build features using ExperimentRunner infrastructure
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    config["training"] = config.get("training", {})
    config["training"]["epochs"] = ABLATION_EPOCHS
    config["training"]["patience"] = PATIENCE

    runner = ExperimentRunner(config)
    runner._split_dates = {
        "train_end": data["dates"][517],  # ~2022-10-31
        "val_end": data["dates"][668],    # ~2023-03-31
    }

    graph = runner.build_graph_and_features(data)

    if tvl_only:
        # Keep only first 8 features (TVL group) per node
        nf = graph["node_features_array"][:, :, :8]
        graph["node_features_array"] = nf
        graph["feature_dim"] = 8

    prepared = runner.prepare_data(graph)
    return runner, prepared, data


def train_tgn_single(prepared, seed=42, epochs=None, edge_index_dict=None):
    """Train a single TGN model. Returns (model, val_preds, test_preds, test_targets)."""
    if epochs is None:
        epochs = ABLATION_EPOCHS

    torch.manual_seed(seed)
    np.random.seed(seed)

    feat_dim = prepared["feature_dim"]
    edge_types = list(prepared["edge_index_dict"].keys())
    eid = edge_index_dict if edge_index_dict is not None else prepared["edge_index_dict"]

    model = TemporalGraphNetwork(
        num_nodes=15,
        node_feature_dim=feat_dim,
        edge_types=edge_types,
        memory_dim=128,
        embedding_dim=128,
        num_attention_heads=4,
        num_gnn_layers=2,
        prediction_horizons=HORIZONS,
        dropout=0.15,
        memory_updater="gru",
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
    horizon_weights = {24: 3.0, 72: 2.0, 168: 1.0, 720: 1.0}

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        if epoch == 0:
            model.reset_memory()

        # Warmup
        if epoch < 10:
            for pg in optimizer.param_groups:
                pg["lr"] = 3e-4 * (epoch + 1) / 10

        window_loss = torch.tensor(0.0, device=DEVICE)
        window_count = 0

        for step, t in enumerate(train_indices):
            x = nf[t].to(DEVICE)
            timestamp = ts[t].expand(15).to(DEVICE)
            preds = model(x, eid, timestamp)

            loss = torch.tensor(0.0, device=DEVICE)
            for h in HORIZONS:
                key = f"cascade_{h}h"
                loss = loss + horizon_weights[h] * criterion(
                    preds[key].unsqueeze(0), la[key][t].unsqueeze(0).to(DEVICE)
                )
            loss = loss + 0.3 * F.mse_loss(preds["severity"], sev[t].to(DEVICE))
            loss = loss + mono_reg(preds)

            window_loss = window_loss + loss
            window_count += 1

            if window_count >= 10 or step == len(train_indices) - 1:
                avg_wl = window_loss / window_count
                optimizer.zero_grad()
                avg_wl.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.memory.detach_memory()
                window_loss = torch.tensor(0.0, device=DEVICE)
                window_count = 0

        # Validate
        model.eval()
        model.reset_memory()
        with torch.no_grad():
            for t in train_indices:
                x = nf[t].to(DEVICE)
                timestamp = ts[t].expand(15).to(DEVICE)
                model(x, eid, timestamp)
                model.memory.detach_memory()

        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for t in range(*val_sl.indices(len(nf))):
                x = nf[t].to(DEVICE)
                timestamp = ts[t].expand(15).to(DEVICE)
                preds = model(x, eid, timestamp)
                loss = sum(
                    criterion(preds[f"cascade_{h}h"].unsqueeze(0),
                              la[f"cascade_{h}h"][t].unsqueeze(0).to(DEVICE))
                    for h in HORIZONS
                )
                model.memory.detach_memory()
                val_loss += loss.item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        if epoch >= 10:
            scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} val={avg_val:.4f} best={best_val:.4f} no_improve={no_improve}")

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # Collect predictions on val and test
    val_preds, test_preds, test_targets = collect_predictions(model, prepared, eid)
    return model, val_preds, test_preds, test_targets


@torch.no_grad()
def collect_predictions(model, prepared, eid):
    """Collect val and test predictions from a trained TGN."""
    model.eval()
    model.reset_memory()

    nf = prepared["node_features"]
    ts = prepared["timestamps"]
    la = prepared["label_arrays"]
    train_sl = prepared["splits"]["train"]
    val_sl = prepared["splits"]["val"]
    test_sl = prepared["splits"]["test"]

    # Warm up memory on training data
    for t in range(*train_sl.indices(len(nf))):
        x = nf[t].to(DEVICE)
        timestamp = ts[t].expand(15).to(DEVICE)
        model(x, eid, timestamp)
        model.memory.detach_memory()

    # Val predictions
    val_preds = {k: [] for k in HORIZON_KEYS}
    val_targets = {k: [] for k in HORIZON_KEYS}
    for t in range(*val_sl.indices(len(nf))):
        x = nf[t].to(DEVICE)
        timestamp = ts[t].expand(15).to(DEVICE)
        out = model(x, eid, timestamp)
        model.memory.detach_memory()
        for k in HORIZON_KEYS:
            val_preds[k].append(torch.sigmoid(out[k]).cpu().item())
            val_targets[k].append(la[k][t].item())

    # Test predictions
    test_preds = {k: [] for k in HORIZON_KEYS}
    test_targets = {k: [] for k in HORIZON_KEYS}
    for t in range(*test_sl.indices(len(nf))):
        x = nf[t].to(DEVICE)
        timestamp = ts[t].expand(15).to(DEVICE)
        out = model(x, eid, timestamp)
        model.memory.detach_memory()
        for k in HORIZON_KEYS:
            test_preds[k].append(torch.sigmoid(out[k]).cpu().item())
            test_targets[k].append(la[k][t].item())

    return (val_preds, val_targets), (test_preds, test_targets), test_targets


def compute_metrics(preds, targets):
    """Compute AUROC, AUPRC for each horizon."""
    results = {}
    for k in HORIZON_KEYS:
        y = np.array(targets[k])
        p = np.array(preds[k])
        if y.sum() == 0 or y.sum() == len(y):
            results[k] = {"auroc": 0.5, "auprc": y.mean()}
            continue
        results[k] = {
            "auroc": roc_auc_score(y, p),
            "auprc": average_precision_score(y, p),
        }
    return results


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += mask.sum() / len(y_true) * abs(avg_conf - avg_acc)
    return ece


# ============================================================
# EXPERIMENT 1: TVL-Only Ablation
# ============================================================
def run_tvl_only_ablation():
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: TVL-ONLY ABLATION")
    print("Trains TGN and XGBoost on 8 raw TVL features only")
    print("=" * 70)

    # Full features
    print("\n--- Training with ALL 46 features ---")
    runner_full, prepared_full, _ = load_and_prepare(tvl_only=False)
    _, _, (test_preds_full, test_targets_full), _ = train_tgn_single(prepared_full, seed=42)
    full_metrics = compute_metrics(test_preds_full, test_targets_full)

    # TVL-only features
    print("\n--- Training with TVL-ONLY 8 features ---")
    runner_tvl, prepared_tvl, data = load_and_prepare(tvl_only=True)
    _, _, (test_preds_tvl, test_targets_tvl), _ = train_tgn_single(prepared_tvl, seed=42)
    tvl_metrics = compute_metrics(test_preds_tvl, test_targets_tvl)

    # XGBoost with full features
    print("\n--- XGBoost (full features) ---")
    xgb_full = _train_xgboost(prepared_full)

    # XGBoost with TVL-only
    print("--- XGBoost (TVL-only features) ---")
    xgb_tvl = _train_xgboost(prepared_tvl)

    print("\n" + "=" * 70)
    print("TVL-ONLY ABLATION RESULTS")
    print("=" * 70)
    print(f"{'Config':<25} | {'1d AUROC':>9} {'1d AUPRC':>9} | {'3d AUROC':>9} {'3d AUPRC':>9} | {'7d AUROC':>9} {'7d AUPRC':>9} | {'30d AUROC':>9} {'30d AUPRC':>9}")
    print("-" * 120)
    for name, m in [("TGN (all 46)", full_metrics), ("TGN (TVL-only 8)", tvl_metrics),
                     ("XGBoost (all 46)", xgb_full), ("XGBoost (TVL-only 8)", xgb_tvl)]:
        row = f"{name:<25}"
        for k in HORIZON_KEYS:
            row += f" | {m[k]['auroc']:>9.3f} {m[k]['auprc']:>9.3f}"
        print(row)

    print("\n--- LaTeX-ready ---")
    for name, m in [("TGN (all)", full_metrics), ("TGN (TVL)", tvl_metrics),
                     ("XGB (all)", xgb_full), ("XGB (TVL)", xgb_tvl)]:
        vals = " & ".join(f"{m[k]['auroc']:.3f} / {m[k]['auprc']:.3f}" for k in HORIZON_KEYS)
        print(f"{name} & {vals} \\\\")

    return {"full": full_metrics, "tvl_only": tvl_metrics,
            "xgb_full": xgb_full, "xgb_tvl": xgb_tvl,
            "prepared_full": prepared_full, "test_preds_full": test_preds_full,
            "test_targets_full": test_targets_full}


def _train_xgboost(prepared):
    """Train XGBoost on prepared features and return test metrics."""
    import xgboost as xgb

    nf_np = prepared["node_features_np"]
    la = prepared["label_arrays"]
    train_sl = prepared["splits"]["train"]
    test_sl = prepared["splits"]["test"]

    xgb_model = XGBoostCascadePredictor(prediction_horizons=HORIZONS)
    window = 30
    X_all = xgb_model.prepare_features(list(nf_np), window=window)

    y_all = {}
    for k in HORIZON_KEYS:
        y = la[k].numpy()[window:window + len(X_all)]
        y_all[k] = y

    min_l = min(len(X_all), min(len(v) for v in y_all.values()))
    X_all = X_all[:min_l]
    y_all = {k: v[:min_l] for k, v in y_all.items()}

    n_train = max(train_sl.stop - window, 1)
    X_train = X_all[:n_train]
    y_train = {k: v[:n_train] for k, v in y_all.items()}
    xgb_model.fit(X_train, y_train)

    n_test_start = test_sl.start - window
    X_test = X_all[max(n_test_start, 0):]
    bp = xgb_model.predict(X_test)
    bt = {}
    for k in HORIZON_KEYS:
        bt[k] = y_all[k][max(n_test_start, 0):]
        min_l2 = min(len(bp[k]), len(bt[k]))
        bp[k] = bp[k][:min_l2]
        bt[k] = bt[k][:min_l2]

    return compute_metrics(bp, bt)


# ============================================================
# EXPERIMENT 2: Platt Scaling Calibration
# ============================================================
def run_platt_scaling(prepared, test_preds=None, test_targets=None):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: PLATT SCALING CALIBRATION")
    print("Fit logistic regression on val predictions, apply to test")
    print("=" * 70)

    if test_preds is None:
        print("Training TGN for calibration experiment...")
        _, (val_preds, val_targets), (test_preds, test_targets), _ = train_tgn_single(prepared, seed=42)
    else:
        # Need val predictions too — retrain
        print("Training TGN to get val predictions...")
        _, (val_preds, val_targets), (test_preds, test_targets), _ = train_tgn_single(prepared, seed=42)

    results = {}
    for k in HORIZON_KEYS:
        y_val = np.array(val_targets[k])
        p_val = np.array(val_preds[k]).reshape(-1, 1)
        y_test = np.array(test_targets[k])
        p_test = np.array(test_preds[k])

        if y_val.sum() == 0 or len(np.unique(y_val)) < 2:
            print(f"  {k}: skipped (no positive val samples)")
            continue

        # Fit Platt scaling (logistic regression on log-odds)
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        lr.fit(p_val, y_val)
        p_test_cal = lr.predict_proba(p_test.reshape(-1, 1))[:, 1]

        # Metrics before calibration
        ece_before = compute_ece(y_test, p_test)
        brier_before = brier_score_loss(y_test, p_test)
        auroc_before = roc_auc_score(y_test, p_test) if y_test.sum() > 0 else 0.5
        auprc_before = average_precision_score(y_test, p_test) if y_test.sum() > 0 else y_test.mean()

        # Metrics after calibration
        ece_after = compute_ece(y_test, p_test_cal)
        brier_after = brier_score_loss(y_test, p_test_cal)
        auroc_after = roc_auc_score(y_test, p_test_cal) if y_test.sum() > 0 else 0.5
        auprc_after = average_precision_score(y_test, p_test_cal) if y_test.sum() > 0 else y_test.mean()

        results[k] = {
            "ece_before": ece_before, "ece_after": ece_after,
            "brier_before": brier_before, "brier_after": brier_after,
            "auroc_before": auroc_before, "auroc_after": auroc_after,
            "auprc_before": auprc_before, "auprc_after": auprc_after,
        }
        print(f"  {k}: ECE {ece_before:.3f} -> {ece_after:.3f}, "
              f"Brier {brier_before:.3f} -> {brier_after:.3f}, "
              f"AUROC {auroc_before:.3f} -> {auroc_after:.3f}")

    print("\n--- LaTeX-ready ---")
    print(f"{'Horizon':<15} {'ECE (raw)':>10} {'ECE (cal)':>10} {'Brier (raw)':>12} {'Brier (cal)':>12} {'AUROC':>8}")
    for k in HORIZON_KEYS:
        if k not in results:
            continue
        r = results[k]
        print(f"{k:<15} {r['ece_before']:>10.3f} {r['ece_after']:>10.3f} "
              f"{r['brier_before']:>12.3f} {r['brier_after']:>12.3f} {r['auroc_after']:>8.3f}")

    return results


# ============================================================
# EXPERIMENT 3: TGN Ensemble (Average 5 Seeds)
# ============================================================
def run_ensemble(prepared):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: TGN ENSEMBLE (5 SEEDS)")
    print("Average predictions across seeds to reduce variance")
    print("=" * 70)

    all_test_preds = {k: [] for k in HORIZON_KEYS}
    individual_metrics = []
    test_targets_ref = None

    for i, seed in enumerate(SEEDS):
        print(f"\n--- Seed {seed} ({i+1}/{len(SEEDS)}) ---")
        t0 = time.time()
        _, _, (test_preds, test_targets), _ = train_tgn_single(
            prepared, seed=seed, epochs=ENSEMBLE_EPOCHS
        )
        elapsed = time.time() - t0
        print(f"  Trained in {elapsed:.0f}s")

        metrics = compute_metrics(test_preds, test_targets)
        individual_metrics.append(metrics)
        test_targets_ref = test_targets

        for k in HORIZON_KEYS:
            all_test_preds[k].append(np.array(test_preds[k]))

        print(f"  1d: AUROC={metrics['cascade_24h']['auroc']:.3f}, AUPRC={metrics['cascade_24h']['auprc']:.3f}")

    # Ensemble: average predictions
    ensemble_preds = {}
    for k in HORIZON_KEYS:
        ensemble_preds[k] = np.mean(all_test_preds[k], axis=0).tolist()
    ensemble_metrics = compute_metrics(ensemble_preds, test_targets_ref)

    # Summary
    print("\n" + "=" * 70)
    print("ENSEMBLE RESULTS")
    print("=" * 70)

    header = f"{'Config':<20}"
    for h in ["1d", "3d", "7d", "30d"]:
        header += f" | {h+' AUROC':>9} {h+' AUPRC':>9}"
    print(header)
    print("-" * 100)

    for i, seed in enumerate(SEEDS):
        row = f"{'Seed '+str(seed):<20}"
        for k in HORIZON_KEYS:
            row += f" | {individual_metrics[i][k]['auroc']:>9.3f} {individual_metrics[i][k]['auprc']:>9.3f}"
        print(row)

    # Mean/std of individual seeds
    for metric_name in ["auroc", "auprc"]:
        vals = {k: [m[k][metric_name] for m in individual_metrics] for k in HORIZON_KEYS}
        means = {k: np.mean(v) for k, v in vals.items()}
        stds = {k: np.std(v) for k, v in vals.items()}
        row = f"{'Mean±Std ('+metric_name+')':<20}"
        for k in HORIZON_KEYS:
            row += f" | {means[k]:>5.3f}±{stds[k]:.3f}         "
        print(row)

    row = f"{'ENSEMBLE (avg 5)':<20}"
    for k in HORIZON_KEYS:
        row += f" | {ensemble_metrics[k]['auroc']:>9.3f} {ensemble_metrics[k]['auprc']:>9.3f}"
    print(row)

    print("\n--- LaTeX-ready ---")
    for k in HORIZON_KEYS:
        aurocs = [m[k]["auroc"] for m in individual_metrics]
        auprcs = [m[k]["auprc"] for m in individual_metrics]
        print(f"{k}: Individual={np.mean(aurocs):.3f}±{np.std(aurocs):.3f} AUROC, "
              f"Ensemble={ensemble_metrics[k]['auroc']:.3f} AUROC, "
              f"{ensemble_metrics[k]['auprc']:.3f} AUPRC")

    return {
        "individual": individual_metrics,
        "ensemble": ensemble_metrics,
        "test_targets": test_targets_ref,
        "ensemble_preds": ensemble_preds,
    }


# ============================================================
# EXPERIMENT 4: Edge-Type Ablation
# ============================================================
def run_edge_ablation(prepared):
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: EDGE-TYPE ABLATION")
    print("Remove one edge type at a time + fully-connected baseline")
    print("=" * 70)

    eid_full = prepared["edge_index_dict"]
    edge_types = list(eid_full.keys())
    print(f"Edge types: {edge_types}")

    results = {}

    # Default (all edges)
    print("\n--- Default (all edges) ---")
    _, _, (test_preds, test_targets), _ = train_tgn_single(prepared, seed=42)
    results["all_edges"] = compute_metrics(test_preds, test_targets)

    # Drop one edge type at a time
    for drop_type in edge_types:
        print(f"\n--- Without {drop_type} ---")
        modified_eid = {}
        for etype, tensor in eid_full.items():
            if etype == drop_type:
                modified_eid[etype] = torch.zeros((2, 0), dtype=torch.long, device=DEVICE)
            else:
                modified_eid[etype] = tensor
        _, _, (test_preds, test_targets), _ = train_tgn_single(
            prepared, seed=42, edge_index_dict=modified_eid
        )
        results[f"no_{drop_type}"] = compute_metrics(test_preds, test_targets)

    # Fully connected (all pairs, single type, ignore heterogeneous structure)
    print("\n--- Fully connected (no structure) ---")
    fc_eid = {}
    # Create complete graph for each edge type
    all_pairs_src = []
    all_pairs_dst = []
    for i in range(15):
        for j in range(15):
            if i != j:
                all_pairs_src.append(i)
                all_pairs_dst.append(j)
    fc_tensor = torch.tensor([all_pairs_src, all_pairs_dst], dtype=torch.long, device=DEVICE)
    for etype in edge_types:
        fc_eid[etype] = fc_tensor
    _, _, (test_preds, test_targets), _ = train_tgn_single(
        prepared, seed=42, edge_index_dict=fc_eid
    )
    results["fully_connected"] = compute_metrics(test_preds, test_targets)

    # Summary
    print("\n" + "=" * 70)
    print("EDGE-TYPE ABLATION RESULTS")
    print("=" * 70)

    header = f"{'Config':<30}"
    for h in ["1d", "3d", "7d", "30d"]:
        header += f" | {h+' AUROC':>9} {h+' AUPRC':>9}"
    print(header)
    print("-" * 110)

    baseline = results["all_edges"]
    for name, m in results.items():
        row = f"{name:<30}"
        for k in HORIZON_KEYS:
            delta_a = m[k]["auroc"] - baseline[k]["auroc"]
            row += f" | {m[k]['auroc']:>9.3f} {m[k]['auprc']:>9.3f}"
        print(row)

    print("\n--- Delta from default ---")
    for name, m in results.items():
        if name == "all_edges":
            continue
        row = f"{name:<30}"
        for k in HORIZON_KEYS:
            da = m[k]["auroc"] - baseline[k]["auroc"]
            dp = m[k]["auprc"] - baseline[k]["auprc"]
            row += f" | {da:>+8.3f}  {dp:>+8.3f} "
        print(row)

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("ABLATION EXPERIMENTS FOR IEEE TCSS REVISION")
    print(f"Device: {DEVICE}")
    print(f"Ablation epochs: {ABLATION_EPOCHS}, Ensemble epochs: {ENSEMBLE_EPOCHS}")
    print("=" * 70)

    t_start = time.time()
    all_results = {}

    # Experiment 1: TVL-only ablation
    exp1 = run_tvl_only_ablation()
    all_results["tvl_ablation"] = {k: v for k, v in exp1.items()
                                    if k not in ("prepared_full", "test_preds_full", "test_targets_full")}

    # Experiment 2: Platt scaling (reuse prepared data from exp 1)
    # Need to reload full features since exp1 ended with TVL-only
    print("\nReloading full features for experiments 2-4...")
    _, prepared_full, _ = load_and_prepare(tvl_only=False)

    exp2 = run_platt_scaling(prepared_full)
    all_results["platt_scaling"] = exp2

    # Experiment 3: TGN Ensemble
    exp3 = run_ensemble(prepared_full)
    all_results["ensemble"] = {k: v for k, v in exp3.items() if k != "test_targets"}

    # Experiment 4: Edge-type ablation (GPU only — too slow on CPU)
    if GPU:
        exp4 = run_edge_ablation(prepared_full)
        all_results["edge_ablation"] = exp4
    else:
        print("\n" + "=" * 70)
        print("SKIPPING EXPERIMENT 4 (Edge Ablation) — CPU too slow")
        print("Run on GPU for full results")
        print("=" * 70)

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed/60:.1f} minutes")
    print("=" * 70)

    # Save results
    import json
    def convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    out_path = os.path.join(os.path.dirname(__file__), "outputs", "ablation_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
