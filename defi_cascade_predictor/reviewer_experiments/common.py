"""
Shared utilities for reviewer experiment scripts.
Handles data loading, splitting, TGN training, and metric evaluation.
"""
import sys, os, copy, time, json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

# Add parent to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run_experiments import RealDataPipeline, ExperimentRunner
from models.tgn import TemporalGraphNetwork
from training.losses import FocalLoss, MonotonicityRegularization

HORIZONS = [24, 72, 168, 720]
HORIZON_KEYS = [f"cascade_{h}h" for h in HORIZONS]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU = torch.cuda.is_available()
EPOCHS = 120 if GPU else 50
PATIENCE = 25 if GPU else 18
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def load_real_data(seed=42, tvl_only=False):
    """Load real TVL data, build graph, prepare tensors."""
    import yaml

    pipeline = RealDataPipeline(seed=seed)
    data = pipeline.load_all()

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
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


def build_tgn(feat_dim, edge_types):
    """Create a TGN model with the paper's architecture."""
    return TemporalGraphNetwork(
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


def train_tgn(prepared, seed=42, epochs=None, patience=None, verbose=True):
    """Train TGN with the paper's training procedure. Returns test metrics."""
    if epochs is None:
        epochs = EPOCHS
    if patience is None:
        patience = PATIENCE

    torch.manual_seed(seed)
    np.random.seed(seed)

    feat_dim = prepared["feature_dim"]
    edge_types = list(prepared["edge_index_dict"].keys())
    eid = prepared["edge_index_dict"]

    model = build_tgn(feat_dim, edge_types)

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
        # LR warmup for first 10 epochs
        if epoch < 10:
            for pg in optimizer.param_groups:
                pg["lr"] = 3e-4 * (epoch + 1) / 10

        wl = torch.tensor(0.0, device=DEVICE)
        wc = 0
        for step, t in enumerate(train_indices):
            x = nf[t].to(DEVICE)
            timestamp = ts[t].expand(15).to(DEVICE)
            preds = model(x, eid, timestamp)
            loss = sum(
                hw[h]
                * criterion(
                    preds[f"cascade_{h}h"].unsqueeze(0),
                    la[f"cascade_{h}h"][t].unsqueeze(0).to(DEVICE),
                )
                for h in HORIZONS
            )
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

        # Validation
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
                vl += sum(
                    criterion(
                        preds[f"cascade_{h}h"].unsqueeze(0),
                        la[f"cascade_{h}h"][t].unsqueeze(0).to(DEVICE),
                    ).item()
                    for h in HORIZONS
                )
                model.memory.detach_memory()
                nv += 1
        avg_val = vl / max(nv, 1)
        if epoch >= 10:
            scheduler.step(avg_val)
        if avg_val < best_val:
            best_val, best_state, no_improve = (
                avg_val,
                copy.deepcopy(model.state_dict()),
                0,
            )
        else:
            no_improve += 1
        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1}/{epochs} val={avg_val:.4f} no_improve={no_improve}"
            )
        if no_improve >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # Test evaluation
    return evaluate_tgn(model, prepared)


def evaluate_tgn(model, prepared):
    """Evaluate trained TGN on test set. Returns metrics dict."""
    eid = prepared["edge_index_dict"]
    nf = prepared["node_features"]
    ts = prepared["timestamps"]
    la = prepared["label_arrays"]
    test_sl = prepared["splits"]["test"]

    model.eval()
    model.reset_memory()
    # Replay history up to test
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

    return compute_metrics(preds_out, targets_out)


def compute_metrics(preds, targets):
    """Compute AUROC and AUPRC for each horizon."""
    results = {}
    for k in HORIZON_KEYS:
        y = np.array(targets[k])
        p = np.array(preds[k])
        if y.sum() == 0 or y.sum() == len(y):
            results[k] = {"auroc": 0.5, "auprc": float(y.mean())}
        else:
            results[k] = {
                "auroc": float(roc_auc_score(y, p)),
                "auprc": float(average_precision_score(y, p)),
            }
    return results


def save_results(results, filename):
    """Save results dict to JSON."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)

    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        raise TypeError

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=conv)
    print(f"Results saved to {path}")


def print_metrics(name, metrics):
    """Pretty-print metrics for one model."""
    print(f"\n{name}:")
    for k, label in zip(HORIZON_KEYS, ["1-day", "3-day", "7-day", "30-day"]):
        print(f"  {label}: AUROC={metrics[k]['auroc']:.3f}  AUPRC={metrics[k]['auprc']:.3f}")
