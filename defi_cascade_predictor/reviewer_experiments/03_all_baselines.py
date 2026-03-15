#!/usr/bin/env python3
"""
Experiment 3: All Baselines (Static GNN, LSTM, Centrality)
Reproduces Table III, baseline rows.

Trains each baseline on the same data and splits as TGN.
Outputs: outputs/reviewer_baselines.json
"""
import sys, os, time, copy
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reviewer_experiments.common import (
    load_real_data, compute_metrics, save_results, print_metrics,
    HORIZONS, HORIZON_KEYS, DEVICE, GPU,
)
from models.baselines.static_gnn import StaticGNN
from models.baselines.lstm_model import LSTMCascadePredictor
from models.baselines.centrality_model import CentralityBaseline
from training.losses import FocalLoss

BASELINE_EPOCHS = 80 if GPU else 50
BASELINE_PATIENCE = 20


def train_static_gnn(prepared):
    """Train Static GNN baseline."""
    feat_dim = prepared["feature_dim"]
    model = StaticGNN(
        node_feature_dim=feat_dim,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        prediction_horizons=HORIZONS,
        dropout=0.15,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )
    criterion = FocalLoss(gamma=2.0, alpha=0.75)

    nf = prepared["node_features"]
    la = prepared["label_arrays"]
    homo_ei = prepared["homo_edge_index"]
    train_sl = prepared["splits"]["train"]
    val_sl = prepared["splits"]["val"]
    test_sl = prepared["splits"]["test"]

    best_state, best_val, no_improve = None, float("inf"), 0

    for epoch in range(BASELINE_EPOCHS):
        model.train()
        epoch_loss, n_steps = 0, 0
        for t in range(*train_sl.indices(len(nf))):
            x = nf[t].to(DEVICE)
            preds = model(x, homo_ei)
            loss = sum(
                criterion(
                    preds[f"cascade_{h}h"].unsqueeze(0),
                    la[f"cascade_{h}h"][t].unsqueeze(0).to(DEVICE),
                )
                for h in HORIZONS
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_loss += loss.item()
            n_steps += 1

        # Validation
        model.eval()
        val_loss, val_steps = 0, 0
        with torch.no_grad():
            for t in range(*val_sl.indices(len(nf))):
                x = nf[t].to(DEVICE)
                preds = model(x, homo_ei)
                loss = sum(
                    criterion(
                        preds[f"cascade_{h}h"].unsqueeze(0),
                        la[f"cascade_{h}h"][t].unsqueeze(0).to(DEVICE),
                    ).item()
                    for h in HORIZONS
                )
                val_loss += loss
                val_steps += 1

        avg_val = val_loss / max(val_steps, 1)
        scheduler.step(avg_val)
        if avg_val < best_val:
            best_val = avg_val
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{BASELINE_EPOCHS} val={avg_val:.4f}")
        if no_improve >= BASELINE_PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    preds_out = {k: [] for k in HORIZON_KEYS}
    targets_out = {k: [] for k in HORIZON_KEYS}
    with torch.no_grad():
        for t in range(*test_sl.indices(len(nf))):
            x = nf[t].to(DEVICE)
            out = model(x, homo_ei)
            for k in HORIZON_KEYS:
                preds_out[k].append(torch.sigmoid(out[k]).cpu().item())
                targets_out[k].append(la[k][t].item())
    return compute_metrics(preds_out, targets_out)


def train_lstm(prepared, seq_len=15):
    """Train LSTM baseline."""
    feat_dim = prepared["feature_dim"]
    model = LSTMCascadePredictor(
        input_dim=feat_dim * 15,  # flattened across nodes
        hidden_dim=256,
        num_layers=2,
        prediction_horizons=HORIZONS,
        dropout=0.2,
        bidirectional=True,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-3)
    criterion = FocalLoss(gamma=2.0, alpha=0.75)

    nf = prepared["node_features"]
    la = prepared["label_arrays"]
    train_sl = prepared["splits"]["train"]
    val_sl = prepared["splits"]["val"]
    test_sl = prepared["splits"]["test"]
    T = len(nf)

    best_state, best_val, no_improve = None, float("inf"), 0

    for epoch in range(BASELINE_EPOCHS):
        model.train()
        epoch_loss, n_steps = 0, 0
        for t in range(train_sl.start + seq_len, train_sl.stop):
            # Build sequence: [seq_len, 1, N*F]
            seq = torch.stack(
                [nf[t - seq_len + s].reshape(-1) for s in range(seq_len)]
            ).unsqueeze(1).to(DEVICE)
            preds = model(seq)
            loss = sum(
                criterion(
                    preds[f"cascade_{h}h"],
                    la[f"cascade_{h}h"][t].unsqueeze(0).to(DEVICE),
                )
                for h in HORIZONS
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_steps += 1

        # Validation
        model.eval()
        val_loss, val_steps = 0, 0
        with torch.no_grad():
            for t in range(val_sl.start + seq_len, val_sl.stop):
                if t - seq_len < 0:
                    continue
                seq = torch.stack(
                    [nf[t - seq_len + s].reshape(-1) for s in range(seq_len)]
                ).unsqueeze(1).to(DEVICE)
                preds = model(seq)
                loss = sum(
                    criterion(
                        preds[f"cascade_{h}h"],
                        la[f"cascade_{h}h"][t].unsqueeze(0).to(DEVICE),
                    ).item()
                    for h in HORIZONS
                )
                val_loss += loss
                val_steps += 1

        avg_val = val_loss / max(val_steps, 1)
        if avg_val < best_val:
            best_val, best_state, no_improve = avg_val, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{BASELINE_EPOCHS} val={avg_val:.4f}")
        if no_improve >= BASELINE_PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    preds_out = {k: [] for k in HORIZON_KEYS}
    targets_out = {k: [] for k in HORIZON_KEYS}
    with torch.no_grad():
        for t in range(*test_sl.indices(T)):
            if t - seq_len < 0:
                continue
            seq = torch.stack(
                [nf[t - seq_len + s].reshape(-1) for s in range(seq_len)]
            ).unsqueeze(1).to(DEVICE)
            out = model(seq)
            for k in HORIZON_KEYS:
                preds_out[k].append(torch.sigmoid(out[k]).cpu().item())
                targets_out[k].append(la[k][t].item())
    return compute_metrics(preds_out, targets_out)


def train_centrality(prepared):
    """Train Centrality baseline."""
    nf_np = prepared["node_features_np"]
    la = prepared["label_arrays"]
    train_sl = prepared["splits"]["train"]
    test_sl = prepared["splits"]["test"]
    homo_ei = prepared["homo_edge_index"]

    model = CentralityBaseline(
        num_nodes=15,
        edge_index=homo_ei,
        prediction_horizons=HORIZONS,
    )

    # Prepare features for centrality model
    train_features = []
    train_labels = {k: [] for k in HORIZON_KEYS}
    for t in range(*train_sl.indices(len(nf_np))):
        train_features.append(nf_np[t].flatten())
        for k in HORIZON_KEYS:
            train_labels[k].append(la[k][t].item())

    X_train = np.array(train_features)
    y_train = {k: np.array(v) for k, v in train_labels.items()}
    model.fit(X_train, y_train)

    # Test
    test_features = []
    test_labels = {k: [] for k in HORIZON_KEYS}
    for t in range(*test_sl.indices(len(nf_np))):
        test_features.append(nf_np[t].flatten())
        for k in HORIZON_KEYS:
            test_labels[k].append(la[k][t].item())

    X_test = np.array(test_features)
    preds = model.predict(X_test)
    return compute_metrics(preds, test_labels)


def main():
    print("=" * 70)
    print("ALL BASELINES (Static GNN, LSTM, Centrality)")
    print(f"Device: {DEVICE} | Epochs: {BASELINE_EPOCHS}")
    print("=" * 70)
    t0 = time.time()

    print("\nLoading real data...")
    prepared = load_real_data(seed=42)
    print(f"  Feature dim: {prepared['feature_dim']}")

    results = {}

    print("\n[1/3] Training Static GNN...")
    try:
        results["static_gnn"] = train_static_gnn(prepared)
        print_metrics("Static GNN", results["static_gnn"])
    except Exception as e:
        print(f"  Static GNN failed: {e}")
        results["static_gnn"] = {"error": str(e)}

    print("\n[2/3] Training LSTM...")
    try:
        results["lstm"] = train_lstm(prepared)
        print_metrics("LSTM", results["lstm"])
    except Exception as e:
        print(f"  LSTM failed: {e}")
        results["lstm"] = {"error": str(e)}

    print("\n[3/3] Training Centrality...")
    try:
        results["centrality"] = train_centrality(prepared)
        print_metrics("Centrality", results["centrality"])
    except Exception as e:
        print(f"  Centrality failed: {e}")
        results["centrality"] = {"error": str(e)}

    elapsed = time.time() - t0
    print(f"\nAll baselines completed in {elapsed/60:.1f} min")
    results["elapsed_minutes"] = elapsed / 60
    save_results(results, "reviewer_baselines.json")


if __name__ == "__main__":
    main()
