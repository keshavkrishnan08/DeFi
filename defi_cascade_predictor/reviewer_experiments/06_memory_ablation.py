#!/usr/bin/env python3
"""
Experiment 6: Memory Ablation
Reproduces Table IV (Memory Ablation).

Trains TGN with and without GRU memory to quantify memory's contribution.
Without memory, the model resets at every timestep (no persistent state).

Outputs: outputs/reviewer_memory_ablation.json
"""
import sys, os, time, copy
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reviewer_experiments.common import (
    load_real_data, train_tgn, build_tgn, evaluate_tgn,
    save_results, print_metrics,
    HORIZONS, HORIZON_KEYS, DEVICE, EPOCHS, PATIENCE,
)
from training.losses import FocalLoss, MonotonicityRegularization
import torch.nn.functional as F


def train_tgn_no_memory(prepared, seed=42):
    """Train TGN but reset memory at every timestep (ablation)."""
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

    for epoch in range(EPOCHS):
        model.train()
        if epoch < 10:
            for pg in optimizer.param_groups:
                pg["lr"] = 3e-4 * (epoch + 1) / 10

        for step, t in enumerate(train_indices):
            # KEY ABLATION: reset memory before every timestep
            model.reset_memory()

            x = nf[t].to(DEVICE)
            timestamp = ts[t].expand(15).to(DEVICE)
            preds = model(x, eid, timestamp)
            loss = sum(
                hw[h] * criterion(
                    preds[f"cascade_{h}h"].unsqueeze(0),
                    la[f"cascade_{h}h"][t].unsqueeze(0).to(DEVICE),
                )
                for h in HORIZONS
            )
            loss = loss + 0.3 * F.mse_loss(preds["severity"], sev[t].to(DEVICE))
            loss = loss + mono_reg(preds)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        vl, nv = 0.0, 0
        with torch.no_grad():
            for t in range(*val_sl.indices(len(nf))):
                model.reset_memory()  # no memory
                preds = model(nf[t].to(DEVICE), eid, ts[t].expand(15).to(DEVICE))
                vl += sum(
                    criterion(
                        preds[f"cascade_{h}h"].unsqueeze(0),
                        la[f"cascade_{h}h"][t].unsqueeze(0).to(DEVICE),
                    ).item()
                    for h in HORIZONS
                )
                nv += 1

        avg_val = vl / max(nv, 1)
        if epoch >= 10:
            scheduler.step(avg_val)
        if avg_val < best_val:
            best_val, best_state, no_improve = avg_val, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} val={avg_val:.4f} no_improve={no_improve}")
        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # Test (no memory)
    model.eval()
    preds_out = {k: [] for k in HORIZON_KEYS}
    targets_out = {k: [] for k in HORIZON_KEYS}
    with torch.no_grad():
        for t in range(*test_sl.indices(len(nf))):
            model.reset_memory()
            out = model(nf[t].to(DEVICE), eid, ts[t].expand(15).to(DEVICE))
            for k in HORIZON_KEYS:
                preds_out[k].append(torch.sigmoid(out[k]).cpu().item())
                targets_out[k].append(la[k][t].item())

    from reviewer_experiments.common import compute_metrics
    return compute_metrics(preds_out, targets_out)


def main():
    print("=" * 70)
    print("MEMORY ABLATION")
    print(f"Device: {DEVICE}")
    print("=" * 70)
    t0 = time.time()

    print("\nLoading real data...")
    prepared = load_real_data(seed=42)

    results = {}

    print("\n[1/2] Full TGN (with memory)...")
    results["with_memory"] = train_tgn(prepared, seed=42)
    print_metrics("TGN (with memory)", results["with_memory"])

    print("\n[2/2] TGN without memory (reset each step)...")
    results["without_memory"] = train_tgn_no_memory(prepared, seed=42)
    print_metrics("TGN (no memory)", results["without_memory"])

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Completed in {elapsed/60:.1f} min")
    print(f"\nMEMORY IMPACT (delta AUROC):")
    for k, label in zip(HORIZON_KEYS, ["1d", "3d", "7d", "30d"]):
        with_m = results["with_memory"][k]["auroc"]
        without_m = results["without_memory"][k]["auroc"]
        delta = with_m - without_m
        print(f"  {label}: {with_m:.3f} -> {without_m:.3f} (delta={delta:+.3f})")

    results["elapsed_minutes"] = elapsed / 60
    save_results(results, "reviewer_memory_ablation.json")


if __name__ == "__main__":
    main()
