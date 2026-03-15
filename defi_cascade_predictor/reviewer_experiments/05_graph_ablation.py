#!/usr/bin/env python3
"""
Experiment 5: Graph Structure Ablation
Tests whether the real composability graph matters vs random/complete graphs.

Trains TGN with three graph structures:
  1. Real composability graph (from protocol documentation)
  2. Random graph (same density, random edges)
  3. Complete graph (all pairs connected)

Outputs: outputs/reviewer_graph_ablation.json
"""
import sys, os, time, copy
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reviewer_experiments.common import (
    load_real_data, train_tgn, save_results, print_metrics,
    HORIZONS, HORIZON_KEYS, DEVICE,
)


def randomize_edges(prepared, seed=42):
    """Replace real edges with random edges of same density."""
    rng = np.random.RandomState(seed)
    prep = copy.deepcopy(prepared)
    eid = prep["edge_index_dict"]

    for k, ei in eid.items():
        n_edges = ei.shape[1]
        src = torch.tensor(rng.randint(0, 15, n_edges))
        dst = torch.tensor(rng.randint(0, 15, n_edges))
        eid[k] = torch.stack([src, dst]).to(ei.device)

    return prep


def complete_graph(prepared):
    """Replace real edges with complete graph."""
    prep = copy.deepcopy(prepared)
    eid = prep["edge_index_dict"]

    # All pairs (15 nodes)
    src = []
    dst = []
    for i in range(15):
        for j in range(15):
            if i != j:
                src.append(i)
                dst.append(j)
    full_ei = torch.tensor([src, dst])

    for k in eid:
        eid[k] = full_ei.to(eid[k].device)

    return prep


def main():
    print("=" * 70)
    print("GRAPH STRUCTURE ABLATION")
    print(f"Device: {DEVICE}")
    print("=" * 70)
    t0 = time.time()

    print("\nLoading real data...")
    prepared = load_real_data(seed=42)

    results = {}

    print("\n[1/3] TGN with real composability graph...")
    results["real"] = train_tgn(prepared)
    print_metrics("Real graph", results["real"])

    print("\n[2/3] TGN with random graph (same density)...")
    prep_random = randomize_edges(prepared)
    results["random"] = train_tgn(prep_random)
    print_metrics("Random graph", results["random"])

    print("\n[3/3] TGN with complete graph...")
    prep_complete = complete_graph(prepared)
    results["complete"] = train_tgn(prep_complete)
    print_metrics("Complete graph", results["complete"])

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Completed in {elapsed/60:.1f} min")
    print(f"\nKEY FINDING:")
    for k, label in zip(HORIZON_KEYS, ["1d", "3d", "7d", "30d"]):
        real_a = results["real"][k]["auprc"]
        rand_a = results["random"][k]["auprc"]
        comp_a = results["complete"][k]["auprc"]
        best = "Real" if real_a >= max(rand_a, comp_a) else ("Random" if rand_a > comp_a else "Complete")
        print(f"  {label}: Real={real_a:.3f}  Random={rand_a:.3f}  Complete={comp_a:.3f}  -> {best} wins")

    results["elapsed_minutes"] = elapsed / 60
    save_results(results, "reviewer_graph_ablation.json")


if __name__ == "__main__":
    main()
