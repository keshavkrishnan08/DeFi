# DeFi Cascade Predictor

A Temporal Graph Network (TGN) for predicting liquidation cascades across decentralized finance protocols. The model treats 15 Ethereum DeFi protocols as nodes in a heterogeneous temporal graph, connected by 6 edge types that capture composability relationships -- shared collateral, liquidity flows, oracle dependencies, governance overlap, price correlations, and liquidation pathways. It predicts cascade events at 1-day, 3-day, 7-day, and 30-day horizons using real TVL data spanning 1,734 days (June 2021 through February 2026).

## Directory Structure

```
defi_cascade_predictor/
  config/
    config.yaml              # All hyperparameters and protocol definitions
  data/
    collectors/              # Data fetchers (DeFiLlama, CoinGecko, FRED, BigQuery)
    processing/              # Feature engineering and graph construction
  models/
    tgn.py                   # Temporal Graph Network (GRU memory, temporal attention,
                             #   feature gating, monotonicity loss, multi-scale context)
    layers/                  # Memory module, message passing, temporal attention
    baselines/               # Static GNN, LSTM, XGBoost, SIR contagion, centrality
  training/
    trainer.py               # Training loop with LR warmup and early stopping
    losses.py                # Focal loss, monotonicity loss, horizon-weighted BCE
  evaluation/
    metrics.py               # AUROC, AUPRC, F1, Brier score, lead time, MCC
    visualization.py         # All figures (ROC/PR curves, training curves, comparisons)
    statistical_tests.py     # Bootstrap CIs, Diebold-Mariano, McNemar, Wilcoxon
    ablation.py              # Ablation study framework
  experiments/
    run_experiments.py        # Main experiment orchestrator (~2700 lines)
  reviewer_experiments/       # 7 standalone scripts for reproducing individual results
  main.py                     # CLI entry point
  fetch_real_data.py          # Downloads TVL, price, volume, and macro data
  requirements.txt
```

## Key Files

| File | What it does |
|------|-------------|
| `config/config.yaml` | Defines all model hyperparameters, protocol list, edge types, training schedule, and evaluation settings |
| `experiments/run_experiments.py` | Orchestrates the full pipeline: data loading, graph construction, TGN training, baseline comparisons, ablations, statistical tests, and visualization |
| `models/tgn.py` | The TGN itself -- GRU-based memory, multi-head temporal attention, feature gating, and multi-horizon prediction heads |
| `evaluation/visualization.py` | Generates every figure: ROC/PR curves, multi-horizon comparisons, training curves, ablation plots |
| `main.py` | Command-line entry point with `--quick`, `--device`, `--seed`, and `--config` flags |

## Reviewer Experiments

The `reviewer_experiments/` directory contains 7 self-contained scripts, each reproducing a specific result. No notebook required -- just run the Python file.

| Script | Reproduces |
|--------|-----------|
| `01_tgn_multiseed.py` | TGN with 5 random seeds + ensemble |
| `02_xgboost_baseline.py` | XGBoost baseline |
| `03_all_baselines.py` | Static GNN, LSTM, and centrality baselines |
| `04_tvl_ablation.py` | TVL-only feature ablation |
| `05_graph_ablation.py` | Graph structure ablation |
| `06_memory_ablation.py` | Memory module ablation |
| `07_best_seed_showcase.py` | Best single-seed results |

Run any of them from the `defi_cascade_predictor/` directory:

```bash
cd defi_cascade_predictor
python reviewer_experiments/01_tgn_multiseed.py
```

Results land in `outputs/reviewer_*.json`.

## Data

All experiments use real Total Value Locked (TVL) data from [DeFiLlama](https://defillama.com/). The dataset covers 15 Ethereum protocols across 5 categories:

- **Lending**: Aave V2/V3, Compound V2/V3, Morpho
- **DEX**: Uniswap V2/V3, Curve, Balancer
- **Liquid staking**: Lido, Rocket Pool
- **Yield**: Convex, Yearn
- **CDP / Stablecoin**: MakerDAO, Frax

Cascade labels are derived from aggregate TVL drawdowns using z-score thresholds. The dataset spans 1,734 daily snapshots. Each protocol node gets 46 base features, augmented to 322 through 7 temporal channels over a 30-day window.

To fetch fresh data:

```bash
cd defi_cascade_predictor
python fetch_real_data.py
```

## How to Run

**Full pipeline** (trains TGN, all baselines, runs ablations, generates figures):

```bash
cd defi_cascade_predictor
python main.py
```

**Quick test** (reduced epochs, faster convergence checks):

```bash
python main.py --quick
```

**Force CPU** (automatically caps epochs for reasonable runtime):

```bash
python main.py --device cpu
```

**Custom config or seed**:

```bash
python main.py --config path/to/custom.yaml --seed 789
```

## Requirements

- Python 3.10+
- PyTorch 2.1+
- PyTorch Geometric 2.4+
- XGBoost 2.0+
- scikit-learn 1.3+

Install everything:

```bash
pip install -r defi_cascade_predictor/requirements.txt
```

GPU recommended for TGN training. CPU works fine but expect longer runtimes (the code auto-adjusts epoch counts).

## Citation

Paper currently under review.

## License

All rights reserved.
