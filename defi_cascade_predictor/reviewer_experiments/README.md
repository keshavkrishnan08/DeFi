# Reviewer Experiment Scripts

Self-contained scripts to reproduce every result in the paper.
Each script loads real TVL data, trains models, and saves metrics to `outputs/`.

## Requirements

```bash
pip install torch numpy scikit-learn pyyaml xgboost pandas loguru
```

GPU recommended but not required. CPU runs use reduced epochs with early stopping.

## Scripts

| Script | What it reproduces | Runtime (CPU / GPU) |
|--------|-------------------|-------------------|
| `01_tgn_multiseed.py` | Table III TGN row (5 seeds, ensemble) | ~2.5h / ~40min |
| `02_xgboost_baseline.py` | Table III XGBoost row | ~5min |
| `03_all_baselines.py` | Table III Static GNN, LSTM, Centrality rows | ~1.5h / ~30min |
| `04_tvl_ablation.py` | TVL-only ablation (Fig. 7) | ~1h / ~20min |
| `05_graph_ablation.py` | Graph structure ablation | ~1h / ~20min |
| `06_memory_ablation.py` | Memory ablation (Table IV) | ~40min / ~15min |

## Usage

Run from the `defi_cascade_predictor/` directory:

```bash
cd /path/to/defi_cascade_predictor
python reviewer_experiments/01_tgn_multiseed.py
python reviewer_experiments/02_xgboost_baseline.py
# etc.
```

Results are saved as JSON in `outputs/reviewer_*.json`.

## Data

All scripts use real TVL data from DeFiLlama (in `data/real/tvl_combined.csv`).
No synthetic data is used. Cascade labels are derived from aggregate TVL drawdowns
using z-score thresholds (z < -4.5, drawdown > 20%).
