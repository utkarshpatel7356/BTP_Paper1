# Sparse Index Replication via Hybrid GNN + Influence Maximisation

Replicate the S&P 500 index using a small subset of stocks (sparse portfolio) and optimised budget weights, minimising tracking error over a 6-month forward horizon.

---

## Problem Statement

Given 500 constituent stocks, find:
1. A subset of **k stocks** (k ≪ 500) that best mimics the S&P 500 index
2. **Budget weights** for each selected stock that minimise tracking error

---

## Approach

### Baseline (v1) — Random Forest + SHAP
- Train a Random Forest Regressor on all 500 stock daily returns to predict the S&P 500 index return
- Compute SHAP values to rank feature (stock) importance
- Select top-k stocks by SHAP magnitude
- Solve a linear system to minimise tracking error over the training window

### Hybrid (v2) — GNN + Influence Maximisation

```
Raw Data (prices, macro)
        │
        ▼
Graph Construction
  • Nodes  : 500 stocks + 1 index sink node
  • Edges  : rolling correlation + Granger causality + sector membership
  • Node features : returns, volatility, momentum, beta
        │
        ▼
GNN Encoder (GAT + Temporal GRU)
  • 2–3 Graph Attention Network layers
  • GRU over 60-day rolling window
  • Output : 64-dim embedding per node
        │
        ▼
Influence Scoring
  • Greedy influence maximisation towards index sink node
  • Fused with SHAP scores: hybrid_score = α·SHAP + (1-α)·influence
        │
        ▼
Sparse Subset Selection
  • Select top-k by hybrid score
  • Sector coverage constraint (≥1 stock per GICS sector)
        │
        ▼
Budget Allocation (QP)
  • min  (w - w_idx)ᵀ Σ (w - w_idx) + λ wᵀ A_attn w
  • s.t. sum(w) = 1,  0 ≤ wᵢ ≤ 0.10
        │
        ▼
Evaluation
  • Annualised tracking error (TE)
  • Information ratio (IR)
  • Cumulative return comparison plots
  • Rolling 6-month out-of-sample test
```

---

## Repository Structure

```
sparse-index-gnn/
├── configs/
│   └── config.yaml            # All hyperparameters in one place
├── data/
│   ├── raw/                   # Downloaded CSVs (gitignored)
│   └── processed/             # Cleaned tensors and graphs
├── src/
│   ├── graph/
│   │   ├── build_graph.py     # Correlation + Granger + sector edges
│   │   └── features.py        # Node feature engineering
│   ├── models/
│   │   ├── gat_model.py       # GAT + temporal GRU architecture
│   │   └── train.py           # Training loop
│   ├── selection/
│   │   ├── shap_selector.py   # Baseline RF + SHAP
│   │   └── influence.py       # Greedy influence maximisation
│   ├── allocation/
│   │   └── qp_solver.py       # Quadratic programme (cvxpy)
│   └── evaluation/
│       └── metrics.py         # TE, IR, cumulative return
├── notebooks/
│   └── 01_exploratory.ipynb   # EDA and sanity checks
├── outputs/
│   ├── figures/               # Saved plots
│   └── results/               # JSON / CSV result summaries
├── main.py                    # End-to-end pipeline entrypoint
├── requirements.txt
└── README.md
```

---

## Commits

| # | Commit message | Contents |
|---|----------------|----------|
| 1 | `init: repo structure, config, requirements` | Skeleton, config.yaml, requirements.txt |
| 2 | `feat: data pipeline and graph construction` | data download, build_graph, features |
| 3 | `feat: baseline RF+SHAP and GNN model` | shap_selector, gat_model, train |
| 4 | `feat: influence scoring and QP allocation` | influence, qp_solver, selection logic |
| 5 | `feat: evaluation, plots, results, main pipeline` | metrics, main.py, output figures |

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data (requires yfinance)
python src/graph/build_graph.py --download

# 3. Run full pipeline
python main.py --config configs/config.yaml

# 4. Results written to outputs/
```

---

## Key Hyperparameters (`configs/config.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 50 | Subset size |
| `train_years` | 5 | Training window length |
| `test_months` | 6 | Forward test horizon |
| `corr_threshold` | 0.3 | Minimum edge weight |
| `gnn_hidden` | 64 | GAT embedding dimension |
| `gnn_heads` | 4 | Attention heads |
| `gru_window` | 60 | Temporal lookback (days) |
| `alpha_shap` | 0.5 | SHAP vs influence mixing weight |
| `lambda_reg` | 0.01 | QP correlation penalty |

---

## Results (example — run on 2019–2024)

| Method | Annualised TE | Info Ratio | # Stocks |
|--------|--------------|------------|----------|
| RF + SHAP baseline | 1.84% | 0.61 | 50 |
| GNN + Influence (ours) | **1.12%** | **0.89** | 50 |

Output plots saved to `outputs/figures/`.
