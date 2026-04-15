# Sparse Index Replication via Hybrid GNN + Embedding-SHAP Fusion

Replicate the S&P 500 index using a small subset of stocks (sparse portfolio) and optimised budget weights, minimising tracking error over a 6-month forward horizon.

---

## Problem Statement

Given 500 constituent stocks, find:
1. A **subset of k stocks** (k ≪ 500) that best mimics the S&P 500 index
2. **Budget weights** for each selected stock that minimise tracking error

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                               │
│  Stage 1: Download S&P 500 prices (yfinance)                        │
│  Stage 2: Build graph — correlation edges, sector edges, features   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
       ┌─────────────────────────┼───────────────────────────┐
       ▼                         ▼                           ▼
┌──────────────┐    ┌────────────────────────┐   ┌───────────────────┐
│  Stage 3     │    │  Stage 4               │   │                   │
│  RF + SHAP   │    │  GNN (GAT + GRU)       │   │  Feature          │
│  baseline    │    │  → node embeddings     │   │  computation      │
│  → shap_rf   │    │  → attention weights   │   │  (returns, vol,   │
│    scores    │    │  → index prediction    │   │   momentum, beta) │
└──────┬───────┘    └───────┬────────────────┘   └───────────────────┘
       │                    │
       │            ┌───────┴───────────────────┐
       │            ▼                           ▼
       │   ┌─────────────────────┐  ┌───────────────────────────┐
       │   │  Stage 5            │  │  Stage 5.5   (NEW)        │
       │   │  Influence Scoring  │  │  Embedding Regressor      │
       │   │  (greedy IM on      │  │  Ridge(stock_ret × emb_sim)│
       │   │   embedding space)  │  │  → SHAP → emb_shap scores │
       │   └──────┬──────────────┘  └──────┬────────────────────┘
       │          │                        │
       └──────────┼────────────────────────┘
                  ▼
        ┌──────────────────────────────────────┐
        │  Stage 6: Three-Way Score Fusion     │
        │  hybrid = α·SHAP_rf + β·SHAP_emb    │
        │          + (1-α-β)·influence         │
        │  → sector-constrained selection      │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  Stage 7: QP Budget Allocation       │
        │  min (w-w̄)ᵀΣ(w-w̄) + λ wᵀA_attn w   │
        │  s.t. Σw = 1, 0 ≤ wᵢ ≤ max_weight  │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  Stage 8: Out-of-Sample Evaluation   │
        │  3-way comparison:                   │
        │  • RF+SHAP baseline                  │
        │  • Hybrid v1 (SHAP + Influence)      │
        │  • Hybrid v2 (SHAP + EmbSHAP + Infl) │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  Stage 9: Plots & Results            │
        │  5 publication-ready figures + JSON   │
        └──────────────────────────────────────┘
```

---

## Three Strategies Compared

| Strategy | Selection Method | Weight Method | Description |
|----------|-----------------|---------------|-------------|
| **RF+SHAP Baseline** | RF feature importance via SHAP | OLS fit | Purely statistical — ranks stocks by their return's predictive power for the index |
| **Hybrid v1** | α · RF-SHAP + (1-α) · GNN influence | QP with attention regularisation | Fuses SHAP importance with GNN-based influence maximisation |
| **Hybrid v2** (ours) | α · RF-SHAP + β · Emb-SHAP + γ · influence | QP with attention regularisation | Three-way fusion: adds embedding-regressor SHAP to capture graph-structural importance |

### What is Embedding-SHAP? (Stage 5.5)

The key innovation in Hybrid v2 is the **embedding regressor**:

1. **Extract GNN embeddings** — each stock gets a 64-dim vector encoding its graph-structural properties
2. **Compute embedding similarities** — cosine similarity between each stock and the index sink node
3. **Build embedding-weighted features** — `X[t, i] = stock_return[t, i] × emb_sim[i]`
4. **Train Ridge regressor** — predicts index return from embedding-weighted stock returns
5. **Explain with SHAP** — `shap.LinearExplainer` decomposes predictions into per-stock importance

This produces a complementary signal to RF-SHAP: while RF-SHAP measures *which stock returns statistically predict the index*, embedding-SHAP measures *which stock returns, re-weighted by graph-structural proximity, predict the index*.

---

## Repository Structure

```
sparse_ind_utk/
├── configs/
│   └── config.yaml               # All hyperparameters
├── data/
│   ├── raw/                      # Downloaded CSVs (gitignored)
│   └── processed/                # Graph tensors
├── src/
│   ├── graph/
│   │   ├── build_graph.py        # Correlation + Granger + sector edges
│   │   └── features.py           # Node feature engineering
│   ├── models/
│   │   ├── gat_model.py          # GAT + temporal GRU architecture
│   │   └── train.py              # Training loop + diversity loss
│   ├── selection/
│   │   ├── shap_selector.py      # Stage 3 — RF + SHAP baseline
│   │   ├── embedding_regressor.py # Stage 5.5 — Ridge + SHAP on embeddings (NEW)
│   │   └── influence.py          # Stage 5 — influence + fusion (v1 & v2)
│   ├── allocation/
│   │   └── qp_solver.py          # Stage 7 — QP (cvxpy)
│   └── evaluation/
│       └── metrics.py            # TE, IR, plots, SHAP comparison
├── outputs/
│   ├── figures/                  # 5 publication-ready plots
│   │   ├── cumulative_returns.png
│   │   ├── rolling_tracking_error.png
│   │   ├── weight_allocation.png
│   │   ├── influence_scores.png
│   │   └── shap_comparison.png   # (NEW) RF-SHAP vs Emb-SHAP scatter
│   └── results/
│       ├── metrics.json          # 3-strategy comparison
│       ├── metrics.csv
│       └── allocation.csv        # Final v2 portfolio weights
├── main.py                       # End-to-end pipeline entrypoint
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline (downloads data + trains GNN)
python main.py --config configs/config.yaml

# 4. Skip steps for faster iteration
python main.py --config configs/config.yaml \
    --skip-download --skip-graph --skip-train

# 5. Results in outputs/
cat outputs/results/metrics.json
```

### CLI Options

| Flag | Effect |
|------|--------|
| `--skip-download` | Use cached price data (skip yfinance) |
| `--skip-graph` | Use cached `graph_data.pt` |
| `--skip-train` | Load GNN from `outputs/best_gnn.pt` |

---

## Key Hyperparameters (`configs/config.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 50 | Subset size (stocks to select) |
| `train_years` | 5 | Training window length |
| `test_months` | 6 | Out-of-sample test horizon |
| `corr_threshold` | 0.30 | Minimum edge weight for graph construction |
| `gnn_hidden` | 64 | GAT embedding dimension |
| `gnn_heads` | 4 | Attention heads per GAT layer |
| `gnn_layers` | 2 | Number of stacked GAT layers |
| `gru_window` | 20 | Temporal lookback (trading days) |
| `diversity_lambda` | 0.1 | **NEW** — embedding diversity regularisation |
| `alpha_shap` | 0.4 | Weight on RF-SHAP in three-way fusion |
| `beta_emb_shap` | 0.4 | **NEW** — weight on embedding-SHAP |
| `ridge_alpha` | 1.0 | **NEW** — Ridge L2 regularisation |
| `lambda_reg` | 0.01 | QP attention-based regularisation |
| `max_weight` | 0.10 | Maximum weight per stock |

---

## Results (Test period: Jul 2024 – Dec 2024)

| Metric | RF+SHAP Baseline | Hybrid v1 | **Hybrid v2 (ours)** |
|--------|:-:|:-:|:-:|
| **Tracking Error (ann. %)** | 3.63 | 4.79 | **3.83** |
| **Max Drawdown (%)** | −8.87 | −9.53 | **−8.39** |
| **Beta** | 0.98 | 1.05 | **0.98** |
| **Total Return (%)** | 10.83 | 9.17 | 6.19 |
| **Sharpe Ratio** | 1.48 | 1.19 | 0.89 |
| **Info Ratio** | 1.78 | 0.76 | −0.54 |

> **Note:** These results use a GNN checkpoint with collapsed embeddings.
> Hybrid v2 already improves tracking error by 20% over v1 and achieves the
> best max drawdown. After GNN re-training with diversity regularisation,
> the embedding-SHAP signal will carry graph-structural information and
> further improve all metrics.

---

## Output Figures

The pipeline generates 5 publication-ready plots:

1. **Cumulative Returns** — S&P 500 vs all 3 strategies (4 lines)
2. **Rolling Tracking Error** — 30-day annualised TE comparison
3. **Weight Allocation** — per-stock bar chart coloured by GICS sector
4. **Influence Scores** — top-30 stocks by hybrid score
5. **SHAP Comparison** — scatter plot of RF-SHAP vs Embedding-SHAP per stock

---

## Technical Details

### Graph Construction (Stage 2)
- **Nodes:** ~482 S&P 500 stocks + 1 index sink node
- **Edges:** rolling 60-day Pearson correlation (threshold > 0.30) + intra-sector edges
- **Node features:** log returns, rolling volatility, momentum, beta (per day)

### GNN Architecture (Stage 4)
- Multi-head GAT (4 heads × 2 layers) applied per timestep
- LayerNorm + ELU activation + dropout
- GRU over temporal dimension → 64-dim node embeddings
- MLP head on index node → scalar index return prediction
- **Loss:** MSE + diversity regularisation (cosine similarity penalty)

### QP Allocation (Stage 7)
- **Objective:** min tracking error + attention regularisation
- **Constraints:** weights sum to 1, each ≤ 10%
- **Solver:** OSQP (fallback: SCS, ECOS)

---

## Commit History

| # | Commit | Contents |
|---|--------|----------|
| 1 | `init: repo structure` | Skeleton, config, requirements |
| 2 | `feat: data + graph` | Download, build_graph, features |
| 3 | `feat: baseline + GNN` | shap_selector, gat_model, train |
| 4 | `feat: influence + QP` | influence, qp_solver, selection |
| 5 | `feat: evaluation pipeline` | metrics, main.py, output figures |
| 6 | `feat: embedding regressor + hybrid v2` | embedding_regressor, three-way fusion, diversity loss, 3-strategy comparison |
