# Code Walkthrough — Sparse Index Replication Pipeline

A file-by-file description of every source file in the project: what it does, how it works, key functions, and how to run the full system

---

## Table of Contents

1. [Repository Layout](#1-repository-layout)
2. [Configuration](#2-configuration--configsconfigyaml)
3. [Data Ingestion & Graph Construction](#3-data-ingestion--graph-construction)
4. [GNN Model Architecture](#4-gnn-model-architecture)
5. [GNN Training Loop](#5-gnn-training-loop)
6. [Stock Selection Modules](#6-stock-selection-modules)
7. [Portfolio Weight Allocation](#7-portfolio-weight-allocation)
8. [Evaluation & Plotting](#8-evaluation--plotting)
9. [Pipeline Entrypoints](#9-pipeline-entrypoints)
10. [How to Run Everything](#10-how-to-run-everything)

---

## 1. Repository Layout

```
sparse_ind_utk/
├── configs/
│   └── config.yaml                  # All hyperparameters (single source of truth)
├── data/
│   ├── raw/prices.csv               # Downloaded stock prices (gitignored)
│   └── processed/graph_data.pt      # Cached PyG graph (gitignored)
├── src/
│   ├── __init__.py
│   ├── graph/
│   │   ├── build_graph.py           # Download data + construct graph
│   │   └── features.py              # Node feature engineering
│   ├── models/
│   │   ├── gat_model.py             # GAT + GRU neural network architecture
│   │   └── train.py                 # Training loop with diversity regularisation
│   ├── selection/
│   │   ├── shap_selector.py         # RF + SHAP baseline stock ranking
│   │   ├── influence.py             # Greedy influence maximisation + fusion
│   │   └── embedding_regressor.py   # Ridge on embedding-weighted returns + SHAP
│   ├── allocation/
│   │   └── qp_solver.py             # QP for tracking-error-minimising weights
│   └── evaluation/
│       └── metrics.py               # Metrics computation + publication-ready plots
├── main.py                          # Primary pipeline (3-strategy comparison)
├── timeline.py                      # Ablation study (6-strategy comparison)
├── timeline_plots.py                # Plotting helpers for timeline.py
├── outputs/
│   ├── best_gnn.pt                  # Saved GNN checkpoint
│   ├── train_losses.npy             # Training loss curve
│   ├── val_losses.npy               # Validation loss curve
│   ├── figures/                     # Plots from main.py
│   ├── results/                     # Metrics from main.py
│   └── timeline/                    # Ablation study outputs
│       ├── figures/                 # 10 ablation plots
│       ├── results/                 # Ablation metrics JSON/CSV
│       └── README.md                # Full methodology documentation
├── notebooks/
│   └── 01_exploratory.ipynb         # Exploratory data analysis
├── requirements.txt
└── README.md
```

---

## 2. Configuration — `configs/config.yaml`

**Purpose**: Single YAML file that controls every hyperparameter in the pipeline. All modules read from this config — nothing is hardcoded.

### Sections

```yaml
data:
  index_ticker: "^GSPC"          # S&P 500 ticker symbol for yfinance
  start_date: "2019-01-01"       # Training data start
  end_date: "2024-12-31"         # Data end (test period ends here)
  train_years: 5                 # Training window = 5 years
  test_months: 6                 # Test window = 6 months (~126 days)
```

- `train_years` and `test_months` determine the train/test split. The last `test_months × 21` trading days are held out.

```yaml
graph:
  corr_window: 60                # Rolling correlation window (days)
  corr_threshold: 0.30           # Minimum |correlation| to create an edge
  use_granger: false             # Granger edges (slow, disabled by default)
  use_sector: true               # Intra-sector edges
```

- `corr_threshold` controls graph density. Lower = more edges = slower GAT but richer structure.

```yaml
model:
  gnn_hidden: 64                 # Node embedding dimension
  gnn_heads: 4                   # GAT attention heads
  gnn_layers: 2                  # Stacked GAT layers
  gru_window: 20                 # Temporal lookback (trading days)
  diversity_lambda: 0.1          # Embedding diversity regularisation strength
  epochs: 100
```

- `gru_window: 20` was reduced from 60 to fit in Tesla T4 GPU memory (16 GB).
- `diversity_lambda: 0.1` penalises embedding collapse (mean cosine similarity > 0.1 margin).

```yaml
selection:
  k: 50                          # Number of stocks to select
  alpha_shap: 0.4                # Weight on RF-SHAP in fusion
  beta_emb_shap: 0.4             # Weight on Embedding-SHAP
  ridge_alpha: 1.0               # Ridge regularisation for embedding regressor
```

- `alpha + beta = 0.8`, so influence gets `gamma = 0.2`.

```yaml
allocation:
  lambda_reg: 0.01               # Attention regularisation in QP
  max_weight: 0.10               # Max 10% in any single stock
  solver: "OSQP"                 # QP solver (fallback: SCS, ECOS)
```

---

## 3. Data Ingestion & Graph Construction

### `src/graph/features.py`

**Purpose**: Engineer per-stock node features from raw prices.

#### `compute_log_returns(prices)`
- Input: DataFrame of adjusted close prices
- Output: DataFrame of log returns (`log(P_t / P_{t-1})`)
- Drops the first row (NaN from differencing)

#### `compute_node_features(prices, index_returns)`
Builds a `(T, N, 5)` feature tensor:

| Index | Feature | Computation |
|-------|---------|-------------|
| 0 | daily_return | `log(P_t / P_{t-1})` |
| 1 | volatility_20 | `returns.rolling(20).std()` |
| 2 | momentum_5 | `returns.rolling(5).sum()` |
| 3 | momentum_20 | `returns.rolling(20).sum()` |
| 4 | beta | `cov(stock, index) / var(index)` over 60-day rolling window |

All features are z-normalised: `(x - mean) / std` across the time axis.

#### `compute_index_node_features(index_returns)`
Features for the S&P 500 sink node: just `[return, volatility]`, padded to match stock feature dimension.

---

### `src/graph/build_graph.py`

**Purpose**: Download price data, construct the heterogeneous graph, save as PyTorch Geometric `Data` object.

#### `fetch_sp500_tickers()`
- Scrapes Wikipedia's S&P 500 constituent table
- Returns DataFrame with `ticker` and `sector` (GICS) columns
- Handles ticker format (e.g., `BRK.B` → `BRK-B` for yfinance)

#### `download_prices(tickers, index_ticker, start, end, raw_dir)`
- Uses `yfinance.download()` to fetch adjusted close prices
- Saves to `data/raw/prices.csv`

#### `build_correlation_edges(returns, window, threshold)`
- Computes pairwise Pearson correlation of stock returns
- Keeps edges where `|corr| >= threshold` (default 0.30)
- Returns `edge_index (2, E)` and `edge_weight (E,)` arrays
- Memory-efficient: uses `returns.corr()` instead of rolling correlation

#### `build_granger_edges(returns, maxlag, pvalue_threshold, sector_map)`
- Tests Granger causality only for **intra-sector** pairs (reduces N² to ~10K tests)
- Uses `statsmodels.grangercausalitytests` with F-test
- An edge `i → j` means stock i's past returns help predict stock j
- Disabled by default (`use_granger: false`) because it's slow

#### `build_sector_edges(sector_map, tickers)`
- Connects all stock pairs within the same GICS sector
- Weight = 1.0 (binary)

#### `build_graph(cfg)` — Main function
Orchestrates the full graph construction:

1. Load `prices.csv`, drop stocks with >5% missing data
2. Compute node features → `(N+1, T, F)` tensor
3. Build edges: correlation + sector + (optional) Granger + stock→index
4. Deduplicate edges (keep max weight)
5. Package into PyG `Data` object with attributes:
   - `x`: `(N+1, T, F)` — node features
   - `edge_index`: `(2, E)` — edge connections
   - `edge_weight`: `(E,)` — edge weights
   - `y`: `(T,)` — index returns (training target)
   - `tickers`: list of stock symbols
   - `n_stocks`: N (number of stocks, excluding index node)
6. Save to `data/processed/graph_data.pt`

---

## 4. GNN Model Architecture

### `src/models/gat_model.py`

**Purpose**: Define the Graph Attention Network + Temporal GRU architecture.

#### `GATTemporalEncoder`

```
Input: x (N+1, T, F) — node features over time
                ↓
    ┌───────────────────────┐
    │ For each timestep t:  │
    │   h_t = GAT(x_t)     │  ← 2 stacked GATConv layers
    │   h_t: (N+1, hidden)  │    4 attention heads per layer
    └───────┬───────────────┘    LayerNorm + ELU + Dropout
            ↓
    Stack h_t over window W
    H: (N+1, W, hidden)
            ↓
    ┌───────────────────────┐
    │ GRU over time dim     │  ← 1-layer GRU
    │ embeddings = H[:,-1]  │    Takes last hidden state
    └───────┬───────────────┘
            ↓
    embeddings: (N+1, hidden)  ← 64-dim per node
```

**GAT layer details**:
- Layer 1: `F → hidden//heads` per head, concat heads → `hidden`
- Layer 2: `hidden → hidden` per head, mean over heads → `hidden`
- Both layers use `add_self_loops=True` (GATConv default)
- Last layer returns attention weights `(E', heads)` for downstream use

**Key design choice**: The GRU processes each node's temporal sequence independently. The GAT handles spatial (cross-stock) interactions, while the GRU handles temporal patterns.

#### `SparseIndexGNN`

Wraps the encoder with a prediction head:

```
embeddings (N+1, hidden)
        ↓
    embeddings[-1]           ← index sink node embedding
        ↓
    Linear(64 → 32) + ELU + Dropout
        ↓
    Linear(32 → 1)           ← predicted daily index return
```

Returns: `(y_hat, embeddings, attn_weights, attn_edge_index)`

---

## 5. GNN Training Loop

### `src/models/train.py`

**Purpose**: Train `SparseIndexGNN` on the rolling-window dataset.

#### `_embedding_diversity_loss(embeddings, margin=0.1)`
Regularisation to prevent all node embeddings from collapsing to the same vector:

```python
sim_matrix = normalised_embeddings @ normalised_embeddings.T
mean_sim = sim_matrix[off_diagonal].mean()
loss = ReLU(mean_sim - margin)  # penalise only if mean_sim > 0.1
```

#### `WindowDataset`
Memory-efficient dataset that slices temporal windows on-the-fly:
- Stores the full `(N+1, T, F)` tensor once
- `__getitem__` returns `x[:, t-W:t, :]` — a view, not a copy
- Avoids pre-materialising all windows (which would be ~66 GB)

#### `train_model(graph_data, cfg)`
Training loop:

1. Split data 90/10 into train/val
2. Create `WindowDataset` for each split
3. For each epoch:
   - Forward pass: iterate over batch, each sample is `(N+1, W, F)`
   - Loss = `MSE(predicted_index_return, actual_index_return) + λ_div × diversity_loss`
   - Mixed precision (AMP) for GPU memory efficiency
   - Gradient clipping (max norm 1.0)
4. `ReduceLROnPlateau` scheduler (patience=5, factor=0.5)
5. Save best checkpoint by validation loss

**Memory optimisations** (for Tesla T4 with 16 GB):
- `torch.cuda.empty_cache()` between batches
- Mixed precision (`autocast` + `GradScaler`)
- `gru_window=20` instead of 60

---

## 6. Stock Selection Modules

### `src/selection/shap_selector.py` — RF+SHAP Baseline

#### `run_shap_selection(stock_returns, index_returns, tickers, k=50)`

**Step-by-step**:

1. **Train Random Forest**: `RandomForestRegressor(n_estimators=200, max_depth=8)`
   - X = stock returns `(T, N)` — each column is a stock
   - y = index returns `(T,)` — daily S&P 500 return
   - The RF learns which combination of stock returns predict the index

2. **Compute SHAP**: `TreeExplainer` with subsampled background (100 rows)
   - Explains 200 random rows
   - SHAP values shape: `(200, N)` — how much each stock contributed to each prediction

3. **Aggregate**: `shap_scores[i] = mean(|SHAP_values[:, i]|)` — average absolute importance

4. **Rank and select**: Top-k stocks by SHAP score

**Returns**: `(selected_tickers, shap_scores, rf_model)`

---

### `src/selection/influence.py` — GNN Influence + Fusion

#### `greedy_influence_maximisation(embeddings, k, tickers)`

Selects k stocks by greedy influence maximisation in embedding space:

```
Initialise: S = {}, aggregate = zero vector

For round r = 1 to k:
    For each candidate stock i not in S:
        new_aggregate = mean(embeddings[S ∪ {i}])
        gain[i] = cosine_sim(new_aggregate, index_embedding) - current_sim
    Add i* = argmax(gain) to S
    Update aggregate
```

**Intuition**: Find the subset whose mean embedding is most aligned with the index node's embedding. This selects stocks that are, collectively, structurally representative of the index.

**Complexity**: O(k × N) cosine similarity computations.

#### `fuse_scores(shap_scores, influence_scores, alpha)`
Two-way linear combination (Hybrid v1):
```
hybrid = α × norm(SHAP) + (1-α) × norm(Influence)
```
where `norm` is min-max normalisation to [0, 1].

#### `fuse_scores_v2(shap_rf, shap_emb, influence_scores, alpha, beta)`
Three-way fusion (Hybrid v2):
```
hybrid = α × norm(SHAP_rf) + β × norm(SHAP_emb) + (1-α-β) × norm(Influence)
```

#### `select_with_sector_constraint(hybrid_scores, tickers, sector_map, k, min_per_sector)`
Two-phase greedy selection:
1. **Phase 1**: Guarantee `min_per_sector` stocks from each GICS sector (picks best-scored stock per sector)
2. **Phase 2**: Fill remaining budget with top-scored stocks regardless of sector

---

### `src/selection/embedding_regressor.py` — Embedding-SHAP

#### `train_embedding_regressor(embeddings, train_stock_returns, train_index_returns, tickers)`

**Step-by-step**:

1. **Compute cosine similarities**:
   ```
   emb_sim[i] = cosine_sim(stock_embedding[i], index_embedding)
   ```
   Higher similarity = stock's graph structure looks more like the index.

2. **Build embedding-weighted features**:
   ```
   X[t, i] = stock_return[t, i] × emb_sim[i]
   ```
   This amplifies returns of structurally-important stocks and dampens peripheral ones.

3. **Standardise**: `StandardScaler` for stable Ridge regression.

4. **Train Ridge regressor**: `Ridge(alpha=1.0)` predicting index returns from weighted features.

5. **SHAP decomposition**: `LinearExplainer` (exact for linear models — no approximation).
   ```
   emb_shap_scores[i] = mean(|SHAP_values[:, i]|)
   ```

**Why Ridge + LinearExplainer?**
- Ridge is closed-form, fast, and stable in the N > T regime
- LinearExplainer gives exact Shapley values (not approximate like TreeExplainer)
- The combination is orders of magnitude faster than RF + TreeExplainer

---

## 7. Portfolio Weight Allocation

### `src/allocation/qp_solver.py`

#### `solve_tracking_error_qp(selected_returns, index_returns, attention_matrix, ...)`

Solves a Quadratic Program to find optimal portfolio weights:

**Objective**:
```
minimise   (w - w̄)ᵀ Σ (w - w̄)  +  λ × wᵀ A_attn w
```

| Term | Meaning |
|------|---------|
| `(w - w̄)ᵀ Σ (w - w̄)` | Tracking error — deviation from index |
| `Σ` | Covariance matrix of excess returns (stock - index) |
| `w̄` | Index proxy weights (equal weight 1/k) |
| `λ × wᵀ A_attn w` | Attention regularisation — penalises concentrating in GAT-connected pairs |
| `A_attn` | Sub-matrix of GNN attention weights for selected stocks |

**Constraints**:
- `Σ wᵢ = 1` (fully invested)
- `0 ≤ wᵢ ≤ 0.10` (no stock gets more than 10%)

**Solver cascade**: OSQP → SCS → ECOS (automatic fallback if a solver fails).

**Implementation details**:
- `_nearest_psd(A)`: Projects Σ to nearest positive semi-definite matrix (eigenvalue clipping)
- Adds `1e-6 × I` to Σ for numerical stability
- If all solvers fail, falls back to equal weights with a warning

#### `build_attention_submatrix(attn_weights, edge_index, selected_indices, n_total)`
Extracts the K×K sub-matrix of GNN attention weights for just the selected stocks:
- Iterates over all edges in the attention output
- Maps source/destination node indices to the selected subset
- Returns a dense `(K, K)` matrix

---

## 8. Evaluation & Plotting

### `src/evaluation/metrics.py`

#### `compute_metrics(portfolio_returns, index_returns, label)`
Computes the full suite of tracking metrics:

| Metric | Computation |
|--------|-------------|
| Tracking Error | `std(excess) × √252 × 100` |
| Information Ratio | `mean(excess) × 252 / TE` |
| Sharpe Ratio | `mean(returns) / std(returns) × √252` |
| Beta | `cov(portfolio, index)[0,1] / var(index)` |
| Max Drawdown | `min((cum_returns - running_max) / running_max)` |
| Total Return | `(∏(1 + r_t) - 1) × 100` |

#### Plotting functions
All plots use matplotlib with a consistent publication-ready style:
- Spines removed (top, right)
- Grid: dashed, alpha=0.3
- DPI: 150
- Colour palette: blue (index), amber (baseline), green (hybrid), purple (v1)

| Function | Output |
|----------|--------|
| `plot_cumulative_returns()` | Cumulative return curves (rebased to 100) |
| `plot_tracking_error_rolling()` | 30-day rolling TE with filled band |
| `plot_weight_allocation()` | Bar chart of weights coloured by GICS sector |
| `plot_loss_curves()` | GNN train/val loss over epochs |
| `plot_influence_scores()` | Top-30 horizontal bar chart |
| `plot_shap_comparison()` | RF-SHAP vs Emb-SHAP scatter plot |

---

### `timeline_plots.py`

**Purpose**: Additional plotting functions for the ablation study. Separated from `metrics.py` to avoid modifying existing code.

| Function | Output |
|----------|--------|
| `plot_score_distributions()` | 3-panel histogram of RF-SHAP, Influence, Emb-SHAP scores |
| `plot_top30_bars()` | 3-panel horizontal bar charts of top-30 stocks per signal |
| `plot_selection_overlap()` | 6×6 Jaccard similarity heatmap between strategy selections |
| `plot_cumulative_ladder()` | All 6 strategies + index on one cumulative return chart |
| `plot_rolling_te_ladder()` | All 6 strategies rolling TE |
| `plot_metrics_bars()` | 2×2 grouped bar chart (TE, Sharpe, Beta, MaxDD) |
| `plot_weight_boxplots()` | Box plots of weight distributions per strategy |
| `plot_sector_comparison()` | Stacked bar chart of sector counts per strategy |
| `plot_alpha_beta_sweep()` | Heatmap of TE vs fusion weights α, β |
| `plot_summary_table()` | matplotlib-rendered comparison table |

---

## 9. Pipeline Entrypoints

### `main.py` — Primary Pipeline

Runs the full 9-stage pipeline end-to-end:

```
Stage 1: Download prices (yfinance)           [--skip-download to skip]
Stage 2: Build graph (correlation + sector)    [--skip-graph to skip]
Stage 3: RF + SHAP baseline                   
Stage 4: Train GNN (100 epochs)               [--skip-train to skip]
Stage 5: Influence scoring (greedy IM)
Stage 5.5: Embedding regressor + SHAP
Stage 6: Three-way fusion + sector selection
Stage 7: QP budget allocation (v1 and v2)
Stage 8: Test evaluation (3 strategies)
Stage 9: Save plots and metrics
```

**Outputs**: `outputs/figures/` (6 plots) + `outputs/results/` (metrics.json, allocation.csv)

**Strategies compared**: RF+SHAP baseline, Hybrid v1, Hybrid v2

---

### `timeline.py` — Ablation Study

Runs 6 strategies independently to tell the progression story:

```
1. Load cached data, graph, GNN checkpoint
2. Compute all 3 signals (RF-SHAP, Influence, Emb-SHAP)
3. Run 6 strategies:
   ├── Strategy 1: RF+SHAP Only (OLS weights)
   ├── Strategy 2: GNN Influence Only (equal weights)
   ├── Strategy 3: Emb-SHAP Only (OLS weights)
   ├── Strategy 4: Hybrid v1 — two-way fusion (QP, no attention)
   ├── Strategy 5: Hybrid v2 — three-way fusion (QP, no attention)
   └── Strategy 6: Hybrid v2 + Attention-regularised QP
4. Generate 10 comparison figures
5. Run α-β sensitivity sweep
6. Save all results
```

**Outputs**: `outputs/timeline/figures/` (10 plots) + `outputs/timeline/results/` (JSON + CSV)

---

## 10. How to Run Everything

### Prerequisites

```bash
# Python 3.9+ recommended
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Key dependencies**: `torch`, `torch_geometric`, `sklearn`, `shap`, `cvxpy`, `yfinance`, `matplotlib`, `seaborn`, `pandas`, `numpy`, `pyyaml`, `statsmodels`

### Run the Full Pipeline (First Time)

```bash
# Downloads data, builds graph, trains GNN, evaluates 3 strategies
python main.py --config configs/config.yaml
```

**Expected runtime**: ~30-60 min (mostly GNN training, 100 epochs on T4 GPU)

**Outputs**:
- `data/raw/prices.csv` — downloaded stock prices
- `data/processed/graph_data.pt` — cached graph
- `outputs/best_gnn.pt` — trained GNN checkpoint
- `outputs/figures/` — 6 publication-ready plots
- `outputs/results/metrics.json` — 3-strategy comparison

### Run with Cached Data (Faster Iteration)

```bash
# Skip download + graph + training (uses cached files)
python main.py --config configs/config.yaml \
    --skip-download --skip-graph --skip-train
```

**Expected runtime**: ~3-5 min (SHAP + influence + QP only)

### Run the Ablation Study

```bash
# Must have already run main.py at least once (needs cached data + checkpoint)
python timeline.py --config configs/config.yaml \
    --skip-download --skip-graph --skip-train
```

**Expected runtime**: ~5-8 min

**Outputs**:
- `outputs/timeline/figures/` — 10 ablation plots
- `outputs/timeline/results/timeline_metrics.json` — 6-strategy results
- `outputs/timeline/results/alpha_beta_sweep.json` — sensitivity grid

### Skip Flags Reference

| Flag | What It Skips | When to Use |
|------|---------------|-------------|
| `--skip-download` | yfinance price download | Already have `data/raw/prices.csv` |
| `--skip-graph` | Graph construction | Already have `data/processed/graph_data.pt` |
| `--skip-train` | GNN training (100 epochs) | Already have `outputs/best_gnn.pt` |

### Changing Hyperparameters

Edit `configs/config.yaml` and re-run. Key parameters to experiment with:

```yaml
# Try different subset sizes
selection:
  k: 30   # or 70, 100

# Try different fusion weights
selection:
  alpha_shap: 0.6      # increase RF-SHAP weight
  beta_emb_shap: 0.4   # keep Emb-SHAP
  # influence gets 0.0

# Try different graph density
graph:
  corr_threshold: 0.20  # denser graph (more edges)
```

### Verifying Results

```bash
# Check metrics
cat outputs/results/metrics.json
cat outputs/timeline/results/timeline_metrics.json

# View figures
ls -la outputs/figures/
ls -la outputs/timeline/figures/
```

### Common Issues

| Issue | Fix |
|-------|-----|
| CUDA OOM during GNN training | Reduce `gru_window` (e.g., 10) or set `device: "cpu"` in config |
| yfinance download fails | Check network; re-run without `--skip-download` |
| QP solver fails | OSQP → SCS → ECOS fallback is automatic; if all fail, check data quality |
| SHAP slow on RF | Reduce `shap_background_size` and `shap_explain_size` in `shap_selector.py` |
| Wikipedia scraping fails | `fetch_sp500_tickers()` may need User-Agent update if Wikipedia blocks |

---

## Data Flow Diagram

```
prices.csv ──→ compute_log_returns() ──→ returns (T, N)
                                              │
                    ┌─────────────────────────┤
                    ▼                         ▼
            compute_node_features()    build_correlation_edges()
            → (T, N, 5) features      → edge_index, edge_weight
                    │                         │
                    ▼                         ▼
            graph_data.pt ◄── build_graph() ──┘
                    │
        ┌───────────┼───────────────────┐
        ▼           ▼                   ▼
  run_shap_selection()   train_model()   (features used in all stages)
  → shap_scores (N,)     → model, embeddings (N+1, 64)
        │                       │
        │               ┌──────┤
        │               ▼      ▼
        │     greedy_influence_max()   train_embedding_regressor()
        │     → influence_scores       → emb_shap_scores
        │               │                    │
        └───────────────┼────────────────────┘
                        ▼
              fuse_scores_v2(α, β)
              → hybrid_scores (N,)
                        │
                        ▼
              select_with_sector_constraint()
              → selected_tickers (k,)
                        │
                        ▼
              solve_tracking_error_qp()
              → weights (k,)
                        │
                        ▼
              compute_metrics()
              → {TE, IR, Sharpe, Beta, MaxDD, Return}
```
