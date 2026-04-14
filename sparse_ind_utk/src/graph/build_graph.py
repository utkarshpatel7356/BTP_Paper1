"""
src/graph/build_graph.py
------------------------
Build the stock-index heterogeneous graph.

Nodes:
  0 .. N-1  : individual S&P 500 stocks
  N         : S&P 500 index sink node

Edges (stock-stock):
  - Rolling Pearson correlation  (weight = abs corr, threshold filtered)
  - Granger causality            (binary, p-value threshold) — intra-sector only
  - Same GICS sector             (binary weight = 1.0)

Edge (stock → index):
  - All stocks connect to the index sink node with weight = abs(beta)

Usage (standalone download):
  python src/graph/build_graph.py --download --config configs/config.yaml
"""

import argparse
import gc
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from statsmodels.tsa.stattools import grangercausalitytests
from torch_geometric.data import Data
from tqdm import tqdm

from src.graph.features import compute_index_node_features, compute_log_returns, compute_node_features


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def fetch_sp500_tickers() -> pd.DataFrame:
    """Scrape S&P 500 tickers and GICS sectors from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(
        url,
        storage_options={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    df = tables[0][["Symbol", "GICS Sector"]].copy()
    df.columns = ["ticker", "sector"]
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df


def download_prices(tickers: List[str], index_ticker: str, start: str, end: str, raw_dir: str) -> None:
    """Download adjusted close prices via yfinance and save to CSV."""
    import yfinance as yf

    os.makedirs(raw_dir, exist_ok=True)
    all_tickers = tickers + [index_ticker]
    print(f"Downloading {len(all_tickers)} tickers from {start} to {end}...")
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=True)["Close"]
    data.to_csv(os.path.join(raw_dir, "prices.csv"))
    print(f"Saved to {raw_dir}/prices.csv  shape={data.shape}")


# ---------------------------------------------------------------------------
# Edge construction helpers
# ---------------------------------------------------------------------------

def build_correlation_edges(
    returns: pd.DataFrame,
    window: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute correlation matrix using standard pairwise correlation (memory-efficient).
    Keeps edges above threshold.
    Returns edge_index (2, E) and edge_weight (E,) arrays.
    """
    n = returns.shape[1]
    # Use standard correlation instead of rolling().corr() to avoid O(T*N*N) memory
    # The rolling corr approach creates a MultiIndex DF of shape (T*N, N) which is ~375M entries
    corr_matrix = returns.corr().values  # (N, N) — simple, memory-efficient
    np.fill_diagonal(corr_matrix, 0)
    # Replace NaN with 0 (stocks with constant returns)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    mask = np.abs(corr_matrix) >= threshold
    src, dst = np.where(mask)
    weights = np.abs(corr_matrix[mask])
    return np.stack([src, dst]), weights


def build_granger_edges(
    returns: pd.DataFrame,
    maxlag: int,
    pvalue_threshold: float,
    sector_map: Dict[str, str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Granger causality: stock i → stock j if i's lags help predict j.
    Only tests intra-sector pairs to keep runtime manageable.
    Returns edge_index (2, E) and edge_weight (E,) arrays (weights = 1.0).
    """
    tickers = returns.columns.tolist()
    n = len(tickers)
    src_list, dst_list = [], []

    # Build sector groups for intra-sector testing only
    if sector_map is not None:
        sector_groups = {}
        for idx, t in enumerate(tickers):
            s = sector_map.get(t, "Unknown")
            sector_groups.setdefault(s, []).append(idx)
    else:
        # Fallback: treat all as one sector (full N² — not recommended)
        sector_groups = {"all": list(range(n))}

    total_pairs = sum(len(v) * (len(v) - 1) for v in sector_groups.values())
    print(f"  Testing {total_pairs} intra-sector pairs (vs {n*(n-1)} full N²)")

    tested = 0
    for sector_name, members in tqdm(sector_groups.items(), desc="Granger (by sector)"):
        for i in members:
            for j in members:
                if i == j:
                    continue
                try:
                    test_data = pd.concat(
                        [returns.iloc[:, j], returns.iloc[:, i]], axis=1
                    ).dropna()
                    if len(test_data) < maxlag + 2:
                        continue
                    result = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
                    min_p = min(result[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1))
                    if min_p < pvalue_threshold:
                        src_list.append(i)
                        dst_list.append(j)
                except Exception:
                    continue
                tested += 1
                # Periodic garbage collection to prevent memory buildup
                if tested % 500 == 0:
                    gc.collect()

    if not src_list:
        return np.empty((2, 0), dtype=np.int64), np.empty(0)

    edge_index = np.stack([src_list, dst_list])
    weights = np.ones(len(src_list))
    return edge_index, weights


def build_sector_edges(sector_map: Dict[str, str], tickers: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Connect all pairs within the same GICS sector."""
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    src_list, dst_list = [], []
    sectors = {}
    for t in tickers:
        s = sector_map.get(t, "Unknown")
        sectors.setdefault(s, []).append(ticker_to_idx[t])

    for members in sectors.values():
        for i in members:
            for j in members:
                if i != j:
                    src_list.append(i)
                    dst_list.append(j)

    if not src_list:
        return np.empty((2, 0), dtype=np.int64), np.empty(0)

    edge_index = np.stack([src_list, dst_list])
    weights = np.ones(len(src_list))
    return edge_index, weights


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

def build_graph(cfg: dict) -> Data:
    """
    Full pipeline: load prices → compute features → build edges → return PyG Data.

    The returned Data object has:
      x            : (N+1, T, F) node features (last node = index sink)
      edge_index   : (2, E)  stock-stock + stock-index edges
      edge_weight  : (E,)    edge weights
      y            : (T,)    index daily returns (prediction target)
      tickers      : list[str] of length N
    """
    dcfg = cfg["data"]
    gcfg = cfg["graph"]
    processed_dir = dcfg["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    # --- Load prices ---
    prices_path = os.path.join(dcfg["raw_dir"], "prices.csv")
    prices_full = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    index_prices = prices_full[dcfg["index_ticker"]]
    stock_prices = prices_full.drop(columns=[dcfg["index_ticker"]])

    # Drop columns with >5% missing
    stock_prices = stock_prices.dropna(thresh=int(0.95 * len(stock_prices)), axis=1)
    stock_prices = stock_prices.ffill().bfill()

    index_returns = compute_log_returns(index_prices.to_frame()).iloc[:, 0]

    # --- Node features ---
    print("Computing node features...")
    node_features, tickers = compute_node_features(stock_prices, index_returns)  # (T, N, F)
    idx_features = compute_index_node_features(index_returns)  # (T, F_idx)
    # Pad idx_features to same F dimension
    F = node_features.shape[-1]
    if idx_features.shape[1] < F:
        pad = np.zeros((idx_features.shape[0], F - idx_features.shape[1]), dtype=np.float32)
        idx_features = np.concatenate([idx_features, pad], axis=1)
    idx_features = idx_features[:, :F]

    # Shape: (N+1, T, F) — we store as (N+1, T, F) then use last T steps
    all_features = np.concatenate(
        [node_features.transpose(1, 0, 2), idx_features[np.newaxis]], axis=0
    )  # (N+1, T, F)

    # --- Edges ---
    returns_df = compute_log_returns(stock_prices)
    n = len(tickers)

    print("Building correlation edges...")
    ei_corr, ew_corr = build_correlation_edges(
        returns_df, gcfg["corr_window"], gcfg["corr_threshold"]
    )

    edge_indices = [ei_corr]
    edge_weights = [ew_corr]

    # Load sector info (needed for both sector edges and Granger)
    sector_df = fetch_sp500_tickers()
    sector_map = dict(zip(sector_df["ticker"], sector_df["sector"]))

    if gcfg.get("use_sector", True):
        print("Building sector edges...")
        ei_sec, ew_sec = build_sector_edges(sector_map, tickers)
        edge_indices.append(ei_sec)
        edge_weights.append(ew_sec)

    if gcfg.get("use_granger", False):
        print("Building Granger edges (intra-sector only)...")
        ei_gr, ew_gr = build_granger_edges(
            returns_df, gcfg["granger_maxlag"], gcfg["granger_pvalue"],
            sector_map=sector_map,
        )
        edge_indices.append(ei_gr)
        edge_weights.append(ew_gr)

    # Stock → index sink edges (weight = mean abs return correlation with index)
    idx_node = n  # index of the sink node
    corr_with_idx = returns_df.corrwith(index_returns).abs().values
    corr_with_idx = np.nan_to_num(corr_with_idx, nan=0.0)
    src_to_idx = np.arange(n)
    dst_to_idx = np.full(n, idx_node)
    ei_to_idx = np.stack([src_to_idx, dst_to_idx])
    ew_to_idx = corr_with_idx

    edge_indices.append(ei_to_idx)
    edge_weights.append(ew_to_idx)

    # Merge and deduplicate
    full_ei = np.concatenate(edge_indices, axis=1)
    full_ew = np.concatenate(edge_weights)

    # Deduplicate: keep max weight for duplicate edges
    edge_df = pd.DataFrame({"src": full_ei[0], "dst": full_ei[1], "w": full_ew})
    edge_df = edge_df.groupby(["src", "dst"])["w"].max().reset_index()
    final_ei = edge_df[["src", "dst"]].values.T
    final_ew = edge_df["w"].values

    # --- Assemble PyG Data ---
    data = Data(
        x=torch.tensor(all_features, dtype=torch.float32),           # (N+1, T, F)
        edge_index=torch.tensor(final_ei, dtype=torch.long),         # (2, E)
        edge_weight=torch.tensor(final_ew, dtype=torch.float32),     # (E,)
        y=torch.tensor(index_returns.values, dtype=torch.float32),   # (T,)
        num_nodes=n + 1,
    )
    data.tickers = tickers
    data.n_stocks = n

    # Save
    out_path = os.path.join(processed_dir, "graph_data.pt")
    torch.save(data, out_path)
    print(f"Graph saved → {out_path}")
    print(f"  Nodes: {n+1} ({n} stocks + 1 index)")
    print(f"  Edges: {final_ei.shape[1]}")

    gc.collect()
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--download", action="store_true", help="Download raw data first")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.download:
        sp500_df = fetch_sp500_tickers()
        tickers = sp500_df["ticker"].tolist()
        download_prices(
            tickers,
            cfg["data"]["index_ticker"],
            cfg["data"]["start_date"],
            cfg["data"]["end_date"],
            cfg["data"]["raw_dir"],
        )

    build_graph(cfg)
