"""
src/graph/features.py
---------------------
Engineer per-stock node features from raw price data.

Features per stock (each as a T-length time series):
  - daily_return       : log return
  - volatility_20      : 20-day rolling std of returns
  - momentum_5         : 5-day cumulative return
  - momentum_20        : 20-day cumulative return
  - beta               : rolling 60-day beta vs index
  - volume_zscore      : z-scored volume (if available)
"""

import numpy as np
import pandas as pd
from typing import Tuple


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns from adjusted close prices."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_node_features(
    prices: pd.DataFrame,
    index_returns: pd.Series,
    vol_window: int = 20,
    mom_windows: Tuple[int, int] = (5, 20),
    beta_window: int = 60,
) -> np.ndarray:
    """
    Build node feature matrix of shape (T, N, F).

    Parameters
    ----------
    prices        : DataFrame (T x N), adjusted close prices, columns = tickers
    index_returns : Series (T,), S&P 500 log returns
    vol_window    : rolling window for volatility
    mom_windows   : (short, long) momentum windows
    beta_window   : rolling window for beta calculation

    Returns
    -------
    features : ndarray (T, N, F) where F = number of features
    tickers  : list of ticker strings (length N)
    """
    returns = compute_log_returns(prices)
    # Align index returns to the same dates
    idx_ret = index_returns.reindex(returns.index).fillna(0)

    T, N = returns.shape
    tickers = returns.columns.tolist()

    # --- per-feature computation ---
    # 1. Daily return
    feat_ret = returns.values  # (T, N)

    # 2. Rolling volatility
    feat_vol = returns.rolling(vol_window).std().bfill().values

    # 3. Momentum (short and long)
    short_w, long_w = mom_windows
    feat_mom5 = returns.rolling(short_w).sum().fillna(0).values
    feat_mom20 = returns.rolling(long_w).sum().fillna(0).values

    # 4. Rolling beta vs index
    feat_beta = np.zeros((T, N))
    for t in range(beta_window, T):
        window_stocks = returns.iloc[t - beta_window : t].values  # (W, N)
        window_idx = idx_ret.iloc[t - beta_window : t].values      # (W,)
        var_idx = np.var(window_idx) + 1e-8
        cov = np.cov(window_stocks.T, window_idx)[:-1, -1]         # (N,)
        feat_beta[t] = cov / var_idx
    # backfill the initial window
    feat_beta[:beta_window] = feat_beta[beta_window]

    # Stack to (T, N, F)
    features = np.stack(
        [feat_ret, feat_vol, feat_mom5, feat_mom20, feat_beta], axis=-1
    )  # (T, N, 5)

    # Normalise each feature across time dimension
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    features = (features - mean) / std

    return features.astype(np.float32), tickers


def compute_index_node_features(
    index_returns: pd.Series,
    vix: pd.Series = None,
    rates: pd.Series = None,
    vol_window: int = 20,
) -> np.ndarray:
    """
    Features for the S&P 500 sink node: (T, F_idx).
    Falls back to just index returns + vol if macro series not provided.
    """
    ret = index_returns.values.reshape(-1, 1)
    vol = pd.Series(index_returns).rolling(vol_window).std().bfill().values.reshape(-1, 1)
    parts = [ret, vol]
    if vix is not None:
        vix_aligned = vix.reindex(index_returns.index).ffill().values.reshape(-1, 1)
        parts.append(vix_aligned)
    if rates is not None:
        rates_aligned = rates.reindex(index_returns.index).ffill().values.reshape(-1, 1)
        parts.append(rates_aligned)
    features = np.concatenate(parts, axis=1).astype(np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    return (features - mean) / std
