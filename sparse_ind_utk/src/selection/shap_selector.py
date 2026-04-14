"""
src/selection/shap_selector.py
-------------------------------
Baseline: Random Forest Regressor + SHAP importance ranking.

Steps:
  1. Flatten stock return matrix → (T, N) feature matrix
  2. Target = S&P 500 daily return
  3. Train RF on training window
  4. Compute SHAP values → importance per stock
  5. Return ranked tickers + importance scores

Memory-efficient: uses a background sample of 100 rows
                  and explains a subsample of 200 rows.
"""

import gc
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple


def run_shap_selection(
    stock_returns: np.ndarray,   # (T, N)
    index_returns: np.ndarray,   # (T,)
    tickers: List[str],
    k: int = 50,
    n_estimators: int = 200,
    max_depth: int = 8,
    random_state: int = 42,
    shap_background_size: int = 100,
    shap_explain_size: int = 200,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Train RF and compute SHAP values to rank stocks.

    Parameters
    ----------
    shap_background_size : number of rows for SHAP background dataset
    shap_explain_size    : number of rows to compute SHAP explanations for

    Returns
    -------
    selected_tickers : list[str]  top-k tickers by mean |SHAP|
    shap_scores      : ndarray (N,) mean absolute SHAP value per stock
    rf_model         : fitted RandomForestRegressor
    """
    T, N = stock_returns.shape
    X = stock_returns          # (T, N)
    y = index_returns          # (T,)

    print(f"Training Random Forest on {T} samples, {N} features...")
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X, y)

    print("Computing SHAP values (using subsampled background)...")
    # Use a subsample for background to avoid O(T*N) memory in SHAP
    rng = np.random.RandomState(random_state)
    bg_idx = rng.choice(T, size=min(shap_background_size, T), replace=False)
    explain_idx = rng.choice(T, size=min(shap_explain_size, T), replace=False)

    explainer = shap.TreeExplainer(rf, X[bg_idx])
    shap_values = explainer.shap_values(X[explain_idx])  # (explain_size, N)
    shap_scores = np.abs(shap_values).mean(axis=0)       # (N,)

    ranked_idx = np.argsort(shap_scores)[::-1]
    selected_idx = ranked_idx[:k]
    selected_tickers = [tickers[i] for i in selected_idx]

    print(f"Top-{k} tickers by SHAP importance:")
    for i, t in enumerate(selected_tickers[:10]):
        print(f"  {i+1:2d}. {t:8s}  SHAP={shap_scores[ranked_idx[i]]:.4f}")

    gc.collect()
    return selected_tickers, shap_scores, rf
