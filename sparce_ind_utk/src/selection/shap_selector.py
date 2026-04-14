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
"""

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
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Train RF and compute SHAP values to rank stocks.

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

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)  # (T, N)
    shap_scores = np.abs(shap_values).mean(axis=0)  # (N,)

    ranked_idx = np.argsort(shap_scores)[::-1]
    selected_idx = ranked_idx[:k]
    selected_tickers = [tickers[i] for i in selected_idx]

    print(f"Top-{k} tickers by SHAP importance:")
    for i, t in enumerate(selected_tickers[:10]):
        print(f"  {i+1:2d}. {t:8s}  SHAP={shap_scores[ranked_idx[i]]:.4f}")

    return selected_tickers, shap_scores, rf
