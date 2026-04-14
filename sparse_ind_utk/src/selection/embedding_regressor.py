"""
src/selection/embedding_regressor.py
-------------------------------------
Stage 5.5: Train a Ridge regressor on GNN-embedding-weighted stock
returns, then explain it with SHAP to get per-stock importance scores
that capture graph-structural information.

Pipeline position:
    Stage 5  → GNN inference → embeddings  (input to this module)
    Stage 5.5 → Embedding regressor → SHAP → emb_shap_scores  (THIS)
    Stage 6  → Three-way fusion → selection

Why this works:
    - Even partially collapsed GNN embeddings encode *some* structural
      information in their individual dimensions.
    - A Ridge regressor trained on embedding-weighted returns learns
      which graph-structural properties are predictive of the index.
    - SHAP on a linear model is exact and instant (LinearExplainer).
    - The resulting scores complement raw-return SHAP (RF baseline)
      by injecting graph-mediated non-linear relationships.
"""

import gc
import numpy as np
import shap
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple


def _cosine_similarities(stock_embs: np.ndarray, idx_emb: np.ndarray) -> np.ndarray:
    """Cosine similarity between each stock embedding and the index sink."""
    norms_s = np.linalg.norm(stock_embs, axis=1, keepdims=True) + 1e-8
    norm_i = np.linalg.norm(idx_emb) + 1e-8
    return (stock_embs @ idx_emb) / (norms_s.squeeze() * norm_i)


def train_embedding_regressor(
    embeddings: np.ndarray,          # (N+1, hidden) — last row = index sink
    train_stock_returns: np.ndarray, # (T_train, N)
    train_index_returns: np.ndarray, # (T_train,)
    tickers: List[str],
    ridge_alpha: float = 1.0,
    shap_background_size: int = 200,
    shap_explain_size: int = 300,
    random_state: int = 42,
) -> Tuple[np.ndarray, object, np.ndarray]:
    """
    Train a Ridge regressor on embedding-weighted features, then
    compute SHAP importance per stock.

    Feature engineering:
        For each training day t, the feature vector is:
            X[t, :] = stock_returns[t, :] * emb_sim[:]
        where emb_sim[i] = cosine_similarity(embedding[i], embedding[index_sink]).

        This re-weights each stock's return by how structurally close
        (in GNN embedding space) it is to the index, giving the
        regressor a graph-informed view of stock importance.

    Parameters
    ----------
    embeddings          : (N+1, hidden) from GNN, last row = index sink
    train_stock_returns : (T_train, N) daily stock returns
    train_index_returns : (T_train,) daily index returns
    tickers             : list of N ticker strings
    ridge_alpha         : L2 regularisation for Ridge
    shap_background_size : rows for SHAP background
    shap_explain_size    : rows to explain

    Returns
    -------
    emb_shap_scores : (N,) per-stock importance from embedding-SHAP
    regressor       : fitted Ridge model
    emb_sim         : (N,) embedding cosine similarities used as weights
    """
    N = len(tickers)
    T_train = train_stock_returns.shape[0]

    stock_embs = embeddings[:N]     # (N, hidden)
    idx_emb = embeddings[N]         # (hidden,)

    # Compute embedding-based similarity weights
    emb_sim = _cosine_similarities(stock_embs, idx_emb)  # (N,)
    print(f"  Embedding similarities: min={emb_sim.min():.4f}, "
          f"max={emb_sim.max():.4f}, std={emb_sim.std():.4f}")

    # Build feature matrix: embedding-weighted returns
    X = train_stock_returns * emb_sim[np.newaxis, :]  # (T, N)

    # Standardise for stable Ridge
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Ridge regressor
    print(f"  Training Ridge regressor (alpha={ridge_alpha}) on "
          f"{T_train} samples, {N} features...")
    reg = Ridge(alpha=ridge_alpha, random_state=random_state)
    reg.fit(X_scaled, train_index_returns)

    train_r2 = reg.score(X_scaled, train_index_returns)
    print(f"  Ridge train R²: {train_r2:.4f}")

    # SHAP — exact for linear models
    print("  Computing SHAP on embedding regressor...")
    rng = np.random.RandomState(random_state)
    bg_idx = rng.choice(T_train, size=min(shap_background_size, T_train), replace=False)
    explain_idx = rng.choice(T_train, size=min(shap_explain_size, T_train), replace=False)

    explainer = shap.LinearExplainer(reg, X_scaled[bg_idx])
    shap_values = explainer.shap_values(X_scaled[explain_idx])  # (explain_size, N)
    emb_shap_scores = np.abs(shap_values).mean(axis=0)          # (N,)

    # Report top stocks
    ranked = np.argsort(emb_shap_scores)[::-1]
    print("  Top-10 by embedding-SHAP:")
    for i in range(min(10, N)):
        print(f"    {i+1:2d}. {tickers[ranked[i]]:8s}  "
              f"emb_SHAP={emb_shap_scores[ranked[i]]:.6f}  "
              f"emb_sim={emb_sim[ranked[i]]:.4f}")

    gc.collect()
    return emb_shap_scores, reg, emb_sim
