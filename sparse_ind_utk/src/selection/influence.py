"""
src/selection/influence.py
--------------------------
Greedy influence maximisation to select the k most "influential" stocks
with respect to the S&P 500 index sink node.

Influence is defined as: how much does including stock i in the seed set
increase the cosine similarity between the seed-set aggregate embedding
and the index sink node embedding?

This mirrors the Independent Cascade / influence spread definition from
the literature, adapted to continuous embedding space.

Algorithm (greedy, O(k·N)):
  1. Start with seed set S = {}
  2. For round r = 1..k:
       For each candidate stock i ∉ S:
           Compute marginal gain Δ(i|S) = sim(agg(S∪{i}), z_idx) - sim(agg(S), z_idx)
       Add i* = argmax Δ(i|S) to S
  3. Return S and per-stock influence scores

The aggregate function is the mean of embeddings in S.
"""

import numpy as np
from typing import List, Tuple


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def greedy_influence_maximisation(
    embeddings: np.ndarray,   # (N+1, hidden) — last row is index sink
    k: int,
    tickers: List[str],
) -> Tuple[List[str], np.ndarray]:
    """
    Greedy influence maximisation in embedding space.

    Parameters
    ----------
    embeddings  : (N+1, hidden) node embeddings from trained GNN
    k           : subset size
    tickers     : list of stock ticker strings (length N)

    Returns
    -------
    selected_tickers  : list[str] of length k
    influence_scores  : (N,) per-stock influence score
    """
    N = len(tickers)
    stock_embs = embeddings[:N]      # (N, hidden)
    idx_emb = embeddings[N]          # (hidden,) index sink

    influence_scores = np.zeros(N)
    selected_idx = []
    candidate_idx = list(range(N))
    current_agg = np.zeros_like(idx_emb)  # mean of selected embeddings

    for r in range(k):
        best_i, best_gain = -1, -float("inf")
        current_sim = cosine_sim(current_agg, idx_emb) if r > 0 else 0.0

        for i in candidate_idx:
            new_agg = (current_agg * r + stock_embs[i]) / (r + 1)
            gain = cosine_sim(new_agg, idx_emb) - current_sim
            if gain > best_gain:
                best_gain = gain
                best_i = i
                best_new_agg = new_agg

        selected_idx.append(best_i)
        candidate_idx.remove(best_i)
        current_agg = best_new_agg
        influence_scores[best_i] = best_gain
        print(f"  Round {r+1:3d}/{k}: selected {tickers[best_i]:8s}  gain={best_gain:.4f}")

    # For unselected stocks, assign a score based on their cosine similarity to idx_emb
    for i in candidate_idx:
        influence_scores[i] = cosine_sim(stock_embs[i], idx_emb)

    selected_tickers = [tickers[i] for i in selected_idx]
    return selected_tickers, influence_scores


def fuse_scores(
    shap_scores: np.ndarray,     # (N,) from RF+SHAP
    influence_scores: np.ndarray, # (N,) from GNN influence
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Combine SHAP and influence scores into a single hybrid ranking.

    hybrid_score = alpha * norm(SHAP) + (1 - alpha) * norm(influence)

    Parameters
    ----------
    alpha : weight on SHAP; (1-alpha) on influence

    Returns
    -------
    hybrid_scores : (N,) normalised combined scores
    """
    def _norm(x):
        r = x - x.min()
        d = r.max() + 1e-8
        return r / d

    return alpha * _norm(shap_scores) + (1 - alpha) * _norm(influence_scores)


def fuse_scores_v2(
    shap_rf: np.ndarray,        # (N,) from RF+SHAP
    shap_emb: np.ndarray,       # (N,) from embedding regressor SHAP
    influence_scores: np.ndarray, # (N,) from GNN influence
    alpha: float = 0.4,
    beta: float = 0.4,
) -> np.ndarray:
    """
    Three-way fusion of scoring signals.

    hybrid = α · norm(shap_rf) + β · norm(shap_emb) + (1-α-β) · norm(influence)

    Parameters
    ----------
    shap_rf          : (N,) SHAP scores from RF baseline
    shap_emb         : (N,) SHAP scores from embedding regressor
    influence_scores : (N,) influence scores from GNN greedy IM
    alpha            : weight on RF-SHAP
    beta             : weight on embedding-SHAP
                       influence gets (1 - alpha - beta)

    Returns
    -------
    hybrid_scores : (N,) normalised combined scores
    """
    gamma = max(1.0 - alpha - beta, 0.0)

    def _norm(x):
        r = x - x.min()
        d = r.max() + 1e-8
        return r / d

    return alpha * _norm(shap_rf) + beta * _norm(shap_emb) + gamma * _norm(influence_scores)


def select_with_sector_constraint(
    hybrid_scores: np.ndarray,
    tickers: List[str],
    sector_map: dict,
    k: int,
    min_per_sector: int = 1,
) -> List[str]:
    """
    Select top-k tickers by hybrid score with a minimum-per-sector constraint.
    Greedy: first guarantee min_per_sector from each sector, then fill remaining
    budget with top-ranked unconstrained picks.

    Parameters
    ----------
    hybrid_scores  : (N,) scores, higher = better
    tickers        : list of ticker strings
    sector_map     : {ticker: sector_name}
    k              : total subset size
    min_per_sector : minimum stocks per GICS sector

    Returns
    -------
    selected : list of selected ticker strings (length k)
    """
    N = len(tickers)
    score_pairs = sorted(enumerate(tickers), key=lambda x: -hybrid_scores[x[0]])

    sectors = {}
    for i, t in enumerate(tickers):
        s = sector_map.get(t, "Unknown")
        sectors.setdefault(s, []).append((i, t, hybrid_scores[i]))

    selected = []
    selected_set = set()

    # Phase 1: guarantee coverage
    for s_name, members in sectors.items():
        members_sorted = sorted(members, key=lambda x: -x[2])
        for i, t, score in members_sorted[:min_per_sector]:
            if t not in selected_set and len(selected) < k:
                selected.append(t)
                selected_set.add(t)

    # Phase 2: fill remainder with top hybrids
    for i, t in score_pairs:
        if len(selected) >= k:
            break
        if t not in selected_set:
            selected.append(t)
            selected_set.add(t)

    return selected[:k]
