"""
src/allocation/qp_solver.py
---------------------------
Solve for portfolio weights that minimise tracking error.

Objective (GNN-regularised QP):
  min   (w - w_idx)ᵀ Σ (w - w_idx)  +  λ · wᵀ A_attn w
  s.t.  sum(w) = 1
        0 ≤ wᵢ ≤ max_weight   for all i

where:
  Σ         = sample covariance of selected stock returns (K x K)
  w_idx     = index weights for selected stocks (proxy = market-cap weights,
               or uniform if not available)
  A_attn    = sub-matrix of GNN attention edge weights between selected stocks
  λ         = regularisation strength (penalises correlated pairs)

If the GNN attention matrix is not available, falls back to pure TE minimisation.
"""

import numpy as np
import cvxpy as cp
from typing import List, Optional, Tuple


def solve_tracking_error_qp(
    selected_returns: np.ndarray,    # (T, K) returns of selected stocks
    index_returns: np.ndarray,       # (T,) index returns
    attention_matrix: Optional[np.ndarray] = None,  # (K, K) GNN attention sub-matrix
    lambda_reg: float = 0.01,
    max_weight: float = 0.10,
    solver: str = "OSQP",
) -> Tuple[np.ndarray, float]:
    """
    Solve the GNN-regularised tracking error QP.

    Parameters
    ----------
    selected_returns  : (T, K) daily returns of the k selected stocks
    index_returns     : (T,) daily index returns
    attention_matrix  : (K, K) optional GNN edge attention weights
    lambda_reg        : regularisation coefficient for attention penalty
    max_weight        : max weight per stock (0.10 = 10%)
    solver            : OSQP (fast) or SCS (robust)

    Returns
    -------
    weights     : (K,) optimal portfolio weights
    tracking_te : annualised tracking error (%)
    """
    T, K = selected_returns.shape

    # Compute covariance matrix Σ of excess returns (stock - index)
    excess = selected_returns - index_returns[:, np.newaxis]  # (T, K)
    Sigma = np.cov(excess.T) + 1e-6 * np.eye(K)              # (K, K)

    # Index proxy weights (equal weight if market caps unavailable)
    w_idx = np.ones(K) / K

    # Decision variable
    w = cp.Variable(K, nonneg=True)

    # Tracking error term
    te_term = cp.quad_form(w - w_idx, Sigma)

    # GNN attention regularisation: penalise correlated pairs
    if attention_matrix is not None:
        A = attention_matrix
        A = (A + A.T) / 2  # symmetrise
        reg_term = lambda_reg * cp.quad_form(w, A)
    else:
        reg_term = 0.0

    objective = cp.Minimize(te_term + reg_term)

    constraints = [
        cp.sum(w) == 1,
        w <= max_weight,
    ]

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=solver, verbose=False)
    except cp.SolverError:
        prob.solve(solver="SCS", verbose=False)

    if w.value is None:
        # Fallback: equal weight
        weights = np.ones(K) / K
        print("Warning: QP solver failed. Using equal weights.")
    else:
        weights = np.clip(w.value, 0, max_weight)
        weights /= weights.sum()

    # Compute annualised tracking error
    portfolio_ret = selected_returns @ weights
    te_daily = np.std(portfolio_ret - index_returns)
    tracking_te = te_daily * np.sqrt(252) * 100  # annualised %

    return weights, tracking_te


def build_attention_submatrix(
    attn_weights: np.ndarray,   # (E,)
    edge_index: np.ndarray,     # (2, E)
    selected_indices: List[int],
    n_total: int,
) -> np.ndarray:
    """
    Extract the K×K sub-matrix of GNN attention weights
    for the selected stock indices.

    Parameters
    ----------
    attn_weights     : mean attention weight per edge (E,)
    edge_index       : (2, E) edge index array
    selected_indices : list of int, indices of selected stocks in full graph
    n_total          : total number of nodes (N+1)

    Returns
    -------
    A_sub : (K, K) attention sub-matrix for selected stocks
    """
    K = len(selected_indices)
    idx_map = {orig: new for new, orig in enumerate(selected_indices)}
    A_sub = np.zeros((K, K))

    src, dst = edge_index
    for e in range(len(attn_weights)):
        s, d = int(src[e]), int(dst[e])
        if s in idx_map and d in idx_map:
            A_sub[idx_map[s], idx_map[d]] += attn_weights[e]

    return A_sub
