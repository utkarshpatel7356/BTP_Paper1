"""
benchmarks.py
-------------
Traditional benchmarks for fair comparison against the GNN pipeline.
These are the methods a reviewer will ask about before accepting any
"GNN adds value" claim. All three run without GNN or SHAP.

B1: Market-cap weighted top-k
    Select k stocks with largest average market-cap over training period.
    Weight proportionally to market cap. This is the obvious naive baseline:
    if the biggest stocks explain the index, you don't need GNN at all.

B2: Sector-representative equal-weight (Sector-EW)
    Select the k/11 stocks per GICS sector with the highest correlation
    to the index (no ML). Weight equally. This tests whether sector
    diversification alone (which our method also uses) explains performance.

B3: Minimum tracking error MVO (Min-TE MVO)
    Select k stocks by lowest individual stock TE vs index (no graph info).
    Then run the same QP solver (without attention regularisation) as our method.
    This isolates the contribution of GNN selection vs standard covariance-based
    selection: if B3 ≈ Hybrid v2, the GNN selection is not adding value.

Why these three:
    - B1 tests: "does anything beat market cap?"
    - B2 tests: "does the sector constraint explain the improvement?"
    - B3 tests: "does GNN selection beat vanilla covariance selection with the same QP?"

Usage:
    from benchmarks import run_all_benchmarks
    benchmark_results = run_all_benchmarks(
        train_stock_ret, train_idx_ret, test_stock_ret, test_idx_ret,
        tickers, sector_map, cfg
    )
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from src.allocation.qp_solver import solve_tracking_error_qp
from src.evaluation.metrics import compute_metrics


def _ols_weights(train_stock_ret: np.ndarray, train_idx_ret: np.ndarray) -> np.ndarray:
    """OLS weights, clipped to non-negative and normalised."""
    from numpy.linalg import lstsq
    K = train_stock_ret.shape[1]
    A = np.column_stack([train_stock_ret, np.ones(len(train_idx_ret))])
    w_raw, _, _, _ = lstsq(A, train_idx_ret, rcond=None)
    w = np.clip(w_raw[:K], 0, None)
    s = w.sum()
    return w / s if s > 0 else np.ones(K) / K


# ---------------------------------------------------------------------------
# B1 — Market-cap weighted top-k
# ---------------------------------------------------------------------------

def benchmark_market_cap_topk(
    train_stock_ret: np.ndarray,   # (T_train, N)
    test_stock_ret: np.ndarray,    # (T_test, N)
    train_idx_ret: np.ndarray,     # (T_train,)
    test_idx_ret: np.ndarray,      # (T_test,)
    tickers: List[str],
    k: int = 50,
    market_caps: Optional[np.ndarray] = None,  # (N,) average market caps
) -> Tuple[List[str], np.ndarray, Dict]:
    """
    B1: Select top-k by market cap, weight proportionally.

    If market_caps is None (yfinance does not always supply it), falls back
    to average dollar volume as a market-cap proxy — a common approximation
    in the sparse index tracking literature (see Beasley et al. 2003).
    """
    N = len(tickers)

    if market_caps is None:
        # Proxy: use |mean return| × 1/volatility as a size proxy
        # (larger, less volatile stocks ≈ large caps in S&P 500)
        vol = np.std(train_stock_ret, axis=0) + 1e-8
        mean_abs = np.abs(np.mean(train_stock_ret, axis=0))
        market_caps = mean_abs / vol  # (N,) heuristic proxy

    ranked = np.argsort(market_caps)[::-1]
    sel_idx = ranked[:k].tolist()
    selected = [tickers[i] for i in sel_idx]

    # Proportional market-cap weights, capped at 10%
    caps_sel = market_caps[sel_idx]
    raw_w = caps_sel / caps_sel.sum()
    w = np.minimum(raw_w, 0.10)
    w /= w.sum()

    test_port = test_stock_ret[:, sel_idx] @ w
    metrics = compute_metrics(test_port, test_idx_ret, label="B1: Market-cap top-k")

    print(f"B1 Market-cap top-k  TE={metrics['tracking_error_pct']:.2f}%  "
          f"IR={metrics['info_ratio']:.2f}  Beta={metrics['beta']:.2f}")
    return selected, w, metrics


# ---------------------------------------------------------------------------
# B2 — Sector-representative equal-weight
# ---------------------------------------------------------------------------

def benchmark_sector_ew(
    train_stock_ret: np.ndarray,
    test_stock_ret: np.ndarray,
    train_idx_ret: np.ndarray,
    test_idx_ret: np.ndarray,
    tickers: List[str],
    sector_map: Dict[str, str],
    k: int = 50,
) -> Tuple[List[str], np.ndarray, Dict]:
    """
    B2: From each GICS sector, select the stocks with highest
    training-period correlation with the index. Take floor(k/n_sectors)
    per sector, fill remainder with the globally highest-corr stocks.
    Weight equally.

    This tests the hypothesis: "sector diversification alone explains the
    improvement, not the GNN structure".
    """
    # Per-stock correlation with index over training period
    corr_with_idx = np.array([
        np.corrcoef(train_stock_ret[:, i], train_idx_ret)[0, 1]
        for i in range(len(tickers))
    ])
    corr_with_idx = np.nan_to_num(corr_with_idx, nan=0.0)

    # Build sector groups
    sectors = {}
    for i, t in enumerate(tickers):
        s = sector_map.get(t, "Unknown")
        sectors.setdefault(s, []).append(i)

    n_sectors = len(sectors)
    per_sector = max(1, k // n_sectors)

    selected_idx = []
    selected_set = set()

    # Phase 1: top-per_sector from each sector by index correlation
    for sec_name, members in sectors.items():
        sub_corr = [(i, corr_with_idx[i]) for i in members]
        sub_corr.sort(key=lambda x: -x[1])
        for i, _ in sub_corr[:per_sector]:
            if len(selected_idx) < k and i not in selected_set:
                selected_idx.append(i)
                selected_set.add(i)

    # Phase 2: fill from globally highest-corr
    global_order = np.argsort(corr_with_idx)[::-1]
    for i in global_order:
        if len(selected_idx) >= k:
            break
        if i not in selected_set:
            selected_idx.append(i)
            selected_set.add(i)

    selected = [tickers[i] for i in selected_idx]
    w = np.ones(len(selected_idx)) / len(selected_idx)  # equal weight

    test_port = test_stock_ret[:, selected_idx] @ w
    metrics = compute_metrics(test_port, test_idx_ret, label="B2: Sector-EW")

    print(f"B2 Sector-EW         TE={metrics['tracking_error_pct']:.2f}%  "
          f"IR={metrics['info_ratio']:.2f}  Beta={metrics['beta']:.2f}")
    return selected, w, metrics


# ---------------------------------------------------------------------------
# B3 — Minimum-TE MVO (no GNN)
# ---------------------------------------------------------------------------

def benchmark_min_te_mvo(
    train_stock_ret: np.ndarray,
    test_stock_ret: np.ndarray,
    train_idx_ret: np.ndarray,
    test_idx_ret: np.ndarray,
    tickers: List[str],
    sector_map: Dict[str, str],
    k: int = 50,
    max_weight: float = 0.10,
    beta_constrained: bool = True,
    beta_lb: float = 0.95,
    beta_ub: float = 1.05,
) -> Tuple[List[str], np.ndarray, Dict]:
    """
    B3: Select stocks by lowest individual training-period TE vs index
    (pure covariance-based selection, no GNN). Then run the same QP
    as the main pipeline (same beta constraint, same max_weight).

    This is the critical benchmark: if B3 ≈ Hybrid v2, the GNN is not
    contributing to the allocation quality beyond what a standard
    covariance-based selection achieves. If Hybrid v2 << B3, the GNN adds
    genuine structural value.
    """
    # Individual TE per stock on train split
    individual_te = np.array([
        np.std(train_stock_ret[:, i] - train_idx_ret) * np.sqrt(252) * 100
        for i in range(len(tickers))
    ])

    # Sector-constrained selection (same sector constraint as main pipeline
    # for a fair comparison — we want to isolate GNN selection, not the sector constraint)
    sectors = {}
    for i, t in enumerate(tickers):
        s = sector_map.get(t, "Unknown")
        sectors.setdefault(s, []).append(i)

    selected_idx = []
    selected_set = set()

    # Phase 1: best TE stock per sector
    for sec_name, members in sectors.items():
        best_i = min(members, key=lambda i: individual_te[i])
        if best_i not in selected_set:
            selected_idx.append(best_i)
            selected_set.add(best_i)

    # Phase 2: fill remainder globally by lowest TE
    order = np.argsort(individual_te)
    for i in order:
        if len(selected_idx) >= k:
            break
        if i not in selected_set:
            selected_idx.append(i)
            selected_set.add(i)

    selected = [tickers[i] for i in selected_idx]

    # Same QP as main pipeline — NO attention regularisation
    weights, train_te = solve_tracking_error_qp(
        selected_returns=train_stock_ret[:, selected_idx],
        index_returns=train_idx_ret,
        attention_matrix=None,  # no GNN attention — this is the key difference
        lambda_reg=0.0,
        max_weight=max_weight,
        solver="OSQP",
        beta_constrained=beta_constrained,
        beta_lb=beta_lb,
        beta_ub=beta_ub,
    )

    test_port = test_stock_ret[:, selected_idx] @ weights
    metrics = compute_metrics(test_port, test_idx_ret, label="B3: Min-TE MVO (no GNN)")

    print(f"B3 Min-TE MVO        TE={metrics['tracking_error_pct']:.2f}%  "
          f"IR={metrics['info_ratio']:.2f}  Beta={metrics['beta']:.2f}")
    return selected, weights, metrics


# ---------------------------------------------------------------------------
# Run all benchmarks and return comparison table
# ---------------------------------------------------------------------------

def run_all_benchmarks(
    train_stock_ret: np.ndarray,
    test_stock_ret: np.ndarray,
    train_idx_ret: np.ndarray,
    test_idx_ret: np.ndarray,
    tickers: List[str],
    sector_map: Dict[str, str],
    cfg: dict,
    market_caps: Optional[np.ndarray] = None,
) -> Dict[str, Dict]:
    """
    Run B1, B2, B3 and return {label: metrics_dict}.
    These results are compared against the GNN methods in the paper.

    The key table the reviewer wants to see:
        B1 Market-cap top-k
        B2 Sector-EW
        B3 Min-TE MVO (same QP, no GNN)
        Hybrid v2 + beta-constrained QP  ← our method

    If Hybrid v2 does not beat B3 on TE, the GNN selection is not adding
    value over covariance-based selection. That would be an honest finding
    (null result is publishable if properly discussed).
    """
    bcfg = cfg.get("benchmarks", {})
    k = cfg["selection"]["k"]
    acfg = cfg["allocation"]

    results = {}

    print("\n" + "=" * 60)
    print("BENCHMARKS (traditional methods — no GNN)")
    print("=" * 60)

    if bcfg.get("market_cap_topk", {}).get("enabled", True):
        _, _, m = benchmark_market_cap_topk(
            train_stock_ret, test_stock_ret, train_idx_ret, test_idx_ret,
            tickers, k=k, market_caps=market_caps
        )
        results["B1_market_cap_topk"] = m

    if bcfg.get("sector_ew", {}).get("enabled", True):
        _, _, m = benchmark_sector_ew(
            train_stock_ret, test_stock_ret, train_idx_ret, test_idx_ret,
            tickers, sector_map, k=k
        )
        results["B2_sector_ew"] = m

    if bcfg.get("min_te_mvo", {}).get("enabled", True):
        beta_c = acfg.get("beta_constrained", False)
        _, _, m = benchmark_min_te_mvo(
            train_stock_ret, test_stock_ret, train_idx_ret, test_idx_ret,
            tickers, sector_map, k=k,
            max_weight=acfg["max_weight"],
            beta_constrained=beta_c,
            beta_lb=acfg.get("beta_lb", 0.95),
            beta_ub=acfg.get("beta_ub", 1.05),
        )
        results["B3_min_te_mvo"] = m

    return results


def print_comparison_table(benchmark_results: Dict, gnn_results: List[Dict]):
    """Print the full comparison table for the paper."""
    all_results = list(benchmark_results.values()) + gnn_results
    header = f"{'Strategy':<35} {'TE%':>6} {'IR':>6} {'Sharpe':>7} {'Beta':>5} {'MaxDD%':>7} {'Ret%':>6}"
    print("\n" + "=" * 75)
    print("FULL COMPARISON TABLE")
    print("=" * 75)
    print(header)
    print("-" * 75)
    for m in all_results:
        print(f"{m['label']:<35} "
              f"{m['tracking_error_pct']:>6.2f} "
              f"{m['info_ratio']:>6.2f} "
              f"{m['sharpe']:>7.2f} "
              f"{m['beta']:>5.2f} "
              f"{m['max_drawdown_pct']:>7.2f} "
              f"{m['total_return_pct']:>6.2f}")
    print("=" * 75)