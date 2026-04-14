"""
src/evaluation/metrics.py
--------------------------
Compute and plot sparse index evaluation metrics.

Metrics:
  - Annualised tracking error (TE)
  - Information ratio (IR = annualised alpha / TE)
  - Cumulative return comparison
  - Maximum drawdown
  - Rolling 30-day TE

Plots saved to outputs/figures/.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional


# --- colour palette (clean, publication-friendly) ---
COLORS = {
    "index":    "#2563EB",   # blue
    "baseline": "#D97706",   # amber
    "hybrid":   "#059669",   # green
    "te_band":  "#DCFCE7",
}


def compute_metrics(
    portfolio_returns: np.ndarray,  # (T,)
    index_returns: np.ndarray,      # (T,)
    label: str = "portfolio",
) -> Dict[str, float]:
    """
    Compute a full suite of tracking metrics.

    Returns dict with keys: tracking_error, info_ratio, total_return,
    max_drawdown, sharpe, beta, alpha.
    """
    excess = portfolio_returns - index_returns

    te_daily = np.std(excess)
    te_annual = te_daily * np.sqrt(252) * 100

    alpha_daily = np.mean(excess)
    alpha_annual = alpha_daily * 252 * 100
    ir = alpha_annual / (te_annual + 1e-8)

    cum_port = np.cumprod(1 + portfolio_returns)
    cum_idx = np.cumprod(1 + index_returns)
    total_return_port = (cum_port[-1] - 1) * 100
    total_return_idx = (cum_idx[-1] - 1) * 100

    # Max drawdown
    rolling_max = np.maximum.accumulate(cum_port)
    drawdowns = (cum_port - rolling_max) / (rolling_max + 1e-8)
    max_drawdown = float(drawdowns.min()) * 100

    # Sharpe (annualised, assuming rf=0)
    sharpe = (np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8)) * np.sqrt(252)

    # Beta
    cov = np.cov(portfolio_returns, index_returns)
    beta = cov[0, 1] / (cov[1, 1] + 1e-8)

    return {
        "label": label,
        "tracking_error_pct": round(te_annual, 4),
        "info_ratio": round(ir, 4),
        "alpha_annual_pct": round(alpha_annual, 4),
        "total_return_pct": round(total_return_port, 2),
        "index_total_return_pct": round(total_return_idx, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "sharpe": round(float(sharpe), 4),
        "beta": round(float(beta), 4),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 150,
    })


def plot_cumulative_returns(
    dates: pd.DatetimeIndex,
    index_returns: np.ndarray,
    baseline_returns: Optional[np.ndarray],
    hybrid_returns: np.ndarray,
    save_dir: str,
    title: str = "Cumulative Returns — Sparse Index vs S&P 500",
):
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    cum_idx = np.cumprod(1 + index_returns) * 100
    cum_hyb = np.cumprod(1 + hybrid_returns) * 100

    ax.plot(dates, cum_idx, color=COLORS["index"],    lw=2.0, label="S&P 500 (full)")
    ax.plot(dates, cum_hyb, color=COLORS["hybrid"],   lw=1.8, label="GNN + Influence (ours)", linestyle="-")
    if baseline_returns is not None:
        cum_bl = np.cumprod(1 + baseline_returns) * 100
        ax.plot(dates, cum_bl, color=COLORS["baseline"], lw=1.5, label="RF + SHAP baseline", linestyle="--")

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_ylabel("Cumulative return (rebased to 100)")
    ax.set_xlabel("")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(framealpha=0.9, fontsize=10)
    fig.tight_layout()
    path = os.path.join(save_dir, "cumulative_returns.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_tracking_error_rolling(
    dates: pd.DatetimeIndex,
    index_returns: np.ndarray,
    baseline_returns: Optional[np.ndarray],
    hybrid_returns: np.ndarray,
    window: int = 30,
    save_dir: str = "outputs/figures",
):
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 4))

    def rolling_te(port, idx, w):
        excess = port - idx
        return pd.Series(excess).rolling(w).std().values * np.sqrt(252) * 100

    te_hyb = rolling_te(hybrid_returns, index_returns, window)
    ax.plot(dates, te_hyb, color=COLORS["hybrid"], lw=1.8, label="GNN + Influence")
    ax.fill_between(dates, 0, te_hyb, color=COLORS["te_band"], alpha=0.4)

    if baseline_returns is not None:
        te_bl = rolling_te(baseline_returns, index_returns, window)
        ax.plot(dates, te_bl, color=COLORS["baseline"], lw=1.5, linestyle="--", label="RF + SHAP baseline")

    ax.set_title(f"{window}-day rolling tracking error (annualised %)", fontsize=13)
    ax.set_ylabel("Tracking error (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(save_dir, "rolling_tracking_error.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_weight_allocation(
    tickers: List[str],
    weights: np.ndarray,
    sector_map: dict,
    save_dir: str,
):
    """Bar chart of portfolio weights, coloured by sector."""
    _setup_style()
    sector_list = [sector_map.get(t, "Unknown") for t in tickers]
    unique_sectors = sorted(set(sector_list))
    palette = sns.color_palette("tab10", len(unique_sectors))
    sector_color = {s: palette[i] for i, s in enumerate(unique_sectors)}
    colors = [sector_color[s] for s in sector_list]

    order = np.argsort(weights)[::-1]
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(tickers)), weights[order] * 100, color=[colors[i] for i in order], edgecolor="none")
    ax.set_xticks(range(len(tickers)))
    ax.set_xticklabels([tickers[i] for i in order], rotation=90, fontsize=7)
    ax.set_ylabel("Weight (%)")
    ax.set_title("Portfolio weight allocation by stock (coloured by GICS sector)", fontsize=13)

    # Legend for sectors
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=sector_color[s], label=s) for s in unique_sectors]
    ax.legend(handles=legend_els, fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    path = os.path.join(save_dir, "weight_allocation.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_loss_curves(
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    save_dir: str,
):
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 4))
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color=COLORS["hybrid"],    lw=1.8, label="Train loss")
    ax.plot(epochs, val_losses,   color=COLORS["baseline"], lw=1.5, linestyle="--", label="Val loss")
    ax.set_title("GNN training and validation loss", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(save_dir, "loss_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_influence_scores(
    tickers: List[str],
    hybrid_scores: np.ndarray,
    selected: List[str],
    save_dir: str,
    top_n: int = 30,
):
    _setup_style()
    order = np.argsort(hybrid_scores)[::-1][:top_n]
    top_tickers = [tickers[i] for i in order]
    top_scores = hybrid_scores[order]
    colors = [COLORS["hybrid"] if t in selected else COLORS["baseline"] for t in top_tickers]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(range(top_n), top_scores[::-1], color=colors[::-1], edgecolor="none")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_tickers[::-1], fontsize=8)
    ax.set_xlabel("Hybrid influence score")
    ax.set_title(f"Top {top_n} stocks by hybrid score (green = selected)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(save_dir, "influence_scores.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def save_results(metrics_list: List[dict], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    print(f"Results saved → {path}")

    # Also save as CSV
    df = pd.DataFrame(metrics_list)
    df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)
