"""
main.py
-------
End-to-end sparse index replication pipeline.

Usage:
    python main.py --config configs/config.yaml [--skip-download] [--skip-graph] [--skip-train]

Pipeline stages:
    1. (Optional) Download raw price data
    2. Build graph (nodes, edges, features)
    3. Run RF + SHAP baseline
    4. Train GNN
    5. Compute influence scores + fuse with SHAP
    6. Select subset with sector constraint
    7. Solve QP for budget allocation
    8. Evaluate on test split
    9. Save plots and results
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import yaml

from src.graph.build_graph import build_graph, download_prices, fetch_sp500_tickers
from src.graph.features import compute_log_returns
from src.selection.shap_selector import run_shap_selection
from src.models.train import train_model
from src.selection.influence import (
    greedy_influence_maximisation,
    fuse_scores,
    select_with_sector_constraint,
)
from src.allocation.qp_solver import solve_tracking_error_qp, build_attention_submatrix
from src.evaluation.metrics import (
    compute_metrics,
    plot_cumulative_returns,
    plot_tracking_error_rolling,
    plot_weight_allocation,
    plot_loss_curves,
    plot_influence_scores,
    save_results,
)


def main(cfg: dict, skip_download: bool, skip_graph: bool, skip_train: bool):
    os.makedirs(cfg["outputs"]["figures_dir"], exist_ok=True)
    os.makedirs(cfg["outputs"]["results_dir"], exist_ok=True)

    dcfg = cfg["data"]
    scfg = cfg["selection"]
    acfg = cfg["allocation"]

    # -----------------------------------------------------------------------
    # Stage 1: Download (optional)
    # -----------------------------------------------------------------------
    if not skip_download:
        sp500_df = fetch_sp500_tickers()
        tickers = sp500_df["ticker"].tolist()
        download_prices(tickers, dcfg["index_ticker"], dcfg["start_date"], dcfg["end_date"], dcfg["raw_dir"])

    # -----------------------------------------------------------------------
    # Stage 2: Build graph
    # -----------------------------------------------------------------------
    graph_path = os.path.join(dcfg["processed_dir"], "graph_data.pt")
    if not skip_graph or not os.path.exists(graph_path):
        graph_data = build_graph(cfg)
    else:
        print(f"Loading cached graph from {graph_path}")
        graph_data = torch.load(graph_path, weights_only=False)

    tickers = graph_data.tickers
    N = graph_data.n_stocks
    T = graph_data.x.shape[1]

    # Reconstruct returns from node features (slot 0 = normalised return)
    # We reload raw prices for clean returns
    prices_path = os.path.join(dcfg["raw_dir"], "prices.csv")
    prices_full = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    index_prices = prices_full[dcfg["index_ticker"]]
    stock_prices = prices_full[tickers].dropna(thresh=int(0.95 * len(prices_full)), axis=1)
    # Realign tickers after dropna
    tickers = [t for t in tickers if t in stock_prices.columns]
    stock_prices = stock_prices[tickers].ffill().bfill()

    all_returns = compute_log_returns(stock_prices).values  # (T, N)
    index_returns = compute_log_returns(index_prices.to_frame()).iloc[:, 0].values  # (T,)
    dates = compute_log_returns(stock_prices).index

    # Train/test split
    T = len(index_returns)
    test_size = int(dcfg["test_months"] * 21)  # ~21 trading days per month
    T_train = T - test_size

    train_stock_ret = all_returns[:T_train]
    test_stock_ret = all_returns[T_train:]
    train_idx_ret = index_returns[:T_train]
    test_idx_ret = index_returns[T_train:]
    train_dates = dates[:T_train]
    test_dates = dates[T_train:]

    print(f"\nTrain: {train_dates[0].date()} → {train_dates[-1].date()}  ({T_train} days)")
    print(f"Test : {test_dates[0].date()} → {test_dates[-1].date()}   ({test_size} days)\n")

    # -----------------------------------------------------------------------
    # Stage 3: RF + SHAP baseline
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("STAGE 3: RF + SHAP baseline")
    print("=" * 60)
    shap_tickers, shap_scores, rf_model = run_shap_selection(
        train_stock_ret, train_idx_ret, tickers, k=scfg["k"]
    )
    shap_idx = [tickers.index(t) for t in shap_tickers]
    shap_test_ret = test_stock_ret[:, shap_idx]

    # Simple OLS weights for baseline
    from numpy.linalg import lstsq
    A = np.column_stack([train_stock_ret[:, shap_idx], np.ones(T_train)])
    bl_weights_raw, _, _, _ = lstsq(A, train_idx_ret, rcond=None)
    bl_weights = np.clip(bl_weights_raw[:scfg["k"]], 0, None)
    if bl_weights.sum() > 0:
        bl_weights /= bl_weights.sum()
    else:
        bl_weights = np.ones(scfg["k"]) / scfg["k"]
    baseline_test_returns = shap_test_ret @ bl_weights

    # -----------------------------------------------------------------------
    # Stage 4: Train GNN
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 4: Train GNN")
    print("=" * 60)
    if not skip_train:
        model, train_losses, val_losses = train_model(graph_data, cfg, checkpoint_dir="outputs")
        np.save("outputs/train_losses.npy", train_losses)
        np.save("outputs/val_losses.npy", val_losses)
    else:
        print("Skipping GNN training (--skip-train). Loading checkpoint...")
        from src.models.gat_model import SparseIndexGNN
        mcfg = cfg["model"]
        F = graph_data.x.shape[2]
        model = SparseIndexGNN(F, mcfg["gnn_hidden"], mcfg["gnn_heads"], mcfg["gnn_layers"],
                               mcfg["gru_window"], mcfg["dropout"])
        model.load_state_dict(torch.load("outputs/best_gnn.pt", map_location="cpu"))
        train_losses = np.load("outputs/train_losses.npy")
        val_losses = np.load("outputs/val_losses.npy")

    # -----------------------------------------------------------------------
    # Stage 5: Influence scoring + fusion
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 5: Influence scoring")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        x_full = graph_data.x[:, -cfg["model"]["gru_window"]:, :]
        _, embeddings, attn = model(x_full, graph_data.edge_index, graph_data.edge_weight)

    embeddings_np = embeddings.detach().cpu().numpy()  # (N+1, hidden)
    attn_np = attn.detach().cpu().mean(dim=-1).numpy() if attn is not None else None  # mean over heads

    print("Running greedy influence maximisation...")
    _, influence_scores = greedy_influence_maximisation(embeddings_np, k=scfg["k"], tickers=tickers)

    hybrid_scores = fuse_scores(shap_scores, influence_scores, alpha=scfg["alpha_shap"])

    # -----------------------------------------------------------------------
    # Stage 6: Subset selection with sector constraint
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 6: Subset selection")
    print("=" * 60)
    sp500_df = fetch_sp500_tickers()
    sector_map = dict(zip(sp500_df["ticker"], sp500_df["sector"]))

    selected = select_with_sector_constraint(
        hybrid_scores, tickers, sector_map, k=scfg["k"],
        min_per_sector=scfg["min_per_sector"]
    )
    print(f"Selected {len(selected)} stocks: {selected[:5]} ...")

    selected_idx = [tickers.index(t) for t in selected]

    # -----------------------------------------------------------------------
    # Stage 7: QP budget allocation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 7: QP budget allocation")
    print("=" * 60)
    A_attn = None
    if attn_np is not None:
        A_attn = build_attention_submatrix(
            attn_np, graph_data.edge_index.numpy(), selected_idx, n_total=N + 1
        )

    weights, train_te = solve_tracking_error_qp(
        selected_returns=train_stock_ret[:, selected_idx],
        index_returns=train_idx_ret,
        attention_matrix=A_attn,
        lambda_reg=acfg["lambda_reg"],
        max_weight=acfg["max_weight"],
        solver=acfg["solver"],
    )
    print(f"Train TE (annualised): {train_te:.2f}%")
    print("Top 5 weights:")
    order = np.argsort(weights)[::-1]
    for i in order[:5]:
        print(f"  {selected[i]:8s}  {weights[i]*100:.2f}%")

    # -----------------------------------------------------------------------
    # Stage 8: Evaluate on test split
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 8: Test evaluation")
    print("=" * 60)
    hybrid_test_returns = test_stock_ret[:, selected_idx] @ weights

    metrics_baseline = compute_metrics(baseline_test_returns, test_idx_ret, label="RF+SHAP baseline")
    metrics_hybrid = compute_metrics(hybrid_test_returns, test_idx_ret, label="GNN+Influence (ours)")

    print("\nBaseline metrics:")
    for k_m, v in metrics_baseline.items():
        if k_m != "label":
            print(f"  {k_m}: {v}")

    print("\nHybrid metrics:")
    for k_m, v in metrics_hybrid.items():
        if k_m != "label":
            print(f"  {k_m}: {v}")

    # -----------------------------------------------------------------------
    # Stage 9: Plots and results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 9: Saving plots and results")
    print("=" * 60)

    figs_dir = cfg["outputs"]["figures_dir"]
    res_dir = cfg["outputs"]["results_dir"]

    plot_loss_curves(train_losses, val_losses, figs_dir)

    plot_cumulative_returns(
        test_dates, test_idx_ret, baseline_test_returns, hybrid_test_returns, figs_dir
    )
    plot_tracking_error_rolling(
        test_dates, test_idx_ret, baseline_test_returns, hybrid_test_returns,
        window=30, save_dir=figs_dir
    )
    plot_weight_allocation(selected, weights, sector_map, figs_dir)
    plot_influence_scores(tickers, hybrid_scores, selected, figs_dir)

    save_results([metrics_baseline, metrics_hybrid], res_dir)

    # Save allocation CSV
    alloc_df = pd.DataFrame({"ticker": selected, "weight": weights, "sector": [sector_map.get(t, "?") for t in selected]})
    alloc_df = alloc_df.sort_values("weight", ascending=False)
    alloc_df.to_csv(os.path.join(res_dir, "allocation.csv"), index=False)
    print(f"Allocation saved → {res_dir}/allocation.csv")

    print("\nPipeline complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparse S&P 500 index replication pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-graph",    action="store_true")
    parser.add_argument("--skip-train",    action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.skip_download, args.skip_graph, args.skip_train)
