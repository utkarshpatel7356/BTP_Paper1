"""
timeline.py — Ablation study: evaluate each approach individually, then show
how combining them step-by-step reaches the final Hybrid v2 results.

Usage:
    python timeline.py --config configs/config.yaml --skip-download --skip-graph --skip-train

Outputs → outputs/timeline/figures/ and outputs/timeline/results/
"""
import argparse, gc, os, json, numpy as np, pandas as pd, torch, yaml
from numpy.linalg import lstsq

from src.graph.build_graph import build_graph, download_prices, fetch_sp500_tickers
from src.graph.features import compute_log_returns
from src.selection.shap_selector import run_shap_selection
from src.models.train import train_model
from src.selection.influence import (
    greedy_influence_maximisation, fuse_scores, fuse_scores_v2,
    select_with_sector_constraint,
)
from src.selection.embedding_regressor import train_embedding_regressor
from src.allocation.qp_solver import solve_tracking_error_qp, build_attention_submatrix
from src.evaluation.metrics import compute_metrics
from timeline_plots import (
    plot_score_distributions, plot_top30_bars, plot_selection_overlap,
    plot_cumulative_ladder, plot_rolling_te_ladder, plot_metrics_bars,
    plot_weight_boxplots, plot_sector_comparison, plot_alpha_beta_sweep,
    plot_summary_table, save_timeline_results, NAMES,
)


def _ols_weights(train_ret, train_idx, selected_idx, k):
    """Compute OLS weights for a selection, clipped and normalised."""
    A = np.column_stack([train_ret[:, selected_idx], np.ones(len(train_idx))])
    w_raw, _, _, _ = lstsq(A, train_idx, rcond=None)
    w = np.clip(w_raw[:k], 0, None)
    return w / w.sum() if w.sum() > 0 else np.ones(k) / k


def _equal_weights(k):
    return np.ones(k) / k


def main(cfg, skip_download, skip_graph, skip_train):
    FIGS = "outputs/timeline/figures"
    RES = "outputs/timeline/results"
    os.makedirs(FIGS, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    dcfg, scfg, acfg = cfg["data"], cfg["selection"], cfg["allocation"]
    k = scfg["k"]

    # === STAGE 1-2: Load data & graph ===
    print("=" * 60 + "\nLoading data and graph\n" + "=" * 60)
    graph_path = os.path.join(dcfg["processed_dir"], "graph_data.pt")
    if not skip_graph or not os.path.exists(graph_path):
        graph_data = build_graph(cfg)
    else:
        print(f"Loading cached graph from {graph_path}")
        graph_data = torch.load(graph_path, weights_only=False)

    tickers = graph_data.tickers
    N = graph_data.n_stocks

    prices_full = pd.read_csv(os.path.join(dcfg["raw_dir"], "prices.csv"),
                              index_col=0, parse_dates=True)
    index_prices = prices_full[dcfg["index_ticker"]]
    stock_prices = prices_full[tickers].dropna(thresh=int(0.95*len(prices_full)), axis=1)
    tickers = [t for t in tickers if t in stock_prices.columns]
    stock_prices = stock_prices[tickers].ffill().bfill()

    all_returns = compute_log_returns(stock_prices).values
    index_returns = compute_log_returns(index_prices.to_frame()).iloc[:, 0].values
    dates = compute_log_returns(stock_prices).index

    T = len(index_returns)
    test_size = int(dcfg["test_months"] * 21)
    T_train = T - test_size

    train_ret = all_returns[:T_train]
    test_ret = all_returns[T_train:]
    train_idx = index_returns[:T_train]
    test_idx = index_returns[T_train:]
    test_dates = dates[T_train:]

    print(f"Train: {dates[:T_train][0].date()} → {dates[:T_train][-1].date()} ({T_train} days)")
    print(f"Test : {test_dates[0].date()} → {test_dates[-1].date()} ({test_size} days)")

    # === STAGE 3: Compute RF+SHAP scores ===
    print("\n" + "=" * 60 + "\nComputing RF+SHAP scores\n" + "=" * 60)
    _, shap_scores, _ = run_shap_selection(train_ret, train_idx, tickers, k=k)
    gc.collect()

    # === STAGE 4: Load GNN ===
    print("\n" + "=" * 60 + "\nLoading GNN\n" + "=" * 60)
    from src.models.gat_model import SparseIndexGNN
    mcfg = cfg["model"]
    F = graph_data.x.shape[2]
    model = SparseIndexGNN(F, mcfg["gnn_hidden"], mcfg["gnn_heads"],
                           mcfg["gnn_layers"], mcfg["gru_window"], mcfg["dropout"])
    model.load_state_dict(torch.load("outputs/best_gnn.pt", map_location="cpu"))
    model.eval(); model.cpu()

    with torch.no_grad():
        x_full = graph_data.x[:, -mcfg["gru_window"]:, :]
        _, embeddings, attn, attn_ei = model(x_full, graph_data.edge_index,
                                              graph_data.edge_weight)
    embeddings_np = embeddings.detach().cpu().numpy()
    if attn is not None and attn_ei is not None:
        attn_np = attn.detach().cpu().mean(dim=-1).numpy()
        attn_ei_np = attn_ei.detach().cpu().numpy()
    else:
        attn_np, attn_ei_np = None, None

    # === STAGE 5: Influence scores ===
    print("\n" + "=" * 60 + "\nComputing influence scores\n" + "=" * 60)
    _, influence_scores = greedy_influence_maximisation(embeddings_np, k=k, tickers=tickers)
    gc.collect()

    # === STAGE 5.5: Embedding-SHAP ===
    print("\n" + "=" * 60 + "\nComputing Embedding-SHAP scores\n" + "=" * 60)
    emb_shap_scores, _, _ = train_embedding_regressor(
        embeddings_np, train_ret, train_idx, tickers,
        ridge_alpha=scfg.get("ridge_alpha", 1.0))
    gc.collect()

    # === PLOT: Individual signal analysis ===
    print("\n" + "=" * 60 + "\nGenerating per-signal figures\n" + "=" * 60)
    plot_score_distributions(shap_scores, influence_scores, emb_shap_scores, FIGS)
    plot_top30_bars(shap_scores, influence_scores, emb_shap_scores, tickers, FIGS)

    # === Run all 6 strategies ===
    print("\n" + "=" * 60 + "\nRunning 6 strategies\n" + "=" * 60)
    sp500_df = fetch_sp500_tickers()
    sector_map = dict(zip(sp500_df["ticker"], sp500_df["sector"]))

    all_selected, all_weights, all_port_rets, all_metrics = [], [], [], []

    # --- Strategy 1: RF+SHAP Only ---
    print("\n--- Strategy 1: RF+SHAP Only ---")
    sel1_idx = np.argsort(shap_scores)[::-1][:k]
    sel1 = [tickers[i] for i in sel1_idx]
    w1 = _ols_weights(train_ret, train_idx, sel1_idx, k)
    ret1 = test_ret[:, sel1_idx] @ w1
    m1 = compute_metrics(ret1, test_idx, label=NAMES[0])
    all_selected.append(sel1); all_weights.append(w1)
    all_port_rets.append(ret1); all_metrics.append(m1)
    print(f"  TE={m1['tracking_error_pct']:.2f}% Sharpe={m1['sharpe']:.2f}")

    # --- Strategy 2: GNN Influence Only ---
    print("\n--- Strategy 2: GNN Influence Only ---")
    sel2_idx = np.argsort(influence_scores)[::-1][:k]
    sel2 = [tickers[i] for i in sel2_idx]
    w2 = _equal_weights(k)
    ret2 = test_ret[:, sel2_idx] @ w2
    m2 = compute_metrics(ret2, test_idx, label=NAMES[1])
    all_selected.append(sel2); all_weights.append(w2)
    all_port_rets.append(ret2); all_metrics.append(m2)
    print(f"  TE={m2['tracking_error_pct']:.2f}% Sharpe={m2['sharpe']:.2f}")

    # --- Strategy 3: Emb-SHAP Only ---
    print("\n--- Strategy 3: Emb-SHAP Only ---")
    sel3_idx = np.argsort(emb_shap_scores)[::-1][:k]
    sel3 = [tickers[i] for i in sel3_idx]
    w3 = _ols_weights(train_ret, train_idx, sel3_idx, k)
    ret3 = test_ret[:, sel3_idx] @ w3
    m3 = compute_metrics(ret3, test_idx, label=NAMES[2])
    all_selected.append(sel3); all_weights.append(w3)
    all_port_rets.append(ret3); all_metrics.append(m3)
    print(f"  TE={m3['tracking_error_pct']:.2f}% Sharpe={m3['sharpe']:.2f}")

    # --- Strategy 4: Hybrid v1 (RF-SHAP + Influence, QP no attn) ---
    print("\n--- Strategy 4: Hybrid v1 (two-way fusion) ---")
    hybrid_v1 = fuse_scores(shap_scores, influence_scores, alpha=scfg["alpha_shap"])
    sel4 = select_with_sector_constraint(hybrid_v1, tickers, sector_map, k=k,
                                          min_per_sector=scfg["min_per_sector"])
    sel4_idx = [tickers.index(t) for t in sel4]
    w4, _ = solve_tracking_error_qp(train_ret[:, sel4_idx], train_idx,
                                     attention_matrix=None,
                                     lambda_reg=0, max_weight=acfg["max_weight"],
                                     solver=acfg["solver"])
    ret4 = test_ret[:, sel4_idx] @ w4
    m4 = compute_metrics(ret4, test_idx, label=NAMES[3])
    all_selected.append(sel4); all_weights.append(w4)
    all_port_rets.append(ret4); all_metrics.append(m4)
    print(f"  TE={m4['tracking_error_pct']:.2f}% Sharpe={m4['sharpe']:.2f}")

    # --- Strategy 5: Hybrid v2 (three-way, QP no attn) ---
    print("\n--- Strategy 5: Hybrid v2 (three-way fusion) ---")
    hybrid_v2 = fuse_scores_v2(shap_scores, emb_shap_scores, influence_scores,
                                alpha=scfg["alpha_shap"],
                                beta=scfg.get("beta_emb_shap", 0.4))
    sel5 = select_with_sector_constraint(hybrid_v2, tickers, sector_map, k=k,
                                          min_per_sector=scfg["min_per_sector"])
    sel5_idx = [tickers.index(t) for t in sel5]
    w5, _ = solve_tracking_error_qp(train_ret[:, sel5_idx], train_idx,
                                     attention_matrix=None,
                                     lambda_reg=0, max_weight=acfg["max_weight"],
                                     solver=acfg["solver"])
    ret5 = test_ret[:, sel5_idx] @ w5
    m5 = compute_metrics(ret5, test_idx, label=NAMES[4])
    all_selected.append(sel5); all_weights.append(w5)
    all_port_rets.append(ret5); all_metrics.append(m5)
    print(f"  TE={m5['tracking_error_pct']:.2f}% Sharpe={m5['sharpe']:.2f}")

    # --- Strategy 6: Hybrid v2 + Attention QP ---
    print("\n--- Strategy 6: Hybrid v2 + Attention-regularised QP ---")
    sel6 = sel5[:]  # same selection as v2
    sel6_idx = sel5_idx[:]
    A_attn = None
    if attn_np is not None and attn_ei_np is not None:
        A_attn = build_attention_submatrix(attn_np, attn_ei_np, sel6_idx, n_total=N+1)
    w6, _ = solve_tracking_error_qp(train_ret[:, sel6_idx], train_idx,
                                     attention_matrix=A_attn,
                                     lambda_reg=acfg["lambda_reg"],
                                     max_weight=acfg["max_weight"],
                                     solver=acfg["solver"])
    ret6 = test_ret[:, sel6_idx] @ w6
    m6 = compute_metrics(ret6, test_idx, label=NAMES[5])
    all_selected.append(sel6); all_weights.append(w6)
    all_port_rets.append(ret6); all_metrics.append(m6)
    print(f"  TE={m6['tracking_error_pct']:.2f}% Sharpe={m6['sharpe']:.2f}")

    # === Generate progression figures ===
    print("\n" + "=" * 60 + "\nGenerating progression figures\n" + "=" * 60)
    plot_cumulative_ladder(test_dates, test_idx, all_port_rets, FIGS)
    plot_rolling_te_ladder(test_dates, test_idx, all_port_rets, FIGS)
    plot_metrics_bars(all_metrics, FIGS)
    plot_selection_overlap(all_selected, FIGS)
    plot_weight_boxplots(all_weights, FIGS)
    plot_sector_comparison(all_selected, sector_map, FIGS)
    plot_summary_table(all_metrics, FIGS)

    # === α-β sensitivity sweep ===
    print("\n" + "=" * 60 + "\nα-β sensitivity sweep\n" + "=" * 60)
    sweep = []
    for a_int in range(0, 11, 2):
        a = a_int / 10.0
        for b_int in range(0, 11 - a_int, 2):
            b = b_int / 10.0
            scores_ab = fuse_scores_v2(shap_scores, emb_shap_scores,
                                        influence_scores, alpha=a, beta=b)
            sel_ab = select_with_sector_constraint(scores_ab, tickers, sector_map,
                                                    k=k, min_per_sector=scfg["min_per_sector"])
            idx_ab = [tickers.index(t) for t in sel_ab]
            w_ab, te_ab = solve_tracking_error_qp(
                train_ret[:, idx_ab], train_idx, attention_matrix=None,
                lambda_reg=0, max_weight=acfg["max_weight"], solver=acfg["solver"])
            # Compute TEST TE
            port_ab = test_ret[:, idx_ab] @ w_ab
            te_test = np.std(port_ab - test_idx) * np.sqrt(252) * 100
            sweep.append({"alpha": round(a, 1), "beta": round(b, 1), "te": round(te_test, 2)})
            print(f"  α={a:.1f} β={b:.1f} → TE={te_test:.2f}%")
    plot_alpha_beta_sweep(sweep, FIGS)

    # === Save results ===
    print("\n" + "=" * 60 + "\nSaving results\n" + "=" * 60)
    save_timeline_results(all_metrics, RES)

    # Save sweep data
    with open(os.path.join(RES, "alpha_beta_sweep.json"), "w") as f:
        json.dump(sweep, f, indent=2)

    # === Print final summary ===
    print("\n" + "=" * 60)
    print("FINAL SUMMARY — ALL 6 STRATEGIES")
    print("=" * 60)
    print(f"{'Strategy':<30s} {'TE%':>7s} {'Sharpe':>7s} {'Beta':>6s} {'MaxDD%':>7s} {'Return%':>8s}")
    print("-" * 68)
    for m in all_metrics:
        print(f"{m['label']:<30s} {m['tracking_error_pct']:7.2f} {m['sharpe']:7.2f} "
              f"{m['beta']:6.2f} {m['max_drawdown_pct']:7.2f} {m['total_return_pct']:8.2f}")

    print(f"\nAll figures saved to {FIGS}/")
    print(f"All results saved to {RES}/")
    print("Timeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timeline ablation study")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.skip_download, args.skip_graph, args.skip_train)
