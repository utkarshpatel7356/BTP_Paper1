import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

def build_hybrid_adjacency_matrix(tickers, X_train_df, alpha=0.3):
    print(f"-> Building Hybrid Topology (Alpha = {alpha})")
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    try:
        sectors_df = pd.read_csv(url)
        sectors_df['Symbol'] = sectors_df['Symbol'].str.replace('.', '-')
        sector_map = dict(zip(sectors_df['Symbol'], sectors_df['GICS Sector']))
    except Exception as e:
        print(f"Warning: Could not load sectors ({e}).")
        sector_map = {}

    n = len(tickers)
    A_GICS = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            sec_i = sector_map.get(tickers[i], f"Unknown_{i}")
            sec_j = sector_map.get(tickers[j], f"Unknown_{j}")
            if sec_i == sec_j:
                A_GICS[i, j] = 1.0

    # ---------------------------------------------------------
    # PART 2: The Correlation Matrix (Mathematical Reality)
    # ---------------------------------------------------------
    print("-> Calculating Pearson Correlation Matrix...")
    # Calculate correlation based strictly on the training data
    corr_matrix = X_train_df.corr().values
    
    # In graph theory, negative edge weights can break the math. 
    # We clip negative correlations to 0 (meaning no structural link).
    A_Corr = np.clip(corr_matrix, 0.0, 1.0)

    # ---------------------------------------------------------
    # PART 3: The Hybrid Blend
    # ---------------------------------------------------------
    A_hybrid = (alpha * A_GICS) + ((1.0 - alpha) * A_Corr)
                
    # Standard Graph Theory Normalization: D^(-1/2) * A * D^(-1/2)
    D = np.diag(1.0 / np.sqrt(A_hybrid.sum(axis=1) + 1e-8))
    A_norm = D @ A_hybrid @ D
    
    return torch.FloatTensor(A_norm), sector_map

# ---------------------------------------------------------
# The Graph Neural Network Architecture
# ---------------------------------------------------------
class StockGNN(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super(StockGNN, self).__init__()
        # Layer 1: Learn patterns from historical returns
        self.W1 = nn.Linear(in_features, hidden_dim)
        # Layer 2: Compress into a single "Centrality Score" for the node
        self.W2 = nn.Linear(hidden_dim, 1)

    def forward(self, X, A):
        # Graph Convolution 1: Node talks to its neighbors
        H = torch.relu(torch.matmul(A, self.W1(X)))
        # Graph Convolution 2: Final Scoring
        scores = torch.sigmoid(torch.matmul(A, self.W2(H))).squeeze()
        return scores

# ---------------------------------------------------------
# Performance Metrics Calculation
# ---------------------------------------------------------

def calculate_tracking_metrics(port_returns, bench_returns, model_name="Model"):
    """Calculates quantitative performance metrics for Out-of-Sample data."""
    # 1. Annualized Tracking Error
    daily_diff = port_returns - bench_returns
    te = np.std(daily_diff) * np.sqrt(252) * 100
    
    # 2. Correlation
    correlation = np.corrcoef(port_returns, bench_returns)[0, 1]
    
    # 3. Beta
    covariance = np.cov(port_returns, bench_returns)[0, 1]
    variance = np.var(bench_returns)
    beta = covariance / variance
    
    # 4. Maximum Drawdown
    def max_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min() * 100
        
    port_mdd = max_drawdown(port_returns)
    bench_mdd = max_drawdown(bench_returns)
    
    print("\n" + "="*50)
    print(f"OUT-OF-SAMPLE METRICS: {model_name}")
    print("="*50)
    print(f"Tracking Error (TE):    {te:.2f}%")
    print(f"Correlation:            {correlation:.4f}")
    print(f"Portfolio Beta:         {beta:.4f}")
    print(f"Max Drawdown (Port):   {port_mdd:.2f}%")
    print(f"Max Drawdown (Bench):  {bench_mdd:.2f}%")
    print("="*50 + "\n")

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def run_gnn_portfolio_selection():
    print("1. Loading Market Data...")
    try:
        X = pd.read_csv('sp500_returns.csv', index_col=0)
        y = pd.read_csv('sp500_benchmark.csv', index_col=0)
    except FileNotFoundError:
        print("Error: Could not find CSV files.")
        return

    X.index = pd.to_datetime(X.index, utc=True).tz_localize(None).normalize()
    y.index = pd.to_datetime(y.index, utc=True).tz_localize(None).normalize()

    aligned_data = X.join(y, how='inner')
    y_target = aligned_data.iloc[:, -1]
    X_features = aligned_data.iloc[:, :-1].fillna(0)
    
    # Chronological Split (70% Train, 30% Test)
    split_idx = int(len(X_features) * 0.7)
    X_train = X_features.iloc[:split_idx]
    y_train = y_target.iloc[:split_idx]
    split_date = X_features.index[split_idx]
    print(f"Dataset split: Training before {split_date.date()}, Testing after.")

    tickers = X_train.columns.tolist()

    print("\n2. Building the S&P 500 Graph...")
    A_tensor, sector_map = build_hybrid_adjacency_matrix(tickers, X_train, alpha=0.3)
    
    # Node Features: Each stock's historical returns in the train period
    X_tensor = torch.FloatTensor(X_train.values.T) # Shape: (500 nodes, T days)
    y_tensor = torch.FloatTensor(y_train.values)   # Shape: (T days)

    print("\n3. Training Graph Neural Network (Node Scoring)...")
    # T days becomes our input feature dimension
    num_days = X_tensor.shape[1] 
    model = StockGNN(in_features=num_days, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train the GNN to score nodes such that they mimic the index
    for epoch in range(150):
        optimizer.zero_grad()
        # Forward pass: Get a score (0 to 1) for all 500 stocks
        node_scores = model(X_tensor, A_tensor)
        
        # Soft-allocate weights based on GNN scores
        soft_weights = node_scores / node_scores.sum()
        
        # Predict the index return
        predicted_index = torch.matmul(X_tensor.T, soft_weights)
        
        # Calculate Tracking Error Loss
        loss = torch.nn.functional.mse_loss(predicted_index, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/150 | Tracking Loss: {loss.item():.6f}")

    # ---------------------------------------------------------
    print("\n4. Extracting Top 50 Stocks via GNN Centrality...")
    # Get final scores without tracking gradients
    with torch.no_grad():
        final_scores = model(X_tensor, A_tensor).numpy()

    # Rank and select Top 50
    importance_df = pd.DataFrame({
        'Ticker': tickers,
        'GNN_Centrality_Score': final_scores,
        'Sector': [sector_map.get(t, 'Unknown') for t in tickers]
    })
    importance_df = importance_df.sort_values(by='GNN_Centrality_Score', ascending=False).reset_index(drop=True)
    
    top_50_df = importance_df.head(50).copy()
    top_50_tickers = top_50_df['Ticker'].tolist()

    # ---------------------------------------------------------
    print("\n5. Optimizing Wealth Allocation (CVXPY Convex Solver)...")
    R = X_train[top_50_tickers].values
    r_b = y_train.values
    T_days = R.shape[0]

    w = cp.Variable(50)
    tracking_error = R @ w - r_b
    objective = cp.Minimize((1/T_days) * cp.sum_squares(tracking_error))

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= 0.15
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    optimal_weights = np.clip(w.value, 0, 1)
    optimal_weights /= np.sum(optimal_weights)

    # ---------------------------------------------------------
    print("\n6. Saving GNN Portfolio Data...")
    top_50_df['Optimal_Weight_Decimal'] = optimal_weights
    top_50_df['Allocation_Percentage'] = (top_50_df['Optimal_Weight_Decimal'] * 100).round(2).astype(str) + '%'
    
    top_50_df.to_csv("gnn_top_50_portfolio.csv", index=False)
    print(" Master GNN portfolio saved to 'gnn_top_50_portfolio.csv'")
    
    print("\nTop 5 Holdings by Weight (GNN Selected):")
    print(top_50_df.sort_values(by='Optimal_Weight_Decimal', ascending=False)[['Ticker', 'Sector', 'Allocation_Percentage']].head(5).to_string(index=False))

    
    X_all_top_50 = X_features[top_50_tickers].values
    portfolio_returns = np.dot(X_all_top_50, optimal_weights)
    portfolio_returns = pd.Series(portfolio_returns, index=X_features.index)

    # Pass only the testing period (post-split) to the evaluator
    calculate_tracking_metrics(
        port_returns=portfolio_returns.iloc[split_idx:], 
        bench_returns=y_target.iloc[split_idx:], 
        model_name="GNN"
    )

    # --- GRAPH 1: Sector Representation Bar Chart ---
    plt.figure(figsize=(10, 6))
    sector_counts = top_50_df['Sector'].value_counts()
    sector_counts.plot(kind='bar', color='teal')
    plt.title("GNN Portfolio: Sector Representation (Coverage)")
    plt.ylabel("Number of Stocks Selected")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("GNN_Graph_1_Sector_Coverage.png", dpi=300)
    plt.close()

    # --- GRAPH 2: Cumulative Tracking Plot (Split & Rebased!) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # IN-SAMPLE PERIOD
    port_train = portfolio_returns.iloc[:split_idx]
    bench_train = y_target.iloc[:split_idx]
    port_cum_train = (1 + port_train).cumprod()
    bench_cum_train = (1 + bench_train).cumprod()

    ax1.plot(bench_cum_train.index, bench_cum_train, label='S&P 500 Benchmark', color='black', linewidth=2)
    ax1.plot(port_cum_train.index, port_cum_train, label='GNN Portfolio', color='green', alpha=0.8, linewidth=2)
    ax1.set_title("GNN In-Sample Training (2018 to Split)")
    ax1.set_ylabel("Cumulative Growth")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # OUT-OF-SAMPLE PERIOD - REBASED TO 1.0
    port_test = portfolio_returns.iloc[split_idx:]
    bench_test = y_target.iloc[split_idx:]
    port_cum_test = (1 + port_test).cumprod()
    bench_cum_test = (1 + bench_test).cumprod()

    ax2.plot(bench_cum_test.index, bench_cum_test, label='S&P 500 Benchmark', color='black', linewidth=2)
    ax2.plot(port_cum_test.index, port_cum_test, label='GNN Portfolio (Locked)', color='green', alpha=0.8, linewidth=2)
    ax2.set_title("GNN Out-of-Sample Testing (Rebased to 1.0)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Topological Index Tracking: GNN Asset Selection + CVXPY", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("GNN_Graph_2_Cumulative_Tracking.png", dpi=300)
    plt.close()

    print("✅ All GNN graphs generated! Pipeline complete.")

if __name__ == "__main__":
    run_gnn_portfolio_selection()