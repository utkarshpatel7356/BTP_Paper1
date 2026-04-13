import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt
import cvxpy as cp
import warnings

warnings.filterwarnings('ignore')

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

def run_shap_portfolio_selection():
    print("1. Loading Market Data...")
    try:
        X = pd.read_csv('sp500_returns.csv', index_col=0)
        y = pd.read_csv('sp500_benchmark.csv', index_col=0)
    except FileNotFoundError:
        print("Error: Could not find the CSV files. Please run generate_data.py first!")
        return

    # Standardize dates
    X.index = pd.to_datetime(X.index, utc=True).tz_localize(None).normalize()
    y.index = pd.to_datetime(y.index, utc=True).tz_localize(None).normalize()

    aligned_data = X.join(y, how='inner')
    y_target = aligned_data.iloc[:, -1]
    X_features = aligned_data.iloc[:, :-1].fillna(0)

    split_idx = int(len(X_features) * 0.7)
    
    X_train = X_features.iloc[:split_idx]
    y_train = y_target.iloc[:split_idx]
    
    X_test = X_features.iloc[split_idx:]
    y_test = y_target.iloc[split_idx:]
    
    split_date = X_features.index[split_idx]
    print(f"Dataset split: Training before {split_date.date()}, Testing after.")

    print("\n2. Training Random Forest Regressor ")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    print("\n3. Calculating SHAP Values (Asset Selection)")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_train) 

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({'Ticker': X_train.columns, 'Mean_Abs_SHAP': mean_abs_shap})
    importance_df = importance_df.sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    
    # Extract just the top 50
    top_50_df = importance_df.head(50).copy()
    top_50_tickers = top_50_df['Ticker'].tolist()
    
    print("\n4. Optimizing Wealth Allocation (CVXPY Convex Solver)...")
    R = X_train[top_50_tickers].values
    r_b = y_train.values
    T = R.shape[0]

    # Define 'w' as a CVXPY math variable of size 50
    w = cp.Variable(50)

    # THE OBJECTIVE: Pure Tracking Error Minimization
    # min 1/T * sum ( (R*w - r_b)^2 )
    tracking_error = R @ w - r_b
    objective = cp.Minimize((1/T) * cp.sum_squares(tracking_error))

    # THE CONSTRAINTS: 
    # 1. Sum of weights == 1
    # 2. All weights >= 0 (No short selling)
    # 3. Optional: w <= 0.15 (Max 15% per stock to prevent extreme concentration. 
    #    You can delete this line if you want pure unconstrained indexing!)
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= 0.15 
    ]

    # SOLVE THE EQUATION
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Extract mathematically perfect weights
    optimal_weights = w.value
    
    # CVXPY sometimes outputs tiny numbers like -1.2e-12 due to floating point math.
    # We clip them to 0 and re-normalize so they add up to exactly 1.0
    optimal_weights = np.clip(optimal_weights, 0, 1)
    optimal_weights /= np.sum(optimal_weights)

    # ---------------------------------------------------------
    print("\n5. Saving Combined Portfolio Data...")
    
    top_50_df['Optimal_Weight_Decimal'] = optimal_weights
    top_50_df['Allocation_Percentage'] = (top_50_df['Optimal_Weight_Decimal'] * 100).round(2).astype(str) + '%'
    
    top_50_df.to_csv("shap_top_50_portfolio.csv", index=False)
    print("Master portfolio saved to 'shap_top_50_portfolio.csv'")
    
    print("\nTop 5 Holdings by Weight:")
    # Sort by weight to see what the solver favored
    print(top_50_df.sort_values(by='Optimal_Weight_Decimal', ascending=False)[['Ticker', 'Mean_Abs_SHAP', 'Allocation_Percentage']].head(5).to_string(index=False))
    
    X_all_top_50 = X_features[top_50_tickers].values
    portfolio_returns = np.dot(X_all_top_50, optimal_weights)
    portfolio_returns = pd.Series(portfolio_returns, index=X_features.index)

    # Pass only the testing period (post-split) to the evaluator
    calculate_tracking_metrics(
        port_returns=portfolio_returns.iloc[split_idx:], 
        bench_returns=y_target.iloc[split_idx:], 
        model_name="Random Forest + SHAP" # (Or "Graph Neural Network")
    )

    # --- GRAPH 1 & 2: SHAP Plots ---
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=20, show=False)
    plt.title("Top 20 Drivers of the S&P 500 (SHAP Absolute Importance)")
    plt.tight_layout()
    plt.savefig("Graph_1_SHAP_Bar.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, max_display=20, show=False)
    plt.title("SHAP Beeswarm: Impact of Stock Returns on S&P 500")
    plt.tight_layout()
    plt.savefig("Graph_2_SHAP_Beeswarm.png", dpi=300)
    plt.close()

    # --- GRAPH 3: Cumulative Tracking Plot (Split & Rebased!) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. IN-SAMPLE PERIOD (Training)
    port_train = portfolio_returns.iloc[:split_idx]
    bench_train = y_target.iloc[:split_idx]
    
    port_cum_train = (1 + port_train).cumprod()
    bench_cum_train = (1 + bench_train).cumprod()

    ax1.plot(bench_cum_train.index, bench_cum_train, label='S&P 500 Benchmark', color='black', linewidth=2)
    #ax1.plot(port_cum_train.index, port_cum_train, label='CVXPY Portfolio', color='blue', alpha=0.8, linewidth=2)
    ax1.set_title("In-Sample Training (2018 to Split)")
    ax1.set_ylabel("Cumulative Growth (1.0 = Initial Investment)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. OUT-OF-SAMPLE PERIOD (Testing) - REBASED TO 1.0
    port_test = portfolio_returns.iloc[split_idx:]
    bench_test = y_target.iloc[split_idx:]
    
    # By calling cumprod() on just the test slice, we automatically rebase day 1 to 1.0!
    port_cum_test = (1 + port_test).cumprod()
    bench_cum_test = (1 + bench_test).cumprod()

    ax2.plot(bench_cum_test.index, bench_cum_test, label='S&P 500 Benchmark', color='black', linewidth=2)
    ax2.plot(port_cum_test.index, port_cum_test, label='CVXPY Portfolio (Locked Weights)', color='blue', alpha=0.8, linewidth=2)
    ax2.set_title("Out-of-Sample Testing (Rebased to 1.0)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add a unified title
    fig.suptitle("Passive Index Tracking: In-Sample vs. True Out-of-Sample (Rebased)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("Graph_3_Cumulative_Tracking_Rebased.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    run_shap_portfolio_selection()