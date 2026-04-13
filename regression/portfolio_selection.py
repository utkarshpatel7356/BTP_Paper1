import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def run_shap_portfolio_selection():
    print("1. Loading Market Data...")
    try:
        X = pd.read_csv('sp500_returns.csv', index_col=0)
        y = pd.read_csv('sp500_benchmark.csv', index_col=0)
    except FileNotFoundError:
        print("Error: Could not find the CSV files. Please run generate_data.py first!")
        return

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

    print("\n2. Training Random Forest Regressor (ON TRAIN DATA ONLY)...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    print("\n3. Calculating SHAP Values (Asset Selection)...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_train) 

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({'Ticker': X_train.columns, 'Mean_Abs_SHAP': mean_abs_shap})
    importance_df = importance_df.sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    
    # Extract just the top 50 into a new dataframe
    top_50_df = importance_df.head(50).copy()
    top_50_tickers = top_50_df['Ticker'].tolist()
    
    print("\n4. Optimizing Capital Allocation (Weighting on TRAIN DATA)...")
    X_train_top_50 = X_train[top_50_tickers].values
    y_train_vals = y_train.values

    # Objective: Minimize Daily Error PLUS Cumulative Drift
    def objective(weights):
        port_ret = np.dot(X_train_top_50, weights)
        daily_mse = np.sum((port_ret - y_train_vals)**2)
        
        port_cum = np.prod(1 + port_ret)
        bench_cum = np.prod(1 + y_train_vals)
        cum_penalty = (port_cum - bench_cum)**2
        
        return daily_mse + (cum_penalty * 10)

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0.001, 0.15) for _ in range(50)) # Max 15% per stock and min is not 0 in top 50 
    init_guess = np.array(50 * [1./50])

    opt_results = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    optimal_weights = opt_results.x

    top_50_df['Optimal_Weight_Decimal'] = optimal_weights
    
    top_50_df['Allocation_Percentage'] = (top_50_df['Optimal_Weight_Decimal'] * 100).round(2).astype(str) + '%'
    
    top_50_df.to_csv("shap_top_50_portfolio.csv", index=False)
    print("portfolio saved to 'shap_top_50_portfolio.csv'")

    print("\nTop 5 Holdings:")
    print(top_50_df[['Ticker', 'Mean_Abs_SHAP', 'Allocation_Percentage']].head(5).to_string(index=False))

    
    X_all_top_50 = X_features[top_50_tickers].values
    portfolio_returns = np.dot(X_all_top_50, optimal_weights)
    portfolio_returns = pd.Series(portfolio_returns, index=X_features.index)

    port_cum = (1 + portfolio_returns).cumprod()
    bench_cum = (1 + y_target).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(bench_cum.index, bench_cum, label='S&P 500 Benchmark', color='black', linewidth=2)
    plt.plot(port_cum.index, port_cum, label='Optimized SHAP Portfolio (50 Stocks)', color='blue', alpha=0.8, linewidth=2)
    
    plt.axvline(x=split_date, color='red', linestyle='--', linewidth=2, label='Out-of-Sample Start')
    plt.text(split_date, plt.ylim()[1]*0.95, '  UNSEEN FUTURE DATA \u2192', color='red', fontweight='bold')

    plt.title("Sparse Index Tracking: In-Sample Training vs. Out-of-Sample Testing")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Growth (1.0 = Initial Investment)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Graph_3_Cumulative_Tracking.png", dpi=300)
    plt.close()
    daily_diff = portfolio_returns - y_target
    rolling_te = daily_diff.rolling(window=30).std() * np.sqrt(252) * 100 

    plt.figure(figsize=(12, 4))
    plt.plot(rolling_te.index, rolling_te, color='purple', linewidth=1.5)
    plt.axvline(x=split_date, color='red', linestyle='--', linewidth=2, label='Out-of-Sample Start')
    plt.title("Annualized 30-Day Rolling Tracking Error (%)")
    plt.xlabel("Date")
    plt.ylabel("Tracking Error (%)")
    plt.axhline(y=rolling_te.mean(), color='black', linestyle='--', label=f'Overall Mean TE: {rolling_te.mean():.2f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Graph_4_Rolling_Tracking_Error.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    run_shap_portfolio_selection()