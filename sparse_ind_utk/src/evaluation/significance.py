import pandas as pd
import numpy as np
import os

def run_significance_test():
    prices_path = 'data/raw/prices.csv'
    alloc_path = 'outputs/results/allocation.csv'
    
    if not os.path.exists(prices_path) or not os.path.exists(alloc_path):
        print("Error: Required data files not found.")
        return

    # Load and fix dates
    prices = pd.read_csv(prices_path, index_col=0)
    prices.index = pd.to_datetime(prices.index, utc=True).tz_localize(None)
    
    alloc = pd.read_csv(alloc_path)
    stocks = alloc['ticker'].tolist()
    weights = alloc['weight'].values
    
    # Calculate returns but DO NOT dropna globally yet!
    returns = prices.pct_change(fill_method=None)
    
    # Only keep the 50 selected stocks + the benchmark, then drop missing days
    cols_to_keep = stocks + ['^GSPC']
    valid_returns = returns[cols_to_keep].dropna()

    port_ret = (valid_returns[stocks] * weights).sum(axis=1)
    bench_ret = valid_returns['^GSPC']

    # Slice for the 24-month In-Validation period (the last run you did)
    test_port = port_ret.loc['2023-01-01':'2024-12-31']
    test_bench = bench_ret.loc['2023-01-01':'2024-12-31']
    active_ret = test_port - test_bench

    n = len(active_ret)
    print(f"DEBUG: Found {n} trading days in the test window.")
    
    if n == 0:
        print("ERROR: Could not find any dates. Check your date range or CSV index.")
        print("Available dates in data:", valid_returns.index[0], "to", valid_returns.index[-1])
        return

    # Block Bootstrap (Block Size = 10 days, 10,000 iterations)
    np.random.seed(42)
    n_boot = 10000
    block_size = 10
    boot_irs = []

    for _ in range(n_boot):
        indices = []
        while len(indices) < n:
            start = np.random.randint(0, n - block_size + 1)
            indices.extend(range(start, start + block_size))
        indices = indices[:n]
        
        sample = active_ret.iloc[indices]
        alpha = sample.mean() * 252
        te = sample.std() * np.sqrt(252)
        boot_irs.append(alpha / te if te > 0 else 0)

    # Results
    ci_low, ci_high = np.percentile(boot_irs, [2.5, 97.5])
    p_value = np.mean(np.array(boot_irs) <= 0)

    print(f"\n--- Significance Results ---")
    print(f"Information Ratio: {active_ret.mean()*252 / (active_ret.std()*np.sqrt(252)):.2f}")
    print(f"95% CI for IR: [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"P-value (H0: IR <= 0): {p_value:.4f}")

if __name__ == "__main__":
    run_significance_test()