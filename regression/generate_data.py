import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings('ignore') # Suppress yfinance spam

def download_sp500_data():
    print("1. Fetching the list of S&P 500 companies...")
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    tickers_df = pd.read_csv(url)
    tickers = tickers_df['Symbol'].str.replace('.', '-').tolist()

    print(f"2. Downloading historical data for {len(tickers)} stocks...")
    stocks_data = yf.download(tickers, start="2018-01-01", end="2024-01-01")['Close']

    print("3. Cleaning up missing data...")
    returns = stocks_data.pct_change()
    thresh = int(0.90 * len(returns))
    returns = returns.dropna(axis=1, thresh=thresh)
    
    X_features = returns.dropna(axis=0)

    print("4. Downloading S&P 500 Index benchmark data (^GSPC)...")
    benchmark_data = yf.download("^GSPC", start="2018-01-01", end="2024-01-01")['Close']
    y_target = benchmark_data.pct_change().dropna()
    
    if isinstance(y_target, pd.Series):
        y_target = pd.DataFrame(y_target, columns=['^GSPC'])

    print("5. Aligning dates and saving to CSV...")
    X_features.index = pd.to_datetime(X_features.index, utc=True).tz_localize(None).normalize()
    y_target.index = pd.to_datetime(y_target.index, utc=True).tz_localize(None).normalize()
    
    aligned_data = X_features.join(y_target, how='inner')
    
    X_final = aligned_data.iloc[:, :-1]
    y_final = aligned_data.iloc[:, -1:]
    
    X_final.to_csv("sp500_returns.csv")
    y_final.to_csv("sp500_benchmark.csv")
    
    print(f"Success! Saved {X_final.shape[0]} trading days and {X_final.shape[1]} valid stocks.")

if __name__ == "__main__":
    download_sp500_data()