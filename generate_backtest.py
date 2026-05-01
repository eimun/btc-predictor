"""
Run this script to generate backtest_results.jsonl from your Colab model.
Uses the same model architecture as app.py.
"""
import numpy as np
import pandas as pd
import requests
import scipy.stats as stats
from arch import arch_model
from datetime import datetime
import json

print("Fetching data...")
url = "https://data-api.binance.vision/api/v3/klines"
params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 1000}
data = requests.get(url, params=params).json()

df = pd.DataFrame(data, columns=[
    "open_time","open","high","low","close","volume",
    "close_time","qav","num_trades","taker_base_vol","taker_quote_vol","ignore"
])
df["close"] = df["close"].astype(float)
df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
df.set_index("open_time", inplace=True)
prices = df["close"]

np.random.seed(42)
train = 280
test = len(prices) - train - 1

log_ret = np.log(prices / prices.shift(1)).dropna()

def rolling_entropy(x, window=60, bins=20):
    def ent(v):
        p, _ = np.histogram(v, bins=bins, density=True)
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    return x.rolling(window).apply(ent, raw=True)

results = []
print(f"Running backtest on {test} bars...")

for i in range(train, train + test):
    train_ret = log_ret.iloc[i - train:i].clip(lower=-0.05, upper=0.05)
    
    am = arch_model(train_ret * 100, vol='FIGARCH', p=1, o=0, q=1, dist='studentst')
    res = am.fit(disp='off')
    
    sigma_fig = res.conditional_volatility / 100
    resid = (train_ret * 100 - res.params['mu']) / res.conditional_volatility
    nu = max(4, stats.t.fit(resid, floc=0, fscale=1)[0])
    
    S0 = prices.iloc[i]
    mu = train_ret.mean()
    sigma2 = sigma_fig.iloc[-1] ** 2
    
    sims = []
    for _ in range(2500):
        Z = np.random.standard_t(nu) * np.sqrt((nu - 2) / nu)
        price = S0 * np.exp((mu - 0.5 * sigma2) + np.sqrt(sigma2) * Z)
        sims.append(price)
    
    S_t1 = np.array(sims)
    low95, high95 = np.percentile(S_t1, [4.5, 95.5])
    actual = prices.iloc[i + 1]
    
    width = high95 - low95
    alpha = 0.05
    if actual < low95:
        winkler = width + (2/alpha) * (low95 - actual)
    elif actual > high95:
        winkler = width + (2/alpha) * (actual - high95)
    else:
        winkler = width
    
    results.append({
        "time": str(prices.index[i + 1]),
        "low": float(low95),
        "high": float(high95),
        "actual": float(actual),
        "coverage": int(low95 <= actual <= high95),
        "width": float(width),
        "winkler_score": float(winkler)
    })
    
    if (i - train) % 50 == 0:
        print(f"  Progress: {i - train}/{test}")

# Save
with open("backtest_results.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

rdf = pd.DataFrame(results)
print(f"\nCoverage:      {rdf['coverage'].mean():.2%}")
print(f"Avg Width:     {rdf['width'].mean():.2f}")
print(f"Winkler Score: {rdf['winkler_score'].mean():.2f}")
print(f"\nSaved {len(results)} predictions to backtest_results.jsonl")
