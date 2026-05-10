import streamlit as st
import pandas as pd
import numpy as np
import requests
from arch import arch_model
import scipy.stats as stats
import plotly.graph_objects as go
import json
from datetime import datetime

st.set_page_config(page_title="BTC Predictor", page_icon="₿", layout="wide")

st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    h1 {
        color: #f2a900;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    div[data-testid="metric-container"] {
        background-color: #1e2130;
        border: 1px solid #2e3246;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    hr {
        border-color: #2e3246;
    }
</style>
""", unsafe_allow_html=True)

st.title("₿ Bitcoin Next Hour Prediction Dashboard")
st.markdown("<p style='text-align: center; color: #a0aab4; font-size: 1.2rem;'>Advanced volatility forecasting using GBM + FIGARCH + Student-t distribution.</p>", unsafe_allow_html=True)

# -----------------------------
# DATA FETCH
# -----------------------------
@st.cache_data
def get_data():
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 600}
    data = requests.get(url, params=params).json()

    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base","taker_quote","ignore"
    ])

    df["close"] = df["close"].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit='ms')
    df.set_index("time", inplace=True)

    return df

df = get_data()
prices = df["close"]

# -----------------------------
# PREDICTION HISTORY LOGIC
# -----------------------------
def update_actual(prices):
    try:
        now = datetime.utcnow()
        with open("predictions_history.jsonl", "r") as f:
            lines = f.readlines()

        updated = []
        changed = False
        for line in lines:
            record = json.loads(line)

            if record["actual"] is None:
                row_time = pd.to_datetime(record["time"])
                target_time = row_time + pd.Timedelta(hours=1)
                # Only verify after the forecast horizon has passed
                if now >= target_time:
                    # Try to find the actual price at target_time from Binance data
                    actual_price = None
                    # Match by nearest index in the fetched prices
                    idx = prices.index.get_indexer([target_time], method='nearest')[0]
                    if idx >= 0 and idx < len(prices):
                        actual_price = float(prices.iloc[idx])
                    
                    if actual_price is None:
                        # Fallback: use latest available price
                        actual_price = float(prices.iloc[-1])
                    
                    record["actual"] = actual_price
                    changed = True

            updated.append(record)

        if changed:
            with open("predictions_history.jsonl", "w") as f:
                for r in updated:
                    f.write(json.dumps(r) + "\n")

    except:
        pass

def save_prediction(low, high, S0):
    current_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    current_hour = str(current_time)
    
    # Duplicate check for the current hour
    try:
        with open("predictions_history.jsonl", "r") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("time") == current_hour:
                    return  # Skip if already exists
    except FileNotFoundError:
        pass

    record = {
        "time": current_hour,
        "current_price": float(S0),
        "low": float(low),
        "high": float(high),
        "actual": None
    }
    with open("predictions_history.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")

def load_history():
    try:
        with open("predictions_history.jsonl", "r") as f:
            lines = f.readlines()
            data = [json.loads(line) for line in lines]
            return pd.DataFrame(data)
    except:
        return pd.DataFrame()

# call this BEFORE saving new prediction
update_actual(prices)

# -----------------------------
# MODEL
# -----------------------------
log_ret = np.log(prices / prices.shift(1)).dropna()

am = arch_model(log_ret * 100, vol='FIGARCH', dist='studentst')
res = am.fit(disp='off')

sigma = res.conditional_volatility / 100
resid = (log_ret * 100 - res.params['mu']) / res.conditional_volatility

nu = max(4, stats.t.fit(resid, floc=0, fscale=1)[0])

S0 = prices.iloc[-1]
mu = log_ret.mean()

# -----------------------------
# SIMULATION
# -----------------------------
def simulate(n_sims=3000):
    sims = []
    for _ in range(n_sims):
        sigma2 = sigma.iloc[-1]**2
        Z = np.random.standard_t(nu) * np.sqrt((nu-2)/nu)
        price = S0 * np.exp((mu - 0.5*sigma2) + np.sqrt(sigma2)*Z)
        sims.append(price)
    return np.array(sims)

S = simulate()

low, high = np.percentile(S, [3.4, 96.6])

# call this
save_prediction(low, high, S0)

# -----------------------------
# UI
# -----------------------------
st.markdown("### 🎯 Live Prediction")
col1, col2, col3 = st.columns(3)

col1.metric("💰 Current BTC Price", f"${S0:,.2f}", delta="Live")
col2.metric("📉 Predicted Low (95% CI)", f"${low:,.2f}", delta=f"{low - S0:.2f}", delta_color="normal")
col3.metric("📈 Predicted High (95% CI)", f"${high:,.2f}", delta=f"{high - S0:.2f}", delta_color="normal")

st.markdown(f"<h3 style='text-align: center; color: #00ff00; margin-top: 20px;'>🟢 Next hour BTC range: ${low:,.0f} – ${high:,.0f}</h3>", unsafe_allow_html=True)
st.caption(f"<div style='text-align: center;'>Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC</div>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# CHART
# -----------------------------
st.subheader("📊 Price Action & Prediction Bounds")

last_prices = prices.tail(50)
fig = go.Figure()

# Add actual price line
fig.add_trace(go.Scatter(
    x=last_prices.index, 
    y=last_prices.values,
    mode='lines',
    name='BTC Price',
    line=dict(color='#f2a900', width=2)
))

# Add prediction area for the next hour
last_time = last_prices.index[-1]
next_time = last_time + pd.Timedelta(hours=1)

fig.add_trace(go.Scatter(
    x=[last_time, next_time, next_time, last_time],
    y=[S0, high, low, S0],
    fill='toself',
    fillcolor='rgba(0, 255, 0, 0.15)',
    line=dict(color='rgba(0, 255, 0, 0.8)', width=1, dash='dash'),
    name='Next Hour 95% CI Range'
))

# Update layout for premium look
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, t=20, b=0),
    xaxis_title="",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# HISTORY & BACKTEST (Side-by-Side)
# -----------------------------
col_backtest, col_history = st.columns([1, 1.5])

with col_backtest:
    st.subheader("🧪 Backtest Performance")
    st.markdown("<p style='color: #a0aab4; font-size: 0.9rem;'>Rolling 30-day evaluation of the FIGARCH model.</p>", unsafe_allow_html=True)
    import os
    if os.path.exists("backtest_results.jsonl"):
        with open("backtest_results.jsonl", "r") as f:
            bt_rows = [json.loads(line) for line in f if line.strip()]
        
        bt_cov = np.mean([r['low'] <= r['actual'] <= r['high'] for r in bt_rows])
        bt_widths = [r['high'] - r['low'] for r in bt_rows]
        bt_winklers = []
        for r in bt_rows:
            w = r['high'] - r['low']
            if r['actual'] < r['low']:
                bt_winklers.append(w + (2/0.05)*(r['low'] - r['actual']))
            elif r['actual'] > r['high']:
                bt_winklers.append(w + (2/0.05)*(r['actual'] - r['high']))
            else:
                bt_winklers.append(w)
        
        st.metric("Coverage (95% Target)", f"{bt_cov:.1%}")
        st.metric("Average Width", f"${np.mean(bt_widths):,.2f}")
        st.metric("Winkler Score", f"{np.mean(bt_winklers):,.2f}")
    else:
        st.metric("Coverage (95% Target)", "N/A")
        st.metric("Average Width", "N/A")
        st.metric("Winkler Score", "N/A")

with col_history:
    st.subheader("📜 Recent Predictions")
    
    history_df = load_history()
    
    if not history_df.empty:
        # Auto clean and format time for beautiful UI
        history_df['time'] = history_df['time'].astype(str).str.replace('T', ' ').str[:13] + ':00'
        history_df.drop_duplicates(subset=['time'], keep='last', inplace=True)
        
        def get_status(row):
            if pd.isna(row['actual']):
                return '⏳ Pending'
            elif row['low'] <= row['actual'] <= row['high']:
                return '✅ Hit'
            else:
                return '❌ Miss'
    
        history_df['Status'] = history_df.apply(get_status, axis=1)
        history_df = history_df.drop(columns=["correct"], errors="ignore")
        
        display_df = history_df.tail(10).copy()
        display_df.rename(columns={
            'time': 'Time (UTC)',
            'current_price': 'Start Price',
            'low': 'Pred Low',
            'high': 'Pred High',
            'actual': 'Actual Price'
        }, inplace=True)
        
        st.dataframe(
            display_df.set_index('Time (UTC)'),
            use_container_width=True,
            column_config={
                "Start Price": st.column_config.NumberColumn(format="$%.2f"),
                "Pred Low": st.column_config.NumberColumn(format="$%.2f"),
                "Pred High": st.column_config.NumberColumn(format="$%.2f"),
                "Actual Price": st.column_config.NumberColumn(format="$%.2f"),
            }
        )
    else:
        st.info("No predictions yet")
