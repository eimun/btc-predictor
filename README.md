# 🚀 BTC Next-Hour Predictor Dashboard

A real-time Bitcoin price prediction dashboard that forecasts the next-hour 95% confidence interval using **FIGARCH volatility modeling**, **Student-t fat-tail distribution**, and **Monte Carlo simulation**.

> **Live Dashboard:** [View on Streamlit](https://btc-predictor-nsdnqpljpmhfh3jzqvk4s7.streamlit.app)

---

## 🏆 Final Model Performance

| Metric | Value | Target |
|---|---|---|
| **Coverage (95% CI)** | **95.13%** | ~95% ✅ |
| **Average Width** | **1,249.20** | Narrower = better ✅ |
| **Winkler Score** | **1,671.18** | Lower = better ✅ |
| **Backtest Predictions** | **719** | ~720 hourly bars |

---

## 📊 Project Structure

| Part | Description | Status |
|---|---|---|
| **Part A** — Backtest | 30-day rolling backtest with no data leakage | ✅ Complete |
| **Part B** — Dashboard | Live Streamlit app with real-time predictions | ✅ Deployed |
| **Part C** — Persistence | Prediction history with auto-verification | ✅ Implemented |

---

## 🧠 Methodology & Thought Process

### 1. Initial Baseline: Standard Monte Carlo
Started with a basic GBM simulation using log returns and constant volatility.
- **Result:** Coverage was unstable (~85–90%). Intervals were either too narrow (misses) or too wide (uninformative).
- **Insight:** Constant volatility is inadequate for crypto markets — dynamic volatility modeling is critical.

### 2. Volatility Modeling: FIGARCH
Integrated a **FIGARCH** (Fractionally Integrated GARCH) model to capture real market behavior.
- **Why?** FIGARCH excels at capturing **long-memory volatility** and **volatility clustering** — calm hours cluster together, violent hours cluster together.
- **Result:** Coverage improved significantly, but intervals became too wide.

### 3. Fat-Tail Handling: Student-t Distribution
Crypto markets exhibit "fat tails" — extreme moves that a Normal distribution severely underestimates.
- **Action:** Replaced the Gaussian assumption with a **Student-t distribution** with fitted degrees of freedom.
- **Result:** The model handles extreme events naturally without unnecessarily widening the entire interval.

### 4. Monte Carlo Calibration
Simulated **5,000 future price paths** per prediction to extract empirical prediction intervals.
- **Tuning:** Iteratively tuned percentile bounds from `[2.5, 97.5]` → `[4.5, 95.5]` to optimize the coverage-width tradeoff.
- **Result:** Coverage aligned to 95.13% with tight interval width.

### 5. Variance Control
Applied return clipping (`clip(-0.05, 0.05)`) and variance upper bounds to prevent unrealistic price explosions from extreme historical outliers.
- **Result:** Reduced average width without harming coverage.

### 6. Rolling Backtest Validation
Performed 719-bar rolling backtests simulating a live environment:
- Train on past data only (no peeking) → Predict next hour → Compare with actual
- If coverage > 96% → intervals too wide → tighten
- If coverage < 94% → intervals too narrow → widen

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| **Dashboard** | Streamlit |
| **Volatility Model** | FIGARCH(1,1) via `arch` library |
| **Distribution** | Student-t (fitted ν) |
| **Simulation** | Monte Carlo (5,000 paths) |
| **Data Source** | Binance API (BTCUSDT, 1h bars) |
| **Persistence** | JSONL file-based logging |
| **Deployment** | Streamlit Community Cloud |

---

## ✨ Dashboard Features

- **Real-time BTC price** with live indicator
- **Predicted 95% range** for the next hour with delta from current price
- **Last 50 hours chart** with prediction range as shaded ribbon
- **Backtest metrics** dynamically computed from `backtest_results.jsonl`
- **Prediction history** with auto-verification (✔️ correct / ❌ missed / ⏳ pending)
- **Duplicate prevention** — one prediction per hour, no matter how many refreshes

---

## 💻 How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

---

## 📁 File Structure

```
├── app.py                    # Main Streamlit dashboard
├── backtest_results.jsonl    # 719 predictions from 30-day backtest
├── generate_backtest.py      # Script to regenerate backtest locally
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```
