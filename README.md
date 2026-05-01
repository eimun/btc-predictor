# 🚀 BTC Next-Hour Predictor Dashboard

A real-time Bitcoin price prediction dashboard built with Streamlit. The core challenge of this project was balancing **prediction accuracy (Coverage)** with **interval sharpness (Average Width)**. 

Simply inflating the prediction range can artificially boost coverage but renders the forecast mathematically useless for practical trading. My objective was to achieve a **~95% coverage with the tightest possible confidence interval**.

Here is a breakdown of my iterative modeling approach and thought process.

---

## 🧠 Methodology & Thought Process

### 1. Initial Baseline: Standard Monte Carlo
I began with a basic Monte Carlo simulation using log returns and constant volatility.
- **Result:** Coverage was unstable (hovering around 85–90%). The intervals were either too narrow (resulting in misses) or too wide (uninformative).
- **Insight:** Constant volatility is inadequate for crypto markets. Dynamic volatility modeling is critical.

### 2. Volatility Modeling: Implementing FIGARCH
To better capture real market behavior, I integrated a **FIGARCH** (Fractionally Integrated GARCH) model.
- **Why?** FIGARCH excels at capturing long-memory volatility and the volatility clustering that is inherent to cryptocurrency markets.
- **Result:** Coverage improved significantly, but the intervals became **too wide**.
- **Insight:** The model became accurate but overly conservative in its range.

### 3. Distribution Adjustment: Switching to Student-t
Crypto markets exhibit "fat tails" (frequent extreme price moves) that a Normal distribution severely underestimates.
- **Action:** Replaced the Gaussian assumption with a **Student-t distribution**.
- **Result:** This allowed the model to handle extreme events naturally without unnecessarily widening the entire interval bounds. The balance between accuracy and width improved dramatically.

### 4. Monte Carlo Calibration
Instead of relying on a static statistical bound, I simulated thousands of future price paths to extract empirical prediction intervals using percentiles.
- **Tuning:** Initially used standard `[2.5%, 97.5%]` bounds, which proved slightly over-safe. I iteratively tuned the percentiles to **`[3.4%, 96.6%]`**.
- **Result:** Coverage aligned much closer to the target 95% while visibly reducing the interval width.

### 5. Controlling Over-Volatility
I observed that rare, extreme volatility spikes in the historical data were disproportionately inflating the interval width of forward projections.
- **Action:** Applied upper bounds to the simulated variance to control extreme outliers.
- **Result:** Prevented unrealistic price explosions in the simulation, significantly reducing average width without harming the 95% coverage target.

### 6. Backtesting & Iterative Tuning
To validate the model, I performed rolling backtests simulating a live environment (train on past data ➔ predict next hour ➔ compare with actual).
- **Strategy:** Tracked **Coverage**, **Average Width**, and **Winkler Score**.
  - If coverage > 96% ➔ Intervals too wide ➔ Tighten bounds.
  - If coverage < 94% ➔ Intervals too narrow ➔ Widen bounds.
- **Result:** This rigorous tuning loop stabilized the final model parameters.

---

## 🏆 Final Model Performance
The core trade-off was increasing coverage while keeping intervals informative. The final backtest results successfully thread this needle:

- **Coverage (95% Target):** ~95.0%
- **Average Width:** ~1,605.00
- **Winkler Score:** ~2,161.00

---

## 💻 How to Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```
