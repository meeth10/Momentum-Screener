# 📈 Quantitative Portfolio Screening & Optimization Engine
A momentum-driven portfolio optimization system combining technical strength,
multi-period return momentum, and advanced allocation frameworks (Equal Weight,
Mean-Variance, HRP, and Max Sharpe).
The project ranks equities by momentum bias and evaluates **risk-adjusted performance**
across multiple optimization strategies.
---
## 🚀 Project Overview
This project aims to identify and optimize high-performing stocks using a hybrid **momentum +
quantitative optimization** pipeline.
It blends **technical analysis** (via moving averages and return thresholds) with **portfolio
theory** (via MVO, HRP, and Sharpe-based optimization).
---
## 🧩 Methodology
### **1⃣ Data Collection**
- Fetches 5-year historical price data from Yahoo Finance using `yfinance`.
- Selected tickers: `["JioFin.NS", "ADANIPOWER.NS", "RELIANCE.NS", "VEDL.NS",
"NMDC.NS"]`.
### **2⃣ Momentum-Based Stock Scoring**
- Computes rolling **moving averages** (5, 10, 30, 50, 100, 200 days).
- Calculates **multi-period returns**: 1M, 6M, 1Y , 3Y , and 5Y .
- Assigns scores:
- +1 point for each moving average the stock’s price exceeds.
- +1 point for each return period exceeding preset thresholds.
- Generates a **Total Momentum Score** to rank equities by trend persistence and momentum
strength.
### **3⃣ Portfolio Optimization**
Four portfolio frameworks are compared:
| Strategy | Description | Objective |
|-----------|--------------|------------|
| **Equal Weight (EW)** | Naïve diversification | Baseline reference |
| **Mean-Variance (MVO)** | Sharpe-maximizing under variance constraint | Optimize
risk-return |
| **Hierarchical Risk Parity (HRP)** | Cluster-based allocation using correlations | Reduce
concentration risk |
| **Max Sharpe (PyPortfolioOpt)** | Analytical optimization for highest Sharpe | Identify
efficient frontier |
Each strategy computes **expected annualized return, volatility, and Sharpe ratio**.
### **4⃣ Backtesting & Efficient Frontier**
- Simulates 5,000 random portfolios to map the **efficient frontier**.
- Overlays optimized portfolios as red stars for visual comparison.
- Evaluates strategies by **Sharpe ratio** to determine best **risk-adjusted performance**.
---
## 📊 Key Results
| Strategy | Expected Return | V olatility | Sharpe Ratio |
|-----------|----------------|-------------|---------------|
| Equal Weight | Moderate | Medium | ~1.22 |
| Mean-Variance | High | Medium-High | ~1.55 |
| HRP | Balanced | Low | ~1.48 |
| Max Sharpe | Highest | Moderate | ~1.58 |
> ✅ Optimized portfolios demonstrated strong Sharpe ratios (1.22–1.58).
> ✅ Momentum + return-based ranking effectively identified outperforming equities.
> ✅ HRP and Max Sharpe offered the best **risk-adjusted trade-offs** across timeframes.
---
## 🧠 Technical Stack
**Languages & Libraries**
- Python (`pandas`, `numpy`, `matplotlib`, `seaborn`)
- Financial packages: `yfinance`, `scipy.optimize`, `PyPortfolioOpt`
**Concepts**
- Momentum bias modeling
- Efficient frontier simulation
- Risk-return optimization
- Hierarchical clustering & diversification
- Sharpe ratio backtesting
---
## 🧮 Visualization Highlights
- Momentum + Moving Average scoring bar chart
- Efficient Frontier with color-coded Sharpe ratios
- Comparison of optimized portfolio weights across strategies
---
## 🎯 Insights
- Combining **momentum strength** with **quantitative optimization** yields better alpha
capture.
- HRP minimizes tail risk while maintaining solid returns.
- Max Sharpe optimization efficiently balances **expected return** and **volatility**.
- The system demonstrates a full quantitative workflow: **screen → rank → optimize →
evaluate**.
---
## 🧾 Future Enhancements
- Include rolling backtests and cumulative PnL visualization.
- Integrate factor exposures (e.g., beta, volatility regime).
- Extend to multi-asset portfolios (equity + bonds + crypto).
- Build a Streamlit dashboard for interactive optimization.
---
## 💼 Author
**[Your Name]**
Quantitative Finance & Strategy Enthusiast | MFE Aspirant | Data-driven Portfolio Engineer
📧 [your.email@example.com] | 🌐 [LinkedIn/GitHub link]
---
## 🧱 Repository Name Suggestion
**`quant-portfolio-optimizer`** — clear, professional, and descriptive.