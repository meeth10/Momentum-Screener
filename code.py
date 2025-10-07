import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from scipy.optimize import minimize
from pypfopt import HRPOpt, EfficientFrontier

def get_stock_data(ticker, period="5y", interval="1d"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    return hist

tickers = ["JioFin.NS", "ADANIPOWER.NS", "RELIANCE.NS", "VEDL.NS", "NMDC.NS"]
moving_average_window = [5, 10, 30, 50, 100, 200]
return_thresholds = {"1M": 3, "6M": 10, "1Y": 20, "3Y": 50, "5Y": 80}

def calc_returns(data):
    current_price = data['Close'].iloc[-1]
    latest_date = data.index[-1]
    tf_days = {"1M": 30, "6M": 182, "1Y": 365, "3Y": 1095, "5Y": 1825}
    returns = {}
    for label, days in tf_days.items():
        past_date = latest_date - timedelta(days=days)
        past_data = data.loc[data.index <= past_date]
        if not past_data.empty:
            past_price = past_data['Close'].iloc[-1]
            ret = (current_price - past_price) / past_price * 100
            returns[label] = ret
        else:
            returns[label] = np.nan
    return returns

scores_list = []
for ticker in tickers:
    data = get_stock_data(ticker)
    for ma in moving_average_window:
        data[f"MA_{ma}"] = data['Close'].rolling(window=ma).mean()
    last_day = data.iloc[-1]
    current_price = last_day['Close']
    score = sum(current_price > last_day[f"MA_{ma}"] for ma in moving_average_window)
    returns = calc_returns(data)
    for period, threshold in return_thresholds.items():
        if period in returns and not np.isnan(returns[period]):
            if returns[period] > threshold:
                score += 1
    scores_list.append({"Ticker": ticker, "Total_Score": score})

scores_df = pd.DataFrame(scores_list).sort_values(by="Total_Score", ascending=False).reset_index(drop=True)
print(scores_df)
price_data = yf.download(tickers, period="5y")['Close']
daily_returns = price_data.pct_change().dropna()
mean_returns = daily_returns.mean() * 252
cov_matrix = daily_returns.cov() * 252
n = len(tickers)

# Equal Weights
ew_weights = np.ones(n) / n
ew_ret = np.dot(ew_weights, mean_returns)
ew_vol = np.sqrt(np.dot(ew_weights.T, np.dot(cov_matrix, ew_weights)))
ew_sharpe = ew_ret / ew_vol

# Mean-Variance Optimization
def portfolio_stats(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = port_return / port_vol
    return np.array([port_return, port_vol, sharpe])

def negative_sharpe(weights, mean_returns, cov_matrix):
    return -portfolio_stats(weights, mean_returns, cov_matrix)[2]

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = tuple((0, 0.5) for _ in range(n))

opt_results = minimize(negative_sharpe, np.ones(n) / n, args=(mean_returns, cov_matrix), 
                       method='SLSQP', bounds=bounds, constraints=constraints)

mvo_weights = opt_results.x
mvo_ret, mvo_vol, mvo_sharpe = portfolio_stats(mvo_weights, mean_returns, cov_matrix)

# HRP
hrp = HRPOpt(daily_returns)
hrp_weights = hrp.optimize()
hrp_ret, hrp_vol, hrp_sharpe = hrp.portfolio_performance()
# Momentum
ef = EfficientFrontier(mean_returns, cov_matrix)
mom_weights = ef.max_sharpe()
mom_ret, mom_vol, mom_sharpe = ef.portfolio_performance()

plt.figure(figsize=(8,5))
sns.barplot(x="Ticker", y="Total_Score", data=scores_df, palette="Blues_d")
plt.title("Momentum + Moving Average Score")
plt.show()

# (c) Efficient Frontier Approximation
port_returns, port_vols = [], []
for _ in range(5000):
    w = np.random.dirichlet(np.ones(n), size=1)[0]
    r, v, _ = portfolio_stats(w, mean_returns, cov_matrix)
    port_returns.append(r); port_vols.append(v)

plt.figure(figsize=(8,6))
plt.scatter(port_vols, port_returns, c=np.array(port_returns)/np.array(port_vols), cmap="viridis", alpha=0.5)
plt.colorbar(label="Sharpe Ratio")
plt.scatter([ew_vol, mvo_vol, hrp_vol, mom_vol],
            [ew_ret, mvo_ret, hrp_ret, mom_ret],
            c="red", marker="*", s=200, label="Optimized Portfolios")
plt.legend()
plt.xlabel("Volatility (Std Dev)")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier & Optimized Portfolios")
plt.show()
