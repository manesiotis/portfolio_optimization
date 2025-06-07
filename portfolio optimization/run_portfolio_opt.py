# run_portfolio_opt.py

from data_loader import load_data, calculate_returns
from optimizer import (
    get_max_sharpe_portfolio,
    get_min_volatility_portfolio,
    portfolio_performance
)
from plotting import (
    generate_random_portfolios,
    plot_efficient_frontier,
    plot_capital_market_line
)
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# === 1. Parameters ===
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
start_date = '2020-01-01'
end_date = '2024-12-31'
risk_free_rate = 0.02  # annualized

# === 2. Load data ===
prices = load_data(tickers, start_date, end_date)
returns = calculate_returns(prices)

# === 3. Compute statistics ===
mean_returns = returns.mean() * 252  # annualized
cov_matrix = returns.cov() * 252     # annualized

# === 4. Max Sharpe Portfolio ===
sharpe_result = get_max_sharpe_portfolio(mean_returns, cov_matrix,
                                         risk_free_rate, no_shorting=True, weight_cap=0.3)
sharpe_weights = sharpe_result.x
sharpe_perf = portfolio_performance(sharpe_weights, mean_returns, cov_matrix, risk_free_rate)

print("📈 Max Sharpe Ratio Portfolio:")
print("Weights:", dict(zip(tickers, np.round(sharpe_weights, 4))))
print("Return: {:.2f}%".format(sharpe_perf[0] * 100))
print("Volatility: {:.2f}%".format(sharpe_perf[1] * 100))
print("Sharpe Ratio: {:.2f}".format(sharpe_perf[2]))
print()

# === 5. Min Volatility Portfolio ===
min_vol_result = get_min_volatility_portfolio(mean_returns, cov_matrix,
                                              no_shorting=True, weight_cap=0.3)
min_vol_weights = min_vol_result.x
min_vol_perf = portfolio_performance(min_vol_weights, mean_returns, cov_matrix, risk_free_rate)

print("🔻 Minimum Volatility Portfolio:")
print("Weights:", dict(zip(tickers, np.round(min_vol_weights, 4))))
print("Return: {:.2f}%".format(min_vol_perf[0] * 100))
print("Volatility: {:.2f}%".format(min_vol_perf[1] * 100))
print("Sharpe Ratio: {:.2f}".format(min_vol_perf[2]))

# === 6. Plot Efficient Frontier + CML ===
results = generate_random_portfolios(10000, mean_returns, cov_matrix, risk_free_rate)
plot_efficient_frontier(results, sharpe_perf, min_vol_perf)
plot_capital_market_line(sharpe_perf, risk_free_rate)
plt.savefig("plots/efficient_frontier_with_cml.png")
plt.show()
