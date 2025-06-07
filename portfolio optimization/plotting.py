# plotting.py

import numpy as np
import matplotlib.pyplot as plt
from optimizer import portfolio_performance

def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate=0.0):
    """
    Simulate random portfolios for Efficient Frontier
    """
    results = {'returns': [], 'volatility': [], 'sharpe': [], 'weights': []}
    num_assets = len(mean_returns)

    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        results['returns'].append(ret)
        results['volatility'].append(vol)
        results['sharpe'].append(sharpe)
        results['weights'].append(weights)

    return results

def plot_efficient_frontier(results, max_sharpe, min_vol):
    """
    Plot the Efficient Frontier and highlight key portfolios
    (Assumes figure is already created by caller)
    """
    returns = np.array(results['returns'])
    volatility = np.array(results['volatility'])
    sharpe = np.array(results['sharpe'])

    scatter = plt.scatter(volatility, returns, c=sharpe, cmap='viridis', marker='o', s=10, alpha=0.6)
    plt.colorbar(scatter, label='Sharpe Ratio')

    # Mark Max Sharpe Portfolio
    plt.scatter(max_sharpe[1], max_sharpe[0], marker='*', color='r', s=200, label='Max Sharpe')

    # Mark Min Volatility Portfolio
    plt.scatter(min_vol[1], min_vol[0], marker='X', color='blue', s=200, label='Min Volatility')

    plt.title('Efficient Frontier')
    plt.xlabel('Volatility (Std. Deviation)')
    plt.ylabel('Expected Return')
    plt.legend()
    plt.grid(True)

def plot_capital_market_line(sharpe_point, risk_free_rate):
    """
    Plot the Capital Market Line (CML) given max Sharpe portfolio
    """
    max_return, max_volatility, max_sharpe = sharpe_point

    # Extend CML from risk-free rate up to 1.5 * volatility of max Sharpe
    x_vals = np.linspace(0, max_volatility * 1.5, 100)
    y_vals = risk_free_rate + max_sharpe * x_vals

    plt.plot(x_vals, y_vals, linestyle='--', color='red', label='Capital Market Line')
