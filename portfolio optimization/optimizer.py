# optimizer.py

import numpy as np
from scipy.optimize import minimize

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    """
    Calculate portfolio return, volatility and Sharpe ratio
    """
    weights = np.array(weights)
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    """
    Negative Sharpe ratio (used for minimization)
    """
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def get_max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate=0.0,
                              no_shorting=True, weight_cap=1.0):
    """
    Maximize Sharpe ratio with optional constraints
    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    lower_bound = 0.0 if no_shorting else -1.0
    bounds = tuple((lower_bound, weight_cap) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]

    result = minimize(negative_sharpe, init_guess,
                      args=args, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    return result

def portfolio_volatility(weights, cov_matrix):
    """
    Calculate portfolio volatility
    """
    weights = np.array(weights)
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def get_min_volatility_portfolio(mean_returns, cov_matrix,
                                  no_shorting=True, weight_cap=1.0):
    """
    Minimize portfolio volatility with optional constraints
    """
    num_assets = len(mean_returns)
    args = (cov_matrix,)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    lower_bound = 0.0 if no_shorting else -1.0
    bounds = tuple((lower_bound, weight_cap) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]

    result = minimize(portfolio_volatility, init_guess,
                      args=args, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    return result

