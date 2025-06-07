# 📊 Project 7: Portfolio Optimization

This project implements Markowitz's Modern Portfolio Theory (MPT) to optimize portfolio allocations. It calculates and visualizes the trade-off between risk and return using the Efficient Frontier and the Capital Market Line (CML).

## 🧠 Goal

Find the optimal portfolio weights to either:
- Maximize the **Sharpe Ratio**
- Minimize the **portfolio volatility**

## 🧱 Project Structure

```bash
portfolio_optimization/
├── data_loader.py          # Load historical price data and compute returns
├── optimizer.py            # Portfolio optimization (Sharpe, Volatility, constraints)
├── metrics.py              # Portfolio metrics (return, volatility, Sharpe)
├── plotting.py             # Efficient Frontier + CML visualization
├── run_portfolio_opt.py    # Main script to run the full workflow
├── plots/                  # Folder for saving generated plots
└── README.md               # Project overview
```

## ⚙️ Features

- Mean-Variance Optimization (Markowitz)
- Max Sharpe Ratio portfolio
- Minimum Volatility portfolio
- Optional constraints:
  - No short selling
  - Max weight per asset
- Efficient Frontier Plot
- Capital Market Line (CML) overlay

## 📈 Output Example

Generated plot saved in `plots/efficient_frontier_with_cml.png`:

- 🟢 Simulated portfolios
- ⭐ Max Sharpe portfolio
- ❌ Minimum volatility portfolio
- 📉 Capital Market Line

## 💾 Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `yfinance`

Install them with:

```bash
pip install numpy pandas matplotlib scipy yfinance
```

## ▶️ How to Run

```bash
python run_portfolio_opt.py
```

This script will:
- Download historical stock data
- Optimize portfolio allocations
- Plot and save the Efficient Frontier with CML
