# 📊 Data-Driven Stock Portfolio Optimization (Brazil)

This project demonstrates a complete data science workflow for building an optimized stock portfolio for the Brazilian market using a mix of:

- ✨ Fundamental analysis (via Fundamentus)
- 🌐 Market data (via Yahoo Finance)
- ⏰ Time-series forecasting (via Facebook Prophet)
- 📈 Modern Portfolio Theory (Markowitz Optimization)

---

## 👤 Who is this for?

This project is designed for:

- Aspiring data scientists interested in applying ML to finance
- People who want to understand portfolio allocation
- Anyone wanting to forecast and optimize investments in Brazil

---

## 🌍 Overview

| Notebook                          | Description                                                  |
|----------------------------------|--------------------------------------------------------------|
| `01_stock_data_download.ipynb`   | Downloads historical prices of Brazilian stocks              |
| `02_risk_return_analysis.ipynb`  | Calculates historical return and volatility                  |
| `03_fundamental_filters.ipynb`   | Filters financially healthy companies using Fundamentus      |
| `04_price_forecast_prophet.ipynb`| Forecasts future stock prices using Prophet                  |
| `05_forecast_risk_return.ipynb`  | Transforms forecasts into expected return and risk           |
| `06_markowitz_optimization.ipynb`| Builds and visualizes optimized portfolios using Markowitz   |

---

## ❓ Problem Statement

> “Based on real and forecasted data, how can we choose the **best stocks to invest in**, and in **what proportion**?”

---

## 🧪 Requirements

To run this project:

```bash
conda create -n datasci python=3.10
conda activate datasci
pip install -r requirements.txt
```

## 🧠 Key Concepts
## ✅ Fundamental Filtering
- Return on Equity > 10%

- Dividend Yield > 3%

- Price/Earnings (P/L) < 20

- Debt/Equity < 1

## ⏳ Forecasting with Prophet
- Predicts stock prices using historical trends

- Calculates projected return and volatility

## 📊 Markowitz Optimization

Builds optimal portfolio allocation by:

- Maximizing Sharpe Ratio
- Minimizing overall volatility

## 🎯 Results
- The optimizer allocates capital based on forecasted performance.

- You get weights per stock and a visual of the efficient frontier.

- Example: if DIRR3.SA has 91% weight, it’s the model’s strongest performer

## 🌐 (Optional) Streamlit App
The app allows you to:

- Visualize forecast results

- Interactively select stocks

- See the optimized portfolio

```bash
streamlit run app.py
```

## 🧾 License
This project is open for educational and portfolio use. No license has been applied — feel free to fork or adapt.

