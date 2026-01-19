import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from scipy.optimize import minimize
from pathlib import Path

st.set_page_config(page_title="Brazilian Stock Portfolio", layout="wide")

st.title("Brazilian Stock Portfolio Optimization")
st.markdown("This app allows you to explore forecasts and optimized portfolios based on fundamental filtering and time series forecasting.")

# === Sidebar ===
st.sidebar.header("Data Loading")
data_dir = st.sidebar.text_input("Data folder path", value="data")

# Load available forecast files
def get_forecast_files():
    folder = Path(data_dir)
    if not folder.exists():
        st.warning(f"Folder '{data_dir}' not found.")
        return []
    files = list(folder.glob("forecast_*.csv"))
    if not files:
        st.warning("No forecast_*.csv files found in the folder.")
    return [f.name for f in files]

forecast_files = get_forecast_files()
selected_files = st.sidebar.multiselect("Select forecast files:", forecast_files)

# === Markowitz Optimization ===
def markowitz_optimize(expected_returns: dict, volatilities: dict):
    tickers = list(expected_returns.keys())
    mu = np.array([expected_returns[t] for t in tickers])
    sigma = np.array([volatilities[t] for t in tickers])

    def portfolio_return(weights):
        return np.sum(weights * mu)

    def portfolio_volatility(weights):
        return np.sqrt(np.sum((weights * sigma) ** 2))

    def neg_sharpe(weights):
        return -portfolio_return(weights) / portfolio_volatility(weights)

    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in tickers)
    initial_weights = np.array([1 / len(tickers)] * len(tickers))

    result = minimize(neg_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_weights = result.x

    df_weights = pd.DataFrame({
        'Ticker': tickers,
        'Weight (Max Sharpe)': opt_weights
    })
    
    df_weights['Weight (Max Sharpe)'] = df_weights['Weight (Max Sharpe)'] / df_weights['Weight (Max Sharpe)'].sum()

    return df_weights, {
        'return': portfolio_return(opt_weights),
        'volatility': portfolio_volatility(opt_weights)
    }

# === Main content ===
if selected_files:
    cols = st.columns(len(selected_files))

    expected_returns = {}
    volatilities = {}

    for i, file in enumerate(selected_files):
        df = pd.read_csv(os.path.join(data_dir, file), parse_dates=['ds'])
        future = df[['ds', 'yhat']].copy()

        future['returns'] = future['yhat'].pct_change()
        expected_return = future['returns'].mean() * 252
        volatility = future['returns'].std() * (252**0.5)

        ticker = file.replace("forecast_", "").replace(".csv", "")
        expected_returns[ticker] = expected_return
        volatilities[ticker] = volatility

        with cols[i]:
            st.subheader(ticker)
            fig = px.line(df, x='ds', y='yhat', title=f"Forecast for {ticker}")
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Expected Return", f"{expected_return:.2%}")
            st.metric("Volatility", f"{volatility:.2%}")

    # === Portfolio Section ===
    st.header("Equal-weighted Portfolio Summary")

    if expected_returns:
        n = len(expected_returns)
        portfolio_return = sum(expected_returns.values()) / n
        portfolio_vol = sum(volatilities.values()) / n

        st.success(f"Expected portfolio return: {portfolio_return:.2%}")
        st.info(f"Expected portfolio volatility: {portfolio_vol:.2%}")

        df_metrics = pd.DataFrame({
            'Expected Return': expected_returns,
            'Volatility': volatilities
        })

        fig = px.scatter(df_metrics, x='Volatility', y='Expected Return', text=df_metrics.index,
                         title="Risk vs Return (Forecast-based)", size_max=60)
        st.plotly_chart(fig, use_container_width=True)

        # === Optimized Portfolio Section ===
        st.header("Markowitz Optimized Portfolio")
        df_opt, metrics = markowitz_optimize(expected_returns, volatilities)

        # Format only when displaying
        df_display = df_opt.copy()
        df_display['Weight (Max Sharpe)'] = df_display['Weight (Max Sharpe)'].apply(lambda x: f"{x * 100:.2f}%")
        st.dataframe(df_display)
        st.success(f"Optimal return: {metrics['return']:.2%}")
        st.info(f"Optimal risk: {metrics['volatility']:.2%}")

else:
    st.info("Please select at least one forecast CSV file from the sidebar.")
