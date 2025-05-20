import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Brazilian Stock Portfolio", layout="wide")

st.title("📈 Brazilian Stock Portfolio Optimization")
st.markdown("This app allows you to explore forecasts and optimized portfolios based on fundamental filtering and time series forecasting.")

# === Sidebar ===
st.sidebar.header("📁 Data Loading")
data_dir = st.sidebar.text_input("Data folder path", value="data")

# Load available forecast files
def get_forecast_files():
    folder = Path(data_dir)
    if not folder.exists():
        st.warning(f"⚠️ Folder '{data_dir}' not found.")
        return []
    files = list(folder.glob("forecast_*.csv"))
    if not files:
        st.warning("⚠️ No forecast_*.csv files found in the folder.")
    return [f.name for f in files]

forecast_files = get_forecast_files()
selected_files = st.sidebar.multiselect("Select forecast files:", forecast_files)

# === Main content ===
if selected_files:
    cols = st.columns(len(selected_files))

    expected_returns = {}
    volatilities = {}

    for i, file in enumerate(selected_files):
        df = pd.read_csv(os.path.join(data_dir, file), parse_dates=['ds'])
        future = df[df['ds'] > df['ds'].max() - pd.Timedelta(days=180)]

        if future.shape[0] < 2:
            st.warning(f"Not enough forecast data in {file}.")
            continue

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
            st.metric("📈 Expected Return", f"{expected_return:.2%}")
            st.metric("⚠️ Volatility", f"{volatility:.2%}")

    # === Portfolio Section ===
    st.header("📊 Equal-weighted Portfolio Summary")

    if expected_returns:
        portfolio_return = sum(expected_returns.values()) / len(expected_returns)
        portfolio_vol = sum(volatilities.values()) / len(volatilities)

        st.success(f"Expected portfolio return: {portfolio_return:.2%}")
        st.info(f"Expected portfolio volatility: {portfolio_vol:.2%}")

        df_metrics = pd.DataFrame({
            'Expected Return': expected_returns,
            'Volatility': volatilities
        })

        fig = px.scatter(df_metrics, x='Volatility', y='Expected Return', text=df_metrics.index,
                         title="Risk vs Return (Forecast-based)", size_max=60)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please select at least one forecast CSV file from the sidebar.")
