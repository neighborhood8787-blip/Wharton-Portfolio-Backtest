import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# =============== Backtest core ===============

DEFAULT_WEIGHTS = {
    "MSFT": 0.15,
    "TSLA": 0.10,
    "XYL": 0.075,
    "CMCSA": 0.075,
    "DSI": 0.20,
    "ICLN": 0.10,
    "ARKK": 0.10,
    "SHY": 0.20,   # proxy for short-term Treasuries
}

def download_price_data(tickers, start, end):
    df = yf.download(
        list(tickers),
        start=start,
        end=end if end else None,
        auto_adjust=False,
        progress=False
    )

    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            data = df["Adj Close"]
        else:
            data = df["Close"]
    else:
        if "Adj Close" in df.columns:
            data = df[["Adj Close"]]
        else:
            data = df[["Close"]]

    return data.dropna(how="all")

def compute_portfolio_returns(price_df, weights_dict):
    weights = pd.Series(weights_dict)
    price_df = price_df[weights.index]
    daily_ret = price_df.pct_change().dropna()
    port_daily_ret = (daily_ret * weights).sum(axis=1)
    return daily_ret, port_daily_ret

def performance_stats(port_ret, rf_rate=0.02):
    n_days = port_ret.shape[0]
    total_return = (1 + port_ret).prod() - 1
    ann_return = (1 + total_return) ** (252 / n_days) - 1

    ann_vol = port_ret.std() * np.sqrt(252)

    ann_rf = rf_rate
    sharpe = (ann_return - ann_rf) / ann_vol if ann_vol != 0 else np.nan

    downside = port_ret[port_ret < 0]
    if len(downside) > 0:
        downside_dev = downside.std() * np.sqrt(252)
        sortino = (ann_return - ann_rf) / downside_dev
    else:
        sortino = np.nan

    cum_value = (1 + port_ret).cumprod()
    rolling_max = cum_value.cummax()
    drawdown = cum_value / rolling_max - 1
    max_dd = drawdown.min()

    stats = {
        "Total Return": total_return,
        "Annualized Return (CAGR)": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd
    }
    return stats, cum_value, drawdown

# =============== Streamlit UI ===============

st.set_page_config(
    page_title="Wharton Portfolio Backtest",
    page_icon=None,
    layout="wide",
)

st.title("Wharton Investment Portfolio Backtest")

st.markdown(
    "This dashboard backtests our Wharton Investment Portfolio and "
    "shows annualized return, volatility, Sharpe, Sortino and maximum drawdown."
)

# ---- Sidebar: parameters ----
st.sidebar.header("Backtest Parameters")

start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End date (optional)", value=pd.to_datetime("today"))
use_end = st.sidebar.checkbox("Use end date?", value=True)

rf_rate = st.sidebar.number_input(
    "Risk-free rate (annual, e.g. 0.02)",
    min_value=-0.05,
    max_value=0.10,
    value=0.02,
    step=0.005,
    format="%.3f"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Portfolio Weights")

weight_inputs = {}
total_w = 0.0
for ticker, w in DEFAULT_WEIGHTS.items():
    val = st.sidebar.number_input(
        f"{ticker} weight",
        min_value=0.0,
        max_value=1.0,
        value=float(w),
        step=0.01,
        format="%.3f"
    )
    weight_inputs[ticker] = val
    total_w += val

st.sidebar.write(f"Sum of weights (before normalize): {total_w:.3f}")

normalize = st.sidebar.checkbox("Normalize weights to 1.0", value=True)

run = st.sidebar.button("Run Backtest")

if run:
    # normalize weights if needed
    weights = weight_inputs.copy()
    if normalize and total_w > 0:
        for t in weights:
            weights[t] /= total_w

    # main backtest
    with st.spinner("Downloading data and running backtest..."):
        try:
            prices = download_price_data(
                weights.keys(),
                start_date,
                end_date if use_end else None
            )
            _, port_daily_ret = compute_portfolio_returns(prices, weights)
            stats, cum_value, drawdown = performance_stats(port_daily_ret, rf_rate)
        except Exception as e:
            st.error(f"Error during backtest: {e}")
        else:
            st.success(
                f"Backtest period: {prices.index[0].date()} to {prices.index[-1].date()}"
            )

            # ---- KPI cards ----
            st.subheader("Performance Summary")

            col1, col2, col3 = st.columns(3)
            col4, col5 = st.columns(2)

            col1.metric(
                "Annualized Return (CAGR)",
                f"{stats['Annualized Return (CAGR)']*100:,.2f} %"
            )
            col2.metric(
                "Annualized Volatility",
                f"{stats['Annualized Volatility']*100:,.2f} %"
            )
            col3.metric(
                "Sharpe Ratio",
                f"{stats['Sharpe Ratio']:,.2f}"
            )
            col4.metric(
                "Sortino Ratio",
                f"{stats['Sortino Ratio']:,.2f}"
            )
            col5.metric(
                "Maximum Drawdown",
                f"{stats['Max Drawdown']*100:,.2f} %"
            )

            st.markdown("---")

            # ---- Charts ----
            left, right = st.columns(2)

            with left:
                st.markdown("#### Equity Curve – Growth of $1")
                st.line_chart(cum_value)

            with right:
                st.markdown("#### Drawdown")
                st.area_chart(drawdown)

            # ---- Show weights + stats table for report ----
            st.markdown("---")
            st.subheader("Current Portfolio Weights")
            w_df = pd.DataFrame.from_dict(weights, orient="index", columns=["Weight"])
            w_df["Weight (%)"] = w_df["Weight"] * 100
            st.dataframe(w_df.style.format({"Weight": "{:.3f}", "Weight (%)": "{:.2f}"}))

            st.subheader("Performance Table (for report)")
            table_df = pd.DataFrame(
                {
                    "Metric": list(stats.keys()),
                    "Value": [
                        f"{stats['Total Return']*100:,.2f} %",
                        f"{stats['Annualized Return (CAGR)']*100:,.2f} %",
                        f"{stats['Annualized Volatility']*100:,.2f} %",
                        f"{stats['Sharpe Ratio']:,.2f}",
                        f"{stats['Sortino Ratio']:,.2f}",
                        f"{stats['Max Drawdown']*100:,.2f} %",
                    ],
                }
            )
            st.table(table_df)
else:
    st.info("Set your parameters in the sidebar and click “Run Backtest” to start.")
