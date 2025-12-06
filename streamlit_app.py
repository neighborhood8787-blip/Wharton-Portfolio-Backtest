import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# =============== Core ===============

DEFAULT_WEIGHTS = {
    "MSFT": 0.15,
    "TSLA": 0.10,
    "XYL": 0.075,
    "CMCSA": 0.075,
    "DSI": 0.20,
    "ICLN": 0.10,
    "ARKK": 0.10,
    "SHY": 0.20,   # short-term Treasuries proxy
}

def download_price_data(tickers, start, end):
    df = yf.download(
        list(tickers),
        start=start,
        end=end if end else None,
        auto_adjust=False,
        progress=False
    )

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
    # 防呆：沒有資料
    if port_ret is None or port_ret.empty:
        empty_stats = {
            "Total Return": 0.0,
            "CAGR": 0.0,
            "Vol": 0.0,
            "Sharpe": 0.0,
            "Sortino": 0.0,
            "MaxDD": 0.0
        }
        return empty_stats, pd.Series(dtype=float), pd.Series(dtype=float)

    n_days = port_ret.shape[0]
    if n_days == 0:
        empty_stats = {
            "Total Return": 0.0,
            "CAGR": 0.0,
            "Vol": 0.0,
            "Sharpe": 0.0,
            "Sortino": 0.0,
            "MaxDD": 0.0
        }
        return empty_stats, pd.Series(dtype=float), pd.Series(dtype=float)

    total_return = (1 + port_ret).prod() - 1
    ann_return = (1 + total_return) ** (252 / n_days) - 1

    ann_vol = port_ret.std() * np.sqrt(252)

    ann_rf = rf_rate
    sharpe = (ann_return - ann_rf) / ann_vol if ann_vol != 0 else 0.0

    downside = port_ret[port_ret < 0]
    if len(downside) > 0:
        downside_dev = downside.std() * np.sqrt(252)
        sortino = (ann_return - ann_rf) / downside_dev if downside_dev != 0 else 0.0
    else:
        sortino = 0.0

    cum_value = (1 + port_ret).cumprod()
    rolling_max = cum_value.cummax()
    drawdown = cum_value / rolling_max - 1
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    stats = {
        "Total Return": total_return,
        "CAGR": ann_return,
        "Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": max_dd
    }
    return stats, cum_value, drawdown

def yearly_returns(port_ret):
    """每一年的報酬（單年 CAGR = 該年總報酬）"""
    if port_ret is None or port_ret.empty:
        return pd.Series(dtype=float)
    df = port_ret.to_frame("r")
    df["year"] = df.index.year
    yr = df.groupby("year")["r"].apply(lambda x: (1 + x).prod() - 1)
    return yr

# =============== UI ===============

st.set_page_config(
    page_title="Portfolio Backtest",
    page_icon=None,
    layout="wide",
)

st.title("Portfolio Backtest")

# Sidebar
st.sidebar.header("Params")

start_date = st.sidebar.date_input("Start", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End", value=pd.to_datetime("today"))
use_end = st.sidebar.checkbox("Use end", value=True)

rf_rate = st.sidebar.number_input(
    "rf",
    min_value=-0.05,
    max_value=0.10,
    value=0.02,
    step=0.005,
    format="%.3f"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Weights")

weight_inputs = {}
total_w = 0.0
for ticker, w in DEFAULT_WEIGHTS.items():
    val = st.sidebar.number_input(
        ticker,
        min_value=0.0,
        max_value=1.0,
        value=float(w),
        step=0.01,
        format="%.3f"
    )
    weight_inputs[ticker] = val
    total_w += val

st.sidebar.write(f"Sum: {total_w:.3f}")

normalize = st.sidebar.checkbox("Norm to 1", value=True)

run = st.sidebar.button("Run")

if run:
    weights = weight_inputs.copy()
    if normalize and total_w > 0:
        for t in weights:
            weights[t] /= total_w

    with st.spinner("Running..."):
        try:
            prices = download_price_data(
                weights.keys(),
                start_date,
                end_date if use_end else None
            )
            _, port_daily_ret = compute_portfolio_returns(prices, weights)
            stats, cum_value, drawdown = performance_stats(port_daily_ret, rf_rate)
            yr_ret = yearly_returns(port_daily_ret)
        except Exception as e:
            st.error(f"Error: {e}")
        else:
            if prices.empty:
                st.error("No price data. Check dates or tickers.")
            else:
                st.success(
                    f"{prices.index[0].date()} → {prices.index[-1].date()}"
                )

                # KPIs
                st.subheader("Stats")

                col1, col2, col3 = st.columns(3)
                col4, col5 = st.columns(2)

                col1.metric("CAGR", f"{stats['CAGR']*100:,.2f} %")
                col2.metric("Vol", f"{stats['Vol']*100:,.2f} %")
                col3.metric("Sharpe", f"{stats['Sharpe']:,.2f}")
                col4.metric("Sortino", f"{stats['Sortino']:,.2f}")
                col5.metric("MaxDD", f"{stats['MaxDD']*100:,.2f} %")

                st.markdown("---")

                # Charts
                left, right = st.columns(2)

                with left:
                    st.markdown("Equity")
                    st.line_chart(cum_value)

                with right:
                    st.markdown("Drawdown")
                    st.area_chart(drawdown)

                st.markdown("---")
                st.subheader("Weights")
                w_df = pd.DataFrame.from_dict(weights, orient="index", columns=["w"])
                w_df["w(%)"] = w_df["w"] * 100
                st.dataframe(w_df.style.format({"w": "{:.3f}", "w(%)": "{:.2f}"}))

                # ===== 每年報酬 + 選擇年份的 CAGR =====
                st.subheader("Year returns")

                if not yr_ret.empty:
                    # 下拉選年份
                    year_list = list(yr_ret.index)
                    year_sel = st.selectbox("Year", options=year_list)

                    year_cagr = yr_ret.loc[year_sel]
                    st.write(f"{year_sel} CAGR: {year_cagr*100:.2f} %")

                    # 顯示全部年份表格
                    yr_df = yr_ret.to_frame("Return")
                    yr_df["Return(%)"] = yr_df["Return"] * 100
                    st.table(yr_df.style.format({"Return": "{:.4f}", "Return(%)": "{:.2f}"}))
                else:
                    st.info("No yearly data.")

                # 總表
                st.subheader("Table")
                table_df = pd.DataFrame(
                    {
                        "Metric": list(stats.keys()),
                        "Value": [
                            f"{stats['Total Return']*100:,.2f} %",
                            f"{stats['CAGR']*100:,.2f} %",
                            f"{stats['Vol']*100:,.2f} %",
                            f"{stats['Sharpe']:,.2f}",
                            f"{stats['Sortino']:,.2f}",
                            f"{stats['MaxDD']*100:,.2f} %",
                        ],
                    }
                )
                st.table(table_df)
else:
    st.info("Set params and click Run.")
