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


@st.cache_data
def download_price_data(tickers, start, end):
    """抓價格資料並回傳 Adj Close"""
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
    """整體績效指標"""
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
    """每年報酬（單年 CAGR）"""
    if port_ret is None or port_ret.empty:
        return pd.Series(dtype=float)
    df = port_ret.to_frame("r")
    df["year"] = df.index.year
    yr = df.groupby("year")["r"].apply(lambda x: (1 + x).prod() - 1)
    return yr


def downsample(series, max_points=400):
    """下採樣：限制點數，讓圖比較順"""
    s = series.dropna()
    if len(s) <= max_points:
        return s
    step = int(np.ceil(len(s) / max_points))
    return s.iloc[::step]


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
    # 權重處理
    weights = weight_inputs.copy()
    if normalize and total_w > 0:
        for t in weights:
            weights[t] /= total_w

    with st.spinner("Running..."):
        try:
            prices = download_price_data(
                tuple(weights.keys()),
                start_date,
                end_date if use_end else None
            )
            if prices.empty:
                raise ValueError("No price data.")
            _, port_daily_ret = compute_portfolio_returns(prices, weights)
            stats, cum_value, drawdown = performance_stats(port_daily_ret, rf_rate)
            yr_ret = yearly_returns(port_daily_ret)
        except Exception as e:
            st.error(f"Error: {e}")
        else:
            st.success(
                f"{prices.index[0].date()} → {prices.index[-1].date()}"
            )

            # ========= Summary stats =========
            st.subheader("Stats")

            col1, col2, col3 = st.columns(3)
            col4, col5 = st.columns(2)

            col1.metric("CAGR", f"{stats['CAGR']*100:,.2f} %")
            col2.metric("Vol", f"{stats['Vol']*100:,.2f} %")
            col3.metric("Sharpe", f"{stats['Sharpe']:,.2f}")
            col4.metric("Sortino", f"{stats['Sortino']:,.2f}")
            col5.metric("MaxDD", f"{stats['MaxDD']*100:,.2f} %")

            st.markdown("---")

            # ========= Equity & Drawdown (優化過) =========
            left, right = st.columns(2)

            # 先決定是否用週資料
            days = (cum_value.index[-1] - cum_value.index[0]).days
            if days > 3 * 365:
                cum_plot = cum_value.resample("W").last()
                dd_plot = drawdown.resample("W").last()
            else:
                cum_plot = cum_value
                dd_plot = drawdown

            # 再下採樣
            cum_plot = downsample(cum_plot, max_points=400)
            dd_plot = downsample(dd_plot, max_points=400)

            with left:
                st.markdown("Equity")
                st.line_chart(cum_plot.to_frame("Equity"))

            with right:
                st.markdown("Drawdown")
                st.area_chart(dd_plot.to_frame("DD"))

            # ========= Weights =========
            st.markdown("---")
            st.subheader("Weights")
            w_df = pd.DataFrame.from_dict(weights, orient="index", columns=["w"])
            w_df["w(%)"] = w_df["w"] * 100
            st.dataframe(w_df.style.format({"w": "{:.3f}", "w(%)": "{:.2f}"}))

            # ========= Yearly returns + 選年份 =========
            st.subheader("Year returns")
            if not yr_ret.empty:
                year_list = list(yr_ret.index)
                year_sel = st.selectbox("Year", options=year_list)
                year_cagr = yr_ret.loc[year_sel]
                st.write(f"{year_sel} CAGR: {year_cagr*100:.2f} %")

                yr_df = yr_ret.to_frame("Return")
                yr_df["Return(%)"] = yr_df["Return"] * 100
                st.table(yr_df.style.format({"Return": "{:.4f}", "Return(%)": "{:.2f}"}))
            else:
                st.info("No yearly data.")

            # ========= 全部指標表格 =========
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
