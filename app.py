import os
from datetime import date

import streamlit as st

from core.data import get_weekly_prices_21wma, fetch_company_news_finnhub
from core.sentiment import load_finbert, score_and_aggregate_weekly
from core.join import join_price_sentiment, compute_trade_events
from core.backtest import weekly_backtest_buy_only, perf_summary
from core.viz import (
    price_sentiment_fig,
    extension_fig,
    equity_vs_bench_fig,
    total_return_bars,
)

st.set_page_config(page_title="Sentiment + 21WMA", layout="wide")

# ---- Sidebar parameters ----
st.sidebar.header("Parameters")

default_start = date(2025, 1, 1)
ticker = st.sidebar.text_input("Ticker", value=os.getenv("DEFAULT_TICKER", "AAPL")).upper()
start_dt = st.sidebar.date_input("Start date", value=default_start, format="YYYY-MM-DD")

use_end = st.sidebar.checkbox("Specify an end date", value=False)
end_dt = st.sidebar.date_input("End date", value=date.today(), format="YYYY-MM-DD") if use_end else None

entry_ext_thr = st.sidebar.number_input(
    "Entry: extension ≤", value=-0.05, step=0.01,
    help="e.g., -0.05 means 5% below the 21 week moving average"
)
neg_thr = st.sidebar.number_input("Negative sentiment ≤", value=-0.05, step=0.01, help="negative sentiment threshold between 0 and -1")
min_headlines = st.sidebar.number_input("Min headlines per week", value=3, min_value=1, step=1)

exit_ext_thr = st.sidebar.number_input(
    "Exit: extension ≥", value=0.05, step=0.01,
    help="e.g., +0.035 means 5% above the 21 week moving average"
)
pos_thr = st.sidebar.number_input("Positive sentiment ≥", value=0.05, step=0.01, help="positive sentiment threshold between 0 and 1")

cost_bps = st.sidebar.number_input("Cost (bps per flip)", value=0.0, step=1.0)

start_str = start_dt.isoformat()
end_str = end_dt.isoformat() if end_dt else None

# ---- Finnhub key ----
FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY") or os.getenv("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    st.warning("Set FINNHUB_API_KEY in Streamlit secrets or environment to fetch headlines.")

# ---- Cache helpers ----
@st.cache_data(show_spinner=False)
def cached_prices(_ticker: str, _start: str, _end: str | None):
    return get_weekly_prices_21wma(_ticker, _start, _end)

@st.cache_data(show_spinner=False)
def cached_news(_ticker: str, _start: str, _end: str | None, _token: str):
    if not _token:
        return None
    return fetch_company_news_finnhub(_ticker, _start, _end, _token)

@st.cache_resource(show_spinner=False)
def cached_finbert():
    return load_finbert(device_preference=None)

st.title("Sentiment + 21WMA (Weekly)")
st.caption(
    "Buy-only: enter when **extension ≤ entry threshold** and **sentiment is negative**; "
    "exit when **extension ≥ exit threshold** and **sentiment is positive**. Backtest vs SPY."
)

if st.button("Run analysis", type="primary"):
    try:
        with st.spinner("Loading prices…"):
            weekly_df = cached_prices(ticker, start_str, end_str)

        with st.spinner("Fetching headlines…"):
            news_df = cached_news(ticker, start_str, end_str, FINNHUB_API_KEY)

        with st.spinner("Scoring sentiment…"):
            if news_df is not None and not news_df.empty:
                finbert = cached_finbert()
                wk = score_and_aggregate_weekly(
                    news_df,
                    min_headlines=int(min_headlines),
                    neg_threshold=float(neg_thr),
                    pos_threshold=float(pos_thr),
                    clf=finbert,
                )
            else:
                import pandas as pd
                wk = pd.DataFrame(columns=["S_wk", "N", "is_negative", "is_positive"])

        # Join + raw signals
        joined = join_price_sentiment(
            weekly_df,
            wk,
            entry_ext_thr=float(entry_ext_thr),
            neg_threshold=float(neg_thr),
            exit_ext_thr=float(exit_ext_thr),
            pos_threshold=float(pos_thr),
            min_headlines=int(min_headlines),
        )

        # Derive stateful events + execution weeks
        joined = compute_trade_events(joined)

        # ---------- Signal health + suggestions ----------
        n_entry = int(joined["entry_signal"].sum())
        n_exit  = int(joined["exit_signal"].sum())

        st.subheader("Signal health")
        st.markdown(
            f"- Entry-signal weeks: **{n_entry}**  \n"
            f"- Exit-signal weeks: **{n_exit}**"
        )

        suggested_entry_ext = min(entry_ext_thr + 0.02, -0.01)   # e.g., -0.07 → -0.05
        suggested_neg_thr   = min(neg_thr + 0.02, -0.01)         # e.g., -0.05 → -0.03
        suggested_min_N     = max(1, int(min_headlines) - 1)

        suggested_exit_ext = max(exit_ext_thr - 0.02, 0.02)      # e.g., +0.12 → +0.10
        suggested_pos_thr  = max(pos_thr - 0.02, 0.00)           # e.g., +0.05 → +0.03

        if n_entry == 0 and n_exit == 0:
            st.warning(
                "No entry **or** exit weeks found with the current thresholds. "
                "Try relaxing thresholds a bit:\n"
                f"- Entry ext ≤ **{suggested_entry_ext:.2f}**, Neg sentiment ≤ **{suggested_neg_thr:.2f}**, Min headlines ≥ **{suggested_min_N}**\n"
                f"- Exit  ext ≥ **{suggested_exit_ext:.2f}**, Pos sentiment ≥ **{suggested_pos_thr:.2f}**\n"
                "You can also extend the date range."
            )
        elif n_entry == 0:
            st.info(
                "No **entry** weeks found. Consider relaxing entry parameters:\n"
                f"- Entry ext ≤ **{suggested_entry_ext:.2f}** (less negative)\n"
                f"- Neg sentiment ≤ **{suggested_neg_thr:.2f}** (less strict)\n"
                f"- Min headlines ≥ **{suggested_min_N}**"
            )
        elif n_exit == 0:
            st.info(
                "No **exit** weeks found. Consider relaxing exit parameters:\n"
                f"- Exit ext ≥ **{suggested_exit_ext:.2f}** (lower positive)\n"
                f"- Pos sentiment ≥ **{suggested_pos_thr:.2f}**"
            )
        # ---------- END Signal health ----------

        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(
                price_sentiment_fig(
                    joined,
                    ticker,
                    neg_threshold=float(neg_thr),
                    min_headlines=int(min_headlines),
                ),
                use_container_width=True,
            )
        with col2:
            st.dataframe(
                joined[
                    [
                        "close_wk",
                        "wma21",
                        "extension_pct",
                        "S_wk",
                        "N",
                        "entry_signal",
                        "exit_signal",
                        "entry_event",
                        "exit_event",
                        "entry_exec",
                        "exit_exec",
                        "position",
                    ]
                ].tail(12)
            )

        st.plotly_chart(
            extension_fig(
                joined, ticker,
                entry_ext_thr=float(entry_ext_thr),
                exit_ext_thr=float(exit_ext_thr),
            ),
            use_container_width=True,
        )

        with st.spinner("Backtesting vs SPY…"):
            bt = weekly_backtest_buy_only(
                joined,
                bench_ticker="SPY",
                start=start_str,
                end=end_str,
                start_equity=1.0,
                cost_bps=float(cost_bps),
            )

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(
                equity_vs_bench_fig(bt, ticker, start_str), use_container_width=True
            )
        with col4:
            st.plotly_chart(total_return_bars(bt, start_str), use_container_width=True)

        s_perf = perf_summary(bt["strat_eq"])
        b_perf = perf_summary(bt["bench_eq"])

        st.markdown(
            f"**Strategy** — Total: {s_perf['total_return']:.2%} | CAGR: {s_perf['cagr']:.2%} | "
            f"Sharpe: {s_perf['sharpe']:.2f} | MaxDD: {s_perf['max_dd']:.2%}"
        )
        st.markdown(
            f"**SPY** — Total: {b_perf['total_return']:.2%} | CAGR: {b_perf['cagr']:.2%} | "
            f"Sharpe: {b_perf['sharpe']:.2f} | MaxDD: {b_perf['max_dd']:.2%}"
        )

    except Exception as e:
        st.exception(e)
else:
    st.info("Set your parameters in the sidebar, then click **Run analysis**.")