import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz

ET = pytz.timezone("America/New_York")

def _tz_index(idx):
    try:
        return idx.tz_convert(ET)
    except Exception:
        return pd.DatetimeIndex(idx).tz_localize(ET)

def _sentiment_color(s, n, neg_thr=-0.05, min_n=3):
    # Always draw a band; lighter for low-headline weeks
    if pd.isna(s):
        return "rgba(160,160,160,0.06)"
    if s <= neg_thr:
        alpha = min(abs(float(s)) / 0.25, 1.0)
        a = 0.06 + 0.14 * alpha
        if (pd.notna(n)) and (n < min_n): a *= 0.6
        return f"rgba(220, 20, 60, {a:.2f})"
    if s >= 0.05:
        alpha = min(float(s) / 0.25, 1.0)
        a = 0.06 + 0.14 * alpha
        if (pd.notna(n)) and (n < min_n): a *= 0.6
        return f"rgba(34, 139, 34, {a:.2f})"
    a = 0.06 if (pd.isna(n) or n >= min_n) else 0.04
    return f"rgba(120,120,120,{a:.2f})"

def price_sentiment_fig(joined_df: pd.DataFrame, ticker: str,
                        neg_threshold: float = -0.05, min_headlines: int = 3) -> go.Figure:
    """Price + 21WMA with sentiment bands, concise hover, and simplified trade markers."""
    df = joined_df.copy()
    df.index = _tz_index(df.index)

    # Hover fields for ALL weeks
    s_disp = df["S_wk"].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "—")
    n_disp = df["N"].fillna(0).astype(int).astype(str)
    ext_pct = 100 * df["extension_pct"]
    custom = np.c_[df["wma21"].values, ext_pct.values, s_disp.values, n_disp.values]

    fig = go.Figure()

    # Main price trace (single rich hover)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close_wk"], name="Weekly Close",
        mode="lines", line=dict(width=2),
        customdata=custom,
        hovertemplate=(
            "Week: %{x|%Y-%m-%d}<br>"
            "Close: %{y:.2f}<br>"
            "21WMA: %{customdata[0]:.2f}<br>"
            "Ext: %{customdata[1]:.1f}%<br>"
            "S_wk: %{customdata[2]} (N=%{customdata[3]})<extra></extra>"
        ),
    ))

    # 21WMA (no own hover to avoid duplicates)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["wma21"], name="21-Week MA",
        mode="lines", line=dict(width=2, dash="dash"),
        hoverinfo="skip"
    ))

    # Sentiment bands for EVERY week
    shapes = []
    for i, t0 in enumerate(df.index):
        t1 = df.index[i+1] if i+1 < len(df.index) else (t0 + pd.Timedelta(days=7))
        s = df.get("S_wk", pd.Series(index=df.index)).iloc[i]
        n = df.get("N", pd.Series(index=df.index)).iloc[i]
        color = _sentiment_color(s, n, neg_thr=neg_threshold, min_n=min_headlines)
        shapes.append(dict(type="rect", xref="x", yref="paper",
                           x0=t0, x1=t1, y0=0, y1=1,
                           fillcolor=color, line=dict(width=0), layer="below"))
    fig.update_layout(shapes=shapes)

    # ---- Decision markers (week t): open red circle = Entry, open green diamond = Exit ----
    entry_sig = df.get("entry_signal", pd.Series(False, index=df.index)).fillna(False)
    exit_sig  = df.get("exit_signal",  pd.Series(False, index=df.index)).fillna(False)

    if entry_sig.any():
        fig.add_trace(go.Scatter(
            x=df.index[entry_sig], y=df.loc[entry_sig, "close_wk"],
            name="Entry (decision)", mode="markers",
            marker=dict(symbol="circle-open", size=10,
                        line=dict(width=2, color="crimson")),
            customdata=custom[entry_sig.values],
            hovertemplate=("Decision (t): Entry<br>"
                           "Ext: %{customdata[1]:.1f}% | "
                           "S_wk: %{customdata[2]} (N=%{customdata[3]})<extra></extra>")
        ))

    if exit_sig.any():
        fig.add_trace(go.Scatter(
            x=df.index[exit_sig], y=df.loc[exit_sig, "close_wk"],
            name="Exit (decision)", mode="markers",
            marker=dict(symbol="diamond-open", size=11,
                        line=dict(width=2, color="green")),
            customdata=custom[exit_sig.values],
            hovertemplate=("Decision (t): Exit<br>"
                           "Ext: %{customdata[1]:.1f}% | "
                           "S_wk: %{customdata[2]} (N=%{customdata[3]})<extra></extra>")
        ))

    # ---- Execution arrows (week t+1): GREEN ▲ open, RED ▼ close ----
    entry_exec = df.get("entry_exec", pd.Series(False, index=df.index)).fillna(False)
    exit_exec  = df.get("exit_exec",  pd.Series(False, index=df.index)).fillna(False)

    if entry_exec.any():
        fig.add_trace(go.Scatter(
            x=df.index[entry_exec], y=df.loc[entry_exec, "close_wk"],
            name="Open position (t+1)", mode="markers",
            marker=dict(size=14, symbol="triangle-up", color="green"),
            hovertemplate=("EXECUTE: Open<br>Week: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>")
        ))
    if exit_exec.any():
        fig.add_trace(go.Scatter(
            x=df.index[exit_exec], y=df.loc[exit_exec, "close_wk"],
            name="Close position (t+1)", mode="markers",
            marker=dict(size=14, symbol="triangle-down", color="crimson"),
            hovertemplate=("EXECUTE: Close<br>Week: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>")
        ))

    fig.update_layout(
        title=f"{ticker} — Weekly Price & 21WMA with Sentiment Bands",
        xaxis_title="Week (ET)", yaxis_title="Price",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=45, r=20, t=70, b=40),
    )
    return fig

def extension_fig(joined_df: pd.DataFrame, ticker: str,
                  entry_ext_thr: float = -0.07,
                  exit_ext_thr: float = 0.12) -> go.Figure:
    """Extension panel with dynamic dotted guides from current thresholds."""
    df = joined_df.copy()
    df.index = _tz_index(df.index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=100 * df["extension_pct"],
        name="Extension vs 21WMA", mode="lines", line=dict(width=2)
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)

    # Dynamic threshold guides
    fig.add_hline(y=100 * entry_ext_thr, line_dash="dot", line_color="crimson", opacity=0.9)
    fig.add_annotation(
        xref="paper", x=1.005, y=100 * entry_ext_thr, yref="y",
        text=f"Entry ≤ {entry_ext_thr:+.0%}", showarrow=False, font=dict(color="crimson", size=12)
    )
    fig.add_hline(y=100 * exit_ext_thr, line_dash="dot", line_color="green", opacity=0.9)
    fig.add_annotation(
        xref="paper", x=1.005, y=100 * exit_ext_thr, yref="y",
        text=f"Exit ≥ {exit_ext_thr:+.0%}", showarrow=False, font=dict(color="green", size=12)
    )

    fig.update_layout(
        title=f"{ticker} — Extension to 21WMA (%)",
        xaxis_title="Week (ET)", yaxis_title="Extension (%)",
        hovermode="x unified",
        margin=dict(l=45, r=20, t=60, b=40),
        yaxis=dict(ticksuffix="%"),
    )
    return fig

def equity_vs_bench_fig(bt_df: pd.DataFrame, ticker: str, start: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df["strat_eq"],
        name=f"{ticker} strategy", mode="lines", line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df["bench_eq"],
        name="SPY", mode="lines", line=dict(width=2, dash="dash")
    ))
    fig.update_layout(
        title=f"{ticker} — Strategy vs SPY (from {start})",
        xaxis_title="Week (ET)",
        yaxis_title="Equity (normalized)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=45, r=20, t=70, b=40),
    )
    return fig


def total_return_bars(bt_df: pd.DataFrame, start: str) -> go.Figure:
    strat_total = bt_df["strat_eq"].iloc[-1] / bt_df["strat_eq"].iloc[0] - 1.0
    bench_total = bt_df["bench_eq"].iloc[-1] / bt_df["bench_eq"].iloc[0] - 1.0
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Strategy", "SPY"],
        y=[100 * strat_total, 100 * bench_total],
        text=[f"{100 * strat_total:.1f}%", f"{100 * bench_total:.1f}%"],
        textposition="auto",
    ))
    fig.update_layout(
        title=f"Total Return since {start}",
        yaxis_title="Total Return (%)",
        xaxis_title="Portfolio",
        margin=dict(l=45, r=20, t=70, b=40),
    )
    return fig