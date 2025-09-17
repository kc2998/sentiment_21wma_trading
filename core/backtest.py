import numpy as np
import pandas as pd

from .data import get_weekly_prices_21wma

def weekly_backtest_buy_only(
    joined: pd.DataFrame,
    bench_ticker: str = "SPY",
    start: str = "2025-01-01",
    end: str | None = None,
    start_equity: float = 1.0,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Simple weekly buy-only backtest:
    - Enter when entry_signal=True at week t -> hold from week t+1
    - Exit when exit_signal=True at week t   -> flat from week t+1
    - Strategy return in week t uses position at t-1 times ret_wk[t]
    """
    df = joined.copy()
    df = df[
        ["close_wk", "extension_pct", "S_wk", "N", "entry_signal", "exit_signal"]
    ].copy()
    df["ret_wk"] = df["close_wk"].pct_change()

    # Position state machine (0/1)
    pos = pd.Series(0, index=df.index, dtype=int)
    for i in range(len(df) - 1):  # last index can't set i+1
        if pos.iloc[i] == 0 and bool(df["entry_signal"].iloc[i]):
            pos.iloc[i + 1] = 1
        elif pos.iloc[i] == 1 and bool(df["exit_signal"].iloc[i]):
            pos.iloc[i + 1] = 0
        else:
            pos.iloc[i + 1] = pos.iloc[i]
    df["position"] = pos

    # Weekly strategy returns
    gross = df["position"].shift(1).fillna(0) * df["ret_wk"].fillna(0)

    # Transaction costs on flips (entry/exit) charged at start of week t
    turnover = df["position"].fillna(0).diff().abs().fillna(0)
    costs = (cost_bps / 1e4) * turnover
    net = gross - costs

    df["strat_ret"] = net
    df["strat_eq"] = start_equity * (1 + df["strat_ret"]).cumprod()

    # Benchmark on same grid
    bench = get_weekly_prices_21wma(bench_ticker, start, end)
    bench_ret = bench["close_wk"].pct_change()
    bench_ret = bench_ret.reindex(df.index).fillna(0.0)
    bench_eq = start_equity * (1 + bench_ret).cumprod()

    df["bench_ret"] = bench_ret
    df["bench_eq"] = bench_eq
    return df

def perf_summary(eq: pd.Series) -> dict:
    eq = eq.dropna()
    r_total = eq.iloc[-1] / eq.iloc[0] - 1
    weeks = max(len(eq) - 1, 1)
    years = weeks / 52.0
    cagr = (1 + r_total) ** (1 / years) - 1 if years > 0 else float("nan")
    rets = eq.pct_change().dropna()
    vol_w = rets.std()
    sharpe = (rets.mean() / vol_w) * np.sqrt(52) if vol_w and vol_w > 0 else float("nan")
    mdd = ((eq / eq.cummax()) - 1).min()
    return dict(total_return=r_total, cagr=cagr, sharpe=sharpe, max_dd=mdd)
