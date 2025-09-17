import pandas as pd
import pytz

ET = pytz.timezone("America/New_York")

def join_price_sentiment(
    weekly_df: pd.DataFrame,
    wk: pd.DataFrame,
    entry_ext_thr: float = -0.10,
    neg_threshold: float = -0.05,
    exit_ext_thr: float = 0.10,
    pos_threshold: float = 0.05,
    min_headlines: int = 3,
) -> pd.DataFrame:
    """Join weekly prices with weekly sentiment + build raw entry/exit conditions."""
    out = weekly_df.copy()

    if len(wk.index) and (wk.index.tz is None):
        wk = wk.copy()
        wk.index = wk.index.tz_localize(ET)

    wk_small = (
        wk.reindex(columns=["S_wk", "N", "is_negative", "is_positive"])
        if not wk.empty
        else pd.DataFrame(columns=["S_wk", "N", "is_negative", "is_positive"])
    )
    out = out.join(wk_small, how="left")

    out["is_negative"] = (
        (out["S_wk"] <= neg_threshold) & (out["N"] >= min_headlines)
    ).fillna(False)
    out["is_positive"] = (
        (out["S_wk"] >= pos_threshold) & (out["N"] >= min_headlines)
    ).fillna(False)

    out["is_undervalued"] = (out["extension_pct"] <= entry_ext_thr).fillna(False)
    out["is_stretched"]   = (out["extension_pct"] >= exit_ext_thr).fillna(False)

    # Raw conditions (not yet stateful)
    out["entry_signal"] = out["is_undervalued"] & out["is_negative"]
    out["exit_signal"]  = out["is_stretched"]   & out["is_positive"]
    return out

def compute_trade_events(joined: pd.DataFrame) -> pd.DataFrame:
    """Derive stateful entry/exit events and execution (t+1) flags from raw signals."""
    df = joined.copy()
    n = len(df)
    pos = pd.Series(0, index=df.index, dtype=int)
    entry_event = pd.Series(False, index=df.index)
    exit_event  = pd.Series(False, index=df.index)
    entry_exec  = pd.Series(False, index=df.index)  # execution week t+1
    exit_exec   = pd.Series(False, index=df.index)

    for i in range(n - 1):  # last row can't set t+1
        if pos.iloc[i] == 0 and bool(df["entry_signal"].iloc[i]):
            entry_event.iloc[i] = True
            entry_exec.iloc[i + 1] = True
            pos.iloc[i + 1] = 1
        elif pos.iloc[i] == 1 and bool(df["exit_signal"].iloc[i]):
            exit_event.iloc[i] = True
            exit_exec.iloc[i + 1] = True
            pos.iloc[i + 1] = 0
        else:
            pos.iloc[i + 1] = pos.iloc[i]

    df["position"]    = pos
    df["entry_event"] = entry_event   # decision week (t)
    df["exit_event"]  = exit_event    # decision week (t)
    df["entry_exec"]  = entry_exec    # execution week (t+1)
    df["exit_exec"]   = exit_exec     # execution week (t+1)
    return df