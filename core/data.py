import os
import re
import time
from datetime import date, datetime, timedelta
from typing import Iterable, Tuple

import pandas as pd
import pytz
import requests
import yfinance as yf

ET = pytz.timezone("America/New_York")

# -------- Prices (weekly, 21WMA, extension) --------

def get_weekly_prices_21wma(ticker: str, start: str, end: str | None = None, buffer_weeks: int = 30) -> pd.DataFrame:
    """Return weekly (W-FRI) DataFrame: close_wk, wma21, extension_pct.
    Uses adjusted close; index tz-aware (ET)."""
    if end is None:
        end = datetime.now(ET).date().isoformat()

    start_ts = pd.Timestamp(start).tz_localize(ET)
    buffered_start = (start_ts - pd.Timedelta(weeks=buffer_weeks)).date().isoformat()

    df = yf.download(ticker, start=buffered_start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No price data for {ticker} between {buffered_start} and {end}")

    # 1-D adjusted close series (robust to MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        close = df['Close'][ticker]
    else:
        close = df['Close']
    close.name = 'close'
    close.index = pd.to_datetime(close.index).tz_localize('UTC').tz_convert(ET).normalize()

    close_wk = close.resample('W-FRI').last()
    wma21 = close_wk.rolling(21, min_periods=1).mean()
    extension_pct = close_wk / wma21 - 1.0

    weekly = pd.DataFrame({'close_wk': close_wk.astype(float), 'wma21': wma21.astype(float), 'extension_pct': extension_pct.astype(float)})
    weekly = weekly[weekly.index >= pd.Timestamp(start, tz=ET)]
    return weekly

# -------- News (Finnhub) --------

def _month_chunks(start_date: str, end_date: str) -> Iterable[Tuple[str, str]]:
    s = pd.to_datetime(start_date).date()
    e = pd.to_datetime(end_date or date.today()).date()
    cur = date(s.year, s.month, 1)
    while cur <= e:
        nxt = (pd.Timestamp(cur) + pd.offsets.MonthEnd(0)).date()
        win_start = max(s, cur)
        win_end = min(e, nxt)
        yield win_start.isoformat(), win_end.isoformat()
        cur = (nxt + timedelta(days=1))

def _norm(s: str) -> str:
    s = (s or '').strip().lower()
    return re.sub(r'\s+', ' ', s)

def _dedupe_rows(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = (_norm(r.get('headline','')), _norm(r.get('url','')))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def fetch_company_news_finnhub(symbol: str, start: str, end: str | None, token: str) -> pd.DataFrame:
    token = token or os.getenv('FINNHUB_API_KEY')
    if not token:
        raise RuntimeError("Missing FINNHUB_API_KEY (env or Streamlit secrets)")

    end = end or date.today().isoformat()
    all_rows: list[dict] = []
    for s, e in _month_chunks(start, end):
        resp = requests.get("https://finnhub.io/api/v1/company-news", params={"symbol": symbol, "from": s, "to": e, "token": token}, timeout=30)
        if resp.status_code != 200:
            print("Finnhub error", resp.status_code, resp.text[:200])
            continue
        rows = resp.json() or []
        all_rows.extend(rows)
        time.sleep(0.2)

    all_rows = _dedupe_rows(all_rows)
    if not all_rows:
        return pd.DataFrame(columns=["dt_et","headline","summary","text","url","source"])  # empty

    df = pd.DataFrame(all_rows)
    ts = pd.to_datetime(df['datetime'], unit='s', errors='coerce', utc=True).dt.tz_convert(ET)
    df['dt_et']   = ts
    df['headline']= df.get('headline','')
    df['summary'] = df.get('summary','')
    df['url']     = df.get('url','')
    df['source']  = df.get('source','')
    df['text']    = (df['headline'].fillna('') + '. ' + df['summary'].fillna('')).str.strip()
    df = df[df['text'].str.len() > 0].copy()
    df.sort_values('dt_et', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df[['dt_et','headline','summary','text','url','source']]
