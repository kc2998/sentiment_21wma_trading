# Sentiment + 21WMA (Weekly) — Streamlit App

Buy-only strategy: enter when a stock is **undervalued** (extension ≤ entry threshold vs 21-week MA) **and** weekly sentiment is **negative**; exit when **extension ≥ exit threshold** **and** weekly sentiment is **positive**. Backtests vs **SPY**.

## Features
- Weekly prices (yfinance), 21WMA, extension %
- Finnhub company news → FinBERT sentiment → weekly `S_wk = mean(p_pos - p_neg)` with min headline count
- Entry/Exit signals with Plotly charts (sentiment color bands, markers, extension panel with ±10% guides)
- Buy-only weekly backtest vs SPY (equity curves + total-return bars)
- Streamlit UI with configurable thresholds

## Quickstart (pip)
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export FINNHUB_API_KEY=YOUR_KEY
streamlit run app.py
