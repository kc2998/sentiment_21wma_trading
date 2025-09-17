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
```

## Try the strategy [HERE](sentiment21wmatrading-gxsxwxmxv82jukjyyxgr9v.streamlit.app)
<img width="1913" height="946" alt="image" src="https://github.com/user-attachments/assets/8a8301c1-579f-4b94-a79f-6b002339c11c" />


---

## Data & Models

### Prices
- **Yahoo Finance** via `yfinance` (adjusted closes).
- Resampled to **weekly (W-FRI)** “close” and rolling **21-week MA (21WMA)**.

**Extension (%)**  
\[
\text{extension}_t \;=\; \frac{\text{Close}_t}{\text{21WMA}_t} - 1
\]

### News
- **Finnhub Company News API** for historical headlines.
- Headlines are **deduplicated** (title+URL), normalized (lowercased, whitespace collapsed).

### Sentiment Model
- **FinBERT** (`ProsusAI/finbert`) 3-class classifier: `positive`, `neutral`, `negative`.
- We use 🤗 Transformers **pipeline**; softmax probs give \(p_{\text{pos}}, p_{\text{neg}}, p_{\text{neu}}\).

**Per-headline score**
\[
s \;=\; p_{\text{pos}} - p_{\text{neg}} \;\in\; [-1, 1]
\]
Neutral confidence shrinks both \(p_{\text{pos}}\) and \(p_{\text{neg}}\), so extreme scores are rarer when text is neutral/ambiguous.

**Weekly aggregation**
- Map each headline to a **“sentiment week” ending Friday** (with a 15:45 ET cutover: Fri headlines at/after 15:45 roll to next week).
- Aggregate \(s\) via **median** → \(S_{\text{wk}}\); also compute **N** = number of headlines that week.
- Weeks with **N < min_headlines** are considered **insufficient** for signaling (still shown visually, lighter).

---

## Trading Logic

- **Entry signal (decision at week t):**  
  `extension ≤ ENTRY_EXT_THR` **AND** `S_wk ≤ NEG_THR` **AND** `N ≥ MIN_HEADLINES`  
  (defaults: `ENTRY_EXT_THR=-0.07`, `NEG_THR=-0.05`, `MIN_HEADLINES=3`)

- **Exit signal (decision at week t):**  
  `extension ≥ EXIT_EXT_THR` **AND** `S_wk ≥ POS_THR` **AND** `N ≥ MIN_HEADLINES`  
  (defaults: `EXIT_EXT_THR=0.12`, `POS_THR=0.05`, `MIN_HEADLINES=3`)

- **Execution timing:** decide at **t**, **apply at t+1** (avoids look-ahead).  
  Extra buys while already long (or sells while flat) are **ignored**. P&L accrues only when `position=1`.

**Backtest details**
- Weekly strategy return = `position_{t-1} * asset_return_t − costs_on_flips`
- Costs: `cost_bps` applied on each entry/exit flip.
- Equity normalized to 1.0 at start; benchmark = **SPY** buy-and-hold over same window.
- Metrics: Total Return, CAGR, Sharpe (weekly → annualized), Max Drawdown.

---

## Pipeline Diagram

```mermaid
flowchart TB
    A[Select Ticker & Dates] --> B[Download Daily Prices (yfinance)]
    B --> C[Resample to Weekly (W-FRI)]
    C --> D[Compute 21WMA & Extension (%)]
    A --> E[Fetch Company News (Finnhub)]
    E --> F[Normalize & Deduplicate Headlines]
    F --> G[Score Headlines with FinBERT<br/>(ProsusAI/finbert)]
    G --> H[Map to Sentiment Week (Fri 15:45 ET cutoff)]
    H --> I[Aggregate Weekly Sentiment<br/>S_wk = median(p_pos - p_neg), N=headlines]
    D --> J[Join Price + Weekly Sentiment]
    I --> J
    J --> K[Entry/Exit Signals (t)]
    K --> L[Execution t+1 -> Position Series]
    L --> M[Weekly Backtest & Benchmark (SPY)]
    M --> N[Plots: Price+Bands, Extension, Equity vs SPY, Totals]
    style N fill:#e8f5ff,stroke:#8ecaff
```
