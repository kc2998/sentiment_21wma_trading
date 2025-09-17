from datetime import timedelta, time as dtime

import pandas as pd
import pytz
from transformers import pipeline
import torch

ET = pytz.timezone("America/New_York")

# Week bucketing: Friday close logic with 15:45 ET cutoff

def week_bucket(dt_et: pd.Timestamp) -> pd.Timestamp:
    dt_et = dt_et.tz_convert(ET)
    cutoff = dtime(15, 45)
    wd = dt_et.weekday()  # Mon=0..Fri=4
    days_to_fri = 4 - wd
    this_fri = (dt_et + timedelta(days=max(days_to_fri, 0))).date()
    use_next = (wd > 4) or (wd == 4 and dt_et.time() >= cutoff)
    if use_next:
        days_to_next_fri = (11 - wd) if wd <= 4 else (6 - wd + 5)
        fri = (dt_et + timedelta(days=days_to_next_fri)).date()
    else:
        fri = this_fri
    return pd.Timestamp(fri, tz=ET)

# FinBERT loader (cache at app scope)

def load_finbert(device_preference: int | None = None):
    """Load FinBERT and choose the best device automatically.
    Order: Apple Silicon (MPS) → CUDA → CPU.
    You can override with device_preference: 0 for GPU, -1 for CPU.
    """
    # Manual override if provided
    if device_preference is not None:
        device = 0 if device_preference >= 0 else -1
        clf = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, device=device)
        return clf

    # Auto-detect
    try:
        if torch.backends.mps.is_available():
            device = torch.device("mps")  # Apple Silicon GPU
            clf = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, device=device)
            return clf
    except Exception:
        pass

    if torch.cuda.is_available():
        device = 0  # first CUDA GPU
    else:
        device = -1  # CPU

    clf = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, device=device)
    return clf

# Score headlines and aggregate weekly sentiment

def score_and_aggregate_weekly(news_df: pd.DataFrame, min_headlines: int = 3,
                                neg_threshold: float = -0.05, pos_threshold: float = 0.05,
                                clf=None) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame(columns=['S_wk','N','is_negative','is_positive'])

    if clf is None:
        clf = load_finbert()

    news_df = news_df.copy()
    news_df['week_end'] = news_df['dt_et'].apply(week_bucket)

    texts = news_df['text'].tolist()
    probs = []
    B = 32
    for i in range(0, len(texts), B):
        chunk = texts[i:i+B]
        out = clf(chunk, truncation=True, max_length=256)
        for triplet in out:
            d = {o['label'].lower(): o['score'] for o in triplet}
            probs.append((d.get('positive',0.0), d.get('neutral',0.0), d.get('negative',0.0)))

    news_df['p_pos'], news_df['p_neu'], news_df['p_neg'] = zip(*probs)
    news_df['score'] = news_df['p_pos'] - news_df['p_neg']

    wk = (news_df.groupby('week_end')
          .agg(S_wk=('score','mean'), N=('score','size'))
          .sort_index())
    wk['is_negative'] = (wk['S_wk'] <= neg_threshold) & (wk['N'] >= min_headlines)
    wk['is_positive'] = (wk['S_wk'] >= pos_threshold) & (wk['N'] >= min_headlines)
    return wk
