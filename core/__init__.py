# core/__init__.py
"""Core package for sentiment-21wma."""
from .data import get_weekly_prices_21wma, fetch_company_news_finnhub, ET
from .sentiment import load_finbert, score_and_aggregate_weekly
from .join import join_price_sentiment
from .backtest import weekly_backtest_buy_only, perf_summary

__all__ = [
    "get_weekly_prices_21wma", "fetch_company_news_finnhub", "ET",
    "load_finbert", "score_and_aggregate_weekly",
    "join_price_sentiment",
    "weekly_backtest_buy_only", "perf_summary",
]

__version__ = "0.1.0"
