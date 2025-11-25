# DS340 Final Project

## Overview
Python project for EUR/USD next-day prediction combining market prices with daily-aligned news sentiment from multiple sources (Yahoo Finance RSS, InvestingLive, ECB). Includes one-week analysis mode and per-source model comparison.

New: Markdown report generation via `generate_markdown.py` that runs the same pipeline and produces a structured `output.md` with weekly prices, per-source news listings, training logs, and predictions.

## Requirements
- Python 3.13
- Packages: feedparser, nltk, vaderSentiment, yfinance, numpy, scikit-learn, torch, requests, beautifulsoup4, pandas, python-dateutil

Install dependencies:
```
python -m pip install feedparser nltk vaderSentiment yfinance numpy scikit-learn torch requests beautifulsoup4 pandas python-dateutil
```

## Run
```
python news_prediction.py
```
Outputs:
- Weekly EUR/USD prices (converted from USDEUR)
- Each source’s news list and average sentiment
- Per-source next-day EUR/USD prediction

## Markdown Report
Generate a Markdown report with the same pipeline and captured logs:
```
python generate_markdown.py
```
This creates `output.md` containing:
- Weekly EUR/USD prices list
- Per-source sections (Yahoo / ForexLive / ECB)
- News items as code blocks (original printed listing)
- Training logs as code blocks
- Next-day EUR/USD prediction and average sentiment

## Files
- `news_prediction.py` — main pipeline with daily sentiment injection, multi-source comparison, early stopping
- `generate_markdown.py` — Markdown generator that preserves core functionality and writes `output.md`
- `withDataset.py` — original dataset-based pipeline and baseline

## Notes
- Network calls are wrapped with basic error handling and timeouts.
- If any source structure changes, parsing may require selector updates.
- The pipeline converts `USDEUR=X` to EUR/USD by `1 / price`.