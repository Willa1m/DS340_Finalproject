import io
import os
import sys
import logging
from contextlib import redirect_stdout
from datetime import datetime

# Import the original pipeline to preserve core functionality
import news_prediction as npmod

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


def _capture(func, *args, **kwargs):
    """Run a function and capture its stdout into a string."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = func(*args, **kwargs)
    return result, buf.getvalue()


def _format_prices_md(dates, prices):
    lines = ["## Weekly EUR/USD Prices", "", "*Data converted from USDEUR to EURUSD by 1/price*", ""]
    for d, p in zip(dates, prices):
        lines.append(f"- {d}: {p:.5f}")
    lines.append("")
    return "\n".join(lines)


def _format_source_header_md(name):
    return f"## Source: {name}\n"


def _format_news_md(captured_listing):
    lines = ["### News Items", "", "```", captured_listing.strip(), "```", ""]
    return "\n".join(lines)


def _format_training_md(captured_training):
    lines = ["### Training Logs", "", "```", captured_training.strip(), "```", ""]
    return "\n".join(lines)


def _format_prediction_md(pred_price, overall_sentiment):
    return f"### Prediction\n\n- Next-day EUR/USD: **{pred_price:.5f}**\n- Average sentiment: **{overall_sentiment:.3f}**\n\n"


def _download_week_prices(start_str, end_str):
    """Download USDEUR, convert to EURUSD, return DataFrame and scaler."""
    df_raw = npmod.yf.download("USDEUR=X", start=start_str, end=end_str, interval="1d")
    df_raw = df_raw.reset_index()
    close_col = df_raw["Close"]
    if isinstance(close_col, pd.DataFrame):
        close_series = close_col.iloc[:, 0]
    else:
        close_series = close_col
    df = pd.DataFrame({"Date": df_raw["Date"], "Price": close_series.astype(float)})
    df = df.sort_values("Date")
    df["Price"] = 1.0 / df["Price"].astype(float)
    return df


def _train_and_predict_for_source(name, fetch_fn, dates, scaled_prices, scaler, start_dt, end_dt):
    """Fetch news, build sentiment series, train LSTM, return markdown blocks."""
    articles, captured_listing = _capture(fetch_fn, 10, start_dt, end_dt)
    daily_avg, overall = npmod._aggregate_daily_sentiment(articles)
    sentiment_series = np.array([[daily_avg.get(d, 0.0)] for d in dates], dtype=float)
    combined = np.hstack([scaled_prices, sentiment_series])
    window = 7 if len(combined) > 7 else max(1, len(combined) - 1)
    X, y = npmod._build_sequences(combined, window)
    if X is None:
        return _format_source_header_md(name) + _format_news_md(captured_listing) + "数据不足，无法训练。\n\n"
    model, captured_train = _capture(npmod._train_with_early_stopping, X, y, 100, 5, 0.001)
    last_w = torch.tensor(combined[-window:], dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(last_w).numpy()
    pred_price = float(scaler.inverse_transform(pred_scaled)[0][0])
    md = _format_source_header_md(name)
    md += _format_news_md(captured_listing)
    md += _format_training_md(captured_train)
    md += _format_prediction_md(pred_price, overall)
    return md


def generate_markdown_output(output_path="output.md"):
    """
    Run the original pipeline in a structured way and produce a Markdown report.
    Preserves logging, adds error handling, and ensures readability.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    start_dt = datetime.strptime(npmod.START_DATE_STR, "%Y-%m-%d")
    end_dt = datetime.strptime(npmod.END_DATE_STR, "%Y-%m-%d")

    try:
        df = _download_week_prices(npmod.START_DATE_STR, npmod.END_DATE_STR)
    except Exception as e:
        logging.error(f"Price download failed: {e}")
        raise

    dates = df["Date"].dt.date.tolist()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(df["Price"].values.reshape(-1, 1))

    md_parts = []
    md_parts.append("# EUR/USD Weekly Prediction Report")
    md_parts.append("")
    md_parts.append(_format_prices_md(dates, df["Price"].astype(float).to_list()))

    sources = [
        ("Yahoo", npmod.fetch_yahoo_rss),
        ("ForexLive", npmod.fetch_investinglive),
        ("ECB", npmod.fetch_ecb),
    ]

    for name, fn in sources:
        try:
            section = _train_and_predict_for_source(name, fn, dates, scaled_prices, scaler, start_dt, end_dt)
        except Exception as e:
            section = _format_source_header_md(name) + f"抓取/训练失败：{e}\n\n"
        md_parts.append(section)

    md_str = "\n".join(md_parts)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_str)


if __name__ == "__main__":
    try:
        generate_markdown_output(os.path.join(os.path.dirname(__file__), "output.md"))
        print("Markdown report generated: output.md")
    except Exception as ex:
        print(f"Failed to generate markdown: {ex}")