import os
import logging
import feedparser
from datetime import datetime
from typing import List, Dict

import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from collections import defaultdict
import pandas as pd

# ---------------- SETTINGS ----------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=EURUSD=X&region=US&lang=en-US"
FOREXLIVE_URL = "https://investinglive.com/live-feed/"
ECB_URL = "https://www.ecb.europa.eu/press/intro/news/html/index.en.html"
START_DATE_STR = "2025-11-18"
END_DATE_STR = "2025-11-25"

# Keywords to filter relevant news
CURRENCY_KEYWORDS = {"eurusd", "euro", "forex", "currency", "dollar", "ecb", "fx", "europe"}

# ---------------- NLTK INIT ----------------
def _download_nltk_resources():
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

_download_nltk_resources()
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

def process_text(text: str) -> str:
    lower = text.lower()
    no_punct = lower.translate(str.maketrans("", "", string.punctuation))
    tokens = [lemmatizer.lemmatize(w) for w in no_punct.split() if w not in stop_words]
    return " ".join(tokens)

def analyze_sentiment(text: str) -> float:
    return analyzer.polarity_scores(text).get("compound", 0.0)

def fetch_yahoo_rss(limit: int, start_dt: datetime, end_dt: datetime) -> List[Dict[str, str]]:
    logging.info("Fetching Yahoo Finance RSS feed for EUR/USD...")
    articles: List[Dict[str, str]] = []
    try:
        feed = feedparser.parse(RSS_URL)
        for entry in feed.entries[: limit * 5]:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            link = entry.get("link", "")
            published = entry.get("published") or entry.get("updated") or entry.get("pubDate") or ""
            try:
                d = dateparser.parse(published).date()
            except Exception:
                d = None
            if d is None or d < start_dt.date() or d > end_dt.date():
                continue
            combined = f"{title} {summary}".lower()
            if not any(k in combined for k in CURRENCY_KEYWORDS):
                continue
            clean = process_text(title + " " + summary)
            sentiment = analyze_sentiment(clean)
            articles.append({
                "headline": title,
                "summary": summary,
                "sentiment": sentiment,
                "link": link,
                "date": d.isoformat(),
            })
            if len(articles) >= limit:
                break
    except Exception as e:
        logging.warning(f"Yahoo RSS fetch failed: {e}")
    logging.info(f"✅ Fetched {len(articles)} Yahoo Finance RSS news items.")
    for i, a in enumerate(articles, 1):
        print(f"{i}. {a['headline']}\n   Sentiment: {a['sentiment']:.3f}\n   Link: {a['link']}\n")
    return articles

def fetch_investinglive(limit: int, start_dt: datetime, end_dt: datetime) -> List[Dict[str, str]]:
    logging.info("Fetching InvestingLive feed...")
    articles: List[Dict[str, str]] = []
    try:
        r = requests.get(FOREXLIVE_URL, timeout=15)
        if r.status_code != 200:
            raise RuntimeError(f"status {r.status_code}")
        soup = BeautifulSoup(r.text, "html.parser")
        nodes = soup.find_all(["article", "li", "div"])
        for node in nodes:
            title = node.get_text(separator=" ", strip=True)
            if not title:
                continue
            d = None
            t = node.find("time")
            if t and (t.get("datetime") or t.text):
                try:
                    d = dateparser.parse(t.get("datetime") or t.text).date()
                except Exception:
                    d = None
            if d is None:
                d = end_dt.date()
            if d < start_dt.date() or d > end_dt.date():
                continue
            clean = process_text(title)
            sentiment = analyze_sentiment(clean)
            articles.append({
                "headline": title,
                "summary": "",
                "sentiment": sentiment,
                "link": FOREXLIVE_URL,
                "date": d.isoformat(),
            })
            if len(articles) >= limit:
                break
    except Exception as e:
        logging.warning(f"InvestingLive fetch failed: {e}")
    logging.info(f"✅ Fetched {len(articles)} InvestingLive items.")
    for i, a in enumerate(articles, 1):
        print(f"{i}. {a['headline'][:120]}\n   Sentiment: {a['sentiment']:.3f}\n   Link: {a['link']}\n")
    return articles

def fetch_ecb(limit: int, start_dt: datetime, end_dt: datetime) -> List[Dict[str, str]]:
    logging.info("Fetching ECB press/news...")
    articles: List[Dict[str, str]] = []
    try:
        r = requests.get(ECB_URL, timeout=15)
        if r.status_code != 200:
            raise RuntimeError(f"status {r.status_code}")
        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.find_all(["article", "li"]) or soup.select(r".news, .press, li")
        for node in items:
            a = node.find("a")
            title = (a.text if a else node.get_text(separator=" ", strip=True))
            dt_node = node.find("time") or node.find("span", class_="date")
            d = None
            if dt_node:
                try:
                    d = dateparser.parse(dt_node.get("datetime") or dt_node.text).date()
                except Exception:
                    d = None
            if d is None:
                d = end_dt.date()
            if d < start_dt.date() or d > end_dt.date():
                continue
            clean = process_text(title)
            sentiment = analyze_sentiment(clean)
            link = a.get("href") if a and a.get("href") else ECB_URL
            articles.append({
                "headline": title,
                "summary": "",
                "sentiment": sentiment,
                "link": link,
                "date": d.isoformat(),
            })
            if len(articles) >= limit:
                break
    except Exception as e:
        logging.warning(f"ECB fetch failed: {e}")
    logging.info(f"✅ Fetched {len(articles)} ECB items.")
    for i, a in enumerate(articles, 1):
        print(f"{i}. {a['headline'][:120]}\n   Sentiment: {a['sentiment']:.3f}\n   Link: {a['link']}\n")
    return articles

# ---------------- LSTM MODEL ----------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def _build_sequences(arr: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i-window:i, :])
        y.append(arr[i, 0])
    if not X:
        return None, None
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)
    return X, y

def _train_with_early_stopping(X: torch.Tensor, y: torch.Tensor, max_epochs=100, patience=5, lr=0.001):
    n = X.shape[0]
    val_size = max(1, int(0.2 * n)) if n >= 5 else 0
    if val_size > 0:
        X_train, y_train = X[:-val_size], y[:-val_size]
        X_val, y_val = X[-val_size:], y[-val_size:]
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_state = None
    no_improve = 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                v = criterion(model(X_val), y_val).item()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}, Val: {v:.6f}")
            if v < best_loss:
                best_loss = v
                best_state = {k: v.clone() if hasattr(v, "clone") else v for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        else:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def _aggregate_daily_sentiment(articles: List[Dict[str, str]]):
    daily = defaultdict(list)
    for a in articles:
        try:
            d = dateparser.parse(a["date"]).date()
        except Exception:
            continue
        daily[d].append(a["sentiment"])
    daily_avg = {k: float(np.mean(v)) for k, v in daily.items()}
    overall = float(np.mean([a["sentiment"] for a in articles])) if articles else 0.0
    return daily_avg, overall

def run_pipeline():
    start_dt = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE_STR, "%Y-%m-%d")
    try:
        df_raw = yf.download("USDEUR=X", start=START_DATE_STR, end=END_DATE_STR, interval="1d")
        df_raw = df_raw.reset_index()
        close_col = df_raw["Close"]
        if isinstance(close_col, pd.DataFrame):
            close_series = close_col.iloc[:, 0]
        else:
            close_series = close_col
        df = pd.DataFrame({"Date": df_raw["Date"], "Price": close_series.astype(float)})
        df = df.sort_values("Date")
    except Exception as e:
        logging.error(f"Price download failed: {e}")
        return
    df["Price"] = 1.0 / df["Price"].astype(float)
    dates = df["Date"].dt.date.tolist()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(df["Price"].values.reshape(-1, 1))
    sources = [
        ("Yahoo", fetch_yahoo_rss),
        ("ForexLive", fetch_investinglive),
        ("ECB", fetch_ecb),
    ]
    window = 7
    print("\nPrices (EUR/USD) for week:")
    price_list = df["Price"].astype(float).to_list()
    for d, p in zip(dates, price_list):
        print(f"{d} \t {p:.5f}")
    for name, fn in sources:
        logging.info(f"Source: {name}")
        articles = fn(limit=10, start_dt=start_dt, end_dt=end_dt)
        daily_avg, overall = _aggregate_daily_sentiment(articles)
        logging.info(f"Average sentiment: {overall:.3f}")
        sentiment_series = np.array([[daily_avg.get(d, 0.0)] for d in dates], dtype=float)
        combined = np.hstack([scaled_prices, sentiment_series])
        if len(combined) <= window:
            window = max(1, len(combined) - 1)
        X, y = _build_sequences(combined, window)
        if X is None:
            print(f"{name}: insufficient data for training")
            continue
        model = _train_with_early_stopping(X, y, max_epochs=100, patience=5, lr=0.001)
        last_w = torch.tensor(combined[-window:], dtype=torch.float32).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            pred_scaled = model(last_w).numpy()
        pred_price = scaler.inverse_transform(pred_scaled)
        print(f"{name} → Predicted next-day EUR/USD: {float(pred_price[0][0]):.5f} | sentiment {overall:.3f}")

if __name__ == "__main__":
    run_pipeline()
