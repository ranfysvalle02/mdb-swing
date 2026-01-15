# Building "The Sentinel": A Hybrid AI Swing Trading Bot

**The Problem:** Automated trading bots are terrifying. If you leave them alone, they can drain your account in minutes due to a single bug or a market crash.

**The Solution:** A "Hybrid" system. The bot does the heavy lifting (scanning, math, AI analysis), but it keeps a human in the loop.

In this guide, we are going to build **The Sentinel**, a Python-based agent that:

1. **Scans the market** every hour for technical setups (Trend + RSI).
2. **Reads the news** using an LLM to validate the move.
3. **Executes a "Shadow Trade"** (Paper Trade) to track performance.
4. **Pings your phone** so you can decide if you want to mirror the trade with real money.

---

## The Architecture

We are ditching complex cloud architectures for a robust **All-In-One** Python script running in Docker.

* **The Brain:** Python (Logic) + OpenAI (Sentiment).
* **The Hands:** Alpaca API (Paper Trading Execution).
* **The Eyes:** Alpaca Data API (Price History & News).
* **The Memory:** MongoDB (Trade History).
* **The Nervous System:** `ntfy.sh` (Push Notifications).

---

## Part 1: The Setup

### 1. `requirements.txt`

We need a lightweight environment. We use `alpaca-trade-api` for both data and execution to keep things consistent.

```text
flask
apscheduler
requests
pymongo
langchain
langchain-openai
alpaca-trade-api
pandas
numpy

```

### 2. `Dockerfile`

This ensures our bot runs identically on your laptop or a cheap cloud VPS.

```dockerfile
FROM python:3.9-slim

# Keep Python from buffering stdout so logs show up immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
CMD ["python", "main.py"]

```

### 3. `docker-compose.yml`

We spin up a local MongoDB instance alongside our bot.

```yaml
version: '3.8'
services:
  mongo:
    image: mongo:latest
    container_name: sentinel_mongo
    restart: always
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

  sentinel:
    build: .
    restart: on-failure
    depends_on:
      - mongo
    environment:
      # SYSTEM CONFIG
      MONGO_URI: "mongodb://mongo:27017/"
      NTFY_TOPIC: "my_sentinel_bot"
      PUBLIC_URL: "http://localhost:5000"
      
      # API KEYS (Replace these!)
      OPENAI_API_KEY: "sk-..."
      ALPACA_API_KEY: "PK..."
      ALPACA_SECRET_KEY: "..."
      # Use Paper URL for safety
      ALPACA_BASE_URL: "https://paper-api.alpaca.markets" 

volumes:
  mongo_data:

```

---

## Part 2: The Code (`main.py`)

This is the core logic. It implements the "Three-Green-Lights" strategy: **Trend + Discount + Sentiment**.

```python
import time
import os
import requests
import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from flask import Flask, request
from apscheduler.schedulers.background import BackgroundScheduler
from pymongo import MongoClient

# --- LANGCHAIN & PYDANTIC ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# --- CONFIGURATION ---
SYMBOL = "NVDA"
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:5000")
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "my_bot")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
PERSONALITY = os.getenv("BOT_PERSONALITY", "Standard")

# --- ALPACA CONFIG ---
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
ALPACA_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Initialize Global Alpaca Client
api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL, api_version='v2')

app = Flask(__name__)
client = MongoClient(MONGO_URI)
db = client['shadow_trading']
signals_col = db['history']

# --- 1. THE AI AGENT ---
PROMPTS = {
    "Standard": "You are a Senior Hedge Fund Analyst. Be professional, skeptical, and concise.",
}

class TradeSignal(BaseModel):
    score: int = Field(description="Sentiment score 0-10 (10 is Bullish).")
    reason: str = Field(description="A concise justification.")

class SentinelAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7) 
        self.chain = self._build_chain()
    
    def _build_chain(self):
        system_prompt = f"""{PROMPTS.get(PERSONALITY)}
        Analyze the news headlines. 
        If news is old/irrelevant, score 5.
        Output valid JSON."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Ticker: {ticker}\nHeadlines:\n{headlines}")
        ])
        return prompt | self.llm.with_structured_output(TradeSignal)

    def analyze(self, ticker, headlines):
        if not headlines: return TradeSignal(score=5, reason="No News Found")
        return self.chain.invoke({"ticker": ticker, "headlines": "\n".join(headlines)})

agent = SentinelAgent()

# --- 2. DATA METHODS (ALPACA) ---
def get_clean_data(symbol):
    """
    Fetches 60 days of daily bars from Alpaca (enough for SMA 50).
    """
    try:
        # Fetch 60 days of data to ensure we have enough for SMA 50
        bars = api.get_bars(symbol, TimeFrame.Day, limit=60).df
        
        if bars.empty:
            print(f"⚠️ No data found for {symbol}")
            return None
            
        # Alpaca returns lowercase columns; rename to Title Case
        bars.rename(columns=lambda x: x.capitalize(), inplace=True)
        return bars
    except Exception as e:
        print(f"❌ Alpaca Data Error: {e}")
        return None

def get_market_news(symbol):
    try:
        news_list = api.get_news(symbol=symbol, limit=3)
        headlines = [n.headline for n in news_list]
        return headlines
    except Exception as e:
        print(f"❌ News Error: {e}")
        return []

# --- 3. MATH & EXECUTION ---
def calculate_technicals(data, rsi_window=14, sma_window=50):
    """Calculates RSI and SMA 50."""
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    loss = loss.replace(0, 0.001)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    # SMA 50
    if len(data) >= sma_window:
        sma = data['Close'].rolling(window=sma_window).mean().iloc[-1]
    else:
        sma = 0 # Not enough data
        
    return rsi, sma

def execute_shadow_trade(symbol, side, reason, score):
    try:
        # Check existing position to avoid stacking
        try:
            pos = api.get_position(symbol)
            if side == 'buy' and float(pos.qty) > 0:
                print(f"✋ Already holding {symbol}. Skipping.")
                return None
        except:
            pass 

        order = api.submit_order(symbol=symbol, qty=1, side=side, type='market', time_in_force='gtc')
        print(f"✅ SHADOW TRADE EXECUTED: {side.upper()} {symbol}")
        
        signals_col.insert_one({
            "action": side,
            "symbol": symbol,
            "reason": reason,
            "ai_score": score,
            "order_id": order.id,
            "timestamp": time.time()
        })
        return order
    except Exception as e:
        print(f"❌ Execution Error: {e}")
        return None

# --- 4. THE LOOP ---
def scan_market():
    print(f"\n🕵️ Scanning {SYMBOL} via Alpaca...")
    
    # 1. Technical Check
    df = get_clean_data(SYMBOL)
    if df is None: return

    price = df['Close'].iloc[-1].item()
    rsi, sma = calculate_technicals(df)
    
    print(f"📉 {SYMBOL} | Price: ${price:.2f} | RSI: {rsi:.1f} | SMA50: ${sma:.2f}")
    
    action = None
    
    # STRATEGY: 
    # 1. Trend: Price > SMA 50 (Uptrend)
    # 2. Discount: RSI < 35 (Oversold)
    if price > sma and rsi < 35:
        action = "buy"
    elif price <= sma:
        print("❌ Trend is Down (Price < SMA50). Skipping.")
        return
    elif rsi >= 35:
        print("💤 Market Neutral (RSI > 35). Skipping.")
        return

    # 3. AI Sentiment Check (Vibe Check)
    print("🚦 Technicals Passed. Asking AI...")
    news_headlines = get_market_news(SYMBOL)
    ai = agent.analyze(SYMBOL, news_headlines)
    print(f"🧠 AI Score: {ai.score} | {ai.reason}")

    # 4. Execution Trigger
    if action == "buy" and ai.score >= 7:
        order = execute_shadow_trade(SYMBOL, "buy", ai.reason, ai.score)
        if order:
            notify_user(price, rsi, ai, order.id)

def notify_user(price, rsi, ai, order_id):
    # Link to Robinhood for manual execution
    rh_link = f"https://robinhood.com/stocks/{SYMBOL}"
    
    try:
        requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", 
            data=f"🤖 Trade Alert: {SYMBOL}\nPrice: ${price}\nRSI: {rsi:.1f}\nScore: {ai.score}/10",
            headers={
                "Title": f"Buy Signal: {SYMBOL}",
                "Priority": "high",
                "Actions": f"view, 📱 Open App, {rh_link}; view, 🔍 Logic, {PUBLIC_URL}/browse?id={order_id}"
            },
            timeout=10
        )
    except Exception as e:
        print(f"⚠️ Notification Failed: {e}")

# --- 5. SERVER ---
@app.route('/browse')
def browse():
    oid = request.args.get('id')
    record = signals_col.find_one({"order_id": oid})
    if not record: return "Trade not found."
    return f"<h1>Reasoning:</h1><p>{record['reason']}</p>"

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(scan_market, 'interval', minutes=60)
    scheduler.start()
    scan_market() 
    app.run(host='0.0.0.0', port=5000)

```

---

## How It Works

### 1. The Trend Filter (SMA 50)

We never trade against the trend. The bot calculates the **50-Day Moving Average**. If the current price is *below* this line, we assume the stock is dying, and we do nothing.

### 2. The Discount Filter (RSI)

If the trend is UP, we look for a dip. We use **RSI (Relative Strength Index)**.

* **RSI < 35:** The stock is "Oversold."
* If RSI > 35, we wait.

### 3. The AI Filter (The Vibe Check)

If the math looks good, the AI reads the last 3 news headlines.

* **Scenario A:** "CEO resigns in scandal" -> AI Score: 2/10 -> **BLOCK TRADE.**
* **Scenario B:** "Market dips on inflation fears" -> AI Score: 8/10 -> **EXECUTE.**

### 4. The Shadow Trade

If all lights are green, the bot executes a **Paper Trade** via Alpaca. This tracks your hypothetical P&L. Simultaneously, it sends a notification to your phone. If you agree with the logic, you open your brokerage app and place the real trade yourself.

---

## Next Steps: Making it Smarter

To take this from a fun project to an enterprise-grade tool, consider implementing **Vector Search**.

Currently, the AI has no memory. By using **MongoDB Atlas Vector Search**, you could embed every news headline you analyze. Before placing a trade, the bot could query its own memory:

> *"Last time I saw headlines about 'CEO Investigation' for this stock, did the price recover?"*

If the answer is "No," the bot can veto the trade based on historical performance, creating a self-improving feedback loop.
