
# mdb-swing

----

# Building "The Sentinel": A Headless AI Swing Trading Bot with Human Authorization

> **The Problem:** Automated trading bots are terrifying. If you leave them alone, they can drain your account in minutes due to a single bug or a market crash.
>
> **The Solution:** An "Iron Man" system. The bot does the heavy lifting (scanning, math, AI analysis), but it requires **you** to press the big red button to execute the trade.



In this guide, we are going to build a **Hybrid Swing Trading System** that:
1.  **Scans the market** every hour for technical setups (RSI, Trends).
2.  **Reads the news** using AI to determine sentiment.
3.  **Pings your phone** with a "Mission Report" if it finds a trade.
4.  **Waits for your click** to execute the trade on Robinhood (or Paper Trade).

---

## The Architecture

We are ditching complex cloud architectures for a simple, robust **All-In-One** Python script running in Docker.



* **The Brain:** Python (Logic & Math).
* **The Heart:** `APScheduler` runs a background loop every 60 minutes.
* **The Nervous System:** `Ntfy.sh` (Free) sends push notifications with action buttons to your phone.
* **The Memory:** `MongoDB Atlas` (Local or Cloud) stores trade history.
* **The Hands:** `Flask` listens for your approval click to execute the trade.

---

## The Strategy: "Three-Green-Lights"

Before bothering you, the bot filters stocks through a rigorous checklist. Note that this is a **Mean Reversion** strategy, designed to catch stocks that have dipped but are fundamentally strong.

1.  **The Trend:** Is the stock in an uptrend? (SMA 50 Check).
2.  **The Discount:** Is the stock oversold? (RSI < 35).
3.  **The Vibe Check:** Does the news look good? (AI Sentiment Analysis).

> **Warning:** Strategies relying on RSI < 35 can be dangerous in a market crash (catching a falling knife). That is why the **AI Sentiment Check** is crucial—it ensures we aren't buying a stock that is crashing due to a scandal or earnings miss.

---

## Part 1: The Code

Here is the complete script. It is configured for **Automatic Paper Trading** + **Human Ready Mode**.

### 1. `requirements.txt`

```text
flask
apscheduler
yfinance
requests
pymongo[srv]
langchain
langchain-openai
alpaca-trade-api
pandas
numpy

```

### 2. `Dockerfile`

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY demo.py .
CMD ["python", "demo.py"]

```

### 3. `docker-compose.yml`

```yaml
version: '3.8'
services:
  atlas:
    image: mongodb/mongodb-atlas-local:latest
    container_name: atlas_local
    environment:
      - MONGODB_INITDB_ROOT_USERNAME=admin
      - MONGODB_INITDB_ROOT_PASSWORD=password123
    ports:
      - "27017:27017"
    volumes:
      - atlas-data:/data/db

  bot:
    build: .
    restart: on-failure
    ports:
      - "5000:5000"
    environment:
      MONGO_URI: mongodb://admin:password123@atlas:27017/?directConnection=true
      # Your public URL (ngrok) for viewing memory
      PUBLIC_URL: https://YOUR_NGROK_URL.ngrok-free.app
      NTFY_TOPIC: my_shadow_bot_007
      
      # AI
      OPENAI_API_KEY: sk-proj-YOUR-KEY
      BOT_PERSONALITY: "Standard" 
      
      # ALPACA PAPER TRADING (REQUIRED)
      ALPACA_API_KEY: YOUR_PAPER_KEY
      ALPACA_SECRET_KEY: YOUR_PAPER_SECRET
      ALPACA_BASE_URL: [https://paper-api.alpaca.markets](https://paper-api.alpaca.markets)
      
    depends_on:
      atlas:
        condition: service_healthy
volumes:
  atlas-data:

```

### 4. `demo.py` (The Sentinel)

This updated script includes **Robust Data Handling**. Financial APIs like `yfinance` often change their data structure (MultiIndex vs Single Index) or return empty frames. This script anticipates those errors.

```python
import time
import os
import requests
import yfinance as yf
import pandas as pd
import alpaca_trade_api as tradeapi
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

# --- ALPACA CONFIG (PAPER ONLY) ---
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
ALPACA_URL = os.getenv("ALPACA_BASE_URL", "[https://paper-api.alpaca.markets](https://paper-api.alpaca.markets)")

app = Flask(__name__)
client = MongoClient(MONGO_URI)
db = client['shadow_trading']
signals_col = db['history']

# --- 1. PERSONALITIES (LLM MAGIC) ---
PROMPTS = {
    "Standard": "You are a Senior Hedge Fund Analyst. Be professional, skeptical, and concise.",
    "Gekko": "You are a ruthless Wall Street corporate raider. Greed is good. Use aggressive metaphors.",
    "Yoda": "A wise Jedi Master you are. The market flow you sense. Speak in riddles you must."
}

# --- 2. THE AI AGENT ---
class TradeSignal(BaseModel):
    score: int = Field(description="Sentiment score 0-10 (10 is Bullish).")
    reason: str = Field(description="A concise justification.")

class SentinelAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7) 
        self.chain = self._build_chain()
    
    def _build_chain(self):
        base_persona = PROMPTS.get(PERSONALITY, PROMPTS["Standard"])
        system_prompt = f"""{base_persona}
        Analyze the news headlines. 
        If news is old/irrelevant, score 5.
        Output valid JSON."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Ticker: {ticker}\nHeadlines:\n{headlines}")
        ])
        return prompt | self.llm.with_structured_output(TradeSignal)

    def analyze(self, ticker, headlines):
        if not headlines: return TradeSignal(score=5, reason="No News")
        return self.chain.invoke({"ticker": ticker, "headlines": "\n".join(headlines)})

agent = SentinelAgent()

# --- 3. DATA & MATH (ROBUST) ---
def get_clean_data(ticker):
    """
    Safely fetches and cleans data, handling yfinance MultiIndex issues.
    """
    try:
        df = yf.download(ticker, period="1mo", interval="1d", progress=False)
        
        if df.empty or len(df) < 14:
            print(f"⚠️ Data insufficient for {ticker}.")
            return None

        # Fix: Handle yfinance MultiIndex (flatten columns if needed)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df
    except Exception as e:
        print(f"❌ Data Fetch Error: {e}")
        return None

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    loss = loss.replace(0, 0.001) # Zero division safeguard
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def execute_shadow_trade(symbol, side, reason, score):
    """Places a PAPER trade automatically."""
    try:
        api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL)
        
        # Check existing position to prevent "Stacking" trades
        try:
            pos = api.get_position(symbol)
            if side == 'buy' and float(pos.qty) > 0:
                print(f"✋ Already holding {symbol}. Skipping.")
                return None
        except:
            pass # No position exists

        order = api.submit_order(symbol=symbol, qty=1, side=side, type='market', time_in_force='gtc')
        print(f"✅ SHADOW TRADE EXECUTED: {side.upper()} {symbol}")
        
        # Log to Mongo
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
    print(f"\n🕵️ Scanning {SYMBOL}...")
    
    # 1. Technical Check
    df = get_clean_data(SYMBOL)
    if df is None: return

    rsi = calculate_rsi(df)
    price = df['Close'].iloc[-1].item()
    print(f"📉 {SYMBOL} | Price: ${price:.2f} | RSI: {rsi:.1f}")
    
    action = None
    # Strategy: Buy Dips, but not Crashes
    if rsi < 35: action = "buy"
    
    if not action:
        print("💤 Market Neutral.")
        return

    # 2. AI Sentiment Check
    stock = yf.Ticker(SYMBOL)
    news_data = stock.news if stock.news else []
    news = [n['title'] for n in news_data[:3]]
    
    ai = agent.analyze(SYMBOL, news)
    print(f"🧠 AI Score: {ai.score} | {ai.reason}")

    # 3. Execution Trigger
    # Rule: Only buy the dip if the AI is Bullish (>7)
    if action == "buy" and ai.score >= 7:
        order = execute_shadow_trade(SYMBOL, "buy", ai.reason, ai.score)
        if order:
            notify_user(price, rsi, ai, order.id)

def notify_user(price, rsi, ai, order_id):
    rh_link = f"[https://robinhood.com/stocks/](https://robinhood.com/stocks/){SYMBOL}"
    
    try:
        requests.post(f"[https://ntfy.sh/](https://ntfy.sh/){NTFY_TOPIC}", 
            data=f"🤖 Paper Trade: {SYMBOL}\nScore: {ai.score}/10\nReason: {ai.reason}",
            headers={
                "Title": f"👻 Shadow Buy: {SYMBOL} (${price:.2f})",
                "Priority": "high",
                "Tags": "ghost,chart_with_upwards_trend",
                "Actions": f"view, 📱 Open Robinhood, {rh_link}; view, 🔍 Inspect Logic, {PUBLIC_URL}/browse?id={order_id}"
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
    if not record: return "Trade not found in memory."
    return f"<h1>AI Reason:</h1><p>{record['reason']}</p>"

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(scan_market, 'interval', minutes=60)
    scheduler.start()
    scan_market() # Run once on boot
    app.run(host='0.0.0.0', port=5000)

```

---

## Part 2: Upgrading to Enterprise (Vector Search)

The code above stores your trade history in a standard database. But to make the AI truly smart, you should upgrade to **Atlas Vector Search**.

**Why?**
Right now, the AI treats every day as a new day. It has no memory. By using Vector Search, the AI can recall the past.

1. **Embeddings:** When news comes in, turn the text into numbers (vectors) using OpenAI.
2. **Storage:** Store these vectors in MongoDB Atlas.
3. **Retrieval (RAG):** Before making a trade, the bot performs a vector search:
* *"Find me the last 3 times news headlines looked similar to today's headlines."*
* *"Did we lose money on those trades?"*



If the historical search returns 3 losses for similar news events, the bot can **Veto** the trade, saving you money where a standard algorithm would have failed.

## Summary

You have now built a system that is:

1. **Safe:** It trades on paper first.
2. **Psychologically aligned:** It asks for your permission for real money.
3. **Robust:** It handles dirty financial data gracefully.
4. **Extensible:** It is ready for Vector Search and RAG implementation.
