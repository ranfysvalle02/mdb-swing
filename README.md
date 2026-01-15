# Building "The Sentinel": A Hybrid AI Swing Trading Bot

**The Problem:** Automated trading bots are terrifying. If you leave them alone, they can drain your account in minutes due to a single bug or a market crash.

**The Solution:** A "Hybrid" system. The bot does the heavy lifting (scanning, math, AI analysis), but it keeps a human in the loop.

In this guide, we are going to build **The Sentinel**, a Python-based agent that:

1. **Scans the market** every hour for technical setups (Trend + RSI).
2. **Reads the news** using an LLM to validate the move.
3. **Executes a "Shadow Trade"** (Paper Trade) to track performance.
4. **Pings your phone** so you can decide if you want to mirror the trade with real money.

## Executive Summary

**The Sentinel** is a hybrid AI trading agent designed to mitigate the risks of fully autonomous trading while leveraging the speed of algorithmic analysis. It solves the "black box" problem by acting as a high-speed analyst rather than a reckless trader—automating the research, but leaving the final financial decision to a human.

* **Goal:** Automate the 99% of trading that is boring (scanning, math, news reading) to focus on the 1% that matters (decision making).
* **Mechanism:** It scans for technical dips in uptrending stocks, uses an LLM (GPT-4) to verify the move against real-time news sentiment, and executes a "paper trade" to track hypothetical performance.
* **Stack:** Python, Docker, MongoDB (Storage), OpenAI (Sentiment Analysis), Alpaca (Market Data & Paper Execution).
* **Outcome:** A push notification on your phone containing a pre-validated, high-probability trade setup with entry, stop-loss, and take-profit targets calculated for you.

---

## The Strategy: Explained Simply

Think of **The Sentinel** like a professional house flipper looking for a deal in a nice neighborhood. It uses three specific checks before it calls you:

### 1. The "Good Neighborhood" Check (Trend)

* **Technical Term:** 50-Day Moving Average (SMA).
* **The Analogy:** We only want to buy houses in neighborhoods where property values are generally going *up* over time. If the whole neighborhood is crashing (Price < SMA), we don't care how cheap the house is—we aren't interested.

### 2. The "Ugly Paint" Check (Discount)

* **Technical Term:** RSI (Relative Strength Index) < 35.
* **The Analogy:** We found a great neighborhood, but we don't want to pay full price. We look for the one house that looks a little ugly right now—maybe the paint is peeling or the lawn is overgrown (Oversold). It’s a good asset temporarily looking bad.

### 3. The "Foundation" Check (AI Analysis)

* **Technical Term:** LLM Sentiment Analysis.
* **The Analogy:** This is the most critical step. Before we buy that ugly house, we hire an inspector ( The AI) to check *why* it's cheap.
* **Good Cheap:** The lawn is overgrown because the owner was lazy. (Market overreaction to minor news). **Result: BUY.**
* **Bad Cheap:** The foundation is cracked and the house is sinking. (CEO fraud, bankruptcy, major lawsuit). **Result: WALK AWAY.**



Only when a stock passes all three checks—it's in a good trend, it's currently cheap, and the news confirms the "foundation" is solid—does the bot ping your phone.

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
import math
import requests
import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from flask import Flask, request
from apscheduler.schedulers.background import BackgroundScheduler
from pymongo import MongoClient
import logging

# --- LANGCHAIN & PYDANTIC ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Sentinel")

# --- CONFIGURATION ---
# Load symbols as a comma-separated string (e.g. "NVDA,AMD,MSFT,TSLA")
SYMBOLS_ENV = os.getenv("TARGET_SYMBOLS", "NVDA,AMD,MSFT,GOOGL,AMZN")
SYMBOLS = [s.strip() for s in SYMBOLS_ENV.split(',')]

# Risk Management
RISK_PER_TRADE_PCT = 0.01  # Risk 1% of total equity per trade
MAX_POS_SIZE_PCT = 0.20    # Never put more than 20% of account in one trade
TP_PCT = float(os.getenv("TAKE_PROFIT_PCT", 1.08)) # Target: +8%
SL_PCT = float(os.getenv("STOP_LOSS_PCT", 0.96))   # Stop: -4%

# Infrastructure
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "sentinel_dev")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:5000")

# --- ALPACA CONFIG ---
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
ALPACA_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL, api_version='v2')
app = Flask(__name__)

# Safe MongoDB Connection
try:
    client = MongoClient(MONGO_URI)
    db = client['shadow_trading']
    history_col = db['history']
except Exception as e:
    logger.error(f"MongoDB Connection Failed: {e}")

# --- 1. THE AI AGENT ---
class TradeSignal(BaseModel):
    score: int = Field(description="Sentiment score 0-10 (10 is Bullish).")
    reason: str = Field(description="A concise justification under 20 words.")

class SentinelAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.5) 
        self.chain = self._build_chain()
    
    def _build_chain(self):
        system_prompt = (
            "You are a Senior Hedge Fund Analyst. Be skeptical. "
            "Analyze the headlines for immediate market impact. "
            "If news is old/irrelevant, score 5. Output JSON."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Ticker: {ticker}\nHeadlines:\n{headlines}")
        ])
        return prompt | self.llm.with_structured_output(TradeSignal)

    def analyze(self, ticker, headlines):
        if not headlines: return TradeSignal(score=5, reason="No News Found")
        return self.chain.invoke({"ticker": ticker, "headlines": "\n".join(headlines)})

agent = SentinelAgent()

# --- 2. DATA METHODS ---
def get_clean_data(symbol):
    try:
        # Fetch 100 days for SMA calculation
        bars = api.get_bars(symbol, TimeFrame.Day, limit=100).df
        if bars.empty: return None
        # Clean MultiIndex if present
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.reset_index()
        return bars
    except Exception as e:
        logger.error(f"Data Error {symbol}: {e}")
        return None

def get_market_news(symbol):
    try:
        news_list = api.get_news(symbol=symbol, limit=3)
        return [n.headline for n in news_list]
    except Exception as e:
        logger.warning(f"News Error {symbol}: {e}")
        return []

# --- 3. MATH & RISK MANAGEMENT ---
def calculate_technicals(data):
    if len(data) < 50:
        return 50.0, float('inf') # Not enough data

    data.columns = [c.lower() for c in data.columns]
    
    # RSI (14)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    loss = loss.replace(0, 0.001)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    # SMA (50)
    sma = data['close'].rolling(window=50).mean().iloc[-1]
    
    return rsi, sma

def calculate_position_size(price):
    """
    Calculates dynamic share quantity based on account equity and risk.
    """
    try:
        account = api.get_account()
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        
        # 1. Calculate Risk Amount (e.g., 1% of $100k = $1,000 risk)
        risk_amt = equity * RISK_PER_TRADE_PCT
        
        # 2. Calculate Stop Loss distance per share
        # If Price is $100 and SL_PCT is 0.96, Stop is at $96. Distance = $4.
        sl_price = price * SL_PCT
        loss_per_share = price - sl_price
        
        if loss_per_share <= 0: return 0 # Avoid division by zero
        
        # 3. Calculate Ideal Qty based on Risk
        # Shares = $1000 Risk / $4 loss per share = 250 shares
        qty_risk = risk_amt / loss_per_share
        
        # 4. Calculate Max Qty based on Total Allocation Cap
        # e.g., Max 20% of account = $20,000. $20,000 / $100 price = 200 shares.
        max_alloc = equity * MAX_POS_SIZE_PCT
        qty_cap = max_alloc / price
        
        # 5. Take the smaller of the two (Risk vs Cap)
        final_qty = math.floor(min(qty_risk, qty_cap))
        
        # 6. Final check against Buying Power
        if (final_qty * price) > buying_power:
            final_qty = math.floor(buying_power / price)
            
        return max(1, final_qty) # Ensure at least 1 share if affordable
        
    except Exception as e:
        logger.error(f"Risk Calc Error: {e}")
        return 1 # Fallback to 1 share

# --- 4. EXECUTION ---
def execute_bracket_trade(symbol, price, side, reason, score):
    try:
        # Check existing position
        try:
            pos = api.get_position(symbol)
            if float(pos.qty) > 0:
                logger.info(f"✋ Existing position in {symbol}. Skipping.")
                return None, 0, 0, 0
        except:
            pass 

        # Calculate Quantity
        qty = calculate_position_size(price)
        
        take_profit = round(price * TP_PCT, 2)
        stop_loss = round(price * SL_PCT, 2)

        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc',
            order_class='bracket',
            take_profit={'limit_price': take_profit},
            stop_loss={'stop_price': stop_loss}
        )
        
        logger.info(f"✅ EXECUTION: Buy {qty} {symbol} @ ~{price}")
        
        # Save to DB
        history_col.insert_one({
            "action": side,
            "symbol": symbol,
            "qty": qty,
            "entry_price": price,
            "tp": take_profit,
            "sl": stop_loss,
            "reason": reason,
            "ai_score": score,
            "order_id": order.id,
            "timestamp": time.time()
        })
        return order, tp, sl, qty

    except Exception as e:
        logger.error(f"Execution Error {symbol}: {e}")
        return None, 0, 0, 0

# --- 5. THE MAIN LOOP ---
def scan_market():
    # --- 1. MARKET HOURS CHECK ---
    try:
        clock = api.get_clock()
        if not clock.is_open:
            logger.info("🌙 Market is closed. Sleeping...")
            return
    except Exception as e:
        logger.error(f"Clock check failed: {e}")
        return

    logger.info("☀️ Market is OPEN. Starting Scan...")

    # --- 2. MULTI-SYMBOL LOOP ---
    for symbol in SYMBOLS:
        try:
            logger.info(f"🔎 Scanning {symbol}...")
            
            # A. Technical Check
            df = get_clean_data(symbol)
            if df is None: continue

            price = df['close'].iloc[-1].item()
            rsi, sma = calculate_technicals(df)

            is_uptrend = price > sma
            is_oversold = rsi < 35
            
            # Skip if technicals don't align
            if not (is_uptrend and is_oversold):
                continue
                
            logger.info(f"⚡ SETUP FOUND: {symbol} | RSI: {rsi:.1f} | Price > SMA")

            # B. AI Analysis
            headlines = get_market_news(symbol)
            ai = agent.analyze(symbol, headlines)
            
            logger.info(f"🧠 {symbol} AI Score: {ai.score}/10 | {ai.reason}")

            # C. Trigger
            if ai.score >= 7:
                order, tp, sl, qty = execute_bracket_trade(symbol, price, "buy", ai.reason, ai.score)
                if order:
                    notify_user(symbol, price, qty, rsi, ai, tp, sl, order.id)
            else:
                logger.info(f"❌ {symbol}: Good Technicals, Bad Sentiment.")

        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            continue

def notify_user(symbol, price, qty, rsi, ai, tp, sl, order_id):
    rh_link = f"https://robinhood.com/stocks/{symbol}"
    msg = (
        f"🚀 BUY: {symbol} ({qty} shares)\n"
        f"💰 Entry: ${price}\n"
        f"🎯 Target: ${tp}\n"
        f"🛑 Stop: ${sl}\n"
        f"🧠 AI: {ai.score}/10 ({ai.reason})"
    )
    
    try:
        requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", 
            data=msg,
            headers={
                "Title": f"Sentinel: {symbol} Trade",
                "Priority": "high",
                "Tags": "money_with_wings",
                "Actions": f"view, 📱 Broker, {rh_link}; view, 🔍 Logic, {PUBLIC_URL}/browse?id={order_id}"
            },
            timeout=5
        )
    except Exception as e:
        logger.error(f"Notification Failed: {e}")

# --- 6. SERVER ---
@app.route('/browse')
def browse():
    oid = request.args.get('id')
    try:
        record = history_col.find_one({"order_id": oid})
        if not record: return "Trade not found."
        return f"""
        <h2>Trade Logic for {record['symbol']}</h2>
        <p><b>AI Score:</b> {record['ai_score']}/10</p>
        <p><b>Reasoning:</b> {record['reason']}</p>
        <hr>
        <p><b>Quantity:</b> {record['qty']}</p>
        <p><b>Entry:</b> ${record['entry_price']}</p>
        <p><b>Risk/Reward:</b> 1:{round((record['tp']-record['entry_price'])/(record['entry_price']-record['sl']), 1)}</p>
        """
    except Exception as e:
        return f"Error loading trade: {e}"

if __name__ == '__main__':
    # Initial scan
    scan_market()
    
    scheduler = BackgroundScheduler()
    # Runs every hour only during weekdays ideally, but the clock check handles the rest
    scheduler.add_job(scan_market, 'interval', minutes=60)
    scheduler.start()
    
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
