# mdb-swing

---

# Building "The Sentinel": A Headless AI Swing Trading Bot

> **The Problem:** Automated trading bots are terrifying. Left alone, a bug can drain your account in minutes.
> **The Solution:** An **"Iron Man" System**. The bot acts as the suit (scanning, math, AI analysis, heavy lifting), but **you** are Tony Stark. You must press the big red button to execute the real trade.

In this guide, we are building a **Hybrid Swing Trading System** that:

1. **Scans the market** hourly for technical setups (RSI).
2. **Analyzes news** using LangChain + OpenAI to determine sentiment.
3. **"Shadow Trades"** in a sandbox (Alpaca Paper) to prove its worth.
4. **Pings your phone** with a "Mission Report."
5. **Waits for your click** to execute the real trade on Robinhood.

---

## The Philosophy: "Trust is Earned"

Most trading tutorials ask you to hand over API keys and pray. We are building **The Sentinel**.

It runs 24/7 on your machine, watching the market and executing trades with **monopoly money** (Alpaca Paper Trading). It builds a resume while you sleep.

When it spots an opportunity, it sends a push notification:

> *"I just bought NVDA on Paper at $120. RSI is 35 and News is Bullish."*

It provides a **Deep Link to Robinhood**. If—and only if—you agree with the bot's thesis, you tap one button to open your app and place the real trade.

---

## The Architecture

We are utilizing a **Microservices** approach running in Docker.

| Component | Tech Stack | Function |
| --- | --- | --- |
| **The Brain** | **LangChain + OpenAI** | Reads news headlines to form a Bullish/Bearish opinion. |
| **The Quant** | **Pandas + NumPy** | Calculates hard math (RSI, SMA) to find entry points. |
| **The Memory** | **MongoDB Atlas** | Stores every "Shadow Trade" and the *exact* news context that triggered it. |
| **The Hands** | **Alpaca API** | Places the simulation trade to track performance. |
| **The Voice** | **Ntfy.sh** | Sends push notifications with actionable buttons to your phone. |

---

## The Strategy: "Three-Green-Lights"

The bot filters stocks through a rigorous checklist. It only bothers you if **all** conditions are met:

1. **The Discount:** Is the stock oversold? (`RSI < 35`)
2. **The Vibe Check:** Does the AI Agent rate the news as Bullish? (`Score > 7/10`)
3. **The Shadow Test:** Was a paper trade successfully executed?

---

## Part 1: The Setup

### 1. `requirements.txt`

```text
flask
apscheduler
yfinance
requests
pymongo
langchain
langchain-openai
alpaca-trade-api
pandas

```

### 2. `Dockerfile`

```dockerfile
FROM python:3.9-slim
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY demo.py .

# Run the Sentinel
CMD ["python", "demo.py"]

```

### 3. `docker-compose.yml`

We use `mongodb-atlas-local` to mimic a production cloud environment locally.

```yaml
version: '3.8'
services:
  atlas:
    image: mongodb/mongodb-atlas-local:latest
    container_name: atlas_local
    ports:
      - "27017:27017"
    volumes:
      - atlas-data:/data/db

  bot:
    build: .
    restart: on-failure
    ports:
      - "5000:5000"
    depends_on:
      - atlas
    environment:
      # Database
      MONGO_URI: mongodb://atlas:27017/?directConnection=true
      
      # Configuration
      PUBLIC_URL: https://YOUR_NGROK_URL.ngrok-free.app
      NTFY_TOPIC: my_shadow_bot_007
      SYMBOL: "NVDA"
      BOT_PERSONALITY: "Standard" 
      
      # AI Keys
      OPENAI_API_KEY: sk-proj-YOUR-KEY
      
      # Alpaca Paper Trading (Required for Shadow Trading)
      ALPACA_API_KEY: YOUR_PAPER_KEY
      ALPACA_SECRET_KEY: YOUR_PAPER_SECRET
      ALPACA_BASE_URL: https://paper-api.alpaca.markets

volumes:
  atlas-data:

```

---

## Part 2: The Code (`demo.py`)

This is the brain of the operation. It combines **Technical Analysis** (Math) with **Fundamental Analysis** (AI).

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

# --- LANGCHAIN IMPORTS ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# --- CONFIGURATION ---
SYMBOL = os.getenv("SYMBOL", "NVDA")
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:5000")
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "my_bot")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
PERSONALITY = os.getenv("BOT_PERSONALITY", "Standard")

# --- ALPACA CONFIG ---
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
ALPACA_URL = os.getenv("ALPACA_BASE_URL")

app = Flask(__name__)
client = MongoClient(MONGO_URI)
db = client['shadow_trading']
signals_col = db['history']

# --- 1. PERSONALITIES (LLM PROMPTS) ---
PROMPTS = {
    "Standard": "You are a Senior Hedge Fund Analyst. Be professional, skeptical, and concise.",
    "Gekko": "You are a ruthless Wall Street corporate raider. Greed is good. Use aggressive metaphors.",
    "Yoda": "A wise Jedi Master you are. The market flow you sense. Speak in riddles you must."
}

# --- 2. THE AI AGENT ---
class TradeSignal(BaseModel):
    """Structured output to ensure the AI creates machine-readable data."""
    score: int = Field(description="Sentiment score 0-10 (10 is Bullish).")
    reason: str = Field(description="A concise justification for the score.")

class SentinelAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.chain = self._build_chain()
    
    def _build_chain(self):
        base_persona = PROMPTS.get(PERSONALITY, PROMPTS["Standard"])
        system_prompt = f"""{base_persona}
        Analyze the news headlines. If news is old/irrelevant, score 5.
        Output valid JSON."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Ticker: {ticker}\nHeadlines:\n{headlines}")
        ])
        # We use strict structured output to prevent hallucinations
        return prompt | self.llm.with_structured_output(TradeSignal)

    def analyze(self, ticker, headlines):
        if not headlines: return TradeSignal(score=5, reason="No News")
        return self.chain.invoke({"ticker": ticker, "headlines": "\n".join(headlines)})

agent = SentinelAgent()

# --- 3. MATH & EXECUTION ---
def calculate_rsi(data, window=14):
    """Calculates Relative Strength Index (Technical Indicator)"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def execute_shadow_trade(symbol, side, reason, score):
    """Places a PAPER trade automatically and logs it to MongoDB."""
    try:
        api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL)
        
        # Avoid stacking orders if we already hold it
        try:
            pos = api.get_position(symbol)
            if side == 'buy' and float(pos.qty) > 0:
                print(f"✋ Already holding {symbol}. Skipping.")
                return None
        except:
            pass # No position exists

        # Submit Paper Order
        order = api.submit_order(symbol=symbol, qty=1, side=side, type='market', time_in_force='gtc')
        print(f"✅ SHADOW TRADE EXECUTED: {side.upper()} {symbol}")
        
        # Log to MongoDB Atlas
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
    try:
        # 1. Technical Analysis
        df = yf.download(SYMBOL, period="1mo", interval="1d", progress=False)
        rsi = calculate_rsi(df)
        price = df['Close'].iloc[-1].item()
        print(f"📉 {SYMBOL} | Price: ${price:.2f} | RSI: {rsi:.1f}")
        
        # Trigger Condition: RSI < 35 (Oversold)
        if rsi >= 35:
            print("💤 Market Neutral (RSI too high).")
            return

        # 2. AI Analysis
        print("💡 Technical Trigger met. Consulting AI Agent...")
        stock = yf.Ticker(SYMBOL)
        news = [n['title'] for n in stock.news[:3]]
        ai = agent.analyze(SYMBOL, news)
        print(f"🧠 AI Score: {ai.score} | {ai.reason}")

        # 3. Execution Decision
        # Buy if RSI is low AND AI is Bullish (>7)
        if ai.score >= 7:
            order = execute_shadow_trade(SYMBOL, "buy", ai.reason, ai.score)
            if order:
                notify_user(price, rsi, ai, order.id)

    except Exception as e:
        print(f"⚠️ Scan Error: {e}")

def notify_user(price, rsi, ai, order_id):
    """Sends Push Notification with Actions"""
    rh_link = f"https://robinhood.com/stocks/{SYMBOL}"
    
    requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", 
        data=f"🤖 I just paper-traded {SYMBOL}!\nScore: {ai.score}/10\nReason: {ai.reason}",
        headers={
            "Title": f"👻 Shadow Buy: {SYMBOL} (${price:.2f})",
            "Priority": "high",
            "Tags": "ghost,chart_with_upwards_trend",
            "Actions": f"view, 📱 Open Robinhood, {rh_link}; view, 🔍 Inspect Logic, {PUBLIC_URL}/browse?id={order_id}"
        }
    )

# --- 5. SERVER & MEMORY ---
@app.route('/browse')
def browse():
    """Web view to inspect the AI's logic from the push notification."""
    oid = request.args.get('id')
    record = signals_col.find_one({"order_id": oid})
    
    if not record: return "Trade not found in memory."
    
    return f"""
    <html>
        <body style="font-family: sans-serif; padding: 20px; max-width: 600px; margin: auto;">
            <h1>🧠 Logic Inspector</h1>
            <h2 style="color: green;">{record['action'].upper()} {record['symbol']}</h2>
            <p><b>AI Confidence:</b> {record['ai_score']}/10</p>
            <div style="background: #f4f4f4; padding: 15px; border-radius: 8px; border-left: 5px solid #333;">
                <h3>The Thesis:</h3>
                <p>"{record['reason']}"</p>
            </div>
            <p style="margin-top:20px; color: #888;"><i>Stored permanently in MongoDB Atlas.</i></p>
        </body>
    </html>
    """

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(scan_market, 'interval', minutes=60)
    scheduler.start()
    scan_market() # Run once immediately on boot
    app.run(host='0.0.0.0', port=5000)

```

---

## Part 3: Advanced Features

### 🧠 A/B Testing Personalities

The `PERSONALITY` variable allows you to test market psychology.

1. **"Gordon Gekko" Mode:**
* *Result:* The bot becomes aggressive. It might interpret a scandal as a "buying opportunity" due to market overreaction.
* *Log:* *"Blood in the streets. I'm buying the fear."*


2. **"Warren Buffett" Mode:**
* *Result:* The bot ignores hype trains. It demands value.
* *Log:* *"RSI is low, but fundamentals are weak. Pass."*



You can run two containers simultaneously with different personalities to see which AI strategy yields a higher P&L on paper.

### ☁️ Cloud Scaling (MongoDB Atlas)

We used the `mongodb-atlas-local` image, which is API-compatible with the cloud. When you are ready to move from a laptop to a cloud server:

1. **Create a Free Cluster** at [MongoDB.com](https://www.mongodb.com/atlas).
2. **Swap the Connection String** in `docker-compose.yml`.
3. **Use MongoDB Charts:** Drag and drop a dashboard to visualize your "Shadow P&L" vs. "Sentiment Score" without writing frontend code.

> **Disclaimer:** *This code is for educational purposes only.*
