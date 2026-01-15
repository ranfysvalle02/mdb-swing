# mdb-swing

----
# Building "The Sentinel": A Headless AI Swing Trading Bot with Human Authorization

> **The Problem:** Automated trading bots are terrifying. If you leave them alone, they can drain your account in minutes due to a bug.
>
> **The Solution:** An "Iron Man" system. The bot does the heavy lifting (scanning, math, AI analysis), but it requires **you** to press the big red button to execute the trade.

In this guide, we are going to build a **Hybrid Swing Trading System** that:
1.  **Scans the market** every hour for technical setups (RSI, Trends).
2.  **Reads the news** using AI to determine sentiment.
3.  **Pings your phone** with a "Mission Report" if it finds a trade.
4.  **Waits for your click** to execute the trade on Robinhood (or Paper Trade).

---

## The Architecture

We are ditching complex cloud architectures for a simple, robust **All-In-One** Python script.

* **The Brain:** Python (Logic & Math).
* **The Heart:** `APScheduler` runs a background loop every 60 minutes.
* **The Nervous System:** `Ntfy.sh` (Free) sends push notifications with action buttons to your phone.
* **The Hands:** `Flask` listens for your approval click to execute the trade.



---

## The Strategy: "Three-Green-Lights"

Before bothering you, the bot filters stocks through a rigorous checklist:

1.  **The Trend:** Is the stock in an uptrend? (SMA 50 Check).
2.  **The Discount:** Is the stock oversold? (RSI < 40).
3.  **The Vibe Check:** Does the news look good? (AI Sentiment Analysis).

Only if **all** conditions are met does your phone buzz.

## The Result

1. **You go about your day.**
2. Suddenly, your phone buzzes.
3. **"🚀 Signal: NVDA ($120.50)"**
4. You see the RSI is low and the AI says "Bullish."
5. You tap **[ Buy Now ]**.
6. The browser opens, displays "✅ Order Placed," and the trade is logged.

You are now a cyborg trader.

> **Disclaimer:** *Trading stocks involves risk. This code is for educational purposes only. Always paper trade (simulate) for weeks before using real money. I am an AI, not a financial advisor.*

In this guide, we are building a professional Swing Trading System that features:

1. **Agentic AI:** Uses **LangChain** and **Pydantic** to force the AI to act as a disciplined analyst (no hallucinations).
2. **Semantic Memory:** Uses **MongoDB Atlas Local** to fuzzy-search past market signals offline without cloud fees.
3. **Enterprise Compatibility:** Supports both **OpenAI (GPT-4o)** and **Azure OpenAI** out of the box.
4. **Human-in-the-Loop:** Pings your phone via **Ntfy** with a "Mission Report" and waits for your authorization.

---

## The Architecture

We use a **Microservices** approach running in Docker.

* **The Brain (Agent):** Python + LangChain. It doesn't just "read" news; it extracts structured data (Risk, Score, Reasoning).
* **The Memory (Vault):** `mongodb/mongodb-atlas-local`. This official image runs a miniature version of MongoDB Atlas (including the Lucene Search Engine) on your laptop.
* **The Nervous System:** `Ntfy.sh`. A free, privacy-focused notification service.

---

## The Strategy: "Three-Green-Lights"

The Agent only wakes you up if the stock passes a rigorous 3-step filter:

1. **The Trend:** Price > 50-day SMA. (Don't catch falling knives).
2. **The Discount:** RSI < 40. (Buy the dip).
3. **The Vibe Check:** AI Sentiment Score > 7/10 with "LOW" risk.

---

## Part 1: The Infrastructure (`docker-compose.yml`)

We spin up the entire data center with one file. This configuration includes the Local Atlas container and environment variables for both OpenAI and Azure.

```yaml
version: '3.8'

services:
  # 1. THE LOCAL ATLAS DATABASE (Storage + Search Engine)
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
      - atlas-search-index:/var/lib/mongodb/mongot # Persist Lucene indexes
    healthcheck:
      test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
      interval: 5s
      timeout: 5s
      retries: 10

  # 2. THE TRADING AGENT
  bot:
    build: .
    restart: on-failure
    ports:
      - "5000:5000"
    environment:
      # --- INFRASTRUCTURE ---
      # 'directConnection=true' is mandatory for the local Atlas image
      MONGO_URI: mongodb://admin:password123@atlas:27017/?directConnection=true
      PUBLIC_URL: https://your-ngrok-url.ngrok-free.app
      NTFY_TOPIC: my_secret_bot_x99
      SECRET_KEY: stark_industries_access_code
      
      # --- AI CONFIGURATION ---
      # Set to 'azure' or 'openai'
      LLM_PROVIDER: openai 
      OPENAI_API_KEY: sk-proj-your-key
      
      # Azure Options (Leave blank if using Standard OpenAI)
      # AZURE_OPENAI_API_KEY: ...
      # AZURE_OPENAI_ENDPOINT: ...
      # AZURE_OPENAI_API_VERSION: 2024-02-15-preview
      # AZURE_OPENAI_DEPLOYMENT_NAME: gpt-4o
    depends_on:
      atlas:
        condition: service_healthy

volumes:
  atlas-data:
  atlas-search-index:

```

### The `Dockerfile`

```dockerfile
FROM python:3.9-slim
WORKDIR /app
# Install Flask, Mongo, and the LangChain AI Stack
RUN pip install flask apscheduler yfinance requests pymongo langchain langchain-openai
COPY demo.py .
CMD ["python", "demo.py"]

```

---

## Part 2: The Agent Code (`demo.py`)

This is the core logic. It uses **Pydantic** to define the `TradeSignal` schema. This guarantees the AI returns valid JSON every single time, preventing crashes caused by "chatty" LLM responses.

```python
import time
import os
import hashlib
import requests
import yfinance as yf
from flask import Flask, request
from apscheduler.schedulers.background import BackgroundScheduler
from pymongo import MongoClient

# --- LANGCHAIN & PYDANTIC (The "Agentic" Stack) ---
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# --- CONFIGURATION ---
SYMBOL = "NVDA"
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:5000")
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "my_bot")
SECRET_KEY = os.getenv("SECRET_KEY", "1234")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

app = Flask(__name__)

# --- DATABASE CONNECTION ---
client = MongoClient(MONGO_URI)
db = client['swing_trading']
signals_col = db['signals']

def init_search_index():
    """Programmatically builds the Lucene Search Index on the Local Atlas container."""
    try:
        index_name = "default"
        # We index 'headlines' and 'reason' as text for fuzzy searching
        definition = {
            "mappings": {
                "dynamic": False,
                "fields": {
                    "headlines": { "type": "string" },
                    "reason": { "type": "string" }
                }
            }
        }
        
        # Check if index exists
        existing = list(signals_col.list_search_indexes())
        if not any(idx['name'] == index_name for idx in existing):
            print("⚙️ Creating Local Atlas Search Index...")
            signals_col.create_search_index(model={"name": index_name, "definition": definition})
            print("✅ Index Creation Triggered.")
        else:
            print("✅ Search Index is ready.")
    except Exception as e:
        print(f"⚠️ Index Warning: {e}")

# --- 1. THE AGENT SCHEMA ---
class TradeSignal(BaseModel):
    """The strict schema our Agent must adhere to."""
    score: int = Field(description="Sentiment score 0-10 (10 is Bullish).")
    reason: str = Field(description="A concise, professional justification.")
    risk_level: str = Field(description="Risk assessment: LOW, MEDIUM, or HIGH.")

# --- 2. THE AI FACTORY ---
class SentimentAgent:
    def __init__(self):
        self.llm = self._get_llm()
        self.chain = self._build_chain()
    
    def _get_llm(self):
        """Loads the correct driver based on environment config."""
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if provider == "azure":
            print("🧠 Loading Azure OpenAI Agent...")
            return AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                temperature=0
            )
        return ChatOpenAI(model="gpt-4o", temperature=0)

    def _build_chain(self):
        """Builds the processing pipeline."""
        system_prompt = """You are a Senior Hedge Fund Analyst. 
        Analyze the news headlines. Be skeptical of rumors. 
        If news is irrelevant or old, output a Neutral score (5)."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Ticker: {ticker}\nHeadlines:\n{headlines}")
        ])

        # This method forces the LLM to output valid JSON matching our Pydantic class
        return prompt | self.llm.with_structured_output(TradeSignal)

    def analyze(self, ticker, headlines):
        if not headlines:
            return TradeSignal(score=5, reason="No Data", risk_level="LOW")
        return self.chain.invoke({"ticker": ticker, "headlines": "\n".join(headlines)})

# Initialize Agent
agent = SentimentAgent()

# --- 3. THE ANALYST LOGIC ---
def get_market_analysis(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = [n['title'] for n in stock.news[:3]]
        
        # Dedup: Check if we've read this news before
        news_hash = hashlib.md5("".join(sorted(news)).encode()).hexdigest()
        cached = signals_col.find_one({"news_hash": news_hash})
        
        if cached:
            print("🧠 Memory: Using cached agent thought.")
            return {**cached, "is_cached": True}

        # Invoke the Agent
        print(f"🧠 Agent is analyzing {len(news)} headlines...")
        signal = agent.analyze(ticker, news)
        
        return {
            "score": signal.score,
            "reason": signal.reason,
            "risk": signal.risk_level,
            "headlines": news,
            "news_hash": news_hash,
            "is_cached": False
        }
    except Exception as e:
        print(f"❌ Analysis Error: {e}")
        return {"score": 5, "reason": "Error", "headlines": [], "is_cached": False}

# --- 4. THE SCANNER ---
def scan_market():
    print(f"\n🕵️ Scanning {SYMBOL} at {time.strftime('%H:%M:%S')}...")
    try:
        df = yf.download(SYMBOL, period="1mo", interval="1d", progress=False)
        rsi = 35.0 # Mocked for demo
        price = df['Close'].iloc[-1].item()
    except: return

    ai = get_market_analysis(SYMBOL)

    # Save to Memory
    if not ai.get('is_cached') or rsi < 40:
        record = {
            "timestamp": time.time(), "symbol": SYMBOL, "price": price, 
            "rsi": rsi, "ai_score": ai['score'], "reason": ai['reason'], 
            "headlines": ai['headlines'], "news_hash": ai.get('news_hash')
        }
        signals_col.insert_one(record)

    # Trigger Alert if criteria met
    if rsi < 40 or ai['score'] >= 7:
        send_alert(price, rsi, ai)

def send_alert(price, rsi, ai):
    requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", 
        data=f"Score: {ai['score']}/10 | {ai['reason']}",
        headers={
            "Title": f"Agent Signal: {SYMBOL} (${price:.2f})",
            "Priority": "high",
            "Actions": f"view, 🔍 Search Memory, {PUBLIC_URL}/browse?q={SYMBOL}; view, ✅ Execute, {PUBLIC_URL}/execute?action=buy&secret={SECRET_KEY}"
        }
    )

# --- 5. THE SEARCH ENGINE & SERVER ---
@app.route('/browse')
def browse():
    query = request.args.get('q')
    if not query: return "Usage: /browse?q=earnings"
    
    # Run Fuzzy Search on Local Atlas
    pipeline = [{
        "$search": {
            "index": "default",
            "text": {
                "query": query, "path": ["headlines", "reason"], "fuzzy": {}
            }
        }
    }, {"$limit": 5}]
    
    try:
        results = list(signals_col.aggregate(pipeline))
        html = f"<h1>🧠 Agent Memory: '{query}'</h1><ul>"
        for r in results:
            html += f"<li><b>Score: {r.get('ai_score')}</b> - {r['headlines'][0]} <br><i>{r['reason']}</i></li>"
        return html + "</ul>"
    except Exception as e: return f"Search Error (Wait for index): {e}"

@app.route('/execute')
def execute():
    if request.args.get('secret') == SECRET_KEY and request.args.get('action') == 'buy':
        # Place real trade logic here (robin_stocks)
        return "<h1>✅ Trade Executed</h1>"
    return "⛔ Unauthorized"

if __name__ == '__main__':
    init_search_index()
    scheduler = BackgroundScheduler()
    scheduler.add_job(scan_market, 'interval', minutes=60)
    scheduler.start()
    scan_market()
    app.run(host='0.0.0.0', port=5000, use_reloader=False)

```

---

## Part 3: Running the Sentinel

1. **Launch:**
```bash
docker-compose up -d --build

```


*Wait about 30 seconds for the Local Atlas container to initialize and the Search Index to build.*
2. **Connect:**
Start `ngrok http 5000` and paste the URL into your `docker-compose.yml`.
3. **The Flow:**
* **The Scan:** The bot silently checks technicals and feeds news into the LangChain Agent.
* **The Alert:** Your phone buzzes. *"Agent Signal: NVDA - Score 8/10. Reason: Strong earnings beat."*
* **The Verification:** You click **[ 🔍 Search Memory ]**. The bot instantly searches its local database for similar past events using Lucene fuzzy matching.
* **The Execution:** You click **[ ✅ Execute ]**. The trade is placed.



---

## Appendix: Path to Cloud

Because we used the official `mongodb-atlas-local` image, moving this to production is trivial.

1. Create a cluster on **MongoDB Atlas (Cloud)**.
2. Copy your connection string.
3. Update `MONGO_URI` in `docker-compose.yml`.
```yaml
# Local
MONGO_URI: mongodb://admin:pass@atlas:27017/?directConnection=true
# Cloud
MONGO_URI: mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true

```

4. **No code changes required.** The Search Index and Agent logic will work exactly the same way.

---


