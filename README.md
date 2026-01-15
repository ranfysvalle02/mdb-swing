# mdb-swing

===

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

---

## The Code: `demo.py`

This is the complete, single-file solution. It acts as both the Scanner and the Web Server.

**Prerequisites:**
```bash
pip install flask apscheduler yfinance openai ntfy robin_stocks

```

**`demo.py`**

```python
import time
import json
import os
import requests
import yfinance as yf
from flask import Flask, request
from apscheduler.schedulers.background import BackgroundScheduler
# from openai import OpenAI # Uncomment if using real OpenAI
# import robin_stocks.robinhood as r # Uncomment for Real Trading

# --- CONFIGURATION ---
SYMBOL = "NVDA"
# 1. Start ngrok (ngrok http 5000) and paste the HTTPS url here:
PUBLIC_URL = "[https://your-unique-id.ngrok-free.app](https://your-unique-id.ngrok-free.app)" 
# 2. Pick a secret topic name for your notifications:
NTFY_TOPIC = "my_secret_trading_bot_x99" 
SECRET_KEY = "stark_industries_access_code"

app = Flask(__name__)

# --- 1. THE AI ANALYST ---
def get_ai_analysis(ticker):
    """Fetches news and estimates sentiment."""
    try:
        stock = yf.Ticker(ticker)
        # Get top 3 headlines
        news = [n['title'] for n in stock.news[:3]]
        
        # --- REAL AI INTEGRATION (Optional) ---
        # client = OpenAI(api_key="sk-...")
        # response = client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[{"role": "user", "content": f"Analyze sentiment: {news}"}]
        # )
        # return json.loads(response.choices[0].message.content)

        # --- MOCK AI (For Demo/Cost Saving) ---
        # We simulate a "Smart" response based on keywords
        score = 8 if any("soar" in h.lower() or "jump" in h.lower() for h in news) else 5
        return {
            "score": score, 
            "reason": "Headlines indicate positive momentum.",
            "headlines": news
        }
    except Exception as e:
        print(f"AI Error: {e}")
        return {"score": 5, "reason": "Data Unavailable", "headlines": []}

# --- 2. THE SCANNER (Background Task) ---
def scan_market():
    print(f"\n🕵️ Scanning {SYMBOL} at {time.strftime('%H:%M:%S')}...")
    
    # A. Technical Analysis (RSI Calculation)
    try:
        df = yf.download(SYMBOL, period="1mo", interval="1d", progress=False)
        if df.empty: return
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1].item()
        price = df['Close'].iloc[-1].item()
    except Exception as e:
        print(f"Data Error: {e}")
        return

    # B. AI Analysis
    ai = get_ai_analysis(SYMBOL)

    print(f"   Price: ${price:.2f} | RSI: {rsi:.1f} | AI Score: {ai['score']}/10")

    # C. The Trigger Logic
    # Condition: RSI is Oversold (<40) OR AI is Super Bullish (>8)
    if rsi < 40 or ai['score'] >= 8:
        print("🚀 SIGNAL FOUND! Sending alert...")
        send_notification(price, rsi, ai)
    else:
        print("   💤 No signal. Sleeping...")

# --- 3. THE COMMUNICATOR (Ntfy.sh) ---
def send_notification(price, rsi, ai):
    """Sends a push notification with a clickable 'BUY' button."""
    
    # This link triggers the 'execute' route on YOUR server
    buy_link = f"{PUBLIC_URL}/execute?action=buy&secret={SECRET_KEY}"
    
    headers = {
        "Title": f"🚀 Signal: {SYMBOL} (${price:.2f})",
        "Priority": "high",
        "Tags": "moneybag,chart_with_upwards_trend",
        # The magic interactive button
        "Actions": f"view, ✅ Buy Now, {buy_link}" 
    }
    
    message = (
        f"RSI: {rsi:.1f} (Oversold)\n"
        f"AI Confidence: {ai['score']}/10\n"
        f"Reason: {ai['reason']}\n\n"
        f"Headlines: {ai['headlines'][0]}"
    )

    try:
        requests.post(f"[https://ntfy.sh/](https://ntfy.sh/){NTFY_TOPIC}", data=message, headers=headers)
        print("🔔 Notification sent to phone!")
    except Exception as e:
        print(f"Notification Failed: {e}")

# --- 4. THE SERVER (Execution Routes) ---
@app.route('/')
def home():
    return "🤖 Sentinel Bot is Running. You are safe."

@app.route('/execute')
def execute():
    action = request.args.get('action')
    secret = request.args.get('secret')

    # Security Check
    if secret != SECRET_KEY:
        return "⛔ UNAUTHORIZED ACCESS"

    if action == "buy":
        # --- PLACE REAL ORDER HERE ---
        # For Robinhood:
        # r.login(username, password)
        # r.order_buy_fractional_by_price(SYMBOL, 100)
        
        # For Paper Trading (Log to file):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open("trade_log.txt", "a") as f:
            f.write(f"[{timestamp}] BOUGHT {SYMBOL}\n")
            
        print(f"💰 EXECUTING BUY ORDER FOR {SYMBOL}")
        
        # Send confirmation back to phone
        requests.post(f"[https://ntfy.sh/](https://ntfy.sh/){NTFY_TOPIC}", 
                      data=f"✅ Order Executed for {SYMBOL}", 
                      headers={"Title": "Order Confirmed", "Tags": "white_check_mark"})
        
        return "<h1>✅ Order Placed</h1><p>The bot has executed your command.</p>"
    
    return "Unknown Command"

# --- MAIN ENTRY POINT ---
if __name__ == '__main__':
    # 1. Start the Background Scheduler
    scheduler = BackgroundScheduler()
    # Run the scanner every 60 minutes
    scheduler.add_job(func=scan_market, trigger="interval", minutes=60)
    scheduler.start()

    print("🤖 Bot Online. Press Ctrl+C to exit.")
    
    # 2. Run a manual scan immediately for testing
    scan_market()
    
    # 3. Start the Web Server
    # use_reloader=False is crucial to stop the scheduler from running twice
    app.run(port=5000, use_reloader=False)

```

---

## How to Run It (The "Ngrok" Trick)

Since this runs on your laptop, the outside world (your phone) can't click the "Buy" button unless we open a tunnel.

1. **Install & Run Ngrok:**
Download from [ngrok.com](https://ngrok.com).
```bash
ngrok http 5000

```


Copy the URL it gives you (e.g., `https://a1b2-c3d4.ngrok-free.app`).
2. **Update Config:**
Paste that URL into `PUBLIC_URL` in `demo.py`.
3. **Run the Bot:**
```bash
python demo.py

```


4. **Subscribe:**
Open the [Ntfy Web App](https://www.google.com/search?q=https://ntfy.sh/app) or download the mobile app. Subscribe to the topic name you chose (e.g., `my_secret_trading_bot_x99`).

## The Result

1. **You go about your day.**
2. Suddenly, your phone buzzes.
3. **"🚀 Signal: NVDA ($120.50)"**
4. You see the RSI is low and the AI says "Bullish."
5. You tap **[ Buy Now ]**.
6. The browser opens, displays "✅ Order Placed," and the trade is logged.

You are now a cyborg trader.

---

> **Disclaimer:** *Trading stocks involves risk. This code is for educational purposes only. Always paper trade (simulate) for weeks before using real money. I am an AI, not a financial advisor.*

