# 👁️ Sauron's Eye: Swing Trading System for Balanced Lows

**An AI-powered swing trading system that captures multi-day to multi-week moves in stocks at balanced lows**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

---

## Quick TLDR

### What is MDB-Engine?

**MDB-Engine** is a declarative framework that eliminates ~200 lines of FastAPI + MongoDB boilerplate. It provides:
- **Automatic app creation** - FastAPI app with MongoDB connection, lifecycle management, and observability in 4 lines
- **Declarative index management** - Define indexes in `manifest.json`, automatically created on startup (including vector search)
- **Scoped database access** - Dependency injection via `Depends(get_scoped_db)` with automatic connection pooling
- **Built-in observability** - Structured logging, correlation IDs, health checks, and metrics out of the box

**Why it matters**: Focus on business logic, not infrastructure. This project demonstrates mdb-engine's power by building a production-ready trading bot with minimal boilerplate.

### Current Strategy: Balanced Low (PRIMARY)

**This is the PRIMARY and ONLY active strategy in the system.** All trading decisions
are made using the Balanced Low approach. Other strategies exist as pseudocode
placeholders for future implementation.

**What is Balanced Low Swing Trading?**

This is a swing trading system focused on "balanced lows" - buying quality stocks
when they're temporarily "on sale" (oversold) but still in an uptrend. This is
NOT buying crashes or distressed stocks - it's buying healthy pullbacks in rising stocks.

**Swing Trading Characteristics**:

1. **Time horizon**: Uses daily bars and a 200-day SMA; positions are held until stop loss or take profit (days to weeks).

2. **Entry strategy: "Balanced Low"**
   - RSI < 35 (oversold)
   - Price > 200-day SMA (uptrend)
   - AI score ≥ 7 (validates opportunity)

3. **Exit strategy**:
   - Stop loss: Entry - (2 × ATR)
   - Take profit: Entry + (3 × ATR)
   - Risk/reward: 1:1.5

4. **Position sizing**: Based on ATR volatility, risking $50 per trade.

5. **Style**: Technical analysis (RSI, SMA-200, ATR) + AI sentiment/news analysis, with manual approval.

**Not Day Trading**:
- Uses daily bars (not intraday)
- Holds positions for multiple days
- Focuses on swing moves, not intraday scalping

**Not Long-Term Investing**:
- Active trading with defined exits
- Technical entry/exit rules
- Risk management via stop losses

**This is swing trading**: Capturing multi-day to multi-week moves in stocks that are temporarily oversold but still in an uptrend.

### How Does Swing Trading Work Here?

**The Swing Trading Watch Cycle**:
1. **Scan** - Fetch 500+ days of daily bar data for popular stocks (swing trading uses daily bars, not intraday)
2. **Technical Analysis** - Calculate RSI, SMA-200, ATR to identify balanced low swing entry signals
3. **AI Validation** - GPT-4o analyzes news headlines and technicals, scores 0-10 for swing opportunity
4. **Historical Learning** - Vector search finds similar past swing signals; if they were profitable, confidence increases
5. **Decision** - If adjusted score (base + confidence boost) ≥ 7, swing opportunity highlighted
6. **Execution** - Manual approval required; bracket orders with stop loss (2×ATR) and take profit (3×ATR)
7. **Hold** - Positions held for days to weeks until stop loss or take profit hit (swing trading time horizon)

**Swing Trading Key Features**:
- **Daily bars** - Uses daily price data, not intraday (swing trading characteristic)
- **Multi-day holds** - Positions held until stop loss or take profit (days to weeks)
- **200-day SMA** - Confirms long-term uptrend for swing entries
- **No auto-trading** - You control when trades execute
- **Risk management** - Position sizing based on volatility (ATR), $50 risk per trade
- **Learning system** - Historical pattern matching improves confidence over time
- **HTMX frontend** - Declarative HTML-first UI, minimal JavaScript

### Key Benefits

**MDB-Engine + Balanced Low = Powerful Combination**:
- **Fast development** - Infrastructure handled, focus on trading logic
- **Production-ready** - Observability, indexes, and error handling built-in
- **Maintainable** - Declarative configuration, consistent patterns
- **Scalable** - Vector search for historical learning, efficient caching
- **Focused strategy** - One thing done exceptionally well (balanced lows)

**Result**: A clean, maintainable trading bot built in days, not weeks, with production-grade infrastructure from day one.

### Convention-Based Architecture (Rails-like)

This project follows **Convention over Configuration** principles, similar to Ruby on Rails:

- **Auto-Discovery**: Strategies, services, and routes are automatically discovered from folder structure
- **Pluggable Strategies**: Add new trading strategies by simply creating a file in `strategies/` folder
- **Zero Configuration**: No manual registration needed - folder structure drives behavior
- **Service Registry**: Services auto-discovered from `services/` folder

**Example**: To add a new strategy, create `strategies/momentum_breakout.py` with a class inheriting from `Strategy`. It's automatically discovered and available!

See [Convention-Based Architecture Guide](docs/convention-based-architecture.md) for details.

---

## The Story Begins

Picture this: You're building a trading bot. You need to scan markets, analyze stocks, store trade history, and present it all in a beautiful dashboard. The usual approach? Spend days wiring together FastAPI, MongoDB connections, logging infrastructure, index management, and dependency injection.

But what if there was a better way?

Enter **mdb-engine** — a framework that handles all the infrastructure so you can focus on what matters: building the Eye that watches markets.

This is the story of how we built Sauron's Eye, a **swing trading system** that identifies stocks at **balanced lows** — those perfect moments when a stock is oversold but stable, in an uptrend, with clear upside potential for multi-day to multi-week moves. And how mdb-engine let us build it faster, cleaner, and more reliably than we ever could have imagined.

---

## What Is This, Really?

Sauron's Eye is a **swing trading system** focused on finding stocks at balanced lows — capturing multi-day to multi-week moves when stocks are temporarily oversold but still in an uptrend.

### Swing Trading Characteristics

This is a swing trading system focused on "balanced lows." Here's exactly what that means:

**1. Time Horizon**
- Uses **daily bars** (not intraday data)
- Positions are held until stop loss or take profit (days to weeks)
- Uses **200-day SMA** to confirm long-term trend direction

**2. Entry Strategy: "Balanced Low"**
- **RSI < 35** (oversold)
- **Price > 200-day SMA** (uptrend)
- **AI score ≥ 7** (validates opportunity)

**3. Exit Strategy**
- **Stop loss**: Entry - (2 × ATR)
- **Take profit**: Entry + (3 × ATR)
- **Risk/reward**: 1:1.5

**4. Position Sizing**
- Based on ATR volatility
- Risking **$50 per trade** (configurable)

**5. Style**
- Technical analysis (RSI, SMA-200, ATR) + AI sentiment/news analysis
- **Manual approval** required — no auto-trading

### What This Is NOT

**This is NOT Day Trading**:
- ❌ Uses daily bars (not intraday)
- ❌ Holds positions for multiple days (not hours)
- ❌ Focuses on swing moves, not intraday scalping
- ❌ No minute-by-minute monitoring required

**This is NOT Long-Term Investing**:
- ❌ Active trading with defined exits
- ❌ Technical entry/exit rules
- ❌ Risk management via stop losses
- ❌ No buy-and-hold strategy

**This IS Swing Trading**: Capturing multi-day to multi-week moves in stocks that are temporarily oversold but still in an uptrend.

### The Philosophy

> *"The lawn is overgrown because the owner was lazy. The foundation is solid. The house is cheap. This is a good buy."*

We're swing traders, not day traders or long-term investors. We're looking for that perfect swing entry when:

1. **A stock hits a balanced low** — oversold (RSI < 35) but not in freefall
2. **It's in an uptrend** — price above 200-day moving average (the foundation is solid)
3. **AI validates swing opportunity** — GPT-4o analyzes news and technicals for multi-day potential
4. **You make the final call** — review opportunities and execute swing trades when ready

---

## How the Eye Watches for Swing Trading Opportunities

The Eye of Sauron doesn't blink. When you trigger a scan, it analyzes popular stocks using **daily bars** to find swing trading opportunities — stocks temporarily oversold but still in an uptrend, perfect for multi-day to multi-week moves.

### The Swing Trading Watch Cycle

**Step 1: The Scan**  
Click "Scan Again" to trigger the Eye. It fetches 500+ days of **daily bar data** for popular stocks (swing trading uses daily bars, not intraday). The Eye checks if you already have a position (no duplicates), then checks its cache before moving to analysis.

**Step 2: Technical Analysis (Swing Trading Indicators)**  
Three key indicators tell the story for swing entries:
- **RSI (Relative Strength Index)**: Is it oversold? (< 35 = balanced low, perfect for swing entry)
- **SMA-200**: Is it in an uptrend? (price > SMA = foundation is solid, confirms swing direction)
- **ATR (Average True Range)**: How volatile? (used for position sizing and stop loss/take profit)

**Swing Entry Signal**: RSI < 35 AND price > SMA-200. Simple. Focused. Perfect for swing trading.

**Step 3: The AI Validates (with Historical Context)**  
Here's where it gets interesting. The Eye checks its Radar cache first — if a recent analysis exists (within 1 hour), it uses that. Otherwise, it fetches recent news headlines and sends everything to Azure OpenAI (GPT-4o). 

But the Eye doesn't work in isolation. It searches its historical database for similar swing signals using vector embeddings, learning from past patterns. If similar swing signals were profitable, the Eye boosts its confidence.

The AI scores the swing opportunity 0-10 based on:
- How balanced the low is (not too extreme, not too mild) — perfect for swing entries
- Strength of the uptrend — confirms swing direction
- News sentiment (catastrophic news = automatic veto)
- Upside potential (room for multi-day to multi-week swing move)
- Historical pattern confidence (boost from similar profitable swing signals)

**Step 4: You Decide**  
If the adjusted AI score (base score + confidence boost) meets the threshold (≥ 7 by default), the Eye highlights the swing opportunity. You see:
- Symbol, price, RSI, trend, risk level
- The AI's reasoning (why it thinks this is a good swing entry)
- The base score and confidence boost
- Historical context (how many similar swing signals were profitable)

One click: **APPROVE** or **REJECT**. You're always in control.

**Step 5: Execution (Swing Trading Style)**  
Approved trades execute with bracket orders designed for swing trading:
- **Stop loss**: Entry - (2 × ATR) — protects your capital during the swing hold
- **Take profit**: Entry + (3 × ATR) — captures the swing move upside
- **Risk/Reward**: 1:1.5 ratio — disciplined swing trading
- **Hold Period**: Days to weeks — positions held until stop loss or take profit hit (swing trading time horizon)


---

## Why mdb-engine Made This Possible

Let me tell you why mdb-engine was a game-changer for this project.

### The Problem We Didn't Have to Solve

When building a trading bot, you need:
- FastAPI app with proper lifecycle management
- MongoDB connection pooling and error handling
- Database indexes for performance (trade history, analysis cache)
- Dependency injection for clean route handlers
- Structured logging and observability

Without mdb-engine, you'd write ~200 lines of boilerplate just to get started. With mdb-engine? **Four lines:**

```python
from mdb_engine import MongoDBEngine

engine = MongoDBEngine(mongo_uri=MONGO_URI, db_name=MONGO_DB)
app = engine.create_app(
    slug="sauron_eye",
    manifest=get_manifest_path(),
    title="Sauron's Eye - Market Watcher"
)
```

That's it. FastAPI app created. MongoDB connected. Logging configured. Lifecycle managed.

### The Indexes That Just Work

Here's where mdb-engine really shines. Our `radar_cache` collection needs to query by `symbol` and check `expires_at` for TTL. Without proper indexes, cache lookups get slow fast.

With mdb-engine, we declare indexes in `config/manifest.json`:

```json
{
  "managed_indexes": {
    "radar_cache": [
      {
        "type": "regular",
        "keys": {"symbol": 1, "strategy_id": 1},
        "name": "symbol_strategy_idx"
      },
      {
        "type": "regular",
        "keys": {"expires_at": 1},
        "name": "ttl_idx",
        "expireAfterSeconds": 3600
      }
    ]
  }
}
```

The Eye starts up, mdb-engine creates the indexes automatically. No migration scripts. No manual index management. No "why is this query slow?" debugging sessions at 2 AM.

**Real impact**: When the Eye scans 50 stocks, cache lookups are instant. Without that index? You'd be waiting seconds. With it? Milliseconds.

### The Dependency Injection That Just Makes Sense

Every route handler needs database access. Without mdb-engine, you'd be managing connections manually:

```python
# The old way (error-prone, verbose)
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
try:
    trades = await db.pending_trades.find(...)
finally:
    client.close()  # Easy to forget!
```

With mdb-engine, it's clean and automatic:

```python
from mdb_engine.dependencies import get_scoped_db

async def get_positions(db = Depends(get_scoped_db)):
    positions = await db.history.find({"status": "open"}).to_list(10)
    # Connection automatically managed, properly scoped to request
```

**The value**: Clean code. Testable code. No connection leaks. No "database connection pool exhausted" errors. Just focus on building features.

### The Observability That's Built-In

Every log message in Sauron's Eye uses mdb-engine's structured logging:

```python
from mdb_engine.observability import get_logger

logger = get_logger(__name__)
logger.info("👁️ The Eye watches... scanning markets...")
```

Structured logs. Correlation IDs. Request tracing. Production-ready observability from day one.

**Without mdb-engine**: You'd spend days setting up logging infrastructure.  
**With mdb-engine**: It just works.

### The Bottom Line

mdb-engine saved us:
- **~200 lines** of infrastructure boilerplate
- **Hours** of debugging connection pool issues
- **Days** of setting up logging and observability
- **Weeks** of maintaining migration scripts for indexes

More importantly, it let us focus on what matters: building the Eye that watches markets and finds opportunities.

---

## Quick Start: Get the Eye Watching

### Prerequisites

- Docker & Docker Compose
- Alpaca Paper Trading Account ([free signup](https://alpaca.markets))
- Azure OpenAI API key ([get one here](https://azure.microsoft.com/en-us/products/ai-services/openai-service))

### Installation

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd mdb-swing
```

**2. Set up environment variables**
```bash
cp env.example .env
# Edit .env with your API keys
```

**3. Start the Eye**
```bash
docker-compose up --build
```

**4. Open the dashboard**
- Web UI: http://localhost:5000
- MongoDB: mongodb://admin:secret@localhost:27017/

The Eye starts watching immediately. You control when the Eye scans via the UI — click "Scan Again" to find new opportunities.

---

## Configuration: Tuning the Eye's Focus

### Environment Variables

```bash
# Alpaca Trading API (Paper Trading)
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_MODEL_NAME=gpt-4o

# Trading Configuration
TARGET_SYMBOLS=NVDA,AMD,MSFT,GOOGL,AMZN,TSLA,COIN,AAPL
MAX_CAPITAL_DEPLOYED=5000.00

```

### Strategy Configuration

The Eye currently focuses on **Balanced Low** opportunities with the following default settings:

- **RSI threshold**: < 35 (oversold but balanced) — configurable via UI/API
- **AI score required**: ≥ 7 (good opportunity with upside potential) — configurable via UI/API
- **Risk per trade**: $50 — configurable via UI/API
- **Max capital**: $5,000 — configurable via UI/API

**Configuration Priority**:
1. **Database** (highest) - Stored in MongoDB `strategy_config` collection, updated via UI/API
2. **Environment Variables** - Set via `.env` file or environment
3. **Defaults** - Hardcoded fallbacks in `src/app/core/config.py`

Strategy configuration is loaded dynamically on each scan, so changes take effect immediately without restart. 

**Current Status**: Balanced Low is the PRIMARY and ONLY active strategy. Other strategies exist as pseudocode placeholders for future implementation. The system uses a pluggable strategy architecture — future strategies can be added in `src/app/strategies/` and configured via `CURRENT_STRATEGY` environment variable. See [Appendix C: Development Guide](#appendix-c-development-guide) for details.


---

## Features: What the Eye Sees

### 🤖 AI-Powered Analysis
- **Azure OpenAI Integration**: GPT-4o analyzes every opportunity
- **News Context**: Fetches recent headlines for sentiment analysis
- **Risk Assessment**: AI provides risk level (Low/Medium/High)
- **Veto Power**: Catastrophic news (fraud, bankruptcy) = automatic rejection

### 📈 Technical Indicators (Swing Trading)
- **RSI**: Identifies oversold conditions — perfect for swing entry signals
- **SMA-200**: Confirms long-term trend direction — confirms swing direction (uptrend)
- **ATR**: Calculates volatility for position sizing and swing exit levels
- **Daily Bars**: Uses daily price data (swing trading characteristic, not intraday)

### 🎯 Risk Management (Swing Trading Style)
- **Position Sizing**: Based on ATR volatility (risk $50 per trade)
- **Stop Loss**: Automatic stop at 2× ATR below entry — protects during swing hold
- **Take Profit**: Automatic target at 3× ATR above entry — captures swing move
- **Capital Limits**: Never risk more than configured max
- **Hold Period**: Days to weeks — positions held until stop loss or take profit (swing trading time horizon)

### 🔔 Manual Trading System
- **No Auto-Trading**: You control when swing trades execute
- **Quick Actions**: Buy/sell directly from stock cards
- **Trade History**: Complete ledger of all executed swing trades
- **Swing Focus**: Designed for multi-day to multi-week positions, not day trading

### 🎯 RadarService: Learning from History
- **TTL-Based Caching**: Analysis results cached for 1 hour to reduce API calls
- **Historical Storage**: Every analysis stored with vector embeddings for similarity search
- **Vector Search**: Find similar signals from history using semantic similarity
- **Confidence Boost**: Historical patterns influence AI scores — if similar signals were profitable, confidence increases
- **Pattern Learning**: The Eye learns which patterns lead to profitable trades over time

### ⚙️ Strategy Configuration
- **Dynamic Configuration**: Strategy parameters stored in MongoDB, updatable via UI/API
- **Presets**: Built-in strategy presets (Conservative, Balanced, Aggressive)
- **Real-Time Updates**: Changes take effect immediately without restart
- **Pluggable System**: Easy to add new strategies — just implement the Strategy interface

### 📱 Real-Time Dashboard
- **Live Account Balance**: Updates automatically
- **Current Positions**: Shows P&L and unrealized gains/losses
- **Trade History**: Complete log of all trades
- **TradingView Charts**: Integrated charting for analysis
- **Strategy Configuration**: Update strategy parameters on the fly

---

## Architecture: How It All Fits Together

### The Tech Stack

- **Backend**: FastAPI (Python 3.11+) — modern, async, fast
- **Frontend**: HTMX + Tailwind CSS — declarative HTML-first approach
- **Database**: MongoDB with mdb-engine — declarative indexes, scoped access
- **AI**: Azure OpenAI (GPT-4o) — intelligent analysis
- **Trading**: Alpaca Paper Trading API — risk-free testing

---

## HTMX Integration: Declarative HTML-First Frontend

Sauron's Eye demonstrates modern web development with **HTMX** — a library that extends HTML to build interactive applications without writing JavaScript.

### Why HTMX?

Traditional Single Page Applications (React, Vue) require:
- Complex state management
- Client-side routing
- JSON API endpoints
- Large JavaScript bundles

HTMX takes a different approach: **Why should only `<a>` and `<form>` be able to make HTTP requests?**

### HTMX Patterns Used

#### 1. **Polling with `hx-trigger="every Xs"`**

```html
<!-- Account balance auto-refreshes every 10 seconds -->
<div id="account-info" 
     hx-get="/api/balance" 
     hx-trigger="load, every 10s" 
     hx-swap="innerHTML">
    <span class="animate-pulse">Connecting...</span>
</div>
```

**Benefits**: No JavaScript needed. Server returns HTML fragments, htmx swaps them in.

#### 2. **POST Requests with `hx-post`**

```html
<!-- Buy button - no onclick handler needed! -->
<button hx-post="/api/quick-buy"
        hx-vals='{"symbol": "AAPL"}'
        hx-target="#positions-list"
        hx-swap="innerHTML">
    BUY
</button>
```

**Benefits**: Server returns HTML that updates multiple elements via `hx-swap-oob`.

#### 3. **Out-of-Band Swaps (`hx-swap-oob`)**

When you buy a stock, the server returns HTML that updates both the positions list AND shows a toast notification:

```python
# Backend returns HTML with multiple updates
return HTMLResponse(content=f"""
    <div id="positions-list" hx-swap-oob="innerHTML">
        {positions_html}
    </div>
    <div id="toast-container" hx-swap-oob="beforeend">
        {toast_html}
    </div>
""")
```

**Benefits**: One request updates multiple UI elements. No JavaScript coordination needed.

#### 4. **Lazy Loading with `hx-trigger="revealed"`**

```html
<!-- Discovery modal content loads when opened -->
<div id="discovery-content" 
     hx-get="/api/discover-stocks" 
     hx-trigger="revealed" 
     hx-swap="innerHTML">
</div>
```

**Benefits**: Content loads on-demand, not on page load. Better performance.

#### 5. **Confirmation Dialogs with `hx-confirm`**

```html
<button hx-post="/api/quick-sell"
        hx-confirm="Close position for AAPL?">
    SELL
</button>
```

**Benefits**: Built-in browser confirm dialog. No custom JavaScript needed.

### HTMX Benefits in Sauron's Eye

- **80% less JavaScript**: Removed ~200 lines of fetch/onclick handlers
- **Server-sent HTML**: Backend returns HTML fragments, not JSON
- **Progressive Enhancement**: Works without JavaScript, enhanced with htmx
- **Locality of Behavior**: See what a button does by reading its HTML attributes
- **Simpler Debugging**: Server logs show HTML responses, easy to trace

### Example: Before vs After

**Before (JavaScript)**:
```javascript
async function quickBuy(symbol, event) {
    const btn = event.target.closest('button');
    btn.disabled = true;
    const response = await fetch('/api/quick-buy', {
        method: 'POST',
        body: formData
    });
    const data = await response.json();
    if (data.success) {
        // Update positions
        // Show toast
        // Handle errors
    }
}
```

**After (HTMX)**:
```html
<button hx-post="/api/quick-buy"
        hx-vals='{"symbol": "AAPL"}'
        hx-target="#positions-list">
    BUY
</button>
```

**Result**: Same functionality, 90% less code, easier to maintain.

---

## MDB-Engine Integration: Declarative MongoDB Framework

Sauron's Eye showcases **mdb-engine** — a framework that simplifies FastAPI + MongoDB integration.

### Why MDB-Engine?

Building a FastAPI + MongoDB app typically requires:
- Connection pooling setup
- Index management scripts
- Dependency injection boilerplate
- Lifecycle management
- Observability setup

MDB-Engine handles all of this declaratively.

### MDB-Engine Features Demonstrated

#### 1. **Declarative Index Management**

Indexes are defined in `config/manifest.json`:

```json
{
  "managed_indexes": {
    "pending_trades": [
      {
        "type": "regular",
        "keys": {"status": 1, "timestamp": -1},
        "name": "status_timestamp_idx"
      }
    ],
    "radar_history": [
      {
        "type": "vectorSearch",
        "name": "vector_idx",
        "definition": {
          "fields": [{
            "type": "vector",
            "path": "embedding",
            "numDimensions": 1536,
            "similarity": "cosine"
          }]
        }
      }
    ]
  }
}
```

**Benefits**: 
- Indexes in version control
- Automatic creation/updates on startup
- Vector search for semantic similarity
- TTL indexes for cache expiration

#### 2. **Scoped Database Access**

```python
from mdb_engine.dependencies import get_scoped_db

async def get_positions(db = Depends(get_scoped_db)) -> HTMLResponse:
    """MDB-Engine automatically manages connection lifecycle."""
    positions = await db.positions.find({}).to_list(10)
    # Connection automatically closed after request
    return HTMLResponse(content=positions_html)
```

**Benefits**:
- No manual connection management
- Automatic connection pooling
- Request-scoped connections
- Type-safe database access

#### 3. **Automatic FastAPI App Creation**

```python
from mdb_engine import MongoDBEngine

engine = MongoDBEngine(mongo_uri=MONGO_URI, db_name=MONGO_DB)
app = engine.create_app(
    slug="sauron_eye",
    manifest=get_manifest_path(),
    title="Sauron's Eye"
)
```

**Benefits**:
- ~200 lines of boilerplate eliminated
- CORS configured from manifest
- Lifecycle callbacks supported
- Health checks and metrics built-in

#### 4. **Built-in Observability**

```python
from mdb_engine.observability import get_logger

logger = get_logger(__name__)
logger.info("👁️ The Eye watches...")
```

**Benefits**:
- Structured logging
- Correlation IDs
- Request tracing
- Production-ready from day one

#### 5. **Vector Search Integration**

The RadarService uses vector embeddings to find similar trading signals:

```python
# MDB-Engine manages the vector search index automatically
similar_signals = await radar_service.find_similar_signals(
    symbol, 
    analysis_data, 
    limit=5
)
```

**Benefits**:
- Semantic similarity search
- Historical pattern matching
- Confidence scoring based on past performance

### MDB-Engine Benefits in Sauron's Eye

- **~200 lines saved**: No connection pooling boilerplate
- **Zero index migrations**: Declarative index management
- **Type safety**: Scoped database access with proper typing
- **Production-ready**: Built-in observability and health checks
- **Vector search**: Semantic similarity for learning from history

### Example: Database Access

**Without MDB-Engine**:
```python
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
try:
    positions = await db.history.find({"status": "open"}).to_list(10)
finally:
    client.close()  # Easy to forget!
```

**With MDB-Engine**:
```python
async def get_positions(db = Depends(get_scoped_db)):
    positions = await db.history.find({"status": "open"}).to_list(10)
    # Connection automatically managed
```

---

## HTMX + MDB-Engine: A Perfect Match

These technologies complement each other beautifully:

- **HTMX**: Declarative HTML-first frontend
- **MDB-Engine**: Declarative MongoDB-first backend
- **Together**: Minimal JavaScript, maximum productivity

### Request Flow

```
Browser → HTMX (hx-post) → FastAPI → MDB-Engine (get_scoped_db) → MongoDB
         ← HTML Fragment ← HTMLResponse ← Scoped DB ← Query Result
```

1. User clicks button with `hx-post="/api/quick-buy"`
2. HTMX sends POST request to FastAPI
3. FastAPI route uses `Depends(get_scoped_db)` for database access
4. MDB-Engine provides scoped database connection
5. Route queries MongoDB (using indexes from manifest.json)
6. Route returns HTMLResponse with HTML fragment
7. HTMX swaps HTML into target element
8. If response includes `hx-swap-oob`, multiple elements update

### Code Example: Complete Flow

**Frontend (HTMX)**:
```html
<button hx-post="/api/quick-buy"
        hx-vals='{"symbol": "AAPL"}'
        hx-target="#positions-list">
    BUY
</button>
```

**Backend (MDB-Engine + HTMX)**:
```python
async def quick_buy(symbol: str = Form(...), db = Depends(get_scoped_db)) -> HTMLResponse:
    """MDB-Engine: Scoped database access"""
    order = await place_order(symbol, 'buy', db)
    
    # Get updated positions (uses index from manifest.json)
    positions = await db.positions.find({}).to_list(10)
    
    # Return HTML with hx-swap-oob for multiple updates
    return HTMLResponse(content=f"""
        <div id="positions-list" hx-swap-oob="innerHTML">
            {render_positions(positions)}
        </div>
        <div id="toast-container" hx-swap-oob="beforeend">
            {render_toast("Order placed!")}
        </div>
    """)
```

**Result**: Clean, maintainable, type-safe code with minimal boilerplate.

---

## Architecture: How It All Fits Together

### The Tech Stack

- **Backend**: FastAPI (Python 3.11+) — modern, async, fast
- **Database**: MongoDB (via mdb-engine) — automatic connection management, declarative indexes
- **AI**: Azure OpenAI (GPT-4o) — strategy-agnostic analysis engine
- **Trading**: Alpaca API (Paper Trading) — free paper trading account
- **Frontend**: HTMX + Tailwind CSS — modern, responsive UI
- **Charts**: TradingView Widgets — professional charting

### The Architecture Story

Sauron's Eye uses a **pluggable strategy architecture** with intelligent caching and learning:

- **The Eye** (`services/eye.py`): Core watchful scanner — watches markets continuously
- **RadarService** (`services/radar.py`): Caching and historical learning — reduces API calls, learns from patterns
- **Strategies** (`strategies/`): Pluggable lenses — define what the Eye seeks
- **EyeAI** (`services/ai.py`): Reusable AI engine — works with any strategy
- **Mechanics** (`services/analysis.py`, `services/trading.py`): Reusable components

The Eye watches. Strategies focus its energy. RadarService remembers and learns. All mechanics are reusable.

**RadarService Powers**:
- **Caching**: 1-hour TTL cache prevents redundant API calls
- **Vector Search**: MongoDB vector search indexes (`radar_history` collection) enable semantic similarity matching
- **Historical Learning**: Every analysis stored with embeddings, outcomes tracked for pattern recognition
- **Confidence Scoring**: Similar historical signals influence current opportunity scores

**Current Focus**: Balanced Low Strategy — finds stocks at balanced lows with upside potential.

**Future Possibilities**: Momentum Breakout, Mean Reversion, Trend Following — all reuse the same Eye engine and RadarService learning.

### Project Structure

```
mdb-swing/
├── src/app/
│   ├── api/              # API routes (HTMX + JSON endpoints)
│   │   └── routes.py     # All route handlers (uses mdb-engine DI)
│   ├── core/             # Configuration & mdb-engine setup
│   │   ├── config.py    # App configuration (env vars, strategy configs)
│   │   └── engine.py    # MDB-engine initialization
│   ├── models/           # Pydantic models
│   │   └── __init__.py  # TradeVerdict model
│   ├── services/         # Business logic (reusable components)
│   │   ├── eye.py       # The Eye — core scanner engine
│   │   ├── radar.py     # RadarService — caching & historical learning
│   │   ├── ai.py        # EyeAI — reusable AI engine (Azure OpenAI)
│   │   ├── analysis.py  # Market data fetching & technical indicators
│   │   ├── indicators.py # Technical indicator calculations
│   │   └── trading.py   # Trade execution & position management
│   ├── strategies/       # Pluggable strategy lenses
│   │   ├── base.py      # Strategy interface (abstract base class)
│   │   └── balanced_low.py  # Balanced Low strategy implementation
│   └── main.py          # Application entry point (FastAPI app creation)
├── frontend/            # HTMX + Tailwind CSS frontend
│   └── index.html       # Single-page application
├── config/              # MongoDB manifest (mdb-engine configuration)
│   └── manifest.json    # Indexes, CORS, WebSocket config
├── docs/                # Additional documentation
│   └── htmx-patterns.md # HTMX pattern examples
├── docker-compose.yaml  # Docker setup
├── Dockerfile           # Application container
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

**Key MongoDB Collections** (managed by mdb-engine):
- `history`: Trade execution records (indexed by timestamp, symbol)
- `radar_cache`: Analysis cache with 1-hour TTL (indexed by symbol+strategy, expires_at)
- `radar_history`: Historical analyses with vector embeddings (vector search index for similarity)
- `strategy_config`: Active strategy configuration (indexed by active flag)

See [Appendix C: Development Guide](#appendix-c-development-guide) for detailed schema reference.

---

## Using the Eye: A Quick Tour

### 1. Market Scanner
Enter a symbol (e.g., AAPL, NVDA), click **SCAN**. The Eye analyzes it using the current strategy, shows you the AI's reasoning, technical indicators, and a TradingView chart.

### 2. Positions
View your active positions with real-time P&L. Click **CHARTS** to see entry, stop loss, and take profit levels visualized.

### 3. Monitoring
- **Positions**: See all open positions with P&L
- **Ledger**: Complete trade history
- **Balance**: Real-time account balance

---

## The mdb-engine Advantage: A Deeper Dive

If you're building a FastAPI + MongoDB application, mdb-engine is worth considering. Here's why:

### 1. Less Code, More Features
~200 lines of boilerplate eliminated. That's 200 lines you don't have to write, test, or maintain.

### 2. Declarative Indexes
Indexes in `manifest.json`, automatically created on startup. No migration scripts. No "why is this slow?" debugging.

### 3. Dependency Injection Done Right
`Depends(get_scoped_db)` — clean, testable, automatic connection management.

### 4. Production-Ready Observability
Structured logging, correlation IDs, request tracing — built-in from day one.

### 5. Consistent Patterns
Same patterns across all projects. New team members? They already know how it works.

**Want to learn more?** Check out `why-mdb-engine.md` for a detailed analysis of the value and considerations.

---

## Risk Disclaimer

**⚠️ IMPORTANT**: This is a trading bot for **educational purposes**.

- Uses **paper trading** (simulated money) by default
- Past performance does not guarantee future results
- Always do your own research
- Never trade with money you can't afford to lose
- The AI is a tool, not a guarantee

---

## Troubleshooting: When the Eye Blinks

### "Insufficient Data" Error
- **Cause**: API only returning limited data
- **Fix**: Using `feed='iex'` parameter (paper trading compatible)
- **Check**: Verify Alpaca API credentials

### "Azure OpenAI Error"
- **Cause**: API version mismatch or missing credentials
- **Fix**: Updated to `2024-08-01-preview` for structured output
- **Check**: Verify Azure OpenAI endpoint and API key

### "No Opportunities Found"
- **Cause**: No signals detected or AI score too low
- **Check**: 
  - Did you run a scan?
  - Is market open?
  - Are RSI conditions met?
  - Is AI score ≥ 7?

---

## The End (Or Is It?)

Sauron's Eye watches markets continuously, finding balanced low swing trading opportunities and presenting them for your approval. Built with mdb-engine, it's clean, fast, and maintainable.

**Remember**: This is swing trading — capturing multi-day to multi-week moves in stocks at balanced lows. Not day trading. Not long-term investing. Swing trading.

The Eye never sleeps. The Eye never blinks. The Eye watches for swing opportunities.

**Ready to let the Eye watch for swing trading opportunities?** Clone the repo, set your API keys, and start scanning for balanced lows.

---

## Appendix

### Appendix A: MDB-Engine Deep Dive

#### Architecture Overview

MDB-Engine provides a declarative layer over FastAPI + MongoDB, handling infrastructure concerns so you can focus on business logic.

**Core Components**:
- **MongoDBEngine**: Singleton engine instance that manages connections and app lifecycle
- **Manifest System**: JSON configuration for indexes, CORS, WebSockets, and app metadata
- **Dependency Injection**: `get_scoped_db()` provides request-scoped database access
- **Observability**: Built-in structured logging, health checks, and metrics

#### Key Patterns Used in This Project

**1. App Creation Pattern**:
```python
from mdb_engine import MongoDBEngine

engine = MongoDBEngine(mongo_uri=MONGO_URI, db_name=MONGO_DB)
app = engine.create_app(
    slug="sauron_eye",
    manifest=get_manifest_path(),
    title="Sauron's Eye - Market Watcher"
)
```

**2. Scoped Database Access**:
```python
from mdb_engine.dependencies import get_scoped_db

async def my_route(db = Depends(get_scoped_db)):
    positions = await db.positions.find({}).to_list(10)
    # Connection automatically managed, closed after request
```

**3. Declarative Indexes**:
Indexes defined in `config/manifest.json`:
- Regular indexes for query optimization
- TTL indexes for cache expiration
- Vector search indexes for semantic similarity

**4. Embedding Service Integration**:
```python
from mdb_engine.embeddings.dependencies import get_embedding_service_for_app

embedding_service = get_embedding_service_for_app(APP_SLUG, engine)
```

#### Index Management Details

**Collections and Indexes**:
- `history`: Timestamp and symbol indexes for trade history queries
- `radar_cache`: Symbol+strategy composite index, TTL index for 1-hour expiration
- `radar_history`: Symbol+timestamp index, vector search index for similarity matching
- `strategy_config`: Active flag index (unique, partial filter)

**Vector Search Index**:
- **Collection**: `radar_history`
- **Dimensions**: 1536 (text-embedding-3-small)
- **Similarity**: Cosine
- **Use Case**: Find similar historical trading signals for confidence scoring

#### Vector Search Implementation

The RadarService uses vector embeddings to find similar trading signals:

1. **Embedding Generation**: Creates embeddings from technical indicators + headlines + AI reasoning
2. **Storage**: Stores embeddings with analysis data in `radar_history`
3. **Query**: Uses MongoDB `$vectorSearch` aggregation to find similar signals
4. **Confidence Scoring**: Calculates boost based on historical win rate

**Example Query**:
```python
pipeline = [
    {
        "$vectorSearch": {
            "index": "sauron_eye_radar_history_vector_idx",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 50,
            "limit": 5,
            "filter": {
                "metadata.symbol": symbol,
                "timestamp": {"$gte": date_threshold}
            }
        }
    }
]
```

### Appendix B: Balanced Low Swing Trading Strategy Details

#### Swing Trading Characteristics

**Time Horizon**: Days to weeks — positions held until stop loss or take profit hit. Uses daily bars and 200-day SMA to identify swing opportunities.

**Entry Strategy**: "Balanced Low" — perfect for swing trading:
- RSI < 35 (oversold — ideal swing entry point)
- Price > 200-day SMA (uptrend — confirms swing direction)
- AI score ≥ 7 (validates swing opportunity)

**Exit Strategy**: Swing trading exits:
- Stop loss: Entry - (2 × ATR) — protects capital during swing hold
- Take profit: Entry + (3 × ATR) — captures swing move upside
- Risk/reward: 1:1.5 — disciplined swing trading

**Position Sizing**: Based on ATR volatility, risking $50 per trade.

**Style**: Technical analysis (RSI, SMA-200, ATR) + AI sentiment/news analysis, with manual approval. Designed for swing trading, not day trading or long-term investing.

#### Technical Indicator Explanations (Swing Trading)

**RSI (Relative Strength Index)**:
- **Range**: 0-100
- **Calculation**: Wilder's smoothing of price gains vs losses over 14 periods
- **Interpretation**: 
  - RSI < 30: Oversold (potential swing buy)
  - RSI > 70: Overbought (potential swing sell)
  - Balanced Low uses RSI < 35 (oversold but not extreme — perfect for swing entries)

**SMA-200 (200-Day Simple Moving Average)**:
- **Calculation**: Average closing price over last 200 days (uses daily bars)
- **Interpretation**: 
  - Price > SMA-200: Uptrend (bullish — confirms swing direction)
  - Price < SMA-200: Downtrend (bearish — avoid swing entries)
  - Balanced Low requires uptrend (price > SMA-200) for swing trading

**ATR (Average True Range)**:
- **Calculation**: Wilder's smoothing of True Range (max of high-low, high-prev_close, low-prev_close)
- **Interpretation**: Measures volatility (calculated from daily bars)
- **Use**: Position sizing and swing trading stop loss/take profit calculation

#### Entry Criteria (Swing Trading)

**Technical Conditions** (for swing entries):
1. RSI < 35 (oversold but balanced — perfect swing entry point)
2. Price > SMA-200 (uptrend confirmed — confirms swing direction)
3. Sufficient data (≥14 days of daily bars for indicators)

**AI Validation** (for swing opportunities):
1. GPT-4o analyzes technicals + news headlines
2. Scores swing opportunity 0-10 based on:
   - How balanced the low is (ideal for swing entries)
   - Strength of uptrend (confirms swing direction)
   - News sentiment (catastrophic news = veto)
   - Upside potential (room for multi-day to multi-week swing move)
3. Historical confidence boost added (based on similar swing signals)
4. **Final Score** = Base Score + Confidence Boost
5. **Required**: Final Score ≥ 7 (configurable)

#### Exit Criteria (Swing Trading)

**Stop Loss** (swing trading protection):
- **Distance**: Entry - (2 × ATR)
- **Purpose**: Limit downside risk during swing hold (days to weeks)
- **Risk/Reward**: 1:1.5 ratio

**Take Profit** (swing trading target):
- **Distance**: Entry + (3 × ATR)
- **Purpose**: Capture swing move upside (multi-day to multi-week move)
- **Risk/Reward**: 1:1.5 ratio

**Manual Exit**:
- User can close swing positions at any time via UI
- No forced holding period (but designed for days to weeks)

**Hold Period**: Positions held for days to weeks until stop loss or take profit hit — this is swing trading, not day trading.

#### Risk Management Rules (Swing Trading)

**Position Sizing** (for swing trades):
- **Risk per Trade**: $50 (configurable)
- **Calculation**: `shares = risk_dollars / (2 × ATR)` — based on daily bar volatility
- **Budget Limit**: Min(buying_power, max_capital)
- **Max Capital**: $5,000 default (configurable)

**Capital Limits**:
- Never risk more than configured max capital
- Position size capped by available buying power
- Multiple swing positions allowed (up to max capital)

**Risk/Reward Ratio** (swing trading):
- **Stop Loss**: 2× ATR below entry — protects during swing hold (days to weeks)
- **Take Profit**: 3× ATR above entry — captures swing move upside
- **Ratio**: 1:1.5 (risk $1 to make $1.50) — disciplined swing trading

**Hold Period**: Positions held for days to weeks until stop loss or take profit hit — this is swing trading, not day trading.

#### Historical Performance Considerations (Swing Trading)

**Learning System**:
- Every swing analysis stored with vector embeddings
- Similar swing signals found via semantic similarity
- Win rate calculated from historical swing trade outcomes
- Confidence boost: (win_rate - 0.5) × 2.0, clamped [-1.0, +1.0]

**Pattern Recognition** (for swing trading):
- Vector search finds swing signals with similar:
  - Technical indicators (RSI, trend, price, ATR) — from daily bars
  - News headlines
  - AI reasoning
- If similar swing signals were profitable, confidence increases
- If similar swing signals lost money, confidence decreases

**Limitations**:
- Historical data limited to 90 days (configurable)
- Requires sufficient historical swing trade data for meaningful patterns
- Market conditions change; past swing performance ≠ future results
- Swing trading results depend on market conditions — not suitable for all market environments

#### What This Is NOT

**This is NOT Day Trading**:
- ❌ Does not use intraday data (uses daily bars)
- ❌ Does not scalp quick moves
- ❌ Does not exit same day
- ❌ Does not require minute-by-minute monitoring

**This is NOT Long-Term Investing**:
- ❌ Does not buy and hold for years
- ❌ Has defined exits (stop loss and take profit)
- ❌ Uses active risk management
- ❌ Focuses on swing moves, not long-term trends

### Appendix C: Development Guide

#### Adding New Strategies

**1. Create Strategy Class**:
```python
# src/app/strategies/my_strategy.py
from .base import Strategy

class MyStrategy(Strategy):
    def check_technical_signal(self, techs: Dict[str, Any]) -> bool:
        # Your technical conditions
        return techs['rsi'] < 30 and techs['trend'] == 'UP'
    
    def get_ai_prompt(self) -> str:
        return "Your strategy-specific AI prompt..."
    
    def get_name(self) -> str:
        return "My Strategy"
```

**2. Register in Config**:
```python
# src/app/core/config.py
STRATEGY_CONFIGS["my_strategy"] = {
    "name": "My Strategy",
    "rsi_threshold": 30,
    "ai_score_required": 7,
    "risk_per_trade": 50.0,
    "max_capital": 5000.0
}
```

**3. Update Environment**:
```bash
CURRENT_STRATEGY=my_strategy
```

**4. Use in Eye**:
The Eye class automatically uses the configured strategy. No code changes needed!

#### Extending the Eye Scanner

**Adding New Indicators**:
1. Add calculation in `src/app/services/indicators.py`
2. Include in `analyze_technicals()` return dict
3. Use in strategy's `check_technical_signal()`

**Adding New Data Sources**:
1. Extend `get_market_data()` in `src/app/services/analysis.py`
2. Return additional data in tuple
3. Use in Eye's `scan_symbol()` method

**Custom AI Prompts**:
- Override `get_ai_prompt()` in your strategy class
- Include historical context if needed
- Eye automatically uses strategy-specific prompts

#### Database Schema Reference

**Collections**:

**`history`**:
- Trade execution records
- Fields: timestamp, symbol, action, qty, price, reason, score, entry_price, stop_loss, take_profit
- Indexes: timestamp (desc), symbol

**`radar_cache`**:
- Cached analysis results (1-hour TTL)
- Fields: symbol, strategy_id, analysis_data, analyzed_at, expires_at
- Indexes: symbol+strategy_id (composite), expires_at (TTL)

**`radar_history`**:
- Historical analyses with embeddings
- Fields: symbol, timestamp, metadata, analysis (techs, headlines, verdict), outcome, embedding
- Indexes: symbol+timestamp, timestamp, vector search

**`strategy_config`**:
- Active strategy configuration
- Fields: active, rsi_threshold, ai_score_required, risk_per_trade, max_capital, name, description, color, preset
- Indexes: active (unique, partial filter)

#### API Endpoint Reference

**Market Analysis**:
- `POST /api/analyze` - Analyze a single symbol
- Backtest functionality removed - keeping UI focused on actionable insights
- `GET /api/discover-stocks` - Discover trending stocks

**Position Management**:
- `GET /api/positions` - Get all open positions
- `POST /api/quick-buy` - Buy a stock (quick action)
- `POST /api/quick-sell` - Sell a position (quick action)
- `POST /api/trade` - Execute a trade (full order)
- `POST /api/panic` - Close all positions

**Strategy Configuration**:
- `GET /api/strategy-config` - Get current strategy config (HTML)
- `GET /api/strategy` - Get strategy config (JSON)
- `POST /api/strategy` - Update strategy config
- `GET /api/strategy/presets` - Get strategy presets

**Trade History**:
- `GET /api/logs` - Get trade history/logs
- `GET /api/balance` - Get account balance

**WebSocket**:
- `WS /ws` - Real-time stock analysis streaming

**Health & Metrics**:
- `GET /health` - Health check (mdb-engine)
- `GET /metrics` - Metrics (mdb-engine)

---

*Built with ❤️ and mdb-engine*
