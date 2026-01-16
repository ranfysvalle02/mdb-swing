# 👁️ Sauron's Eye - AI-Augmented Swing Trading Bot

**The Watching Eye - Extracting Capital Through Swing Strategies**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 What Is This?

Sauron's Eye is an **AI-powered swing trading bot** that automates the boring 99% of trading (scanning markets, calculating indicators, reading news) so you can focus on the 1% that matters: **decision making**.

### The Philosophy

> *"The lawn is overgrown because the owner was lazy. The foundation is solid. The house is cheap. This is a good buy."*

The bot identifies stocks that are:
1. **In a good trend** (price above 200-day moving average)
2. **Currently cheap** (RSI < 35, oversold)
3. **News confirms** the "foundation" is solid (AI analyzes sentiment)

Only when all three align does it create a **pending trade** for your approval.

---

## 🧠 How It Works

### The Strategy: Swing Trading with AI Validation

**Swing Trading** = Buy dips in uptrending stocks, hold for days/weeks, sell when profit target or stop loss is hit.

#### Step 1: Market Scanning (Every 15 Minutes)
- Scans your watchlist (default: NVDA, AMD, MSFT, GOOGL, AMZN, TSLA, COIN, AAPL)
- Checks if you already have a position (skips if you do)
- Fetches 500+ days of historical data using Alpaca's IEX feed

#### Step 2: Technical Analysis
Calculates three key indicators:
- **RSI (Relative Strength Index)**: Measures if stock is oversold (< 30) or overbought (> 70)
- **SMA-200 (200-day Simple Moving Average)**: Determines long-term trend
- **ATR (Average True Range)**: Measures volatility for position sizing

**Entry Signal**: RSI < 35 AND price > SMA-200 (uptrending but oversold)

#### Step 3: News Analysis with AI
- Fetches latest 3 news headlines from Alpaca
- **Scrapes article content** using Firecrawl (first ~500 chars)
- Sends to Azure OpenAI (GPT-4o) for sentiment analysis
- AI scores the trade 0-10 based on:
  - Technical indicators
  - News sentiment
  - Risk assessment
  - Catastrophic news detection (fraud, bankruptcy = automatic veto)

#### Step 4: Manual Approval (YOU DECIDE)
- If AI score >= 8, creates a **pending trade**
- Shows in UI with:
  - Symbol, price, RSI, trend, risk level
  - AI reasoning
  - Score (0-10)
- You click **APPROVE** or **REJECT**
- Approved trades execute with bracket orders (limit + stop loss + take profit)

#### Step 5: Position Management
- **Position sizing** based on ATR volatility (risk $50 per trade by default)
- **Stop loss**: Entry price - (2 × ATR)
- **Take profit**: Entry price + (3 × ATR)
- **Risk/Reward**: 1:1.5 ratio

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Alpaca Paper Trading Account (free)
- Azure OpenAI API key (or OpenAI API key)
- Firecrawl API key (optional, for article scraping)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd mdb-swing
   ```

2. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

3. **Start the application**
   ```bash
   docker-compose up --build
   ```

4. **Access the dashboard**
   - Web UI: http://localhost:5000
   - MongoDB: mongodb://admin:secret@localhost:27017/

---

## ⚙️ Configuration

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

# Firecrawl (for article content scraping)
FIRECRAWL_API_KEY=fc-your_key_here

# Trading Configuration
TARGET_SYMBOLS=NVDA,AMD,MSFT,GOOGL,AMZN,TSLA,COIN,AAPL
MAX_CAPITAL_DEPLOYED=5000.00
```

### Strategy Presets

The bot supports different strategy presets (configurable via UI):

1. **Conservative** (Default)
   - RSI threshold: < 30 (very oversold)
   - AI score required: >= 9
   - Risk per trade: $25
   - Max capital: $2,500

2. **Moderate**
   - RSI threshold: < 35
   - AI score required: >= 8
   - Risk per trade: $50
   - Max capital: $5,000

3. **Aggressive**
   - RSI threshold: < 40
   - AI score required: >= 7
   - Risk per trade: $100
   - Max capital: $10,000

---

## 📊 Features

### 🤖 AI-Powered Analysis
- **Azure OpenAI Integration**: Uses GPT-4o for trade analysis
- **News Scraping**: Firecrawl extracts article content (not just headlines)
- **Sentiment Analysis**: AI reads news and validates trade signals
- **Risk Assessment**: AI provides risk level (Low/Medium/High)

### 📈 Technical Indicators
- **RSI**: Identifies oversold/overbought conditions
- **SMA-200**: Confirms long-term trend direction
- **ATR**: Calculates volatility for position sizing

### 🎯 Risk Management
- **Position Sizing**: Based on ATR volatility
- **Stop Loss**: Automatic stop at 2× ATR below entry
- **Take Profit**: Automatic target at 3× ATR above entry
- **Capital Limits**: Configurable max capital per trade

### 🔔 Manual Approval System
- **No Auto-Trading**: All trades require your approval
- **Pending Trades Panel**: See all AI recommendations
- **One-Click Approval**: Approve or reject with a button
- **Trade History**: Complete ledger of all executed trades

### 📱 Real-Time Dashboard
- **Live Account Balance**: Updates every 10 seconds
- **Current Positions**: Shows P&L and unrealized gains/losses
- **Trade History**: Complete log of all trades
- **TradingView Charts**: Integrated charting for analysis
- **Backtesting**: Test strategies on historical data

---

## 🎨 UI/UX Features

### Glassmorphism Design
- Modern glassmorphic UI with backdrop blur effects
- Smooth animations and transitions
- Responsive layout for all screen sizes

### Educational Elements
- **Tooltips**: Hover over icons for explanations
- **Modal Dialogs**: Detailed explanations of features
- **Status Badges**: Visual indicators for risk, trends, scores
- **Icons**: Font Awesome icons throughout for clarity

### Transparency
- **AI Reasoning**: See exactly why the AI recommended a trade
- **Technical Details**: All indicators visible and explained
- **News Context**: Read the actual news articles analyzed
- **Risk Levels**: Clear risk assessment for each trade

---

## 🔧 Architecture

### Tech Stack
- **Backend**: FastAPI (Python 3.11+)
- **Database**: MongoDB (via mdb-engine)
- **AI**: Azure OpenAI (GPT-4o)
- **Trading**: Alpaca API (Paper Trading)
- **Scraping**: Firecrawl API
- **Frontend**: HTMX + Tailwind CSS
- **Charts**: TradingView Widgets

### Project Structure
```
mdb-swing/
├── src/app/
│   ├── api/          # API routes
│   ├── core/         # Configuration & engine
│   ├── models/       # Pydantic models
│   ├── services/     # Business logic
│   │   ├── ai.py     # AI analysis
│   │   ├── analysis.py # Market data & indicators
│   │   ├── indicators.py # Technical indicators
│   │   └── trading.py # Trade execution
│   └── main.py       # Application entry point
├── frontend/         # HTML/CSS/JS
├── config/           # MongoDB manifest
└── docker-compose.yaml
```

---

## 📖 How to Use

### 1. Market Scanner
- Enter a symbol (e.g., AAPL, NVDA)
- Click **SCAN**
- See AI analysis with score, reasoning, and technical indicators
- Chart updates automatically

### 2. Pending Trades
- When the Eye detects a signal, it appears in **Pending Trades**
- Review the AI's reasoning and score
- Click **APPROVE** to execute or **REJECT** to dismiss
- Trades auto-refresh every 3 seconds

### 3. Manual Trading
- Use **Manual Override** for direct execution
- Enter symbol and select BUY/SELL
- Click **EXECUTE**

### 4. Backtesting
- Enter a symbol in **Time Machine**
- Click **TEST**
- See 1-year performance with win rate and return %
- Review trade-by-trade results

### 5. Monitoring
- **Positions**: See all open positions with P&L
- **Ledger**: Complete trade history
- **Balance**: Real-time account balance

---

## 🛡️ Risk Disclaimer

**⚠️ IMPORTANT**: This is a trading bot for **educational purposes**. 

- Uses **paper trading** (simulated money) by default
- Past performance does not guarantee future results
- Always do your own research
- Never trade with money you can't afford to lose
- The AI is a tool, not a guarantee

---

## 🔍 Troubleshooting

### "Insufficient Data" Error
- **Cause**: API only returning 1 day of data
- **Fix**: Using `feed='iex'` parameter (paper trading compatible)
- **Check**: Verify Alpaca API credentials

### "Azure OpenAI Error"
- **Cause**: API version mismatch
- **Fix**: Updated to `2024-08-01-preview` for structured output
- **Check**: Verify Azure OpenAI endpoint and API key

### "Firecrawl Not Working"
- **Cause**: API key not set in Docker environment
- **Fix**: Added `FIRECRAWL_API_KEY` to docker-compose.yaml
- **Check**: Restart container after adding key

### "No Pending Trades"
- **Cause**: No signals detected or AI score too low
- **Check**: 
  - Are symbols in your watchlist?
  - Is market open?
  - Are RSI conditions met?
  - Is AI score >= 8?

---
