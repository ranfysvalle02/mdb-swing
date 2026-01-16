"""Trading execution services."""
import math
import io
import base64
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from alpaca_trade_api.rest import TimeFrame
from .indicators import rsi, sma, atr
from mdb_engine.observability import get_logger
from ..core.config import ALPACA_KEY, ALPACA_SECRET, ALPACA_URL, MAX_CAPITAL_DEPLOYED, STRATEGY_CONFIG
from .analysis import api, get_market_data, analyze_technicals

logger = get_logger(__name__)

def calculate_position_size(price: float, atr: float, risk_per_trade_dollars: float = None) -> int:
    """Calculate position size based on volatility (ATR)."""
    if risk_per_trade_dollars is None:
        risk_per_trade_dollars = STRATEGY_CONFIG['risk_per_trade']
    """Calculate position size based on volatility (ATR)."""
    stop_distance = 2 * atr
    if stop_distance == 0:
        return 0
    
    shares_risk = math.floor(risk_per_trade_dollars / stop_distance)
    
    buying_power = 10000.0  # Default fallback
    if api:
        try:
            acct = api.get_account()
            buying_power = float(acct.buying_power)
        except Exception as e:
            logger.warning(f"Could not get account info: {e}, using default buying power")
    
    # Use strategy preset max capital
    max_capital = min(STRATEGY_CONFIG['max_capital'], MAX_CAPITAL_DEPLOYED)
    max_shares_budget = math.floor(min(buying_power, max_capital) / price)
    
    return int(min(shares_risk, max_shares_budget))

async def place_order(symbol: str, side: str, reason: str, score: int, db) -> Optional[Any]:
    """Place a trading order."""
    if api is None:
        logger.error("Cannot place order: Alpaca API not initialized")
        return None
    
    df, _ = get_market_data(symbol)
    if df is None:
        logger.error(f"Cannot place order for {symbol}: No market data")
        return None
    
    try:
        techs = analyze_technicals(df)
    except Exception as e:
        logger.error(f"Technical analysis failed for {symbol}: {e}")
        return None
    
    qty = calculate_position_size(techs['price'], techs['atr'])
    if qty < 1:
        logger.info(f"Skipping {symbol}: Position size 0 (Risk/Budget)")
        return None

    limit_price = techs['price']
    stop_price = round(techs['price'] - (2 * techs['atr']), 2)
    take_profit = round(techs['price'] + (3 * techs['atr']), 2)
    
    try:
        order = api.submit_order(
            symbol=symbol, qty=qty, side=side, type='limit', limit_price=limit_price,
            time_in_force='day', order_class='bracket',
            take_profit={'limit_price': take_profit},
            stop_loss={'stop_price': stop_price}
        )
        
        # Log to DB using mdb-engine scoped database
        await db.history.insert_one({
            "timestamp": datetime.now(),
            "symbol": symbol,
            "action": side,
            "qty": qty,
            "price": limit_price,
            "reason": reason,
            "score": score
        })
        return order
    except Exception as e:
        logger.error(f"Order failed: {e}")
        return None

def run_backtest_simulation(symbol: str, days: int = 365) -> Tuple[Optional[Dict[str, Any]], Any]:
    """Run a backtest simulation for a symbol."""
    if api is None:
        return None, "Alpaca API not initialized"
    
    try:
        limit = days + 250
        # Use 'iex' feed for paper trading (avoids SIP subscription error)
        bars = api.get_bars(symbol, TimeFrame.Day, limit=limit, feed='iex').df
        if bars.empty:
            return None, "No Data"
        
        bars = bars.reset_index()
        bars.columns = [c.lower() for c in bars.columns]
        
        bars['rsi'] = rsi(bars['close'], length=14)
        bars['sma_200'] = sma(bars['close'], length=200)
        bars['atr'] = atr(bars['high'], bars['low'], bars['close'], length=14)
        
        data = bars.iloc[-days:].copy().reset_index(drop=True)
        
    except Exception as e:
        return None, str(e)

    initial_capital = 10000.0
    cash = initial_capital
    equity_curve = [initial_capital]
    trades = []
    
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    state = 'SCANNING'

    for index, row in data.iterrows():
        current_price = row['close']
        current_date = row['timestamp'].strftime('%Y-%m-%d')
        
        if math.isnan(row['sma_200']) or math.isnan(row['atr']):
            equity_curve.append(cash + (position * current_price))
            continue

        if state == 'SCANNING':
            if current_price > row['sma_200'] and row['rsi'] < 35:
                atr = row['atr']
                stop_dist = 2 * atr
                target_dist = 3 * atr
                
                risk_dollars = 100
                shares = math.floor(risk_dollars / stop_dist)
                cost = shares * current_price
                
                if shares > 0 and cost < cash:
                    state = 'HOLDING'
                    position = shares
                    entry_price = current_price
                    stop_loss = current_price - stop_dist
                    take_profit = current_price + target_dist
                    cash -= cost
                    trades.append({"date": current_date, "type": "BUY", "price": entry_price})

        elif state == 'HOLDING':
            sell_price = None
            result = ""
            
            if row['low'] <= stop_loss:
                sell_price = stop_loss
                result = "STOP LOSS"
            elif row['high'] >= take_profit:
                sell_price = take_profit
                result = "WIN"
            
            if sell_price:
                pnl = (sell_price - entry_price) * position
                cash += (position * sell_price)
                state = 'SCANNING'
                trades.append({"date": current_date, "type": "SELL", "price": sell_price, "result": result, "pnl": pnl})
                position = 0

        current_val = cash + (position * current_price)
        equity_curve.append(current_val)

    sells = [t for t in trades if t['type'] == 'SELL']
    total_trades = len(sells)
    wins = len([t for t in sells if t.get('result') == 'WIN'])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    total_return = ((equity_curve[-1] - initial_capital) / initial_capital) * 100
    
    plt.figure(figsize=(6, 2.5), facecolor='#111827')
    ax = plt.gca()
    ax.set_facecolor('#111827')
    plt.plot(equity_curve, color='#ef4444', linewidth=1.5)
    plt.title(f"{symbol} - The Eye's Vision (1yr Swing Strategy)", color='white', fontsize=10)
    plt.tick_params(colors='gray', labelsize=8)
    plt.grid(color='#374151', linestyle='--', linewidth=0.5)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=False, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return {
        "trades": total_trades,
        "win_rate": round(win_rate, 1),
        "return": round(total_return, 2),
        "plot": plot_url
    }, trades
