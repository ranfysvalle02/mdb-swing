# Architecture Implementation Summary

**Status:** ✅ Core Components Implemented

## What Was Added

### 1. MongoDB Schema Definitions (`src/app/core/schema.py`)
- **strategies**: Configuration storage (versioned, auditable)
- **signals**: Decision history (what the algorithm decided)
- **trades**: Trade journal (executed trades with lifecycle)

### 2. Data Ingestion Layer (`src/app/data/market_data.py`)
- **load_daily_bars()**: Loads OHLCV from cache/API, NOT MongoDB
- **load_ai_scores()**: Loads AI scores from MongoDB signals collection
- Uses file-based cache (Parquet/pickle) for performance
- Architecture principle: Market data is disposable computation fuel

### 3. Strategy Engine (`src/app/strategies/balanced_low.py`)
- **generate_signals()**: Uses vectorbt for vectorized indicator calculation
- **calculate_position_size()**: Pure computation, no MongoDB
- **run_portfolio()**: In-memory portfolio simulation
- Architecture principle: Indicators computed on-the-fly, not stored

### 4. Persistence Layer (`src/app/services/persistence.py`)
- **persist_signal()**: Stores decisions (BUY/SELL/HOLD/SKIP)
- **persist_trade()**: Stores trade facts (entry, stop, target)
- **update_trade_status()**: Trade lifecycle management
- Architecture principle: MongoDB stores what *happened*, not computation

### 5. Orchestration (`src/app/api/strategy_orchestrator.py`)
- **run_balanced_low_strategy()**: Ties everything together
- Follows the architecture pattern:
  1. Fetch Config (MongoDB)
  2. Load Data (Cache/API)
  3. Compute (Memory - vectorbt)
  4. Persist Results (MongoDB - decisions only)
  5. Return for UI rendering

### 6. MongoDB Collections (Updated `config/manifest.json`)
- Added indexes for `strategies`, `signals`, and `trades` collections
- Proper indexing for query performance

## Architecture Principles Applied

✅ **MongoDB remembers what *happened*. Python decides what *should happen*.**

- OHLCV data: NOT in MongoDB (uses cache/API)
- Indicators: NOT in MongoDB (computed on-the-fly with vectorbt)
- Decisions: YES in MongoDB (signals collection)
- Trades: YES in MongoDB (trades collection)
- Config: YES in MongoDB (strategies collection)

## Next Steps

1. **Trade Lifecycle State Machine**: Implement state transitions (Open -> Monitoring -> Closed)
2. **Integration**: Wire up the orchestrator to existing routes
3. **Backtesting**: Use vectorbt for historical simulation
4. **Migration**: Gradually migrate existing code to use new architecture

## Usage Example

```python
from ..api.strategy_orchestrator import run_balanced_low_strategy

# Run strategy following architecture pattern
result = await run_balanced_low_strategy(
    symbols=["AAPL", "NVDA"],
    date="2026-01-17",
    db=db
)

# Result contains:
# - signals: List of BUY signals created
# - portfolio_stats: Simulation results
# - config_used: Configuration that was applied
```

## Dependencies Added

- `vectorbt>=0.25.0`: For vectorized indicator calculations
