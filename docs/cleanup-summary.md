# Code Cleanup Summary - Future-Proof Architecture

**Status:** ✅ Complete - All backwards compatibility removed, dead code eliminated

## Files Deleted (Dead Code Removed)

1. **`src/app/strategies/balanced_low.py`** → Upgraded to use vectorbt
2. **`src/app/strategies/example_strategy.py`** - Pseudocode placeholder
3. **`src/app/strategies/golden_cross.py`** - Pseudocode placeholder
4. **`src/app/strategies/momentum_breakout.py`** - Pseudocode placeholder
5. **`src/app/strategies/support_bounce.py`** - Pseudocode placeholder
6. **`src/app/strategies/base.py`** - Old Strategy class interface
7. **`src/app/services/eye.py`** - Old Eye class (not used)
8. **`src/app/core/registry.py`** - Strategy registry (not needed)

## Files Created (New Architecture)

1. **`src/app/core/schema.py`** - MongoDB schema definitions
2. **`src/app/data/market_data.py`** - Data ingestion (NOT MongoDB)
3. **`src/app/strategies/balanced_low.py`** - vectorbt-based strategy
4. **`src/app/services/persistence.py`** - MongoDB persistence layer
5. **`src/app/api/strategy_orchestrator.py`** - Orchestration following architecture
6. **`src/app/services/ai_prompts.py`** - AI prompts (no class dependencies)

## Code Updated

### Core Changes

1. **`src/app/core/config.py`**
   - Removed `get_strategy_instance()` - replaced with `get_strategy_config()`
   - Removed `STRATEGY_CONFIGS` dict - simplified to `STRATEGY_CONFIG`
   - Config now loads from MongoDB or env vars

2. **`src/app/services/analysis.py`**
   - Updated `analyze_technicals()` to use vectorbt instead of custom indicators
   - Removed dependency on `indicators.py` functions

3. **`src/app/api/routes.py`**
   - Removed all `get_strategy_instance()` calls
   - Replaced `strategy.get_ai_prompt()` with `get_balanced_low_prompt()`
   - Replaced `strategy.get_config()` with `get_strategy_config(db)`
   - Updated to use new architecture pattern

4. **`src/app/main.py`**
   - Removed `get_strategy_instance()` import and usage
   - Simplified startup logging

5. **`src/app/strategies/__init__.py`**
   - Removed old Strategy class imports
   - Exports vectorbt functions: `generate_signals`, `calculate_position_size`, `run_portfolio`

6. **`config/manifest.json`**
   - Added indexes for `strategies`, `signals`, `trades` collections

## Architecture Principles Applied

✅ **MongoDB remembers what *happened*. Python decides what *should happen*.**

- **OHLCV data**: NOT in MongoDB (uses cache/API via `data/market_data.py`)
- **Indicators**: NOT in MongoDB (computed with vectorbt on-the-fly)
- **Decisions**: YES in MongoDB (`signals` collection)
- **Trades**: YES in MongoDB (`trades` collection)
- **Config**: YES in MongoDB (`strategies` collection)

## Dependencies

- Added: `vectorbt>=0.25.0` for vectorized indicator calculations
- Removed: Dependencies on old Strategy class pattern

## Migration Path

All code now uses the new architecture:
- Strategy logic: `src/app/strategies/balanced_low.py` (vectorbt)
- Data loading: `src/app/data/market_data.py` (cache/API)
- Persistence: `src/app/services/persistence.py` (MongoDB)
- Orchestration: `src/app/api/strategy_orchestrator.py` (follows pattern)

## Next Steps

1. Wire up orchestrator to routes (optional - can use existing routes)
2. Add backtesting using vectorbt portfolio simulation
3. Migrate existing trade execution to use new persistence layer
