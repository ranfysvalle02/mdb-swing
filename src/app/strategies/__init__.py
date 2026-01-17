"""Strategy implementations for the Eye of Sauron.

PRIMARY STRATEGY: Balanced Low (ONLY active strategy)
-------------------------------------------------------
- balanced_low.py: ACTIVE - This is the primary and only active strategy
- All trading decisions use the Balanced Low approach

PSEUDOCODE STRATEGIES (Future Implementation):
-----------------------------------------------
These are pseudocode placeholders for future strategies. They are NOT active
and will NOT be loaded by the strategy registry:

- momentum_breakout.py: PSEUDOCODE - Buy breakouts with strong momentum
- golden_cross.py: PSEUDOCODE - Buy on 50-day/200-day SMA crossover
- support_bounce.py: PSEUDOCODE - Buy bounces off support levels
- example_strategy.py: PSEUDOCODE - Template/example for creating new strategies

Convention: Strategies are auto-discovered via the StrategyRegistry.
Only files containing proper Strategy class implementations are loaded.
Pseudocode files (with "# PSEUDOCODE" header) are ignored.

To add a new active strategy:
1. Create a new file in this directory
2. Implement a class inheriting from Strategy base class
3. The registry will auto-discover it
4. Set CURRENT_STRATEGY environment variable to use it

Example:
    # strategies/my_strategy.py
    from .base import Strategy
    
    class MyStrategy(Strategy):
        def get_name(self):
            return "My Strategy"
        # ... implement required methods
"""
from .base import Strategy
from .balanced_low import BalancedLowStrategy

# Auto-discovery happens via registry - these are for explicit imports if needed
# NOTE: Only BalancedLowStrategy is currently active
__all__ = ['Strategy', 'BalancedLowStrategy']
