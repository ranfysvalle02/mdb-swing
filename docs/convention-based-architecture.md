# Convention-Based Architecture (Rails-like)

This project follows a **Convention over Configuration** approach, similar to Ruby on Rails, where folder structure drives functionality and behavior.

## Core Principles

1. **Auto-Discovery**: Components are automatically discovered from folder structure
2. **Convention-Based**: Follow naming and structure conventions for automatic registration
3. **Pluggable**: Add new strategies, services, or routes by simply adding files to the appropriate folder
4. **Zero Configuration**: No manual registration needed - folder structure is the configuration

## Folder Structure

```
src/app/
├── strategies/          # Trading strategies (auto-discovered)
│   ├── base.py         # Strategy base class
│   ├── balanced_low.py # Balanced Low strategy
│   └── my_strategy.py  # Add new strategies here
├── services/           # Business logic services (auto-discovered)
│   ├── ai.py           # AI service
│   ├── eye.py          # Eye service
│   └── trading.py      # Trading service
├── api/                # API routes
│   └── routes.py       # Main routes (future: domain-driven subdirectories)
├── core/               # Core infrastructure
│   ├── registry.py     # Strategy registry (auto-discovery)
│   ├── plugins.py      # Service registry (auto-discovery)
│   ├── routing.py      # Route registry (auto-discovery)
│   └── config.py       # Configuration management
└── models/             # Data models
```

## Adding a New Strategy

### Step 1: Create Strategy File

Create a new file in `src/app/strategies/`:

```python
# src/app/strategies/momentum_breakout.py
from typing import Dict, Any
from .base import Strategy

class MomentumBreakoutStrategy(Strategy):
    """Momentum breakout strategy - buy on strong momentum breakouts."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "rsi_threshold": 70,  # Overbought breakout
            "ai_score_required": 8,
            "risk_per_trade": 50.0,
            "max_capital": 5000.0,
        }
    
    def check_technical_signal(self, techs: Dict[str, Any]) -> bool:
        """Check if momentum breakout conditions are met."""
        rsi_threshold = self.config.get("rsi_threshold", 70)
        return techs.get('rsi', 50) > rsi_threshold and techs.get('trend') == "UP"
    
    def get_ai_prompt(self) -> str:
        """Get AI prompt for momentum breakout analysis."""
        return (
            "You are analyzing momentum breakout opportunities for SWING TRADING.\n"
            "Focus on stocks with strong momentum breaking out to new highs.\n"
            # ... rest of prompt
        )
    
    def get_name(self) -> str:
        return "Momentum Breakout"
    
    def get_description(self) -> str:
        return "Buy stocks on strong momentum breakouts with continuation potential"
    
    def get_config(self) -> Dict[str, Any]:
        return {
            **self.config,
            "name": self.get_name(),
            "description": self.get_description(),
            "color": "blue"
        }
```

### Step 2: Use the Strategy

The strategy is automatically discovered! No registration needed.

```python
from src.app.core.registry import get_registry

# Get strategy by name (auto-discovered)
registry = get_registry()
strategy = registry.get_strategy(name="momentum_breakout")

# Or use via config
from src.app.core.config import get_strategy_instance
strategy = get_strategy_instance(strategy_name="momentum_breakout")
```

### Step 3: Set Active Strategy

Set via environment variable:

```bash
export CURRENT_STRATEGY=momentum_breakout
```

Or programmatically:

```python
from src.app.core.registry import get_registry
registry = get_registry()
registry.set_active_strategy("momentum_breakout")
```

## Adding a New Service

### Step 1: Create Service File

Create a new file in `src/app/services/`:

```python
# src/app/services/sentiment.py
from typing import Dict, Any

class SentimentService:
    """Service for analyzing market sentiment."""
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment from text."""
        # Implementation
        return {"sentiment": "positive", "score": 0.8}
```

### Step 2: Use the Service

The service is automatically discovered:

```python
from src.app.core.plugins import get_service

# Get service by name (auto-discovered)
sentiment_service = get_service("sentiment")
result = sentiment_service.analyze_sentiment("Great earnings report!")
```

## Registry System

### Strategy Registry

Auto-discovers strategies from `strategies/` folder:

```python
from src.app.core.registry import get_registry

registry = get_registry()

# List all discovered strategies
strategies = registry.list_strategies()
# ['balanced_low', 'momentum_breakout', ...]

# Get strategy instance
strategy = registry.get_strategy("balanced_low")

# Get strategy info
info = registry.get_strategy_info("balanced_low")
# {'name': 'Balanced Low', 'description': '...', 'config': {...}}
```

### Service Registry

Auto-discovers services from `services/` folder:

```python
from src.app.core.plugins import get_service_registry

registry = get_service_registry()

# List all discovered services
services = registry.list_services()
# ['ai', 'eye', 'trading', 'sentiment', ...]

# Get service
service = registry.get_service("sentiment")
```

## Configuration Priority

1. **Database** (highest priority) - Strategy config stored in MongoDB
2. **Environment Variables** - `CURRENT_STRATEGY`, etc.
3. **Default Values** - Hardcoded fallbacks

## Conventions

### File Naming

- **Strategies**: `{name}_strategy.py` or `{name}.py` (e.g., `balanced_low.py`)
- **Services**: `{name}.py` (e.g., `sentiment.py`)
- **Private Files**: Start with `_` (ignored by auto-discovery)

### Class Naming

- **Strategies**: `{Name}Strategy` (e.g., `BalancedLowStrategy`)
- **Services**: `{Name}Service` or descriptive name (e.g., `SentimentService`)

### Strategy Requirements

All strategies must:
1. Inherit from `Strategy` base class
2. Implement required abstract methods:
   - `check_technical_signal(techs) -> bool`
   - `get_ai_prompt() -> str`
   - `get_name() -> str`
3. Optionally implement:
   - `get_description() -> str`
   - `get_config() -> Dict[str, Any]`

## Benefits

1. **Zero Configuration**: Add files, they're automatically discovered
2. **Pluggable**: Easy to add new strategies without modifying core code
3. **Testable**: Each strategy is isolated and testable
4. **Maintainable**: Clear separation of concerns
5. **Scalable**: Add unlimited strategies without code changes

## Example: Complete Strategy Plugin

```python
# src/app/strategies/golden_cross.py
from typing import Dict, Any
from .base import Strategy

class GoldenCrossStrategy(Strategy):
    """Golden Cross strategy - buy when 50-day SMA crosses above 200-day SMA."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "sma_short": 50,
            "sma_long": 200,
            "ai_score_required": 7,
            "risk_per_trade": 50.0,
        }
    
    def check_technical_signal(self, techs: Dict[str, Any]) -> bool:
        """Check for golden cross signal."""
        # Implementation checks for SMA crossover
        return techs.get('sma_50', 0) > techs.get('sma_200', 0)
    
    def get_ai_prompt(self) -> str:
        return "Analyze golden cross opportunities for swing trading..."
    
    def get_name(self) -> str:
        return "Golden Cross"
    
    def get_description(self) -> str:
        return "Buy on golden cross (50-day SMA crosses above 200-day SMA)"
    
    def get_config(self) -> Dict[str, Any]:
        return {
            **self.config,
            "name": self.get_name(),
            "description": self.get_description(),
            "color": "gold"
        }
```

That's it! The strategy is automatically discovered and available via the registry.

## Pseudocode Strategies

The `strategies/` folder contains both active strategies and pseudocode placeholders:

### Active Strategies

- **balanced_low.py**: PRIMARY and ONLY active strategy
  - Fully implemented and tested
  - Used for all trading decisions
  - Well-documented with comprehensive comments

### Pseudocode Strategies (Future)

These files contain pseudocode for future strategies. They are NOT active and will NOT be loaded by the registry:

- **momentum_breakout.py**: Buy breakouts with strong momentum
- **golden_cross.py**: Buy on 50-day/200-day SMA crossover
- **support_bounce.py**: Buy bounces off support levels
- **example_strategy.py**: Template for creating new strategies

### How Pseudocode Works

Pseudocode files:
1. Contain `# PSEUDOCODE` header marker
2. Document strategy concept, entry/exit criteria, risk characteristics
3. Include implementation structure showing what the class would look like
4. Are automatically ignored by the strategy registry
5. Can be converted to active strategies by implementing the Strategy interface

### Converting Pseudocode to Active Strategy

To convert a pseudocode strategy to active:

1. **Remove PSEUDOCODE markers** from the file
2. **Implement Strategy interface**:
   - `check_technical_signal(techs) -> bool`
   - `get_ai_prompt() -> str`
   - `get_name() -> str`
   - `get_description() -> str`
   - `get_config() -> Dict[str, Any]`
3. **Test the strategy** thoroughly
4. **Update registry** (auto-discovery will pick it up)
5. **Set as active** via `CURRENT_STRATEGY` environment variable

### Why Pseudocode?

- **Planning**: Document future strategies without implementing them
- **Clarity**: Show what strategies might look like
- **Architecture**: Demonstrate the pluggable system's flexibility
- **Future-proofing**: Easy to implement when ready

## Future Enhancements

- **Domain-Driven Routes**: Organize routes by domain (positions/, strategies/, etc.)
- **Plugin System**: External plugins in `plugins/` folder
- **Strategy Presets**: Pre-configured strategy variants
- **Strategy Marketplace**: Share strategies via package system
