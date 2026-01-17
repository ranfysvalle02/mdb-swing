"""Strategy and Plugin Registry - Rails-like Convention over Configuration.

This module provides auto-discovery of strategies and plugins based on folder structure.
Following Rails conventions, the folder structure drives functionality and behavior.

Conventions:
- Strategies: Any class in `strategies/` folder that inherits from Strategy base class
- Plugins: Any module in `plugins/` folder (future expansion)
- Auto-discovery: Scans folders and registers available components
- Configuration: Active strategy selected via environment variable or config
"""
import importlib
import inspect
from pathlib import Path
from typing import Dict, Type, Optional, List, Any
from mdb_engine.observability import get_logger

logger = get_logger(__name__)


class StrategyRegistry:
    """Registry for auto-discovering and managing trading strategies.
    
    Rails-like convention: Strategies are auto-discovered from the strategies/ folder.
    Each strategy file should contain a class that inherits from Strategy base class.
    The class name should follow convention: {Name}Strategy (e.g., BalancedLowStrategy)
    """
    
    def __init__(self):
        self._strategies: Dict[str, Type] = {}
        self._instances: Dict[str, Any] = {}
        self._active_strategy_name: Optional[str] = None
        
    def discover_strategies(self, strategies_dir: Path) -> Dict[str, Type]:
        """Auto-discover strategies from the strategies directory.
        
        Convention: Scans strategies/ folder for Python files containing Strategy classes.
        Files starting with _ or __ are ignored (private/convention).
        
        Args:
            strategies_dir: Path to strategies directory
            
        Returns:
            Dictionary mapping strategy names to strategy classes
        """
        discovered = {}
        
        if not strategies_dir.exists():
            logger.warning(f"Strategies directory not found: {strategies_dir}")
            return discovered
        
        # Import base Strategy class for isinstance checks
        try:
            from ..strategies.base import Strategy
        except ImportError as e:
            logger.error(f"Failed to import Strategy base class: {e}")
            return discovered
        
        # Scan for Python files in strategies directory
        for file_path in strategies_dir.glob("*.py"):
            # Skip private files (Rails convention: _* files are private)
            if file_path.name.startswith("_") or file_path.name.startswith("__"):
                continue
            
            # Skip base.py (it's the base class, not a strategy)
            if file_path.name == "base.py":
                continue
            
            # Skip pseudocode files (they contain "# PSEUDOCODE" header)
            # Read first few lines to check for pseudocode marker
            try:
                with open(file_path, 'r') as f:
                    first_lines = ''.join(f.readlines()[:5])
                    if '# PSEUDOCODE' in first_lines.upper() or 'PSEUDOCODE PLACEHOLDER' in first_lines.upper():
                        logger.debug(f"Skipping pseudocode file: {file_path.name}")
                        continue
            except Exception:
                # If we can't read it, try to load it anyway (let import handle errors)
                pass
            
            try:
                # Import module (e.g., strategies.balanced_low)
                module_name = f"src.app.strategies.{file_path.stem}"
                module = importlib.import_module(module_name)
                
                # Find all Strategy subclasses in the module
                import re  # Import at top of loop for reuse
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Check if it's a Strategy subclass and not the base class itself
                    if (issubclass(obj, Strategy) and 
                        obj is not Strategy and 
                        obj.__module__ == module_name):
                        
                        # Get strategy name (class name without "Strategy" suffix, or full name)
                        # Normalize: remove "Strategy" suffix, convert to lowercase, handle CamelCase
                        if name.endswith("Strategy"):
                            base_name = name[:-8]  # Remove "Strategy"
                            # Convert CamelCase to snake_case (e.g., BalancedLow -> balanced_low)
                            snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', base_name).lower()
                            strategy_name = snake_case
                        else:
                            strategy_name = name.lower()
                        
                        # Store by normalized name (e.g., "balanced_low")
                        discovered[strategy_name] = obj
                        # Also store by full class name (e.g., "balancedlowstrategy")
                        discovered[name.lower()] = obj
                        # Store by filename convention (e.g., "balanced_low" from balanced_low.py)
                        file_based_name = file_path.stem.lower()
                        if file_based_name != strategy_name:
                            discovered[file_based_name] = obj
                            
                        logger.info(f"✅ Discovered strategy: {name} (registered as '{strategy_name}')")
                        
            except Exception as e:
                logger.warning(f"Failed to load strategy from {file_path.name}: {e}")
                continue
        
        self._strategies.update(discovered)
        return discovered
    
    def register_strategy(self, name: str, strategy_class: Type):
        """Manually register a strategy class.
        
        Args:
            name: Strategy identifier (lowercase, e.g., 'balanced_low')
            strategy_class: Strategy class that inherits from Strategy base class
        """
        self._strategies[name.lower()] = strategy_class
        logger.info(f"✅ Registered strategy: {name}")
    
    def get_strategy(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get strategy instance (singleton per strategy name).
        
        Convention: If name is None, returns the active strategy.
        Active strategy is set via set_active_strategy() or environment variable.
        
        Args:
            name: Strategy name (optional, defaults to active strategy)
            config: Strategy configuration dict (optional)
            
        Returns:
            Strategy instance
            
        Raises:
            ValueError: If strategy not found
        """
        # Use active strategy if name not provided
        if name is None:
            name = self._active_strategy_name
        
        if name is None:
            # Default to first discovered strategy or 'balanced_low'
            if self._strategies:
                name = list(self._strategies.keys())[0]
                logger.info(f"Using default strategy: {name}")
            else:
                raise ValueError("No strategies available and no active strategy set")
        
        name = name.lower()
        
        # Return cached instance if available
        if name in self._instances:
            return self._instances[name]
        
        # Get strategy class
        strategy_class = self._strategies.get(name)
        if strategy_class is None:
            # Try to find by partial match
            for key, cls in self._strategies.items():
                if name in key or key in name:
                    strategy_class = cls
                    name = key
                    break
        
        if strategy_class is None:
            available = ", ".join(self._strategies.keys())
            raise ValueError(f"Strategy '{name}' not found. Available: {available}")
        
        # Create instance with config
        try:
            instance = strategy_class(config=config)
            self._instances[name] = instance
            logger.info(f"✅ Created strategy instance: {name}")
            return instance
        except Exception as e:
            logger.error(f"Failed to create strategy instance {name}: {e}", exc_info=True)
            raise
    
    def set_active_strategy(self, name: str):
        """Set the active strategy (used when name is not specified).
        
        Args:
            name: Strategy name (will be lowercased)
        """
        name = name.lower()
        if name not in self._strategies:
            available = ", ".join(self._strategies.keys())
            raise ValueError(f"Strategy '{name}' not found. Available: {available}")
        self._active_strategy_name = name
        logger.info(f"✅ Active strategy set to: {name}")
    
    def list_strategies(self) -> List[str]:
        """List all discovered strategy names.
        
        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())
    
    def get_strategy_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get strategy metadata.
        
        Args:
            name: Strategy name (optional, defaults to active)
            
        Returns:
            Dictionary with strategy metadata
        """
        strategy = self.get_strategy(name)
        return {
            "name": strategy.get_name(),
            "description": strategy.get_description(),
            "config": strategy.get_config(),
            "class_name": strategy.__class__.__name__,
        }


# Global registry instance (singleton pattern)
_registry: Optional[StrategyRegistry] = None


def get_registry() -> StrategyRegistry:
    """Get the global strategy registry instance.
    
    Convention: Auto-discovers strategies on first access.
    
    Returns:
        StrategyRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = StrategyRegistry()
        # Auto-discover strategies
        strategies_dir = Path(__file__).parent.parent / "strategies"
        _registry.discover_strategies(strategies_dir)
    return _registry


def discover_strategies() -> Dict[str, Type]:
    """Convenience function to discover strategies.
    
    Returns:
        Dictionary of discovered strategies
    """
    return get_registry().discover_strategies(
        Path(__file__).parent.parent / "strategies"
    )
