"""Custom strategy evaluation service."""
from typing import Dict, Any, List, Optional, Tuple
from mdb_engine.observability import get_logger

logger = get_logger(__name__)


def evaluate_strategy(
    stats: Dict[str, Any],
    strategy_config: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """Evaluate if a stock meets the criteria defined in a custom strategy.
    
    Args:
        stats: Dictionary of stock statistics from analyze_comprehensive_stats()
        strategy_config: Strategy configuration with 'conditions' array
        
    Returns:
        Tuple of (meets_criteria: bool, rejection_reasons: List[str])
        rejection_reasons is empty if meets_criteria is True
    """
    conditions = strategy_config.get('conditions', [])
    
    if not conditions:
        logger.warning("Strategy has no conditions - rejecting by default")
        return False, ["Strategy has no conditions defined"]
    
    rejection_reasons = []
    all_conditions_met = True
    
    for condition in conditions:
        metric = condition.get('metric')
        operator = condition.get('operator')
        
        if not metric or not operator:
            logger.warning(f"Invalid condition: missing metric or operator: {condition}")
            all_conditions_met = False
            rejection_reasons.append(f"Invalid condition: {condition}")
            continue
        
        # Get metric value from stats
        metric_value = stats.get(metric)
        
        # Handle None/missing values
        if metric_value is None:
            all_conditions_met = False
            rejection_reasons.append(f"{metric} is not available")
            continue
        
        # Evaluate condition based on operator
        condition_met = False
        
        try:
            if operator == 'between':
                min_val = condition.get('min')
                max_val = condition.get('max')
                if min_val is not None and max_val is not None:
                    condition_met = min_val <= metric_value <= max_val
                    if not condition_met:
                        rejection_reasons.append(
                            f"{metric} ({metric_value:.2f}) not between {min_val} and {max_val}"
                        )
                else:
                    logger.warning(f"Between operator missing min/max: {condition}")
                    all_conditions_met = False
                    rejection_reasons.append(f"{metric}: Invalid 'between' condition")
                    
            elif operator == 'less_than':
                threshold = condition.get('value')
                if threshold is not None:
                    condition_met = metric_value < threshold
                    if not condition_met:
                        rejection_reasons.append(
                            f"{metric} ({metric_value:.2f}) >= {threshold}"
                        )
                else:
                    logger.warning(f"less_than operator missing value: {condition}")
                    all_conditions_met = False
                    rejection_reasons.append(f"{metric}: Invalid 'less_than' condition")
                    
            elif operator == 'less_than_or_equal':
                threshold = condition.get('value')
                if threshold is not None:
                    condition_met = metric_value <= threshold
                    if not condition_met:
                        rejection_reasons.append(
                            f"{metric} ({metric_value:.2f}) > {threshold}"
                        )
                else:
                    logger.warning(f"less_than_or_equal operator missing value: {condition}")
                    all_conditions_met = False
                    rejection_reasons.append(f"{metric}: Invalid 'less_than_or_equal' condition")
                    
            elif operator == 'greater_than':
                threshold = condition.get('value')
                if threshold is not None:
                    condition_met = metric_value > threshold
                    if not condition_met:
                        rejection_reasons.append(
                            f"{metric} ({metric_value:.2f}) <= {threshold}"
                        )
                else:
                    logger.warning(f"greater_than operator missing value: {condition}")
                    all_conditions_met = False
                    rejection_reasons.append(f"{metric}: Invalid 'greater_than' condition")
                    
            elif operator == 'greater_than_or_equal':
                threshold = condition.get('value')
                if threshold is not None:
                    condition_met = metric_value >= threshold
                    if not condition_met:
                        rejection_reasons.append(
                            f"{metric} ({metric_value:.2f}) < {threshold}"
                        )
                else:
                    logger.warning(f"greater_than_or_equal operator missing value: {condition}")
                    all_conditions_met = False
                    rejection_reasons.append(f"{metric}: Invalid 'greater_than_or_equal' condition")
                    
            elif operator == 'equals':
                expected_value = condition.get('value')
                if expected_value is not None:
                    # For string values (like trend), use case-insensitive comparison
                    if isinstance(metric_value, str) and isinstance(expected_value, str):
                        condition_met = metric_value.upper() == expected_value.upper()
                    else:
                        condition_met = metric_value == expected_value
                    if not condition_met:
                        rejection_reasons.append(
                            f"{metric} ({metric_value}) != {expected_value}"
                        )
                else:
                    logger.warning(f"equals operator missing value: {condition}")
                    all_conditions_met = False
                    rejection_reasons.append(f"{metric}: Invalid 'equals' condition")
                    
            elif operator == 'not_equals':
                not_value = condition.get('value')
                if not_value is not None:
                    if isinstance(metric_value, str) and isinstance(not_value, str):
                        condition_met = metric_value.upper() != not_value.upper()
                    else:
                        condition_met = metric_value != not_value
                    if not condition_met:
                        rejection_reasons.append(
                            f"{metric} ({metric_value}) == {not_value}"
                        )
                else:
                    logger.warning(f"not_equals operator missing value: {condition}")
                    all_conditions_met = False
                    rejection_reasons.append(f"{metric}: Invalid 'not_equals' condition")
                    
            else:
                logger.warning(f"Unknown operator: {operator}")
                all_conditions_met = False
                rejection_reasons.append(f"{metric}: Unknown operator '{operator}'")
                
        except Exception as e:
            logger.error(f"Error evaluating condition {condition}: {e}", exc_info=True)
            all_conditions_met = False
            rejection_reasons.append(f"{metric}: Evaluation error - {str(e)[:50]}")
            continue
        
        if not condition_met:
            all_conditions_met = False
    
    return all_conditions_met, rejection_reasons if not all_conditions_met else []


def get_available_metrics() -> Dict[str, List[Dict[str, Any]]]:
    """Get list of available metrics organized by category.
    
    Returns:
        Dictionary mapping category names to lists of metric definitions
    """
    return {
        "Momentum Indicators": [
            {
                "key": "rsi",
                "name": "RSI (14)",
                "description": "Relative Strength Index. <30 = oversold (buy), >70 = overbought (sell)",
                "type": "number",
                "range": (0, 100),
                "default_operator": "between",
                "default_min": 20,
                "default_max": 35
            },
            {
                "key": "stoch_k",
                "name": "Stochastic %K",
                "description": "Momentum oscillator. <20 = oversold, >80 = overbought",
                "type": "number",
                "range": (0, 100),
                "default_operator": "between",
                "default_min": 0,
                "default_max": 20
            },
            {
                "key": "stoch_d",
                "name": "Stochastic %D",
                "description": "Smoothed %K. %K crosses above %D = bullish",
                "type": "number",
                "range": (0, 100),
                "default_operator": "between",
                "default_min": 0,
                "default_max": 20
            },
            {
                "key": "macd",
                "name": "MACD Line",
                "description": "Fast (12) - slow (26) EMAs. Positive = bullish",
                "type": "number",
                "range": None,
                "default_operator": "greater_than",
                "default_value": 0
            },
            {
                "key": "macd_signal",
                "name": "MACD Signal",
                "description": "9-period EMA of MACD",
                "type": "number",
                "range": None,
                "default_operator": "less_than",
                "default_value": 0
            },
            {
                "key": "macd_histogram",
                "name": "MACD Histogram",
                "description": "MACD - Signal. Positive = momentum increasing",
                "type": "number",
                "range": None,
                "default_operator": "greater_than",
                "default_value": 0
            }
        ],
        "Moving Averages": [
            {
                "key": "sma_20",
                "name": "SMA-20",
                "description": "20-day Simple Moving Average",
                "type": "price",
                "range": None,
                "default_operator": "less_than",
                "default_value": None
            },
            {
                "key": "sma_50",
                "name": "SMA-50",
                "description": "50-day Simple Moving Average",
                "type": "price",
                "range": None,
                "default_operator": "less_than",
                "default_value": None
            },
            {
                "key": "sma_100",
                "name": "SMA-100",
                "description": "100-day Simple Moving Average",
                "type": "price",
                "range": None,
                "default_operator": "less_than",
                "default_value": None
            },
            {
                "key": "sma_200",
                "name": "SMA-200",
                "description": "200-day Simple Moving Average (long-term trend)",
                "type": "price",
                "range": None,
                "default_operator": "less_than",
                "default_value": None
            },
            {
                "key": "price_vs_sma_200_pct",
                "name": "Price vs SMA-200 %",
                "description": "Distance from SMA-200. 0-3% = near support (ideal)",
                "type": "percentage",
                "range": (-50, 50),
                "default_operator": "between",
                "default_min": 0,
                "default_max": 3
            },
            {
                "key": "price_vs_sma_100_pct",
                "name": "Price vs SMA-100 %",
                "description": "Distance from SMA-100",
                "type": "percentage",
                "range": (-50, 50),
                "default_operator": "between",
                "default_min": -5,
                "default_max": 5
            },
            {
                "key": "price_vs_sma_50_pct",
                "name": "Price vs SMA-50 %",
                "description": "Distance from SMA-50",
                "type": "percentage",
                "range": (-50, 50),
                "default_operator": "between",
                "default_min": -5,
                "default_max": 5
            },
            {
                "key": "ema_12",
                "name": "EMA-12",
                "description": "12-day Exponential Moving Average",
                "type": "price",
                "range": None,
                "default_operator": "less_than",
                "default_value": None
            },
            {
                "key": "ema_26",
                "name": "EMA-26",
                "description": "26-day Exponential Moving Average",
                "type": "price",
                "range": None,
                "default_operator": "less_than",
                "default_value": None
            }
        ],
        "Bollinger Bands": [
            {
                "key": "bb_percent_b",
                "name": "%B Position",
                "description": "Position within Bollinger Bands. <20% = oversold, >80% = overbought",
                "type": "percentage",
                "range": (0, 100),
                "default_operator": "less_than",
                "default_value": 20
            },
            {
                "key": "bb_width_pct",
                "name": "Band Width %",
                "description": "Volatility measure. <3% = low vol (expect breakout), >5% = high vol",
                "type": "percentage",
                "range": (0, 20),
                "default_operator": "between",
                "default_min": 0,
                "default_max": 5
            }
        ],
        "Volatility & Range": [
            {
                "key": "atr",
                "name": "ATR (14)",
                "description": "Average True Range. Use for stop-loss: Entry - (2×ATR)",
                "type": "price",
                "range": None,
                "default_operator": "greater_than",
                "default_value": 0
            },
            {
                "key": "atr_pct",
                "name": "ATR %",
                "description": "Volatility relative to price. <2% = low vol, >5% = high vol",
                "type": "percentage",
                "range": (0, 20),
                "default_operator": "less_than",
                "default_value": 5
            },
            {
                "key": "price_position_in_range",
                "name": "Price Position in Range",
                "description": "Position in 20-day range. <30% = near bottom (buy), >70% = near top (sell)",
                "type": "percentage",
                "range": (0, 100),
                "default_operator": "less_than",
                "default_value": 30
            }
        ],
        "Volume": [
            {
                "key": "volume_ratio",
                "name": "Volume Ratio",
                "description": "Current volume vs 20-day average. >1.5x = high volume, <0.5x = low volume",
                "type": "number",
                "range": (0, 10),
                "default_operator": "greater_than",
                "default_value": 1.2
            }
        ],
        "Trend": [
            {
                "key": "trend",
                "name": "Trend",
                "description": "UP = price above SMA-200, DOWN = price below SMA-200",
                "type": "string",
                "range": None,
                "default_operator": "equals",
                "default_value": "UP"
            }
        ],
        "Price": [
            {
                "key": "price",
                "name": "Price",
                "description": "Current stock price",
                "type": "price",
                "range": None,
                "default_operator": "greater_than",
                "default_value": 0
            },
            {
                "key": "change_pct",
                "name": "Change %",
                "description": "Daily price change percentage",
                "type": "percentage",
                "range": (-20, 20),
                "default_operator": "between",
                "default_min": -10,
                "default_max": 10
            }
        ]
    }


def format_strategy_description(strategy_config: Dict[str, Any]) -> str:
    """Generate a plain English description of a strategy.
    
    Args:
        strategy_config: Strategy configuration with 'conditions' array
        
    Returns:
        Human-readable description of the strategy
    """
    conditions = strategy_config.get('conditions', [])
    
    if not conditions:
        return "No conditions defined"
    
    descriptions = []
    for condition in conditions:
        metric = condition.get('metric', 'unknown')
        operator = condition.get('operator', 'unknown')
        
        metric_name = metric.replace('_', ' ').title()
        
        if operator == 'between':
            min_val = condition.get('min')
            max_val = condition.get('max')
            descriptions.append(f"{metric_name} between {min_val} and {max_val}")
        elif operator in ['less_than', 'less_than_or_equal']:
            threshold = condition.get('value')
            descriptions.append(f"{metric_name} < {threshold}")
        elif operator in ['greater_than', 'greater_than_or_equal']:
            threshold = condition.get('value')
            descriptions.append(f"{metric_name} > {threshold}")
        elif operator == 'equals':
            value = condition.get('value')
            descriptions.append(f"{metric_name} = {value}")
        elif operator == 'not_equals':
            value = condition.get('value')
            descriptions.append(f"{metric_name} ≠ {value}")
        else:
            descriptions.append(f"{metric_name} ({operator})")
    
    return " AND ".join(descriptions)
