"""Signal Hunter service for Finviz screener integration."""
from typing import List, Dict, Any, Optional
import pandas as pd
from mdb_engine.observability import get_logger

logger = get_logger(__name__)

# Filter Presets - Easy-to-use strategies for finding opportunities
# NOTE: All filter values must match Finviz's exact filter option strings
FILTER_PRESETS = {
    "Buy Low": {
        'RSI (14)': 'Oversold (30)',
        'Performance': 'Month Down',
        'Average Volume': 'Over 500K',
        'Price': 'Over $5'
    },
    "Catch Momentum": {
        'RSI (14)': 'Not Overbought (<60)',
        'Performance': 'Month Up',
        'Average Volume': 'Over 500K',
        'Price': 'Over $5'
    }
}

# Default preset to auto-load
DEFAULT_PRESET = "Buy Low"


def get_screener_results(
    filters_dict: Optional[Dict[str, str]] = None,
    preset_name: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Fetch stocks from Finviz using Overview screener with preset filters.
    
    Args:
        filters_dict: Dictionary of Finviz filters (e.g., {'P/E': 'Under 20', 'RSI (14)': 'Oversold (40)'})
                     If None, uses preset_name instead
        preset_name: Name of preset to use (e.g., "Super Filter", "Deep Value")
                     If None and filters_dict is None, uses DEFAULT_PRESET
        limit: Maximum number of results to return (default: 100)
        
    Returns:
        List of dictionaries with stock data including symbol and metadata
        Returns empty list on error
    """
    try:
        from finvizfinance.screener.overview import Overview
        
        # Determine which filters to use
        if filters_dict is None:
            if preset_name and preset_name in FILTER_PRESETS:
                filters_dict = FILTER_PRESETS[preset_name].copy()
                logger.info(f"[SIGNAL_HUNTER] Using preset: {preset_name}")
            else:
                filters_dict = FILTER_PRESETS[DEFAULT_PRESET].copy()
                logger.info(f"[SIGNAL_HUNTER] Using default preset: {DEFAULT_PRESET}")
        
        logger.info(f"[SIGNAL_HUNTER] Fetching stocks with filters: {filters_dict}")
        
        foverview = Overview()
        
        try:
            foverview.set_filter(filters_dict=filters_dict)
            logger.info(f"[SIGNAL_HUNTER] Filter applied successfully")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[SIGNAL_HUNTER] Error setting filter: {error_msg}", exc_info=True)
            
            # Try to extract valid options from error message
            import re
            if "Invalid filter option" in error_msg and "Possible filter options" in error_msg:
                options_match = re.search(r"Possible filter options: \[(.*?)\]", error_msg)
                if options_match:
                    possible_options = [f.strip().strip("'\"") for f in options_match.group(1).split(',')]
                    logger.error(f"[SIGNAL_HUNTER] Valid options: {possible_options[:10]}...")
                    # Try to find which filter caused the issue
                    filter_match = re.search(r"Invalid filter option '([^']+)'", error_msg)
                    if filter_match:
                        invalid_option = filter_match.group(1)
                        logger.error(f"[SIGNAL_HUNTER] Invalid option '{invalid_option}' was used. Check preset filters.")
            
            return []
        
        # Get screener results
        try:
            logger.info(f"[SIGNAL_HUNTER] Calling screener_view()...")
            df = foverview.screener_view()
            logger.info(f"[SIGNAL_HUNTER] Retrieved {len(df) if df is not None and not df.empty else 0} stocks")
        except Exception as e:
            logger.error(f"[SIGNAL_HUNTER] Error fetching screener view: {e}", exc_info=True)
            return []
        
        if df is None or df.empty:
            logger.warning(f"[SIGNAL_HUNTER] No stocks found matching filters: {filters_dict}")
            return []
        
        # Fix Volume data for sorting (remove commas, convert to numeric)
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].astype(str).str.replace(',', '', regex=False)
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        
        # Sort by Volume (descending) and limit
        if 'Volume' in df.columns:
            df = df.sort_values(by='Volume', ascending=False)
        
        df = df.head(limit)
        
        # Convert DataFrame to list of dicts
        results = []
        for _, row in df.iterrows():
            # Get ticker/symbol
            ticker = None
            for col in ['Ticker', 'ticker', 'Symbol', 'symbol']:
                if col in df.columns:
                    ticker = str(row[col]).upper().strip()
                    break
            
            if not ticker or ticker == 'NAN' or ticker == 'NONE':
                continue
            
            stock_data = {
                'symbol': ticker,
                'raw_data': row.to_dict()
            }
            
            # Extract common fields
            for field in ['Company', 'Sector', 'Price', 'Volume', 'Market Cap', 'P/E']:
                if field in df.columns:
                    value = row[field]
                    # Handle numeric fields
                    if field == 'Price' and isinstance(value, str):
                        try:
                            value = float(value.replace('$', '').replace(',', ''))
                        except (ValueError, TypeError):
                            pass
                    stock_data[field.lower().replace(' ', '_')] = value
            
            results.append(stock_data)
        
        logger.info(f"[SIGNAL_HUNTER] Returning {len(results)} stocks")
        return results
        
    except Exception as e:
        logger.error(f"[SIGNAL_HUNTER] Unexpected error: {e}", exc_info=True)
        return []
