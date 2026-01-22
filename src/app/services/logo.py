"""Logo service with SVG initials fallback."""
import hashlib
from typing import Dict
from mdb_engine.observability import get_logger

logger = get_logger(__name__)

# In-memory cache for SVG initials (avoid regenerating)
_svg_cache: Dict[str, str] = {}


def _generate_svg_initials(ticker: str) -> str:
    """Generate SVG initials as fallback logo.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL' -> 'A')
        
    Returns:
        SVG string with initials
    """
    # Check in-memory cache first
    if ticker in _svg_cache:
        return _svg_cache[ticker]
    
    # Get first letter of ticker
    initial = ticker[0].upper() if ticker else "?"
    
    # Generate deterministic color based on ticker
    # Use hash to get consistent color per ticker
    hash_val = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16)
    
    # Color palette (good contrast on dark background)
    colors = [
        "#00f0ff",  # Primary cyan
        "#7000ff",  # Secondary purple
        "#ff0055",  # Accent red
        "#22c55e",  # Success green
        "#eab308",  # Warning yellow
        "#3b82f6",  # Info blue
    ]
    color = colors[hash_val % len(colors)]
    
    # Generate SVG
    svg = f'''<svg width="40" height="40" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
  <circle cx="20" cy="20" r="18" fill="{color}" opacity="0.2"/>
  <circle cx="20" cy="20" r="18" stroke="{color}" stroke-width="1.5" fill="none" opacity="0.4"/>
  <text x="20" y="20" font-family="monospace" font-size="18" font-weight="bold" 
        fill="{color}" text-anchor="middle" dominant-baseline="central">{initial}</text>
</svg>'''
    
    # Cache in memory
    _svg_cache[ticker] = svg
    
    return svg


def get_svg_initials(ticker: str) -> str:
    """Get SVG initials fallback for a ticker.
    
    Always returns SVG (never fails).
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        SVG string with initials
    """
    if not ticker:
        ticker = "?"
    
    return _generate_svg_initials(ticker.upper().strip())


def get_logo_html(ticker: str, css_class: str = "company-logo") -> str:
    """Get logo as HTML (SVG initials).
    
    Returns ready-to-use HTML fragment with SVG initials.
    No external API calls, instant rendering.
    
    Args:
        ticker: Stock ticker symbol
        css_class: CSS class for container div
        
    Returns:
        HTML string with <div> containing SVG element
    """
    svg = get_svg_initials(ticker)
    return f'<div class="{css_class}" data-ticker="{ticker}">{svg}</div>'
