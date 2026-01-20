"""Jinja2 Templates Configuration.

Centralized template configuration to avoid circular imports.
Enables autoescape for security and adds custom filters.
"""
import json
from pathlib import Path
from fastapi import Request
from fastapi.templating import Jinja2Templates

# Templates directory relative to this file
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Enable autoescape for security (prevents XSS)
templates.env.autoescape = True

# Note: CSRF tokens are available via request.state.csrf_token in route handlers
# Pass csrf_token explicitly to templates when needed

# Import logo service for SVG initials filter (import at module level to avoid circular imports)
try:
    from ..services.logo import get_svg_initials
    LOGO_SERVICE_AVAILABLE = True
except ImportError:
    LOGO_SERVICE_AVAILABLE = False
    def get_svg_initials(ticker: str) -> str:
        """Fallback if logo service not available."""
        return f'<svg width="40" height="40" viewBox="0 0 40 40"><text x="20" y="20">?</text></svg>'

# Add custom filters
def tojson_filter(value):
    """Convert Python object to JSON string, properly escaped for HTML attributes."""
    return json.dumps(value)

def round_filter(value, precision=0):
    """Round a number to specified precision."""
    try:
        return round(float(value), precision)
    except (ValueError, TypeError):
        return value

def int_filter(value):
    """Convert value to integer."""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return value

def svg_initials_filter(value):
    """Generate SVG initials for a ticker symbol."""
    if not value:
        value = "?"
    return get_svg_initials(str(value))

# Register filters with Jinja2 environment
templates.env.filters["tojson"] = tojson_filter
templates.env.filters["round"] = round_filter
templates.env.filters["int"] = int_filter
templates.env.filters["svg_initials"] = svg_initials_filter
