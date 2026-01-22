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

def render_template(template_name: str, request: Request, **context) -> str:
    """Render a template with CSRF token automatically included.
    
    HTMX Best Practice: CSRF tokens should be available in all templates.
    This helper ensures csrf_token is always in the template context.
    
    Args:
        template_name: Name of template file (e.g., "pages/account_balance.html")
        request: FastAPI request object (required for CSRF token)
        **context: Additional template variables
        
    Returns:
        Rendered HTML string
        
    Example:
        html = render_template("pages/account_balance.html", request, balance=1000)
    """
    # Ensure CSRF token is available
    csrf_token = getattr(request.state, "csrf_token", None)
    if not csrf_token:
        from .security import get_csrf_token_for_template
        csrf_token = get_csrf_token_for_template(request)
        request.state.csrf_token = csrf_token
    
    # Add CSRF token and request to context
    context["csrf_token"] = csrf_token
    context["request"] = request
    
    # Render template
    return templates.get_template(template_name).render(**context)

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

def min_filter(value1, value2):
    """Return the minimum of two values."""
    try:
        return min(float(value1), float(value2))
    except (ValueError, TypeError):
        return value1

def max_filter(value1, value2):
    """Return the maximum of two values."""
    try:
        return max(float(value1), float(value2))
    except (ValueError, TypeError):
        return value1

# Register filters with Jinja2 environment
templates.env.filters["tojson"] = tojson_filter
templates.env.filters["round"] = round_filter
templates.env.filters["int"] = int_filter
templates.env.filters["svg_initials"] = svg_initials_filter
templates.env.filters["min"] = min_filter
templates.env.filters["max"] = max_filter
