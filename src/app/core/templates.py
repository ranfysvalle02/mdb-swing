"""Jinja2 Templates Configuration."""
import json
from pathlib import Path
from fastapi import Request
from fastapi.templating import Jinja2Templates

templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))
templates.env.autoescape = True

def render_template(template_name: str, request: Request, **context) -> str:
    """Render a template with CSRF token automatically included."""
    csrf_token = getattr(request.state, "csrf_token", None)
    if not csrf_token:
        from .security import get_csrf_token_for_template
        csrf_token = get_csrf_token_for_template(request)
        request.state.csrf_token = csrf_token
    
    context["csrf_token"] = csrf_token
    context["request"] = request
    return templates.get_template(template_name).render(**context)

try:
    from ..services.logo import get_svg_initials
    LOGO_SERVICE_AVAILABLE = True
except ImportError:
    LOGO_SERVICE_AVAILABLE = False
    def get_svg_initials(ticker: str) -> str:
        return f'<svg width="40" height="40" viewBox="0 0 40 40"><text x="20" y="20">?</text></svg>'
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

templates.env.filters["tojson"] = tojson_filter
templates.env.filters["round"] = round_filter
templates.env.filters["int"] = int_filter
templates.env.filters["svg_initials"] = svg_initials_filter
templates.env.filters["min"] = min_filter
templates.env.filters["max"] = max_filter
