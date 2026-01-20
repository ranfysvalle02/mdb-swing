# Proper Template Architecture - Staff Engineer Approach

## The Problem

Mixing f-strings with HTML/JavaScript generation is fundamentally flawed:
- **No escaping**: Manual escaping is error-prone
- **Syntax errors**: Curly braces conflict with f-string syntax
- **Unmaintainable**: HTML mixed with Python logic
- **Security risks**: XSS vulnerabilities from improper escaping
- **Testing**: Can't test templates independently

## The Solution: Pure Jinja2 Templates

### Architecture Pattern

```
Python Route (Logic Only)
    ↓
Prepare Data Structure
    ↓
Render Jinja2 Template
    ↓
Return HTMLResponse
```

### Key Principles

1. **Zero f-strings for HTML**: All HTML comes from templates
2. **Data, not HTML**: Routes prepare data, templates render HTML
3. **Autoescape enabled**: Jinja2 handles XSS prevention
4. **Template includes**: Reusable components via `{% include %}`
5. **Custom filters**: Use `tojson` filter for JSON in attributes

### Example: Before vs After

**Before (Hacky)**:
```python
return HTMLResponse(content=f"""
    <button hx-vals='{{"ticker": "{symbol}"}}'>Buy</button>
    <script>
        function doSomething() {{
            console.log("{value}");
        }}
    </script>
""")
```

**After (Proper)**:
```python
# Route prepares data
return HTMLResponse(content=templates.get_template("pages/buy_button.html").render(
    symbol=symbol,
    value=value
))
```

```jinja2
{# templates/pages/buy_button.html #}
<button hx-vals='{{ {"ticker": symbol} | tojson }}'>Buy</button>
<script>
    function doSomething() {
        console.log("{{ value|e }}");
    }
</script>
```

### Benefits

1. **No syntax errors**: Jinja2 handles all escaping
2. **Security**: Autoescape prevents XSS
3. **Maintainability**: HTML in templates, logic in Python
4. **Testability**: Can test templates independently
5. **Reusability**: Components via includes
6. **Type safety**: Template errors caught at render time

### Migration Strategy

1. **Start with new routes**: Use templates from day one
2. **Convert incrementally**: One route at a time
3. **Extract components**: Common HTML → template includes
4. **Add filters**: Custom Jinja2 filters for common patterns

### Current Status

✅ **Converted to Templates**:
- Strategy config form (`pages/strategy_config.html`)
- Script includes (`scripts/*.html`)
- Component templates (`components/*.html`)

⚠️ **Still Using f-strings** (needs conversion):
- `stock_card()` function in `templates.py`
- `analyze_symbol()` route in `routes.py`
- `get_watch_list()` route in `routes.py`
- Other large HTML responses

### Next Steps

1. Convert `stock_card()` to `components/stock_card.html`
2. Convert `analyze_symbol()` to `pages/analysis.html`
3. Convert `get_watch_list()` to `pages/watchlist.html`
4. Extract common patterns to includes
5. Add more custom filters as needed
