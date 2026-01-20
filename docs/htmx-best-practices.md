# HTMX Best Practices Implementation

This document tracks the implementation of HTMX best practices across all HTML generation in the codebase.

## Core Principles

1. **Zero f-string HTML**: All HTML comes from Jinja2 templates
2. **HTMX attributes only**: No onclick, no fetch(), no addEventListener for server interactions
3. **Server-rendered HTML**: Server returns HTML fragments, not JSON
4. **hx-swap-oob for multi-updates**: One response updates multiple elements
5. **Progressive enhancement**: Works without JavaScript, enhanced with HTMX
6. **Locality of behavior**: See what a button does by reading its HTML attributes

## HTMX Patterns Used

### 1. Buttons with Server Actions
```jinja2
<button 
    hx-post="/api/analyze"
    hx-vals='{{ {"ticker": symbol} | tojson }}'
    hx-target="#analysis-results"
    hx-swap="beforeend">
    Analyze
</button>
```

### 2. Forms with HTMX
```jinja2
<form 
    hx-post="/api/endpoint"
    hx-target="#result"
    hx-swap="innerHTML">
    <input name="field" value="{{ value }}">
    <button type="submit">Submit</button>
</form>
```

### 3. Multi-Element Updates
```jinja2
{# Server response includes multiple hx-swap-oob elements #}
<div id="positions-list" hx-swap-oob="innerHTML">
    {{ positions_html }}
</div>
<div id="toast-container" hx-swap-oob="beforeend">
    {{ toast_html }}
</div>
```

### 4. Confirmation Dialogs
```jinja2
<button 
    hx-post="/api/quick-sell"
    hx-vals='{{ {"symbol": symbol} | tojson }}'
    hx-confirm="Close position for {{ symbol }}?">
    SELL
</button>
```

### 5. Polling with Visibility Check
```jinja2
<div 
    hx-get="/api/balance" 
    hx-trigger="load, every 10s[document.visibilityState === 'visible']" 
    hx-swap="innerHTML">
    Loading...
</div>
```

## Template Structure

```
templates/
├── pages/           # Full page responses
│   ├── account_balance.html
│   ├── analysis_result.html
│   ├── analysis_no_ai.html
│   ├── strategy_config.html
│   └── ...
├── components/      # Reusable components
│   ├── position_card.html
│   ├── stock_card.html (TODO)
│   └── ...
├── partials/        # Small HTML fragments
│   ├── error_message.html
│   ├── status_message.html
│   └── ...
└── scripts/         # JavaScript snippets
    ├── htmx_trigger.html
    ├── tradingview_widget.html
    └── ...
```

## Conversion Status

### ✅ Completed

1. **Error Responses**: All error messages use `partials/error_message.html`
2. **Account Balance**: Uses `pages/account_balance.html`
3. **Analysis Result**: Uses `pages/analysis_result.html` with proper HTMX attributes
4. **Strategy Config**: Already using `pages/strategy_config.html`
5. **Templates Infrastructure**: Autoescape enabled, `tojson` filter added

### ⚠️ In Progress

1. **stock_card()**: Large function (~400 lines) needs conversion to `components/stock_card.html`
2. **get_watch_list()**: Partially templated, needs full conversion
3. **analyze_rejection()**: Needs template conversion
4. **get_explanation()**: Large f-string needs template
5. **get_strategy_display_html()**: Needs template conversion

### 🔴 Remaining Violations

1. **onclick handlers**: 12 locations still using onclick instead of HTMX/Alpine.js
   - Modal buttons: Can use Alpine.js `@click` (UI-only)
   - Toast close buttons: Should use Alpine.js `@click`
   - Form submissions: Should use HTMX `hx-post`

2. **f-string HTML**: ~15 locations still generating HTML with f-strings

## HTMX Attribute Reference

| Attribute | Purpose | Example |
|-----------|---------|---------|
| `hx-get` | GET request | `hx-get="/api/data"` |
| `hx-post` | POST request | `hx-post="/api/analyze"` |
| `hx-vals` | JSON data | `hx-vals='{{ data \| tojson }}'` |
| `hx-target` | Target element | `hx-target="#results"` |
| `hx-swap` | Swap strategy | `hx-swap="innerHTML"` |
| `hx-swap-oob` | Out-of-band swap | `hx-swap-oob="beforeend"` |
| `hx-trigger` | Trigger events | `hx-trigger="click, every 5s"` |
| `hx-confirm` | Confirmation | `hx-confirm="Are you sure?"` |

## Best Practices Checklist

- [x] All HTML from templates (not f-strings)
- [x] Autoescape enabled for security
- [x] `tojson` filter for JSON in HTML attributes
- [x] Server returns HTML fragments
- [x] Multi-updates use `hx-swap-oob`
- [ ] All onclick handlers replaced with HTMX/Alpine.js
- [ ] All forms use HTMX
- [ ] All buttons use HTMX for server actions
- [ ] Progressive enhancement (works without JS)
- [ ] No JavaScript syntax errors

## Security Considerations

1. **Autoescape**: Enabled by default, prevents XSS
2. **tojson filter**: Properly escapes JSON for HTML attributes
3. **Template isolation**: Templates can't execute arbitrary code
4. **Server-side validation**: All inputs validated on server

## Performance Considerations

1. **Server-rendered HTML**: Faster initial load
2. **Minimal JavaScript**: Less client-side processing
3. **Selective updates**: Only update changed elements
4. **Visibility checks**: Don't poll when tab is hidden
