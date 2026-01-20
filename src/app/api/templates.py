"""HTML template helpers for HTMX responses using Jinja2.

MDB-Engine Pattern: Separate presentation logic from routes.
These helpers generate clean, reusable HTML components for HTMX responses.

Best Practice: Using Jinja2 templates instead of f-strings for:
- Better separation of concerns
- Cleaner template syntax
- Easier maintenance
- Better support for HTMX patterns
"""
import json
import re
import time
from typing import Optional, Dict, Any
from fastapi.responses import HTMLResponse
from ..core.templates import templates
from ..services.positions import PositionMetrics, SellSignal


def empty_positions() -> str:
    """Empty state for positions list."""
    return templates.get_template("components/empty_positions.html").render()


def pending_order_card(
    symbol: str,
    qty: int,
    side: str,
    order_type: str,
    limit_price: Optional[float],
    order_id: str,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None
) -> str:
    """Generate HTML for a pending order card."""
    return templates.get_template("components/pending_order_card.html").render(
        symbol=symbol,
        qty=qty,
        side=side,
        order_type=order_type,
        limit_price=limit_price,
        order_id=order_id,
        stop_loss=stop_loss,
        take_profit=take_profit
    )


def progress_bar(
    label: str,
    icon: str,
    progress_pct: float,
    distance_display: str,
    entry_price: float,
    target_price: float,
    target_label: str,
    color: str,
    warning_class: str = "",
    warning_msg: str = ""
) -> str:
    """Generate a progress bar component.
    
    HTMX Security Best Practice: Sanitize warning_msg before using |safe filter.
    """
    # Sanitize warning_msg since it's rendered with |safe in template
    sanitized_warning_msg = _sanitize_html_for_htmx(warning_msg) if warning_msg else ""
    
    return templates.get_template("components/progress_bar.html").render(
        label=label,
        icon=icon,
        progress_pct=progress_pct,
        distance_display=distance_display,
        entry_price=entry_price,
        target_price=target_price,
        target_label=target_label,
        color=color,
        warning_class=warning_class,
        warning_msg=sanitized_warning_msg
    )


def position_card(
    symbol: str,
    metrics: PositionMetrics,
    sell_signal: Optional[SellSignal] = None
) -> str:
    """Generate HTML for a position card with progress indicators."""
    from ..services.logo import get_svg_initials
    svg_initials = get_svg_initials(symbol)
    return templates.get_template("components/position_card.html").render(
        symbol=symbol,
        metrics=metrics,
        sell_signal=sell_signal,
        logo_svg=svg_initials
    )


def lucide_init_script() -> str:
    """Generate script tag to initialize Lucide icons."""
    return templates.get_template("scripts/lucide_init.html").render()


def toast_notification(
    message: str,
    type: str = "info",
    duration: int = 3000,
    target_id: str = "toast-container",
    undo_action: Optional[str] = None,
    undo_symbol: Optional[str] = None
) -> str:
    """Generate toast notification HTML with hx-swap-oob support.
    
    HTMX Security Best Practice: This function returns HTML with data attributes only.
    - NO inline scripts (prevents XSS attacks)
    - Uses Alpine.js x-data for initialization (already in DOM, not injected)
    - All user input is escaped via template filters (|e)
    - Uses data attributes to pass data safely
    
    The toast is initialized by Alpine.js event handlers listening for htmx:afterSwap events.
    
    Args:
        message: Toast message text
        type: Toast type (success, info, warning, error)
        duration: Auto-dismiss duration in milliseconds
        target_id: Target container ID for hx-swap-oob
        undo_action: Optional HTMX action URL for undo button
        undo_symbol: Optional symbol to pass to undo action
    """
    toast_id = f"toast-{int(time.time() * 1000000)}"
    
    return templates.get_template("components/toast_notification.html").render(
        toast_id=toast_id,
        message=message,
        type=type,
        duration=duration,
        target_id=target_id,
        undo_action=undo_action,
        undo_symbol=undo_symbol
    )


def _sanitize_html_for_htmx(html_content: str) -> str:
    """Sanitize HTML content to ensure it's valid for HTMX parsing.
    
    Fixes common issues that cause HTMX insertBefore errors:
    - Escapes standalone & characters that aren't part of valid HTML entities
    - Preserves existing HTML entities (like &amp;, &lt;, &quot;, etc.)
    
    Args:
        html_content: HTML string that may contain unescaped entities
        
    Returns:
        Sanitized HTML string safe for HTMX parsing
    """
    if not html_content:
        return html_content
    
    # Pattern to escape & characters that are NOT part of valid HTML entities
    # Valid entities: &name; or &#number; or &#xhex;
    # This regex uses negative lookahead to avoid matching valid entities
    # It matches & that's NOT followed by:
    #   - A letter (start of named entity like &amp;)
    #   - # followed by digits (numeric entity like &#123;)
    #   - # followed by x/X and hex digits (hex entity like &#x1F;)
    #   - And ending with ;
    result = re.sub(
        r'&(?!(?:[a-zA-Z][a-zA-Z0-9]{1,31}|#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6});)',
        '&amp;',
        html_content
    )
    
    return result


def error_response(
    message: str,
    status_code: int = 500,
    target_id: Optional[str] = None,
    hx_trigger: Optional[str] = None
) -> HTMLResponse:
    """Standardized error response for HTMX.
    
    HTMX Gold Standard: Consistent error handling across all routes.
    
    Args:
        message: Error message to display
        status_code: HTTP status code (default: 500)
        target_id: Optional element ID to target with hx-swap-oob
        hx_trigger: Optional HTMX event to trigger
        
    Returns:
        HTMLResponse with error message
    """
    error_html = templates.get_template("partials/error_message.html").render(
        message=message
    )
    
    if target_id:
        # Use htmx_response for multi-element updates
        content = htmx_response(updates={target_id: error_html})
        response = HTMLResponse(content=content, status_code=status_code)
    else:
        response = HTMLResponse(content=error_html, status_code=status_code)
    
    if hx_trigger:
        response.headers["HX-Trigger"] = hx_trigger
    
    return response


def htmx_html_response(
    content: str,
    status_code: int = 200,
    hx_trigger: Optional[str] = None,
    sanitize: bool = True
) -> HTMLResponse:
    """Wrapper for HTMX HTML responses with automatic sanitization.
    
    HTMX Gold Standard: All HTML responses go through sanitization.
    
    Args:
        content: HTML content to return
        status_code: HTTP status code (default: 200)
        hx_trigger: Optional HTMX event to trigger via header
        sanitize: Whether to sanitize content (default: True)
        
    Returns:
        HTMLResponse with sanitized content
    """
    if sanitize:
        content = _sanitize_html_for_htmx(content)
    
    response = HTMLResponse(content=content, status_code=status_code)
    
    if hx_trigger:
        response.headers["HX-Trigger"] = hx_trigger
    
    return response


def htmx_response(
    updates: Dict[str, str]
) -> str:
    """Generate HTMX response with multiple hx-swap-oob updates.
    
    HTMX Security Best Practice: This function returns HTML only - no inline scripts.
    Use HX-Trigger response headers to trigger events instead of inline scripts.
    
    MDB-Engine Pattern: Standardize multi-element HTMX responses.
    This helper generates HTML with multiple hx-swap-oob elements,
    allowing a single response to update multiple parts of the page.
    
    Args:
        updates: Dict mapping element IDs to HTML content.
                 Keys are element IDs, values are HTML strings.
                 For toast-container, use "beforeend" strategy.
        
    Returns:
        HTML string with multiple hx-swap-oob elements.
        
    Example:
        response_content = htmx_response(updates={
            "positions-list": positions_html,
            "toast-container": toast_html
        })
        response = HTMLResponse(content=response_content)
        response.headers["HX-Trigger"] = "refreshPositions"  # Set trigger via header
        return response
    """
    # Prepare updates data for template
    updates_data = {}
    for element_id, content in updates.items():
        # Sanitize content to ensure valid HTML for HTMX parsing
        content = _sanitize_html_for_htmx(content)
        
        if element_id == "toast-container":
            # toast_notification already includes hx-swap-oob="beforeend"
            updates_data[element_id] = {
                "content": content,
                "swap_strategy": "beforeend"
            }
        else:
            # Determine swap strategy based on content
            # If content starts with a div with the same ID, use outerHTML
            # Otherwise, use innerHTML for content replacement
            content_stripped = content.strip()
            if content_stripped.startswith(f'<div id="{element_id}"') or content_stripped.startswith(f"<div id='{element_id}'"):
                # Content already has wrapper div with ID - add hx-swap-oob to it
                # Find the opening div tag and add hx-swap-oob attribute
                pattern = rf'(<div\s+id=["\']{re.escape(element_id)}["\'])([^>]*>)'
                replacement = r'\1 hx-swap-oob="outerHTML"\2'
                content = re.sub(pattern, replacement, content, count=1)
                updates_data[element_id] = {
                    "content": content,
                    "swap_strategy": "outerHTML"
                }
            else:
                # Content doesn't have wrapper - wrap it
                swap_strategy = "innerHTML"
                updates_data[element_id] = {
                    "content": content,
                    "swap_strategy": swap_strategy
                }
    
    # Render template (no trigger - use HX-Trigger response headers instead)
    return templates.get_template("partials/htmx_swap_oob.html").render(
        updates=updates_data
    )


def htmx_modal_wrapper(
    modal_id: str,
    title: str,
    content: str,
    size: str = "medium",
    icon: Optional[str] = None,
    icon_color: str = "text-purple-400"
) -> str:
    """Generate HTMX modal wrapper HTML with native dialog element.
    
    Clean HTMX Pattern: Uses native <dialog> with CSS3 animations.
    No Alpine.js, no Hyperscript - just pure CSS3 + HTMX magic.
    
    HTMX Security Best Practice: Sanitize content before using |safe filter.
    """
    size_classes = {
        "small": "max-w-md",
        "medium": "max-w-2xl",
        "large": "max-w-4xl",
        "xl": "max-w-6xl",
        "full": "max-w-95vw"
    }
    max_width = size_classes.get(size, size_classes["medium"])
    
    # Sanitize content since it's rendered with |safe in template
    sanitized_content = _sanitize_html_for_htmx(content)
    
    return templates.get_template("components/htmx_modal_wrapper.html").render(
        modal_id=modal_id,
        title=title,
        content=sanitized_content,
        max_width=max_width,
        icon=icon,
        icon_color=icon_color
    )


def stock_card(stock_data: Dict[str, Any]) -> str:
    """Generate stock card HTML from stock data.
    
    MDB-Engine Pattern: Server-rendered components for better performance.
    This replaces the client-side createStockCard() JavaScript function.
    
    Uses Jinja2 template for proper HTMX best practices.
    """
    symbol = stock_data.get('symbol', '')
    price = stock_data.get('price')
    rsi = stock_data.get('rsi')
    atr = stock_data.get('atr', 0)
    trend = stock_data.get('trend', '')
    sma = stock_data.get('sma', 0)
    ai_score = stock_data.get('ai_score')
    ai_reason = stock_data.get('ai_reason', '')
    risk_level = stock_data.get('risk_level', 'medium')
    has_limited_data = stock_data.get('has_limited_data', False) or (price and not ai_score and not rsi)
    cache_hit = stock_data.get('cache_hit', False)
    similar_signals = stock_data.get('similar_signals', [])
    confidence_boost = stock_data.get('confidence_boost', 0.0)
    win_rate = stock_data.get('win_rate', 0.0)
    similar_count = stock_data.get('similar_count', 0)
    meets_criteria = stock_data.get('meets_criteria', False)
    
    # Calculate profit potential and risk metrics
    stop_loss = price and atr > 0 and (price - (2 * atr)) or None
    take_profit = price and atr > 0 and (price + (3 * atr)) or None
    profit_potential = take_profit and price and (take_profit - price) or None
    profit_potential_pct = profit_potential and price and ((profit_potential / price) * 100) or None
    risk_amount = stop_loss and price and (price - stop_loss) or None
    risk_reward_ratio = profit_potential and risk_amount and risk_amount > 0 and round(profit_potential / risk_amount, 2) or None
    
    # Determine score styling
    score_color = "text-gray-400"
    score_badge = "badge-info"
    score_bg = "bg-gray-500/20"
    score_border = "border-gray-500/30"
    score_icon = "info"
    score_glow = ""
    overall_signal = "neutral"
    description = ai_reason.strip() if ai_reason and ai_reason.strip() else "Trending stock - check analysis"
    
    if ai_score is not None:
        if ai_score >= 9:
            score_color = "text-emerald-400"
            score_badge = "badge-success"
            score_bg = "bg-emerald-500/25"
            score_border = "border-emerald-500/50"
            score_icon = "star"
            score_glow = "shadow-lg shadow-emerald-500/30"
            description = ai_reason.strip() if ai_reason and ai_reason.strip() else "Excellent Low Buy - Strong Bounce Signals"
            overall_signal = "excellent"
        elif ai_score >= 8:
            score_color = "text-green-400"
            score_badge = "badge-success"
            score_bg = "bg-green-500/20"
            score_border = "border-green-500/40"
            score_icon = "check-circle-2"
            score_glow = "shadow-md shadow-green-500/20" if meets_criteria else ""
            description = ai_reason.strip() if ai_reason and ai_reason.strip() else ("Low Buy Opportunity - Signals Suggest Bounce" if meets_criteria else "Strong Score - Monitor Entry")
            overall_signal = "strong"
        elif ai_score >= 7:
            score_color = "text-lime-400"
            score_badge = "badge-info"
            score_bg = "bg-lime-500/15"
            score_border = "border-lime-500/30"
            score_icon = "check"
            description = ai_reason.strip() if ai_reason and ai_reason.strip() else "Good Opportunity - Review Details"
            overall_signal = "good"
        elif ai_score >= 5:
            score_color = "text-yellow-400"
            score_badge = "badge-warning"
            score_bg = "bg-yellow-500/15"
            score_border = "border-yellow-500/30"
            score_icon = "alert-triangle"
            description = ai_reason.strip() if ai_reason and ai_reason.strip() else "Moderate Signal - Proceed with Caution"
            overall_signal = "moderate"
        elif ai_score >= 3:
            score_color = "text-orange-400"
            score_badge = "badge-warning"
            score_bg = "bg-orange-500/15"
            score_border = "border-orange-500/30"
            score_icon = "alert-circle"
            description = ai_reason.strip() if ai_reason and ai_reason.strip() else "Weak Signal - High Risk"
            overall_signal = "weak"
        else:
            score_color = "text-red-400"
            score_badge = "badge-danger"
            score_bg = "bg-red-500/15"
            score_border = "border-red-500/30"
            score_icon = "x-circle"
            description = ai_reason.strip() if ai_reason and ai_reason.strip() else "Poor Signal - Not Recommended"
            overall_signal = "poor"
    
    # Trend styling
    trend_color = "text-gray-400"
    trend_icon = "minus"
    trend_bg = "bg-gray-500/10"
    if trend == "UP":
        trend_color = "text-green-400"
        trend_icon = "trending-up"
        trend_bg = "bg-green-500/10"
    elif trend == "DOWN":
        trend_color = "text-red-400"
        trend_icon = "trending-down"
        trend_bg = "bg-red-500/10"
    
    # RSI styling
    rsi_status = "Neutral"
    rsi_color = "text-gray-400"
    rsi_icon = "equal"
    rsi_bg = "bg-gray-500/10"
    if rsi is not None:
        if rsi < 30:
            rsi_status = "Oversold"
            rsi_color = "text-green-400"
            rsi_icon = "arrow-down"
            rsi_bg = "bg-green-500/10"
        elif rsi > 70:
            rsi_status = "Overbought"
            rsi_color = "text-red-400"
            rsi_icon = "arrow-up"
            rsi_bg = "bg-red-500/10"
    
    # Border color based on signal
    border_class = "border-gray-500/30"
    if overall_signal == "excellent":
        border_class = "border-emerald-500/60 shadow-lg shadow-emerald-500/20"
    elif overall_signal == "strong":
        border_class = "border-green-500/50" + (" shadow-md shadow-green-500/15" if meets_criteria else "")
    elif overall_signal == "good":
        border_class = "border-lime-500/40"
    elif overall_signal == "moderate":
        border_class = "border-yellow-500/40"
    elif overall_signal == "weak":
        border_class = "border-orange-500/40"
    elif overall_signal == "poor":
        border_class = "border-red-500/40"
    
    # Calculate score label
    score_label = "Excellent" if ai_score and ai_score >= 9 else "Strong" if ai_score and ai_score >= 8 else "Good" if ai_score and ai_score >= 7 else "Moderate" if ai_score and ai_score >= 5 else "Weak" if ai_score and ai_score >= 3 else "Poor" if ai_score is not None else "N/A"
    
    # Render template with all calculated values
    return templates.get_template("components/stock_card.html").render(
        symbol=symbol,
        price=price,
        rsi=rsi,
        atr=atr,
        trend=trend,
        sma=sma,
        ai_score=ai_score,
        ai_reason=ai_reason,
        description=description,
        has_limited_data=has_limited_data,
        cache_hit=cache_hit,
        similar_signals=similar_signals,
        similar_count=similar_count,
        confidence_boost=confidence_boost,
        win_rate=win_rate,
        meets_criteria=meets_criteria,
        stop_loss=stop_loss,
        take_profit=take_profit,
        profit_potential=profit_potential,
        profit_potential_pct=profit_potential_pct,
        risk_amount=risk_amount,
        risk_reward_ratio=risk_reward_ratio,
        score_color=score_color,
        score_badge=score_badge,
        score_bg=score_bg,
        score_border=score_border,
        score_icon=score_icon,
        score_glow=score_glow,
        score_label=score_label,
        overall_signal=overall_signal,
        border_class=border_class,
        trend_color=trend_color,
        trend_icon=trend_icon,
        trend_bg=trend_bg,
        rsi_status=rsi_status,
        rsi_color=rsi_color,
        rsi_icon=rsi_icon,
        rsi_bg=rsi_bg
    )
