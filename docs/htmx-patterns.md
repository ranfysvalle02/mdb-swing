# HTMX Patterns in Sauron's Eye

This document showcases the HTMX patterns used throughout Sauron's Eye, demonstrating how to build interactive web applications with minimal JavaScript.

## Table of Contents

1. [Basic Patterns](#basic-patterns)
2. [Advanced Patterns](#advanced-patterns)
3. [Real-World Examples](#real-world-examples)
4. [Before/After Comparisons](#beforeafter-comparisons)

---

## Basic Patterns

### 1. GET Requests with Auto-Refresh

**Pattern**: Polling with `hx-trigger="every Xs"`

```html
<!-- Account balance refreshes every 10 seconds -->
<div id="account-info" 
     hx-get="/api/balance" 
     hx-trigger="load, every 10s" 
     hx-swap="innerHTML">
    <span class="animate-pulse">Connecting...</span>
</div>
```

**What Happens**:
1. On page load (`load`), fetches balance
2. Every 10 seconds (`every 10s`), fetches balance again
3. Server returns HTML fragment
4. HTMX swaps it into `#account-info`

**Benefits**:
- No JavaScript needed
- Automatic refresh
- Server controls the HTML format

**Used In**: Account balance, positions list, trade logs

---

### 2. POST Requests with Form Data

**Pattern**: `hx-post` with `hx-vals` for form data

```html
<!-- Buy button sends POST with symbol -->
<button hx-post="/api/quick-buy"
        hx-vals='{"symbol": "AAPL"}'
        hx-target="#positions-list"
        hx-swap="innerHTML">
    BUY
</button>
```

**What Happens**:
1. User clicks button
2. HTMX sends POST to `/api/quick-buy` with `symbol=AAPL`
3. Server processes order and returns HTML
4. HTMX swaps response into `#positions-list`

**Benefits**:
- No form element needed
- No JavaScript event handlers
- Server returns updated HTML

**Used In**: Quick buy/sell buttons, watchlist add/remove

---

### 3. Confirmation Dialogs

**Pattern**: `hx-confirm` for user confirmation

```html
<button hx-post="/api/quick-sell"
        hx-vals='{"symbol": "AAPL"}'
        hx-confirm="Close position for AAPL?">
    SELL
</button>
```

**What Happens**:
1. User clicks button
2. Browser shows confirm dialog
3. If confirmed, POST request is sent
4. If cancelled, request is not sent

**Benefits**:
- Built-in browser confirm
- No custom JavaScript needed
- Works without JavaScript (falls back to form submit)

**Used In**: Sell buttons, cancel orders

---

### 4. Lazy Loading

**Pattern**: `hx-trigger="revealed"` for on-demand loading

```html
<!-- Content loads when modal is opened -->
<div id="discovery-content" 
     hx-get="/api/discover-stocks" 
     hx-trigger="revealed" 
     hx-swap="innerHTML">
    <div class="text-center py-8">
        <i class="fas fa-spinner fa-spin"></i>
        <p>Loading...</p>
    </div>
</div>
```

**What Happens**:
1. Element is hidden initially
2. When element becomes visible (`revealed`), request is sent
3. Server returns HTML
4. HTMX swaps it in

**Benefits**:
- Content loads on-demand
- Better initial page load performance
- No manual visibility detection needed

**Used In**: Discovery modal, strategy config modal

---

## Advanced Patterns

### 5. Out-of-Band Swaps (`hx-swap-oob`)

**Pattern**: Update multiple elements from one response

**Problem**: When you buy a stock, you need to:
- Update the positions list
- Show a success toast notification
- Update the account balance

**Solution**: Server returns HTML with `hx-swap-oob` elements

**Frontend**:
```html
<div id="positions-list"></div>
<div id="toast-container"></div>
```

**Backend Response**:
```python
return HTMLResponse(content=f"""
    <div id="positions-list" hx-swap-oob="innerHTML">
        {positions_html}
    </div>
    <div id="toast-container" hx-swap-oob="beforeend">
        {toast_html}
    </div>
""")
```

**What Happens**:
1. HTMX receives response
2. Finds elements with `hx-swap-oob="true"`
3. Updates `#positions-list` with new HTML
4. Appends toast to `#toast-container`
5. All from one request!

**Benefits**:
- One request updates multiple elements
- No JavaScript coordination needed
- Server controls what updates

**Used In**: Quick buy/sell (updates positions + toast), watchlist add (updates list + button)

---

### 6. Button State Updates

**Pattern**: Update button after action using `hx-swap-oob`

**Problem**: When you add a stock to watchlist, the button should change to "In Watchlist" and be disabled.

**Solution**: Server returns button update with `hx-swap-oob`

**Initial Button**:
```html
<button id="watchlist-btn-AAPL"
        hx-post="/api/watchlist/add"
        hx-vals='{"symbol": "AAPL", "name": "Apple Inc."}'>
    Add to Watchlist
</button>
```

**Backend Response**:
```python
# Returns watchlist HTML + button update
return HTMLResponse(content=f"""
    <div id="watchlist-display" hx-swap-oob="innerHTML">
        {watchlist_html}
    </div>
    <button id="watchlist-btn-{symbol}" hx-swap-oob="true" disabled>
        <i class="fas fa-check"></i> In Watchlist
    </button>
""")
```

**What Happens**:
1. User clicks "Add to Watchlist"
2. Server adds stock and returns HTML
3. HTMX updates watchlist display
4. HTMX replaces button with disabled "In Watchlist" version
5. Button is now disabled and shows correct state

**Benefits**:
- Immediate visual feedback
- No JavaScript state management
- Server is source of truth

**Used In**: Watchlist add/remove buttons

---

### 7. Modal Content Loading

**Pattern**: Load modal content with HTMX

**Problem**: Explanation modal needs to load detailed analysis when opened.

**Solution**: Button opens modal and triggers HTMX request

**Button**:
```html
<button hx-post="/api/explanation"
        hx-vals='{"symbol": "AAPL"}'
        hx-target="#explanation-content"
        hx-swap="innerHTML"
        onclick="openModal('explanation-modal');">
    EXPLAIN
</button>
```

**Modal**:
```html
<div id="explanation-modal" class="modal">
    <div class="modal-content">
        <h2 id="explanation-symbol">Detailed Explanation</h2>
        <div id="explanation-content">
            <!-- HTMX loads content here -->
        </div>
    </div>
</div>
```

**What Happens**:
1. User clicks "EXPLAIN"
2. `onclick` opens modal (minimal JS for UI)
3. HTMX sends POST to `/api/explanation`
4. Server returns HTML explanation
5. HTMX swaps it into `#explanation-content`

**Benefits**:
- Content loads on-demand
- Server controls HTML format
- No JavaScript for data fetching

**Used In**: Explanation modal, strategy config modal

---

## Real-World Examples

### Example 1: Positions List Auto-Refresh

**Location**: `frontend/index.html`

```html
<!-- Positions refresh every 5 seconds -->
<div id="positions-list" 
     hx-get="/api/positions" 
     hx-trigger="load, every 5s" 
     hx-swap="innerHTML">
</div>
```

**Backend**: `src/app/api/routes.py`

```python
async def get_positions() -> HTMLResponse:
    """Returns HTML with htmx buttons for closing positions."""
    pos = api.list_positions()
    html = ""
    for p in pos:
        html += f"""
        <div class="position-card">
            <span>{p.symbol}</span>
            <button hx-post="/api/quick-sell"
                    hx-vals='{{"symbol": "{p.symbol}"}}'
                    hx-target="#positions-list"
                    hx-confirm="Close position?">
                CLOSE
            </button>
        </div>
        """
    return HTMLResponse(content=html)
```

**Flow**:
1. Page loads → positions fetched
2. Every 5s → positions refreshed
3. User clicks CLOSE → POST sent
4. Server closes position → returns updated positions HTML
5. HTMX swaps in new HTML

---

### Example 2: Watchlist Management

**Location**: Discovery modal

**Add to Watchlist**:
```html
<button id="watchlist-btn-AAPL"
        hx-post="/api/watchlist/add"
        hx-vals='{"symbol": "AAPL", "name": "Apple Inc."}'
        hx-target="#watchlist-display"
        hx-swap="innerHTML">
    Add to Watchlist
</button>
```

**Backend Response**:
```python
async def add_to_watchlist(...) -> HTMLResponse:
    # Add to database
    await db.watchlist.update_one(...)
    
    # Get updated watchlist HTML
    watchlist_html = await get_watchlist_html(db)
    
    # Return both watchlist + button update
    return HTMLResponse(content=f"""
        <div id="watchlist-display" hx-swap-oob="innerHTML">
            {watchlist_html}
        </div>
        <button id="watchlist-btn-{symbol}" hx-swap-oob="true" disabled>
            <i class="fas fa-check"></i> In Watchlist
        </button>
    """)
```

**Flow**:
1. User clicks "Add to Watchlist"
2. Server adds stock to MongoDB
3. Server returns watchlist HTML + button update
4. HTMX updates watchlist display
5. HTMX replaces button with disabled version

---

### Example 3: Quick Buy with Multiple Updates

**Location**: Stock cards in "The Eye" section

**Button**:
```html
<button hx-post="/api/quick-buy"
        hx-vals='{"symbol": "AAPL"}'
        hx-target="#positions-list"
        hx-swap="innerHTML">
    BUY
</button>
```

**Backend Response**:
```python
async def quick_buy(...) -> HTMLResponse:
    # Place order
    order = await place_order(symbol, 'buy', db)
    
    # Get updated positions
    positions_html = await get_positions_html()
    
    # Return positions + toast
    return HTMLResponse(content=f"""
        <div id="positions-list" hx-swap-oob="innerHTML">
            {positions_html}
        </div>
        <div id="toast-container" hx-swap-oob="beforeend">
            <div class="toast success">
                Buy order placed for {symbol}
            </div>
        </div>
    """)
```

**Flow**:
1. User clicks BUY
2. Server places order via Alpaca API
3. Server returns positions HTML + toast HTML
4. HTMX updates positions list
5. HTMX appends toast notification
6. User sees updated positions and success message

---

## Before/After Comparisons

### Before: JavaScript Fetch

```javascript
async function quickBuy(symbol, event) {
    const btn = event.target.closest('button');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    
    try {
        const formData = new URLSearchParams();
        formData.append('symbol', symbol);
        
        const response = await fetch('/api/quick-buy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update positions
            await refreshPositions();
            // Show toast
            showToast('Order placed!');
            // Reset button
            btn.innerHTML = 'BUY';
            btn.disabled = false;
        } else {
            showError(data.error);
        }
    } catch (err) {
        console.error('Error:', err);
        showError('Failed to place order');
    } finally {
        btn.disabled = false;
    }
}
```

**Lines of Code**: ~35 lines  
**Complexity**: High (error handling, state management)  
**Maintainability**: Medium (scattered logic)

### After: HTMX

```html
<button hx-post="/api/quick-buy"
        hx-vals='{"symbol": "AAPL"}'
        hx-target="#positions-list"
        hx-swap="innerHTML">
    BUY
</button>
```

**Lines of Code**: 4 lines  
**Complexity**: Low (declarative)  
**Maintainability**: High (locality of behavior)

**Reduction**: 88% less code, 100% less JavaScript

---

### Before: Manual Polling

```javascript
function startPolling() {
    setInterval(async () => {
        try {
            const response = await fetch('/api/balance');
            const data = await response.json();
            document.getElementById('account-info').innerHTML = 
                `$${data.balance}`;
        } catch (err) {
            console.error('Polling error:', err);
        }
    }, 10000);
}

// Don't forget to clean up!
window.addEventListener('beforeunload', () => {
    clearInterval(pollingInterval);
});
```

**Lines of Code**: ~15 lines  
**Complexity**: Medium (cleanup needed)  
**Memory Leaks**: Possible if not cleaned up

### After: HTMX Polling

```html
<div id="account-info" 
     hx-get="/api/balance" 
     hx-trigger="load, every 10s" 
     hx-swap="innerHTML">
</div>
```

**Lines of Code**: 4 lines  
**Complexity**: Low (automatic cleanup)  
**Memory Leaks**: None (HTMX handles cleanup)

**Reduction**: 73% less code, zero memory leak risk

---

## Key Takeaways

1. **HTMX reduces JavaScript by 80%+**: Most interactions don't need custom JS
2. **Server controls HTML**: Backend returns HTML fragments, not JSON
3. **Progressive Enhancement**: Works without JavaScript, enhanced with HTMX
4. **Locality of Behavior**: See what a button does by reading its HTML
5. **Less State Management**: Server is source of truth, not client

## Further Reading

- [HTMX Documentation](https://htmx.org/docs/)
- [HTMX Examples](https://htmx.org/examples/)
- [HTMX Philosophy](https://htmx.org/essays/locality-of-behaviour/)
