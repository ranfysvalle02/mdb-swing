# The HTMX + MDB-Engine Stack

## Overview

This project demonstrates a **gold standard** implementation of the **HTMX + MDB-Engine** stack—a powerful combination that enables rapid development of modern, interactive web applications with minimal JavaScript and maximum productivity.

## What is This Stack?

### HTMX: Declarative HTML-First Frontend

**HTMX** is a library that extends HTML to enable dynamic interactions without writing JavaScript. Instead of building Single Page Applications (SPAs) with complex state management, HTMX lets you:

- Make HTTP requests directly from HTML attributes (`hx-get`, `hx-post`, etc.)
- Swap HTML fragments into the DOM without JavaScript
- Handle complex interactions declaratively
- Maintain **locality of behavior**—see what an element does by reading its HTML

### MDB-Engine: Declarative MongoDB-First Backend

**MDB-Engine** is a framework that eliminates ~200 lines of FastAPI + MongoDB boilerplate by providing:

- Automatic FastAPI app creation with lifecycle management
- Declarative index management via `manifest.json`
- Scoped database access with dependency injection
- Built-in observability (logging, health checks, metrics)
- Vector search support for semantic similarity

## Why This Stack?

### The Problem with Traditional Approaches

**Traditional SPA Stack (React/Vue + REST API):**
- Complex state management (Redux, Vuex)
- Client-side routing
- Large JavaScript bundles
- JSON API endpoints
- Separate frontend/backend codebases
- Type mismatches between frontend and backend

**Traditional Backend Stack (FastAPI + MongoDB):**
- ~200 lines of boilerplate for connection management
- Manual index creation and migration scripts
- Scattered configuration
- Manual observability setup
- Complex dependency injection patterns

### The HTMX + MDB-Engine Solution

**HTMX eliminates frontend complexity:**
- **80% less JavaScript**—no state management, no routing, no complex logic
- **Server-rendered HTML**—faster initial load, better SEO
- **Progressive enhancement**—works without JavaScript, enhanced with HTMX
- **Locality of behavior**—HTML attributes show what elements do

**MDB-Engine eliminates backend boilerplate:**
- **~200 lines saved**—infrastructure handled automatically
- **Declarative indexes**—version-controlled in `manifest.json`
- **Scoped database access**—automatic connection lifecycle management
- **Production-ready**—observability built-in from day one

## Value Proposition

### 1. Rapid Development

**Before (Traditional Stack):**
```python
# Backend: 50+ lines for MongoDB setup
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
# ... connection pooling, error handling, lifecycle management ...

# Frontend: 100+ lines of JavaScript
async function buyStock(symbol) {
    const response = await fetch('/api/buy', {
        method: 'POST',
        body: JSON.stringify({symbol}),
        headers: {'Content-Type': 'application/json'}
    });
    const data = await response.json();
    updateUI(data);
    showToast('Order placed!');
    // ... error handling, loading states, etc.
}
```

**After (HTMX + MDB-Engine):**
```python
# Backend: 4 lines
from mdb_engine import MongoDBEngine
engine = MongoDBEngine(mongo_uri=MONGO_URI, db_name=MONGO_DB)
app = engine.create_app(slug="app", manifest=get_manifest_path(), title="App")
```

```html
<!-- Frontend: 1 line -->
<button hx-post="/api/buy" hx-vals='{"symbol": "AAPL"}' hx-target="#positions">Buy</button>
```

**Result:** Same functionality, 90% less code, easier to maintain.

### 2. Declarative Configuration

**Indexes in `manifest.json` (not scattered migration scripts):**
```json
{
  "managed_indexes": {
    "radar_cache": [
      {
        "type": "regular",
        "keys": {"symbol": 1, "strategy_id": 1},
        "name": "symbol_strategy_idx"
      },
      {
        "type": "regular",
        "keys": {"expires_at": 1},
        "name": "ttl_idx",
        "expireAfterSeconds": 3600
      }
    ],
    "radar_history": [
      {
        "type": "vectorSearch",
        "name": "vector_idx",
        "definition": {
          "fields": [{
            "type": "vector",
            "path": "embedding",
            "numDimensions": 1536,
            "similarity": "cosine"
          }]
        }
      }
    ]
  }
}
```

**HTMX attributes in HTML (not JavaScript):**
```html
<div hx-get="/api/balance" 
     hx-trigger="load, every 10s[document.visibilityState === 'visible']" 
     hx-swap="innerHTML">
    Loading...
</div>
```

### 3. Production-Ready from Day One

**MDB-Engine provides:**
- Structured logging with correlation IDs
- Health checks (`/health`)
- Metrics (`/metrics`)
- Automatic connection pooling
- Index management

**HTMX provides:**
- Built-in error handling
- Request/response lifecycle events
- Out-of-band swaps for multi-element updates
- Polling with visibility checks
- Confirmation dialogs

### 4. Type Safety and Maintainability

**Backend:**
- Type hints on all route handlers
- Scoped database access (no connection leaks)
- Declarative indexes (version-controlled)

**Frontend:**
- HTML-first (no JavaScript type mismatches)
- Server-rendered templates (type-safe on backend)
- Progressive enhancement (works without JS)

## Architecture Overview

### Request Flow

```
Browser → HTMX (hx-post) → FastAPI → MDB-Engine (get_scoped_db) → MongoDB
         ← HTML Fragment ← HTMLResponse ← Scoped DB ← Query Result
```

1. User clicks button with `hx-post="/api/buy"`
2. HTMX sends POST request to FastAPI
3. FastAPI route uses `Depends(get_scoped_db)` for database access
4. MDB-Engine provides scoped database connection
5. Route queries MongoDB (using indexes from `manifest.json`)
6. Route returns `HTMLResponse` with HTML fragment
7. HTMX swaps HTML into target element
8. If response includes `hx-swap-oob`, multiple elements update

### Code Example: Complete Flow

**Frontend (HTMX):**
```html
<button hx-post="/api/quick-buy"
        hx-vals='{"symbol": "AAPL"}'
        hx-target="#positions-list"
        hx-swap="innerHTML">
    BUY
</button>
```

**Backend (MDB-Engine + HTMX):**
```python
from mdb_engine.dependencies import get_scoped_db
from fastapi.responses import HTMLResponse

async def quick_buy(
    symbol: str = Form(...), 
    db: Any = Depends(get_scoped_db)
) -> HTMLResponse:
    """MDB-Engine: Scoped database access"""
    order = await place_order(symbol, 'buy', db)
    
    # Get updated positions (uses index from manifest.json)
    positions = await db.positions.find({}).to_list(10)
    
    # Return HTML with hx-swap-oob for multiple updates
    return HTMLResponse(content=templates.get_template("components/positions.html").render(
        positions=positions,
        toast_message="Order placed!"
    ))
```

**Result:** Clean, maintainable, type-safe code with minimal boilerplate.

## Extensibility

### Adding New Features

#### 1. New Route Handler

```python
async def new_feature(
    param: str = Form(...),
    db: Any = Depends(get_scoped_db)
) -> HTMLResponse:
    """New feature using HTMX + MDB-Engine patterns."""
    # Database access via scoped connection
    data = await db.collection.find({}).to_list(10)
    
    # Return HTML fragment for HTMX
    return HTMLResponse(content=templates.get_template("pages/feature.html").render(
        data=data
    ))
```

#### 2. New Database Index

Add to `config/manifest.json`:
```json
{
  "managed_indexes": {
    "new_collection": [
      {
        "type": "regular",
        "keys": {"field": 1},
        "name": "field_idx"
      }
    ]
  }
}
```

Index is automatically created on startup—no migration scripts needed.

#### 3. New HTMX Interaction

```html
<!-- Polling -->
<div hx-get="/api/data" 
     hx-trigger="every 5s" 
     hx-swap="innerHTML">
</div>

<!-- Form submission -->
<form hx-post="/api/submit" 
      hx-target="#result" 
      hx-swap="innerHTML">
    <input name="field" required>
    <button type="submit">Submit</button>
</form>

<!-- Multi-element update -->
<div id="result" hx-swap-oob="innerHTML">Updated</div>
<div id="toast" hx-swap-oob="beforeend">Notification</div>
```

### Extending Patterns

#### Custom HTMX Behaviors

```html
<!-- Confirmation dialog -->
<button hx-post="/api/delete"
        hx-confirm="Are you sure?">
    Delete
</button>

<!-- Loading state -->
<button hx-post="/api/action"
        hx-indicator="#spinner">
    Action
</button>
<div id="spinner" class="htmx-indicator">Loading...</div>

<!-- Error handling -->
<div hx-get="/api/data"
     hx-target="#content"
     hx-swap="innerHTML"
     hx-on::htmx:response-error="alert('Error occurred')">
</div>
```

#### Custom MDB-Engine Patterns

```python
# Custom health check
@app.get("/health/custom")
async def custom_health(db: Any = Depends(get_scoped_db)):
    """Custom health check using scoped database."""
    try:
        await db.collection.find_one({})
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# Custom metrics
@app.get("/metrics/custom")
async def custom_metrics(db: Any = Depends(get_scoped_db)):
    """Custom metrics using scoped database."""
    count = await db.collection.count_documents({})
    return {"collection_count": count}
```

## Best Practices

### HTMX Best Practices

1. **Zero onclick handlers**—use `@click` (Alpine.js) for UI-only interactions, HTMX for server actions
2. **Server-rendered HTML**—all HTML comes from Jinja2 templates, not f-strings
3. **Progressive enhancement**—works without JavaScript, enhanced with HTMX
4. **Locality of behavior**—see what an element does by reading its HTML attributes
5. **Out-of-band swaps**—use `hx-swap-oob` for multi-element updates
6. **Visibility checks**—don't poll when tab is hidden: `hx-trigger="every 10s[document.visibilityState === 'visible']"`
7. **Error handling**—return HTML error fragments, not JSON

### MDB-Engine Best Practices

1. **Scoped database access**—always use `Depends(get_scoped_db)`, never global connections
2. **Declarative indexes**—define all indexes in `manifest.json`, never in code
3. **Type hints**—add type hints to all route handlers and service functions
4. **Error handling**—use standardized error response helpers
5. **Observability**—use `get_logger(__name__)` for structured logging
6. **Vector search**—leverage vector search indexes for semantic similarity

### Security Best Practices

1. **CSRF protection**—enable CSRF validation for all POST/PUT/DELETE requests
2. **Template autoescape**—always enabled in Jinja2 templates
3. **Input validation**—validate all inputs using Pydantic models or Form validation
4. **Security headers**—use SecurityHeadersMiddleware for CSP, XSS protection, etc.

## Performance Optimization

### Database Query Optimization

1. **Use indexes**—ensure queries use indexes defined in `manifest.json`
2. **Limit results**—use `.limit()` and `.skip()` for pagination
3. **Project fields**—only fetch needed fields: `.find({}, {"field": 1})`
4. **Connection pooling**—handled automatically by MDB-Engine

### Template Optimization

1. **Minimize includes**—reduce nested template includes
2. **Cache templates**—Jinja2 caches compiled templates automatically
3. **Lazy loading**—use `hx-trigger="revealed"` for content that loads on-demand

### HTMX Optimization

1. **Selective updates**—only update changed elements
2. **Polling optimization**—use visibility checks to avoid unnecessary requests
3. **Request debouncing**—use `hx-trigger="click delay:500ms"` for debouncing

## Real-World Examples

### Example 1: Real-Time Dashboard

**Frontend:**
```html
<div id="balance" 
     hx-get="/api/balance" 
     hx-trigger="load, every 10s[document.visibilityState === 'visible']" 
     hx-swap="innerHTML">
    Loading...
</div>
```

**Backend:**
```python
async def get_balance(db: Any = Depends(get_scoped_db)) -> HTMLResponse:
    """Get account balance using scoped database."""
    account = await get_account_data(db)
    return HTMLResponse(content=templates.get_template("pages/balance.html").render(
        balance=account.balance
    ))
```

### Example 2: Form Submission with Multi-Update

**Frontend:**
```html
<form hx-post="/api/trade"
      hx-target="#positions"
      hx-swap="innerHTML">
    <input name="symbol" required>
    <button type="submit">Trade</button>
</form>
```

**Backend:**
```python
async def execute_trade(
    symbol: str = Form(...),
    db: Any = Depends(get_scoped_db)
) -> HTMLResponse:
    """Execute trade and update multiple UI elements."""
    order = await place_order(symbol, db)
    positions = await db.positions.find({}).to_list(10)
    
    # Return HTML with out-of-band swaps
    return HTMLResponse(content=f"""
        <div id="positions" hx-swap-oob="innerHTML">
            {render_positions(positions)}
        </div>
        <div id="toast" hx-swap-oob="beforeend">
            {render_toast("Order placed!")}
        </div>
    """)
```

### Example 3: Vector Search for Similarity

**Backend:**
```python
async def find_similar_signals(
    symbol: str,
    analysis_data: Dict[str, Any],
    db: Any = Depends(get_scoped_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> List[Dict[str, Any]]:
    """Find similar historical signals using vector search."""
    # Generate embedding
    embedding = await embedding_service.create_embedding(
        f"{symbol} {analysis_data['rsi']} {analysis_data['trend']}"
    )
    
    # Vector search (uses index from manifest.json)
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_idx",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 50,
                "limit": 5
            }
        }
    ]
    
    results = await db.radar_history.aggregate(pipeline).to_list(5)
    return results
```

## Migration Guide

### From Traditional SPA to HTMX

1. **Replace API endpoints**—return HTML fragments instead of JSON
2. **Remove JavaScript**—replace fetch/axios calls with HTMX attributes
3. **Update templates**—ensure all HTML comes from server-rendered templates
4. **Add HTMX attributes**—use `hx-get`, `hx-post`, etc. for interactions

### From Manual MongoDB to MDB-Engine

1. **Replace connection code**—use `Depends(get_scoped_db)` instead of manual connections
2. **Move indexes**—define indexes in `manifest.json` instead of migration scripts
3. **Update logging**—use `get_logger(__name__)` instead of manual logging setup
4. **Remove boilerplate**—delete ~200 lines of infrastructure code

## Conclusion

The **HTMX + MDB-Engine** stack provides:

- **Rapid development**—90% less code than traditional approaches
- **Declarative configuration**—indexes and interactions defined declaratively
- **Production-ready**—observability and security built-in
- **Type-safe**—type hints and server-rendered templates
- **Extensible**—easy to add new features and patterns
- **Maintainable**—clean code with minimal boilerplate

This stack enables developers to build modern, interactive web applications faster, with less code, and with better maintainability than traditional SPA + REST API approaches.

## Resources

- [HTMX Documentation](https://htmx.org/)
- [MDB-Engine Documentation](https://github.com/your-org/mdb-engine)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MongoDB Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/)

---

*This document reflects the gold standard implementation patterns used in this codebase.*
