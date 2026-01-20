# Static Assets - Future-Proof Dependency Management

## Zero-Build Rule: Vendor All Dependencies

This directory contains **vendored** (downloaded and committed) JavaScript libraries.

**Why Vendor?**
- In 2055, npm might be gone. GitHub might be gone. CDNs might be down.
- A file sitting in your repo included via `<script src="/static/js/htmx.js">` will work forever.
- No build chain = no broken dependencies = future-proof code.

## Current Dependencies (To Be Vendored)

### Required Files:
1. **htmx.min.js** (v1.9.10)
   - Download from: https://unpkg.com/htmx.org@1.9.10/dist/htmx.min.js
   - Save to: `frontend/static/js/htmx.min.js`

2. **alpinejs.min.js** (v3.13.5)
   - Download from: https://cdn.jsdelivr.net/npm/alpinejs@3.13.5/dist/cdn.min.js
   - Save to: `frontend/static/js/alpinejs.min.js`

3. **lucide.min.js** (latest)
   - Download from: https://unpkg.com/lucide@latest/dist/umd/lucide.min.js
   - Save to: `frontend/static/js/lucide.min.js`

## How to Vendor:

```bash
# Download and save dependencies
curl -o frontend/static/js/htmx.min.js https://unpkg.com/htmx.org@1.9.10/dist/htmx.min.js
curl -o frontend/static/js/alpinejs.min.js https://cdn.jsdelivr.net/npm/alpinejs@3.13.5/dist/cdn.min.js
curl -o frontend/static/js/lucide.min.js https://unpkg.com/lucide@latest/dist/umd/lucide.min.js
```

## Update index.html:

Replace CDN links with local paths:
```html
<!-- Before (CDN - fragile) -->
<script src="https://unpkg.com/htmx.org@1.9.10"></script>

<!-- After (Vendored - future-proof) -->
<script src="/static/js/htmx.min.js"></script>
```

## Benefits:

✅ **Works offline** - No internet required to run the app
✅ **Version locked** - Dependencies won't change unexpectedly  
✅ **Faster loads** - No external CDN requests
✅ **Future-proof** - Files in repo work forever, even if CDNs disappear
✅ **No build step** - Pure HTML/CSS/JS, no npm/webpack/vite needed

## Note on Tailwind:

Tailwind CDN is currently used for development. For production:
- Consider using Tailwind CLI to generate a static CSS file
- Or use a stable utility CSS framework
- Or write vanilla CSS (most future-proof)
