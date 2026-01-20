"""Security utilities for CSRF protection and security headers.

HTMX Best Practice: CSRF protection is essential for POST endpoints.
This module provides CSRF token generation, validation, and middleware.

CSRF tokens are automatically generated and stored per session.
HTMX requests should include the token in the X-CSRF-Token header.
"""
import os
import secrets
from typing import Optional
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse
from mdb_engine.observability import get_logger

logger = get_logger(__name__)

# CSRF protection can be disabled for development/testing
# Set ENABLE_CSRF_PROTECTION=false to disable (not recommended for production)
# Default to false - enable after adding CSRF tokens to HTMX requests via hx-headers
ENABLE_CSRF_PROTECTION = os.getenv("ENABLE_CSRF_PROTECTION", "false").lower() == "true"

# CSRF token storage (in production, use Redis or database)
# Using in-memory dict for simplicity - consider Redis for multi-instance deployments
_csrf_tokens: dict[str, str] = {}


def generate_csrf_token() -> str:
    """Generate a secure CSRF token.
    
    Returns:
        A URL-safe random token string.
    """
    return secrets.token_urlsafe(32)


def store_csrf_token(session_id: str, token: str) -> None:
    """Store a CSRF token for a session.
    
    Args:
        session_id: Session identifier (can use client IP + User-Agent hash)
        token: CSRF token to store
    """
    _csrf_tokens[session_id] = token


def get_csrf_token(session_id: str) -> Optional[str]:
    """Get stored CSRF token for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        CSRF token if found, None otherwise
    """
    return _csrf_tokens.get(session_id)


def validate_csrf_token(request: Request, token: Optional[str]) -> bool:
    """Validate CSRF token from request.
    
    HTMX sends CSRF token in X-CSRF-Token header or form data.
    
    Args:
        request: FastAPI request object
        token: CSRF token from request (header or form)
        
    Returns:
        True if token is valid, False otherwise
    """
    if not token:
        return False
    
    # Get session ID from request (simplified - use IP + User-Agent hash)
    session_id = _get_session_id(request)
    stored_token = get_csrf_token(session_id)
    
    if not stored_token:
        return False
    
    # Use secrets.compare_digest to prevent timing attacks
    return secrets.compare_digest(token, stored_token)


def _get_session_id(request: Request) -> str:
    """Generate a session ID from request.
    
    In production, use proper session management.
    For now, use IP + User-Agent hash.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Session identifier string
    """
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    import hashlib
    return hashlib.sha256(f"{client_ip}:{user_agent}".encode()).hexdigest()


class CSRFMiddleware(BaseHTTPMiddleware):
    """CSRF protection middleware for POST/PUT/DELETE requests.
    
    HTMX Best Practice: Protect all state-changing operations.
    """
    
    async def dispatch(self, request: Request, call_next):
        """Process request and validate CSRF token for state-changing methods.
        
        HTMX Gold Standard: CSRF protection for POST/PUT/DELETE requests.
        
        CSRF tokens are expected in the X-CSRF-Token header for HTMX requests.
        Tokens are automatically generated and stored per session.
        
        To disable CSRF protection (not recommended), set ENABLE_CSRF_PROTECTION=false.
        """
        # Skip CSRF check if disabled
        if not ENABLE_CSRF_PROTECTION:
            response = await call_next(request)
            return response
        
        # Skip CSRF check for GET, HEAD, OPTIONS (safe methods)
        if request.method in ("GET", "HEAD", "OPTIONS"):
            response = await call_next(request)
            return response
        
        # Skip CSRF check for health/metrics endpoints (monitoring)
        if request.url.path in ("/health", "/metrics"):
            response = await call_next(request)
            return response
        
        # Skip CSRF check for WebSocket endpoints (different protocol)
        if request.url.path.startswith("/ws"):
            response = await call_next(request)
            return response
        
        # Get CSRF token from header (HTMX sends in X-CSRF-Token)
        # HTMX Gold Standard: CSRF tokens sent via headers, not form data
        # Also check form data as fallback for non-HTMX requests
        csrf_token = (
            request.headers.get("X-CSRF-Token") or 
            request.headers.get("X-Csrf-Token") or
            request.headers.get("X-Csrf-Token")
        )
        
        # Try to get from form data if not in headers (for non-HTMX requests)
        if not csrf_token:
            try:
                # Note: This requires reading the request body, which FastAPI handles
                # For HTMX requests, tokens should be in headers
                form_data = await request.form()
                csrf_token = form_data.get("csrf_token")
            except Exception:
                # Request body already consumed or not form data
                pass
        
        # Validate CSRF token
        if not csrf_token:
            logger.warning(
                f"CSRF token missing for {request.method} {request.url.path}. "
                "HTMX requests should include X-CSRF-Token header."
            )
            return JSONResponse(
                content={"error": "CSRF token missing", "detail": "Please include X-CSRF-Token header"},
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        if not validate_csrf_token(request, csrf_token):
            logger.warning(
                f"CSRF validation failed for {request.method} {request.url.path}. "
                "Token may be expired or invalid."
            )
            return JSONResponse(
                content={"error": "CSRF token validation failed", "detail": "Invalid or expired CSRF token"},
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        # Token is valid, proceed with request
        response = await call_next(request)
        return response


def get_csrf_token_for_template(request: Request) -> str:
    """Get or generate CSRF token for template rendering.
    
    This function ensures a CSRF token exists for the current session.
    Tokens are automatically generated if missing.
    
    HTMX Best Practice: Include CSRF token in all forms and HTMX requests.
    Use hx-headers attribute to include token: hx-headers='{"X-CSRF-Token": "{{ csrf_token }}"}'
    
    Args:
        request: FastAPI request object
        
    Returns:
        CSRF token string for the current session
    """
    session_id = _get_session_id(request)
    token = get_csrf_token(session_id)
    
    if not token:
        token = generate_csrf_token()
        store_csrf_token(session_id, token)
        logger.debug(f"Generated new CSRF token for session {session_id[:8]}...")
    
    return token
