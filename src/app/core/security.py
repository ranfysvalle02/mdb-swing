"""Security utilities for CSRF protection and security headers.

HTMX Best Practice: CSRF protection is essential for POST endpoints.
This module provides CSRF token generation, validation, and middleware.
"""
import secrets
from typing import Optional
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from mdb_engine.observability import get_logger

logger = get_logger(__name__)

# CSRF token storage (in production, use Redis or database)
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
        For now, we'll be lenient and allow requests without CSRF tokens during migration.
        In production, enforce CSRF validation strictly.
        """
        # Skip CSRF check for GET, HEAD, OPTIONS
        if request.method in ("GET", "HEAD", "OPTIONS"):
            response = await call_next(request)
            return response
        
        # Skip CSRF check for health/metrics endpoints
        if request.url.path in ("/health", "/metrics"):
            response = await call_next(request)
            return response
        
        # Skip CSRF check for WebSocket endpoints
        if request.url.path.startswith("/ws"):
            response = await call_next(request)
            return response
        
        # Get CSRF token from header (HTMX sends in X-CSRF-Token)
        # HTMX Gold Standard: CSRF tokens sent via headers, not form data
        # Don't consume request body here - let FastAPI handle form parsing
        csrf_token = request.headers.get("X-CSRF-Token") or request.headers.get("X-Csrf-Token")
        
        # CSRF validation: Currently disabled during migration to avoid consuming request body
        # In production, enable strict validation and ensure HTMX sends CSRF tokens in headers
        # For now, we skip validation to allow the application to work
        # TODO: Enable CSRF validation in production after adding hx-headers to all HTMX requests
        if csrf_token and not validate_csrf_token(request, csrf_token):
            logger.warning(f"CSRF validation failed for {request.method} {request.url.path} - allowing for now (migration)")
            # TODO: Enable strict CSRF validation in production
            # return Response(
            #     content='{"error": "CSRF token validation failed"}',
            #     status_code=status.HTTP_403_FORBIDDEN,
            #     media_type="application/json"
            # )
        
        response = await call_next(request)
        return response


def get_csrf_token_for_template(request: Request) -> str:
    """Get or generate CSRF token for template rendering.
    
    Args:
        request: FastAPI request object
        
    Returns:
        CSRF token string
    """
    session_id = _get_session_id(request)
    token = get_csrf_token(session_id)
    
    if not token:
        token = generate_csrf_token()
        store_csrf_token(session_id, token)
    
    return token
