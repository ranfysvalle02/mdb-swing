"""Security utilities for CSRF protection and security headers."""
import os
import secrets
from typing import Optional
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse
from mdb_engine.observability import get_logger

logger = get_logger(__name__)

ENABLE_CSRF_PROTECTION = os.getenv("ENABLE_CSRF_PROTECTION", "false").lower() == "true"
_csrf_tokens: dict[str, str] = {}


def generate_csrf_token() -> str:
    """Generate a secure CSRF token."""
    return secrets.token_urlsafe(32)


def store_csrf_token(session_id: str, token: str) -> None:
    """Store a CSRF token for a session."""
    _csrf_tokens[session_id] = token


def get_csrf_token(session_id: str) -> Optional[str]:
    """Get stored CSRF token for a session."""
    return _csrf_tokens.get(session_id)


def validate_csrf_token(request: Request, token: Optional[str]) -> bool:
    """Validate CSRF token from request."""
    if not token:
        return False
    
    session_id = _get_session_id(request)
    stored_token = get_csrf_token(session_id)
    
    if not stored_token:
        return False
    
    return secrets.compare_digest(token, stored_token)


def _get_session_id(request: Request) -> str:
    """Generate a session ID from request."""
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    import hashlib
    return hashlib.sha256(f"{client_ip}:{user_agent}".encode()).hexdigest()


class CSRFMiddleware(BaseHTTPMiddleware):
    """CSRF protection middleware for POST/PUT/DELETE requests."""
    
    async def dispatch(self, request: Request, call_next):
        if not ENABLE_CSRF_PROTECTION:
            return await call_next(request)
        
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return await call_next(request)
        
        if request.url.path in ("/health", "/metrics"):
            return await call_next(request)
        
        if request.url.path.startswith("/ws"):
            return await call_next(request)
        
        csrf_token = (
            request.headers.get("X-CSRF-Token") or 
            request.headers.get("X-Csrf-Token") or
            request.headers.get("X-Csrf-Token")
        )
        
        if not csrf_token:
            try:
                form_data = await request.form()
                csrf_token = form_data.get("csrf_token")
            except Exception:
                pass
        
        if not csrf_token:
            logger.warning(f"CSRF token missing for {request.method} {request.url.path}")
            return JSONResponse(
                content={"error": "CSRF token missing", "detail": "Please include X-CSRF-Token header"},
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        if not validate_csrf_token(request, csrf_token):
            logger.warning(f"CSRF validation failed for {request.method} {request.url.path}")
            return JSONResponse(
                content={"error": "CSRF token validation failed", "detail": "Invalid or expired CSRF token"},
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        return await call_next(request)


def get_csrf_token_for_template(request: Request) -> str:
    """Get or generate CSRF token for template rendering."""
    session_id = _get_session_id(request)
    token = get_csrf_token(session_id)
    
    if not token:
        token = generate_csrf_token()
        store_csrf_token(session_id, token)
        logger.debug(f"Generated new CSRF token for session {session_id[:8]}...")
    
    return token
