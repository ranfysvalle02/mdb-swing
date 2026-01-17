"""Route Discovery System - Rails-like Convention over Configuration.

This module provides auto-discovery of routes based on folder structure.
Following Rails conventions, the folder structure drives route registration.

Conventions:
- Routes: Can be organized by domain/resource in api/ subdirectories
- Auto-discovery: Scans api/ folder for route modules
- Convention: Route files should export route functions or FastAPI routers

Note: Currently routes are in a single routes.py file, but this system
allows for future expansion to domain-driven route organization.

Example Future Structure:
    api/
        routes.py          # Main routes (current)
        positions/
            routes.py      # Position-related routes
        strategies/
            routes.py      # Strategy-related routes
"""
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Callable, Any, Optional
from fastapi import APIRouter
from mdb_engine.observability import get_logger

logger = get_logger(__name__)


class RouteRegistry:
    """Registry for auto-discovering and managing routes.
    
    Rails-like convention: Routes can be auto-discovered from the api/ folder.
    Currently supports single routes.py file, but can be extended for subdirectories.
    """
    
    def __init__(self):
        self._routes: Dict[str, Any] = {}
        self._routers: List[APIRouter] = []
        
    def discover_routes(self, api_dir: Path) -> Dict[str, Any]:
        """Auto-discover routes from the API directory.
        
        Convention: Scans api/ folder for Python files containing route functions.
        Currently supports single routes.py file, but can be extended.
        
        Args:
            api_dir: Path to API directory
            
        Returns:
            Dictionary mapping route modules to their contents
        """
        discovered = {}
        
        if not api_dir.exists():
            logger.warning(f"API directory not found: {api_dir}")
            return discovered
        
        # Currently, we have a single routes.py file
        # Future: Could scan subdirectories for domain-driven routes
        routes_file = api_dir / "routes.py"
        if routes_file.exists():
            try:
                module = importlib.import_module("src.app.api.routes")
                discovered["routes"] = module
                logger.info("✅ Discovered routes module: routes.py")
            except Exception as e:
                logger.warning(f"Failed to load routes module: {e}")
        
        # Future: Scan subdirectories for domain routes
        # for subdir in api_dir.iterdir():
        #     if subdir.is_dir() and not subdir.name.startswith("_"):
        #         routes_file = subdir / "routes.py"
        #         if routes_file.exists():
        #             # Load domain routes
        #             ...
        
        self._routes.update(discovered)
        return discovered
    
    def register_router(self, router: APIRouter, prefix: str = ""):
        """Register a FastAPI router.
        
        Args:
            router: FastAPI APIRouter instance
            prefix: URL prefix for the router (optional)
        """
        router.prefix = prefix
        self._routers.append(router)
        logger.info(f"✅ Registered router with prefix: {prefix}")
    
    def get_routes_module(self) -> Any:
        """Get the main routes module.
        
        Returns:
            Routes module (currently routes.py)
        """
        return self._routes.get("routes")


# Global registry instance (singleton pattern)
_route_registry: Optional[RouteRegistry] = None


def get_route_registry() -> RouteRegistry:
    """Get the global route registry instance.
    
    Convention: Auto-discovers routes on first access.
    
    Returns:
        RouteRegistry instance
    """
    global _route_registry
    if _route_registry is None:
        _route_registry = RouteRegistry()
        # Auto-discover routes
        api_dir = Path(__file__).parent.parent / "api"
        _route_registry.discover_routes(api_dir)
    return _route_registry
