"""Plugin Discovery System - Rails-like Convention over Configuration.

This module provides auto-discovery of plugins and services based on folder structure.
Following Rails conventions, the folder structure drives functionality.

Conventions:
- Services: Auto-discovered from `services/` folder
- Plugins: Future expansion for `plugins/` folder
- Auto-discovery: Scans folders and registers available components
- Lazy loading: Components loaded on-demand

Example Usage:
    from .core.plugins import get_service
    
    # Get a service by name (auto-discovered)
    service = get_service('eye')
"""
import importlib
import inspect
from pathlib import Path
from typing import Dict, Type, Optional, Any, Callable
from mdb_engine.observability import get_logger

logger = get_logger(__name__)


class ServiceRegistry:
    """Registry for auto-discovering and managing services.
    
    Rails-like convention: Services are auto-discovered from the services/ folder.
    Each service file should contain classes or functions that can be registered.
    """
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._instances: Dict[str, Any] = {}
        
    def discover_services(self, services_dir: Path) -> Dict[str, Any]:
        """Auto-discover services from the services directory.
        
        Convention: Scans services/ folder for Python files containing service classes/functions.
        Files starting with _ or __ are ignored (private/convention).
        
        Args:
            services_dir: Path to services directory
            
        Returns:
            Dictionary mapping service names to service classes/functions
        """
        discovered = {}
        
        if not services_dir.exists():
            logger.warning(f"Services directory not found: {services_dir}")
            return discovered
        
        # Scan for Python files in services directory
        for file_path in services_dir.glob("*.py"):
            # Skip private files (Rails convention: _* files are private)
            if file_path.name.startswith("_") or file_path.name.startswith("__"):
                continue
            
            try:
                # Import module (e.g., services.eye)
                module_name = f"src.app.services.{file_path.stem}"
                module = importlib.import_module(module_name)
                
                # Find main classes/functions in the module
                # Convention: Look for classes with descriptive names or main functions
                for name, obj in inspect.getmembers(module):
                    # Skip private members
                    if name.startswith("_"):
                        continue
                    
                    # Register classes (e.g., Eye, RadarService)
                    if inspect.isclass(obj) and obj.__module__ == module_name:
                        service_name = name.lower().replace("service", "").replace("class", "")
                        if service_name:
                            discovered[service_name] = obj
                            discovered[name.lower()] = obj
                            logger.debug(f"✅ Discovered service class: {name} (registered as '{service_name}')")
                    
                    # Register module-level functions (e.g., analyze_technicals)
                    elif inspect.isfunction(obj) and obj.__module__ == module_name:
                        service_name = name.lower()
                        discovered[service_name] = obj
                        logger.debug(f"✅ Discovered service function: {name}")
                
                # Also register the module itself by filename
                service_name = file_path.stem.lower()
                discovered[service_name] = module
                
            except Exception as e:
                logger.warning(f"Failed to load service from {file_path.name}: {e}")
                continue
        
        self._services.update(discovered)
        return discovered
    
    def register_service(self, name: str, service: Any):
        """Manually register a service.
        
        Args:
            name: Service identifier (lowercase)
            service: Service class, function, or module
        """
        self._services[name.lower()] = service
        logger.info(f"✅ Registered service: {name}")
    
    def get_service(self, name: str) -> Any:
        """Get a service by name.
        
        Args:
            name: Service name
            
        Returns:
            Service class, function, or module
            
        Raises:
            ValueError: If service not found
        """
        name = name.lower()
        service = self._services.get(name)
        
        if service is None:
            # Try partial match
            for key, svc in self._services.items():
                if name in key or key in name:
                    service = svc
                    break
        
        if service is None:
            available = ", ".join(self._services.keys())
            raise ValueError(f"Service '{name}' not found. Available: {available}")
        
        return service
    
    def list_services(self) -> list[str]:
        """List all discovered service names.
        
        Returns:
            List of service names
        """
        return list(self._services.keys())


# Global registry instance (singleton pattern)
_service_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """Get the global service registry instance.
    
    Convention: Auto-discovers services on first access.
    
    Returns:
        ServiceRegistry instance
    """
    global _service_registry
    if _service_registry is None:
        _service_registry = ServiceRegistry()
        # Auto-discover services
        services_dir = Path(__file__).parent.parent / "services"
        _service_registry.discover_services(services_dir)
    return _service_registry


def get_service(name: str) -> Any:
    """Convenience function to get a service by name.
    
    Args:
        name: Service name
        
    Returns:
        Service class, function, or module
    """
    return get_service_registry().get_service(name)


def discover_services() -> Dict[str, Any]:
    """Convenience function to discover services.
    
    Returns:
        Dictionary of discovered services
    """
    return get_service_registry().discover_services(
        Path(__file__).parent.parent / "services"
    )
