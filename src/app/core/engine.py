"""MongoDB Engine initialization.

MDB-Engine Showcase:
This module demonstrates mdb-engine's core features:
1. Simplified FastAPI app setup - engine.create_app() handles everything
2. Declarative index management - indexes defined in manifest.json
3. Scoped database access - get_scoped_db() for dependency injection
4. Connection lifecycle - automatic pooling and cleanup
5. Observability - built-in logging via get_logger()

See main.py for app creation and config/manifest.json for index definitions.
"""
from pathlib import Path
from mdb_engine import MongoDBEngine
from .config import MONGO_URI, MONGO_DB

# MDB-Engine: Initialize the engine once, use everywhere
# This handles connection pooling, lifecycle management, and app configuration
engine = MongoDBEngine(
    mongo_uri=MONGO_URI,
    db_name=MONGO_DB
)

def get_manifest_path() -> Path:
    """Get the path to manifest.json."""
    # In Docker, manifest is mounted at /app/config/manifest.json
    # In local dev, it's at project root/config/manifest.json
    docker_path = Path("/app/config/manifest.json")
    if docker_path.exists():
        return docker_path
    return Path(__file__).parent.parent.parent.parent / "config" / "manifest.json"

# App slug constant - used throughout the application
APP_SLUG = "flux"

async def get_scoped_db():
    """Get scoped database for FLUX app using mdb-engine.
    
    MDB-Engine Pattern: Dependency Injection for Database Access
    
    This is a FastAPI dependency function that automatically provides the scoped database.
    Use with FastAPI's Depends():
    
        async def my_route(db = Depends(get_scoped_db)):
            await db.collection.find({}).to_list(10)
    
    Benefits:
    - Automatic connection lifecycle management (no manual cleanup needed)
    - Connection pooling handled by mdb-engine
    - Scoped to request lifecycle (connections closed after request)
    - Type-safe database access
    
    Example usage in routes:
        async def get_positions(db = Depends(get_scoped_db)) -> HTMLResponse:
            positions = await db.positions.find({}).to_list(10)
            # Connection automatically managed - no cleanup needed!
    
    The database is automatically scoped to the app (APP_SLUG), with proper connection
    lifecycle management via mdb-engine. All queries are automatically filtered
    by app_id, and collection names are prefixed.
    """
    return engine.get_scoped_db(APP_SLUG)
