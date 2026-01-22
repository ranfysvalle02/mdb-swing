"""MongoDB Engine initialization."""
from pathlib import Path
from mdb_engine import MongoDBEngine
from .config import MONGO_URI, MONGO_DB

engine = MongoDBEngine(
    mongo_uri=MONGO_URI,
    db_name=MONGO_DB
)

def get_manifest_path() -> Path:
    """Get the path to manifest.json."""
    docker_path = Path("/app/config/manifest.json")
    if docker_path.exists():
        return docker_path
    return Path(__file__).parent.parent.parent.parent / "config" / "manifest.json"

APP_SLUG = "flux"

async def get_scoped_db():
    """Get scoped database for FLUX app."""
    return engine.get_scoped_db(APP_SLUG)
