"""MongoDB Engine initialization."""
from pathlib import Path
from mdb_engine import MongoDBEngine
from .config import MONGO_URI, MONGO_DB

# Initialize MongoDB Engine
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
