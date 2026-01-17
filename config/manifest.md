# Manifest.json Documentation

This file explains the mdb-engine manifest.json configuration for Sauron's Eye.

## Overview

The manifest.json file is mdb-engine's declarative configuration system. It defines:
- Database indexes (including vector search)
- CORS settings
- WebSocket configuration (if needed)
- Collection-specific settings

## Indexes Explained

### `history` Collection
- **timestamp_idx**: Sorts by timestamp descending for recent-first queries
- **symbol_idx**: Fast lookups by stock symbol

### `radar_cache` Collection
- **symbol_time_idx**: Composite index for cache lookups by symbol and time
- **symbol_idx**: Unique index ensuring one cache entry per symbol
- **ttl_idx**: TTL index that auto-deletes entries after 1 hour (3600 seconds)
  - MDB-Engine automatically manages this - no manual cleanup needed!

### `radar_history` Collection
- **symbol_time_idx**: Composite index for historical queries by symbol and time
- **timestamp_idx**: General timestamp sorting
- **vector_idx**: Vector search index for semantic similarity
  - **Type**: Vector search (MongoDB Atlas Vector Search)
  - **Dimensions**: 1536 (OpenAI embedding dimensions)
  - **Similarity**: Cosine similarity
  - **Path**: `embedding` field
  - **Filter**: Only indexes documents with `metadata.symbol`
  - Used by RadarService.find_similar_signals() for finding similar trading signals

### `strategy_config` Collection
- **active_idx**: Unique partial index ensuring only one active strategy
  - Partial filter: Only indexes documents where `active: true`
  - Ensures data integrity at the database level

## CORS Configuration

```json
"cors": {
  "enabled": true,
  "allow_origins": ["*"],
  "allow_credentials": true,
  "allow_methods": ["*"],
  "allow_headers": ["*"]
}
```

MDB-Engine automatically applies these CORS settings to the FastAPI app.
No manual CORS middleware configuration needed!

## WebSocket Configuration

WebSocket endpoints are configured programmatically in `main.py`:
- Path: `/ws`
- Handler: Custom handler for real-time stock analysis streaming
- MDB-Engine provides `register_message_handler()` for message routing

## Benefits of Declarative Configuration

1. **Version Control**: Index definitions are in code, not scattered in database
2. **Automatic Management**: MDB-Engine creates/updates indexes on startup
3. **Type Safety**: JSON schema validation ensures correct configuration
4. **Documentation**: This file serves as living documentation
5. **Reproducibility**: Same indexes in dev, staging, and production

## MDB-Engine Features Demonstrated

- **Declarative Index Management**: All indexes defined here
- **Vector Search**: Semantic similarity for finding similar signals
- **TTL Indexes**: Automatic cache expiration
- **Partial Indexes**: Efficient unique constraints
- **Composite Indexes**: Optimized query performance
