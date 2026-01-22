"""Radar Service - MongoDB-backed caching and historical learning."""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from mdb_engine.observability import get_logger

logger = get_logger(__name__)


class RadarService:
    """Radar Service for caching analysis results and historical learning."""
    
    def __init__(self, db, embedding_service=None):
        """Initialize RadarService with database and embedding service."""
        self.db = db
        self.embedding_service = embedding_service
    
    def _get_embedding_service(self):
        """Get embedding service, with fallback if not injected."""
        if self.embedding_service:
            return self.embedding_service
        
        try:
            from mdb_engine.embeddings.dependencies import get_embedding_service_for_app
            from ..core.engine import engine, APP_SLUG
            
            embedding_service = get_embedding_service_for_app(APP_SLUG, engine)
            if embedding_service:
                logger.debug("Using mdb-engine EmbeddingService (fallback)")
                return embedding_service
        except (ImportError, AttributeError) as e:
            logger.debug(f"mdb-engine EmbeddingService not available: {e}")
        
        try:
            from openai import OpenAI
            from ..core.config import OPENAI_API_KEY
            if OPENAI_API_KEY:
                logger.debug("Using OpenAI client directly (fallback)")
                return OpenAI(api_key=OPENAI_API_KEY)
        except Exception as fallback_error:
            logger.error(f"Failed to initialize OpenAI embedding service: {fallback_error}")
        
        return None
    
    async def get_cached_analysis(self, symbol: str, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Check cache for analysis results.
        
        Args:
            symbol: Stock symbol
            strategy_id: Strategy identifier
            
        Returns:
            Dictionary with 'fresh' (bool) and 'analysis' (dict) if cached, None otherwise
        """
        try:
            cached = await self.db.radar_cache.find_one({
                "symbol": symbol,
                "strategy_id": strategy_id
            })
            
            if cached:
                expires_at = cached.get('expires_at')
                if expires_at and datetime.now() < expires_at:
                    logger.debug(f"Cache hit for {symbol} with strategy {strategy_id}")
                    return {
                        "fresh": True,
                        "analysis": cached.get('analysis_data', {})
                    }
                else:
                    logger.debug(f"Cache expired for {symbol} with strategy {strategy_id}")
                    await self.db.radar_cache.delete_one({"_id": cached["_id"]})
            
            return None
        except Exception as e:
            logger.error(f"Error checking cache for {symbol}: {e}", exc_info=True)
            return None
    
    async def cache_analysis(self, symbol: str, strategy_id: str, analysis_data: Dict[str, Any]) -> bool:
        """Store analysis results in cache with TTL.
        
        Args:
            symbol: Stock symbol
            strategy_id: Strategy identifier
            analysis_data: Analysis results to cache
            
        Returns:
            True if cached successfully, False otherwise
        """
        try:
            analyzed_at = datetime.now()
            expires_at = analyzed_at + timedelta(hours=1)
            
            cache_doc = {
                "symbol": symbol,
                "strategy_id": strategy_id,
                "analysis_data": analysis_data,
                "analyzed_at": analyzed_at,
                "expires_at": expires_at
            }
            
            await self.db.radar_cache.update_one(
                {"symbol": symbol, "strategy_id": strategy_id},
                {"$set": cache_doc},
                upsert=True
            )
            
            logger.debug(f"Cached analysis for {symbol} with strategy {strategy_id}")
            return True
        except Exception as e:
            logger.error(f"Error caching analysis for {symbol}: {e}", exc_info=True)
            return False
    
    async def invalidate_cache(self, symbol: str) -> bool:
        """Manually invalidate cache for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if invalidated successfully, False otherwise
        """
        try:
            result = await self.db.radar_cache.delete_many({"symbol": symbol})
            logger.info(f"Invalidated cache for {symbol}: {result.deleted_count} entries")
            return True
        except Exception as e:
            logger.error(f"Error invalidating cache for {symbol}: {e}", exc_info=True)
            return False
    
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            1536-dimensional embedding vector or None if failed
        """
        embedding_service = self._get_embedding_service()
        if not embedding_service:
            logger.warning("Embedding service not available")
            return None
        
        try:
            # Try mdb-engine EmbeddingService interface first (embed_chunks method)
            if hasattr(embedding_service, 'embed_chunks'):
                embeddings = await embedding_service.embed_chunks([text], model="text-embedding-3-small")
                if embeddings and len(embeddings) > 0:
                    embedding = embeddings[0]
                else:
                    logger.error("embed_chunks returned empty result")
                    return None
            elif hasattr(embedding_service, 'embed'):
                embedding = await embedding_service.embed(text)
            elif hasattr(embedding_service, 'embed_query'):
                embedding = await embedding_service.embed_query(text)
            elif hasattr(embedding_service, 'embeddings'):
                response = embedding_service.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                embedding = response.data[0].embedding
            else:
                logger.error("Unknown embedding service interface")
                return None
            
            if isinstance(embedding, list):
                return [float(x) for x in embedding]
            elif hasattr(embedding, 'tolist'):
                return embedding.tolist()
            else:
                logger.error(f"Unexpected embedding type: {type(embedding)}")
                return None
        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)
            return None
    
    async def store_to_history(self, symbol: str, analysis_data: Dict[str, Any], outcome: Optional[Dict[str, Any]] = None) -> bool:
        """Store analysis to historical database with embedding.
        
        Args:
            symbol: Stock symbol
            analysis_data: Full analysis context (techs, headlines, verdict, etc.)
            outcome: Optional trade outcome (pnl, days_held, exit_reason, profitable)
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            techs = analysis_data.get('techs', {})
            headlines = analysis_data.get('headlines', [])
            verdict = analysis_data.get('verdict')
            
            headlines_text = "\n".join(headlines[:5]) if isinstance(headlines, list) else str(headlines)
            
            if isinstance(verdict, dict):
                reason = verdict.get('reason', '') if verdict else analysis_data.get('reason', '')
                verdict_score = verdict.get('score', 0) if verdict else analysis_data.get('score', 0)
                verdict_risk = verdict.get('risk_level', 'UNKNOWN') if verdict else analysis_data.get('risk_level', 'UNKNOWN')
                verdict_action = verdict.get('action', 'NOT BUY') if verdict else analysis_data.get('action', 'NOT BUY')
            else:
                reason = verdict.reason if verdict and hasattr(verdict, 'reason') else analysis_data.get('reason', '')
                verdict_score = verdict.score if verdict and hasattr(verdict, 'score') else analysis_data.get('score', 0)
                verdict_risk = verdict.risk_level if verdict and hasattr(verdict, 'risk_level') else analysis_data.get('risk_level', 'UNKNOWN')
                verdict_action = verdict.action if verdict and hasattr(verdict, 'action') else analysis_data.get('action', 'NOT BUY')
            
            embedding_text = (
                f"{symbol} RSI:{techs.get('rsi', 0):.1f} "
                f"Trend:{techs.get('trend', 'UNKNOWN')} "
                f"Price:${techs.get('price', 0):.2f} "
                f"ATR:${techs.get('atr', 0):.2f} "
                f"Headlines:{headlines_text[:200]} "
                f"AI:{reason[:200]}"
            )
            
            embedding = await self._generate_embedding(embedding_text)
            
            timestamp = datetime.now()
            history_doc = {
                "symbol": symbol,
                "timestamp": timestamp,
                "metadata": {
                    "symbol": symbol,
                    "strategy": analysis_data.get('strategy', 'unknown')
                },
                "analysis": {
                    "techs": techs,
                    "headlines": headlines_text,
                    "verdict": {
                        "score": verdict_score,
                        "reason": reason,
                        "risk_level": verdict_risk,
                        "action": verdict_action
                    }
                },
                "outcome": outcome
            }
            
            # Add embedding if available
            if embedding:
                history_doc["embedding"] = embedding
            
            await self.db.radar_history.insert_one(history_doc)
            logger.debug(f"Stored analysis to history for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error storing to history for {symbol}: {e}", exc_info=True)
            return False
    
    async def find_similar_signals(self, symbol: str, analysis_data: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar signals using vector search.
        
        Args:
            symbol: Stock symbol
            analysis_data: Current analysis data
            limit: Maximum number of similar signals to return
            
        Returns:
            List of similar historical signals with outcomes
        """
        try:
            techs = analysis_data.get('techs', {})
            headlines = analysis_data.get('headlines', [])
            verdict = analysis_data.get('verdict')
            
            headlines_text = "\n".join(headlines[:5]) if isinstance(headlines, list) else str(headlines)
            
            if isinstance(verdict, dict):
                reason = verdict.get('reason', '') if verdict else analysis_data.get('reason', '')
            else:
                reason = verdict.reason if verdict and hasattr(verdict, 'reason') else analysis_data.get('reason', '')
            
            embedding_text = (
                f"{symbol} RSI:{techs.get('rsi', 0):.1f} "
                f"Trend:{techs.get('trend', 'UNKNOWN')} "
                f"Price:${techs.get('price', 0):.2f} "
                f"ATR:${techs.get('atr', 0):.2f} "
                f"Headlines:{headlines_text[:200]} "
                f"AI:{reason[:200]}"
            )
            
            query_embedding = await self._generate_embedding(embedding_text)
            if not query_embedding:
                logger.warning(f"Could not generate embedding for vector search: {symbol}")
                return []
            
            date_threshold = datetime.now() - timedelta(days=90)
            
            from ..core.engine import APP_SLUG
            vector_index_name = f"{APP_SLUG}_radar_history_vector_idx"
            
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": vector_index_name,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 10,
                        "limit": limit,
                        "filter": {
                            "metadata.symbol": symbol,
                            "timestamp": {"$gte": date_threshold}
                        }
                    }
                },
                {
                    "$project": {
                        "symbol": 1,
                        "timestamp": 1,
                        "analysis": 1,
                        "outcome": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            results = await self.db.radar_history.aggregate(pipeline).to_list(length=limit)
            
            logger.debug(f"Found {len(results)} similar signals for {symbol}")
            return results
        except Exception as e:
            logger.error(f"Error finding similar signals for {symbol}: {e}", exc_info=True)
            try:
                date_threshold = datetime.now() - timedelta(days=90)
                results = await self.db.radar_history.find({
                    "symbol": symbol,
                    "timestamp": {"$gte": date_threshold}
                }).sort("timestamp", -1).limit(limit).to_list(length=limit)
                return results
            except Exception as fallback_error:
                logger.error(f"Fallback query also failed: {fallback_error}", exc_info=True)
                return []
    
    async def get_signal_confidence(self, symbol: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence boost based on historical patterns.
        
        Args:
            symbol: Stock symbol
            analysis_data: Current analysis data
            
        Returns:
            Dictionary with 'boost' (float), 'win_rate' (float), 'similar_count' (int), 'reason' (str)
        """
        try:
            similar_signals = await self.find_similar_signals(symbol, analysis_data, limit=10)
            
            if not similar_signals:
                return {
                    "boost": 0.0,
                    "win_rate": 0.0,
                    "similar_count": 0,
                    "reason": "No historical data"
                }
            
            # Filter to signals with outcomes
            signals_with_outcomes = [
                s for s in similar_signals
                if s.get('outcome') and s['outcome'] is not None
            ]
            
            if not signals_with_outcomes:
                return {
                    "boost": 0.0,
                    "win_rate": 0.0,
                    "similar_count": 0,
                    "reason": "No outcomes yet"
                }
            
            profitable_count = sum(
                1 for s in signals_with_outcomes
                if s.get('outcome', {}).get('profitable', False)
            )
            total_count = len(signals_with_outcomes)
            win_rate = profitable_count / total_count if total_count > 0 else 0.0
            
            boost = (win_rate - 0.5) * 2.0
            boost = max(-1.0, min(1.0, boost))
            
            return {
                "boost": boost,
                "win_rate": win_rate,
                "similar_count": total_count,
                "reason": f"{profitable_count}/{total_count} similar signals were profitable"
            }
        except Exception as e:
            logger.error(f"Error calculating confidence for {symbol}: {e}", exc_info=True)
            return {
                "boost": 0.0,
                "win_rate": 0.0,
                "similar_count": 0,
                "reason": f"Error: {str(e)[:50]}"
            }
    
    async def save_daily_scan(
        self, 
        stocks: List[Dict[str, Any]], 
        strategy_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save daily scan results to MongoDB.
        
        Args:
            stocks: List of stock analysis results
            strategy_id: Strategy identifier
            metadata: Optional metadata (duration, cache hits/misses, symbols_scanned)
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            from ..core.config import is_after_final_scan_time, get_today_date_et
            
            scan_timestamp = datetime.now()
            scan_date = get_today_date_et()
            is_final = is_after_final_scan_time()
            
            scan_doc = {
                "date": scan_date,
                "timestamp": scan_timestamp,
                "is_final": is_final,
                "strategy_id": strategy_id,
                "stocks": stocks,
                "metadata": metadata or {
                    "symbols_scanned": len(stocks),
                    "duration_seconds": 0.0,
                    "cache_hits": 0,
                    "cache_misses": 0
                }
            }
            
            if is_final:
                result = await self.db.daily_scans.update_one(
                    {"date": scan_date, "is_final": True},
                    {"$set": scan_doc},
                    upsert=True
                )
                logger.info(f"Saved {'final' if is_final else 'regular'} scan for {scan_date}: {len(stocks)} stocks")
            else:
                await self.db.daily_scans.insert_one(scan_doc)
                logger.info(f"Saved scan for {scan_date}: {len(stocks)} stocks")
            
            return True
        except Exception as e:
            logger.error(f"Error saving daily scan: {e}", exc_info=True)
            return False
    
    async def get_latest_scan(self, date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get latest scan for a given date (or today if not specified).
        
        Logic:
        - If time < 6pm ET: return latest scan for date
        - If time >= 6pm ET: return final scan for date (or latest if no final exists)
        - If date not specified: use today's date
        
        Args:
            date: Optional date string (YYYY-MM-DD). If None, uses today.
            
        Returns:
            Scan document or None if not found
        """
        try:
            from ..core.config import is_after_final_scan_time, get_today_date_et
            
            if date is None:
                date = get_today_date_et()
            
            is_after_6pm = is_after_final_scan_time()
            
            if is_after_6pm:
                final_scan = await self.db.daily_scans.find_one(
                    {"date": date, "is_final": True},
                    sort=[("timestamp", -1)]
                )
                if final_scan:
                    logger.debug(f"Found final scan for {date}")
                    return final_scan
                
                latest_scan = await self.db.daily_scans.find_one(
                    {"date": date},
                    sort=[("timestamp", -1)]
                )
                if latest_scan:
                    logger.debug(f"Found latest scan for {date} (no final scan)")
                    return latest_scan
            else:
                latest_scan = await self.db.daily_scans.find_one(
                    {"date": date},
                    sort=[("timestamp", -1)]
                )
                if latest_scan:
                    logger.debug(f"Found latest scan for {date}")
                    return latest_scan
            
            latest_overall = await self.db.daily_scans.find_one(
                {},
                sort=[("timestamp", -1)]
            )
            if latest_overall:
                logger.debug(f"Found latest scan overall (date: {latest_overall.get('date')})")
                return latest_overall
            
            return None
        except Exception as e:
            logger.error(f"Error getting latest scan: {e}", exc_info=True)
            return None
    
    async def get_scan_by_date(self, date: str) -> Optional[Dict[str, Any]]:
        """Get scan for a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD)
            
        Returns:
            Scan document or None if not found
        """
        try:
            # Prefer final scan if available, otherwise latest
            final_scan = await self.db.daily_scans.find_one(
                {"date": date, "is_final": True},
                sort=[("timestamp", -1)]
            )
            if final_scan:
                return final_scan
            
            latest_scan = await self.db.daily_scans.find_one(
                {"date": date},
                sort=[("timestamp", -1)]
            )
            return latest_scan
        except Exception as e:
            logger.error(f"Error getting scan for date {date}: {e}", exc_info=True)
            return None
