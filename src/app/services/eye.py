"""The Eye of Sauron - Core watchful scanner engine.

The Eye is the central component that orchestrates market scanning, technical analysis,
AI validation, and opportunity identification. It works with pluggable strategies that
define what patterns to seek, while all mechanics (data fetching, indicators, AI) are reusable.

Architecture:
- Eye class: Orchestrates the scanning process
- Strategy: Defines what patterns to seek (pluggable)
- RadarService: Handles caching and historical learning
- EyeAI: Provides AI analysis (strategy-agnostic)
- Analysis/Trading services: Reusable market data and execution logic

MDB-Engine Pattern: Uses scoped database access for all MongoDB operations.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from mdb_engine.observability import get_logger
from ..strategies.base import Strategy
from .ai import EyeAI
from .analysis import get_market_data, analyze_technicals
from .trading import place_order
from .radar import RadarService

logger = get_logger(__name__)


class Eye:
    """The Eye of Sauron - watches markets continuously.
    
    The Eye is the core scanner engine that watches markets. Strategies are
    pluggable lenses that focus the Eye's energy on specific patterns.
    All mechanics (data fetching, indicators, AI) are reusable.
    """
    
    def __init__(self, strategy: Strategy, db, alpaca_api=None):
        """Initialize the Eye with a strategy.
        
        Args:
            strategy: Strategy instance that defines what the Eye seeks
            db: MongoDB database instance
            alpaca_api: Alpaca API instance (optional, for position checking)
        """
        self.strategy = strategy
        self.db = db
        self.alpaca_api = alpaca_api
        
        # Initialize AI engine
        try:
            self.ai = EyeAI()
        except Exception as e:
            logger.warning(f"Azure OpenAI not configured. The Eye cannot see: {e}")
            self.ai = None
    
    async def scan_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Watch a single symbol - reusable scanning logic.
        
        The Eye watches the symbol, checks technical conditions via strategy,
        validates with AI, and creates pending trade if opportunity found.
        
        Args:
            symbol: Stock symbol to scan
            
        Returns:
            Dictionary with scan result, or None if no opportunity
        """
        try:
            # Initialize RadarService
            # MDB-Engine Pattern: Get EmbeddingService for non-route context
            from mdb_engine.embeddings.dependencies import get_embedding_service_for_app
            from ..core.engine import engine, APP_SLUG
            
            embedding_service = get_embedding_service_for_app(APP_SLUG, engine)
            radar_service = RadarService(self.db, embedding_service=embedding_service)
            strategy_id = self.strategy.get_name()
            
            # Check cache before expensive operations
            cached = await radar_service.get_cached_analysis(symbol, strategy_id)
            if cached and cached['fresh']:
                logger.debug(f"👁️ The Eye sees cached analysis for {symbol} - using cache")
                return cached['analysis']
            
            # Skip if we already have a position
            if self.alpaca_api:
                try:
                    pos = self.alpaca_api.get_position(symbol)
                    if float(pos.qty) > 0:
                        logger.debug(f"👁️ The Eye sees {symbol} already held - skipping")
                        return None
                except Exception:
                    # No position exists, continue watching
                    pass
            
            # Fetch market data
            bars, headlines, news_objects = get_market_data(symbol)
            if bars is None or bars.empty:
                logger.debug(f"👁️ The Eye sees no data for {symbol} - skipping")
                return None
            
            if len(bars) < 14:
                logger.debug(f"👁️ The Eye sees insufficient data for {symbol} ({len(bars)} days) - skipping")
                return None
            
            # Calculate technical indicators
            try:
                techs = analyze_technicals(bars)
            except Exception as e:
                logger.error(f"👁️ The Eye's technical analysis failed for {symbol}: {e}")
                return None
            
            # Check if strategy's technical conditions are met
            if not self.strategy.check_technical_signal(techs):
                logger.debug(f"👁️ The Eye watches {symbol} - no {self.strategy.get_name()} signal")
                return None
            
            logger.info(f"👁️ The Eye focuses on {symbol} - {self.strategy.get_name()} signal detected")
            
            # Find similar historical signals for context
            analysis_data_for_search = {
                'techs': techs,
                'headlines': headlines
            }
            similar_signals = await radar_service.find_similar_signals(symbol, analysis_data_for_search, limit=5)
            
            # Validate with AI if available
            if not self.ai:
                logger.warning(f"⚠️ The Eye is blind - cannot validate {symbol} without AI")
                return None
            
            try:
                # Get strategy-specific AI prompt
                strategy_prompt = self.strategy.get_ai_prompt()
                
                # Include historical context in prompt if available
                if similar_signals:
                    profitable_count = sum(
                        1 for s in similar_signals
                        if s.get('outcome', {}).get('profitable', False)
                    )
                    if profitable_count > 0:
                        strategy_prompt += f"\n\nHistorical Context: {profitable_count} out of {len(similar_signals)} similar signals were profitable."
                
                # Analyze with AI using strategy prompt
                verdict = self.ai.analyze(
                    ticker=symbol,
                    techs=techs,
                    headlines="\n".join(headlines),
                    strategy_prompt=strategy_prompt
                )
                
                # Calculate confidence boost from historical patterns
                confidence = await radar_service.get_signal_confidence(symbol, {
                    'techs': techs,
                    'headlines': headlines,
                    'verdict': verdict
                })
                
                # Check if score meets strategy requirements (with confidence boost)
                config = self.strategy.get_config()
                ai_score_required = config.get('ai_score_required', 7)
                adjusted_score = verdict.score + confidence['boost']
                
                # Store analysis to cache and history
                analysis_data = {
                    'symbol': symbol,
                    'techs': techs,
                    'headlines': headlines,
                    'verdict': verdict,
                    'timestamp': datetime.now(),
                    'strategy': strategy_id,
                    'confidence': confidence
                }
                await radar_service.cache_analysis(symbol, strategy_id, analysis_data)
                await radar_service.store_to_history(symbol, analysis_data)
                
                if adjusted_score >= ai_score_required:
                    logger.info(
                        f"🔥 The Eye commands: {symbol} - {self.strategy.get_name()} opportunity "
                        f"(Score: {verdict.score}/10 + {confidence['boost']:.2f} boost = {adjusted_score:.2f}, Required: {ai_score_required})"
                    )
                    
                    return {
                        "symbol": symbol,
                        "strategy": self.strategy.get_name(),
                        "score": verdict.score,
                        "adjusted_score": adjusted_score,
                        "confidence_boost": confidence['boost'],
                        "action": "opportunity_found"
                    }
                else:
                    logger.info(
                        f"⚫ The Eye sees weakness in {symbol} - {self.strategy.get_name()} rejected "
                        f"(Score: {verdict.score} + {confidence['boost']:.2f} = {adjusted_score:.2f}, Required: {ai_score_required})"
                    )
                    return None
                    
            except Exception as e:
                logger.error(f"👁️ The Eye's AI analysis failed for {symbol}: {e}", exc_info=True)
                return None
                
        except Exception as e:
            logger.error(f"👁️ The Eye encountered an error watching {symbol}: {e}", exc_info=True)
            return None
    
    async def scan_watchlist(self, symbols: List[str]) -> Dict[str, Any]:
        """Scan multiple symbols - the Eye watches all.
        
        Args:
            symbols: List of stock symbols to scan
            
        Returns:
            Dictionary with scan summary
        """
        logger.info(f"👁️ The Eye watches... scanning {len(symbols)} symbols for {self.strategy.get_name()} opportunities...")
        
        opportunities_found = 0
        errors = 0
        
        for symbol in symbols:
            result = await self.scan_symbol(symbol)
            if result:
                opportunities_found += 1
            elif result is None:
                # Normal case - no opportunity found
                pass
            else:
                errors += 1
        
        summary = {
            "symbols_scanned": len(symbols),
            "opportunities_found": opportunities_found,
            "errors": errors,
            "strategy": self.strategy.get_name()
        }
        
        logger.info(
            f"👁️ The Eye's watch complete - {opportunities_found} {self.strategy.get_name()} "
            f"opportunities found from {len(symbols)} symbols"
        )
        
        return summary
