"""Intelligent Symbol Discovery Service - Scouting Tool.

This is a scouting/discovery tool that:
1. Generates intelligent queries based on strategy
2. Searches web/news for relevant articles
3. Extracts tickers directly from search results
4. Reasons about each ticker with strategy context
5. Ranks and selects top 5 based on strategy criteria
6. Auto-populates target symbols

Architecture:
- Scouting-focused: Designed as a discovery tool, not manual input
- Direct ticker extraction: Extracts tickers directly, not company names first
- Intelligent ranking: Uses LLM to reason about each ticker
- Strategy-aware: All steps consider current trading strategy
- Fully automated: No manual comma-separated input needed

MDB-Engine Integration:
- Logging: Uses `get_logger(__name__)` from mdb_engine.observability for structured logging
"""
from typing import List, Dict, Any, Optional
from mdb_engine.observability import get_logger
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator
import re
import asyncio

logger = get_logger(__name__)

# Try to import Firecrawl
try:
    from firecrawl import FirecrawlApp
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False
    logger.warning("firecrawl-py not installed - symbol discovery will be limited")

from ..core.config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_MODEL_NAME,
    FIRECRAWL_API_KEY,
    CURRENT_STRATEGY
)


class StockOpportunity(BaseModel):
    """Stock opportunity discovered by agent."""
    ticker: str = Field(description="Stock ticker symbol (e.g., 'AAPL', 'NVDA')")
    company_name: str = Field(description="Company name")
    why_mentioned: str = Field(description="Why this stock was mentioned as a buy opportunity")
    potential_reason: str = Field(description="Reasoning for why it's a good buy (oversold, uptrend, etc.)")
    relevance_score: int = Field(description="Relevance score 1-10", ge=1, le=10)
    
    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Validate ticker format: 1-5 uppercase letters."""
        v = v.strip().upper()
        if not re.match(r'^[A-Z]{1,5}$', v):
            raise ValueError(f"Invalid ticker format: {v}")
        return v


class StockOpportunitiesSchema(BaseModel):
    """Schema for agent to return discovered stocks."""
    stocks: List[StockOpportunity] = Field(description="List of stock opportunities discovered", min_length=1, max_length=20)
    
    @field_validator('stocks')
    @classmethod
    def validate_stocks(cls, v: List[StockOpportunity]) -> List[StockOpportunity]:
        """Deduplicate tickers."""
        seen = set()
        unique = []
        for stock in v:
            if stock.ticker not in seen:
                seen.add(stock.ticker)
                unique.append(stock)
        return unique


class TickerRanking(BaseModel):
    """Ranked ticker with reasoning."""
    ticker: str = Field(description="Stock ticker symbol")
    rank: int = Field(description="Rank (1 = best match for strategy)", ge=1)
    reasoning: str = Field(description="Why this ticker was selected/ranked")
    match_score: float = Field(description="How well it matches strategy criteria (0-10)", ge=0.0, le=10.0)
    
    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Validate ticker format."""
        v = v.strip().upper()
        if not re.match(r'^[A-Z]{1,5}$', v):
            raise ValueError(f"Invalid ticker format: {v}")
        return v


class TickerRankingList(BaseModel):
    """List of ranked tickers."""
    rankings: List[TickerRanking] = Field(description="Ranked tickers (top 5)", min_length=1, max_length=5)


class SymbolDiscoveryService:
    """Intelligent symbol discovery scouting tool.
    
    Pipeline:
    1. Generate intelligent scouting query based on strategy
    2. Search web/news for articles (first few high-quality results)
    3. Extract tickers directly from search results
    4. Reason about each ticker with strategy context
    5. Rank and select top 5 based on strategy criteria
    6. Return ready-to-use ticker symbols
    """
    
    def __init__(self):
        """Initialize the discovery service."""
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are required")
        
        self.llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
            model_name=AZURE_OPENAI_MODEL_NAME,
            api_version="2024-08-01-preview",
            temperature=0.3  # Lower temperature for more consistent extraction
        )
        
        self.firecrawl_client = None
        if FIRECRAWL_AVAILABLE and FIRECRAWL_API_KEY:
            try:
                self.firecrawl_client = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
                logger.info("Firecrawl client initialized for symbol discovery")
            except Exception as e:
                logger.warning(f"Could not initialize Firecrawl client: {e}")
    
    def _generate_search_query(self, strategy_name: str, strategy_config: Dict[str, Any]) -> str:
        """Generate a search query based on strategy.
        
        Creates search queries for Firecrawl Search API (faster than Agent).
        Uses discovery_prompt if available, otherwise falls back to default logic.
        
        Args:
            strategy_name: Name of the strategy (e.g., 'balanced_low')
            strategy_config: Strategy configuration dictionary
            
        Returns:
            Search query string optimized for the strategy
        """
        # Check if discovery_prompt exists and use it
        discovery_prompt = strategy_config.get('discovery_prompt', '')
        if discovery_prompt:
            # Extract key search terms from prompt
            # Take first few non-comment lines and combine into search query
            lines = discovery_prompt.split('\n')
            search_terms = []
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                # Skip empty lines, comments, and section headers
                if line and not line.startswith('#') and not line.startswith('-') and not line[0].isdigit():
                    # Extract quoted search queries if present
                    if '"' in line:
                        import re
                        quoted = re.findall(r'"([^"]+)"', line)
                        search_terms.extend(quoted)
                    else:
                        # Use the line if it's not a header
                        if not line.endswith(':'):
                            search_terms.append(line)
            
            if search_terms:
                # Combine first few search terms
                query = ' '.join(search_terms[:3]) + " today swing trading"
                logger.info(f"📝 [SCOUTING] Using discovery_prompt to generate query: '{query[:100]}...'")
                return query
        
        # Fallback to default logic
        rsi_threshold = strategy_config.get('rsi_threshold', 35)
        description = strategy_config.get('description', 'swing trading opportunities')
        
        if strategy_name == "balanced_low":
            return (
                f"oversold stocks RSI below {rsi_threshold} swing trading buy opportunities today "
                f"stocks in uptrend oversold bounce potential technical analysis"
            )
        else:
            return (
                f"{description} stocks buy opportunities today swing trading technical analysis"
            )
    
    async def _discover_stocks_with_search(self, query: str, max_retries: int = 2) -> List[StockOpportunity]:
        """Use Firecrawl Search + LLM to discover stocks (faster than Agent).
        
        Uses search API to get results quickly, then LLM extracts tickers.
        Much faster than Agent approach - doesn't freeze the system.
        
        Args:
            query: Search query for finding stocks
            max_retries: Maximum retry attempts
            
        Returns:
            List of StockOpportunity objects
        """
        if not self.firecrawl_client:
            logger.warning("Firecrawl not available, cannot use search")
            return []
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔍 [SCOUTING] Using Firecrawl Search to discover stocks...")
                logger.info(f"   Query: {query[:100]}...")
                
                # Use Firecrawl Search API (faster than Agent)
                search_response = self.firecrawl_client.search(
                    query=query,
                    limit=10  # Get top 10 results
                )
                
                # Firecrawl search returns SearchData object with data attribute
                # data contains: web, images, news arrays
                if not search_response:
                    logger.warning("⚠️ [SCOUTING] Search returned no response")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                    return []
                
                # Access the data attribute (SearchData object)
                search_data = search_response.data if hasattr(search_response, 'data') else search_response
                
                # Extract web results (SearchData.data.web is a list)
                web_results = []
                if hasattr(search_data, 'web') and search_data.web:
                    web_results = search_data.web
                elif isinstance(search_data, dict):
                    web_results = search_data.get('web', [])
                elif isinstance(search_data, list):
                    web_results = search_data
                
                if not web_results:
                    logger.warning("⚠️ [SCOUTING] No web results found in search response")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                    return []
                
                # Extract text from search results
                search_texts = []
                for result in web_results[:10]:  # Limit to first 10 results
                    # Handle both dict and object access
                    if isinstance(result, dict):
                        title = result.get('title', '')
                        description = result.get('description', '')
                        url = result.get('url', '')
                    else:
                        # Pydantic model or object
                        title = getattr(result, 'title', '') or ''
                        description = getattr(result, 'description', '') or ''
                        url = getattr(result, 'url', '') or ''
                    
                    if title or description:
                        search_texts.append(f"Title: {title}\nDescription: {description}\nURL: {url}")
                
                if not search_texts:
                    logger.warning("⚠️ [SCOUTING] No text extracted from search results")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                    return []
                
                # Use LLM to extract stock tickers from search results
                combined_text = "\n\n---\n\n".join(search_texts[:5])  # Use top 5 results
                
                extraction_prompt = ChatPromptTemplate.from_messages([
                    ("system", (
                        "You are a stock ticker extraction expert. Extract stock ticker symbols from search results. "
                        "Look for ticker symbols (1-5 uppercase letters like AAPL, NVDA, MSFT) mentioned in the context. "
                        "Return tickers that are mentioned as buy opportunities, oversold stocks, or swing trading setups."
                    )),
                    ("human", (
                        f"Search Results:\n{combined_text}\n\n"
                        f"Extract stock ticker symbols mentioned in these search results. "
                        f"For each ticker, provide: ticker symbol, company name, why it was mentioned, potential reason, and relevance score (1-10). "
                        f"Focus on stocks mentioned for swing trading, oversold conditions, or buy opportunities."
                    ))
                ])
                
                chain = extraction_prompt | self.llm.with_structured_output(StockOpportunitiesSchema)
                result = chain.invoke({})
                
                # Extract stocks from LLM response
                opportunities = []
                if hasattr(result, 'stocks') and isinstance(result.stocks, list):
                    for stock in result.stocks:
                        if isinstance(stock, dict):
                            try:
                                opp = StockOpportunity(**stock)
                                opportunities.append(opp)
                            except Exception as e:
                                logger.warning(f"⚠️ [SCOUTING] Invalid stock data: {e}")
                        elif isinstance(stock, StockOpportunity):
                            opportunities.append(stock)
                elif isinstance(result, dict) and 'stocks' in result:
                    for stock in result['stocks']:
                        try:
                            opp = StockOpportunity(**stock) if isinstance(stock, dict) else stock
                            opportunities.append(opp)
                        except Exception as e:
                            logger.warning(f"⚠️ [SCOUTING] Invalid stock data: {e}")
                
                if opportunities:
                    logger.info(f"✅ [SCOUTING] Search + LLM discovered {len(opportunities)} stocks: {[s.ticker for s in opportunities[:10]]}")
                    return opportunities
                else:
                    logger.warning("⚠️ [SCOUTING] LLM extraction returned no stocks")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                    return []
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 * (attempt + 1)
                    logger.warning(f"⚠️ [SCOUTING] Search attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ [SCOUTING] All search attempts failed: {e}", exc_info=True)
        
        return []
    
    async def _rank_tickers_by_strategy(self, opportunities: List[StockOpportunity], strategy_name: str, strategy_config: Dict[str, Any], max_retries: int = 3) -> List[str]:
        """Rank tickers based on strategy criteria and select top 5.
        
        Uses LLM to reason about each ticker and rank them based on how well
        they match the trading strategy.
        
        Args:
            opportunities: List of discovered stock opportunities
            strategy_name: Name of the strategy
            strategy_config: Strategy configuration
            max_retries: Maximum retry attempts
            
        Returns:
            List of top 5 ticker symbols (ranked)
        """
        if not opportunities:
            return []
        
        # Build strategy context
        rsi_threshold = strategy_config.get('rsi_threshold', 35)
        ai_score_required = strategy_config.get('ai_score_required', 7)
        description = strategy_config.get('description', 'swing trading opportunities')
        
        ticker_list = "\n".join([
            f"- {o.ticker} ({o.company_name}): {o.why_mentioned} | Potential: {o.potential_reason} [Relevance: {o.relevance_score}/10]"
            for o in opportunities[:15]  # Limit to top 15 for ranking
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a trading strategy expert. Rank tickers based on how well they match "
                "a specific trading strategy. Consider the strategy criteria and each ticker's context. "
                "Select the top 5 tickers that best match the strategy."
            )),
            ("human", (
                f"Strategy: {strategy_name}\n"
                f"Description: {description}\n"
                f"RSI Threshold: < {rsi_threshold} (oversold)\n"
                f"AI Score Required: ≥ {ai_score_required}\n"
                f"Type: Swing Trading (multi-day to multi-week positions)\n\n"
                f"Extracted Tickers:\n{ticker_list}\n\n"
                "Rank these tickers based on how well they match the strategy. "
                "Consider: relevance to query, potential for oversold bounce, uptrend potential, "
                "swing trading suitability. Select top 5 and provide reasoning for each."
            ))
        ])
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                chain = prompt | self.llm.with_structured_output(TickerRankingList)
                result = chain.invoke({})
                
                # Extract ranked tickers
                rankings = []
                if hasattr(result, 'rankings') and isinstance(result.rankings, list):
                    rankings = result.rankings
                elif isinstance(result, dict) and 'rankings' in result:
                    rankings = [TickerRanking(**r) if isinstance(r, dict) else r for r in result['rankings']]
                
                if rankings:
                    # Sort by rank and extract tickers
                    rankings.sort(key=lambda x: x.rank)
                    top_tickers = [r.ticker for r in rankings[:5]]
                    logger.info(f"✅ [SCOUTING] Ranked and selected top {len(top_tickers)} tickers: {top_tickers}")
                    for r in rankings[:5]:
                        logger.info(f"   {r.rank}. {r.ticker} (score: {r.match_score:.1f}) - {r.reasoning[:80]}")
                    return top_tickers
                else:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    # Fallback: return top tickers by relevance score
                    sorted_opps = sorted(opportunities, key=lambda x: x.relevance_score, reverse=True)
                    return [o.ticker for o in sorted_opps[:5]]
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"⚠️ [SCOUTING] Ranking attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ [SCOUTING] All ranking attempts failed: {e}", exc_info=True)
                    # Fallback: return top tickers by relevance score
                    sorted_opps = sorted(opportunities, key=lambda x: x.relevance_score, reverse=True)
                    return [o.ticker for o in sorted_opps[:5]]
        
        return []
    
    async def discover_symbols(
        self,
        strategy_name: Optional[str] = None,
        strategy_config: Optional[Dict[str, Any]] = None,
        max_symbols: int = 5
    ) -> List[str]:
        """Discover symbols intelligently based on strategy - Scouting Tool.
        
        This is the main entry point for intelligent symbol discovery.
        Pipeline: Query -> Search -> Extract Tickers -> Rank -> Select Top 5
        
        Args:
            strategy_name: Optional strategy name (defaults to CURRENT_STRATEGY)
            strategy_config: Optional strategy config (defaults to empty dict)
            max_symbols: Maximum number of symbols to return (default: 5)
            
        Returns:
            List of ticker symbols (ready to use)
        """
        try:
            # Use defaults if not provided
            if strategy_name is None:
                strategy_name = CURRENT_STRATEGY
            if strategy_config is None:
                strategy_config = {}
            
            logger.info(f"🔍 [SCOUTING] Starting intelligent symbol discovery for strategy: {strategy_name}")
            
            # Step 1: Generate search query
            search_query = self._generate_search_query(strategy_name, strategy_config)
            logger.info(f"📝 [SCOUTING] Generated search query: '{search_query[:100]}...'")
            
            # Step 2: Use Firecrawl Search + LLM to discover stocks (faster than Agent)
            # Search API is much faster and doesn't freeze the system
            opportunities = await self._discover_stocks_with_search(search_query)
            if not opportunities:
                logger.warning("⚠️ [SCOUTING] Agent found no stocks, returning empty list")
                return []
            
            # Step 3: Rank opportunities based on strategy and select top 5
            top_tickers = await self._rank_tickers_by_strategy(opportunities, strategy_name, strategy_config)
            if not top_tickers:
                logger.warning("⚠️ [SCOUTING] No tickers ranked, returning empty list")
                return []
            
            # Step 5: Limit and return
            final_symbols = top_tickers[:max_symbols]
            logger.info(f"✅ [SCOUTING] Discovered {len(final_symbols)} symbols: {final_symbols}")
            return final_symbols
            
        except Exception as e:
            logger.error(f"❌ [SCOUTING] Error in discover_symbols: {e}", exc_info=True)
            return []
