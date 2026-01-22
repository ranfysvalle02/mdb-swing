"""Input validation helpers for API routes."""
import re
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from fastapi import HTTPException, status


def validate_symbol(symbol: str) -> str:
    """Validate and normalize stock symbol."""
    if not symbol:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Symbol cannot be empty"
        )
    
    symbol = symbol.strip().upper()
    
    if not re.match(r'^[A-Z0-9]{1,5}$', symbol):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid symbol format: {symbol}. Must be 1-5 alphanumeric characters."
        )
    
    return symbol


def validate_symbols(symbols: str, max_count: int = 5) -> List[str]:
    """Validate and normalize comma-separated stock symbols."""
    if not symbols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Symbols cannot be empty"
        )
    
    symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    
    if not symbol_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one symbol is required"
        )
    
    if len(symbol_list) > max_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {max_count} symbols allowed"
        )
    
    validated_symbols = []
    for symbol in symbol_list:
        validated_symbols.append(validate_symbol(symbol))
    
    return validated_symbols


def validate_price(price: float, min_price: float = 0.01, max_price: float = 1000000.0) -> float:
    """Validate price value."""
    if price is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Price cannot be empty"
        )
    
    try:
        price_float = float(price)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid price format: {price}"
        )
    
    if price_float < min_price or price_float > max_price:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Price must be between ${min_price:.2f} and ${max_price:.2f}"
        )
    
    return round(price_float, 2)


def validate_quantity(qty: int, min_qty: int = 1, max_qty: int = 10000) -> int:
    """Validate share quantity."""
    if qty is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Quantity cannot be empty"
        )
    
    try:
        qty_int = int(qty)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid quantity format: {qty}"
        )
    
    if qty_int < min_qty or qty_int > max_qty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Quantity must be between {min_qty} and {max_qty}"
        )
    
    return qty_int


class TradeRequest(BaseModel):
    """Request model for trade execution."""
    ticker: str = Field(..., min_length=1, max_length=5, description="Stock symbol")
    action: str = Field(..., regex="^(BUY|SELL)$", description="Trade action")
    
    @validator('ticker')
    def validate_ticker(cls, v):
        return validate_symbol(v)
    
    @validator('action')
    def validate_action(cls, v):
        if v.upper() not in ('BUY', 'SELL'):
            raise ValueError("Action must be BUY or SELL")
        return v.upper()


class QuickBuyRequest(BaseModel):
    """Request model for quick buy."""
    symbol: str = Field(..., min_length=1, max_length=5, description="Stock symbol")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return validate_symbol(v)


class QuickSellRequest(BaseModel):
    """Request model for quick sell."""
    symbol: str = Field(..., min_length=1, max_length=5, description="Stock symbol")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return validate_symbol(v)


class WatchListRequest(BaseModel):
    """Request model for watch list update."""
    symbols: str = Field(..., min_length=1, description="Comma-separated stock symbols")
    
    @validator('symbols')
    def validate_symbols(cls, v):
        validated = validate_symbols(v, max_count=5)
        return ','.join(validated)


class AnalyzeRequest(BaseModel):
    """Request model for symbol analysis."""
    ticker: str = Field(..., min_length=1, max_length=5, description="Stock symbol")
    
    @validator('ticker')
    def validate_ticker(cls, v):
        return validate_symbol(v)
