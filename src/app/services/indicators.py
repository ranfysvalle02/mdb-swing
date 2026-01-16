"""Technical indicators implementation - stateless utility class for calculating indicators."""
import pandas as pd
import numpy as np
from typing import Optional


class TechnicalIndicators:
    """
    A stateless utility class for calculating technical indicators.
    Optimized for Pandas Series/DataFrames.
    """

    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        """
        Calculates Simple Moving Average (SMA).
        
        Args:
            series: Pandas Series of prices.
            period: The window size for the moving average.
            
        Returns:
            Pandas Series containing the SMA. Returns NaN for periods with insufficient data.
        """
        if len(series) == 0:
            return pd.Series(dtype=float)
        
        # Use rolling window - it will return NaN for periods with insufficient data
        # This is the expected behavior for moving averages
        return series.rolling(window=min(period, len(series)), min_periods=1).mean()

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculates Relative Strength Index (RSI) using Wilder's Smoothing.
        
        Note: The first average is a simple mean, subsequent are smoothed.
        Formula: RSI = 100 - (100 / (1 + RS))
        
        Args:
            series: Pandas Series of prices (usually Close).
            period: The lookback period (standard is 14).
            
        Returns:
            Pandas Series (0-100).
        """
        delta = series.diff()

        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)

        # Calculate initial Simple Average for the first period
        # We use a trick with ewm(alpha=1/period) to emulate Wilder's Smoothing
        # accurately after the initial mean.
        alpha = 1.0 / period
        
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Clean up initial NaN values
        return rsi

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculates Average True Range (ATR).
        
        Args:
            df: DataFrame containing 'high', 'low', 'close' columns (case-insensitive).
            period: The smoothing period.
            
        Returns:
            Pandas Series representing volatility.
        """
        df = df.rename(columns=str.lower)
        required_cols = {'high', 'low', 'close'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)

        # True Range is the max of three distinct value differences
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR is usually a Smoothed Moving Average of the TR
        # Wilder's Smoothing: alpha = 1/period
        atr = true_range.ewm(alpha=1/period, adjust=False).mean()
        
        return atr


# Convenience functions for backward compatibility
def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average - convenience wrapper."""
    return TechnicalIndicators.calculate_sma(series, length)


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index - convenience wrapper."""
    return TechnicalIndicators.calculate_rsi(series, length)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range - convenience wrapper."""
    df = pd.DataFrame({'high': high, 'low': low, 'close': close})
    return TechnicalIndicators.calculate_atr(df, length)
