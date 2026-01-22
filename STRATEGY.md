# Balanced Low Swing Trading Strategy Guide

**A comprehensive guide to using VIEW ALL STATS metrics to make better trading decisions**

---

## Table of Contents

1. [Strategy Overview](#strategy-overview)
2. [Core Entry Criteria](#core-entry-criteria)
3. [Understanding VIEW ALL STATS Metrics](#understanding-view-all-stats-metrics)
4. [Decision-Making Framework](#decision-making-framework)
5. [Practical Examples](#practical-examples)
6. [Advanced Techniques](#advanced-techniques)
7. [Common Mistakes to Avoid](#common-mistakes-to-avoid)
8. [Quick Reference Guide](#quick-reference-guide)

---

## Strategy Overview

### What is Balanced Low Swing Trading?

**Balanced Low** is a swing trading strategy focused on finding stocks that are temporarily "on sale" (oversold) but still in a healthy uptrend. This is NOT buying crashes or distressed stocks — it's buying quality stocks during healthy pullbacks.

### Core Philosophy

> *"The lawn is overgrown because the owner was lazy. The foundation is solid. The house is cheap. This is a good buy."*

We're looking for stocks that:
- Are **oversold** (temporarily down) but not in freefall
- Are in an **uptrend** (foundation is solid)
- Are **near support** (SMA-200) — ready to bounce back
- Have **catalysts** (earnings, news, sector rotation) — triggers for the bounce

### Time Horizon

- **Swing Trading**: Days to weeks (not day trading, not long-term investing)
- Uses **daily bars** (not intraday data)
- Positions held until **stop loss** or **take profit** hit
- Typical hold period: 3-14 days, sometimes longer

### What This Strategy Is NOT

**NOT Day Trading**:
- ❌ Does not use intraday data
- ❌ Does not scalp quick moves
- ❌ Does not exit same day
- ❌ Does not require minute-by-minute monitoring

**NOT Long-Term Investing**:
- ❌ Does not buy and hold for years
- ❌ Has defined exits (stop loss and take profit)
- ❌ Uses active risk management
- ❌ Focuses on swing moves, not long-term trends

**THIS IS Swing Trading**: Capturing multi-day to multi-week moves in stocks at balanced lows.

---

## Core Entry Criteria

The Balanced Low strategy requires **all four** of these conditions to be met:

### 1. RSI Sweet Spot (20-35)

**What it means**: Stock is oversold but not extreme.

- **RSI < 20**: Too extreme — may indicate fundamental issues, avoid
- **RSI 20-35**: **Sweet spot** — oversold enough to bounce, not so extreme it's broken
- **RSI > 35**: Not oversold — wait for pullback

**Why it matters**: 
- RSI < 20 can mean the stock is fundamentally broken
- RSI 20-35 means temporary oversold condition (perfect for bounce)
- RSI > 35 means not enough discount for entry

**In VIEW ALL STATS**: Check the **RSI (14)** value in Momentum Indicators section.

### 2. Uptrend Confirmation (Price > SMA-200)

**What it means**: Stock is in a long-term uptrend.

- **Price > SMA-200**: Uptrend confirmed ✅
- **Price < SMA-200**: Downtrend — avoid ❌

**Why it matters**:
- Uptrend = foundation is solid
- Downtrend = stock may continue falling
- We want to buy pullbacks in uptrends, not catch falling knives

**In VIEW ALL STATS**: 
- Check **SMA-200** value in Moving Averages section
- Check **Price vs SMA-200 %** — should be positive
- Check **Trend** indicator — should be "UP"

### 3. Support Proximity (within 3% of SMA-200)

**What it means**: Price is near the SMA-200 support level.

- **0-3% above SMA-200**: **Ideal** — near support, high bounce probability
- **3-5% above SMA-200**: Acceptable but less ideal
- **> 5% above SMA-200**: Too far from support — wait for pullback

**Why it matters**:
- SMA-200 acts as support in uptrends
- Price near support = higher probability of bounce
- Price far from support = more downside risk before bounce

**In VIEW ALL STATS**: 
- Calculate: `(Price - SMA-200) / SMA-200 × 100`
- Or check **Price vs SMA-200 %** if displayed
- Should be between 0% and 3%

### 4. AI Score Validation (≥ 7)

**What it means**: AI analysis confirms bounce-back probability.

- **Score 7-10**: Strong bounce-back setup ✅
- **Score 4-6**: Decent potential, but missing key factors ⚠️
- **Score 0-3**: Weak or no bounce potential ❌

**Why it matters**:
- AI analyzes news, catalysts, and technicals together
- Validates that bounce-back probability is high
- Considers factors beyond just technical indicators

**In VIEW ALL STATS**: Not shown in stats view — check AI analysis results.

---

## Understanding VIEW ALL STATS Metrics

When you click **VIEW ALL STATS**, you see comprehensive technical indicators. Here's how to interpret each metric and use it to influence your trading decisions.

### Price & Volume

#### Price
- **What it shows**: Latest closing price
- **How to use**: 
  - Compare to moving averages (SMA-20, SMA-50, SMA-100, SMA-200)
  - Compare to Bollinger Bands (upper, middle, lower)
  - Check if price is near support levels

#### Change %
- **What it shows**: Daily price movement percentage
- **How to use**:
  - Green (positive) = up today — may indicate momentum
  - Red (negative) = down today — pullback opportunity
  - Large moves (> 5%) = high volatility, wider stops needed

#### Volume
- **What it shows**: Shares traded today
- **How to use**: 
  - High volume = strong move (institutional interest)
  - Low volume = weak move (may reverse)

#### Volume Ratio
- **What it shows**: Current volume vs 20-day average
- **How to use**:
  - **> 1.5x**: High volume — strong move, confirms direction
  - **0.5x - 1.5x**: Normal volume — neutral
  - **< 0.5x**: Low volume — weak move, may reverse

**Decision Influence**:
- ✅ **High volume on pullback** = Strong support, good entry
- ⚠️ **Low volume on pullback** = Weak support, wait for confirmation
- ✅ **High volume on bounce** = Confirmation of move

---

### Momentum Indicators

#### RSI (14) — Primary Oversold Indicator

**What it shows**: Relative Strength Index — measures momentum on 0-100 scale

**Interpretation**:
- **< 30**: Oversold (buy signal) ✅
- **20-35**: **Sweet spot for Balanced Low** ✅✅
- **35-70**: Neutral
- **> 70**: Overbought (sell signal) ❌

**How to use**:
- RSI < 30 = oversold condition
- RSI 20-35 = perfect for Balanced Low entry
- RSI < 20 = too extreme, may indicate fundamental issues
- RSI > 70 = overbought, avoid entry

**Decision Influence**:
- ✅ **RSI 25-35** = Ideal entry zone
- ⚠️ **RSI 20-25** = Good but watch for fundamental issues
- ❌ **RSI < 20** = Too extreme, avoid
- ❌ **RSI > 35** = Not oversold, wait for pullback

#### Stochastic %K & %D — Momentum Confirmation

**What it shows**: Stochastic Oscillator — confirms RSI signals

**Interpretation**:
- **< 20**: Oversold (buy signal) ✅
- **20-80**: Neutral
- **> 80**: Overbought (sell signal) ❌
- **%K crosses above %D**: Bullish momentum ✅

**How to use**:
- Stochastic < 20 = oversold confirmation
- %K > %D = bullish momentum building
- Use with RSI for stronger signals

**Decision Influence**:
- ✅ **RSI < 30 + Stochastic < 20** = Strong oversold signal (convergence)
- ✅ **%K crosses above %D** = Momentum shift, bullish
- ⚠️ **RSI oversold but Stochastic neutral** = Weaker signal, wait

**Signal Convergence Method**:
- Multiple momentum indicators pointing same direction = stronger signal
- RSI < 30 + Stochastic < 20 = Strong buy signal
- RSI > 70 + Stochastic > 80 = Strong sell signal (avoid entry)

---

### MACD — Momentum Trend Indicator

#### MACD Line
- **What it shows**: Fast (12-day EMA) minus slow (26-day EMA) momentum
- **Positive** = Bullish momentum ✅
- **Negative** = Bearish momentum ❌

#### Signal Line
- **What it shows**: 9-day EMA of MACD line
- **MACD > Signal** = Bullish momentum ✅
- **MACD < Signal** = Bearish momentum ❌

#### Histogram
- **What it shows**: MACD - Signal (momentum strength)
- **Positive** = Momentum increasing ✅
- **Negative** = Momentum decreasing ❌

**How to use**:
- **MACD > Signal** = Bullish momentum (buy signal)
- **MACD < Signal** = Bearish momentum (avoid entry)
- **Histogram turning positive** = Momentum shift, bullish
- **Histogram positive and increasing** = Strong momentum

**Decision Influence**:
- ✅ **MACD > Signal + Positive Histogram** = Strong bullish momentum
- ✅ **MACD crossing above Signal** = Momentum shift, good entry
- ⚠️ **MACD < Signal but improving** = Wait for crossover
- ❌ **MACD < Signal + Negative Histogram** = Avoid entry

**Combined with RSI**:
- ✅ **RSI oversold + MACD turning positive** = Strong buy signal
- ✅ **RSI oversold + MACD > Signal** = Momentum confirmation
- ⚠️ **RSI oversold + MACD negative** = Wait for MACD improvement

---

### Moving Averages — Trend Confirmation

#### SMA-200 (200-Day Simple Moving Average)
- **What it shows**: Long-term trend direction
- **Price > SMA-200** = Uptrend ✅ (required for Balanced Low)
- **Price < SMA-200** = Downtrend ❌ (avoid entry)

#### SMA-50, SMA-100
- **What it shows**: Medium-term trend
- **Price > SMA-50** = Medium-term uptrend ✅
- **Price < SMA-50** = Medium-term downtrend ⚠️

#### EMA-12, EMA-26
- **What it shows**: Short-term momentum (used in MACD)
- **EMA-12 > EMA-26** = Short-term bullish ✅
- **EMA-12 < EMA-26** = Short-term bearish ❌

#### Price vs SMA %
- **What it shows**: Distance from moving average (support/resistance)
- **0-3% above SMA-200** = Near support ✅ (ideal for Balanced Low)
- **3-5% above SMA-200** = Acceptable ⚠️
- **> 5% above SMA-200** = Too far from support ❌

**How to use**:
- **SMA-200 alignment**: Price > SMA-200 = uptrend (required)
- **Support proximity**: Price within 0-3% of SMA-200 = near support (ideal)
- **Multiple MAs aligned**: Price > SMA-200 > SMA-100 > SMA-50 = strong uptrend
- **MA crossover**: EMA-12 crossing above EMA-26 = bullish momentum

**Decision Influence**:
- ✅ **Price > SMA-200 + within 3%** = Perfect Balanced Low setup
- ✅ **Price > SMA-200 + SMA-200 > SMA-100 > SMA-50** = Strong uptrend
- ⚠️ **Price > SMA-200 but > 5% above** = Wait for pullback to support
- ❌ **Price < SMA-200** = Downtrend, avoid entry

---

### Bollinger Bands — Volatility & Overbought/Oversold

#### Upper/Lower Bands
- **What it shows**: Volatility-based overbought/oversold levels
- **Price touching upper band** = Overbought (sell signal) ❌
- **Price touching lower band** = Oversold (buy signal) ✅

#### %B Position
- **What it shows**: Where price sits in band (0-100%)
- **< 20%** = Oversold (buy signal) ✅
- **20-80%** = Neutral
- **> 80%** = Overbought (sell signal) ❌

#### Band Width
- **What it shows**: Volatility measure
- **< 3%** = Low volatility, expect breakout ✅
- **3-5%** = Normal volatility
- **> 5%** = High volatility, expect consolidation ⚠️

**How to use**:
- **%B < 20%** = Oversold, buy signal
- **Price touching lower band** = Support level
- **Narrow bands** = Low volatility, expect breakout soon
- **Wide bands** = High volatility, expect consolidation

**Decision Influence**:
- ✅ **%B < 20% + Price near lower band** = Strong oversold signal
- ✅ **Narrow bands (< 3%)** = Low volatility, tighter stops possible
- ⚠️ **Wide bands (> 5%)** = High volatility, wider stops needed
- ❌ **%B > 80%** = Overbought, avoid entry

**Combined with RSI**:
- ✅ **RSI < 30 + %B < 20%** = Strong oversold convergence
- ✅ **RSI oversold + Price touching lower Bollinger Band** = Support level confirmed

---

### Volatility & Range

#### ATR (14) — Average True Range
- **What it shows**: Average price movement (volatility measure)
- **Higher ATR** = More volatile (wider stops needed)
- **Lower ATR** = Less volatile (tighter stops possible)

#### ATR %
- **What it shows**: Volatility relative to price
- **< 2%** = Low volatility ✅
- **2-5%** = Normal volatility
- **> 5%** = High volatility ⚠️

**How to use for stops**:
- **Stop Loss**: Entry - (2 × ATR)
- **Take Profit**: Entry + (3 × ATR)
- **Position Sizing**: Based on ATR volatility

**Decision Influence**:
- ✅ **Low ATR% (< 2%)** = Tighter stops, lower risk per trade
- ⚠️ **High ATR% (> 5%)** = Wider stops needed, higher risk
- ✅ **ATR for position sizing** = Risk $50 per trade based on ATR

#### 20-Day High/Low
- **What it shows**: Recent price range
- **20-Day High** = Recent resistance level
- **20-Day Low** = Recent support level

#### Price Position in Range
- **What it shows**: Where price sits in 20-day range (0-100%)
- **< 30%** = Near bottom (buy signal) ✅
- **30-70%** = Middle of range
- **> 70%** = Near top (sell signal) ❌

**How to use**:
- **Price Position < 30%** = Near bottom, good entry
- **Price Position > 70%** = Near top, avoid entry
- **20-Day Low** = Recent support level

**Decision Influence**:
- ✅ **Price Position < 30%** = Near bottom, buy signal
- ✅ **Price near 20-Day Low** = Support level, good entry
- ⚠️ **Price Position 30-70%** = Middle of range, neutral
- ❌ **Price Position > 70%** = Near top, avoid entry

**Combined Support Analysis**:
- ✅ **Price Position < 30% + Price near SMA-200 + Price near Bollinger Lower Band** = Multiple support levels, strong buy signal

---

## Decision-Making Framework

### Signal Convergence Method

**Principle**: Multiple indicators pointing the same direction = stronger signal

**How to use**:
1. Count how many indicators confirm the signal
2. More confirmations = stronger signal
3. Weight the signals: Core indicators (RSI, SMA-200) > Supporting indicators

**Example Strong Buy Signal**:
- ✅ RSI < 30 (oversold)
- ✅ Stochastic < 20 (oversold confirmation)
- ✅ %B < 20% (Bollinger oversold)
- ✅ Price Position < 30% (near bottom)
- ✅ Price > SMA-200 (uptrend)
- ✅ Price within 3% of SMA-200 (near support)
- ✅ MACD > Signal (bullish momentum)
- ✅ Volume Ratio > 1.5x (high volume)

**Result**: 8 confirmations = **Very Strong Buy Signal**

**Example Weak Signal**:
- ✅ RSI < 30 (oversold)
- ❌ Stochastic > 30 (not oversold)
- ⚠️ %B 35% (neutral)
- ⚠️ Price Position 45% (middle of range)
- ✅ Price > SMA-200 (uptrend)
- ❌ Price 6% above SMA-200 (too far from support)
- ❌ MACD < Signal (bearish momentum)
- ❌ Volume Ratio < 0.8x (low volume)

**Result**: 2 confirmations = **Weak Signal — Wait**

---

### Support Level Analysis

**Principle**: Price near multiple support levels = higher bounce probability

**Primary Support**: SMA-200
- Price within 0-3% of SMA-200 = near primary support ✅

**Secondary Support**: Bollinger Lower Band
- Price touching lower band = secondary support ✅

**Recent Support**: 20-Day Low
- Price near 20-day low = recent support level ✅

**How to use**:
1. Identify all support levels
2. Check if price is near multiple supports
3. More supports = stronger bounce probability

**Example**:
- Price: $100
- SMA-200: $98 (2% above — primary support) ✅
- Bollinger Lower: $99.50 (near lower band — secondary support) ✅
- 20-Day Low: $99 (near recent low — recent support) ✅

**Result**: Price near **3 support levels** = **Very High Bounce Probability**

---

### Momentum Confirmation

**Principle**: Oversold + Momentum shift = Strong buy signal

**Key Combinations**:

1. **RSI Oversold + MACD Turning Positive**
   - RSI < 30 (oversold)
   - MACD crossing above Signal (momentum shift)
   - **Result**: Strong buy signal ✅

2. **Stochastic Oversold + MACD Histogram Positive**
   - Stochastic < 20 (oversold)
   - MACD Histogram > 0 (momentum building)
   - **Result**: Momentum confirmation ✅

3. **Volume Spike on Pullback**
   - High volume on pullback = institutional interest
   - **Result**: Strong support, good entry ✅

**Example**:
- RSI: 28 (oversold) ✅
- MACD: 0.3, Signal: 0.1 (MACD > Signal, positive) ✅
- MACD Histogram: 0.2 (positive, momentum increasing) ✅
- Volume Ratio: 1.8x (high volume) ✅

**Result**: **Strong Momentum Confirmation** — all indicators align

---

### Risk Assessment Using Metrics

**High Risk Indicators**:
- ❌ High ATR% (> 5%) = Wider stops needed
- ❌ Low volume (< 0.5x avg) = Weak move, may reverse
- ❌ Price far from SMA-200 (> 5%) = Not near support
- ❌ Multiple indicators overbought = Avoid entry
- ❌ MACD strongly negative = Bearish momentum

**Low Risk Indicators**:
- ✅ Low ATR% (< 2%) = Tighter stops possible
- ✅ High volume (> 1.5x avg) = Strong move
- ✅ Price near SMA-200 (0-3%) = Near support
- ✅ Multiple indicators oversold = Good entry
- ✅ MACD positive = Bullish momentum

**Risk Score Calculation**:
- Count risk factors (high risk = +1, low risk = -1)
- Lower score = lower risk
- Only enter if risk score is acceptable

**Example Low Risk Setup**:
- ATR%: 1.8% (low volatility) ✅
- Volume Ratio: 1.6x (high volume) ✅
- Price vs SMA-200: 2.1% (near support) ✅
- RSI: 28 (oversold) ✅
- MACD: 0.4 (positive) ✅

**Risk Score**: -5 (very low risk) ✅

**Example High Risk Setup**:
- ATR%: 6.2% (high volatility) ❌
- Volume Ratio: 0.6x (low volume) ❌
- Price vs SMA-200: 7.5% (far from support) ❌
- RSI: 45 (not oversold) ❌
- MACD: -0.8 (negative) ❌

**Risk Score**: +5 (high risk) ❌ — **Avoid Entry**

---

## Practical Examples

### Example 1: Strong Buy Signal

**Scenario**: Stock meets all Balanced Low criteria with multiple confirmations

**Metrics**:
- **Price**: $100.50
- **SMA-200**: $98.20 (Price 2.3% above — near support) ✅
- **RSI**: 28 (sweet spot, oversold) ✅
- **Stochastic %K**: 18, %D: 20 (%K < %D but both oversold) ✅
- **MACD**: 0.5, Signal: 0.2 (MACD > Signal, positive) ✅
- **MACD Histogram**: 0.3 (positive, momentum increasing) ✅
- **%B Position**: 15% (oversold) ✅
- **Price Position in Range**: 25% (near bottom) ✅
- **Volume Ratio**: 1.8x (high volume on pullback) ✅
- **ATR**: $2.10, ATR%: 2.1% (normal volatility) ✅
- **Trend**: UP (Price > SMA-200) ✅

**Analysis**:
- ✅ RSI in sweet spot (28)
- ✅ Price near SMA-200 support (2.3% above)
- ✅ Multiple momentum confirmations (RSI, Stochastic, MACD)
- ✅ Multiple support levels (SMA-200, Bollinger Lower, 20-Day Low)
- ✅ High volume confirms move
- ✅ Normal volatility (manageable risk)

**Decision**: **STRONG BUY** — 9 confirmations, low risk

**Action**: 
- Enter position
- Stop Loss: $100.50 - (2 × $2.10) = $96.30
- Take Profit: $100.50 + (3 × $2.10) = $106.80

---

### Example 2: Wait Signal

**Scenario**: Stock partially meets criteria but missing key factors

**Metrics**:
- **Price**: $105.00
- **SMA-200**: $100.00 (Price 5% above — too far from support) ⚠️
- **RSI**: 32 (sweet spot, oversold) ✅
- **Stochastic %K**: 35, %D: 38 (neutral, not oversold) ⚠️
- **MACD**: -0.2, Signal: -0.1 (negative, but improving) ⚠️
- **MACD Histogram**: -0.1 (negative, but improving) ⚠️
- **%B Position**: 35% (neutral) ⚠️
- **Price Position in Range**: 45% (middle of range) ⚠️
- **Volume Ratio**: 0.8x (low volume) ❌
- **ATR**: $2.50, ATR%: 2.4% (normal volatility) ✅
- **Trend**: UP (Price > SMA-200) ✅

**Analysis**:
- ✅ RSI in sweet spot (32)
- ✅ Uptrend confirmed
- ⚠️ Price too far from SMA-200 (5% vs 3% target)
- ⚠️ Stochastic not oversold (no confirmation)
- ⚠️ MACD negative (no momentum confirmation)
- ⚠️ Low volume (weak move)
- ⚠️ Neutral Bollinger position

**Decision**: **WAIT** — Missing support proximity and volume confirmation

**Action**: 
- Monitor for pullback to SMA-200 (target: $100-103)
- Wait for volume confirmation (> 1.2x avg)
- Wait for MACD to turn positive
- Re-evaluate when price gets within 3% of SMA-200

---

### Example 3: Avoid Signal

**Scenario**: Stock appears oversold but in downtrend with negative momentum

**Metrics**:
- **Price**: $95.00
- **SMA-200**: $105.00 (Price 9.5% below — downtrend) ❌
- **RSI**: 18 (too extreme, may indicate fundamental issues) ❌
- **Stochastic %K**: 12, %D: 15 (extremely oversold) ⚠️
- **MACD**: -1.5, Signal: -1.0 (strongly negative) ❌
- **MACD Histogram**: -0.5 (negative, momentum decreasing) ❌
- **%B Position**: 5% (extremely oversold) ⚠️
- **Price Position in Range**: 10% (at bottom, but in downtrend) ⚠️
- **Volume Ratio**: 2.5x (high volume on decline) ❌
- **ATR**: $3.20, ATR%: 3.4% (elevated volatility) ⚠️
- **Trend**: DOWN (Price < SMA-200) ❌

**Analysis**:
- ❌ Downtrend (Price < SMA-200) — violates core requirement
- ❌ RSI too extreme (18) — may indicate fundamental issues
- ❌ Strongly negative MACD — bearish momentum
- ❌ High volume on decline — selling pressure
- ⚠️ Extremely oversold — but in downtrend, not uptrend

**Decision**: **AVOID** — Downtrend, extreme oversold, negative momentum

**Action**: 
- Do not enter
- Stock may continue falling despite oversold condition
- Wait for trend reversal (Price > SMA-200) before considering entry

---

## Advanced Techniques

### Divergence Analysis

**Bullish Divergence**:
- Price making **lower lows**
- RSI making **higher lows**
- **Interpretation**: Momentum improving despite price decline
- **Use**: Confirm with other metrics before entry

**Bearish Divergence**:
- Price making **higher highs**
- RSI making **lower highs**
- **Interpretation**: Momentum weakening despite price rise
- **Use**: Warning sign, consider taking profits

**How to use**:
- Divergence alone is not enough — use with other confirmations
- Bullish divergence + oversold + near support = Strong buy signal
- Bearish divergence + overbought + far from support = Avoid entry

---

### Multi-Timeframe Confirmation

**Daily Bars** (Primary):
- Use for swing trading entry signals
- Check all VIEW ALL STATS metrics on daily timeframe

**Weekly Bars** (Confirmation):
- Check if weekly trend aligns with daily trend
- Weekly uptrend + Daily oversold = Strong setup

**How to use**:
- Daily RSI oversold + Weekly uptrend = Strong buy signal
- Daily oversold but Weekly downtrend = Weaker signal, be cautious

---

### Volume Profile Analysis

**High Volume on Pullback**:
- Volume > 1.5x average on down day
- **Interpretation**: Institutional support, strong bounce probability ✅

**Low Volume on Pullback**:
- Volume < 0.5x average on down day
- **Interpretation**: Weak support, may continue falling ⚠️

**Volume Spike on Bounce**:
- Volume > 2x average on up day
- **Interpretation**: Confirmation of move, strong momentum ✅

**How to use**:
- High volume pullback = Good entry opportunity
- Low volume pullback = Wait for volume confirmation
- Volume spike on bounce = Hold position, momentum confirmed

---

## Common Mistakes to Avoid

### 1. Relying on Single Indicator
- **Mistake**: Entering based only on RSI < 30
- **Fix**: Check multiple indicators for convergence
- **Better**: RSI + Stochastic + %B + Price Position all oversold

### 2. Ignoring Volume Confirmation
- **Mistake**: Entering on low volume pullback
- **Fix**: Wait for volume confirmation (> 1.2x average)
- **Better**: High volume pullback = stronger support

### 3. Entering When Price Too Far from SMA-200
- **Mistake**: Entering when price 8% above SMA-200
- **Fix**: Wait for pullback to within 3% of SMA-200
- **Better**: Price near support = higher bounce probability

### 4. Not Checking MACD Momentum
- **Mistake**: Entering when MACD strongly negative
- **Fix**: Wait for MACD to turn positive or at least improve
- **Better**: MACD > Signal + Positive Histogram = Momentum confirmation

### 5. Ignoring Bollinger Band Position
- **Mistake**: Entering when %B > 50% (middle of range)
- **Fix**: Wait for %B < 20% (oversold)
- **Better**: %B < 20% + Price touching lower band = Support confirmed

### 6. Not Considering ATR for Position Sizing
- **Mistake**: Using fixed stop loss regardless of volatility
- **Fix**: Use ATR for dynamic stop loss (Entry - 2×ATR)
- **Better**: Lower ATR% = tighter stops, better risk/reward

### 7. Entering in Downtrend
- **Mistake**: Entering when Price < SMA-200
- **Fix**: Only enter when Price > SMA-200 (uptrend)
- **Better**: Uptrend + Oversold = Balanced Low setup

### 8. Ignoring Multiple Support Levels
- **Mistake**: Only checking SMA-200
- **Fix**: Check all support levels (SMA-200, Bollinger Lower, 20-Day Low)
- **Better**: Price near multiple supports = stronger bounce probability

---

## Quick Reference Guide

### Entry Checklist

Use this checklist before entering a trade:

**Core Requirements** (All Required):
- [ ] RSI 20-35 (sweet spot)
- [ ] Price > SMA-200 (uptrend)
- [ ] Price within 3% of SMA-200 (near support)
- [ ] AI Score ≥ 7 (bounce-back probability)

**Momentum Confirmations** (At least 2):
- [ ] Stochastic < 20 (oversold)
- [ ] MACD > Signal (bullish momentum)
- [ ] MACD Histogram positive (momentum increasing)
- [ ] %K crosses above %D (bullish crossover)

**Support Confirmations** (At least 2):
- [ ] Price near SMA-200 (0-3% above)
- [ ] %B < 20% (Bollinger oversold)
- [ ] Price Position < 30% (near bottom)
- [ ] Price near 20-Day Low (recent support)

**Volume Confirmation**:
- [ ] Volume Ratio > 1.2x (high volume)

**Risk Assessment**:
- [ ] ATR% < 5% (manageable volatility)
- [ ] Not multiple overbought indicators
- [ ] No catastrophic news

**Decision**:
- **6+ checkmarks**: Strong buy signal ✅
- **4-5 checkmarks**: Decent signal, proceed with caution ⚠️
- **< 4 checkmarks**: Wait for better setup ❌

---

### Signal Strength Matrix

| Indicator | Strong Buy | Decent Buy | Wait | Avoid |
|-----------|------------|------------|------|-------|
| **RSI** | 25-35 | 20-25, 35-40 | 40-50 | < 20, > 50 |
| **Price vs SMA-200** | 0-2% | 2-3% | 3-5% | > 5%, < 0% |
| **MACD** | > Signal, Positive | > Signal, Near 0 | < Signal, Improving | < Signal, Negative |
| **%B Position** | < 15% | 15-25% | 25-40% | > 40% |
| **Price Position** | < 25% | 25-35% | 35-50% | > 50% |
| **Volume Ratio** | > 1.5x | 1.2-1.5x | 0.8-1.2x | < 0.8x |
| **Stochastic** | < 15 | 15-25 | 25-40 | > 40 |

**How to use**:
- Count "Strong Buy" signals
- Count "Decent Buy" signals
- **5+ Strong Buy** = Very Strong Signal ✅
- **3-4 Strong Buy + 2+ Decent Buy** = Good Signal ✅
- **< 3 Strong Buy** = Wait ⚠️

---

### Risk Levels

**Low Risk** (Green):
- ATR% < 2%
- Volume Ratio > 1.5x
- Price within 2% of SMA-200
- Multiple support levels
- Multiple momentum confirmations

**Medium Risk** (Yellow):
- ATR% 2-4%
- Volume Ratio 1.0-1.5x
- Price 2-4% above SMA-200
- Some support levels
- Some momentum confirmations

**High Risk** (Red):
- ATR% > 5%
- Volume Ratio < 0.8x
- Price > 5% above SMA-200
- No clear support levels
- Negative momentum indicators

**Decision**:
- ✅ **Low Risk**: Proceed with confidence
- ⚠️ **Medium Risk**: Proceed with caution, smaller position
- ❌ **High Risk**: Avoid entry, wait for better setup

---

### Quick Decision Tree

```
Is Price > SMA-200?
├─ NO → AVOID (Downtrend)
└─ YES → Continue

Is RSI 20-35?
├─ NO → WAIT (Not in sweet spot)
└─ YES → Continue

Is Price within 3% of SMA-200?
├─ NO → WAIT (Too far from support)
└─ YES → Continue

Is MACD > Signal?
├─ NO → WAIT (No momentum confirmation)
└─ YES → Continue

Is Volume Ratio > 1.2x?
├─ NO → WAIT (Low volume)
└─ YES → Continue

Is %B < 20%?
├─ NO → WAIT (Not oversold in Bollinger)
└─ YES → Continue

Is Price Position < 30%?
├─ NO → WAIT (Not near bottom)
└─ YES → STRONG BUY SIGNAL ✅
```

---

## Conclusion

The Balanced Low strategy is about finding quality stocks at temporary lows in uptrends. VIEW ALL STATS provides comprehensive metrics to validate these opportunities.

**Key Takeaways**:
1. **Convergence is key**: Multiple indicators pointing same direction = stronger signal
2. **Support matters**: Price near multiple support levels = higher bounce probability
3. **Momentum confirms**: Oversold + Momentum shift = Strong buy signal
4. **Risk assessment**: Use ATR, volume, and support proximity to assess risk
5. **Patience pays**: Wait for all criteria to align before entering

**Remember**: Not every oversold stock is a buy. Look for the convergence of multiple factors pointing to a high-probability bounce-back opportunity.

For more information on the technical implementation, see [README.md](README.md).

---

*Happy Trading! ⚡*
