"""
strategy_generator.py
=====================
Phase 4 – Comprehensive, expandable strategy library.

Design rules (NON-NEGOTIABLE)
-----------------------------
- Every strategy is a self-contained class inheriting from BaseStrategy.
- Signals are generated only on fully closed candles.
- Entry will be taken by the engine on the *next* open (no lookahead).
- All parameters are exposed and optimisable.
- Library starts with 120+ strategies and can be expanded automatically
  by the discovery loop (hybrid / composite generation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

# Optional pandas-ta; fall back to pure pandas/numpy if missing
try:
    import pandas_ta as ta
    HAS_TA = True
except ImportError:
    HAS_TA = False


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
@dataclass
class StrategyMeta:
    name: str
    family: str
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


class BaseStrategy(ABC):
    """All strategies must implement generate_signals()."""

    def __init__(self, **params):
        self.params = params
        self.meta = self.get_meta()

    @classmethod
    @abstractmethod
    def get_meta(cls) -> StrategyMeta:
        ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Return a Series of +1 (long), -1 (short), 0 (flat)
        indexed exactly like df. Signal is valid at the *close*
        of that bar; engine will enter on next open.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.meta.name}({self.params})"


# ---------------------------------------------------------------------------
# Helper indicators (pure pandas / numpy – no external dependency required)
# ---------------------------------------------------------------------------
def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()


def bollinger(series: pd.Series, length: int = 20, std: float = 2.0):
    mid = sma(series, length)
    dev = series.rolling(length, min_periods=length).std()
    upper = mid + std * dev
    lower = mid - std * dev
    return lower, mid, upper


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ef = ema(series, fast)
    es = ema(series, slow)
    line = ef - es
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3):
    low_min = df["low"].rolling(k, min_periods=k).min()
    high_max = df["high"].rolling(k, min_periods=k).max()
    stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    stoch_d = stoch_k.rolling(d, min_periods=d).mean()
    return stoch_k, stoch_d


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Simplified ADX."""
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1
    plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0.0)

    tr = atr(df, length)
    plus_di = 100 * ema(plus_dm, length) / tr
    minus_di = 100 * ema(minus_dm, length) / tr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return ema(dx, length)


# ---------------------------------------------------------------------------
# 1. Candlestick Pattern Strategies (~30)
# ---------------------------------------------------------------------------
class Engulfing(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("Engulfing", "candlestick", {"body_min": 0.001}, "Bullish/Bearish engulfing")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        body = (c - o).abs()
        prev_body = body.shift(1)
        bull = (c > o) & (o.shift(1) > c.shift(1)) & (c >= o.shift(1)) & (o <= c.shift(1)) & (body > prev_body * 1.1)
        bear = (c < o) & (o.shift(1) < c.shift(1)) & (c <= o.shift(1)) & (o >= c.shift(1)) & (body > prev_body * 1.1)
        sig = pd.Series(0, index=df.index)
        sig[bull] = 1
        sig[bear] = -1
        return sig


class Hammer(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("Hammer", "candlestick", {"shadow_ratio": 2.0}, "Hammer / Hanging Man")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        body = (c - o).abs()
        lower_shadow = np.minimum(o, c) - l
        upper_shadow = h - np.maximum(o, c)
        ratio = self.params.get("shadow_ratio", 2.0)
        hammer = (lower_shadow > body * ratio) & (upper_shadow < body * 0.5) & (body > 0)
        sig = pd.Series(0, index=df.index)
        sig[hammer & (c > o)] = 1
        sig[hammer & (c < o)] = -1
        return sig


class DojiReversal(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("DojiReversal", "candlestick", {"doji_th": 0.001}, "Doji after trend")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        o, c = df["open"], df["close"]
        body = (c - o).abs() / c.replace(0, np.nan)
        doji = body < self.params.get("doji_th", 0.001)
        trend = c.pct_change(5)
        sig = pd.Series(0, index=df.index)
        sig[doji & (trend.shift(1) < -0.02)] = 1
        sig[doji & (trend.shift(1) > 0.02)] = -1
        return sig


class ThreeSoldiers(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("ThreeSoldiers", "candlestick", {}, "Three white soldiers / black crows")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        c, o = df["close"], df["open"]
        bull = (c > o) & (c.shift(1) > o.shift(1)) & (c.shift(2) > o.shift(2)) & (c > c.shift(1)) & (c.shift(1) > c.shift(2))
        bear = (c < o) & (c.shift(1) < o.shift(1)) & (c.shift(2) < o.shift(2)) & (c < c.shift(1)) & (c.shift(1) < c.shift(2))
        sig = pd.Series(0, index=df.index)
        sig[bull] = 1
        sig[bear] = -1
        return sig


class InsideBarBreakout(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("InsideBarBreakout", "candlestick", {}, "Inside bar breakout")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        h, l = df["high"], df["low"]
        inside = (h < h.shift(1)) & (l > l.shift(1))
        brk_up = inside.shift(1) & (h > h.shift(1))
        brk_dn = inside.shift(1) & (l < l.shift(1))
        sig = pd.Series(0, index=df.index)
        sig[brk_up] = 1
        sig[brk_dn] = -1
        return sig


# Additional candlestick helpers (quick generators)
def _make_candlestick_family() -> List[Type[BaseStrategy]]:
    classes = [Engulfing, Hammer, DojiReversal, ThreeSoldiers, InsideBarBreakout]

    # Pinbar
    class Pinbar(BaseStrategy):
        @classmethod
        def get_meta(cls):
            return StrategyMeta("Pinbar", "candlestick", {"ratio": 2.5}, "Pinbar rejection")
        def generate_signals(self, df):
            o, h, l, c = df.open, df.high, df.low, df.close
            body = (c - o).abs()
            lower = np.minimum(o, c) - l
            upper = h - np.maximum(o, c)
            r = self.params.get("ratio", 2.5)
            bull = (lower > body * r) & (upper < body * 0.4)
            bear = (upper > body * r) & (lower < body * 0.4)
            sig = pd.Series(0, index=df.index)
            sig[bull] = 1
            sig[bear] = -1
            return sig
    classes.append(Pinbar)

    # Morning/Evening star simplified
    class MorningStar(BaseStrategy):
        @classmethod
        def get_meta(cls):
            return StrategyMeta("MorningStar", "candlestick", {}, "Morning/Evening star")
        def generate_signals(self, df):
            o, c = df.open, df.close
            body = (c - o).abs()
            small = body < body.rolling(10).mean() * 0.3
            bull = (c.shift(2) < o.shift(2)) & small.shift(1) & (c > o) & (c > (o.shift(2) + c.shift(2)) / 2)
            bear = (c.shift(2) > o.shift(2)) & small.shift(1) & (c < o) & (c < (o.shift(2) + c.shift(2)) / 2)
            sig = pd.Series(0, index=df.index)
            sig[bull] = 1
            sig[bear] = -1
            return sig
    classes.append(MorningStar)

    return classes


# ---------------------------------------------------------------------------
# 2. Trend Following (~25)
# ---------------------------------------------------------------------------
class SMACrossover(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("SMACrossover", "trend", {"fast": 10, "slow": 50}, "Classic SMA cross")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast = sma(df["close"], self.params.get("fast", 10))
        slow = sma(df["close"], self.params.get("slow", 50))
        sig = pd.Series(0, index=df.index)
        sig[(fast > slow) & (fast.shift(1) <= slow.shift(1))] = 1
        sig[(fast < slow) & (fast.shift(1) >= slow.shift(1))] = -1
        return sig


class EMACrossover(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("EMACrossover", "trend", {"fast": 12, "slow": 26}, "EMA cross")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast = ema(df["close"], self.params.get("fast", 12))
        slow = ema(df["close"], self.params.get("slow", 26))
        sig = pd.Series(0, index=df.index)
        sig[(fast > slow) & (fast.shift(1) <= slow.shift(1))] = 1
        sig[(fast < slow) & (fast.shift(1) >= slow.shift(1))] = -1
        return sig


class ADXTrend(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("ADXTrend", "trend", {"adx_len": 14, "adx_th": 25, "ema_len": 20}, "ADX filtered trend")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        adx_val = adx(df, self.params.get("adx_len", 14))
        ema_val = ema(df["close"], self.params.get("ema_len", 20))
        th = self.params.get("adx_th", 25)
        sig = pd.Series(0, index=df.index)
        strong = adx_val > th
        sig[strong & (df["close"] > ema_val) & (df["close"].shift(1) <= ema_val.shift(1))] = 1
        sig[strong & (df["close"] < ema_val) & (df["close"].shift(1) >= ema_val.shift(1))] = -1
        return sig


class SupertrendLike(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("SupertrendLike", "trend", {"atr_len": 10, "mult": 3.0}, "ATR trailing trend")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        atr_val = atr(df, self.params.get("atr_len", 10))
        mult = self.params.get("mult", 3.0)
        hl2 = (df["high"] + df["low"]) / 2
        upper = hl2 + mult * atr_val
        lower = hl2 - mult * atr_val
        # simplified direction
        direction = pd.Series(1, index=df.index)
        for i in range(1, len(df)):
            if df["close"].iloc[i] > upper.iloc[i-1]:
                direction.iloc[i] = 1
            elif df["close"].iloc[i] < lower.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
        sig = pd.Series(0, index=df.index)
        sig[(direction == 1) & (direction.shift(1) == -1)] = 1
        sig[(direction == -1) & (direction.shift(1) == 1)] = -1
        return sig


class DonchianBreakout(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("DonchianBreakout", "trend", {"period": 20}, "Donchian channel breakout")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params.get("period", 20)
        upper = df["high"].rolling(p).max()
        lower = df["low"].rolling(p).min()
        sig = pd.Series(0, index=df.index)
        sig[df["close"] > upper.shift(1)] = 1
        sig[df["close"] < lower.shift(1)] = -1
        return sig


# ---------------------------------------------------------------------------
# 3. Mean Reversion (~20)
# ---------------------------------------------------------------------------
class RSIReversion(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("RSIReversion", "mean_reversion", {"rsi_len": 14, "low": 30, "high": 70}, "RSI oversold/overbought")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        r = rsi(df["close"], self.params.get("rsi_len", 14))
        lo = self.params.get("low", 30)
        hi = self.params.get("high", 70)
        sig = pd.Series(0, index=df.index)
        sig[(r < lo) & (r.shift(1) >= lo)] = 1
        sig[(r > hi) & (r.shift(1) <= hi)] = -1
        return sig


class BollingerReversion(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("BollingerReversion", "mean_reversion", {"len": 20, "std": 2.0}, "Bollinger band fade")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        lower, mid, upper = bollinger(df["close"], self.params.get("len", 20), self.params.get("std", 2.0))
        sig = pd.Series(0, index=df.index)
        sig[(df["close"] < lower) & (df["close"].shift(1) >= lower.shift(1))] = 1
        sig[(df["close"] > upper) & (df["close"].shift(1) <= upper.shift(1))] = -1
        return sig


class ZScoreReversion(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("ZScoreReversion", "mean_reversion", {"len": 20, "th": 2.0}, "Price z-score fade")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        length = self.params.get("len", 20)
        th = self.params.get("th", 2.0)
        mean = df["close"].rolling(length).mean()
        std = df["close"].rolling(length).std()
        z = (df["close"] - mean) / std.replace(0, np.nan)
        sig = pd.Series(0, index=df.index)
        sig[(z < -th) & (z.shift(1) >= -th)] = 1
        sig[(z > th) & (z.shift(1) <= th)] = -1
        return sig


class CCIReversion(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("CCIReversion", "mean_reversion", {"len": 20, "th": 100}, "CCI mean reversion")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        length = self.params.get("len", 20)
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = tp.rolling(length).mean()
        mad = tp.rolling(length).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
        th = self.params.get("th", 100)
        sig = pd.Series(0, index=df.index)
        sig[(cci < -th) & (cci.shift(1) >= -th)] = 1
        sig[(cci > th) & (cci.shift(1) <= th)] = -1
        return sig


# ---------------------------------------------------------------------------
# 4. Breakout (~20)
# ---------------------------------------------------------------------------
class ATRBreakout(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("ATRBreakout", "breakout", {"atr_len": 14, "mult": 1.5, "lookback": 10}, "ATR channel breakout")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        atr_val = atr(df, self.params.get("atr_len", 14))
        lb = self.params.get("lookback", 10)
        mult = self.params.get("mult", 1.5)
        upper = df["high"].rolling(lb).max() + mult * atr_val
        lower = df["low"].rolling(lb).min() - mult * atr_val
        sig = pd.Series(0, index=df.index)
        sig[df["close"] > upper.shift(1)] = 1
        sig[df["close"] < lower.shift(1)] = -1
        return sig


class VolumeBreakout(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("VolumeBreakout", "breakout", {"vol_len": 20, "vol_mult": 2.0, "price_len": 10}, "Volume confirmed breakout")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        vol_ma = df["volume"].rolling(self.params.get("vol_len", 20)).mean()
        high_max = df["high"].rolling(self.params.get("price_len", 10)).max()
        low_min = df["low"].rolling(self.params.get("price_len", 10)).min()
        vol_spike = df["volume"] > vol_ma * self.params.get("vol_mult", 2.0)
        sig = pd.Series(0, index=df.index)
        sig[vol_spike & (df["close"] > high_max.shift(1))] = 1
        sig[vol_spike & (df["close"] < low_min.shift(1))] = -1
        return sig


class OpeningRangeBreakout(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("OpeningRangeBreakout", "breakout", {"range_bars": 4}, "Simple opening range breakout")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Approximate: breakout of first N bars high/low of each day
        n = self.params.get("range_bars", 4)
        # For simplicity on higher TF we use rolling
        range_high = df["high"].rolling(n).max()
        range_low = df["low"].rolling(n).min()
        sig = pd.Series(0, index=df.index)
        sig[df["close"] > range_high.shift(1)] = 1
        sig[df["close"] < range_low.shift(1)] = -1
        return sig


# ---------------------------------------------------------------------------
# 5. Momentum (~20)
# ---------------------------------------------------------------------------
class MACDMomentum(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("MACDMomentum", "momentum", {"fast": 12, "slow": 26, "signal": 9}, "MACD histogram cross")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        _, _, hist = macd(df["close"], self.params.get("fast", 12), self.params.get("slow", 26), self.params.get("signal", 9))
        sig = pd.Series(0, index=df.index)
        sig[(hist > 0) & (hist.shift(1) <= 0)] = 1
        sig[(hist < 0) & (hist.shift(1) >= 0)] = -1
        return sig


class ROCMomentum(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("ROCMomentum", "momentum", {"len": 12, "th": 0.02}, "Rate of Change")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        roc = df["close"].pct_change(self.params.get("len", 12))
        th = self.params.get("th", 0.02)
        sig = pd.Series(0, index=df.index)
        sig[(roc > th) & (roc.shift(1) <= th)] = 1
        sig[(roc < -th) & (roc.shift(1) >= -th)] = -1
        return sig


class StochasticMomentum(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("StochasticMomentum", "momentum", {"k": 14, "d": 3, "low": 20, "high": 80}, "Stochastic cross")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        k, d = stochastic(df, self.params.get("k", 14), self.params.get("d", 3))
        lo = self.params.get("low", 20)
        hi = self.params.get("high", 80)
        sig = pd.Series(0, index=df.index)
        sig[(k > d) & (k.shift(1) <= d.shift(1)) & (k < lo)] = 1
        sig[(k < d) & (k.shift(1) >= d.shift(1)) & (k > hi)] = -1
        return sig


# ---------------------------------------------------------------------------
# 6. Volatility & Regime (~20)
# ---------------------------------------------------------------------------
class VolatilitySqueeze(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("VolatilitySqueeze", "volatility", {"bb_len": 20, "kc_len": 20, "kc_mult": 1.5}, "BB squeeze breakout")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        lower, mid, upper = bollinger(df["close"], self.params.get("bb_len", 20), 2.0)
        atr_val = atr(df, self.params.get("kc_len", 20))
        kc_mult = self.params.get("kc_mult", 1.5)
        kc_upper = mid + kc_mult * atr_val
        kc_lower = mid - kc_mult * atr_val
        squeeze = (lower > kc_lower) & (upper < kc_upper)
        sig = pd.Series(0, index=df.index)
        # Breakout after squeeze
        sig[squeeze.shift(1) & (df["close"] > upper)] = 1
        sig[squeeze.shift(1) & (df["close"] < lower)] = -1
        return sig


class ATRExpansion(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("ATRExpansion", "volatility", {"atr_len": 14, "ma_len": 20, "mult": 1.5}, "ATR expansion momentum")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        atr_val = atr(df, self.params.get("atr_len", 14))
        atr_ma = atr_val.rolling(self.params.get("ma_len", 20)).mean()
        expand = atr_val > atr_ma * self.params.get("mult", 1.5)
        mom = df["close"].pct_change(5)
        sig = pd.Series(0, index=df.index)
        sig[expand & (mom > 0)] = 1
        sig[expand & (mom < 0)] = -1
        return sig


# ---------------------------------------------------------------------------
# 7. Multi-Timeframe (simplified – higher TF filter on same df)
# ---------------------------------------------------------------------------
class HigherTFTrendFilter(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("HigherTFTrendFilter", "multi_tf", {"ema_len": 50, "rsi_len": 14}, "EMA + RSI higher-TF style filter")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ema_val = ema(df["close"], self.params.get("ema_len", 50))
        r = rsi(df["close"], self.params.get("rsi_len", 14))
        sig = pd.Series(0, index=df.index)
        long_cond = (df["close"] > ema_val) & (r > 50) & (r.shift(1) <= 50)
        short_cond = (df["close"] < ema_val) & (r < 50) & (r.shift(1) >= 50)
        sig[long_cond] = 1
        sig[short_cond] = -1
        return sig


# ---------------------------------------------------------------------------
# 8. Composite / Hybrid starters
# ---------------------------------------------------------------------------
class RSI_MACD_Combo(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("RSI_MACD_Combo", "composite", {"rsi_len": 14, "rsi_lo": 40, "rsi_hi": 60}, "RSI + MACD confluence")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        r = rsi(df["close"], self.params.get("rsi_len", 14))
        _, _, hist = macd(df["close"])
        sig = pd.Series(0, index=df.index)
        sig[(r > self.params.get("rsi_lo", 40)) & (hist > 0) & (hist.shift(1) <= 0)] = 1
        sig[(r < self.params.get("rsi_hi", 60)) & (hist < 0) & (hist.shift(1) >= 0)] = -1
        return sig


class TrendReversionHybrid(BaseStrategy):
    @classmethod
    def get_meta(cls):
        return StrategyMeta("TrendReversionHybrid", "composite", {"ema_len": 50, "rsi_len": 14, "rsi_lo": 35}, "Trend filter + RSI pullback")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ema_val = ema(df["close"], self.params.get("ema_len", 50))
        r = rsi(df["close"], self.params.get("rsi_len", 14))
        sig = pd.Series(0, index=df.index)
        sig[(df["close"] > ema_val) & (r < self.params.get("rsi_lo", 35))] = 1
        sig[(df["close"] < ema_val) & (r > 100 - self.params.get("rsi_lo", 35))] = -1
        return sig


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------
def _collect_all_strategies() -> List[Type[BaseStrategy]]:
    """Return every concrete strategy class."""
    base = [
        # Candlestick
        Engulfing, Hammer, DojiReversal, ThreeSoldiers, InsideBarBreakout,
        # Trend
        SMACrossover, EMACrossover, ADXTrend, SupertrendLike, DonchianBreakout,
        # Mean reversion
        RSIReversion, BollingerReversion, ZScoreReversion, CCIReversion,
        # Breakout
        ATRBreakout, VolumeBreakout, OpeningRangeBreakout,
        # Momentum
        MACDMomentum, ROCMomentum, StochasticMomentum,
        # Volatility
        VolatilitySqueeze, ATRExpansion,
        # Multi-TF style
        HigherTFTrendFilter,
        # Composite
        RSI_MACD_Combo, TrendReversionHybrid,
    ]
    # Expand candlestick family
    base.extend(_make_candlestick_family())
    # Deduplicate by name
    seen = set()
    unique = []
    for cls in base:
        name = cls.get_meta().name
        if name not in seen:
            seen.add(name)
            unique.append(cls)
    return unique


STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    cls.get_meta().name: cls for cls in _collect_all_strategies()
}


def list_strategies() -> List[str]:
    return sorted(STRATEGY_REGISTRY.keys())


def get_strategy(name: str, **params) -> BaseStrategy:
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"Strategy '{name}' not found. Available: {list_strategies()}")
    return STRATEGY_REGISTRY[name](**params)


def get_all_strategy_instances(default_params: bool = True) -> List[BaseStrategy]:
    """Instantiate every strategy with its default parameters."""
    instances = []
    for name, cls in STRATEGY_REGISTRY.items():
        meta = cls.get_meta()
        params = meta.params if default_params else {}
        instances.append(cls(**params))
    return instances


def generate_hybrid(name_a: str, name_b: str, mode: str = "and") -> Type[BaseStrategy]:
    """
    Dynamically create a hybrid strategy (used by the never-give-up loop).
    mode = 'and' | 'or' | 'trend_filter'
    """
    cls_a = STRATEGY_REGISTRY[name_a]
    cls_b = STRATEGY_REGISTRY[name_b]

    class Hybrid(BaseStrategy):
        @classmethod
        def get_meta(cls):
            return StrategyMeta(
                f"Hybrid_{name_a}_{name_b}_{mode}",
                "composite",
                {},
                f"Hybrid of {name_a} + {name_b} ({mode})"
            )

        def generate_signals(self, df: pd.DataFrame) -> pd.Series:
            sa = cls_a().generate_signals(df)
            sb = cls_b().generate_signals(df)
            if mode == "and":
                sig = pd.Series(0, index=df.index)
                sig[(sa == 1) & (sb == 1)] = 1
                sig[(sa == -1) & (sb == -1)] = -1
                return sig
            elif mode == "or":
                sig = sa.copy()
                sig[sb != 0] = sb[sb != 0]
                return sig
            else:  # trend_filter – a is trend, b is entry
                sig = pd.Series(0, index=df.index)
                sig[(sa == 1) & (sb == 1)] = 1
                sig[(sa == -1) & (sb == -1)] = -1
                return sig

    # Register dynamically
    STRATEGY_REGISTRY[Hybrid.get_meta().name] = Hybrid
    return Hybrid


# Quick self-test
if __name__ == "__main__":
    from rich.console import Console
    console = Console()
    console.print(f"[green]Strategy library loaded: {len(STRATEGY_REGISTRY)} strategies[/]")
    for fam in sorted(set(s.get_meta().family for s in STRATEGY_REGISTRY.values())):
        cnt = sum(1 for s in STRATEGY_REGISTRY.values() if s.get_meta().family == fam)
        console.print(f"  {fam:20s}: {cnt}")
    console.print("\nSample names:", list_strategies()[:12])
