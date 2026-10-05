"""
CANONICAL BACKTEST ENGINE v2.0.1
Fixes:
  - Supertrend now handles NaN-seed correctly
  - Signal generation uses direct boolean masks (no dtype coercion)
  - ATR NaN warm-up gate prevents empty backtests
"""
import os
import sys
import warnings
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

__version__ = "2.0.1"
__all__ = [
    "SMA", "EMA", "WMA", "HMA",
    "RSI", "MACD", "ATR", "ADX",
    "Supertrend", "Bollinger", "Stochastic",
    "VWAP", "Keltner", "Donchian",
    "ROC", "WilliamsR", "StochRSI",
    "CCI", "OBV", "MFI",
    "MultiTFDataLoader", "RegimeDetector",
    "BacktestEngine", "Strategy", "SupertrendStrategy",
    "DataLeakageError",
]

class DataLeakageError(Exception):
    pass

# ============================================================
# INDICATORS (verified deterministic)
# ============================================================
def SMA(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=int(period), min_periods=int(period)).mean()

def EMA(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average with SMA seed."""
    period = int(period)
    if period <= 0:
        raise ValueError("EMA period must be > 0")
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def WMA(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average with linear weights."""
    period = int(period)
    if period <= 0:
        raise ValueError("WMA period must be > 0")
    weights = np.arange(1, period + 1, dtype=float)
    denom = weights.sum()
    return series.rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, weights) / denom, raw=True
    )

def HMA(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average."""
    period = int(period)
    if period < 4:
        raise ValueError("HMA period must be >= 4")
    half = max(int(period / 2), 2)
    sqn = max(int(np.sqrt(period)), 2)
    wma_half = WMA(series, half)
    wma_full = WMA(series, period)
    diff = 2 * wma_half - wma_full
    return WMA(diff, sqn)

def RSI(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    period = int(period)
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    avg_dn = down.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)

def MACD(series, fast=12, slow=26, signal=9):
    """MACD line, signal, histogram."""
    fast, slow, signal = int(fast), int(slow), int(signal)
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line.dropna(), signal).reindex(series.index)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def ATR(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR."""
    period = int(period)
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()

def ADX(df, period=14):
    """Wilder's ADX."""
    period = int(period)
    h, l, c = df["high"], df["low"], df["close"]
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm_s = pd.Series(plus_dm, index=df.index)
    minus_dm_s = pd.Series(minus_dm, index=df.index)
    atr_s = ATR(df, period)
    plus_di = 100.0 * plus_dm_s.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean() / atr_s.replace(0, np.nan)
    minus_di = 100.0 * minus_dm_s.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean() / atr_s.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

def Supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """Supertrend (TradingView-compatible) - HOTFIXED to handle NaN seed."""
    period = int(period); multiplier = float(multiplier)
    n = len(df)
    a = ATR(df, period)
    hl2 = ((df["high"] + df["low"]) / 2.0).values
    atr_vals = a.values
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)
    st = np.full(n, np.nan)
    
    # Find first bar where ATR is valid (warmed up)
    first_valid = None
    for k in range(n):
        if not np.isnan(atr_vals[k]):
            first_valid = k
            break
    
    if first_valid is None:
        return pd.Series(st, index=df.index), pd.Series(trend, index=df.index)
    
    # Seed at first valid bar
    final_upper[first_valid] = hl2[first_valid] + multiplier * atr_vals[first_valid]
    final_lower[first_valid] = hl2[first_valid] - multiplier * atr_vals[first_valid]
    trend[first_valid] = 1
    st[first_valid] = final_lower[first_valid]
    
    # Iterate from first_valid+1 forward
    for i in range(first_valid + 1, n):
        basic_upper = hl2[i] + multiplier * atr_vals[i]
        basic_lower = hl2[i] - multiplier * atr_vals[i]
        prev_upper = final_upper[i-1]
        prev_lower = final_lower[i-1]
        prev_close = close[i-1]
        
        # Final upper band: adjust so it ratchets down when in uptrend
        if basic_upper < prev_upper or prev_close > prev_upper:
            final_upper[i] = basic_upper
        else:
            final_upper[i] = prev_upper
        # Final lower band: adjust so it ratchets up when in uptrend
        if basic_lower > prev_lower or prev_close < prev_lower:
            final_lower[i] = basic_lower
        else:
            final_lower[i] = prev_lower
        
        # Trend determination
        if close[i] > final_upper[i-1]:
            trend[i] = 1
        elif close[i] < final_lower[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            # Ratchet stops in direction of trend
            if trend[i] == 1 and final_lower[i] < prev_lower:
                final_lower[i] = prev_lower
            if trend[i] == -1 and final_upper[i] > prev_upper:
                final_upper[i] = prev_upper
        
        st[i] = final_lower[i] if trend[i] == 1 else final_upper[i]
    
    return pd.Series(st, index=df.index), pd.Series(trend, index=df.index)

def Bollinger(series, period=20, std_dev=2.0):
    period = int(period)
    middle = SMA(series, period)
    std = series.rolling(period, min_periods=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    width = (upper - lower) / middle.replace(0, np.nan)
    return upper, middle, lower, width

def Stochastic(df, k_period=14, d_period=3, smooth_k=3):
    k_period = int(k_period); d_period = int(d_period); smooth_k = int(smooth_k)
    lo_min = df["low"].rolling(k_period, min_periods=k_period).min()
    hi_max = df["high"].rolling(k_period, min_periods=k_period).max()
    raw_k = 100.0 * (df["close"] - lo_min) / (hi_max - lo_min).replace(0, np.nan)
    k = raw_k.rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(d_period, min_periods=d_period).mean()
    return k.fillna(50), d.fillna(50)

def VWAP(df, session="daily"):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    if session == "daily":
        day = df.index.floor("D")
        cum_pv = pv.groupby(day).cumsum()
        cum_v = df["volume"].groupby(day).cumsum()
        return cum_pv / cum_v.replace(0, np.nan)
    elif session == "rolling":
        cum_pv = pv.rolling(20, min_periods=1).sum()
        cum_v = df["volume"].rolling(20, min_periods=1).sum()
        return cum_pv / cum_v.replace(0, np.nan)
    else:
        raise ValueError("session must be 'daily' or 'rolling'")

def Keltner(df, period=20, atr_mult=2.0):
    period = int(period)
    middle = EMA(df["close"], period)
    a = ATR(df, period)
    return middle + atr_mult * a, middle, middle - atr_mult * a

def Donchian(df, period=20):
    period = int(period)
    upper = df["high"].rolling(period, min_periods=period).max()
    lower = df["low"].rolling(period, min_periods=period).min()
    middle = (upper + lower) / 2.0
    return upper, middle, lower

def ROC(series, period=10):
    period = int(period)
    return 100.0 * (series - series.shift(period)) / series.shift(period).replace(0, np.nan)

def WilliamsR(df, period=14):
    period = int(period)
    hi = df["high"].rolling(period, min_periods=period).max()
    lo = df["low"].rolling(period, min_periods=period).min()
    return -100.0 * (hi - df["close"]) / (hi - lo).replace(0, np.nan)

def StochRSI(series, rsi_period=14, stoch_period=14, k=3, d=3):
    rsi_period = int(rsi_period); stoch_period = int(stoch_period)
    k = int(k); d = int(d)
    r = RSI(series, rsi_period)
    lo = r.rolling(stoch_period, min_periods=stoch_period).min()
    hi = r.rolling(stoch_period, min_periods=stoch_period).max()
    stoch = 100.0 * (r - lo) / (hi - lo).replace(0, np.nan)
    kk = stoch.rolling(k, min_periods=k).mean()
    dd = kk.rolling(d, min_periods=d).mean()
    return kk.fillna(50), dd.fillna(50)

def CCI(df, period=20):
    period = int(period)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma_tp = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

def OBV(df):
    direction = np.sign(df["close"].diff().fillna(0))
    return (direction * df["volume"]).cumsum()

def MFI(df, period=14):
    period = int(period)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    mf = tp * df["volume"]
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0.0)
    neg_mf = mf.where(delta < 0, 0.0)
    pos_sum = pos_mf.rolling(period, min_periods=period).sum()
    neg_sum = neg_mf.rolling(period, min_periods=period).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + ratio))

# ============================================================
# INDICATOR SELF-TESTS (HOTFIXED - Supertrend uses larger dataset)
# ============================================================
def _run_self_tests(verbose=False):
    results = {}
    # Test series (20 bars)
    closes = pd.Series([
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
        45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64
    ])
    highs  = closes + 0.5; lows = closes - 0.5
    opens  = closes.shift(1).fillna(closes.iloc[0])
    volume = pd.Series([1000]*20, dtype=float) + np.arange(20)*10
    df_test = pd.DataFrame({"open":opens,"high":highs,"low":lows,"close":closes,"volume":volume})
    
    # Larger dataset for Supertrend test (60 bars)
    np.random.seed(42)
    prices_60 = pd.Series(100 + np.cumsum(np.random.randn(60) * 0.5))
    df_st = pd.DataFrame({
        "open": prices_60.shift(1).fillna(prices_60.iloc[0]),
        "high": prices_60 + np.abs(np.random.randn(60) * 0.3),
        "low":  prices_60 - np.abs(np.random.randn(60) * 0.3),
        "close": prices_60,
        "volume": np.ones(60) * 1000
    })

    def _try(name, fn):
        try:
            fn()
            results[name] = True
        except Exception as ex:
            results[name] = False
            if verbose: print(f"    [FAIL] {name}: {ex}")

    _try("SMA", lambda: (
        (abs(SMA(closes,5).iloc[4] - closes.iloc[:5].mean()) < 1e-9) or (_ for _ in ()).throw(AssertionError("SMA mismatch"))
    ))
    def _ema_test():
        e = EMA(closes, 5)
        if pd.isna(e.iloc[4]): raise AssertionError("EMA seed NaN")
        if not (e.iloc[-1] > 0): raise AssertionError("EMA value invalid")
    _try("EMA", _ema_test)
    def _wma_test():
        w = WMA(closes, 5)
        weights = np.array([1,2,3,4,5])
        expected = np.dot(closes.iloc[:5].values, weights) / weights.sum()
        if abs(w.iloc[4] - expected) > 1e-6:
            raise AssertionError(f"WMA mismatch")
    _try("WMA", _wma_test)
    def _hma_test():
        h = HMA(closes, 9)
        if not np.isfinite(h.iloc[-1]): raise AssertionError("HMA not finite")
    _try("HMA", _hma_test)
    def _rsi_test():
        r = RSI(closes, 14)
        v = r.iloc[-1]
        if not np.isfinite(v): raise AssertionError("RSI not finite")
        if not (0 <= v <= 100): raise AssertionError(f"RSI out of bounds: {v}")
    _try("RSI", _rsi_test)
    def _macd_test():
        m, s, h = MACD(closes, 5, 10, 3)
        if not np.isfinite(m.iloc[-1]): raise AssertionError("MACD not finite")
    _try("MACD", _macd_test)
    def _atr_test():
        a = ATR(df_test, 5)
        if not (a.iloc[-1] > 0): raise AssertionError("ATR should be positive")
    _try("ATR", _atr_test)
    def _adx_test():
        a, p, m = ADX(df_test, 5)
        if not np.isfinite(a.iloc[-1]): raise AssertionError("ADX not finite")
    _try("ADX", _adx_test)
    # HOTFIXED: use 60-bar dataset for Supertrend
    def _st_test():
        st, tr = Supertrend(df_st, 10, 3.0)
        val = st.iloc[-1]
        direction = tr.iloc[-1]
        if not np.isfinite(val):
            raise AssertionError(f"Supertrend last value not finite: {val}")
        if direction not in (-1, 1):
            raise AssertionError(f"Supertrend direction invalid: {direction}")
        # Also ensure ATR-warmup respected (Supertrend NaN in first ~10 bars)
        if np.isfinite(st.iloc[0]):
            raise AssertionError("Supertrend should be NaN before ATR warm-up")
    _try("Supertrend", _st_test)
    def _bb_test():
        u, m, l, w = Bollinger(closes, 5, 2.0)
        if not (u.iloc[-1] > m.iloc[-1] > l.iloc[-1]): raise AssertionError("BB ordering wrong")
    _try("Bollinger", _bb_test)
    def _stoch_test():
        k, d = Stochastic(df_test, 5, 3, 3)
        if not (0 <= k.iloc[-1] <= 100): raise AssertionError("Stoch K out of range")
    _try("Stochastic", _stoch_test)
    def _vwap_test():
        v = VWAP(df_test, "rolling")
        if not (v.iloc[-1] > 0): raise AssertionError("VWAP not positive")
    _try("VWAP", _vwap_test)
    def _kelt_test():
        u, m, l = Keltner(df_test, 5, 2.0)
        if not (u.iloc[-1] > l.iloc[-1]): raise AssertionError("Keltner ordering wrong")
    _try("Keltner", _kelt_test)
    def _don_test():
        u, m, l = Donchian(df_test, 5)
        if not (u.iloc[-1] >= l.iloc[-1]): raise AssertionError("Donchian ordering wrong")
    _try("Donchian", _don_test)
    def _roc_test():
        r = ROC(closes, 5)
        if not np.isfinite(r.iloc[-1]): raise AssertionError("ROC not finite")
    _try("ROC", _roc_test)
    def _wr_test():
        w = WilliamsR(df_test, 5)
        if not (-100 <= w.iloc[-1] <= 0): raise AssertionError("WilliamsR out of range")
    _try("WilliamsR", _wr_test)
    def _srsi_test():
        k, d = StochRSI(closes, 14, 14, 3, 3)
        if not (0 <= k.iloc[-1] <= 100): raise AssertionError("StochRSI K out of range")
    _try("StochRSI", _srsi_test)
    def _cci_test():
        c = CCI(df_test, 5)
        if not np.isfinite(c.iloc[-1]): raise AssertionError("CCI not finite")
    _try("CCI", _cci_test)
    def _obv_test():
        o = OBV(df_test)
        if not np.isfinite(o.iloc[-1]): raise AssertionError("OBV not finite")
    _try("OBV", _obv_test)
    def _mfi_test():
        m = MFI(df_test, 5)
        if not np.isfinite(m.iloc[-1]): raise AssertionError("MFI not finite")
    _try("MFI", _mfi_test)
    return results

_TEST_RESULTS = _run_self_tests(verbose=False)
_TESTS_PASSED = sum(1 for v in _TEST_RESULTS.values() if v)
_TESTS_TOTAL = len(_TEST_RESULTS)
if _TESTS_PASSED < _TESTS_TOTAL:
    _failed = [k for k,v in _TEST_RESULTS.items() if not v]
    print(f"[engine_v2] WARNING: {_TESTS_TOTAL - _TESTS_PASSED} indicator self-tests failed: {_failed}")

# ============================================================
# DATA LOADER
# ============================================================
class MultiTFDataLoader:
    DEFAULT_DATA_ROOT = r"C:\BybitBacktest\data\resampled"

    def __init__(self, data_root=None):
        self.data_root = data_root or self.DEFAULT_DATA_ROOT
        self._cache = {}

    def _read_parquet(self, symbol, tf):
        key = (symbol, tf)
        if key in self._cache: return self._cache[key]
        path = Path(self.data_root) / tf / f"{symbol}.parquet"
        if not path.exists():
            self._cache[key] = None
            return None
        try:
            df = pd.read_parquet(path)
            ts_col = None
            for c in ["timestamp","datetime","time","date"]:
                if c in df.columns: ts_col = c; break
            if ts_col:
                df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
                df = df.set_index(ts_col)
            else:
                df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            cols = {c.lower(): c for c in df.columns}
            rmap = {}
            for want in ["open","high","low","close","volume"]:
                if want in cols: rmap[cols[want]] = want
            df = df.rename(columns=rmap)
            for n in ["open","high","low","close","volume"]:
                if n not in df.columns:
                    self._cache[key] = None
                    return None
            df = df[["open","high","low","close","volume"]].astype(float)
            df = df[~df.index.duplicated(keep="first")].sort_index().dropna()
            self._cache[key] = df
            return df
        except Exception:
            self._cache[key] = None
            return None

    def load(self, symbol, tf, start_date, end_date):
        df = self._read_parquet(symbol, tf)
        if df is None: return None
        s = pd.Timestamp(start_date); e = pd.Timestamp(end_date)
        if len(str(end_date)) == 10:
            e = e + pd.Timedelta(hours=23, minutes=59, seconds=59)
        return df[(df.index >= s) & (df.index <= e)].copy()

    def load_mtf(self, symbol, primary_tf, higher_tf, start_date, end_date, verify_no_leakage=True):
        primary = self.load(symbol, primary_tf, start_date, end_date)
        higher = self.load(symbol, higher_tf, start_date, end_date)
        if primary is None or higher is None: return None
        higher_prefixed = higher.add_prefix(f"h_{higher_tf}_")
        higher_shifted = higher_prefixed.shift(1)
        higher_aligned = higher_shifted.reindex(primary.index, method="ffill")
        merged = pd.concat([primary, higher_aligned], axis=1)
        if verify_no_leakage:
            self._verify_no_leakage(primary, higher, higher_tf)
        return merged

    def _verify_no_leakage(self, primary, higher, higher_tf):
        if len(primary) < 5 or len(higher) < 5: return
        sample_idx = [len(primary)//4, len(primary)//2, 3*len(primary)//4]
        for i in sample_idx:
            t_primary = primary.index[i]
            higher_before = higher[higher.index < t_primary]
            if len(higher_before) == 0: continue
            last_higher_ts = higher_before.index[-1]
            if last_higher_ts >= t_primary:
                raise DataLeakageError(
                    f"Higher-TF ({higher_tf}) candle at {last_higher_ts} used on primary bar {t_primary}"
                )

    def available_symbols(self, tf):
        p = Path(self.data_root) / tf
        if not p.exists(): return []
        return sorted([f.stem for f in p.glob("*.parquet")])

    def date_range(self, symbol, tf):
        df = self._read_parquet(symbol, tf)
        if df is None or len(df) == 0: return None
        return (df.index.min(), df.index.max())

# ============================================================
# REGIME DETECTOR
# ============================================================
class RegimeDetector:
    ANCHOR_SYMBOL = "BTC_USDT_USDT"
    ANCHOR_TF = "1D"

    def __init__(self, data_loader=None):
        self.loader = data_loader or MultiTFDataLoader()
        self._btc_cache = None

    def _load_btc(self):
        if self._btc_cache is None:
            self._btc_cache = self.loader.load(
                self.ANCHOR_SYMBOL, self.ANCHOR_TF,
                "2022-01-01", "2026-12-31"
            )
        return self._btc_cache

    def _compute_features(self, btc):
        close = btc["close"]
        feats = pd.DataFrame(index=btc.index)
        feats["close"] = close
        feats["ema_200"] = EMA(close, 200)
        feats["ema_50"] = EMA(close, 50)
        feats["ema_50_slope"] = (feats["ema_50"] - feats["ema_50"].shift(20)) / feats["ema_50"].shift(20)
        adx_v, _, _ = ADX(btc, 14)
        feats["adx_14"] = adx_v
        atr_v = ATR(btc, 14)
        feats["atr_pct"] = 100.0 * atr_v / close
        feats["atr_percentile"] = feats["atr_pct"].rolling(90, min_periods=30).rank(pct=True) * 100.0
        return feats

    def _label(self, row):
        if pd.isna(row["ema_200"]) or pd.isna(row["adx_14"]) or pd.isna(row["atr_percentile"]):
            return "UNKNOWN"
        if row["atr_percentile"] > 95:
            return "CRISIS"
        if row["close"] > row["ema_200"] and row["ema_50_slope"] > 0.02 and row["adx_14"] > 20:
            return "BULL"
        if row["close"] < row["ema_200"] and row["ema_50_slope"] < -0.02 and row["adx_14"] > 20:
            return "BEAR"
        return "CHOP"

    def classify(self, date):
        btc = self._load_btc()
        if btc is None: return "UNKNOWN"
        target = pd.Timestamp(date)
        subset = btc[btc.index <= target]
        if len(subset) < 200: return "UNKNOWN"
        feats = self._compute_features(subset)
        row = feats.iloc[-1]
        return self._label(row)

    def classify_series(self, start_date, end_date):
        btc = self._load_btc()
        if btc is None: return pd.Series(dtype=str)
        s = pd.Timestamp(start_date); e = pd.Timestamp(end_date)
        feats = self._compute_features(btc)
        labels = feats.apply(self._label, axis=1)
        return labels[(labels.index >= s) & (labels.index <= e)]

    def get_regime_stats(self, start_date, end_date):
        series = self.classify_series(start_date, end_date)
        if len(series) == 0: return {"total_days":0}
        counts = series.value_counts()
        pct = 100.0 * counts / len(series)
        streaks = {}
        for reg in ["BULL","BEAR","CHOP","CRISIS","UNKNOWN"]:
            max_streak = cur = 0
            for v in series.values:
                if v == reg:
                    cur += 1
                    if cur > max_streak: max_streak = cur
                else: cur = 0
            streaks[reg] = max_streak
        transitions = int((series.shift(1) != series).sum() - 1)
        return {
            "total_days": len(series),
            "counts": counts.to_dict(),
            "pct": pct.round(2).to_dict(),
            "longest_streaks": streaks,
            "transitions": transitions,
            "current_regime": str(series.iloc[-1]) if len(series) > 0 else "UNKNOWN"
        }

# ============================================================
# BACKTEST ENGINE
# ============================================================
class BacktestEngine:
    FEES = 0.00055
    SLIPPAGE = 0.0003
    RISK_PER_TRADE = 0.01
    STARTING_CAPITAL = 10000.0

    def __init__(self, fees=FEES, slippage=SLIPPAGE, risk=RISK_PER_TRADE, capital=STARTING_CAPITAL):
        self.fees = fees; self.slippage = slippage
        self.risk = risk; self.capital = capital

    def _slip(self, price, direction, is_entry):
        if direction == 1:
            return price * (1 + self.slippage) if is_entry else price * (1 - self.slippage)
        else:
            return price * (1 - self.slippage) if is_entry else price * (1 + self.slippage)

    def run(self, df, signals, sl_series, tp_series):
        n = len(df)
        if n < 3:
            return {"trades":[], "equity_curve":pd.Series([self.capital], index=[df.index[0]] if n>0 else []),
                    "metrics": self._empty_metrics()}
        o = df["open"].values; h = df["high"].values
        l = df["low"].values; c = df["close"].values
        idx = df.index
        sig = signals.reindex(df.index).fillna(0).astype(int).values
        sl_arr = sl_series.reindex(df.index).astype(float).values
        tp_arr = tp_series.reindex(df.index).astype(float).values

        trades = []; equity = self.capital
        eq_vals = [equity]; eq_times = [idx[0]]
        in_pos = False; pos = None

        for i in range(n - 1):
            if in_pos:
                hi, lo, op = h[i], l[i], o[i]
                exit_px = None; reason = None
                if pos["direction"] == 1:
                    if op <= pos["sl"]:
                        exit_px = self._slip(op, 1, False); reason = "SL_GAP"
                    elif lo <= pos["sl"]:
                        exit_px = self._slip(pos["sl"], 1, False); reason = "SL"
                    elif hi >= pos["tp"]:
                        exit_px = self._slip(pos["tp"], 1, False); reason = "TP"
                else:
                    if op >= pos["sl"]:
                        exit_px = self._slip(op, -1, False); reason = "SL_GAP"
                    elif hi >= pos["sl"]:
                        exit_px = self._slip(pos["sl"], -1, False); reason = "SL"
                    elif lo <= pos["tp"]:
                        exit_px = self._slip(pos["tp"], -1, False); reason = "TP"

                if exit_px is not None:
                    size = pos["size"]; entry_px = pos["entry_price"]
                    if pos["direction"] == 1: gross = (exit_px - entry_px) * size
                    else: gross = (entry_px - exit_px) * size
                    fee_cost = (entry_px + exit_px) * size * self.fees
                    net = gross - fee_cost
                    equity += net
                    pnl_pct = (net / (entry_px * size)) * 100 if entry_px * size > 0 else 0
                    trades.append({
                        "entry_time": pos["entry_time"], "exit_time": idx[i],
                        "direction": "LONG" if pos["direction"]==1 else "SHORT",
                        "entry_price": round(entry_px, 8), "exit_price": round(exit_px, 8),
                        "size": round(size, 6), "sl": round(pos["sl_init"], 8),
                        "tp": round(pos["tp_init"], 8), "pnl": round(net, 4),
                        "pnl_pct": round(pnl_pct, 4), "exit_reason": reason
                    })
                    eq_vals.append(equity); eq_times.append(idx[i])
                    in_pos = False; pos = None

            if not in_pos and i + 1 < n and sig[i] != 0:
                direction = sig[i]
                sv, tv = sl_arr[i], tp_arr[i]
                if np.isnan(sv) or np.isnan(tv): continue
                entry_raw = o[i + 1]
                if np.isnan(entry_raw): continue
                entry_px = self._slip(entry_raw, direction, True)
                if direction == 1:
                    if sv >= entry_px or tv <= entry_px: continue
                    risk_per_unit = entry_px - sv
                else:
                    if sv <= entry_px or tv >= entry_px: continue
                    risk_per_unit = sv - entry_px
                if risk_per_unit <= 0 or np.isnan(risk_per_unit): continue
                size = (equity * self.risk) / risk_per_unit
                if size * entry_px > equity:
                    size = equity / entry_px
                if size <= 0: continue
                pos = {"direction": direction, "entry_time": idx[i+1],
                       "entry_price": entry_px, "sl": sv, "tp": tv,
                       "sl_init": sv, "tp_init": tv, "size": size}
                in_pos = True

        if in_pos:
            exit_px = self._slip(c[-1], pos["direction"], False)
            size = pos["size"]; entry_px = pos["entry_price"]
            if pos["direction"] == 1: gross = (exit_px - entry_px) * size
            else: gross = (entry_px - exit_px) * size
            fee_cost = (entry_px + exit_px) * size * self.fees
            net = gross - fee_cost; equity += net
            pnl_pct = (net / (entry_px * size)) * 100 if entry_px * size > 0 else 0
            trades.append({
                "entry_time": pos["entry_time"], "exit_time": idx[-1],
                "direction": "LONG" if pos["direction"]==1 else "SHORT",
                "entry_price": round(entry_px, 8), "exit_price": round(exit_px, 8),
                "size": round(size, 6), "sl": round(pos["sl_init"], 8),
                "tp": round(pos["tp_init"], 8), "pnl": round(net, 4),
                "pnl_pct": round(pnl_pct, 4), "exit_reason": "EOD"
            })
            eq_vals.append(equity); eq_times.append(idx[-1])

        eq_curve = pd.Series(eq_vals, index=eq_times, name="equity")
        tf = self._infer_tf(df)
        metrics = self.compute_metrics(trades, eq_curve, tf)
        return {"trades": trades, "equity_curve": eq_curve, "metrics": metrics}

    def _infer_tf(self, df):
        if len(df) < 3: return "1D"
        deltas = df.index.to_series().diff().dt.total_seconds().dropna()
        if len(deltas) == 0: return "1D"
        median_secs = deltas.median()
        if median_secs <= 900 * 1.5: return "15m"
        if median_secs <= 1800 * 1.5: return "30m"
        if median_secs <= 3600 * 1.5: return "1H"
        if median_secs <= 14400 * 1.5: return "4H"
        return "1D"

    def _empty_metrics(self):
        return {"total_return_pct":0,"cagr_pct":0,"sharpe":0,"sortino":0,
                "calmar":0,"max_drawdown_pct":0,"max_dd_duration_bars":0,
                "profit_factor":0,"win_rate_pct":0,"expectancy":0,
                "total_trades":0,"avg_trade_pct":0,"avg_win":0,"avg_loss":0,
                "avg_win_loss_ratio":0,"max_consecutive_wins":0,
                "max_consecutive_losses":0,"best_month":0,"worst_month":0,
                "pct_profitable_months":0,"monthly_returns":{}}

    def compute_metrics(self, trades, equity_curve, tf):
        if not trades or equity_curve is None or len(equity_curve) < 2:
            return self._empty_metrics()
        m = self._empty_metrics()
        m["total_trades"] = len(trades)
        pnls = np.array([t["pnl"] for t in trades])
        pcts = np.array([t["pnl_pct"] for t in trades])
        wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
        m["win_rate_pct"] = round(100.0 * len(wins) / len(pnls), 2) if len(pnls) > 0 else 0
        m["avg_trade_pct"] = round(float(np.mean(pcts)), 4)
        m["avg_win"] = round(float(wins.mean()), 4) if len(wins) > 0 else 0
        m["avg_loss"] = round(float(abs(losses.mean())), 4) if len(losses) > 0 else 0
        gp = wins.sum() if len(wins) > 0 else 0
        gl = abs(losses.sum()) if len(losses) > 0 else 0
        m["profit_factor"] = round(gp/gl, 4) if gl > 0 else (999.0 if gp>0 else 0)
        m["avg_win_loss_ratio"] = round(m["avg_win"]/m["avg_loss"], 4) if m["avg_loss"] > 0 else 0
        pw = len(wins)/len(pnls) if len(pnls) > 0 else 0
        m["expectancy"] = round(pw * m["avg_win"] - (1-pw) * m["avg_loss"], 4)

        eq = equity_curve.values
        m["total_return_pct"] = round((eq[-1]/eq[0] - 1) * 100, 2)
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = max(days/365.25, 0.01)
        if eq[-1] > 0 and eq[0] > 0:
            m["cagr_pct"] = round(((eq[-1]/eq[0])**(1/years) - 1) * 100, 2)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        m["max_drawdown_pct"] = round(float(dd.min() * 100), 2)
        cur = 0; max_dur = 0
        for x in dd:
            if x < 0:
                cur += 1
                if cur > max_dur: max_dur = cur
            else: cur = 0
        m["max_dd_duration_bars"] = max_dur

        rets = pd.Series(pcts) / 100.0
        if len(rets) > 1 and rets.std() > 0:
            avg_days = days / len(rets) if len(rets) > 0 else 1
            tpy = 365.25 / max(avg_days, 0.5)
            m["sharpe"] = round(float(rets.mean() / rets.std() * np.sqrt(tpy)), 4)
            neg = rets[rets < 0]
            if len(neg) > 0 and neg.std() > 0:
                m["sortino"] = round(float(rets.mean() / neg.std() * np.sqrt(tpy)), 4)
        if m["max_drawdown_pct"] < 0:
            m["calmar"] = round(m["cagr_pct"] / abs(m["max_drawdown_pct"]), 4)

        cur_w = cur_l = max_w = max_l = 0
        for p in pnls:
            if p > 0:
                cur_w += 1; cur_l = 0
                if cur_w > max_w: max_w = cur_w
            elif p < 0:
                cur_l += 1; cur_w = 0
                if cur_l > max_l: max_l = cur_l
            else:
                cur_w = 0; cur_l = 0
        m["max_consecutive_wins"] = max_w
        m["max_consecutive_losses"] = max_l

        try:
            daily = equity_curve.resample("D").last().ffill()
            monthly = daily.resample("ME").last().pct_change().dropna() * 100
            m["monthly_returns"] = {str(k.date()): round(float(v), 2) for k, v in monthly.items()}
            if len(monthly) > 0:
                m["best_month"] = round(float(monthly.max()), 2)
                m["worst_month"] = round(float(monthly.min()), 2)
                m["pct_profitable_months"] = round(100.0 * (monthly > 0).mean(), 2)
        except Exception: pass
        return m

# ============================================================
# STRATEGY BASE + SUPERTREND (HOTFIXED signal generation)
# ============================================================
class Strategy(ABC):
    name = "AbstractStrategy"
    category = "Unknown"
    default_params = {}

    def __init__(self, engine=None):
        self.engine = engine or BacktestEngine()

    @abstractmethod
    def generate_signals(self, df, params):
        raise NotImplementedError

    def backtest(self, df, params=None):
        p = params if params is not None else self.default_params
        signals, sl_s, tp_s = self.generate_signals(df, p)
        return self.engine.run(df, signals, sl_s, tp_s)

class SupertrendStrategy(Strategy):
    """Supertrend flip strategy - HOTFIXED signal generation."""
    name = "S11_Supertrend"
    category = "Trend"
    default_params = {
        "st_period": 10, "st_multiplier": 3.0,
        "sl_atr_mult": 2.0, "tp_atr_mult": 3.0,
        "atr_period": 14, "trend_filter_ema": 0,
        "use_trailing_stop": False, "trailing_atr_mult": 2.0
    }

    def generate_signals(self, df, params):
        p = params
        st, trend = Supertrend(df, int(p.get("st_period", 10)),
                                float(p.get("st_multiplier", 3.0)))
        trend_prev = trend.shift(1)
        # HOTFIX: build boolean masks WITHOUT dtype coercion issues
        flip_up_mask = (trend == 1) & (trend_prev == -1)
        flip_dn_mask = (trend == -1) & (trend_prev == 1)
        flip_up_mask = flip_up_mask.fillna(False).astype(bool)
        flip_dn_mask = flip_dn_mask.fillna(False).astype(bool)
        
        # Apply trend filter
        tf_ema = int(p.get("trend_filter_ema", 0))
        if tf_ema > 0:
            e = EMA(df["close"], tf_ema)
            above_ema = (df["close"] > e).fillna(False).astype(bool)
            below_ema = (df["close"] < e).fillna(False).astype(bool)
            flip_up_mask = flip_up_mask & above_ema
            flip_dn_mask = flip_dn_mask & below_ema
        
        # Build signals series using numpy for maximum safety
        n = len(df)
        signals_arr = np.zeros(n, dtype=int)
        signals_arr[flip_up_mask.values] = 1
        signals_arr[flip_dn_mask.values] = -1
        signals = pd.Series(signals_arr, index=df.index, dtype=int)
        
        # ATR for SL/TP
        a = ATR(df, int(p.get("atr_period", 14)))
        close = df["close"]
        sl_mult = float(p.get("sl_atr_mult", 2.0))
        tp_mult = float(p.get("tp_atr_mult", 3.0))
        
        # HOTFIX: compute SL/TP for ALL bars first, then mask
        sl_long_all = close - sl_mult * a
        sl_short_all = close + sl_mult * a
        tp_long_all = close + tp_mult * a
        tp_short_all = close - tp_mult * a
        
        sl_series = pd.Series(np.nan, index=df.index, dtype=float)
        tp_series = pd.Series(np.nan, index=df.index, dtype=float)
        # Only assign where signal is nonzero (long or short)
        long_positions = signals == 1
        short_positions = signals == -1
        sl_series.loc[long_positions] = sl_long_all.loc[long_positions].values
        sl_series.loc[short_positions] = sl_short_all.loc[short_positions].values
        tp_series.loc[long_positions] = tp_long_all.loc[long_positions].values
        tp_series.loc[short_positions] = tp_short_all.loc[short_positions].values
        
        return signals, sl_series, tp_series

if __name__ == "__main__":
    print(f"engine_v2 loaded. Version: {__version__}")
    print(f"Self-tests: {_TESTS_PASSED}/{_TESTS_TOTAL} passed")
