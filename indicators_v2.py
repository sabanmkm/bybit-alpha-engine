"""
ADVANCED INDICATORS LIBRARY v2.
BTC Macro Filter, Volume Confirmation, RSI Divergence, Volatility Regime,
Liquidation Cascade Proxy, and Rolling VWAP Deviation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def sma(s: pd.Series, p: int) -> pd.Series:
    return s.rolling(p, min_periods=p).mean()

def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()

def wma(s: pd.Series, p: int) -> pd.Series:
    w = np.arange(1, p+1, dtype=float); w /= w.sum()
    r = np.full_like(s.values, np.nan)
    if len(s) >= p: r[p-1:] = np.convolve(s.values, w[::-1], mode="valid")
    return pd.Series(r, index=s.index)

def rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    ag = pd.Series(g, index=s.index).ewm(alpha=1/p, adjust=False).mean()
    al = pd.Series(l, index=s.index).ewm(alpha=1/p, adjust=False).mean()
    return 100.0 - (100.0 / (1.0 + ag / (al + 1e-9)))

def atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr, index=df.index).ewm(alpha=1/p, adjust=False).mean()

def bollinger_bands(s: pd.Series, p: int = 20, std: float = 2.0):
    m = s.rolling(p).mean(); sd = s.rolling(p).std()
    return m + std*sd, m, m - std*sd

def keltner_channels(df: pd.DataFrame, ep: int = 20, ap: int = 10, m: float = 2.0):
    mid = ema(df["close"], ep); a = atr(df, ap)
    return mid + m*a, mid, mid - m*a

def donchian_channels(df: pd.DataFrame, p: int = 20):
    u = df["high"].rolling(p).max(); l = df["low"].rolling(p).min()
    return u, (u+l)*0.5, l

def supertrend(df: pd.DataFrame, p: int = 10, m: float = 3.0):
    a = atr(df, p).values; hl2 = (df["high"].values + df["low"].values) * 0.5
    ub_b, lb_b = hl2 + m*a, hl2 - m*a
    n = len(df); ub, lb, tr = np.copy(ub_b), np.copy(lb_b), np.ones(n, dtype=int)
    c = df["close"].values
    for i in range(1, n):
        ub[i] = ub_b[i] if ub_b[i] < ub[i-1] or c[i-1] > ub[i-1] else ub[i-1]
        lb[i] = lb_b[i] if lb_b[i] > lb[i-1] or c[i-1] < lb[i-1] else lb[i-1]
        tr[i] = 1 if c[i] > ub[i-1] else (-1 if c[i] < lb[i-1] else tr[i-1])
    return pd.Series(np.where(tr==1, lb, ub), index=df.index), pd.Series(tr, index=df.index)

def macd(s: pd.Series, f: int = 12, sl: int = 26, sig: int = 9):
    m = ema(s, f) - ema(s, sl); return m, ema(m, sig), m - ema(m, sig)

def btc_macro_filter(df_asset: pd.DataFrame, df_btc: pd.DataFrame,
                     fast_ema: int = 50, slow_ema: int = 200) -> pd.Series:
    btc_close = df_btc["close"].reindex(df_asset.index, method="ffill")
    e50 = ema(btc_close, fast_ema)
    e200 = ema(btc_close, slow_ema)
    regime = pd.Series(0, index=df_asset.index)
    regime[(btc_close > e50) & (e50 > e200)] = 1   # Bull
    regime[(btc_close < e50) & (e50 < e200)] = -1  # Bear
    return regime

def volume_confirmation(df: pd.DataFrame, lookback: int = 20, mult: float = 1.5) -> pd.Series:
    vol_ma = df["volume"].rolling(lookback).mean()
    return df["volume"] > (mult * vol_ma)

def rsi_divergence_bullish(df: pd.DataFrame, rsi_period: int = 14, lookback: int = 10) -> pd.Series:
    r = rsi(df["close"], rsi_period)
    price_low = df["low"].rolling(lookback).min()
    rsi_low = r.rolling(lookback).min()
    return (df["low"] <= price_low) & (rsi_low > rsi_low.shift(lookback))

def rsi_divergence_bearish(df: pd.DataFrame, rsi_period: int = 14, lookback: int = 10) -> pd.Series:
    r = rsi(df["close"], rsi_period)
    price_high = df["high"].rolling(lookback).max()
    rsi_high = r.rolling(lookback).max()
    return (df["high"] >= price_high) & (rsi_high < rsi_high.shift(lookback))

def volatility_regime(df: pd.DataFrame, short_window: int = 20, long_window: int = 60) -> pd.Series:
    rv_short = df["close"].pct_change().rolling(short_window).std()
    rv_long = df["close"].pct_change().rolling(long_window).std()
    regime = pd.Series(0, index=df.index)
    regime[rv_short > rv_long * 1.3] = 1   # High vol
    regime[rv_short < rv_long * 0.7] = -1  # Low vol
    return regime

def liquidation_cascade_proxy(df: pd.DataFrame, vol_mult: float = 3.0, drop_pct: float = 0.03) -> pd.Series:
    bar_return = df["close"].pct_change()
    vol_ma = df["volume"].rolling(20).mean()
    return (bar_return < -drop_pct) & (df["volume"] > (vol_mult * vol_ma))

def vwap_deviation(df: pd.DataFrame, window: int = 48) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_pv = (tp * df["volume"]).rolling(window).sum()
    cum_vol = df["volume"].rolling(window).sum()
    vwap = cum_pv / (cum_vol + 1e-9)
    return (df["close"] - vwap) / (vwap + 1e-9)
