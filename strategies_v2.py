"""
ENHANCED STRATEGY LIBRARY v2 — 20 Alpha-Enhanced Strategies.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from indicators_v2 import *

STRATEGY_REGISTRY_V2 = {}

def register(name, category):
    def deco(func):
        STRATEGY_REGISTRY_V2[name] = {"func": func, "category": category}
        return func
    return deco

@register("btc_filtered_ema_cross", "MacroFiltered")
def s_btc_ema_cross(df, df_btc=None, fast=9, slow=21, **kw):
    ef, es = ema(df["close"], fast), ema(df["close"], slow)
    cross_up = (ef > es) & (ef.shift(1) <= es.shift(1))
    cross_dn = (ef < es) & (ef.shift(1) >= es.shift(1))
    sig = pd.Series(0, index=df.index)
    if df_btc is not None:
        regime = btc_macro_filter(df, df_btc)
        sig[cross_up & (regime >= 0)] = 1
        sig[cross_dn & (regime <= 0)] = -1
    else:
        sig[cross_up] = 1; sig[cross_dn] = -1
    return sig

@register("btc_filtered_supertrend", "MacroFiltered")
def s_btc_supertrend(df, df_btc=None, period=10, mult=3.0, **kw):
    _, tr = supertrend(df, period, mult)
    flip_up = (tr == 1) & (tr.shift(1) == -1)
    flip_dn = (tr == -1) & (tr.shift(1) == 1)
    sig = pd.Series(0, index=df.index)
    if df_btc is not None:
        regime = btc_macro_filter(df, df_btc)
        sig[flip_up & (regime >= 0)] = 1
        sig[flip_dn & (regime <= 0)] = -1
    else:
        sig[flip_up] = 1; sig[flip_dn] = -1
    return sig

@register("btc_filtered_donchian", "MacroFiltered")
def s_btc_donchian(df, df_btc=None, period=20, **kw):
    u, _, l = donchian_channels(df, period)
    sig = pd.Series(0, index=df.index)
    brk_up = df["close"] > u.shift(1)
    brk_dn = df["close"] < l.shift(1)
    if df_btc is not None:
        regime = btc_macro_filter(df, df_btc)
        sig[brk_up & (regime >= 0)] = 1
        sig[brk_dn & (regime <= 0)] = -1
    else:
        sig[brk_up] = 1; sig[brk_dn] = -1
    return sig

@register("vol_confirmed_breakout", "VolumeConfirmed")
def s_vol_breakout(df, df_btc=None, period=20, vol_mult=1.5, **kw):
    u, _, l = donchian_channels(df, period)
    vol_ok = volume_confirmation(df, mult=vol_mult)
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > u.shift(1)) & vol_ok] = 1
    sig[(df["close"] < l.shift(1)) & vol_ok] = -1
    return sig

@register("vol_confirmed_squeeze", "VolumeConfirmed")
def s_vol_squeeze(df, df_btc=None, bb_p=20, bb_s=2.0, kc_m=1.5, vol_mult=1.5, **kw):
    bu, _, bl = bollinger_bands(df["close"], bb_p, bb_s)
    ku, _, kl = keltner_channels(df, bb_p, 10, kc_m)
    squeeze = (bu < ku) & (bl > kl)
    fired = ~squeeze & squeeze.shift(1)
    mom = df["close"] - sma(df["close"], 20)
    vol_ok = volume_confirmation(df, mult=vol_mult)
    sig = pd.Series(0, index=df.index)
    sig[fired & (mom > 0) & vol_ok] = 1
    sig[fired & (mom < 0) & vol_ok] = -1
    return sig

@register("vol_spike_reversal", "VolumeConfirmed")
def s_vol_spike_rev(df, df_btc=None, vol_mult=2.5, rsi_p=14, **kw):
    vol_ok = volume_confirmation(df, mult=vol_mult)
    r = rsi(df["close"], rsi_p)
    body = df["close"] - df["open"]
    rng = df["high"] - df["low"]
    bullish_candle = (body > 0) & (body > 0.5 * rng)
    bearish_candle = (body < 0) & (body < -0.5 * rng)
    sig = pd.Series(0, index=df.index)
    sig[vol_ok & bullish_candle & (r < 35)] = 1
    sig[vol_ok & bearish_candle & (r > 65)] = -1
    return sig

@register("rsi_divergence_reversal", "Divergence")
def s_rsi_div(df, df_btc=None, rsi_p=14, lookback=10, **kw):
    bull_div = rsi_divergence_bullish(df, rsi_p, lookback)
    bear_div = rsi_divergence_bearish(df, rsi_p, lookback)
    sig = pd.Series(0, index=df.index)
    sig[bull_div] = 1
    sig[bear_div] = -1
    return sig

@register("rsi_div_with_vol", "Divergence")
def s_rsi_div_vol(df, df_btc=None, rsi_p=14, lookback=10, vol_mult=1.3, **kw):
    bull_div = rsi_divergence_bullish(df, rsi_p, lookback)
    bear_div = rsi_divergence_bearish(df, rsi_p, lookback)
    vol_ok = volume_confirmation(df, mult=vol_mult)
    sig = pd.Series(0, index=df.index)
    sig[bull_div & vol_ok] = 1
    sig[bear_div & vol_ok] = -1
    return sig

@register("regime_adaptive_trend", "RegimeAdaptive")
def s_regime_trend(df, df_btc=None, fast=9, slow=21, **kw):
    ef, es = ema(df["close"], fast), ema(df["close"], slow)
    regime = volatility_regime(df)
    cross_up = (ef > es) & (ef.shift(1) <= es.shift(1))
    cross_dn = (ef < es) & (ef.shift(1) >= es.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up & (regime == 1)] = 1
    sig[cross_dn & (regime == 1)] = -1
    return sig

@register("regime_adaptive_meanrev", "RegimeAdaptive")
def s_regime_mr(df, df_btc=None, bb_p=20, bb_s=2.0, **kw):
    u, m, l = bollinger_bands(df["close"], bb_p, bb_s)
    regime = volatility_regime(df)
    sig = pd.Series(0, index=df.index)
    sig[(df["low"] <= l) & (df["close"] > l) & (regime == -1)] = 1
    sig[(df["high"] >= u) & (df["close"] < u) & (regime == -1)] = -1
    return sig

@register("regime_hybrid_switch", "RegimeAdaptive")
def s_regime_hybrid(df, df_btc=None, fast=9, slow=21, bb_p=20, bb_s=2.0, **kw):
    regime = volatility_regime(df)
    ef, es = ema(df["close"], fast), ema(df["close"], slow)
    u, m, l = bollinger_bands(df["close"], bb_p, bb_s)
    sig = pd.Series(0, index=df.index)
    sig[(ef > es) & (ef.shift(1) <= es.shift(1)) & (regime == 1)] = 1
    sig[(ef < es) & (ef.shift(1) >= es.shift(1)) & (regime == 1)] = -1
    sig[(df["low"] <= l) & (df["close"] > l) & (regime == -1)] = 1
    sig[(df["high"] >= u) & (df["close"] < u) & (regime == -1)] = -1
    return sig

@register("liquidation_cascade_buy", "Liquidation")
def s_liq_cascade(df, df_btc=None, vol_mult=3.0, drop_pct=0.03, **kw):
    cascade = liquidation_cascade_proxy(df, vol_mult, drop_pct)
    sig = pd.Series(0, index=df.index)
    sig[cascade.shift(1) & (df["close"] > df["open"])] = 1
    return sig

@register("liquidation_cascade_both", "Liquidation")
def s_liq_both(df, df_btc=None, vol_mult=3.0, drop_pct=0.03, **kw):
    vol_ma = df["volume"].rolling(20).mean()
    ret = df["close"].pct_change()
    big_drop = (ret < -drop_pct) & (df["volume"] > (vol_mult * vol_ma))
    big_pump = (ret > drop_pct) & (df["volume"] > (vol_mult * vol_ma))
    sig = pd.Series(0, index=df.index)
    sig[big_drop.shift(1) & (df["close"] > df["open"])] = 1
    sig[big_pump.shift(1) & (df["close"] < df["open"])] = -1
    return sig

@register("vwap_mean_reversion", "VWAP")
def s_vwap_mr(df, df_btc=None, window=48, dev_thresh=0.025, **kw):
    dev = vwap_deviation(df, window)
    sig = pd.Series(0, index=df.index)
    sig[(dev < -dev_thresh) & (dev.shift(1) >= -dev_thresh)] = 1
    sig[(dev > dev_thresh) & (dev.shift(1) <= dev_thresh)] = -1
    return sig

@register("vwap_trend_pullback", "VWAP")
def s_vwap_pullback(df, df_btc=None, window=48, ema_p=50, **kw):
    dev = vwap_deviation(df, window)
    e = ema(df["close"], ema_p)
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > e) & (dev < -0.01) & (dev.shift(1) >= -0.01)] = 1
    sig[(df["close"] < e) & (dev > 0.01) & (dev.shift(1) <= 0.01)] = -1
    return sig

@register("triple_confirmation_long", "Composite")
def s_triple_long(df, df_btc=None, **kw):
    r = rsi(df["close"], 14)
    vol_ok = volume_confirmation(df, mult=1.5)
    rsi_bounce = (r > 30) & (r.shift(1) <= 30)
    sig = pd.Series(0, index=df.index)
    if df_btc is not None:
        regime = btc_macro_filter(df, df_btc)
        sig[rsi_bounce & vol_ok & (regime == 1)] = 1
    else:
        sig[rsi_bounce & vol_ok] = 1
    return sig

@register("quad_confirmation", "Composite")
def s_quad(df, df_btc=None, **kw):
    _, tr = supertrend(df, 10, 3.0)
    r = rsi(df["close"], 14)
    vol_ok = volume_confirmation(df, mult=1.3)
    st_flip_up = (tr == 1) & (tr.shift(1) == -1)
    st_flip_dn = (tr == -1) & (tr.shift(1) == 1)
    sig = pd.Series(0, index=df.index)
    if df_btc is not None:
        regime = btc_macro_filter(df, df_btc)
        sig[st_flip_up & vol_ok & (r > 40) & (regime >= 0)] = 1
        sig[st_flip_dn & vol_ok & (r < 60) & (regime <= 0)] = -1
    else:
        sig[st_flip_up & vol_ok & (r > 40)] = 1
        sig[st_flip_dn & vol_ok & (r < 60)] = -1
    return sig

@register("full_stack_alpha", "Composite")
def s_full_stack(df, df_btc=None, **kw):
    _, tr = supertrend(df, 10, 3.0)
    vol_ok = volume_confirmation(df, mult=1.3)
    dev = vwap_deviation(df, 48)
    liq_against = liquidation_cascade_proxy(df, 3.0, 0.03)
    st_up = (tr == 1) & (tr.shift(1) == -1)
    st_dn = (tr == -1) & (tr.shift(1) == 1)
    sig = pd.Series(0, index=df.index)
    if df_btc is not None:
        regime = btc_macro_filter(df, df_btc)
        sig[st_up & vol_ok & (dev > -0.02) & ~liq_against & (regime >= 0)] = 1
        sig[st_dn & vol_ok & (dev < 0.02) & ~liq_against & (regime <= 0)] = -1
    else:
        sig[st_up & vol_ok & (dev > -0.02) & ~liq_against] = 1
        sig[st_dn & vol_ok & (dev < 0.02) & ~liq_against] = -1
    return sig

@register("tweezer_with_btc_filter", "Enhanced")
def s_tweezer_btc(df, df_btc=None, tolerance=0.005, ema_period=50, **kw):
    body = (df["close"] - df["open"]).abs()
    ls = df[["open","close"]].min(axis=1) - df["low"]
    us = df["high"] - df[["open","close"]].max(axis=1)
    hammer = (ls >= 2.0 * body) & (us <= 0.3 * body) & (body > 0)
    star = (us >= 2.0 * body) & (ls <= 0.3 * body) & (body > 0)
    e = ema(df["close"], ema_period)
    sig = pd.Series(0, index=df.index)
    if df_btc is not None:
        regime = btc_macro_filter(df, df_btc)
        sig[hammer & (df["close"] > e) & (regime >= 0)] = 1
        sig[star & (df["close"] < e) & (regime <= 0)] = -1
    else:
        sig[hammer & (df["close"] > e)] = 1
        sig[star & (df["close"] < e)] = -1
    return sig

@register("ema_cross_with_regime", "Enhanced")
def s_ema_regime(df, df_btc=None, fast=20, slow=50, **kw):
    ef, es = ema(df["close"], fast), ema(df["close"], slow)
    regime = volatility_regime(df)
    cross_up = (ef > es) & (ef.shift(1) <= es.shift(1))
    cross_dn = (ef < es) & (ef.shift(1) >= es.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up & (regime >= 0)] = 1
    sig[cross_dn & (regime <= 0)] = -1
    return sig
