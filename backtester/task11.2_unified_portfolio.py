"""
TASK 11.2: UNIFIED PORTFOLIO SIMULATION
Single cash pool, chronological event stream, compound sizing.
Every trade risks a fixed % of TOTAL account equity (not siloed).
"""
import os
import sys
import time
import warnings
import traceback
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

warnings.filterwarnings("ignore")

sys.path.insert(0, r"C:\BybitBacktest\backtester")
try:
    from engine_v2 import (
        __version__ as ENGINE_VER,
        MultiTFDataLoader, BacktestEngine, Strategy, RegimeDetector,
        RSI, MACD, ATR, EMA, SMA, ADX, HMA, Supertrend, Bollinger,
        Stochastic, StochRSI, VWAP, Keltner, Donchian, WilliamsR
    )
except ImportError as e:
    print(f"FATAL: engine_v2 import failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
    from colorama import Fore, Style, init as colorama_init
    from tabulate import tabulate
    colorama_init(autoreset=True)
except ImportError as e:
    print(f"Missing package: {e}"); sys.exit(1)

# ============================================================
# CONFIGURATION
# ============================================================
DATA_ROOT = r"C:\BybitBacktest\data\resampled"
RESULTS_ROOT = r"C:\BybitBacktest\results"
os.makedirs(RESULTS_ROOT, exist_ok=True)

IS_START = pd.Timestamp("2022-01-01")
IS_END   = pd.Timestamp("2025-12-31 23:59:59")
OOS_START = pd.Timestamp("2026-01-01")
OOS_END   = pd.Timestamp("2026-08-31 23:59:59")
FULL_START = IS_START
FULL_END = OOS_END

STARTING_CAPITAL = 10000.0
FEES = 0.00055
SLIPPAGE = 0.0003
MAX_GROSS_LEVERAGE = 3.0

RISK_TIERS = [
    ("Conservative_0.5%", 0.005),
    ("Standard_1.0%",     0.010),
    ("Growth_1.5%",       0.015),
    ("Aggressive_2.0%",   0.020),
]

C_CYAN=Fore.CYAN; C_YEL=Fore.YELLOW; C_GRN=Fore.GREEN
C_RED=Fore.RED; C_MAG=Fore.MAGENTA; C_WHT=Fore.WHITE
S_BR=Style.BRIGHT; S_RS=Style.RESET_ALL

def box(txt, color=C_CYAN):
    line = "=" * max(60, len(txt) + 4)
    print(color + S_BR + "+" + line + "+")
    print(color + S_BR + "|  " + txt.ljust(len(line) - 2) + "|")
    print(color + S_BR + "+" + line + "+" + S_RS)

def section(n, txt, color=C_MAG):
    print()
    print(color + S_BR + "-" * 78)
    print(color + S_BR + f"  SECTION {n}: {txt}")
    print(color + S_BR + "-" * 78 + S_RS)

# ============================================================
# THE 16 EDGES
# ============================================================
PORTFOLIO_16 = [
    {"id":1,  "strategy":"S30b_MidWeek_MeanReversion", "symbol":"XMR_USDT_USDT",  "tf":"1H", "family":"TimeBased"},
    {"id":2,  "strategy":"S30a_DOW_Seasonality",       "symbol":"ATOM_USDT_USDT", "tf":"4H", "family":"TimeBased"},
    {"id":3,  "strategy":"S30a_DOW_Seasonality",       "symbol":"ADA_USDT_USDT",  "tf":"4H", "family":"TimeBased"},
    {"id":4,  "strategy":"S30a_DOW_Seasonality",       "symbol":"BSV_USDT_USDT",  "tf":"4H", "family":"TimeBased"},
    {"id":5,  "strategy":"S30a_DOW_Seasonality",       "symbol":"ETC_USDT_USDT",  "tf":"4H", "family":"TimeBased"},
    {"id":6,  "strategy":"S30a_DOW_Seasonality",       "symbol":"AVAX_USDT_USDT", "tf":"4H", "family":"TimeBased"},
    {"id":7,  "strategy":"S30a_DOW_Seasonality",       "symbol":"QTUM_USDT_USDT", "tf":"4H", "family":"TimeBased"},
    {"id":8,  "strategy":"S30b_MidWeek_MeanReversion", "symbol":"TWT_USDT_USDT",  "tf":"4H", "family":"TimeBased"},
    {"id":9,  "strategy":"S30d_NY_London_Momentum",    "symbol":"DOT_USDT_USDT",  "tf":"1H", "family":"TimeBased"},
    {"id":10, "strategy":"S30d_NY_London_Momentum",    "symbol":"XLM_USDT_USDT",  "tf":"1H", "family":"TimeBased"},
    {"id":11, "strategy":"R13_ATR_Range_Expansion",    "symbol":"ETH_USDT_USDT",  "tf":"4H", "family":"RegimeAdaptive"},
    {"id":12, "strategy":"R15_Volume_Breakout",        "symbol":"SAND_USDT_USDT", "tf":"4H", "family":"RegimeAdaptive"},
    {"id":13, "strategy":"R01_EMA_Cross",              "symbol":"OP_USDT_USDT",   "tf":"4H", "family":"RegimeAdaptive"},
    {"id":14, "strategy":"R15_Volume_Breakout",        "symbol":"OP_USDT_USDT",   "tf":"4H", "family":"RegimeAdaptive"},
    {"id":15, "strategy":"R01_EMA_Cross",              "symbol":"INJ_USDT_USDT",  "tf":"1H", "family":"RegimeAdaptive"},
    {"id":16, "strategy":"R16_Trend_Plus_RSI_Pullback","symbol":"DOGE_USDT_USDT", "tf":"4H", "family":"RegimeAdaptive"},
]

# ============================================================
# STRATEGY IMPLEMENTATIONS
# ============================================================
def _sig(df, lm, sm):
    n = len(df); a = np.zeros(n, dtype=int)
    a[lm.values] = 1; a[sm.values] = -1
    both = lm.values & sm.values; a[both] = 1
    return pd.Series(a, index=df.index, dtype=int)

def _sltp(df, sig, atr_s, sl_mult, tp_mult):
    close = df["close"]
    sl = pd.Series(np.nan, index=df.index, dtype=float)
    tp = pd.Series(np.nan, index=df.index, dtype=float)
    L = sig == 1; S = sig == -1
    sl.loc[L] = (close - sl_mult * atr_s).loc[L].values
    sl.loc[S] = (close + sl_mult * atr_s).loc[S].values
    tp.loc[L] = (close + tp_mult * atr_s).loc[L].values
    tp.loc[S] = (close - tp_mult * atr_s).loc[S].values
    return sl, tp

def _regime_align(df, regime_series):
    if regime_series is None or len(regime_series) == 0:
        return pd.Series(["UNKNOWN"] * len(df), index=df.index)
    rs = regime_series.copy()
    if not isinstance(rs.index, pd.DatetimeIndex):
        rs.index = pd.to_datetime(rs.index)
    return rs.reindex(df.index, method="ffill").fillna("UNKNOWN")

def _gate(sig, reg, allow_L, allow_S):
    rv = reg.values; sv = sig.values.copy()
    lm = sv == 1; sm = sv == -1
    aL = np.isin(rv, allow_L); aS = np.isin(rv, allow_S)
    sv[lm & ~aL] = 0; sv[sm & ~aS] = 0
    return pd.Series(sv, index=sig.index, dtype=int)

def strat_S30a_DOW(df, regime=None):
    dow = pd.Series(df.index.dayofweek, index=df.index)
    r = RSI(df["close"], 14); a = ATR(df, 14)
    lm = ((dow == 4) & (r > 50)).fillna(False).astype(bool)
    sm = ((dow == 3) & (r < 50)).fillna(False).astype(bool)
    sig = _sig(df, lm, sm)
    sl, tp = _sltp(df, sig, a, 1.5, 2.5)
    return sig, sl, tp

def strat_S30b_MidWeek(df, regime=None):
    sma = SMA(df["close"], 20); a = ATR(df, 14)
    dow = pd.Series(df.index.dayofweek, index=df.index)
    dist = df["close"] - sma
    far_above = (dist > 1.5 * a).fillna(False).astype(bool)
    far_below = (dist < -1.5 * a).fillna(False).astype(bool)
    trig = ((dow == 1) | (dow == 2)).astype(bool)
    sm = (far_above & trig).fillna(False).astype(bool)
    lm = (far_below & trig).fillna(False).astype(bool)
    sig = _sig(df, lm, sm)
    sl, tp = _sltp(df, sig, a, 2.0, 1.5)
    return sig, sl, tp

def strat_S30d_NY_London(df, regime=None):
    a = ATR(df, 14)
    hour = pd.Series(df.index.hour, index=df.index)
    date = pd.Series(df.index.date, index=df.index)
    df_tmp = df.copy(); df_tmp["date"] = date.values
    in_london = (hour >= 8) & (hour < 12)
    df_tmp["in_london"] = in_london.values
    grp = df_tmp[df_tmp["in_london"]].groupby("date")
    lhi = grp["high"].max(); llo = grp["low"].min()
    lh = pd.Series(date.map(lhi).values, index=df.index)
    ll = pd.Series(date.map(llo).values, index=df.index)
    atr_ref = a.shift(4).bfill()
    atr_exp = a > 1.2 * atr_ref
    in_trade = (hour >= 12) & (hour < 16)
    long_b = df["close"] > lh; short_b = df["close"] < ll
    lm = (in_trade & long_b & atr_exp).fillna(False).astype(bool)
    sm = (in_trade & short_b & atr_exp).fillna(False).astype(bool)
    sig = _sig(df, lm, sm)
    sl, tp = _sltp(df, sig, a, 1.5, 2.5)
    return sig, sl, tp

def strat_R01_EMA_Cross(df, regime):
    e_fast = EMA(df["close"], 12); e_slow = EMA(df["close"], 26)
    a = ATR(df, 14)
    cu = ((e_fast > e_slow) & (e_fast.shift(1) <= e_slow.shift(1))).fillna(False).astype(bool)
    cd = ((e_fast < e_slow) & (e_fast.shift(1) >= e_slow.shift(1))).fillna(False).astype(bool)
    sig = _sig(df, cu, cd)
    reg = _regime_align(df, regime)
    sig = _gate(sig, reg, ["BULL"], ["BEAR"])
    close = df["close"]
    sl = pd.Series(np.nan, index=df.index, dtype=float)
    tp = pd.Series(np.nan, index=df.index, dtype=float)
    L = sig == 1; S = sig == -1
    sl_l_raw = df["low"].rolling(20).min(); sl_s_raw = df["high"].rolling(20).max()
    sl.loc[L] = sl_l_raw.loc[L].values
    sl.loc[S] = sl_s_raw.loc[S].values
    tp.loc[L] = (close + 3.0 * a).loc[L].values
    tp.loc[S] = (close - 1.5 * a).loc[S].values
    return sig, sl, tp

def strat_R13_ATR_Range_Expansion(df, regime):
    a = ATR(df, 14)
    br = df["high"] - df["low"]
    big = br > 1.8 * a
    up = df["close"] > df["open"]; dn = df["close"] < df["open"]
    lm = (big & up).fillna(False).astype(bool)
    sm = (big & dn).fillna(False).astype(bool)
    sig = _sig(df, lm, sm)
    reg = _regime_align(df, regime)
    sig = _gate(sig, reg, ["BULL"], ["BEAR"])
    sl, tp = _sltp(df, sig, a, 2.0, 3.0)
    return sig, sl, tp

def strat_R15_Volume_Breakout(df, regime):
    vol_avg = SMA(df["volume"], 20)
    hv = df["volume"] > 2.5 * vol_avg
    br = df["high"] - df["low"]
    pos = (df["close"] - df["low"]) / br.replace(0, np.nan)
    near_high = pos > 0.7; near_low = pos < 0.3
    lm = (hv & near_high).fillna(False).astype(bool)
    sm = (hv & near_low).fillna(False).astype(bool)
    sig = _sig(df, lm, sm)
    reg = _regime_align(df, regime)
    sig = _gate(sig, reg, ["BULL","CHOP"], ["BEAR","CHOP"])
    a = ATR(df, 14)
    sl, tp = _sltp(df, sig, a, 1.5, 2.5)
    return sig, sl, tp

def strat_R16_Trend_Plus_RSI_Pullback(df, regime):
    e = EMA(df["close"], 50); es = e.diff(5)
    r = RSI(df["close"], 14)
    bp = ((es > 0) & (r >= 40) & (r <= 50)).fillna(False).astype(bool)
    sp = ((es < 0) & (r >= 50) & (r <= 60)).fillna(False).astype(bool)
    lm = (bp & ~bp.shift(1).fillna(True)).fillna(False).astype(bool)
    sm = (sp & ~sp.shift(1).fillna(True)).fillna(False).astype(bool)
    sig = _sig(df, lm, sm)
    reg = _regime_align(df, regime)
    sig = _gate(sig, reg, ["BULL"], ["BEAR"])
    a = ATR(df, 14)
    sl, tp = _sltp(df, sig, a, 1.5, 2.5)
    return sig, sl, tp

STRATEGY_MAP = {
    "S30a_DOW_Seasonality": strat_S30a_DOW,
    "S30b_MidWeek_MeanReversion": strat_S30b_MidWeek,
    "S30d_NY_London_Momentum": strat_S30d_NY_London,
    "R01_EMA_Cross": strat_R01_EMA_Cross,
    "R13_ATR_Range_Expansion": strat_R13_ATR_Range_Expansion,
    "R15_Volume_Breakout": strat_R15_Volume_Breakout,
    "R16_Trend_Plus_RSI_Pullback": strat_R16_Trend_Plus_RSI_Pullback,
}

# ============================================================
# SIGNAL EXTRACTION: Build raw signal events per edge (with SL/TP at signal bar close, entry at next bar open)
# ============================================================
def extract_signal_events(edge, loader, regime_series, start, end):
    """Returns list of pending signal events:
       [{edge_id, symbol, tf, signal_time, entry_time, entry_price_raw, sl_price, tp_price, direction}, ...]
    """
    sname = edge["strategy"]; sym = edge["symbol"]; tf = edge["tf"]
    fn = STRATEGY_MAP.get(sname)
    if fn is None: return []
    df = loader.load(sym, tf, start, end)
    if df is None or len(df) < 50: return []
    try:
        needs_regime = sname.startswith("R")
        if needs_regime:
            sig, sl, tp = fn(df, regime_series)
        else:
            sig, sl, tp = fn(df)
    except Exception as e:
        return []

    events = []
    sig_arr = sig.values
    sl_arr = sl.values
    tp_arr = tp.values
    idx = df.index
    o = df["open"].values
    n = len(df)
    for i in range(n - 1):  # need i+1 for entry
        s = sig_arr[i]
        if s == 0: continue
        if np.isnan(sl_arr[i]) or np.isnan(tp_arr[i]): continue
        entry_raw = o[i+1]
        if np.isnan(entry_raw): continue
        events.append({
            "edge_id": edge["id"],
            "strategy": edge["strategy"],
            "symbol": sym,
            "tf": tf,
            "family": edge["family"],
            "signal_time": idx[i],
            "entry_time": idx[i+1],
            "entry_price_raw": float(entry_raw),
            "sl_price": float(sl_arr[i]),
            "tp_price": float(tp_arr[i]),
            "direction": int(s),  # 1 or -1
        })
    return events

# ============================================================
# LOAD FULL OHLCV FOR EACH SYMBOL/TF for exit-checking
# ============================================================
def load_all_ohlcv(loader, edges, start, end):
    """Preloads OHLCV DataFrames keyed by (symbol, tf)."""
    cache = {}
    keys = set((e["symbol"], e["tf"]) for e in edges)
    for sym, tf in keys:
        df = loader.load(sym, tf, start, end)
        if df is not None and len(df) > 0:
            cache[(sym, tf)] = df
    return cache

# ============================================================
# UNIFIED PORTFOLIO SIMULATOR
# ============================================================
def simulate_unified_portfolio(all_events, ohlcv_cache, risk_pct, tier_label,
                                start_capital=STARTING_CAPITAL, verbose=False):
    """
    Runs a single-cash-pool simulation.
    - Sorts events by signal_time.
    - For each signal, checks if symbol already has open position (skip if yes).
    - Checks portfolio gross leverage constraint.
    - Computes position size using CURRENT equity (compound).
    - Tracks exit via bar-by-bar SL/TP scan on the entry TF.

    Returns: {trades:list, equity_curve:pd.Series, final_equity:float, max_concurrent:int,
              peak_leverage:float, avg_utilization:float}
    """
    # Sort events chronologically by entry_time
    events = sorted(all_events, key=lambda e: e["entry_time"])

    equity = start_capital
    open_positions = {}  # symbol -> position dict
    closed_trades = []
    eq_history = []  # [(timestamp, equity, gross_notional)]
    eq_history.append((events[0]["entry_time"] if events else start, equity, 0.0))

    max_concurrent = 0
    peak_leverage = 0.0
    utilization_samples = []

    # Precompute per-position exit tracking: for each open position, we need to iterate its TF bars
    # We'll do this lazily by tracking each position's TF DataFrame reference and next bar index.

    def open_gross_notional(open_pos_dict):
        return sum(p["size"] * p["current_ref_price"] for p in open_pos_dict.values())

    def try_close_positions_before(check_time):
        """Advance all open positions bar-by-bar until check_time, checking SL/TP hits."""
        nonlocal equity
        to_close = []
        for sym, pos in open_positions.items():
            df = pos["df_ref"]
            tf_idx = pos["current_bar_idx"]
            n = len(df)
            while tf_idx < n and df.index[tf_idx] < check_time:
                # Skip bars before entry itself
                if df.index[tf_idx] <= pos["entry_time"]:
                    tf_idx += 1
                    continue
                bar = df.iloc[tf_idx]
                hi = bar["high"]; lo = bar["low"]; op = bar["open"]
                exit_px = None; reason = None
                if pos["direction"] == 1:
                    if op <= pos["sl"]:
                        exit_px = op * (1 - SLIPPAGE); reason = "SL_GAP"
                    elif lo <= pos["sl"]:
                        exit_px = pos["sl"] * (1 - SLIPPAGE); reason = "SL"
                    elif hi >= pos["tp"]:
                        exit_px = pos["tp"] * (1 - SLIPPAGE); reason = "TP"
                else:
                    if op >= pos["sl"]:
                        exit_px = op * (1 + SLIPPAGE); reason = "SL_GAP"
                    elif hi >= pos["sl"]:
                        exit_px = pos["sl"] * (1 + SLIPPAGE); reason = "SL"
                    elif lo <= pos["tp"]:
                        exit_px = pos["tp"] * (1 + SLIPPAGE); reason = "TP"
                if exit_px is not None:
                    size = pos["size"]; entry_px = pos["entry_price"]
                    if pos["direction"] == 1:
                        gross = (exit_px - entry_px) * size
                    else:
                        gross = (entry_px - exit_px) * size
                    fee_cost = (entry_px + exit_px) * size * FEES
                    net = gross - fee_cost
                    equity += net
                    pnl_pct = (net / (entry_px * size)) * 100 if entry_px * size > 0 else 0
                    trade_rec = {
                        "edge_id": pos["edge_id"],
                        "strategy": pos["strategy"],
                        "symbol": sym,
                        "tf": pos["tf"],
                        "family": pos["family"],
                        "direction": "LONG" if pos["direction"] == 1 else "SHORT",
                        "entry_time": pos["entry_time"],
                        "exit_time": df.index[tf_idx],
                        "entry_price": round(entry_px, 8),
                        "exit_price": round(exit_px, 8),
                        "size": round(size, 6),
                        "sl": round(pos["sl_init"], 8),
                        "tp": round(pos["tp_init"], 8),
                        "pnl": round(net, 4),
                        "pnl_pct": round(pnl_pct, 4),
                        "exit_reason": reason,
                        "equity_after": round(equity, 4),
                        "risk_pct": risk_pct,
                        "tier": tier_label,
                    }
                    closed_trades.append(trade_rec)
                    eq_history.append((df.index[tf_idx], equity, open_gross_notional({k:v for k,v in open_positions.items() if k != sym})))
                    to_close.append(sym)
                    break
                else:
                    pos["current_ref_price"] = bar["close"]
                    tf_idx += 1
            pos["current_bar_idx"] = tf_idx

        for s in to_close:
            del open_positions[s]

    # Process events chronologically
    for ev_idx, event in enumerate(events):
        entry_time = event["entry_time"]
        sym = event["symbol"]; tf = event["tf"]

        # First close any positions that would have exited before this event
        try_close_positions_before(entry_time)

        # Skip if symbol already has open position
        if sym in open_positions:
            continue

        # Compute position size using CURRENT equity
        direction = event["direction"]
        entry_raw = event["entry_price_raw"]
        sl_v = event["sl_price"]
        tp_v = event["tp_price"]

        # Apply slippage to entry
        if direction == 1:
            entry_px = entry_raw * (1 + SLIPPAGE)
            if sl_v >= entry_px or tp_v <= entry_px: continue
            risk_per_unit = entry_px - sl_v
        else:
            entry_px = entry_raw * (1 - SLIPPAGE)
            if sl_v <= entry_px or tp_v >= entry_px: continue
            risk_per_unit = sl_v - entry_px

        if risk_per_unit <= 0 or np.isnan(risk_per_unit): continue

        risk_dollars = equity * risk_pct
        size = risk_dollars / risk_per_unit
        if size <= 0: continue

        # Check gross leverage constraint
        proposed_notional = size * entry_px
        current_gross = open_gross_notional(open_positions)
        if (current_gross + proposed_notional) > (equity * MAX_GROSS_LEVERAGE):
            # Reduce size to fit or skip
            available = max(equity * MAX_GROSS_LEVERAGE - current_gross, 0)
            if available < 1.0:
                continue
            size = available / entry_px
            if size * risk_per_unit < risk_dollars * 0.1:
                continue

        # Load exit-tracking OHLCV
        df_ref = ohlcv_cache.get((sym, tf))
        if df_ref is None or len(df_ref) < 5: continue
        # Find index of entry_time in df_ref
        try:
            bar_idx = df_ref.index.get_indexer([entry_time], method="nearest")[0]
        except Exception:
            continue
        if bar_idx < 0 or bar_idx >= len(df_ref): continue

        # Open position
        open_positions[sym] = {
            "edge_id": event["edge_id"],
            "strategy": event["strategy"],
            "tf": tf,
            "family": event["family"],
            "direction": direction,
            "entry_time": entry_time,
            "entry_price": entry_px,
            "sl": sl_v, "tp": tp_v,
            "sl_init": sl_v, "tp_init": tp_v,
            "size": size,
            "df_ref": df_ref,
            "current_bar_idx": bar_idx,
            "current_ref_price": entry_px,
        }
        # Track concurrency
        if len(open_positions) > max_concurrent:
            max_concurrent = len(open_positions)
        # Track leverage
        gross_now = open_gross_notional(open_positions)
        lev_now = gross_now / equity if equity > 0 else 0
        if lev_now > peak_leverage:
            peak_leverage = lev_now
        utilization_samples.append((gross_now / equity) if equity > 0 else 0)

    # Close any remaining positions at end (mark to last close)
    for sym, pos in list(open_positions.items()):
        df = pos["df_ref"]
        last_close = df["close"].iloc[-1]
        if pos["direction"] == 1:
            exit_px = last_close * (1 - SLIPPAGE)
            gross = (exit_px - pos["entry_price"]) * pos["size"]
        else:
            exit_px = last_close * (1 + SLIPPAGE)
            gross = (pos["entry_price"] - exit_px) * pos["size"]
        fee_cost = (pos["entry_price"] + exit_px) * pos["size"] * FEES
        net = gross - fee_cost
        equity += net
        pnl_pct = (net / (pos["entry_price"] * pos["size"])) * 100 if pos["entry_price"] * pos["size"] > 0 else 0
        closed_trades.append({
            "edge_id": pos["edge_id"],
            "strategy": pos["strategy"],
            "symbol": sym,
            "tf": pos["tf"],
            "family": pos["family"],
            "direction": "LONG" if pos["direction"] == 1 else "SHORT",
            "entry_time": pos["entry_time"],
            "exit_time": df.index[-1],
            "entry_price": round(pos["entry_price"], 8),
            "exit_price": round(exit_px, 8),
            "size": round(pos["size"], 6),
            "sl": round(pos["sl_init"], 8),
            "tp": round(pos["tp_init"], 8),
            "pnl": round(net, 4),
            "pnl_pct": round(pnl_pct, 4),
            "exit_reason": "EOD",
            "equity_after": round(equity, 4),
            "risk_pct": risk_pct,
            "tier": tier_label,
        })
        eq_history.append((df.index[-1], equity, 0.0))
    open_positions.clear()

    # Build equity Series
    if not eq_history:
        return None
    eq_history.sort(key=lambda x: x[0])
    times = [h[0] for h in eq_history]
    vals = [h[1] for h in eq_history]
    eq_series = pd.Series(vals, index=pd.DatetimeIndex(times))
    # Deduplicate index
    eq_series = eq_series[~eq_series.index.duplicated(keep="last")]

    avg_util = float(np.mean(utilization_samples)) if utilization_samples else 0.0

    return {
        "trades": closed_trades,
        "equity_curve": eq_series,
        "final_equity": equity,
        "max_concurrent": max_concurrent,
        "peak_leverage": round(peak_leverage, 3),
        "avg_utilization_pct": round(avg_util * 100, 2),
    }

# ============================================================
# METRICS
# ============================================================
def compute_metrics(sim_result, start_cap):
    if sim_result is None: return None
    trades = sim_result["trades"]
    eq = sim_result["equity_curve"]
    m = {"final_equity": round(sim_result["final_equity"], 2),
         "total_trades": len(trades),
         "net_profit": round(sim_result["final_equity"] - start_cap, 2),
         "total_return_pct": round((sim_result["final_equity"] / start_cap - 1) * 100, 3),
         "max_concurrent": sim_result["max_concurrent"],
         "peak_leverage": sim_result["peak_leverage"],
         "avg_utilization_pct": sim_result["avg_utilization_pct"]}
    if len(eq) < 3 or not trades:
        m.update({"cagr_pct":0,"sharpe":0,"sortino":0,"calmar":0,
                 "max_dd_pct":0,"win_rate":0,"profit_factor":0,"expectancy_dollar":0})
        return m
    days = (eq.index[-1] - eq.index[0]).days
    years = max(days / 365.25, 0.01)
    if eq.iloc[0] > 0:
        m["cagr_pct"] = round((eq.iloc[-1] / eq.iloc[0]) ** (1/years) * 100 - 100, 2)
    else:
        m["cagr_pct"] = 0
    # Daily equity for sharpe
    try:
        d = eq.resample("D").last().ffill()
        dr = d.pct_change().dropna()
        if dr.std() > 0:
            m["sharpe"] = round(float(dr.mean() / dr.std() * np.sqrt(365)), 3)
        else:
            m["sharpe"] = 0
        neg = dr[dr < 0]
        if len(neg) > 0 and neg.std() > 0:
            m["sortino"] = round(float(dr.mean() / neg.std() * np.sqrt(365)), 3)
        else:
            m["sortino"] = 999
    except Exception:
        m["sharpe"] = 0; m["sortino"] = 0
    peak = eq.cummax()
    dd = (eq - peak) / peak
    m["max_dd_pct"] = round(float(dd.min() * 100), 2)
    m["calmar"] = round(m["cagr_pct"] / abs(m["max_dd_pct"]), 3) if m["max_dd_pct"] < 0 else 0
    pnls = np.array([t["pnl"] for t in trades])
    wins = (pnls > 0).sum()
    m["win_rate"] = round(100 * wins / len(pnls), 2)
    gp = pnls[pnls > 0].sum(); gl = abs(pnls[pnls < 0].sum())
    m["profit_factor"] = round(gp / gl, 3) if gl > 0 else 999
    aw = pnls[pnls > 0].mean() if wins > 0 else 0
    al = abs(pnls[pnls < 0].mean()) if (pnls < 0).sum() > 0 else 0
    pw = wins / len(pnls)
    m["expectancy_dollar"] = round(pw * aw - (1 - pw) * al, 4)
    return m

def slice_trades_period(trades, start, end):
    return [t for t in trades if start <= t["exit_time"] <= end]

def slice_equity_period(eq, start, end):
    return eq[(eq.index >= start) & (eq.index <= end)]

# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    box("TASK 11.2: UNIFIED PORTFOLIO SIMULATION", C_CYAN)
    print(f"{C_CYAN}  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{C_CYAN}  Engine: v{ENGINE_VER}")
    print(f"{C_CYAN}  Starting capital: ${STARTING_CAPITAL:,.2f}")
    print(f"{C_CYAN}  Max gross leverage: {MAX_GROSS_LEVERAGE}x")
    print(f"{C_CYAN}  Risk tiers: {[t[0] for t in RISK_TIERS]}")

    loader = MultiTFDataLoader(DATA_ROOT)

    # Regime
    print(f"\n  Precomputing BTC regime classification...")
    detector = RegimeDetector(data_loader=loader)
    regime_series = detector.classify_series("2022-01-01", "2026-08-31")
    if not isinstance(regime_series.index, pd.DatetimeIndex):
        regime_series.index = pd.to_datetime(regime_series.index)
    print(f"  {C_GRN}[OK] Regime series: {len(regime_series)} days{S_RS}")

    # Preload OHLCV
    section(1, "PRELOADING OHLCV FOR ALL 16 EDGE SYMBOL/TFs", C_CYAN)
    ohlcv_cache = load_all_ohlcv(loader, PORTFOLIO_16, FULL_START, FULL_END)
    print(f"  {C_GRN}[OK] Loaded {len(ohlcv_cache)} unique (symbol,tf) datasets{S_RS}")

    # Extract signal events for all 16 edges
    section(2, "EXTRACTING SIGNAL EVENTS FROM 16 EDGES (2022-2026)", C_CYAN)
    all_events = []
    per_edge_count = {}
    for edge in PORTFOLIO_16:
        evs = extract_signal_events(edge, loader, regime_series, FULL_START, FULL_END)
        all_events.extend(evs)
        per_edge_count[edge["id"]] = len(evs)
        print(f"  Edge #{edge['id']:2d} {edge['strategy'][:26]:<26} | {edge['symbol']:<14} | {edge['tf']:<3} | Signals: {len(evs)}")
    print(f"\n  {C_GRN}[OK] Total raw signal events across 16 edges: {len(all_events):,}{S_RS}")

    # Sort chronologically
    all_events.sort(key=lambda e: e["entry_time"])

    # Run simulation for each risk tier
    section(3, "SIMULATING 4 RISK TIERS (Unified Cash Pool + Chronological)", C_MAG)
    tier_results = {}
    for tier_label, risk_pct in RISK_TIERS:
        print(f"\n  {C_YEL}Simulating {tier_label} (risk={risk_pct*100:.1f}% per trade)...{S_RS}")
        sim = simulate_unified_portfolio(all_events, ohlcv_cache, risk_pct, tier_label,
                                          start_capital=STARTING_CAPITAL)
        if sim is None:
            print(f"    {C_RED}[FAIL] Simulation returned None{S_RS}")
            continue
        m_full = compute_metrics(sim, STARTING_CAPITAL)
        # OOS-only metrics
        oos_trades = slice_trades_period(sim["trades"], OOS_START, OOS_END)
        oos_eq = slice_equity_period(sim["equity_curve"], OOS_START, OOS_END)
        if len(oos_eq) > 2:
            oos_start_eq = oos_eq.iloc[0]
            oos_sim = {"trades": oos_trades, "equity_curve": oos_eq,
                       "final_equity": oos_eq.iloc[-1], "max_concurrent": sim["max_concurrent"],
                       "peak_leverage": sim["peak_leverage"],
                       "avg_utilization_pct": sim["avg_utilization_pct"]}
            m_oos = compute_metrics(oos_sim, oos_start_eq)
        else:
            m_oos = None
        tier_results[tier_label] = {"sim": sim, "metrics_full": m_full, "metrics_oos": m_oos, "risk_pct": risk_pct}
        print(f"    Full: Final ${m_full['final_equity']:,.2f} | "
              f"Return {m_full['total_return_pct']:+.2f}% | "
              f"Sharpe {m_full['sharpe']:.2f} | MaxDD {m_full['max_dd_pct']:.2f}%")
        if m_oos:
            print(f"    OOS : Return {m_oos['total_return_pct']:+.2f}% | "
                  f"Sharpe {m_oos['sharpe']:.2f} | MaxDD {m_oos['max_dd_pct']:.2f}% | "
                  f"Trades {m_oos['total_trades']}")

    if not tier_results:
        print(f"  {C_RED}[!] No tier simulations succeeded. Aborting.{S_RS}")
        return

    # SECTION 4: Sizing tier comparison
    section(4, "SIZING TIER COMPARISON MATRIX", C_CYAN)
    print(f"\n  {C_CYAN}Full Period (2022-2026):{S_RS}")
    full_rows = []
    for tier_label, res in tier_results.items():
        m = res["metrics_full"]
        col = C_GRN if m["total_return_pct"] > 0 else C_RED
        full_rows.append([
            tier_label, f"${STARTING_CAPITAL:,.0f}",
            f"${m['final_equity']:,.2f}",
            f"{col}${m['net_profit']:+,.2f}{S_RS}",
            f"{col}{m['total_return_pct']:+.2f}%{S_RS}",
            f"{m['cagr_pct']:.2f}%",
            f"{m['sharpe']:.2f}",
            f"{m['max_dd_pct']:.2f}%",
            f"{m['calmar']:.2f}",
            f"{m['win_rate']:.1f}%",
            f"{m['total_trades']:,}",
        ])
    print(tabulate(full_rows,
        headers=["Tier","Start","Final Equity","Net Profit","Return","CAGR","Sharpe","MaxDD","Calmar","Win%","Trades"],
        tablefmt="grid"))

    print(f"\n  {C_MAG}2026 OOS Only:{S_RS}")
    oos_rows = []
    for tier_label, res in tier_results.items():
        m = res["metrics_oos"]
        if m is None:
            oos_rows.append([tier_label, "N/A"]*11); continue
        col = C_GRN if m["total_return_pct"] > 0 else C_RED
        oos_rows.append([
            tier_label,
            f"{col}{m['total_return_pct']:+.2f}%{S_RS}",
            f"${m['final_equity']:,.2f}",
            f"{col}${m['net_profit']:+,.2f}{S_RS}",
            f"{m['sharpe']:.2f}",
            f"{m['max_dd_pct']:.2f}%",
            f"{m['calmar']:.2f}",
            f"{m['win_rate']:.1f}%",
            f"{m['profit_factor']:.2f}",
            f"{m['total_trades']:,}",
        ])
    print(tabulate(oos_rows,
        headers=["Tier","OOS Ret","End Eq","Net","Sharpe","MaxDD","Calmar","Win%","PF","Trades"],
        tablefmt="grid"))

    # SECTION 5: 2026 month-by-month for 1.0% and 2.0%
    section(5, "2026 OOS MONTH-BY-MONTH HEATMAP (1.0% and 2.0% tiers)", C_MAG)
    for target_tier in ["Standard_1.0%", "Aggressive_2.0%"]:
        if target_tier not in tier_results: continue
        res = tier_results[target_tier]
        sim = res["sim"]
        oos_eq = slice_equity_period(sim["equity_curve"], OOS_START, OOS_END)
        oos_trades = slice_trades_period(sim["trades"], OOS_START, OOS_END)
        if len(oos_eq) < 2: continue
        print(f"\n  {C_CYAN}Tier: {target_tier}{S_RS}")
        mo_rows = []
        mo_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug"]
        for mi in range(1, 9):
            ms = pd.Timestamp(f"2026-{mi:02d}-01")
            if mi == 12:
                me = pd.Timestamp("2027-01-01") - pd.Timedelta(seconds=1)
            else:
                me = pd.Timestamp(f"2026-{mi+1:02d}-01") - pd.Timedelta(seconds=1)
            mo_eq = slice_equity_period(oos_eq, ms, me)
            mo_trades = [t for t in oos_trades if ms <= t["exit_time"] <= me]
            if len(mo_eq) < 2:
                mo_ret_pct = 0; mo_dollar = 0
            else:
                mo_start = mo_eq.iloc[0]; mo_end = mo_eq.iloc[-1]
                mo_ret_pct = (mo_end / mo_start - 1) * 100 if mo_start > 0 else 0
                mo_dollar = mo_end - mo_start
            mo_wins = sum(1 for t in mo_trades if t["pnl"] > 0)
            wr = 100 * mo_wins / len(mo_trades) if mo_trades else 0
            col = C_GRN if mo_ret_pct > 0 else (C_RED if mo_ret_pct < 0 else C_YEL)
            mo_rows.append([
                mo_names[mi-1],
                f"{col}{mo_ret_pct:+.2f}%{S_RS}",
                f"{col}${mo_dollar:+.2f}{S_RS}",
                len(mo_trades),
                f"{wr:.1f}%" if mo_trades else "-"
            ])
        print(tabulate(mo_rows, headers=["Month","Return %","Return $","Trades","Win%"], tablefmt="grid"))

    # SECTION 6: Annual progression
    section(6, "ANNUAL EQUITY PROGRESSION (compound growth)", C_CYAN)
    for target_tier in ["Standard_1.0%", "Aggressive_2.0%"]:
        if target_tier not in tier_results: continue
        res = tier_results[target_tier]
        eq = res["sim"]["equity_curve"]
        trades = res["sim"]["trades"]
        print(f"\n  {C_CYAN}Tier: {target_tier}{S_RS}")
        ann_rows = []
        year_start_eq = STARTING_CAPITAL
        for yr in [2022, 2023, 2024, 2025, 2026]:
            y_start = pd.Timestamp(f"{yr}-01-01")
            y_end = pd.Timestamp(f"{yr}-12-31 23:59:59") if yr < 2026 else OOS_END
            y_eq = slice_equity_period(eq, y_start, y_end)
            if len(y_eq) < 2:
                ann_rows.append([yr, "-", "-", "-", "-", "N/A"]); continue
            y_start_val = y_eq.iloc[0]
            y_end_val = y_eq.iloc[-1]
            y_ret = (y_end_val / y_start_val - 1) * 100 if y_start_val > 0 else 0
            y_dollar = y_end_val - y_start_val
            y_trades = [t for t in trades if y_start <= t["exit_time"] <= y_end]
            col = C_GRN if y_ret > 0 else C_RED
            ann_rows.append([
                yr,
                f"${y_start_val:,.2f}",
                f"${y_end_val:,.2f}",
                f"{col}${y_dollar:+,.2f}{S_RS}",
                f"{col}{y_ret:+.2f}%{S_RS}",
                len(y_trades),
                "OOS" if yr == 2026 else "IS"
            ])
        print(tabulate(ann_rows,
            headers=["Year","Start Equity","End Equity","Net","Return","Trades","Period"],
            tablefmt="grid"))

    # SECTION 7: Exposure audit
    section(7, "PORTFOLIO EXPOSURE & CONCURRENCY AUDIT", C_CYAN)
    aud_rows = []
    for tier_label, res in tier_results.items():
        sim = res["sim"]
        aud_rows.append([
            tier_label,
            sim["max_concurrent"],
            f"{sim['peak_leverage']:.2f}x",
            f"{sim['avg_utilization_pct']:.1f}%",
            f"{100 - sim['avg_utilization_pct']:.1f}%",
            len(sim["trades"])
        ])
    print(tabulate(aud_rows,
        headers=["Tier","Peak Concurrent","Peak Leverage","Avg Utilization","Avg Idle Cash","Total Trades"],
        tablefmt="grid"))

    # SECTION 8: Best tier recommendation
    section(8, "RECOMMENDED SIZING TIER", C_GRN)
    # Score by full-period Calmar (return/DD balance)
    best_tier = max(tier_results.keys(),
                    key=lambda k: tier_results[k]["metrics_full"]["calmar"])
    best_m = tier_results[best_tier]["metrics_full"]
    best_oos = tier_results[best_tier]["metrics_oos"]
    best_risk = tier_results[best_tier]["risk_pct"]
    print(f"\n  {C_GRN}{S_BR}RECOMMENDED: {best_tier} (risk={best_risk*100:.1f}% per trade){S_RS}")
    print(f"  Selection criterion: Highest Calmar ratio (best risk-adjusted growth)")
    print(f"\n  Full Period Metrics:")
    print(f"    Final Equity : ${best_m['final_equity']:,.2f}")
    print(f"    Total Return : {best_m['total_return_pct']:+.2f}%")
    print(f"    CAGR         : {best_m['cagr_pct']:.2f}%")
    print(f"    Sharpe       : {best_m['sharpe']:.3f}")
    print(f"    Sortino      : {best_m['sortino']:.3f}")
    print(f"    Max Drawdown : {best_m['max_dd_pct']:.2f}%")
    print(f"    Calmar       : {best_m['calmar']:.3f}")
    print(f"    Profit Factor: {best_m['profit_factor']:.3f}")
    print(f"    Win Rate     : {best_m['win_rate']:.2f}%")
    print(f"    Total Trades : {best_m['total_trades']:,}")
    if best_oos:
        print(f"\n  2026 OOS Metrics:")
        print(f"    OOS Return   : {best_oos['total_return_pct']:+.2f}%")
        print(f"    OOS Sharpe   : {best_oos['sharpe']:.3f}")
        print(f"    OOS Max DD   : {best_oos['max_dd_pct']:.2f}%")
        print(f"    OOS Trades   : {best_oos['total_trades']:,}")
        print(f"    OOS Win Rate : {best_oos['win_rate']:.2f}%")

    # SECTION 9: Ready-to-trade blueprint
    section(9, "READY-TO-TRADE BLUEPRINT ($10k / $50k / $100k accounts)", C_MAG)
    print(f"\n  Formula: Position Size = (Account_Equity * Risk_Pct) / abs(Entry_Price - Stop_Loss)")
    print(f"  Constraint: Total open position notional <= {MAX_GROSS_LEVERAGE}x Account Equity")
    print(f"  Max 1 concurrent position per symbol\n")

    for tier_label in ["Standard_1.0%", "Aggressive_2.0%"]:
        if tier_label not in tier_results: continue
        risk = tier_results[tier_label]["risk_pct"]
        print(f"  {C_CYAN}{S_BR}--- {tier_label} (risk = {risk*100:.1f}% per trade) ---{S_RS}")
        bp_rows = []
        for acct in [10000, 50000, 100000]:
            risk_per_trade_dollar = acct * risk
            max_gross_notional = acct * MAX_GROSS_LEVERAGE
            # Example: with 1x ATR SL of 2% of price on a $30k BTC entry
            # size = risk_dollar / (0.02 * 30000) = risk_dollar / 600
            bp_rows.append([
                f"${acct:,}",
                f"${risk_per_trade_dollar:.2f}",
                f"${max_gross_notional:,.0f}",
                f"~{risk_per_trade_dollar / (30000 * 0.02):.4f} BTC (@ $30k, 2% SL)",
                f"~${risk_per_trade_dollar / 0.02:,.0f}",
            ])
        print(tabulate(bp_rows,
            headers=["Account","Risk/Trade $","Max Gross Notional","Ex. BTC Size","Ex. Total Notional (2% SL)"],
            tablefmt="simple"))
        print()

    # SECTION 10: Exports
    section(10, "EXPORTS", C_CYAN)
    # Master trade log (all tiers)
    log_rows = []
    for tier_label, res in tier_results.items():
        for t in res["sim"]["trades"]:
            log_rows.append(t)
    trade_log_csv = os.path.join(RESULTS_ROOT, "task11.2_true_portfolio_trade_log.csv")
    pd.DataFrame(log_rows).to_csv(trade_log_csv, index=False)
    print(f"  {C_GRN}[SAVED] {trade_log_csv} ({len(log_rows):,} rows){S_RS}")

    # Monthly returns comparison
    mo_comp_rows = []
    for tier_label, res in tier_results.items():
        eq = res["sim"]["equity_curve"]
        try:
            monthly = eq.resample("ME").last().pct_change().dropna() * 100
            for idx, val in monthly.items():
                mo_comp_rows.append({
                    "tier": tier_label, "year_month": idx.strftime("%Y-%m"),
                    "return_pct": round(float(val), 3),
                    "period": "IS" if idx <= IS_END else "OOS"
                })
        except Exception:
            pass
    mo_csv = os.path.join(RESULTS_ROOT, "task11.2_monthly_returns_comparison.csv")
    pd.DataFrame(mo_comp_rows).to_csv(mo_csv, index=False)
    print(f"  {C_GRN}[SAVED] {mo_csv}{S_RS}")

    # Portfolio summary
    summary_rows = []
    for tier_label, res in tier_results.items():
        row = {"tier": tier_label, "risk_pct": res["risk_pct"]}
        for k, v in res["metrics_full"].items():
            row[f"full_{k}"] = v
        if res["metrics_oos"]:
            for k, v in res["metrics_oos"].items():
                row[f"oos_{k}"] = v
        summary_rows.append(row)
    sum_csv = os.path.join(RESULTS_ROOT, "task11.2_portfolio_summary.csv")
    pd.DataFrame(summary_rows).to_csv(sum_csv, index=False)
    print(f"  {C_GRN}[SAVED] {sum_csv}{S_RS}")

    elapsed = time.time() - t0
    print()
    box(f"TASK 11.2 COMPLETE - Runtime: {elapsed/60:.2f} minutes", C_GRN)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C_RED}Interrupted{S_RS}")
    except Exception as e:
        print(f"\n{C_RED}FATAL: {e}{S_RS}")
        traceback.print_exc()
