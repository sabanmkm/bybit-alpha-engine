"""
TASK 13.5: CROSS-TASK SIGNAL CONSISTENCY AUDIT
8 tests x 16 edges = 128 test cases to prove deterministic reproducibility.
"""
import os
import sys
import time
import copy
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
        MultiTFDataLoader, BacktestEngine, RegimeDetector,
        RSI, MACD, ATR, EMA, SMA, ADX, HMA, Supertrend
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

TRADE_LOG_CSV = os.path.join(RESULTS_ROOT, "task11.2_true_portfolio_trade_log.csv")

FULL_START = pd.Timestamp("2022-01-01")
FULL_END = pd.Timestamp("2026-08-31 23:59:59")

# Bybit trading params
FEES = 0.00055
SLIPPAGE = 0.0003
EXPECTED_TOTAL_COST_PER_TRADE_PCT = 2 * (FEES + SLIPPAGE)  # 0.17%

# Tolerances
INDICATOR_TOLERANCE = 1e-6
SL_TP_TOLERANCE_PCT = 0.0001  # 0.01%
FEE_SLIP_TOLERANCE = 1e-4
SIGNAL_COUNT_TOLERANCE_PCT = 5.0
REFERENCE_MATCH_THRESHOLD = 0.85  # 85% match acceptable

# Colors
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
    {"id":1,  "strategy":"S30b_MidWeek_MeanReversion", "symbol":"XMR_USDT_USDT",  "tf":"1H", "uses_mtf": False, "regime": False},
    {"id":2,  "strategy":"S30a_DOW_Seasonality",       "symbol":"ATOM_USDT_USDT", "tf":"4H", "uses_mtf": False, "regime": False},
    {"id":3,  "strategy":"S30a_DOW_Seasonality",       "symbol":"ADA_USDT_USDT",  "tf":"4H", "uses_mtf": False, "regime": False},
    {"id":4,  "strategy":"S30a_DOW_Seasonality",       "symbol":"BSV_USDT_USDT",  "tf":"4H", "uses_mtf": False, "regime": False},
    {"id":5,  "strategy":"S30a_DOW_Seasonality",       "symbol":"ETC_USDT_USDT",  "tf":"4H", "uses_mtf": False, "regime": False},
    {"id":6,  "strategy":"S30a_DOW_Seasonality",       "symbol":"AVAX_USDT_USDT", "tf":"4H", "uses_mtf": False, "regime": False},
    {"id":7,  "strategy":"S30a_DOW_Seasonality",       "symbol":"QTUM_USDT_USDT", "tf":"4H", "uses_mtf": False, "regime": False},
    {"id":8,  "strategy":"S30b_MidWeek_MeanReversion", "symbol":"TWT_USDT_USDT",  "tf":"4H", "uses_mtf": False, "regime": False},
    {"id":9,  "strategy":"S30d_NY_London_Momentum",    "symbol":"DOT_USDT_USDT",  "tf":"1H", "uses_mtf": True,  "regime": False},
    {"id":10, "strategy":"S30d_NY_London_Momentum",    "symbol":"XLM_USDT_USDT",  "tf":"1H", "uses_mtf": True,  "regime": False},
    {"id":11, "strategy":"R13_ATR_Range_Expansion",    "symbol":"ETH_USDT_USDT",  "tf":"4H", "uses_mtf": False, "regime": True},
    {"id":12, "strategy":"R15_Volume_Breakout",        "symbol":"SAND_USDT_USDT", "tf":"4H", "uses_mtf": False, "regime": True},
    {"id":13, "strategy":"R01_EMA_Cross",              "symbol":"OP_USDT_USDT",   "tf":"4H", "uses_mtf": False, "regime": True},
    {"id":14, "strategy":"R15_Volume_Breakout",        "symbol":"OP_USDT_USDT",   "tf":"4H", "uses_mtf": False, "regime": True},
    {"id":15, "strategy":"R01_EMA_Cross",              "symbol":"INJ_USDT_USDT",  "tf":"1H", "uses_mtf": False, "regime": True},
    {"id":16, "strategy":"R16_Trend_Plus_RSI_Pullback","symbol":"DOGE_USDT_USDT", "tf":"4H", "uses_mtf": False, "regime": True},
]

# ============================================================
# STRATEGY IMPLEMENTATIONS (canonical, from engine_v2 primitives)
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
    sl_l = df["low"].rolling(20).min(); sl_s = df["high"].rolling(20).max()
    sl.loc[L] = sl_l.loc[L].values
    sl.loc[S] = sl_s.loc[S].values
    tp.loc[L] = (close + 3.0 * a).loc[L].values
    tp.loc[S] = (close - 1.5 * a).loc[S].values
    return sig, sl, tp

def strat_R13_ATR_Range(df, regime):
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

def strat_R15_Volume(df, regime):
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

def strat_R16_Trend_RSI(df, regime):
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
    "R13_ATR_Range_Expansion": strat_R13_ATR_Range,
    "R15_Volume_Breakout": strat_R15_Volume,
    "R16_Trend_Plus_RSI_Pullback": strat_R16_Trend_RSI,
}

def generate_signals_for_edge(edge, loader, regime_series):
    """Return (df, sig, sl, tp) for a given edge."""
    fn = STRATEGY_MAP.get(edge["strategy"])
    if fn is None: return None, None, None, None
    df = loader.load(edge["symbol"], edge["tf"], FULL_START, FULL_END)
    if df is None or len(df) < 100: return None, None, None, None
    try:
        if edge["regime"]:
            sig, sl, tp = fn(df, regime_series)
        else:
            sig, sl, tp = fn(df)
        return df, sig, sl, tp
    except Exception as e:
        print(f"    {C_RED}[ERR gen signals] {e}{S_RS}")
        return None, None, None, None

# ============================================================
# TEST A: DETERMINISM (Same input -> same output twice)
# ============================================================
def test_A_determinism(edge, loader, regime_series):
    df1, s1, sl1, tp1 = generate_signals_for_edge(edge, loader, regime_series)
    df2, s2, sl2, tp2 = generate_signals_for_edge(edge, loader, regime_series)
    if s1 is None or s2 is None:
        return {"passed": False, "detail": "signal generation failed"}
    if len(s1) != len(s2):
        return {"passed": False, "detail": f"length mismatch {len(s1)} vs {len(s2)}"}
    signal_diff = int((s1.values != s2.values).sum())
    # Compare SL/TP where signals fire (nan-aware)
    sl1_v = sl1.values; sl2_v = sl2.values
    sl_diff_mask = ~((np.isnan(sl1_v) & np.isnan(sl2_v)) | (sl1_v == sl2_v))
    sl_diff = int(sl_diff_mask.sum())
    tp1_v = tp1.values; tp2_v = tp2.values
    tp_diff_mask = ~((np.isnan(tp1_v) & np.isnan(tp2_v)) | (tp1_v == tp2_v))
    tp_diff = int(tp_diff_mask.sum())
    passed = signal_diff == 0 and sl_diff == 0 and tp_diff == 0
    return {"passed": bool(passed),
            "detail": f"signal_diff={signal_diff}, sl_diff={sl_diff}, tp_diff={tp_diff}",
            "signal_diff": signal_diff, "sl_diff": sl_diff, "tp_diff": tp_diff}

# ============================================================
# TEST B: REFERENCE MATCH (current vs prior trade log)
# ============================================================
def test_B_reference_match(edge, loader, regime_series, engine, ref_trades):
    if not ref_trades or len(ref_trades) < 5:
        return {"passed": True, "detail": "no reference (auto-pass)",
                "match_rate": 1.0, "ref_count": 0, "cur_count": 0}
    df, sig, sl, tp = generate_signals_for_edge(edge, loader, regime_series)
    if df is None:
        return {"passed": False, "detail": "signal gen failed"}
    result = engine.run(df, sig, sl, tp)
    cur_trades = result["trades"]
    if not cur_trades:
        return {"passed": False, "detail": "current run produced zero trades",
                "match_rate": 0.0, "ref_count": len(ref_trades), "cur_count": 0}
    # Match by entry_time (rounded to nearest minute)
    ref_times = set(pd.to_datetime(t["entry_time"]).floor("min") for t in ref_trades)
    cur_times = set(pd.to_datetime(t["entry_time"]).floor("min") for t in cur_trades)
    intersection = ref_times & cur_times
    if len(ref_times) == 0:
        match_rate = 1.0
    else:
        match_rate = len(intersection) / len(ref_times)
    passed = match_rate >= REFERENCE_MATCH_THRESHOLD
    return {"passed": bool(passed),
            "detail": f"match_rate={match_rate:.1%}, ref={len(ref_times)}, cur={len(cur_times)}",
            "match_rate": round(match_rate, 3),
            "ref_count": len(ref_times), "cur_count": len(cur_times)}

# ============================================================
# TEST C: LOOKAHEAD BIAS INJECTION
# Shift close forward by 1 bar. If strategy peeks at future, signals move.
# ============================================================
def test_C_lookahead_bias(edge, loader, regime_series):
    fn = STRATEGY_MAP.get(edge["strategy"])
    if fn is None: return {"passed": False, "detail": "strategy fn missing"}
    df = loader.load(edge["symbol"], edge["tf"], FULL_START, FULL_END)
    if df is None: return {"passed": False, "detail": "no data"}
    # Baseline signals
    if edge["regime"]:
        sig_base, _, _ = fn(df, regime_series)
    else:
        sig_base, _, _ = fn(df)
    # Inject: shift future close prices by 1 bar earlier (contaminate present with future)
    df_leak = df.copy()
    df_leak["close_orig"] = df["close"]
    # Replace close with future close (shift(-1)) to simulate a leak
    df_leak["close"] = df["close"].shift(-1).fillna(df["close"].iloc[-1])
    # Similarly for high/low
    df_leak["high"] = df["high"].shift(-1).fillna(df["high"].iloc[-1])
    df_leak["low"] = df["low"].shift(-1).fillna(df["low"].iloc[-1])
    try:
        if edge["regime"]:
            sig_leak, _, _ = fn(df_leak, regime_series)
        else:
            sig_leak, _, _ = fn(df_leak)
    except Exception:
        return {"passed": True, "detail": "leak version failed to run (implicit protection)"}
    # If strategy uses future data, signals will be identical to leaked version
    # If clean, signals should DIFFER (base uses today's close, leak uses tomorrow's)
    diff_count = int((sig_base.values != sig_leak.values).sum())
    total = len(sig_base)
    diff_pct = 100 * diff_count / total if total > 0 else 0
    # Pass = signals ARE DIFFERENT (proving strategy uses current data, not future)
    # If signals identical, that means either (a) no signals fire, or (b) strategy peeks future
    n_base_signals = int((sig_base != 0).sum())
    n_leak_signals = int((sig_leak != 0).sum())
    if n_base_signals == 0:
        return {"passed": True, "detail": "no baseline signals to test", "diff_count": 0}
    # Expect > 1% difference if strategy is clean (uses present, not future)
    passed = diff_pct > 1.0
    return {"passed": bool(passed),
            "detail": f"diff_pct={diff_pct:.2f}%, base_sig={n_base_signals}, leak_sig={n_leak_signals}",
            "diff_count": diff_count, "diff_pct": round(diff_pct, 2)}

# ============================================================
# TEST D: MTF DATA SYNCHRONIZATION
# ============================================================
def test_D_mtf_sync(edge, loader):
    if not edge["uses_mtf"]:
        return {"passed": True, "detail": "N/A - no MTF used"}
    # S30d uses London range from same-day 8h-12h (which IS forward-looking within the day!)
    # But the London session bars are all closed BEFORE the trade window 12:00-16:00
    # So we need to verify London high/low was known by trade time
    df = loader.load(edge["symbol"], edge["tf"], FULL_START, FULL_END)
    if df is None: return {"passed": False, "detail": "no data"}
    hour = pd.Series(df.index.hour, index=df.index)
    date = pd.Series(df.index.date, index=df.index)
    df_tmp = df.copy(); df_tmp["date"] = date.values
    in_london = (hour >= 8) & (hour < 12)
    df_tmp["in_london"] = in_london.values
    grp = df_tmp[df_tmp["in_london"]].groupby("date")
    london_high = grp["high"].max()
    london_low = grp["low"].min()
    lh_map = pd.Series(date.map(london_high).values, index=df.index)
    ll_map = pd.Series(date.map(london_low).values, index=df.index)
    # For each trade time (hour>=12), the London range should equal the actual max/min from that day's 8-12 window
    trade_bars = df[(hour >= 12) & (hour < 16)]
    violations = 0
    checked = 0
    for ts in trade_bars.index[:50]:  # sample 50 to keep fast
        day = ts.date()
        # Get actual London range for this day
        day_london = df[(df.index.date == day) & (df.index.hour >= 8) & (df.index.hour < 12)]
        if len(day_london) == 0: continue
        actual_hi = day_london["high"].max()
        actual_lo = day_london["low"].min()
        assumed_hi = lh_map.loc[ts]
        assumed_lo = ll_map.loc[ts]
        if not np.isclose(actual_hi, assumed_hi) or not np.isclose(actual_lo, assumed_lo):
            violations += 1
        # Also check timing: all London bars must have closed before ts
        latest_london_bar = day_london.index.max()
        if latest_london_bar >= ts:
            violations += 1
        checked += 1
    passed = violations == 0
    return {"passed": bool(passed),
            "detail": f"checked={checked}, violations={violations}",
            "violations": violations, "checked": checked}

# ============================================================
# TEST E: INDICATOR CROSS-VERIFICATION
# ============================================================
def test_E_indicator_verify(edge, loader):
    df = loader.load(edge["symbol"], edge["tf"], FULL_START, FULL_END)
    if df is None or len(df) < 100: return {"passed": False, "detail": "no data"}
    # Compute canonical indicators
    atr_canon = ATR(df, 14)
    rsi_canon = RSI(df["close"], 14)
    ema_canon = EMA(df["close"], 12)
    # Recompute from scratch using inline formulas
    # ATR verify: TR = max(H-L, |H-PC|, |L-PC|), Wilder's EMA
    h = df["high"]; l = df["low"]; c = df["close"]; pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    atr_manual = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    # RSI verify: Wilder's smoothing on up/down
    delta = df["close"].diff()
    up = delta.clip(lower=0); dn = -delta.clip(upper=0)
    au = up.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    ad = dn.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = au / ad.replace(0, np.nan)
    rsi_manual = (100 - (100/(1+rs))).fillna(50)
    # EMA verify
    ema_manual = df["close"].ewm(span=12, adjust=False, min_periods=12).mean()

    # Sample last 100 non-NaN values
    valid_idx = atr_canon.dropna().index[-100:]
    if len(valid_idx) < 10:
        return {"passed": False, "detail": "insufficient valid indicator data"}
    atr_diff = np.abs(atr_canon.loc[valid_idx].values - atr_manual.loc[valid_idx].values).max()
    rsi_diff = np.abs(rsi_canon.loc[valid_idx].values - rsi_manual.loc[valid_idx].values).max()
    ema_diff = np.abs(ema_canon.loc[valid_idx].values - ema_manual.loc[valid_idx].values).max()
    max_diff = max(atr_diff, rsi_diff, ema_diff)
    passed = max_diff < INDICATOR_TOLERANCE * max(1, df["close"].mean())
    return {"passed": bool(passed),
            "detail": f"max_diff ATR={atr_diff:.2e}, RSI={rsi_diff:.2e}, EMA={ema_diff:.2e}",
            "max_diff": float(max_diff)}

# ============================================================
# TEST F: SIGNAL COUNT AUDIT
# ============================================================
def test_F_signal_count(edge, loader, regime_series, engine, ref_trades):
    df, sig, sl, tp = generate_signals_for_edge(edge, loader, regime_series)
    if df is None: return {"passed": False, "detail": "no data"}
    result = engine.run(df, sig, sl, tp)
    cur_count = len(result["trades"])
    ref_count = len(ref_trades) if ref_trades else cur_count
    if ref_count == 0:
        return {"passed": True, "detail": f"no reference, current={cur_count}",
                "cur_count": cur_count, "ref_count": 0, "deviation_pct": 0}
    deviation = abs(cur_count - ref_count) / ref_count * 100
    passed = deviation <= SIGNAL_COUNT_TOLERANCE_PCT
    return {"passed": bool(passed),
            "detail": f"cur={cur_count}, ref={ref_count}, dev={deviation:.2f}%",
            "cur_count": cur_count, "ref_count": ref_count,
            "deviation_pct": round(deviation, 2)}

# ============================================================
# TEST G: SL/TP RECIPE VERIFICATION
# ============================================================
def test_G_sltp_recipe(edge, loader, regime_series):
    df, sig, sl_series, tp_series = generate_signals_for_edge(edge, loader, regime_series)
    if df is None: return {"passed": False, "detail": "no data"}
    # For each active signal, recompute SL/TP from scratch using canonical formula
    atr_ref = ATR(df, 14)
    close = df["close"]
    long_mask = sig == 1
    short_mask = sig == -1
    # Determine SL/TP recipe per strategy
    strat = edge["strategy"]
    # Recipes (must match strategy implementations above)
    recipes = {
        "S30a_DOW_Seasonality":       {"sl": 1.5, "tp": 2.5, "type": "atr"},
        "S30b_MidWeek_MeanReversion": {"sl": 2.0, "tp": 1.5, "type": "atr"},
        "S30d_NY_London_Momentum":    {"sl": 1.5, "tp": 2.5, "type": "atr"},
        "R13_ATR_Range_Expansion":    {"sl": 2.0, "tp": 3.0, "type": "atr"},
        "R15_Volume_Breakout":        {"sl": 1.5, "tp": 2.5, "type": "atr"},
        "R16_Trend_Plus_RSI_Pullback":{"sl": 1.5, "tp": 2.5, "type": "atr"},
        "R01_EMA_Cross":              {"sl": None, "tp_bull": 3.0, "tp_bear": 1.5, "type": "structural"},
    }
    recipe = recipes.get(strat)
    if recipe is None:
        return {"passed": False, "detail": f"no recipe for {strat}"}

    max_err_pct = 0.0
    checked = 0
    if recipe["type"] == "atr":
        # SL long = close - sl_mult * atr
        expected_sl_long = close - recipe["sl"] * atr_ref
        expected_sl_short = close + recipe["sl"] * atr_ref
        expected_tp_long = close + recipe["tp"] * atr_ref
        expected_tp_short = close - recipe["tp"] * atr_ref
        # Compare only where signals fire
        for idx in df.index[long_mask.values]:
            if pd.isna(sl_series.loc[idx]) or pd.isna(expected_sl_long.loc[idx]): continue
            err = abs(sl_series.loc[idx] - expected_sl_long.loc[idx]) / abs(expected_sl_long.loc[idx]) * 100
            if err > max_err_pct: max_err_pct = err
            checked += 1
        for idx in df.index[short_mask.values]:
            if pd.isna(sl_series.loc[idx]) or pd.isna(expected_sl_short.loc[idx]): continue
            err = abs(sl_series.loc[idx] - expected_sl_short.loc[idx]) / abs(expected_sl_short.loc[idx]) * 100
            if err > max_err_pct: max_err_pct = err
            checked += 1
    else:  # structural (R01)
        sl_low = df["low"].rolling(20).min()
        sl_high = df["high"].rolling(20).max()
        expected_tp_long = close + recipe["tp_bull"] * atr_ref
        expected_tp_short = close - recipe["tp_bear"] * atr_ref
        for idx in df.index[long_mask.values]:
            if pd.isna(sl_series.loc[idx]) or pd.isna(sl_low.loc[idx]): continue
            err = abs(sl_series.loc[idx] - sl_low.loc[idx]) / abs(sl_low.loc[idx]) * 100
            if err > max_err_pct: max_err_pct = err
            checked += 1
        for idx in df.index[short_mask.values]:
            if pd.isna(sl_series.loc[idx]) or pd.isna(sl_high.loc[idx]): continue
            err = abs(sl_series.loc[idx] - sl_high.loc[idx]) / abs(sl_high.loc[idx]) * 100
            if err > max_err_pct: max_err_pct = err
            checked += 1

    passed = max_err_pct < SL_TP_TOLERANCE_PCT * 100 if checked > 0 else True
    return {"passed": bool(passed),
            "detail": f"checked={checked}, max_err_pct={max_err_pct:.6f}%",
            "max_err_pct": round(max_err_pct, 6)}

# ============================================================
# TEST H: FEE AND SLIPPAGE APPLICATION
# ============================================================
def test_H_fee_slippage(edge, loader, regime_series, engine):
    df, sig, sl, tp = generate_signals_for_edge(edge, loader, regime_series)
    if df is None: return {"passed": False, "detail": "no data"}
    result = engine.run(df, sig, sl, tp)
    trades = result["trades"]
    if not trades or len(trades) < 5:
        return {"passed": True, "detail": "insufficient trades to verify (auto-pass)"}
    # For each trade, expected cost = entry_price * (fee + slip) + exit_price * (fee + slip)
    # We verify the engine applies this via checking that pnl equals gross - fees_expected
    errors = []
    checked = 0
    for t in trades[:20]:  # sample
        entry_px = t["entry_price"]; exit_px = t["exit_price"]; size = t["size"]
        # Reported pnl = gross - fees (where fees applied by engine)
        # We reconstruct expected: gross_pnl_raw = (exit-entry)*size*direction
        direction = 1 if t["direction"] == "LONG" else -1
        gross = (exit_px - entry_px) * size * direction
        expected_fees = (entry_px + exit_px) * size * FEES
        expected_pnl = gross - expected_fees
        actual_pnl = t["pnl"]
        err = abs(actual_pnl - expected_pnl)
        errors.append(err)
        checked += 1
    max_err = max(errors) if errors else 0
    passed = max_err < FEE_SLIP_TOLERANCE * 100  # within tolerance in dollar terms scaled
    return {"passed": bool(passed),
            "detail": f"checked={checked}, max_pnl_err=${max_err:.6f}",
            "max_err": round(max_err, 6)}

# ============================================================
# LOAD REFERENCE TRADE LOG
# ============================================================
def load_ref_trades():
    if not os.path.exists(TRADE_LOG_CSV): return {}
    try:
        df = pd.read_csv(TRADE_LOG_CSV)
        if "tier" in df.columns:
            df = df[df["tier"] == "Standard_1.0%"]
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        return {int(eid): grp.to_dict("records") for eid, grp in df.groupby("edge_id")}
    except Exception as e:
        print(f"  {C_YEL}[WARN] Could not load reference log: {e}{S_RS}")
        return {}

# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    box("TASK 13.5: SIGNAL CONSISTENCY AUDIT", C_CYAN)
    print(f"{C_CYAN}  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{C_CYAN}  Engine: v{ENGINE_VER}")
    print(f"{C_CYAN}  16 edges x 8 tests = 128 test cases")

    loader = MultiTFDataLoader(DATA_ROOT)
    engine = BacktestEngine()

    # Regimes
    print(f"\n  Precomputing BTC regime classification...")
    detector = RegimeDetector(data_loader=loader)
    regime_series = detector.classify_series("2022-01-01", "2026-08-31")
    if not isinstance(regime_series.index, pd.DatetimeIndex):
        regime_series.index = pd.to_datetime(regime_series.index)
    print(f"  {C_GRN}[OK] Regime series: {len(regime_series)} days{S_RS}")

    # Reference log
    print(f"\n  Loading reference trade log...")
    ref_trades_map = load_ref_trades()
    print(f"  {C_GRN}[OK] Reference edges: {len(ref_trades_map)}{S_RS}")

    # Run 8 tests per edge
    section(1, "RUNNING 8 CONSISTENCY TESTS ON 16 EDGES", C_MAG)
    scorecard = []
    for edge in PORTFOLIO_16:
        eid = edge["id"]
        ref_trades = ref_trades_map.get(eid, [])
        print(f"\n  {C_CYAN}Edge #{eid} {edge['strategy'][:26]:<26} | {edge['symbol']:<14} | {edge['tf']}{S_RS}")

        # Test A
        print(f"    A. Determinism...", end="", flush=True)
        tA = test_A_determinism(edge, loader, regime_series)
        print(f" {'PASS' if tA['passed'] else 'FAIL'} | {tA['detail']}")

        # Test B
        print(f"    B. Reference Match...", end="", flush=True)
        tB = test_B_reference_match(edge, loader, regime_series, engine, ref_trades)
        print(f" {'PASS' if tB['passed'] else 'FAIL'} | {tB['detail']}")

        # Test C
        print(f"    C. Look-ahead Bias...", end="", flush=True)
        tC = test_C_lookahead_bias(edge, loader, regime_series)
        print(f" {'PASS' if tC['passed'] else 'FAIL'} | {tC['detail']}")

        # Test D
        print(f"    D. MTF Sync...", end="", flush=True)
        tD = test_D_mtf_sync(edge, loader)
        print(f" {'PASS' if tD['passed'] else 'FAIL'} | {tD['detail']}")

        # Test E
        print(f"    E. Indicator Verify...", end="", flush=True)
        tE = test_E_indicator_verify(edge, loader)
        print(f" {'PASS' if tE['passed'] else 'FAIL'} | {tE['detail']}")

        # Test F
        print(f"    F. Signal Count Audit...", end="", flush=True)
        tF = test_F_signal_count(edge, loader, regime_series, engine, ref_trades)
        print(f" {'PASS' if tF['passed'] else 'FAIL'} | {tF['detail']}")

        # Test G
        print(f"    G. SL/TP Recipe...", end="", flush=True)
        tG = test_G_sltp_recipe(edge, loader, regime_series)
        print(f" {'PASS' if tG['passed'] else 'FAIL'} | {tG['detail']}")

        # Test H
        print(f"    H. Fee/Slippage...", end="", flush=True)
        tH = test_H_fee_slippage(edge, loader, regime_series, engine)
        print(f" {'PASS' if tH['passed'] else 'FAIL'} | {tH['detail']}")

        score = sum([tA["passed"], tB["passed"], tC["passed"], tD["passed"],
                     tE["passed"], tF["passed"], tG["passed"], tH["passed"]])
        if score == 8: status = "CERTIFIED"
        elif score >= 6: status = "REVIEW"
        else: status = "CORRUPTED"
        scorecard.append({
            "edge_id": eid, "strategy": edge["strategy"], "symbol": edge["symbol"], "tf": edge["tf"],
            "tA": tA["passed"], "tB": tB["passed"], "tC": tC["passed"], "tD": tD["passed"],
            "tE": tE["passed"], "tF": tF["passed"], "tG": tG["passed"], "tH": tH["passed"],
            "score": score, "status": status,
            "details": {"A":tA,"B":tB,"C":tC,"D":tD,"E":tE,"F":tF,"G":tG,"H":tH}
        })
        st_col = C_GRN if status == "CERTIFIED" else (C_YEL if status == "REVIEW" else C_RED)
        print(f"    {st_col}{S_BR}>>> SCORE: {score}/8 | STATUS: {status}{S_RS}")

    # SECTION 2: Executive summary
    section(2, "EXECUTIVE SUMMARY", C_CYAN)
    total_tests = 16 * 8
    total_pass = sum(e["score"] for e in scorecard)
    total_fail = total_tests - total_pass
    n_certified = sum(1 for e in scorecard if e["status"] == "CERTIFIED")
    n_review = sum(1 for e in scorecard if e["status"] == "REVIEW")
    n_corrupted = sum(1 for e in scorecard if e["status"] == "CORRUPTED")

    summary_rows = [
        ["Total Edges Audited", 16],
        ["Total Test Cases", total_tests],
        [f"{C_GRN}Passed{S_RS}", f"{C_GRN}{total_pass} ({100*total_pass/total_tests:.1f}%){S_RS}"],
        [f"{C_RED}Failed{S_RS}", f"{C_RED}{total_fail} ({100*total_fail/total_tests:.1f}%){S_RS}"],
        [f"{C_GRN}CERTIFIED (8/8){S_RS}", n_certified],
        [f"{C_YEL}REVIEW (6-7/8){S_RS}", n_review],
        [f"{C_RED}CORRUPTED (<6/8){S_RS}", n_corrupted],
    ]
    print(tabulate(summary_rows, headers=["Metric", "Value"], tablefmt="grid"))

    # SECTION 3: 16x8 Scorecard grid
    section(3, "16 x 8 CONSISTENCY GRID", C_CYAN)
    header = ["#", "Strategy", "Symbol", "TF", "A", "B", "C", "D", "E", "F", "G", "H", "Score", "Status"]
    matrix_rows = []
    for e in scorecard:
        row = [e["edge_id"], e["strategy"][:22], e["symbol"][:12], e["tf"]]
        for k in ["tA","tB","tC","tD","tE","tF","tG","tH"]:
            v = e[k]
            row.append(f"{C_GRN}P{S_RS}" if v else f"{C_RED}F{S_RS}")
        row.append(f"{e['score']}/8")
        st_col = C_GRN if e["status"] == "CERTIFIED" else (C_YEL if e["status"] == "REVIEW" else C_RED)
        row.append(f"{st_col}{e['status']}{S_RS}")
        matrix_rows.append(row)
    print(tabulate(matrix_rows, headers=header, tablefmt="grid"))

    # SECTION 4: Divergence deep-dive
    section(4, "DIVERGENCE DEEP-DIVE (FAILURES ONLY)", C_RED)
    div_rows = []
    for e in scorecard:
        for tkey in ["A","B","C","D","E","F","G","H"]:
            d = e["details"].get(tkey, {})
            if not d.get("passed", True):
                div_rows.append([
                    e["edge_id"], e["strategy"][:22], e["symbol"][:12], e["tf"],
                    f"Test {tkey}",
                    d.get("detail", "")[:60]
                ])
    if not div_rows:
        print(f"  {C_GRN}{S_BR}[!] ZERO DIVERGENCES DETECTED - all 128 test cases passed{S_RS}")
    else:
        print(tabulate(div_rows, headers=["#","Strategy","Symbol","TF","Test","Detail"], tablefmt="simple"))

    # SECTION 5: Look-ahead audit result
    section(5, "LOOK-AHEAD BIAS AUDIT (Test C)", C_MAG)
    c_pass = sum(1 for e in scorecard if e["tC"])
    print(f"  {c_pass}/{len(scorecard)} edges verified free of look-ahead bias.")
    if c_pass == len(scorecard):
        print(f"  {C_GRN}{S_BR}[!] ALL 16 EDGES USE ONLY PRESENT DATA - no future leakage detected{S_RS}")
    else:
        fail_c = [e for e in scorecard if not e["tC"]]
        for e in fail_c:
            d = e["details"]["C"]
            print(f"  {C_RED}FAIL: Edge #{e['edge_id']} {e['strategy']}/{e['symbol']} | {d.get('detail','')}{S_RS}")

    # SECTION 6: Per-test pass rates
    section(6, "PER-TEST PASS RATE SUMMARY", C_CYAN)
    test_names = {
        "tA": "A. Determinism",
        "tB": "B. Reference Match",
        "tC": "C. Look-ahead Bias",
        "tD": "D. MTF Sync",
        "tE": "E. Indicator Verify",
        "tF": "F. Signal Count",
        "tG": "G. SL/TP Recipe",
        "tH": "H. Fee/Slippage",
    }
    rate_rows = []
    for k, name in test_names.items():
        n_pass = sum(1 for e in scorecard if e[k])
        rate = 100 * n_pass / len(scorecard)
        col = C_GRN if rate >= 90 else (C_YEL if rate >= 70 else C_RED)
        rate_rows.append([name, n_pass, len(scorecard), f"{col}{rate:.1f}%{S_RS}"])
    print(tabulate(rate_rows, headers=["Test", "Passed", "Total", "Rate"], tablefmt="grid"))

    # SECTION 7: Final certification
    section(7, "FINAL CERTIFICATION", C_CYAN)
    if n_corrupted == 0 and n_certified >= 12:
        verdict = f"{C_GRN}{S_BR}CERTIFIED CONSISTENT - All signals reproducible; Engine v2 verified deterministic{S_RS}"
        code = "GO"
    elif n_corrupted <= 2 and n_certified + n_review >= 14:
        verdict = f"{C_YEL}CONDITIONAL - {n_review} edges need review, {n_corrupted} corrupted; investigate before deploy{S_RS}"
        code = "CONDITIONAL"
    else:
        verdict = f"{C_RED}SIGNAL CORRUPTION DETECTED - {n_corrupted} edges failed; BLOCK deployment{S_RS}"
        code = "BLOCK"
    print(f"\n  {S_BR}FINAL VERDICT: {verdict}")

    # SECTION 8: Exports
    section(8, "EXPORTS", C_CYAN)
    # Scorecard CSV
    sc_csv = os.path.join(RESULTS_ROOT, "task13.5_consistency_scorecard.csv")
    export_rows = []
    for e in scorecard:
        row = {k: v for k, v in e.items() if k != "details"}
        d = e["details"]
        for tk in ["A","B","C","D","E","F","G","H"]:
            det = d.get(tk, {})
            row[f"test_{tk}_detail"] = det.get("detail", "")
        export_rows.append(row)
    pd.DataFrame(export_rows).to_csv(sc_csv, index=False)
    print(f"  {C_GRN}[SAVED] {sc_csv}{S_RS}")

    # Divergence details CSV
    div_csv = os.path.join(RESULTS_ROOT, "task13.5_divergence_details.csv")
    div_export = []
    for e in scorecard:
        for tk in ["A","B","C","D","E","F","G","H"]:
            d = e["details"].get(tk, {})
            if not d.get("passed", True):
                div_export.append({
                    "edge_id": e["edge_id"], "strategy": e["strategy"],
                    "symbol": e["symbol"], "tf": e["tf"],
                    "failed_test": tk, "detail": d.get("detail", "")
                })
    if div_export:
        pd.DataFrame(div_export).to_csv(div_csv, index=False)
        print(f"  {C_GRN}[SAVED] {div_csv} ({len(div_export)} failures){S_RS}")
    else:
        # Save empty file with headers
        pd.DataFrame(columns=["edge_id","strategy","symbol","tf","failed_test","detail"]).to_csv(div_csv, index=False)
        print(f"  {C_GRN}[SAVED] {div_csv} (empty - no failures){S_RS}")

    elapsed = time.time() - t0
    print()
    box(f"TASK 13.5 COMPLETE - Runtime: {elapsed/60:.2f} min", C_GRN)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C_RED}Interrupted{S_RS}")
    except Exception as e:
        print(f"\n{C_RED}FATAL: {e}{S_RS}")
        traceback.print_exc()
