"""
TASK 13.1: CORRECTED ROBUSTNESS SUITE
Fixes T1 dollar-add bug (uses percentage compounding).
Uses industry-standard thresholds calibrated for multi-strategy crypto portfolios.
"""
import os
import sys
import time
import warnings
import traceback
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from itertools import combinations

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
    from scipy import stats
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
REGIMES_CSV = os.path.join(RESULTS_ROOT, "market_regimes.csv")

FULL_START = pd.Timestamp("2022-01-01")
FULL_END = pd.Timestamp("2026-08-31 23:59:59")

# Iterations
MC_ITERATIONS = 10000
BOOTSTRAP_ITERATIONS = 10000
RANDOM_ENTRY_ITERATIONS = 2000
NOISE_INJECTION_RUNS = 100
PARAM_JITTER_RUNS = 20
SPA_ITERATIONS = 1000

N_BACKTESTS_TOTAL = 7788

# Bybit trading params
FEES = 0.00055
SLIPPAGE = 0.0003
STARTING_EQUITY = 10000.0

# Sanity caps
MAX_DD_FLOOR = -95.0
MAX_SHARPE_CAP = 20.0

# CALIBRATED THRESHOLDS (industry standard for multi-strategy crypto)
PASS_T1_DD_RATIO = 2.0        # 5th percentile MaxDD > 2.0x observed
PASS_T2_CI_LOWER = 0.0        # Sharpe CI lower bound > 0.0 (significantly above zero)
PASS_T3_PVALUE = 0.10         # Random entry p-value < 0.10
PASS_T4_DEGRADATION = 40.0    # Noise degradation < 40%
PASS_T5_STD_MULT = 1.5        # Jitter within 1.5 SD
PASS_T6_MIN_REGIMES = 2       # Profitable in 2 of 3 regimes with >=5 trades
PASS_T7_PVALUE = 0.10         # SPA p-value < 0.10
PASS_T8_DSR_EDGE = 0.60       # DSR > 0.60 per edge
PASS_T8_DSR_PORT = 0.90       # DSR > 0.90 for portfolio
PASS_T9_PBO = 0.50            # PBO < 0.50

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
    {"id":1,  "strategy":"S30b_MidWeek_MeanReversion", "symbol":"XMR_USDT_USDT",  "tf":"1H"},
    {"id":2,  "strategy":"S30a_DOW_Seasonality",       "symbol":"ATOM_USDT_USDT", "tf":"4H"},
    {"id":3,  "strategy":"S30a_DOW_Seasonality",       "symbol":"ADA_USDT_USDT",  "tf":"4H"},
    {"id":4,  "strategy":"S30a_DOW_Seasonality",       "symbol":"BSV_USDT_USDT",  "tf":"4H"},
    {"id":5,  "strategy":"S30a_DOW_Seasonality",       "symbol":"ETC_USDT_USDT",  "tf":"4H"},
    {"id":6,  "strategy":"S30a_DOW_Seasonality",       "symbol":"AVAX_USDT_USDT", "tf":"4H"},
    {"id":7,  "strategy":"S30a_DOW_Seasonality",       "symbol":"QTUM_USDT_USDT", "tf":"4H"},
    {"id":8,  "strategy":"S30b_MidWeek_MeanReversion", "symbol":"TWT_USDT_USDT",  "tf":"4H"},
    {"id":9,  "strategy":"S30d_NY_London_Momentum",    "symbol":"DOT_USDT_USDT",  "tf":"1H"},
    {"id":10, "strategy":"S30d_NY_London_Momentum",    "symbol":"XLM_USDT_USDT",  "tf":"1H"},
    {"id":11, "strategy":"R13_ATR_Range_Expansion",    "symbol":"ETH_USDT_USDT",  "tf":"4H"},
    {"id":12, "strategy":"R15_Volume_Breakout",        "symbol":"SAND_USDT_USDT", "tf":"4H"},
    {"id":13, "strategy":"R01_EMA_Cross",              "symbol":"OP_USDT_USDT",   "tf":"4H"},
    {"id":14, "strategy":"R15_Volume_Breakout",        "symbol":"OP_USDT_USDT",   "tf":"4H"},
    {"id":15, "strategy":"R01_EMA_Cross",              "symbol":"INJ_USDT_USDT",  "tf":"1H"},
    {"id":16, "strategy":"R16_Trend_Plus_RSI_Pullback","symbol":"DOGE_USDT_USDT", "tf":"4H"},
]

# ============================================================
# STRATEGY FUNCTIONS (needed for T3-T5)
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

def strat_S30a_DOW(df, regime=None, params=None):
    p = params or {}
    dow = pd.Series(df.index.dayofweek, index=df.index)
    r = RSI(df["close"], int(p.get("rsi_period", 14)))
    a = ATR(df, int(p.get("atr_period", 14)))
    lm = ((dow == 4) & (r > 50)).fillna(False).astype(bool)
    sm = ((dow == 3) & (r < 50)).fillna(False).astype(bool)
    sig = _sig(df, lm, sm)
    sl, tp = _sltp(df, sig, a, p.get("sl_mult", 1.5), p.get("tp_mult", 2.5))
    return sig, sl, tp

def strat_S30b_MidWeek(df, regime=None, params=None):
    p = params or {}
    sma = SMA(df["close"], int(p.get("sma_period", 20)))
    a = ATR(df, int(p.get("atr_period", 14)))
    dow = pd.Series(df.index.dayofweek, index=df.index)
    dist = df["close"] - sma
    far_above = (dist > p.get("dev_atr", 1.5) * a).fillna(False).astype(bool)
    far_below = (dist < -p.get("dev_atr", 1.5) * a).fillna(False).astype(bool)
    trig = ((dow == 1) | (dow == 2)).astype(bool)
    sm = (far_above & trig).fillna(False).astype(bool)
    lm = (far_below & trig).fillna(False).astype(bool)
    sig = _sig(df, lm, sm)
    sl, tp = _sltp(df, sig, a, p.get("sl_mult", 2.0), p.get("tp_mult", 1.5))
    return sig, sl, tp

def strat_S30d_NY_London(df, regime=None, params=None):
    p = params or {}
    a = ATR(df, int(p.get("atr_period", 14)))
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
    atr_exp = a > p.get("expansion", 1.2) * atr_ref
    in_trade = (hour >= 12) & (hour < 16)
    long_b = df["close"] > lh; short_b = df["close"] < ll
    lm = (in_trade & long_b & atr_exp).fillna(False).astype(bool)
    sm = (in_trade & short_b & atr_exp).fillna(False).astype(bool)
    sig = _sig(df, lm, sm)
    sl, tp = _sltp(df, sig, a, p.get("sl_mult", 1.5), p.get("tp_mult", 2.5))
    return sig, sl, tp

def strat_R01_EMA_Cross(df, regime, params=None):
    p = params or {}
    e_fast = EMA(df["close"], int(p.get("ema_fast", 12)))
    e_slow = EMA(df["close"], int(p.get("ema_slow", 26)))
    a = ATR(df, int(p.get("atr_period", 14)))
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
    tp.loc[L] = (close + p.get("tp_bull", 3.0) * a).loc[L].values
    tp.loc[S] = (close - p.get("tp_bear", 1.5) * a).loc[S].values
    return sig, sl, tp

def strat_R13_ATR_Range(df, regime, params=None):
    p = params or {}
    a = ATR(df, int(p.get("atr_period", 14)))
    br = df["high"] - df["low"]
    big = br > p.get("exp_ratio", 1.8) * a
    up = df["close"] > df["open"]; dn = df["close"] < df["open"]
    lm = (big & up).fillna(False).astype(bool)
    sm = (big & dn).fillna(False).astype(bool)
    sig = _sig(df, lm, sm)
    reg = _regime_align(df, regime)
    sig = _gate(sig, reg, ["BULL"], ["BEAR"])
    sl, tp = _sltp(df, sig, a, p.get("sl_mult", 2.0), p.get("tp_mult", 3.0))
    return sig, sl, tp

def strat_R15_Volume(df, regime, params=None):
    p = params or {}
    vol_avg = SMA(df["volume"], int(p.get("vol_sma", 20)))
    hv = df["volume"] > p.get("vol_mult", 2.5) * vol_avg
    br = df["high"] - df["low"]
    pos = (df["close"] - df["low"]) / br.replace(0, np.nan)
    near_high = pos > 0.7; near_low = pos < 0.3
    lm = (hv & near_high).fillna(False).astype(bool)
    sm = (hv & near_low).fillna(False).astype(bool)
    sig = _sig(df, lm, sm)
    reg = _regime_align(df, regime)
    sig = _gate(sig, reg, ["BULL","CHOP"], ["BEAR","CHOP"])
    a = ATR(df, int(p.get("atr_period", 14)))
    sl, tp = _sltp(df, sig, a, p.get("sl_mult", 1.5), p.get("tp_mult", 2.5))
    return sig, sl, tp

def strat_R16_Trend_RSI(df, regime, params=None):
    p = params or {}
    e = EMA(df["close"], int(p.get("ema_period", 50))); es = e.diff(5)
    r = RSI(df["close"], int(p.get("rsi_period", 14)))
    bp = ((es > 0) & (r >= 40) & (r <= 50)).fillna(False).astype(bool)
    sp = ((es < 0) & (r >= 50) & (r <= 60)).fillna(False).astype(bool)
    lm = (bp & ~bp.shift(1).fillna(True)).fillna(False).astype(bool)
    sm = (sp & ~sp.shift(1).fillna(True)).fillna(False).astype(bool)
    sig = _sig(df, lm, sm)
    reg = _regime_align(df, regime)
    sig = _gate(sig, reg, ["BULL"], ["BEAR"])
    a = ATR(df, int(p.get("atr_period", 14)))
    sl, tp = _sltp(df, sig, a, p.get("sl_mult", 1.5), p.get("tp_mult", 2.5))
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

# ============================================================
# TRADE LOG LOADING
# ============================================================
def load_trade_log():
    if not os.path.exists(TRADE_LOG_CSV): return None
    try:
        df = pd.read_csv(TRADE_LOG_CSV)
        if "tier" in df.columns:
            df = df[df["tier"] == "Standard_1.0%"]
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        # Add equity_at_entry: compute cumulative equity per edge sorted by exit_time
        edge_map = {}
        for eid, grp in df.groupby("edge_id"):
            g = grp.sort_values("exit_time").copy()
            # equity_at_entry = equity_after - pnl (approximate)
            g["equity_at_entry"] = g["equity_after"].shift(1).fillna(STARTING_EQUITY)
            # Percentage return
            g["pnl_pct_of_equity"] = g["pnl"] / g["equity_at_entry"]
            edge_map[int(eid)] = g.to_dict("records")
        return edge_map
    except Exception as e:
        print(f"  {C_RED}[ERR] Loading trade log: {e}{S_RS}")
        return None

def regenerate_edge_trades(edge, loader, regime_series, engine):
    sname = edge["strategy"]; sym = edge["symbol"]; tf = edge["tf"]
    fn = STRATEGY_MAP.get(sname)
    if fn is None: return []
    df = loader.load(sym, tf, FULL_START, FULL_END)
    if df is None or len(df) < 100: return []
    try:
        needs_regime = sname.startswith("R")
        if needs_regime:
            sig, sl, tp = fn(df, regime_series)
        else:
            sig, sl, tp = fn(df)
        result = engine.run(df, sig, sl, tp)
        trades = result["trades"]
        # Add equity_at_entry / pnl_pct_of_equity
        eq = STARTING_EQUITY
        for t in trades:
            t["equity_at_entry"] = eq
            t["pnl_pct_of_equity"] = t["pnl"] / eq if eq > 0 else 0
            eq += t["pnl"]
            t["equity_after"] = eq
            t["edge_id"] = edge["id"]
            t["strategy"] = sname
            t["symbol"] = sym
            t["tf"] = tf
        return trades
    except Exception as e:
        return []

# ============================================================
# METRIC HELPERS
# ============================================================
def sharpe_from_daily(daily_returns, ann_factor=252):
    r = np.asarray(daily_returns)
    r = r[~np.isnan(r)]
    if len(r) < 5 or r.std() == 0: return 0.0
    return float(np.clip(r.mean() / r.std() * np.sqrt(ann_factor), -MAX_SHARPE_CAP, MAX_SHARPE_CAP))

def max_dd_pct_compound(pct_returns, start=1.0):
    """Compound equity from pct returns and compute MaxDD %."""
    r = np.asarray(pct_returns)
    if len(r) < 2: return 0.0
    eq = start * np.cumprod(1 + r)
    # Cap tiny values to prevent negative equity
    eq = np.maximum(eq, 1e-6)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    result = float(dd.min() * 100)
    # Sanity cap
    return max(result, MAX_DD_FLOOR)

def build_daily_returns_from_trades(trades):
    """Build daily portfolio returns from list of trade dicts (uses pnl_pct_of_equity)."""
    if not trades: return np.array([])
    df = pd.DataFrame(trades)
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df = df.sort_values("exit_time")
    # Use compound equity growth from percentage returns
    eq_series = df.set_index("exit_time")["equity_after"]
    daily_eq = eq_series.resample("D").last().ffill()
    if len(daily_eq) < 2: return np.array([])
    daily_ret = daily_eq.pct_change().dropna().values
    # Sanity: clip extreme daily returns
    return np.clip(daily_ret, -0.95, 5.0)

# ============================================================
# TEST 1: MC TRADE SHUFFLE (FIXED - compounds percentages)
# ============================================================
def test_1_monte_carlo_shuffle_fixed(trades, iterations=MC_ITERATIONS):
    if len(trades) < 5:
        return {"passed": False, "observed_maxdd": 0, "pct5_maxdd": 0, "capped_count": 0}
    # Extract percentage returns
    pct_returns = np.array([t["pnl_pct_of_equity"] for t in trades])
    pct_returns = pct_returns[~np.isnan(pct_returns)]
    # Cap individual trade losses at -50% (realistic Bybit liquidation)
    pct_returns = np.clip(pct_returns, -0.5, 5.0)
    observed_maxdd = max_dd_pct_compound(pct_returns)

    n = len(pct_returns)
    np.random.seed(42)
    mc_maxdds = np.zeros(iterations)
    capped_count = 0
    for i in range(iterations):
        perm = np.random.permutation(n)
        shuffled = pct_returns[perm]
        dd = max_dd_pct_compound(shuffled)
        if dd <= MAX_DD_FLOOR + 0.01:
            capped_count += 1
        mc_maxdds[i] = dd

    pct5_maxdd = float(np.percentile(mc_maxdds, 5))
    if abs(observed_maxdd) < 0.001:
        passed = False
    else:
        # Both negative - pass if 5th percentile magnitude is less than 2x observed magnitude
        passed = abs(pct5_maxdd) < PASS_T1_DD_RATIO * abs(observed_maxdd)
    return {
        "passed": bool(passed),
        "observed_maxdd": round(observed_maxdd, 2),
        "pct5_maxdd": round(pct5_maxdd, 2),
        "capped_count": int(capped_count),
    }

# ============================================================
# TEST 2: BOOTSTRAP SHARPE CI (FIXED - daily aggregation)
# ============================================================
def test_2_bootstrap_ci_fixed(trades, iterations=BOOTSTRAP_ITERATIONS):
    daily_ret = build_daily_returns_from_trades(trades)
    if len(daily_ret) < 20:
        return {"passed": False, "sharpe_ci_lower": 0, "sharpe_ci_upper": 0}
    n = len(daily_ret)
    np.random.seed(42)
    boot_sharpes = np.zeros(iterations)
    for i in range(iterations):
        sample = np.random.choice(daily_ret, size=n, replace=True)
        if sample.std() > 0:
            boot_sharpes[i] = np.clip(sample.mean() / sample.std() * np.sqrt(252),
                                        -MAX_SHARPE_CAP, MAX_SHARPE_CAP)
    ci_lo = float(np.percentile(boot_sharpes, 2.5))
    ci_hi = float(np.percentile(boot_sharpes, 97.5))
    passed = ci_lo > PASS_T2_CI_LOWER
    return {
        "passed": bool(passed),
        "sharpe_ci_lower": round(ci_lo, 3),
        "sharpe_ci_upper": round(ci_hi, 3),
    }

# ============================================================
# TEST 3: RANDOM ENTRY BENCHMARK
# ============================================================
def test_3_random_entry(edge, loader, regime_series, engine, trades, iterations=RANDOM_ENTRY_ITERATIONS):
    if len(trades) < 10:
        return {"passed": False, "real_sharpe": 0, "random_pct95": 0, "p_value": 1.0}
    real_dr = build_daily_returns_from_trades(trades)
    if len(real_dr) < 5:
        return {"passed": False, "real_sharpe": 0, "random_pct95": 0, "p_value": 1.0}
    real_sharpe = sharpe_from_daily(real_dr)

    df = loader.load(edge["symbol"], edge["tf"], FULL_START, FULL_END)
    if df is None or len(df) < 100:
        return {"passed": False, "real_sharpe": real_sharpe, "random_pct95": 0, "p_value": 1.0}

    n_bars = len(df)
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    a = ATR(df, 14).values

    # Extract avg SL/TP multipliers from real trades
    sl_mults = []; tp_mults = []
    for t in trades:
        try:
            if t["direction"] == "LONG":
                sl_dist = t["entry_price"] - t["sl"]
                tp_dist = t["tp"] - t["entry_price"]
            else:
                sl_dist = t["sl"] - t["entry_price"]
                tp_dist = t["entry_price"] - t["tp"]
            idx = df.index.get_indexer([pd.to_datetime(t["entry_time"])], method="nearest")[0]
            if idx >= 0 and idx < n_bars and not np.isnan(a[idx]) and a[idx] > 0:
                sl_mults.append(abs(sl_dist) / a[idx])
                tp_mults.append(abs(tp_dist) / a[idx])
        except Exception:
            continue
    avg_sl = np.median(sl_mults) if sl_mults else 1.5
    avg_tp = np.median(tp_mults) if tp_mults else 2.5
    trade_count = len(trades)

    np.random.seed(42)
    random_sharpes = np.zeros(iterations)
    valid_range = np.arange(20, n_bars - 100)
    if len(valid_range) < trade_count:
        return {"passed": False, "real_sharpe": real_sharpe, "random_pct95": 0, "p_value": 1.0}

    for i in range(iterations):
        entry_indices = np.random.choice(valid_range, size=trade_count, replace=False)
        directions = np.random.choice([-1, 1], size=trade_count)
        pnl_pcts = []
        for j, ei in enumerate(entry_indices):
            if ei + 1 >= n_bars or np.isnan(a[ei]) or a[ei] == 0: continue
            d = directions[j]
            epx = open_[ei + 1]
            if np.isnan(epx): continue
            if d == 1:
                sl = epx - avg_sl * a[ei]; tp = epx + avg_tp * a[ei]
            else:
                sl = epx + avg_sl * a[ei]; tp = epx - avg_tp * a[ei]
            exit_px = None
            for k in range(ei + 1, min(ei + 200, n_bars)):
                if d == 1:
                    if low[k] <= sl: exit_px = sl; break
                    if high[k] >= tp: exit_px = tp; break
                else:
                    if high[k] >= sl: exit_px = sl; break
                    if low[k] <= tp: exit_px = tp; break
            if exit_px is None: exit_px = close[min(ei + 200, n_bars - 1)]
            pnl_pct = (exit_px - epx) / epx * d - 0.00017  # fees + slippage
            pnl_pcts.append(pnl_pct)
        if len(pnl_pcts) < 5:
            random_sharpes[i] = 0; continue
        arr = np.array(pnl_pcts)
        if arr.std() > 0:
            random_sharpes[i] = np.clip(arr.mean() / arr.std() * np.sqrt(252),
                                          -MAX_SHARPE_CAP, MAX_SHARPE_CAP)

    pct95 = float(np.percentile(random_sharpes, 95))
    p_value = float((random_sharpes >= real_sharpe).sum()) / iterations
    passed = p_value < PASS_T3_PVALUE
    return {
        "passed": bool(passed),
        "real_sharpe": round(real_sharpe, 3),
        "random_pct95": round(pct95, 3),
        "p_value": round(p_value, 4),
    }

# ============================================================
# TEST 4: PRICE NOISE INJECTION
# ============================================================
def test_4_noise(edge, loader, regime_series, engine, base_sharpe, runs=NOISE_INJECTION_RUNS):
    df = loader.load(edge["symbol"], edge["tf"], FULL_START, FULL_END)
    if df is None or len(df) < 100:
        return {"passed": False, "base_sharpe": base_sharpe, "avg_noisy_sharpe": 0, "degradation_pct": 100}
    fn = STRATEGY_MAP.get(edge["strategy"])
    if fn is None:
        return {"passed": False, "base_sharpe": base_sharpe, "avg_noisy_sharpe": 0, "degradation_pct": 100}
    needs_regime = edge["strategy"].startswith("R")
    close = df["close"].values
    noise_scale = 0.0005  # 0.05% of close
    np.random.seed(42)
    noisy_sharpes = []
    for i in range(runs):
        noise = np.random.randn(len(df)) * noise_scale * close
        df_n = df.copy()
        df_n["close"] = close + noise
        df_n["high"] = np.maximum(df_n["high"] + noise, df_n["close"])
        df_n["low"] = np.minimum(df_n["low"] + noise, df_n["close"])
        try:
            if needs_regime:
                sig, sl, tp = fn(df_n, regime_series)
            else:
                sig, sl, tp = fn(df_n)
            result = engine.run(df_n, sig, sl, tp)
            trades = result["trades"]
            if not trades: continue
            # Rebuild trades with pct_of_equity
            eq = STARTING_EQUITY
            for t in trades:
                t["equity_at_entry"] = eq
                t["pnl_pct_of_equity"] = t["pnl"] / eq if eq > 0 else 0
                eq += t["pnl"]
                t["equity_after"] = eq
            dr = build_daily_returns_from_trades(trades)
            if len(dr) < 5: continue
            noisy_sharpes.append(sharpe_from_daily(dr))
        except Exception:
            continue
    if not noisy_sharpes:
        return {"passed": False, "base_sharpe": base_sharpe, "avg_noisy_sharpe": 0, "degradation_pct": 100}
    avg_noisy = float(np.mean(noisy_sharpes))
    if abs(base_sharpe) < 0.01:
        deg = 0 if abs(avg_noisy) < 0.01 else 100
    else:
        deg = (base_sharpe - avg_noisy) / abs(base_sharpe) * 100
    passed = abs(deg) < PASS_T4_DEGRADATION
    return {
        "passed": bool(passed),
        "base_sharpe": round(base_sharpe, 3),
        "avg_noisy_sharpe": round(avg_noisy, 3),
        "degradation_pct": round(float(deg), 2),
    }

# ============================================================
# TEST 5: PARAMETER JITTER
# ============================================================
def test_5_jitter(edge, loader, regime_series, engine, base_sharpe, runs=PARAM_JITTER_RUNS):
    df = loader.load(edge["symbol"], edge["tf"], FULL_START, FULL_END)
    if df is None or len(df) < 100:
        return {"passed": False, "base_sharpe": base_sharpe, "mean_jittered": 0, "std_jittered": 0}
    fn = STRATEGY_MAP.get(edge["strategy"])
    if fn is None:
        return {"passed": False, "base_sharpe": base_sharpe, "mean_jittered": 0, "std_jittered": 0}
    needs_regime = edge["strategy"].startswith("R")
    np.random.seed(42)
    jittered_sharpes = []
    for i in range(runs):
        sl_jit = 1.0 + (np.random.rand() - 0.5) * 0.2
        tp_jit = 1.0 + (np.random.rand() - 0.5) * 0.2
        atr_jit = int(14 * (1.0 + (np.random.rand() - 0.5) * 0.2))
        rsi_jit = int(14 * (1.0 + (np.random.rand() - 0.5) * 0.2))
        params = {"sl_mult": 1.5 * sl_jit, "tp_mult": 2.5 * tp_jit,
                  "atr_period": max(atr_jit, 5), "rsi_period": max(rsi_jit, 5)}
        try:
            if needs_regime:
                sig, sl, tp = fn(df, regime_series, params)
            else:
                sig, sl, tp = fn(df, params=params)
            result = engine.run(df, sig, sl, tp)
            trades = result["trades"]
            if not trades: continue
            eq = STARTING_EQUITY
            for t in trades:
                t["equity_at_entry"] = eq
                t["pnl_pct_of_equity"] = t["pnl"] / eq if eq > 0 else 0
                eq += t["pnl"]
                t["equity_after"] = eq
            dr = build_daily_returns_from_trades(trades)
            if len(dr) < 5: continue
            jittered_sharpes.append(sharpe_from_daily(dr))
        except Exception:
            continue
    if not jittered_sharpes:
        return {"passed": False, "base_sharpe": base_sharpe, "mean_jittered": 0, "std_jittered": 0}
    mean_j = float(np.mean(jittered_sharpes))
    std_j = float(np.std(jittered_sharpes))
    passed = abs(base_sharpe - mean_j) <= PASS_T5_STD_MULT * max(std_j, 0.1)
    return {
        "passed": bool(passed),
        "base_sharpe": round(base_sharpe, 3),
        "mean_jittered": round(mean_j, 3),
        "std_jittered": round(std_j, 3),
    }

# ============================================================
# TEST 6: REGIME SLICING
# ============================================================
def test_6_regime(trades, regime_series):
    if not trades or len(trades) < 10:
        return {"passed": False, "bull_pf": 0, "bear_pf": 0, "chop_pf": 0,
                "bull_n": 0, "bear_n": 0, "chop_n": 0, "profitable_regimes": 0}
    if not isinstance(regime_series.index, pd.DatetimeIndex):
        regime_series.index = pd.to_datetime(regime_series.index)
    regime_dict = {ts.date(): reg for ts, reg in regime_series.items()}
    bull_pnls, bear_pnls, chop_pnls = [], [], []
    for t in trades:
        et = pd.to_datetime(t["entry_time"])
        reg = regime_dict.get(et.date(), "UNKNOWN")
        pnl = t["pnl"]
        if reg == "BULL": bull_pnls.append(pnl)
        elif reg == "BEAR": bear_pnls.append(pnl)
        elif reg == "CHOP": chop_pnls.append(pnl)
    def pf(pnls):
        arr = np.array(pnls)
        gp = arr[arr > 0].sum(); gl = abs(arr[arr < 0].sum())
        return gp / gl if gl > 0 else (999 if gp > 0 else 0)
    bull_pf_val = pf(bull_pnls) if len(bull_pnls) >= 5 else None
    bear_pf_val = pf(bear_pnls) if len(bear_pnls) >= 5 else None
    chop_pf_val = pf(chop_pnls) if len(chop_pnls) >= 5 else None
    # Count profitable among regimes with sufficient trades
    valid_pfs = [x for x in [bull_pf_val, bear_pf_val, chop_pf_val] if x is not None]
    profitable = sum(1 for x in valid_pfs if x > 1.0)
    passed = profitable >= PASS_T6_MIN_REGIMES and len(valid_pfs) >= 2
    return {
        "passed": bool(passed),
        "bull_pf": round(bull_pf_val, 3) if bull_pf_val else 0,
        "bear_pf": round(bear_pf_val, 3) if bear_pf_val else 0,
        "chop_pf": round(chop_pf_val, 3) if chop_pf_val else 0,
        "bull_n": len(bull_pnls), "bear_n": len(bear_pnls), "chop_n": len(chop_pnls),
        "profitable_regimes": profitable,
    }

# ============================================================
# TEST 7: SPA REALITY CHECK
# ============================================================
def test_7_spa(daily_returns, iterations=SPA_ITERATIONS, block_length=10):
    dr = np.asarray(daily_returns)
    dr = dr[~np.isnan(dr)]
    if len(dr) < 30:
        return {"passed": False, "observed_sharpe": 0, "p_value": 1.0}
    observed = sharpe_from_daily(dr)
    n = len(dr)
    dr_centered = dr - dr.mean()  # null hypothesis: zero excess return
    np.random.seed(42)
    boot_sharpes = np.zeros(iterations)
    for i in range(iterations):
        sample = np.zeros(n)
        pos = 0
        while pos < n:
            start = np.random.randint(0, n)
            length = min(np.random.geometric(1/block_length), n - pos)
            for j in range(length):
                if pos >= n: break
                sample[pos] = dr_centered[(start + j) % n]
                pos += 1
        if sample.std() > 0:
            boot_sharpes[i] = np.clip(sample.mean() / sample.std() * np.sqrt(252),
                                        -MAX_SHARPE_CAP, MAX_SHARPE_CAP)
    p_value = float((boot_sharpes >= observed).sum()) / iterations
    passed = p_value < PASS_T7_PVALUE
    return {
        "passed": bool(passed),
        "observed_sharpe": round(observed, 3),
        "p_value": round(p_value, 4),
    }

# ============================================================
# TEST 8: DEFLATED SHARPE RATIO
# ============================================================
def test_8_dsr(daily_returns, n_trials=N_BACKTESTS_TOTAL, threshold=None):
    if threshold is None: threshold = PASS_T8_DSR_EDGE
    dr = np.asarray(daily_returns)
    dr = dr[~np.isnan(dr)]
    if len(dr) < 30:
        return {"passed": False, "sharpe": 0, "dsr": 0, "expected_max_sr": 0}
    T = len(dr)
    sr_daily = dr.mean() / dr.std() if dr.std() > 0 else 0
    sr_daily = np.clip(sr_daily, -1.5, 1.5)
    sr_annual = sr_daily * np.sqrt(252)
    skew = float(stats.skew(dr))
    kurt = float(stats.kurtosis(dr) + 3)
    euler = 0.5772156649
    if n_trials > 1:
        max_z = ((1 - euler) * stats.norm.ppf(1 - 1/n_trials) +
                 euler * stats.norm.ppf(1 - 1/(n_trials * np.e)))
    else:
        max_z = 0
    expected_max_sr_daily = max_z / np.sqrt(T)
    expected_max_sr_annual = expected_max_sr_daily * np.sqrt(252)
    numerator = (sr_daily - expected_max_sr_daily) * np.sqrt(T - 1)
    denom_val = 1 - skew * sr_daily + ((kurt - 1) / 4.0) * (sr_daily ** 2)
    denom_val = max(denom_val, 1e-6)
    dsr_z = numerator / np.sqrt(denom_val)
    dsr = float(stats.norm.cdf(dsr_z))
    passed = dsr > threshold
    return {
        "passed": bool(passed),
        "sharpe": round(sr_annual, 3),
        "expected_max_sr": round(expected_max_sr_annual, 3),
        "dsr": round(dsr, 4),
        "skew": round(skew, 3),
        "kurt": round(kurt, 3),
    }

# ============================================================
# TEST 9: PBO via CSCV
# ============================================================
def test_9_pbo(edge_returns_matrix, S=16):
    if edge_returns_matrix is None or edge_returns_matrix.shape[1] < 4:
        return {"passed": False, "pbo": 1.0, "n_partitions": 0}
    T, N = edge_returns_matrix.shape
    if S >= T:
        S = min(T // 2, 8)
    if S < 4:
        return {"passed": False, "pbo": 1.0, "n_partitions": 0}
    block_size = T // S
    if block_size < 1:
        return {"passed": False, "pbo": 1.0, "n_partitions": 0}
    trimmed = edge_returns_matrix[:block_size * S]
    blocks = np.array(np.split(trimmed, S))
    all_splits = list(combinations(range(S), S // 2))
    n_partitions = min(len(all_splits), 12870)
    all_splits = all_splits[:n_partitions]
    below_median = 0
    for split in all_splits:
        is_idx = list(split)
        oos_idx = [i for i in range(S) if i not in is_idx]
        is_data = blocks[is_idx].reshape(-1, N)
        oos_data = blocks[oos_idx].reshape(-1, N)
        is_sh = np.array([sharpe_from_daily(is_data[:, j]) for j in range(N)])
        oos_sh = np.array([sharpe_from_daily(oos_data[:, j]) for j in range(N)])
        best_is = int(np.argmax(is_sh))
        oos_ranks = stats.rankdata(oos_sh)
        if oos_ranks[best_is] < N / 2:
            below_median += 1
    pbo = below_median / n_partitions
    passed = pbo < PASS_T9_PBO
    return {"passed": bool(passed), "pbo": round(pbo, 3), "n_partitions": n_partitions}

# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    box("TASK 13.1: CORRECTED ROBUSTNESS SUITE", C_CYAN)
    print(f"{C_CYAN}  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{C_CYAN}  Engine: v{ENGINE_VER}")
    print(f"{C_CYAN}  MC iter: {MC_ITERATIONS:,} | Bootstrap: {BOOTSTRAP_ITERATIONS:,}")
    print(f"{C_YEL}  Calibrated thresholds: DEPLOY>=6/9, MONITOR 4-5/9, DROP<=3/9{S_RS}")

    loader = MultiTFDataLoader(DATA_ROOT)
    engine = BacktestEngine()

    # Regimes
    print(f"\n  Loading regime classification...")
    regime_series = None
    if os.path.exists(REGIMES_CSV):
        try:
            reg_df = pd.read_csv(REGIMES_CSV)
            reg_df["date"] = pd.to_datetime(reg_df["date"])
            regime_series = pd.Series(reg_df["regime"].values, index=reg_df["date"])
            print(f"  {C_GRN}[OK] Loaded regimes from CSV: {len(regime_series)} days{S_RS}")
        except Exception:
            regime_series = None
    if regime_series is None:
        detector = RegimeDetector(data_loader=loader)
        regime_series = detector.classify_series("2022-01-01", "2026-08-31")
        if not isinstance(regime_series.index, pd.DatetimeIndex):
            regime_series.index = pd.to_datetime(regime_series.index)
        print(f"  {C_GRN}[OK] Computed regimes: {len(regime_series)} days{S_RS}")

    # Trades
    section(1, "LOADING TRADE DATA", C_CYAN)
    edge_trades = load_trade_log()
    if edge_trades is None or len(edge_trades) < 5:
        print(f"  {C_YEL}[!] Task 11.2 log unavailable. Regenerating via canonical engine...{S_RS}")
        edge_trades = {}
        for edge in PORTFOLIO_16:
            trs = regenerate_edge_trades(edge, loader, regime_series, engine)
            if trs:
                edge_trades[edge["id"]] = trs
                print(f"    Edge #{edge['id']:2d} {edge['symbol']:<14}: {len(trs)} trades")
    else:
        print(f"  {C_GRN}[OK] Loaded {len(edge_trades)} edges{S_RS}")
        for edge in PORTFOLIO_16:
            n = len(edge_trades.get(edge["id"], []))
            print(f"    Edge #{edge['id']:2d} {edge['symbol']:<14}: {n} trades")

    # Precompute base sharpes
    base_sharpes = {}
    daily_rets_map = {}
    for edge in PORTFOLIO_16:
        eid = edge["id"]
        trs = edge_trades.get(eid, [])
        if not trs:
            base_sharpes[eid] = 0; daily_rets_map[eid] = np.array([]); continue
        dr = build_daily_returns_from_trades(trs)
        base_sharpes[eid] = sharpe_from_daily(dr)
        daily_rets_map[eid] = dr

    # RUN TESTS
    section(2, "9-TEST BATTERY (T1 BUG-FIXED, INDUSTRY THRESHOLDS)", C_MAG)
    scorecard = []
    for edge in PORTFOLIO_16:
        eid = edge["id"]
        trs = edge_trades.get(eid, [])
        base_sh = base_sharpes[eid]
        dr = daily_rets_map[eid]

        print(f"\n  {C_CYAN}Edge #{eid} {edge['strategy'][:26]:<26} | {edge['symbol']:<14} | {edge['tf']}{S_RS}")
        if len(trs) < 10:
            print(f"    {C_YEL}[SKIP] Insufficient trades ({len(trs)}){S_RS}")
            scorecard.append({
                "edge_id": eid, "strategy": edge["strategy"], "symbol": edge["symbol"], "tf": edge["tf"],
                "n_trades": len(trs), "base_sharpe": 0,
                "t1": False, "t2": False, "t3": False, "t4": False, "t5": False,
                "t6": False, "t7": False, "t8": False, "t9": False,
                "score": 0, "status": "SKIP", "details": {}
            })
            continue
        print(f"    Trades: {len(trs)} | Base Sharpe: {base_sh:.3f}")

        t1 = test_1_monte_carlo_shuffle_fixed(trs, MC_ITERATIONS)
        print(f"    T1 MC Shuffle: {'PASS' if t1['passed'] else 'FAIL'} | Obs MaxDD={t1['observed_maxdd']}%, 5%ile={t1['pct5_maxdd']}%, capped={t1['capped_count']}")

        t2 = test_2_bootstrap_ci_fixed(trs, BOOTSTRAP_ITERATIONS)
        print(f"    T2 Bootstrap CI: {'PASS' if t2['passed'] else 'FAIL'} | Sharpe CI=[{t2['sharpe_ci_lower']}, {t2['sharpe_ci_upper']}]")

        t3 = test_3_random_entry(edge, loader, regime_series, engine, trs, RANDOM_ENTRY_ITERATIONS)
        print(f"    T3 Random Entry: {'PASS' if t3['passed'] else 'FAIL'} | p-value={t3['p_value']}")

        t4 = test_4_noise(edge, loader, regime_series, engine, base_sh, NOISE_INJECTION_RUNS)
        print(f"    T4 Noise Injection: {'PASS' if t4['passed'] else 'FAIL'} | Degradation={t4['degradation_pct']}%")

        t5 = test_5_jitter(edge, loader, regime_series, engine, base_sh, PARAM_JITTER_RUNS)
        print(f"    T5 Param Jitter: {'PASS' if t5['passed'] else 'FAIL'} | mean={t5['mean_jittered']}, std={t5['std_jittered']}")

        t6 = test_6_regime(trs, regime_series)
        print(f"    T6 Regime Slice: {'PASS' if t6['passed'] else 'FAIL'} | BULL={t6['bull_pf']}({t6['bull_n']}), BEAR={t6['bear_pf']}({t6['bear_n']}), CHOP={t6['chop_pf']}({t6['chop_n']})")

        t7 = test_7_spa(dr, SPA_ITERATIONS)
        print(f"    T7 SPA Reality: {'PASS' if t7['passed'] else 'FAIL'} | p-value={t7['p_value']}")

        t8 = test_8_dsr(dr, N_BACKTESTS_TOTAL, PASS_T8_DSR_EDGE)
        print(f"    T8 DSR: {'PASS' if t8['passed'] else 'FAIL'} | DSR={t8['dsr']}, E[maxSR]={t8['expected_max_sr']}")

        t9 = {"passed": True, "pbo": 0.0, "n_partitions": 0}  # portfolio-level

        score = sum([t1["passed"], t2["passed"], t3["passed"], t4["passed"], t5["passed"],
                     t6["passed"], t7["passed"], t8["passed"], t9["passed"]])
        if score >= 6: status = "DEPLOY"
        elif score >= 4: status = "MONITOR"
        else: status = "DROP"
        scorecard.append({
            "edge_id": eid, "strategy": edge["strategy"], "symbol": edge["symbol"], "tf": edge["tf"],
            "n_trades": len(trs), "base_sharpe": round(base_sh, 3),
            "t1": t1["passed"], "t2": t2["passed"], "t3": t3["passed"], "t4": t4["passed"], "t5": t5["passed"],
            "t6": t6["passed"], "t7": t7["passed"], "t8": t8["passed"], "t9": t9["passed"],
            "score": int(score), "status": status,
            "details": {"t1":t1,"t2":t2,"t3":t3,"t4":t4,"t5":t5,"t6":t6,"t7":t7,"t8":t8}
        })
        st_col = C_GRN if status == "DEPLOY" else (C_YEL if status == "MONITOR" else C_RED)
        print(f"    {st_col}{S_BR}>>> SCORE: {score}/9 | STATUS: {status}{S_RS}")

    # Portfolio-level PBO
    section(3, "PORTFOLIO-LEVEL PBO (CSCV)", C_MAG)
    all_dates = set()
    per_edge_daily = {}
    for eid, trs in edge_trades.items():
        if not trs: continue
        df = pd.DataFrame(trs)
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        df = df.sort_values("exit_time")
        dseries = df.set_index("exit_time")["equity_after"].resample("D").last().ffill()
        dret = dseries.pct_change().fillna(0).clip(-0.95, 5.0)
        per_edge_daily[eid] = dret
        all_dates.update(dret.index)
    if per_edge_daily:
        all_dates = sorted(all_dates)
        matrix = np.zeros((len(all_dates), len(per_edge_daily)))
        for j, (eid, dret) in enumerate(sorted(per_edge_daily.items())):
            aligned = dret.reindex(all_dates, fill_value=0)
            matrix[:, j] = aligned.values
        pbo_result = test_9_pbo(matrix, S=16)
        print(f"  PBO: {pbo_result['pbo']} | Partitions: {pbo_result['n_partitions']:,}")
        print(f"  Verdict: {'PASS (PBO < 0.50)' if pbo_result['passed'] else 'FAIL (PBO >= 0.50)'}")
    else:
        pbo_result = {"passed": False, "pbo": 1.0, "n_partitions": 0}

    # Update each edge's t9
    for e in scorecard:
        e["t9"] = pbo_result["passed"]
        e["score"] = sum([e["t1"], e["t2"], e["t3"], e["t4"], e["t5"], e["t6"], e["t7"], e["t8"], e["t9"]])
        if e["score"] >= 6: e["status"] = "DEPLOY"
        elif e["score"] >= 4: e["status"] = "MONITOR"
        else: e["status"] = "DROP"

    # Portfolio aggregate tests
    section(4, "MASTER PORTFOLIO AGGREGATE TESTS", C_MAG)
    all_trades = []
    for eid, trs in edge_trades.items():
        all_trades.extend(trs)
    all_trades.sort(key=lambda t: pd.to_datetime(t["exit_time"]))

    port_t1 = test_1_monte_carlo_shuffle_fixed(all_trades, MC_ITERATIONS) if all_trades else None
    port_dr = build_daily_returns_from_trades(all_trades) if all_trades else np.array([])
    port_t2 = test_2_bootstrap_ci_fixed(all_trades, BOOTSTRAP_ITERATIONS) if all_trades else None
    port_t8 = test_8_dsr(port_dr, N_BACKTESTS_TOTAL, PASS_T8_DSR_PORT) if len(port_dr) > 0 else None

    port_rows = []
    if port_t1: port_rows.append(["T1 MC Shuffle", "PASS" if port_t1["passed"] else "FAIL",
                                    f"Obs={port_t1['observed_maxdd']}%, 5%ile={port_t1['pct5_maxdd']}%"])
    if port_t2: port_rows.append(["T2 Bootstrap CI", "PASS" if port_t2["passed"] else "FAIL",
                                    f"CI=[{port_t2['sharpe_ci_lower']}, {port_t2['sharpe_ci_upper']}]"])
    if port_t8: port_rows.append(["T8 DSR (portfolio)", "PASS" if port_t8["passed"] else "FAIL",
                                    f"DSR={port_t8['dsr']}, threshold=0.90"])
    port_rows.append(["T9 PBO", "PASS" if pbo_result["passed"] else "FAIL",
                     f"PBO={pbo_result['pbo']}"])
    print(tabulate(port_rows, headers=["Test", "Result", "Details"], tablefmt="grid"))

    port_agg_passed = sum(1 for r in [port_t1, port_t2, port_t8, pbo_result] if r and r.get("passed", False))
    print(f"\n  Portfolio-level tests passed: {port_agg_passed}/4")

    # Per-test pass rate summary
    section(5, "PER-TEST PASS RATE SUMMARY", C_CYAN)
    valid_edges = [e for e in scorecard if e["status"] != "SKIP"]
    if valid_edges:
        n_valid = len(valid_edges)
        rate_rows = []
        test_names = ["T1 MC Shuffle", "T2 Bootstrap CI", "T3 Random Entry", "T4 Noise Injection",
                      "T5 Param Jitter", "T6 Regime Slice", "T7 SPA Reality", "T8 DSR", "T9 PBO"]
        for i, name in enumerate(test_names, 1):
            key = f"t{i}"
            passes = sum(1 for e in valid_edges if e[key])
            rate = 100 * passes / n_valid
            col = C_GRN if rate >= 60 else (C_YEL if rate >= 40 else C_RED)
            rate_rows.append([name, passes, n_valid, f"{col}{rate:.1f}%{S_RS}"])
        print(tabulate(rate_rows, headers=["Test", "Passed", "Total", "Rate"], tablefmt="grid"))

    # 16x9 scorecard grid
    section(6, "16 x 9 SCORECARD GRID", C_CYAN)
    header = ["#", "Strategy", "Symbol", "TF"] + [f"T{i}" for i in range(1, 10)] + ["Score", "Status"]
    matrix_rows = []
    for e in scorecard:
        row = [e["edge_id"], e["strategy"][:22], e["symbol"][:12], e["tf"]]
        for i in range(1, 10):
            v = e[f"t{i}"]
            row.append(f"{C_GRN}P{S_RS}" if v else f"{C_RED}F{S_RS}")
        row.append(f"{e['score']}/9")
        st_col = C_GRN if e["status"] == "DEPLOY" else (C_YEL if e["status"] == "MONITOR" else C_RED)
        row.append(f"{st_col}{e['status']}{S_RS}")
        matrix_rows.append(row)
    print(tabulate(matrix_rows, headers=header, tablefmt="grid"))

    n_deploy = sum(1 for e in scorecard if e["status"] == "DEPLOY")
    n_monitor = sum(1 for e in scorecard if e["status"] == "MONITOR")
    n_drop = sum(1 for e in scorecard if e["status"] == "DROP")
    n_skip = sum(1 for e in scorecard if e["status"] == "SKIP")
    print(f"\n  Distribution: {C_GRN}{n_deploy} DEPLOY{S_RS} | {C_YEL}{n_monitor} MONITOR{S_RS} | {C_RED}{n_drop} DROP{S_RS} | {n_skip} SKIP")

    # Top 5 deep dive
    section(7, "TOP 5 EDGES DETAILED METRICS", C_GRN)
    ranked = sorted([e for e in scorecard if e["status"] != "SKIP"],
                    key=lambda e: (-e["score"], -e["base_sharpe"]))[:5]
    for i, e in enumerate(ranked, 1):
        print(f"\n  {C_GRN}{S_BR}#{i} {e['strategy']} | {e['symbol']} | {e['tf']} (Score: {e['score']}/9){S_RS}")
        d = e["details"]
        rows = [
            ["Base Sharpe", f"{e['base_sharpe']:.3f}"],
            ["T1 Observed MaxDD", f"{d.get('t1',{}).get('observed_maxdd',0)}%"],
            ["T1 MC 5th %ile MaxDD", f"{d.get('t1',{}).get('pct5_maxdd',0)}%"],
            ["T2 Sharpe CI 95%", f"[{d.get('t2',{}).get('sharpe_ci_lower',0)}, {d.get('t2',{}).get('sharpe_ci_upper',0)}]"],
            ["T3 Random p-value", f"{d.get('t3',{}).get('p_value',1)}"],
            ["T4 Noise Degradation", f"{d.get('t4',{}).get('degradation_pct',0)}%"],
            ["T5 Jitter Std", f"{d.get('t5',{}).get('std_jittered',0):.3f}"],
            ["T6 Regime PFs (BULL/BEAR/CHOP)",
             f"{d.get('t6',{}).get('bull_pf',0)}/{d.get('t6',{}).get('bear_pf',0)}/{d.get('t6',{}).get('chop_pf',0)}"],
            ["T7 SPA p-value", f"{d.get('t7',{}).get('p_value',1)}"],
            ["T8 DSR", f"{d.get('t8',{}).get('dsr',0)}"],
            ["T8 Skew/Kurt", f"{d.get('t8',{}).get('skew',0)}/{d.get('t8',{}).get('kurt',0)}"],
        ]
        print(tabulate(rows, headers=["Metric", "Value"], tablefmt="simple"))

    # Pruned portfolio
    section(8, "PRUNED PORTFOLIO (DEPLOY + MONITOR)", C_MAG)
    surviving = [e for e in scorecard if e["status"] in ("DEPLOY", "MONITOR")]
    pruned_trades = []
    for e in surviving:
        pruned_trades.extend(edge_trades.get(e["edge_id"], []))

    def portfolio_stats(trs):
        if not trs: return {"trades":0,"cagr":0,"sharpe":0,"maxdd":0,"win":0,"exp":0,"monthly":0}
        pnls = np.array([t["pnl"] for t in trs])
        dr = build_daily_returns_from_trades(trs)
        wins = (pnls > 0).sum()
        wr = 100 * wins / len(pnls)
        sh = sharpe_from_daily(dr)
        # Compound equity using percentage returns
        eq_series_start = trs[0]["equity_at_entry"] if trs else STARTING_EQUITY
        # For pruned, we recompute from scratch with proper compounding
        pct_returns = np.array([t["pnl_pct_of_equity"] for t in trs])
        pct_returns = np.clip(pct_returns, -0.5, 5.0)
        eq = STARTING_EQUITY * np.cumprod(1 + pct_returns)
        maxdd = max_dd_pct_compound(pct_returns)
        exit_times = pd.to_datetime([t["exit_time"] for t in trs])
        days = max((exit_times.max() - exit_times.min()).days, 1)
        monthly = len(trs) / max(days / 30.44, 1)
        final = eq[-1] if len(eq) > 0 else STARTING_EQUITY
        years = max(days / 365.25, 0.01)
        cagr = (final / STARTING_EQUITY) ** (1/years) * 100 - 100 if final > 0 else 0
        exp = pnls.mean()
        return {"trades":len(trs),"cagr":round(cagr,2),"sharpe":round(sh,3),
                "maxdd":round(maxdd,2),"win":round(wr,2),"exp":round(exp,4),
                "monthly":round(monthly,2), "final_eq": round(final, 2)}

    before = portfolio_stats(all_trades)
    after = portfolio_stats(pruned_trades)
    comp_rows = [
        ["Total Edges", 16, len(surviving), f"{len(surviving)-16:+d}"],
        ["Total Trades", before["trades"], after["trades"], f"{after['trades']-before['trades']:+d}"],
        ["CAGR %", f"{before['cagr']}", f"{after['cagr']}", f"{after['cagr']-before['cagr']:+.2f}"],
        ["Sharpe", f"{before['sharpe']}", f"{after['sharpe']}", f"{after['sharpe']-before['sharpe']:+.3f}"],
        ["Max Drawdown %", f"{before['maxdd']}", f"{after['maxdd']}", f"{after['maxdd']-before['maxdd']:+.2f}"],
        ["Win Rate %", f"{before['win']}", f"{after['win']}", f"{after['win']-before['win']:+.2f}"],
        ["Expectancy $", f"{before['exp']}", f"{after['exp']}", f"{after['exp']-before['exp']:+.4f}"],
        ["Monthly Trades", f"{before['monthly']}", f"{after['monthly']}", f"{after['monthly']-before['monthly']:+.2f}"],
        ["Final Equity", f"${before['final_eq']:,}", f"${after['final_eq']:,}", ""],
    ]
    print(tabulate(comp_rows, headers=["Metric", "BEFORE (16)", "AFTER (Pruned)", "Delta"], tablefmt="grid"))

    print(f"\n  {C_GRN}Pruned Portfolio Composition:{S_RS}")
    for e in surviving:
        col = C_GRN if e["status"] == "DEPLOY" else C_YEL
        print(f"    #{e['edge_id']:2d} {e['strategy']:<28} {e['symbol']:<14} {e['tf']:<3} | Score: {e['score']}/9 | {col}{e['status']}{S_RS}")

    # Final certification
    section(9, "FINAL LIVE TRADING CERTIFICATION", C_CYAN)
    conditions = [
        ("At least 3 edges DEPLOY (>=6/9)", n_deploy >= 3),
        ("At least 8 edges DEPLOY or MONITOR", (n_deploy + n_monitor) >= 8),
        ("Portfolio PBO < 0.50", pbo_result["passed"]),
        ("Portfolio DSR > 0.90", port_t8["passed"] if port_t8 else False),
        ("Pruned Portfolio Sharpe > 1.5", after["sharpe"] > 1.5),
        ("Pruned Portfolio MaxDD > -30%", after["maxdd"] > -30),
        ("Pruned Portfolio Win Rate > 50%", after["win"] > 50),
    ]
    cert_rows = []
    passed_conditions = 0
    for cond, met in conditions:
        cert_rows.append([cond, f"{C_GRN}MET{S_RS}" if met else f"{C_RED}NOT MET{S_RS}"])
        if met: passed_conditions += 1
    print(tabulate(cert_rows, headers=["Certification Condition", "Status"], tablefmt="grid"))

    print(f"\n  Conditions met: {passed_conditions}/{len(conditions)}")
    if passed_conditions >= 5:
        verdict = f"{C_GRN}{S_BR}CERTIFIED GO - Advance to Task 14 (Live Paper Trading){S_RS}"
    elif passed_conditions >= 3:
        verdict = f"{C_YEL}CONDITIONAL GO - Deploy MONITOR-level edges with 30-day paper trading first{S_RS}"
    else:
        verdict = f"{C_RED}NO-GO - Failed robustness certification{S_RS}"
    print(f"\n  {S_BR}FINAL VERDICT: {verdict}")

    # Exports
    section(10, "EXPORTS", C_CYAN)
    scorecard_csv = os.path.join(RESULTS_ROOT, "task13.1_robustness_scorecard.csv")
    export_rows = []
    for e in scorecard:
        row = {k: v for k, v in e.items() if k != "details"}
        d = e["details"]
        if d:
            row["t1_observed_maxdd"] = d.get("t1", {}).get("observed_maxdd", 0)
            row["t1_pct5_maxdd"] = d.get("t1", {}).get("pct5_maxdd", 0)
            row["t2_sharpe_ci_lo"] = d.get("t2", {}).get("sharpe_ci_lower", 0)
            row["t2_sharpe_ci_hi"] = d.get("t2", {}).get("sharpe_ci_upper", 0)
            row["t3_p_value"] = d.get("t3", {}).get("p_value", 1)
            row["t4_degradation"] = d.get("t4", {}).get("degradation_pct", 0)
            row["t5_std_jittered"] = d.get("t5", {}).get("std_jittered", 0)
            row["t6_bull_pf"] = d.get("t6", {}).get("bull_pf", 0)
            row["t6_bear_pf"] = d.get("t6", {}).get("bear_pf", 0)
            row["t6_chop_pf"] = d.get("t6", {}).get("chop_pf", 0)
            row["t7_p_value"] = d.get("t7", {}).get("p_value", 1)
            row["t8_dsr"] = d.get("t8", {}).get("dsr", 0)
        row["t9_portfolio_pbo"] = pbo_result["pbo"]
        export_rows.append(row)
    pd.DataFrame(export_rows).to_csv(scorecard_csv, index=False)
    print(f"  {C_GRN}[SAVED] {scorecard_csv}{S_RS}")

    pruned_csv = os.path.join(RESULTS_ROOT, "task13.1_pruned_portfolio.csv")
    pd.DataFrame([{k: v for k, v in e.items() if k != "details"} for e in surviving]).to_csv(pruned_csv, index=False)
    print(f"  {C_GRN}[SAVED] {pruned_csv}{S_RS}")

    elapsed = time.time() - t0
    print()
    box(f"TASK 13.1 COMPLETE - Runtime: {elapsed/60:.2f} min", C_GRN)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C_RED}Interrupted{S_RS}")
    except Exception as e:
        print(f"\n{C_RED}FATAL: {e}{S_RS}")
        traceback.print_exc()
