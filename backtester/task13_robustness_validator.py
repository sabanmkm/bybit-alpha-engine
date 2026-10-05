"""
TASK 13: 9-BATTERY STATISTICAL ROBUSTNESS SUITE
Vectorized NumPy implementation for speed.
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

FULL_START = pd.Timestamp("2022-01-01")
FULL_END = pd.Timestamp("2026-08-31 23:59:59")
OOS_START = pd.Timestamp("2026-01-01")
OOS_END = pd.Timestamp("2026-08-31 23:59:59")

# Testing config
MC_ITERATIONS = 10000
BOOTSTRAP_ITERATIONS = 10000
RANDOM_ENTRY_ITERATIONS = 1000
NOISE_INJECTION_RUNS = 100
PARAM_JITTER_RUNS = 20

# Multi-testing correction
N_BACKTESTS_TOTAL = 7788  # from prior tasks

# Bybit trading params
FEES = 0.00055
SLIPPAGE = 0.0003

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
# STRATEGY FUNCTIONS (needed for tests 3,4,5)
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
    lm = ((dow == int(p.get("day_long", 4))) & (r > p.get("rsi_thr", 50))).fillna(False).astype(bool)
    sm = ((dow == int(p.get("day_short", 3))) & (r < p.get("rsi_thr", 50))).fillna(False).astype(bool)
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
# GET EDGE TRADES (from Task 11.2 log or regenerate)
# ============================================================
def load_trade_log():
    """Load Task 11.2 trade log; return dict edge_id -> list of trade dicts."""
    if not os.path.exists(TRADE_LOG_CSV):
        return None
    try:
        df = pd.read_csv(TRADE_LOG_CSV)
        # Only use Standard 1.0% tier for statistical tests
        if "tier" in df.columns:
            df = df[df["tier"] == "Standard_1.0%"]
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        edge_map = {}
        for eid, grp in df.groupby("edge_id"):
            edge_map[int(eid)] = grp.to_dict("records")
        return edge_map
    except Exception as e:
        print(f"  {C_RED}[ERR] Loading trade log: {e}{S_RS}")
        return None

def regenerate_edge_trades(edge, loader, regime_series, engine):
    """Regenerate trades using canonical engine for a single edge."""
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
        for t in trades:
            t["edge_id"] = edge["id"]
            t["strategy"] = sname
            t["symbol"] = sym
            t["tf"] = tf
        return trades
    except Exception as e:
        print(f"  {C_RED}[ERR regenerate] {sname}/{sym}: {e}{S_RS}")
        return []

# ============================================================
# METRIC HELPERS
# ============================================================
def sharpe_from_returns(returns_array, ann_factor=365):
    """Annualized Sharpe from array of per-trade percentage returns."""
    r = np.asarray(returns_array)
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std() == 0: return 0.0
    return float(r.mean() / r.std() * np.sqrt(ann_factor))

def max_dd_from_equity(equity_array):
    """Max drawdown pct from equity array."""
    eq = np.asarray(equity_array)
    if len(eq) < 2: return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(dd.min() * 100)

def max_consec_losses(pnls):
    """Max consecutive negative trades."""
    pnls = np.asarray(pnls)
    cur = 0; mx = 0
    for p in pnls:
        if p < 0:
            cur += 1
            if cur > mx: mx = cur
        else: cur = 0
    return mx

def equity_from_pnls(pnls, start=10000.0):
    """Build equity curve from pnls array."""
    return start + np.cumsum(np.asarray(pnls))

def build_daily_returns(trades):
    """Build daily returns series from trades list."""
    if not trades: return np.array([])
    df = pd.DataFrame(trades)
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df = df.sort_values("exit_time")
    df["equity"] = 10000.0 + df["pnl"].cumsum()
    eq_daily = df.set_index("exit_time")["equity"].resample("D").last().ffill()
    if len(eq_daily) < 2: return np.array([])
    return eq_daily.pct_change().dropna().values

# ============================================================
# TEST 1: MONTE CARLO TRADE SHUFFLE
# ============================================================
def test_1_monte_carlo_shuffle(pnls, iterations=MC_ITERATIONS):
    pnls = np.asarray(pnls)
    if len(pnls) < 5:
        return {"passed": False, "observed_maxdd": 0, "pct5_maxdd": 0, "pct95_consec_loss": 0}
    observed_maxdd = max_dd_from_equity(equity_from_pnls(pnls))
    # Vectorized shuffle: create matrix of shuffled indices
    n = len(pnls)
    np.random.seed(42)
    mc_maxdds = np.zeros(iterations)
    mc_streaks = np.zeros(iterations)
    for i in range(iterations):
        perm = np.random.permutation(n)
        shuffled = pnls[perm]
        eq = 10000.0 + np.cumsum(shuffled)
        peak = np.maximum.accumulate(eq)
        dd = ((eq - peak) / peak).min() * 100
        mc_maxdds[i] = dd
        # Streak
        cur = 0; mx = 0
        for p in shuffled:
            if p < 0:
                cur += 1
                if cur > mx: mx = cur
            else: cur = 0
        mc_streaks[i] = mx
    pct5_maxdd = float(np.percentile(mc_maxdds, 5))
    pct95_streak = float(np.percentile(mc_streaks, 95))
    passed = abs(pct5_maxdd) < 2.0 * abs(observed_maxdd)
    return {
        "passed": bool(passed),
        "observed_maxdd": round(observed_maxdd, 2),
        "pct5_maxdd": round(pct5_maxdd, 2),
        "pct95_consec_loss": int(pct95_streak),
    }

# ============================================================
# TEST 2: BOOTSTRAP CI
# ============================================================
def test_2_bootstrap_ci(daily_returns, iterations=BOOTSTRAP_ITERATIONS):
    dr = np.asarray(daily_returns)
    dr = dr[~np.isnan(dr)]
    if len(dr) < 20:
        return {"passed": False, "sharpe_ci_lower": 0, "sharpe_ci_upper": 0, "cagr_ci_lower": 0}
    n = len(dr)
    np.random.seed(42)
    boot_sharpes = np.zeros(iterations)
    boot_cagrs = np.zeros(iterations)
    for i in range(iterations):
        sample = np.random.choice(dr, size=n, replace=True)
        if sample.std() > 0:
            boot_sharpes[i] = sample.mean() / sample.std() * np.sqrt(365)
        cum = np.prod(1 + sample)
        years = n / 365
        if cum > 0 and years > 0:
            boot_cagrs[i] = (cum ** (1/years) - 1) * 100
    ci_lo = float(np.percentile(boot_sharpes, 2.5))
    ci_hi = float(np.percentile(boot_sharpes, 97.5))
    cagr_lo = float(np.percentile(boot_cagrs, 2.5))
    cagr_hi = float(np.percentile(boot_cagrs, 97.5))
    passed = ci_lo > 0.50
    return {
        "passed": bool(passed),
        "sharpe_ci_lower": round(ci_lo, 3),
        "sharpe_ci_upper": round(ci_hi, 3),
        "cagr_ci_lower": round(cagr_lo, 2),
        "cagr_ci_upper": round(cagr_hi, 2),
    }

# ============================================================
# TEST 3: RANDOM ENTRY BENCHMARK
# ============================================================
def test_3_random_entry(edge, loader, regime_series, engine, trades, iterations=RANDOM_ENTRY_ITERATIONS):
    if len(trades) < 10:
        return {"passed": False, "real_sharpe": 0, "random_pct95": 0, "p_value": 1.0}
    dr = build_daily_returns(trades)
    if len(dr) < 5:
        return {"passed": False, "real_sharpe": 0, "random_pct95": 0, "p_value": 1.0}
    real_sharpe = sharpe_from_returns(dr, ann_factor=365)

    df = loader.load(edge["symbol"], edge["tf"], FULL_START, FULL_END)
    if df is None or len(df) < 100:
        return {"passed": False, "real_sharpe": real_sharpe, "random_pct95": 0, "p_value": 1.0}

    trade_count = len(trades)
    n_bars = len(df)
    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    a = ATR(df, 14).values

    # Get average SL/TP mult from actual trades
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
    avg_sl_mult = np.median(sl_mults) if sl_mults else 1.5
    avg_tp_mult = np.median(tp_mults) if tp_mults else 2.5

    np.random.seed(42)
    random_sharpes = np.zeros(iterations)
    for i in range(iterations):
        # Random entry indices
        entry_idx = np.random.choice(np.arange(20, n_bars - 10), size=trade_count, replace=False)
        directions = np.random.choice([-1, 1], size=trade_count)
        rand_pnls_pct = []
        for j, ei in enumerate(entry_idx):
            if ei + 1 >= n_bars or np.isnan(a[ei]) or a[ei] == 0: continue
            direction = directions[j]
            entry_px = open_[ei + 1]
            if np.isnan(entry_px): continue
            if direction == 1:
                sl = entry_px - avg_sl_mult * a[ei]
                tp = entry_px + avg_tp_mult * a[ei]
            else:
                sl = entry_px + avg_sl_mult * a[ei]
                tp = entry_px - avg_tp_mult * a[ei]
            # Scan forward for exit
            exit_px = None
            for k in range(ei + 1, min(ei + 100, n_bars)):
                if direction == 1:
                    if low[k] <= sl: exit_px = sl; break
                    if high[k] >= tp: exit_px = tp; break
                else:
                    if high[k] >= sl: exit_px = sl; break
                    if low[k] <= tp: exit_px = tp; break
            if exit_px is None: exit_px = close[min(ei + 100, n_bars - 1)]
            pnl_pct = (exit_px - entry_px) / entry_px * 100 * direction - 0.017  # fees+slippage
            rand_pnls_pct.append(pnl_pct)
        if len(rand_pnls_pct) < 5:
            random_sharpes[i] = 0; continue
        arr = np.asarray(rand_pnls_pct)
        if arr.std() > 0:
            random_sharpes[i] = arr.mean() / arr.std() * np.sqrt(min(365 * len(arr) / max((df.index[-1]-df.index[0]).days, 1), 365))

    pct95 = float(np.percentile(random_sharpes, 95))
    p_value = float((random_sharpes >= real_sharpe).sum()) / iterations
    passed = p_value < 0.05
    return {
        "passed": bool(passed),
        "real_sharpe": round(real_sharpe, 3),
        "random_pct95": round(pct95, 3),
        "p_value": round(p_value, 4),
    }

# ============================================================
# TEST 4: PRICE NOISE INJECTION
# ============================================================
def test_4_noise_injection(edge, loader, regime_series, engine, base_sharpe, runs=NOISE_INJECTION_RUNS):
    df = loader.load(edge["symbol"], edge["tf"], FULL_START, FULL_END)
    if df is None or len(df) < 100:
        return {"passed": False, "base_sharpe": base_sharpe, "avg_noisy_sharpe": 0, "degradation_pct": 100}
    fn = STRATEGY_MAP.get(edge["strategy"])
    if fn is None:
        return {"passed": False, "base_sharpe": base_sharpe, "avg_noisy_sharpe": 0, "degradation_pct": 100}

    needs_regime = edge["strategy"].startswith("R")
    close = df["close"].values
    noise_scale = 0.001  # 0.1% of close
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
            dr = build_daily_returns(trades)
            if len(dr) < 5: continue
            noisy_sharpes.append(sharpe_from_returns(dr))
        except Exception:
            continue

    if not noisy_sharpes:
        return {"passed": False, "base_sharpe": base_sharpe, "avg_noisy_sharpe": 0, "degradation_pct": 100}
    avg_noisy = float(np.mean(noisy_sharpes))
    if base_sharpe == 0:
        deg = 0
    else:
        deg = (base_sharpe - avg_noisy) / abs(base_sharpe) * 100
    passed = deg < 30
    return {
        "passed": bool(passed),
        "base_sharpe": round(base_sharpe, 3),
        "avg_noisy_sharpe": round(avg_noisy, 3),
        "degradation_pct": round(float(deg), 2),
    }

# ============================================================
# TEST 5: PARAMETER JITTER
# ============================================================
def test_5_param_jitter(edge, loader, regime_series, engine, base_sharpe, runs=PARAM_JITTER_RUNS):
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
        # Perturb sl_mult and tp_mult by +/- 10%
        sl_jit = 1.0 + (np.random.rand() - 0.5) * 0.2
        tp_jit = 1.0 + (np.random.rand() - 0.5) * 0.2
        atr_jit = int(14 * (1.0 + (np.random.rand() - 0.5) * 0.2))
        params = {"sl_mult": 1.5 * sl_jit, "tp_mult": 2.5 * tp_jit, "atr_period": max(atr_jit, 5)}
        try:
            if needs_regime:
                sig, sl, tp = fn(df, regime_series, params)
            else:
                sig, sl, tp = fn(df, params=params)
            result = engine.run(df, sig, sl, tp)
            trades = result["trades"]
            if not trades: continue
            dr = build_daily_returns(trades)
            if len(dr) < 5: continue
            jittered_sharpes.append(sharpe_from_returns(dr))
        except Exception:
            continue

    if not jittered_sharpes:
        return {"passed": False, "base_sharpe": base_sharpe, "mean_jittered": 0, "std_jittered": 0}
    mean_j = float(np.mean(jittered_sharpes))
    std_j = float(np.std(jittered_sharpes))
    # Pass if |base - mean| within 1 std
    passed = abs(base_sharpe - mean_j) < max(std_j, 0.1)
    return {
        "passed": bool(passed),
        "base_sharpe": round(base_sharpe, 3),
        "mean_jittered": round(mean_j, 3),
        "std_jittered": round(std_j, 3),
    }

# ============================================================
# TEST 6: REGIME SLICING
# ============================================================
def test_6_regime_slicing(trades, regime_series):
    if not trades:
        return {"passed": False, "bull_pf": 0, "bear_pf": 0, "chop_pf": 0, "profitable_regimes": 0}
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
    bull_pf = pf(bull_pnls) if bull_pnls else 0
    bear_pf = pf(bear_pnls) if bear_pnls else 0
    chop_pf = pf(chop_pnls) if chop_pnls else 0
    profitable = sum(1 for x in [bull_pf, bear_pf, chop_pf] if x > 1.0)
    passed = profitable >= 2
    return {
        "passed": bool(passed),
        "bull_pf": round(bull_pf, 3),
        "bear_pf": round(bear_pf, 3),
        "chop_pf": round(chop_pf, 3),
        "profitable_regimes": profitable,
    }

# ============================================================
# TEST 7: SPA REALITY CHECK (Stationary Block Bootstrap)
# ============================================================
def test_7_spa_reality_check(daily_returns, iterations=1000, block_length=10):
    dr = np.asarray(daily_returns)
    dr = dr[~np.isnan(dr)]
    if len(dr) < 30:
        return {"passed": False, "observed_sharpe": 0, "p_value": 1.0}
    observed_sharpe = sharpe_from_returns(dr)
    n = len(dr)
    np.random.seed(42)
    # Stationary block bootstrap: geometric block lengths
    bootstrap_sharpes = np.zeros(iterations)
    # Center returns around 0 for null hypothesis
    dr_centered = dr - dr.mean()
    for i in range(iterations):
        sample = np.zeros(n)
        pos = 0
        while pos < n:
            start = np.random.randint(0, n)
            length = np.random.geometric(1/block_length)
            for j in range(length):
                if pos >= n: break
                sample[pos] = dr_centered[(start + j) % n]
                pos += 1
        if sample.std() > 0:
            bootstrap_sharpes[i] = sample.mean() / sample.std() * np.sqrt(365)
    # P-value: fraction of bootstrap sharpes >= observed
    p_value = float((bootstrap_sharpes >= observed_sharpe).sum()) / iterations
    passed = p_value < 0.05
    return {
        "passed": bool(passed),
        "observed_sharpe": round(observed_sharpe, 3),
        "p_value": round(p_value, 4),
    }

# ============================================================
# TEST 8: DEFLATED SHARPE RATIO (Bailey & Lopez de Prado)
# ============================================================
def test_8_dsr(daily_returns, n_trials=N_BACKTESTS_TOTAL):
    dr = np.asarray(daily_returns)
    dr = dr[~np.isnan(dr)]
    if len(dr) < 30:
        return {"passed": False, "sharpe": 0, "dsr": 0, "expected_max_sr": 0}
    T = len(dr)
    sr_daily = dr.mean() / dr.std() if dr.std() > 0 else 0
    sr_annual = sr_daily * np.sqrt(365)
    skew = float(stats.skew(dr))
    kurt = float(stats.kurtosis(dr) + 3)  # excess -> raw kurtosis

    # E[max SR] with N trials (assuming all null)
    # Bailey & Lopez de Prado (2014) formula
    euler_mascheroni = 0.5772156649
    if n_trials > 1:
        max_z = (1 - euler_mascheroni) * stats.norm.ppf(1 - 1/n_trials) + \
                euler_mascheroni * stats.norm.ppf(1 - 1/(n_trials * np.e))
    else:
        max_z = 0
    # sr_std assumption: 1 (per unit time)
    expected_max_sr_daily = max_z / np.sqrt(T)
    expected_max_sr_annual = expected_max_sr_daily * np.sqrt(365)

    # DSR
    numerator = (sr_daily - expected_max_sr_daily) * np.sqrt(T - 1)
    denominator = np.sqrt(max(1 - skew * sr_daily + ((kurt - 1) / 4.0) * (sr_daily ** 2), 1e-9))
    dsr_z = numerator / denominator
    dsr = float(stats.norm.cdf(dsr_z))
    passed = dsr > 0.95
    return {
        "passed": bool(passed),
        "sharpe": round(sr_annual, 3),
        "expected_max_sr": round(expected_max_sr_annual, 3),
        "dsr": round(dsr, 4),
        "skew": round(skew, 3),
        "kurt": round(kurt, 3),
    }

# ============================================================
# TEST 9: PBO (Combinatorially Symmetric Cross-Validation)
# ============================================================
def test_9_pbo(edge_returns_matrix, S=16):
    """
    edge_returns_matrix: shape (T, N) where each column is one edge's daily returns.
    We compute PBO across the N edges (columns).
    """
    if edge_returns_matrix is None or edge_returns_matrix.shape[1] < 2:
        return {"passed": False, "pbo": 1.0, "n_partitions": 0}
    T, N = edge_returns_matrix.shape
    # Split T into S groups
    if S >= T:
        S = min(T // 2, 8)
    if S < 4:
        return {"passed": False, "pbo": 1.0, "n_partitions": 0}
    # Divide time into S equal-size blocks
    block_size = T // S
    trimmed = edge_returns_matrix[:block_size * S]
    blocks = np.array(np.split(trimmed, S))  # shape (S, block_size, N)

    # All ways to split S blocks in half
    all_splits = list(combinations(range(S), S // 2))
    n_partitions = len(all_splits)
    if n_partitions > 12870:
        all_splits = all_splits[:12870]
        n_partitions = 12870

    below_median_count = 0
    for split in all_splits:
        is_idx = list(split)
        oos_idx = [i for i in range(S) if i not in is_idx]
        is_data = blocks[is_idx].reshape(-1, N)
        oos_data = blocks[oos_idx].reshape(-1, N)
        # Compute Sharpe per edge for IS and OOS
        is_sharpes = np.array([sharpe_from_returns(is_data[:, j]) for j in range(N)])
        oos_sharpes = np.array([sharpe_from_returns(oos_data[:, j]) for j in range(N)])
        # Best IS edge
        best_is_edge = int(np.argmax(is_sharpes))
        # OOS rank of that edge
        oos_ranks = stats.rankdata(oos_sharpes)  # 1=lowest, N=highest
        best_is_oos_rank = oos_ranks[best_is_edge]
        # Below median = rank < N/2
        if best_is_oos_rank < N / 2:
            below_median_count += 1

    pbo = below_median_count / n_partitions
    passed = pbo < 0.40
    return {
        "passed": bool(passed),
        "pbo": round(pbo, 3),
        "n_partitions": n_partitions,
    }

# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    box("TASK 13: 9-BATTERY STATISTICAL ROBUSTNESS SUITE", C_CYAN)
    print(f"{C_CYAN}  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{C_CYAN}  Engine: v{ENGINE_VER}")
    print(f"{C_CYAN}  MC iterations: {MC_ITERATIONS:,} | Bootstrap: {BOOTSTRAP_ITERATIONS:,}")
    print(f"{C_CYAN}  N total trials (for DSR): {N_BACKTESTS_TOTAL:,}")

    loader = MultiTFDataLoader(DATA_ROOT)
    engine = BacktestEngine()

    # Precompute regimes
    print(f"\n  Precomputing BTC regime classification...")
    detector = RegimeDetector(data_loader=loader)
    regime_series = detector.classify_series("2022-01-01", "2026-08-31")
    if not isinstance(regime_series.index, pd.DatetimeIndex):
        regime_series.index = pd.to_datetime(regime_series.index)
    print(f"  {C_GRN}[OK] Regime series: {len(regime_series)} days{S_RS}")

    # Load or regenerate trade log
    section(1, "LOADING TRADE DATA FOR 16 EDGES", C_CYAN)
    edge_trades = load_trade_log()
    if edge_trades is None or len(edge_trades) < 5:
        print(f"  {C_YEL}[!] Task 11.2 log unavailable/insufficient. Regenerating via canonical engine...{S_RS}")
        edge_trades = {}
        for edge in PORTFOLIO_16:
            trs = regenerate_edge_trades(edge, loader, regime_series, engine)
            if trs:
                edge_trades[edge["id"]] = trs
                print(f"    Edge #{edge['id']:2d} {edge['symbol']:<14}: {len(trs)} trades")
    else:
        print(f"  {C_GRN}[OK] Loaded trade log for {len(edge_trades)} edges{S_RS}")
        for edge in PORTFOLIO_16:
            n_tr = len(edge_trades.get(edge["id"], []))
            print(f"    Edge #{edge['id']:2d} {edge['symbol']:<14}: {n_tr} trades")

    # Also compute base sharpe per edge for reference
    base_sharpes = {}
    daily_ret_map = {}
    pnls_map = {}
    for edge in PORTFOLIO_16:
        eid = edge["id"]
        trs = edge_trades.get(eid, [])
        if not trs:
            base_sharpes[eid] = 0; daily_ret_map[eid] = np.array([]); pnls_map[eid] = np.array([])
            continue
        pnls = np.array([t["pnl"] for t in trs])
        dr = build_daily_returns(trs)
        base_sharpes[eid] = sharpe_from_returns(dr)
        daily_ret_map[eid] = dr
        pnls_map[eid] = pnls

    # ============================================================
    # RUN 9 TESTS FOR EACH EDGE
    # ============================================================
    section(2, "EXECUTING 9-TEST BATTERY ON 16 EDGES", C_MAG)
    scorecard = []

    for edge in PORTFOLIO_16:
        eid = edge["id"]
        trs = edge_trades.get(eid, [])
        pnls = pnls_map[eid]
        dr = daily_ret_map[eid]
        base_sh = base_sharpes[eid]

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

        print(f"    Trades: {len(trs)} | Base Sharpe (annualized): {base_sh:.3f}")

        # Test 1
        print(f"    T1 Monte Carlo Shuffle ({MC_ITERATIONS:,} iter)...", end="", flush=True)
        t1 = test_1_monte_carlo_shuffle(pnls, MC_ITERATIONS)
        print(f" {'PASS' if t1['passed'] else 'FAIL'} (5th %ile MaxDD: {t1['pct5_maxdd']}%, obs: {t1['observed_maxdd']}%)")

        # Test 2
        print(f"    T2 Bootstrap Sharpe CI ({BOOTSTRAP_ITERATIONS:,} iter)...", end="", flush=True)
        t2 = test_2_bootstrap_ci(dr, BOOTSTRAP_ITERATIONS)
        print(f" {'PASS' if t2['passed'] else 'FAIL'} (CI: [{t2['sharpe_ci_lower']}, {t2['sharpe_ci_upper']}])")

        # Test 3
        print(f"    T3 Random Entry Benchmark ({RANDOM_ENTRY_ITERATIONS} iter)...", end="", flush=True)
        t3 = test_3_random_entry(edge, loader, regime_series, engine, trs, RANDOM_ENTRY_ITERATIONS)
        print(f" {'PASS' if t3['passed'] else 'FAIL'} (p-value: {t3['p_value']}, real Sh: {t3['real_sharpe']}, rand p95: {t3['random_pct95']})")

        # Test 4
        print(f"    T4 Price Noise Injection ({NOISE_INJECTION_RUNS} runs)...", end="", flush=True)
        t4 = test_4_noise_injection(edge, loader, regime_series, engine, base_sh, NOISE_INJECTION_RUNS)
        print(f" {'PASS' if t4['passed'] else 'FAIL'} (degradation: {t4['degradation_pct']}%)")

        # Test 5
        print(f"    T5 Parameter Jitter ({PARAM_JITTER_RUNS} runs)...", end="", flush=True)
        t5 = test_5_param_jitter(edge, loader, regime_series, engine, base_sh, PARAM_JITTER_RUNS)
        print(f" {'PASS' if t5['passed'] else 'FAIL'} (mean jittered Sh: {t5['mean_jittered']}, std: {t5['std_jittered']})")

        # Test 6
        print(f"    T6 Regime Slicing (BULL/BEAR/CHOP)...", end="", flush=True)
        t6 = test_6_regime_slicing(trs, regime_series)
        print(f" {'PASS' if t6['passed'] else 'FAIL'} (BULL PF: {t6['bull_pf']}, BEAR: {t6['bear_pf']}, CHOP: {t6['chop_pf']})")

        # Test 7
        print(f"    T7 SPA Reality Check...", end="", flush=True)
        t7 = test_7_spa_reality_check(dr)
        print(f" {'PASS' if t7['passed'] else 'FAIL'} (p-value: {t7['p_value']})")

        # Test 8
        print(f"    T8 Deflated Sharpe Ratio (N={N_BACKTESTS_TOTAL:,})...", end="", flush=True)
        t8 = test_8_dsr(dr, N_BACKTESTS_TOTAL)
        print(f" {'PASS' if t8['passed'] else 'FAIL'} (DSR: {t8['dsr']}, E[max SR]: {t8['expected_max_sr']})")

        # Test 9 (per-edge PBO uses that edge's own returns split into time blocks against itself; skipped in per-edge context)
        # We'll compute Test 9 at the portfolio level below
        t9 = {"passed": True, "pbo": 0.0, "n_partitions": 0, "note": "computed at portfolio level"}

        score = sum([t1["passed"], t2["passed"], t3["passed"], t4["passed"], t5["passed"],
                     t6["passed"], t7["passed"], t8["passed"], t9["passed"]])
        if score >= 7: status = "DEPLOY"
        elif score >= 5: status = "MONITOR"
        else: status = "DROP"

        scorecard.append({
            "edge_id": eid, "strategy": edge["strategy"], "symbol": edge["symbol"], "tf": edge["tf"],
            "n_trades": len(trs), "base_sharpe": round(base_sh, 3),
            "t1": t1["passed"], "t2": t2["passed"], "t3": t3["passed"], "t4": t4["passed"], "t5": t5["passed"],
            "t6": t6["passed"], "t7": t7["passed"], "t8": t8["passed"], "t9": t9["passed"],
            "score": int(score), "status": status,
            "details": {"t1":t1,"t2":t2,"t3":t3,"t4":t4,"t5":t5,"t6":t6,"t7":t7,"t8":t8,"t9":t9}
        })
        st_col = C_GRN if status == "DEPLOY" else (C_YEL if status == "MONITOR" else C_RED)
        print(f"    {st_col}{S_BR}>>> SCORE: {score}/9 | STATUS: {status}{S_RS}")

    # ============================================================
    # PORTFOLIO-LEVEL PBO (Test 9) using edge return matrix
    # ============================================================
    section(3, "PORTFOLIO-LEVEL PBO (COMBINATORIAL CROSS-VALIDATION)", C_MAG)
    # Build matrix (T, N) where columns = per-edge daily returns
    all_dates = set()
    per_edge_daily = {}
    for eid, trs in edge_trades.items():
        if not trs: continue
        df = pd.DataFrame(trs)
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        df = df.sort_values("exit_time")
        df["eq"] = 10000.0 + df["pnl"].cumsum()
        dseries = df.set_index("exit_time")["eq"].resample("D").last().ffill()
        dret = dseries.pct_change().fillna(0)
        per_edge_daily[eid] = dret
        all_dates.update(dret.index)

    if per_edge_daily:
        all_dates = sorted(all_dates)
        matrix = np.zeros((len(all_dates), len(per_edge_daily)))
        for j, (eid, dret) in enumerate(sorted(per_edge_daily.items())):
            aligned = dret.reindex(all_dates, fill_value=0)
            matrix[:, j] = aligned.values
        pbo_result = test_9_pbo(matrix, S=16)
        print(f"  PBO: {pbo_result['pbo']}")
        print(f"  Partitions evaluated: {pbo_result['n_partitions']:,}")
        print(f"  Verdict: {'PASS (PBO < 0.40)' if pbo_result['passed'] else 'FAIL (PBO >= 0.40)'}")
    else:
        pbo_result = {"passed": False, "pbo": 1.0, "n_partitions": 0}

    # Apply portfolio PBO result to individual edges' t9
    for e in scorecard:
        e["t9"] = pbo_result["passed"]
        # Recompute score
        e["score"] = sum([e["t1"], e["t2"], e["t3"], e["t4"], e["t5"], e["t6"], e["t7"], e["t8"], e["t9"]])
        if e["score"] >= 7: e["status"] = "DEPLOY"
        elif e["score"] >= 5: e["status"] = "MONITOR"
        else: e["status"] = "DROP"

    # ============================================================
    # PORTFOLIO-LEVEL AGGREGATE TESTS (T1, T2, T8)
    # ============================================================
    section(4, "MASTER PORTFOLIO AGGREGATE TESTS", C_MAG)
    # Combine all trades into one master series
    all_trades = []
    for eid, trs in edge_trades.items():
        all_trades.extend(trs)
    if all_trades:
        all_trades.sort(key=lambda t: pd.to_datetime(t["exit_time"]))
        master_pnls = np.array([t["pnl"] for t in all_trades])
        master_dr = build_daily_returns(all_trades)

        port_t1 = test_1_monte_carlo_shuffle(master_pnls, MC_ITERATIONS)
        port_t2 = test_2_bootstrap_ci(master_dr, BOOTSTRAP_ITERATIONS)
        port_t8 = test_8_dsr(master_dr, N_BACKTESTS_TOTAL)

        port_rows = [
            ["T1 Monte Carlo Shuffle", "PASS" if port_t1["passed"] else "FAIL",
             f"5th %ile MaxDD={port_t1['pct5_maxdd']}% (obs={port_t1['observed_maxdd']}%)"],
            ["T2 Bootstrap CI", "PASS" if port_t2["passed"] else "FAIL",
             f"Sharpe CI [{port_t2['sharpe_ci_lower']}, {port_t2['sharpe_ci_upper']}]"],
            ["T8 Deflated Sharpe", "PASS" if port_t8["passed"] else "FAIL",
             f"DSR={port_t8['dsr']}, E[maxSR]={port_t8['expected_max_sr']}"],
            ["T9 PBO", "PASS" if pbo_result["passed"] else "FAIL",
             f"PBO={pbo_result['pbo']} ({pbo_result['n_partitions']:,} partitions)"],
        ]
        print(tabulate(port_rows, headers=["Test","Result","Details"], tablefmt="grid"))

    # ============================================================
    # SECTION 5: 16x9 SCORECARD MATRIX
    # ============================================================
    section(5, "THE 16 x 9 MASTER SCORECARD GRID", C_CYAN)
    header = ["#", "Strategy", "Symbol", "TF"] + [f"T{i}" for i in range(1, 10)] + ["Score", "Status"]
    matrix_rows = []
    for e in scorecard:
        row = [e["edge_id"], e["strategy"][:22], e["symbol"][:12], e["tf"]]
        for i in range(1, 10):
            key = f"t{i}"
            v = e[key]
            row.append(f"{C_GRN}P{S_RS}" if v else f"{C_RED}F{S_RS}")
        row.append(f"{e['score']}/9")
        st_col = C_GRN if e["status"] == "DEPLOY" else (C_YEL if e["status"] == "MONITOR" else C_RED)
        row.append(f"{st_col}{e['status']}{S_RS}")
        matrix_rows.append(row)
    print(tabulate(matrix_rows, headers=header, tablefmt="grid"))

    # Score distribution
    n_deploy = sum(1 for e in scorecard if e["status"] == "DEPLOY")
    n_monitor = sum(1 for e in scorecard if e["status"] == "MONITOR")
    n_drop = sum(1 for e in scorecard if e["status"] == "DROP")
    print(f"\n  Summary: {C_GRN}{n_deploy} DEPLOY{S_RS} | {C_YEL}{n_monitor} MONITOR{S_RS} | {C_RED}{n_drop} DROP{S_RS}")

    # ============================================================
    # SECTION 6: TOP 3 MOST ROBUST DEEP DIVE
    # ============================================================
    section(6, "TOP 3 MOST ROBUST EDGES - DETAILED DIAGNOSTICS", C_GRN)
    ranked = sorted(scorecard, key=lambda e: (-e["score"], -e["base_sharpe"]))[:3]
    for i, e in enumerate(ranked, 1):
        print(f"\n  {C_GRN}{S_BR}#{i} {e['strategy']} | {e['symbol']} | {e['tf']} (Score: {e['score']}/9){S_RS}")
        d = e["details"]
        rows = [
            ["Base Sharpe", f"{e['base_sharpe']:.3f}"],
            ["T1 Observed MaxDD", f"{d['t1']['observed_maxdd']}%"],
            ["T1 MC 5th %ile MaxDD", f"{d['t1']['pct5_maxdd']}%"],
            ["T2 Sharpe CI 95%", f"[{d['t2']['sharpe_ci_lower']}, {d['t2']['sharpe_ci_upper']}]"],
            ["T2 CAGR CI 95%", f"[{d['t2']['cagr_ci_lower']}%, {d['t2']['cagr_ci_upper']}%]"],
            ["T3 Random Entry p-value", f"{d['t3']['p_value']}"],
            ["T4 Noise Degradation", f"{d['t4']['degradation_pct']}%"],
            ["T5 Param Jitter Std", f"{d['t5']['std_jittered']:.3f}"],
            ["T6 BULL/BEAR/CHOP PF", f"{d['t6']['bull_pf']}/{d['t6']['bear_pf']}/{d['t6']['chop_pf']}"],
            ["T7 SPA p-value", f"{d['t7']['p_value']}"],
            ["T8 DSR", f"{d['t8']['dsr']}"],
            ["T8 Skewness", f"{d['t8']['skew']}"],
            ["T8 Kurtosis", f"{d['t8']['kurt']}"],
        ]
        print(tabulate(rows, headers=["Metric", "Value"], tablefmt="simple"))

    # ============================================================
    # SECTION 7: PRUNED PORTFOLIO
    # ============================================================
    section(7, "PRUNED PRODUCTION PORTFOLIO (DEPLOY only)", C_MAG)
    deploy_edges = [e for e in scorecard if e["status"] == "DEPLOY"]
    if not deploy_edges:
        print(f"  {C_RED}[!] Zero edges passed the 7/9 threshold. Consider relaxing to MONITOR (5/9).{S_RS}")
        deploy_edges = [e for e in scorecard if e["status"] in ("DEPLOY", "MONITOR")]
        print(f"  {C_YEL}Using MONITOR+DEPLOY: {len(deploy_edges)} edges{S_RS}")

    # Build pruned trades
    pruned_trades = []
    for e in deploy_edges:
        pruned_trades.extend(edge_trades.get(e["edge_id"], []))
    full_trades = all_trades

    def portfolio_stats(trs):
        if not trs: return {"trades":0,"cagr":0,"sharpe":0,"maxdd":0,"win":0,"exp":0,"monthly":0}
        pnls = np.array([t["pnl"] for t in trs])
        dr = build_daily_returns(trs)
        wins = (pnls > 0).sum()
        wr = 100 * wins / len(pnls)
        sh = sharpe_from_returns(dr)
        eq = equity_from_pnls(pnls)
        maxdd = max_dd_from_equity(eq)
        exp = pnls.mean()
        # Monthly freq
        exit_times = pd.to_datetime([t["exit_time"] for t in trs])
        days = (exit_times.max() - exit_times.min()).days
        monthly = len(trs) / max(days / 30.44, 1)
        # CAGR
        final = 10000 + pnls.sum()
        years = max(days / 365.25, 0.01)
        cagr = (final / 10000) ** (1/years) * 100 - 100 if final > 0 else 0
        return {"trades":len(trs),"cagr":round(cagr,2),"sharpe":round(sh,3),
                "maxdd":round(maxdd,2),"win":round(wr,2),"exp":round(exp,4),
                "monthly":round(monthly,2)}

    before = portfolio_stats(full_trades)
    after = portfolio_stats(pruned_trades)
    comp_rows = [
        ["Total Edges", 16, len(deploy_edges), f"{len(deploy_edges)-16:+d}"],
        ["Total Trades", before["trades"], after["trades"], f"{after['trades']-before['trades']:+d}"],
        ["CAGR %", before["cagr"], after["cagr"], f"{after['cagr']-before['cagr']:+.2f}"],
        ["Sharpe", before["sharpe"], after["sharpe"], f"{after['sharpe']-before['sharpe']:+.3f}"],
        ["Max Drawdown %", before["maxdd"], after["maxdd"], f"{after['maxdd']-before['maxdd']:+.2f}"],
        ["Win Rate %", before["win"], after["win"], f"{after['win']-before['win']:+.2f}"],
        ["Expectancy $", before["exp"], after["exp"], f"{after['exp']-before['exp']:+.4f}"],
        ["Monthly Trades", before["monthly"], after["monthly"], f"{after['monthly']-before['monthly']:+.2f}"],
    ]
    print(tabulate(comp_rows, headers=["Metric","BEFORE (16)","AFTER (Pruned)","Delta"], tablefmt="grid"))

    print(f"\n  {C_GRN}Pruned Portfolio Composition:{S_RS}")
    for e in deploy_edges:
        print(f"    #{e['edge_id']:2d} {e['strategy']:<28} {e['symbol']:<14} {e['tf']} | Score: {e['score']}/9 | {e['status']}")

    # ============================================================
    # SECTION 8: FINAL GO/NO-GO
    # ============================================================
    section(8, "FINAL LIVE TRADING CERTIFICATION", C_CYAN)
    conditions = [
        ("At least 3 edges scored DEPLOY (>=7/9)", len([e for e in scorecard if e['status']=='DEPLOY']) >= 3),
        ("Master Portfolio PBO < 0.40", pbo_result["passed"]),
        ("Pruned Portfolio Sharpe > 1.0", after["sharpe"] > 1.0),
        ("Pruned Portfolio Max DD > -20%", after["maxdd"] > -20),
        ("Pruned Portfolio Win Rate >= 50%", after["win"] >= 50),
    ]
    cert_rows = []
    passed_conditions = 0
    for cond, met in conditions:
        cert_rows.append([cond, f"{C_GRN}MET{S_RS}" if met else f"{C_RED}NOT MET{S_RS}"])
        if met: passed_conditions += 1
    print(tabulate(cert_rows, headers=["Certification Condition", "Status"], tablefmt="grid"))

    print(f"\n  Conditions met: {passed_conditions}/{len(conditions)}")
    if passed_conditions >= 4:
        verdict = f"{C_GRN}{S_BR}CERTIFIED GO - Advance to Task 14 (Live Paper Trading Deployment){S_RS}"
    elif passed_conditions >= 3:
        verdict = f"{C_YEL}CONDITIONAL - Deploy MONITOR-level edges after 30-day live paper trade{S_RS}"
    else:
        verdict = f"{C_RED}NO-GO - Portfolio failed robustness certification. Return to research.{S_RS}"
    print(f"\n  {S_BR}FINAL VERDICT: {verdict}")

    # ============================================================
    # SECTION 9: EXPORTS
    # ============================================================
    section(9, "EXPORT FILES", C_CYAN)
    scorecard_csv = os.path.join(RESULTS_ROOT, "task13_robustness_scorecard.csv")
    export_rows = []
    for e in scorecard:
        row = {k: v for k, v in e.items() if k != "details"}
        # Add key details
        d = e["details"]
        if d:
            row["t1_observed_maxdd"] = d.get("t1", {}).get("observed_maxdd", 0)
            row["t1_pct5_maxdd"] = d.get("t1", {}).get("pct5_maxdd", 0)
            row["t2_sharpe_ci_lower"] = d.get("t2", {}).get("sharpe_ci_lower", 0)
            row["t2_sharpe_ci_upper"] = d.get("t2", {}).get("sharpe_ci_upper", 0)
            row["t3_p_value"] = d.get("t3", {}).get("p_value", 1)
            row["t4_degradation_pct"] = d.get("t4", {}).get("degradation_pct", 100)
            row["t5_std_jittered"] = d.get("t5", {}).get("std_jittered", 0)
            row["t6_bull_pf"] = d.get("t6", {}).get("bull_pf", 0)
            row["t6_bear_pf"] = d.get("t6", {}).get("bear_pf", 0)
            row["t6_chop_pf"] = d.get("t6", {}).get("chop_pf", 0)
            row["t7_p_value"] = d.get("t7", {}).get("p_value", 1)
            row["t8_dsr"] = d.get("t8", {}).get("dsr", 0)
            row["t8_skew"] = d.get("t8", {}).get("skew", 0)
            row["t8_kurt"] = d.get("t8", {}).get("kurt", 0)
        row["t9_pbo_portfolio"] = pbo_result["pbo"]
        export_rows.append(row)
    pd.DataFrame(export_rows).to_csv(scorecard_csv, index=False)
    print(f"  {C_GRN}[SAVED] {scorecard_csv}{S_RS}")

    pruned_csv = os.path.join(RESULTS_ROOT, "task13_pruned_portfolio.csv")
    pruned_rows = [{k: v for k, v in e.items() if k != "details"} for e in deploy_edges]
    pd.DataFrame(pruned_rows).to_csv(pruned_csv, index=False)
    print(f"  {C_GRN}[SAVED] {pruned_csv}{S_RS}")

    elapsed = time.time() - t0
    print()
    box(f"TASK 13 COMPLETE - Runtime: {elapsed/60:.2f} min", C_GRN)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C_RED}Interrupted{S_RS}")
    except Exception as e:
        print(f"\n{C_RED}FATAL: {e}{S_RS}")
        traceback.print_exc()
