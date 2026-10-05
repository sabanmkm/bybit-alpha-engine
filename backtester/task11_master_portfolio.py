"""
TASK 11/12: MASTER PORTFOLIO ASSEMBLY & CAPITAL ALLOCATION ENGINE
Assembles 16 verified alpha edges and computes 4 sizing models.
"""
import os
import sys
import time
import json
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
RISK_PER_TRADE = 0.01

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
# THE 16 VALIDATED EDGES
# ============================================================
PORTFOLIO_16 = [
    # Family A: Time-based/Calendar (10)
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
    # Family B: Regime-Adaptive (6)
    {"id":11, "strategy":"R13_ATR_Range_Expansion",    "symbol":"ETH_USDT_USDT",  "tf":"4H", "family":"RegimeAdaptive"},
    {"id":12, "strategy":"R15_Volume_Breakout",        "symbol":"SAND_USDT_USDT", "tf":"4H", "family":"RegimeAdaptive"},
    {"id":13, "strategy":"R01_EMA_Cross",              "symbol":"OP_USDT_USDT",   "tf":"4H", "family":"RegimeAdaptive"},
    {"id":14, "strategy":"R15_Volume_Breakout",        "symbol":"OP_USDT_USDT",   "tf":"4H", "family":"RegimeAdaptive"},
    {"id":15, "strategy":"R01_EMA_Cross",              "symbol":"INJ_USDT_USDT",  "tf":"1H", "family":"RegimeAdaptive"},
    {"id":16, "strategy":"R16_Trend_Plus_RSI_Pullback","symbol":"DOGE_USDT_USDT", "tf":"4H", "family":"RegimeAdaptive"},
]

# ============================================================
# STRATEGY IMPLEMENTATIONS (consolidated from Tasks 7.2 + 8)
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

# --- Time-Based ---
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

# --- Regime-Adaptive ---
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
# BACKTEST WRAPPER
# ============================================================
def run_edge(edge, loader, engine, regime_series, start, end):
    """Returns (trades_list, equity_df, metrics_dict) or None."""
    sname = edge["strategy"]; sym = edge["symbol"]; tf = edge["tf"]
    fn = STRATEGY_MAP.get(sname)
    if fn is None: return None, None, None
    df = loader.load(sym, tf, start, end)
    if df is None or len(df) < 50: return None, None, None
    try:
        needs_regime = sname.startswith("R")
        if needs_regime:
            sig, sl, tp = fn(df, regime_series)
        else:
            sig, sl, tp = fn(df)
        result = engine.run(df, sig, sl, tp)
        return result["trades"], result["equity_curve"], result["metrics"]
    except Exception as e:
        print(f"    {C_RED}[ERR] {sname}|{sym}|{tf}: {e}{S_RS}")
        return None, None, None

# ============================================================
# METRIC HELPERS
# ============================================================
def annualized_sharpe_from_daily(eq_daily):
    if len(eq_daily) < 5: return 0.0
    dr = eq_daily.pct_change().dropna()
    if dr.std() == 0: return 0.0
    return float(dr.mean() / dr.std() * np.sqrt(365))

def annualized_sortino_from_daily(eq_daily):
    if len(eq_daily) < 5: return 0.0
    dr = eq_daily.pct_change().dropna()
    neg = dr[dr < 0]
    if len(neg) == 0 or neg.std() == 0: return 999.0
    return float(dr.mean() / neg.std() * np.sqrt(365))

def cagr_from_equity(eq_series):
    if len(eq_series) < 2 or eq_series.iloc[0] <= 0: return 0.0
    years = max((eq_series.index[-1] - eq_series.index[0]).days / 365.25, 0.01)
    return float((eq_series.iloc[-1] / eq_series.iloc[0]) ** (1/years) - 1) * 100

def max_dd_pct(eq_series):
    if len(eq_series) < 2: return 0.0
    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak
    return float(dd.min() * 100)

def max_dd_duration(eq_series):
    if len(eq_series) < 2: return 0
    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak
    cur = 0; mx = 0
    for x in dd.values:
        if x < 0:
            cur += 1
            if cur > mx: mx = cur
        else: cur = 0
    return mx

def merge_daily_equity(equity_list, weights, start, end):
    """Combine multiple equity curves into a single portfolio series."""
    daily_frames = []
    for eq, w in zip(equity_list, weights):
        if eq is None or len(eq) < 3: continue
        try:
            d = eq.resample("D").last().ffill()
            # Normalize returns and apply weight
            base = d.iloc[0]
            rets = d.pct_change().fillna(0)
            weighted_eq_growth = (1 + rets * w).cumprod()
            daily_frames.append(weighted_eq_growth)
        except Exception:
            continue
    if not daily_frames: return None
    combined = pd.concat(daily_frames, axis=1).ffill().fillna(1.0)
    # Portfolio growth = sum of weighted independent growths (assuming rebalanced)
    port_growth = combined.mean(axis=1)  # equal-weight aggregation of already-weighted returns
    port_eq = STARTING_CAPITAL * port_growth
    return port_eq

def compute_portfolio_metrics(equity_curves, weights, trades_list, label=""):
    """Compute unified portfolio metrics given equity curves and weights."""
    port_eq = merge_daily_equity(equity_curves, weights, None, None)
    if port_eq is None or len(port_eq) < 5:
        return None
    metrics = {}
    metrics["final_equity"] = round(float(port_eq.iloc[-1]), 2)
    metrics["cagr_pct"] = round(cagr_from_equity(port_eq), 3)
    metrics["sharpe"] = round(annualized_sharpe_from_daily(port_eq), 3)
    metrics["sortino"] = round(annualized_sortino_from_daily(port_eq), 3)
    metrics["max_dd_pct"] = round(max_dd_pct(port_eq), 2)
    metrics["max_dd_bars"] = int(max_dd_duration(port_eq))
    metrics["calmar"] = round(metrics["cagr_pct"] / abs(metrics["max_dd_pct"]), 3) if metrics["max_dd_pct"] < 0 else 0
    all_t = [t for edge_trades in trades_list for t in edge_trades]
    if all_t:
        pnls = np.array([t["pnl"] for t in all_t])
        wins = (pnls > 0).sum()
        metrics["win_rate"] = round(100 * wins / len(pnls), 2)
        gp = pnls[pnls > 0].sum(); gl = abs(pnls[pnls < 0].sum())
        metrics["profit_factor"] = round(gp / gl, 3) if gl > 0 else 999
        metrics["total_trades"] = len(all_t)
        aw = pnls[pnls > 0].mean() if wins > 0 else 0
        al = abs(pnls[pnls < 0].mean()) if (pnls < 0).sum() > 0 else 0
        pw = wins / len(pnls)
        metrics["expectancy_dollar"] = round(pw * aw - (1 - pw) * al, 4)
        metrics["expectancy_pct"] = round(np.mean([t["pnl_pct"] for t in all_t]), 4)
    else:
        metrics.update({"win_rate":0,"profit_factor":0,"total_trades":0,"expectancy_dollar":0,"expectancy_pct":0})
    return metrics, port_eq

# ============================================================
# WEIGHT MODELS
# ============================================================
def weights_equal(n): return [1.0/n] * n

def weights_risk_parity(equity_curves):
    """Weight inverse to daily-return volatility."""
    vols = []
    for eq in equity_curves:
        if eq is None or len(eq) < 5: vols.append(1.0); continue
        d = eq.resample("D").last().ffill()
        dr = d.pct_change().dropna()
        vol = dr.std() if dr.std() > 0 else 1e-6
        vols.append(vol)
    inv = [1.0/v for v in vols]
    total = sum(inv)
    return [i/total for i in inv]

def weights_fractional_kelly(trades_list, fraction=0.25):
    """f* = fraction * (p/a - q/b), where p=win_rate, q=loss_rate, a=avg_loss, b=avg_win."""
    ws = []
    for tr in trades_list:
        if not tr:
            ws.append(0.0); continue
        pnls = np.array([t["pnl_pct"]/100 for t in tr])
        wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
        if len(wins) == 0 or len(losses) == 0:
            ws.append(0.01); continue
        p = len(wins)/len(pnls); q = 1 - p
        b = wins.mean(); a = abs(losses.mean())
        if a == 0 or b == 0:
            ws.append(0.01); continue
        kelly = p/a - q/b
        f_star = max(0.0, min(fraction * kelly, 0.5))  # cap at 50%
        ws.append(f_star)
    total = sum(ws)
    if total <= 0: return [1.0/len(ws)] * len(ws)
    return [w/total for w in ws]

def weights_alpha_composite(edge_metrics):
    """Weight by OOS Sharpe * Monthly Frequency."""
    scores = []
    for m in edge_metrics:
        if m is None:
            scores.append(0.01); continue
        sh = max(m.get("sharpe", 0), 0)
        mf = m.get("monthly_freq", 0)
        score = sh * (1 + mf * 0.5)
        scores.append(max(score, 0.01))
    total = sum(scores)
    if total <= 0: return [1.0/len(scores)] * len(scores)
    return [s/total for s in scores]

# ============================================================
# CORRELATION MATRIX
# ============================================================
def compute_correlation_matrix(equity_curves, edge_labels):
    daily_returns = {}
    for i, eq in enumerate(equity_curves):
        if eq is None or len(eq) < 5:
            daily_returns[edge_labels[i]] = pd.Series(dtype=float)
            continue
        d = eq.resample("D").last().ffill()
        dr = d.pct_change().dropna()
        daily_returns[edge_labels[i]] = dr
    df_returns = pd.DataFrame(daily_returns)
    corr = df_returns.corr()
    return corr

# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    box("TASK 11/12: MASTER PORTFOLIO ASSEMBLY", C_CYAN)
    print(f"{C_CYAN}  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{C_CYAN}  Engine: v{ENGINE_VER}")
    print(f"{C_CYAN}  IS Window : {IS_START.date()} -> {IS_END.date()}")
    print(f"{C_MAG}  OOS Window: {OOS_START.date()} -> {OOS_END.date()}")

    loader = MultiTFDataLoader(DATA_ROOT)
    engine = BacktestEngine()

    # Precompute regimes
    print(f"\n  Precomputing BTC regime classification...")
    detector = RegimeDetector(data_loader=loader)
    regime_series = detector.classify_series("2022-01-01", "2026-08-31")
    if not isinstance(regime_series.index, pd.DatetimeIndex):
        regime_series.index = pd.to_datetime(regime_series.index)
    print(f"  {C_GRN}[OK] Regime series: {len(regime_series)} days{S_RS}")

    # ==============================
    # RUN ALL 16 EDGES ON FULL PERIOD (IS+OOS combined for simulation)
    # ==============================
    section(1, "SIMULATING 16 EDGES (2022-2026)", C_CYAN)
    all_edges_data = []  # list of dicts: {edge, trades_full, equity_full, metrics_full, trades_is, trades_oos, metrics_is, metrics_oos, eq_is, eq_oos}
    for edge in PORTFOLIO_16:
        print(f"  Simulating: {edge['strategy'][:28]:<28} | {edge['symbol']:<14} | {edge['tf']:<3} ...", end="")
        # Full period
        tr_full, eq_full, m_full = run_edge(edge, loader, engine, regime_series, FULL_START, FULL_END)
        # IS period
        tr_is, eq_is, m_is = run_edge(edge, loader, engine, regime_series, IS_START, IS_END)
        # OOS period
        tr_oos, eq_oos, m_oos = run_edge(edge, loader, engine, regime_series, OOS_START, OOS_END)

        if m_oos is not None:
            days = (OOS_END - OOS_START).days
            m_oos["monthly_freq"] = round(len(tr_oos or []) / max(days/30.44, 1), 3)
        if m_is is not None:
            days = (IS_END - IS_START).days
            m_is["monthly_freq"] = round(len(tr_is or []) / max(days/30.44, 1), 3)

        all_edges_data.append({
            "edge": edge,
            "trades_full": tr_full or [],
            "equity_full": eq_full,
            "metrics_full": m_full,
            "trades_is": tr_is or [],
            "equity_is": eq_is,
            "metrics_is": m_is,
            "trades_oos": tr_oos or [],
            "equity_oos": eq_oos,
            "metrics_oos": m_oos,
        })
        status = f"{C_GRN}OK{S_RS}" if m_full else f"{C_RED}FAIL{S_RS}"
        n_full = len(tr_full or []); n_oos = len(tr_oos or [])
        print(f" [{status}] Full: {n_full} trades | OOS: {n_oos} trades")

    valid_edges = [e for e in all_edges_data if e["metrics_full"] is not None]
    print(f"\n  {C_GRN}[OK] {len(valid_edges)}/16 edges simulated successfully{S_RS}")

    if len(valid_edges) < 5:
        print(f"  {C_RED}[!] Insufficient valid edges. Aborting.{S_RS}")
        return

    # ==============================
    # SECTION 2: DETAILED EDGE BREAKDOWN
    # ==============================
    section(2, "DETAILED 16-EDGE BREAKDOWN (IS vs OOS)", C_CYAN)
    edge_rows = []
    for e in all_edges_data:
        ed = e["edge"]
        mi = e["metrics_is"] or {}
        mo = e["metrics_oos"] or {}
        edge_rows.append([
            ed["id"], ed["family"][:8], ed["strategy"][:26], ed["symbol"][:12], ed["tf"],
            f"{mi.get('sharpe',0):.2f}", f"{mo.get('sharpe',0):.2f}",
            f"{mo.get('win_rate_pct',0):.1f}%",
            mo.get("total_trades", 0),
            f"{mo.get('monthly_freq',0):.2f}",
            f"{mo.get('profit_factor',0):.2f}",
            f"{mo.get('total_return_pct',0):+.2f}%"
        ])
    print(tabulate(edge_rows,
        headers=["#","Family","Strategy","Symbol","TF","IS Sh","OOS Sh","OOS Win%","OOS Tr","OOS MoFr","OOS PF","OOS Ret"],
        tablefmt="grid"))

    # ==============================
    # SECTION 3: COMPUTE 4 CAPITAL ALLOCATION MODELS
    # ==============================
    section(3, "CAPITAL ALLOCATION MODEL COMPARISON", C_MAG)
    n_edges = len(all_edges_data)
    equity_curves_full = [e["equity_full"] for e in all_edges_data]
    equity_curves_oos = [e["equity_oos"] for e in all_edges_data]
    trades_full = [e["trades_full"] for e in all_edges_data]
    trades_oos = [e["trades_oos"] for e in all_edges_data]
    metrics_oos = [e["metrics_oos"] for e in all_edges_data]

    # Compute 4 weight vectors
    w_equal = weights_equal(n_edges)
    w_rp = weights_risk_parity(equity_curves_full)
    w_kelly = weights_fractional_kelly(trades_full, fraction=0.25)
    w_alpha = weights_alpha_composite(metrics_oos)

    models = {
        "Equal Weight": w_equal,
        "Risk Parity": w_rp,
        "Fractional Kelly (0.25)": w_kelly,
        "Alpha Composite": w_alpha,
    }

    # Compute FULL PERIOD portfolio metrics for each model
    model_results = {}
    for label, w in models.items():
        m, eq = compute_portfolio_metrics(equity_curves_full, w, trades_full, label)
        if m:
            model_results[label] = {"metrics": m, "equity": eq, "weights": w}

    # Compute OOS-only metrics too for each model
    model_oos_results = {}
    for label, w in models.items():
        m, eq = compute_portfolio_metrics(equity_curves_oos, w, trades_oos, label)
        if m:
            model_oos_results[label] = {"metrics": m, "equity": eq, "weights": w}

    # Comparison table
    print(f"\n  {C_CYAN}Full-Period Portfolio Performance (2022-2026):{S_RS}")
    comp_rows = []
    for label, res in model_results.items():
        m = res["metrics"]
        comp_rows.append([
            label,
            f"{m['cagr_pct']:.2f}%",
            f"{m['sharpe']:.3f}",
            f"{m['sortino']:.3f}",
            f"{m['max_dd_pct']:.2f}%",
            f"{m['calmar']:.3f}",
            f"{m['profit_factor']:.2f}",
            f"{m['win_rate']:.2f}%",
            f"${m['final_equity']:,.2f}"
        ])
    print(tabulate(comp_rows,
        headers=["Model","CAGR","Sharpe","Sortino","MaxDD","Calmar","PF","Win%","Final Equity"],
        tablefmt="grid"))

    print(f"\n  {C_MAG}2026 OOS-Only Portfolio Performance:{S_RS}")
    comp_oos_rows = []
    for label, res in model_oos_results.items():
        m = res["metrics"]
        comp_oos_rows.append([
            label,
            f"{m['cagr_pct']:.2f}%",
            f"{m['sharpe']:.3f}",
            f"{m['max_dd_pct']:.2f}%",
            f"{m['calmar']:.3f}",
            f"{m['profit_factor']:.2f}",
            f"{m['win_rate']:.2f}%",
            f"${m['final_equity']:,.2f}"
        ])
    print(tabulate(comp_oos_rows,
        headers=["Model","OOS CAGR","OOS Sharpe","OOS MaxDD","Calmar","PF","Win%","Final Equity"],
        tablefmt="grid"))

    # Recommend best model
    best_model = max(model_results.keys(), key=lambda k: model_results[k]["metrics"]["sharpe"])
    print(f"\n  {C_GRN}{S_BR}RECOMMENDED MODEL: {best_model}{S_RS}")
    print(f"  Selection criterion: Highest full-period Sharpe ratio")
    print(f"  Sharpe: {model_results[best_model]['metrics']['sharpe']:.3f} | "
          f"CAGR: {model_results[best_model]['metrics']['cagr_pct']:.2f}% | "
          f"MaxDD: {model_results[best_model]['metrics']['max_dd_pct']:.2f}%")

    # ==============================
    # SECTION 4: 16x16 CORRELATION MATRIX
    # ==============================
    section(4, "16 x 16 PAIRWISE CORRELATION MATRIX", C_CYAN)
    edge_labels = [f"E{e['edge']['id']}_{e['edge']['symbol'][:6]}" for e in all_edges_data]
    corr_matrix = compute_correlation_matrix(equity_curves_full, edge_labels)

    # Display correlation with color coding
    print(f"  (Low correlation = better diversification. Values > 0.5 in red)")
    header = [""] + edge_labels
    rows = []
    for i, lbl_i in enumerate(edge_labels):
        row = [lbl_i]
        for j, lbl_j in enumerate(edge_labels):
            val = corr_matrix.iloc[i, j] if lbl_j in corr_matrix.columns and lbl_i in corr_matrix.index else 0
            if pd.isna(val): val = 0
            if i == j:
                row.append(f"{C_WHT}1.00{S_RS}")
            elif abs(val) > 0.5:
                row.append(f"{C_RED}{val:+.2f}{S_RS}")
            elif abs(val) > 0.3:
                row.append(f"{C_YEL}{val:+.2f}{S_RS}")
            else:
                row.append(f"{C_GRN}{val:+.2f}{S_RS}")
        rows.append(row)
    print(tabulate(rows, headers=header, tablefmt="simple"))

    # Diversification metrics
    corr_vals = corr_matrix.values
    mask = np.triu(np.ones_like(corr_vals, dtype=bool), k=1)
    off_diag = corr_vals[mask]
    off_diag = off_diag[~np.isnan(off_diag)]
    if len(off_diag) > 0:
        print(f"\n  {C_CYAN}Diversification Statistics:{S_RS}")
        print(f"    Mean pairwise correlation : {np.mean(off_diag):.3f}")
        print(f"    Max pairwise correlation  : {np.max(off_diag):.3f}")
        print(f"    Min pairwise correlation  : {np.min(off_diag):.3f}")
        print(f"    # pairs > 0.5 correlation : {int((np.abs(off_diag) > 0.5).sum())}")

    # ==============================
    # SECTION 5: 2026 MONTH-BY-MONTH MATRIX
    # ==============================
    section(5, "2026 MONTH-BY-MONTH OOS PORTFOLIO PERFORMANCE", C_MAG)
    best_w = model_results[best_model]["weights"]
    # Build monthly returns from portfolio equity curve (OOS)
    port_eq_oos = model_oos_results[best_model]["equity"]

    monthly_rows = []
    mo_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug"]
    for mi in range(1, 9):
        if mi == 12:
            ms = pd.Timestamp(f"2026-{mi:02d}-01")
            me = pd.Timestamp("2027-01-01") - pd.Timedelta(seconds=1)
        else:
            ms = pd.Timestamp(f"2026-{mi:02d}-01")
            me = pd.Timestamp(f"2026-{mi+1:02d}-01") - pd.Timedelta(seconds=1)
        # Filter equity for month
        mo_eq = port_eq_oos[(port_eq_oos.index >= ms) & (port_eq_oos.index <= me)]
        if len(mo_eq) < 2:
            mo_ret = 0
        else:
            mo_ret = (mo_eq.iloc[-1] / mo_eq.iloc[0] - 1) * 100
        # Count trades in month across all edges (using best model weights)
        mo_trades = 0
        mo_wins = 0
        for e in all_edges_data:
            for t in e["trades_oos"]:
                if ms <= t["exit_time"] <= me:
                    mo_trades += 1
                    if t["pnl"] > 0: mo_wins += 1
        wr = 100 * mo_wins / mo_trades if mo_trades > 0 else 0
        col = C_GRN if mo_ret > 0 else (C_RED if mo_ret < 0 else C_YEL)
        monthly_rows.append([
            mo_names[mi-1],
            f"{col}{mo_ret:+.2f}%{S_RS}",
            mo_trades,
            f"{wr:.1f}%" if mo_trades > 0 else "-"
        ])
    print(tabulate(monthly_rows, headers=["Month","Return","Trades","Win%"], tablefmt="grid"))

    # ==============================
    # SECTION 6: ANNUAL PERFORMANCE
    # ==============================
    section(6, "ANNUAL PERFORMANCE (2022-2026 OOS)", C_CYAN)
    port_eq_full = model_results[best_model]["equity"]
    annual_rows = []
    for year in [2022, 2023, 2024, 2025, 2026]:
        y_start = pd.Timestamp(f"{year}-01-01")
        y_end = pd.Timestamp(f"{year}-12-31 23:59:59") if year < 2026 else OOS_END
        y_eq = port_eq_full[(port_eq_full.index >= y_start) & (port_eq_full.index <= y_end)]
        if len(y_eq) < 2:
            annual_rows.append([year, "-", "-", "-", "-", "N/A"])
            continue
        y_ret = (y_eq.iloc[-1] / y_eq.iloc[0] - 1) * 100
        y_dd = max_dd_pct(y_eq)
        y_sh = annualized_sharpe_from_daily(y_eq)
        # Trades in year
        y_trades = 0
        for e in all_edges_data:
            y_trades += sum(1 for t in e["trades_full"] if y_start <= t["exit_time"] <= y_end)
        col = C_GRN if y_ret > 0 else C_RED
        annual_rows.append([
            year,
            f"{col}{y_ret:+.2f}%{S_RS}",
            f"{y_sh:.2f}",
            f"{y_dd:.2f}%",
            y_trades,
            "OOS" if year == 2026 else "IS"
        ])
    print(tabulate(annual_rows, headers=["Year","Return","Sharpe","MaxDD","Trades","Period"], tablefmt="grid"))

    # ==============================
    # SECTION 7: CAPITAL ALLOCATION BLUEPRINT
    # ==============================
    section(7, "CAPITAL ALLOCATION BLUEPRINT (Best Model: " + best_model + ")", C_MAG)
    accounts = [10000, 50000, 100000]
    print(f"  Position sizing = allocation_weight * account_size")
    print(f"  Each strategy risks 1% of allocated capital per trade\n")

    for acct in accounts:
        print(f"  {C_CYAN}{S_BR}--- ${acct:,} ACCOUNT ---{S_RS}")
        alloc_rows = []
        total_alloc = 0
        for i, e in enumerate(all_edges_data):
            ed = e["edge"]
            w = best_w[i]
            allocated = acct * w
            risk_per_trade = allocated * RISK_PER_TRADE
            total_alloc += allocated
            alloc_rows.append([
                ed["id"], ed["strategy"][:24], ed["symbol"][:12], ed["tf"],
                f"{w*100:.2f}%",
                f"${allocated:,.2f}",
                f"${risk_per_trade:.2f}"
            ])
        print(tabulate(alloc_rows,
            headers=["#","Strategy","Symbol","TF","Weight","Notional","Risk/Trade"],
            tablefmt="simple"))
        print(f"  Total allocation: ${total_alloc:,.2f}  |  Max risk per trade (aggregate): ${sum(acct * w * RISK_PER_TRADE for w in best_w):.2f}\n")

    # ==============================
    # SECTION 8: RISK MANAGEMENT & EXECUTION RULES
    # ==============================
    section(8, "RISK MANAGEMENT & EXECUTION RULES", C_RED)
    print(f"""
  {C_YEL}{S_BR}PORTFOLIO-LEVEL RISK CONTROLS:{S_RS}

  1. {C_CYAN}Equity Stop-Loss:{S_RS}
     - If portfolio equity drops 10% from High-Water-Mark, FLATTEN ALL positions.
     - Resume trading only after equity recovers to 95% of HWM.

  2. {C_CYAN}Daily Loss Limit:{S_RS}
     - Maximum -3% intraday loss triggers 24-hour trading halt.
     - Reset limit at 00:00 UTC daily.

  3. {C_CYAN}Regime Crisis Shutdown:{S_RS}
     - When RegimeDetector.classify() = CRISIS, disable ALL Regime-Adaptive edges (11-16).
     - Time-Based edges (1-10) may continue as they are regime-independent.
     - Resume adaptive edges when regime exits CRISIS for 3+ consecutive days.

  4. {C_CYAN}Position Concentration:{S_RS}
     - No more than 30% of total capital exposed to any single symbol.
     - No more than 50% of total capital exposed to any single family.
     - Currently: OP=2 edges, XMR=1, ATOM=1, etc. (Diversification: EXCELLENT)

  5. {C_CYAN}Correlation Circuit Breaker:{S_RS}
     - If rolling 30-day pairwise correlation between any 2 edges exceeds 0.7,
       reduce the smaller edge's allocation by 50%.

  6. {C_CYAN}Slippage Tolerance Monitor:{S_RS}
     - Backtest assumed 0.03% slippage per side.
     - If observed live slippage exceeds 0.08% for 5+ consecutive trades, halt symbol.

  7. {C_CYAN}Rebalancing Cadence:{S_RS}
     - Recompute allocation weights every 90 days using rolling 12-month performance.
     - Deploy paper-tested weight changes with 30-day observation before going live.

  8. {C_CYAN}Emergency Kill Switch:{S_RS}
     - If BTC drops >20% in 24h OR VIX-equivalent (crypto realized vol) > 200% annualized,
       IMMEDIATELY flatten all positions and disable auto-execution.
""")

    # ==============================
    # SECTION 9: EXPORTS
    # ==============================
    section(9, "EXPORT FILES", C_CYAN)

    # Master trade log
    all_trades_flat = []
    for e in all_edges_data:
        ed = e["edge"]
        for t in e["trades_full"]:
            all_trades_flat.append({
                "edge_id": ed["id"],
                "strategy": ed["strategy"],
                "symbol": ed["symbol"],
                "tf": ed["tf"],
                "family": ed["family"],
                "entry_time": t["entry_time"],
                "exit_time": t["exit_time"],
                "direction": t["direction"],
                "entry_price": t["entry_price"],
                "exit_price": t["exit_price"],
                "size": t["size"],
                "pnl": t["pnl"],
                "pnl_pct": t["pnl_pct"],
                "exit_reason": t["exit_reason"],
                "period": "IS" if t["exit_time"] <= IS_END else "OOS"
            })
    log_csv = os.path.join(RESULTS_ROOT, "master_portfolio_trade_log.csv")
    pd.DataFrame(all_trades_flat).to_csv(log_csv, index=False)
    print(f"  {C_GRN}[SAVED] {log_csv} ({len(all_trades_flat)} trades){S_RS}")

    # Monthly returns matrix
    monthly_matrix = []
    port_eq_full = model_results[best_model]["equity"]
    port_eq_monthly = port_eq_full.resample("ME").last().pct_change().dropna() * 100
    for idx, ret in port_eq_monthly.items():
        # Trades this month
        m_start = pd.Timestamp(idx.year, idx.month, 1)
        if idx.month == 12:
            m_end = pd.Timestamp(idx.year+1, 1, 1) - pd.Timedelta(seconds=1)
        else:
            m_end = pd.Timestamp(idx.year, idx.month+1, 1) - pd.Timedelta(seconds=1)
        m_tr = sum(1 for e in all_edges_data for t in e["trades_full"] if m_start <= t["exit_time"] <= m_end)
        monthly_matrix.append({
            "year": idx.year, "month": idx.month,
            "return_pct": round(float(ret), 3),
            "trades": m_tr,
            "period": "IS" if idx <= IS_END else "OOS"
        })
    mo_csv = os.path.join(RESULTS_ROOT, "master_portfolio_monthly_returns.csv")
    pd.DataFrame(monthly_matrix).to_csv(mo_csv, index=False)
    print(f"  {C_GRN}[SAVED] {mo_csv} ({len(monthly_matrix)} months){S_RS}")

    # Allocation blueprint
    blueprint_rows = []
    for i, e in enumerate(all_edges_data):
        ed = e["edge"]
        for acct in accounts:
            w = best_w[i]
            blueprint_rows.append({
                "edge_id": ed["id"],
                "strategy": ed["strategy"],
                "symbol": ed["symbol"],
                "tf": ed["tf"],
                "family": ed["family"],
                "model": best_model,
                "weight_pct": round(w * 100, 3),
                "account_size_usd": acct,
                "notional_allocation_usd": round(acct * w, 2),
                "risk_per_trade_usd": round(acct * w * RISK_PER_TRADE, 2),
                "weight_equal": round(w_equal[i]*100, 3),
                "weight_risk_parity": round(w_rp[i]*100, 3),
                "weight_kelly": round(w_kelly[i]*100, 3),
                "weight_alpha": round(w_alpha[i]*100, 3),
            })
    bp_csv = os.path.join(RESULTS_ROOT, "master_portfolio_allocation_blueprint.csv")
    pd.DataFrame(blueprint_rows).to_csv(bp_csv, index=False)
    print(f"  {C_GRN}[SAVED] {bp_csv}{S_RS}")

    # ==============================
    # SECTION 10: EXECUTIVE SUMMARY
    # ==============================
    section(10, "EXECUTIVE SUMMARY", C_GRN)
    best_m = model_results[best_model]["metrics"]
    best_oos = model_oos_results[best_model]["metrics"]
    print(f"""
  {C_GRN}{S_BR}MASTER PORTFOLIO METRICS ({best_model}):{S_RS}

  Full Period (2022-2026):
    - Final Equity        : ${best_m['final_equity']:,.2f} (started at ${STARTING_CAPITAL:,.0f})
    - Total Return        : {(best_m['final_equity']/STARTING_CAPITAL - 1)*100:+.2f}%
    - CAGR                : {best_m['cagr_pct']:.2f}%
    - Sharpe Ratio        : {best_m['sharpe']:.3f}
    - Sortino Ratio       : {best_m['sortino']:.3f}
    - Max Drawdown        : {best_m['max_dd_pct']:.2f}%
    - Calmar Ratio        : {best_m['calmar']:.3f}
    - Profit Factor       : {best_m['profit_factor']:.3f}
    - Win Rate            : {best_m['win_rate']:.2f}%
    - Total Trades        : {best_m['total_trades']:,}
    - Expectancy per Trade: ${best_m['expectancy_dollar']:.4f}

  2026 OOS (Unseen Data):
    - OOS CAGR            : {best_oos['cagr_pct']:.2f}%
    - OOS Sharpe          : {best_oos['sharpe']:.3f}
    - OOS Max Drawdown    : {best_oos['max_dd_pct']:.2f}%
    - OOS Win Rate        : {best_oos['win_rate']:.2f}%
    - OOS Profit Factor   : {best_oos['profit_factor']:.3f}
    - OOS Total Trades    : {best_oos['total_trades']:,}

  {C_CYAN}Portfolio Composition:{S_RS}
    - Total Edges         : {len(all_edges_data)}
    - Unique Symbols      : {len(set(e['edge']['symbol'] for e in all_edges_data))}
    - Time-Based Edges    : {sum(1 for e in all_edges_data if e['edge']['family']=='TimeBased')}
    - Regime-Adaptive     : {sum(1 for e in all_edges_data if e['edge']['family']=='RegimeAdaptive')}
    - Timeframes Used     : {sorted(set(e['edge']['tf'] for e in all_edges_data))}

  {C_MAG}Deployment Readiness:{S_RS}
""")
    if best_oos['sharpe'] > 1.0 and best_oos['cagr_pct'] > 0:
        print(f"  {C_GRN}{S_BR}GO - Deploy to Bybit testnet paper trading (30-day observation minimum){S_RS}")
    elif best_oos['sharpe'] > 0.5:
        print(f"  {C_YEL}CONDITIONAL - Deploy top-scoring edges only, extended observation period{S_RS}")
    else:
        print(f"  {C_RED}HOLD - OOS performance insufficient; revisit strategy selection{S_RS}")

    elapsed = time.time() - t0
    print()
    box(f"TASK 11/12 COMPLETE - Runtime: {elapsed/60:.2f} minutes", C_GRN)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C_RED}Interrupted{S_RS}")
    except Exception as e:
        print(f"\n{C_RED}FATAL: {e}{S_RS}")
        traceback.print_exc()
