"""
TASK 7.1: HIGH-FREQUENCY TIME-BASED ALPHA SCALING
Uses canonical engine_v2. Fixes pandas 2.2+ fillna deprecation in S30d.
Prioritizes 15m/30m for high trade frequency.
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
        RSI, MACD, ATR, EMA, SMA, ADX
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

IS_START = pd.Timestamp("2022-01-01")
IS_END   = pd.Timestamp("2025-12-31 23:59:59")
OOS_START = pd.Timestamp("2026-01-01")
OOS_END   = pd.Timestamp("2026-08-31 23:59:59")

# IS gates (stricter for high-frequency confidence)
IS_MIN_TRADES = 80
IS_MIN_WIN_RATE = 51.0
IS_MIN_PF = 1.25
IS_MIN_SHARPE = 0.65
IS_MAX_DD_LIMIT = -22.0

# OOS gates
OOS_MIN_SHARPE = 0.50
OOS_MIN_RETURN = 0.0
OOS_MIN_PF = 1.15
OOS_MIN_WIN_RATE = 50.0
OOS_MIN_TRADES = 15

TOP_LIQUID_SYMBOLS = [
    "BTC_USDT_USDT","ETH_USDT_USDT","SOL_USDT_USDT","XRP_USDT_USDT","DOGE_USDT_USDT",
    "ADA_USDT_USDT","AVAX_USDT_USDT","SHIB_USDT_USDT","LINK_USDT_USDT","BNB_USDT_USDT",
    "MATIC_USDT_USDT","LTC_USDT_USDT","DOT_USDT_USDT","BCH_USDT_USDT","ATOM_USDT_USDT",
    "NEAR_USDT_USDT","UNI_USDT_USDT","OP_USDT_USDT","ARB_USDT_USDT","APT_USDT_USDT",
    "FIL_USDT_USDT","INJ_USDT_USDT","TRX_USDT_USDT","ETC_USDT_USDT","XLM_USDT_USDT",
    "SUI_USDT_USDT","SEI_USDT_USDT","TIA_USDT_USDT","RUNE_USDT_USDT","AAVE_USDT_USDT",
    "MKR_USDT_USDT","LDO_USDT_USDT","SAND_USDT_USDT","MANA_USDT_USDT","CRV_USDT_USDT",
    "GALA_USDT_USDT","APE_USDT_USDT","FTM_USDT_USDT","AXS_USDT_USDT","GRT_USDT_USDT",
    "CHZ_USDT_USDT","1000LUNC_USDT_USDT","TWT_USDT_USDT","ZRX_USDT_USDT","ANKR_USDT_USDT",
    "STX_USDT_USDT","ORDI_USDT_USDT","WIF_USDT_USDT","PEPE_USDT_USDT","1000PEPE_USDT_USDT",
    "BONK_USDT_USDT","FLOKI_USDT_USDT","EGLD_USDT_USDT","MINA_USDT_USDT","ZEN_USDT_USDT",
    "ALICE_USDT_USDT","JST_USDT_USDT","SUN_USDT_USDT","ONT_USDT_USDT","AGLD_USDT_USDT"
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
# STRATEGIES (pandas 2.2+ compatible)
# ============================================================
def _build_signals_series(df, long_mask, short_mask):
    n = len(df)
    signals_arr = np.zeros(n, dtype=int)
    signals_arr[long_mask.values] = 1
    signals_arr[short_mask.values] = -1
    return pd.Series(signals_arr, index=df.index, dtype=int)

def _build_sl_tp(df, signals, atr_series, sl_mult, tp_mult):
    close = df["close"]
    sl_series = pd.Series(np.nan, index=df.index, dtype=float)
    tp_series = pd.Series(np.nan, index=df.index, dtype=float)
    lm = signals == 1; sm = signals == -1
    sl_series.loc[lm] = (close - sl_mult * atr_series).loc[lm].values
    sl_series.loc[sm] = (close + sl_mult * atr_series).loc[sm].values
    tp_series.loc[lm] = (close + tp_mult * atr_series).loc[lm].values
    tp_series.loc[sm] = (close - tp_mult * atr_series).loc[sm].values
    return sl_series, tp_series

class S30a_DOW_Seasonality(Strategy):
    name = "S30a_DOW_Seasonality"
    category = "Seasonal"
    default_params = {
        "entry_day_long": 4, "entry_day_short": 3, "rsi_period": 14,
        "rsi_long_threshold": 50.0, "rsi_short_threshold": 50.0,
        "sl_atr_mult": 1.5, "tp_atr_mult": 2.5, "atr_period": 14
    }
    def generate_signals(self, df, params):
        p = params
        dow = pd.Series(df.index.dayofweek, index=df.index)
        r = RSI(df["close"], int(p.get("rsi_period", 14)))
        a = ATR(df, int(p.get("atr_period", 14)))
        long_mask = ((dow == int(p.get("entry_day_long", 4))) & (r > p.get("rsi_long_threshold", 50.0))).fillna(False).astype(bool)
        short_mask = ((dow == int(p.get("entry_day_short", 3))) & (r < p.get("rsi_short_threshold", 50.0))).fillna(False).astype(bool)
        signals = _build_signals_series(df, long_mask, short_mask)
        sl_series, tp_series = _build_sl_tp(df, signals, a, float(p.get("sl_atr_mult", 1.5)), float(p.get("tp_atr_mult", 2.5)))
        return signals, sl_series, tp_series

class S30b_MidWeek_MeanReversion(Strategy):
    name = "S30b_MidWeek_MeanReversion"
    category = "Seasonal_MeanRev"
    default_params = {
        "sma_period": 20, "atr_period": 14, "deviation_atr": 1.5,
        "sl_atr_mult": 2.0, "tp_atr_mult": 1.5
    }
    def generate_signals(self, df, params):
        p = params
        sma = SMA(df["close"], int(p.get("sma_period", 20)))
        a = ATR(df, int(p.get("atr_period", 14)))
        dev = float(p.get("deviation_atr", 1.5))
        dow = pd.Series(df.index.dayofweek, index=df.index)
        dist = (df["close"] - sma)
        far_above = (dist > dev * a).fillna(False).astype(bool)
        far_below = (dist < -dev * a).fillna(False).astype(bool)
        trigger = ((dow == 1) | (dow == 2)).astype(bool)
        short_mask = (far_above & trigger).fillna(False).astype(bool)
        long_mask = (far_below & trigger).fillna(False).astype(bool)
        signals = _build_signals_series(df, long_mask, short_mask)
        sl_series, tp_series = _build_sl_tp(df, signals, a, float(p.get("sl_atr_mult", 2.0)), float(p.get("tp_atr_mult", 1.5)))
        return signals, sl_series, tp_series

class S30c_Asian_Session_Fade(Strategy):
    name = "S30c_Asian_Session_Fade"
    category = "Session_Fade"
    default_params = {
        "session_start_hour": 0, "session_end_hour": 8,
        "breakout_start_hour": 3, "breakout_end_hour": 6,
        "range_bars_hours": 2, "breakout_pct_atr": 0.3,
        "sl_atr_mult": 1.0, "tp_atr_mult": 1.5, "atr_period": 14
    }
    def generate_signals(self, df, params):
        p = params
        a = ATR(df, int(p.get("atr_period", 14)))
        hour = pd.Series(df.index.hour, index=df.index)
        date = pd.Series(df.index.date, index=df.index)
        breakout_start = int(p.get("breakout_start_hour", 3))
        breakout_end = int(p.get("breakout_end_hour", 6))
        range_hours = int(p.get("range_bars_hours", 2))
        session_start = int(p.get("session_start_hour", 0))
        first_range_end = session_start + range_hours
        in_first_range = (hour >= session_start) & (hour < first_range_end)
        df_tmp = df.copy()
        df_tmp["date"] = date.values
        df_tmp["in_first"] = in_first_range.values
        grp = df_tmp[df_tmp["in_first"]].groupby("date")
        first_high = grp["high"].max()
        first_low = grp["low"].min()
        range_high_series = pd.Series(date.map(first_high).values, index=df.index)
        range_low_series = pd.Series(date.map(first_low).values, index=df.index)
        in_breakout_window = (hour >= breakout_start) & (hour < breakout_end)
        vol_prev3_mean = df["volume"].rolling(3).mean().shift(1)
        vol_declining = df["volume"] < vol_prev3_mean
        bp = float(p.get("breakout_pct_atr", 0.3))
        upside_break = df["close"] > (range_high_series + bp * a)
        downside_break = df["close"] < (range_low_series - bp * a)
        short_mask = (in_breakout_window & upside_break & vol_declining).fillna(False).astype(bool)
        long_mask = (in_breakout_window & downside_break & vol_declining).fillna(False).astype(bool)
        signals = _build_signals_series(df, long_mask, short_mask)
        sl_series, tp_series = _build_sl_tp(df, signals, a, float(p.get("sl_atr_mult", 1.0)), float(p.get("tp_atr_mult", 1.5)))
        return signals, sl_series, tp_series

class S30d_NY_London_Overlap_Momentum(Strategy):
    """HOTFIXED: replaced deprecated fillna(method='bfill') with .bfill()"""
    name = "S30d_NY_London_Overlap_Momentum"
    category = "Session_Breakout"
    default_params = {
        "london_start_hour": 8, "london_end_hour": 12,
        "trade_start_hour": 12, "trade_end_hour": 16,
        "atr_expansion_ratio": 1.2, "sl_atr_mult": 1.5, "tp_atr_mult": 2.5,
        "atr_period": 14
    }
    def generate_signals(self, df, params):
        p = params
        a = ATR(df, int(p.get("atr_period", 14)))
        hour = pd.Series(df.index.hour, index=df.index)
        date = pd.Series(df.index.date, index=df.index)
        london_start = int(p.get("london_start_hour", 8))
        london_end = int(p.get("london_end_hour", 12))
        trade_start = int(p.get("trade_start_hour", 12))
        trade_end = int(p.get("trade_end_hour", 16))
        df_tmp = df.copy()
        df_tmp["date"] = date.values
        in_london = (hour >= london_start) & (hour < london_end)
        df_tmp["in_london"] = in_london.values
        grp = df_tmp[df_tmp["in_london"]].groupby("date")
        london_high = grp["high"].max()
        london_low = grp["low"].min()
        lh = pd.Series(date.map(london_high).values, index=df.index)
        ll = pd.Series(date.map(london_low).values, index=df.index)
        # HOTFIX: replaced fillna(method="bfill") with .bfill()
        atr_ref = a.shift(4).bfill()
        atr_expand = a > float(p.get("atr_expansion_ratio", 1.2)) * atr_ref
        in_trade_window = (hour >= trade_start) & (hour < trade_end)
        long_break = df["close"] > lh
        short_break = df["close"] < ll
        long_mask = (in_trade_window & long_break & atr_expand).fillna(False).astype(bool)
        short_mask = (in_trade_window & short_break & atr_expand).fillna(False).astype(bool)
        signals = _build_signals_series(df, long_mask, short_mask)
        sl_series, tp_series = _build_sl_tp(df, signals, a, float(p.get("sl_atr_mult", 1.5)), float(p.get("tp_atr_mult", 2.5)))
        return signals, sl_series, tp_series

class S30e_Weekend_Gap_Fade(Strategy):
    """HOTFIXED: replaced ffill(method=...) with .ffill()"""
    name = "S30e_Weekend_Gap_Fade"
    category = "Weekend_Gap"
    default_params = {
        "friday_reference_hour": 20, "sunday_window_start_hour": 18,
        "sunday_window_end_hour": 23, "gap_threshold_atr": 1.5,
        "sl_atr_mult": 2.0, "tp_atr_mult": 1.5, "atr_period": 14
    }
    def generate_signals(self, df, params):
        p = params
        a = ATR(df, int(p.get("atr_period", 14)))
        dow = pd.Series(df.index.dayofweek, index=df.index)
        hour = pd.Series(df.index.hour, index=df.index)
        fri_hr = int(p.get("friday_reference_hour", 20))
        sun_start = int(p.get("sunday_window_start_hour", 18))
        sun_end = int(p.get("sunday_window_end_hour", 23))
        fri_ref = pd.Series(np.nan, index=df.index)
        friday_mask = (dow == 4) & (hour == fri_hr)
        fri_ref[friday_mask] = df["close"][friday_mask].values
        # HOTFIX: .ffill() instead of fillna(method="ffill")
        fri_ref = fri_ref.ffill()
        in_sunday = (dow == 6) & (hour >= sun_start) & (hour <= sun_end)
        gap = df["close"] - fri_ref
        thr = float(p.get("gap_threshold_atr", 1.5)) * a
        gap_up = gap > thr
        gap_down = gap < -thr
        short_mask = (in_sunday & gap_up).fillna(False).astype(bool)
        long_mask = (in_sunday & gap_down).fillna(False).astype(bool)
        signals = _build_signals_series(df, long_mask, short_mask)
        sl_series, tp_series = _build_sl_tp(df, signals, a, float(p.get("sl_atr_mult", 2.0)), float(p.get("tp_atr_mult", 1.5)))
        return signals, sl_series, tp_series

class S30f_Funding_Rate_Proxy_Cycle(Strategy):
    name = "S30f_Funding_Rate_Proxy_Cycle"
    category = "Funding_Arbitrage"
    default_params = {
        "settlement_hours": [0, 8, 16], "pre_settlement_minutes": 45,
        "rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30,
        "sl_atr_mult": 1.0, "tp_atr_mult": 1.2, "atr_period": 14
    }
    def generate_signals(self, df, params):
        p = params
        a = ATR(df, int(p.get("atr_period", 14)))
        r = RSI(df["close"], int(p.get("rsi_period", 14)))
        hour = pd.Series(df.index.hour, index=df.index)
        minute = pd.Series(df.index.minute, index=df.index)
        pre_min = int(p.get("pre_settlement_minutes", 45))
        settlement_hours = p.get("settlement_hours", [0, 8, 16])
        pre_settlement_mask = pd.Series(False, index=df.index)
        for sh in settlement_hours:
            pre_hour = (sh - 1) % 24
            in_window = (hour == pre_hour) & (minute >= (60 - pre_min))
            pre_settlement_mask = pre_settlement_mask | in_window
        ob = float(p.get("rsi_overbought", 70))
        os_ = float(p.get("rsi_oversold", 30))
        short_mask = (pre_settlement_mask & (r > ob)).fillna(False).astype(bool)
        long_mask = (pre_settlement_mask & (r < os_)).fillna(False).astype(bool)
        signals = _build_signals_series(df, long_mask, short_mask)
        sl_series, tp_series = _build_sl_tp(df, signals, a, float(p.get("sl_atr_mult", 1.0)), float(p.get("tp_atr_mult", 1.2)))
        return signals, sl_series, tp_series

# Prioritize 15m/30m for HF; also include 1H/4H for completeness
STRATEGY_REGISTRY = [
    (S30a_DOW_Seasonality,           ["15m", "30m", "1H", "4H"]),
    (S30b_MidWeek_MeanReversion,     ["15m", "30m", "1H", "4H"]),
    (S30c_Asian_Session_Fade,        ["15m", "30m"]),
    (S30d_NY_London_Overlap_Momentum,["15m", "30m", "1H"]),
    (S30e_Weekend_Gap_Fade,          ["15m", "30m", "1H"]),
    (S30f_Funding_Rate_Proxy_Cycle,  ["15m", "30m"]),
]

def compute_monthly_freq(trades, days):
    if not trades or days <= 0: return 0.0
    months = days / 30.44
    return round(len(trades) / max(months, 1), 3)

def get_month_range_2026(month):
    if month == 12:
        start = pd.Timestamp(f"2026-{month:02d}-01")
        end = pd.Timestamp("2027-01-01") - pd.Timedelta(seconds=1)
    else:
        start = pd.Timestamp(f"2026-{month:02d}-01")
        end = pd.Timestamp(f"2026-{month+1:02d}-01") - pd.Timedelta(seconds=1)
    return start, end

# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    box("TASK 7.1: HIGH-FREQUENCY TIME-BASED ALPHA SCALING", C_CYAN)
    print(f"{C_CYAN}  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{C_CYAN}  Engine: v{ENGINE_VER}")

    # SECTION 0: Bug fix confirmation
    section(0, "BUG FIX CONFIRMATION (S30d & S30e pandas 2.2+ deprecation)", C_GRN)
    print(f"  {C_GRN}[FIXED] S30d line: 'atr_ref = a.shift(4).bfill()'{S_RS}")
    print(f"          Previously: 'a.shift(4).fillna(method=\"bfill\")'  <-- DEPRECATED")
    print(f"  {C_GRN}[FIXED] S30e line: 'fri_ref = fri_ref.ffill()'{S_RS}")
    print(f"          Previously: 'fri_ref.fillna(method=\"ffill\")'  <-- DEPRECATED")

    loader = MultiTFDataLoader(DATA_ROOT)
    engine = BacktestEngine()

    # SECTION 1: Symbol discovery
    section(1, "SYMBOL UNIVERSE (Top-Liquidity Filter)", C_CYAN)
    all_available = set(loader.available_symbols("1H"))
    universe = [s for s in TOP_LIQUID_SYMBOLS if s in all_available]
    print(f"  Requested top-liquidity symbols: {len(TOP_LIQUID_SYMBOLS)}")
    print(f"  Available in data:                 {len(universe)}")
    if len(universe) < 20:
        print(f"  {C_YEL}[!] Fewer than 20 symbols available; supplementing from full pool{S_RS}")
        extra = [s for s in all_available if s not in universe][:40 - len(universe)]
        universe = universe + extra
    universe = universe[:60]  # cap for runtime
    print(f"  Final symbol universe:             {len(universe)}")

    total_combos = 0
    for strat_cls, tfs in STRATEGY_REGISTRY:
        for tf in tfs:
            total_combos += len(universe)
    print(f"  Total (strategy x symbol x TF) combos: {total_combos}")

    # SECTION 2: IS Screening
    section(2, "IN-SAMPLE SCREENING (2022-2025)", C_MAG)
    is_results = []
    counter = 0
    err_count = 0
    skip_count = 0

    for strat_cls, tfs in STRATEGY_REGISTRY:
        strat_instance = strat_cls(engine=engine)
        for tf in tfs:
            print(f"\n  {C_CYAN}Strategy: {strat_cls.name} | TF: {tf} | Symbols: {len(universe)}{S_RS}")
            for sym in universe:
                counter += 1
                try:
                    df = loader.load(sym, tf, IS_START, IS_END)
                    if df is None or len(df) < 500:
                        skip_count += 1
                        continue
                    result = strat_instance.backtest(df, strat_cls.default_params)
                    m = result["metrics"]
                    days_span = (df.index.max() - df.index.min()).days
                    m["monthly_freq"] = compute_monthly_freq(result["trades"], days_span)
                    is_results.append({
                        "strategy": strat_cls.name, "symbol": sym, "tf": tf, **m
                    })
                    if counter % 200 == 0:
                        elapsed_min = (time.time()-t0)/60
                        print(f"    Progress: {counter}/{total_combos} | Elapsed: {elapsed_min:.1f}m | "
                              f"Qualified so far: {sum(1 for r in is_results if r['sharpe']>=IS_MIN_SHARPE and r['total_trades']>=IS_MIN_TRADES)}")
                except Exception as e:
                    err_count += 1
                    if err_count <= 3:
                        print(f"    {C_RED}[ERR] {strat_cls.name}|{sym}|{tf}: {e}{S_RS}")

    print(f"\n  {C_GRN}[OK] IS screening: {len(is_results)} setups, {err_count} errors, {skip_count} skipped{S_RS}")

    is_df = pd.DataFrame(is_results)
    all_csv = os.path.join(RESULTS_ROOT, "task7.1_high_frequency_results.csv")
    is_df.to_csv(all_csv, index=False)
    print(f"  {C_GRN}[SAVED] {all_csv}{S_RS}")

    # SECTION 3: Apply IS gates
    section(3, "IS GATE FILTERING", C_MAG)
    if len(is_df) == 0:
        print(f"  {C_RED}[!] No IS results.{S_RS}")
        return

    print(f"  Criteria: Trades>={IS_MIN_TRADES}, Win>={IS_MIN_WIN_RATE}%, PF>={IS_MIN_PF}, Sharpe>={IS_MIN_SHARPE}, MaxDD>{IS_MAX_DD_LIMIT}%")
    qualified = is_df[
        (is_df["total_trades"] >= IS_MIN_TRADES) &
        (is_df["win_rate_pct"] >= IS_MIN_WIN_RATE) &
        (is_df["profit_factor"] >= IS_MIN_PF) &
        (is_df["sharpe"] >= IS_MIN_SHARPE) &
        (is_df["max_drawdown_pct"] > IS_MAX_DD_LIMIT)
    ].copy()
    qualified = qualified.sort_values(["sharpe", "monthly_freq"], ascending=[False, False]).reset_index(drop=True)
    print(f"  Before: {len(is_df)} -> Qualified: {C_GRN}{len(qualified)}{S_RS}")

    if len(qualified) == 0:
        print(f"  {C_YEL}[!] No IS-qualified setups. Consider relaxing thresholds.{S_RS}")
        return

    # SECTION 4: OOS validation
    section(4, "OOS 2026 VALIDATION (FROZEN PARAMS)", C_MAG)
    strat_map = {cls.name: cls for cls, _ in STRATEGY_REGISTRY}
    oos_verified = []
    oos_all = []
    oos_tested = 0

    for i, row in qualified.iterrows():
        oos_tested += 1
        strat_name = row["strategy"]; sym = row["symbol"]; tf = row["tf"]
        cls = strat_map.get(strat_name)
        if cls is None: continue
        try:
            strat_instance = cls(engine=engine)
            df_oos = loader.load(sym, tf, OOS_START, OOS_END)
            if df_oos is None or len(df_oos) < 50:
                continue
            result = strat_instance.backtest(df_oos, cls.default_params)
            m = result["metrics"]
            days_span = (df_oos.index.max() - df_oos.index.min()).days
            m["monthly_freq"] = compute_monthly_freq(result["trades"], days_span)
            entry = {
                "strategy": strat_name, "symbol": sym, "tf": tf,
                "is_sharpe": round(row["sharpe"], 3),
                "is_win_rate": round(row["win_rate_pct"], 2),
                "is_pf": round(row["profit_factor"], 3),
                "is_trades": int(row["total_trades"]),
                "is_return": round(row["total_return_pct"], 2),
                "is_monthly_freq": round(row["monthly_freq"], 3),
                "oos_sharpe": round(m["sharpe"], 3),
                "oos_win_rate": round(m["win_rate_pct"], 2),
                "oos_pf": round(m["profit_factor"], 3),
                "oos_trades": int(m["total_trades"]),
                "oos_return": round(m["total_return_pct"], 2),
                "oos_max_dd": round(m["max_drawdown_pct"], 2),
                "oos_monthly_freq": round(m["monthly_freq"], 3),
                "oos_expectancy": round(m["expectancy"], 4),
                "trades_ref": result["trades"],
            }
            oos_all.append(entry)
            passes = (
                m["sharpe"] >= OOS_MIN_SHARPE and
                m["total_return_pct"] > OOS_MIN_RETURN and
                m["profit_factor"] >= OOS_MIN_PF and
                m["win_rate_pct"] >= OOS_MIN_WIN_RATE and
                m["total_trades"] >= OOS_MIN_TRADES and
                m["expectancy"] > 0
            )
            if passes:
                deg = m["sharpe"] / row["sharpe"] if row["sharpe"] > 0 else 0
                entry["degradation"] = round(deg, 3)
                entry["status"] = "STRONG_PASS" if deg >= 0.7 else "PASS"
                oos_verified.append(entry)
        except Exception as e:
            if oos_tested <= 3:
                print(f"    {C_RED}[ERR OOS] {strat_name}|{sym}|{tf}: {e}{S_RS}")

    print(f"  OOS tested: {oos_tested} | Passed: {C_GRN}{len(oos_verified)}{S_RS}")

    # Composite score for ranking
    for r in oos_verified:
        r["composite"] = round(
            r["oos_sharpe"] * 30 +
            (r["oos_pf"] - 1) * 20 +
            r["oos_win_rate"] * 0.5 +
            min(r["oos_monthly_freq"], 10) * 3 +
            r["degradation"] * 10
        , 2)
    oos_verified.sort(key=lambda x: -x["composite"])

    # SECTION 5: Funnel summary
    section(5, "EXECUTION FUNNEL SUMMARY", C_CYAN)
    funnel = [
        ["Total combos scanned", total_combos],
        ["IS backtests completed", len(is_results)],
        ["Passed IS gates", len(qualified)],
        ["Tested on OOS 2026", oos_tested],
        ["Passed OOS gates", len(oos_verified)],
        ["Yield rate (OOS-verified / total)", f"{100*len(oos_verified)/max(total_combos,1):.2f}%"],
    ]
    print(tabulate(funnel, headers=["Stage", "Count"], tablefmt="grid"))

    if len(oos_verified) == 0:
        print(f"\n  {C_RED}[!] No OOS-verified edges. Report saved with IS-only data.{S_RS}")
        return

    # SECTION 6: Top 25
    section(6, "TOP 25 OOS-VERIFIED EDGES (Ranked by Composite)", C_GRN)
    top_rows = []
    for i, r in enumerate(oos_verified[:25], 1):
        st_col = C_GRN if r["status"] == "STRONG_PASS" else C_YEL
        top_rows.append([
            i, r["strategy"][:22], r["symbol"][:14], r["tf"],
            f"{r['is_sharpe']:.2f}", f"{r['oos_sharpe']:.2f}",
            f"{r['oos_win_rate']:.1f}%", f"{r['oos_pf']:.2f}",
            r["oos_trades"], f"{r['oos_return']:.2f}%",
            f"{r['oos_monthly_freq']:.1f}",
            f"{st_col}{r['status']}{S_RS}"
        ])
    print(tabulate(top_rows,
        headers=["#","Strategy","Symbol","TF","IS Sh","OOS Sh","OOS Win%","OOS PF","OOS Tr","OOS Ret","Mo Fr","Status"],
        tablefmt="grid"))

    # SECTION 7: Portfolio construction targeting 25+ combined monthly trades
    section(7, "PORTFOLIO CONSTRUCTION (>=25 combined trades/month)", C_MAG)
    portfolio = []
    combined_monthly = 0.0
    sym_count = {}; strat_count = {}
    MAX_PER_SYMBOL = 3
    MAX_PER_STRATEGY = 6
    TARGET_MONTHLY = 25.0

    for r in oos_verified:
        if sym_count.get(r["symbol"], 0) >= MAX_PER_SYMBOL: continue
        if strat_count.get(r["strategy"], 0) >= MAX_PER_STRATEGY: continue
        # Only add if it improves diversity or if we still need frequency
        if combined_monthly < TARGET_MONTHLY or len(portfolio) < 8:
            portfolio.append(r)
            combined_monthly += r["oos_monthly_freq"]
            sym_count[r["symbol"]] = sym_count.get(r["symbol"], 0) + 1
            strat_count[r["strategy"]] = strat_count.get(r["strategy"], 0) + 1
        if combined_monthly >= TARGET_MONTHLY * 2 and len(portfolio) >= 12:
            break

    print(f"  Portfolio size            : {len(portfolio)}")
    print(f"  Combined monthly trades   : {combined_monthly:.1f}")
    print(f"  Target                    : {TARGET_MONTHLY}+")
    print(f"  Unique symbols            : {len(sym_count)}")
    print(f"  Unique strategies         : {len(strat_count)}")

    prt_rows = []
    for r in portfolio:
        prt_rows.append([
            r["strategy"][:22], r["symbol"][:14], r["tf"],
            f"{r['oos_sharpe']:.2f}", f"{r['oos_win_rate']:.1f}%",
            f"{r['oos_pf']:.2f}", r["oos_trades"], f"{r['oos_return']:.2f}%",
            f"{r['oos_monthly_freq']:.1f}"
        ])
    print("\n  Portfolio Members:")
    print(tabulate(prt_rows,
        headers=["Strategy","Symbol","TF","OOS Sh","OOS Win%","OOS PF","Trades","Return","Mo Fr"],
        tablefmt="simple"))

    # Combined stats
    all_pf_trades = []
    for r in portfolio:
        all_pf_trades.extend(r["trades_ref"])

    if all_pf_trades:
        pnls = np.array([t["pnl"] for t in all_pf_trades])
        wins = (pnls > 0).sum()
        pf_win_rate = 100 * wins / len(pnls) if len(pnls) > 0 else 0
        gp = pnls[pnls > 0].sum(); gl = abs(pnls[pnls < 0].sum())
        pf_pf = gp / gl if gl > 0 else 999
        pf_avg_win = pnls[pnls>0].mean() if wins > 0 else 0
        pf_avg_loss = abs(pnls[pnls<0].mean()) if (pnls<0).sum() > 0 else 0
        pf_expect = (wins/len(pnls))*pf_avg_win - (1-wins/len(pnls))*pf_avg_loss

        trades_by_time = sorted(all_pf_trades, key=lambda t: t["exit_time"])
        eq_values = [10000.0]; eq_times = [OOS_START]
        eq = 10000.0
        for t in trades_by_time:
            eq += t["pnl"]
            eq_values.append(eq); eq_times.append(t["exit_time"])
        eq_series = pd.Series(eq_values, index=eq_times)
        eq_daily = eq_series.resample("D").last().ffill()
        dr = eq_daily.pct_change().dropna()
        port_sh = float(dr.mean()/dr.std()*np.sqrt(365)) if dr.std() > 0 else 0
        peak = eq_daily.cummax(); dd = (eq_daily - peak)/peak
        port_dd = float(dd.min()*100)
        port_ret = (eq_series.iloc[-1]/eq_series.iloc[0] - 1) * 100

        print(f"\n  {C_GRN}{S_BR}Portfolio 2026 OOS Combined Metrics:{S_RS}")
        port_stats = [
            ["Combined Trades", len(all_pf_trades)],
            ["Combined Win Rate %", f"{pf_win_rate:.2f}"],
            ["Combined Profit Factor", f"{pf_pf:.3f}"],
            ["Combined Sharpe (daily)", f"{port_sh:.3f}"],
            ["Combined Max DD %", f"{port_dd:.2f}"],
            ["Combined Total Return %", f"{port_ret:.2f}"],
            ["Combined Expectancy $", f"{pf_expect:.4f}"],
            ["Combined Monthly Trades", f"{combined_monthly:.1f}"],
            ["Start Equity", f"${eq_series.iloc[0]:.2f}"],
            ["End Equity", f"${eq_series.iloc[-1]:.2f}"],
        ]
        print(tabulate(port_stats, headers=["Metric","Value"], tablefmt="grid"))

        # SECTION 8: Month-by-month 2026 heatmap
        section(8, "2026 MONTH-BY-MONTH PORTFOLIO PERFORMANCE HEATMAP", C_CYAN)
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug"]
        monthly_rows = []
        for mi in range(1, 9):
            ms, me = get_month_range_2026(mi)
            m_trades = [t for t in all_pf_trades if ms <= t["exit_time"] <= me]
            if m_trades:
                m_pnl = sum(t["pnl"] for t in m_trades)
                m_wins = sum(1 for t in m_trades if t["pnl"] > 0)
                m_wr = 100 * m_wins / len(m_trades)
                m_ret = m_pnl / 10000 * 100
                col = C_GRN if m_ret > 0 else C_RED
            else:
                m_ret = 0; m_wr = 0; col = C_YEL
            monthly_rows.append([
                month_names[mi-1],
                f"{col}{m_ret:+.2f}%{S_RS}",
                len(m_trades),
                f"{m_wr:.1f}%" if m_trades else "-"
            ])
        print(tabulate(monthly_rows, headers=["Month","Return","Trades","Win%"], tablefmt="grid"))

        # SECTION 9: Executive recommendation
        section(9, "EXECUTIVE RECOMMENDATION: CORE DEPLOYMENT PORTFOLIO", C_GRN)
        target_met = combined_monthly >= TARGET_MONTHLY
        wr_met = pf_win_rate >= 50
        pf_met = pf_pf > 1.15
        if target_met and wr_met and pf_met:
            verdict = f"{C_GRN}{S_BR}GO - Deploy to live paper trading (30-day observation){S_RS}"
        elif target_met and wr_met:
            verdict = f"{C_YEL}CONDITIONAL - Meets frequency+win but PF marginal; monitor closely{S_RS}"
        elif target_met:
            verdict = f"{C_YEL}FREQUENCY OK but WIN RATE fails - review sub-strategies{S_RS}"
        else:
            verdict = f"{C_RED}INSUFFICIENT FREQUENCY - relax gates or expand universe{S_RS}"
        print(f"\n  Verdict: {verdict}\n")

        # Suggested weights based on composite scores
        total_composite = sum(r["composite"] for r in portfolio)
        weight_rows = []
        for r in portfolio:
            w = r["composite"] / total_composite * 100 if total_composite > 0 else 0
            weight_rows.append([
                r["strategy"][:22], r["symbol"][:14], r["tf"],
                f"{r['oos_sharpe']:.2f}", f"{r['oos_win_rate']:.1f}%",
                f"{r['oos_monthly_freq']:.1f}", f"{r['composite']:.1f}",
                f"{w:.2f}%"
            ])
        print("  Composite-Weighted Allocation:")
        print(tabulate(weight_rows,
            headers=["Strategy","Symbol","TF","OOS Sh","Win%","MoFr","Comp","Weight"],
            tablefmt="grid"))

        print(f"\n  {C_CYAN}Next Steps:{S_RS}")
        print(f"    1. Save weights and monitor 30 days paper trading on Bybit testnet")
        print(f"    2. Track live vs backtest slippage (session boundaries expected +0.02-0.05%)")
        print(f"    3. Portfolio-level 10% equity stop-loss")
        print(f"    4. Re-run walk-forward every 90 days on rolling windows")

    # Save portfolio
    if portfolio:
        pf_csv = os.path.join(RESULTS_ROOT, "task7.1_qualified_portfolio.csv")
        pd.DataFrame([{k:v for k,v in r.items() if k != "trades_ref"} for r in portfolio]).to_csv(pf_csv, index=False)
        print(f"\n  {C_GRN}[SAVED] {pf_csv}{S_RS}")

    # Save all OOS verified
    if oos_verified:
        oos_csv = os.path.join(RESULTS_ROOT, "task7.1_all_oos_verified.csv")
        pd.DataFrame([{k:v for k,v in r.items() if k != "trades_ref"} for r in oos_verified]).to_csv(oos_csv, index=False)
        print(f"  {C_GRN}[SAVED] {oos_csv}{S_RS}")

    elapsed = time.time() - t0
    print()
    box(f"TASK 7.1 COMPLETE - Runtime: {elapsed/60:.2f} minutes", C_GRN)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C_RED}Interrupted{S_RS}")
    except Exception as e:
        print(f"\n{C_RED}FATAL: {e}{S_RS}")
        traceback.print_exc()
