"""
TASK 7: TIME-BASED ALPHA DISCOVERY ENGINE
Uses canonical engine_v2 (do NOT reimplement indicators or backtest).
Implements 6 time/session-based strategies and scans across all symbols.
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

# Import canonical engine
sys.path.insert(0, r"C:\BybitBacktest\backtester")
try:
    from engine_v2 import (
        __version__ as ENGINE_VER,
        MultiTFDataLoader, BacktestEngine, Strategy,
        RSI, MACD, ATR, EMA, SMA, ADX
    )
except ImportError as e:
    print(f"FATAL: engine_v2 import failed: {e}")
    print("Ensure Task 6 was completed successfully.")
    sys.exit(1)

# Dependencies
try:
    import pandas as pd
    import numpy as np
    from scipy import stats
    from colorama import Fore, Style, init as colorama_init
    from tabulate import tabulate
    colorama_init(autoreset=True)
except ImportError as e:
    print(f"Missing: {e}"); sys.exit(1)

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

# IS gate criteria
IS_MIN_TRADES = 40
IS_MIN_WIN_RATE = 48.0
IS_MIN_PF = 1.20
IS_MIN_SHARPE = 0.70
IS_MAX_DD_LIMIT = -25.0

# OOS gate criteria
OOS_MIN_SHARPE = 0.40
OOS_MIN_RETURN = 0.0
OOS_MIN_PF = 1.10
OOS_MIN_WIN_RATE = 48.0
OOS_MIN_TRADES = 10

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
# STRATEGY 1: DAY-OF-WEEK STRUCTURAL CYCLE
# ============================================================
class S30a_DOW_Seasonality(Strategy):
    name = "S30a_DOW_Seasonality"
    category = "Seasonal"
    default_params = {
        "entry_day_long": 4,      # Friday (0=Mon)
        "entry_day_short": 3,     # Thursday
        "rsi_period": 14,
        "rsi_long_threshold": 50.0,
        "rsi_short_threshold": 50.0,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 2.5,
        "atr_period": 14,
        "hold_bars": 6,           # exit after N bars if neither SL nor TP hit
    }

    def generate_signals(self, df, params):
        p = params
        dow = pd.Series(df.index.dayofweek, index=df.index)
        r = RSI(df["close"], int(p.get("rsi_period", 14)))
        a = ATR(df, int(p.get("atr_period", 14)))

        long_day = dow == int(p.get("entry_day_long", 4))
        short_day = dow == int(p.get("entry_day_short", 3))
        rsi_long_ok = r > p.get("rsi_long_threshold", 50.0)
        rsi_short_ok = r < p.get("rsi_short_threshold", 50.0)

        long_mask = (long_day & rsi_long_ok).fillna(False).astype(bool)
        short_mask = (short_day & rsi_short_ok).fillna(False).astype(bool)

        n = len(df)
        signals = np.zeros(n, dtype=int)
        signals[long_mask.values] = 1
        signals[short_mask.values] = -1
        signals = pd.Series(signals, index=df.index, dtype=int)

        close = df["close"]
        sl_mult = float(p.get("sl_atr_mult", 1.5))
        tp_mult = float(p.get("tp_atr_mult", 2.5))
        sl_series = pd.Series(np.nan, index=df.index, dtype=float)
        tp_series = pd.Series(np.nan, index=df.index, dtype=float)
        lm = signals == 1; sm = signals == -1
        sl_series.loc[lm] = (close - sl_mult * a).loc[lm].values
        sl_series.loc[sm] = (close + sl_mult * a).loc[sm].values
        tp_series.loc[lm] = (close + tp_mult * a).loc[lm].values
        tp_series.loc[sm] = (close - tp_mult * a).loc[sm].values
        return signals, sl_series, tp_series

# ============================================================
# STRATEGY 2: MID-WEEK MEAN REVERSION
# ============================================================
class S30b_MidWeek_MeanReversion(Strategy):
    name = "S30b_MidWeek_MeanReversion"
    category = "Seasonal_MeanRev"
    default_params = {
        "sma_period": 20,
        "atr_period": 14,
        "deviation_atr": 1.5,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 1.5,
        "trigger_day": 2,   # 0=Mon, 2=Wednesday
    }

    def generate_signals(self, df, params):
        p = params
        sma = SMA(df["close"], int(p.get("sma_period", 20)))
        a = ATR(df, int(p.get("atr_period", 14)))
        dev = float(p.get("deviation_atr", 1.5))
        dow = pd.Series(df.index.dayofweek, index=df.index)

        # Distance from SMA in ATR units
        dist = (df["close"] - sma)
        far_above = (dist > dev * a).fillna(False).astype(bool)
        far_below = (dist < -dev * a).fillna(False).astype(bool)
        trigger_day = dow == int(p.get("trigger_day", 2))
        # Also allow Tuesday (1) as trigger
        trigger_day_alt = dow == 1
        trigger = (trigger_day | trigger_day_alt).astype(bool)

        # Short when far above (fade the extension), long when far below
        short_mask = (far_above & trigger).fillna(False).astype(bool)
        long_mask = (far_below & trigger).fillna(False).astype(bool)

        n = len(df)
        signals = np.zeros(n, dtype=int)
        signals[long_mask.values] = 1
        signals[short_mask.values] = -1
        signals = pd.Series(signals, index=df.index, dtype=int)

        close = df["close"]
        sl_mult = float(p.get("sl_atr_mult", 2.0))
        tp_mult = float(p.get("tp_atr_mult", 1.5))
        sl_series = pd.Series(np.nan, index=df.index, dtype=float)
        tp_series = pd.Series(np.nan, index=df.index, dtype=float)
        lm = signals == 1; sm = signals == -1
        # Long targets: SL below, TP back at SMA (or ATR)
        sl_series.loc[lm] = (close - sl_mult * a).loc[lm].values
        sl_series.loc[sm] = (close + sl_mult * a).loc[sm].values
        # TP: mean-reversion target = at least tp_mult ATR back toward SMA
        tp_series.loc[lm] = (close + tp_mult * a).loc[lm].values
        tp_series.loc[sm] = (close - tp_mult * a).loc[sm].values
        return signals, sl_series, tp_series

# ============================================================
# STRATEGY 3: ASIAN SESSION FADE
# ============================================================
class S30c_Asian_Session_Fade(Strategy):
    name = "S30c_Asian_Session_Fade"
    category = "Session_Fade"
    default_params = {
        "session_start_hour": 0,
        "session_end_hour": 8,
        "breakout_start_hour": 3,
        "breakout_end_hour": 6,
        "range_bars_hours": 2,      # First 2 hours of Asian session define the range
        "breakout_pct_atr": 0.3,
        "sl_atr_mult": 1.0,
        "tp_atr_mult": 1.5,
        "atr_period": 14,
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

        # For each date, compute the high/low of first N hours of Asian session
        # Filter to bars in the first range window
        in_first_range = (hour >= session_start) & (hour < first_range_end)
        # Rolling per-day high/low of first range
        df_tmp = df.copy()
        df_tmp["date"] = date.values
        df_tmp["in_first"] = in_first_range.values
        # Group by date, get first-range high/low
        grp = df_tmp[df_tmp["in_first"]].groupby("date")
        first_high = grp["high"].max()
        first_low = grp["low"].min()
        # Map back to full frame by date
        range_high_series = pd.Series(date.map(first_high).values, index=df.index)
        range_low_series = pd.Series(date.map(first_low).values, index=df.index)

        # In breakout window
        in_breakout_window = (hour >= breakout_start) & (hour < breakout_end)
        # Volume declining vs prior 3 bars
        vol_prev3_mean = df["volume"].rolling(3).mean().shift(1)
        vol_declining = df["volume"] < vol_prev3_mean

        # Breakout above by 0.3% ATR (fade short) / below (fade long)
        bp = float(p.get("breakout_pct_atr", 0.3))
        upside_break = df["close"] > (range_high_series + bp * a)
        downside_break = df["close"] < (range_low_series - bp * a)

        short_mask = (in_breakout_window & upside_break & vol_declining).fillna(False).astype(bool)
        long_mask = (in_breakout_window & downside_break & vol_declining).fillna(False).astype(bool)

        n = len(df)
        signals = np.zeros(n, dtype=int)
        signals[long_mask.values] = 1
        signals[short_mask.values] = -1
        signals = pd.Series(signals, index=df.index, dtype=int)

        close = df["close"]
        sl_mult = float(p.get("sl_atr_mult", 1.0))
        tp_mult = float(p.get("tp_atr_mult", 1.5))
        sl_series = pd.Series(np.nan, index=df.index, dtype=float)
        tp_series = pd.Series(np.nan, index=df.index, dtype=float)
        lm = signals == 1; sm = signals == -1
        sl_series.loc[lm] = (close - sl_mult * a).loc[lm].values
        sl_series.loc[sm] = (close + sl_mult * a).loc[sm].values
        tp_series.loc[lm] = (close + tp_mult * a).loc[lm].values
        tp_series.loc[sm] = (close - tp_mult * a).loc[sm].values
        return signals, sl_series, tp_series

# ============================================================
# STRATEGY 4: NY/LONDON OVERLAP MOMENTUM
# ============================================================
class S30d_NY_London_Overlap_Momentum(Strategy):
    name = "S30d_NY_London_Overlap_Momentum"
    category = "Session_Breakout"
    default_params = {
        "london_start_hour": 8,
        "london_end_hour": 12,
        "trade_start_hour": 12,
        "trade_end_hour": 16,
        "atr_expansion_ratio": 1.2,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 2.5,
        "atr_period": 14,
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

        # Compute London range per day
        df_tmp = df.copy()
        df_tmp["date"] = date.values
        in_london = (hour >= london_start) & (hour < london_end)
        df_tmp["in_london"] = in_london.values
        grp = df_tmp[df_tmp["in_london"]].groupby("date")
        london_high = grp["high"].max()
        london_low = grp["low"].min()
        london_atr = grp["close"].apply(lambda x: (x.max() - x.min()) if len(x)>0 else np.nan)
        lh = pd.Series(date.map(london_high).values, index=df.index)
        ll = pd.Series(date.map(london_low).values, index=df.index)
        lr = pd.Series(date.map(london_atr).values, index=df.index)

        # ATR expansion: current ATR > multiplier * ATR at London close reference
        atr_expand = a > float(p.get("atr_expansion_ratio", 1.2)) * a.shift(4).fillna(method="bfill")

        in_trade_window = (hour >= trade_start) & (hour < trade_end)
        # Break above London high -> long momentum, break below -> short
        long_break = df["close"] > lh
        short_break = df["close"] < ll

        long_mask = (in_trade_window & long_break & atr_expand).fillna(False).astype(bool)
        short_mask = (in_trade_window & short_break & atr_expand).fillna(False).astype(bool)

        n = len(df)
        signals = np.zeros(n, dtype=int)
        signals[long_mask.values] = 1
        signals[short_mask.values] = -1
        signals = pd.Series(signals, index=df.index, dtype=int)

        close = df["close"]
        sl_mult = float(p.get("sl_atr_mult", 1.5))
        tp_mult = float(p.get("tp_atr_mult", 2.5))
        sl_series = pd.Series(np.nan, index=df.index, dtype=float)
        tp_series = pd.Series(np.nan, index=df.index, dtype=float)
        lm = signals == 1; sm = signals == -1
        sl_series.loc[lm] = (close - sl_mult * a).loc[lm].values
        sl_series.loc[sm] = (close + sl_mult * a).loc[sm].values
        tp_series.loc[lm] = (close + tp_mult * a).loc[lm].values
        tp_series.loc[sm] = (close - tp_mult * a).loc[sm].values
        return signals, sl_series, tp_series

# ============================================================
# STRATEGY 5: WEEKEND GAP FADE
# ============================================================
class S30e_Weekend_Gap_Fade(Strategy):
    name = "S30e_Weekend_Gap_Fade"
    category = "Weekend_Gap"
    default_params = {
        "friday_reference_hour": 20,
        "sunday_window_start_hour": 18,
        "sunday_window_end_hour": 23,
        "gap_threshold_atr": 1.5,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 1.5,
        "atr_period": 14,
    }

    def generate_signals(self, df, params):
        p = params
        a = ATR(df, int(p.get("atr_period", 14)))
        dow = pd.Series(df.index.dayofweek, index=df.index)  # 4=Fri, 6=Sun
        hour = pd.Series(df.index.hour, index=df.index)

        # Friday reference close at friday_reference_hour
        fri_hr = int(p.get("friday_reference_hour", 20))
        sun_start = int(p.get("sunday_window_start_hour", 18))
        sun_end = int(p.get("sunday_window_end_hour", 23))

        fri_ref = pd.Series(np.nan, index=df.index)
        friday_mask = (dow == 4) & (hour == fri_hr)
        fri_ref[friday_mask] = df["close"][friday_mask].values
        # Forward-fill last known Friday reference to all subsequent bars
        fri_ref = fri_ref.ffill()

        in_sunday = (dow == 6) & (hour >= sun_start) & (hour <= sun_end)
        gap = df["close"] - fri_ref
        thr = float(p.get("gap_threshold_atr", 1.5)) * a

        gap_up = gap > thr
        gap_down = gap < -thr

        # Fade: if gap up on Sunday, short expecting close to close gap; opposite for gap down
        short_mask = (in_sunday & gap_up).fillna(False).astype(bool)
        long_mask = (in_sunday & gap_down).fillna(False).astype(bool)

        n = len(df)
        signals = np.zeros(n, dtype=int)
        signals[long_mask.values] = 1
        signals[short_mask.values] = -1
        signals = pd.Series(signals, index=df.index, dtype=int)

        close = df["close"]
        sl_mult = float(p.get("sl_atr_mult", 2.0))
        tp_mult = float(p.get("tp_atr_mult", 1.5))
        sl_series = pd.Series(np.nan, index=df.index, dtype=float)
        tp_series = pd.Series(np.nan, index=df.index, dtype=float)
        lm = signals == 1; sm = signals == -1
        sl_series.loc[lm] = (close - sl_mult * a).loc[lm].values
        sl_series.loc[sm] = (close + sl_mult * a).loc[sm].values
        tp_series.loc[lm] = (close + tp_mult * a).loc[lm].values
        tp_series.loc[sm] = (close - tp_mult * a).loc[sm].values
        return signals, sl_series, tp_series

# ============================================================
# STRATEGY 6: FUNDING RATE PROXY CYCLE (8-HOUR SETTLEMENT)
# ============================================================
class S30f_Funding_Rate_Proxy_Cycle(Strategy):
    name = "S30f_Funding_Rate_Proxy_Cycle"
    category = "Funding_Arbitrage"
    default_params = {
        "settlement_hours": [0, 8, 16],  # Bybit funding times UTC
        "pre_settlement_minutes": 45,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "sl_atr_mult": 1.0,
        "tp_atr_mult": 1.2,
        "atr_period": 14,
    }

    def generate_signals(self, df, params):
        p = params
        a = ATR(df, int(p.get("atr_period", 14)))
        r = RSI(df["close"], int(p.get("rsi_period", 14)))
        hour = pd.Series(df.index.hour, index=df.index)
        minute = pd.Series(df.index.minute, index=df.index)

        # Pre-settlement window: e.g., 23:15-23:59, 07:15-07:59, 15:15-15:59
        pre_min = int(p.get("pre_settlement_minutes", 45))
        settlement_hours = p.get("settlement_hours", [0, 8, 16])
        # Pre-settlement windows: hour = (settlement_hour - 1), minute >= (60 - pre_min)
        pre_settlement_mask = pd.Series(False, index=df.index)
        for sh in settlement_hours:
            pre_hour = (sh - 1) % 24
            in_window = (hour == pre_hour) & (minute >= (60 - pre_min))
            pre_settlement_mask = pre_settlement_mask | in_window

        ob = float(p.get("rsi_overbought", 70))
        os_ = float(p.get("rsi_oversold", 30))
        # Counter-momentum: overbought -> short, oversold -> long
        short_mask = (pre_settlement_mask & (r > ob)).fillna(False).astype(bool)
        long_mask = (pre_settlement_mask & (r < os_)).fillna(False).astype(bool)

        n = len(df)
        signals = np.zeros(n, dtype=int)
        signals[long_mask.values] = 1
        signals[short_mask.values] = -1
        signals = pd.Series(signals, index=df.index, dtype=int)

        close = df["close"]
        sl_mult = float(p.get("sl_atr_mult", 1.0))
        tp_mult = float(p.get("tp_atr_mult", 1.2))
        sl_series = pd.Series(np.nan, index=df.index, dtype=float)
        tp_series = pd.Series(np.nan, index=df.index, dtype=float)
        lm = signals == 1; sm = signals == -1
        sl_series.loc[lm] = (close - sl_mult * a).loc[lm].values
        sl_series.loc[sm] = (close + sl_mult * a).loc[sm].values
        tp_series.loc[lm] = (close + tp_mult * a).loc[lm].values
        tp_series.loc[sm] = (close - tp_mult * a).loc[sm].values
        return signals, sl_series, tp_series

# ============================================================
# STRATEGY REGISTRY & TF ASSIGNMENT
# ============================================================
STRATEGY_REGISTRY = [
    (S30a_DOW_Seasonality,           ["1H", "4H"]),
    (S30b_MidWeek_MeanReversion,     ["1H", "4H"]),
    (S30c_Asian_Session_Fade,        ["15m", "30m"]),
    (S30d_NY_London_Overlap_Momentum,["15m", "30m"]),
    (S30e_Weekend_Gap_Fade,          ["30m", "1H"]),
    (S30f_Funding_Rate_Proxy_Cycle,  ["15m", "30m"]),
]

# ============================================================
# HELPERS
# ============================================================
def compute_monthly_freq(trades: List[Dict], days: int) -> float:
    if not trades or days <= 0: return 0.0
    months = days / 30.44
    return round(len(trades) / max(months, 1), 3)

def get_month_range_2026(month: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if month == 12:
        start = pd.Timestamp(f"2026-{month:02d}-01")
        end = pd.Timestamp(f"2027-01-01") - pd.Timedelta(seconds=1)
    else:
        start = pd.Timestamp(f"2026-{month:02d}-01")
        end = pd.Timestamp(f"2026-{month+1:02d}-01") - pd.Timedelta(seconds=1)
    return start, end

def month_pct_return(trades: List[Dict], month_start, month_end) -> float:
    m_trades = [t for t in trades if month_start <= t["exit_time"] <= month_end]
    if not m_trades: return 0.0
    return round(sum(t["pnl_pct"] for t in m_trades), 3)

# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    box("TASK 7: TIME-BASED ALPHA DISCOVERY ENGINE", C_CYAN)
    print(f"{C_CYAN}  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{C_CYAN}  Engine: v{ENGINE_VER}")
    print(f"{C_CYAN}  IS Window : {IS_START.date()} -> {IS_END.date()}")
    print(f"{C_MAG}  OOS Window: {OOS_START.date()} -> {OOS_END.date()}")
    print(f"{C_CYAN}  Strategies: {len(STRATEGY_REGISTRY)}")

    loader = MultiTFDataLoader(DATA_ROOT)
    engine = BacktestEngine()

    section(1, "AUTO-DISCOVERY OF SYMBOLS", C_CYAN)
    tfs_used = sorted(set(tf for _, tfs in STRATEGY_REGISTRY for tf in tfs))
    print(f"  Timeframes to scan: {tfs_used}")
    tf_symbols = {}
    for tf in tfs_used:
        syms = loader.available_symbols(tf)
        tf_symbols[tf] = syms
        print(f"    {tf:<4}: {len(syms)} symbols available")

    total_combos = 0
    for strat_cls, tfs in STRATEGY_REGISTRY:
        for tf in tfs:
            total_combos += len(tf_symbols.get(tf, []))
    print(f"\n  Total (strategy x symbol x TF) combos to scan: {total_combos}")

    # ================================================================
    # STAGE 1: IN-SAMPLE SCREENING
    # ================================================================
    section(2, "STAGE 1: IN-SAMPLE SCREENING (2022-2025)", C_MAG)
    is_results = []
    counter = 0
    err_count = 0
    skip_count = 0

    for strat_cls, tfs in STRATEGY_REGISTRY:
        strat_instance = strat_cls(engine=engine)
        strat_name = strat_cls.name
        for tf in tfs:
            symbols = tf_symbols.get(tf, [])
            print(f"\n  {C_CYAN}Strategy: {strat_name} | TF: {tf} | Symbols: {len(symbols)}{S_RS}")
            for sym in symbols:
                counter += 1
                try:
                    df = loader.load(sym, tf, IS_START, IS_END)
                    if df is None or len(df) < 200:
                        skip_count += 1
                        continue
                    result = strat_instance.backtest(df, strat_cls.default_params)
                    m = result["metrics"]
                    days_span = (df.index.max() - df.index.min()).days
                    m["monthly_freq"] = compute_monthly_freq(result["trades"], days_span)
                    entry = {
                        "strategy": strat_name, "symbol": sym, "tf": tf,
                        **m
                    }
                    is_results.append(entry)
                    if counter % 100 == 0:
                        elapsed_min = (time.time()-t0)/60
                        print(f"    Progress: {counter}/{total_combos} | Elapsed: {elapsed_min:.1f}m")
                except Exception as e:
                    err_count += 1
                    if err_count <= 5:
                        print(f"    {C_RED}[ERR] {strat_name}|{sym}|{tf}: {e}{S_RS}")

    print(f"\n  {C_GRN}[OK] IS screening complete: {len(is_results)} setups{S_RS}")
    print(f"  Errors: {err_count} | Skipped (insufficient data): {skip_count}")

    # Save all IS results
    is_df = pd.DataFrame(is_results)
    all_is_csv = os.path.join(RESULTS_ROOT, "task7_all_is_results.csv")
    is_df.to_csv(all_is_csv, index=False)
    print(f"  {C_GRN}[SAVED] {all_is_csv} ({len(is_df)} rows){S_RS}")

    # Apply IS gates
    section(3, "IS GATE FILTERING", C_MAG)
    if len(is_df) == 0:
        print(f"  {C_RED}[!] No IS results. Aborting.{S_RS}")
        return

    before = len(is_df)
    qualified = is_df[
        (is_df["total_trades"] >= IS_MIN_TRADES) &
        (is_df["win_rate_pct"] >= IS_MIN_WIN_RATE) &
        (is_df["profit_factor"] >= IS_MIN_PF) &
        (is_df["sharpe"] >= IS_MIN_SHARPE) &
        (is_df["max_drawdown_pct"] > IS_MAX_DD_LIMIT)
    ].copy()
    qualified = qualified.sort_values("sharpe", ascending=False).reset_index(drop=True)

    print(f"  Before gates: {before}")
    print(f"  IS gate criteria:")
    print(f"    Trades >= {IS_MIN_TRADES}")
    print(f"    Win Rate >= {IS_MIN_WIN_RATE}%")
    print(f"    PF >= {IS_MIN_PF}")
    print(f"    Sharpe >= {IS_MIN_SHARPE}")
    print(f"    MaxDD > {IS_MAX_DD_LIMIT}%")
    print(f"  {C_GRN}Qualified: {len(qualified)}{S_RS}")

    qual_csv = os.path.join(RESULTS_ROOT, "task7_qualified_is_candidates.csv")
    qualified.to_csv(qual_csv, index=False)
    print(f"  {C_GRN}[SAVED] {qual_csv}{S_RS}")

    if len(qualified) == 0:
        print(f"\n  {C_YEL}[!] No candidates passed IS gates. Consider relaxing thresholds.{S_RS}")
        return

    # ================================================================
    # STAGE 2: OOS 2026 VALIDATION
    # ================================================================
    section(4, "STAGE 2: OOS 2026 VALIDATION (FROZEN PARAMS)", C_MAG)
    strat_map = {cls.name: cls for cls, _ in STRATEGY_REGISTRY}
    oos_verified = []
    oos_all = []
    oos_tested = 0
    oos_pass = 0

    print(f"  Testing {len(qualified)} qualified setups on 2026 OOS...")
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
            oos_entry = {
                "strategy": strat_name, "symbol": sym, "tf": tf,
                "is_sharpe": round(row["sharpe"], 3),
                "is_win_rate": round(row["win_rate_pct"], 2),
                "is_pf": round(row["profit_factor"], 3),
                "is_trades": int(row["total_trades"]),
                "is_return": round(row["total_return_pct"], 2),
                "oos_sharpe": round(m["sharpe"], 3),
                "oos_win_rate": round(m["win_rate_pct"], 2),
                "oos_pf": round(m["profit_factor"], 3),
                "oos_trades": int(m["total_trades"]),
                "oos_return": round(m["total_return_pct"], 2),
                "oos_max_dd": round(m["max_drawdown_pct"], 2),
                "oos_monthly_freq": m["monthly_freq"],
                "oos_expectancy": round(m["expectancy"], 4),
                "trades_ref": result["trades"],
            }
            oos_all.append(oos_entry)

            # Apply OOS gates
            passes = (
                m["sharpe"] >= OOS_MIN_SHARPE and
                m["total_return_pct"] > OOS_MIN_RETURN and
                m["profit_factor"] >= OOS_MIN_PF and
                m["win_rate_pct"] >= OOS_MIN_WIN_RATE and
                m["total_trades"] >= OOS_MIN_TRADES and
                m["expectancy"] > 0
            )
            if passes:
                oos_pass += 1
                deg = m["sharpe"] / row["sharpe"] if row["sharpe"] > 0 else 0
                status = "STRONG_PASS" if deg >= 0.7 else "PASS"
                oos_entry["status"] = status
                oos_entry["degradation"] = round(deg, 3)
                oos_verified.append(oos_entry)
        except Exception as e:
            if oos_tested <= 5:
                print(f"    {C_RED}[ERR OOS] {strat_name}|{sym}|{tf}: {e}{S_RS}")

    print(f"\n  {C_GRN}OOS Tested: {oos_tested} | Passed: {oos_pass}{S_RS}")

    if len(oos_verified) == 0:
        print(f"\n  {C_YEL}[!] No setups passed OOS gates.{S_RS}")
        # Still save whatever we have
        pd.DataFrame(oos_all).drop(columns=["trades_ref"], errors="ignore").to_csv(
            os.path.join(RESULTS_ROOT, "task7_oos_verified_edges.csv"), index=False
        )
        return

    # Sort verified by OOS composite: (OOS Sharpe + PF + degradation) / 3
    for r in oos_verified:
        r["composite"] = round(
            (r["oos_sharpe"] * 30 + (r["oos_pf"] - 1) * 20 + r["oos_win_rate"] * 0.5 + r["degradation"] * 10)
        , 2)
    oos_verified.sort(key=lambda x: -x["composite"])

    # Save OOS verified
    oos_df = pd.DataFrame([{k:v for k,v in r.items() if k != "trades_ref"} for r in oos_verified])
    oos_csv = os.path.join(RESULTS_ROOT, "task7_oos_verified_edges.csv")
    oos_df.to_csv(oos_csv, index=False)
    print(f"  {C_GRN}[SAVED] {oos_csv} ({len(oos_df)} verified edges){S_RS}")

    # ================================================================
    # SECTION 5: EXECUTION FUNNEL SUMMARY
    # ================================================================
    section(5, "EXECUTION FUNNEL SUMMARY", C_CYAN)
    funnel = [
        ["1. Total combos scanned", total_combos],
        ["2. IS backtests completed", len(is_results)],
        ["3. Passed IS gates", len(qualified)],
        ["4. Tested on OOS 2026", oos_tested],
        ["5. Passed OOS gates", oos_pass],
        ["6. Yield rate (OOS-verified / total scanned)", f"{100*oos_pass/max(total_combos,1):.2f}%"],
    ]
    print(tabulate(funnel, headers=["Stage", "Count"], tablefmt="grid"))

    # ================================================================
    # SECTION 6: TOP 30 OOS-VERIFIED EDGES
    # ================================================================
    section(6, "TOP 30 OOS-VERIFIED EDGES", C_GRN)
    top_rows = []
    for i, r in enumerate(oos_verified[:30], 1):
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

    # ================================================================
    # SECTION 7: PORTFOLIO CONSTRUCTION (>=25 trades/month combined)
    # ================================================================
    section(7, "PORTFOLIO CONSTRUCTION (>= 25 combined trades/month)", C_MAG)
    # Greedy: pick top-composite edges avoiding same-symbol clustering (max 2 per symbol)
    portfolio = []
    combined_monthly_trades = 0
    sym_count = {}
    strat_count = {}
    MAX_PER_SYMBOL = 2
    MAX_PER_STRATEGY = 5
    TARGET_MONTHLY = 25

    for r in oos_verified:
        if sym_count.get(r["symbol"], 0) >= MAX_PER_SYMBOL: continue
        if strat_count.get(r["strategy"], 0) >= MAX_PER_STRATEGY: continue
        portfolio.append(r)
        combined_monthly_trades += r["oos_monthly_freq"]
        sym_count[r["symbol"]] = sym_count.get(r["symbol"], 0) + 1
        strat_count[r["strategy"]] = strat_count.get(r["strategy"], 0) + 1
        # Stop if we reach target monthly trades AND at least 5 edges
        if combined_monthly_trades >= TARGET_MONTHLY and len(portfolio) >= 5:
            break

    print(f"  Portfolio size: {len(portfolio)} edges")
    print(f"  Combined monthly trades: {combined_monthly_trades:.1f}")
    print(f"  Unique symbols: {len(sym_count)}")
    print(f"  Unique strategies: {len(strat_count)}")

    if portfolio:
        prt_rows = []
        for r in portfolio:
            prt_rows.append([
                r["strategy"][:22], r["symbol"][:14], r["tf"],
                f"{r['oos_sharpe']:.2f}", f"{r['oos_win_rate']:.1f}%",
                f"{r['oos_pf']:.2f}", r["oos_trades"], f"{r['oos_return']:.2f}%",
                f"{r['oos_monthly_freq']:.1f}"
            ])
        print("\n  Portfolio members:")
        print(tabulate(prt_rows,
            headers=["Strategy","Symbol","TF","OOS Sh","OOS Win%","OOS PF","OOS Tr","OOS Ret","Mo Fr"],
            tablefmt="simple"))

        # Compute portfolio-level combined stats from raw trades
        all_pf_trades = []
        for r in portfolio:
            all_pf_trades.extend(r["trades_ref"])
        if all_pf_trades:
            pnls = np.array([t["pnl"] for t in all_pf_trades])
            wins = (pnls > 0).sum()
            pf_win_rate = 100 * wins / len(pnls) if len(pnls)>0 else 0
            gp = pnls[pnls>0].sum(); gl = abs(pnls[pnls<0].sum())
            pf_pf = gp/gl if gl>0 else 999
            # Combined equity curve
            trades_by_time = sorted(all_pf_trades, key=lambda t: t["exit_time"])
            eq_values = [10000.0]
            eq_times = [OOS_START]
            eq = 10000.0
            for t in trades_by_time:
                eq += t["pnl"]
                eq_values.append(eq)
                eq_times.append(t["exit_time"])
            eq_series = pd.Series(eq_values, index=eq_times)
            eq_daily = eq_series.resample("D").last().ffill()
            dr = eq_daily.pct_change().dropna()
            port_sh = float(dr.mean()/dr.std()*np.sqrt(365)) if dr.std()>0 else 0
            peak = eq_daily.cummax(); dd = (eq_daily - peak)/peak
            port_dd = float(dd.min()*100)
            port_ret = (eq_series.iloc[-1]/eq_series.iloc[0] - 1) * 100

            print(f"\n  {C_GRN}{S_BR}Portfolio 2026 OOS Combined Metrics:{S_RS}")
            port_stats = [
                ["Combined Trades", len(all_pf_trades)],
                ["Combined Win Rate %", f"{pf_win_rate:.2f}"],
                ["Combined Profit Factor", f"{pf_pf:.3f}"],
                ["Combined Sharpe (daily-based)", f"{port_sh:.3f}"],
                ["Combined Max DD %", f"{port_dd:.2f}"],
                ["Combined Total Return %", f"{port_ret:.2f}"],
                ["Combined Monthly Trades", f"{combined_monthly_trades:.1f}"],
                ["Start Equity", f"${eq_series.iloc[0]:.2f}"],
                ["End Equity", f"${eq_series.iloc[-1]:.2f}"],
            ]
            print(tabulate(port_stats, headers=["Metric","Value"], tablefmt="grid"))

            # 2026 month-by-month
            print(f"\n  {C_CYAN}2026 Month-by-Month Portfolio Returns:{S_RS}")
            month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug"]
            monthly_rows = []
            for mi in range(1, 9):
                ms, me = get_month_range_2026(mi)
                m_trades = [t for t in all_pf_trades if ms <= t["exit_time"] <= me]
                if m_trades:
                    m_pnl_sum = sum(t["pnl"] for t in m_trades)
                    m_wins = sum(1 for t in m_trades if t["pnl"]>0)
                    m_wr = 100*m_wins/len(m_trades)
                    m_ret_pct = m_pnl_sum / 10000 * 100
                    col = C_GRN if m_ret_pct > 0 else C_RED
                else:
                    m_ret_pct = 0; m_wr = 0; col = C_YEL
                monthly_rows.append([
                    month_names[mi-1],
                    f"{col}{m_ret_pct:+.2f}%{S_RS}",
                    len(m_trades),
                    f"{m_wr:.1f}%" if m_trades else "-"
                ])
            print(tabulate(monthly_rows, headers=["Month","Return","Trades","Win%"], tablefmt="grid"))

        # Save portfolio
        pf_csv = os.path.join(RESULTS_ROOT, "task7_top_portfolio.csv")
        pd.DataFrame([{k:v for k,v in r.items() if k != "trades_ref"} for r in portfolio]).to_csv(pf_csv, index=False)
        print(f"\n  {C_GRN}[SAVED] {pf_csv}{S_RS}")

    # ================================================================
    # SECTION 8: STRATEGY-CATEGORY PERFORMANCE SUMMARY
    # ================================================================
    section(8, "PER-STRATEGY OOS PERFORMANCE SUMMARY", C_CYAN)
    strat_stats = {}
    for r in oos_verified:
        s = r["strategy"]
        if s not in strat_stats:
            strat_stats[s] = {"count":0,"avg_sharpe":[],"avg_pf":[],"avg_win":[],"avg_ret":[]}
        strat_stats[s]["count"] += 1
        strat_stats[s]["avg_sharpe"].append(r["oos_sharpe"])
        strat_stats[s]["avg_pf"].append(r["oos_pf"])
        strat_stats[s]["avg_win"].append(r["oos_win_rate"])
        strat_stats[s]["avg_ret"].append(r["oos_return"])
    strat_rows = []
    for s, d in strat_stats.items():
        strat_rows.append([
            s[:28], d["count"],
            f"{np.mean(d['avg_sharpe']):.2f}",
            f"{np.mean(d['avg_pf']):.2f}",
            f"{np.mean(d['avg_win']):.1f}%",
            f"{np.mean(d['avg_ret']):.2f}%",
        ])
    strat_rows.sort(key=lambda x: -float(x[2]))
    print(tabulate(strat_rows,
        headers=["Strategy","Verified Edges","AvgOOSSh","AvgOOSPF","AvgOOSWin","AvgOOSRet"],
        tablefmt="grid"))

    # ================================================================
    # SECTION 9: EXECUTIVE SUMMARY
    # ================================================================
    section(9, "EXECUTIVE SUMMARY", C_GRN)
    print(f"  Total setups scanned          : {total_combos}")
    print(f"  IS-qualified setups           : {len(qualified)}  ({100*len(qualified)/max(total_combos,1):.1f}%)")
    print(f"  OOS 2026-verified edges       : {len(oos_verified)}  ({100*len(oos_verified)/max(total_combos,1):.2f}%)")
    print(f"  Portfolio size                : {len(portfolio)}")
    print(f"  Portfolio combined monthly trades: {combined_monthly_trades:.1f}")

    if len(oos_verified) >= 5:
        print(f"\n  {C_GRN}{S_BR}Verdict: TIME-BASED EDGES SCALE. Deploy filtered portfolio to paper trading.{S_RS}")
    elif len(oos_verified) >= 1:
        print(f"\n  {C_YEL}Verdict: LIMITED EDGES SURVIVE. Consider expanded scan or looser gates.{S_RS}")
    else:
        print(f"\n  {C_RED}Verdict: NO EDGES SURVIVED. Time-based strategies did not scale on this dataset.{S_RS}")

    print(f"\n  Next Steps:")
    print(f"    1. Review portfolio_top_portfolio.csv for deployment candidates")
    print(f"    2. Run walk-forward on rolling 6-month windows for further robustness")
    print(f"    3. Paper trade top 5 edges on Bybit testnet (30 days min)")
    print(f"    4. Monitor live slippage on session-boundary trades (may exceed 0.03%)")

    elapsed = time.time() - t0
    print()
    box(f"TASK 7 COMPLETE - Runtime: {elapsed/60:.2f} minutes", C_GRN)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C_RED}Interrupted{S_RS}")
    except Exception as e:
        print(f"\n{C_RED}FATAL: {e}{S_RS}")
        traceback.print_exc()
