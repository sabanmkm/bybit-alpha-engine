"""
TASK 7.2: HIGH-FREQUENCY TIME-BASED PORTFOLIO BUILDER
Lenient IS gates (4/5 must pass) -> strict OOS -> portfolio construction.
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
        MultiTFDataLoader, BacktestEngine, Strategy,
        RSI, MACD, ATR, EMA, SMA, ADX
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

# LENIENT IS gates - need 4 out of 5 to pass
IS_GATES = {
    "trades_min": 30,
    "win_rate_min": 48.0,
    "pf_min": 1.15,
    "sharpe_min": 0.50,
    "dd_max": -28.0,
}
IS_GATES_REQUIRED = 4  # pass at least 4 of 5

# STRICT OOS gates - must pass ALL
OOS_GATES = {
    "win_rate_min": 50.0,
    "sharpe_min": 0.50,
    "pf_min": 1.15,
    "trades_min": 12,
    "return_min": 0.0,
}

# Portfolio construction targets
PORTFOLIO_TARGETS = {
    "min_combined_monthly_trades": 25,
    "min_combined_win_rate": 50.0,
    "min_combined_sharpe": 1.2,
    "max_combined_dd": -15.0,
    "max_per_symbol": 3,
    "max_per_strategy": 6,
    "target_portfolio_size": 10,
}

# Prefer 15m/30m for high frequency
TF_WEIGHTS = {"15m": 4, "30m": 4, "1H": 2, "4H": 1}

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
# HELPERS FOR SIGNAL/SL/TP BUILDING (dtype-safe)
# ============================================================
def _build_signals(df, long_mask, short_mask):
    n = len(df)
    arr = np.zeros(n, dtype=int)
    arr[long_mask.values] = 1
    arr[short_mask.values] = -1
    return pd.Series(arr, index=df.index, dtype=int)

def _build_sl_tp(df, signals, atr_series, sl_mult, tp_mult):
    close = df["close"]
    sl = pd.Series(np.nan, index=df.index, dtype=float)
    tp = pd.Series(np.nan, index=df.index, dtype=float)
    lm = signals == 1; sm = signals == -1
    sl.loc[lm] = (close - sl_mult * atr_series).loc[lm].values
    sl.loc[sm] = (close + sl_mult * atr_series).loc[sm].values
    tp.loc[lm] = (close + tp_mult * atr_series).loc[lm].values
    tp.loc[sm] = (close - tp_mult * atr_series).loc[sm].values
    return sl, tp

# ============================================================
# STRATEGIES (pandas 2.2+ compatible)
# ============================================================
class S30a_DOW_Seasonality(Strategy):
    name = "S30a_DOW_Seasonality"
    category = "Seasonal"
    default_params = {"entry_day_long":4,"entry_day_short":3,"rsi_period":14,
                      "rsi_long_threshold":50.0,"rsi_short_threshold":50.0,
                      "sl_atr_mult":1.5,"tp_atr_mult":2.5,"atr_period":14}
    def generate_signals(self, df, params):
        p = params
        dow = pd.Series(df.index.dayofweek, index=df.index)
        r = RSI(df["close"], int(p["rsi_period"]))
        a = ATR(df, int(p["atr_period"]))
        long_mask = ((dow == int(p["entry_day_long"])) & (r > p["rsi_long_threshold"])).fillna(False).astype(bool)
        short_mask = ((dow == int(p["entry_day_short"])) & (r < p["rsi_short_threshold"])).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class S30b_MidWeek_MeanReversion(Strategy):
    name = "S30b_MidWeek_MeanReversion"
    category = "Seasonal_MeanRev"
    default_params = {"sma_period":20,"atr_period":14,"deviation_atr":1.5,
                      "sl_atr_mult":2.0,"tp_atr_mult":1.5}
    def generate_signals(self, df, params):
        p = params
        sma = SMA(df["close"], int(p["sma_period"]))
        a = ATR(df, int(p["atr_period"]))
        dev = float(p["deviation_atr"])
        dow = pd.Series(df.index.dayofweek, index=df.index)
        dist = (df["close"] - sma)
        far_above = (dist > dev * a).fillna(False).astype(bool)
        far_below = (dist < -dev * a).fillna(False).astype(bool)
        trigger = ((dow == 1) | (dow == 2)).astype(bool)
        short_mask = (far_above & trigger).fillna(False).astype(bool)
        long_mask = (far_below & trigger).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class S30c_Asian_Session_Fade(Strategy):
    name = "S30c_Asian_Session_Fade"
    category = "Session_Fade"
    default_params = {"session_start_hour":0,"session_end_hour":8,
                      "breakout_start_hour":3,"breakout_end_hour":6,
                      "range_bars_hours":2,"breakout_pct_atr":0.3,
                      "sl_atr_mult":1.0,"tp_atr_mult":1.5,"atr_period":14}
    def generate_signals(self, df, params):
        p = params
        a = ATR(df, int(p["atr_period"]))
        hour = pd.Series(df.index.hour, index=df.index)
        date = pd.Series(df.index.date, index=df.index)
        breakout_start = int(p["breakout_start_hour"])
        breakout_end = int(p["breakout_end_hour"])
        range_hours = int(p["range_bars_hours"])
        session_start = int(p["session_start_hour"])
        first_end = session_start + range_hours
        in_first = (hour >= session_start) & (hour < first_end)
        df_tmp = df.copy(); df_tmp["date"] = date.values; df_tmp["in_first"] = in_first.values
        grp = df_tmp[df_tmp["in_first"]].groupby("date")
        first_high = grp["high"].max(); first_low = grp["low"].min()
        rh = pd.Series(date.map(first_high).values, index=df.index)
        rl = pd.Series(date.map(first_low).values, index=df.index)
        in_win = (hour >= breakout_start) & (hour < breakout_end)
        vol_avg = df["volume"].rolling(3).mean().shift(1)
        vol_dec = df["volume"] < vol_avg
        bp = float(p["breakout_pct_atr"])
        up_break = df["close"] > (rh + bp * a)
        dn_break = df["close"] < (rl - bp * a)
        short_mask = (in_win & up_break & vol_dec).fillna(False).astype(bool)
        long_mask = (in_win & dn_break & vol_dec).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class S30d_NY_London_Overlap_Momentum(Strategy):
    name = "S30d_NY_London_Overlap_Momentum"
    category = "Session_Breakout"
    default_params = {"london_start_hour":8,"london_end_hour":12,
                      "trade_start_hour":12,"trade_end_hour":16,
                      "atr_expansion_ratio":1.2,"sl_atr_mult":1.5,
                      "tp_atr_mult":2.5,"atr_period":14}
    def generate_signals(self, df, params):
        p = params
        a = ATR(df, int(p["atr_period"]))
        hour = pd.Series(df.index.hour, index=df.index)
        date = pd.Series(df.index.date, index=df.index)
        london_start = int(p["london_start_hour"]); london_end = int(p["london_end_hour"])
        trade_start = int(p["trade_start_hour"]); trade_end = int(p["trade_end_hour"])
        df_tmp = df.copy(); df_tmp["date"] = date.values
        in_london = (hour >= london_start) & (hour < london_end)
        df_tmp["in_london"] = in_london.values
        grp = df_tmp[df_tmp["in_london"]].groupby("date")
        london_high = grp["high"].max(); london_low = grp["low"].min()
        lh = pd.Series(date.map(london_high).values, index=df.index)
        ll = pd.Series(date.map(london_low).values, index=df.index)
        atr_ref = a.shift(4).bfill()
        atr_exp = a > float(p["atr_expansion_ratio"]) * atr_ref
        in_trade = (hour >= trade_start) & (hour < trade_end)
        long_break = df["close"] > lh
        short_break = df["close"] < ll
        long_mask = (in_trade & long_break & atr_exp).fillna(False).astype(bool)
        short_mask = (in_trade & short_break & atr_exp).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class S30e_Weekend_Gap_Fade(Strategy):
    name = "S30e_Weekend_Gap_Fade"
    category = "Weekend_Gap"
    default_params = {"friday_reference_hour":20,"sunday_window_start_hour":18,
                      "sunday_window_end_hour":23,"gap_threshold_atr":1.5,
                      "sl_atr_mult":2.0,"tp_atr_mult":1.5,"atr_period":14}
    def generate_signals(self, df, params):
        p = params
        a = ATR(df, int(p["atr_period"]))
        dow = pd.Series(df.index.dayofweek, index=df.index)
        hour = pd.Series(df.index.hour, index=df.index)
        fri_hr = int(p["friday_reference_hour"])
        sun_start = int(p["sunday_window_start_hour"])
        sun_end = int(p["sunday_window_end_hour"])
        fri_ref = pd.Series(np.nan, index=df.index)
        fmask = (dow == 4) & (hour == fri_hr)
        fri_ref[fmask] = df["close"][fmask].values
        fri_ref = fri_ref.ffill()
        in_sun = (dow == 6) & (hour >= sun_start) & (hour <= sun_end)
        gap = df["close"] - fri_ref
        thr = float(p["gap_threshold_atr"]) * a
        gap_up = gap > thr; gap_dn = gap < -thr
        short_mask = (in_sun & gap_up).fillna(False).astype(bool)
        long_mask = (in_sun & gap_dn).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class S30f_Funding_Rate_Proxy_Cycle(Strategy):
    name = "S30f_Funding_Rate_Proxy_Cycle"
    category = "Funding_Arbitrage"
    default_params = {"settlement_hours":[0,8,16],"pre_settlement_minutes":45,
                      "rsi_period":14,"rsi_overbought":70,"rsi_oversold":30,
                      "sl_atr_mult":1.0,"tp_atr_mult":1.2,"atr_period":14}
    def generate_signals(self, df, params):
        p = params
        a = ATR(df, int(p["atr_period"]))
        r = RSI(df["close"], int(p["rsi_period"]))
        hour = pd.Series(df.index.hour, index=df.index)
        minute = pd.Series(df.index.minute, index=df.index)
        pre_min = int(p["pre_settlement_minutes"])
        pre_mask = pd.Series(False, index=df.index)
        for sh in p["settlement_hours"]:
            pre_hour = (sh - 1) % 24
            in_w = (hour == pre_hour) & (minute >= (60 - pre_min))
            pre_mask = pre_mask | in_w
        ob = float(p["rsi_overbought"]); os_ = float(p["rsi_oversold"])
        short_mask = (pre_mask & (r > ob)).fillna(False).astype(bool)
        long_mask = (pre_mask & (r < os_)).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

STRATEGY_REGISTRY = [
    (S30a_DOW_Seasonality,           ["15m","30m","1H","4H"]),
    (S30b_MidWeek_MeanReversion,     ["15m","30m","1H","4H"]),
    (S30c_Asian_Session_Fade,        ["15m","30m"]),
    (S30d_NY_London_Overlap_Momentum,["15m","30m","1H"]),
    (S30e_Weekend_Gap_Fade,          ["15m","30m","1H"]),
    (S30f_Funding_Rate_Proxy_Cycle,  ["15m","30m"]),
]

def monthly_freq(trades, days):
    if not trades or days <= 0: return 0.0
    return round(len(trades) / max(days / 30.44, 1), 3)

def month_range_2026(m):
    if m == 12:
        s = pd.Timestamp(f"2026-{m:02d}-01")
        e = pd.Timestamp("2027-01-01") - pd.Timedelta(seconds=1)
    else:
        s = pd.Timestamp(f"2026-{m:02d}-01")
        e = pd.Timestamp(f"2026-{m+1:02d}-01") - pd.Timedelta(seconds=1)
    return s, e

def passes_is_lenient(row):
    """Return count of gates passed (out of 5)."""
    c = 0
    if row["total_trades"] >= IS_GATES["trades_min"]: c += 1
    if row["win_rate_pct"] >= IS_GATES["win_rate_min"]: c += 1
    if row["profit_factor"] >= IS_GATES["pf_min"]: c += 1
    if row["sharpe"] >= IS_GATES["sharpe_min"]: c += 1
    if row["max_drawdown_pct"] > IS_GATES["dd_max"]: c += 1
    return c

def passes_oos_strict(m):
    return (
        m["win_rate_pct"] >= OOS_GATES["win_rate_min"] and
        m["sharpe"] >= OOS_GATES["sharpe_min"] and
        m["profit_factor"] >= OOS_GATES["pf_min"] and
        m["total_trades"] >= OOS_GATES["trades_min"] and
        m["total_return_pct"] > OOS_GATES["return_min"]
    )

# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    box("TASK 7.2: HIGH-FREQUENCY TIME-BASED PORTFOLIO BUILDER", C_CYAN)
    print(f"{C_CYAN}  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{C_CYAN}  Engine: v{ENGINE_VER}")

    loader = MultiTFDataLoader(DATA_ROOT)
    engine = BacktestEngine()

    section(1, "SYMBOL UNIVERSE DISCOVERY", C_CYAN)
    # Union of symbols across all TFs used
    tfs_used = sorted(set(tf for _, tfs in STRATEGY_REGISTRY for tf in tfs))
    all_syms_set = set()
    for tf in tfs_used:
        for s in loader.available_symbols(tf):
            all_syms_set.add(s)
    universe = sorted(all_syms_set)
    print(f"  Total unique symbols in dataset: {len(universe)}")
    print(f"  Timeframes to scan: {tfs_used}")

    total_combos = 0
    for strat_cls, tfs in STRATEGY_REGISTRY:
        for tf in tfs:
            avail = [s for s in universe if s in loader.available_symbols(tf)]
            total_combos += len(avail)
    print(f"  Total (strategy x symbol x TF) combos: {total_combos}")

    # SECTION 2: IS Screening
    section(2, "IN-SAMPLE SCREENING (2022-2025) - LENIENT GATES", C_MAG)
    is_results = []
    counter = 0; err_count = 0; skip_count = 0
    t_scan = time.time()

    for strat_cls, tfs in STRATEGY_REGISTRY:
        strat_inst = strat_cls(engine=engine)
        for tf in tfs:
            avail = [s for s in universe if s in loader.available_symbols(tf)]
            print(f"\n  {C_CYAN}{strat_cls.name} | {tf} | {len(avail)} symbols{S_RS}")
            for sym in avail:
                counter += 1
                try:
                    df = loader.load(sym, tf, IS_START, IS_END)
                    if df is None or len(df) < 300:
                        skip_count += 1; continue
                    res = strat_inst.backtest(df, strat_cls.default_params)
                    m = res["metrics"]
                    days = (df.index.max() - df.index.min()).days
                    m["monthly_freq"] = monthly_freq(res["trades"], days)
                    is_results.append({
                        "strategy": strat_cls.name, "symbol": sym, "tf": tf, **m
                    })
                    if counter % 250 == 0:
                        elap = (time.time() - t0)/60
                        eta = (elap / counter) * (total_combos - counter)
                        n_qual = sum(1 for r in is_results if passes_is_lenient(r) >= IS_GATES_REQUIRED)
                        print(f"    Progress: {counter}/{total_combos} | Elapsed {elap:.1f}m | ETA {eta:.1f}m | Qualified {n_qual}")
                except Exception as e:
                    err_count += 1
                    if err_count <= 3:
                        print(f"    {C_RED}[ERR] {strat_cls.name}|{sym}|{tf}: {e}{S_RS}")

    print(f"\n  {C_GRN}[OK] IS scan complete: {len(is_results)} setups, {err_count} errors, {skip_count} skipped{S_RS}")

    is_df = pd.DataFrame(is_results)
    all_csv = os.path.join(RESULTS_ROOT, "task7.2_high_frequency_results.csv")
    is_df.to_csv(all_csv, index=False)
    print(f"  {C_GRN}[SAVED] {all_csv}{S_RS}")

    if len(is_df) == 0:
        print(f"  {C_RED}[!] No results. Aborting.{S_RS}"); return

    # SECTION 3: Apply lenient IS gates (4 of 5 must pass)
    section(3, "IS GATE FILTERING (4 of 5 must pass)", C_MAG)
    is_df["gates_passed"] = is_df.apply(passes_is_lenient, axis=1)
    qualified = is_df[is_df["gates_passed"] >= IS_GATES_REQUIRED].copy()
    qualified = qualified.sort_values(["sharpe","monthly_freq"], ascending=[False,False]).reset_index(drop=True)

    print(f"  Lenient criteria (need {IS_GATES_REQUIRED}/5):")
    print(f"    Trades >= {IS_GATES['trades_min']}")
    print(f"    Win Rate >= {IS_GATES['win_rate_min']}%")
    print(f"    PF >= {IS_GATES['pf_min']}")
    print(f"    Sharpe >= {IS_GATES['sharpe_min']}")
    print(f"    MaxDD > {IS_GATES['dd_max']}%")
    print(f"\n  IS total: {len(is_df)} -> Qualified: {C_GRN}{len(qualified)}{S_RS}")

    # Gate distribution histogram
    gate_dist = is_df["gates_passed"].value_counts().sort_index()
    print(f"\n  Gate-pass distribution:")
    for gp, cnt in gate_dist.items():
        marker = f"{C_GRN}<-- QUALIFIED{S_RS}" if gp >= IS_GATES_REQUIRED else ""
        print(f"    {int(gp)}/5 gates: {cnt} setups {marker}")

    if len(qualified) == 0:
        print(f"  {C_RED}[!] Zero setups met lenient IS gates. Aborting.{S_RS}")
        return

    # SECTION 4: OOS validation
    section(4, "OOS 2026 VALIDATION (STRICT GATES, FROZEN PARAMS)", C_MAG)
    strat_map = {cls.name: cls for cls, _ in STRATEGY_REGISTRY}
    oos_verified = []
    oos_all = []
    oos_tested = 0

    print(f"  Testing {len(qualified)} IS-qualified setups on 2026...")
    for i, row in qualified.iterrows():
        oos_tested += 1
        sname = row["strategy"]; sym = row["symbol"]; tf = row["tf"]
        cls = strat_map.get(sname)
        if cls is None: continue
        try:
            strat_inst = cls(engine=engine)
            df_oos = loader.load(sym, tf, OOS_START, OOS_END)
            if df_oos is None or len(df_oos) < 50: continue
            res = strat_inst.backtest(df_oos, cls.default_params)
            m = res["metrics"]
            days = (df_oos.index.max() - df_oos.index.min()).days
            m["monthly_freq"] = monthly_freq(res["trades"], days)
            entry = {
                "strategy": sname, "symbol": sym, "tf": tf,
                "is_sharpe": round(row["sharpe"], 3),
                "is_win_rate": round(row["win_rate_pct"], 2),
                "is_pf": round(row["profit_factor"], 3),
                "is_trades": int(row["total_trades"]),
                "is_return": round(row["total_return_pct"], 2),
                "is_monthly_freq": round(row["monthly_freq"], 3),
                "is_gates_passed": int(row["gates_passed"]),
                "oos_sharpe": round(m["sharpe"], 3),
                "oos_win_rate": round(m["win_rate_pct"], 2),
                "oos_pf": round(m["profit_factor"], 3),
                "oos_trades": int(m["total_trades"]),
                "oos_return": round(m["total_return_pct"], 2),
                "oos_max_dd": round(m["max_drawdown_pct"], 2),
                "oos_monthly_freq": round(m["monthly_freq"], 3),
                "oos_expectancy": round(m["expectancy"], 4),
                "trades_ref": res["trades"],
            }
            oos_all.append(entry)
            if passes_oos_strict(m):
                deg = m["sharpe"] / row["sharpe"] if row["sharpe"] > 0 else 0
                entry["degradation"] = round(deg, 3)
                entry["status"] = "STRONG_PASS" if deg >= 0.7 else "PASS"
                entry["freq_score"] = round(m["monthly_freq"] * 10 + m["sharpe"] * 20, 2)
                oos_verified.append(entry)
        except Exception as e:
            if oos_tested <= 3:
                print(f"    {C_RED}[ERR OOS] {sname}|{sym}|{tf}: {e}{S_RS}")

    oos_verified.sort(key=lambda x: -x["freq_score"])
    print(f"  Tested: {oos_tested} | {C_GRN}Passed OOS: {len(oos_verified)}{S_RS}")

    # SECTION 5: Funnel
    section(5, "EXECUTION FUNNEL", C_CYAN)
    funnel = [
        ["Total combos scanned", total_combos],
        ["IS backtests completed", len(is_results)],
        ["Passed IS gates (>=4/5)", len(qualified)],
        ["Tested on OOS 2026", oos_tested],
        ["Passed OOS strict gates", len(oos_verified)],
        ["Overall yield", f"{100*len(oos_verified)/max(total_combos,1):.2f}%"],
    ]
    print(tabulate(funnel, headers=["Stage","Count"], tablefmt="grid"))

    if len(oos_verified) == 0:
        print(f"\n  {C_RED}[!] No OOS-verified edges. IS-only data saved.{S_RS}")
        if oos_all:
            oos_csv = os.path.join(RESULTS_ROOT, "task7.2_all_oos_tested.csv")
            pd.DataFrame([{k:v for k,v in r.items() if k != "trades_ref"} for r in oos_all]).to_csv(oos_csv, index=False)
            print(f"  {C_GRN}[SAVED] All OOS-tested (not passing): {oos_csv}{S_RS}")
        return

    # SECTION 6: Top 30 OOS-verified edges
    section(6, "TOP 30 OOS-VERIFIED EDGES (RANKED BY FREQUENCY SCORE)", C_GRN)
    top_rows = []
    for i, r in enumerate(oos_verified[:30], 1):
        st_col = C_GRN if r["status"] == "STRONG_PASS" else C_YEL
        top_rows.append([
            i, r["strategy"][:22], r["symbol"][:14], r["tf"],
            f"{r['is_sharpe']:.2f}", f"{r['oos_sharpe']:.2f}",
            f"{r['oos_win_rate']:.1f}%", f"{r['oos_pf']:.2f}",
            r["oos_trades"], f"{r['oos_return']:.2f}%",
            f"{r['oos_monthly_freq']:.1f}",
            f"{r['freq_score']:.1f}",
            f"{st_col}{r['status']}{S_RS}"
        ])
    print(tabulate(top_rows,
        headers=["#","Strategy","Symbol","TF","IS Sh","OOS Sh","OOS Win%","OOS PF","Tr","Ret","MoFr","FrScore","Status"],
        tablefmt="grid"))

    # Save all OOS verified
    oos_csv = os.path.join(RESULTS_ROOT, "task7.2_all_oos_verified.csv")
    pd.DataFrame([{k:v for k,v in r.items() if k != "trades_ref"} for r in oos_verified]).to_csv(oos_csv, index=False)
    print(f"\n  {C_GRN}[SAVED] {oos_csv}{S_RS}")

    # SECTION 7: Portfolio construction targeting 25+ combined monthly trades
    section(7, "PORTFOLIO CONSTRUCTION", C_MAG)
    portfolio = []
    combined_monthly = 0.0
    sym_count = {}; strat_count = {}
    tgt_size = PORTFOLIO_TARGETS["target_portfolio_size"]

    for r in oos_verified:
        if sym_count.get(r["symbol"], 0) >= PORTFOLIO_TARGETS["max_per_symbol"]: continue
        if strat_count.get(r["strategy"], 0) >= PORTFOLIO_TARGETS["max_per_strategy"]: continue
        portfolio.append(r)
        combined_monthly += r["oos_monthly_freq"]
        sym_count[r["symbol"]] = sym_count.get(r["symbol"], 0) + 1
        strat_count[r["strategy"]] = strat_count.get(r["strategy"], 0) + 1
        if len(portfolio) >= tgt_size and combined_monthly >= PORTFOLIO_TARGETS["min_combined_monthly_trades"]:
            break

    print(f"  Portfolio size          : {len(portfolio)}")
    print(f"  Combined monthly trades : {combined_monthly:.1f}")
    print(f"  Target monthly trades   : {PORTFOLIO_TARGETS['min_combined_monthly_trades']}+")
    print(f"  Unique symbols          : {len(sym_count)}")
    print(f"  Unique strategies       : {len(strat_count)}")

    if not portfolio:
        print(f"  {C_RED}[!] Empty portfolio. Aborting.{S_RS}")
        return

    prt_rows = []
    for r in portfolio:
        prt_rows.append([
            r["strategy"][:22], r["symbol"][:14], r["tf"],
            f"{r['oos_sharpe']:.2f}", f"{r['oos_win_rate']:.1f}%",
            f"{r['oos_pf']:.2f}", r["oos_trades"], f"{r['oos_return']:.2f}%",
            f"{r['oos_monthly_freq']:.1f}", f"{r['freq_score']:.1f}"
        ])
    print("\n  Portfolio Members:")
    print(tabulate(prt_rows,
        headers=["Strategy","Symbol","TF","OOS Sh","Win%","PF","Trades","Return","MoFr","FrScore"],
        tablefmt="simple"))

    # Combined stats
    all_trades = []
    for r in portfolio: all_trades.extend(r["trades_ref"])
    if not all_trades:
        print(f"  {C_RED}[!] No trades in portfolio.{S_RS}")
        return

    pnls = np.array([t["pnl"] for t in all_trades])
    wins = (pnls > 0).sum()
    pf_wr = 100 * wins / len(pnls) if len(pnls) > 0 else 0
    gp = pnls[pnls>0].sum(); gl = abs(pnls[pnls<0].sum())
    pf_pf = gp/gl if gl > 0 else 999
    pf_aw = pnls[pnls>0].mean() if wins > 0 else 0
    pf_al = abs(pnls[pnls<0].mean()) if (pnls<0).sum() > 0 else 0
    pf_expect = (wins/len(pnls))*pf_aw - (1-wins/len(pnls))*pf_al

    trades_time = sorted(all_trades, key=lambda t: t["exit_time"])
    eq_v = [10000.0]; eq_t = [OOS_START]
    eq = 10000.0
    for t in trades_time:
        eq += t["pnl"]; eq_v.append(eq); eq_t.append(t["exit_time"])
    eq_s = pd.Series(eq_v, index=eq_t)
    eq_d = eq_s.resample("D").last().ffill()
    dr = eq_d.pct_change().dropna()
    port_sh = float(dr.mean()/dr.std()*np.sqrt(365)) if dr.std() > 0 else 0
    peak = eq_d.cummax(); dd = (eq_d - peak)/peak
    port_dd = float(dd.min()*100)
    port_ret = (eq_s.iloc[-1]/eq_s.iloc[0] - 1) * 100

    # Verify portfolio meets ALL targets
    tgt_check = [
        ["Combined Trades", len(all_trades), "", ""],
        ["Combined Win Rate %", f"{pf_wr:.2f}", f">={PORTFOLIO_TARGETS['min_combined_win_rate']}",
         f"{C_GRN}OK{S_RS}" if pf_wr >= PORTFOLIO_TARGETS["min_combined_win_rate"] else f"{C_RED}FAIL{S_RS}"],
        ["Combined PF", f"{pf_pf:.3f}", ">=1.15",
         f"{C_GRN}OK{S_RS}" if pf_pf >= 1.15 else f"{C_RED}FAIL{S_RS}"],
        ["Combined Sharpe", f"{port_sh:.3f}", f">={PORTFOLIO_TARGETS['min_combined_sharpe']}",
         f"{C_GRN}OK{S_RS}" if port_sh >= PORTFOLIO_TARGETS["min_combined_sharpe"] else f"{C_RED}FAIL{S_RS}"],
        ["Combined Max DD %", f"{port_dd:.2f}", f">{PORTFOLIO_TARGETS['max_combined_dd']}",
         f"{C_GRN}OK{S_RS}" if port_dd > PORTFOLIO_TARGETS["max_combined_dd"] else f"{C_RED}FAIL{S_RS}"],
        ["Combined Total Return %", f"{port_ret:.2f}", ">0",
         f"{C_GRN}OK{S_RS}" if port_ret > 0 else f"{C_RED}FAIL{S_RS}"],
        ["Combined Expectancy $", f"{pf_expect:.4f}", ">0",
         f"{C_GRN}OK{S_RS}" if pf_expect > 0 else f"{C_RED}FAIL{S_RS}"],
        ["Combined Monthly Trades", f"{combined_monthly:.1f}", f">={PORTFOLIO_TARGETS['min_combined_monthly_trades']}",
         f"{C_GRN}OK{S_RS}" if combined_monthly >= PORTFOLIO_TARGETS["min_combined_monthly_trades"] else f"{C_RED}FAIL{S_RS}"],
        ["Start Equity", f"${eq_s.iloc[0]:.2f}", "", ""],
        ["End Equity", f"${eq_s.iloc[-1]:.2f}", "", ""],
    ]
    print(f"\n  {C_GRN}{S_BR}Portfolio 2026 OOS Combined Metrics (vs Targets):{S_RS}")
    print(tabulate(tgt_check, headers=["Metric","Value","Target","Pass"], tablefmt="grid"))

    all_targets_met = (pf_wr >= PORTFOLIO_TARGETS["min_combined_win_rate"]
                       and port_sh >= PORTFOLIO_TARGETS["min_combined_sharpe"]
                       and port_dd > PORTFOLIO_TARGETS["max_combined_dd"]
                       and combined_monthly >= PORTFOLIO_TARGETS["min_combined_monthly_trades"]
                       and pf_pf >= 1.15 and port_ret > 0)

    # SECTION 8: Month-by-month heatmap
    section(8, "2026 MONTH-BY-MONTH PORTFOLIO HEATMAP (JAN-AUG)", C_CYAN)
    mo_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug"]
    monthly_rows = []
    profitable_mo = 0
    for mi in range(1, 9):
        ms, me = month_range_2026(mi)
        m_trades = [t for t in all_trades if ms <= t["exit_time"] <= me]
        if m_trades:
            m_pnl = sum(t["pnl"] for t in m_trades)
            m_wins = sum(1 for t in m_trades if t["pnl"] > 0)
            m_wr = 100 * m_wins / len(m_trades)
            m_ret = m_pnl / 10000 * 100
            col = C_GRN if m_ret > 0 else C_RED
            if m_ret > 0: profitable_mo += 1
        else:
            m_ret = 0; m_wr = 0; col = C_YEL
        monthly_rows.append([
            mo_names[mi-1],
            f"{col}{m_ret:+.2f}%{S_RS}",
            len(m_trades),
            f"{m_wr:.1f}%" if m_trades else "-"
        ])
    print(tabulate(monthly_rows, headers=["Month","Return","Trades","Win%"], tablefmt="grid"))
    print(f"\n  Profitable months: {profitable_mo}/8 ({100*profitable_mo/8:.1f}%)")

    # SECTION 9: Executive recommendation with weights
    section(9, "EXECUTIVE RECOMMENDATION & ALLOCATION WEIGHTS", C_GRN)
    if all_targets_met:
        verdict = f"{C_GRN}{S_BR}GO - Portfolio meets ALL user targets. Deploy to paper trading.{S_RS}"
    else:
        verdict = f"{C_YEL}CONDITIONAL - Some targets missed. Deploy top-scoring subset only.{S_RS}"
    print(f"\n  Verdict: {verdict}\n")

    # Frequency-score weighted allocation
    total_score = sum(r["freq_score"] for r in portfolio)
    weight_rows = []
    for r in portfolio:
        w = r["freq_score"] / total_score * 100 if total_score > 0 else 0
        weight_rows.append([
            r["strategy"][:22], r["symbol"][:14], r["tf"],
            f"{r['oos_sharpe']:.2f}", f"{r['oos_win_rate']:.1f}%",
            f"{r['oos_monthly_freq']:.1f}", f"{r['freq_score']:.1f}",
            f"{w:.2f}%"
        ])
    print("  Recommended Frequency-Score-Weighted Allocation:")
    print(tabulate(weight_rows,
        headers=["Strategy","Symbol","TF","OOS Sh","Win%","MoFr","FrScore","Weight"],
        tablefmt="grid"))

    print(f"\n  {C_CYAN}Deployment Instructions:{S_RS}")
    print(f"    1. Apply weights as position-size multipliers per edge")
    print(f"    2. Start Bybit testnet paper trading immediately")
    print(f"    3. Run for 30+ days before live capital deployment")
    print(f"    4. Portfolio-level equity stop-loss at -10% from HWM")
    print(f"    5. Re-run walk-forward monthly on rolling window")

    # Save portfolio CSV with weights
    pf_csv_rows = []
    for r in portfolio:
        w = r["freq_score"] / total_score * 100 if total_score > 0 else 0
        row = {k: v for k, v in r.items() if k != "trades_ref"}
        row["allocation_weight_pct"] = round(w, 3)
        pf_csv_rows.append(row)
    pf_csv = os.path.join(RESULTS_ROOT, "task7.2_final_portfolio.csv")
    pd.DataFrame(pf_csv_rows).to_csv(pf_csv, index=False)
    print(f"\n  {C_GRN}[SAVED] {pf_csv}{S_RS}")

    elapsed = time.time() - t0
    print()
    box(f"TASK 7.2 COMPLETE - Runtime: {elapsed/60:.2f} minutes", C_GRN)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C_RED}Interrupted{S_RS}")
    except Exception as e:
        print(f"\n{C_RED}FATAL: {e}{S_RS}")
        traceback.print_exc()
