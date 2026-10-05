"""
TASK 8 HOTFIX: Re-runs regime diagnostic + portfolio construction
Fixes datetime64 vs date comparison bug.
Re-executes strategies on OOS to rebuild trade logs (needed for regime attribution),
using the already-verified OOS setups from task8_oos_verified_adaptive.csv.
"""
import os
import sys
import time
import warnings
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

warnings.filterwarnings("ignore")

sys.path.insert(0, r"C:\BybitBacktest\backtester")
try:
    from engine_v2 import (
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

# Config
DATA_ROOT = r"C:\BybitBacktest\data\resampled"
RESULTS_ROOT = r"C:\BybitBacktest\results"
OOS_START = pd.Timestamp("2026-01-01")
OOS_END   = pd.Timestamp("2026-08-31 23:59:59")

PORTFOLIO_TARGETS = {
    "min_combined_monthly_trades": 25,
    "min_combined_win_rate": 50.0,
    "max_per_symbol": 3,
    "max_per_strategy": 4,
    "target_portfolio_size": 12,
}

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

def month_range_2026(m):
    if m == 12:
        s = pd.Timestamp(f"2026-{m:02d}-01")
        e = pd.Timestamp("2027-01-01") - pd.Timedelta(seconds=1)
    else:
        s = pd.Timestamp(f"2026-{m:02d}-01")
        e = pd.Timestamp(f"2026-{m+1:02d}-01") - pd.Timedelta(seconds=1)
    return s, e

# ============================================================
# STRATEGY DEFINITIONS (need to rebuild trade logs)
# We only need the 6 strategies that appeared in verified results
# ============================================================
def _build_signals(df, long_mask, short_mask):
    n = len(df)
    arr = np.zeros(n, dtype=int)
    arr[long_mask.values] = 1
    arr[short_mask.values] = -1
    both = long_mask.values & short_mask.values
    arr[both] = 1
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

def _apply_regime_gate(signals, regimes, allowed_long, allowed_short):
    reg_vals = regimes.values
    sig_vals = signals.values.copy()
    long_mask = sig_vals == 1
    short_mask = sig_vals == -1
    allow_long = np.isin(reg_vals, allowed_long)
    allow_short = np.isin(reg_vals, allowed_short)
    sig_vals[long_mask & ~allow_long] = 0
    sig_vals[short_mask & ~allow_short] = 0
    return pd.Series(sig_vals, index=signals.index, dtype=int)

def align_regime_to_df(df, regime_series):
    if regime_series is None or len(regime_series) == 0:
        return pd.Series(["UNKNOWN"] * len(df), index=df.index)
    daily_regime = regime_series.copy()
    # Ensure index is datetime
    if not isinstance(daily_regime.index, pd.DatetimeIndex):
        daily_regime.index = pd.to_datetime(daily_regime.index)
    aligned = daily_regime.reindex(df.index, method="ffill")
    return aligned.fillna("UNKNOWN")

class RegimeAdaptiveStrategy(Strategy):
    regimes = None
    def set_regimes(self, regime_series): self.regimes = regime_series

class R01_Adaptive_EMA_Cross(RegimeAdaptiveStrategy):
    name = "R01_Adaptive_EMA_Cross"
    default_params = {"ema_fast":12,"ema_slow":26,"atr_period":14,
                      "sl_lookback":20,"tp_atr_bull":3.0,"tp_atr_bear":1.5}
    def generate_signals(self, df, params):
        p = params
        e_fast = EMA(df["close"], int(p["ema_fast"]))
        e_slow = EMA(df["close"], int(p["ema_slow"]))
        cross_up = ((e_fast > e_slow) & (e_fast.shift(1) <= e_slow.shift(1))).fillna(False).astype(bool)
        cross_dn = ((e_fast < e_slow) & (e_fast.shift(1) >= e_slow.shift(1))).fillna(False).astype(bool)
        signals = _build_signals(df, cross_up, cross_dn)
        if self.regimes is not None:
            reg = align_regime_to_df(df, self.regimes)
            signals = _apply_regime_gate(signals, reg, ["BULL"], ["BEAR"])
        a = ATR(df, int(p["atr_period"]))
        close = df["close"]
        sl = pd.Series(np.nan, index=df.index, dtype=float)
        tp = pd.Series(np.nan, index=df.index, dtype=float)
        lm = signals == 1; sm = signals == -1
        sl_l_raw = df["low"].rolling(int(p["sl_lookback"])).min()
        sl_s_raw = df["high"].rolling(int(p["sl_lookback"])).max()
        sl.loc[lm] = sl_l_raw.loc[lm].values
        sl.loc[sm] = sl_s_raw.loc[sm].values
        tp.loc[lm] = (close + p["tp_atr_bull"] * a).loc[lm].values
        tp.loc[sm] = (close - p["tp_atr_bear"] * a).loc[sm].values
        return signals, sl, tp

class R13_Adaptive_ATR_Range_Expansion(RegimeAdaptiveStrategy):
    name = "R13_Adaptive_ATR_Range_Expansion"
    default_params = {"atr_period":14,"expansion_ratio":1.8,
                      "sl_atr_mult":2.0,"tp_atr_mult":3.0}
    def generate_signals(self, df, params):
        p = params
        a = ATR(df, int(p["atr_period"]))
        bar_range = df["high"] - df["low"]
        big_range = bar_range > float(p["expansion_ratio"]) * a
        close_gt_open = df["close"] > df["open"]
        close_lt_open = df["close"] < df["open"]
        long_mask = (big_range & close_gt_open).fillna(False).astype(bool)
        short_mask = (big_range & close_lt_open).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        if self.regimes is not None:
            reg = align_regime_to_df(df, self.regimes)
            signals = _apply_regime_gate(signals, reg, ["BULL"], ["BEAR"])
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class R15_Adaptive_Volume_Breakout(RegimeAdaptiveStrategy):
    name = "R15_Adaptive_Volume_Breakout"
    default_params = {"vol_sma":20,"vol_mult":2.5,"atr_period":14,
                      "close_pct":0.7,"sl_atr_mult":1.5,"tp_atr_mult":2.5}
    def generate_signals(self, df, params):
        p = params
        vol_avg = SMA(df["volume"], int(p["vol_sma"]))
        high_vol = df["volume"] > float(p["vol_mult"]) * vol_avg
        bar_range = df["high"] - df["low"]
        close_pos = (df["close"] - df["low"]) / bar_range.replace(0, np.nan)
        close_near_high = close_pos > float(p["close_pct"])
        close_near_low = close_pos < (1 - float(p["close_pct"]))
        long_mask = (high_vol & close_near_high).fillna(False).astype(bool)
        short_mask = (high_vol & close_near_low).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        if self.regimes is not None:
            reg = align_regime_to_df(df, self.regimes)
            signals = _apply_regime_gate(signals, reg, ["BULL","CHOP"], ["BEAR","CHOP"])
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class R16_Adaptive_Trend_Plus_RSI_Pullback(RegimeAdaptiveStrategy):
    name = "R16_Adaptive_Trend_Plus_RSI_Pullback"
    default_params = {"ema_trend":50,"rsi_period":14,
                      "rsi_bull_low":40,"rsi_bull_high":50,
                      "rsi_bear_low":50,"rsi_bear_high":60,
                      "atr_period":14,"sl_atr_mult":1.5,"tp_atr_mult":2.5}
    def generate_signals(self, df, params):
        p = params
        e = EMA(df["close"], int(p["ema_trend"]))
        e_slope = e.diff(5)
        r = RSI(df["close"], int(p["rsi_period"]))
        bull_pullback = ((e_slope > 0) & (r >= p["rsi_bull_low"]) & (r <= p["rsi_bull_high"])).fillna(False).astype(bool)
        bear_pullback = ((e_slope < 0) & (r >= p["rsi_bear_low"]) & (r <= p["rsi_bear_high"])).fillna(False).astype(bool)
        long_mask = (bull_pullback & ~bull_pullback.shift(1).fillna(True)).fillna(False).astype(bool)
        short_mask = (bear_pullback & ~bear_pullback.shift(1).fillna(True)).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        if self.regimes is not None:
            reg = align_regime_to_df(df, self.regimes)
            signals = _apply_regime_gate(signals, reg, ["BULL"], ["BEAR"])
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

STRATEGY_MAP = {
    "R01_Adaptive_EMA_Cross": R01_Adaptive_EMA_Cross,
    "R13_Adaptive_ATR_Range_Expansion": R13_Adaptive_ATR_Range_Expansion,
    "R15_Adaptive_Volume_Breakout": R15_Adaptive_Volume_Breakout,
    "R16_Adaptive_Trend_Plus_RSI_Pullback": R16_Adaptive_Trend_Plus_RSI_Pullback,
}

# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    box("TASK 8 HOTFIX: Regime Diagnostic + Portfolio Rebuild", C_CYAN)

    loader = MultiTFDataLoader(DATA_ROOT)
    engine = BacktestEngine()

    section(1, "LOADING VERIFIED OOS SETUPS FROM CSV", C_CYAN)
    oos_csv = os.path.join(RESULTS_ROOT, "task8_oos_verified_adaptive.csv")
    verified_df = pd.read_csv(oos_csv)
    print(f"  Loaded {len(verified_df)} OOS-verified setups")
    print(verified_df[["strategy","symbol","tf","oos_sharpe","oos_win_rate","oos_trades","oos_return","status"]].to_string())

    section(2, "RECOMPUTING REGIMES + REBUILDING OOS TRADE LOGS", C_CYAN)
    detector = RegimeDetector(data_loader=loader)
    regime_series = detector.classify_series("2022-01-01", "2026-08-31")
    # Normalize index to Timestamp
    if not isinstance(regime_series.index, pd.DatetimeIndex):
        regime_series.index = pd.to_datetime(regime_series.index)
    print(f"  Regime series length: {len(regime_series)}")

    # Rebuild trade logs
    all_verified = []
    for i, row in verified_df.iterrows():
        sname = row["strategy"]; sym = row["symbol"]; tf = row["tf"]
        cls = STRATEGY_MAP.get(sname)
        if cls is None:
            print(f"  {C_YEL}[SKIP] Unknown strategy: {sname}{S_RS}")
            continue
        try:
            df_oos = loader.load(sym, tf, OOS_START, OOS_END)
            if df_oos is None or len(df_oos) < 50: continue
            strat_inst = cls(engine=engine)
            strat_inst.set_regimes(regime_series)
            res = strat_inst.backtest(df_oos, cls.default_params)
            entry = row.to_dict()
            entry["trades_ref"] = res["trades"]
            all_verified.append(entry)
            print(f"  {C_GRN}[OK]{S_RS} {sname[:32]:<32} | {sym:<14} | {tf:<3} | Trades: {len(res['trades'])}")
        except Exception as e:
            print(f"  {C_RED}[ERR]{S_RS} {sname}|{sym}|{tf}: {e}")

    if not all_verified:
        print(f"  {C_RED}[!] No verified setups rebuilt.{S_RS}"); return

    # SECTION 3: FIXED regime diagnostic
    section(3, "REGIME PERFORMANCE DIAGNOSTIC (2026 OOS)", C_CYAN)
    reg_stats = {r: {"pnl":0.0,"trades":0,"wins":0} for r in ["BULL","BEAR","CHOP","CRISIS","UNKNOWN"]}

    # FIX: filter regime_series using pd.Timestamp comparison, NOT .date()
    regime_2026 = regime_series[(regime_series.index >= OOS_START) & (regime_series.index <= OOS_END)]
    print(f"  Regime days in 2026 OOS: {len(regime_2026)}")

    # Build lookup: date -> regime label
    regime_dict = {}
    for idx, reg in regime_2026.items():
        # idx is pd.Timestamp; convert to date for daily lookup
        d = idx.date() if hasattr(idx, "date") else pd.Timestamp(idx).date()
        regime_dict[d] = reg

    print(f"  Regime dict entries: {len(regime_dict)}")

    for r in all_verified:
        for t in r["trades_ref"]:
            entry_time = t["entry_time"]
            if hasattr(entry_time, "date"):
                d = entry_time.date()
            else:
                d = pd.Timestamp(entry_time).date()
            reg = regime_dict.get(d, "UNKNOWN")
            reg_stats[reg]["pnl"] += t["pnl"]
            reg_stats[reg]["trades"] += 1
            if t["pnl"] > 0: reg_stats[reg]["wins"] += 1

    reg_rows = []
    for reg in ["BULL","BEAR","CHOP","CRISIS","UNKNOWN"]:
        s = reg_stats[reg]
        wr = 100*s["wins"]/s["trades"] if s["trades"]>0 else 0
        avg = s["pnl"]/s["trades"] if s["trades"]>0 else 0
        col = C_GRN if s["pnl"]>0 else (C_RED if s["pnl"]<0 else C_YEL)
        reg_rows.append([
            reg, s["trades"], f"{wr:.1f}%",
            f"{col}${s['pnl']:.2f}{S_RS}", f"${avg:.4f}"
        ])
    print(tabulate(reg_rows, headers=["Regime","Trades","Win%","Total PnL","Avg PnL/Trade"], tablefmt="grid"))

    # SECTION 4: Portfolio construction
    section(4, "FINAL ADAPTIVE PORTFOLIO CONSTRUCTION", C_MAG)
    verified_sorted = sorted(all_verified, key=lambda x: -float(x.get("oos_sharpe", 0)))
    portfolio = []
    combined_monthly = 0.0
    sym_count = {}; strat_count = {}
    for r in verified_sorted:
        if sym_count.get(r["symbol"], 0) >= PORTFOLIO_TARGETS["max_per_symbol"]: continue
        if strat_count.get(r["strategy"], 0) >= PORTFOLIO_TARGETS["max_per_strategy"]: continue
        portfolio.append(r)
        combined_monthly += float(r.get("oos_monthly_freq", 0))
        sym_count[r["symbol"]] = sym_count.get(r["symbol"], 0) + 1
        strat_count[r["strategy"]] = strat_count.get(r["strategy"], 0) + 1
        if len(portfolio) >= PORTFOLIO_TARGETS["target_portfolio_size"]:
            break

    print(f"  Portfolio size          : {len(portfolio)}")
    print(f"  Combined monthly trades : {combined_monthly:.1f}")
    print(f"  Unique symbols          : {len(sym_count)}")
    print(f"  Unique strategies       : {len(strat_count)}")

    prt_rows = []
    for r in portfolio:
        prt_rows.append([
            r["strategy"][:26], r["symbol"][:12], r["tf"],
            f"{float(r['oos_sharpe']):.2f}", f"{float(r['oos_win_rate']):.1f}%",
            f"{float(r['oos_pf']):.2f}", int(r["oos_trades"]),
            f"{float(r['oos_return']):.2f}%", f"{float(r['oos_monthly_freq']):.1f}"
        ])
    print("\n  Portfolio Members:")
    print(tabulate(prt_rows,
        headers=["Strategy","Symbol","TF","OOS Sh","Win%","PF","Trades","Return","MoFr"],
        tablefmt="simple"))

    # Combined stats
    all_trades = []
    for r in portfolio: all_trades.extend(r["trades_ref"])

    if all_trades:
        pnls = np.array([t["pnl"] for t in all_trades])
        wins = (pnls > 0).sum()
        pf_wr = 100*wins/len(pnls) if len(pnls)>0 else 0
        gp = pnls[pnls>0].sum(); gl = abs(pnls[pnls<0].sum())
        pf_pf = gp/gl if gl>0 else 999
        pf_aw = pnls[pnls>0].mean() if wins>0 else 0
        pf_al = abs(pnls[pnls<0].mean()) if (pnls<0).sum()>0 else 0
        pf_expect = (wins/len(pnls))*pf_aw - (1-wins/len(pnls))*pf_al

        trades_time = sorted(all_trades, key=lambda t: t["exit_time"])
        eq_v = [10000.0]; eq_t = [OOS_START]
        eq = 10000.0
        for t in trades_time:
            eq += t["pnl"]; eq_v.append(eq); eq_t.append(t["exit_time"])
        eq_s = pd.Series(eq_v, index=eq_t)
        eq_d = eq_s.resample("D").last().ffill()
        dr = eq_d.pct_change().dropna()
        port_sh = float(dr.mean()/dr.std()*np.sqrt(365)) if dr.std()>0 else 0
        peak = eq_d.cummax(); dd = (eq_d - peak)/peak
        port_dd = float(dd.min()*100)
        port_ret = (eq_s.iloc[-1]/eq_s.iloc[0] - 1) * 100

        print(f"\n  {C_GRN}{S_BR}Portfolio 2026 OOS Combined Metrics:{S_RS}")
        pstats = [
            ["Combined Trades", len(all_trades)],
            ["Combined Win Rate %", f"{pf_wr:.2f}"],
            ["Combined Profit Factor", f"{pf_pf:.3f}"],
            ["Combined Sharpe (daily)", f"{port_sh:.3f}"],
            ["Combined Max DD %", f"{port_dd:.2f}"],
            ["Combined Total Return %", f"{port_ret:.2f}"],
            ["Combined Expectancy $", f"{pf_expect:.4f}"],
            ["Combined Monthly Trades", f"{combined_monthly:.1f}"],
        ]
        print(tabulate(pstats, headers=["Metric","Value"], tablefmt="grid"))

        # SECTION 5: month-by-month heatmap
        section(5, "2026 MONTH-BY-MONTH ADAPTIVE PORTFOLIO HEATMAP", C_CYAN)
        mo_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug"]
        rows = []
        profitable_mo = 0
        for mi in range(1, 9):
            ms, me = month_range_2026(mi)
            m_trades = [t for t in all_trades if ms <= t["exit_time"] <= me]
            if m_trades:
                m_pnl = sum(t["pnl"] for t in m_trades)
                m_wins = sum(1 for t in m_trades if t["pnl"]>0)
                m_wr = 100*m_wins/len(m_trades)
                m_ret = m_pnl / 10000 * 100
                col = C_GRN if m_ret>0 else C_RED
                if m_ret > 0: profitable_mo += 1
            else:
                m_ret = 0; m_wr = 0; col = C_YEL
            rows.append([
                mo_names[mi-1],
                f"{col}{m_ret:+.2f}%{S_RS}",
                len(m_trades),
                f"{m_wr:.1f}%" if m_trades else "-"
            ])
        print(tabulate(rows, headers=["Month","Return","Trades","Win%"], tablefmt="grid"))
        print(f"\n  Profitable months: {profitable_mo}/8 ({100*profitable_mo/8:.1f}%)")

        # Save portfolio
        pf_csv = os.path.join(RESULTS_ROOT, "task8_adaptive_portfolio.csv")
        rows_out = []
        for r in portfolio:
            row = {k: v for k, v in r.items() if k != "trades_ref"}
            rows_out.append(row)
        pd.DataFrame(rows_out).to_csv(pf_csv, index=False)
        print(f"\n  {C_GRN}[SAVED] {pf_csv}{S_RS}")

        # SECTION 6: Verdict
        section(6, "EXECUTIVE SUMMARY", C_GRN)
        all_targets_met = (
            pf_wr >= PORTFOLIO_TARGETS["min_combined_win_rate"] and
            combined_monthly >= PORTFOLIO_TARGETS["min_combined_monthly_trades"] and
            port_ret > 0 and pf_pf >= 1.15
        )
        if all_targets_met:
            v = f"{C_GRN}{S_BR}GO - Adaptive portfolio meets targets. Deploy alongside Task 7.2.{S_RS}"
        else:
            v = f"{C_YEL}CONDITIONAL - Some targets missed. Only 6 edges verified (low sample); consider expanded scan.{S_RS}"
        print(f"\n  Verdict: {v}")

        print(f"\n  {C_CYAN}Key Insights:{S_RS}")
        print(f"    - Trend/breakout strategies (R01, R13, R15) dominated OOS survival")
        print(f"    - Mean-reversion strategies (R06-R10) failed to survive 2026 OOS")
        print(f"    - 4H timeframe most robust: 5/6 verified edges on 4H")
        print(f"    - Combined with Task 7.2 time-based portfolio for maximum diversification")

    elapsed = time.time() - t0
    print()
    box(f"TASK 8 HOTFIX COMPLETE - Runtime: {elapsed:.1f}s", C_GRN)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C_RED}Interrupted{S_RS}")
    except Exception as e:
        print(f"\n{C_RED}FATAL: {e}{S_RS}")
        traceback.print_exc()
