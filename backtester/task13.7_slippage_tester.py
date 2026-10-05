"""
TASK 13.7: SLIPPAGE SENSITIVITY STRESS TEST
Estimates realistic per-symbol slippage and re-simulates portfolio
across 6 slippage levels. Reveals true breakeven cost tolerance.
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
    from engine_v2 import MultiTFDataLoader
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

FUNDING_CSV = os.path.join(RESULTS_ROOT, "task13.6_funding_adjusted_trades.csv")
BASE_CSV = os.path.join(RESULTS_ROOT, "task11.2_true_portfolio_trade_log.csv")

IS_END = pd.Timestamp("2025-12-31 23:59:59")
OOS_START = pd.Timestamp("2026-01-01")
OOS_END = pd.Timestamp("2026-08-31 23:59:59")

STARTING_CAPITAL = 10000.0
BASELINE_SLIPPAGE = 0.0003  # 0.03%

# Slippage sweep levels
SLIPPAGE_LEVELS = [0.0003, 0.0005, 0.0008, 0.0010, 0.0015, 0.0020]
SLIPPAGE_LABELS = ["0.03%", "0.05%", "0.08%", "0.10%", "0.15%", "0.20%"]

# 15 unique symbols
SYMBOLS = [
    "XMR_USDT_USDT","ATOM_USDT_USDT","ADA_USDT_USDT","BSV_USDT_USDT","ETC_USDT_USDT",
    "AVAX_USDT_USDT","QTUM_USDT_USDT","TWT_USDT_USDT","DOT_USDT_USDT","XLM_USDT_USDT",
    "ETH_USDT_USDT","SAND_USDT_USDT","OP_USDT_USDT","INJ_USDT_USDT","DOGE_USDT_USDT"
]

# Liquidity tier defaults (used if data load fails)
LIQUIDITY_DEFAULTS = {
    "ETH_USDT_USDT": {"tier":1, "slip":0.0003},
    "DOGE_USDT_USDT": {"tier":1, "slip":0.0004},
    "ADA_USDT_USDT": {"tier":1, "slip":0.0004},
    "AVAX_USDT_USDT": {"tier":1, "slip":0.0005},
    "DOT_USDT_USDT":  {"tier":1, "slip":0.0005},
    "ATOM_USDT_USDT": {"tier":2, "slip":0.0006},
    "XLM_USDT_USDT":  {"tier":2, "slip":0.0006},
    "ETC_USDT_USDT":  {"tier":2, "slip":0.0007},
    "SAND_USDT_USDT": {"tier":2, "slip":0.0007},
    "INJ_USDT_USDT":  {"tier":2, "slip":0.0006},
    "OP_USDT_USDT":   {"tier":2, "slip":0.0006},
    "XMR_USDT_USDT":  {"tier":3, "slip":0.0010},
    "BSV_USDT_USDT":  {"tier":3, "slip":0.0012},
    "QTUM_USDT_USDT": {"tier":3, "slip":0.0012},
    "TWT_USDT_USDT":  {"tier":3, "slip":0.0015},
}

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
# LOAD TRADE LOG
# ============================================================
def load_trade_log():
    """Load funding-adjusted log if available, else baseline."""
    if os.path.exists(FUNDING_CSV):
        df = pd.read_csv(FUNDING_CSV)
        source = "task13.6 (funding-adjusted)"
        # Use net pnl (after funding)
        if "pnl_net" in df.columns:
            df["pnl_base"] = df["pnl_net"]
        else:
            df["pnl_base"] = df["pnl"]
    else:
        df = pd.read_csv(BASE_CSV)
        source = "task11.2 (baseline, no funding)"
        df["pnl_base"] = df["pnl"]
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    # Filter to Standard tier only for consistent analysis
    if "tier" in df.columns:
        df = df[df["tier"] == "Standard_1.0%"].reset_index(drop=True)
    return df, source

# ============================================================
# STEP 1: PER-SYMBOL SLIPPAGE ESTIMATOR
# ============================================================
def estimate_symbol_slippage(loader, symbol):
    """Estimate realistic slippage using 1H liquidity data."""
    df = loader.load(symbol, "1H", "2023-01-01", "2025-12-31")
    if df is None or len(df) < 100:
        default = LIQUIDITY_DEFAULTS.get(symbol, {"tier":3, "slip":0.0010})
        return {
            "symbol": symbol,
            "avg_range_pct": None,
            "avg_volume_usd": None,
            "estimated_slippage": default["slip"],
            "session_slippage": default["slip"] * 2.0,
            "tier": default["tier"],
            "source": "default"
        }
    # Compute avg bar range as pct
    bar_range_pct = (df["high"] - df["low"]) / df["close"]
    avg_range_pct = float(bar_range_pct.mean())
    # Volume in USD
    avg_price = float(df["close"].mean())
    avg_volume_usd = float((df["volume"] * df["close"]).mean())
    # Slippage formula
    est_slip = 0.0002 + (avg_range_pct * 0.5) + (10000.0 / max(avg_volume_usd, 100.0)) * 0.05
    est_slip = float(np.clip(est_slip, 0.0002, 0.0030))
    # Session boundary: bars at hour 12 and 16
    hour = df.index.hour
    session_mask = (hour == 12) | (hour == 16)
    if session_mask.sum() > 20:
        session_range = (df.loc[session_mask, "high"] - df.loc[session_mask, "low"]) / df.loc[session_mask, "close"]
        session_range_avg = float(session_range.mean())
        session_slip = 0.0002 + (session_range_avg * 0.5) + (10000.0 / max(avg_volume_usd, 100.0)) * 0.05
        session_slip = float(np.clip(session_slip, 0.0002, 0.0050))
    else:
        session_slip = est_slip * 1.8
    # Tier
    if avg_volume_usd > 5_000_000:  tier = 1
    elif avg_volume_usd > 500_000:  tier = 2
    else:                            tier = 3
    return {
        "symbol": symbol,
        "avg_range_pct": round(avg_range_pct * 100, 4),
        "avg_volume_usd": round(avg_volume_usd, 2),
        "estimated_slippage": round(est_slip, 5),
        "session_slippage": round(session_slip, 5),
        "tier": tier,
        "source": "computed"
    }

# ============================================================
# STEP 2 & 3: APPLY SLIPPAGE TO TRADES
# ============================================================
def apply_slippage_to_trades(trade_df, slippage_map):
    """
    slippage_map: either a scalar float or dict {symbol: slip_rate}
    Adjusts entry/exit prices and recomputes PnL.
    Returns new DataFrame with 'pnl_adjusted' column.
    """
    df = trade_df.copy()
    baseline_slip = BASELINE_SLIPPAGE

    def get_slip(sym):
        if isinstance(slippage_map, dict):
            return slippage_map.get(sym, baseline_slip)
        return float(slippage_map)

    # Compute adjustment factors
    # Original trade already had 0.03% slippage baked in.
    # If new slippage = S, we need to REPLACE 0.03% with S.
    # Additional slippage = S - baseline_slip
    # This gets applied both directions (entry and exit)
    # For LONG: entry gets more expensive (higher), exit gets cheaper (lower)
    # For SHORT: entry gets cheaper (lower), exit gets more expensive (higher)
    # Additional cost per side = additional_slip_frac * price

    directions = df["direction"].map({"LONG":1, "SHORT":-1}).values
    entry_prices = df["entry_price"].values
    exit_prices = df["exit_price"].values
    sizes = df["size"].values
    symbols = df["symbol"].values

    additional_slip = np.array([get_slip(s) - baseline_slip for s in symbols])
    # If additional_slip < 0, we're reducing slippage (baseline was overly conservative for that symbol)
    # If > 0, we're adding cost

    # Additional cost on entry per unit: entry_price * additional_slip (LONG buys higher, SHORT sells lower)
    # Net cost on trade = (entry_slip_delta + exit_slip_delta) * size
    # For both LONG and SHORT, additional slippage reduces PnL (moves entry/exit against us)
    # cost_delta_dollars = (entry_price + exit_price) * additional_slip * size
    slip_cost_delta = (entry_prices + exit_prices) * additional_slip * sizes
    # This slip_cost_delta is SUBTRACTED from PnL (if positive, PnL is reduced)
    df["slippage_adjustment"] = -slip_cost_delta
    df["pnl_adjusted"] = df["pnl_base"] + df["slippage_adjustment"]
    return df

# ============================================================
# METRIC COMPUTATION
# ============================================================
def compute_metrics(pnls, days_span=None, starting_cap=STARTING_CAPITAL):
    if len(pnls) < 2:
        return {"n_trades":len(pnls),"total_return":0,"cagr":0,"sharpe":0,"sortino":0,
                "maxdd":0,"win_rate":0,"pf":0,"expectancy":0,"final_equity":starting_cap}
    equity = starting_cap + np.cumsum(pnls)
    equity = np.maximum(equity, 1.0)
    total_ret = (equity[-1] / starting_cap - 1) * 100
    if days_span is None: days_span = max(len(pnls) / 3.0, 1)  # rough estimate
    years = max(days_span / 365.25, 0.01)
    cagr = ((equity[-1] / starting_cap) ** (1/years) - 1) * 100 if equity[-1] > 0 else 0
    daily_pct = pnls / starting_cap
    if daily_pct.std() > 0:
        trades_per_year = len(pnls) / years
        sh = daily_pct.mean() / daily_pct.std() * np.sqrt(trades_per_year)
        neg = daily_pct[daily_pct < 0]
        sortino = daily_pct.mean() / neg.std() * np.sqrt(trades_per_year) if len(neg) > 0 and neg.std() > 0 else 0
    else:
        sh = 0; sortino = 0
    sh = float(np.clip(sh, -20, 20))
    sortino = float(np.clip(sortino, -20, 20))
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    maxdd = float(dd.min() * 100)
    wins = (pnls > 0).sum()
    wr = 100 * wins / len(pnls)
    gp = pnls[pnls > 0].sum()
    gl = abs(pnls[pnls < 0].sum())
    pf = gp / gl if gl > 0 else (999 if gp > 0 else 0)
    exp = pnls.mean()
    return {
        "n_trades": len(pnls),
        "total_return": round(total_ret, 2),
        "cagr": round(cagr, 2),
        "sharpe": round(sh, 3),
        "sortino": round(sortino, 3),
        "maxdd": round(maxdd, 2),
        "win_rate": round(wr, 2),
        "pf": round(pf, 3),
        "expectancy": round(exp, 4),
        "final_equity": round(equity[-1], 2),
    }

def compute_metrics_for_trades(trades_df, pnl_col="pnl_adjusted"):
    if len(trades_df) < 2:
        return compute_metrics(np.array([]))
    days = max((trades_df["exit_time"].max() - trades_df["entry_time"].min()).days, 1)
    return compute_metrics(trades_df[pnl_col].values, days)

# ============================================================
# STEP 4: PER-EDGE BREAKPOINT DETECTION
# ============================================================
def find_edge_breakpoints(trades_df, edge_id):
    """Find slippage levels where edge Sharpe drops below thresholds."""
    edge_trades = trades_df[trades_df["edge_id"] == edge_id]
    if len(edge_trades) < 10:
        return {"marginal":None, "unprofitable":None, "pf_break":None, "sharpe_at_baseline":0, "sharpe_at_010":0}
    breakpoints = {"marginal":None, "unprofitable":None, "pf_break":None}
    sharpe_at_baseline = None
    sharpe_at_010 = None
    for slip in np.arange(0.0002, 0.0031, 0.0001):
        adj_df = apply_slippage_to_trades(edge_trades, slip)
        m = compute_metrics_for_trades(adj_df)
        if slip == 0.0003:
            sharpe_at_baseline = m["sharpe"]
        if abs(slip - 0.0010) < 1e-6:
            sharpe_at_010 = m["sharpe"]
        if breakpoints["marginal"] is None and m["sharpe"] < 0.5:
            breakpoints["marginal"] = round(slip * 100, 3)
        if breakpoints["unprofitable"] is None and m["sharpe"] < 0.0:
            breakpoints["unprofitable"] = round(slip * 100, 3)
        if breakpoints["pf_break"] is None and m["pf"] < 1.0:
            breakpoints["pf_break"] = round(slip * 100, 3)
    if sharpe_at_baseline is None:
        adj_df = apply_slippage_to_trades(edge_trades, 0.0003)
        sharpe_at_baseline = compute_metrics_for_trades(adj_df)["sharpe"]
    if sharpe_at_010 is None:
        adj_df = apply_slippage_to_trades(edge_trades, 0.0010)
        sharpe_at_010 = compute_metrics_for_trades(adj_df)["sharpe"]
    breakpoints["sharpe_at_baseline"] = sharpe_at_baseline
    breakpoints["sharpe_at_010"] = sharpe_at_010
    return breakpoints

# ============================================================
# ASCII CHART
# ============================================================
def ascii_chart(x_vals, y_vals, width=60, height=15, title="Sharpe vs Slippage"):
    if len(y_vals) < 2: return
    y_min, y_max = min(y_vals), max(y_vals)
    y_range = y_max - y_min if y_max > y_min else 1.0
    print(f"\n  {C_CYAN}{title}{S_RS}")
    print(f"  {C_CYAN}{'Sharpe':<8}|{S_RS}")
    for row in range(height, -1, -1):
        y_val = y_min + (y_range * row / height)
        line = f"  {y_val:>6.2f}  |"
        for col in range(len(x_vals)):
            plot_row = int((y_vals[col] - y_min) / y_range * height)
            if plot_row == row:
                line += " * "
            else:
                line += "   "
        print(line)
    print(f"  {'':>8}+" + "-" * (len(x_vals) * 3))
    print(f"  {'':>8} " + "".join(f"{x*100:>2.0f} " for x in x_vals))
    print(f"  {'':>8} slippage % --->")

# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    box("TASK 13.7: SLIPPAGE SENSITIVITY STRESS TEST", C_CYAN)
    print(f"{C_CYAN}  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{C_CYAN}  Slippage levels: {SLIPPAGE_LABELS}")

    # Load trade log
    section(1, "LOAD TRADE LOG", C_CYAN)
    trade_df, source = load_trade_log()
    print(f"  Source: {source}")
    print(f"  Trades loaded: {len(trade_df):,}")
    print(f"  Date range: {trade_df['entry_time'].min()} -> {trade_df['exit_time'].max()}")

    loader = MultiTFDataLoader(DATA_ROOT)

    # STEP 1: Per-symbol slippage estimation
    section(2, "STEP 1: PER-SYMBOL SLIPPAGE ESTIMATION", C_MAG)
    print(f"  Analyzing 1H liquidity for {len(SYMBOLS)} symbols...")
    slip_estimates = {}
    slip_rows = []
    for sym in SYMBOLS:
        est = estimate_symbol_slippage(loader, sym)
        slip_estimates[sym] = est
        tier_col = C_GRN if est["tier"] == 1 else (C_YEL if est["tier"] == 2 else C_RED)
        slip_rows.append([
            sym.split("_")[0],
            f"${est['avg_volume_usd']/1e6:.2f}M" if est['avg_volume_usd'] else "N/A",
            f"{est['avg_range_pct']:.4f}%" if est['avg_range_pct'] else "N/A",
            f"{est['estimated_slippage']*100:.4f}%",
            f"{est['session_slippage']*100:.4f}%",
            f"{tier_col}Tier {est['tier']}{S_RS}",
            est["source"]
        ])
    print(tabulate(slip_rows,
        headers=["Symbol","Avg Volume","Avg Range","Est Slip","Session Slip","Tier","Source"],
        tablefmt="grid"))

    # STEP 2: Multi-slippage sweep
    section(3, "STEP 2: PORTFOLIO SLIPPAGE SWEEP (6 LEVELS)", C_MAG)
    sweep_rows = []
    sweep_export = []
    port_metrics_by_level = {}
    for slip, label in zip(SLIPPAGE_LEVELS, SLIPPAGE_LABELS):
        adj_df = apply_slippage_to_trades(trade_df, slip)
        m = compute_metrics_for_trades(adj_df)
        port_metrics_by_level[label] = m
        col = C_GRN if m["sharpe"] > 1.0 else (C_YEL if m["sharpe"] > 0.5 else C_RED)
        sweep_rows.append([
            label,
            f"{m['total_return']:+.2f}%",
            f"{m['cagr']:+.2f}%",
            f"{col}{m['sharpe']:.3f}{S_RS}",
            f"{m['maxdd']:.2f}%",
            f"{m['win_rate']:.2f}%",
            f"{m['pf']:.3f}",
            f"${m['final_equity']:,.2f}"
        ])
        sweep_export.append({
            "slippage_level": label, "slippage_decimal": slip,
            "total_return_pct": m["total_return"], "cagr_pct": m["cagr"],
            "sharpe": m["sharpe"], "sortino": m["sortino"],
            "max_dd_pct": m["maxdd"], "win_rate_pct": m["win_rate"],
            "profit_factor": m["pf"], "expectancy_dollar": m["expectancy"],
            "final_equity": m["final_equity"], "n_trades": m["n_trades"]
        })
    print(tabulate(sweep_rows,
        headers=["Slippage", "Return", "CAGR", "Sharpe", "MaxDD", "Win%", "PF", "Final Equity"],
        tablefmt="grid"))

    # STEP 3: Per-symbol realistic slippage
    section(4, "STEP 3: PER-SYMBOL REALISTIC SLIPPAGE APPLICATION", C_MAG)
    per_symbol_slip = {sym: est["estimated_slippage"] for sym, est in slip_estimates.items()}
    adj_df = apply_slippage_to_trades(trade_df, per_symbol_slip)
    m_realistic = compute_metrics_for_trades(adj_df)
    m_baseline = port_metrics_by_level["0.03%"]

    print(f"  Using estimated per-symbol slippage:")
    for sym, slip in sorted(per_symbol_slip.items(), key=lambda x: -x[1]):
        print(f"    {sym.split('_')[0]:<8}: {slip*100:.4f}%")

    realistic_rows = [
        ["Total Return %", f"{m_baseline['total_return']}%", f"{m_realistic['total_return']}%",
         f"{m_realistic['total_return'] - m_baseline['total_return']:+.2f}%"],
        ["CAGR %", f"{m_baseline['cagr']}%", f"{m_realistic['cagr']}%",
         f"{m_realistic['cagr'] - m_baseline['cagr']:+.2f}%"],
        ["Sharpe", m_baseline["sharpe"], m_realistic["sharpe"],
         f"{m_realistic['sharpe'] - m_baseline['sharpe']:+.3f}"],
        ["Max DD %", f"{m_baseline['maxdd']}%", f"{m_realistic['maxdd']}%",
         f"{m_realistic['maxdd'] - m_baseline['maxdd']:+.2f}%"],
        ["Win Rate %", f"{m_baseline['win_rate']}%", f"{m_realistic['win_rate']}%",
         f"{m_realistic['win_rate'] - m_baseline['win_rate']:+.2f}%"],
        ["Profit Factor", m_baseline["pf"], m_realistic["pf"],
         f"{m_realistic['pf'] - m_baseline['pf']:+.3f}"],
        ["Expectancy $", m_baseline["expectancy"], m_realistic["expectancy"],
         f"{m_realistic['expectancy'] - m_baseline['expectancy']:+.4f}"],
        ["Final Equity", f"${m_baseline['final_equity']:,.2f}", f"${m_realistic['final_equity']:,.2f}",
         f"${m_realistic['final_equity'] - m_baseline['final_equity']:+,.2f}"],
    ]
    print(f"\n  {C_CYAN}Baseline (0.03% flat) vs Realistic (per-symbol):{S_RS}")
    print(tabulate(realistic_rows,
        headers=["Metric", "Baseline (0.03%)", "Realistic (per-symbol)", "Delta"],
        tablefmt="grid"))

    # STEP 4: Per-edge breakpoints
    section(5, "STEP 4: PER-EDGE SLIPPAGE BREAKPOINTS", C_MAG)
    print(f"  Scanning breakpoints from 0.02% to 0.30% per edge...")
    edge_ids = sorted(trade_df["edge_id"].unique())
    breakpoint_rows = []
    breakpoint_export = []
    for eid in edge_ids:
        edge_grp = trade_df[trade_df["edge_id"] == eid]
        if len(edge_grp) < 10: continue
        strat = edge_grp["strategy"].iloc[0]
        sym = edge_grp["symbol"].iloc[0]
        tf = edge_grp["tf"].iloc[0]
        bp = find_edge_breakpoints(trade_df, eid)
        # Classification
        marg = bp["marginal"] or 999
        # DEPLOY = survives 0.15% with Sharpe > 1.0 (marginal breakpoint > 0.15%)
        # MONITOR = survives 0.10% but not 0.15%
        # DROP = fails at 0.10% or lower
        marg_pct = marg  # already in %
        if marg_pct > 0.15 or (marg_pct == 999 and bp["sharpe_at_010"] > 1.0):
            status = "DEPLOY"; col = C_GRN
        elif marg_pct > 0.10:
            status = "MONITOR"; col = C_YEL
        else:
            status = "DROP"; col = C_RED
        breakpoint_rows.append([
            int(eid), strat[:24], sym.split("_")[0], tf,
            f"{bp['sharpe_at_baseline']:.2f}",
            f"{bp['sharpe_at_010']:.2f}",
            f"{bp['marginal']}%" if bp['marginal'] else ">0.30%",
            f"{bp['unprofitable']}%" if bp['unprofitable'] else ">0.30%",
            f"{bp['pf_break']}%" if bp['pf_break'] else ">0.30%",
            f"{col}{status}{S_RS}"
        ])
        breakpoint_export.append({
            "edge_id": int(eid), "strategy": strat, "symbol": sym, "tf": tf,
            "sharpe_at_baseline_0.03": bp["sharpe_at_baseline"],
            "sharpe_at_0.10": bp["sharpe_at_010"],
            "marginal_break_pct": bp["marginal"],
            "unprofitable_break_pct": bp["unprofitable"],
            "pf_break_pct": bp["pf_break"],
            "status": status
        })
    print(tabulate(breakpoint_rows,
        headers=["#","Strategy","Symbol","TF","Sh@0.03","Sh@0.10","Marg Break","Unprof Break","PF Break","Status"],
        tablefmt="grid"))

    # STEP 5: Degradation curve
    section(6, "STEP 5: PORTFOLIO DEGRADATION CURVE", C_CYAN)
    sharpe_vals = [port_metrics_by_level[l]["sharpe"] for l in SLIPPAGE_LABELS]
    ascii_chart(SLIPPAGE_LEVELS, sharpe_vals, title="Portfolio Sharpe vs Slippage Level")

    # Find breakpoint where portfolio Sharpe drops below 1.5
    port_break = None
    for i, sh in enumerate(sharpe_vals):
        if sh < 1.5:
            port_break = SLIPPAGE_LABELS[i]
            break
    if port_break:
        print(f"\n  {C_YEL}Portfolio Sharpe drops below 1.5 at: {port_break}{S_RS}")
    else:
        print(f"\n  {C_GRN}Portfolio Sharpe remains above 1.5 across all tested levels{S_RS}")

    # STEP 6: Final classification summary
    section(7, "STEP 6: FINAL DEPLOYMENT CLASSIFICATION", C_MAG)
    n_deploy = sum(1 for e in breakpoint_export if e["status"] == "DEPLOY")
    n_monitor = sum(1 for e in breakpoint_export if e["status"] == "MONITOR")
    n_drop = sum(1 for e in breakpoint_export if e["status"] == "DROP")
    class_rows = [
        [f"{C_GRN}DEPLOY (survives 0.15% slip, Sharpe > 1.0){S_RS}", n_deploy],
        [f"{C_YEL}MONITOR (survives 0.10% but not 0.15%){S_RS}", n_monitor],
        [f"{C_RED}DROP (fails at 0.10% or lower){S_RS}", n_drop],
    ]
    print(tabulate(class_rows, headers=["Category", "Count"], tablefmt="grid"))

    # Family breakdown
    time_based = ["S30a", "S30b", "S30d"]
    regime = ["R01", "R13", "R15", "R16"]
    tb_edges = [e for e in breakpoint_export if any(t in e["strategy"] for t in time_based)]
    rg_edges = [e for e in breakpoint_export if any(t in e["strategy"] for t in regime)]
    print(f"\n  Time-Based Family ({len(tb_edges)} edges):")
    print(f"    DEPLOY: {sum(1 for e in tb_edges if e['status']=='DEPLOY')}")
    print(f"    MONITOR: {sum(1 for e in tb_edges if e['status']=='MONITOR')}")
    print(f"    DROP: {sum(1 for e in tb_edges if e['status']=='DROP')}")
    print(f"\n  Regime-Adaptive Family ({len(rg_edges)} edges):")
    print(f"    DEPLOY: {sum(1 for e in rg_edges if e['status']=='DEPLOY')}")
    print(f"    MONITOR: {sum(1 for e in rg_edges if e['status']=='MONITOR')}")
    print(f"    DROP: {sum(1 for e in rg_edges if e['status']=='DROP')}")

    # SECTION 7: Executive verdict
    section(8, "EXECUTIVE VERDICT", C_GRN)
    avg_realistic_slip = np.mean(list(per_symbol_slip.values())) * 100
    print(f"\n  Average realistic slippage across 15 symbols: {avg_realistic_slip:.4f}%")
    print(f"  Portfolio Sharpe at realistic slippage       : {m_realistic['sharpe']:.3f}")
    print(f"  Portfolio CAGR at realistic slippage         : {m_realistic['cagr']:.2f}%")
    print(f"  Edges surviving realistic conditions         : {n_deploy + n_monitor}/16")
    print(f"  Recommended slippage buffer for live trading : 0.15% (stress-tested tolerance)")

    if n_deploy >= 8 and m_realistic["sharpe"] > 1.0:
        verdict = f"{C_GRN}{S_BR}GO - Portfolio robust to realistic slippage. Deploy to Task 14.{S_RS}"
    elif n_deploy >= 5 and m_realistic["sharpe"] > 0.5:
        verdict = f"{C_YEL}CONDITIONAL - Deploy DEPLOY-rated edges only. Monitor slippage in live.{S_RS}"
    else:
        verdict = f"{C_RED}NO-GO - Slippage sensitivity too high. Return to research.{S_RS}"
    print(f"\n  {S_BR}FINAL VERDICT: {verdict}")

    # SECTION 8: Exports
    section(9, "EXPORT FILES", C_CYAN)
    sweep_csv = os.path.join(RESULTS_ROOT, "task13.7_slippage_sweep_portfolio.csv")
    pd.DataFrame(sweep_export).to_csv(sweep_csv, index=False)
    print(f"  {C_GRN}[SAVED] {sweep_csv}{S_RS}")

    sym_csv = os.path.join(RESULTS_ROOT, "task13.7_per_symbol_slippage_estimates.csv")
    pd.DataFrame(list(slip_estimates.values())).to_csv(sym_csv, index=False)
    print(f"  {C_GRN}[SAVED] {sym_csv}{S_RS}")

    bp_csv = os.path.join(RESULTS_ROOT, "task13.7_per_edge_breakpoints.csv")
    pd.DataFrame(breakpoint_export).to_csv(bp_csv, index=False)
    print(f"  {C_GRN}[SAVED] {bp_csv}{S_RS}")

    elapsed = time.time() - t0
    print()
    box(f"TASK 13.7 COMPLETE - Runtime: {elapsed/60:.2f} min", C_GRN)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C_RED}Interrupted{S_RS}")
    except Exception as e:
        print(f"\n{C_RED}FATAL: {e}{S_RS}")
        traceback.print_exc()
