"""
TASK 13.6: FUNDING RATE COST INTEGRATION
Fetches historical Bybit funding rates and applies them to every trade.
Reveals the TRUE net-of-cost portfolio performance.
"""
import os
import sys
import time
import json
import warnings
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from colorama import Fore, Style, init as colorama_init
    from tabulate import tabulate
    colorama_init(autoreset=True)
except ImportError as e:
    print(f"Missing package: {e}"); sys.exit(1)

# ============================================================
# CONFIGURATION
# ============================================================
DATA_ROOT = r"C:\BybitBacktest\data\resampled"
FUNDING_ROOT = r"C:\BybitBacktest\data\funding_rates"
RESULTS_ROOT = r"C:\BybitBacktest\results"
os.makedirs(FUNDING_ROOT, exist_ok=True)
os.makedirs(RESULTS_ROOT, exist_ok=True)

TRADE_LOG_CSV = os.path.join(RESULTS_ROOT, "task11.2_true_portfolio_trade_log.csv")

BYBIT_API = "https://api.bybit.com/v5/market/funding/history"
RATE_LIMIT_SLEEP = 0.6  # 100 req/min safe

# Funding settlement times UTC (hours)
FUNDING_HOURS = [0, 8, 16]

# Portfolio dates
IS_START = pd.Timestamp("2022-01-01")
IS_END = pd.Timestamp("2025-12-31 23:59:59")
OOS_START = pd.Timestamp("2026-01-01")
OOS_END = pd.Timestamp("2026-08-31 23:59:59")
FULL_START = IS_START
FULL_END = OOS_END

STARTING_CAPITAL = 10000.0

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
# THE 15 UNIQUE BYBIT SYMBOLS (mapped from portfolio symbol names)
# Portfolio uses "XMR_USDT_USDT" format; Bybit uses "XMRUSDT"
# ============================================================
PORTFOLIO_SYMBOL_MAP = {
    "XMR_USDT_USDT":  "XMRUSDT",
    "ATOM_USDT_USDT": "ATOMUSDT",
    "ADA_USDT_USDT":  "ADAUSDT",
    "BSV_USDT_USDT":  "BSVUSDT",
    "ETC_USDT_USDT":  "ETCUSDT",
    "AVAX_USDT_USDT": "AVAXUSDT",
    "QTUM_USDT_USDT": "QTUMUSDT",
    "TWT_USDT_USDT":  "TWTUSDT",
    "DOT_USDT_USDT":  "DOTUSDT",
    "XLM_USDT_USDT":  "XLMUSDT",
    "ETH_USDT_USDT":  "ETHUSDT",
    "SAND_USDT_USDT": "SANDUSDT",
    "OP_USDT_USDT":   "OPUSDT",
    "INJ_USDT_USDT":  "INJUSDT",
    "DOGE_USDT_USDT": "DOGEUSDT",
}

UNIQUE_SYMBOLS = list(PORTFOLIO_SYMBOL_MAP.values())

# ============================================================
# BYBIT SESSION SETUP
# ============================================================
def build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "Task13.6-FundingIntegrator/1.0"})
    return s

# ============================================================
# FETCH FUNDING RATES (paginated with cache)
# ============================================================
def fetch_funding_history(session, symbol: str, start_ts_ms: int, end_ts_ms: int) -> pd.DataFrame:
    """Fetches all funding rates for symbol in [start_ts, end_ts] range."""
    all_records = []
    current_end = end_ts_ms
    calls = 0
    max_calls = 300  # safety cap ~ 60,000 records
    consecutive_empty = 0

    while calls < max_calls:
        params = {
            "category": "linear",
            "symbol": symbol,
            "startTime": start_ts_ms,
            "endTime": current_end,
            "limit": 200,
        }
        try:
            resp = session.get(BYBIT_API, params=params, timeout=15)
            calls += 1
            if resp.status_code != 200:
                print(f"    {C_YEL}[HTTP {resp.status_code}] {symbol}, page {calls}{S_RS}")
                time.sleep(2)
                consecutive_empty += 1
                if consecutive_empty >= 3: break
                continue
            data = resp.json()
            if data.get("retCode") != 0:
                print(f"    {C_YEL}[API err] {symbol}: {data.get('retMsg')}{S_RS}")
                break
            recs = data.get("result", {}).get("list", [])
            if not recs:
                consecutive_empty += 1
                if consecutive_empty >= 2: break
                time.sleep(RATE_LIMIT_SLEEP)
                continue
            consecutive_empty = 0
            all_records.extend(recs)
            # Bybit returns most recent first; advance current_end backwards
            oldest_ts_in_page = int(recs[-1]["fundingRateTimestamp"])
            if oldest_ts_in_page <= start_ts_ms: break
            current_end = oldest_ts_in_page - 1
            time.sleep(RATE_LIMIT_SLEEP)
        except Exception as e:
            print(f"    {C_RED}[EXC] {symbol}: {e}{S_RS}")
            time.sleep(2)
            consecutive_empty += 1
            if consecutive_empty >= 3: break

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["fundingRateTimestamp"] = pd.to_numeric(df["fundingRateTimestamp"])
    df["fundingRate"] = pd.to_numeric(df["fundingRate"])
    df["datetime"] = pd.to_datetime(df["fundingRateTimestamp"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.drop_duplicates(subset=["fundingRateTimestamp"]).sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "fundingRate", "fundingRateTimestamp", "symbol"]]

def get_or_fetch_funding(session, symbol: str, start_ts_ms: int, end_ts_ms: int) -> pd.DataFrame:
    """Returns funding data using local cache if available."""
    cache_path = os.path.join(FUNDING_ROOT, f"{symbol}.csv")
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            cache_start = df["datetime"].min()
            cache_end = df["datetime"].max()
            start_dt = pd.Timestamp(start_ts_ms, unit="ms")
            end_dt = pd.Timestamp(end_ts_ms, unit="ms")
            if cache_start <= start_dt + pd.Timedelta(days=7) and cache_end >= end_dt - pd.Timedelta(days=7):
                return df
        except Exception:
            pass
    # Fetch fresh
    print(f"    {C_YEL}Fetching {symbol}...{S_RS}", end="", flush=True)
    df = fetch_funding_history(session, symbol, start_ts_ms, end_ts_ms)
    if len(df) > 0:
        df.to_csv(cache_path, index=False)
        print(f" {C_GRN}[OK] {len(df):,} records{S_RS}")
    else:
        print(f" {C_RED}[FAIL] no data{S_RS}")
    return df

# ============================================================
# FUNDING APPLICATION TO TRADES
# ============================================================
def find_funding_events_in_range(entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> List[pd.Timestamp]:
    """Returns list of funding settlement timestamps between entry and exit (exclusive of entry, inclusive of exit)."""
    if pd.isna(entry_time) or pd.isna(exit_time) or entry_time >= exit_time:
        return []
    # Round entry_time up to next funding hour
    events = []
    cur = entry_time.replace(minute=0, second=0, microsecond=0)
    # Move to next 8h boundary
    while cur.hour not in FUNDING_HOURS or cur <= entry_time:
        cur = cur + pd.Timedelta(hours=1)
        if cur > exit_time + pd.Timedelta(hours=1): break
    while cur <= exit_time:
        if cur.hour in FUNDING_HOURS:
            events.append(cur)
        cur = cur + pd.Timedelta(hours=8)
    return events

def lookup_funding_rate(funding_df: pd.DataFrame, ts: pd.Timestamp) -> float:
    """Return funding rate at (or nearest before) given timestamp."""
    if funding_df is None or len(funding_df) == 0: return 0.0
    # Nearest lookup: find row where datetime is closest to ts, within +/- 4 hours
    diffs = (funding_df["datetime"] - ts).abs()
    idx = diffs.idxmin()
    nearest_ts = funding_df.loc[idx, "datetime"]
    if abs((nearest_ts - ts).total_seconds()) > 4 * 3600:
        return 0.0
    return float(funding_df.loc[idx, "fundingRate"])

def compute_funding_cost_for_trade(trade: dict, funding_df: pd.DataFrame) -> Tuple[float, int]:
    """Returns (funding_cost_dollars, num_settlements). Positive cost = money OUT."""
    entry_time = pd.to_datetime(trade["entry_time"])
    exit_time = pd.to_datetime(trade["exit_time"])
    events = find_funding_events_in_range(entry_time, exit_time)
    if not events: return 0.0, 0

    notional = float(trade["entry_price"]) * float(trade["size"])
    direction = 1 if trade["direction"] == "LONG" else -1

    total_cost = 0.0
    for evt_ts in events:
        rate = lookup_funding_rate(funding_df, evt_ts)
        # If LONG (direction=1) and rate > 0: LONG pays -> cost = +notional*rate
        # If LONG and rate < 0: LONG receives -> cost = +notional*rate (negative = income)
        # If SHORT (direction=-1) and rate > 0: SHORT receives -> cost = -notional*rate
        # If SHORT and rate < 0: SHORT pays -> cost = -notional*rate
        cost = notional * rate * direction
        total_cost += cost
    return total_cost, len(events)

# ============================================================
# METRIC COMPUTATION
# ============================================================
def compute_metrics(pnls: np.ndarray, days_span: float) -> Dict:
    """Compute portfolio metrics from pnl array."""
    if len(pnls) < 2:
        return {"total_return":0,"cagr":0,"sharpe":0,"sortino":0,"maxdd":0,
                "win_rate":0,"pf":0,"expectancy":0,"n_trades":len(pnls)}
    equity = STARTING_CAPITAL + np.cumsum(pnls)
    equity = np.maximum(equity, 1.0)  # prevent negative
    total_ret = (equity[-1] / STARTING_CAPITAL - 1) * 100
    years = max(days_span / 365.25, 0.01)
    cagr = ((equity[-1] / STARTING_CAPITAL) ** (1/years) - 1) * 100 if equity[-1] > 0 else 0
    # Sharpe from daily returns approx
    daily_pnl_pct = pnls / STARTING_CAPITAL  # rough
    if len(daily_pnl_pct) > 1 and daily_pnl_pct.std() > 0:
        # Annualize by trades per year
        trades_per_year = len(pnls) / years
        sharpe = daily_pnl_pct.mean() / daily_pnl_pct.std() * np.sqrt(trades_per_year)
        neg = daily_pnl_pct[daily_pnl_pct < 0]
        sortino = daily_pnl_pct.mean() / neg.std() * np.sqrt(trades_per_year) if len(neg) > 0 and neg.std() > 0 else 0
    else:
        sharpe = 0; sortino = 0
    sharpe = float(np.clip(sharpe, -20, 20))
    sortino = float(np.clip(sortino, -20, 20))
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    maxdd = float(dd.min() * 100)
    wins = (pnls > 0).sum()
    wr = 100 * wins / len(pnls)
    gp = pnls[pnls > 0].sum()
    gl = abs(pnls[pnls < 0].sum())
    pf = gp / gl if gl > 0 else (999 if gp > 0 else 0)
    exp_val = pnls.mean()
    return {
        "total_return": round(total_ret, 2),
        "cagr": round(cagr, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "maxdd": round(maxdd, 2),
        "win_rate": round(wr, 2),
        "pf": round(pf, 3),
        "expectancy": round(exp_val, 4),
        "n_trades": len(pnls),
        "final_equity": round(equity[-1], 2),
        "gross_profit": round(gp, 2),
        "gross_loss": round(gl, 2),
    }

# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    box("TASK 13.6: FUNDING RATE COST INTEGRATION", C_CYAN)
    print(f"{C_CYAN}  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{C_CYAN}  Unique symbols to fetch: {len(UNIQUE_SYMBOLS)}")
    print(f"{C_CYAN}  Portfolio symbols: {len(PORTFOLIO_SYMBOL_MAP)}")

    # SECTION 1: Fetch funding data
    section(1, "FETCHING BYBIT FUNDING HISTORY (2022-01 to 2026-08)", C_MAG)
    session = build_session()
    start_ts = int(FULL_START.timestamp() * 1000)
    end_ts = int(FULL_END.timestamp() * 1000)

    funding_data = {}
    coverage_rows = []
    for sym in UNIQUE_SYMBOLS:
        df = get_or_fetch_funding(session, sym, start_ts, end_ts)
        funding_data[sym] = df
        if len(df) > 0:
            coverage_rows.append([
                sym, len(df),
                df["datetime"].min().strftime("%Y-%m-%d"),
                df["datetime"].max().strftime("%Y-%m-%d"),
                f"{C_GRN}OK{S_RS}"
            ])
        else:
            coverage_rows.append([sym, 0, "-", "-", f"{C_RED}MISSING{S_RS}"])

    print(f"\n  {C_CYAN}Funding Data Coverage Report:{S_RS}")
    print(tabulate(coverage_rows, headers=["Symbol","Records","Start","End","Status"], tablefmt="grid"))

    # SECTION 2: Funding rate statistics
    section(2, "FUNDING RATE STATISTICS PER SYMBOL", C_CYAN)
    stat_rows = []
    for sym in UNIQUE_SYMBOLS:
        df = funding_data.get(sym)
        if df is None or len(df) == 0:
            stat_rows.append([sym, "-", "-", "-", "-", "-"])
            continue
        rates = df["fundingRate"].values
        mean_r = float(np.mean(rates))
        med_r = float(np.median(rates))
        max_r = float(np.max(rates))
        min_r = float(np.min(rates))
        pos_pct = 100 * (rates > 0).sum() / len(rates)
        # Annualized: 3 settlements per day * 365 days
        mean_annual = mean_r * 3 * 365 * 100
        stat_rows.append([
            sym,
            f"{mean_r*10000:+.2f} bps",   # per settlement in basis points
            f"{med_r*10000:+.2f} bps",
            f"{max_r*10000:+.2f} bps",
            f"{min_r*10000:+.2f} bps",
            f"{pos_pct:.1f}%",
            f"{mean_annual:+.2f}%"
        ])
    print(tabulate(stat_rows,
        headers=["Symbol","Mean/settle","Median","Max spike","Min spike","% Positive","Annualized"],
        tablefmt="grid"))

    # SECTION 3: Load trade log
    section(3, "LOADING TRADE LOG & APPLYING FUNDING", C_MAG)
    trade_df = pd.read_csv(TRADE_LOG_CSV)
    print(f"  Loaded {len(trade_df):,} trades from Task 11.2")
    trade_df["entry_time"] = pd.to_datetime(trade_df["entry_time"])
    trade_df["exit_time"] = pd.to_datetime(trade_df["exit_time"])

    # Apply funding to each trade
    print(f"  Computing funding cost per trade...")
    funding_costs = []
    funding_events = []
    for i, row in trade_df.iterrows():
        sym_portfolio = row["symbol"]
        bybit_sym = PORTFOLIO_SYMBOL_MAP.get(sym_portfolio)
        if bybit_sym is None:
            funding_costs.append(0.0)
            funding_events.append(0)
            continue
        fund_df = funding_data.get(bybit_sym)
        cost, n_events = compute_funding_cost_for_trade(row.to_dict(), fund_df)
        funding_costs.append(cost)
        funding_events.append(n_events)
        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1:,}/{len(trade_df):,} trades")
    trade_df["funding_cost"] = funding_costs
    trade_df["n_funding_events"] = funding_events
    trade_df["pnl_gross"] = trade_df["pnl"]
    trade_df["pnl_net"] = trade_df["pnl"] - trade_df["funding_cost"]

    total_funding = trade_df["funding_cost"].sum()
    print(f"\n  {C_GRN}[OK] Funding applied. Total cost: ${total_funding:,.2f}{S_RS}")
    print(f"  Trades with funding events: {(trade_df['n_funding_events'] > 0).sum():,} / {len(trade_df):,}")
    print(f"  Avg funding events per trade: {trade_df['n_funding_events'].mean():.2f}")

    # SECTION 4: Per-edge funding impact
    section(4, "PER-EDGE FUNDING IMPACT", C_CYAN)
    per_edge_rows = []
    per_edge_export = []
    # Use Standard_1.0% tier if available
    trade_df_std = trade_df[trade_df["tier"] == "Standard_1.0%"] if "tier" in trade_df.columns else trade_df

    for eid, grp in trade_df_std.groupby("edge_id"):
        strat = grp["strategy"].iloc[0]
        sym = grp["symbol"].iloc[0]
        tf = grp["tf"].iloc[0]
        n_trades = len(grp)
        n_long = (grp["direction"] == "LONG").sum()
        n_short = n_trades - n_long
        long_pct = 100 * n_long / n_trades if n_trades > 0 else 0
        gross_pnl = grp["pnl_gross"].sum()
        fund_cost = grp["funding_cost"].sum()
        net_pnl = grp["pnl_net"].sum()
        fund_pct = 100 * fund_cost / abs(gross_pnl) if abs(gross_pnl) > 0.01 else 0

        # Compute original vs net Sharpe
        days_span = max((grp["exit_time"].max() - grp["entry_time"].min()).days, 1)
        m_gross = compute_metrics(grp["pnl_gross"].values, days_span)
        m_net = compute_metrics(grp["pnl_net"].values, days_span)

        # Classify
        if m_net["sharpe"] > 1.0 and m_net["pf"] > 1.2:
            status = "DEPLOY"; col = C_GRN
        elif m_net["sharpe"] > 0.5 and m_net["pf"] > 1.0:
            status = "MARGINAL"; col = C_YEL
        else:
            status = "DROP"; col = C_RED

        per_edge_rows.append([
            int(eid), strat[:26], sym.split("_")[0], tf,
            n_trades, f"{long_pct:.0f}%",
            f"${gross_pnl:+,.2f}",
            f"{col}${fund_cost:+,.2f}{S_RS}",
            f"${net_pnl:+,.2f}",
            f"{m_gross['sharpe']:.2f}",
            f"{col}{m_net['sharpe']:.2f}{S_RS}",
            f"{col}{status}{S_RS}"
        ])
        per_edge_export.append({
            "edge_id": int(eid), "strategy": strat, "symbol": sym, "tf": tf,
            "n_trades": n_trades, "n_long": int(n_long), "n_short": int(n_short),
            "gross_pnl": round(gross_pnl, 2), "funding_cost": round(fund_cost, 2),
            "net_pnl": round(net_pnl, 2), "fund_pct_of_gross": round(fund_pct, 2),
            "gross_sharpe": m_gross["sharpe"], "net_sharpe": m_net["sharpe"],
            "gross_pf": m_gross["pf"], "net_pf": m_net["pf"],
            "gross_expectancy": m_gross["expectancy"], "net_expectancy": m_net["expectancy"],
            "status": status,
        })
    print(tabulate(per_edge_rows,
        headers=["#","Strategy","Symbol","TF","Trades","Long%","GrossPnL","FundingCost","NetPnL","GrossSh","NetSh","Status"],
        tablefmt="grid"))

    # SECTION 5: Portfolio before/after
    section(5, "PORTFOLIO-LEVEL BEFORE vs AFTER FUNDING", C_MAG)
    days_full = (trade_df_std["exit_time"].max() - trade_df_std["entry_time"].min()).days
    m_before = compute_metrics(trade_df_std["pnl_gross"].values, days_full)
    m_after = compute_metrics(trade_df_std["pnl_net"].values, days_full)
    total_gross_profit = trade_df_std[trade_df_std["pnl_gross"] > 0]["pnl_gross"].sum()
    total_funding = trade_df_std["funding_cost"].sum()
    fund_impact_pct = 100 * total_funding / total_gross_profit if total_gross_profit > 0 else 0

    comp_rows = [
        ["Total Trades", m_before["n_trades"], m_after["n_trades"], "0"],
        ["Final Equity", f"${m_before['final_equity']:,.2f}", f"${m_after['final_equity']:,.2f}",
         f"${m_after['final_equity'] - m_before['final_equity']:+,.2f}"],
        ["Total Return %", f"{m_before['total_return']}%", f"{m_after['total_return']}%",
         f"{m_after['total_return'] - m_before['total_return']:+.2f}%"],
        ["CAGR %", f"{m_before['cagr']}%", f"{m_after['cagr']}%",
         f"{m_after['cagr'] - m_before['cagr']:+.2f}%"],
        ["Sharpe", m_before["sharpe"], m_after["sharpe"],
         f"{m_after['sharpe'] - m_before['sharpe']:+.3f}"],
        ["Sortino", m_before["sortino"], m_after["sortino"],
         f"{m_after['sortino'] - m_before['sortino']:+.3f}"],
        ["Max Drawdown %", f"{m_before['maxdd']}%", f"{m_after['maxdd']}%",
         f"{m_after['maxdd'] - m_before['maxdd']:+.2f}%"],
        ["Win Rate %", f"{m_before['win_rate']}%", f"{m_after['win_rate']}%",
         f"{m_after['win_rate'] - m_before['win_rate']:+.2f}%"],
        ["Profit Factor", m_before["pf"], m_after["pf"],
         f"{m_after['pf'] - m_before['pf']:+.3f}"],
        ["Expectancy $", f"${m_before['expectancy']:.4f}", f"${m_after['expectancy']:.4f}",
         f"${m_after['expectancy'] - m_before['expectancy']:+.4f}"],
        ["Total Funding Cost", "$0.00", f"${total_funding:,.2f}", f"${total_funding:+,.2f}"],
        ["Funding as % of Gross Profit", "0%", f"{fund_impact_pct:.2f}%", f"{fund_impact_pct:+.2f}%"],
    ]
    print(tabulate(comp_rows, headers=["Metric", "BEFORE (no funding)", "AFTER (with funding)", "Delta"], tablefmt="grid"))

    # SECTION 6: Funding-adjusted sizing tiers
    section(6, "FUNDING-ADJUSTED SIZING TIER COMPARISON", C_CYAN)
    if "tier" in trade_df.columns:
        tier_rows = []
        for tier in sorted(trade_df["tier"].unique()):
            tier_grp = trade_df[trade_df["tier"] == tier]
            days_span = max((tier_grp["exit_time"].max() - tier_grp["entry_time"].min()).days, 1)
            m_g = compute_metrics(tier_grp["pnl_gross"].values, days_span)
            m_n = compute_metrics(tier_grp["pnl_net"].values, days_span)
            fund_sum = tier_grp["funding_cost"].sum()
            oos_grp = tier_grp[tier_grp["exit_time"] >= OOS_START]
            if len(oos_grp) > 5:
                days_oos = max((oos_grp["exit_time"].max() - oos_grp["entry_time"].min()).days, 1)
                m_oos = compute_metrics(oos_grp["pnl_net"].values, days_oos)
            else:
                m_oos = {"cagr":0,"sharpe":0,"maxdd":0}
            tier_rows.append([
                tier,
                f"${m_g['final_equity']:,.0f}",
                f"${m_n['final_equity']:,.0f}",
                f"${fund_sum:,.2f}",
                f"{m_n['cagr']}%",
                f"{m_n['sharpe']}",
                f"{m_n['maxdd']}%",
                f"{m_oos['cagr']}%",
                f"{m_oos['sharpe']}",
            ])
        print(tabulate(tier_rows,
            headers=["Tier","Gross Final","Net Final","Funding","Net CAGR","Net Sharpe","Net MaxDD","OOS CAGR","OOS Sharpe"],
            tablefmt="grid"))
    else:
        print(f"  {C_YEL}[!] No 'tier' column in trade log{S_RS}")

    # SECTION 7: Edge survival
    section(7, "EDGE SURVIVAL ANALYSIS", C_CYAN)
    n_deploy = sum(1 for e in per_edge_export if e["status"] == "DEPLOY")
    n_marg = sum(1 for e in per_edge_export if e["status"] == "MARGINAL")
    n_drop = sum(1 for e in per_edge_export if e["status"] == "DROP")

    # By family
    time_based = ["S30a", "S30b", "S30d"]
    time_edges = [e for e in per_edge_export if any(t in e["strategy"] for t in time_based)]
    regime_edges = [e for e in per_edge_export if any(t in e["strategy"] for t in ["R01","R13","R15","R16"])]

    survival_rows = [
        ["Total DEPLOY (Sharpe>1.0, PF>1.2)", n_deploy, "of 16"],
        ["Total MARGINAL (Sharpe 0.5-1.0)", n_marg, "of 16"],
        ["Total DROP (Sharpe<0.5 or PF<1.0)", n_drop, "of 16"],
        ["", "", ""],
        ["Time-Based edges DEPLOY", sum(1 for e in time_edges if e["status"]=="DEPLOY"), f"of {len(time_edges)}"],
        ["Time-Based edges MARGINAL", sum(1 for e in time_edges if e["status"]=="MARGINAL"), f"of {len(time_edges)}"],
        ["Time-Based edges DROP", sum(1 for e in time_edges if e["status"]=="DROP"), f"of {len(time_edges)}"],
        ["", "", ""],
        ["Regime-Adaptive edges DEPLOY", sum(1 for e in regime_edges if e["status"]=="DEPLOY"), f"of {len(regime_edges)}"],
        ["Regime-Adaptive edges MARGINAL", sum(1 for e in regime_edges if e["status"]=="MARGINAL"), f"of {len(regime_edges)}"],
        ["Regime-Adaptive edges DROP", sum(1 for e in regime_edges if e["status"]=="DROP"), f"of {len(regime_edges)}"],
    ]
    print(tabulate(survival_rows, headers=["Category", "Count", "Total"], tablefmt="grid"))

    # SECTION 8: Executive verdict
    section(8, "FINAL EXECUTIVE VERDICT", C_GRN)
    print(f"\n  Total gross portfolio profit    : ${total_gross_profit:,.2f}")
    print(f"  Total funding cost              : ${total_funding:,.2f}")
    print(f"  Funding as % of gross profit    : {fund_impact_pct:.2f}%")
    print(f"  Net portfolio Sharpe            : {m_after['sharpe']:.3f} (was {m_before['sharpe']:.3f})")
    print(f"  Net portfolio CAGR              : {m_after['cagr']:.2f}% (was {m_before['cagr']:.2f}%)")
    print(f"  Edges surviving DEPLOY threshold: {n_deploy}/16")

    if n_deploy >= 8 and m_after["sharpe"] > 1.0 and fund_impact_pct < 30:
        verdict = f"{C_GRN}{S_BR}GO - Portfolio remains profitable after funding costs. Deploy to paper.{S_RS}"
    elif n_deploy >= 5 and m_after["sharpe"] > 0.5:
        verdict = f"{C_YEL}CONDITIONAL - Deploy DEPLOY-rated edges only; monitor funding closely.{S_RS}"
    else:
        verdict = f"{C_RED}NO-GO - Funding costs materially impair edge. Return to research.{S_RS}"
    print(f"\n  {S_BR}FINAL VERDICT: {verdict}")

    # SECTION 9: Exports
    section(9, "EXPORT FILES", C_CYAN)
    trades_csv = os.path.join(RESULTS_ROOT, "task13.6_funding_adjusted_trades.csv")
    trade_df.to_csv(trades_csv, index=False)
    print(f"  {C_GRN}[SAVED] {trades_csv} ({len(trade_df):,} trades){S_RS}")

    edge_csv = os.path.join(RESULTS_ROOT, "task13.6_per_edge_funding_impact.csv")
    pd.DataFrame(per_edge_export).to_csv(edge_csv, index=False)
    print(f"  {C_GRN}[SAVED] {edge_csv}{S_RS}")

    port_csv = os.path.join(RESULTS_ROOT, "task13.6_funding_adjusted_portfolio.csv")
    port_rows = [
        {"metric":"total_return_pct", "before":m_before["total_return"], "after":m_after["total_return"]},
        {"metric":"cagr_pct",         "before":m_before["cagr"],         "after":m_after["cagr"]},
        {"metric":"sharpe",           "before":m_before["sharpe"],       "after":m_after["sharpe"]},
        {"metric":"sortino",          "before":m_before["sortino"],      "after":m_after["sortino"]},
        {"metric":"max_dd_pct",       "before":m_before["maxdd"],        "after":m_after["maxdd"]},
        {"metric":"win_rate_pct",     "before":m_before["win_rate"],     "after":m_after["win_rate"]},
        {"metric":"profit_factor",    "before":m_before["pf"],           "after":m_after["pf"]},
        {"metric":"expectancy_dollar","before":m_before["expectancy"],   "after":m_after["expectancy"]},
        {"metric":"total_funding_cost","before":0,                        "after":round(total_funding,2)},
        {"metric":"funding_pct_of_gross","before":0,                     "after":round(fund_impact_pct,2)},
        {"metric":"final_equity",     "before":m_before["final_equity"], "after":m_after["final_equity"]},
    ]
    pd.DataFrame(port_rows).to_csv(port_csv, index=False)
    print(f"  {C_GRN}[SAVED] {port_csv}{S_RS}")

    elapsed = time.time() - t0
    print()
    box(f"TASK 13.6 COMPLETE - Runtime: {elapsed/60:.2f} min", C_GRN)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C_RED}Interrupted{S_RS}")
    except Exception as e:
        print(f"\n{C_RED}FATAL: {e}{S_RS}")
        traceback.print_exc()
