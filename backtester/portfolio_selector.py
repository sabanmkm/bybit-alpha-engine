# ============================================================
# OPTIMAL PORTFOLIO SELECTOR — FULL PYTHON ENGINE
# Version 1.1 - Pandas 2.2+ Compatible
# ============================================================
import os
import sys
import time
import warnings
import traceback
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings("ignore")

# Dependency check
missing = []
try: import pandas as pd
except ImportError: missing.append("pandas")
try: import numpy as np
except ImportError: missing.append("numpy")
try: from scipy import stats
except ImportError: missing.append("scipy")
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    missing.append("colorama")
try: from tabulate import tabulate
except ImportError: missing.append("tabulate")

if missing:
    print("Missing packages: " + " ".join(missing))
    sys.exit(1)

# ============================================================
# PANDAS VERSION COMPATIBILITY LAYER
# ============================================================
def _pd_version_tuple():
    try:
        return tuple(int(x) for x in pd.__version__.split(".")[:2])
    except Exception:
        return (2, 2)

_PD_VER = _pd_version_tuple()

# Pandas 2.2+ deprecated 'M', 'Q', 'Y' in favor of 'ME', 'QE', 'YE'
FREQ_MONTHLY = "ME" if _PD_VER >= (2, 2) else "M"
FREQ_QUARTER = "QE" if _PD_VER >= (2, 2) else "Q"
FREQ_YEARLY = "YE" if _PD_VER >= (2, 2) else "Y"

def safe_resample(series_or_df, freq, agg_fn=None):
    """Version-safe resample that gracefully handles pandas API changes."""
    try:
        r = series_or_df.resample(freq)
        if agg_fn is None:
            return r
        return r.apply(agg_fn)
    except (ValueError, KeyError) as e:
        # Try fallback frequency
        fallback = "M" if freq == "ME" else ("Q" if freq == "QE" else ("Y" if freq == "YE" else freq))
        try:
            r = series_or_df.resample(fallback)
            if agg_fn is None:
                return r
            return r.apply(agg_fn)
        except Exception:
            return None
    except Exception:
        return None

# ============================================================
# CONFIGURATION
# ============================================================
RESULTS_ROOT = r"C:\BybitBacktest\results"
MASTER_CSV = os.path.join(RESULTS_ROOT, "master_ranking.csv")
TRADE_LOG_CSV = os.path.join(RESULTS_ROOT, "trade_log.csv")

# Gate thresholds
MIN_TRADES = 30
MIN_MONTHLY_FREQ = 0.5
MIN_ACTIVE_MONTHS = 12
MIN_PROFIT_FACTOR = 1.20
MIN_TOTAL_RETURN = 0.0
MIN_EXPECTANCY = 0.0
MIN_CAGR = 2.0
MIN_SHARPE = 0.50
MIN_SORTINO = 0.75
MIN_CALMAR = 0.50
MAX_DD_LIMIT = -25.0
MAX_DD_DURATION = 500
MIN_RECOVERY_FACTOR = 1.0
MIN_WIN_RATE = 40.0
MIN_AVG_WIN_LOSS = 1.0
MAX_LOSS_STREAK = 15
MIN_SKEWNESS = -1.0
CATASTROPHE_DD = -50.0
EXCLUDED_STRATEGIES = ["S23_StochRSI"]
MAX_ULCER_INDEX = 20.0

PORTFOLIO_A_SIZE = 5
PORTFOLIO_B_SIZE = 10
MAX_PER_SYMBOL = 2
MAX_PER_STRATEGY = 3
MIN_TIMEFRAMES = 2
MIN_SYMBOLS = 3
MAX_PAIRWISE_CORR = 0.70

TRADE_LOG_MAX_MB = 500
CHUNK_SIZE = 500000

# ============================================================
# COLORS
# ============================================================
C_CYAN = Fore.CYAN; C_YEL = Fore.YELLOW; C_GRN = Fore.GREEN
C_RED = Fore.RED; C_MAG = Fore.MAGENTA; C_WHT = Fore.WHITE
S_BR = Style.BRIGHT; S_RS = Style.RESET_ALL

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
# STEP 1: LOAD MASTER RANKING
# ============================================================
def load_master_ranking():
    if not os.path.exists(MASTER_CSV):
        print(f"{C_RED}FATAL: Master ranking not found at {MASTER_CSV}{S_RS}")
        sys.exit(1)
    df = pd.read_csv(MASTER_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    alias_map = {
        "tf": "timeframe", "pf": "profit_factor", "maxdd": "max_dd_pct",
        "winrate": "win_rate_pct", "totalreturn": "total_return_pct",
        "avgtrade": "avg_trade_pct", "avg_win_loss_ratio": "avg_win_loss",
        "dd_bars": "max_dd_bars", "return": "total_return_pct"
    }
    for old, new in alias_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    return df

# ============================================================
# STEP 2: LOAD TRADE LOG (chunked)
# ============================================================
def load_trade_log():
    if not os.path.exists(TRADE_LOG_CSV):
        print(f"{C_YEL}[WARN] Trade log not found. Using master_ranking-only mode.{S_RS}")
        return None
    size_mb = os.path.getsize(TRADE_LOG_CSV) / (1024 * 1024)
    print(f"  Trade log size: {size_mb:.1f} MB")
    try:
        if size_mb > TRADE_LOG_MAX_MB:
            print(f"  {C_YEL}Large file detected. Using chunked read (chunksize={CHUNK_SIZE}){S_RS}")
            chunks = []
            for i, chunk in enumerate(pd.read_csv(TRADE_LOG_CSV, chunksize=CHUNK_SIZE)):
                chunks.append(chunk)
                if (i+1) % 2 == 0:
                    print(f"    Loaded chunk {i+1}...")
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(TRADE_LOG_CSV)
        df.columns = [c.strip().lower() for c in df.columns]
        for c in ["entry_time", "exit_time"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        if "tf" in df.columns and "timeframe" not in df.columns:
            df = df.rename(columns={"tf": "timeframe"})
        if "pnl" in df.columns and "pnl_dollar" not in df.columns:
            df = df.rename(columns={"pnl": "pnl_dollar"})
        print(f"  {C_GRN}[OK] Loaded {len(df):,} trades{S_RS}")
        return df
    except Exception as e:
        print(f"  {C_RED}[ERR] Trade log load failed: {e}{S_RS}")
        return None

# ============================================================
# STEP 3: ENRICH FROM TRADE LOG
# ============================================================
def compute_ulcer(equity_series):
    if len(equity_series) < 2: return 0.0
    peak = equity_series.cummax()
    dd_pct = 100 * (equity_series - peak) / peak
    return float(np.sqrt((dd_pct ** 2).mean()))

def enrich_from_trade_log(master_df, trade_df):
    if trade_df is None or len(trade_df) == 0:
        for c in ["monthly_freq", "active_months", "avg_hold_hours",
                  "pct_overnight", "skewness", "tail_ratio",
                  "recovery_factor", "ulcer_index", "consec_win",
                  "consec_loss"]:
            master_df[c] = np.nan
        return master_df

    print(f"  Enriching {len(master_df)} setups from trade log...")
    if "timeframe" not in trade_df.columns and "tf" in trade_df.columns:
        trade_df = trade_df.rename(columns={"tf": "timeframe"})

    enrich_rows = []
    grouped = trade_df.groupby(["strategy", "symbol", "timeframe"])
    total_groups = len(grouped)
    counter = 0
    for (strat, sym, tf), grp in grouped:
        counter += 1
        if counter % 100 == 0:
            print(f"    Processed {counter}/{total_groups} groups...")
        try:
            grp = grp.sort_values("entry_time").reset_index(drop=True)
            n = len(grp)
            if n < 3: continue
            pnl_col = "pnl_dollar" if "pnl_dollar" in grp.columns else "pnl"
            if pnl_col not in grp.columns: continue
            pnls = grp[pnl_col].astype(float).values

            months = pd.to_datetime(grp["entry_time"]).dt.to_period("M").nunique()
            date_span_days = (grp["exit_time"].max() - grp["entry_time"].min()).days
            date_span_months = max(date_span_days / 30.44, 1)
            monthly_freq = n / date_span_months

            hold_hrs = (grp["exit_time"] - grp["entry_time"]).dt.total_seconds() / 3600
            avg_hold = float(hold_hrs.mean())
            pct_overnight = 100.0 * float((hold_hrs > 24).mean())

            pnl_pct_col = "pnl_pct" if "pnl_pct" in grp.columns else pnl_col
            pcts = grp[pnl_pct_col].astype(float).values
            skewness = float(stats.skew(pcts)) if len(pcts) > 2 else 0.0

            p95 = np.percentile(pcts, 95)
            p05 = np.percentile(pcts, 5)
            tail_ratio = abs(p95) / abs(p05) if abs(p05) > 0.0001 else 0.0

            wins_bool = pnls > 0
            cur_w = cur_l = max_w = max_l = 0
            for w in wins_bool:
                if w:
                    cur_w += 1; cur_l = 0
                    if cur_w > max_w: max_w = cur_w
                else:
                    cur_l += 1; cur_w = 0
                    if cur_l > max_l: max_l = cur_l

            eq = 10000 + np.cumsum(pnls)
            eq_series = pd.Series(eq)
            ulcer = compute_ulcer(eq_series)

            peak = np.maximum.accumulate(eq)
            dd_dollars = eq - peak
            max_dd_dollars = abs(dd_dollars.min())
            net_profit = eq[-1] - 10000
            recovery = net_profit / max_dd_dollars if max_dd_dollars > 0 else (999 if net_profit > 0 else 0)

            enrich_rows.append({
                "strategy": strat, "symbol": sym, "timeframe": tf,
                "monthly_freq": round(monthly_freq, 3),
                "active_months": int(months),
                "avg_hold_hours": round(avg_hold, 2),
                "pct_overnight": round(pct_overnight, 2),
                "skewness": round(skewness, 3),
                "tail_ratio": round(tail_ratio, 3),
                "recovery_factor": round(recovery, 3),
                "ulcer_index": round(ulcer, 3),
                "consec_win": max_w,
                "consec_loss": max_l
            })
        except Exception:
            continue

    enrich_df = pd.DataFrame(enrich_rows)
    print(f"  {C_GRN}[OK] Enriched {len(enrich_df)} setups{S_RS}")
    merged = master_df.merge(enrich_df, on=["strategy", "symbol", "timeframe"], how="left")
    return merged

# ============================================================
# STEP 4: SIX-GATE FILTERING
# ============================================================
def apply_gates(df):
    funnel = {}
    initial = len(df)
    print(f"\n  Starting pool: {initial} setups")

    critical = ["sharpe", "trades", "profit_factor", "max_dd_pct"]
    for c in critical:
        if c not in df.columns:
            print(f"  {C_RED}[ERR] Missing critical column: {c}{S_RS}")
            return df, funnel
    df = df.dropna(subset=critical).copy()
    print(f"  After NaN cleanup: {len(df)} setups")
    funnel["initial"] = initial
    funnel["after_nan"] = len(df)

    before = len(df)
    df = df[df["trades"] >= MIN_TRADES].copy()
    if "monthly_freq" in df.columns:
        df = df[(df["monthly_freq"] >= MIN_MONTHLY_FREQ) | (df["monthly_freq"].isna())].copy()
    if "active_months" in df.columns:
        df = df[(df["active_months"] >= MIN_ACTIVE_MONTHS) | (df["active_months"].isna())].copy()
    funnel["gate1_statistical"] = len(df)
    print(f"  Gate 1 (Statistical): {before} -> {len(df)} ({before-len(df)} eliminated)")

    before = len(df)
    df = df[df["profit_factor"] >= MIN_PROFIT_FACTOR].copy()
    if "total_return_pct" in df.columns:
        df = df[df["total_return_pct"] > MIN_TOTAL_RETURN].copy()
    if "expectancy" in df.columns:
        df = df[df["expectancy"] > MIN_EXPECTANCY].copy()
    if "cagr_pct" in df.columns:
        df = df[df["cagr_pct"] > MIN_CAGR].copy()
    funnel["gate2_profitability"] = len(df)
    print(f"  Gate 2 (Profitability): {before} -> {len(df)} ({before-len(df)} eliminated)")

    before = len(df)
    df = df[df["sharpe"] >= MIN_SHARPE].copy()
    if "sortino" in df.columns:
        df = df[df["sortino"] >= MIN_SORTINO].copy()
    if "calmar" in df.columns:
        df = df[df["calmar"] >= MIN_CALMAR].copy()
    funnel["gate3_risk_adj"] = len(df)
    print(f"  Gate 3 (Risk-Adjusted): {before} -> {len(df)} ({before-len(df)} eliminated)")

    before = len(df)
    df = df[df["max_dd_pct"] > MAX_DD_LIMIT].copy()
    if "max_dd_bars" in df.columns:
        df = df[df["max_dd_bars"] < MAX_DD_DURATION].copy()
    if "recovery_factor" in df.columns:
        df = df[(df["recovery_factor"] > MIN_RECOVERY_FACTOR) | (df["recovery_factor"].isna())].copy()
    funnel["gate4_dd_survival"] = len(df)
    print(f"  Gate 4 (Drawdown Survival): {before} -> {len(df)} ({before-len(df)} eliminated)")

    before = len(df)
    if "win_rate_pct" in df.columns:
        df = df[df["win_rate_pct"] >= MIN_WIN_RATE].copy()
    if "avg_win_loss" in df.columns:
        df = df[df["avg_win_loss"] >= MIN_AVG_WIN_LOSS].copy()
    if "consec_loss" in df.columns:
        df = df[(df["consec_loss"] <= MAX_LOSS_STREAK) | (df["consec_loss"].isna())].copy()
    if "skewness" in df.columns:
        df = df[(df["skewness"] > MIN_SKEWNESS) | (df["skewness"].isna())].copy()
    funnel["gate5_trade_quality"] = len(df)
    print(f"  Gate 5 (Trade Quality): {before} -> {len(df)} ({before-len(df)} eliminated)")

    before = len(df)
    df = df[df["max_dd_pct"] > CATASTROPHE_DD].copy()
    for ex in EXCLUDED_STRATEGIES:
        df = df[~df["strategy"].str.contains(ex, case=False, na=False)].copy()
    if "ulcer_index" in df.columns:
        df = df[(df["ulcer_index"] < MAX_ULCER_INDEX) | (df["ulcer_index"].isna())].copy()
    funnel["gate6_catastrophe"] = len(df)
    print(f"  Gate 6 (Catastrophe): {before} -> {len(df)} ({before-len(df)} eliminated)")

    return df, funnel

# ============================================================
# STEP 5: ALPHA SCORING
# ============================================================
def minmax(series):
    s = pd.to_numeric(series, errors="coerce").fillna(series.median() if len(series) > 0 else 0)
    mn, mx = s.min(), s.max()
    if mx - mn < 1e-9:
        return pd.Series([50.0] * len(s), index=s.index)
    return 100 * (s - mn) / (mx - mn)

def minmax_inverse(series):
    return 100 - minmax(series)

def calc_alpha_score(df):
    if len(df) == 0: return df
    df = df.copy()
    df["_sharpe_s"] = minmax(df.get("sharpe", pd.Series([0]*len(df))))
    df["_sortino_s"] = minmax(df.get("sortino", pd.Series([0]*len(df))))
    df["_calmar_s"] = minmax(df.get("calmar", pd.Series([0]*len(df))))
    df["_pf_s"] = minmax(df.get("profit_factor", pd.Series([0]*len(df))))
    df["_expec_s"] = minmax(df.get("expectancy", pd.Series([0]*len(df))))
    df["_tail_s"] = minmax(df.get("tail_ratio", pd.Series([0]*len(df))))
    df["_wr_s"] = minmax(df.get("win_rate_pct", pd.Series([0]*len(df))))
    df["_freq_s"] = minmax(df.get("monthly_freq", pd.Series([0]*len(df))))
    df["_streak_s"] = minmax_inverse(df.get("consec_loss", pd.Series([0]*len(df))))
    df["_ulcer_s"] = minmax_inverse(df.get("ulcer_index", pd.Series([0]*len(df))))
    df["_dd_s"] = minmax(df.get("max_dd_pct", pd.Series([0]*len(df))))
    df["_recov_s"] = minmax(df.get("recovery_factor", pd.Series([0]*len(df))))
    df["_wl_s"] = minmax(df.get("avg_win_loss", pd.Series([0]*len(df))))
    df["_skew_s"] = minmax(df.get("skewness", pd.Series([0]*len(df))))

    df["alpha_score"] = (
        0.15 * df["_sharpe_s"] + 0.10 * df["_sortino_s"] + 0.10 * df["_calmar_s"] +
        0.10 * df["_pf_s"] + 0.05 * df["_expec_s"] + 0.05 * df["_tail_s"] +
        0.05 * df["_wr_s"] + 0.05 * df["_freq_s"] + 0.05 * df["_streak_s"] +
        0.05 * df["_ulcer_s"] + 0.08 * df["_dd_s"] + 0.07 * df["_recov_s"] +
        0.05 * df["_wl_s"] + 0.05 * df["_skew_s"]
    )
    df["alpha_score"] = df["alpha_score"].round(2)
    drop_cols = [c for c in df.columns if c.startswith("_") and c.endswith("_s")]
    df = df.drop(columns=drop_cols)
    return df.sort_values("alpha_score", ascending=False).reset_index(drop=True)

# ============================================================
# STEP 6: PORTFOLIO CONSTRUCTION
# ============================================================
def build_equity_curve(trade_df, strategy, symbol, tf):
    if trade_df is None: return None
    grp = trade_df[
        (trade_df["strategy"] == strategy) &
        (trade_df["symbol"] == symbol) &
        (trade_df["timeframe"] == tf)
    ].copy()
    if len(grp) < 3: return None
    grp = grp.sort_values("exit_time")
    pnl_col = "pnl_dollar" if "pnl_dollar" in grp.columns else "pnl"
    if pnl_col not in grp.columns: return None
    grp["equity"] = 10000 + grp[pnl_col].cumsum()
    return grp[["exit_time", "equity"]].set_index("exit_time")["equity"]

def compute_correlation(eq1, eq2):
    try:
        d1 = eq1.resample("D").last().ffill().pct_change().dropna()
        d2 = eq2.resample("D").last().ffill().pct_change().dropna()
        common = d1.index.intersection(d2.index)
        if len(common) < 20: return 0.0
        return float(d1.loc[common].corr(d2.loc[common]))
    except Exception:
        return 0.0

def select_portfolio(candidates, trade_df, target_size, relax_corr=False):
    selected = []
    selected_eq = []
    sym_count = {}
    strat_count = {}

    for _, row in candidates.iterrows():
        if len(selected) >= target_size: break
        sym, strat, tf = row["symbol"], row["strategy"], row["timeframe"]

        if sym_count.get(sym, 0) >= MAX_PER_SYMBOL: continue
        if strat_count.get(strat, 0) >= MAX_PER_STRATEGY: continue

        if not relax_corr and trade_df is not None:
            new_eq = build_equity_curve(trade_df, strat, sym, tf)
            if new_eq is not None:
                too_correlated = False
                for existing_eq in selected_eq:
                    if existing_eq is None: continue
                    corr = compute_correlation(new_eq, existing_eq)
                    if corr > MAX_PAIRWISE_CORR:
                        too_correlated = True
                        break
                if too_correlated: continue
                selected.append(row)
                selected_eq.append(new_eq)
            else:
                selected.append(row)
                selected_eq.append(None)
        else:
            selected.append(row)
            if trade_df is not None:
                selected_eq.append(build_equity_curve(trade_df, strat, sym, tf))
            else:
                selected_eq.append(None)

        sym_count[sym] = sym_count.get(sym, 0) + 1
        strat_count[strat] = strat_count.get(strat, 0) + 1

    if len(selected) < target_size and not relax_corr:
        print(f"  {C_YEL}[!] Could not fill portfolio of {target_size} with correlation gate. Relaxing correlation constraint.{S_RS}")
        return select_portfolio(candidates, trade_df, target_size, relax_corr=True)

    if len(selected) == 0:
        return pd.DataFrame(), []
    return pd.DataFrame(selected).reset_index(drop=True), selected_eq

# ============================================================
# STEP 7: PORTFOLIO METRICS (Pandas 2.2+ safe)
# ============================================================
def portfolio_metrics(port_df, port_equities, trade_df):
    m = {"cagr": 0, "sharpe": 0, "max_dd": 0, "calmar": 0,
         "win_rate": 0, "pf": 0, "avg_corr": 0,
         "n_symbols": 0, "n_strategies": 0, "n_timeframes": 0,
         "monthly_trades": 0, "worst_month": 0, "best_month": 0,
         "pct_profitable_months": 0, "start_equity": 0, "end_equity": 0,
         "peak_equity": 0, "trough_equity": 0}

    if len(port_df) == 0: return m

    m["n_symbols"] = int(port_df["symbol"].nunique())
    m["n_strategies"] = int(port_df["strategy"].nunique())
    m["n_timeframes"] = int(port_df["timeframe"].nunique())

    valid_eqs = [eq for eq in port_equities if eq is not None]
    if len(valid_eqs) == 0:
        return m

    daily_frames = []
    for eq in valid_eqs:
        try:
            d = eq.resample("D").last().ffill()
            daily_frames.append(d)
        except Exception:
            continue
    if len(daily_frames) == 0: return m

    try:
        combined = pd.concat(daily_frames, axis=1).ffill().fillna(10000)
        combined.columns = [f"eq{i}" for i in range(len(daily_frames))]
        n_members = len(daily_frames)
        port_eq = combined.sum(axis=1) / n_members
        port_eq = port_eq.dropna()
    except Exception as e:
        print(f"  {C_YEL}[WARN] Equity combination failed: {e}{S_RS}")
        return m

    if len(port_eq) < 5: return m

    m["start_equity"] = round(float(port_eq.iloc[0]), 2)
    m["end_equity"] = round(float(port_eq.iloc[-1]), 2)
    m["peak_equity"] = round(float(port_eq.max()), 2)
    m["trough_equity"] = round(float(port_eq.min()), 2)

    days = (port_eq.index[-1] - port_eq.index[0]).days
    years = max(days / 365.25, 0.1)
    if port_eq.iloc[0] > 0:
        m["cagr"] = round(((port_eq.iloc[-1] / port_eq.iloc[0]) ** (1/years) - 1) * 100, 2)

    daily_ret = port_eq.pct_change().dropna()
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        m["sharpe"] = round(float(daily_ret.mean() / daily_ret.std() * np.sqrt(365)), 3)

    peak = port_eq.cummax()
    dd = (port_eq - peak) / peak
    m["max_dd"] = round(float(dd.min() * 100), 2)
    if m["max_dd"] < 0:
        m["calmar"] = round(m["cagr"] / abs(m["max_dd"]), 3)

    # Trade-level stats
    if trade_df is not None:
        try:
            keys = set(zip(port_df["strategy"], port_df["symbol"], port_df["timeframe"]))
            mask = trade_df.apply(
                lambda r: (r["strategy"], r["symbol"], r["timeframe"]) in keys, axis=1
            )
            pt_trades = trade_df[mask]
            pnl_col = "pnl_dollar" if "pnl_dollar" in pt_trades.columns else "pnl"
            if pnl_col in pt_trades.columns and len(pt_trades) > 0:
                wins = (pt_trades[pnl_col] > 0).sum()
                m["win_rate"] = round(100 * wins / len(pt_trades), 2)
                gp = pt_trades[pt_trades[pnl_col] > 0][pnl_col].sum()
                gl = abs(pt_trades[pt_trades[pnl_col] < 0][pnl_col].sum())
                m["pf"] = round(gp / gl, 3) if gl > 0 else 999
                if "entry_time" in pt_trades.columns:
                    months = pt_trades["entry_time"].dt.to_period("M").nunique()
                    m["monthly_trades"] = round(len(pt_trades) / max(months, 1), 1)
        except Exception as e:
            print(f"  {C_YEL}[WARN] Trade stats failed: {e}{S_RS}")

    # Monthly returns — SAFE resample
    try:
        monthly = safe_resample(daily_ret, FREQ_MONTHLY, lambda x: (1+x).prod() - 1)
        if monthly is not None and len(monthly) > 0:
            m["worst_month"] = round(float(monthly.min()) * 100, 2)
            m["best_month"] = round(float(monthly.max()) * 100, 2)
            m["pct_profitable_months"] = round(100 * (monthly > 0).mean(), 2)
    except Exception as e:
        print(f"  {C_YEL}[WARN] Monthly resample failed: {e}{S_RS}")

    # Avg pairwise correlation
    if len(valid_eqs) >= 2:
        try:
            corrs = []
            for i in range(len(valid_eqs)):
                for j in range(i+1, len(valid_eqs)):
                    c = compute_correlation(valid_eqs[i], valid_eqs[j])
                    corrs.append(c)
            if corrs:
                m["avg_corr"] = round(float(np.mean(corrs)), 3)
        except Exception:
            pass

    return m

# ============================================================
# PRINTERS
# ============================================================
def print_portfolio_table(port_df, name):
    if len(port_df) == 0:
        print(f"{C_RED}  [!] Portfolio {name} is EMPTY{S_RS}")
        return
    disp_cols = ["strategy", "symbol", "timeframe", "trades", "sharpe",
                 "calmar", "profit_factor", "max_dd_pct", "win_rate_pct",
                 "alpha_score"]
    disp_cols = [c for c in disp_cols if c in port_df.columns]
    tbl = []
    for i, row in port_df.iterrows():
        r = [i+1] + [row.get(c, "-") for c in disp_cols]
        tbl.append(r)
    headers = ["#"] + [c.replace("_", " ").title() for c in disp_cols]
    print(tabulate(tbl, headers=headers, tablefmt="grid"))

def print_portfolio_metrics(m):
    tbl = [
        ["Portfolio CAGR %", m["cagr"]],
        ["Portfolio Sharpe", m["sharpe"]],
        ["Portfolio Max DD %", m["max_dd"]],
        ["Portfolio Calmar", m["calmar"]],
        ["Portfolio Win Rate %", m["win_rate"]],
        ["Portfolio Profit Factor", m["pf"]],
        ["Avg Pairwise Correlation", m["avg_corr"]],
        ["Unique Symbols", m["n_symbols"]],
        ["Unique Strategies", m["n_strategies"]],
        ["Unique Timeframes", m["n_timeframes"]],
        ["Monthly Trades (combined)", m["monthly_trades"]],
        ["Worst Month %", m["worst_month"]],
        ["Best Month %", m["best_month"]],
        ["% Profitable Months", m["pct_profitable_months"]],
        ["Start Equity", m["start_equity"]],
        ["Peak Equity", m["peak_equity"]],
        ["Trough Equity", m["trough_equity"]],
        ["End Equity", m["end_equity"]],
    ]
    print(tabulate(tbl, headers=["Metric", "Value"], tablefmt="simple"))

def print_correlation_matrix(port_df, port_equities):
    valid_pairs = [(i, eq) for i, eq in enumerate(port_equities) if eq is not None]
    if len(valid_pairs) < 2:
        print(f"  {C_YEL}[!] Insufficient equity curves for correlation matrix{S_RS}")
        return
    n = len(valid_pairs)
    labels = [f"{port_df.iloc[i]['strategy'][:12]}/{port_df.iloc[i]['symbol'][:8]}"
              for i, _ in valid_pairs]
    matrix = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b: matrix[a][b] = 1.0
            else: matrix[a][b] = compute_correlation(valid_pairs[a][1], valid_pairs[b][1])
    rows = []
    for a in range(n):
        row = [labels[a]] + [f"{matrix[a][b]:.2f}" for b in range(n)]
        rows.append(row)
    print(tabulate(rows, headers=[""] + labels, tablefmt="simple"))

# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    box("OPTIMAL PORTFOLIO SELECTOR - FINAL REPORT", C_CYAN)
    print(f"{C_CYAN}  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{C_CYAN}  Pandas version: {pd.__version__} (Monthly Freq: '{FREQ_MONTHLY}')")
    print(f"{C_CYAN}  Master: {MASTER_CSV}")
    print(f"{C_CYAN}  Trade log: {TRADE_LOG_CSV}")

    section(1, "DATA LOADING & ENRICHMENT")
    master = load_master_ranking()
    print(f"  {C_GRN}[OK] Loaded {len(master)} setups from master_ranking{S_RS}")

    trade_df = load_trade_log()
    master = enrich_from_trade_log(master, trade_df)

    section(2, "SIX-GATE FUNNEL ANALYSIS")
    qualified, funnel = apply_gates(master)

    print(f"\n  {C_CYAN}Funnel Summary:{S_RS}")
    funnel_tbl = [[k, v] for k, v in funnel.items()]
    print(tabulate(funnel_tbl, headers=["Stage", "Setups Remaining"], tablefmt="simple"))

    if len(qualified) < PORTFOLIO_A_SIZE:
        print(f"\n  {C_YEL}[!] Only {len(qualified)} setups passed all 6 gates. Relaxing gate 3 by 20%.{S_RS}")
        MIN_SHARPE_RELAXED = MIN_SHARPE * 0.8
        master2 = master.copy()
        master2 = master2.dropna(subset=["sharpe", "trades", "profit_factor", "max_dd_pct"]).copy()
        master2 = master2[master2["trades"] >= MIN_TRADES].copy()
        master2 = master2[master2["profit_factor"] >= MIN_PROFIT_FACTOR].copy()
        master2 = master2[master2["sharpe"] >= MIN_SHARPE_RELAXED].copy()
        master2 = master2[master2["max_dd_pct"] > MAX_DD_LIMIT].copy()
        master2 = master2[master2["max_dd_pct"] > CATASTROPHE_DD].copy()
        for ex in EXCLUDED_STRATEGIES:
            master2 = master2[~master2["strategy"].str.contains(ex, case=False, na=False)].copy()
        qualified = master2

    section(3, "ALPHA SCORING (WEIGHTED)")
    qualified = calc_alpha_score(qualified)
    print(f"  {C_GRN}[OK] Alpha scored {len(qualified)} candidates{S_RS}")

    section(4, "QUALIFIED CANDIDATES RANKING")
    if len(qualified) > 0:
        disp_cols = ["strategy", "symbol", "timeframe", "trades",
                     "monthly_freq", "sharpe", "sortino", "calmar",
                     "profit_factor", "max_dd_pct", "win_rate_pct", "alpha_score"]
        disp_cols = [c for c in disp_cols if c in qualified.columns]
        tbl = []
        for i, row in qualified.iterrows():
            r = [i+1] + [row.get(c, "-") for c in disp_cols]
            tbl.append(r)
        headers = ["#"] + [c.replace("_", " ").title() for c in disp_cols]
        print(tabulate(tbl[:40], headers=headers, tablefmt="grid"))
        if len(qualified) > 40:
            print(f"  ... ({len(qualified) - 40} more not shown)")
    else:
        print(f"  {C_RED}[!] NO SETUPS QUALIFIED. Aborted.{S_RS}")
        return

    qual_path = os.path.join(RESULTS_ROOT, "qualified_candidates.csv")
    qualified.to_csv(qual_path, index=False)
    print(f"  {C_GRN}[SAVED] Qualified -> {qual_path}{S_RS}")

    section(5, "PORTFOLIO A - TOP 5 (CONCENTRATED, HIGH CONVICTION)")
    port5, eq5 = select_portfolio(qualified, trade_df, PORTFOLIO_A_SIZE)
    print_portfolio_table(port5, "TOP 5")
    m5 = portfolio_metrics(port5, eq5, trade_df)
    print(f"\n  {C_CYAN}Portfolio-Level Metrics:{S_RS}")
    print_portfolio_metrics(m5)
    if len(port5) > 0:
        print(f"\n  {C_CYAN}Diversity Breakdown:{S_RS}")
        print(f"    Symbols    : {sorted(port5['symbol'].unique().tolist())}")
        print(f"    Strategies : {sorted(port5['strategy'].unique().tolist())}")
        print(f"    Timeframes : {sorted(port5['timeframe'].unique().tolist())}")
    top5_path = os.path.join(RESULTS_ROOT, "portfolio_top5.csv")
    port5.to_csv(top5_path, index=False)
    print(f"  {C_GRN}[SAVED] Top 5 -> {top5_path}{S_RS}")

    section(6, "PORTFOLIO B - TOP 10 (DIVERSIFIED, BALANCED)")
    port10, eq10 = select_portfolio(qualified, trade_df, PORTFOLIO_B_SIZE)
    print_portfolio_table(port10, "TOP 10")
    m10 = portfolio_metrics(port10, eq10, trade_df)
    print(f"\n  {C_CYAN}Portfolio-Level Metrics:{S_RS}")
    print_portfolio_metrics(m10)
    if len(port10) > 0:
        print(f"\n  {C_CYAN}Diversity Breakdown:{S_RS}")
        print(f"    Symbols    : {sorted(port10['symbol'].unique().tolist())}")
        print(f"    Strategies : {sorted(port10['strategy'].unique().tolist())}")
        print(f"    Timeframes : {sorted(port10['timeframe'].unique().tolist())}")
        print(f"\n  {C_CYAN}Pairwise Correlation Matrix:{S_RS}")
        print_correlation_matrix(port10, eq10)
    top10_path = os.path.join(RESULTS_ROOT, "portfolio_top10.csv")
    port10.to_csv(top10_path, index=False)
    print(f"  {C_GRN}[SAVED] Top 10 -> {top10_path}{S_RS}")

    section(7, "PORTFOLIO COMPARISON")
    comp = [
        ["Members", len(port5), len(port10)],
        ["CAGR %", m5["cagr"], m10["cagr"]],
        ["Sharpe", m5["sharpe"], m10["sharpe"]],
        ["Max DD %", m5["max_dd"], m10["max_dd"]],
        ["Calmar", m5["calmar"], m10["calmar"]],
        ["Profit Factor", m5["pf"], m10["pf"]],
        ["Win Rate %", m5["win_rate"], m10["win_rate"]],
        ["Monthly Trades", m5["monthly_trades"], m10["monthly_trades"]],
        ["Unique Symbols", m5["n_symbols"], m10["n_symbols"]],
        ["Avg Correlation", m5["avg_corr"], m10["avg_corr"]],
        ["% Profitable Months", m5["pct_profitable_months"], m10["pct_profitable_months"]],
    ]
    print(tabulate(comp, headers=["Metric", "Top 5", "Top 10"], tablefmt="grid"))

    reco = "TOP 5" if m5["sharpe"] > m10["sharpe"] else "TOP 10"
    print(f"\n  {C_GRN}RECOMMENDATION: {reco} portfolio has superior risk-adjusted return.{S_RS}")

    section(8, "RISK PROFILE RECOMMENDATIONS")
    conserv = qualified[qualified["max_dd_pct"] > -10].head(5)
    balanced = qualified[qualified["max_dd_pct"] > -20].head(5)
    aggress = qualified[qualified["max_dd_pct"] > -30].head(5)

    print(f"\n  {C_GRN}[CONSERVATIVE] MaxDD < 10% — Capital Preservation Priority:{S_RS}")
    if len(conserv) > 0:
        for _, r in conserv.iterrows():
            print(f"    - {r['strategy']:<28} {r['symbol']:<22} {r['timeframe']:<3} "
                  f"Sharpe={r['sharpe']:.2f} DD={r['max_dd_pct']:.2f}% Alpha={r['alpha_score']:.1f}")
        print(f"    Suggested allocation: Equal weight across all 5 members.")

    print(f"\n  {C_YEL}[BALANCED] MaxDD < 20% — Growth with Controlled Risk:{S_RS}")
    if len(balanced) > 0:
        for _, r in balanced.iterrows():
            print(f"    - {r['strategy']:<28} {r['symbol']:<22} {r['timeframe']:<3} "
                  f"Sharpe={r['sharpe']:.2f} DD={r['max_dd_pct']:.2f}% Alpha={r['alpha_score']:.1f}")
        print(f"    Suggested allocation: Inverse-volatility weighted.")

    print(f"\n  {C_RED}[AGGRESSIVE] MaxDD < 30% — Maximum Return Focus:{S_RS}")
    if len(aggress) > 0:
        for _, r in aggress.iterrows():
            print(f"    - {r['strategy']:<28} {r['symbol']:<22} {r['timeframe']:<3} "
                  f"Sharpe={r['sharpe']:.2f} DD={r['max_dd_pct']:.2f}% Alpha={r['alpha_score']:.1f}")
        print(f"    Suggested allocation: Alpha-score weighted.")

    section(9, "MONTHLY EXPECTATIONS (based on TOP 5)")
    if m5["cagr"] > 0:
        monthly_exp = ((1 + m5["cagr"]/100) ** (1/12) - 1) * 100
        print(f"  Expected monthly return    : {monthly_exp:.2f}%")
        print(f"  Historical worst month     : {m5['worst_month']}%")
        print(f"  Historical best month      : {m5['best_month']}%")
        print(f"  Probability of profit month: {m5['pct_profitable_months']}%")
        print(f"  Expected trades / month    : {m5['monthly_trades']}")

    section(10, "EXECUTIVE SUMMARY & NEXT STEPS")
    if len(qualified) > 0 and m5["sharpe"] > 0.8:
        verdict = f"{C_GRN}GO — Portfolio has statistically robust edge.{S_RS}"
    elif len(qualified) > 0:
        verdict = f"{C_YEL}CONDITIONAL GO — Edge exists but demands walk-forward validation.{S_RS}"
    else:
        verdict = f"{C_RED}NO-GO — Insufficient qualified candidates.{S_RS}"

    print(f"\n  Verdict: {verdict}")

    if len(qualified) > 0:
        print(f"\n  {C_GRN}Top 3 Edges to Trust:{S_RS}")
        for i, (_, r) in enumerate(qualified.head(3).iterrows(), 1):
            print(f"    {i}. {r['strategy']} on {r['symbol']} @ {r['timeframe']} "
                  f"(Alpha={r['alpha_score']:.1f}, Sharpe={r['sharpe']:.2f})")

        print(f"\n  {C_RED}Top 3 Risks to Monitor:{S_RS}")
        print(f"    1. Live slippage exceeding 0.03% assumption (retest with 0.10%)")
        print(f"    2. Regime shift breaking 2022-2025 patterns (walk-forward on 2026)")
        print(f"    3. Correlation spikes during market crashes (all crypto -> 1.0)")

        print(f"\n  {C_CYAN}Recommended Next Task:{S_RS}")
        print(f"    -> Walk-forward validation on sealed 2026 OOS data")
        print(f"    -> Optuna hyperparameter tuning on the Top 3 setups")
        print(f"    -> Monte Carlo stress testing (2,000 randomized trade sequences)")

    elapsed = time.time() - t0
    print()
    box(f"REPORT COMPLETE - Runtime: {elapsed:.2f} seconds", C_GRN)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C_RED}Interrupted by user{S_RS}")
    except Exception as e:
        print(f"\n{C_RED}FATAL: {e}{S_RS}")
        traceback.print_exc()
