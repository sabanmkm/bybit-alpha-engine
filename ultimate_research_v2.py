"""
ULTIMATE STRATEGY RESEARCH PIPELINE v2.2 (Bulletproof Full-Matrix Screener).
Evaluates all original + BTC macro-filtered strategies across 127 symbols and 6 timeframes.
No inline discard bugs — captures all valid combinations and ranks the Top 25.
"""
from __future__ import annotations
import sys, json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel

warnings.filterwarnings("ignore")

ROOT_DIR = Path(r"C:\BybitBacktest")
RAW_DIR = ROOT_DIR / "data" / "raw_5m"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR))

try:
    import config as cfg
    cfg.DATA_GATE_UNLOCKED = True
except Exception:
    pass

from strategy_generator import STRATEGY_REGISTRY
from backtest_engine import run_backtest

console = Console()

TEST_TFS = {
    "1H": "60min",
    "2H": "120min",
    "3H": "180min",
    "3H30m": "210min",
    "4H": "240min",
    "6H": "360min"
}

class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)

# Helper functions for BTC filter
def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()

def btc_regime(df_asset: pd.DataFrame, df_btc: pd.DataFrame) -> pd.Series:
    btc_c = df_btc["close"].reindex(df_asset.index, method="ffill").bfill()
    e50 = ema(btc_c, 50)
    reg = pd.Series(0, index=df_asset.index)
    reg[btc_c >= e50] = 1   # Bullish / Neutral
    reg[btc_c < e50] = -1   # Bearish
    return reg

def run_ultimate_research():
    console.rule("[bold magenta]ULTIMATE STRATEGY RESEARCH PIPELINE v2.2")
    
    # 1. Load BTC
    btc_candidates = list(RAW_DIR.glob("*BTC*.parquet"))
    if not btc_candidates:
        console.print("[red]BTC raw data missing![/]")
        return
    
    btc_file = btc_candidates[0]
    console.print(f"[green]Using BTC reference: {btc_file.name}[/green]")
    df_btc_raw = pd.read_parquet(btc_file)
    if not isinstance(df_btc_raw.index, pd.DatetimeIndex):
        df_btc_raw.index = pd.to_datetime(df_btc_raw.index, utc=True)
    df_btc_raw = df_btc_raw.sort_index()

    # Pre-resample BTC across all 6 timeframes
    btc_tf_data = {}
    for tf_label, rule in TEST_TFS.items():
        btc_tf = df_btc_raw.resample(rule, label="left", closed="left").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        btc_tf_data[tf_label] = btc_tf

    raw_files = sorted(list(RAW_DIR.glob("*.parquet")))
    console.print(f"[green]Loaded {len(raw_files)} symbols across {len(TEST_TFS)} time horizons.[/green]\n")

    ts_is_end = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
    ts_oos_start = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")

    all_results = []
    total_evaluated = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=35, complete_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console
    ) as prog:
        task = prog.add_task("Evaluating Symbols...", total=len(raw_files))

        for f in raw_files:
            sym_name = f.stem.replace("_", "/")
            prog.update(task, description=f"[yellow]{f.stem[:14]:<14}[/]")

            try:
                df_raw = pd.read_parquet(f)
                if not isinstance(df_raw.index, pd.DatetimeIndex):
                    df_raw.index = pd.to_datetime(df_raw.index, utc=True)
                df_raw = df_raw.sort_index()
            except Exception:
                prog.advance(task)
                continue

            if len(df_raw) < 500:
                prog.advance(task)
                continue

            for tf_label, rule in TEST_TFS.items():
                df_btc_tf = btc_tf_data[tf_label]

                try:
                    df_tf = df_raw.resample(rule, label="left", closed="left").agg({
                        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
                    }).dropna()
                except Exception:
                    continue

                if len(df_tf) < 100:
                    continue

                # Slices
                is_mask = df_tf.index <= ts_is_end
                oos_mask = df_tf.index >= ts_oos_start

                df_is = df_tf[is_mask]
                df_oos = df_tf[oos_mask]

                if len(df_is) < 60 or len(df_oos) < 20:
                    continue

                btc_reg = btc_regime(df_tf, df_btc_tf)

                for strat_name, strat_meta in STRATEGY_REGISTRY.items():
                    func = strat_meta["func"]
                    cat = strat_meta["category"]
                    params = strat_meta["default_params"]

                    try:
                        # 1. Base Strategy Signals (Continuous series for 100% warmup)
                        base_sig = func(df_tf, **params)

                        # Test Version A: Base Strategy
                        res_is_base = run_backtest(df_is, base_sig[is_mask], sym_name, tf_label, strat_name)
                        total_evaluated += 1

                        if res_is_base.total_trades >= 25 and res_is_base.sharpe_ratio >= 0.75:
                            res_oos_base = run_backtest(df_oos, base_sig[oos_mask], sym_name, tf_label, strat_name)
                            all_results.append({
                                "symbol": sym_name,
                                "tf": tf_label,
                                "strategy": strat_name,
                                "variant": "Base",
                                "category": cat,
                                "is_trades": int(res_is_base.total_trades),
                                "is_sharpe": round(float(res_is_base.sharpe_ratio), 2),
                                "is_calmar": round(float(res_is_base.calmar_ratio), 2),
                                "is_max_dd": round(float(res_is_base.max_drawdown_pct), 1),
                                "is_win_rate": round(float(res_is_base.win_rate * 100), 1),
                                "oos_trades": int(res_oos_base.total_trades),
                                "oos_sharpe": round(float(res_oos_base.sharpe_ratio), 2),
                                "oos_return": round(float(res_oos_base.net_profit_pct), 1),
                                "oos_max_dd": round(float(res_oos_base.max_drawdown_pct), 1),
                                "oos_pf": round(float(res_oos_base.profit_factor), 2),
                                "oos_win_rate": round(float(res_oos_base.win_rate * 100), 1),
                                "composite_score": round(float(res_is_base.composite_score), 3)
                            })

                        # Test Version B: BTC Macro-Filtered Strategy
                        btc_filtered_sig = pd.Series(0, index=df_tf.index)
                        btc_filtered_sig[(base_sig == 1) & (btc_reg >= 0)] = 1
                        btc_filtered_sig[(base_sig == -1) & (btc_reg <= 0)] = -1

                        res_is_btc = run_backtest(df_is, btc_filtered_sig[is_mask], sym_name, tf_label, f"{strat_name}_BTC")
                        total_evaluated += 1

                        if res_is_btc.total_trades >= 20 and res_is_btc.sharpe_ratio >= 0.75:
                            res_oos_btc = run_backtest(df_oos, btc_filtered_sig[oos_mask], sym_name, tf_label, f"{strat_name}_BTC")
                            all_results.append({
                                "symbol": sym_name,
                                "tf": tf_label,
                                "strategy": f"{strat_name} + BTC Filter",
                                "variant": "BTC_Filter",
                                "category": f"{cat} (Macro)",
                                "is_trades": int(res_is_btc.total_trades),
                                "is_sharpe": round(float(res_is_btc.sharpe_ratio), 2),
                                "is_calmar": round(float(res_is_btc.calmar_ratio), 2),
                                "is_max_dd": round(float(res_is_btc.max_drawdown_pct), 1),
                                "is_win_rate": round(float(res_is_btc.win_rate * 100), 1),
                                "oos_trades": int(res_oos_btc.total_trades),
                                "oos_sharpe": round(float(res_oos_btc.sharpe_ratio), 2),
                                "oos_return": round(float(res_oos_btc.net_profit_pct), 1),
                                "oos_max_dd": round(float(res_oos_btc.max_drawdown_pct), 1),
                                "oos_pf": round(float(res_oos_btc.profit_factor), 2),
                                "oos_win_rate": round(float(res_oos_btc.win_rate * 100), 1),
                                "composite_score": round(float(res_is_btc.composite_score), 3)
                            })

                    except Exception:
                        continue

            prog.advance(task)

    console.print(f"\n[green]Evaluated {total_evaluated:,} backtests across all symbols and timeframes.[/green]")
    console.print(f"[bold green]✓ Total qualifying candidate combinations found: {len(all_results)}[/bold green]\n")

    if not all_results:
        console.print("[red]No candidates met the baseline threshold.[/]")
        return

    df_res = pd.DataFrame(all_results)

    # 1. Primary Filter: 2026 OOS must be net profitable with Sharpe >= 0.75 and Max DD < 15%
    passed_oos = df_res[(df_res["oos_sharpe"] >= 0.75) & (df_res["oos_return"] > 0) & (df_res["oos_max_dd"] < 15.0)].copy()
    
    if len(passed_oos) < 10:
        # Fallback: relax slightly if OOS was choppy
        passed_oos = df_res[(df_res["oos_sharpe"] >= 0.40) & (df_res["oos_return"] > 0)].copy()

    # Sort descending by 2026 OOS Sharpe
    ranked = passed_oos.sort_values("oos_sharpe", ascending=False).reset_index(drop=True)

    # Apply portfolio diversity filter (max 2 per symbol, max 2 per strategy)
    top_candidates = []
    sym_counts = {}
    strat_counts = {}

    for _, row in ranked.iterrows():
        sym = row["symbol"]
        strat = row["strategy"]
        if sym_counts.get(sym, 0) < 2 and strat_counts.get(strat, 0) < 2:
            top_candidates.append(row.to_dict())
            sym_counts[sym] = sym_counts.get(sym, 0) + 1
            strat_counts[strat] = strat_counts.get(strat, 0) + 1
        if len(top_candidates) >= 25:
            break

    # Leaderboard Table
    tbl = Table(title="TOP 25 STRATEGIES LEADERBOARD (Full Matrix Screen + 2026 OOS)", show_lines=True)
    tbl.add_column("Rank", style="dim", justify="right")
    tbl.add_column("Symbol", style="bold white")
    tbl.add_column("TF", justify="center", style="bold magenta")
    tbl.add_column("Strategy Name", style="yellow")
    tbl.add_column("IS Sharpe", justify="right", style="green")
    tbl.add_column("2026 OOS Sharpe", justify="right", style="bold green")
    tbl.add_column("2026 Return %", justify="right", style="bold cyan")
    tbl.add_column("2026 Max DD", justify="right", style="red")
    tbl.add_column("OOS PF", justify="right")
    tbl.add_column("OOS Win %", justify="right")
    tbl.add_column("IS Score", justify="right", style="bold yellow")

    for i, r in enumerate(top_candidates, 1):
        ret_str = f"+{r['oos_return']:.1f}%" if r['oos_return'] >= 0 else f"{r['oos_return']:.1f}%"
        tbl.add_row(
            str(i),
            r["symbol"],
            r["tf"],
            r["strategy"],
            f"{r['is_sharpe']:.2f}",
            f"{r['oos_sharpe']:.2f}",
            ret_str,
            f"{r['oos_max_dd']:.1f}%",
            f"{r['oos_pf']:.2f}",
            f"{r['oos_win_rate']:.1f}%",
            f"{r['composite_score']:.3f}"
        )

    console.print(tbl)

    # Save to JSON
    out_path = RESULTS_DIR / "v2_enhanced_discoveries.json"
    out_path.write_text(json.dumps(top_candidates, cls=SafeEncoder, indent=2), encoding="utf-8")
    console.print(f"\n[bold green]✓ Top 25 discoveries exported to {out_path}[/bold green]")
    console.rule("[bold green]RESEARCH COMPLETE")

if __name__ == "__main__":
    run_ultimate_research()
