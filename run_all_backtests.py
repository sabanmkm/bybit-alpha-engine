"""
run_all_backtests.py
====================
Improved version - uses ThreadPoolExecutor (Windows friendly).
"""

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from config import (
    IS_END, MIN_CANDIDATES_AFTER_SCREEN, RESAMPLED_DIR, RESULTS_DIR,
    SHARPE_SCREEN_THRESHOLD, TIMEFRAMES, BT, assert_no_2026
)
from backtest_engine import run_backtest, composite_score
from strategy_generator import get_all_strategy_instances, get_strategy

console = Console()

BACKTEST_DIR = RESULTS_DIR / "backtests"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SL = 0.02
DEFAULT_TP = 0.04
DEFAULT_TRAIL = 0.015
DEFAULT_TIME_STOP = None


def load_is_data(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    path = RESAMPLED_DIR / timeframe / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            return None
        df = df[df.index <= IS_END].copy()
        if len(df) < 300:
            return None
        assert_no_2026(df)
        return df
    except Exception:
        return None


def list_symbols(timeframe: str = "1H") -> List[str]:
    p = RESAMPLED_DIR / timeframe
    if not p.exists():
        return []
    return sorted([f.stem for f in p.glob("*.parquet")])


def _run_single(task: tuple) -> Optional[dict]:
    symbol, timeframe, strategy_name, params = task
    try:
        df = load_is_data(symbol, timeframe)
        if df is None:
            return None

        strat = get_strategy(strategy_name, **params)
        signals = strat.generate_signals(df)

        result = run_backtest(
            df=df,
            signals=signals,
            symbol=symbol,
            timeframe=timeframe,
            strategy_name=strategy_name,
            sl_pct=DEFAULT_SL,
            tp_pct=DEFAULT_TP,
            trailing_pct=DEFAULT_TRAIL,
            time_stop=DEFAULT_TIME_STOP,
            enforce_date_gate=True,
        )

        m = result.metrics
        if m["n_trades"] < BT.min_trades:
            return None

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy_name,
            "params": params,
            "n_trades": int(m["n_trades"]),
            "sharpe": round(m["sharpe"], 4),
            "sortino": round(m.get("sortino", 0.0), 4),
            "calmar": round(m["calmar"], 4),
            "profit_factor": round(m["profit_factor"], 4),
            "max_dd": round(m["max_dd"], 4),
            "win_rate": round(m["win_rate"], 4),
            "ulcer_index": round(m.get("ulcer_index", 0.0), 4),
            "total_return": round(m["total_return"], 4),
            "composite": round(m.get("composite", 0.0), 4),
        }
    except Exception as e:
        return {"symbol": symbol, "timeframe": timeframe, "strategy": strategy_name, "error": str(e)}


def run_screening(
    timeframes: Optional[List[str]] = None,
    max_symbols: Optional[int] = None,
    n_jobs: int = 4,
    save_prefix: str = "screen_v2",
) -> pd.DataFrame:
    timeframes = timeframes or TIMEFRAMES
    symbols = list_symbols("1H")
    if max_symbols:
        symbols = symbols[:max_symbols]

    strategies = get_all_strategy_instances(default_params=True)
    console.print(f"[cyan]Strategies : {len(strategies)}[/]")
    console.print(f"[cyan]Symbols    : {len(symbols)}[/]")
    console.print(f"[cyan]Timeframes : {timeframes}[/]")

    tasks = [(sym, tf, strat.meta.name, strat.params)
             for strat in strategies
             for tf in timeframes
             for sym in symbols]

    console.print(f"[cyan]Total combinations : {len(tasks):,}[/]")

    results: List[dict] = []
    errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as prog:
        task_id = prog.add_task("Screening…", total=len(tasks))

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            future_to_task = {executor.submit(_run_single, t): t for t in tasks}
            for future in as_completed(future_to_task):
                res = future.result()
                if res is None:
                    pass
                elif "error" in res:
                    errors += 1
                    console.print(f"[red]Error on {res['strategy']} {res['symbol']} {res['timeframe']}: {res['error'][:80]}[/]")
                else:
                    results.append(res)
                prog.advance(task_id)

    if not results:
        console.print("[red]No valid results.[/]")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)

    out_path = BACKTEST_DIR / f"{save_prefix}_full.parquet"
    df.to_parquet(out_path, index=False)

    top_path = BACKTEST_DIR / f"{save_prefix}_top_candidates.parquet"
    top = df[df["sharpe"] >= SHARPE_SCREEN_THRESHOLD].head(50)
    top.to_parquet(top_path, index=False)

    # === Rich Table ===
    table = Table(title="Top 15 by Composite Score (New Scoring)", header_style="bold magenta")
    table.add_column("#", justify="right")
    table.add_column("Strategy", style="cyan", width=24)
    table.add_column("Symbol", width=18)
    table.add_column("TF")
    table.add_column("Trades", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("PF", justify="right")
    table.add_column("MaxDD", justify="right")
    table.add_column("Composite", justify="right")

    for i, row in df.head(15).iterrows():
        table.add_row(
            str(i+1),
            str(row["strategy"])[:22],
            str(row["symbol"])[:16],
            str(row["timeframe"]),
            str(int(row["n_trades"])),
            f"{row['sharpe']:.2f}",
            f"{row['profit_factor']:.2f}",
            f"{row['max_dd']:.1%}",
            f"{row['composite']:.3f}",
        )
    console.print(table)

    n_good = len(df[df["sharpe"] >= SHARPE_SCREEN_THRESHOLD])
    console.print(f"\n[bold]Candidates with Sharpe ≥ {SHARPE_SCREEN_THRESHOLD}: {n_good}  (target ≥ {MIN_CANDIDATES_AFTER_SCREEN})[/]")
    console.print(f"Errors encountered: {errors}")
    console.print(f"[green]Results saved to results/backtests/{save_prefix}_*.parquet[/]")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", nargs="+", default=None, help="Timeframes (e.g. 1H 4H)")
    parser.add_argument("--max-symbols", type=int, default=50, help="Limit symbols for faster testing")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--prefix", default="screen_v2")
    args = parser.parse_args()

    run_screening(
        timeframes=args.tf,
        max_symbols=args.max_symbols,
        n_jobs=args.jobs,
        save_prefix=args.prefix,
    )
