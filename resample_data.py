"""
resample_data.py
================
Phase 2 – Optional OHLCV resampler.

Reads raw 5-minute parquet files and produces:
  15m, 30m, 1H, 4H, 1D

Rules
-----
- Open  = first open of the period
- High  = max high
- Low   = min low
- Close = last close
- Volume = sum of volume
- No lookahead bias
- Gaps are left as missing bars (no artificial candles are invented)
- Existing files are skipped unless --force is used

If the target resampled folders already contain data the script can exit early.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from config import (
    RAW_5M_DIR,
    RESAMPLED_DIR,
    TIMEFRAMES,
    TF_TO_MINUTES,
)

console = Console()

# pandas resample rule mapping
RESAMPLE_RULES: Dict[str, str] = {
    "15m": "15min",
    "30m": "30min",
    "1H": "1h",
    "4H": "4h",
    "1D": "1D",
}


def list_raw_symbols() -> List[Path]:
    return sorted(RAW_5M_DIR.glob("*.parquet"))


def already_resampled(symbol_stem: str, tf: str) -> bool:
    target = RESAMPLED_DIR / tf / f"{symbol_stem}.parquet"
    return target.exists() and target.stat().st_size > 1000


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Proper OHLCV aggregation.
    Assumes df is indexed by UTC DatetimeIndex and sorted.
    """
    if df.empty:
        return df

    # Ensure timezone-aware UTC
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")

    ohlcv = df.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # Drop periods that had no data at all (true gaps)
    ohlcv = ohlcv.dropna(subset=["open", "high", "low", "close"])

    # Volume can legitimately be zero; leave it
    ohlcv["volume"] = ohlcv["volume"].fillna(0.0)

    return ohlcv


def process_symbol(raw_path: Path, force: bool = False) -> Dict[str, str]:
    """
    Resample one symbol into all target timeframes.
    Returns a status dict.
    """
    stem = raw_path.stem  # e.g. BTC_USDT_USDT
    result = {"symbol": stem, "status": "ok"}

    try:
        df = pd.read_parquet(raw_path)

        # Basic sanitisation
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            else:
                raise ValueError("No DatetimeIndex and no timestamp column")

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns: {required - set(df.columns)}")

        for tf in TIMEFRAMES:
            if not force and already_resampled(stem, tf):
                result[tf] = "skipped"
                continue

            rule = RESAMPLE_RULES[tf]
            resampled = resample_ohlcv(df, rule)

            out_dir = RESAMPLED_DIR / tf
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}.parquet"
            resampled.to_parquet(out_path, engine="pyarrow", compression="zstd")
            result[tf] = f"{len(resampled):,}"

    except Exception as e:
        result["status"] = f"error: {type(e).__name__}: {e}"
        for tf in TIMEFRAMES:
            result[tf] = "–"

    return result


def main(force: bool = False, skip_if_populated: bool = True) -> None:
    console.rule("[bold cyan]Phase 2 – Resample 5m → higher timeframes (Optional)")

    raw_files = list_raw_symbols()
    if not raw_files:
        console.print("[yellow]No raw 5m parquet files found. Nothing to resample.[/]")
        return

    # Early-exit check
    if skip_if_populated and not force:
        all_populated = True
        for tf in TIMEFRAMES:
            tf_dir = RESAMPLED_DIR / tf
            if not tf_dir.exists() or len(list(tf_dir.glob("*.parquet"))) < 10:
                all_populated = False
                break
        if all_populated:
            console.print(
                "[green]All resampled folders already contain data. "
                "Skipping resampling (use --force to override).[/]"
            )
            return

    console.print(f"Found {len(raw_files)} raw symbols. Force={force}")

    results: List[Dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=6,
    ) as prog:
        task = prog.add_task("Resampling…", total=len(raw_files))

        for raw_path in raw_files:
            prog.update(task, description=f"[cyan]{raw_path.stem}[/]")
            res = process_symbol(raw_path, force=force)
            results.append(res)
            prog.advance(task)

    # Summary table
    table = Table(title="Resample Summary", show_header=True, header_style="bold magenta")
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Status")
    for tf in TIMEFRAMES:
        table.add_column(tf, justify="right")

    ok_count = 0
    for r in results:
        status = r["status"]
        style = "green" if status == "ok" else "red"
        row = [r["symbol"], f"[{style}]{status}[/]"]
        for tf in TIMEFRAMES:
            row.append(str(r.get(tf, "–")))
        table.add_row(*row)
        if status == "ok":
            ok_count += 1

    console.print(table)
    console.print(f"\n[bold green]Finished.[/] Successfully processed: {ok_count}/{len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample Bybit 5m data")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-create all resampled files even if they already exist",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Do not exit early when folders already look populated",
    )
    args = parser.parse_args()
    main(force=args.force, skip_if_populated=not args.no_skip)
