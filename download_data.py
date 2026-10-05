"""
download_data.py
================
Phase 1 – Optional raw 5-minute OHLCV downloader for Bybit USDT Perpetuals.

Features
--------
- Fully resumable via JSON progress log
- Rate-limit aware with exponential backoff (up to 5 retries)
- Skips symbols that already have ≥ MIN_CANDLES_RAW candles
- Only keeps symbols whose first candle is in 2022 or earlier
- Rich live progress bar + final summary table
- Zero API key required (public endpoints via ccxt)

If the raw_5m folder already contains sufficient data the script exits early.
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ccxt
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
    BACKOFF_FACTOR,
    BASE_SLEEP,
    DATA_END,
    DATA_START,
    EXCHANGE_ID,
    LOGS_DIR,
    MAX_RETRIES,
    MIN_CANDLES_RAW,
    QUOTE,
    RAW_5M_DIR,
    ROOT,
)

console = Console()
PROGRESS_LOG = LOGS_DIR / "download_progress.json"
RAW_5M_DIR.mkdir(parents=True, exist_ok=True)


def load_progress() -> Dict[str, dict]:
    if PROGRESS_LOG.exists():
        try:
            with open(PROGRESS_LOG, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_progress(progress: Dict[str, dict]) -> None:
    with open(PROGRESS_LOG, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, default=str)


def create_exchange() -> ccxt.Exchange:
    exchange = getattr(ccxt, EXCHANGE_ID)(
        {
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},  # USDT perpetuals
        }
    )
    return exchange


def fetch_markets(exchange: ccxt.Exchange) -> List[str]:
    """Return list of active USDT linear perpetual symbols."""
    markets = exchange.load_markets()
    symbols = []
    for symbol, m in markets.items():
        if (
            m.get("quote") == QUOTE
            and m.get("swap") is True
            and m.get("linear") is True
            and m.get("active") is True
        ):
            symbols.append(symbol)
    return sorted(symbols)


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def fetch_ohlcv_resumable(
    exchange: ccxt.Exchange,
    symbol: str,
    since_ms: int,
    end_ms: int,
    timeframe: str = "5m",
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch OHLCV with automatic pagination, retries and exponential backoff.
    Returns a clean DataFrame indexed by UTC timestamp.
    """
    all_candles: List[list] = []
    current = since_ms
    retries = 0

    while current < end_ms:
        try:
            candles = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=current, limit=limit
            )
            if not candles:
                break

            # Filter out candles beyond end_ms
            candles = [c for c in candles if c[0] <= end_ms]
            if not candles:
                break

            all_candles.extend(candles)
            last_ts = candles[-1][0]
            current = last_ts + 1  # next candle

            # Safety: if exchange returns same candle repeatedly
            if len(candles) < 2:
                break

            time.sleep(BASE_SLEEP)
            retries = 0  # reset on success

        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
            retries += 1
            if retries > MAX_RETRIES:
                raise RuntimeError(
                    f"{symbol}: max retries exceeded – {type(e).__name__}: {e}"
                ) from e
            sleep_for = BASE_SLEEP * (BACKOFF_FACTOR ** retries)
            console.log(
                f"[yellow]{symbol} retry {retries}/{MAX_RETRIES} "
                f"after {type(e).__name__} – sleeping {sleep_for:.1f}s[/]"
            )
            time.sleep(sleep_for)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]
    return df


def already_sufficient(symbol: str) -> Tuple[bool, int]:
    """Check if a parquet already exists and has enough rows."""
    path = RAW_5M_DIR / f"{symbol.replace('/', '_')}.parquet"
    if not path.exists():
        return False, 0
    try:
        df = pd.read_parquet(path)
        n = len(df)
        return n >= MIN_CANDLES_RAW, n
    except Exception:
        return False, 0


def save_symbol(symbol: str, df: pd.DataFrame) -> None:
    path = RAW_5M_DIR / f"{symbol.replace('/', '_')}.parquet"
    df.to_parquet(path, engine="pyarrow", compression="zstd")


def main() -> None:
    console.rule("[bold cyan]Phase 1 – Bybit 5m Data Downloader (Optional)")

    # Quick exit if we already have a healthy set of files
    existing = list(RAW_5M_DIR.glob("*.parquet"))
    if len(existing) >= 30:
        console.print(
            f"[green]Found {len(existing)} existing raw files. "
            "Assuming data is sufficient – skipping download.[/]"
        )
        console.print(
            "[dim]Delete files in data\\raw_5m or the progress log if you want a full re-download.[/]"
        )
        return

    exchange = create_exchange()
    console.print(f"Loading markets from {EXCHANGE_ID}…")
    symbols = fetch_markets(exchange)
    console.print(f"Found {len(symbols)} active USDT linear perpetuals.")

    progress = load_progress()
    since_ms = ms(DATA_START)
    end_ms = ms(DATA_END)

    results: List[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    ) as prog:
        task = prog.add_task("Downloading…", total=len(symbols))

        for symbol in symbols:
            prog.update(task, description=f"[cyan]{symbol}[/]")

            # Resume / skip logic
            ok, n_exist = already_sufficient(symbol)
            if ok:
                results.append(
                    {
                        "symbol": symbol,
                        "status": "skipped (sufficient)",
                        "candles": n_exist,
                        "first": None,
                        "last": None,
                    }
                )
                progress[symbol] = {"status": "done", "candles": n_exist}
                save_progress(progress)
                prog.advance(task)
                continue

            try:
                df = fetch_ohlcv_resumable(exchange, symbol, since_ms, end_ms)

                if df.empty:
                    results.append(
                        {
                            "symbol": symbol,
                            "status": "empty",
                            "candles": 0,
                            "first": None,
                            "last": None,
                        }
                    )
                    progress[symbol] = {"status": "empty"}
                    save_progress(progress)
                    prog.advance(task)
                    continue

                first_ts = df.index[0]
                last_ts = df.index[-1]
                n = len(df)

                # Only keep symbols whose history starts in 2022 or earlier
                if first_ts.year > 2022:
                    results.append(
                        {
                            "symbol": symbol,
                            "status": "rejected (starts after 2022)",
                            "candles": n,
                            "first": str(first_ts.date()),
                            "last": str(last_ts.date()),
                        }
                    )
                    progress[symbol] = {"status": "rejected_late_start"}
                    save_progress(progress)
                    prog.advance(task)
                    continue

                if n < MIN_CANDLES_RAW:
                    results.append(
                        {
                            "symbol": symbol,
                            "status": f"too short ({n})",
                            "candles": n,
                            "first": str(first_ts.date()),
                            "last": str(last_ts.date()),
                        }
                    )
                    progress[symbol] = {"status": "too_short", "candles": n}
                    save_progress(progress)
                    prog.advance(task)
                    continue

                # Success
                save_symbol(symbol, df)
                results.append(
                    {
                        "symbol": symbol,
                        "status": "downloaded",
                        "candles": n,
                        "first": str(first_ts.date()),
                        "last": str(last_ts.date()),
                    }
                )
                progress[symbol] = {
                    "status": "done",
                    "candles": n,
                    "first": str(first_ts),
                    "last": str(last_ts),
                }
                save_progress(progress)

            except Exception as e:
                console.log(f"[red]{symbol} failed: {e}[/]")
                results.append(
                    {
                        "symbol": symbol,
                        "status": f"error: {type(e).__name__}",
                        "candles": 0,
                        "first": None,
                        "last": None,
                    }
                )
                progress[symbol] = {"status": "error", "error": str(e)}
                save_progress(progress)

            prog.advance(task)

    # Final rich summary table
    table = Table(title="Download Summary", show_header=True, header_style="bold magenta")
    table.add_column("Symbol", style="cyan")
    table.add_column("Status")
    table.add_column("Candles", justify="right")
    table.add_column("First")
    table.add_column("Last")

    downloaded = 0
    for r in sorted(results, key=lambda x: x["symbol"]):
        status = r["status"]
        style = "green" if "downloaded" in status or "skipped" in status else "yellow"
        if "error" in status or "rejected" in status:
            style = "red"
        table.add_row(
            r["symbol"],
            f"[{style}]{status}[/]",
            f"{r['candles']:,}" if r["candles"] else "–",
            r["first"] or "–",
            r["last"] or "–",
        )
        if "downloaded" in status or "skipped" in status:
            downloaded += 1

    console.print(table)
    console.print(
        f"\n[bold green]Finished.[/] Usable symbols: {downloaded}  |  "
        f"Progress log: {PROGRESS_LOG}"
    )


if __name__ == "__main__":
    main()
