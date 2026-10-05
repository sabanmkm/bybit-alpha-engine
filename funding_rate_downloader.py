"""
FUNDING RATE DOWNLOADER — Fetches historical 8h funding rates from Bybit.
Saves to C:\BybitBacktest\data\funding_rates\ as Parquet files.
"""
import sys, time, json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import ccxt
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn

console = Console()

ROOT_DIR = Path(r"C:\BybitBacktest")
FUNDING_DIR = ROOT_DIR / "data" / "funding_rates"
FUNDING_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "GALA/USDT:USDT", "STORJ/USDT:USDT", "ALGO/USDT:USDT",
    "LUNA2/USDT:USDT", "CHZ/USDT:USDT", "CRO/USDT:USDT",
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
    "DOGE/USDT:USDT", "XRP/USDT:USDT", "ADA/USDT:USDT",
    "AVAX/USDT:USDT", "DOT/USDT:USDT", "LINK/USDT:USDT",
    "MATIC/USDT:USDT", "UNI/USDT:USDT", "ATOM/USDT:USDT",
    "LTC/USDT:USDT", "BCH/USDT:USDT", "FIL/USDT:USDT",
    "APT/USDT:USDT", "ARB/USDT:USDT", "OP/USDT:USDT",
    "NEAR/USDT:USDT", "SAND/USDT:USDT", "MANA/USDT:USDT",
    "AXS/USDT:USDT", "GRT/USDT:USDT", "FET/USDT:USDT",
]

def download_funding_rates():
    console.rule("[bold cyan]DOWNLOADING HISTORICAL FUNDING RATES FROM BYBIT")

    ex = ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    })
    ex.load_markets()

    since = ex.parse8601("2022-01-01T00:00:00Z")
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    downloaded = 0
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=30, complete_style="green"),
        MofNCompleteColumn(),
        console=console
    ) as prog:
        task = prog.add_task("Downloading...", total=len(SYMBOLS))

        for sym in SYMBOLS:
            base = sym.split("/")[0]
            out_file = FUNDING_DIR / f"{base}_funding.parquet"

            if out_file.exists() and out_file.stat().st_size > 1000:
                skipped += 1
                prog.advance(task)
                continue

            try:
                all_rates = []
                current_since = since

                while current_since < now_ms:
                    try:
                        rates = ex.fetch_funding_rate_history(sym, since=current_since, limit=200)
                        if not rates:
                            break
                        all_rates.extend(rates)
                        current_since = rates[-1]["timestamp"] + 1
                        time.sleep(0.15)
                    except Exception:
                        time.sleep(1)
                        break

                if all_rates:
                    df = pd.DataFrame(all_rates)
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    df = df.set_index("timestamp").sort_index()
                    df = df[["fundingRate"]].rename(columns={"fundingRate": "funding_rate"})
                    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
                    df = df.dropna()
                    df.to_parquet(out_file, compression="snappy")
                    downloaded += 1
                    console.print(f"  [green]✓ {base}: {len(df)} funding records[/]")
                else:
                    console.print(f"  [yellow]⊘ {base}: No data[/]")

            except Exception as e:
                console.print(f"  [red]✗ {base}: {e}[/]")

            prog.advance(task)

    console.print(f"\n[bold green]✓ Downloaded: {downloaded} | Skipped (cached): {skipped}[/]")
    console.print(f"[bold green]✓ Funding rates saved to {FUNDING_DIR}[/]")
    console.rule("[bold green]FUNDING DOWNLOAD COMPLETE")

if __name__ == "__main__":
    download_funding_rates()
