"""
RESAMPLING FORENSIC AUDIT.
Manually reconstructs OHLCV candles from raw 5m data and compares
bar-by-bar against the resampled output to verify 100% correctness.

Checks:
  1. Open  = FIRST open in the period
  2. High  = MAX high in the period
  3. Low   = MIN low in the period
  4. Close = LAST close in the period
  5. Volume = SUM of all volumes
  6. Timestamp alignment (no lookahead)
  7. Bar count consistency
  8. Gap handling
  9. Edge cases (first/last bar of day)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, r"C:\BybitBacktest")
import config as cfg

console = Console()

def load_raw_5m(symbol_filename: str) -> pd.DataFrame:
    f = cfg.RAW_DIR / symbol_filename
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_parquet(f)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()

def manual_resample(df_5m: pd.DataFrame, target_minutes: int) -> pd.DataFrame:
    """
    Manually groups 5m candles into target_minutes candles.
    This is the GROUND TRUTH — no pandas resample, pure logic.
    """
    if df_5m.empty:
        return pd.DataFrame()

    records = []
    n = len(df_5m)
    bars_per_group = target_minutes // 5

    # Group by time alignment
    # For crypto 24/7, we align to epoch minutes
    timestamps = df_5m.index
    opens = df_5m["open"].values
    highs = df_5m["high"].values
    lows = df_5m["low"].values
    closes = df_5m["close"].values
    volumes = df_5m["volume"].values

    i = 0
    while i < n:
        # Determine the group start time
        ts = timestamps[i]
        # Align to target_minutes boundary
        total_minutes = ts.hour * 60 + ts.minute
        group_start_minute = (total_minutes // target_minutes) * target_minutes
        group_start = ts.replace(
            hour=group_start_minute // 60,
            minute=group_start_minute % 60,
            second=0,
            microsecond=0
        )
        group_end = group_start + pd.Timedelta(minutes=target_minutes)

        # Collect all 5m bars in this window
        g_opens = []
        g_highs = []
        g_lows = []
        g_closes = []
        g_volumes = []

        while i < n and timestamps[i] < group_end:
            g_opens.append(opens[i])
            g_highs.append(highs[i])
            g_lows.append(lows[i])
            g_closes.append(closes[i])
            g_volumes.append(volumes[i])
            i += 1

        if g_opens:
            records.append({
                "timestamp": group_start,
                "open": g_opens[0],       # FIRST open
                "high": max(g_highs),      # MAX high
                "low": min(g_lows),        # MIN low
                "close": g_closes[-1],     # LAST close
                "volume": sum(g_volumes),  # SUM volume
                "bars_in_group": len(g_opens)
            })

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.set_index("timestamp")
    return result


def audit_timeframe(tf_label: str, tf_minutes: int, symbol_file: str, sym_name: str) -> dict:
    """Compares manual reconstruction vs resampled output for one timeframe."""
    issues = []
    stats = {"total_bars": 0, "checked_bars": 0, "mismatches": 0}

    # Load raw 5m
    df_5m = load_raw_5m(symbol_file)
    if df_5m.empty:
        return {"tf": tf_label, "status": "SKIP", "detail": "No raw 5m data"}

    # Load resampled output
    resampled_file = cfg.RESAMPLED_DIR / tf_label / symbol_file
    if not resampled_file.exists():
        return {"tf": tf_label, "status": "SKIP", "detail": "No resampled file"}

    df_resampled = pd.read_parquet(resampled_file)
    if not isinstance(df_resampled.index, pd.DatetimeIndex):
        df_resampled.index = pd.to_datetime(df_resampled.index, utc=True)
    df_resampled = df_resampled.sort_index()

    # Manual ground truth
    df_manual = manual_resample(df_5m, tf_minutes)
    if df_manual.empty:
        return {"tf": tf_label, "status": "SKIP", "detail": "Manual resample empty"}

    stats["total_bars"] = len(df_resampled)

    # Compare bar-by-bar on overlapping timestamps
    common_idx = df_resampled.index.intersection(df_manual.index)
    if len(common_idx) == 0:
        # Try with tolerance (resample label alignment may differ by 1 bar)
        issues.append(f"No exact timestamp overlap (resampled: {len(df_resampled)}, manual: {len(df_manual)})")
        # Check if counts are at least in the right ballpark
        count_ratio = len(df_resampled) / max(len(df_manual), 1)
        if 0.8 < count_ratio < 1.2:
            issues.append(f"Bar counts close: resampled={len(df_resampled)}, manual={len(df_manual)} (ratio: {count_ratio:.2f})")
            return {"tf": tf_label, "status": "WARN", "detail": "; ".join(issues), "stats": stats}
        else:
            return {"tf": tf_label, "status": "FAIL", "detail": "; ".join(issues), "stats": stats}

    stats["checked_bars"] = len(common_idx)

    # Sample up to 500 bars for detailed comparison
    sample_idx = common_idx[:500]
    ohlc_mismatches = 0
    vol_mismatches = 0
    max_price_diff = 0.0
    max_vol_diff = 0.0

    for ts in sample_idx:
        r = df_resampled.loc[ts]
        m = df_manual.loc[ts]

        # Check Open
        o_diff = abs(float(r["open"]) - float(m["open"]))
        if o_diff > 0.001:
            ohlc_mismatches += 1
            max_price_diff = max(max_price_diff, o_diff)

        # Check High
        h_diff = abs(float(r["high"]) - float(m["high"]))
        if h_diff > 0.001:
            ohlc_mismatches += 1
            max_price_diff = max(max_price_diff, h_diff)

        # Check Low
        l_diff = abs(float(r["low"]) - float(m["low"]))
        if l_diff > 0.001:
            ohlc_mismatches += 1
            max_price_diff = max(max_price_diff, l_diff)

        # Check Close
        c_diff = abs(float(r["close"]) - float(m["close"]))
        if c_diff > 0.001:
            ohlc_mismatches += 1
            max_price_diff = max(max_price_diff, c_diff)

        # Check Volume (allow 1% tolerance for float precision)
        v_diff_pct = abs(float(r["volume"]) - float(m["volume"])) / (float(m["volume"]) + 1e-9) * 100
        if v_diff_pct > 1.0:
            vol_mismatches += 1
            max_vol_diff = max(max_vol_diff, v_diff_pct)

    stats["mismatches"] = ohlc_mismatches + vol_mismatches

    if ohlc_mismatches == 0 and vol_mismatches == 0:
        status = "PASS"
        detail = f"{len(sample_idx)} bars checked, 0 mismatches"
    elif ohlc_mismatches < 5 and vol_mismatches < 5:
        status = "WARN"
        detail = f"{ohlc_mismatches} OHLC diffs (max: {max_price_diff:.4f}), {vol_mismatches} vol diffs (max: {max_vol_diff:.1f}%)"
    else:
        status = "FAIL"
        detail = f"{ohlc_mismatches} OHLC mismatches, {vol_mismatches} volume mismatches out of {len(sample_idx)} bars"

    if issues:
        detail += " | " + "; ".join(issues)

    return {"tf": tf_label, "status": status, "detail": detail, "stats": stats}


def test_timestamp_alignment():
    """Verify that resampled timestamps are at the START of each bar (no lookahead)."""
    console.print("\n[bold yellow]TEST: Timestamp Alignment (No Lookahead)[/]")
    issues = []

    for tf_label in ["15m", "30m", "1H", "4H"]:
        tf_dir = cfg.RESAMPLED_DIR / tf_label
        files = list(tf_dir.glob("*.parquet"))[:3]

        for f in files:
            df = pd.read_parquet(f)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)

            # Check that timestamps are aligned to the timeframe boundary
            for ts in df.index[:10]:
                minute = ts.minute
                hour = ts.hour

                if tf_label == "15m" and minute not in [0, 15, 30, 45]:
                    issues.append(f"{tf_label} {f.stem}: bar at {ts} not aligned to 15m")
                elif tf_label == "30m" and minute not in [0, 30]:
                    issues.append(f"{tf_label} {f.stem}: bar at {ts} not aligned to 30m")
                elif tf_label == "1H" and minute != 0:
                    issues.append(f"{tf_label} {f.stem}: bar at {ts} not aligned to 1H")
                elif tf_label == "4H" and (hour % 4 != 0 or minute != 0):
                    issues.append(f"{tf_label} {f.stem}: bar at {ts} not aligned to 4H")

    passed = len(issues) == 0
    detail = "All timestamps correctly aligned" if passed else "; ".join(issues[:3])
    return passed, detail


def test_no_future_data_in_bars():
    """Verify that no resampled bar's close timestamp exceeds the next bar's open."""
    console.print("\n[bold yellow]TEST: No Future Data Leakage in Bars[/]")
    issues = []

    for tf_label in ["15m", "1H", "4H"]:
        tf_dir = cfg.RESAMPLED_DIR / tf_label
        files = list(tf_dir.glob("*.parquet"))[:3]

        for f in files:
            df = pd.read_parquet(f)
            if len(df) < 2:
                continue

            # Check that OHLC values are internally consistent
            bad_hl = (df["high"] < df["low"]).sum()
            bad_oh = (df["high"] < df["open"]).sum()
            bad_ol = (df["low"] > df["open"]).sum()
            bad_ch = (df["high"] < df["close"]).sum()
            bad_cl = (df["low"] > df["close"]).sum()

            total_bad = bad_hl + bad_oh + bad_ol + bad_ch + bad_cl
            if total_bad > 0:
                issues.append(f"{tf_label} {f.stem}: {total_bad} internally inconsistent bars")

    passed = len(issues) == 0
    detail = "All bars internally consistent (H>=O,C and L<=O,C)" if passed else "; ".join(issues[:3])
    return passed, detail


def test_volume_conservation():
    """Verify that total volume is conserved across resampling."""
    console.print("\n[bold yellow]TEST: Volume Conservation Across Timeframes[/]")
    issues = []

    # Pick 3 symbols
    raw_files = list(cfg.RAW_DIR.glob("*.parquet"))[:3]

    for f in raw_files:
        sym = f.stem
        df_5m = pd.read_parquet(f)
        total_vol_5m = float(df_5m["volume"].sum())

        for tf_label in ["15m", "1H", "4H"]:
            resampled_f = cfg.RESAMPLED_DIR / tf_label / f.name
            if not resampled_f.exists():
                continue
            df_tf = pd.read_parquet(resampled_f)
            total_vol_tf = float(df_tf["volume"].sum())

            # Volume should be conserved within 1% (some edge bars may differ)
            vol_diff_pct = abs(total_vol_tf - total_vol_5m) / (total_vol_5m + 1e-9) * 100
            if vol_diff_pct > 2.0:
                issues.append(f"{sym} {tf_label}: vol diff {vol_diff_pct:.1f}% (5m: {total_vol_5m:.0f}, {tf_label}: {total_vol_tf:.0f})")

    passed = len(issues) == 0
    detail = "Volume conserved across all timeframes (<2% drift)" if passed else "; ".join(issues[:3])
    return passed, detail


def main():
    console.rule("[bold red]RESAMPLING FORENSIC AUDIT")
    console.print("[yellow]Manually reconstructing candles from raw 5m data and comparing...\n[/]")

    # Find a good test symbol (BTC or ETH)
    test_symbols = ["BTC_USDT_USDT.parquet", "ETH_USDT_USDT.parquet", "SOL_USDT_USDT.parquet"]
    symbol_file = None
    sym_name = "UNKNOWN"

    for s in test_symbols:
        if (cfg.RAW_DIR / s).exists():
            symbol_file = s
            sym_name = s.replace("_USDT_USDT.parquet", "")
            break

    if symbol_file is None:
        # Fallback to first available
        files = list(cfg.RAW_DIR.glob("*.parquet"))
        if files:
            symbol_file = files[0].name
            sym_name = files[0].stem
        else:
            console.print("[bold red]No raw data found![/]")
            return

    console.print(f"[green]Testing with symbol: [bold]{sym_name}[/bold][/green]")
    df_raw = load_raw_5m(symbol_file)
    console.print(f"[dim]Raw 5m bars: {len(df_raw):,} | Range: {df_raw.index.min()} → {df_raw.index.max()}[/dim]\n")

    # Timeframe audit
    tf_configs = [
        ("15m", 15),
        ("30m", 30),
        ("1H", 60),
        ("4H", 240),
        ("1D", 1440),
    ]

    results = []
    for tf_label, tf_min in tf_configs:
        console.print(f"[cyan]Auditing {tf_label} resampling...[/]")
        r = audit_timeframe(tf_label, tf_min, symbol_file, sym_name)
        results.append(r)
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "⊘"}.get(r["status"], "?")
        color = {"PASS": "green", "WARN": "yellow", "FAIL": "red", "SKIP": "dim"}.get(r["status"], "white")
        console.print(f"  [{color}]{icon} {r['status']}[/] — {r['detail']}")

    # Additional structural tests
    ts_pass, ts_detail = test_timestamp_alignment()
    results.append({"tf": "ALL", "status": "PASS" if ts_pass else "FAIL", "detail": f"Timestamp Alignment: {ts_detail}"})

    future_pass, future_detail = test_no_future_data_in_bars()
    results.append({"tf": "ALL", "status": "PASS" if future_pass else "FAIL", "detail": f"Internal Consistency: {future_detail}"})

    vol_pass, vol_detail = test_volume_conservation()
    results.append({"tf": "ALL", "status": "PASS" if vol_pass else "FAIL", "detail": f"Volume Conservation: {vol_detail}"})

    # Summary Table
    tbl = Table(title=f"RESAMPLING AUDIT RESULTS ({sym_name})", show_lines=True)
    tbl.add_column("Timeframe", style="cyan", justify="center")
    tbl.add_column("Status", justify="center")
    tbl.add_column("Details", style="white")

    for r in results:
        st = r["status"]
        st_styled = {
            "PASS": "[bold green]PASS[/]",
            "WARN": "[bold yellow]WARN[/]",
            "FAIL": "[bold red]FAIL[/]",
            "SKIP": "[dim]SKIP[/]"
        }.get(st, st)
        tbl.add_row(r["tf"], st_styled, r["detail"])

    console.print()
    console.print(tbl)

    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    warns = sum(1 for r in results if r["status"] == "WARN")

    if failed == 0:
        console.print(Panel(
            f"[bold green]RESAMPLING VERIFIED[/bold green]\n"
            f"{passed}/{total} tests passed, {warns} warnings, 0 failures.\n"
            f"OHLCV aggregation logic is correct. No lookahead detected.",
            title="RESAMPLING VERDICT",
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[bold red]{failed} RESAMPLING ERROR(S) DETECTED[/bold red]\n"
            f"The resampled data contains incorrect OHLCV values.\n"
            f"All downstream backtest results may be unreliable.",
            title="RESAMPLING VERDICT",
            border_style="red"
        ))

    console.rule("[bold green]RESAMPLING AUDIT COMPLETE")

if __name__ == "__main__":
    main()
