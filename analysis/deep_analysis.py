#!/usr/bin/env python3
r"""
═══════════════════════════════════════════════════════════════════════════════════
  MULTI-TIMEFRAME CRYPTO CANDLESTICK DEEP ANALYSIS ENGINE (v2.3 - Clean Paths)
  - Outputs full color report to Terminal
  - Automatically saves clean ASCII report to C:\BybitBacktest\analysis\reports\
═══════════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import re
import time
import warnings
from datetime import datetime, timedelta
from collections import defaultdict
import traceback

warnings.filterwarnings("ignore")

# ─── Dependency Check ───────────────────────────────────────────────────────────
REQUIRED = ["pandas", "numpy", "pyarrow", "scipy", "tabulate", "colorama"]
missing = []
for pkg in REQUIRED:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print("=" * 70)
    print("  MISSING REQUIRED PACKAGES")
    print(f"  Missing: {', '.join(missing)}")
    print(f"\n  Run this command to install requirements inside your virtual environment:\n")
    print(f"    pip install {' '.join(missing)}")
    print("=" * 70)
    sys.exit(1)

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from tabulate import tabulate
from colorama import init, Fore, Back, Style

init(autoreset=True)

# ─── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR = r"C:\BybitBacktest\data\resampled"
REPORT_DIR = r"C:\BybitBacktest\analysis\reports"
TIMEFRAMES = ["15m", "30m", "1H", "4H", "1D"]
TF_MINUTES = {"15m": 15, "30m": 30, "1H": 60, "4H": 240, "1D": 1440}
PATTERN_LOOKAHEAD = 10
MAX_SYMBOLS_DISPLAY = 30
REPORT_WIDTH = 120

# ─── Dual Logger (Terminal + File Auto-Save) ────────────────────────────────────
class DualOutput:
    """Streams output to console and automatically writes clean text to file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.log_file = open(filepath, "w", encoding="utf-8")
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, message):
        self.terminal.write(message)
        # Strip color escape codes before saving to plain text file
        clean_message = self.ansi_escape.sub('', message)
        self.log_file.write(clean_message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# Set up logging file
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
log_path = os.path.join(REPORT_DIR, f"deep_analysis_{timestamp_str}.txt")
sys.stdout = DualOutput(log_path)

# ─── Color Helpers ──────────────────────────────────────────────────────────────
C_BULL = Fore.GREEN
C_BEAR = Fore.RED
C_NEUT = Fore.YELLOW
C_INFO = Fore.CYAN
C_HEAD = Fore.WHITE + Style.BRIGHT
C_DIM = Fore.WHITE + Style.DIM
C_RESET = Style.RESET_ALL
C_MAG = Fore.MAGENTA

def header_box(title, width=REPORT_WIDTH):
    print()
    print(C_HEAD + "╔" + "═" * (width - 2) + "╗")
    print(C_HEAD + "║" + title.center(width - 2) + "║")
    print(C_HEAD + "╚" + "═" * (width - 2) + "╝" + C_RESET)
    print()

def sub_header(title, width=REPORT_WIDTH):
    print()
    print(C_INFO + "┌" + "─" * (width - 2) + "┐")
    print(C_INFO + "│ " + title.ljust(width - 4) + " │")
    print(C_INFO + "└" + "─" * (width - 2) + "┘" + C_RESET)

def mini_header(title):
    print()
    print(C_MAG + f"  ▸ {title}" + C_RESET)
    print(C_DIM + "  " + "─" * (len(title) + 4) + C_RESET)

def stars(n, max_n=5):
    n = max(0, min(max_n, int(round(n))))
    return C_NEUT + "★" * n + "☆" * (max_n - n) + C_RESET

def safe_series_div(a, b, default=0.0):
    try:
        b_safe = b.replace(0, np.nan)
        result = a / b_safe
        return result.fillna(default)
    except Exception:
        return pd.Series([default] * len(a), index=a.index)

def print_table(headers, rows, fmt="simple"):
    if not rows:
        return
    try:
        print(tabulate(rows, headers=headers, tablefmt=fmt, floatfmt=".4f"))
    except Exception:
        print(tabulate(rows, headers=headers, tablefmt="simple", floatfmt=".4f"))

# ─── Data Loading ───────────────────────────────────────────────────────────────
class DataStore:
    def __init__(self):
        self.data = {}
        self.symbols_per_tf = {}
        self.all_symbols = set()
        self.load_errors = []
        self.health_report = {}

    def discover_and_load(self):
        header_box("DATA LOADING & DISCOVERY")

        for tf in TIMEFRAMES:
            tf_dir = os.path.join(BASE_DIR, tf)
            self.symbols_per_tf[tf] = []

            if not os.path.exists(tf_dir):
                print(C_BEAR + f"  ✗ Directory not found: {tf_dir}" + C_RESET)
                continue

            parquet_files = [f for f in os.listdir(tf_dir) if f.endswith(".parquet")]
            if not parquet_files:
                print(C_NEUT + f"  ⚠ No parquet files in {tf_dir}" + C_RESET)
                continue

            print(C_INFO + f"  Loading {tf} timeframe ({len(parquet_files)} files)..." + C_RESET)

            for fname in sorted(parquet_files):
                symbol = fname.replace(".parquet", "")
                fpath = os.path.join(tf_dir, fname)
                try:
                    df = pd.read_parquet(fpath)

                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df.reset_index()

                    df.columns = [str(c).lower().strip() for c in df.columns]

                    ts_col = None
                    for candidate in ["timestamp", "datetime", "date", "time", "open_time", "candle_time", "ts", "index"]:
                        if candidate in df.columns:
                            ts_col = candidate
                            break

                    if ts_col is None:
                        first_col = df.columns[0]
                        try:
                            pd.to_datetime(df[first_col].iloc[0])
                            ts_col = first_col
                        except Exception:
                            self.load_errors.append((symbol, tf, "No timestamp column"))
                            continue

                    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
                        if pd.api.types.is_numeric_dtype(df[ts_col]):
                            sample = df[ts_col].iloc[0]
                            if sample > 1e12:
                                df[ts_col] = pd.to_datetime(df[ts_col], unit="ms", errors="coerce")
                            elif sample > 1e9:
                                df[ts_col] = pd.to_datetime(df[ts_col], unit="s", errors="coerce")
                            else:
                                df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
                        else:
                            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

                    df = df.rename(columns={ts_col: "datetime"})
                    df = df.dropna(subset=["datetime"])
                    df = df.set_index("datetime")
                    df = df.sort_index()

                    col_map = {}
                    for rc in ["open", "high", "low", "close", "volume"]:
                        if rc in df.columns:
                            col_map[rc] = rc
                        else:
                            for c in df.columns:
                                if rc == c.lower() or rc in c.lower().split("_"):
                                    col_map[rc] = c
                                    break

                    if not all(k in col_map for k in ["open", "high", "low", "close"]):
                        self.load_errors.append((symbol, tf, f"Missing OHLC cols. Found: {list(df.columns)[:5]}"))
                        continue

                    rename_dict = {v: k for k, v in col_map.items() if k != v}
                    if rename_dict:
                        df = df.rename(columns=rename_dict)

                    for col in ["open", "high", "low", "close", "volume"]:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")

                    if "volume" not in df.columns:
                        df["volume"] = 0.0

                    df = df.dropna(subset=["open", "high", "low", "close"])

                    if len(df) < 10:
                        self.load_errors.append((symbol, tf, f"Too few rows: {len(df)}"))
                        continue

                    df = df[["open", "high", "low", "close", "volume"]].copy()

                    df["returns"] = df["close"].pct_change()
                    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
                    df["body"] = df["close"] - df["open"]
                    df["body_abs"] = df["body"].abs()

                    oc_max = df[["open", "close"]].max(axis=1)
                    oc_min = df[["open", "close"]].min(axis=1)
                    df["upper_shadow"] = df["high"] - oc_max
                    df["lower_shadow"] = oc_min - df["low"]
                    df["total_range"] = df["high"] - df["low"]
                    df["body_pct"] = safe_series_div(df["body_abs"], df["total_range"])

                    self.data[(symbol, tf)] = df
                    self.symbols_per_tf[tf].append(symbol)
                    self.all_symbols.add(symbol)

                except Exception as e:
                    self.load_errors.append((symbol, tf, str(e)[:100]))

            loaded = len(self.symbols_per_tf[tf])
            print(C_BULL + f"    ✓ {tf}: Successfully loaded {loaded} symbols" + C_RESET)

        print()
        total_datasets = len(self.data)
        total_symbols = len(self.all_symbols)
        print(C_HEAD + f"  Dataset Inventory: {total_datasets} total series across {total_symbols} unique assets" + C_RESET)

        if self.load_errors:
            print(C_BEAR + f"\n  ⚠ {len(self.load_errors)} loading errors logged (first 5 shown):" + C_RESET)
            for sym, tf, err in self.load_errors[:5]:
                print(C_DIM + f"    - {sym} [{tf}]: {err}" + C_RESET)

        return total_datasets > 0

# ─── Section 1: Health Check ───────────────────────────────────────────────────
def section1_health_check(store):
    header_box("SECTION 1: DATA INVENTORY & HEALTH CHECK")

    for tf in TIMEFRAMES:
        symbols = store.symbols_per_tf.get(tf, [])
        if not symbols:
            continue

        sub_header(f"Timeframe Health: {tf} ({len(symbols)} symbols)")
        tf_rows = []

        for sym in sorted(symbols):
            key = (sym, tf)
            df = store.data[key]

            total_candles = len(df)
            date_start = df.index.min()
            date_end = df.index.max()

            try:
                total_minutes = (date_end - date_start).total_seconds() / 60
                expected = int(total_minutes / TF_MINUTES[tf]) + 1
            except Exception:
                expected = total_candles

            missing_pct = max(0, (1 - total_candles / max(expected, 1)) * 100)

            try:
                diffs = df.index.to_series().diff().dropna()
                expected_diff = pd.Timedelta(minutes=TF_MINUTES[tf])
                gaps = int((diffs > expected_diff * 1.5).sum())
            except Exception:
                gaps = 0

            score = 100.0
            score -= min(30, missing_pct * 0.5)
            score -= min(20, gaps * 0.5)
            if total_candles < 100: score -= 20
            if total_candles < 50: score -= 20

            try:
                zero_vol_pct = float((df["volume"] == 0).mean() * 100)
            except Exception:
                zero_vol_pct = 0.0
            score -= min(10, zero_vol_pct * 0.2)

            score = max(0, min(100, score))

            flags = []
            if total_candles < 50: flags.append("SHORT")
            if missing_pct > 20: flags.append("GAPS")
            if zero_vol_pct > 50: flags.append("NO_VOL")
            if score < 50: flags.append("LOW_Q")
            flag_str = ",".join(flags) if flags else "OK"

            store.health_report[key] = {
                "candles": total_candles, "start": date_start, "end": date_end,
                "missing_pct": missing_pct, "gaps": gaps, "score": score,
                "flags": flag_str, "zero_vol_pct": zero_vol_pct
            }

            start_str = date_start.strftime("%Y-%m-%d") if pd.notna(date_start) else "N/A"
            end_str = date_end.strftime("%Y-%m-%d") if pd.notna(date_end) else "N/A"

            tf_rows.append([sym, total_candles, start_str, end_str,
                            f"{missing_pct:.1f}%", gaps, f"{score:.0f}", flag_str])

        headers = ["Symbol", "Candles", "Start", "End", "Missing%", "Gaps", "Score", "Flags"]
        print_table(headers, tf_rows[:MAX_SYMBOLS_DISPLAY])
        if len(tf_rows) > MAX_SYMBOLS_DISPLAY:
            print(C_DIM + f"    ... [{len(tf_rows) - MAX_SYMBOLS_DISPLAY} additional symbols written to full report file]" + C_RESET)

    mini_header("Global Data Health Overview")
    scores = [v["score"] for v in store.health_report.values()]
    if scores:
        print(f"  Average Quality Score: {np.mean(scores):.1f}/100")
        print(f"  High Quality Datasets (≥80): {C_BULL}{sum(1 for s in scores if s >= 80)}{C_RESET}")
        print(f"  Degraded Datasets (<50):     {C_BEAR}{sum(1 for s in scores if s < 50)}{C_RESET}")

# ─── Section 2: Patterns ────────────────────────────────────────────────────────
def detect_patterns(df):
    patterns = {}
    n = len(df)
    if n < 5:
        return patterns

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    body = c - o
    body_abs = np.abs(body)
    total_range = h - l
    upper_shadow = h - np.maximum(o, c)
    lower_shadow = np.minimum(o, c) - l

    tr_safe = np.where(total_range == 0, 1e-10, total_range)
    body_ratio = body_abs / tr_safe
    upper_ratio = upper_shadow / tr_safe
    lower_ratio = lower_shadow / tr_safe

    avg_body = pd.Series(body_abs).rolling(20, min_periods=5).mean().values
    avg_body_safe = np.where((avg_body == 0) | np.isnan(avg_body), 1e-10, avg_body)

    patterns["Doji"] = (body_ratio < 0.1) & (total_range > 0)
    patterns["Hammer"] = ((lower_shadow >= 2 * body_abs) & (upper_ratio < 0.15) & (body_ratio > 0.05) & (body_ratio < 0.4))
    patterns["Inverted Hammer"] = ((upper_shadow >= 2 * body_abs) & (lower_ratio < 0.15) & (body_ratio > 0.05) & (body_ratio < 0.4))
    patterns["Shooting Star"] = patterns["Inverted Hammer"].copy()
    patterns["Bullish Marubozu"] = (body_ratio > 0.95) & (body > 0)
    patterns["Bearish Marubozu"] = (body_ratio > 0.95) & (body < 0)
    patterns["Spinning Top"] = ((body_ratio < 0.3) & (upper_ratio > 0.2) & (lower_ratio > 0.2) & (total_range > 0))
    patterns["Hanging Man"] = patterns["Hammer"].copy()

    be = np.zeros(n, dtype=bool)
    bee = np.zeros(n, dtype=bool)
    bh = np.zeros(n, dtype=bool)
    bearh = np.zeros(n, dtype=bool)
    pl = np.zeros(n, dtype=bool)
    dcc = np.zeros(n, dtype=bool)
    tt = np.zeros(n, dtype=bool)
    tb = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if body[i-1] < 0 and body[i] > 0 and o[i] <= c[i-1] and c[i] >= o[i-1] and body_abs[i] > body_abs[i-1]: be[i] = True
        if body[i-1] > 0 and body[i] < 0 and o[i] >= c[i-1] and c[i] <= o[i-1] and body_abs[i] > body_abs[i-1]: bee[i] = True
        if body[i-1] < 0 and body[i] > 0 and body_abs[i-1] > body_abs[i] and o[i] > c[i-1] and c[i] < o[i-1]: bh[i] = True
        if body[i-1] > 0 and body[i] < 0 and body_abs[i-1] > body_abs[i] and o[i] < c[i-1] and c[i] > o[i-1]: bearh[i] = True
        mid_prev = (o[i-1] + c[i-1]) / 2
        if body[i-1] < 0 and body[i] > 0 and o[i] < c[i-1] and c[i] > mid_prev and c[i] < o[i-1]: pl[i] = True
        if body[i-1] > 0 and body[i] < 0 and o[i] > c[i-1] and c[i] < mid_prev and c[i] > o[i-1]: dcc[i] = True
        if abs(h[i] - h[i-1]) / tr_safe[i] < 0.05 and body[i-1] > 0 and body[i] < 0: tt[i] = True
        if abs(l[i] - l[i-1]) / tr_safe[i] < 0.05 and body[i-1] < 0 and body[i] > 0: tb[i] = True

    patterns["Bullish Engulfing"] = be
    patterns["Bearish Engulfing"] = bee
    patterns["Bullish Harami"] = bh
    patterns["Bearish Harami"] = bearh
    patterns["Piercing Line"] = pl
    patterns["Dark Cloud Cover"] = dcc
    patterns["Tweezer Tops"] = tt
    patterns["Tweezer Bottoms"] = tb

    ms = np.zeros(n, dtype=bool)
    es = np.zeros(n, dtype=bool)
    tws = np.zeros(n, dtype=bool)
    tbc = np.zeros(n, dtype=bool)
    tiu = np.zeros(n, dtype=bool)
    tid = np.zeros(n, dtype=bool)

    for i in range(2, n):
        if body[i-2] < 0 and body_abs[i-2] > avg_body_safe[i-2] * 0.5 and body_ratio[i-1] < 0.3 and body[i] > 0 and body_abs[i] > avg_body_safe[i] * 0.5 and c[i] > (o[i-2] + c[i-2]) / 2: ms[i] = True
        if body[i-2] > 0 and body_abs[i-2] > avg_body_safe[i-2] * 0.5 and body_ratio[i-1] < 0.3 and body[i] < 0 and body_abs[i] > avg_body_safe[i] * 0.5 and c[i] < (o[i-2] + c[i-2]) / 2: es[i] = True
        if body[i-2] > 0 and body[i-1] > 0 and body[i] > 0 and c[i-1] > c[i-2] and c[i] > c[i-1] and o[i-1] > o[i-2] and o[i] > o[i-1] and upper_ratio[i] < 0.3 and upper_ratio[i-1] < 0.3: tws[i] = True
        if body[i-2] < 0 and body[i-1] < 0 and body[i] < 0 and c[i-1] < c[i-2] and c[i] < c[i-1] and o[i-1] < o[i-2] and o[i] < o[i-1] and lower_ratio[i] < 0.3 and lower_ratio[i-1] < 0.3: tbc[i] = True
        if bh[i-1] and body[i] > 0 and c[i] > c[i-1] and c[i] > o[i-2]: tiu[i] = True
        if bearh[i-1] and body[i] < 0 and c[i] < c[i-1] and c[i] < o[i-2]: tid[i] = True

    patterns["Morning Star"] = ms
    patterns["Evening Star"] = es
    patterns["Three White Soldiers"] = tws
    patterns["Three Black Crows"] = tbc
    patterns["Three Inside Up"] = tiu
    patterns["Three Inside Down"] = tid

    return patterns

def evaluate_pattern_success(df, pattern_mask, is_bullish, lookahead=PATTERN_LOOKAHEAD):
    indices = np.where(pattern_mask)[0]
    if len(indices) == 0:
        return 0, 0.0, 0.0

    close_vals = df["close"].values
    n = len(close_vals)
    successes = 0
    total_return = 0.0
    valid = 0

    for idx in indices:
        if idx + lookahead >= n:
            continue
        entry_price = close_vals[idx]
        if entry_price == 0:
            continue
        future_price = close_vals[idx + 1: idx + lookahead + 1]
        if len(future_price) == 0:
            continue

        if is_bullish:
            max_future = np.max(future_price)
            ret = (max_future - entry_price) / entry_price
            if ret > 0.001:
                successes += 1
        else:
            min_future = np.min(future_price)
            ret = (entry_price - min_future) / entry_price
            if ret > 0.001:
                successes += 1

        end_price = close_vals[min(idx + lookahead, n - 1)]
        actual_ret = (end_price - entry_price) / entry_price
        if not is_bullish:
            actual_ret = -actual_ret
        total_return += actual_ret
        valid += 1

    if valid == 0:
        return len(indices), 0.0, 0.0
    return len(indices), successes / valid * 100, total_return / valid * 100

BULLISH_PATTERNS = {"Doji", "Hammer", "Inverted Hammer", "Bullish Marubozu", "Bullish Engulfing",
                    "Morning Star", "Three White Soldiers", "Bullish Harami", "Piercing Line",
                    "Tweezer Bottoms", "Three Inside Up"}
BEARISH_PATTERNS = {"Shooting Star", "Bearish Marubozu", "Bearish Engulfing", "Evening Star",
                    "Three Black Crows", "Bearish Harami", "Dark Cloud Cover", "Tweezer Tops",
                    "Three Inside Down", "Hanging Man"}

def section2_patterns(store):
    header_box("SECTION 2: CANDLESTICK PATTERN RECOGNITION")
    all_pattern_stats = defaultdict(list)
    sym_pattern_scores = {}
    confluence_data = defaultdict(lambda: defaultdict(list))

    for tf in TIMEFRAMES:
        symbols = store.symbols_per_tf.get(tf, [])
        if not symbols:
            continue

        sub_header(f"Pattern Analysis: {tf}")
        tf_summary_rows = []

        for sym in sorted(symbols):
            key = (sym, tf)
            df = store.data[key]
            if len(df) < 20:
                continue

            try:
                patterns = detect_patterns(df)
            except Exception:
                continue

            sym_total_patterns = 0
            sym_total_success = 0
            sym_pattern_count = 0
            best_pattern = ("None", 0, 0)

            for pname, pmask in patterns.items():
                is_bullish = pname in BULLISH_PATTERNS
                count, success_rate, avg_ret = evaluate_pattern_success(df, pmask, is_bullish)

                if count > 0:
                    all_pattern_stats[pname].append((sym, tf, count, success_rate, avg_ret))
                    sym_total_patterns += count
                    sym_total_success += success_rate * count
                    sym_pattern_count += 1
                    if count >= 3 and success_rate > best_pattern[1]:
                        best_pattern = (pname, success_rate, avg_ret)
                    if count >= 2:
                        confluence_data[sym][pname].append(tf)

            if sym_pattern_count > 0 and sym_total_patterns > 0:
                avg_success = sym_total_success / sym_total_patterns
                reliability = min(100, avg_success * (1 + np.log1p(sym_total_patterns) / 10))
            else:
                avg_success = 0
                reliability = 0

            sym_pattern_scores[key] = reliability
            tf_summary_rows.append([
                sym, sym_total_patterns, sym_pattern_count,
                f"{avg_success:.1f}%" if sym_pattern_count > 0 else "N/A",
                best_pattern[0],
                f"{best_pattern[1]:.1f}%" if best_pattern[0] != "None" else "N/A",
                f"{reliability:.0f}"
            ])

        if tf_summary_rows:
            headers = ["Symbol", "Total Signals", "Patterns Found", "Avg Success", "Best Pattern", "Best %", "Reliability"]
            tf_summary_rows.sort(key=lambda x: float(x[-1]), reverse=True)
            print_table(headers, tf_summary_rows[:MAX_SYMBOLS_DISPLAY])

    mini_header("Global Pattern Reliability Table")
    pattern_rank_rows = []
    for pname, stats_list in sorted(all_pattern_stats.items()):
        total_count = sum(s[2] for s in stats_list)
        if total_count == 0: continue
        weighted_success = sum(s[3] * s[2] for s in stats_list) / total_count
        avg_ret = np.mean([s[4] for s in stats_list if s[2] > 0])
        n_symbols = len(set(s[0] for s in stats_list))
        direction = "BULL" if pname in BULLISH_PATTERNS else ("BEAR" if pname in BEARISH_PATTERNS else "NEUT")
        pattern_rank_rows.append([pname, direction, total_count, n_symbols, f"{weighted_success:.1f}%", f"{avg_ret:+.3f}%"])

    pattern_rank_rows.sort(key=lambda x: float(x[4].replace("%", "")), reverse=True)
    print_table(["Pattern", "Direction", "Occurrences", "Symbols", "Weighted Win%", "Avg Ret"], pattern_rank_rows)

    mini_header("Cross-Timeframe Pattern Confluence")
    confluence_rows = []
    for sym in sorted(confluence_data.keys()):
        for pname, tfs in confluence_data[sym].items():
            if len(tfs) >= 2:
                confluence_rows.append([sym, pname, ", ".join(tfs), len(tfs)])

    if confluence_rows:
        confluence_rows.sort(key=lambda x: x[3], reverse=True)
        print_table(["Symbol", "Pattern", "Timeframes", "TF Count"], confluence_rows[:25])
    else:
        print(C_DIM + "  No multi-timeframe confluence observed." + C_RESET)

    return sym_pattern_scores

# ─── Section 3: Trends ─────────────────────────────────────────────────────────
def calc_adx(df, period=14):
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    if n < period * 2:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

    tr, plus_dm, minus_dm = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        up_move, down_move = high[i] - high[i-1], low[i-1] - low[i]
        if up_move > down_move and up_move > 0: plus_dm[i] = up_move
        if down_move > up_move and down_move > 0: minus_dm[i] = down_move

    atr, sp, sm = np.zeros(n), np.zeros(n), np.zeros(n)
    atr[period] = np.mean(tr[1:period+1])
    sp[period] = np.mean(plus_dm[1:period+1])
    sm[period] = np.mean(minus_dm[1:period+1])

    for i in range(period+1, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        sp[i] = (sp[i-1] * (period-1) + plus_dm[i]) / period
        sm[i] = (sm[i-1] * (period-1) + minus_dm[i]) / period

    plus_di = np.where(atr > 0, 100 * sp / atr, 0)
    minus_di = np.where(atr > 0, 100 * sm / atr, 0)
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)

    adx = np.zeros(n)
    start = period * 2
    if start < n:
        adx[start] = np.mean(dx[period+1:start+1])
        for i in range(start+1, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period

    return adx, plus_di, minus_di

def detect_hh_hl(df, lookback=20):
    if len(df) < lookback * 3: return "Unknown", 0
    high, low = df["high"].values, df["low"].values
    swing_highs, swing_lows = [], []
    half = lookback // 2

    for i in range(half, len(df) - half):
        if high[i] == np.max(high[i-half:i+half+1]): swing_highs.append((i, high[i]))
        if low[i] == np.min(low[i-half:i+half+1]): swing_lows.append((i, low[i]))

    if len(swing_highs) < 2 or len(swing_lows) < 2: return "Unknown", 0

    rh, rl = swing_highs[-4:], swing_lows[-4:]
    hh = sum(1 for i in range(1, len(rh)) if rh[i][1] > rh[i-1][1])
    ll = sum(1 for i in range(1, len(rl)) if rl[i][1] < rl[i-1][1])
    hl = sum(1 for i in range(1, len(rl)) if rl[i][1] > rl[i-1][1])
    lh = sum(1 for i in range(1, len(rh)) if rh[i][1] < rh[i-1][1])

    total = max(1, len(rh) + len(rl) - 2)
    if hh + hl > lh + ll: return "Uptrend", (hh+hl)/total*100
    elif lh + ll > hh + hl: return "Downtrend", (lh+ll)/total*100
    else: return "Sideways", 50

def section3_trends(store):
    header_box("SECTION 3: TREND ANALYSIS")
    trend_results = {}

    for tf in TIMEFRAMES:
        symbols = store.symbols_per_tf.get(tf, [])
        if not symbols: continue

        sub_header(f"Trend Breakdown: {tf}")
        trend_rows = []

        for sym in sorted(symbols):
            key = (sym, tf)
            df = store.data[key]
            if len(df) < 50: continue

            try:
                close = df["close"]
                last_close = close.iloc[-1]

                sma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else np.nan
                sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
                sma100 = close.rolling(100).mean().iloc[-1] if len(close) >= 100 else np.nan
                sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
                ema12 = close.ewm(span=12).mean().iloc[-1]
                ema26 = close.ewm(span=26).mean().iloc[-1]

                ma_bullish, ma_total = 0, 0
                for ma_val in [sma20, sma50, sma100, sma200, ema12, ema26]:
                    if not np.isnan(ma_val):
                        ma_total += 1
                        if last_close > ma_val: ma_bullish += 1

                ma_score = (ma_bullish / ma_total * 100) if ma_total > 0 else 50
                adx_vals, plus_di, minus_di = calc_adx(df)
                adx_cur = float(adx_vals[-1]) if len(adx_vals) > 0 else 0

                adx_strength = "Very Strong" if adx_cur >= 50 else ("Strong" if adx_cur >= 25 else ("Moderate" if adx_cur >= 20 else "Weak"))
                structure, _ = detect_hh_hl(df)

                if ma_score >= 70 and structure == "Uptrend": trend = "Uptrend"
                elif ma_score <= 30 and structure == "Downtrend": trend = "Downtrend"
                elif ma_score >= 60: trend = "Mild Uptrend"
                elif ma_score <= 40: trend = "Mild Downtrend"
                else: trend = "Sideways"

                trend_duration = 0
                if len(close) >= 50:
                    diff = (close.rolling(20).mean() - close.rolling(50).mean()).dropna()
                    if len(diff) > 1:
                        signs = np.sign(diff.values)
                        changes = np.where(signs[:-1] != signs[1:])[0]
                        trend_duration = len(diff) - changes[-1] - 1 if len(changes) > 0 else len(diff)

                reversal_risk = "High" if min([abs(last_close - m)/last_close*100 for m in [sma20, sma50, ema26] if not np.isnan(m)], default=100) < 1.0 else "Normal"

                trend_results[key] = {
                    "trend": trend, "ma_score": ma_score, "adx": adx_cur,
                    "adx_strength": adx_strength, "structure": structure,
                    "duration": trend_duration, "reversal_risk": reversal_risk
                }

                trend_color = C_BULL if "Up" in trend else (C_BEAR if "Down" in trend else C_NEUT)
                trend_rows.append([sym, trend_color + trend + C_RESET, f"{ma_score:.0f}%", f"{adx_cur:.1f}", adx_strength, structure, trend_duration, reversal_risk])
            except Exception:
                continue

        if trend_rows:
            print_table(["Symbol", "Trend", "MA Score", "ADX", "ADX Strength", "Structure", "Duration", "Reversal"], trend_rows[:MAX_SYMBOLS_DISPLAY])

    mini_header("Cross-Timeframe Trend Agreement")
    align_rows = []
    for sym in sorted(store.all_symbols):
        tf_trends = [trend_results[(sym, tf)]["trend"] for tf in TIMEFRAMES if (sym, tf) in trend_results]
        if len(tf_trends) >= 2:
            bull = sum(1 for t in tf_trends if "Up" in t)
            bear = sum(1 for t in tf_trends if "Down" in t)
            total = len(tf_trends)
            align = max(bull, bear) / total * 100
            dir_str = "BULLISH" if bull > bear else ("BEARISH" if bear > bull else "MIXED")
            align_rows.append([sym, dir_str, f"{align:.0f}%", f"{bull} Bull / {bear} Bear / {total - bull - bear} Neut"])

    if align_rows:
        align_rows.sort(key=lambda x: float(x[2].replace("%", "")), reverse=True)
        print_table(["Symbol", "Consensus Direction", "Agreement %", "TF Breakdown"], align_rows[:25])

    return trend_results

# ─── Section 4: Statistics ─────────────────────────────────────────────────────
def section4_statistics(store):
    header_box("SECTION 4: STATISTICAL ANALYSIS & DISTRIBUTIONS")
    stat_results = {}

    for tf in TIMEFRAMES:
        symbols = store.symbols_per_tf.get(tf, [])
        if not symbols: continue

        sub_header(f"Distributions & Volatility: {tf}")
        stat_rows = []

        for sym in sorted(symbols):
            key = (sym, tf)
            df = store.data[key]
            returns = df["returns"].dropna()
            if len(returns) < 20: continue

            try:
                mean_ret = float(returns.mean() * 100)
                std_ret = float(returns.std() * 100)
                skew = float(returns.skew())
                kurt = float(returns.kurtosis())
                hist_vol = float(returns.std() * np.sqrt(252 * (1440 / TF_MINUTES[tf])) * 100)

                tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift(1)).abs(), (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
                atr_14 = float(tr.rolling(14).mean().iloc[-1])
                lc = float(df["close"].iloc[-1])
                atr_pct = (atr_14 / lc * 100) if lc != 0 else 0

                avg_vol = float(df["volume"].mean())
                vol_spikes = int((df["volume"] > avg_vol * 2).sum()) if avg_vol > 0 else 0

                stat_results[key] = {"mean_ret": mean_ret, "std_ret": std_ret, "skew": skew, "kurt": kurt, "hist_vol": hist_vol, "atr_pct": atr_pct, "avg_vol": avg_vol}
                stat_rows.append([sym, f"{mean_ret:+.4f}%", f"{std_ret:.4f}%", f"{skew:+.2f}", f"{kurt:.2f}", f"{hist_vol:.1f}%", f"{atr_pct:.2f}%", f"{avg_vol:,.0f}", vol_spikes])
            except Exception:
                continue

        if stat_rows:
            print_table(["Symbol", "Mean Ret", "Std Dev", "Skew", "Kurt", "Ann.Vol%", "ATR%", "Avg Vol", "Spikes"], stat_rows[:MAX_SYMBOLS_DISPLAY])

    return stat_results

# ─── Section 5: Correlations ───────────────────────────────────────────────────
def section5_correlations(store):
    header_box("SECTION 5: CORRELATIONS & RELATIONSHIPS")
    corr_results = {}

    for tf in ["1H", "4H", "1D"]:
        symbols = store.symbols_per_tf.get(tf, [])
        if len(symbols) < 2: continue

        sub_header(f"Correlation Matrix & Beta ({tf})")
        returns_dict = {sym: store.data[(sym, tf)]["returns"].dropna() for sym in symbols if len(store.data.get((sym, tf), [])) > 30}
        if len(returns_dict) < 2: continue

        returns_df = pd.DataFrame(returns_dict).dropna(how="all")
        valid = [c for c in returns_df.columns if returns_df[c].count() >= 30]
        if len(valid) < 2: continue

        corr_matrix = returns_df[valid].corr(method="pearson")
        pairs = []
        syms_list = list(corr_matrix.columns)
        for i in range(len(syms_list)):
            for j in range(i+1, len(syms_list)):
                c = corr_matrix.iloc[i, j]
                if not np.isnan(c): pairs.append((syms_list[i], syms_list[j], float(c)))

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        mini_header(f"Top 10 Highly Correlated Pairs (>0.8) [{tf}]")
        print_table(["Asset A", "Asset B", "Pearson Corr"], [(a, b, f"{c:.4f}") for a, b, c in pairs if c > 0.8][:10])

        btc_sym = next((s for s in valid if "BTC" in s.upper()), None)
        if btc_sym:
            mini_header(f"Beta Exposure vs {btc_sym} [{tf}]")
            btc_ret = returns_df[btc_sym].dropna()
            beta_rows = []
            for sym in valid:
                if sym == btc_sym: continue
                aligned = pd.concat([btc_ret, returns_df[sym].dropna()], axis=1).dropna()
                if len(aligned) < 20: continue
                beta = aligned.iloc[:, 1].cov(aligned.iloc[:, 0]) / (aligned.iloc[:, 0].var() or 1e-10)
                beta_rows.append([sym, f"{beta:.3f}", "High (Aggressive)" if abs(beta) > 1.3 else ("Moderate" if abs(beta) > 0.7 else "Defensive")])
            beta_rows.sort(key=lambda x: abs(float(x[1])), reverse=True)
            print_table(["Symbol", "Beta vs BTC", "Classification"], beta_rows[:15])

        corr_results[tf] = {"pairs": pairs}
        break

    return corr_results

# ─── Section 6: Cycles & Seasonality ───────────────────────────────────────────
def section6_cycles(store):
    header_box("SECTION 6: CYCLES & SEASONALITY")
    for tf in ["1D", "4H"]:
        symbols = store.symbols_per_tf.get(tf, [])
        if not symbols: continue
        sub_header(f"Day-of-Week Seasonality [{tf}] (Top 10 Assets)")

        dow_rows = []
        for sym in sorted(symbols)[:10]:
            df = store.data[(sym, tf)]
            if len(df) < 30: continue
            dfc = df.copy()
            dfc["dow"] = dfc.index.dayofweek
            mean_dow = dfc.groupby("dow")["returns"].mean() * 100
            best_day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][mean_dow.idxmax()] if len(mean_dow) > 0 else "N/A"
            worst_day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][mean_dow.idxmin()] if len(mean_dow) > 0 else "N/A"
            dow_rows.append([sym, f"{mean_dow.get(0, 0):+.2f}%", f"{mean_dow.get(1, 0):+.2f}%", f"{mean_dow.get(2, 0):+.2f}%", f"{mean_dow.get(3, 0):+.2f}%", f"{mean_dow.get(4, 0):+.2f}%", best_day, worst_day])

        if dow_rows:
            print_table(["Symbol", "Mon", "Tue", "Wed", "Thu", "Fri", "Optimal Day", "Weakest Day"], dow_rows)
        break

# ─── Section 7: Anomalies ──────────────────────────────────────────────────────
def section7_anomalies(store):
    header_box("SECTION 7: ANOMALIES & OUTLIER DETECTION")
    anomaly_results = {}
    extreme_events = []

    for tf in TIMEFRAMES:
        symbols = store.symbols_per_tf.get(tf, [])
        if not symbols: continue
        sub_header(f"Anomaly Audit: {tf}")
        rows = []

        for sym in sorted(symbols):
            key = (sym, tf)
            df = store.data[key]
            returns = df["returns"].dropna()
            if len(returns) < 20: continue

            try:
                rstd = returns.std()
                if rstd == 0: continue
                z = (returns - returns.mean()) / rstd
                pu, pd_ = int((z > 3).sum()), int((z < -3).sum())
                eu, ed = int((z > 5).sum()), int((z < -5).sum())

                avg_vol = float(df["volume"].mean())
                va = int((df["volume"] > avg_vol * 3).sum()) if avg_vol > 0 else 0

                tot_ext = eu + ed
                sev = "EXTREME" if tot_ext > 3 else ("SEVERE" if tot_ext > 0 or (pu+pd_) > 15 else ("MODERATE" if (pu+pd_) > 5 else "CLEAN"))
                anomaly_results[key] = {"severity": sev, "extreme": tot_ext}

                sev_color = C_BEAR if sev in ["EXTREME", "SEVERE"] else (C_NEUT if sev == "MODERATE" else C_BULL)
                rows.append([sym, pu, pd_, va, sev_color + sev + C_RESET])

                if tot_ext > 0:
                    for idx in z[z.abs() > 5].index:
                        extreme_events.append([sym, tf, idx.strftime("%Y-%m-%d %H:%M"), f"{returns.get(idx, 0)*100:+.2f}%", "PUMP" if returns.get(idx, 0) > 0 else "CRASH"])
            except Exception:
                continue

        if rows:
            rows.sort(key=lambda x: x[1] + x[2], reverse=True)
            print_table(["Symbol", "Spikes (Z>3)", "Crashes (Z<-3)", "Vol Anomalies", "Severity"], rows[:MAX_SYMBOLS_DISPLAY])

    if extreme_events:
        mini_header("Flash Crashes & Extreme Pumps Registry (Z > 5)")
        print_table(["Symbol", "TF", "Timestamp", "Return %", "Classification"], extreme_events[:15])

    return anomaly_results

# ─── Section 8: Levels ─────────────────────────────────────────────────────────
def section8_levels(store):
    header_box("SECTION 8: SUPPORT/RESISTANCE & KEY LEVELS")
    level_results = {}

    for tf in ["1D", "4H", "1H"]:
        symbols = store.symbols_per_tf.get(tf, [])
        if not symbols: continue
        sub_header(f"Calculated Pivot & Key Structural Levels ({tf})")

        for sym in sorted(symbols)[:10]:
            key = (sym, tf)
            df = store.data[key]
            if len(df) < 30: continue

            try:
                last_close = float(df["close"].iloc[-1])
                h, l, c = float(df["high"].iloc[-2]), float(df["low"].iloc[-2]), float(df["close"].iloc[-2])
                pivot = (h + l + c) / 3
                r1 = 2 * pivot - l
                s1 = 2 * pivot - h
                r2 = pivot + (h - l)
                s2 = pivot - (h - l)

                level_results[key] = {"pivot": pivot, "r1": r1, "r2": r2, "s1": s1, "s2": s2, "last_close": last_close}
                print(f"  {C_HEAD}{sym:15s} [{tf}]{C_RESET} Last: {last_close:.4f} | Pivot: {pivot:.4f} | R1: {r1:.4f} | S1: {s1:.4f} | R2: {r2:.4f} | S2: {s2:.4f}")
            except Exception:
                continue
        break

    return level_results

# ─── Section 9: Momentum ───────────────────────────────────────────────────────
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def section9_momentum(store):
    header_box("SECTION 9: MOMENTUM & OSCILLATOR ANALYSIS")
    momentum_results = {}

    for tf in TIMEFRAMES:
        symbols = store.symbols_per_tf.get(tf, [])
        if not symbols: continue
        sub_header(f"Momentum Matrix: {tf}")
        rows = []

        for sym in sorted(symbols):
            key = (sym, tf)
            df = store.data[key]
            if len(df) < 50: continue

            try:
                close = df["close"]
                rsi = calc_rsi(close)
                rsi_cur = float(rsi.iloc[-1]) if not rsi.empty else np.nan

                ema12 = close.ewm(span=12).mean()
                ema26 = close.ewm(span=26).mean()
                hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
                hc = float(hist.iloc[-1])

                macd_sig = "Bullish" if hc > 0 else "Bearish"
                score = 0
                if not np.isnan(rsi_cur): score += (rsi_cur - 50) * 1.5
                if hc != 0: score += 25 if hc > 0 else -25
                score = np.clip(score, -100, 100)

                mom_class = "Strong Bull" if score > 50 else ("Mild Bull" if score > 15 else ("Neutral" if score > -15 else ("Mild Bear" if score > -50 else "Strong Bear")))
                momentum_results[key] = {"rsi": rsi_cur, "macd_sig": macd_sig, "score": score, "classification": mom_class}

                rows.append([sym, f"{rsi_cur:.1f}" if not np.isnan(rsi_cur) else "N/A", macd_sig, f"{score:+.1f}", mom_class])
            except Exception:
                continue

        if rows:
            rows.sort(key=lambda x: float(x[3].replace("+", "")), reverse=True)
            print_table(["Symbol", "RSI (14)", "MACD State", "Composite Score", "Regime"], rows[:MAX_SYMBOLS_DISPLAY])

    return momentum_results

# ─── Section 10: Classifications & Tradability ─────────────────────────────────
def section10_classifications(store, stat_results, trend_results, momentum_results, pattern_scores, anomaly_results):
    header_box("SECTION 10: CLASSIFICATIONS & TRADABILITY RANKINGS")
    classifications, tradability_scores = {}, {}

    for sym in sorted(store.all_symbols):
        best_tf = next((tf for tf in ["1H", "4H", "30m", "15m", "1D"] if (sym, tf) in store.data and len(store.data[(sym, tf)]) >= 50), None)
        if best_tf is None: continue

        key = (sym, best_tf)
        s = stat_results.get(key, {})
        t = trend_results.get(key, {})
        m = momentum_results.get(key, {})
        p = pattern_scores.get(key, 0)
        a = anomaly_results.get(key, {})

        hv = s.get("hist_vol", 50)
        vc = "Ultra-Low" if hv < 20 else ("Low" if hv < 40 else ("Medium" if hv < 80 else ("High" if hv < 150 else "Ultra-High")))
        adx = t.get("adx", 0)
        tc = "Strong Trend" if adx >= 25 else "Ranging"
        av = s.get("avg_vol", 0)
        lc = "High Liq" if av > 1e7 else ("Medium" if av > 5e5 else "Low Liq")

        ts = 50.0
        if 30 < hv < 100: ts += 15
        elif hv > 180: ts -= 15
        if adx >= 25: ts += 15
        if av > 1e6: ts += 10
        elif av < 1e4: ts -= 20
        ts += (p - 50) * 0.1
        if a.get("severity") in ["EXTREME", "SEVERE"]: ts -= 20

        ts = max(0, min(100, ts))
        classifications[sym] = {"tf": best_tf, "vol_class": vc, "trend_class": tc, "liq_class": lc, "tradability": ts, "adx": adx, "trend": t.get("trend", "Unknown"), "mom": m.get("score", 0)}
        tradability_scores[sym] = ts

    ranked = sorted(tradability_scores.items(), key=lambda x: x[1], reverse=True)
    rows = [[i, sym, f"{sc:.1f}", classifications[sym]["trend"], f"{classifications[sym]['adx']:.0f}", classifications[sym]["vol_class"], classifications[sym]["liq_class"]] for i, (sym, sc) in enumerate(ranked[:25], 1)]
    print_table(["#", "Symbol", "Tradability", "Primary Trend", "ADX", "Volatility Class", "Liquidity Tier"], rows)

    return classifications, tradability_scores

# ─── Section 11: Cross-Symbol Comparisons ──────────────────────────────────────
def section11_comparisons(store, classifications, tradability_scores, corr_results):
    header_box("SECTION 11: CROSS-SYMBOL & COINTEGRATION DISCOVERY")
    ranked = sorted(tradability_scores.items(), key=lambda x: x[1], reverse=True)
    top_syms = [s for s, _ in ranked[:15]]

    pair_rows = []
    for tf in ["1D", "4H"]:
        symbols = [s for s in top_syms if (s, tf) in store.data]
        if len(symbols) < 2: continue
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                sa, sb = symbols[i], symbols[j]
                aligned = pd.concat([store.data[(sa, tf)]["close"], store.data[(sb, tf)]["close"]], axis=1).dropna()
                if len(aligned) < 60: continue
                try:
                    x, y = aligned.iloc[:, 0].values, aligned.iloc[:, 1].values
                    beta = np.sum((x - x.mean()) * (y - y.mean())) / (np.sum((x - x.mean())**2) or 1e-10)
                    res = y - (y.mean() - beta * x.mean() + beta * x)
                    slope, _, _, _, se = scipy_stats.linregress(res[:-1], np.diff(res))
                    tstat = slope / (se or 1e-10)
                    if tstat < -2.8:
                        hl = -np.log(2) / slope if slope < 0 else np.nan
                        pair_rows.append([sa, sb, tf, f"{tstat:.2f}", "Yes (Mean-Reverting)" if tstat < -3.3 else "Moderate", f"{beta:.4f}", f"{hl:.1f} bars"])
                except Exception:
                    continue
        if pair_rows: break

    if pair_rows:
        print_table(["Asset A", "Asset B", "TF", "ADF Stat", "Cointegration", "Hedge Ratio", "Half Life"], pair_rows[:10])
    else:
        print(C_DIM + "  No statistically significant cointegrated pairs identified." + C_RESET)

# ─── Section 12: Signals ───────────────────────────────────────────────────────
def section12_signals(store, classifications, tradability_scores, trend_results, momentum_results, stat_results, level_results, pattern_scores, anomaly_results):
    header_box("SECTION 12: CONFLUENCE STRATEGY SIGNALS")
    signals = []

    for sym, c in classifications.items():
        best_tf = c["tf"]
        key = (sym, best_tf)
        df = store.data[key]
        lc = float(df["close"].iloc[-1])

        t = trend_results.get(key, {})
        m = momentum_results.get(key, {})
        s = stat_results.get(key, {})
        lv = level_results.get(key, {})

        bull_f, bear_f = [], []
        if "Up" in t.get("trend", ""): bull_f.append(f"Trend: {t.get('trend')} [S3]")
        if "Down" in t.get("trend", ""): bear_f.append(f"Trend: {t.get('trend')} [S3]")
        if t.get("adx", 0) >= 25: (bull_f if "Up" in t.get("trend", "") else bear_f).append(f"Strong ADX: {t.get('adx'):.0f} [S3]")
        if m.get("score", 0) > 20: bull_f.append(f"Bullish Momentum: {m.get('score'):+.0f} [S9]")
        if m.get("score", 0) < -20: bear_f.append(f"Bearish Momentum: {m.get('score'):+.0f} [S9]")

        direction = "LONG" if len(bull_f) > len(bear_f) else ("SHORT" if len(bear_f) > len(bull_f) else None)
        if not direction: continue

        conf = min(5, max(len(bull_f), len(bear_f)))
        atr_val = lc * s.get("atr_pct", 2.0) / 100

        sl = (lc - 2 * atr_val) if direction == "LONG" else (lc + 2 * atr_val)
        tp1 = (lc + 2 * atr_val) if direction == "LONG" else (lc - 2 * atr_val)
        tp2 = (lc + 3.5 * atr_val) if direction == "LONG" else (lc - 3.5 * atr_val)

        signals.append({
            "symbol": sym, "direction": direction, "confidence": conf, "tf": best_tf,
            "entry": lc, "sl": sl, "tp1": tp1, "tp2": tp2, "tradability": c["tradability"],
            "factors": bull_f if direction == "LONG" else bear_f
        })

    signals.sort(key=lambda x: x["confidence"] * x["tradability"], reverse=True)

    sub_header("Top High-Conviction Setups")
    for sig in signals[:10]:
        dc = C_BULL if sig["direction"] == "LONG" else C_BEAR
        print(f"  {dc}▶ {sig['symbol']} — {sig['direction']} | {stars(sig['confidence'])} | TF: {sig['tf']} | Tradability: {sig['tradability']:.0f}/100{C_RESET}")
        print(f"    Entry: {sig['entry']:.5f} | SL: {sig['sl']:.5f} | TP1: {sig['tp1']:.5f} | TP2: {sig['tp2']:.5f}")
        print(f"    Evidence: {', '.join(sig['factors'])}")
        print()

    return signals

# ─── Section 13: Summary ───────────────────────────────────────────────────────
def section13_summary(store, signals, classifications):
    header_box("SECTION 13: EXECUTIVE SUMMARY & BLUEPRINT")

    longs = sum(1 for s in signals if s["direction"] == "LONG")
    shorts = sum(1 for s in signals if s["direction"] == "SHORT")
    regime = "BULLISH EXPANSION" if longs > shorts * 1.5 else ("BEARISH CONTRACTION" if shorts > longs * 1.5 else "ROTATIONAL / RANGE")

    print(f"  Macro Market Posture: {C_HEAD}{regime}{C_RESET} ({longs} Long Setups / {shorts} Short Setups)")
    print(f"  Active Universe: {len(store.all_symbols)} Symbols Analyzed")
    print(f"\n  {C_HEAD}Portfolio Allocation Model:{C_RESET}")
    for sig in signals[:5]:
        dc = C_BULL if sig["direction"] == "LONG" else C_BEAR
        print(f"    • {sig['symbol']:15s} [{dc}{sig['direction']}{C_RESET}] Weight: 20% | SL: {sig['sl']:.4f} | Target: {sig['tp1']:.4f}")

    print(f"\n  {C_HEAD}Execution Directives:{C_RESET}")
    print("    1. Risk max 1-2% account balance per position.")
    print("    2. Move Stop-Loss to Breakeven upon TP1 reach.")
    print("    3. Avoid adding more than 3 simultaneous positions with Pearson Correlation > 0.8.")

    print(f"\n  {C_BULL}✓ Full report automatically written to disk:{C_RESET}")
    print(f"    {C_INFO}{os.path.abspath(log_path)}{C_RESET}")

# ─── Main Routine ──────────────────────────────────────────────────────────────
def main():
    start = time.time()

    print()
    print(C_HEAD + "═" * REPORT_WIDTH)
    print("║" + " MULTI-TIMEFRAME CRYPTO CANDLESTICK ENGINE ".center(REPORT_WIDTH-2) + "║")
    print("║" + f" Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(REPORT_WIDTH-2) + "║")
    print("═" * REPORT_WIDTH + C_RESET)

    store = DataStore()
    if not store.discover_and_load():
        print(C_BEAR + "Data loading failed." + C_RESET)
        sys.exit(1)

    section1_health_check(store)
    p_scores = section2_patterns(store)
    t_res = section3_trends(store)
    s_res = section4_statistics(store)
    c_res = section5_correlations(store)
    section6_cycles(store)
    a_res = section7_anomalies(store)
    lv_res = section8_levels(store)
    m_res = section9_momentum(store)
    cls, trad = section10_classifications(store, s_res, t_res, m_res, p_scores, a_res)
    section11_comparisons(store, cls, trad, c_res)
    sigs = section12_signals(store, cls, trad, t_res, m_res, s_res, lv_res, p_scores, a_res)
    section13_summary(store, sigs, cls)

    elapsed = time.time() - start
    print()
    print(C_HEAD + "═" * REPORT_WIDTH)
    print("║" + f" Analysis Completed in {elapsed:.1f}s ".center(REPORT_WIDTH-2) + "║")
    print("═" * REPORT_WIDTH + C_RESET)
    print()

if __name__ == "__main__":
    main()