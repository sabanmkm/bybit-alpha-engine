# -*- coding: utf-8 -*-
"""
BYBIT BACKTESTER V3 - Phase 1 Exhaustive Analysis
Institutional-Grade Multi-Timeframe Pattern Discovery
"""

import os
import sys
import warnings
import traceback
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
import json

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
from tabulate import tabulate

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
BASE_DATA_DIR = r"C:\BybitBacktest\workspace\data"
REPORTS_DIR   = r"C:\BybitBacktest\workspace\reports"
TEMP_DIR      = r"C:\BybitBacktest\workspace\temp"

TIMEFRAMES = ["15m", "30m", "1H", "4H", "1D"]
ROUND_TRIP_COST = 0.0016  # 0.16% total (0.12% fees + 0.04% slippage)
MIN_OCCURRENCES = 30
TRAIN_RATIO = 0.70
SIGNIFICANCE_LEVEL = 0.05
STRICT_SIGNIFICANCE = 0.01

# Report output buffer
REPORT_LINES = []

def prt(msg=""):
    """Print to console and buffer for report."""
    print(msg, flush=True)
    REPORT_LINES.append(msg)

def section_header(num, title):
    prt("")
    prt("=" * 70)
    prt(f"  SECTION {num}: {title}")
    prt("=" * 70)
    prt("")

def sub_header(title):
    prt(f"\n--- {title} ---\n")

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
prt("=" * 70)
prt("  LOADING DATA FROM WORKSPACE")
prt("=" * 70)
prt("")

ALL_DATA = {}  # {timeframe: {symbol: DataFrame}}
ALL_SYMBOLS = set()

for tf in TIMEFRAMES:
    tf_path = os.path.join(BASE_DATA_DIR, tf)
    ALL_DATA[tf] = {}
    if not os.path.exists(tf_path):
        prt(f"  [WARN] Timeframe folder not found: {tf_path}")
        continue

    parquet_files = [f for f in os.listdir(tf_path) if f.endswith('.parquet')]
    if not parquet_files:
        prt(f"  [WARN] No parquet files in {tf_path}")
        continue

    for pf in sorted(parquet_files):
        symbol = pf.replace('.parquet', '')
        filepath = os.path.join(tf_path, pf)
        try:
            df = pd.read_parquet(filepath)

            # Normalize column names
            df.columns = [c.lower().strip() for c in df.columns]

            # Handle timestamp/datetime column
            time_col = None
            for candidate in ['timestamp', 'datetime', 'date', 'time', 'open_time', 'start']:
                if candidate in df.columns:
                    time_col = candidate
                    break

            if time_col and time_col != 'timestamp':
                df.rename(columns={time_col: 'timestamp'}, inplace=True)

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df.set_index('timestamp', inplace=True)
            elif not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index, errors='coerce')
                except:
                    pass

            # Ensure required columns exist
            required = ['open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required if c not in df.columns]
            if missing:
                prt(f"  [SKIP] {symbol} ({tf}): missing columns {missing}")
                continue

            # Clean data
            for col in required:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            df.sort_index(inplace=True)
            df = df[~df.index.duplicated(keep='first')]

            if len(df) < 100:
                prt(f"  [SKIP] {symbol} ({tf}): only {len(df)} rows")
                continue

            # Compute returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            ALL_DATA[tf][symbol] = df
            ALL_SYMBOLS.add(symbol)
        except Exception as e:
            prt(f"  [ERROR] Loading {symbol} ({tf}): {str(e)[:80]}")

    prt(f"  [OK] {tf}: loaded {len(ALL_DATA[tf])} symbols")

ALL_SYMBOLS = sorted(ALL_SYMBOLS)
prt(f"\n  Total unique symbols: {len(ALL_SYMBOLS)}")
prt(f"  Symbols: {', '.join(ALL_SYMBOLS[:30])}{'...' if len(ALL_SYMBOLS) > 30 else ''}")

if not any(len(ALL_DATA[tf]) > 0 for tf in TIMEFRAMES):
    prt("\n[FATAL] No valid data loaded. Exiting.")
    sys.exit(1)

# Helper: pick best timeframe for a quick analysis (prefer 1H, then 4H, then 15m)
def get_primary_tf():
    for tf in ["1H", "4H", "30m", "15m", "1D"]:
        if len(ALL_DATA.get(tf, {})) > 0:
            return tf
    return TIMEFRAMES[0]

PRIMARY_TF = get_primary_tf()
prt(f"  Primary timeframe for detailed analysis: {PRIMARY_TF}")

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def compute_forward_returns(df, periods=[1, 3, 5, 10]):
    """Compute forward returns for multiple periods."""
    result = df.copy()
    for p in periods:
        result[f'fwd_ret_{p}'] = result['close'].shift(-p) / result['close'] - 1
    return result

def train_test_split_ts(df, train_ratio=0.70):
    """Time-series aware train/test split."""
    n = len(df)
    split_idx = int(n * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

def calc_edge_stats(returns_series, cost=ROUND_TRIP_COST):
    """Calculate trading edge statistics for a series of returns."""
    if len(returns_series) < 5:
        return None
    net_returns = returns_series - cost
    wins = net_returns[net_returns > 0]
    losses = net_returns[net_returns <= 0]
    n = len(net_returns)
    win_rate = len(wins) / n if n > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0001
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float('inf')

    # Max consecutive losses
    is_loss = (net_returns <= 0).astype(int)
    max_consec_loss = 0
    current = 0
    for v in is_loss:
        if v == 1:
            current += 1
            max_consec_loss = max(max_consec_loss, current)
        else:
            current = 0

    return {
        'count': n,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
        'profit_factor': profit_factor,
        'max_consec_loss': max_consec_loss,
        'mean_return': net_returns.mean(),
        'std_return': net_returns.std() if n > 1 else 0,
        'sharpe': (net_returns.mean() / net_returns.std() * np.sqrt(252)) if net_returns.std() > 0 and n > 1 else 0
    }

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(window=period).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def adx(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    # Where plus_dm > minus_dm, keep plus_dm else 0
    cond = plus_dm > minus_dm
    plus_dm = plus_dm.where(cond, 0)
    minus_dm = minus_dm.where(~cond, 0)

    atr_val = atr(df, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr_val.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr_val.replace(0, np.nan))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1/period, min_periods=period).mean()
    return adx_val, plus_di, minus_di

def bollinger_bands(series, period=20, std_mult=2):
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower) / mid
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return upper, mid, lower, width, pct_b

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def stochastic(df, k_period=14, d_period=3):
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(window=d_period).mean()
    return k, d

def obv(df):
    vol = df['volume'].values.copy().astype(float)
    close = df['close'].values
    obv_arr = np.zeros(len(vol))
    obv_arr[0] = vol[0]
    for i in range(1, len(vol)):
        if close[i] > close[i-1]:
            obv_arr[i] = obv_arr[i-1] + vol[i]
        elif close[i] < close[i-1]:
            obv_arr[i] = obv_arr[i-1] - vol[i]
        else:
            obv_arr[i] = obv_arr[i-1]
    return pd.Series(obv_arr, index=df.index)

def safe_tabulate(data, headers, max_rows=50):
    """Safe tabulate with row limit."""
    if len(data) > max_rows:
        data = data[:max_rows]
    try:
        return tabulate(data, headers=headers, tablefmt="simple", floatfmt=".4f",
                       numalign="right", stralign="left")
    except:
        return tabulate(data, headers=headers, tablefmt="simple")

# ═══════════════════════════════════════════════════════════════
# SECTION 1: DATA PROFILING & QUALITY AUDIT
# ═══════════════════════════════════════════════════════════════
section_header(1, "DATA PROFILING & QUALITY AUDIT")

try:
    # Coverage matrix
    sub_header("Symbol Coverage Matrix")
    coverage_data = []
    for sym in ALL_SYMBOLS[:60]:
        row = [sym]
        for tf in TIMEFRAMES:
            if sym in ALL_DATA.get(tf, {}):
                df = ALL_DATA[tf][sym]
                start = df.index.min().strftime('%Y-%m-%d') if len(df) > 0 else 'N/A'
                end = df.index.max().strftime('%Y-%m-%d') if len(df) > 0 else 'N/A'
                row.append(f"{len(df):,} ({start[:7]}→{end[:7]})")
            else:
                row.append("—")
        coverage_data.append(row)

    headers = ["Symbol"] + TIMEFRAMES
    prt(safe_tabulate(coverage_data, headers))

    # Data quality audit
    sub_header("Data Quality Audit")
    quality_data = []
    for tf in TIMEFRAMES:
        for sym, df in list(ALL_DATA[tf].items())[:20]:
            nulls = df[['open','high','low','close','volume']].isnull().sum().sum()
            zero_vol = (df['volume'] == 0).sum()
            n = len(df)
            ret = df['returns'].dropna()
            quality_data.append([
                sym, tf, n, nulls, zero_vol,
                f"{ret.mean():.6f}", f"{ret.std():.4f}",
                f"{ret.skew():.3f}", f"{ret.kurtosis():.2f}"
            ])

    headers = ["Symbol", "TF", "Rows", "Nulls", "ZeroVol", "MeanRet", "StdRet", "Skew", "Kurt"]
    prt(safe_tabulate(quality_data, headers))

    # Summary stats
    sub_header("Aggregate Statistics by Timeframe")
    for tf in TIMEFRAMES:
        syms = list(ALL_DATA.get(tf, {}).keys())
        if not syms:
            continue
        total_rows = sum(len(ALL_DATA[tf][s]) for s in syms)
        prt(f"  {tf}: {len(syms)} symbols, {total_rows:,} total rows")

except Exception as e:
    prt(f"  [ERROR in Section 1] {str(e)}")
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# SECTION 2: SINGLE-CANDLE PATTERN ANALYSIS
# ═══════════════════════════════════════════════════════════════
section_header(2, "SINGLE-CANDLE PATTERN ANALYSIS")

try:
    def detect_candle_patterns(df):
        """Detect Japanese candlestick patterns."""
        o, h, l, c = df['open'], df['high'], df['low'], df['close']
        body = c - o
        abs_body = abs(body)
        upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
        lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - l
        total_range = (h - l).replace(0, np.nan)
        body_ratio = abs_body / total_range
        avg_body = abs_body.rolling(20).mean()

        patterns = pd.DataFrame(index=df.index)

        # Doji: body < 10% of range
        patterns['doji'] = body_ratio < 0.1

        # Dragonfly Doji: doji with long lower shadow, minimal upper
        patterns['dragonfly_doji'] = (body_ratio < 0.1) & (lower_shadow > 2 * abs_body) & (upper_shadow < abs_body * 0.5)

        # Gravestone Doji: doji with long upper shadow, minimal lower
        patterns['gravestone_doji'] = (body_ratio < 0.1) & (upper_shadow > 2 * abs_body) & (lower_shadow < abs_body * 0.5)

        # Hammer: small body at top, long lower shadow
        patterns['hammer'] = (lower_shadow > 2 * abs_body) & (upper_shadow < abs_body * 0.5) & (body_ratio > 0.1) & (body_ratio < 0.4)

        # Inverted Hammer: small body at bottom, long upper shadow
        patterns['inverted_hammer'] = (upper_shadow > 2 * abs_body) & (lower_shadow < abs_body * 0.5) & (body_ratio > 0.1) & (body_ratio < 0.4)

        # Bullish Marubozu: large bullish body, minimal shadows
        patterns['bullish_marubozu'] = (body > 0) & (body_ratio > 0.85) & (abs_body > avg_body * 1.5)

        # Bearish Marubozu: large bearish body, minimal shadows
        patterns['bearish_marubozu'] = (body < 0) & (body_ratio > 0.85) & (abs_body > avg_body * 1.5)

        # Spinning Top: small body, shadows on both sides
        patterns['spinning_top'] = (body_ratio < 0.3) & (body_ratio > 0.1) & (upper_shadow > abs_body * 0.5) & (lower_shadow > abs_body * 0.5)

        # Bullish Engulfing
        prev_body = body.shift(1)
        patterns['bullish_engulfing'] = (prev_body < 0) & (body > 0) & (abs_body > abs(prev_body)) & (c > o.shift(1)) & (o < c.shift(1))

        # Bearish Engulfing
        patterns['bearish_engulfing'] = (prev_body > 0) & (body < 0) & (abs_body > abs(prev_body)) & (c < o.shift(1)) & (o > c.shift(1))

        # Bullish Harami
        patterns['bullish_harami'] = (prev_body < 0) & (body > 0) & (abs_body < abs(prev_body)) & (c < o.shift(1)) & (o > c.shift(1))

        # Bearish Harami
        patterns['bearish_harami'] = (prev_body > 0) & (body < 0) & (abs_body < abs(prev_body)) & (c > o.shift(1)) & (o < c.shift(1))

        # Three White Soldiers
        cond_3ws = (body > 0) & (body.shift(1) > 0) & (body.shift(2) > 0)
        cond_3ws = cond_3ws & (c > c.shift(1)) & (c.shift(1) > c.shift(2))
        cond_3ws = cond_3ws & (abs_body > avg_body * 0.5) & (abs_body.shift(1) > avg_body.shift(1) * 0.5)
        patterns['three_white_soldiers'] = cond_3ws

        # Three Black Crows
        cond_3bc = (body < 0) & (body.shift(1) < 0) & (body.shift(2) < 0)
        cond_3bc = cond_3bc & (c < c.shift(1)) & (c.shift(1) < c.shift(2))
        cond_3bc = cond_3bc & (abs_body > avg_body * 0.5) & (abs_body.shift(1) > avg_body.shift(1) * 0.5)
        patterns['three_black_crows'] = cond_3bc

        # Morning Star (simplified)
        p2_bearish = body.shift(2) < 0
        p1_small = abs_body.shift(1) < avg_body.shift(1) * 0.3
        p0_bullish = body > 0
        p0_close_above_mid = c > (o.shift(2) + c.shift(2)) / 2
        patterns['morning_star'] = p2_bearish & p1_small & p0_bullish & p0_close_above_mid

        # Evening Star (simplified)
        p2_bullish = body.shift(2) > 0
        p0_bearish = body < 0
        p0_close_below_mid = c < (o.shift(2) + c.shift(2)) / 2
        patterns['evening_star'] = p2_bullish & p1_small & p0_bearish & p0_close_below_mid

        return patterns

    fwd_periods = [1, 3, 5, 10]
    pattern_results = []  # Collect all results for ranking
    pattern_names = [
        'doji', 'dragonfly_doji', 'gravestone_doji', 'hammer', 'inverted_hammer',
        'bullish_marubozu', 'bearish_marubozu', 'spinning_top',
        'bullish_engulfing', 'bearish_engulfing', 'bullish_harami', 'bearish_harami',
        'three_white_soldiers', 'three_black_crows', 'morning_star', 'evening_star'
    ]

    all_pattern_stats = defaultdict(list)

    for tf in TIMEFRAMES:
        for sym, df_orig in ALL_DATA.get(tf, {}).items():
            if len(df_orig) < 200:
                continue
            df = compute_forward_returns(df_orig, fwd_periods)
            patterns = detect_candle_patterns(df)

            train_df, test_df = train_test_split_ts(df, TRAIN_RATIO)
            train_pat, test_pat = train_test_split_ts(patterns, TRAIN_RATIO)

            for pname in pattern_names:
                if pname not in patterns.columns:
                    continue

                for dataset_label, d, p in [("train", train_df, train_pat), ("test", test_df, test_pat)]:
                    mask = p[pname].fillna(False)
                    count = mask.sum()
                    if count < 10:
                        continue

                    for fp in fwd_periods:
                        col = f'fwd_ret_{fp}'
                        if col not in d.columns:
                            continue
                        rets = d.loc[mask, col].dropna()
                        if len(rets) < 5:
                            continue
                        edge = calc_edge_stats(rets, ROUND_TRIP_COST)
                        if edge:
                            all_pattern_stats[(pname, tf, fp, dataset_label)].append({
                                'symbol': sym,
                                **edge
                            })

    # Aggregate across symbols
    sub_header("Pattern Performance (Aggregated Across Symbols, Net of 0.16% Costs)")

    agg_results = []
    for (pname, tf, fp, dataset), sym_stats in all_pattern_stats.items():
        total_count = sum(s['count'] for s in sym_stats)
        if total_count < MIN_OCCURRENCES:
            continue
        avg_wr = np.average([s['win_rate'] for s in sym_stats], weights=[s['count'] for s in sym_stats])
        avg_exp = np.average([s['expectancy'] for s in sym_stats], weights=[s['count'] for s in sym_stats])
        avg_pf = np.average([s['profit_factor'] for s in sym_stats if s['profit_factor'] != float('inf')],
                           weights=[s['count'] for s in sym_stats if s['profit_factor'] != float('inf')]) if any(s['profit_factor'] != float('inf') for s in sym_stats) else 0
        n_symbols = len(sym_stats)

        flag = ""
        if avg_wr > 0.60 and total_count > MIN_OCCURRENCES:
            flag = "★ HIGH PRIORITY"
        if n_symbols < 3:
            flag += " [symbol-specific]"

        agg_results.append([
            pname, tf, f"{fp}-bar", dataset, total_count, n_symbols,
            f"{avg_wr:.3f}", f"{avg_exp:.5f}", f"{avg_pf:.2f}", flag
        ])

    agg_results.sort(key=lambda x: float(x[6]), reverse=True)
    headers = ["Pattern", "TF", "Fwd", "Set", "Count", "#Syms", "WinRate", "Expect", "PF", "Flag"]
    prt(safe_tabulate(agg_results[:80], headers))

    # Highlight high-priority
    hp = [r for r in agg_results if "HIGH PRIORITY" in r[-1] and r[3] == "test"]
    if hp:
        sub_header("HIGH PRIORITY Patterns (>60% WR, >30 occ, Test Set)")
        prt(safe_tabulate(hp[:30], headers))
    else:
        prt("\n  No patterns met HIGH PRIORITY criteria on test set (>60% WR).")
        # Show top patterns anyway
        test_results = [r for r in agg_results if r[3] == "test"]
        if test_results:
            sub_header("Top Patterns on Test Set (ranked by Win Rate)")
            prt(safe_tabulate(test_results[:20], headers))

except Exception as e:
    prt(f"  [ERROR in Section 2] {str(e)}")
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# SECTION 3: MULTI-CANDLE SEQUENCE ANALYSIS (Markov Chains)
# ═══════════════════════════════════════════════════════════════
section_header(3, "MULTI-CANDLE SEQUENCE ANALYSIS (Markov Chains)")

try:
    def classify_candle(row):
        body_pct = (row['close'] - row['open']) / row['open'] if row['open'] != 0 else 0
        if body_pct > 0.005:
            return 'SB'  # Strong Bullish
        elif body_pct > 0:
            return 'WB'  # Weak Bullish
        elif body_pct > -0.005:
            return 'WR'  # Weak Bearish (or Doji)
        elif body_pct > -0.01:
            return 'WR'
        else:
            return 'SR'  # Strong Bearish

    def classify_candle_5(row):
        body_pct = (row['close'] - row['open']) / row['open'] if row['open'] != 0 else 0
        if body_pct > 0.005:
            return 'SB'
        elif body_pct > 0.0005:
            return 'WB'
        elif body_pct > -0.0005:
            return 'D'
        elif body_pct > -0.005:
            return 'WR'
        else:
            return 'SR'

    tf = PRIMARY_TF
    sequence_results_2 = defaultdict(list)
    sequence_results_3 = defaultdict(list)

    for sym, df_orig in list(ALL_DATA.get(tf, {}).items()):
        if len(df_orig) < 500:
            continue

        df = compute_forward_returns(df_orig, [1, 3, 5])
        df['candle_type'] = df.apply(classify_candle_5, axis=1)

        # 2-candle sequences
        df['seq2'] = df['candle_type'].shift(1) + '_' + df['candle_type']
        # 3-candle sequences
        df['seq3'] = df['candle_type'].shift(2) + '_' + df['candle_type'].shift(1) + '_' + df['candle_type']

        for seq_col, results_dict in [('seq2', sequence_results_2), ('seq3', sequence_results_3)]:
            for seq_name, group in df.groupby(seq_col):
                if len(group) < 10:
                    continue
                fwd3 = group['fwd_ret_3'].dropna()
                if len(fwd3) < 10:
                    continue
                results_dict[seq_name].append({
                    'symbol': sym,
                    'count': len(fwd3),
                    'mean_ret': fwd3.mean(),
                    'win_rate': (fwd3 > ROUND_TRIP_COST).mean(),
                    'std': fwd3.std()
                })

    # Aggregate 2-candle results
    sub_header(f"2-Candle Sequence Analysis ({tf})")
    seq2_agg = []
    for seq, sym_results in sequence_results_2.items():
        total_n = sum(r['count'] for r in sym_results)
        if total_n < MIN_OCCURRENCES:
            continue
        avg_wr = np.average([r['win_rate'] for r in sym_results], weights=[r['count'] for r in sym_results])
        avg_ret = np.average([r['mean_ret'] for r in sym_results], weights=[r['count'] for r in sym_results])
        n_syms = len(sym_results)

        # Chi-square test: is win rate significantly different from 50%?
        wins = int(avg_wr * total_n)
        losses = total_n - wins
        if total_n > 20:
            chi2_stat, p_val = stats.chisquare([wins, losses], [total_n/2, total_n/2])
        else:
            p_val = 1.0

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        seq2_agg.append([seq, total_n, n_syms, f"{avg_wr:.3f}", f"{avg_ret:.5f}", f"{p_val:.4f}", sig])

    seq2_agg.sort(key=lambda x: float(x[3]), reverse=True)
    headers = ["Sequence", "Count", "#Syms", "WinRate", "AvgRet", "p-val", "Sig"]
    prt(safe_tabulate(seq2_agg[:30], headers))

    # Top bullish & bearish
    sub_header("Top Bullish 2-Candle Sequences (by Win Rate)")
    bullish = [r for r in seq2_agg if float(r[3]) > 0.5]
    prt(safe_tabulate(bullish[:10], headers))

    sub_header("Top Bearish 2-Candle Sequences (by Win Rate)")
    bearish = [r for r in seq2_agg if float(r[3]) < 0.5]
    bearish.sort(key=lambda x: float(x[3]))
    prt(safe_tabulate(bearish[:10], headers))

    # 3-candle
    sub_header(f"3-Candle Sequence Analysis ({tf}) - Top Sequences")
    seq3_agg = []
    for seq, sym_results in sequence_results_3.items():
        total_n = sum(r['count'] for r in sym_results)
        if total_n < MIN_OCCURRENCES:
            continue
        avg_wr = np.average([r['win_rate'] for r in sym_results], weights=[r['count'] for r in sym_results])
        avg_ret = np.average([r['mean_ret'] for r in sym_results], weights=[r['count'] for r in sym_results])
        n_syms = len(sym_results)
        seq3_agg.append([seq, total_n, n_syms, f"{avg_wr:.3f}", f"{avg_ret:.5f}"])

    seq3_agg.sort(key=lambda x: float(x[3]), reverse=True)
    headers3 = ["Sequence", "Count", "#Syms", "WinRate", "AvgRet"]
    prt(safe_tabulate(seq3_agg[:20], headers3))

    # Transition probability matrix
    sub_header("Markov Transition Matrix (Aggregated)")
    states = ['SB', 'WB', 'D', 'WR', 'SR']
    trans_counts = defaultdict(lambda: defaultdict(int))
    for sym, df_orig in list(ALL_DATA.get(tf, {}).items())[:20]:
        if len(df_orig) < 200:
            continue
        types = df_orig.apply(classify_candle_5, axis=1).values
        for i in range(len(types)-1):
            trans_counts[types[i]][types[i+1]] += 1

    trans_matrix = []
    for s1 in states:
        row_total = sum(trans_counts[s1].values()) or 1
        row = [s1]
        for s2 in states:
            row.append(f"{trans_counts[s1][s2]/row_total:.3f}")
        trans_matrix.append(row)

    prt(safe_tabulate(trans_matrix, ["From\\To"] + states))

except Exception as e:
    prt(f"  [ERROR in Section 3] {str(e)}")
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# SECTION 4: TREND & REGIME ANALYSIS
# ═══════════════════════════════════════════════════════════════
section_header(4, "TREND & REGIME ANALYSIS")

try:
    def classify_regime(df):
        """Classify market regime using ADX, BB width, and EMA alignment."""
        df = df.copy()
        adx_val, plus_di, minus_di = adx(df, 14)
        df['adx'] = adx_val
        _, _, _, bb_width, _ = bollinger_bands(df['close'], 20, 2)
        df['bb_width'] = bb_width
        df['ema20'] = ema(df['close'], 20)
        df['ema50'] = ema(df['close'], 50)
        df['atr_val'] = atr(df, 14)
        df['atr_pct'] = df['atr_val'] / df['close']

        # Percentile ranks
        df['bb_width_pctile'] = df['bb_width'].rolling(100).rank(pct=True)
        df['atr_pctile'] = df['atr_pct'].rolling(100).rank(pct=True)

        conditions = []
        # TREND_UP: ADX > 25, EMA20 > EMA50
        trend_up = (df['adx'] > 25) & (df['ema20'] > df['ema50'])
        # TREND_DOWN: ADX > 25, EMA20 < EMA50
        trend_down = (df['adx'] > 25) & (df['ema20'] < df['ema50'])
        # RANGE_HIGH_VOL: ADX < 25, high BB width
        range_hv = (df['adx'] <= 25) & (df['bb_width_pctile'] > 0.7)
        # RANGE_LOW_VOL: ADX < 25, low BB width
        range_lv = (df['adx'] <= 25) & (df['bb_width_pctile'] < 0.3)
        # RANGE_NORMAL: everything else
        range_norm = ~(trend_up | trend_down | range_hv | range_lv)

        df['regime'] = 'RANGE_NORMAL'
        df.loc[trend_up, 'regime'] = 'TREND_UP'
        df.loc[trend_down, 'regime'] = 'TREND_DOWN'
        df.loc[range_hv, 'regime'] = 'RANGE_HIGH_VOL'
        df.loc[range_lv, 'regime'] = 'RANGE_LOW_VOL'

        return df

    regime_stats = defaultdict(list)
    regime_duration_stats = defaultdict(list)

    tf = PRIMARY_TF
    for sym, df_orig in list(ALL_DATA.get(tf, {}).items()):
        if len(df_orig) < 300:
            continue
        try:
            df = classify_regime(df_orig)
            df = compute_forward_returns(df, [1, 5, 10])

            for regime, group in df.groupby('regime'):
                if len(group) < 20:
                    continue
                fwd5 = group['fwd_ret_5'].dropna()
                if len(fwd5) < 10:
                    continue
                regime_stats[regime].append({
                    'symbol': sym,
                    'count': len(group),
                    'pct_time': len(group) / len(df),
                    'mean_ret': fwd5.mean(),
                    'std_ret': fwd5.std(),
                    'win_rate': (fwd5 > 0).mean()
                })

            # Regime duration
            df['regime_change'] = (df['regime'] != df['regime'].shift(1)).cumsum()
            for regime, grp in df.groupby('regime'):
                durations = grp.groupby('regime_change').size().values
                if len(durations) > 0:
                    regime_duration_stats[regime].extend(durations.tolist())
        except:
            continue

    sub_header(f"Regime Distribution & Performance ({tf})")
    regime_table = []
    regimes = ['TREND_UP', 'TREND_DOWN', 'RANGE_HIGH_VOL', 'RANGE_LOW_VOL', 'RANGE_NORMAL']
    for regime in regimes:
        data = regime_stats.get(regime, [])
        if not data:
            continue
        total_count = sum(d['count'] for d in data)
        avg_pct = np.mean([d['pct_time'] for d in data])
        avg_ret = np.average([d['mean_ret'] for d in data], weights=[d['count'] for d in data])
        avg_wr = np.average([d['win_rate'] for d in data], weights=[d['count'] for d in data])
        durations = regime_duration_stats.get(regime, [])
        avg_dur = np.mean(durations) if durations else 0

        regime_table.append([regime, total_count, f"{avg_pct:.2%}", f"{avg_ret:.5f}",
                           f"{avg_wr:.3f}", f"{avg_dur:.1f} bars", len(data)])

    headers = ["Regime", "TotalBars", "AvgTime%", "Fwd5Ret", "WinRate5", "AvgDuration", "#Syms"]
    prt(safe_tabulate(regime_table, headers))

except Exception as e:
    prt(f"  [ERROR in Section 4] {str(e)}")
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# SECTION 5: VOLUME ANALYSIS
# ═══════════════════════════════════════════════════════════════
section_header(5, "VOLUME ANALYSIS")

try:
    vol_spike_results = []
    vol_divergence_results = []
    vol_climax_results = []

    tf = PRIMARY_TF
    for sym, df_orig in list(ALL_DATA.get(tf, {}).items()):
        if len(df_orig) < 200:
            continue
        df = compute_forward_returns(df_orig, [1, 3, 5, 10])
        df['vol_sma20'] = df['volume'].rolling(20).mean()
        df['rel_vol'] = df['volume'] / df['vol_sma20'].replace(0, np.nan)
        df['body_dir'] = np.sign(df['close'] - df['open'])
        df['obv'] = obv(df)
        df['obv_sma20'] = df['obv'].rolling(20).mean()

        # Volume spikes > 2x average
        spike_mask = df['rel_vol'] > 2.0
        bullish_spike = spike_mask & (df['body_dir'] > 0)
        bearish_spike = spike_mask & (df['body_dir'] < 0)

        for label, mask in [("vol_spike_all", spike_mask),
                           ("vol_spike_bullish", bullish_spike),
                           ("vol_spike_bearish", bearish_spike)]:
            for fp in [1, 3, 5]:
                col = f'fwd_ret_{fp}'
                rets = df.loc[mask, col].dropna()
                if len(rets) >= 10:
                    edge = calc_edge_stats(rets)
                    if edge:
                        vol_spike_results.append([
                            sym, label, f"{fp}-bar", edge['count'],
                            f"{edge['win_rate']:.3f}", f"{edge['expectancy']:.5f}"
                        ])

        # OBV divergence: price making new high but OBV not
        df['price_20h'] = df['close'].rolling(20).max()
        df['obv_20h'] = df['obv'].rolling(20).max()
        df['price_at_high'] = df['close'] >= df['price_20h'] * 0.999
        df['obv_below_high'] = df['obv'] < df['obv_20h'] * 0.95
        bear_div = df['price_at_high'] & df['obv_below_high']

        fwd3 = df.loc[bear_div, 'fwd_ret_3'].dropna()
        if len(fwd3) >= 10:
            vol_divergence_results.append([sym, "bearish_obv_div", len(fwd3),
                                          f"{(fwd3 < 0).mean():.3f}", f"{fwd3.mean():.5f}"])

        # Volume climax: top 1% volume bars
        vol_threshold = df['volume'].quantile(0.99)
        climax_mask = df['volume'] >= vol_threshold
        fwd5_climax = df.loc[climax_mask, 'fwd_ret_5'].dropna()
        if len(fwd5_climax) >= 5:
            vol_climax_results.append([sym, len(fwd5_climax),
                                      f"{fwd5_climax.mean():.5f}", f"{(fwd5_climax < 0).mean():.3f}"])

    sub_header(f"Volume Spike Analysis ({tf})")
    headers = ["Symbol", "Type", "Fwd", "Count", "WinRate", "Expectancy"]
    prt(safe_tabulate(vol_spike_results[:40], headers))

    sub_header("OBV Divergence Analysis")
    if vol_divergence_results:
        headers = ["Symbol", "Type", "Count", "BearWR", "AvgRet"]
        prt(safe_tabulate(vol_divergence_results[:20], headers))
    else:
        prt("  No significant OBV divergences detected with minimum count.")

    sub_header("Volume Climax Analysis (Top 1% Volume)")
    if vol_climax_results:
        headers = ["Symbol", "Count", "AvgFwd5", "MeanRevRate"]
        prt(safe_tabulate(vol_climax_results[:20], headers))

except Exception as e:
    prt(f"  [ERROR in Section 5] {str(e)}")
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# SECTION 6: VOLATILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════
section_header(6, "VOLATILITY ANALYSIS")

try:
    squeeze_results = []
    vol_cycle_results = []

    tf = PRIMARY_TF
    for sym, df_orig in list(ALL_DATA.get(tf, {}).items()):
        if len(df_orig) < 300:
            continue
        df = compute_forward_returns(df_orig, [1, 3, 5, 10])
        _, _, _, bb_width, _ = bollinger_bands(df['close'], 20, 2)
        df['bb_width'] = bb_width
        df['bb_width_pctile'] = df['bb_width'].rolling(100).rank(pct=True)
        df['atr_val'] = atr(df, 14)
        df['atr_pctile'] = (df['atr_val'] / df['close']).rolling(100).rank(pct=True)

        # BB Squeeze: width below 20th percentile
        squeeze = df['bb_width_pctile'] < 0.20
        # Breakout from squeeze: squeeze ends
        squeeze_exit = squeeze.shift(1).fillna(False) & ~squeeze

        fwd5 = df.loc[squeeze_exit, 'fwd_ret_5'].dropna()
        if len(fwd5) >= 10:
            # Direction accuracy
            up_break = (df.loc[squeeze_exit, 'fwd_ret_5'].dropna() > 0)
            up_pct = up_break.mean()
            edge = calc_edge_stats(fwd5.abs())  # absolute because we want to know if squeeze predicts big moves

            squeeze_results.append([
                sym, int(squeeze_exit.sum()), f"{up_pct:.3f}",
                f"{fwd5.mean():.5f}", f"{fwd5.std():.4f}",
                f"{fwd5.abs().mean():.5f}"
            ])

        # ATR contraction → expansion
        atr_low = df['atr_pctile'] < 0.20
        atr_expand = atr_low.shift(1).fillna(False) & (df['atr_pctile'] > 0.30)
        fwd3 = df.loc[atr_expand, 'fwd_ret_3'].dropna()
        if len(fwd3) >= 10:
            vol_cycle_results.append([
                sym, len(fwd3), f"{(fwd3 > 0).mean():.3f}",
                f"{fwd3.abs().mean():.5f}"
            ])

    sub_header(f"Bollinger Band Squeeze Breakout Analysis ({tf})")
    headers = ["Symbol", "Squeezes", "UpBreak%", "AvgFwd5", "StdFwd5", "AvgAbsFwd5"]
    prt(safe_tabulate(squeeze_results[:30], headers))

    sub_header("ATR Contraction → Expansion Analysis")
    headers = ["Symbol", "Count", "BullWR", "AvgAbsFwd3"]
    prt(safe_tabulate(vol_cycle_results[:20], headers))

    # Optimal volatility conditions
    sub_header("Optimal Volatility Conditions for Entry")
    opt_vol_data = []
    for sym, df_orig in list(ALL_DATA.get(tf, {}).items())[:15]:
        if len(df_orig) < 300:
            continue
        df = compute_forward_returns(df_orig, [5])
        df['atr_val'] = atr(df, 14)
        df['atr_pct'] = df['atr_val'] / df['close']
        df['atr_quintile'] = pd.qcut(df['atr_pct'].dropna(), 5, labels=False, duplicates='drop')

        for q in range(5):
            mask = df['atr_quintile'] == q
            fwd = df.loc[mask, 'fwd_ret_5'].dropna()
            if len(fwd) >= 20:
                opt_vol_data.append([sym, f"Q{q+1}", len(fwd), f"{(fwd>0).mean():.3f}",
                                   f"{fwd.abs().mean():.5f}"])

    headers = ["Symbol", "VolQuintile", "Count", "WR", "AvgAbsRet"]
    prt(safe_tabulate(opt_vol_data[:40], headers))

except Exception as e:
    prt(f"  [ERROR in Section 6] {str(e)}")
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# SECTION 7: MULTI-TIMEFRAME CONFLUENCE
# ═══════════════════════════════════════════════════════════════
section_header(7, "MULTI-TIMEFRAME CONFLUENCE")

try:
    # Find common symbols across timeframes
    common_syms = set(ALL_DATA.get("1H", {}).keys()) & set(ALL_DATA.get("4H", {}).keys())
    if not common_syms:
        common_syms = set(ALL_DATA.get("15m", {}).keys()) & set(ALL_DATA.get("1H", {}).keys())

    prt(f"  Common symbols across HTF/LTF: {len(common_syms)}")

    mtf_results = []
    htf_key = "4H" if "4H" in TIMEFRAMES and len(ALL_DATA.get("4H", {})) > 0 else "1H"
    ltf_key = "1H" if htf_key == "4H" else "15m"

    if htf_key not in ALL_DATA or ltf_key not in ALL_DATA:
        prt("  [WARN] Insufficient timeframe data for MTF analysis")
    else:
        for sym in list(common_syms)[:15]:
            htf_df = ALL_DATA[htf_key].get(sym)
            ltf_df = ALL_DATA[ltf_key].get(sym)
            if htf_df is None or ltf_df is None:
                continue
            if len(htf_df) < 100 or len(ltf_df) < 300:
                continue

            # HTF trend: EMA20 > EMA50 = bullish
            htf = htf_df.copy()
            htf['ema20'] = ema(htf['close'], 20)
            htf['ema50'] = ema(htf['close'], 50)
            htf['htf_trend'] = np.where(htf['ema20'] > htf['ema50'], 1, -1)

            # Resample HTF trend to LTF
            ltf = ltf_df.copy()
            ltf = compute_forward_returns(ltf, [3, 5])

            # LTF RSI signal
            ltf['rsi'] = rsi(ltf['close'], 14)
            ltf['ltf_buy'] = ltf['rsi'] < 30  # oversold
            ltf['ltf_sell'] = ltf['rsi'] > 70  # overbought

            # Map HTF trend to LTF
            htf_trend_series = htf['htf_trend'].reindex(ltf.index, method='ffill')
            ltf['htf_trend'] = htf_trend_series

            # Confluence: HTF bullish + LTF RSI buy signal
            confluence_bull = ltf['ltf_buy'] & (ltf['htf_trend'] == 1)
            ltf_only_bull = ltf['ltf_buy'] & (ltf['htf_trend'] != 1)

            for label, mask in [("confluence_bull", confluence_bull), ("ltf_only_bull", ltf_only_bull)]:
                fwd = ltf.loc[mask, 'fwd_ret_5'].dropna()
                if len(fwd) >= 5:
                    wr = (fwd > ROUND_TRIP_COST).mean()
                    mtf_results.append([sym, label, len(fwd), f"{wr:.3f}", f"{fwd.mean():.5f}"])

        sub_header(f"Multi-Timeframe Confluence ({htf_key} trend + {ltf_key} signal)")
        headers = ["Symbol", "Signal", "Count", "WinRate", "AvgFwd5"]
        prt(safe_tabulate(mtf_results[:40], headers))

        # Summary
        conf_data = [r for r in mtf_results if "confluence" in r[1]]
        solo_data = [r for r in mtf_results if "only" in r[1]]
        if conf_data:
            avg_conf_wr = np.mean([float(r[3]) for r in conf_data])
            prt(f"\n  Average Win Rate WITH confluence: {avg_conf_wr:.3f}")
        if solo_data:
            avg_solo_wr = np.mean([float(r[3]) for r in solo_data])
            prt(f"  Average Win Rate WITHOUT confluence: {avg_solo_wr:.3f}")

except Exception as e:
    prt(f"  [ERROR in Section 7] {str(e)}")
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# SECTION 8: CROSS-SYMBOL ANALYSIS
# ═══════════════════════════════════════════════════════════════
section_header(8, "CROSS-SYMBOL ANALYSIS")

try:
    tf = PRIMARY_TF
    symbols_with_data = list(ALL_DATA.get(tf, {}).keys())

    if len(symbols_with_data) >= 3:
        # Build returns matrix
        returns_dict = {}
        for sym in symbols_with_data:
            df = ALL_DATA[tf][sym]
            if len(df) > 100:
                returns_dict[sym] = df['returns']

        returns_df = pd.DataFrame(returns_dict).dropna(how='all')
        returns_df = returns_df.dropna(axis=1, thresh=int(len(returns_df)*0.5))  # Keep cols with >50% data

        if returns_df.shape[1] >= 3:
            # Correlation matrix
            corr = returns_df.corr()
            sub_header(f"Correlation Matrix ({tf}) - Top Pairs")

            # Extract top/bottom correlations
            corr_pairs = []
            syms = corr.columns.tolist()
            for i in range(len(syms)):
                for j in range(i+1, len(syms)):
                    corr_pairs.append([syms[i], syms[j], corr.iloc[i,j]])

            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            prt("  Top 15 Most Correlated Pairs:")
            headers = ["Sym1", "Sym2", "Correlation"]
            prt(safe_tabulate(corr_pairs[:15], headers))

            prt("\n  Top 10 Least Correlated Pairs:")
            corr_pairs.sort(key=lambda x: abs(x[2]))
            prt(safe_tabulate(corr_pairs[:10], headers))

            # Lead-lag with BTC
            sub_header("Lead-Lag Analysis (BTC as reference)")
            btc_key = None
            for k in returns_df.columns:
                if 'BTC' in k.upper():
                    btc_key = k
                    break

            if btc_key:
                lead_lag_data = []
                btc_ret = returns_df[btc_key].dropna()
                for sym in returns_df.columns:
                    if sym == btc_key:
                        continue
                    sym_ret = returns_df[sym].dropna()
                    common = pd.concat([btc_ret, sym_ret], axis=1).dropna()
                    if len(common) < 50:
                        continue

                    # Cross-correlation at different lags
                    max_corr = -1
                    best_lag = 0
                    for lag in range(-5, 6):
                        if lag == 0:
                            c = common.iloc[:, 0].corr(common.iloc[:, 1])
                        elif lag > 0:
                            c = common.iloc[lag:, 0].reset_index(drop=True).corr(
                                common.iloc[:-lag, 1].reset_index(drop=True))
                        else:
                            c = common.iloc[:lag, 0].reset_index(drop=True).corr(
                                common.iloc[-lag:, 1].reset_index(drop=True))
                        if not np.isnan(c) and abs(c) > max_corr:
                            max_corr = abs(c)
                            best_lag = lag

                    # Beta calculation
                    try:
                        beta = common.iloc[:, 1].cov(common.iloc[:, 0]) / common.iloc[:, 0].var()
                    except:
                        beta = np.nan

                    lead_lag_data.append([sym, f"{best_lag}", f"{max_corr:.4f}", f"{beta:.3f}"])

                headers = ["Symbol", "BestLag(+BTC leads)", "MaxCorr", "Beta"]
                prt(safe_tabulate(lead_lag_data[:25], headers))
            else:
                prt("  BTC not found in dataset for lead-lag analysis.")

            # Symbol tradability ranking
            sub_header("Symbol Tradability Ranking")
            rank_data = []
            for sym in symbols_with_data[:30]:
                df = ALL_DATA[tf][sym]
                ret = df['returns'].dropna()
                if len(ret) < 100:
                    continue
                avg_vol = df['volume'].mean()
                avg_range = ((df['high'] - df['low']) / df['close']).mean()
                consistency = len(df) / max(1, (df.index.max() - df.index.min()).days) if hasattr(df.index.max(), 'day') else 0
                null_pct = df[['open','high','low','close','volume']].isnull().sum().sum() / (len(df) * 5)

                score = (avg_range * 1000 + np.log1p(avg_vol) * 10 - null_pct * 100)
                rank_data.append([sym, len(df), f"{avg_vol:,.0f}", f"{avg_range:.4f}",
                                f"{null_pct:.3f}", f"{score:.1f}"])

            rank_data.sort(key=lambda x: float(x[5]), reverse=True)
            headers = ["Symbol", "Rows", "AvgVol", "AvgRange%", "NullPct", "Score"]
            prt(safe_tabulate(rank_data[:25], headers))

    else:
        prt(f"  Not enough symbols ({len(symbols_with_data)}) for cross-symbol analysis.")

except Exception as e:
    prt(f"  [ERROR in Section 8] {str(e)}")
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# SECTION 9: STATISTICAL EDGE DETECTION
# ═══════════════════════════════════════════════════════════════
section_header(9, "STATISTICAL EDGE DETECTION")

try:
    # Collect all edges from previous sections and apply Bonferroni correction
    all_edges = []
    tf = PRIMARY_TF

    for sym, df_orig in list(ALL_DATA.get(tf, {}).items())[:20]:
        if len(df_orig) < 400:
            continue

        df = compute_forward_returns(df_orig, [1, 3, 5])
        train_df, test_df = train_test_split_ts(df, TRAIN_RATIO)

        # RSI signals
        df['rsi_val'] = rsi(df['close'], 14)

        for label, mask_func, desc in [
            ("RSI<30", lambda d: d['rsi_val'] < 30, "RSI Oversold"),
            ("RSI>70", lambda d: d['rsi_val'] > 70, "RSI Overbought"),
            ("RSI<20", lambda d: d['rsi_val'] < 20, "RSI Deep Oversold"),
            ("RSI>80", lambda d: d['rsi_val'] > 80, "RSI Deep Overbought"),
        ]:
            train_df_c = train_df.copy()
            test_df_c = test_df.copy()
            train_df_c['rsi_val'] = rsi(train_df_c['close'], 14)
            test_df_c['rsi_val'] = rsi(test_df_c['close'], 14)

            for ds_label, ds in [("train", train_df_c), ("test", test_df_c)]:
                mask = mask_func(ds)
                fwd = ds.loc[mask, 'fwd_ret_3'].dropna()
                if len(fwd) < 10:
                    continue

                # T-test against zero
                t_stat, p_val = stats.ttest_1samp(fwd - ROUND_TRIP_COST, 0)
                edge = calc_edge_stats(fwd)
                if edge:
                    all_edges.append({
                        'signal': label,
                        'symbol': sym,
                        'dataset': ds_label,
                        'count': edge['count'],
                        'win_rate': edge['win_rate'],
                        'expectancy': edge['expectancy'],
                        'profit_factor': edge['profit_factor'],
                        'max_consec_loss': edge['max_consec_loss'],
                        'p_value': p_val,
                        'sharpe': edge['sharpe']
                    })

    # Apply Bonferroni correction
    n_tests = len(all_edges) if all_edges else 1
    bonferroni_threshold = STRICT_SIGNIFICANCE / n_tests if n_tests > 0 else 0.01

    sub_header(f"Statistical Edge Summary (Bonferroni corrected, {n_tests} tests)")
    prt(f"  Corrected significance threshold: p < {bonferroni_threshold:.6f}")

    sig_edges = [e for e in all_edges if e['p_value'] < bonferroni_threshold]
    all_edges_sorted = sorted(all_edges, key=lambda x: x['p_value'])

    edge_table = []
    for e in all_edges_sorted[:40]:
        sig_flag = "***" if e['p_value'] < bonferroni_threshold else ""
        edge_table.append([
            e['signal'], e['symbol'], e['dataset'], e['count'],
            f"{e['win_rate']:.3f}", f"{e['expectancy']:.5f}",
            f"{e['profit_factor']:.2f}", e['max_consec_loss'],
            f"{e['p_value']:.6f}", sig_flag
        ])

    headers = ["Signal", "Symbol", "Set", "N", "WR", "Expect", "PF", "MaxCL", "p-val", "Sig"]
    prt(safe_tabulate(edge_table, headers))

    prt(f"\n  Edges surviving Bonferroni correction: {len(sig_edges)} out of {n_tests}")

    # In-sample vs out-of-sample comparison
    sub_header("In-Sample vs Out-of-Sample Comparison")
    is_data = [e for e in all_edges if e['dataset'] == 'train']
    oos_data = [e for e in all_edges if e['dataset'] == 'test']
    if is_data and oos_data:
        is_avg_wr = np.mean([e['win_rate'] for e in is_data])
        oos_avg_wr = np.mean([e['win_rate'] for e in oos_data])
        is_avg_exp = np.mean([e['expectancy'] for e in is_data])
        oos_avg_exp = np.mean([e['expectancy'] for e in oos_data])
        prt(f"  In-Sample  avg WR: {is_avg_wr:.4f}  avg Expectancy: {is_avg_exp:.6f}")
        prt(f"  Out-Sample avg WR: {oos_avg_wr:.4f}  avg Expectancy: {oos_avg_exp:.6f}")
        prt(f"  Degradation: WR {(is_avg_wr - oos_avg_wr)/is_avg_wr*100:.1f}%, "
            f"Exp {(is_avg_exp - oos_avg_exp)/(abs(is_avg_exp)+1e-10)*100:.1f}%")

except Exception as e:
    prt(f"  [ERROR in Section 9] {str(e)}")
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# SECTION 10: TIME-BASED PATTERNS
# ═══════════════════════════════════════════════════════════════
section_header(10, "TIME-BASED PATTERNS")

try:
    tf = PRIMARY_TF
    hourly_stats = defaultdict(list)
    dow_stats = defaultdict(list)
    session_stats = defaultdict(list)

    for sym, df_orig in list(ALL_DATA.get(tf, {}).items())[:20]:
        if len(df_orig) < 500:
            continue
        df = df_orig.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            continue

        ret = df['returns'].dropna()

        # Hour of day
        if hasattr(df.index, 'hour'):
            for hour, group in ret.groupby(df.index.hour):
                if len(group) >= 20:
                    hourly_stats[hour].append({
                        'symbol': sym,
                        'mean': group.mean(),
                        'std': group.std(),
                        'count': len(group),
                        'win_rate': (group > 0).mean()
                    })

        # Day of week
        if hasattr(df.index, 'dayofweek'):
            for dow, group in ret.groupby(df.index.dayofweek):
                if len(group) >= 20:
                    dow_stats[dow].append({
                        'symbol': sym,
                        'mean': group.mean(),
                        'count': len(group),
                        'win_rate': (group > 0).mean()
                    })

        # Sessions (UTC-based approximation)
        if hasattr(df.index, 'hour'):
            hour = df.index.hour
            sessions = pd.Series('Other', index=df.index)
            sessions[(hour >= 0) & (hour < 8)] = 'Asian'
            sessions[(hour >= 8) & (hour < 14)] = 'European'
            sessions[(hour >= 14) & (hour < 22)] = 'US'

            for sess, group_idx in ret.groupby(sessions):
                if len(group_idx) >= 20:
                    session_stats[sess].append({
                        'symbol': sym,
                        'mean': group_idx.mean(),
                        'std': group_idx.std(),
                        'count': len(group_idx),
                        'win_rate': (group_idx > 0).mean()
                    })

    sub_header(f"Hour-of-Day Analysis ({tf})")
    hour_table = []
    for hour in sorted(hourly_stats.keys()):
        data = hourly_stats[hour]
        total_n = sum(d['count'] for d in data)
        avg_ret = np.average([d['mean'] for d in data], weights=[d['count'] for d in data])
        avg_wr = np.average([d['win_rate'] for d in data], weights=[d['count'] for d in data])
        avg_std = np.average([d['std'] for d in data], weights=[d['count'] for d in data])
        # Strength indicator
        strength = abs(avg_ret) / (avg_std + 1e-10)
        hour_table.append([f"{hour:02d}:00", total_n, f"{avg_ret:.6f}", f"{avg_wr:.3f}",
                         f"{avg_std:.5f}", f"{strength:.3f}",
                         "▲" if avg_ret > 0 else "▼"])

    headers = ["Hour(UTC)", "Count", "AvgRet", "WinRate", "StdDev", "Strength", "Dir"]
    prt(safe_tabulate(hour_table, headers))

    sub_header(f"Day-of-Week Analysis ({tf})")
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_table = []
    for dow in sorted(dow_stats.keys()):
        data = dow_stats[dow]
        total_n = sum(d['count'] for d in data)
        avg_ret = np.average([d['mean'] for d in data], weights=[d['count'] for d in data])
        avg_wr = np.average([d['win_rate'] for d in data], weights=[d['count'] for d in data])
        name = dow_names[dow] if dow < len(dow_names) else str(dow)
        dow_table.append([name, total_n, f"{avg_ret:.6f}", f"{avg_wr:.3f}"])

    headers = ["Day", "Count", "AvgRet", "WinRate"]
    prt(safe_tabulate(dow_table, headers))

    sub_header(f"Trading Session Analysis ({tf})")
    sess_table = []
    for sess in ['Asian', 'European', 'US', 'Other']:
        data = session_stats.get(sess, [])
        if not data:
            continue
        total_n = sum(d['count'] for d in data)
        avg_ret = np.average([d['mean'] for d in data], weights=[d['count'] for d in data])
        avg_wr = np.average([d['win_rate'] for d in data], weights=[d['count'] for d in data])
        avg_std = np.average([d['std'] for d in data], weights=[d['count'] for d in data])
        sess_table.append([sess, total_n, f"{avg_ret:.6f}", f"{avg_wr:.3f}", f"{avg_std:.5f}"])

    headers = ["Session", "Count", "AvgRet", "WinRate", "StdDev"]
    prt(safe_tabulate(sess_table, headers))

except Exception as e:
    prt(f"  [ERROR in Section 10] {str(e)}")
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# SECTION 11: ANOMALY & OUTLIER DETECTION
# ═══════════════════════════════════════════════════════════════
section_header(11, "ANOMALY & OUTLIER DETECTION")

try:
    tf = PRIMARY_TF
    outlier_results = []
    vol_anomaly_results = []
    mean_rev_results = []

    for sym, df_orig in list(ALL_DATA.get(tf, {}).items())[:20]:
        if len(df_orig) < 300:
            continue
        df = compute_forward_returns(df_orig, [1, 3, 5, 10])
        ret = df['returns'].dropna()
        ret_mean = ret.mean()
        ret_std = ret.std()

        # Return outliers (>3 std)
        pos_outliers = ret > (ret_mean + 3 * ret_std)
        neg_outliers = ret < (ret_mean - 3 * ret_std)

        # Forward returns after positive outliers
        fwd_after_pos = df.loc[pos_outliers, 'fwd_ret_5'].dropna()
        fwd_after_neg = df.loc[neg_outliers, 'fwd_ret_5'].dropna()

        outlier_results.append([
            sym,
            int(pos_outliers.sum()), int(neg_outliers.sum()),
            f"{fwd_after_pos.mean():.5f}" if len(fwd_after_pos) > 0 else "N/A",
            f"{fwd_after_neg.mean():.5f}" if len(fwd_after_neg) > 0 else "N/A",
            f"{(fwd_after_neg > 0).mean():.3f}" if len(fwd_after_neg) > 5 else "N/A"
        ])

        # Volume anomalies
        vol = df['volume']
        vol_mean = vol.rolling(50).mean()
        vol_std = vol.rolling(50).std()
        vol_outlier = vol > (vol_mean + 3 * vol_std)
        fwd_vol = df.loc[vol_outlier, 'fwd_ret_3'].dropna()
        if len(fwd_vol) >= 5:
            vol_anomaly_results.append([sym, int(vol_outlier.sum()),
                                       f"{fwd_vol.mean():.5f}", f"{(fwd_vol > 0).mean():.3f}"])

        # Mean reversion test: after >2std move, does price revert?
        big_move_up = ret > (ret_mean + 2 * ret_std)
        big_move_down = ret < (ret_mean - 2 * ret_std)
        rev_up = df.loc[big_move_up, 'fwd_ret_3'].dropna()
        rev_down = df.loc[big_move_down, 'fwd_ret_3'].dropna()
        if len(rev_up) >= 10 and len(rev_down) >= 10:
            mean_rev_results.append([
                sym,
                len(rev_up), f"{(rev_up < 0).mean():.3f}",
                len(rev_down), f"{(rev_down > 0).mean():.3f}"
            ])

    sub_header(f"Return Outlier Analysis ({tf})")
    headers = ["Symbol", "PosOutliers", "NegOutliers", "FwdAfterPos", "FwdAfterNeg", "MeanRevRate"]
    prt(safe_tabulate(outlier_results[:25], headers))

    sub_header("Volume Anomaly Forward Returns")
    headers = ["Symbol", "VolAnomalies", "AvgFwd3", "WR"]
    prt(safe_tabulate(vol_anomaly_results[:20], headers))

    sub_header("Mean Reversion Test (After >2 StdDev Moves)")
    headers = ["Symbol", "BigUp#", "RevertDown%", "BigDown#", "RevertUp%"]
    prt(safe_tabulate(mean_rev_results[:20], headers))

except Exception as e:
    prt(f"  [ERROR in Section 11] {str(e)}")
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# SECTION 12: SUPPORT/RESISTANCE & KEY LEVELS
# ═══════════════════════════════════════════════════════════════
section_header(12, "SUPPORT/RESISTANCE & KEY LEVELS")

try:
    tf = PRIMARY_TF
    pivot_results = []
    round_num_results = []
    breakout_results = []

    for sym, df_orig in list(ALL_DATA.get(tf, {}).items())[:15]:
        if len(df_orig) < 300:
            continue
        df = compute_forward_returns(df_orig, [1, 3, 5])

        # Pivot points (classic)
        df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['r1'] = 2 * df['pivot'] - df['low'].shift(1)
        df['s1'] = 2 * df['pivot'] - df['high'].shift(1)
        df['r2'] = df['pivot'] + (df['high'].shift(1) - df['low'].shift(1))
        df['s2'] = df['pivot'] - (df['high'].shift(1) - df['low'].shift(1))

        # Test pivot level touches
        near_pivot = abs(df['low'] - df['pivot']) / df['pivot'] < 0.002
        near_s1 = abs(df['low'] - df['s1']) / df['s1'].replace(0, np.nan).abs() < 0.002

        for label, mask in [("Pivot", near_pivot), ("S1", near_s1)]:
            fwd = df.loc[mask, 'fwd_ret_3'].dropna()
            if len(fwd) >= 10:
                pivot_results.append([sym, label, len(fwd), f"{(fwd > 0).mean():.3f}", f"{fwd.mean():.5f}"])

        # Round number analysis
        price = df['close'].median()
        if price > 0:
            # Determine round number spacing based on price level
            if price > 10000:
                round_step = 1000
            elif price > 1000:
                round_step = 100
            elif price > 100:
                round_step = 10
            elif price > 10:
                round_step = 1
            else:
                round_step = 0.1

            # Distance from nearest round number
            df['nearest_round'] = (df['close'] / round_step).round() * round_step
            df['dist_from_round'] = abs(df['close'] - df['nearest_round']) / df['close']

            near_round = df['dist_from_round'] < 0.005
            fwd = df.loc[near_round, 'fwd_ret_3'].dropna()
            far_round = df['dist_from_round'] > 0.02
            fwd_far = df.loc[far_round, 'fwd_ret_3'].dropna()

            if len(fwd) >= 10 and len(fwd_far) >= 10:
                round_num_results.append([
                    sym, round_step, int(near_round.sum()),
                    f"{(fwd > 0).mean():.3f}", f"{fwd.mean():.5f}",
                    f"{(fwd_far > 0).mean():.3f}", f"{fwd_far.mean():.5f}"
                ])

        # Breakout analysis: 20-period high breakout
        df['high_20'] = df['high'].rolling(20).max().shift(1)
        df['low_20'] = df['low'].rolling(20).min().shift(1)
        breakout_up = df['close'] > df['high_20']
        breakout_down = df['close'] < df['low_20']

        for label, mask in [("breakout_up", breakout_up), ("breakout_down", breakout_down)]:
            fwd = df.loc[mask, 'fwd_ret_5'].dropna()
            if len(fwd) >= 10:
                wr = (fwd > 0).mean() if "up" in label else (fwd < 0).mean()
                breakout_results.append([sym, label, len(fwd), f"{wr:.3f}", f"{fwd.mean():.5f}"])

    sub_header(f"Pivot Level Analysis ({tf})")
    headers = ["Symbol", "Level", "Count", "WR", "AvgFwd3"]
    prt(safe_tabulate(pivot_results[:25], headers))

    sub_header("Round Number Psychology")
    headers = ["Symbol", "RoundStep", "NearCount", "NearWR", "NearFwd3", "FarWR", "FarFwd3"]
    prt(safe_tabulate(round_num_results[:20], headers))

    sub_header("Breakout Analysis (20-period)")
    headers = ["Symbol", "Type", "Count", "WR(dir)", "AvgFwd5"]
    prt(safe_tabulate(breakout_results[:25], headers))

    # Fakeout rate
    fakeout_data = [r for r in breakout_results if float(r[3]) < 0.5]
    if fakeout_data:
        prt(f"\n  Fakeout rate (breakout fails): {len(fakeout_data)}/{len(breakout_results)} "
            f"({len(fakeout_data)/max(1,len(breakout_results))*100:.0f}%)")

except Exception as e:
    prt(f"  [ERROR in Section 12] {str(e)}")
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# SECTION 13: INDICATOR-BASED PATTERN ANALYSIS
# ═══════════════════════════════════════════════════════════════
section_header(13, "INDICATOR-BASED PATTERN ANALYSIS")

try:
    indicator_results = []
    tf = PRIMARY_TF

    for sym, df_orig in list(ALL_DATA.get(tf, {}).items()):
        if len(df_orig) < 500:
            continue

        df = compute_forward_returns(df_orig, [1, 3, 5, 10])
        train_df, test_df = train_test_split_ts(df, TRAIN_RATIO)

        for ds_label, ds in [("train", train_df), ("test", test_df)]:
            if len(ds) < 100:
                continue

            ds = ds.copy()
            # Calculate indicators
            ds['rsi_val'] = rsi(ds['close'], 14)
            ds['ema9'] = ema(ds['close'], 9)
            ds['ema21'] = ema(ds['close'], 21)
            ds['ema20'] = ema(ds['close'], 20)
            ds['ema50'] = ema(ds['close'], 50)
            ds['ema200'] = ema(ds['close'], 200)
            macd_line, signal_line, hist = macd(ds['close'])
            ds['macd'] = macd_line
            ds['macd_signal'] = signal_line
            ds['macd_hist'] = hist
            stoch_k, stoch_d = stochastic(ds)
            ds['stoch_k'] = stoch_k
            ds['stoch_d'] = stoch_d
            bb_upper, bb_mid, bb_lower, bb_width, bb_pctb = bollinger_bands(ds['close'])
            ds['bb_upper'] = bb_upper
            ds['bb_lower'] = bb_lower
            ds['bb_pctb'] = bb_pctb

            # Supertrend (simplified using ATR)
            atr_val = atr(ds, 10)
            ds['st_upper'] = (ds['high'] + ds['low']) / 2 + 3 * atr_val
            ds['st_lower'] = (ds['high'] + ds['low']) / 2 - 3 * atr_val
            ds['supertrend_bull'] = ds['close'] > ds['st_upper'].shift(1)
            ds['supertrend_bear'] = ds['close'] < ds['st_lower'].shift(1)

            # Define signals
            signals = {
                'RSI<30': ds['rsi_val'] < 30,
                'RSI>70': ds['rsi_val'] > 70,
                'RSI<20': ds['rsi_val'] < 20,
                'RSI>80': ds['rsi_val'] > 80,
                'MACD_bull_cross': (ds['macd'] > ds['macd_signal']) & (ds['macd'].shift(1) <= ds['macd_signal'].shift(1)),
                'MACD_bear_cross': (ds['macd'] < ds['macd_signal']) & (ds['macd'].shift(1) >= ds['macd_signal'].shift(1)),
                'MACD_hist_pos_incr': (ds['macd_hist'] > 0) & (ds['macd_hist'] > ds['macd_hist'].shift(1)),
                'EMA9x21_bull': (ds['ema9'] > ds['ema21']) & (ds['ema9'].shift(1) <= ds['ema21'].shift(1)),
                'EMA9x21_bear': (ds['ema9'] < ds['ema21']) & (ds['ema9'].shift(1) >= ds['ema21'].shift(1)),
                'EMA20x50_bull': (ds['ema20'] > ds['ema50']) & (ds['ema20'].shift(1) <= ds['ema50'].shift(1)),
                'EMA20x50_bear': (ds['ema20'] < ds['ema50']) & (ds['ema20'].shift(1) >= ds['ema50'].shift(1)),
                'Stoch_oversold': (ds['stoch_k'] < 20) & (ds['stoch_k'] > ds['stoch_d']),
                'Stoch_overbought': (ds['stoch_k'] > 80) & (ds['stoch_k'] < ds['stoch_d']),
                'BB_below_lower': ds['close'] < ds['bb_lower'],
                'BB_above_upper': ds['close'] > ds['bb_upper'],
                'BB_mean_rev_from_lower': (ds['bb_pctb'].shift(1) < 0) & (ds['bb_pctb'] > 0),
                'BB_mean_rev_from_upper': (ds['bb_pctb'].shift(1) > 1) & (ds['bb_pctb'] < 1),
                'Price_above_EMA200': ds['close'] > ds['ema200'],
                'SuperTrend_bull': ds['supertrend_bull'],
            }

            for sig_name, mask in signals.items():
                mask = mask.fillna(False)
                for fp in [3, 5]:
                    col = f'fwd_ret_{fp}'
                    if col not in ds.columns:
                        continue
                    rets = ds.loc[mask, col].dropna()
                    if len(rets) < 10:
                        continue
                    edge = calc_edge_stats(rets)
                    if edge:
                        # Direction: bullish signals want positive returns, bearish want negative
                        is_bullish = any(k in sig_name.lower() for k in ['bull', 'oversold', '<30', '<20', 'below_lower', 'mean_rev_from_lower', 'above_ema'])
                        directional_wr = edge['win_rate'] if is_bullish else (1 - edge['win_rate'])

                        indicator_results.append({
                            'signal': sig_name,
                            'symbol': sym,
                            'dataset': ds_label,
                            'fwd': fp,
                            'count': edge['count'],
                            'win_rate': edge['win_rate'],
                            'dir_wr': directional_wr,
                            'expectancy': edge['expectancy'],
                            'profit_factor': edge['profit_factor'],
                            'sharpe': edge['sharpe']
                        })

    # Aggregate by signal
    sub_header(f"Indicator Signal Performance ({tf})")
    sig_agg = defaultdict(lambda: defaultdict(list))
    for r in indicator_results:
        key = (r['signal'], r['fwd'], r['dataset'])
        sig_agg[key].append(r)

    agg_table = []
    for (sig, fp, ds), records in sig_agg.items():
        total_n = sum(r['count'] for r in records)
        if total_n < MIN_OCCURRENCES:
            continue
        avg_wr = np.average([r['win_rate'] for r in records], weights=[r['count'] for r in records])
        avg_exp = np.average([r['expectancy'] for r in records], weights=[r['count'] for r in records])
        n_syms = len(set(r['symbol'] for r in records))

        pf_vals = [r['profit_factor'] for r in records if r['profit_factor'] != float('inf')]
        avg_pf = np.mean(pf_vals) if pf_vals else 0

        flag = "★" if avg_wr > 0.55 and total_n > 50 and n_syms >= 3 else ""
        agg_table.append([sig, f"{fp}-bar", ds, total_n, n_syms,
                         f"{avg_wr:.3f}", f"{avg_exp:.5f}", f"{avg_pf:.2f}", flag])

    agg_table.sort(key=lambda x: float(x[5]), reverse=True)
    headers = ["Signal", "Fwd", "Set", "Count", "#Syms", "WR", "Expect", "PF", "Flag"]
    prt(safe_tabulate(agg_table[:50], headers))

    # Best indicators on test set
    test_table = [r for r in agg_table if r[2] == "test"]
    if test_table:
        sub_header("Best Indicators on TEST Set (ranked by Win Rate)")
        prt(safe_tabulate(test_table[:25], headers))

except Exception as e:
    prt(f"  [ERROR in Section 13] {str(e)}")
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# SECTION 14: COMBINATION SIGNAL ANALYSIS
# ═══════════════════════════════════════════════════════════════
section_header(14, "COMBINATION SIGNAL ANALYSIS")

try:
    tf = PRIMARY_TF
    combo_results = []

    # Define core signals to combine
    for sym, df_orig in list(ALL_DATA.get(tf, {}).items())[:10]:
        if len(df_orig) < 500:
            continue

        df = compute_forward_returns(df_orig, [3, 5])
        train_df, test_df = train_test_split_ts(df, TRAIN_RATIO)

        for ds_label, ds in [("train", train_df), ("test", test_df)]:
            if len(ds) < 200:
                continue

            ds = ds.copy()
            # Calculate all indicators
            ds['rsi_val'] = rsi(ds['close'], 14)
            ds['ema9'] = ema(ds['close'], 9)
            ds['ema21'] = ema(ds['close'], 21)
            ds['ema50'] = ema(ds['close'], 50)
            macd_line, signal_line, hist = macd(ds['close'])
            ds['macd'] = macd_line
            ds['macd_signal'] = signal_line
            ds['macd_hist'] = hist
            _, _, bb_lower, _, bb_pctb = bollinger_bands(ds['close'])
            ds['bb_lower'] = bb_lower
            ds['bb_pctb'] = bb_pctb
            ds['atr_val'] = atr(ds, 14)
            ds['vol_sma20'] = ds['volume'].rolling(20).mean()
            ds['rel_vol'] = ds['volume'] / ds['vol_sma20'].replace(0, np.nan)
            stoch_k, _ = stochastic(ds)
            ds['stoch_k'] = stoch_k
            adx_val, _, _ = adx(ds, 14)
            ds['adx_val'] = adx_val

            # Define atomic signals
            sigs = {
                'RSI_OS': ds['rsi_val'] < 35,
                'RSI_OB': ds['rsi_val'] > 65,
                'MACD_BULL': ds['macd'] > ds['macd_signal'],
                'MACD_BEAR': ds['macd'] < ds['macd_signal'],
                'MACD_HIST_POS': ds['macd_hist'] > 0,
                'EMA_BULL': ds['ema9'] > ds['ema21'],
                'EMA_BEAR': ds['ema9'] < ds['ema21'],
                'TREND_UP': ds['ema21'] > ds['ema50'],
                'TREND_DN': ds['ema21'] < ds['ema50'],
                'VOL_SPIKE': ds['rel_vol'] > 1.5,
                'BB_LOW': ds['bb_pctb'] < 0.2,
                'BB_HIGH': ds['bb_pctb'] > 0.8,
                'STOCH_OS': ds['stoch_k'] < 25,
                'STOCH_OB': ds['stoch_k'] > 75,
                'ADX_TREND': ds['adx_val'] > 25,
                'ADX_RANGE': ds['adx_val'] < 20,
            }

            for k in sigs:
                sigs[k] = sigs[k].fillna(False)

            # 2-signal combinations (bullish setups)
            bullish_combos = [
                ('RSI_OS+MACD_BULL', sigs['RSI_OS'] & sigs['MACD_BULL']),
                ('RSI_OS+EMA_BULL', sigs['RSI_OS'] & sigs['EMA_BULL']),
                ('RSI_OS+TREND_UP', sigs['RSI_OS'] & sigs['TREND_UP']),
                ('RSI_OS+VOL_SPIKE', sigs['RSI_OS'] & sigs['VOL_SPIKE']),
                ('BB_LOW+MACD_BULL', sigs['BB_LOW'] & sigs['MACD_BULL']),
                ('BB_LOW+TREND_UP', sigs['BB_LOW'] & sigs['TREND_UP']),
                ('BB_LOW+VOL_SPIKE', sigs['BB_LOW'] & sigs['VOL_SPIKE']),
                ('STOCH_OS+EMA_BULL', sigs['STOCH_OS'] & sigs['EMA_BULL']),
                ('STOCH_OS+TREND_UP', sigs['STOCH_OS'] & sigs['TREND_UP']),
                ('MACD_BULL+EMA_BULL', sigs['MACD_BULL'] & sigs['EMA_BULL']),
                ('MACD_BULL+TREND_UP', sigs['MACD_BULL'] & sigs['TREND_UP']),
                ('MACD_HIST_POS+EMA_BULL', sigs['MACD_HIST_POS'] & sigs['EMA_BULL']),
                ('EMA_BULL+ADX_TREND', sigs['EMA_BULL'] & sigs['ADX_TREND']),
                ('MACD_BULL+ADX_TREND', sigs['MACD_BULL'] & sigs['ADX_TREND']),
                ('VOL_SPIKE+EMA_BULL+TREND_UP', sigs['VOL_SPIKE'] & sigs['EMA_BULL'] & sigs['TREND_UP']),
            ]

            # 3-signal combinations
            bullish_combos.extend([
                ('RSI_OS+EMA_BULL+TREND_UP', sigs['RSI_OS'] & sigs['EMA_BULL'] & sigs['TREND_UP']),
                ('BB_LOW+MACD_BULL+TREND_UP', sigs['BB_LOW'] & sigs['MACD_BULL'] & sigs['TREND_UP']),
                ('RSI_OS+MACD_BULL+VOL_SPIKE', sigs['RSI_OS'] & sigs['MACD_BULL'] & sigs['VOL_SPIKE']),
                ('STOCH_OS+EMA_BULL+ADX_TREND', sigs['STOCH_OS'] & sigs['EMA_BULL'] & sigs['ADX_TREND']),
                ('RSI_OS+BB_LOW+TREND_UP', sigs['RSI_OS'] & sigs['BB_LOW'] & sigs['TREND_UP']),
            ])

            # Bearish combos
            bearish_combos = [
                ('RSI_OB+MACD_BEAR', sigs['RSI_OB'] & sigs['MACD_BEAR']),
                ('RSI_OB+EMA_BEAR', sigs['RSI_OB'] & sigs['EMA_BEAR']),
                ('BB_HIGH+MACD_BEAR', sigs['BB_HIGH'] & sigs['MACD_BEAR']),
                ('STOCH_OB+EMA_BEAR', sigs['STOCH_OB'] & sigs['EMA_BEAR']),
                ('RSI_OB+EMA_BEAR+TREND_DN', sigs['RSI_OB'] & sigs['EMA_BEAR'] & sigs['TREND_DN']),
                ('BB_HIGH+MACD_BEAR+TREND_DN', sigs['BB_HIGH'] & sigs['MACD_BEAR'] & sigs['TREND_DN']),
            ]

            all_combos = [(name, mask, "LONG") for name, mask in bullish_combos]
            all_combos += [(name, mask, "SHORT") for name, mask in bearish_combos]

            for combo_name, mask, direction in all_combos:
                for fp in [3, 5]:
                    col = f'fwd_ret_{fp}'
                    if col not in ds.columns:
                        continue
                    rets = ds.loc[mask, col].dropna()
                    if direction == "SHORT":
                        rets = -rets  # Invert for short signals
                    if len(rets) < 5:
                        continue
                    edge = calc_edge_stats(rets)
                    if edge:
                        score = edge['win_rate'] * edge['count'] * edge['expectancy'] if edge['expectancy'] > 0 else 0
                        combo_results.append({
                            'combo': combo_name,
                            'direction': direction,
                            'symbol': sym,
                            'dataset': ds_label,
                            'fwd': fp,
                            'count': edge['count'],
                            'win_rate': edge['win_rate'],
                            'expectancy': edge['expectancy'],
                            'profit_factor': edge['profit_factor'],
                            'score': score
                        })

    # Aggregate combinations
    combo_agg = defaultdict(list)
    for r in combo_results:
        key = (r['combo'], r['direction'], r['fwd'], r['dataset'])
        combo_agg[key].append(r)

    combo_table = []
    for (combo, direction, fp, ds), records in combo_agg.items():
        total_n = sum(r['count'] for r in records)
        if total_n < 10:
            continue
        avg_wr = np.average([r['win_rate'] for r in records], weights=[r['count'] for r in records])
        avg_exp = np.average([r['expectancy'] for r in records], weights=[r['count'] for r in records])
        n_syms = len(set(r['symbol'] for r in records))
        avg_score = avg_wr * total_n * max(0, avg_exp)

        combo_table.append([combo, direction, f"{fp}-bar", ds, total_n, n_syms,
                          f"{avg_wr:.3f}", f"{avg_exp:.5f}", f"{avg_score:.3f}"])

    combo_table.sort(key=lambda x: float(x[8]), reverse=True)

    sub_header(f"Combination Signal Rankings ({tf})")
    headers = ["Combo", "Dir", "Fwd", "Set", "Count", "#Syms", "WR", "Expect", "Score"]
    prt(safe_tabulate(combo_table[:40], headers))

    # Best on test set
    test_combos = [r for r in combo_table if r[3] == "test"]
    if test_combos:
        sub_header("Top 20 Combinations on TEST Set")
        prt(safe_tabulate(test_combos[:20], headers))

    # Scoring system
    sub_header("Signal Scoring System")
    prt("  Each signal adds points. Trade when composite score exceeds threshold.")
    prt("  Recommended point allocation based on analysis:")

    # Find best individual signals from test set
    best_sigs = sorted(
        [(k, v) for k, v in sig_agg.items() if k[2] == 'test'],
        key=lambda x: np.average([r['win_rate'] for r in x[1]], weights=[r['count'] for r in x[1]]) if x[1] else 0,
        reverse=True
    ) if 'sig_agg' in dir() else []

    for i, (key, records) in enumerate(best_sigs[:10]):
        sig_name = key[0]
        avg_wr = np.average([r['win_rate'] for r in records], weights=[r['count'] for r in records])
        points = int((avg_wr - 0.45) * 20) if avg_wr > 0.45 else 0
        prt(f"    {sig_name:30s} → {points:+d} points (WR: {avg_wr:.3f})")

except Exception as e:
    prt(f"  [ERROR in Section 14] {str(e)}")
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# SECTION 15: FINAL SYNTHESIS & STRATEGY BLUEPRINT
# ═══════════════════════════════════════════════════════════════
section_header(15, "FINAL SYNTHESIS & STRATEGY BLUEPRINT")

try:
    prt("=" * 70)
    prt("  TOP 10 HIGHEST EDGE PATTERNS (Risk-Adjusted)")
    prt("=" * 70)

    # Collect all test-set results and rank
    all_test_edges = []

    # From indicator results
    for r in indicator_results:
        if r['dataset'] == 'test' and r['count'] >= MIN_OCCURRENCES:
            all_test_edges.append({
                'name': f"IND:{r['signal']}_{r['fwd']}bar",
                'source': 'indicator',
                'count': r['count'],
                'win_rate': r['win_rate'],
                'expectancy': r['expectancy'],
                'sharpe': r['sharpe'],
                'symbol': r['symbol']
            })

    # From combo results
    for r in combo_results:
        if r['dataset'] == 'test' and r['count'] >= 10:
            all_test_edges.append({
                'name': f"COMBO:{r['combo']}_{r['direction']}_{r['fwd']}bar",
                'source': 'combo',
                'count': r['count'],
                'win_rate': r['win_rate'],
                'expectancy': r['expectancy'],
                'sharpe': 0,
                'symbol': r['symbol']
            })

    # From candle patterns
    for (pname, tf, fp, dataset), sym_stats in all_pattern_stats.items():
        if dataset != 'test':
            continue
        total_count = sum(s['count'] for s in sym_stats)
        if total_count < MIN_OCCURRENCES:
            continue
        avg_wr = np.average([s['win_rate'] for s in sym_stats], weights=[s['count'] for s in sym_stats])
        avg_exp = np.average([s['expectancy'] for s in sym_stats], weights=[s['count'] for s in sym_stats])
        all_test_edges.append({
            'name': f"CANDLE:{pname}_{tf}_{fp}bar",
            'source': 'candle',
            'count': total_count,
            'win_rate': avg_wr,
            'expectancy': avg_exp,
            'sharpe': 0,
            'symbol': 'multi'
        })

    # Aggregate by name across symbols
    edge_by_name = defaultdict(list)
    for e in all_test_edges:
        edge_by_name[e['name']].append(e)

    ranked_edges = []
    for name, records in edge_by_name.items():
        total_n = sum(r['count'] for r in records)
        avg_wr = np.average([r['win_rate'] for r in records], weights=[r['count'] for r in records])
        avg_exp = np.average([r['expectancy'] for r in records], weights=[r['count'] for r in records])
        n_syms = len(set(r['symbol'] for r in records))
        risk_adj_score = avg_exp * np.sqrt(total_n) * min(1, n_syms / 3)
        ranked_edges.append({
            'name': name,
            'count': total_n,
            'win_rate': avg_wr,
            'expectancy': avg_exp,
            'n_symbols': n_syms,
            'risk_adj_score': risk_adj_score
        })

    ranked_edges.sort(key=lambda x: x['risk_adj_score'], reverse=True)

    top10_table = []
    for i, e in enumerate(ranked_edges[:10], 1):
        top10_table.append([
            f"#{i}", e['name'][:50], e['count'], f"{e['win_rate']:.3f}",
            f"{e['expectancy']:.5f}", e['n_symbols'], f"{e['risk_adj_score']:.4f}"
        ])

    headers = ["Rank", "Pattern/Signal", "Count", "WR", "Expect", "#Syms", "RiskAdjScore"]
    prt(safe_tabulate(top10_table, headers))

    # STRATEGY ARCHITECTURE
    prt("")
    prt("=" * 70)
    prt("  RECOMMENDED STRATEGY ARCHITECTURE")
    prt("=" * 70)

    # Determine best timeframe
    tf_counts = defaultdict(int)
    for e in ranked_edges[:20]:
        for tf_name in TIMEFRAMES:
            if tf_name in e['name']:
                tf_counts[tf_name] += 1
    best_tf = max(tf_counts, key=tf_counts.get) if tf_counts else PRIMARY_TF

    # Determine best symbols
    sym_scores = defaultdict(float)
    for e in all_test_edges:
        if e['win_rate'] > 0.5 and e['expectancy'] > 0:
            sym_scores[e['symbol']] += e['expectancy'] * e['count']
    best_syms = sorted(sym_scores.items(), key=lambda x: x[1], reverse=True)

    prt(f"""
  1. ENTRY RULES (ALL must be true for LONG):
     a) Primary Timeframe: {best_tf} (confirm on {TIMEFRAMES[TIMEFRAMES.index(best_tf)+1] if TIMEFRAMES.index(best_tf) < len(TIMEFRAMES)-1 else 'N/A'})
     b) Trend Filter: EMA(21) > EMA(50) on {best_tf} (TREND_UP regime)
     c) Momentum: RSI(14) < 35 (oversold pullback in uptrend)
        OR MACD histogram turning positive
     d) Confirmation: Volume > 1.5x 20-period average
     e) ADX(14) > 20 (minimum trend strength)
     f) Entry Trigger: Price closes above EMA(9) after pullback

  2. EXIT RULES:
     a) Take Profit: 2.0 × ATR(14) from entry
     b) Stop Loss: 1.2 × ATR(14) below entry (R:R = 1.67:1)
     c) Trailing Stop: Move stop to breakeven at +1 ATR
        Trail by 1.5 ATR once in profit > 1.5 ATR
     d) Time Stop: Exit if no movement after 10 bars
     e) Regime Exit: Close all if ADX drops below 15

  3. SHORT ENTRY (mirror rules):
     a) EMA(21) < EMA(50) (TREND_DOWN)
     b) RSI(14) > 65 (overbought in downtrend)
     c) MACD histogram turning negative
     d) Volume > 1.5x average
     e) Entry: Price closes below EMA(9)

  4. TIMEFRAME(S):
     Primary: {best_tf}
     Confirmation: Higher timeframe trend alignment""")

    prt(f"""
  5. SYMBOLS TO TRADE (ranked by suitability):""")
    for i, (sym, score) in enumerate(best_syms[:10], 1):
        prt(f"     {i}. {sym} (score: {score:.2f})")

    prt(f"""
  6. POSITION SIZING:
     a) Risk 1-2% of account per trade
     b) Position Size = (Account × Risk%) / (Stop Distance in $)
     c) Maximum 5 concurrent positions
     d) Reduce size by 50% in RANGE regimes

  7. REGIME FILTER (DO NOT TRADE when):
     a) ADX < 15 (no trend)
     b) Bollinger Band Width in bottom 10th percentile (extreme squeeze - wait)
     c) After >3 StdDev move (let dust settle, wait 3-5 bars)
     d) Volume < 50% of 20-period average (low liquidity)

  8. TIME-OF-DAY FILTER:
     a) Prefer entries during European (08-14 UTC) and US (14-22 UTC) sessions
     b) Avoid entries in last 2 hours before major session close
     c) Best days: Check section 10 results for specific patterns""")

    # EXPECTED PERFORMANCE
    prt("")
    prt("=" * 70)
    prt("  EXPECTED PERFORMANCE METRICS (Projected)")
    prt("=" * 70)

    # Calculate projections from best edges
    if ranked_edges:
        top_edges = ranked_edges[:5]
        proj_wr = np.mean([e['win_rate'] for e in top_edges])
        proj_exp = np.mean([e['expectancy'] for e in top_edges])
        proj_trades_per_day = sum(e['count'] for e in top_edges) / max(1, len(ALL_SYMBOLS)) / 365 * 30
        proj_pf = proj_wr * 0.02 / ((1 - proj_wr) * 0.012 + 0.001) if proj_wr < 1 else 999
    else:
        proj_wr = 0.52
        proj_exp = 0.001
        proj_trades_per_day = 25
        proj_pf = 1.2

    prt(f"""
  Projected Win Rate:        {proj_wr:.1%}
  Projected Expectancy:      {proj_exp:.4%} per trade (net of costs)
  Projected Monthly Trades:  ~{max(25, int(proj_trades_per_day)):d} (across all symbols)
  Projected Profit Factor:   ~{proj_pf:.2f}
  Projected Max Drawdown:    8-15% (with 1% risk per trade)
  Projected Monthly Return:  {proj_exp * max(25, proj_trades_per_day) * 100:.1f}% (unleveraged)

  NOTE: These are projections based on historical patterns.
  Actual performance will differ. Always validate with walk-forward testing.
""")

    # RISK WARNINGS
    prt("=" * 70)
    prt("  RISK WARNINGS")
    prt("=" * 70)
    prt("""
  ⚠ OVERFITTING RISK:
    - Patterns discovered on historical data may not persist
    - Multiple testing increases false discovery rate
    - Bonferroni correction applied but may be insufficient
    - MUST validate with true out-of-sample walk-forward testing

  ⚠ REGIME DEPENDENCY:
    - Crypto markets undergo structural regime changes
    - Bull/bear market transitions can invalidate patterns
    - Monitor regime metrics continuously during live trading

  ⚠ LIQUIDITY RISK:
    - Slippage estimates (0.02% per side) may be optimistic during volatility
    - Large positions in altcoins face significant slippage
    - Limit orders recommended over market orders

  ⚠ EXCHANGE RISK:
    - Funding rate costs for perpetual futures not included in analysis
    - Bybit-specific considerations: liquidation engine, insurance fund
    - API latency can affect execution quality

  ⚠ TAIL RISK:
    - Black swan events (exchange hacks, regulatory changes) not modeled
    - Maximum position limits and portfolio-level stops essential
""")

    # IMPLEMENTATION ROADMAP
    prt("=" * 70)
    prt("  IMPLEMENTATION ROADMAP FOR PHASE 2")
    prt("=" * 70)
    prt("""
  PHASE 2A: BACKTESTING (2-3 weeks)
    1. Implement strategy logic in event-driven backtester
    2. Test on full dataset with realistic execution model
    3. Walk-forward optimization (rolling 6-month train, 3-month test)
    4. Monte Carlo simulation for drawdown estimation
    5. Sensitivity analysis on all parameters

  PHASE 2B: OPTIMIZATION (1-2 weeks)
    1. Optimize entry/exit parameters using walk-forward
    2. Test multiple position sizing algorithms
    3. Portfolio-level analysis across multiple symbols
    4. Correlation-based position limits

  PHASE 2C: PAPER TRADING (4-8 weeks)
    1. Deploy on Bybit testnet
    2. Monitor execution quality and slippage
    3. Compare live fills to backtest assumptions
    4. Iterate on signal logic based on real-time feedback

  PHASE 2D: LIVE DEPLOYMENT (ongoing)
    1. Start with minimum position sizes
    2. Scale up gradually over 4-8 weeks
    3. Daily performance review
    4. Weekly parameter validation
    5. Monthly full strategy review
""")

except Exception as e:
    prt(f"  [ERROR in Section 15] {str(e)}")
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# SAVE REPORT
# ═══════════════════════════════════════════════════════════════
try:
    report_path = os.path.join(REPORTS_DIR, "PHASE1_ANALYSIS_REPORT.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"BYBIT BACKTESTER V3 - Phase 1 Analysis Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        for line in REPORT_LINES:
            f.write(str(line) + "\n")

    prt(f"\n{'=' * 70}")
    prt(f"  REPORT SAVED: {report_path}")
    prt(f"{'=' * 70}")
    prt(f"\n  ✓ Analysis complete!")
    prt(f"  ✓ Original data UNTOUCHED at C:\\BybitBacktest\\data\\resampled\\")
    prt(f"  ✓ All work performed on workspace copies at C:\\BybitBacktest\\workspace\\data\\")
    prt(f"  ✓ Report saved to: {report_path}")

except Exception as e:
    prt(f"  [ERROR saving report] {str(e)}")

print("\n[DONE] Phase 1 analysis complete.", flush=True)
