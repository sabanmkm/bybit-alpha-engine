# ============================================================
# BYBIT BACKTESTING BOT — FULL PYTHON ENGINE
# ============================================================
import os
import sys
import time
import warnings
import traceback
from datetime import datetime
from pathlib import Path

# Force stdout/stderr to UTF-8 to prevent encoding crashes on GBK/CP936/Windows-1252 systems
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings("ignore")

# --- Dependency check ---
missing = []
try:
    import pandas as pd
except ImportError:
    missing.append("pandas")
try:
    import numpy as np
except ImportError:
    missing.append("numpy")
try:
    import pyarrow
except ImportError:
    missing.append("pyarrow")
try:
    from scipy import stats
except ImportError:
    missing.append("scipy")
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    missing.append("colorama")
try:
    from tabulate import tabulate
except ImportError:
    missing.append("tabulate")

if missing:
    print("Missing packages. Install via:")
    print("  pip install " + " ".join(missing))
    sys.exit(1)

# ============================================================
# CONFIGURATION
# ============================================================
DATA_ROOT = r"C:\BybitBacktest\data\resampled"
RESULTS_ROOT = r"C:\BybitBacktest\results"
os.makedirs(RESULTS_ROOT, exist_ok=True)

IS_START = pd.Timestamp("2022-01-01")
IS_END   = pd.Timestamp("2025-12-31 23:59:59")
OOS_START = pd.Timestamp("2026-01-01")

STARTING_CAPITAL = 10000.0
RISK_PER_TRADE = 0.01
FEE_RATE = 0.00055
SLIPPAGE = 0.0003
MAX_TOTAL_POSITIONS = 5
LEVERAGE = 1.0

SYMBOLS = [
    "1000LUNC_USDT_USDT", "ANKR_USDT_USDT", "CHZ_USDT_USDT", "CRV_USDT_USDT",
    "JST_USDT_USDT", "ONT_USDT_USDT", "OP_USDT_USDT", "SUN_USDT_USDT", "TRX_USDT_USDT",
    "APT_USDT_USDT", "AVAX_USDT_USDT", "DYDX_USDT_USDT", "FLOW_USDT_USDT",
    "MANA_USDT_USDT", "MINA_USDT_USDT", "TWT_USDT_USDT", "ZEN_USDT_USDT",
    "ZRX_USDT_USDT", "ALICE_USDT_USDT",
    "BTC_USDT_USDT", "ETH_USDT_USDT", "BNB_USDT_USDT",
    "1000BTT_USDT_USDT", "AGLD_USDT_USDT", "ATOM_USDT_USDT",
]

TIMEFRAMES = ["1H", "4H", "1D"]

BARS_PER_YEAR = {"15m": 35040, "30m": 17520, "1H": 8760, "4H": 2190, "1D": 365}

# ============================================================
# UTILITIES: ASCII safe borders & colors
# ============================================================
C_CYAN = Fore.CYAN; C_YEL = Fore.YELLOW; C_GRN = Fore.GREEN
C_RED = Fore.RED; C_MAG = Fore.MAGENTA; C_WHT = Fore.WHITE
S_BR = Style.BRIGHT; S_RS = Style.RESET_ALL

def box(txt, color=C_CYAN):
    line = "=" * max(60, len(txt) + 4)
    print(color + S_BR + "+" + line + "+")
    print(color + S_BR + "|  " + txt.ljust(len(line) - 2) + "|")
    print(color + S_BR + "+" + line + "+" + S_RS)

def section(txt, color=C_MAG):
    print()
    print(color + S_BR + "-" * 70)
    print(color + S_BR + "  " + txt)
    print(color + S_BR + "-" * 70 + S_RS)

# ============================================================
# DATE GATE
# ============================================================
def enforce_date_gate(df):
    """Ensures no data past IS_END leaks into the backtest."""
    if df is None or len(df) == 0:
        return df
    df = df[df.index <= IS_END].copy()
    df = df[df.index >= IS_START].copy()
    if len(df) > 0 and df.index.max() > IS_END:
        raise ValueError("DATE GATE VIOLATION: 2026 data leaked into backtest!")
    return df

# ============================================================
# DATA LOADING
# ============================================================
_data_cache = {}

def load_data(symbol, tf):
    key = (symbol, tf)
    if key in _data_cache:
        return _data_cache[key]
    path = Path(DATA_ROOT) / tf / f"{symbol}.parquet"
    if not path.exists():
        _data_cache[key] = None
        return None
    try:
        df = pd.read_parquet(path)
        # Find timestamp column
        ts_col = None
        for c in ["timestamp", "datetime", "time", "date"]:
            if c in df.columns:
                ts_col = c
                break
        if ts_col is not None:
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            df = df.set_index(ts_col)
        else:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
        cols = {c.lower(): c for c in df.columns}
        rename_map = {}
        for want in ["open", "high", "low", "close", "volume"]:
            if want in cols:
                rename_map[cols[want]] = want
        df = df.rename(columns=rename_map)
        needed = ["open", "high", "low", "close", "volume"]
        for n in needed:
            if n not in df.columns:
                _data_cache[key] = None
                return None
        df = df[needed].astype(float)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        df = df.dropna()
        df = enforce_date_gate(df)
        if len(df) < 200:
            _data_cache[key] = None
            return None
        _data_cache[key] = df
        return df
    except Exception as e:
        _data_cache[key] = None
        return None

# ============================================================
# INDICATORS (vectorized, no lookahead)
# ============================================================
def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def sma(s, n):
    return s.rolling(n, min_periods=n).mean()

def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    ru = up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rd = dn.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = ru / rd.replace(0, np.nan)
    r = 100 - (100 / (1 + rs))
    return r.fillna(50)

def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def adx(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr_.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr_.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean().fillna(0)

def macd(close, fast=12, slow=26, sig=9):
    e_fast = ema(close, fast)
    e_slow = ema(close, slow)
    line = e_fast - e_slow
    signal = ema(line, sig)
    hist = line - signal
    return line, signal, hist

def bollinger(close, n=20, k=2.0):
    m = sma(close, n)
    s = close.rolling(n, min_periods=n).std()
    up = m + k * s
    lo = m - k * s
    width = (up - lo) / m.replace(0, np.nan)
    return up, m, lo, width

def keltner(df, n=20, k=1.5):
    m = ema(df["close"], n)
    a = atr(df, n)
    up = m + k * a
    lo = m - k * a
    return up, m, lo

def supertrend(df, period=10, mult=3.0):
    a = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    close = df["close"].values
    up = upper.values.copy()
    lo = lower.values.copy()
    trend = np.ones(len(df))
    for i in range(1, len(df)):
        if close[i] > up[i-1]:
            trend[i] = 1
        elif close[i] < lo[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            if trend[i] == 1 and lo[i] < lo[i-1]:
                lo[i] = lo[i-1]
            if trend[i] == -1 and up[i] > up[i-1]:
                up[i] = up[i-1]
    st = np.where(trend == 1, lo, up)
    return pd.Series(st, index=df.index), pd.Series(trend, index=df.index)

def hull_ma(close, n=21):
    half = int(n / 2)
    sqn = int(np.sqrt(n))
    wma_half = close.rolling(half).apply(lambda x: np.dot(x, np.arange(1, half+1)) / (half*(half+1)/2), raw=True)
    wma_full = close.rolling(n).apply(lambda x: np.dot(x, np.arange(1, n+1)) / (n*(n+1)/2), raw=True)
    diff = 2 * wma_half - wma_full
    hma = diff.rolling(sqn).apply(lambda x: np.dot(x, np.arange(1, sqn+1)) / (sqn*(sqn+1)/2), raw=True)
    return hma

def psar(df, af0=0.02, af_max=0.2):
    high = df["high"].values; low = df["low"].values
    n = len(df)
    psar_v = np.zeros(n); trend = np.zeros(n)
    if n < 2:
        return pd.Series(psar_v, index=df.index), pd.Series(trend, index=df.index)
    trend[0] = 1
    psar_v[0] = low[0]
    ep = high[0]; af = af0
    for i in range(1, n):
        psar_v[i] = psar_v[i-1] + af * (ep - psar_v[i-1])
        if trend[i-1] == 1:
            if low[i] < psar_v[i]:
                trend[i] = -1
                psar_v[i] = ep
                ep = low[i]; af = af0
            else:
                trend[i] = 1
                if high[i] > ep:
                    ep = high[i]; af = min(af + af0, af_max)
        else:
            if high[i] > psar_v[i]:
                trend[i] = 1
                psar_v[i] = ep
                ep = high[i]; af = af0
            else:
                trend[i] = -1
                if low[i] < ep:
                    ep = low[i]; af = min(af + af0, af_max)
    return pd.Series(psar_v, index=df.index), pd.Series(trend, index=df.index)

def stoch_rsi(close, n=14, k=3, d=3):
    r = rsi(close, n)
    lo = r.rolling(n).min()
    hi = r.rolling(n).max()
    stoch = 100 * (r - lo) / (hi - lo).replace(0, np.nan)
    kk = stoch.rolling(k).mean()
    dd = kk.rolling(d).mean()
    return kk.fillna(50), dd.fillna(50)

def donchian(df, n=20):
    up = df["high"].rolling(n).max()
    lo = df["low"].rolling(n).min()
    mid = (up + lo) / 2
    return up, mid, lo

def roc(close, n=10):
    return 100 * (close / close.shift(n) - 1)

# ============================================================
# CANDLESTICK PATTERNS (vectorized, no lookahead)
# ============================================================
def _body(df): return (df["close"] - df["open"]).abs()
def _range(df): return (df["high"] - df["low"]).replace(0, np.nan)
def _upper_wick(df): return df["high"] - df[["open","close"]].max(axis=1)
def _lower_wick(df): return df[["open","close"]].min(axis=1) - df["low"]

def hanging_man(df):
    body = _body(df); rng = _range(df)
    lw = _lower_wick(df); uw = _upper_wick(df)
    cond = (lw >= 2*body) & (uw <= 0.3*body) & (body/rng < 0.4)
    # In uptrend context: prior 5-bar close < current
    trend = df["close"] > df["close"].shift(5)
    return cond & trend

def inverted_hammer(df):
    body = _body(df); rng = _range(df)
    lw = _lower_wick(df); uw = _upper_wick(df)
    cond = (uw >= 2*body) & (lw <= 0.3*body) & (body/rng < 0.4)
    trend = df["close"] < df["close"].shift(5)
    return cond & trend

def shooting_star(df):
    body = _body(df); rng = _range(df)
    lw = _lower_wick(df); uw = _upper_wick(df)
    cond = (uw >= 2*body) & (lw <= 0.3*body) & (body/rng < 0.4)
    trend = df["close"] > df["close"].shift(5)
    return cond & trend

def three_white_soldiers(df):
    up1 = df["close"] > df["open"]
    up2 = df["close"].shift(1) > df["open"].shift(1)
    up3 = df["close"].shift(2) > df["open"].shift(2)
    higher = (df["close"] > df["close"].shift(1)) & (df["close"].shift(1) > df["close"].shift(2))
    return up1 & up2 & up3 & higher

def three_black_crows(df):
    d1 = df["close"] < df["open"]
    d2 = df["close"].shift(1) < df["open"].shift(1)
    d3 = df["close"].shift(2) < df["open"].shift(2)
    lower = (df["close"] < df["close"].shift(1)) & (df["close"].shift(1) > df["close"].shift(2))
    return d1 & d2 & d3 & lower

def bullish_engulfing(df):
    prev_bear = df["close"].shift(1) < df["open"].shift(1)
    curr_bull = df["close"] > df["open"]
    engulf = (df["open"] < df["close"].shift(1)) & (df["close"] > df["open"].shift(1))
    return prev_bear & curr_bull & engulf

def bearish_engulfing(df):
    prev_bull = df["close"].shift(1) > df["open"].shift(1)
    curr_bear = df["close"] < df["open"]
    engulf = (df["open"] > df["close"].shift(1)) & (df["close"] < df["open"].shift(1))
    return prev_bull & curr_bear & engulf

def tweezer_tops(df, tol=0.001):
    diff = (df["high"] - df["high"].shift(1)).abs() / df["high"]
    prev_bull = df["close"].shift(1) > df["open"].shift(1)
    curr_bear = df["close"] < df["open"]
    return (diff <= tol) & prev_bull & curr_bear

# ============================================================
# BACKTEST ENGINE
# ============================================================
class Trade:
    __slots__ = ["symbol","tf","strategy","direction","entry_time","entry_price",
                 "exit_time","exit_price","size","sl","tp","pnl","pnl_pct","fees","reason"]
    def __init__(self, **k):
        for key, val in k.items(): setattr(self, key, val)

def apply_slippage(price, direction, is_entry):
    if direction == "LONG":
        return price * (1 + SLIPPAGE) if is_entry else price * (1 - SLIPPAGE)
    else:
        return price * (1 - SLIPPAGE) if is_entry else price * (1 + SLIPPAGE)

def run_backtest(df, signals_long, signals_short, sl_long, sl_short, tp_long, tp_short,
                 symbol, tf, strat_name, trail_atr=None, trail_donch=None,
                 exit_signal_long=None, exit_signal_short=None, time_stop_bars=None):
    n = len(df)
    if n < 50:
        return [], None
    o = df["open"].values; h = df["high"].values; l = df["low"].values; c = df["close"].values
    idx = df.index

    sig_l = signals_long.reindex(df.index).fillna(False).values if signals_long is not None else np.zeros(n, dtype=bool)
    sig_s = signals_short.reindex(df.index).fillna(False).values if signals_short is not None else np.zeros(n, dtype=bool)
    sl_l = sl_long.reindex(df.index).values if sl_long is not None else np.full(n, np.nan)
    sl_s = sl_short.reindex(df.index).values if sl_short is not None else np.full(n, np.nan)
    tp_l = tp_long.reindex(df.index).values if tp_long is not None else np.full(n, np.nan)
    tp_s = tp_short.reindex(df.index).values if tp_short is not None else np.full(n, np.nan)
    ex_l = exit_signal_long.reindex(df.index).fillna(False).values if exit_signal_long is not None else np.zeros(n, dtype=bool)
    ex_s = exit_signal_short.reindex(df.index).fillna(False).values if exit_signal_short is not None else np.zeros(n, dtype=bool)
    trail_a = trail_atr.reindex(df.index).values if trail_atr is not None else None
    tr_don_up = trail_donch[0].reindex(df.index).values if trail_donch is not None else None
    tr_don_lo = trail_donch[1].reindex(df.index).values if trail_donch is not None else None

    trades = []
    equity = STARTING_CAPITAL
    equity_curve = [equity]
    equity_times = [idx[0]]

    in_pos = False
    pos = None  # dict

    for i in range(n - 1):
        # Check exit first if in position
        if in_pos:
            hi = h[i]; lo = l[i]; op = o[i]
            hit_sl = False; hit_tp = False; exit_price = None; reason = None
            # Update trailing stop
            if pos["trail_type"] == "atr" and trail_a is not None and not np.isnan(trail_a[i]):
                if pos["direction"] == "LONG":
                    new_sl = c[i-1] - 2.0 * trail_a[i-1] if i > 0 and not np.isnan(trail_a[i-1]) else pos["sl"]
                    if new_sl > pos["sl"]: pos["sl"] = new_sl
                else:
                    new_sl = c[i-1] + 2.0 * trail_a[i-1] if i > 0 and not np.isnan(trail_a[i-1]) else pos["sl"]
                    if new_sl < pos["sl"]: pos["sl"] = new_sl
            elif pos["trail_type"] == "donch" and tr_don_up is not None:
                if pos["direction"] == "LONG" and not np.isnan(tr_don_lo[i-1]):
                    if tr_don_lo[i-1] > pos["sl"]: pos["sl"] = tr_don_lo[i-1]
                elif pos["direction"] == "SHORT" and not np.isnan(tr_don_up[i-1]):
                    if tr_don_up[i-1] < pos["sl"]: pos["sl"] = tr_don_up[i-1]

            # Check gap
            if pos["direction"] == "LONG":
                if op <= pos["sl"]:
                    exit_price = apply_slippage(op, "LONG", False)
                    reason = "SL_GAP"; hit_sl = True
                elif lo <= pos["sl"]:
                    exit_price = apply_slippage(pos["sl"], "LONG", False)
                    reason = "SL"; hit_sl = True
                elif hi >= pos["tp"]:
                    exit_price = apply_slippage(pos["tp"], "LONG", False)
                    reason = "TP"; hit_tp = True
                elif ex_l[i]:
                    exit_price = apply_slippage(c[i], "LONG", False)
                    reason = "SIGNAL"
                elif time_stop_bars is not None and (i - pos["entry_i"]) >= time_stop_bars:
                    exit_price = apply_slippage(c[i], "LONG", False)
                    reason = "TIME"
            else:  # SHORT
                if op >= pos["sl"]:
                    exit_price = apply_slippage(op, "SHORT", False)
                    reason = "SL_GAP"; hit_sl = True
                elif hi >= pos["sl"]:
                    exit_price = apply_slippage(pos["sl"], "SHORT", False)
                    reason = "SL"; hit_sl = True
                elif lo <= pos["tp"]:
                    exit_price = apply_slippage(pos["tp"], "SHORT", False)
                    reason = "TP"; hit_tp = True
                elif ex_s[i]:
                    exit_price = apply_slippage(c[i], "SHORT", False)
                    reason = "SIGNAL"
                elif time_stop_bars is not None and (i - pos["entry_i"]) >= time_stop_bars:
                    exit_price = apply_slippage(c[i], "SHORT", False)
                    reason = "TIME"

            if exit_price is not None:
                size = pos["size"]
                entry_px = pos["entry_price"]
                if pos["direction"] == "LONG":
                    gross = (exit_price - entry_px) * size
                else:
                    gross = (entry_px - exit_price) * size
                fees = (entry_px + exit_price) * size * FEE_RATE
                net = gross - fees
                equity += net
                pnl_pct = net / (entry_px * size) * 100 if entry_px * size > 0 else 0
                trades.append(Trade(
                    symbol=symbol, tf=tf, strategy=strat_name,
                    direction=pos["direction"], entry_time=pos["entry_time"],
                    entry_price=round(entry_px, 8), exit_time=idx[i],
                    exit_price=round(exit_price, 8), size=round(size, 4),
                    sl=round(pos["sl_init"], 8), tp=round(pos["tp_init"], 8),
                    pnl=round(net, 4), pnl_pct=round(pnl_pct, 4),
                    fees=round(fees, 4), reason=reason
                ))
                equity_curve.append(equity)
                equity_times.append(idx[i])
                in_pos = False
                pos = None

        # Check entry (only when flat)
        if not in_pos and i + 1 < n:
            direction = None
            if sig_l[i] and not np.isnan(sl_l[i]) and not np.isnan(tp_l[i]):
                direction = "LONG"
                sl_v = sl_l[i]; tp_v = tp_l[i]
            elif sig_s[i] and not np.isnan(sl_s[i]) and not np.isnan(tp_s[i]):
                direction = "SHORT"
                sl_v = sl_s[i]; tp_v = tp_s[i]
            if direction is not None:
                entry_price_raw = o[i+1]
                if np.isnan(entry_price_raw):
                    continue
                entry_price = apply_slippage(entry_price_raw, direction, True)
                if direction == "LONG":
                    if sl_v >= entry_price or tp_v <= entry_price: continue
                    risk_per_unit = entry_price - sl_v
                else:
                    if sl_v <= entry_price or tp_v >= entry_price: continue
                    risk_per_unit = sl_v - entry_price
                if risk_per_unit <= 0 or np.isnan(risk_per_unit):
                    continue
                risk_dollars = equity * RISK_PER_TRADE
                size = risk_dollars / risk_per_unit
                notional = size * entry_price
                max_notional = equity * LEVERAGE
                if notional > max_notional:
                    size = max_notional / entry_price
                if size <= 0:
                    continue
                trail_type = "none"
                if strat_name in ["S09_EMA_Cross_Long", "S10_EMA_Cross_Short", "S24_ATR_Regime"]:
                    trail_type = "atr"
                elif strat_name == "S12_Donchian_Breakout":
                    trail_type = "donch"
                pos = {
                    "direction": direction, "entry_time": idx[i+1],
                    "entry_price": entry_price, "sl": sl_v, "tp": tp_v,
                    "sl_init": sl_v, "tp_init": tp_v, "size": size,
                    "entry_i": i+1, "trail_type": trail_type
                }
                in_pos = True

    # Close open position at last bar
    if in_pos:
        exit_price = apply_slippage(c[-1], pos["direction"], False)
        size = pos["size"]; entry_px = pos["entry_price"]
        if pos["direction"] == "LONG":
            gross = (exit_price - entry_px) * size
        else:
            gross = (entry_px - exit_price) * size
        fees = (entry_px + exit_price) * size * FEE_RATE
        net = gross - fees
        equity += net
        pnl_pct = net / (entry_px * size) * 100 if entry_px * size > 0 else 0
        trades.append(Trade(
            symbol=symbol, tf=tf, strategy=strat_name,
            direction=pos["direction"], entry_time=pos["entry_time"],
            entry_price=round(entry_px, 8), exit_time=idx[-1],
            exit_price=round(exit_price, 8), size=round(size, 4),
            sl=round(pos["sl_init"], 8), tp=round(pos["tp_init"], 8),
            pnl=round(net, 4), pnl_pct=round(pnl_pct, 4),
            fees=round(fees, 4), reason="EOD"
        ))
        equity_curve.append(equity)
        equity_times.append(idx[-1])

    eq_df = pd.DataFrame({"equity": equity_curve}, index=equity_times)
    return trades, eq_df

# ============================================================
# METRICS
# ============================================================
def compute_metrics(trades, eq_df, tf):
    m = {"trades": len(trades), "total_return_pct": 0.0, "cagr_pct": 0.0,
         "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
         "max_dd_pct": 0.0, "max_dd_bars": 0, "profit_factor": 0.0,
         "win_rate_pct": 0.0, "avg_trade_pct": 0.0, "avg_win_loss": 0.0,
         "expectancy": 0.0, "composite": 0.0}
    if not trades or eq_df is None or len(eq_df) < 2:
        return m
    pnls = np.array([t.pnl for t in trades])
    pcts = np.array([t.pnl_pct for t in trades])
    wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
    m["win_rate_pct"] = 100.0 * len(wins) / len(pnls) if len(pnls) > 0 else 0
    m["avg_trade_pct"] = float(np.mean(pcts))
    gp = wins.sum() if len(wins) > 0 else 0
    gl = abs(losses.sum()) if len(losses) > 0 else 0
    m["profit_factor"] = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    aw = wins.mean() if len(wins) > 0 else 0
    al = abs(losses.mean()) if len(losses) > 0 else 0
    m["avg_win_loss"] = aw / al if al > 0 else 0
    p_win = len(wins) / len(pnls) if len(pnls) > 0 else 0
    m["expectancy"] = p_win * aw - (1 - p_win) * al

    eq = eq_df["equity"].values
    m["total_return_pct"] = (eq[-1] / eq[0] - 1) * 100
    dur_days = (eq_df.index[-1] - eq_df.index[0]).days
    years = dur_days / 365.25 if dur_days > 0 else 1
    if eq[-1] > 0 and eq[0] > 0 and years > 0:
        m["cagr_pct"] = ((eq[-1] / eq[0]) ** (1/years) - 1) * 100
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    m["max_dd_pct"] = float(dd.min() * 100)
    dd_bars = 0; max_bars = 0
    for x in dd:
        if x < 0:
            dd_bars += 1
            if dd_bars > max_bars: max_bars = dd_bars
        else:
            dd_bars = 0
    m["max_dd_bars"] = max_bars
    rets = pd.Series(pcts) / 100.0
    if len(rets) > 1 and rets.std() > 0:
        avg_days = dur_days / len(rets) if len(rets) > 0 else 1
        trades_per_year = 365.25 / max(avg_days, 0.5)
        m["sharpe"] = float(rets.mean() / rets.std() * np.sqrt(trades_per_year))
        neg = rets[rets < 0]
        if len(neg) > 0 and neg.std() > 0:
            m["sortino"] = float(rets.mean() / neg.std() * np.sqrt(trades_per_year))
    if m["max_dd_pct"] < 0:
        m["calmar"] = m["cagr_pct"] / abs(m["max_dd_pct"])
    
    # Composite Score Calculation (0-100)
    sh = np.clip(m["sharpe"] / 3.0, 0, 1) * 30
    ca = np.clip(m["calmar"] / 3.0, 0, 1) * 20
    pf = np.clip((m["profit_factor"] - 1) / 2.0, 0, 1) * 15
    dd_pen = np.clip(1 - abs(m["max_dd_pct"]) / 50.0, 0, 1) * 15
    wr = np.clip(m["win_rate_pct"] / 70.0, 0, 1) * 10
    if m["trades"] < 20: tc = 0
    elif m["trades"] < 50: tc = 5 * (m["trades"] - 20) / 30
    else: tc = 10
    m["composite"] = round(sh + ca + pf + dd_pen + wr + tc, 2)
    for k in ["total_return_pct","cagr_pct","max_dd_pct","win_rate_pct","avg_trade_pct"]:
        m[k] = round(m[k], 2)
    for k in ["sharpe","sortino","calmar","profit_factor","avg_win_loss","expectancy"]:
        m[k] = round(m[k], 4)
    return m

# ============================================================
# STRATEGIES (30 total)
# ============================================================
def _stub(): return None, None, None, None, None, None

def strat_S01(df):
    a = atr(df, 14); adx_ = adx(df, 14)
    pat = hanging_man(df)
    sig_s = pat & (adx_ > 20)
    sl_s = df["high"] + 0.5 * a
    entry_est = df["close"]
    tp_s = entry_est - 1.5 * a
    return None, sig_s, None, sl_s, None, tp_s

def strat_S02(df):
    a = atr(df, 14); adx_ = adx(df, 14)
    pat = inverted_hammer(df)
    sig_l = pat & (adx_ > 20)
    sl_l = df["low"] - 0.5 * a
    entry_est = df["close"]
    tp_l = entry_est + 1.5 * a
    return sig_l, None, sl_l, None, tp_l, None

def strat_S03(df):
    a = atr(df, 14)
    pat = three_white_soldiers(df)
    vol_avg = df["volume"].rolling(20).mean()
    vol_ok = df["volume"] > 1.2 * vol_avg
    sig_l = pat & vol_ok
    lo3 = df["low"].rolling(3).min()
    sl_l = lo3
    tp_l = df["close"] + 2.0 * a
    return sig_l, None, sl_l, None, tp_l, None

def strat_S04(df):
    a = atr(df, 14); r = rsi(df["close"], 14)
    pat = tweezer_tops(df, tol=0.001)
    sig_s = pat & (r > 65)
    sl_s = df["high"] + 0.3 * a
    tp_s = df["close"] - 1.5 * a
    return None, sig_s, None, sl_s, None, tp_s

def strat_S05(df):
    a = atr(df, 14)
    pat = bearish_engulfing(df)
    up_seq = (df["close"] > df["open"]).rolling(3).sum().shift(1) >= 3
    sig_s = pat & up_seq
    sl_s = df["high"]
    tp_s = df["close"] - 1.5 * a
    return None, sig_s, None, sl_s, None, tp_s

def strat_S06(df):
    a = atr(df, 14)
    pat = bullish_engulfing(df)
    dn_seq = (df["close"] < df["open"]).rolling(3).sum().shift(1) >= 3
    sig_l = pat & dn_seq
    sl_l = df["low"]
    tp_l = df["close"] + 1.5 * a
    return sig_l, None, sl_l, None, tp_l, None

def strat_S07(df):
    a = atr(df, 14)
    pat = three_black_crows(df)
    vol_avg = df["volume"].rolling(20).mean()
    vol_ok = df["volume"] > vol_avg
    sig_s = pat & vol_ok
    hi3 = df["high"].rolling(3).max()
    sl_s = hi3
    tp_s = df["close"] - 2.0 * a
    return None, sig_s, None, sl_s, None, tp_s

def strat_S08(df):
    a = atr(df, 14); r = rsi(df["close"], 14)
    pat = shooting_star(df)
    sig_s = pat & (r > 60)
    sl_s = df["high"]
    tp_s = df["close"] - 1.5 * a
    return None, sig_s, None, sl_s, None, tp_s

def strat_S09(df):
    e12 = ema(df["close"], 12); e26 = ema(df["close"], 26); adx_ = adx(df, 14)
    cross_up = (e12 > e26) & (e12.shift(1) <= e26.shift(1))
    sig_l = cross_up & (adx_ > 25)
    swing_low = df["low"].rolling(20).min()
    a = atr(df, 14)
    sl_l = swing_low
    tp_l = df["close"] + 3.0 * a
    return sig_l, None, sl_l, None, tp_l, None

def strat_S10(df):
    e12 = ema(df["close"], 12); e26 = ema(df["close"], 26); adx_ = adx(df, 14)
    cross_dn = (e12 < e26) & (e12.shift(1) >= e26.shift(1))
    sig_s = cross_dn & (adx_ > 25)
    swing_high = df["high"].rolling(20).max()
    a = atr(df, 14)
    sl_s = swing_high
    tp_s = df["close"] - 3.0 * a
    return None, sig_s, None, sl_s, None, tp_s

def strat_S11(df):
    st, tr = supertrend(df, 10, 3.0)
    flip_up = (tr == 1) & (tr.shift(1) == -1)
    flip_dn = (tr == -1) & (tr.shift(1) == 1)
    sig_l = flip_up; sig_s = flip_dn
    sl_l = st; sl_s = st
    risk_l = df["close"] - st
    risk_s = st - df["close"]
    tp_l = df["close"] + 2.0 * risk_l.abs()
    tp_s = df["close"] - 2.0 * risk_s.abs()
    return sig_l, sig_s, sl_l, sl_s, tp_l, tp_s

def strat_S12(df):
    up20, _, lo20 = donchian(df, 20)
    up10, _, lo10 = donchian(df, 10)
    adx_ = adx(df, 14)
    a = atr(df, 14)
    brk_up = (df["close"] > up20.shift(1)) & (adx_ > 20)
    brk_dn = (df["close"] < lo20.shift(1)) & (adx_ > 20)
    sl_l = lo10
    sl_s = up10
    tp_l = df["close"] + 3.0 * a
    tp_s = df["close"] - 3.0 * a
    return brk_up, brk_dn, sl_l, sl_s, tp_l, tp_s

def strat_S13(df):
    h = hull_ma(df["close"], 21)
    slope = h.diff()
    turn_up = (slope > 0) & (slope.shift(1) <= 0)
    turn_dn = (slope < 0) & (slope.shift(1) >= 0)
    a = atr(df, 14)
    sl_l = df["close"] - 2.0 * a
    sl_s = df["close"] + 2.0 * a
    tp_l = df["close"] + 3.0 * a
    tp_s = df["close"] - 3.0 * a
    return turn_up, turn_dn, sl_l, sl_s, tp_l, tp_s

def strat_S14(df):
    p, tr = psar(df)
    e50 = ema(df["close"], 50)
    flip_up = (tr == 1) & (tr.shift(1) == -1) & (df["close"] > e50)
    flip_dn = (tr == -1) & (tr.shift(1) == 1) & (df["close"] < e50)
    sl_l = p; sl_s = p
    risk_l = (df["close"] - p).abs()
    risk_s = (p - df["close"]).abs()
    tp_l = df["close"] + 2.0 * risk_l
    tp_s = df["close"] - 2.0 * risk_s
    return flip_up, flip_dn, sl_l, sl_s, tp_l, tp_s

def strat_S15(df):
    up, mid, lo, _ = bollinger(df["close"], 20, 2.0)
    r = rsi(df["close"], 14); adx_ = adx(df, 14); a = atr(df, 14)
    sig_l = (df["close"] <= lo) & (r < 35) & (adx_ < 25)
    sl_l = lo - 0.5 * a
    tp_l = mid
    return sig_l, None, sl_l, None, tp_l, None

def strat_S16(df):
    up, mid, lo, _ = bollinger(df["close"], 20, 2.0)
    r = rsi(df["close"], 14); adx_ = adx(df, 14); a = atr(df, 14)
    sig_s = (df["close"] >= up) & (r > 65) & (adx_ < 25)
    sl_s = up + 0.5 * a
    tp_s = mid
    return None, sig_s, None, sl_s, None, tp_s

def strat_S17(df):
    r = rsi(df["close"], 14); a = atr(df, 14)
    up, mid, lo, w = bollinger(df["close"], 20, 2.0)
    w_med = w.rolling(50, min_periods=20).median()
    low_vol = w < w_med
    sig_l = (r < 25) & low_vol
    sig_s = (r > 75) & low_vol
    sl_l = df["close"] - 2.0 * a
    sl_s = df["close"] + 2.0 * a
    tp_l = df["close"] + 2.0 * a
    tp_s = df["close"] - 2.0 * a
    return sig_l, sig_s, sl_l, sl_s, tp_l, tp_s

def strat_S18(df):
    z = (df["close"] - df["close"].rolling(50).mean()) / df["close"].rolling(50).std()
    a = atr(df, 14)
    sig_l = z < -2.0
    sig_s = z > 2.0
    sl_l = df["close"] - 3.0 * a
    sl_s = df["close"] + 3.0 * a
    tp_l = df["close"] + 2.0 * a
    tp_s = df["close"] - 2.0 * a
    return sig_l, sig_s, sl_l, sl_s, tp_l, tp_s

def strat_S19(df):
    up, mid, lo = keltner(df, 20, 1.5)
    adx_ = adx(df, 14); a = atr(df, 14)
    sig_l = (df["close"] < lo) & (adx_ < 20)
    sig_s = (df["close"] > up) & (adx_ < 20)
    sl_l = lo - 1.5 * a
    sl_s = up + 1.5 * a
    tp_l = mid
    tp_s = mid
    return sig_l, sig_s, sl_l, sl_s, tp_l, tp_s

def strat_S20(df):
    r = rsi(df["close"], 14)
    _, _, hist = macd(df["close"])
    incr3 = (hist > hist.shift(1)) & (hist.shift(1) > hist.shift(2)) & (hist.shift(2) > hist.shift(3))
    sig_l = (r > 55) & (hist > 0) & incr3
    a = atr(df, 14)
    sl_l = df["low"].rolling(20).min()
    tp_l = df["close"] + 3.0 * a
    return sig_l, None, sl_l, None, tp_l, None

def strat_S21(df):
    r = rsi(df["close"], 14)
    _, _, hist = macd(df["close"])
    decr3 = (hist < hist.shift(1)) & (hist.shift(1) < hist.shift(2)) & (hist.shift(2) < hist.shift(3))
    sig_s = (r < 45) & (hist < 0) & decr3
    a = atr(df, 14)
    sl_s = df["high"].rolling(20).max()
    tp_s = df["close"] - 3.0 * a
    return None, sig_s, None, sl_s, None, tp_s

def strat_S22(df):
    r10 = roc(df["close"], 10)
    a = atr(df, 14)
    vol_avg = df["volume"].rolling(20).mean()
    vol_ok = df["volume"] > 1.5 * vol_avg
    sig_l = (r10 > 5) & vol_ok
    sig_s = (r10 < -5) & vol_ok
    sl_l = df["close"] - 2.0 * a
    sl_s = df["close"] + 2.0 * a
    tp_l = df["close"] + 3.0 * a
    tp_s = df["close"] - 3.0 * a
    return sig_l, sig_s, sl_l, sl_s, tp_l, tp_s

def strat_S23(df):
    k, d = stoch_rsi(df["close"], 14, 3, 3)
    a = atr(df, 14)
    cross_up = (k > d) & (k.shift(1) <= d.shift(1)) & (k.shift(1) < 20)
    cross_dn = (k < d) & (k.shift(1) >= d.shift(1)) & (k.shift(1) > 80)
    sl_l = df["close"] - 1.5 * a
    sl_s = df["close"] + 1.5 * a
    tp_l = df["close"] + 2.5 * a
    tp_s = df["close"] - 2.5 * a
    return cross_up, cross_dn, sl_l, sl_s, tp_l, tp_s

def strat_S24(df):
    a14 = atr(df, 14); a50 = atr(df, 50)
    high_vol = a14 > 1.5 * a50
    e12 = ema(df["close"], 12); e26 = ema(df["close"], 26)
    sig_l = high_vol & (e12 > e26) & (e12.shift(1) <= e26.shift(1))
    sig_s = high_vol & (e12 < e26) & (e12.shift(1) >= e26.shift(1))
    sl_l = df["close"] - 2.0 * a14
    sl_s = df["close"] + 2.0 * a14
    tp_l = df["close"] + 4.0 * a14
    tp_s = df["close"] - 4.0 * a14
    return sig_l, sig_s, sl_l, sl_s, tp_l, tp_s

def strat_S25(df):
    up, mid, lo, w = bollinger(df["close"], 20, 2.0)
    w_min = w.rolling(20).min()
    squeeze = w <= w_min * 1.01
    squeeze_prev = squeeze.shift(1).fillna(False)
    brk_up = squeeze_prev & (df["close"] > up.shift(1))
    brk_dn = squeeze_prev & (df["close"] < lo.shift(1))
    bwidth_px = (up - lo)
    sl_l = mid
    sl_s = mid
    tp_l = df["close"] + 2.0 * bwidth_px
    tp_s = df["close"] - 2.0 * bwidth_px
    return brk_up, brk_dn, sl_l, sl_s, tp_l, tp_s

def strat_S26(df):
    a = atr(df, 14)
    atr_pct = a / df["close"]
    med = atr_pct.rolling(100, min_periods=30).median()
    norm_vol = atr_pct / med.replace(0, np.nan)
    e12 = ema(df["close"], 12); e26 = ema(df["close"], 26)
    sig_l = (e12 > e26) & (e12.shift(1) <= e26.shift(1))
    sig_s = (e12 < e26) & (e12.shift(1) >= e26.shift(1))
    dyn_mult = (2.0 / norm_vol.clip(0.5, 3.0)).fillna(2.0)
    sl_l = df["close"] - dyn_mult * a
    sl_s = df["close"] + dyn_mult * a
    tp_l = df["close"] + 2.0 * dyn_mult * a
    tp_s = df["close"] - 2.0 * dyn_mult * a
    return sig_l, sig_s, sl_l, sl_s, tp_l, tp_s

def strat_S27(df, symbol, tf):
    if tf != "1H":
        return None, None, None, None, None, None
    df_1d = load_data(symbol, "1D")
    df_4h = load_data(symbol, "4H")
    if df_1d is None or df_4h is None:
        return None, None, None, None, None, None
    e50_d = ema(df_1d["close"], 50)
    trend_d = (e50_d - e50_d.shift(3)).reindex(df.index, method="ffill")
    r_4h = rsi(df_4h["close"], 14).reindex(df.index, method="ffill")
    e12 = ema(df["close"], 12); e26 = ema(df["close"], 26)
    cross_up = (e12 > e26) & (e12.shift(1) <= e26.shift(1))
    cross_dn = (e12 < e26) & (e12.shift(1) >= e26.shift(1))
    sig_l = cross_up & (trend_d > 0) & (r_4h > 50)
    sig_s = cross_dn & (trend_d < 0) & (r_4h < 50)
    a = atr(df, 14)
    sl_l = df["close"] - 2.0 * a
    sl_s = df["close"] + 2.0 * a
    tp_l = df["close"] + 3.0 * a
    tp_s = df["close"] - 3.0 * a
    return sig_l, sig_s, sl_l, sl_s, tp_l, tp_s

def strat_S28(df, symbol, tf):
    if tf != "1H":
        return None, None, None, None, None, None
    df_4h = load_data(symbol, "4H")
    if df_4h is None:
        return None, None, None, None, None, None
    r_4h = rsi(df_4h["close"], 14).reindex(df.index, method="ffill")
    r_1h = rsi(df["close"], 14)
    sig_l = (r_4h > 55) & (r_1h > 55) & (r_1h.shift(1) <= 50) & (r_1h > 50)
    sig_s = (r_4h < 45) & (r_1h < 45) & (r_1h.shift(1) >= 50) & (r_1h < 50)
    a = atr(df, 14)
    sl_l = df["close"] - 1.5 * a
    sl_s = df["close"] + 1.5 * a
    tp_l = df["close"] + 2.5 * a
    tp_s = df["close"] - 2.5 * a
    return sig_l, sig_s, sl_l, sl_s, tp_l, tp_s

def strat_S29(df):
    a = atr(df, 14); adx_ = adx(df, 14); r = rsi(df["close"], 14)
    pat_bear = hanging_man(df) | shooting_star(df)
    pat_bull = inverted_hammer(df)
    sig_s = pat_bear & (adx_ > 25) & (r > 65)
    sig_l = pat_bull & (adx_ > 25) & (r < 35)
    sl_l = df["low"] - 0.5 * a
    sl_s = df["high"] + 0.5 * a
    tp_l = df["close"] + 2.0 * a
    tp_s = df["close"] - 2.0 * a
    return sig_l, sig_s, sl_l, sl_s, tp_l, tp_s

def strat_S30(df):
    dow = pd.Series(df.index.dayofweek, index=df.index)
    r = rsi(df["close"], 14)
    _, _, hist = macd(df["close"])
    a = atr(df, 14)
    sig_l = (dow == 4) & (r > 50) & (hist > 0)
    sig_s = (dow == 3) & (r < 50) & (hist < 0)
    sl_l = df["close"] - 1.5 * a
    sl_s = df["close"] + 1.5 * a
    tp_l = df["close"] + 2.5 * a
    tp_s = df["close"] - 2.5 * a
    return sig_l, sig_s, sl_l, sl_s, tp_l, tp_s

STRATEGIES = [
    ("S01_Hanging_Man_Short",    strat_S01, "Pattern"),
    ("S02_Inverted_Hammer_Long", strat_S02, "Pattern"),
    ("S03_Three_White_Soldiers", strat_S03, "Pattern"),
    ("S04_Tweezer_Tops_Short",   strat_S04, "Pattern"),
    ("S05_Bearish_Engulfing",    strat_S05, "Pattern"),
    ("S06_Bullish_Engulfing",    strat_S06, "Pattern"),
    ("S07_Three_Black_Crows",    strat_S07, "Pattern"),
    ("S08_Shooting_Star_Short",  strat_S08, "Pattern"),
    ("S09_EMA_Cross_Long",       strat_S09, "Trend"),
    ("S10_EMA_Cross_Short",      strat_S10, "Trend"),
    ("S11_Supertrend",           strat_S11, "Trend"),
    ("S12_Donchian_Breakout",    strat_S12, "Trend"),
    ("S13_Hull_MA_Trend",        strat_S13, "Trend"),
    ("S14_PSAR_EMA",             strat_S14, "Trend"),
    ("S15_BB_Bounce_Long",       strat_S15, "MeanRev"),
    ("S16_BB_Bounce_Short",      strat_S16, "MeanRev"),
    ("S17_RSI_Extreme",          strat_S17, "MeanRev"),
    ("S18_ZScore_Reversion",     strat_S18, "MeanRev"),
    ("S19_Keltner_MeanRev",      strat_S19, "MeanRev"),
    ("S20_RSI_MACD_Long",        strat_S20, "Momentum"),
    ("S21_RSI_MACD_Short",       strat_S21, "Momentum"),
    ("S22_ROC_Breakout",         strat_S22, "Momentum"),
    ("S23_StochRSI",             strat_S23, "Momentum"),
    ("S24_ATR_Regime",           strat_S24, "Volatility"),
    ("S25_BB_Squeeze",           strat_S25, "Volatility"),
    ("S26_VolAdj_Sizing",        strat_S26, "Volatility"),
    ("S27_Triple_Screen",        strat_S27, "MTF"),
    ("S28_MTF_RSI_Align",        strat_S28, "MTF"),
    ("S29_Confluence",           strat_S29, "Hybrid"),
    ("S30_Seasonality",          strat_S30, "Hybrid"),
]

MTF_STRATS = {"S27_Triple_Screen", "S28_MTF_RSI_Align"}

# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    box("BYBIT BACKTESTING BOT v1.0", C_CYAN)
    print(f"{C_CYAN}  Run date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{C_CYAN}  Data path: {DATA_ROOT}")
    print(f"{C_CYAN}  Symbols  : {len(SYMBOLS)}")
    print(f"{C_CYAN}  Timeframes: {TIMEFRAMES}")
    print(f"{C_CYAN}  Strategies: {len(STRATEGIES)}")
    print(f"{C_CYAN}  IS window : {IS_START.date()} -> {IS_END.date()}")
    print(f"{C_YEL}  [!] DATE GATE ACTIVE: 2026 data is sealed for walk-forward validation{S_RS}")
    print()

    all_results = []
    all_trades = []

    total_iters = len(STRATEGIES) * len(SYMBOLS) * len(TIMEFRAMES)
    counter = 0
    errors = 0
    skipped_data = 0

    for strat_name, strat_fn, category in STRATEGIES:
        section(f"CATEGORY: {category}  |  STRATEGY: {strat_name}", C_MAG)
        cat_results = []
        for tf in TIMEFRAMES:
            for symbol in SYMBOLS:
                counter += 1
                try:
                    df = load_data(symbol, tf)
                    if df is None:
                        skipped_data += 1
                        continue
                    if strat_name in MTF_STRATS:
                        out = strat_fn(df, symbol, tf)
                    else:
                        out = strat_fn(df)
                    if out is None or all(x is None for x in out):
                        continue
                    sig_l, sig_s, sl_l, sl_s, tp_l, tp_s = out
                    
                    trail_atr_series = None; trail_donch = None
                    if strat_name in ["S09_EMA_Cross_Long", "S10_EMA_Cross_Short", "S24_ATR_Regime"]:
                        trail_atr_series = atr(df, 14)
                    elif strat_name == "S12_Donchian_Breakout":
                        up10, _, lo10 = donchian(df, 10)
                        trail_donch = (up10, lo10)

                    time_stop = None
                    if strat_name == "S30_Seasonality":
                        time_stop = 3

                    trades, eq_df = run_backtest(
                        df, sig_l, sig_s, sl_l, sl_s, tp_l, tp_s,
                        symbol, tf, strat_name,
                        trail_atr=trail_atr_series, trail_donch=trail_donch,
                        time_stop_bars=time_stop
                    )
                    metrics = compute_metrics(trades, eq_df, tf)
                    metrics["strategy"] = strat_name
                    metrics["symbol"] = symbol
                    metrics["timeframe"] = tf
                    metrics["category"] = category
                    all_results.append(metrics)
                    all_trades.extend(trades)
                    cat_results.append(metrics)

                    if metrics["trades"] >= 20:
                        mark = f"{C_GRN}[OK]{S_RS}"
                    else:
                        mark = f"{C_YEL}.{S_RS}"
                    tag = f"[{counter:>4}/{total_iters}]"
                    print(f"{tag} {strat_name:<28} | {tf:<3} | {symbol:<22} | "
                          f"Trades: {metrics['trades']:>4} | "
                          f"Sharpe: {metrics['sharpe']:>6.2f} | "
                          f"Comp: {metrics['composite']:>5.1f} {mark}")
                except Exception as e:
                    errors += 1
                    print(f"{C_RED}[ERR] {strat_name} | {tf} | {symbol}: {e}{S_RS}")

        # Category summary
        valid = [r for r in cat_results if r["trades"] >= 20]
        if valid:
            valid_sorted = sorted(valid, key=lambda x: -x["composite"])[:5]
            tbl = [[r["symbol"], r["timeframe"], r["trades"], r["sharpe"],
                    r["profit_factor"], r["win_rate_pct"], r["max_dd_pct"], r["composite"]]
                   for r in valid_sorted]
            print(f"\n{C_CYAN}  Top 5 for {strat_name}:{S_RS}")
            print(tabulate(tbl, headers=["Symbol","TF","Trades","Sharpe","PF","Win%","MaxDD%","Comp"],
                           tablefmt="simple"))

    # ================= MASTER REPORT =================
    section("MASTER RANKING - TOP 50 (composite score)", C_CYAN)
    valid_all = [r for r in all_results if r["trades"] >= 20]
    ranked = sorted(valid_all, key=lambda x: -x["composite"])[:50]
    tbl = []
    for i, r in enumerate(ranked, 1):
        tbl.append([i, r["strategy"], r["symbol"], r["timeframe"], r["trades"],
                    r["sharpe"], r["calmar"], r["profit_factor"],
                    r["max_dd_pct"], r["win_rate_pct"], r["composite"]])
    print(tabulate(tbl,
        headers=["#","Strategy","Symbol","TF","Trades","Sharpe","Calmar","PF","MaxDD%","Win%","Comp"],
        tablefmt="grid"))

    section("TOP 10 DETAILED BREAKDOWN", C_CYAN)
    for i, r in enumerate(ranked[:10], 1):
        print(f"\n{C_YEL}#{i} - {r['strategy']} | {r['symbol']} | {r['timeframe']}{S_RS}")
        detail = [
            ["Total Return %", r["total_return_pct"]],
            ["CAGR %", r["cagr_pct"]],
            ["Sharpe", r["sharpe"]],
            ["Sortino", r["sortino"]],
            ["Calmar", r["calmar"]],
            ["Max DD %", r["max_dd_pct"]],
            ["Max DD Bars", r["max_dd_bars"]],
            ["Profit Factor", r["profit_factor"]],
            ["Win Rate %", r["win_rate_pct"]],
            ["Trades", r["trades"]],
            ["Avg Trade %", r["avg_trade_pct"]],
            ["Avg Win/Loss", r["avg_win_loss"]],
            ["Expectancy", r["expectancy"]],
            ["Composite", r["composite"]],
        ]
        print(tabulate(detail, headers=["Metric","Value"], tablefmt="simple"))

    section("STRATEGY CATEGORY PERFORMANCE", C_CYAN)
    cat_map = {}
    for r in valid_all:
        cat_map.setdefault(r["category"], []).append(r)
    cat_tbl = []
    for cat, rs in cat_map.items():
        avg_sh = np.mean([x["sharpe"] for x in rs])
        avg_cp = np.mean([x["composite"] for x in rs])
        avg_pf = np.mean([x["profit_factor"] for x in rs])
        avg_wr = np.mean([x["win_rate_pct"] for x in rs])
        cat_tbl.append([cat, len(rs), round(avg_sh,3), round(avg_pf,3),
                        round(avg_wr,2), round(avg_cp,2)])
    cat_tbl.sort(key=lambda x: -x[5])
    print(tabulate(cat_tbl, headers=["Category","N","AvgSharpe","AvgPF","AvgWin%","AvgComp"],
                   tablefmt="grid"))

    section("SYMBOL RANKING (avg composite)", C_CYAN)
    sym_map = {}
    for r in valid_all:
        sym_map.setdefault(r["symbol"], []).append(r["composite"])
    sym_tbl = [[s, len(v), round(np.mean(v),2), round(max(v),2)] for s, v in sym_map.items()]
    sym_tbl.sort(key=lambda x: -x[2])
    print(tabulate(sym_tbl[:25], headers=["Symbol","Setups","AvgComp","BestComp"], tablefmt="simple"))

    section("TIMEFRAME COMPARISON", C_CYAN)
    tf_map = {}
    for r in valid_all:
        tf_map.setdefault(r["timeframe"], []).append(r)
    tf_tbl = []
    for tf, rs in tf_map.items():
        tf_tbl.append([tf, len(rs),
                       round(np.mean([x["sharpe"] for x in rs]),3),
                       round(np.mean([x["composite"] for x in rs]),2),
                       round(np.mean([x["win_rate_pct"] for x in rs]),2)])
    print(tabulate(tf_tbl, headers=["TF","N","AvgSharpe","AvgComp","AvgWin%"], tablefmt="simple"))

    section("RISK ANALYSIS", C_RED)
    worst_dd = sorted(valid_all, key=lambda x: x["max_dd_pct"])[:10]
    print(f"{C_RED}Worst 10 Drawdowns:{S_RS}")
    tbl = [[r["strategy"], r["symbol"], r["timeframe"], r["max_dd_pct"], r["max_dd_bars"], r["trades"]]
           for r in worst_dd]
    print(tabulate(tbl, headers=["Strategy","Symbol","TF","MaxDD%","DDBars","Trades"], tablefmt="simple"))

    print(f"\n{C_RED}Longest Consecutive Losing Trades (top strategies):{S_RS}")
    streak_tbl = []
    for r in ranked[:10]:
        tr = [t for t in all_trades if t.strategy == r["strategy"] and t.symbol == r["symbol"] and t.tf == r["timeframe"]]
        cur = 0; mx = 0
        for t in tr:
            if t.pnl < 0:
                cur += 1
                if cur > mx: mx = cur
            else:
                cur = 0
        streak_tbl.append([r["strategy"], r["symbol"], r["timeframe"], mx, len(tr)])
    print(tabulate(streak_tbl, headers=["Strategy","Symbol","TF","MaxLossStreak","TotalTrades"], tablefmt="simple"))

    section("EXECUTIVE SUMMARY", C_GRN)
    total_setups = len(all_results)
    tradeable = len(valid_all)
    positive_sharpe = len([r for r in valid_all if r["sharpe"] > 1.0])
    print(f"  Total setups tested       : {total_setups}")
    print(f"  Setups with >= 20 trades  : {tradeable}")
    print(f"  Setups with Sharpe > 1.0  : {positive_sharpe}")
    print(f"  Errors                    : {errors}")
    print(f"  Skipped (missing data)    : {skipped_data}")
    print()
    print(f"{C_GRN}  >>> TOP 5 SETUPS TO ADVANCE TO OPTIMIZATION / OOS VALIDATION:{S_RS}")
    for i, r in enumerate(ranked[:5], 1):
        verdict = "GO" if r["sharpe"] > 1.0 and r["profit_factor"] > 1.3 and r["trades"] >= 30 else "REVIEW"
        col = C_GRN if verdict == "GO" else C_YEL
        print(f"  {col}{i}. {r['strategy']:<28} {r['symbol']:<22} {r['timeframe']:<3}  "
              f"Sharpe={r['sharpe']:.2f}  PF={r['profit_factor']:.2f}  "
              f"Win={r['win_rate_pct']:.1f}%  DD={r['max_dd_pct']:.1f}%  -> {verdict}{S_RS}")
    print()

    # Save trade log
    trade_log_path = os.path.join(RESULTS_ROOT, "trade_log.csv")
    if all_trades:
        rows = []
        for t in all_trades:
            rows.append({
                "strategy": t.strategy, "symbol": t.symbol, "timeframe": t.tf,
                "direction": t.direction, "entry_time": t.entry_time,
                "exit_time": t.exit_time, "entry_price": t.entry_price,
                "exit_price": t.exit_price, "size": t.size,
                "sl": t.sl, "tp": t.tp, "pnl": t.pnl, "pnl_pct": t.pnl_pct,
                "fees": t.fees, "reason": t.reason
            })
        pd.DataFrame(rows).to_csv(trade_log_path, index=False)
        print(f"{C_GRN}  [OK] Trade log saved: {trade_log_path}  ({len(rows)} trades){S_RS}")

    # Save master ranking CSV
    ranking_path = os.path.join(RESULTS_ROOT, "master_ranking.csv")
    pd.DataFrame(all_results).to_csv(ranking_path, index=False)
    print(f"{C_GRN}  [OK] Master ranking saved: {ranking_path}{S_RS}")

    elapsed = time.time() - t0
    box(f"BACKTEST COMPLETE — Runtime: {elapsed/60:.2f} minutes", C_GRN)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C_RED}Interrupted by user{S_RS}")
    except Exception as e:
        print(f"\n{C_RED}FATAL: {e}{S_RS}")
        traceback.print_exc()
