"""
CANONICAL BACKTEST ENGINE v2.1
Identical logic to the engine used in Tasks 6-13.7.
This file is the single source of truth for all indicator calculations.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from abc import ABC, abstractmethod

__version__ = "2.1.0"

# ============================================================
# INDICATORS
# ============================================================
def SMA(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(int(period), min_periods=int(period)).mean()

def EMA(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=int(period), adjust=False, min_periods=int(period)).mean()

def WMA(series: pd.Series, period: int) -> pd.Series:
    period = int(period)
    weights = np.arange(1, period + 1, dtype=float)
    denom = weights.sum()
    return series.rolling(period, min_periods=period).apply(
        lambda x: np.dot(x, weights) / denom, raw=True
    )

def HMA(series: pd.Series, period: int) -> pd.Series:
    period = int(period)
    half = max(int(period / 2), 2)
    sqn = max(int(np.sqrt(period)), 2)
    wma_half = WMA(series, half)
    wma_full = WMA(series, period)
    diff = 2 * wma_half - wma_full
    return WMA(diff, sqn)

def RSI(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI - alpha = 1/period"""
    period = int(period)
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    au = up.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    ad = down.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    rs = au / ad.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def MACD(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ef = EMA(close, fast)
    es = EMA(close, slow)
    macd_line = ef - es
    signal_line = EMA(macd_line.dropna(), signal).reindex(close.index)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def ATR(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR"""
    period = int(period)
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()

def ADX(df: pd.DataFrame, period: int = 14):
    period = int(period)
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0/period, adjust=False, min_periods=period).mean() / atr_s.replace(0, np.nan)
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0/period, adjust=False, min_periods=period).mean() / atr_s.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    return adx.fillna(0), pdi.fillna(0), mdi.fillna(0)

def Supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    period = int(period)
    n = len(df)
    a = ATR(df, period)
    hl2 = ((df["high"] + df["low"]) / 2.0).values
    atr_vals = a.values
    close = df["close"].values
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)
    st = np.full(n, np.nan)
    first_valid = None
    for k in range(n):
        if not np.isnan(atr_vals[k]):
            first_valid = k
            break
    if first_valid is None:
        return pd.Series(st, index=df.index), pd.Series(trend, index=df.index)
    final_upper[first_valid] = hl2[first_valid] + multiplier * atr_vals[first_valid]
    final_lower[first_valid] = hl2[first_valid] - multiplier * atr_vals[first_valid]
    trend[first_valid] = 1
    st[first_valid] = final_lower[first_valid]
    for i in range(first_valid + 1, n):
        basic_upper = hl2[i] + multiplier * atr_vals[i]
        basic_lower = hl2[i] - multiplier * atr_vals[i]
        prev_upper = final_upper[i-1]
        prev_lower = final_lower[i-1]
        prev_close = close[i-1]
        final_upper[i] = basic_upper if (basic_upper < prev_upper or prev_close > prev_upper) else prev_upper
        final_lower[i] = basic_lower if (basic_lower > prev_lower or prev_close < prev_lower) else prev_lower
        if close[i] > final_upper[i-1]:
            trend[i] = 1
        elif close[i] < final_lower[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            if trend[i] == 1 and final_lower[i] < prev_lower:
                final_lower[i] = prev_lower
            if trend[i] == -1 and final_upper[i] > prev_upper:
                final_upper[i] = prev_upper
        st[i] = final_lower[i] if trend[i] == 1 else final_upper[i]
    return pd.Series(st, index=df.index), pd.Series(trend, index=df.index)

def Bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    period = int(period)
    middle = SMA(close, period)
    std = close.rolling(period, min_periods=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    width = (upper - lower) / middle.replace(0, np.nan)
    return upper, middle, lower, width

def Stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth_k: int = 3):
    k_period, d_period, smooth_k = int(k_period), int(d_period), int(smooth_k)
    lo_min = df["low"].rolling(k_period, min_periods=k_period).min()
    hi_max = df["high"].rolling(k_period, min_periods=k_period).max()
    raw_k = 100 * (df["close"] - lo_min) / (hi_max - lo_min).replace(0, np.nan)
    k = raw_k.rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(d_period, min_periods=d_period).mean()
    return k.fillna(50), d.fillna(50)

def VWAP(df: pd.DataFrame, session: str = "daily") -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    if session == "daily":
        day = df.index.floor("D")
        cum_pv = pv.groupby(day).cumsum()
        cum_v = df["volume"].groupby(day).cumsum()
        return cum_pv / cum_v.replace(0, np.nan)
    else:
        cum_pv = pv.rolling(20, min_periods=1).sum()
        cum_v = df["volume"].rolling(20, min_periods=1).sum()
        return cum_pv / cum_v.replace(0, np.nan)

def Keltner(df: pd.DataFrame, period: int = 20, atr_mult: float = 2.0):
    period = int(period)
    middle = EMA(df["close"], period)
    a = ATR(df, period)
    return middle + atr_mult * a, middle, middle - atr_mult * a

def Donchian(df: pd.DataFrame, period: int = 20):
    period = int(period)
    upper = df["high"].rolling(period, min_periods=period).max()
    lower = df["low"].rolling(period, min_periods=period).min()
    middle = (upper + lower) / 2.0
    return upper, middle, lower

def ROC(close: pd.Series, period: int = 10) -> pd.Series:
    period = int(period)
    return 100 * (close - close.shift(period)) / close.shift(period).replace(0, np.nan)

def WilliamsR(df: pd.DataFrame, period: int = 14) -> pd.Series:
    period = int(period)
    hi = df["high"].rolling(period, min_periods=period).max()
    lo = df["low"].rolling(period, min_periods=period).min()
    return -100 * (hi - df["close"]) / (hi - lo).replace(0, np.nan)

def StochRSI(close: pd.Series, rsi_period: int = 14, stoch_period: int = 14, k: int = 3, d: int = 3):
    rsi_period, stoch_period, k, d = int(rsi_period), int(stoch_period), int(k), int(d)
    r = RSI(close, rsi_period)
    lo = r.rolling(stoch_period, min_periods=stoch_period).min()
    hi = r.rolling(stoch_period, min_periods=stoch_period).max()
    stoch = 100 * (r - lo) / (hi - lo).replace(0, np.nan)
    kk = stoch.rolling(k, min_periods=k).mean()
    dd = kk.rolling(d, min_periods=d).mean()
    return kk.fillna(50), dd.fillna(50)

def CCI(df: pd.DataFrame, period: int = 20) -> pd.Series:
    period = int(period)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma_tp = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

def OBV(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff().fillna(0))
    return (direction * df["volume"]).cumsum()

def MFI(df: pd.DataFrame, period: int = 14) -> pd.Series:
    period = int(period)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    mf = tp * df["volume"]
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0.0)
    neg_mf = mf.where(delta < 0, 0.0)
    pos_sum = pos_mf.rolling(period, min_periods=period).sum()
    neg_sum = neg_mf.rolling(period, min_periods=period).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))

# ============================================================
# STRATEGY BASE CLASS
# ============================================================
class Strategy(ABC):
    name: str = "AbstractStrategy"
    default_params: Dict = {}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, params: Dict = None,
                          regime_series: Optional[pd.Series] = None):
        raise NotImplementedError

# ============================================================
# BACKTEST ENGINE (for parity checks)
# ============================================================
class BacktestEngine:
    FEES = 0.00055
    SLIPPAGE = 0.0008
    RISK_PER_TRADE = 0.01
    STARTING_CAPITAL = 10000.0

    def __init__(self, fees=FEES, slippage=SLIPPAGE, risk=RISK_PER_TRADE, capital=STARTING_CAPITAL):
        self.fees = fees; self.slippage = slippage
        self.risk = risk; self.capital = capital

    def _slip(self, price, direction, is_entry):
        if direction == 1:
            return price * (1 + self.slippage) if is_entry else price * (1 - self.slippage)
        else:
            return price * (1 - self.slippage) if is_entry else price * (1 + self.slippage)

    def run(self, df, signals, sl_series, tp_series):
        n = len(df)
        if n < 3:
            return {"trades":[], "equity_curve":pd.Series([self.capital], index=[df.index[0]] if n>0 else []), "metrics":{}}
        o = df["open"].values; h = df["high"].values
        l = df["low"].values; c = df["close"].values
        idx = df.index
        sig = signals.reindex(df.index).fillna(0).astype(int).values
        sl_arr = sl_series.reindex(df.index).astype(float).values
        tp_arr = tp_series.reindex(df.index).astype(float).values
        trades = []
        equity = self.capital
        eq_vals = [equity]; eq_times = [idx[0]]
        in_pos = False; pos = None
        for i in range(n-1):
            if in_pos:
                hi, lo, op = h[i], l[i], o[i]
                exit_px = None; reason = None
                if pos["direction"] == 1:
                    if op <= pos["sl"]: exit_px = self._slip(op, 1, False); reason = "SL_GAP"
                    elif lo <= pos["sl"]: exit_px = self._slip(pos["sl"], 1, False); reason = "SL"
                    elif hi >= pos["tp"]: exit_px = self._slip(pos["tp"], 1, False); reason = "TP"
                else:
                    if op >= pos["sl"]: exit_px = self._slip(op, -1, False); reason = "SL_GAP"
                    elif hi >= pos["sl"]: exit_px = self._slip(pos["sl"], -1, False); reason = "SL"
                    elif lo <= pos["tp"]: exit_px = self._slip(pos["tp"], -1, False); reason = "TP"
                if exit_px is not None:
                    size = pos["size"]; entry_px = pos["entry_price"]
                    gross = (exit_px - entry_px) * size if pos["direction"] == 1 else (entry_px - exit_px) * size
                    fee_cost = (entry_px + exit_px) * size * self.fees
                    net = gross - fee_cost
                    equity += net
                    pnl_pct = (net / (entry_px * size)) * 100 if entry_px * size > 0 else 0
                    trades.append({
                        "entry_time": pos["entry_time"], "exit_time": idx[i],
                        "direction": "LONG" if pos["direction"] == 1 else "SHORT",
                        "entry_price": round(entry_px, 8), "exit_price": round(exit_px, 8),
                        "size": round(size, 6), "sl": round(pos["sl_init"], 8),
                        "tp": round(pos["tp_init"], 8), "pnl": round(net, 4),
                        "pnl_pct": round(pnl_pct, 4), "exit_reason": reason
                    })
                    eq_vals.append(equity); eq_times.append(idx[i])
                    in_pos = False; pos = None
            if not in_pos and i + 1 < n and sig[i] != 0:
                direction = sig[i]
                sv, tv = sl_arr[i], tp_arr[i]
                if np.isnan(sv) or np.isnan(tv): continue
                entry_raw = o[i + 1]
                if np.isnan(entry_raw): continue
                entry_px = self._slip(entry_raw, direction, True)
                if direction == 1:
                    if sv >= entry_px or tv <= entry_px: continue
                    risk_per_unit = entry_px - sv
                else:
                    if sv <= entry_px or tv >= entry_px: continue
                    risk_per_unit = sv - entry_px
                if risk_per_unit <= 0: continue
                size = (equity * self.risk) / risk_per_unit
                if size * entry_px > equity:
                    size = equity / entry_px
                if size <= 0: continue
                pos = {"direction": direction, "entry_time": idx[i+1], "entry_price": entry_px,
                       "sl": sv, "tp": tv, "sl_init": sv, "tp_init": tv, "size": size}
                in_pos = True
        eq_curve = pd.Series(eq_vals, index=eq_times)
        return {"trades": trades, "equity_curve": eq_curve, "metrics": {}}
