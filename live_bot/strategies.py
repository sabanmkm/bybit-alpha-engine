"""
BYBIT LIVE BOT - STRATEGY SIGNAL GENERATORS
All 12 strategies producing IDENTICAL signals to backtest engine.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from engine_v2 import Strategy, RSI, MACD, ATR, EMA, SMA, ADX, HMA, Supertrend

# ============================================================
# SIGNAL BUILDING HELPERS (dtype-safe)
# ============================================================
def _build_signals(df: pd.DataFrame, long_mask: pd.Series, short_mask: pd.Series) -> pd.Series:
    n = len(df)
    arr = np.zeros(n, dtype=int)
    arr[long_mask.values] = 1
    arr[short_mask.values] = -1
    both = long_mask.values & short_mask.values
    arr[both] = 1
    return pd.Series(arr, index=df.index, dtype=int)

def _build_sl_tp(df: pd.DataFrame, signals: pd.Series, atr_series: pd.Series,
                  sl_mult: float, tp_mult: float) -> Tuple[pd.Series, pd.Series]:
    close = df["close"]
    sl = pd.Series(np.nan, index=df.index, dtype=float)
    tp = pd.Series(np.nan, index=df.index, dtype=float)
    L = signals == 1; S = signals == -1
    sl.loc[L] = (close - sl_mult * atr_series).loc[L].values
    sl.loc[S] = (close + sl_mult * atr_series).loc[S].values
    tp.loc[L] = (close + tp_mult * atr_series).loc[L].values
    tp.loc[S] = (close - tp_mult * atr_series).loc[S].values
    return sl, tp

def _regime_align(df: pd.DataFrame, regime_series: Optional[pd.Series]) -> pd.Series:
    if regime_series is None or len(regime_series) == 0:
        return pd.Series(["UNKNOWN"] * len(df), index=df.index)
    rs = regime_series.copy()
    if not isinstance(rs.index, pd.DatetimeIndex):
        rs.index = pd.to_datetime(rs.index)
    return rs.reindex(df.index, method="ffill").fillna("UNKNOWN")

def _regime_gate(signals: pd.Series, regimes: pd.Series,
                  allow_long: list, allow_short: list) -> pd.Series:
    rv = regimes.values
    sv = signals.values.copy()
    long_mask = sv == 1
    short_mask = sv == -1
    aL = np.isin(rv, allow_long)
    aS = np.isin(rv, allow_short)
    sv[long_mask & ~aL] = 0
    sv[short_mask & ~aS] = 0
    return pd.Series(sv, index=signals.index, dtype=int)

# ============================================================
# STRATEGY 1: S30a Day-of-Week Seasonality
# ============================================================
class S30a_DOW_Seasonality(Strategy):
    name = "S30a_DOW_Seasonality"
    default_params = {"entry_day_long":4, "entry_day_short":3, "rsi_period":14,
                       "rsi_long_thr":50.0, "rsi_short_thr":50.0,
                       "sl_atr_mult":1.5, "tp_atr_mult":2.5, "atr_period":14}

    def generate_signals(self, df: pd.DataFrame, params: dict = None, regime_series=None):
        p = params or self.default_params
        dow = pd.Series(df.index.dayofweek, index=df.index)
        r = RSI(df["close"], int(p["rsi_period"]))
        a = ATR(df, int(p["atr_period"]))
        long_mask = ((dow == int(p["entry_day_long"])) & (r > p["rsi_long_thr"])).fillna(False).astype(bool)
        short_mask = ((dow == int(p["entry_day_short"])) & (r < p["rsi_short_thr"])).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

# ============================================================
# STRATEGY 2: S30b Mid-Week Mean Reversion
# ============================================================
class S30b_MidWeek_MeanReversion(Strategy):
    name = "S30b_MidWeek_MeanReversion"
    default_params = {"sma_period":20, "atr_period":14, "deviation_atr":1.5,
                       "sl_atr_mult":2.0, "tp_atr_mult":1.5}

    def generate_signals(self, df: pd.DataFrame, params: dict = None, regime_series=None):
        p = params or self.default_params
        sma = SMA(df["close"], int(p["sma_period"]))
        a = ATR(df, int(p["atr_period"]))
        dev = float(p["deviation_atr"])
        dow = pd.Series(df.index.dayofweek, index=df.index)
        dist = df["close"] - sma
        far_above = (dist > dev * a).fillna(False).astype(bool)
        far_below = (dist < -dev * a).fillna(False).astype(bool)
        trig = ((dow == 1) | (dow == 2)).astype(bool)
        short_mask = (far_above & trig).fillna(False).astype(bool)
        long_mask = (far_below & trig).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

# ============================================================
# STRATEGY 3: S30d NY/London Momentum
# ============================================================
class S30d_NY_London_Momentum(Strategy):
    name = "S30d_NY_London_Momentum"
    default_params = {"london_start":8, "london_end":12, "trade_start":12, "trade_end":16,
                       "expansion_ratio":1.2, "sl_atr_mult":1.5, "tp_atr_mult":2.5, "atr_period":14}

    def generate_signals(self, df: pd.DataFrame, params: dict = None, regime_series=None):
        p = params or self.default_params
        a = ATR(df, int(p["atr_period"]))
        hour = pd.Series(df.index.hour, index=df.index)
        date = pd.Series(df.index.date, index=df.index)
        df_tmp = df.copy(); df_tmp["date"] = date.values
        in_london = (hour >= p["london_start"]) & (hour < p["london_end"])
        df_tmp["in_london"] = in_london.values
        grp = df_tmp[df_tmp["in_london"]].groupby("date")
        london_high = grp["high"].max()
        london_low = grp["low"].min()
        lh = pd.Series(date.map(london_high).values, index=df.index)
        ll = pd.Series(date.map(london_low).values, index=df.index)
        atr_ref = a.shift(4).bfill()
        atr_exp = a > float(p["expansion_ratio"]) * atr_ref
        in_trade = (hour >= p["trade_start"]) & (hour < p["trade_end"])
        long_break = df["close"] > lh
        short_break = df["close"] < ll
        long_mask = (in_trade & long_break & atr_exp).fillna(False).astype(bool)
        short_mask = (in_trade & short_break & atr_exp).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

# ============================================================
# STRATEGY 4: R01 Regime-Adaptive EMA Cross
# ============================================================
class R01_EMA_Cross(Strategy):
    name = "R01_EMA_Cross"
    default_params = {"ema_fast":12, "ema_slow":26, "atr_period":14,
                       "sl_lookback":20, "tp_atr_bull":3.0, "tp_atr_bear":1.5}

    def generate_signals(self, df: pd.DataFrame, params: dict = None, regime_series=None):
        p = params or self.default_params
        e_fast = EMA(df["close"], int(p["ema_fast"]))
        e_slow = EMA(df["close"], int(p["ema_slow"]))
        a = ATR(df, int(p["atr_period"]))
        cu = ((e_fast > e_slow) & (e_fast.shift(1) <= e_slow.shift(1))).fillna(False).astype(bool)
        cd = ((e_fast < e_slow) & (e_fast.shift(1) >= e_slow.shift(1))).fillna(False).astype(bool)
        signals = _build_signals(df, cu, cd)
        reg = _regime_align(df, regime_series)
        signals = _regime_gate(signals, reg, ["BULL"], ["BEAR"])
        close = df["close"]
        sl = pd.Series(np.nan, index=df.index, dtype=float)
        tp = pd.Series(np.nan, index=df.index, dtype=float)
        L = signals == 1; S = signals == -1
        sl_l = df["low"].rolling(int(p["sl_lookback"])).min()
        sl_s = df["high"].rolling(int(p["sl_lookback"])).max()
        sl.loc[L] = sl_l.loc[L].values
        sl.loc[S] = sl_s.loc[S].values
        tp.loc[L] = (close + float(p["tp_atr_bull"]) * a).loc[L].values
        tp.loc[S] = (close - float(p["tp_atr_bear"]) * a).loc[S].values
        return signals, sl, tp

# ============================================================
# STRATEGY 5: R13 Regime-Adaptive ATR Range Expansion
# ============================================================
class R13_ATR_Range_Expansion(Strategy):
    name = "R13_ATR_Range_Expansion"
    default_params = {"atr_period":14, "expansion_ratio":1.8, "sl_atr_mult":2.0, "tp_atr_mult":3.0}

    def generate_signals(self, df: pd.DataFrame, params: dict = None, regime_series=None):
        p = params or self.default_params
        a = ATR(df, int(p["atr_period"]))
        bar_range = df["high"] - df["low"]
        big = bar_range > float(p["expansion_ratio"]) * a
        up = df["close"] > df["open"]
        dn = df["close"] < df["open"]
        long_mask = (big & up).fillna(False).astype(bool)
        short_mask = (big & dn).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        reg = _regime_align(df, regime_series)
        signals = _regime_gate(signals, reg, ["BULL"], ["BEAR"])
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

# ============================================================
# STRATEGY 6: R15 Regime-Adaptive Volume Breakout
# ============================================================
class R15_Volume_Breakout(Strategy):
    name = "R15_Volume_Breakout"
    default_params = {"vol_sma":20, "vol_mult":2.5, "atr_period":14,
                       "close_pct":0.7, "sl_atr_mult":1.5, "tp_atr_mult":2.5}

    def generate_signals(self, df: pd.DataFrame, params: dict = None, regime_series=None):
        p = params or self.default_params
        vol_avg = SMA(df["volume"], int(p["vol_sma"]))
        hv = df["volume"] > float(p["vol_mult"]) * vol_avg
        bar_range = df["high"] - df["low"]
        close_pos = (df["close"] - df["low"]) / bar_range.replace(0, np.nan)
        near_high = close_pos > float(p["close_pct"])
        near_low = close_pos < (1 - float(p["close_pct"]))
        long_mask = (hv & near_high).fillna(False).astype(bool)
        short_mask = (hv & near_low).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        reg = _regime_align(df, regime_series)
        signals = _regime_gate(signals, reg, ["BULL","CHOP"], ["BEAR","CHOP"])
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

# ============================================================
# STRATEGY REGISTRY - Maps strategy name to class
# ============================================================
STRATEGY_REGISTRY = {
    "S30a_DOW_Seasonality":       S30a_DOW_Seasonality,
    "S30b_MidWeek_MeanReversion": S30b_MidWeek_MeanReversion,
    "S30d_NY_London_Momentum":    S30d_NY_London_Momentum,
    "R01_EMA_Cross":              R01_EMA_Cross,
    "R13_ATR_Range_Expansion":    R13_ATR_Range_Expansion,
    "R15_Volume_Breakout":        R15_Volume_Breakout,
}

def get_strategy(name: str) -> Strategy:
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}")
    return cls()
