"""
TASK 8: REGIME-ADAPTIVE STRATEGY LIBRARY
20 strategies dynamically switching behavior based on BTC regime.
Uses canonical engine_v2 + RegimeDetector.
"""
import os
import sys
import time
import warnings
import traceback
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

warnings.filterwarnings("ignore")

sys.path.insert(0, r"C:\BybitBacktest\backtester")
try:
    from engine_v2 import (
        __version__ as ENGINE_VER,
        MultiTFDataLoader, BacktestEngine, Strategy, RegimeDetector,
        RSI, MACD, ATR, EMA, SMA, ADX, HMA, Supertrend, Bollinger,
        Stochastic, StochRSI, VWAP, Keltner, Donchian, WilliamsR
    )
except ImportError as e:
    print(f"FATAL: engine_v2 import failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
    from colorama import Fore, Style, init as colorama_init
    from tabulate import tabulate
    colorama_init(autoreset=True)
except ImportError as e:
    print(f"Missing package: {e}"); sys.exit(1)

# ============================================================
# CONFIGURATION
# ============================================================
DATA_ROOT = r"C:\BybitBacktest\data\resampled"
RESULTS_ROOT = r"C:\BybitBacktest\results"
os.makedirs(RESULTS_ROOT, exist_ok=True)

IS_START = pd.Timestamp("2022-01-01")
IS_END   = pd.Timestamp("2025-12-31 23:59:59")
OOS_START = pd.Timestamp("2026-01-01")
OOS_END   = pd.Timestamp("2026-08-31 23:59:59")

# Lenient IS: 4 of 5
IS_GATES = {"trades_min":35,"win_rate_min":48.0,"pf_min":1.15,"sharpe_min":0.50,"dd_max":-28.0}
IS_GATES_REQUIRED = 4

# Strict OOS: all
OOS_GATES = {"win_rate_min":50.0,"sharpe_min":0.50,"pf_min":1.15,"trades_min":12,"return_min":0.0}

# Portfolio construction
PORTFOLIO_TARGETS = {
    "min_combined_monthly_trades": 25,
    "min_combined_win_rate": 50.0,
    "max_per_symbol": 3,
    "max_per_strategy": 4,
    "target_portfolio_size": 12,
}

TOP_LIQUID_SYMBOLS = [
    "BTC_USDT_USDT","ETH_USDT_USDT","SOL_USDT_USDT","XRP_USDT_USDT","DOGE_USDT_USDT",
    "ADA_USDT_USDT","AVAX_USDT_USDT","SHIB_USDT_USDT","LINK_USDT_USDT","BNB_USDT_USDT",
    "MATIC_USDT_USDT","LTC_USDT_USDT","DOT_USDT_USDT","BCH_USDT_USDT","ATOM_USDT_USDT",
    "NEAR_USDT_USDT","UNI_USDT_USDT","OP_USDT_USDT","ARB_USDT_USDT","APT_USDT_USDT",
    "FIL_USDT_USDT","INJ_USDT_USDT","TRX_USDT_USDT","ETC_USDT_USDT","XLM_USDT_USDT",
    "SUI_USDT_USDT","SEI_USDT_USDT","TIA_USDT_USDT","RUNE_USDT_USDT","AAVE_USDT_USDT",
    "MKR_USDT_USDT","LDO_USDT_USDT","SAND_USDT_USDT","MANA_USDT_USDT","CRV_USDT_USDT",
    "GALA_USDT_USDT","APE_USDT_USDT","FTM_USDT_USDT","AXS_USDT_USDT","GRT_USDT_USDT"
]

C_CYAN=Fore.CYAN; C_YEL=Fore.YELLOW; C_GRN=Fore.GREEN
C_RED=Fore.RED; C_MAG=Fore.MAGENTA; C_WHT=Fore.WHITE
S_BR=Style.BRIGHT; S_RS=Style.RESET_ALL

def box(txt, color=C_CYAN):
    line = "=" * max(60, len(txt) + 4)
    print(color + S_BR + "+" + line + "+")
    print(color + S_BR + "|  " + txt.ljust(len(line) - 2) + "|")
    print(color + S_BR + "+" + line + "+" + S_RS)

def section(n, txt, color=C_MAG):
    print()
    print(color + S_BR + "-" * 78)
    print(color + S_BR + f"  SECTION {n}: {txt}")
    print(color + S_BR + "-" * 78 + S_RS)

# ============================================================
# REGIME LOOKUP (precomputed once, reused for all strategies)
# ============================================================
_REGIME_CACHE = None

def get_regime_series(start, end, loader):
    """Returns pd.Series indexed by date with regime labels (no lookahead)."""
    global _REGIME_CACHE
    if _REGIME_CACHE is None:
        detector = RegimeDetector(data_loader=loader)
        _REGIME_CACHE = detector.classify_series(start, end)
    return _REGIME_CACHE

def align_regime_to_df(df, regime_series):
    """Aligns daily regime series to intraday df index using forward-fill."""
    if regime_series is None or len(regime_series) == 0:
        return pd.Series(["UNKNOWN"] * len(df), index=df.index)
    # regime_series is indexed by date; reindex to df's intraday index with ffill
    daily_regime = regime_series.copy()
    daily_regime.index = pd.to_datetime(daily_regime.index)
    aligned = daily_regime.reindex(df.index, method="ffill")
    return aligned.fillna("UNKNOWN")

# ============================================================
# SIGNAL/SL/TP HELPERS
# ============================================================
def _build_signals(df, long_mask, short_mask):
    n = len(df)
    arr = np.zeros(n, dtype=int)
    arr[long_mask.values] = 1
    arr[short_mask.values] = -1
    # Prevent conflict: if both long and short flagged, prefer long (arr already set)
    both = long_mask.values & short_mask.values
    arr[both] = 1
    return pd.Series(arr, index=df.index, dtype=int)

def _build_sl_tp(df, signals, atr_series, sl_mult, tp_mult):
    close = df["close"]
    sl = pd.Series(np.nan, index=df.index, dtype=float)
    tp = pd.Series(np.nan, index=df.index, dtype=float)
    lm = signals == 1; sm = signals == -1
    sl.loc[lm] = (close - sl_mult * atr_series).loc[lm].values
    sl.loc[sm] = (close + sl_mult * atr_series).loc[sm].values
    tp.loc[lm] = (close + tp_mult * atr_series).loc[lm].values
    tp.loc[sm] = (close - tp_mult * atr_series).loc[sm].values
    return sl, tp

def _apply_regime_gate(signals, regimes, allowed_regimes_for_long, allowed_regimes_for_short):
    """Zero out signals not permitted by regime."""
    reg_vals = regimes.values
    sig_vals = signals.values.copy()
    long_mask = sig_vals == 1
    short_mask = sig_vals == -1
    allow_long = np.isin(reg_vals, allowed_regimes_for_long)
    allow_short = np.isin(reg_vals, allowed_regimes_for_short)
    sig_vals[long_mask & ~allow_long] = 0
    sig_vals[short_mask & ~allow_short] = 0
    return pd.Series(sig_vals, index=signals.index, dtype=int)

# ============================================================
# BASE REGIME-ADAPTIVE STRATEGY MIXIN
# ============================================================
class RegimeAdaptiveStrategy(Strategy):
    """Base class that injects regime info via `regimes` attribute set externally."""
    regimes = None  # pd.Series aligned to df, set by scanner

    def set_regimes(self, regime_series):
        self.regimes = regime_series

# ============================================================
# FAMILY 1: ADAPTIVE TREND (R01-R05)
# ============================================================
class R01_Adaptive_EMA_Cross(RegimeAdaptiveStrategy):
    name = "R01_Adaptive_EMA_Cross"
    category = "Adaptive_Trend"
    default_params = {"ema_fast":12,"ema_slow":26,"atr_period":14,
                      "sl_lookback":20,"tp_atr_bull":3.0,"tp_atr_bear":1.5}
    def generate_signals(self, df, params):
        p = params
        e_fast = EMA(df["close"], int(p["ema_fast"]))
        e_slow = EMA(df["close"], int(p["ema_slow"]))
        cross_up = ((e_fast > e_slow) & (e_fast.shift(1) <= e_slow.shift(1))).fillna(False).astype(bool)
        cross_dn = ((e_fast < e_slow) & (e_fast.shift(1) >= e_slow.shift(1))).fillna(False).astype(bool)
        signals = _build_signals(df, cross_up, cross_dn)
        # Regime gate: LONG only in BULL, SHORT only in BEAR
        if self.regimes is not None:
            reg = align_regime_to_df(df, self.regimes)
            signals = _apply_regime_gate(signals, reg, ["BULL"], ["BEAR"])
        a = ATR(df, int(p["atr_period"]))
        close = df["close"]
        # BULL longs: SL at 20-bar low, wide TP
        # BEAR shorts: SL at 20-bar high, fast TP
        sl = pd.Series(np.nan, index=df.index, dtype=float)
        tp = pd.Series(np.nan, index=df.index, dtype=float)
        lm = signals == 1; sm = signals == -1
        sl_l_raw = df["low"].rolling(int(p["sl_lookback"])).min()
        sl_s_raw = df["high"].rolling(int(p["sl_lookback"])).max()
        sl.loc[lm] = sl_l_raw.loc[lm].values
        sl.loc[sm] = sl_s_raw.loc[sm].values
        tp.loc[lm] = (close + p["tp_atr_bull"] * a).loc[lm].values
        tp.loc[sm] = (close - p["tp_atr_bear"] * a).loc[sm].values
        return signals, sl, tp

class R02_Adaptive_Supertrend_Dynamic(RegimeAdaptiveStrategy):
    name = "R02_Adaptive_Supertrend_Dynamic"
    category = "Adaptive_Trend"
    default_params = {"st_period":10,"st_mult_bull":3.0,"st_mult_bear":2.5,
                      "atr_period":14,"sl_atr_mult":2.0,"tp_atr_bull":4.0,"tp_atr_bear":2.0}
    def generate_signals(self, df, params):
        p = params
        st_bull, tr_bull = Supertrend(df, int(p["st_period"]), float(p["st_mult_bull"]))
        st_bear, tr_bear = Supertrend(df, int(p["st_period"]), float(p["st_mult_bear"]))
        flip_up = ((tr_bull == 1) & (tr_bull.shift(1) == -1)).fillna(False).astype(bool)
        flip_dn = ((tr_bear == -1) & (tr_bear.shift(1) == 1)).fillna(False).astype(bool)
        signals = _build_signals(df, flip_up, flip_dn)
        if self.regimes is not None:
            reg = align_regime_to_df(df, self.regimes)
            signals = _apply_regime_gate(signals, reg, ["BULL"], ["BEAR"])
        a = ATR(df, int(p["atr_period"]))
        close = df["close"]
        sl = pd.Series(np.nan, index=df.index, dtype=float)
        tp = pd.Series(np.nan, index=df.index, dtype=float)
        lm = signals == 1; sm = signals == -1
        sl.loc[lm] = (close - p["sl_atr_mult"] * a).loc[lm].values
        sl.loc[sm] = (close + p["sl_atr_mult"] * a).loc[sm].values
        tp.loc[lm] = (close + p["tp_atr_bull"] * a).loc[lm].values
        tp.loc[sm] = (close - p["tp_atr_bear"] * a).loc[sm].values
        return signals, sl, tp

class R03_Adaptive_Hull_MA_Slope(RegimeAdaptiveStrategy):
    name = "R03_Adaptive_Hull_MA_Slope"
    category = "Adaptive_Trend"
    default_params = {"hma_period":21,"ema_filter":50,"slope_lookback":2,
                      "atr_period":14,"sl_atr_mult":2.0,"tp_atr_mult":3.0}
    def generate_signals(self, df, params):
        p = params
        h = HMA(df["close"], int(p["hma_period"]))
        e = EMA(df["close"], int(p["ema_filter"]))
        slope = h.diff(int(p["slope_lookback"]))
        turn_up = ((slope > 0) & (slope.shift(1) <= 0) & (df["close"] > e)).fillna(False).astype(bool)
        turn_dn = ((slope < 0) & (slope.shift(1) >= 0) & (df["close"] < e)).fillna(False).astype(bool)
        signals = _build_signals(df, turn_up, turn_dn)
        if self.regimes is not None:
            reg = align_regime_to_df(df, self.regimes)
            signals = _apply_regime_gate(signals, reg, ["BULL"], ["BEAR"])
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class R04_Adaptive_Triple_EMA_Ribbon(RegimeAdaptiveStrategy):
    name = "R04_Adaptive_Triple_EMA_Ribbon"
    category = "Adaptive_Trend"
    default_params = {"ema_short":8,"ema_mid":21,"ema_long":55,
                      "atr_period":14,"sl_atr_mult":2.0,"tp_atr_mult":3.0}
    def generate_signals(self, df, params):
        p = params
        e1 = EMA(df["close"], int(p["ema_short"]))
        e2 = EMA(df["close"], int(p["ema_mid"]))
        e3 = EMA(df["close"], int(p["ema_long"]))
        aligned_up = (e1 > e2) & (e2 > e3)
        aligned_dn = (e1 < e2) & (e2 < e3)
        # Fire on transition into alignment
        long_mask = (aligned_up & (~aligned_up).shift(1).fillna(True)).fillna(False).astype(bool)
        short_mask = (aligned_dn & (~aligned_dn).shift(1).fillna(True)).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        if self.regimes is not None:
            reg = align_regime_to_df(df, self.regimes)
            signals = _apply_regime_gate(signals, reg, ["BULL"], ["BEAR"])
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class R05_Adaptive_ADX_Trend_Strength(RegimeAdaptiveStrategy):
    name = "R05_Adaptive_ADX_Trend_Strength"
    category = "Adaptive_Trend"
    default_params = {"adx_period":14,"adx_threshold":25,"atr_period":14,
                      "sl_atr_mult":2.0,"tp_atr_mult":3.0}
    def generate_signals(self, df, params):
        p = params
        adx_val, pdi, mdi = ADX(df, int(p["adx_period"]))
        strong = adx_val > float(p["adx_threshold"])
        # Fire on new signal (transition)
        pdi_gt = pdi > mdi
        long_new = (strong & pdi_gt & ~(pdi_gt.shift(1).fillna(False))).fillna(False).astype(bool)
        short_new = (strong & (~pdi_gt) & pdi_gt.shift(1).fillna(True)).fillna(False).astype(bool)
        signals = _build_signals(df, long_new, short_new)
        if self.regimes is not None:
            reg = align_regime_to_df(df, self.regimes)
            signals = _apply_regime_gate(signals, reg, ["BULL"], ["BEAR"])
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

# ============================================================
# FAMILY 2: ADAPTIVE MEAN REVERSION (R06-R10)
# ============================================================
class R06_Adaptive_RSI_Reversion(RegimeAdaptiveStrategy):
    name = "R06_Adaptive_RSI_Reversion"
    category = "Adaptive_MeanRev"
    default_params = {"rsi_period":14,"bull_buy":38,"bear_sell":62,
                      "chop_buy":30,"chop_sell":70,"atr_period":14,
                      "sl_atr_mult":1.5,"tp_atr_mult":2.0}
    def generate_signals(self, df, params):
        p = params
        r = RSI(df["close"], int(p["rsi_period"]))
        reg = align_regime_to_df(df, self.regimes) if self.regimes is not None else pd.Series(["UNKNOWN"]*len(df), index=df.index)
        is_bull = reg == "BULL"; is_bear = reg == "BEAR"; is_chop = reg == "CHOP"
        # Bull: buy dips only
        bull_long = (is_bull & (r < p["bull_buy"]) & (r.shift(1) >= p["bull_buy"])).fillna(False).astype(bool)
        # Bear: sell rallies only
        bear_short = (is_bear & (r > p["bear_sell"]) & (r.shift(1) <= p["bear_sell"])).fillna(False).astype(bool)
        # Chop: both boundaries
        chop_long = (is_chop & (r < p["chop_buy"]) & (r.shift(1) >= p["chop_buy"])).fillna(False).astype(bool)
        chop_short = (is_chop & (r > p["chop_sell"]) & (r.shift(1) <= p["chop_sell"])).fillna(False).astype(bool)
        long_mask = (bull_long | chop_long).astype(bool)
        short_mask = (bear_short | chop_short).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class R07_Adaptive_StochRSI_Snapback(RegimeAdaptiveStrategy):
    name = "R07_Adaptive_StochRSI_Snapback"
    category = "Adaptive_MeanRev"
    default_params = {"rsi_period":14,"stoch_period":14,"k":3,"d":3,
                      "os":20,"ob":80,"atr_period":14,
                      "sl_atr_mult":1.5,"tp_atr_mult":1.5}
    def generate_signals(self, df, params):
        p = params
        k, d = StochRSI(df["close"], int(p["rsi_period"]), int(p["stoch_period"]), int(p["k"]), int(p["d"]))
        cross_up = ((k > d) & (k.shift(1) <= d.shift(1)) & (k < p["os"] + 10)).fillna(False).astype(bool)
        cross_dn = ((k < d) & (k.shift(1) >= d.shift(1)) & (k > p["ob"] - 10)).fillna(False).astype(bool)
        reg = align_regime_to_df(df, self.regimes) if self.regimes is not None else pd.Series(["UNKNOWN"]*len(df), index=df.index)
        is_bull = reg == "BULL"; is_bear = reg == "BEAR"; is_chop = reg == "CHOP"
        long_mask = ((cross_up & is_bull) | (cross_up & is_chop)).fillna(False).astype(bool)
        short_mask = ((cross_dn & is_bear) | (cross_dn & is_chop)).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class R08_Adaptive_Bollinger_Band_Fade(RegimeAdaptiveStrategy):
    name = "R08_Adaptive_Bollinger_Band_Fade"
    category = "Adaptive_MeanRev"
    default_params = {"bb_period":20,"bb_std":2.0,"atr_period":14,
                      "sl_atr_mult":1.5,"tp_atr_mult":2.0}
    def generate_signals(self, df, params):
        p = params
        upper, middle, lower, _ = Bollinger(df["close"], int(p["bb_period"]), float(p["bb_std"]))
        reg = align_regime_to_df(df, self.regimes) if self.regimes is not None else pd.Series(["UNKNOWN"]*len(df), index=df.index)
        is_bull = reg == "BULL"; is_bear = reg == "BEAR"; is_chop = reg == "CHOP"
        # BULL: buy dips at middle band (crossing down through)
        touch_mid_down = ((df["close"] < middle) & (df["close"].shift(1) >= middle.shift(1))).fillna(False).astype(bool)
        touch_mid_up = ((df["close"] > middle) & (df["close"].shift(1) <= middle.shift(1))).fillna(False).astype(bool)
        # CHOP: fade outer bands
        touch_upper = ((df["close"] >= upper)).fillna(False).astype(bool)
        touch_lower = ((df["close"] <= lower)).fillna(False).astype(bool)
        bull_long = (is_bull & touch_mid_down).fillna(False).astype(bool)
        bear_short = (is_bear & touch_mid_up).fillna(False).astype(bool)
        chop_long = (is_chop & touch_lower).fillna(False).astype(bool)
        chop_short = (is_chop & touch_upper).fillna(False).astype(bool)
        long_mask = (bull_long | chop_long).astype(bool)
        short_mask = (bear_short | chop_short).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class R09_Adaptive_Keltner_Mean_Reversion(RegimeAdaptiveStrategy):
    name = "R09_Adaptive_Keltner_Mean_Reversion"
    category = "Adaptive_MeanRev"
    default_params = {"kelt_period":20,"kelt_mult":2.0,"atr_period":14,
                      "sl_atr_mult":1.5,"tp_atr_mult":2.0}
    def generate_signals(self, df, params):
        p = params
        upper, middle, lower = Keltner(df, int(p["kelt_period"]), float(p["kelt_mult"]))
        below_lower = ((df["close"] < lower)).fillna(False).astype(bool)
        above_upper = ((df["close"] > upper)).fillna(False).astype(bool)
        reg = align_regime_to_df(df, self.regimes) if self.regimes is not None else pd.Series(["UNKNOWN"]*len(df), index=df.index)
        is_bull = reg == "BULL"; is_bear = reg == "BEAR"; is_chop = reg == "CHOP"
        long_mask = ((is_bull & below_lower) | (is_chop & below_lower)).fillna(False).astype(bool)
        short_mask = ((is_bear & above_upper) | (is_chop & above_upper)).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class R10_Adaptive_WilliamsR_OB_OS(RegimeAdaptiveStrategy):
    name = "R10_Adaptive_WilliamsR_OB_OS"
    category = "Adaptive_MeanRev"
    default_params = {"wr_period":14,"os":-80,"ob":-20,
                      "atr_period":14,"sl_atr_mult":1.5,"tp_atr_mult":2.0}
    def generate_signals(self, df, params):
        p = params
        wr = WilliamsR(df, int(p["wr_period"]))
        os_touch = ((wr < p["os"]) & (wr.shift(1) >= p["os"])).fillna(False).astype(bool)
        ob_touch = ((wr > p["ob"]) & (wr.shift(1) <= p["ob"])).fillna(False).astype(bool)
        reg = align_regime_to_df(df, self.regimes) if self.regimes is not None else pd.Series(["UNKNOWN"]*len(df), index=df.index)
        is_bull = reg == "BULL"; is_bear = reg == "BEAR"; is_chop = reg == "CHOP"
        long_mask = ((is_bull & os_touch) | (is_chop & os_touch)).fillna(False).astype(bool)
        short_mask = ((is_bear & ob_touch) | (is_chop & ob_touch)).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

# ============================================================
# FAMILY 3: ADAPTIVE BREAKOUT (R11-R15)
# ============================================================
class R11_Adaptive_Donchian_Breakout(RegimeAdaptiveStrategy):
    name = "R11_Adaptive_Donchian_Breakout"
    category = "Adaptive_Breakout"
    default_params = {"don_period":20,"atr_period":14,
                      "sl_atr_mult":2.0,"tp_atr_mult":3.0}
    def generate_signals(self, df, params):
        p = params
        upper, middle, lower = Donchian(df, int(p["don_period"]))
        break_up = ((df["close"] > upper.shift(1))).fillna(False).astype(bool)
        break_dn = ((df["close"] < lower.shift(1))).fillna(False).astype(bool)
        reg = align_regime_to_df(df, self.regimes) if self.regimes is not None else pd.Series(["UNKNOWN"]*len(df), index=df.index)
        is_bull = reg == "BULL"; is_bear = reg == "BEAR"; is_chop = reg == "CHOP"
        # BULL: momentum breakouts long; BEAR: momentum breakdowns short; CHOP: fade extremes
        long_mask = ((is_bull & break_up) | (is_chop & break_dn)).fillna(False).astype(bool)
        short_mask = ((is_bear & break_dn) | (is_chop & break_up)).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class R12_Adaptive_Bollinger_Squeeze_Breakout(RegimeAdaptiveStrategy):
    name = "R12_Adaptive_Bollinger_Squeeze_Breakout"
    category = "Adaptive_Breakout"
    default_params = {"bb_period":20,"bb_std":2.0,"squeeze_lookback":20,
                      "atr_period":14,"sl_atr_mult":2.0,"tp_atr_mult":3.0}
    def generate_signals(self, df, params):
        p = params
        upper, middle, lower, width = Bollinger(df["close"], int(p["bb_period"]), float(p["bb_std"]))
        width_min = width.rolling(int(p["squeeze_lookback"])).min()
        squeeze = (width <= width_min * 1.05).fillna(False)
        squeeze_prev = squeeze.shift(1).fillna(False).astype(bool)
        break_up = (squeeze_prev & (df["close"] > upper.shift(1))).fillna(False).astype(bool)
        break_dn = (squeeze_prev & (df["close"] < lower.shift(1))).fillna(False).astype(bool)
        signals = _build_signals(df, break_up, break_dn)
        if self.regimes is not None:
            reg = align_regime_to_df(df, self.regimes)
            signals = _apply_regime_gate(signals, reg, ["BULL"], ["BEAR"])
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class R13_Adaptive_ATR_Range_Expansion(RegimeAdaptiveStrategy):
    name = "R13_Adaptive_ATR_Range_Expansion"
    category = "Adaptive_Breakout"
    default_params = {"atr_period":14,"expansion_ratio":1.8,
                      "sl_atr_mult":2.0,"tp_atr_mult":3.0}
    def generate_signals(self, df, params):
        p = params
        a = ATR(df, int(p["atr_period"]))
        bar_range = df["high"] - df["low"]
        big_range = bar_range > float(p["expansion_ratio"]) * a
        close_gt_open = df["close"] > df["open"]
        close_lt_open = df["close"] < df["open"]
        long_mask = (big_range & close_gt_open).fillna(False).astype(bool)
        short_mask = (big_range & close_lt_open).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        if self.regimes is not None:
            reg = align_regime_to_df(df, self.regimes)
            signals = _apply_regime_gate(signals, reg, ["BULL"], ["BEAR"])
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class R14_Adaptive_Opening_Range_Breakout(RegimeAdaptiveStrategy):
    name = "R14_Adaptive_Opening_Range_Breakout"
    category = "Adaptive_Breakout"
    default_params = {"or_hours":4,"atr_period":14,
                      "sl_atr_mult":1.5,"tp_atr_mult":2.5}
    def generate_signals(self, df, params):
        p = params
        or_hours = int(p["or_hours"])
        hour = pd.Series(df.index.hour, index=df.index)
        date = pd.Series(df.index.date, index=df.index)
        in_or = hour < or_hours
        df_tmp = df.copy()
        df_tmp["date"] = date.values
        df_tmp["in_or"] = in_or.values
        grp = df_tmp[df_tmp["in_or"]].groupby("date")
        or_high = grp["high"].max()
        or_low = grp["low"].min()
        oh = pd.Series(date.map(or_high).values, index=df.index)
        ol = pd.Series(date.map(or_low).values, index=df.index)
        after_or = hour >= or_hours
        break_up = (after_or & (df["close"] > oh)).fillna(False).astype(bool)
        break_dn = (after_or & (df["close"] < ol)).fillna(False).astype(bool)
        reg = align_regime_to_df(df, self.regimes) if self.regimes is not None else pd.Series(["UNKNOWN"]*len(df), index=df.index)
        is_bull = reg == "BULL"; is_bear = reg == "BEAR"; is_chop = reg == "CHOP"
        long_mask = ((is_bull & break_up) | (is_chop & break_dn)).fillna(False).astype(bool)
        short_mask = ((is_bear & break_dn) | (is_chop & break_up)).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class R15_Adaptive_Volume_Breakout(RegimeAdaptiveStrategy):
    name = "R15_Adaptive_Volume_Breakout"
    category = "Adaptive_Breakout"
    default_params = {"vol_sma":20,"vol_mult":2.5,"atr_period":14,
                      "close_pct":0.7,"sl_atr_mult":1.5,"tp_atr_mult":2.5}
    def generate_signals(self, df, params):
        p = params
        vol_avg = SMA(df["volume"], int(p["vol_sma"]))
        high_vol = df["volume"] > float(p["vol_mult"]) * vol_avg
        bar_range = df["high"] - df["low"]
        close_pos = (df["close"] - df["low"]) / bar_range.replace(0, np.nan)
        close_near_high = close_pos > float(p["close_pct"])
        close_near_low = close_pos < (1 - float(p["close_pct"]))
        long_mask = (high_vol & close_near_high).fillna(False).astype(bool)
        short_mask = (high_vol & close_near_low).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        if self.regimes is not None:
            reg = align_regime_to_df(df, self.regimes)
            signals = _apply_regime_gate(signals, reg, ["BULL","CHOP"], ["BEAR","CHOP"])
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

# ============================================================
# FAMILY 4: ADAPTIVE MULTI-FACTOR HYBRIDS (R16-R20)
# ============================================================
class R16_Adaptive_Trend_Plus_RSI_Pullback(RegimeAdaptiveStrategy):
    name = "R16_Adaptive_Trend_Plus_RSI_Pullback"
    category = "Adaptive_Hybrid"
    default_params = {"ema_trend":50,"rsi_period":14,
                      "rsi_bull_low":40,"rsi_bull_high":50,
                      "rsi_bear_low":50,"rsi_bear_high":60,
                      "atr_period":14,"sl_atr_mult":1.5,"tp_atr_mult":2.5}
    def generate_signals(self, df, params):
        p = params
        e = EMA(df["close"], int(p["ema_trend"]))
        e_slope = e.diff(5)
        r = RSI(df["close"], int(p["rsi_period"]))
        bull_pullback = ((e_slope > 0) & (r >= p["rsi_bull_low"]) & (r <= p["rsi_bull_high"])).fillna(False).astype(bool)
        bear_pullback = ((e_slope < 0) & (r >= p["rsi_bear_low"]) & (r <= p["rsi_bear_high"])).fillna(False).astype(bool)
        # Fire on entry into zone
        long_mask = (bull_pullback & ~bull_pullback.shift(1).fillna(True)).fillna(False).astype(bool)
        short_mask = (bear_pullback & ~bear_pullback.shift(1).fillna(True)).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        if self.regimes is not None:
            reg = align_regime_to_df(df, self.regimes)
            signals = _apply_regime_gate(signals, reg, ["BULL"], ["BEAR"])
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class R17_Adaptive_MACD_Histogram_Acceleration(RegimeAdaptiveStrategy):
    name = "R17_Adaptive_MACD_Histogram_Acceleration"
    category = "Adaptive_Hybrid"
    default_params = {"macd_fast":12,"macd_slow":26,"macd_signal":9,
                      "atr_period":14,"sl_atr_mult":1.5,"tp_atr_mult":2.5}
    def generate_signals(self, df, params):
        p = params
        macd_line, sig_line, hist = MACD(df["close"], int(p["macd_fast"]), int(p["macd_slow"]), int(p["macd_signal"]))
        hist_up_accel = (hist > 0) & (hist > hist.shift(1)) & (hist.shift(1) > hist.shift(2))
        hist_dn_accel = (hist < 0) & (hist < hist.shift(1)) & (hist.shift(1) < hist.shift(2))
        # For CHOP: zero-line reversion
        cross_up_zero = (hist > 0) & (hist.shift(1) <= 0)
        cross_dn_zero = (hist < 0) & (hist.shift(1) >= 0)
        reg = align_regime_to_df(df, self.regimes) if self.regimes is not None else pd.Series(["UNKNOWN"]*len(df), index=df.index)
        is_bull = reg == "BULL"; is_bear = reg == "BEAR"; is_chop = reg == "CHOP"
        long_mask = ((is_bull & hist_up_accel) | (is_chop & cross_up_zero)).fillna(False).astype(bool)
        short_mask = ((is_bear & hist_dn_accel) | (is_chop & cross_dn_zero)).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class R18_Adaptive_VWAP_Deviation(RegimeAdaptiveStrategy):
    name = "R18_Adaptive_VWAP_Deviation"
    category = "Adaptive_Hybrid"
    default_params = {"atr_period":14,"sl_atr_mult":1.5,"tp_atr_mult":2.0,
                      "vwap_std_mult":2.0}
    def generate_signals(self, df, params):
        p = params
        try:
            vwap = VWAP(df, session="daily")
        except Exception:
            vwap = SMA(df["close"], 20)
        # Rough std bands using rolling std of close over 20
        std20 = df["close"].rolling(20).std()
        upper_dev = vwap + p["vwap_std_mult"] * std20
        lower_dev = vwap - p["vwap_std_mult"] * std20
        # BULL: buy pullbacks to VWAP
        pullback_to_vwap = ((df["close"] <= vwap) & (df["close"].shift(1) > vwap.shift(1))).fillna(False).astype(bool)
        rally_to_vwap = ((df["close"] >= vwap) & (df["close"].shift(1) < vwap.shift(1))).fillna(False).astype(bool)
        # CHOP: fade 2-std bands
        touch_upper = ((df["close"] >= upper_dev)).fillna(False).astype(bool)
        touch_lower = ((df["close"] <= lower_dev)).fillna(False).astype(bool)
        reg = align_regime_to_df(df, self.regimes) if self.regimes is not None else pd.Series(["UNKNOWN"]*len(df), index=df.index)
        is_bull = reg == "BULL"; is_bear = reg == "BEAR"; is_chop = reg == "CHOP"
        long_mask = ((is_bull & pullback_to_vwap) | (is_chop & touch_lower)).fillna(False).astype(bool)
        short_mask = ((is_bear & rally_to_vwap) | (is_chop & touch_upper)).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class R19_Adaptive_Triple_Oscillator_Confluence(RegimeAdaptiveStrategy):
    name = "R19_Adaptive_Triple_Oscillator_Confluence"
    category = "Adaptive_Hybrid"
    default_params = {"rsi_period":14,"stoch_rsi_period":14,"stoch_k":3,"stoch_d":3,
                      "macd_fast":12,"macd_slow":26,"macd_signal":9,
                      "atr_period":14,"sl_atr_mult":1.5,"tp_atr_mult":2.5}
    def generate_signals(self, df, params):
        p = params
        r = RSI(df["close"], int(p["rsi_period"]))
        k, d_ = StochRSI(df["close"], int(p["rsi_period"]), int(p["stoch_rsi_period"]), int(p["stoch_k"]), int(p["stoch_d"]))
        _, _, hist = MACD(df["close"], int(p["macd_fast"]), int(p["macd_slow"]), int(p["macd_signal"]))
        long_cond = (r > 50) & (k > 50) & (hist > 0)
        short_cond = (r < 50) & (k < 50) & (hist < 0)
        # Fire on transition
        long_mask = (long_cond & ~long_cond.shift(1).fillna(True)).fillna(False).astype(bool)
        short_mask = (short_cond & ~short_cond.shift(1).fillna(True)).fillna(False).astype(bool)
        signals = _build_signals(df, long_mask, short_mask)
        if self.regimes is not None:
            reg = align_regime_to_df(df, self.regimes)
            signals = _apply_regime_gate(signals, reg, ["BULL"], ["BEAR"])
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

class R20_Adaptive_Supertrend_Plus_Volume_Confirm(RegimeAdaptiveStrategy):
    name = "R20_Adaptive_Supertrend_Plus_Volume_Confirm"
    category = "Adaptive_Hybrid"
    default_params = {"st_period":10,"st_mult":3.0,"vol_sma":20,"vol_mult":1.3,
                      "atr_period":14,"sl_atr_mult":2.0,"tp_atr_mult":3.0}
    def generate_signals(self, df, params):
        p = params
        st, tr = Supertrend(df, int(p["st_period"]), float(p["st_mult"]))
        vol_avg = SMA(df["volume"], int(p["vol_sma"]))
        vol_confirm = df["volume"] > float(p["vol_mult"]) * vol_avg
        flip_up = ((tr == 1) & (tr.shift(1) == -1) & vol_confirm).fillna(False).astype(bool)
        flip_dn = ((tr == -1) & (tr.shift(1) == 1) & vol_confirm).fillna(False).astype(bool)
        signals = _build_signals(df, flip_up, flip_dn)
        if self.regimes is not None:
            reg = align_regime_to_df(df, self.regimes)
            signals = _apply_regime_gate(signals, reg, ["BULL"], ["BEAR"])
        a = ATR(df, int(p["atr_period"]))
        sl, tp = _build_sl_tp(df, signals, a, float(p["sl_atr_mult"]), float(p["tp_atr_mult"]))
        return signals, sl, tp

STRATEGY_REGISTRY = [
    R01_Adaptive_EMA_Cross, R02_Adaptive_Supertrend_Dynamic, R03_Adaptive_Hull_MA_Slope,
    R04_Adaptive_Triple_EMA_Ribbon, R05_Adaptive_ADX_Trend_Strength,
    R06_Adaptive_RSI_Reversion, R07_Adaptive_StochRSI_Snapback, R08_Adaptive_Bollinger_Band_Fade,
    R09_Adaptive_Keltner_Mean_Reversion, R10_Adaptive_WilliamsR_OB_OS,
    R11_Adaptive_Donchian_Breakout, R12_Adaptive_Bollinger_Squeeze_Breakout,
    R13_Adaptive_ATR_Range_Expansion, R14_Adaptive_Opening_Range_Breakout,
    R15_Adaptive_Volume_Breakout,
    R16_Adaptive_Trend_Plus_RSI_Pullback, R17_Adaptive_MACD_Histogram_Acceleration,
    R18_Adaptive_VWAP_Deviation, R19_Adaptive_Triple_Oscillator_Confluence,
    R20_Adaptive_Supertrend_Plus_Volume_Confirm,
]

TIMEFRAMES = ["15m", "30m", "1H", "4H"]

# ============================================================
# HELPERS
# ============================================================
def monthly_freq(trades, days):
    if not trades or days <= 0: return 0.0
    return round(len(trades) / max(days / 30.44, 1), 3)

def month_range_2026(m):
    if m == 12:
        s = pd.Timestamp(f"2026-{m:02d}-01")
        e = pd.Timestamp("2027-01-01") - pd.Timedelta(seconds=1)
    else:
        s = pd.Timestamp(f"2026-{m:02d}-01")
        e = pd.Timestamp(f"2026-{m+1:02d}-01") - pd.Timedelta(seconds=1)
    return s, e

def passes_is_lenient(row):
    c = 0
    if row["total_trades"] >= IS_GATES["trades_min"]: c += 1
    if row["win_rate_pct"] >= IS_GATES["win_rate_min"]: c += 1
    if row["profit_factor"] >= IS_GATES["pf_min"]: c += 1
    if row["sharpe"] >= IS_GATES["sharpe_min"]: c += 1
    if row["max_drawdown_pct"] > IS_GATES["dd_max"]: c += 1
    return c

def passes_oos_strict(m):
    return (
        m["win_rate_pct"] >= OOS_GATES["win_rate_min"] and
        m["sharpe"] >= OOS_GATES["sharpe_min"] and
        m["profit_factor"] >= OOS_GATES["pf_min"] and
        m["total_trades"] >= OOS_GATES["trades_min"] and
        m["total_return_pct"] > OOS_GATES["return_min"]
    )

# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    box("TASK 8: REGIME-ADAPTIVE STRATEGY LIBRARY", C_CYAN)
    print(f"{C_CYAN}  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{C_CYAN}  Engine: v{ENGINE_VER}")
    print(f"{C_CYAN}  Strategies: {len(STRATEGY_REGISTRY)}")
    print(f"{C_CYAN}  Timeframes: {TIMEFRAMES}")

    loader = MultiTFDataLoader(DATA_ROOT)
    engine = BacktestEngine()

    # Precompute regime series once
    section(1, "PRECOMPUTING BTC REGIME CLASSIFICATION", C_CYAN)
    print(f"  Loading BTC 1D and classifying every day (no lookahead)...")
    try:
        regime_series = get_regime_series("2022-01-01", "2026-08-31", loader)
        print(f"  {C_GRN}[OK] Regime series length: {len(regime_series)} days{S_RS}")
        reg_counts = regime_series.value_counts()
        for reg, cnt in reg_counts.items():
            print(f"    {reg}: {cnt} days ({100*cnt/len(regime_series):.1f}%)")
    except Exception as e:
        print(f"  {C_RED}[FAIL] Regime detection: {e}{S_RS}")
        return

    # Symbol universe
    section(2, "SYMBOL UNIVERSE (TOP 40 LIQUID)", C_CYAN)
    all_syms_1h = set(loader.available_symbols("1H"))
    universe = [s for s in TOP_LIQUID_SYMBOLS if s in all_syms_1h][:40]
    if len(universe) < 20:
        extra = [s for s in all_syms_1h if s not in universe][:40 - len(universe)]
        universe.extend(extra)
    print(f"  Universe size: {len(universe)}")
    print(f"  Symbols: {', '.join(universe[:20])}{'...' if len(universe)>20 else ''}")

    total_combos = 0
    for strat_cls in STRATEGY_REGISTRY:
        for tf in TIMEFRAMES:
            avail = [s for s in universe if s in loader.available_symbols(tf)]
            total_combos += len(avail)
    print(f"  Total (strategy x symbol x TF) combos: {total_combos}")

    # STAGE 1: IS screening
    section(3, "STAGE 1: IN-SAMPLE SCREENING (2022-2025)", C_MAG)
    is_results = []
    counter = 0
    err_count = 0
    skip_count = 0

    for strat_cls in STRATEGY_REGISTRY:
        strat_inst = strat_cls(engine=engine)
        strat_inst.set_regimes(regime_series)
        for tf in TIMEFRAMES:
            avail = [s for s in universe if s in loader.available_symbols(tf)]
            print(f"\n  {C_CYAN}{strat_cls.name[:36]:<36} | {tf} | {len(avail)} symbols{S_RS}")
            for sym in avail:
                counter += 1
                try:
                    df = loader.load(sym, tf, IS_START, IS_END)
                    if df is None or len(df) < 300:
                        skip_count += 1; continue
                    res = strat_inst.backtest(df, strat_cls.default_params)
                    m = res["metrics"]
                    days = (df.index.max() - df.index.min()).days
                    m["monthly_freq"] = monthly_freq(res["trades"], days)
                    is_results.append({
                        "strategy": strat_cls.name, "category": strat_cls.category,
                        "symbol": sym, "tf": tf, **m
                    })
                    if counter % 250 == 0:
                        elap = (time.time()-t0)/60
                        eta = (elap/counter)*(total_combos-counter) if counter > 0 else 0
                        n_qual = sum(1 for r in is_results if passes_is_lenient(r) >= IS_GATES_REQUIRED)
                        print(f"    Progress: {counter}/{total_combos} | Elapsed {elap:.1f}m | ETA {eta:.1f}m | Qualified {n_qual}")
                except Exception as e:
                    err_count += 1
                    if err_count <= 3:
                        print(f"    {C_RED}[ERR] {strat_cls.name}|{sym}|{tf}: {e}{S_RS}")

    print(f"\n  {C_GRN}[OK] IS scan complete: {len(is_results)} setups, {err_count} errors, {skip_count} skipped{S_RS}")
    is_df = pd.DataFrame(is_results)
    all_csv = os.path.join(RESULTS_ROOT, "task8_all_is_results.csv")
    is_df.to_csv(all_csv, index=False)
    print(f"  {C_GRN}[SAVED] {all_csv}{S_RS}")

    if len(is_df) == 0:
        print(f"  {C_RED}[!] No IS results.{S_RS}"); return

    # IS gates
    section(4, "IS GATE FILTERING (4 of 5 must pass)", C_MAG)
    is_df["gates_passed"] = is_df.apply(passes_is_lenient, axis=1)
    qualified = is_df[is_df["gates_passed"] >= IS_GATES_REQUIRED].copy()
    qualified = qualified.sort_values(["sharpe","monthly_freq"], ascending=[False,False]).reset_index(drop=True)
    print(f"  Total: {len(is_df)} -> Qualified: {C_GRN}{len(qualified)}{S_RS}")
    gate_dist = is_df["gates_passed"].value_counts().sort_index()
    print(f"  Gate-pass distribution:")
    for gp, cnt in gate_dist.items():
        marker = f"{C_GRN}<-- QUALIFIED{S_RS}" if gp >= IS_GATES_REQUIRED else ""
        print(f"    {int(gp)}/5 gates: {cnt} setups {marker}")

    if len(qualified) == 0:
        print(f"  {C_RED}[!] Zero setups met IS gates.{S_RS}"); return

    # STAGE 2: OOS validation
    section(5, "STAGE 2: OOS 2026 VALIDATION (STRICT GATES)", C_MAG)
    strat_map = {c.name: c for c in STRATEGY_REGISTRY}
    oos_verified = []
    oos_all = []
    oos_tested = 0

    print(f"  Testing {len(qualified)} qualified setups on 2026 OOS...")
    for i, row in qualified.iterrows():
        oos_tested += 1
        sname = row["strategy"]; sym = row["symbol"]; tf = row["tf"]
        cls = strat_map.get(sname)
        if cls is None: continue
        try:
            strat_inst = cls(engine=engine)
            strat_inst.set_regimes(regime_series)
            df_oos = loader.load(sym, tf, OOS_START, OOS_END)
            if df_oos is None or len(df_oos) < 50: continue
            res = strat_inst.backtest(df_oos, cls.default_params)
            m = res["metrics"]
            days = (df_oos.index.max() - df_oos.index.min()).days
            m["monthly_freq"] = monthly_freq(res["trades"], days)
            entry = {
                "strategy": sname, "category": cls.category,
                "symbol": sym, "tf": tf,
                "is_sharpe": round(row["sharpe"], 3),
                "is_win_rate": round(row["win_rate_pct"], 2),
                "is_pf": round(row["profit_factor"], 3),
                "is_trades": int(row["total_trades"]),
                "is_return": round(row["total_return_pct"], 2),
                "is_monthly_freq": round(row["monthly_freq"], 3),
                "is_gates_passed": int(row["gates_passed"]),
                "oos_sharpe": round(m["sharpe"], 3),
                "oos_win_rate": round(m["win_rate_pct"], 2),
                "oos_pf": round(m["profit_factor"], 3),
                "oos_trades": int(m["total_trades"]),
                "oos_return": round(m["total_return_pct"], 2),
                "oos_max_dd": round(m["max_drawdown_pct"], 2),
                "oos_monthly_freq": round(m["monthly_freq"], 3),
                "oos_expectancy": round(m["expectancy"], 4),
                "trades_ref": res["trades"],
            }
            oos_all.append(entry)
            if passes_oos_strict(m):
                deg = m["sharpe"]/row["sharpe"] if row["sharpe"]>0 else 0
                entry["degradation"] = round(deg, 3)
                entry["status"] = "STRONG_PASS" if deg >= 0.7 else "PASS"
                entry["freq_score"] = round(m["monthly_freq"]*10 + m["sharpe"]*20, 2)
                oos_verified.append(entry)
        except Exception as e:
            if oos_tested <= 3:
                print(f"    {C_RED}[ERR OOS] {sname}|{sym}|{tf}: {e}{S_RS}")

    oos_verified.sort(key=lambda x: -x["freq_score"])
    print(f"  Tested: {oos_tested} | {C_GRN}Passed: {len(oos_verified)}{S_RS}")

    # SECTION 6: Funnel
    section(6, "EXECUTION FUNNEL SUMMARY", C_CYAN)
    funnel = [
        ["Total combos scanned", total_combos],
        ["IS backtests completed", len(is_results)],
        ["Passed IS gates (>=4/5)", len(qualified)],
        ["Tested on OOS 2026", oos_tested],
        ["Passed OOS gates", len(oos_verified)],
        ["Overall yield", f"{100*len(oos_verified)/max(total_combos,1):.2f}%"],
    ]
    print(tabulate(funnel, headers=["Stage","Count"], tablefmt="grid"))

    if len(oos_verified) == 0:
        print(f"  {C_RED}[!] No OOS-verified edges.{S_RS}")
        if oos_all:
            fallback_csv = os.path.join(RESULTS_ROOT, "task8_all_oos_tested.csv")
            pd.DataFrame([{k:v for k,v in r.items() if k != "trades_ref"} for r in oos_all]).to_csv(fallback_csv, index=False)
            print(f"  {C_GRN}[SAVED] All OOS (unqualified) -> {fallback_csv}{S_RS}")
        return

    # Save all OOS verified
    oos_csv = os.path.join(RESULTS_ROOT, "task8_oos_verified_adaptive.csv")
    pd.DataFrame([{k:v for k,v in r.items() if k != "trades_ref"} for r in oos_verified]).to_csv(oos_csv, index=False)
    print(f"  {C_GRN}[SAVED] {oos_csv}{S_RS}")

    # SECTION 7: Top 30
    section(7, "TOP 30 OOS-VERIFIED ADAPTIVE EDGES", C_GRN)
    top_rows = []
    for i, r in enumerate(oos_verified[:30], 1):
        st_col = C_GRN if r["status"] == "STRONG_PASS" else C_YEL
        top_rows.append([
            i, r["strategy"][:26], r["symbol"][:12], r["tf"],
            f"{r['is_sharpe']:.2f}", f"{r['oos_sharpe']:.2f}",
            f"{r['oos_win_rate']:.1f}%", f"{r['oos_pf']:.2f}",
            r["oos_trades"], f"{r['oos_return']:.2f}%",
            f"{r['oos_monthly_freq']:.1f}", f"{r['freq_score']:.1f}",
            f"{st_col}{r['status']}{S_RS}"
        ])
    print(tabulate(top_rows,
        headers=["#","Strategy","Symbol","TF","IS Sh","OOS Sh","Win%","PF","Tr","Ret","MoFr","FrScr","Status"],
        tablefmt="grid"))

    # SECTION 8: Regime performance diagnostic
    section(8, "REGIME PERFORMANCE DIAGNOSTIC (2026 OOS)", C_CYAN)
    reg_stats = {"BULL":{"pnl":0,"trades":0,"wins":0},
                 "BEAR":{"pnl":0,"trades":0,"wins":0},
                 "CHOP":{"pnl":0,"trades":0,"wins":0},
                 "CRISIS":{"pnl":0,"trades":0,"wins":0},
                 "UNKNOWN":{"pnl":0,"trades":0,"wins":0}}
    # Precompute daily regime lookup for 2026
    regime_2026 = regime_series[(regime_series.index >= OOS_START.date()) &
                                  (regime_series.index <= OOS_END.date())]
    regime_dict = {}
    for dt, reg in regime_2026.items():
        regime_dict[pd.Timestamp(dt).date()] = reg

    for r in oos_verified:
        for t in r["trades_ref"]:
            entry_date = t["entry_time"].date() if hasattr(t["entry_time"], "date") else pd.Timestamp(t["entry_time"]).date()
            reg = regime_dict.get(entry_date, "UNKNOWN")
            reg_stats[reg]["pnl"] += t["pnl"]
            reg_stats[reg]["trades"] += 1
            if t["pnl"] > 0: reg_stats[reg]["wins"] += 1

    reg_rows = []
    for reg in ["BULL","BEAR","CHOP","CRISIS","UNKNOWN"]:
        s = reg_stats[reg]
        wr = 100*s["wins"]/s["trades"] if s["trades"]>0 else 0
        avg = s["pnl"]/s["trades"] if s["trades"]>0 else 0
        col = C_GRN if s["pnl"]>0 else (C_RED if s["pnl"]<0 else C_YEL)
        reg_rows.append([
            reg, s["trades"], f"{wr:.1f}%",
            f"{col}${s['pnl']:.2f}{S_RS}", f"${avg:.4f}"
        ])
    print(tabulate(reg_rows, headers=["Regime","Trades","Win%","Total PnL","Avg PnL/Trade"], tablefmt="grid"))

    # SECTION 9: Portfolio construction
    section(9, "FINAL ADAPTIVE PORTFOLIO CONSTRUCTION", C_MAG)
    portfolio = []
    combined_monthly = 0.0
    sym_count = {}; strat_count = {}
    for r in oos_verified:
        if sym_count.get(r["symbol"], 0) >= PORTFOLIO_TARGETS["max_per_symbol"]: continue
        if strat_count.get(r["strategy"], 0) >= PORTFOLIO_TARGETS["max_per_strategy"]: continue
        portfolio.append(r)
        combined_monthly += r["oos_monthly_freq"]
        sym_count[r["symbol"]] = sym_count.get(r["symbol"], 0) + 1
        strat_count[r["strategy"]] = strat_count.get(r["strategy"], 0) + 1
        if len(portfolio) >= PORTFOLIO_TARGETS["target_portfolio_size"] and combined_monthly >= PORTFOLIO_TARGETS["min_combined_monthly_trades"]:
            break

    if not portfolio:
        print(f"  {C_RED}[!] Empty portfolio.{S_RS}"); return

    print(f"  Portfolio size          : {len(portfolio)}")
    print(f"  Combined monthly trades : {combined_monthly:.1f}")
    print(f"  Unique symbols          : {len(sym_count)}")
    print(f"  Unique strategies       : {len(strat_count)}")

    prt_rows = []
    for r in portfolio:
        prt_rows.append([
            r["strategy"][:26], r["symbol"][:12], r["tf"],
            f"{r['oos_sharpe']:.2f}", f"{r['oos_win_rate']:.1f}%",
            f"{r['oos_pf']:.2f}", r["oos_trades"], f"{r['oos_return']:.2f}%",
            f"{r['oos_monthly_freq']:.1f}", f"{r['freq_score']:.1f}"
        ])
    print("\n  Portfolio Members:")
    print(tabulate(prt_rows,
        headers=["Strategy","Symbol","TF","OOS Sh","Win%","PF","Tr","Ret","MoFr","FrScr"],
        tablefmt="simple"))

    # Combined stats
    all_trades = []
    for r in portfolio: all_trades.extend(r["trades_ref"])

    if all_trades:
        pnls = np.array([t["pnl"] for t in all_trades])
        wins = (pnls > 0).sum()
        pf_wr = 100*wins/len(pnls) if len(pnls)>0 else 0
        gp = pnls[pnls>0].sum(); gl = abs(pnls[pnls<0].sum())
        pf_pf = gp/gl if gl>0 else 999
        pf_aw = pnls[pnls>0].mean() if wins>0 else 0
        pf_al = abs(pnls[pnls<0].mean()) if (pnls<0).sum()>0 else 0
        pf_expect = (wins/len(pnls))*pf_aw - (1-wins/len(pnls))*pf_al

        trades_time = sorted(all_trades, key=lambda t: t["exit_time"])
        eq_v = [10000.0]; eq_t = [OOS_START]
        eq = 10000.0
        for t in trades_time:
            eq += t["pnl"]; eq_v.append(eq); eq_t.append(t["exit_time"])
        eq_s = pd.Series(eq_v, index=eq_t)
        eq_d = eq_s.resample("D").last().ffill()
        dr = eq_d.pct_change().dropna()
        port_sh = float(dr.mean()/dr.std()*np.sqrt(365)) if dr.std()>0 else 0
        peak = eq_d.cummax(); dd = (eq_d - peak)/peak
        port_dd = float(dd.min()*100)
        port_ret = (eq_s.iloc[-1]/eq_s.iloc[0] - 1) * 100

        print(f"\n  {C_GRN}{S_BR}Portfolio 2026 OOS Combined Metrics:{S_RS}")
        pstats = [
            ["Combined Trades", len(all_trades)],
            ["Combined Win Rate %", f"{pf_wr:.2f}"],
            ["Combined Profit Factor", f"{pf_pf:.3f}"],
            ["Combined Sharpe (daily)", f"{port_sh:.3f}"],
            ["Combined Max DD %", f"{port_dd:.2f}"],
            ["Combined Total Return %", f"{port_ret:.2f}"],
            ["Combined Expectancy $", f"{pf_expect:.4f}"],
            ["Combined Monthly Trades", f"{combined_monthly:.1f}"],
        ]
        print(tabulate(pstats, headers=["Metric","Value"], tablefmt="grid"))

        # SECTION 10: 2026 month-by-month heatmap
        section(10, "2026 MONTH-BY-MONTH ADAPTIVE PORTFOLIO HEATMAP", C_CYAN)
        mo_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug"]
        rows = []
        profitable_mo = 0
        for mi in range(1, 9):
            ms, me = month_range_2026(mi)
            m_trades = [t for t in all_trades if ms <= t["exit_time"] <= me]
            if m_trades:
                m_pnl = sum(t["pnl"] for t in m_trades)
                m_wins = sum(1 for t in m_trades if t["pnl"]>0)
                m_wr = 100*m_wins/len(m_trades)
                m_ret = m_pnl / 10000 * 100
                col = C_GRN if m_ret>0 else C_RED
                if m_ret > 0: profitable_mo += 1
            else:
                m_ret = 0; m_wr = 0; col = C_YEL
            rows.append([
                mo_names[mi-1],
                f"{col}{m_ret:+.2f}%{S_RS}",
                len(m_trades),
                f"{m_wr:.1f}%" if m_trades else "-"
            ])
        print(tabulate(rows, headers=["Month","Return","Trades","Win%"], tablefmt="grid"))
        print(f"\n  Profitable months: {profitable_mo}/8 ({100*profitable_mo/8:.1f}%)")

        # Save portfolio
        pf_csv = os.path.join(RESULTS_ROOT, "task8_adaptive_portfolio.csv")
        total_score = sum(r["freq_score"] for r in portfolio)
        rows = []
        for r in portfolio:
            w = r["freq_score"]/total_score*100 if total_score>0 else 0
            row = {k:v for k,v in r.items() if k != "trades_ref"}
            row["allocation_weight_pct"] = round(w, 3)
            rows.append(row)
        pd.DataFrame(rows).to_csv(pf_csv, index=False)
        print(f"\n  {C_GRN}[SAVED] {pf_csv}{S_RS}")

    # SECTION 11: Executive verdict
    section(11, "EXECUTIVE SUMMARY", C_GRN)
    all_targets_met = (
        pf_wr >= PORTFOLIO_TARGETS["min_combined_win_rate"] and
        port_sh >= 1.0 and
        combined_monthly >= PORTFOLIO_TARGETS["min_combined_monthly_trades"] and
        port_ret > 0 and pf_pf >= 1.15
    ) if all_trades else False
    if all_targets_met:
        verdict = f"{C_GRN}{S_BR}GO - Adaptive portfolio meets ALL targets. Deploy alongside Task 7.2 portfolio.{S_RS}"
    else:
        verdict = f"{C_YEL}CONDITIONAL - Some targets missed. Review top-scored subset only.{S_RS}"
    print(f"\n  Verdict: {verdict}\n")

    print(f"  {C_CYAN}Regime-adaptive edges will complement Task 7.2 time-based portfolio:{S_RS}")
    print(f"  - Time-based edges (Task 7.2) exploit calendar/session anomalies (regime-independent)")
    print(f"  - Adaptive edges (Task 8) dynamically follow trends and fade extremes based on BTC regime")
    print(f"  - Combined portfolio provides diversification across BOTH structural and directional alpha sources")

    elapsed = time.time() - t0
    print()
    box(f"TASK 8 COMPLETE - Runtime: {elapsed/60:.2f} minutes", C_GRN)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C_RED}Interrupted{S_RS}")
    except Exception as e:
        print(f"\n{C_RED}FATAL: {e}{S_RS}")
        traceback.print_exc()
