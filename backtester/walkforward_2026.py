# ============================================================
# WALK-FORWARD 2026 OOS VALIDATION - Full Python Engine
# ============================================================
import os, sys, json, time, warnings, traceback
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

warnings.filterwarnings("ignore")

# Dependency check
missing = []
try: import pandas as pd
except ImportError: missing.append("pandas")
try: import numpy as np
except ImportError: missing.append("numpy")
try: from scipy import stats
except ImportError: missing.append("scipy")
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except ImportError: missing.append("colorama")
try: from tabulate import tabulate
except ImportError: missing.append("tabulate")
if missing:
    print("Missing packages: " + " ".join(missing))
    sys.exit(1)

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = r"C:\BybitBacktest\data\resampled"
RESULTS_PATH = r"C:\BybitBacktest\results"
os.makedirs(RESULTS_PATH, exist_ok=True)

IS_START = pd.Timestamp("2022-01-01")
IS_END = pd.Timestamp("2025-12-31 23:59:59")
OOS_START = pd.Timestamp("2026-01-01")
OOS_END = pd.Timestamp("2026-08-31 23:59:59")
FULL_END = pd.Timestamp("2026-12-31 23:59:59")

STARTING_CAPITAL = 10000.0
RISK_PER_TRADE = 0.01
FEES = 0.00055
SLIPPAGE = 0.0003

# Colors
C_CYAN=Fore.CYAN; C_YEL=Fore.YELLOW; C_GRN=Fore.GREEN; C_RED=Fore.RED
C_MAG=Fore.MAGENTA; C_WHT=Fore.WHITE; S_BR=Style.BRIGHT; S_RS=Style.RESET_ALL

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

def color_num(v, hi_good=True, pos_col=C_GRN, neg_col=C_RED):
    if v is None or (isinstance(v,float) and np.isnan(v)): return f"{C_YEL}N/A{S_RS}"
    if v > 0: return f"{pos_col if hi_good else neg_col}+{v:.2f}{S_RS}"
    if v < 0: return f"{neg_col if hi_good else pos_col}{v:.2f}{S_RS}"
    return f"{v:.2f}"

# ============================================================
# THE 10 PORTFOLIO EDGES + FROZEN PARAMETERS + KNOWN IS METRICS
# ============================================================
PORTFOLIO = [
    {
        "idx":1, "strategy":"S30_Seasonality", "symbol":"TWT_USDT_USDT", "tf":"4H",
        "source":"ORIGINAL",
        "params":{"entry_day_long":4,"entry_day_short":3,"rsi_period":14,
                  "rsi_long_threshold":50,"rsi_short_threshold":50,
                  "macd_fast":12,"macd_slow":26,"macd_signal":9,
                  "hold_bars":2,"sl_atr_mult":1.5,"tp_atr_mult":2.5,"atr_period":14},
        "known_is":{"sharpe":1.56,"cagr":10.86,"max_dd":-4.22,"trades":278}
    },
    {
        "idx":2, "strategy":"S13_Hull_MA_Trend", "symbol":"1000LUNC_USDT_USDT", "tf":"1D",
        "source":"ORIGINAL",
        "params":{"hma_period":21,"slope_lookback":2,"sl_atr_mult":2.0,
                  "tp_atr_mult":3.0,"atr_period":14,"trend_filter_ema":0,
                  "min_slope_threshold":0.0},
        "known_is":{"sharpe":0.95,"cagr":3.68,"max_dd":-2.01,"trades":31}
    },
    {
        "idx":3, "strategy":"S13_Hull_MA_Trend", "symbol":"ZEN_USDT_USDT", "tf":"1D",
        "source":"ORIGINAL",
        "params":{"hma_period":21,"slope_lookback":2,"sl_atr_mult":2.0,
                  "tp_atr_mult":3.0,"atr_period":14,"trend_filter_ema":0,
                  "min_slope_threshold":0.0},
        "known_is":{"sharpe":1.05,"cagr":5.08,"max_dd":-3.00,"trades":51}
    },
    {
        "idx":4, "strategy":"S28_MTF_RSI_Align", "symbol":"MANA_USDT_USDT", "tf":"1H",
        "source":"TUNED_OPTUNA",
        "params":{"rsi_period":7,"rsi_higher_tf_threshold":54.61,"rsi_entry_threshold":50.44,
                  "higher_tf":"1D","sl_atr_mult":2.13,"tp_atr_mult":3.55,"atr_period":13,
                  "rsi_exit_overbought":71.80,"rsi_exit_oversold":20.02,"cooldown_bars":2},
        "known_is":{"sharpe":2.76,"cagr":73.39,"max_dd":-9.29,"trades":696},
        "original_params":{"rsi_period":14,"rsi_higher_tf_threshold":55,"rsi_entry_threshold":50,
                           "higher_tf":"4H","sl_atr_mult":1.5,"tp_atr_mult":2.5,"atr_period":14,
                           "rsi_exit_overbought":75,"rsi_exit_oversold":25,"cooldown_bars":0}
    },
    {
        "idx":5, "strategy":"S13_Hull_MA_Trend", "symbol":"CHZ_USDT_USDT", "tf":"1D",
        "source":"ORIGINAL",
        "params":{"hma_period":21,"slope_lookback":2,"sl_atr_mult":2.0,
                  "tp_atr_mult":3.0,"atr_period":14,"trend_filter_ema":0,
                  "min_slope_threshold":0.0},
        "known_is":{"sharpe":0.99,"cagr":4.89,"max_dd":-3.52,"trades":50}
    },
    {
        "idx":6, "strategy":"S28_MTF_RSI_Align", "symbol":"AVAX_USDT_USDT", "tf":"1H",
        "source":"TUNED_STABILITY",
        "params":{"rsi_period":14,"rsi_higher_tf_threshold":56.0,"rsi_entry_threshold":50,
                  "higher_tf":"4H","sl_atr_mult":1.6,"tp_atr_mult":2.5,"atr_period":14,
                  "rsi_exit_overbought":75,"rsi_exit_oversold":25,"cooldown_bars":0},
        "known_is":{"sharpe":1.49,"cagr":15.86,"max_dd":-4.14,"trades":163},
        "original_params":{"rsi_period":14,"rsi_higher_tf_threshold":55,"rsi_entry_threshold":50,
                           "higher_tf":"4H","sl_atr_mult":1.5,"tp_atr_mult":2.5,"atr_period":14,
                           "rsi_exit_overbought":75,"rsi_exit_oversold":25,"cooldown_bars":0}
    },
    {
        "idx":7, "strategy":"S11_Supertrend", "symbol":"ANKR_USDT_USDT", "tf":"4H",
        "source":"ORIGINAL",
        "params":{"st_period":10,"st_multiplier":3.0,"sl_atr_mult":2.0,"tp_atr_mult":3.0,
                  "atr_period":14,"trend_filter_ema":0,"use_trailing_stop":False,
                  "trailing_atr_mult":2.0},
        "known_is":{"sharpe":1.41,"cagr":8.23,"max_dd":-9.93,"trades":77}
    },
    {
        "idx":8, "strategy":"S28_MTF_RSI_Align", "symbol":"ZRX_USDT_USDT", "tf":"1H",
        "source":"TUNED_OPTUNA",
        "params":{"rsi_period":8,"rsi_higher_tf_threshold":54.68,"rsi_entry_threshold":46.64,
                  "higher_tf":"1D","sl_atr_mult":2.11,"tp_atr_mult":3.67,"atr_period":13,
                  "rsi_exit_overbought":78.82,"rsi_exit_oversold":28.69,"cooldown_bars":0},
        "known_is":{"sharpe":2.63,"cagr":43.04,"max_dd":-6.13,"trades":324},
        "original_params":{"rsi_period":14,"rsi_higher_tf_threshold":55,"rsi_entry_threshold":50,
                           "higher_tf":"4H","sl_atr_mult":1.5,"tp_atr_mult":2.5,"atr_period":14,
                           "rsi_exit_overbought":75,"rsi_exit_oversold":25,"cooldown_bars":0}
    },
    {
        "idx":9, "strategy":"S11_Supertrend", "symbol":"AVAX_USDT_USDT", "tf":"4H",
        "source":"TUNED_WALKFORWARD",
        "params":{"st_period":12,"st_multiplier":3.57,"sl_atr_mult":1.68,"tp_atr_mult":4.17,
                  "atr_period":14,"trend_filter_ema":100,"use_trailing_stop":False,
                  "trailing_atr_mult":2.70},
        "known_is":{"sharpe":1.31,"cagr":11.82,"max_dd":-8.09,"trades":93},
        "original_params":{"st_period":10,"st_multiplier":3.0,"sl_atr_mult":2.0,"tp_atr_mult":3.0,
                           "atr_period":14,"trend_filter_ema":0,"use_trailing_stop":False,
                           "trailing_atr_mult":2.0}
    },
    {
        "idx":10, "strategy":"S02_Inverted_Hammer_Long", "symbol":"CHZ_USDT_USDT", "tf":"4H",
        "source":"ORIGINAL",
        "params":{"body_ratio_max":0.33,"upper_wick_ratio_min":2.0,"lower_wick_ratio_max":0.5,
                  "adx_period":14,"adx_threshold":20,"sl_atr_mult":1.0,"tp_atr_mult":1.5,
                  "atr_period":14,"volume_confirm_mult":1.0},
        "known_is":{"sharpe":0.94,"cagr":4.94,"max_dd":-6.28,"trades":30}
    },
]

# ============================================================
# DATE GATE (now allows both IS and OOS periods)
# ============================================================
def gate_data(df, mode="ALL"):
    if df is None or len(df) == 0: return df
    if mode == "IS":
        df = df[(df.index >= IS_START) & (df.index <= IS_END)].copy()
    elif mode == "OOS":
        df = df[(df.index >= OOS_START) & (df.index <= OOS_END)].copy()
    else:  # ALL (up through OOS_END)
        df = df[(df.index >= IS_START) & (df.index <= OOS_END)].copy()
    return df

# ============================================================
# DATA LOADER (caches full extent through OOS_END)
# ============================================================
_cache = {}
def load_data(symbol, tf):
    key = (symbol, tf)
    if key in _cache: return _cache[key]
    path = Path(DATA_PATH) / tf / f"{symbol}.parquet"
    if not path.exists():
        _cache[key] = None; return None
    try:
        df = pd.read_parquet(path)
        ts_col = None
        for c in ["timestamp","datetime","time","date"]:
            if c in df.columns: ts_col = c; break
        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            df = df.set_index(ts_col)
        else:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
        cols = {c.lower(): c for c in df.columns}
        rmap = {}
        for w in ["open","high","low","close","volume"]:
            if w in cols: rmap[cols[w]] = w
        df = df.rename(columns=rmap)
        for n in ["open","high","low","close","volume"]:
            if n not in df.columns: _cache[key]=None; return None
        df = df[["open","high","low","close","volume"]].astype(float)
        df = df[~df.index.duplicated(keep="first")].sort_index().dropna()
        df = gate_data(df, mode="ALL")
        if len(df) < 100: _cache[key]=None; return None
        _cache[key] = df; return df
    except Exception:
        _cache[key] = None; return None

def slice_period(df, start, end):
    if df is None: return None
    return df[(df.index >= start) & (df.index <= end)].copy()

# ============================================================
# INDICATORS
# ============================================================
def ema(s,n): return s.ewm(span=n, adjust=False, min_periods=n).mean()
def sma(s,n): return s.rolling(n, min_periods=n).mean()

def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0.0); dn = -delta.clip(upper=0.0)
    ru = up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rd = dn.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = ru / rd.replace(0, np.nan)
    return (100 - (100/(1+rs))).fillna(50)

def atr(df, n=14):
    h,l,c = df["high"],df["low"],df["close"]; pc = c.shift(1)
    tr = pd.concat([h-l,(h-pc).abs(),(l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def adx(df, n=14):
    h,l,c = df["high"],df["low"],df["close"]
    up = h.diff(); dn = -l.diff()
    pdm = np.where((up>dn)&(up>0), up, 0.0)
    mdm = np.where((dn>up)&(dn>0), dn, 0.0)
    pc = c.shift(1)
    tr = pd.concat([h-l,(h-pc).abs(),(l-pc).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    pdi = 100*pd.Series(pdm, index=df.index).ewm(alpha=1/n, adjust=False, min_periods=n).mean()/a.replace(0,np.nan)
    mdi = 100*pd.Series(mdm, index=df.index).ewm(alpha=1/n, adjust=False, min_periods=n).mean()/a.replace(0,np.nan)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean().fillna(0)

def macd(close, fast=12, slow=26, sig=9):
    ef = ema(close, fast); es = ema(close, slow)
    line = ef - es; signal = ema(line, sig); hist = line - signal
    return line, signal, hist

def hull_ma(close, n=21):
    n = max(int(n), 4)
    half = max(int(n/2), 2); sqn = max(int(np.sqrt(n)), 2)
    def wma(x, length):
        return x.rolling(length).apply(
            lambda a: np.dot(a, np.arange(1,length+1))/(length*(length+1)/2), raw=True)
    wh = wma(close, half); wf = wma(close, n)
    diff = 2*wh - wf
    return wma(diff, sqn)

def supertrend(df, period=10, mult=3.0):
    a = atr(df, period)
    hl2 = (df["high"]+df["low"])/2
    upper = hl2 + mult*a; lower = hl2 - mult*a
    close = df["close"].values
    up = upper.values.copy(); lo = lower.values.copy()
    trend = np.ones(len(df))
    for i in range(1, len(df)):
        if close[i] > up[i-1]: trend[i] = 1
        elif close[i] < lo[i-1]: trend[i] = -1
        else:
            trend[i] = trend[i-1]
            if trend[i]==1 and lo[i]<lo[i-1]: lo[i] = lo[i-1]
            if trend[i]==-1 and up[i]>up[i-1]: up[i] = up[i-1]
    st = np.where(trend==1, lo, up)
    return pd.Series(st, index=df.index), pd.Series(trend, index=df.index)

# ============================================================
# STRATEGY IMPLEMENTATIONS (identical to Task 4 optimizer)
# ============================================================
def strat_S30(df, p):
    dow = pd.Series(df.index.dayofweek, index=df.index)
    r = rsi(df["close"], int(p.get("rsi_period",14)))
    _,_,h = macd(df["close"], int(p.get("macd_fast",12)),
                 int(p.get("macd_slow",26)), int(p.get("macd_signal",9)))
    a = atr(df, int(p.get("atr_period",14)))
    sig_l = (dow == int(p.get("entry_day_long",4))) & (r > p.get("rsi_long_threshold",50)) & (h > 0)
    sig_s = (dow == int(p.get("entry_day_short",3))) & (r < p.get("rsi_short_threshold",50)) & (h < 0)
    sl_l = df["close"] - p.get("sl_atr_mult",1.5)*a
    sl_s = df["close"] + p.get("sl_atr_mult",1.5)*a
    tp_l = df["close"] + p.get("tp_atr_mult",2.5)*a
    tp_s = df["close"] - p.get("tp_atr_mult",2.5)*a
    return sig_l, sig_s, sl_l, sl_s, tp_l, tp_s

def strat_S13(df, p):
    hma = hull_ma(df["close"], int(p.get("hma_period",21)))
    lb = int(p.get("slope_lookback",2))
    slope = hma.diff(lb); sp = slope.shift(1)
    thr = p.get("min_slope_threshold", 0.0)
    turn_up = (slope > thr) & (sp <= thr)
    turn_dn = (slope < -thr) & (sp >= -thr)
    tf_ema = int(p.get("trend_filter_ema", 0))
    if tf_ema > 0:
        e = ema(df["close"], tf_ema)
        turn_up = turn_up & (df["close"] > e)
        turn_dn = turn_dn & (df["close"] < e)
    a = atr(df, int(p.get("atr_period",14)))
    sl_l = df["close"] - p.get("sl_atr_mult",2.0)*a
    sl_s = df["close"] + p.get("sl_atr_mult",2.0)*a
    tp_l = df["close"] + p.get("tp_atr_mult",3.0)*a
    tp_s = df["close"] - p.get("tp_atr_mult",3.0)*a
    return turn_up, turn_dn, sl_l, sl_s, tp_l, tp_s

def strat_S28(df, symbol, p, period_mode="ALL"):
    higher_tf = p.get("higher_tf", "4H")
    df_h_full = load_data(symbol, higher_tf)
    if df_h_full is None:
        return None, None, None, None, None, None
    # Slice higher TF to match same period as df
    df_h = slice_period(df_h_full, df.index.min(), df.index.max())
    if df_h is None or len(df_h) < 20:
        return None, None, None, None, None, None
    n = int(p.get("rsi_period",14))
    # Compute higher-TF RSI, then shift by 1 to avoid lookahead
    r_h_raw = rsi(df_h["close"], n)
    r_h_shifted = r_h_raw.shift(1)  # only use previously-closed higher-TF bar
    r_h = r_h_shifted.reindex(df.index, method="ffill")
    r_1 = rsi(df["close"], n)
    ht_thr = p.get("rsi_higher_tf_threshold", 55)
    en_thr = p.get("rsi_entry_threshold", 50)
    sig_l = (r_h > ht_thr) & (r_1 > ht_thr) & (r_1.shift(1) <= en_thr) & (r_1 > en_thr)
    sig_s = (r_h < (100-ht_thr)) & (r_1 < (100-ht_thr)) & (r_1.shift(1) >= (100-en_thr)) & (r_1 < (100-en_thr))
    a = atr(df, int(p.get("atr_period",14)))
    sl_l = df["close"] - p.get("sl_atr_mult",1.5)*a
    sl_s = df["close"] + p.get("sl_atr_mult",1.5)*a
    tp_l = df["close"] + p.get("tp_atr_mult",2.5)*a
    tp_s = df["close"] - p.get("tp_atr_mult",2.5)*a
    return sig_l, sig_s, sl_l, sl_s, tp_l, tp_s

def strat_S11(df, p):
    st, tr = supertrend(df, int(p.get("st_period",10)), p.get("st_multiplier",3.0))
    flip_up = (tr==1) & (tr.shift(1)==-1)
    flip_dn = (tr==-1) & (tr.shift(1)==1)
    tf_ema = int(p.get("trend_filter_ema", 0))
    if tf_ema > 0:
        e = ema(df["close"], tf_ema)
        flip_up = flip_up & (df["close"] > e)
        flip_dn = flip_dn & (df["close"] < e)
    a = atr(df, int(p.get("atr_period",14)))
    sl_l = df["close"] - p.get("sl_atr_mult",2.0)*a
    sl_s = df["close"] + p.get("sl_atr_mult",2.0)*a
    tp_l = df["close"] + p.get("tp_atr_mult",3.0)*a
    tp_s = df["close"] - p.get("tp_atr_mult",3.0)*a
    return flip_up, flip_dn, sl_l, sl_s, tp_l, tp_s

def strat_S02(df, p):
    body = (df["close"]-df["open"]).abs()
    rng = (df["high"]-df["low"]).replace(0, np.nan)
    uw = df["high"] - df[["open","close"]].max(axis=1)
    lw = df[["open","close"]].min(axis=1) - df["low"]
    bs = body.replace(0, 1e-9)
    pat = ((uw >= p.get("upper_wick_ratio_min",2.0)*bs) &
           (lw <= p.get("lower_wick_ratio_max",0.5)*bs) &
           (body/rng < p.get("body_ratio_max",0.33)))
    trend = df["close"] < df["close"].shift(5)
    av = adx(df, int(p.get("adx_period",14)))
    vol_avg = df["volume"].rolling(20).mean()
    vo = df["volume"] > p.get("volume_confirm_mult",1.0)*vol_avg
    sig_l = pat & trend & (av > p.get("adx_threshold",20)) & vo
    a = atr(df, int(p.get("atr_period",14)))
    sl_l = df["low"] - p.get("sl_atr_mult",1.0)*a
    tp_l = df["close"] + p.get("tp_atr_mult",1.5)*a
    return sig_l, None, sl_l, None, tp_l, None

def build_signals(strategy, symbol, df, params):
    try:
        if strategy == "S30_Seasonality":
            return strat_S30(df, params)
        elif strategy == "S13_Hull_MA_Trend":
            return strat_S13(df, params)
        elif strategy == "S28_MTF_RSI_Align":
            return strat_S28(df, symbol, params)
        elif strategy == "S11_Supertrend":
            return strat_S11(df, params)
        elif strategy == "S02_Inverted_Hammer_Long":
            return strat_S02(df, params)
    except Exception as e:
        print(f"    {C_RED}[Signal ERR] {strategy}/{symbol}: {e}{S_RS}")
    return None, None, None, None, None, None

# ============================================================
# BACKTEST ENGINE
# ============================================================
def slip(price, direction, is_entry):
    if direction == "LONG":
        return price*(1+SLIPPAGE) if is_entry else price*(1-SLIPPAGE)
    else:
        return price*(1-SLIPPAGE) if is_entry else price*(1+SLIPPAGE)

def run_backtest(df, sig_l, sig_s, sl_l, sl_s, tp_l, tp_s):
    n = len(df)
    if n < 20: return [], None
    o=df["open"].values; h=df["high"].values; l=df["low"].values; c=df["close"].values
    idx = df.index

    def rvb(x): return x.reindex(df.index).fillna(False).values if x is not None else np.zeros(n,dtype=bool)
    def rv(x): return x.reindex(df.index).values if x is not None else np.full(n,np.nan)
    sig_l_v = rvb(sig_l); sig_s_v = rvb(sig_s)
    sl_l_v = rv(sl_l); sl_s_v = rv(sl_s); tp_l_v = rv(tp_l); tp_s_v = rv(tp_s)

    trades = []; equity = STARTING_CAPITAL
    eq_curve = [equity]; eq_times = [idx[0]]
    in_pos = False; pos = None

    for i in range(n-1):
        if in_pos:
            hi,lo,op = h[i],l[i],o[i]
            ex = None; reason = None
            if pos["direction"] == "LONG":
                if op <= pos["sl"]: ex = slip(op,"LONG",False); reason="SL_GAP"
                elif lo <= pos["sl"]: ex = slip(pos["sl"],"LONG",False); reason="SL"
                elif hi >= pos["tp"]: ex = slip(pos["tp"],"LONG",False); reason="TP"
            else:
                if op >= pos["sl"]: ex = slip(op,"SHORT",False); reason="SL_GAP"
                elif hi >= pos["sl"]: ex = slip(pos["sl"],"SHORT",False); reason="SL"
                elif lo <= pos["tp"]: ex = slip(pos["tp"],"SHORT",False); reason="TP"

            if ex is not None:
                size = pos["size"]; entry_px = pos["entry_price"]
                if pos["direction"]=="LONG": gross = (ex-entry_px)*size
                else: gross = (entry_px-ex)*size
                fees = (entry_px+ex)*size*FEES
                net = gross - fees
                equity += net
                pnl_pct = net/(entry_px*size)*100 if entry_px*size>0 else 0
                trades.append({
                    "entry_time":pos["entry_time"], "exit_time":idx[i],
                    "direction":pos["direction"], "entry_price":entry_px,
                    "exit_price":ex, "size":size, "sl":pos["sl_init"],
                    "tp":pos["tp_init"], "pnl":net, "pnl_pct":pnl_pct,
                    "reason":reason
                })
                eq_curve.append(equity); eq_times.append(idx[i])
                in_pos = False; pos = None

        if not in_pos and i+1 < n:
            direction = None
            if sig_l_v[i] and not np.isnan(sl_l_v[i]) and not np.isnan(tp_l_v[i]):
                direction="LONG"; sl_v=sl_l_v[i]; tp_v=tp_l_v[i]
            elif sig_s_v[i] and not np.isnan(sl_s_v[i]) and not np.isnan(tp_s_v[i]):
                direction="SHORT"; sl_v=sl_s_v[i]; tp_v=tp_s_v[i]
            if direction is not None:
                epx_raw = o[i+1]
                if np.isnan(epx_raw): continue
                epx = slip(epx_raw, direction, True)
                if direction=="LONG":
                    if sl_v >= epx or tp_v <= epx: continue
                    risk = epx - sl_v
                else:
                    if sl_v <= epx or tp_v >= epx: continue
                    risk = sl_v - epx
                if risk <= 0 or np.isnan(risk): continue
                size = (equity * RISK_PER_TRADE) / risk
                if size * epx > equity: size = equity / epx
                if size <= 0: continue
                pos = {"direction":direction, "entry_time":idx[i+1], "entry_price":epx,
                       "sl":sl_v, "tp":tp_v, "sl_init":sl_v, "tp_init":tp_v,
                       "size":size}
                in_pos = True

    if in_pos:
        ex = slip(c[-1], pos["direction"], False)
        size = pos["size"]; entry_px = pos["entry_price"]
        if pos["direction"]=="LONG": gross = (ex-entry_px)*size
        else: gross = (entry_px-ex)*size
        fees = (entry_px+ex)*size*FEES
        net = gross - fees; equity += net
        pnl_pct = net/(entry_px*size)*100 if entry_px*size>0 else 0
        trades.append({
            "entry_time":pos["entry_time"], "exit_time":idx[-1],
            "direction":pos["direction"], "entry_price":entry_px,
            "exit_price":ex, "size":size, "sl":pos["sl_init"],
            "tp":pos["tp_init"], "pnl":net, "pnl_pct":pnl_pct, "reason":"EOD"
        })
        eq_curve.append(equity); eq_times.append(idx[-1])

    eq_df = pd.DataFrame({"equity":eq_curve}, index=eq_times)
    return trades, eq_df

# ============================================================
# METRICS
# ============================================================
def compute_metrics(trades, eq_df):
    m = {"trades":0,"sharpe":0.0,"sortino":0.0,"calmar":0.0,
         "pf":0.0,"max_dd":0.0,"win_rate":0.0,"cagr":0.0,
         "total_return":0.0,"avg_trade":0.0,"expectancy":0.0,
         "monthly_freq":0.0,"max_loss_streak":0,"avg_win":0.0,"avg_loss":0.0}
    if not trades or eq_df is None or len(eq_df) < 2: return m
    m["trades"] = len(trades)
    pnls = np.array([t["pnl"] for t in trades])
    pcts = np.array([t["pnl_pct"] for t in trades])
    wins = pnls[pnls>0]; losses = pnls[pnls<0]
    m["win_rate"] = 100.0*len(wins)/len(pnls) if len(pnls)>0 else 0
    m["avg_trade"] = float(np.mean(pcts))
    m["avg_win"] = float(wins.mean()) if len(wins)>0 else 0
    m["avg_loss"] = float(abs(losses.mean())) if len(losses)>0 else 0
    gp = wins.sum() if len(wins)>0 else 0
    gl = abs(losses.sum()) if len(losses)>0 else 0
    m["pf"] = gp/gl if gl>0 else (999 if gp>0 else 0)
    pw = len(wins)/len(pnls) if len(pnls)>0 else 0
    m["expectancy"] = pw*m["avg_win"] - (1-pw)*m["avg_loss"]

    eq = eq_df["equity"].values
    m["total_return"] = (eq[-1]/eq[0]-1)*100
    days = (eq_df.index[-1]-eq_df.index[0]).days
    years = days/365.25 if days>0 else 1
    if eq[-1]>0 and eq[0]>0 and years>0:
        m["cagr"] = ((eq[-1]/eq[0])**(1/years)-1)*100
    peak = np.maximum.accumulate(eq)
    dd = (eq-peak)/peak
    m["max_dd"] = float(dd.min()*100)

    rets = pd.Series(pcts)/100.0
    if len(rets)>1 and rets.std()>0:
        avg_days = days/len(rets) if len(rets)>0 else 1
        tpy = 365.25 / max(avg_days, 0.5)
        m["sharpe"] = float(rets.mean()/rets.std()*np.sqrt(tpy))
        neg = rets[rets<0]
        if len(neg)>0 and neg.std()>0:
            m["sortino"] = float(rets.mean()/neg.std()*np.sqrt(tpy))
    if m["max_dd"]<0: m["calmar"] = m["cagr"]/abs(m["max_dd"])

    months = (days/30.44) if days>0 else 1
    m["monthly_freq"] = len(trades)/max(months, 1)

    cur = 0; mx = 0
    for p in pnls:
        if p < 0:
            cur += 1
            if cur > mx: mx = cur
        else: cur = 0
    m["max_loss_streak"] = mx

    for k in ["sharpe","sortino","calmar","pf","expectancy","avg_trade","monthly_freq","avg_win","avg_loss"]:
        m[k] = round(float(m[k]), 4)
    for k in ["max_dd","win_rate","cagr","total_return"]:
        m[k] = round(float(m[k]), 2)
    return m

# ============================================================
# HELPER: Run edge on given period
# ============================================================
def run_edge(edge, params_key="params", period="IS"):
    df_full = load_data(edge["symbol"], edge["tf"])
    if df_full is None: return None, None, "NO_DATA"
    if period == "IS":
        df = slice_period(df_full, IS_START, IS_END)
    elif period == "OOS":
        df = slice_period(df_full, OOS_START, OOS_END)
    else: df = df_full
    if df is None or len(df) < 20: return None, None, "INSUFFICIENT_BARS"
    p = edge.get(params_key, edge.get("params", {}))
    sigs = build_signals(edge["strategy"], edge["symbol"], df, p)
    if sigs[0] is None: return None, None, "SIGNAL_FAIL"
    trades, eq = run_backtest(df, *sigs)
    return trades, eq, "OK"

# ============================================================
# MONTHLY METRICS
# ============================================================
def monthly_metrics_from_trades(trades, month_start, month_end):
    """Compute per-month metrics from a slice of trades."""
    m_trades = [t for t in trades if month_start <= t["exit_time"] <= month_end]
    if not m_trades:
        return {"trades":0,"return_pct":0,"win_rate":0,"pf":0,"max_dd":0,
                "best":0,"worst":0,"net_pnl":0}
    pnls = np.array([t["pnl"] for t in m_trades])
    pcts = np.array([t["pnl_pct"] for t in m_trades])
    wins = (pnls>0).sum()
    gp = pnls[pnls>0].sum(); gl = abs(pnls[pnls<0].sum())
    pf = gp/gl if gl>0 else (999 if gp>0 else 0)
    # Intra-month equity DD
    eq = STARTING_CAPITAL + np.cumsum(pnls)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return {
        "trades":len(m_trades),
        "return_pct": round(float(np.sum(pcts)), 2),
        "win_rate": round(100*wins/len(m_trades), 2),
        "pf": round(pf, 3),
        "max_dd": round(float(dd.min()*100), 2),
        "best": round(float(pcts.max()), 2),
        "worst": round(float(pcts.min()), 2),
        "net_pnl": round(float(pnls.sum()), 2),
    }

# ============================================================
# PASS/FAIL SCORER
# ============================================================
def verdict(oos_m, is_m):
    if oos_m is None or oos_m["trades"] == 0:
        return "STRONG_FAIL", 0, ["No OOS trades executed"]
    criteria = []
    passed = []

    # 1. OOS Sharpe > 0
    c1 = oos_m["sharpe"] > 0
    passed.append(c1); criteria.append(("OOS Sharpe > 0", c1, f"{oos_m['sharpe']:.2f}"))

    # 2. Sharpe Degradation > 0.50
    if is_m["sharpe"] > 0:
        deg = oos_m["sharpe"] / is_m["sharpe"]
    else:
        deg = 0
    c2 = deg > 0.50
    passed.append(c2); criteria.append(("Sharpe Deg > 0.50", c2, f"{deg:.2f}"))

    # 3. OOS MaxDD > -25%
    c3 = oos_m["max_dd"] > -25
    passed.append(c3); criteria.append(("OOS MaxDD > -25%", c3, f"{oos_m['max_dd']:.2f}%"))

    # 4. MaxDD Ratio < 2.0
    if is_m["max_dd"] < 0:
        dd_ratio = abs(oos_m["max_dd"]) / abs(is_m["max_dd"])
    else:
        dd_ratio = 999
    c4 = dd_ratio < 2.0
    passed.append(c4); criteria.append(("DD Ratio < 2.0", c4, f"{dd_ratio:.2f}"))

    # 5. OOS PF > 1.0
    c5 = oos_m["pf"] > 1.0
    passed.append(c5); criteria.append(("OOS PF > 1.0", c5, f"{oos_m['pf']:.2f}"))

    # 6. OOS Trades >= 10
    c6 = oos_m["trades"] >= 10
    passed.append(c6); criteria.append(("OOS Trades >= 10", c6, f"{oos_m['trades']}"))

    n_pass = sum(passed)
    if n_pass == 6 and deg > 0.70:
        v = "STRONG_PASS"
    elif n_pass == 6:
        v = "WEAK_PASS"
    elif n_pass >= 4:
        v = "MARGINAL_FAIL"
    else:
        v = "STRONG_FAIL"
    return v, n_pass, criteria

# ============================================================
# SYMBOL PRICE CONTEXT
# ============================================================
def price_context(symbol, tf, period_start, period_end):
    df_full = load_data(symbol, tf)
    if df_full is None: return None
    df = slice_period(df_full, period_start, period_end)
    if df is None or len(df) < 5: return None
    start_px = float(df["close"].iloc[0])
    end_px = float(df["close"].iloc[-1])
    hi = float(df["high"].max()); lo = float(df["low"].min())
    change_pct = (end_px/start_px - 1)*100
    # Annualized volatility
    log_rets = np.log(df["close"]/df["close"].shift(1)).dropna()
    if tf == "1D": ann_factor = np.sqrt(365)
    elif tf == "4H": ann_factor = np.sqrt(365*6)
    elif tf == "1H": ann_factor = np.sqrt(365*24)
    else: ann_factor = np.sqrt(365)
    vol_ann = float(log_rets.std() * ann_factor * 100) if log_rets.std() > 0 else 0
    # Trend direction via 50-EMA slope
    if len(df) >= 60:
        e50 = ema(df["close"], 50)
        slope = float(e50.iloc[-1] - e50.iloc[-min(20,len(e50)-1)])
        trend = "BULL" if slope > 0 else "BEAR" if slope < 0 else "FLAT"
    else:
        trend = "N/A"
    # Regime changes (EMA cross count)
    regime_changes = 0
    if len(df) >= 60:
        e_fast = ema(df["close"], 12); e_slow = ema(df["close"], 26)
        cross = ((e_fast > e_slow) != (e_fast.shift(1) > e_slow.shift(1))).fillna(False).sum()
        regime_changes = int(cross)
    return {
        "start_price": start_px, "end_price": end_px,
        "high": hi, "low": lo, "change_pct": round(change_pct,2),
        "vol_ann_pct": round(vol_ann, 2), "trend": trend,
        "regime_changes": regime_changes, "bars": len(df)
    }

# ============================================================
# MAIN ORCHESTRATOR
# ============================================================
def main():
    t0 = time.time()
    box("WALK-FORWARD 2026 OOS VALIDATION", C_CYAN)
    print(f"{C_CYAN}  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{C_CYAN}  IS Window : {IS_START.date()} -> {IS_END.date()}")
    print(f"{C_MAG}  OOS Window: {OOS_START.date()} -> {OOS_END.date()} [UNLOCKED]")
    print(f"{C_YEL}  Edges to validate: {len(PORTFOLIO)}")

    # ================================================================
    # SECTION 1: IS BASELINE VERIFICATION
    # ================================================================
    section(1, "IS BASELINE VERIFICATION (2022-2025)", C_CYAN)
    is_results = []
    for e in PORTFOLIO:
        print(f"  Running IS: {e['strategy'][:22]:<22} | {e['symbol'][:14]:<14} | {e['tf']} ...")
        trades_is, eq_is, status = run_edge(e, "params", "IS")
        m_is = compute_metrics(trades_is, eq_is)
        # Compare to known
        known = e["known_is"]
        sh_diff_pct = abs(m_is["sharpe"] - known["sharpe"]) / max(abs(known["sharpe"]),0.01) * 100
        match_flag = "MATCH" if sh_diff_pct < 20 else "MISMATCH"
        match_col = C_GRN if match_flag == "MATCH" else C_YEL
        is_results.append({"edge":e, "trades":trades_is, "eq":eq_is,
                           "metrics":m_is, "match":match_flag, "status":status,
                           "sh_diff_pct":sh_diff_pct})
        print(f"    {match_col}[{match_flag}] IS Sharpe: computed={m_is['sharpe']:.2f} "
              f"vs known={known['sharpe']:.2f} (diff={sh_diff_pct:.1f}%){S_RS}")

    print(f"\n  {C_CYAN}IS Baseline Summary Table:{S_RS}")
    tbl = []
    for r in is_results:
        e = r["edge"]; m = r["metrics"]; kn = e["known_is"]
        match_col = C_GRN if r["match"]=="MATCH" else C_YEL
        tbl.append([
            e["idx"], e["strategy"][:22], e["symbol"][:12], e["tf"],
            m["trades"], f"{m['sharpe']:.2f}", f"{m['cagr']:.2f}%",
            f"{m['max_dd']:.2f}%", f"{kn['sharpe']:.2f}",
            f"{match_col}{r['match']}{S_RS}"
        ])
    print(tabulate(tbl, headers=["#","Strategy","Symbol","TF","Trades","IS Sh",
                                  "IS CAGR","IS DD","Known Sh","Match"], tablefmt="grid"))

    # ================================================================
    # SECTION 2: OOS RESULTS OVERVIEW
    # ================================================================
    section(2, "OOS 2026 RESULTS OVERVIEW", C_MAG)
    oos_results = []
    all_oos_trades = []
    for e in PORTFOLIO:
        print(f"  Running OOS: {e['strategy'][:22]:<22} | {e['symbol'][:14]:<14} | {e['tf']} ...")
        trades_oos, eq_oos, status = run_edge(e, "params", "OOS")
        m_oos = compute_metrics(trades_oos, eq_oos)
        oos_results.append({"edge":e, "trades":trades_oos, "eq":eq_oos,
                            "metrics":m_oos, "status":status})
        # attach edge info for combined log
        if trades_oos:
            for t in trades_oos:
                t["strategy"] = e["strategy"]
                t["symbol"] = e["symbol"]
                t["tf"] = e["tf"]
                all_oos_trades.append(t)
        prof_col = C_GRN if m_oos["total_return"]>0 else C_RED
        print(f"    Trades: {m_oos['trades']:>4} | Sharpe: {m_oos['sharpe']:>6.2f} | "
              f"PF: {m_oos['pf']:>5.2f} | MaxDD: {m_oos['max_dd']:>6.2f}% | "
              f"{prof_col}Return: {m_oos['total_return']:.2f}%{S_RS}")

    print(f"\n  {C_MAG}OOS Overview Table:{S_RS}")
    tbl = []
    for r in oos_results:
        e = r["edge"]; m = r["metrics"]
        ret_col = C_GRN if m["total_return"]>0 else C_RED
        tbl.append([
            e["idx"], e["strategy"][:22], e["symbol"][:12], e["tf"],
            m["trades"], f"{m['sharpe']:.2f}",
            f"{ret_col}{m['total_return']:.2f}%{S_RS}",
            f"{m['max_dd']:.2f}%", f"{m['pf']:.2f}", f"{m['win_rate']:.1f}%"
        ])
    print(tabulate(tbl, headers=["#","Strategy","Symbol","TF","Trades","OOS Sh",
                                  "Return","MaxDD","PF","Win%"], tablefmt="grid"))

    # ================================================================
    # SECTION 3: IS vs OOS COMPARISON
    # ================================================================
    section(3, "IS vs OOS COMPARISON (THE CRITICAL TABLE)", C_CYAN)
    comparison_rows = []
    for i in range(len(PORTFOLIO)):
        e = PORTFOLIO[i]
        m_is = is_results[i]["metrics"]
        m_oos = oos_results[i]["metrics"]
        deg = m_oos["sharpe"]/m_is["sharpe"] if m_is["sharpe"]>0 else 0
        v, n_pass, _ = verdict(m_oos, m_is)
        v_col = {"STRONG_PASS":C_GRN, "WEAK_PASS":C_YEL,
                 "MARGINAL_FAIL":C_MAG, "STRONG_FAIL":C_RED}.get(v, C_WHT)
        comparison_rows.append([
            e["idx"], e["strategy"][:20], e["symbol"][:12], e["tf"],
            f"{m_is['sharpe']:.2f}", f"{m_oos['sharpe']:.2f}",
            f"{deg*100:.1f}%",
            f"{m_is['max_dd']:.2f}%", f"{m_oos['max_dd']:.2f}%",
            f"{v_col}{v}{S_RS}"
        ])
    print(tabulate(comparison_rows, headers=["#","Strategy","Symbol","TF","IS Sh",
                                              "OOS Sh","Deg%","IS DD","OOS DD","Verdict"],
                   tablefmt="grid"))

    # ================================================================
    # SECTION 4: MONTH-BY-MONTH 2026 HEATMAP
    # ================================================================
    section(4, "MONTH-BY-MONTH 2026 HEATMAP", C_CYAN)
    months_2026 = [
        (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-31 23:59:59"), "Jan"),
        (pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-28 23:59:59"), "Feb"),
        (pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-31 23:59:59"), "Mar"),
        (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-30 23:59:59"), "Apr"),
        (pd.Timestamp("2026-05-01"), pd.Timestamp("2026-05-31 23:59:59"), "May"),
        (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-30 23:59:59"), "Jun"),
        (pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-31 23:59:59"), "Jul"),
        (pd.Timestamp("2026-08-01"), pd.Timestamp("2026-08-31 23:59:59"), "Aug"),
    ]

    monthly_data = []  # for CSV export
    heatmap_rows = []
    for r in oos_results:
        e = r["edge"]
        row = [f"{e['idx']}: {e['strategy'][:14]}/{e['symbol'][:8]}"]
        total_ret = 0
        for (ms, me, label) in months_2026:
            mm = monthly_metrics_from_trades(r["trades"] or [], ms, me)
            ret = mm["return_pct"]
            total_ret += ret
            col = C_GRN if ret>0.1 else (C_RED if ret<-0.1 else C_YEL)
            row.append(f"{col}{ret:+.2f}{S_RS}")
            monthly_data.append({
                "strategy":e["strategy"], "symbol":e["symbol"], "tf":e["tf"],
                "month":label, **mm
            })
        row.append(f"{C_CYAN}{total_ret:+.2f}{S_RS}")
        heatmap_rows.append(row)

    headers = ["Strategy/Symbol"] + [m[2] for m in months_2026] + ["Total"]
    print(tabulate(heatmap_rows, headers=headers, tablefmt="simple"))

    # ================================================================
    # SECTION 5: PORTFOLIO-LEVEL 2026 PERFORMANCE
    # ================================================================
    section(5, "PORTFOLIO-LEVEL 2026 PERFORMANCE (EQUAL WEIGHT)", C_MAG)
    daily_frames = []
    for r in oos_results:
        eq = r["eq"]
        if eq is None or len(eq) < 3: continue
        try:
            d = eq["equity"].resample("D").last().ffill()
            daily_frames.append(d)
        except Exception:
            continue
    if daily_frames:
        combined = pd.concat(daily_frames, axis=1).ffill().fillna(STARTING_CAPITAL)
        combined.columns = [f"e{i}" for i in range(len(daily_frames))]
        port_eq = combined.mean(axis=1)
        days = (port_eq.index[-1] - port_eq.index[0]).days
        years = max(days/365.25, 0.1)
        port_cagr = ((port_eq.iloc[-1]/port_eq.iloc[0])**(1/years)-1)*100 if port_eq.iloc[0]>0 else 0
        dr = port_eq.pct_change().dropna()
        port_sh = float(dr.mean()/dr.std()*np.sqrt(365)) if dr.std()>0 else 0
        peak = port_eq.cummax(); dd = (port_eq - peak)/peak
        port_dd = float(dd.min()*100)
        port_calmar = port_cagr/abs(port_dd) if port_dd < 0 else 0
        # monthly returns
        try:
            monthly = dr.resample("ME").apply(lambda x: (1+x).prod() - 1) * 100
        except Exception:
            monthly = pd.Series([])
        best_m = round(float(monthly.max()),2) if len(monthly)>0 else 0
        worst_m = round(float(monthly.min()),2) if len(monthly)>0 else 0
        pct_prof = round(100*(monthly>0).mean(),2) if len(monthly)>0 else 0

        # Combine trade stats
        pnls = np.array([t["pnl"] for t in all_oos_trades]) if all_oos_trades else np.array([])
        port_wr = 100*(pnls>0).sum()/len(pnls) if len(pnls)>0 else 0
        gp = pnls[pnls>0].sum(); gl = abs(pnls[pnls<0].sum())
        port_pf = gp/gl if gl>0 else (999 if gp>0 else 0)

        tbl = [
            ["Portfolio CAGR (annualized)", f"{port_cagr:.2f}%"],
            ["Portfolio Sharpe (annualized)", f"{port_sh:.3f}"],
            ["Portfolio Max Drawdown", f"{port_dd:.2f}%"],
            ["Portfolio Calmar", f"{port_calmar:.3f}"],
            ["Portfolio Win Rate", f"{port_wr:.2f}%"],
            ["Portfolio Profit Factor", f"{port_pf:.3f}"],
            ["Total Trades (combined)", len(all_oos_trades)],
            ["Best Monthly Return", f"{best_m:.2f}%"],
            ["Worst Monthly Return", f"{worst_m:.2f}%"],
            ["% Profitable Months", f"{pct_prof:.2f}%"],
            ["Combined Equity Start", f"${port_eq.iloc[0]:.2f}"],
            ["Combined Equity End", f"${port_eq.iloc[-1]:.2f}"],
            ["Combined Equity Peak", f"${port_eq.max():.2f}"],
            ["Combined Equity Trough", f"${port_eq.min():.2f}"],
        ]
        print(tabulate(tbl, headers=["Metric","Value"], tablefmt="grid"))

        if len(monthly)>0:
            print(f"\n  {C_CYAN}Monthly Portfolio Returns:{S_RS}")
            for idx, val in monthly.items():
                col = C_GRN if val > 0 else C_RED
                print(f"    {idx.strftime('%Y-%m'):<10} {col}{val:+.2f}%{S_RS}")
    else:
        print(f"  {C_RED}[!] No portfolio equity curves available for OOS.{S_RS}")

    # ================================================================
    # SECTION 6: SYMBOL-BY-SYMBOL DEEP DIVE
    # ================================================================
    section(6, "SYMBOL-BY-SYMBOL 2026 DEEP DIVE", C_CYAN)
    unique_symbols = sorted(set(e["symbol"] for e in PORTFOLIO))
    symbol_details = []

    for sym in unique_symbols:
        edges_on_sym = [e for e in PORTFOLIO if e["symbol"]==sym]
        # Get price context from primary TF
        primary_tf = edges_on_sym[0]["tf"]
        ctx = price_context(sym, primary_tf, OOS_START, OOS_END)

        print(f"\n  {C_CYAN}{S_BR}+{'='*70}+{S_RS}")
        print(f"  {C_CYAN}{S_BR}|  SYMBOL: {sym} - 2026 DETAILED ANALYSIS{' '*max(0,25-len(sym))}|{S_RS}")
        print(f"  {C_CYAN}{S_BR}+{'='*70}+{S_RS}")

        # 1. Price action
        print(f"\n  {C_YEL}1. PRICE ACTION SUMMARY (2026 OOS, TF={primary_tf}):{S_RS}")
        if ctx:
            pa_rows = [
                ["Start Price", f"${ctx['start_price']:.6f}"],
                ["End Price", f"${ctx['end_price']:.6f}"],
                ["High 2026", f"${ctx['high']:.6f}"],
                ["Low 2026", f"${ctx['low']:.6f}"],
                ["% Change 2026", f"{ctx['change_pct']:+.2f}%"],
                ["Annualized Vol 2026", f"{ctx['vol_ann_pct']:.2f}%"],
                ["Trend (50-EMA slope)", ctx["trend"]],
                ["Regime Changes (12/26 cross)", ctx["regime_changes"]],
                ["Bars in Period", ctx["bars"]],
            ]
            print(tabulate(pa_rows, headers=["Metric","Value"], tablefmt="simple"))
        else:
            print(f"    {C_RED}[!] No price data available{S_RS}")

        # 2. Strategy perf on this symbol
        print(f"\n  {C_YEL}2. STRATEGIES ON THIS SYMBOL:{S_RS}")
        sym_rows = []
        for e in edges_on_sym:
            idx_e = e["idx"] - 1
            m_is = is_results[idx_e]["metrics"]
            m_oos = oos_results[idx_e]["metrics"]
            deg = m_oos["sharpe"]/m_is["sharpe"] if m_is["sharpe"]>0 else 0
            sym_rows.append([
                e["strategy"][:22], e["tf"], e["source"],
                f"{m_is['sharpe']:.2f}", f"{m_oos['sharpe']:.2f}", f"{deg*100:.1f}%",
                m_oos["trades"], f"{m_oos['total_return']:.2f}%"
            ])
        print(tabulate(sym_rows, headers=["Strategy","TF","ParamSrc","IS Sh",
                                            "OOS Sh","Deg%","OOS Trades","OOS Ret"],
                       tablefmt="simple"))

        # 3. Verdict
        pass_count = 0; total = len(edges_on_sym)
        for e in edges_on_sym:
            idx_e = e["idx"] - 1
            m_is = is_results[idx_e]["metrics"]; m_oos = oos_results[idx_e]["metrics"]
            v, _, _ = verdict(m_oos, m_is)
            if v in ("STRONG_PASS","WEAK_PASS"): pass_count += 1
        if pass_count == total:
            action = "KEEP"; col = C_GRN
        elif pass_count >= total/2:
            action = "REDUCE"; col = C_YEL
        else:
            action = "DROP"; col = C_RED
        print(f"\n  {col}{S_BR}VERDICT for {sym}: {action}  ({pass_count}/{total} edges passed){S_RS}")

        # Store detail row
        symbol_details.append({
            "symbol": sym, "primary_tf": primary_tf,
            "start_price": ctx["start_price"] if ctx else None,
            "end_price": ctx["end_price"] if ctx else None,
            "change_pct": ctx["change_pct"] if ctx else None,
            "vol_ann_pct": ctx["vol_ann_pct"] if ctx else None,
            "trend": ctx["trend"] if ctx else None,
            "edges_on_symbol": total,
            "edges_passed": pass_count,
            "action": action
        })

    # ================================================================
    # SECTION 7: TUNED vs ORIGINAL COMPARISON ON OOS
    # ================================================================
    section(7, "TUNED vs ORIGINAL COMPARISON ON 2026 OOS", C_MAG)
    tuned_edges = [e for e in PORTFOLIO if e["source"].startswith("TUNED")]
    tuned_rows = []
    for e in tuned_edges:
        print(f"  Testing {e['strategy']} | {e['symbol']} | {e['tf']} ...")
        # Run tuned
        tr_t, eq_t, _ = run_edge(e, "params", "OOS")
        m_t = compute_metrics(tr_t, eq_t)
        # Run original
        e_orig = dict(e); e_orig["params"] = e["original_params"]
        tr_o, eq_o, _ = run_edge(e_orig, "params", "OOS")
        m_o = compute_metrics(tr_o, eq_o)
        winner = "TUNED" if m_t["sharpe"] > m_o["sharpe"] else "ORIGINAL"
        win_col = C_GRN if winner == "TUNED" else C_RED
        tuned_rows.append([
            e["strategy"][:22], e["symbol"][:14], e["tf"],
            e["source"].replace("TUNED_",""),
            f"{m_o['sharpe']:.2f}", f"{m_t['sharpe']:.2f}",
            f"{m_o['total_return']:.2f}%", f"{m_t['total_return']:.2f}%",
            f"{win_col}{winner}{S_RS}"
        ])
        print(f"    Original OOS Sharpe: {m_o['sharpe']:.2f} | Return: {m_o['total_return']:.2f}%")
        print(f"    Tuned    OOS Sharpe: {m_t['sharpe']:.2f} | Return: {m_t['total_return']:.2f}%")
        print(f"    {win_col}Winner: {winner}{S_RS}")
    print(f"\n  {C_MAG}Tuning Validity Test Summary:{S_RS}")
    print(tabulate(tuned_rows, headers=["Strategy","Symbol","TF","TuneMethod",
                                          "Orig Sh","Tuned Sh","Orig Ret","Tuned Ret","Winner"],
                   tablefmt="grid"))

    # ================================================================
    # SECTION 8: PASS/FAIL SCORECARD
    # ================================================================
    section(8, "PASS/FAIL SCORECARD - FINAL VERDICT", C_CYAN)
    scorecard_rows = []
    verdicts_all = []
    for i in range(len(PORTFOLIO)):
        e = PORTFOLIO[i]
        m_is = is_results[i]["metrics"]; m_oos = oos_results[i]["metrics"]
        v, n_pass, criteria = verdict(m_oos, m_is)
        verdicts_all.append({"edge":e, "verdict":v, "n_pass":n_pass,
                             "criteria":criteria, "m_is":m_is, "m_oos":m_oos})
        deg = m_oos["sharpe"]/m_is["sharpe"] if m_is["sharpe"]>0 else 0
        v_col = {"STRONG_PASS":C_GRN, "WEAK_PASS":C_YEL,
                 "MARGINAL_FAIL":C_MAG, "STRONG_FAIL":C_RED}.get(v, C_WHT)
        scorecard_rows.append([
            e["idx"], e["strategy"][:20], e["symbol"][:12],
            f"{m_oos['sharpe']:.2f}", f"{deg*100:.1f}%",
            f"{m_oos['max_dd']:.2f}%", f"{m_oos['pf']:.2f}",
            m_oos["trades"], f"{n_pass}/6",
            f"{v_col}{v}{S_RS}"
        ])
    print(tabulate(scorecard_rows, headers=["#","Strategy","Symbol","OOS Sh","Deg",
                                              "OOS DD","OOS PF","Trades","Passed","VERDICT"],
                   tablefmt="grid"))

    # Detail criteria per edge
    print(f"\n  {C_CYAN}Per-Edge Criteria Breakdown:{S_RS}")
    for vd in verdicts_all:
        e = vd["edge"]
        v_col = {"STRONG_PASS":C_GRN, "WEAK_PASS":C_YEL,
                 "MARGINAL_FAIL":C_MAG, "STRONG_FAIL":C_RED}.get(vd["verdict"], C_WHT)
        print(f"\n    {v_col}[{vd['verdict']}]{S_RS} #{e['idx']} {e['strategy']} | {e['symbol']} | {e['tf']}")
        for name, ok, val in vd["criteria"]:
            mk = f"{C_GRN}OK{S_RS}" if ok else f"{C_RED}FAIL{S_RS}"
            print(f"      [{mk}] {name}: {val}")

    n_strong = sum(1 for v in verdicts_all if v["verdict"]=="STRONG_PASS")
    n_weak = sum(1 for v in verdicts_all if v["verdict"]=="WEAK_PASS")
    n_marg = sum(1 for v in verdicts_all if v["verdict"]=="MARGINAL_FAIL")
    n_fail = sum(1 for v in verdicts_all if v["verdict"]=="STRONG_FAIL")
    n_pass = n_strong + n_weak
    print(f"\n  {C_CYAN}{S_BR}FINAL COUNTS:{S_RS}")
    print(f"    {C_GRN}Strong Pass: {n_strong}/10{S_RS}")
    print(f"    {C_YEL}Weak Pass  : {n_weak}/10{S_RS}")
    print(f"    {C_MAG}Marginal   : {n_marg}/10{S_RS}")
    print(f"    {C_RED}Strong Fail: {n_fail}/10{S_RS}")
    print(f"    Total Passed: {n_pass}/10")

    # ================================================================
    # SECTION 9: PORTFOLIO RECONSTRUCTION
    # ================================================================
    section(9, "PORTFOLIO RECONSTRUCTION FOR LIVE DEPLOYMENT", C_CYAN)
    keep_list = [v["edge"] for v in verdicts_all if v["verdict"] in ("STRONG_PASS","WEAK_PASS")]
    drop_list = [v["edge"] for v in verdicts_all if v["verdict"]=="STRONG_FAIL"]
    marginal_list = [v["edge"] for v in verdicts_all if v["verdict"]=="MARGINAL_FAIL"]

    print(f"\n  {C_GRN}KEEP ({len(keep_list)} edges - deploy live):{S_RS}")
    for e in keep_list:
        idx_e = e["idx"] - 1
        m_oos = oos_results[idx_e]["metrics"]
        print(f"    #{e['idx']} {e['strategy']:<25} {e['symbol']:<20} {e['tf']:<3} "
              f"OOS Sh: {m_oos['sharpe']:.2f} | OOS Ret: {m_oos['total_return']:.2f}%")

    print(f"\n  {C_MAG}MARGINAL ({len(marginal_list)} edges - monitor & test more):{S_RS}")
    for e in marginal_list:
        idx_e = e["idx"] - 1
        m_oos = oos_results[idx_e]["metrics"]
        print(f"    #{e['idx']} {e['strategy']:<25} {e['symbol']:<20} {e['tf']:<3} "
              f"OOS Sh: {m_oos['sharpe']:.2f} | OOS Ret: {m_oos['total_return']:.2f}%")

    print(f"\n  {C_RED}DROP ({len(drop_list)} edges - do not deploy):{S_RS}")
    for e in drop_list:
        idx_e = e["idx"] - 1
        m_oos = oos_results[idx_e]["metrics"]
        print(f"    #{e['idx']} {e['strategy']:<25} {e['symbol']:<20} {e['tf']:<3} "
              f"OOS Sh: {m_oos['sharpe']:.2f} | OOS Ret: {m_oos['total_return']:.2f}%")

    # ================================================================
    # SECTION 10: EXECUTIVE SUMMARY
    # ================================================================
    section(10, "EXECUTIVE SUMMARY", C_MAG)
    if n_pass >= 6:
        verdict_final = f"{C_GRN}GO - Deploy passing edges to live trading{S_RS}"
        confidence = min(10, 5 + n_pass)
    elif n_pass >= 3:
        verdict_final = f"{C_YEL}CONDITIONAL GO - Deploy only STRONG_PASS edges, paper trade the rest{S_RS}"
        confidence = 4 + n_pass
    else:
        verdict_final = f"{C_RED}NO-GO - Insufficient survival on OOS data. Return to research phase.{S_RS}"
        confidence = max(1, n_pass * 2)

    print(f"\n  Final Verdict: {verdict_final}")
    print(f"  Confidence Level: {confidence}/10")

    # Top 3 edges most likely to survive
    ranked = sorted(verdicts_all, key=lambda x: -x["m_oos"]["sharpe"])
    print(f"\n  {C_GRN}Top 3 Edges Most Likely to Survive Live Trading:{S_RS}")
    for v in ranked[:3]:
        e = v["edge"]
        print(f"    - {e['strategy']:<25} {e['symbol']:<20} {e['tf']:<3} "
              f"OOS Sh={v['m_oos']['sharpe']:.2f} | PF={v['m_oos']['pf']:.2f} | "
              f"Trades={v['m_oos']['trades']}")

    # Top 3 risks
    print(f"\n  {C_RED}Top 3 Risks to Monitor in Live Deployment:{S_RS}")
    risks = []
    for v in verdicts_all:
        e = v["edge"]; m = v["m_oos"]
        if m["max_dd"] < -10:
            risks.append((abs(m["max_dd"]), f"{e['strategy']} on {e['symbol']}: OOS DD = {m['max_dd']:.2f}%"))
    risks.sort(reverse=True)
    for r in risks[:3]:
        print(f"    - {r[1]}")
    print(f"    - Live slippage may exceed backtest assumption of 0.03% during volatile events")
    print(f"    - MTF strategies may face data-feed synchronization issues live")

    print(f"\n  {C_CYAN}Recommended Next Steps:{S_RS}")
    print(f"    1. Deploy passing edges on Bybit testnet for 30 days paper trading")
    print(f"    2. Monitor live slippage vs backtest assumption")
    print(f"    3. Set portfolio-level equity stop-loss at 10% from high water mark")
    print(f"    4. Re-run walk-forward every 3 months on rolling data")

    # ================================================================
    # SECTION 11: EXPORT
    # ================================================================
    section(11, "EXPORT RESULTS", C_CYAN)

    # IS vs OOS CSV
    iso_rows = []
    for i, e in enumerate(PORTFOLIO):
        m_is = is_results[i]["metrics"]; m_oos = oos_results[i]["metrics"]
        row = {"strategy":e["strategy"], "symbol":e["symbol"], "tf":e["tf"],
               "source":e["source"]}
        for k in ["sharpe","sortino","calmar","pf","max_dd","win_rate","cagr","total_return","trades","monthly_freq"]:
            row[f"is_{k}"] = m_is.get(k)
            row[f"oos_{k}"] = m_oos.get(k)
        v, np_v, _ = verdict(m_oos, m_is)
        row["verdict"] = v
        row["criteria_passed"] = np_v
        iso_rows.append(row)
    iso_csv = os.path.join(RESULTS_PATH, "walkforward_is_vs_oos.csv")
    pd.DataFrame(iso_rows).to_csv(iso_csv, index=False)
    print(f"  {C_GRN}[SAVED] {iso_csv}{S_RS}")

    # Monthly 2026 CSV
    monthly_csv = os.path.join(RESULTS_PATH, "walkforward_monthly_2026.csv")
    pd.DataFrame(monthly_data).to_csv(monthly_csv, index=False)
    print(f"  {C_GRN}[SAVED] {monthly_csv}{S_RS}")

    # Symbol details CSV
    sym_csv = os.path.join(RESULTS_PATH, "walkforward_symbol_detail.csv")
    pd.DataFrame(symbol_details).to_csv(sym_csv, index=False)
    print(f"  {C_GRN}[SAVED] {sym_csv}{S_RS}")

    # Trade log CSV
    trade_log_csv = os.path.join(RESULTS_PATH, "walkforward_trade_log_2026.csv")
    if all_oos_trades:
        pd.DataFrame(all_oos_trades).to_csv(trade_log_csv, index=False)
        print(f"  {C_GRN}[SAVED] {trade_log_csv} ({len(all_oos_trades)} trades){S_RS}")
    else:
        print(f"  {C_YEL}[!] No OOS trades to save{S_RS}")

    # Verdict CSV
    verdict_csv = os.path.join(RESULTS_PATH, "walkforward_verdict.csv")
    verdict_rows = []
    for v in verdicts_all:
        e = v["edge"]; m = v["m_oos"]
        verdict_rows.append({
            "idx":e["idx"], "strategy":e["strategy"], "symbol":e["symbol"],
            "tf":e["tf"], "source":e["source"], "verdict":v["verdict"],
            "criteria_passed":v["n_pass"], "oos_sharpe":m["sharpe"],
            "oos_max_dd":m["max_dd"], "oos_pf":m["pf"],
            "oos_trades":m["trades"], "oos_return_pct":m["total_return"]
        })
    pd.DataFrame(verdict_rows).to_csv(verdict_csv, index=False)
    print(f"  {C_GRN}[SAVED] {verdict_csv}{S_RS}")

    elapsed = time.time() - t0
    print()
    box(f"WALK-FORWARD VALIDATION COMPLETE - Runtime: {elapsed/60:.2f} min", C_GRN)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C_RED}Interrupted by user{S_RS}")
    except Exception as e:
        print(f"\n{C_RED}FATAL: {e}{S_RS}")
        traceback.print_exc()
