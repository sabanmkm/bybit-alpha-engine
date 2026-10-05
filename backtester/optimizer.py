# ============================================================
# HYPERPARAMETER OPTIMIZATION ENGINE
# Runs 3 methods per strategy, compares results
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
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError: missing.append("optuna")
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

STARTING_CAPITAL = 10000.0
RISK_PER_TRADE = 0.01
FEES = 0.00055
SLIPPAGE = 0.0003

OPTUNA_TRIALS = 300  # reduced from 500 for time budget
WALK_FORWARD_TRIALS = 100
GRID_POINTS = 6

# Colors
C_CYAN = Fore.CYAN; C_YEL = Fore.YELLOW; C_GRN = Fore.GREEN
C_RED = Fore.RED; C_MAG = Fore.MAGENTA; C_WHT = Fore.WHITE
S_BR = Style.BRIGHT; S_RS = Style.RESET_ALL

# Portfolio top 10 (from portfolio_top10.csv reference values)
TOP10 = [
    {"strategy":"S30_Seasonality","symbol":"TWT_USDT_USDT","tf":"4H",
     "orig":{"trades":278,"sharpe":1.56,"sortino":3.18,"calmar":2.57,"pf":1.47,
             "max_dd":-4.22,"win_rate":52.52,"cagr":10.86,"total_return":37.71,
             "avg_trade":0.54,"expectancy":13.57,"monthly_freq":7.56,"max_loss_streak":5}},
    {"strategy":"S13_Hull_MA_Trend","symbol":"1000LUNC_USDT_USDT","tf":"1D",
     "orig":{"trades":31,"sharpe":0.95,"sortino":4.17,"calmar":1.84,"pf":1.92,
             "max_dd":-2.01,"win_rate":58.06,"cagr":3.68,"total_return":11.10,
             "avg_trade":1.55,"expectancy":55.5,"monthly_freq":0.80,"max_loss_streak":2}},
    {"strategy":"S13_Hull_MA_Trend","symbol":"ZEN_USDT_USDT","tf":"1D",
     "orig":{"trades":51,"sharpe":1.05,"sortino":3.72,"calmar":1.70,"pf":1.91,
             "max_dd":-3.00,"win_rate":56.86,"cagr":5.08,"total_return":21.92,
             "avg_trade":7.37,"expectancy":42.99,"monthly_freq":1.09,"max_loss_streak":3}},
    {"strategy":"S28_MTF_RSI_Align","symbol":"MANA_USDT_USDT","tf":"1H",
     "orig":{"trades":290,"sharpe":1.24,"sortino":4.07,"calmar":2.07,"pf":1.40,
             "max_dd":-8.17,"win_rate":49.66,"cagr":16.92,"total_return":86.43,
             "avg_trade":0.44,"expectancy":29.80,"monthly_freq":6.09,"max_loss_streak":8}},
    {"strategy":"S13_Hull_MA_Trend","symbol":"CHZ_USDT_USDT","tf":"1D",
     "orig":{"trades":50,"sharpe":0.99,"sortino":3.12,"calmar":1.39,"pf":1.86,
             "max_dd":-3.52,"win_rate":56.00,"cagr":4.89,"total_return":20.87,
             "avg_trade":6.85,"expectancy":40.15,"monthly_freq":1.06,"max_loss_streak":3}},
    {"strategy":"S28_MTF_RSI_Align","symbol":"AVAX_USDT_USDT","tf":"1H",
     "orig":{"trades":255,"sharpe":1.06,"sortino":3.10,"calmar":2.34,"pf":1.46,
             "max_dd":-7.20,"win_rate":50.59,"cagr":16.82,"total_return":86.01,
             "avg_trade":0.41,"expectancy":33.73,"monthly_freq":5.38,"max_loss_streak":6}},
    {"strategy":"S11_Supertrend","symbol":"ANKR_USDT_USDT","tf":"4H",
     "orig":{"trades":77,"sharpe":1.41,"sortino":6.20,"calmar":0.83,"pf":1.75,
             "max_dd":-9.93,"win_rate":48.05,"cagr":8.23,"total_return":36.5,
             "avg_trade":1.20,"expectancy":39.12,"monthly_freq":1.61,"max_loss_streak":5}},
    {"strategy":"S28_MTF_RSI_Align","symbol":"ZRX_USDT_USDT","tf":"1H",
     "orig":{"trades":239,"sharpe":1.33,"sortino":4.17,"calmar":1.89,"pf":1.51,
             "max_dd":-9.38,"win_rate":51.05,"cagr":17.72,"total_return":84.39,
             "avg_trade":0.58,"expectancy":35.31,"monthly_freq":5.33,"max_loss_streak":8}},
    {"strategy":"S11_Supertrend","symbol":"AVAX_USDT_USDT","tf":"4H",
     "orig":{"trades":79,"sharpe":1.00,"sortino":3.58,"calmar":1.41,"pf":1.63,
             "max_dd":-5.02,"win_rate":45.57,"cagr":7.09,"total_return":30.4,
             "avg_trade":0.98,"expectancy":32.44,"monthly_freq":1.65,"max_loss_streak":4}},
    {"strategy":"S02_Inverted_Hammer_Long","symbol":"CHZ_USDT_USDT","tf":"4H",
     "orig":{"trades":30,"sharpe":0.94,"sortino":3.25,"calmar":0.79,"pf":2.08,
             "max_dd":-6.28,"win_rate":50.00,"cagr":4.94,"total_return":21.0,
             "avg_trade":1.35,"expectancy":39.15,"monthly_freq":0.71,"max_loss_streak":4}}
]

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

def color_delta(val, better_higher=True):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return f"{C_YEL}N/A{S_RS}"
    if val > 0:
        return (f"{C_GRN}+{val:.3f}{S_RS}" if better_higher else f"{C_RED}+{val:.3f}{S_RS}")
    elif val < 0:
        return (f"{C_RED}{val:.3f}{S_RS}" if better_higher else f"{C_GRN}{val:.3f}{S_RS}")
    return f"{val:.3f}"

# ============================================================
# DATE GATE
# ============================================================
def enforce_date_gate(df):
    if df is None or len(df) == 0: return df
    df = df[(df.index >= IS_START) & (df.index <= IS_END)].copy()
    if len(df) > 0 and df.index.max() > IS_END:
        raise ValueError("DATE GATE VIOLATION: 2026 data leaked into IS backtest!")
    return df

# ============================================================
# DATA LOADER
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
        df = enforce_date_gate(df)
        if len(df) < 200: _cache[key]=None; return None
        _cache[key] = df; return df
    except Exception:
        _cache[key] = None; return None

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
    plus_dm = np.where((up>dn)&(up>0), up, 0.0)
    minus_dm = np.where((dn>up)&(dn>0), dn, 0.0)
    pc = c.shift(1)
    tr = pd.concat([h-l,(h-pc).abs(),(l-pc).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    pdi = 100*pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False, min_periods=n).mean()/a.replace(0,np.nan)
    mdi = 100*pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False, min_periods=n).mean()/a.replace(0,np.nan)
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
# BACKTEST ENGINE
# ============================================================
def slip(price, direction, is_entry):
    if direction == "LONG":
        return price*(1+SLIPPAGE) if is_entry else price*(1-SLIPPAGE)
    else:
        return price*(1-SLIPPAGE) if is_entry else price*(1+SLIPPAGE)

def run_backtest(df, sig_l, sig_s, sl_l, sl_s, tp_l, tp_s, time_stop=None):
    n = len(df)
    if n < 50: return [], None
    o=df["open"].values; h=df["high"].values; l=df["low"].values; c=df["close"].values
    idx = df.index
    sl_l_v = sl_l.reindex(df.index).values if sl_l is not None else np.full(n,np.nan)
    sl_s_v = sl_s.reindex(df.index).values if sl_s is not None else np.full(n,np.nan)
    tp_l_v = tp_l.reindex(df.index).values if tp_l is not None else np.full(n,np.nan)
    tp_s_v = tp_s.reindex(df.index).values if tp_s is not None else np.full(n,np.nan)
    sig_l_v = sig_l.reindex(df.index).fillna(False).values if sig_l is not None else np.zeros(n,dtype=bool)
    sig_s_v = sig_s.reindex(df.index).fillna(False).values if sig_s is not None else np.zeros(n,dtype=bool)

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
                elif time_stop is not None and (i - pos["entry_i"]) >= time_stop:
                    ex = slip(c[i],"LONG",False); reason="TIME"
            else:
                if op >= pos["sl"]: ex = slip(op,"SHORT",False); reason="SL_GAP"
                elif hi >= pos["sl"]: ex = slip(pos["sl"],"SHORT",False); reason="SL"
                elif lo <= pos["tp"]: ex = slip(pos["tp"],"SHORT",False); reason="TP"
                elif time_stop is not None and (i - pos["entry_i"]) >= time_stop:
                    ex = slip(c[i],"SHORT",False); reason="TIME"
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
                       "size":size, "entry_i":i+1}
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
    m = {"trades":len(trades), "sharpe":0, "sortino":0, "calmar":0,
         "pf":0, "max_dd":0, "win_rate":0, "cagr":0, "total_return":0,
         "avg_trade":0, "expectancy":0, "monthly_freq":0, "max_loss_streak":0}
    if not trades or eq_df is None or len(eq_df) < 2: return m
    pnls = np.array([t["pnl"] for t in trades])
    pcts = np.array([t["pnl_pct"] for t in trades])
    wins = pnls[pnls>0]; losses = pnls[pnls<0]
    m["win_rate"] = 100*len(wins)/len(pnls) if len(pnls)>0 else 0
    m["avg_trade"] = float(np.mean(pcts))
    gp = wins.sum() if len(wins)>0 else 0
    gl = abs(losses.sum()) if len(losses)>0 else 0
    m["pf"] = gp/gl if gl>0 else (999 if gp>0 else 0)
    aw = wins.mean() if len(wins)>0 else 0
    al = abs(losses.mean()) if len(losses)>0 else 0
    pw = len(wins)/len(pnls) if len(pnls)>0 else 0
    m["expectancy"] = pw*aw - (1-pw)*al

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

    # Monthly freq
    months = (days/30.44) if days>0 else 1
    m["monthly_freq"] = len(trades)/max(months, 1)

    # Max loss streak
    cur = 0; mx = 0
    for p in pnls:
        if p < 0:
            cur += 1
            if cur > mx: mx = cur
        else: cur = 0
    m["max_loss_streak"] = mx

    for k in ["sharpe","sortino","calmar","pf","expectancy","avg_trade","monthly_freq"]:
        m[k] = round(float(m[k]), 4)
    for k in ["max_dd","win_rate","cagr","total_return"]:
        m[k] = round(float(m[k]), 2)
    return m

# ============================================================
# STRATEGY IMPLEMENTATIONS (fully parameterized)
# ============================================================
def strat_S30(df, p):
    dow = pd.Series(df.index.dayofweek, index=df.index)
    r = rsi(df["close"], int(p.get("rsi_period",14)))
    _, _, hist = macd(df["close"], int(p.get("macd_fast",12)),
                      int(p.get("macd_slow",26)), int(p.get("macd_signal",9)))
    a = atr(df, int(p.get("atr_period",14)))
    sig_l = (dow == int(p.get("entry_day_long",4))) & (r > p.get("rsi_long_threshold",50)) & (hist > 0)
    sig_s = (dow == int(p.get("entry_day_short",3))) & (r < p.get("rsi_short_threshold",50)) & (hist < 0)
    sl_l = df["close"] - p.get("sl_atr_mult",1.5)*a
    sl_s = df["close"] + p.get("sl_atr_mult",1.5)*a
    tp_l = df["close"] + p.get("tp_atr_mult",2.5)*a
    tp_s = df["close"] - p.get("tp_atr_mult",2.5)*a
    return sig_l, sig_s, sl_l, sl_s, tp_l, tp_s, int(p.get("hold_bars",2))

def strat_S13(df, p):
    hma = hull_ma(df["close"], int(p.get("hma_period",21)))
    lookback = int(p.get("slope_lookback",2))
    slope = hma.diff(lookback)
    slope_prev = slope.shift(1)
    thr = p.get("min_slope_threshold", 0.0)
    turn_up = (slope > thr) & (slope_prev <= thr)
    turn_dn = (slope < -thr) & (slope_prev >= -thr)
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
    return turn_up, turn_dn, sl_l, sl_s, tp_l, tp_s, None

def strat_S28(df, symbol, p):
    higher_tf = p.get("higher_tf", "4H")
    df_h = load_data(symbol, higher_tf)
    if df_h is None: return None, None, None, None, None, None, None
    n = int(p.get("rsi_period",14))
    r_h = rsi(df_h["close"], n).reindex(df.index, method="ffill")
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
    return sig_l, sig_s, sl_l, sl_s, tp_l, tp_s, None

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
    return flip_up, flip_dn, sl_l, sl_s, tp_l, tp_s, None

def strat_S02(df, p):
    body = (df["close"]-df["open"]).abs()
    rng = (df["high"]-df["low"]).replace(0, np.nan)
    uw = df["high"] - df[["open","close"]].max(axis=1)
    lw = df[["open","close"]].min(axis=1) - df["low"]
    br_max = p.get("body_ratio_max", 0.33)
    uw_min = p.get("upper_wick_ratio_min", 2.0)
    lw_max = p.get("lower_wick_ratio_max", 0.5)
    body_safe = body.replace(0, 1e-9)
    pat = (uw >= uw_min*body_safe) & (lw <= lw_max*body_safe) & (body/rng < br_max)
    trend = df["close"] < df["close"].shift(5)
    adx_v = adx(df, int(p.get("adx_period",14)))
    vol_avg = df["volume"].rolling(20).mean()
    vol_ok = df["volume"] > p.get("volume_confirm_mult",1.0)*vol_avg
    sig_l = pat & trend & (adx_v > p.get("adx_threshold",20)) & vol_ok
    a = atr(df, int(p.get("atr_period",14)))
    sl_l = df["low"] - p.get("sl_atr_mult",1.0)*a
    tp_l = df["close"] + p.get("tp_atr_mult",1.5)*a
    return sig_l, None, sl_l, None, tp_l, None, None

# ============================================================
# STRATEGY DISPATCHER
# ============================================================
def run_strategy(strategy, symbol, df, p):
    if strategy == "S30_Seasonality":
        sig_l, sig_s, sl_l, sl_s, tp_l, tp_s, ts = strat_S30(df, p)
    elif strategy == "S13_Hull_MA_Trend":
        sig_l, sig_s, sl_l, sl_s, tp_l, tp_s, ts = strat_S13(df, p)
    elif strategy == "S28_MTF_RSI_Align":
        out = strat_S28(df, symbol, p)
        if out[0] is None: return [], None
        sig_l, sig_s, sl_l, sl_s, tp_l, tp_s, ts = out
    elif strategy == "S11_Supertrend":
        sig_l, sig_s, sl_l, sl_s, tp_l, tp_s, ts = strat_S11(df, p)
    elif strategy == "S02_Inverted_Hammer_Long":
        sig_l, sig_s, sl_l, sl_s, tp_l, tp_s, ts = strat_S02(df, p)
    else:
        return [], None
    trades, eq_df = run_backtest(df, sig_l, sig_s, sl_l, sl_s, tp_l, tp_s, time_stop=ts)
    return trades, eq_df

# ============================================================
# PARAMETER SPACES
# ============================================================
def sample_params(trial, strategy):
    p = {}
    if strategy == "S30_Seasonality":
        p["entry_day_long"] = trial.suggest_int("entry_day_long", 0, 6)
        p["entry_day_short"] = trial.suggest_int("entry_day_short", 0, 6)
        p["rsi_period"] = trial.suggest_int("rsi_period", 7, 21)
        p["rsi_long_threshold"] = trial.suggest_float("rsi_long_threshold", 40, 55)
        p["rsi_short_threshold"] = trial.suggest_float("rsi_short_threshold", 45, 60)
        p["macd_fast"] = trial.suggest_int("macd_fast", 8, 16)
        p["macd_slow"] = trial.suggest_int("macd_slow", 20, 32)
        p["macd_signal"] = trial.suggest_int("macd_signal", 5, 13)
        p["hold_bars"] = trial.suggest_int("hold_bars", 1, 6)
        p["sl_atr_mult"] = trial.suggest_float("sl_atr_mult", 1.0, 3.0)
        p["tp_atr_mult"] = trial.suggest_float("tp_atr_mult", 1.5, 4.0)
        p["atr_period"] = trial.suggest_int("atr_period", 10, 20)
    elif strategy == "S13_Hull_MA_Trend":
        p["hma_period"] = trial.suggest_int("hma_period", 9, 55)
        p["slope_lookback"] = trial.suggest_int("slope_lookback", 1, 5)
        p["sl_atr_mult"] = trial.suggest_float("sl_atr_mult", 1.0, 3.0)
        p["tp_atr_mult"] = trial.suggest_float("tp_atr_mult", 2.0, 5.0)
        p["atr_period"] = trial.suggest_int("atr_period", 10, 20)
        p["trend_filter_ema"] = trial.suggest_categorical("trend_filter_ema", [0,50,100,200])
        p["min_slope_threshold"] = trial.suggest_float("min_slope_threshold", 0.0, 0.005)
    elif strategy == "S28_MTF_RSI_Align":
        p["rsi_period"] = trial.suggest_int("rsi_period", 7, 21)
        p["rsi_higher_tf_threshold"] = trial.suggest_float("rsi_higher_tf_threshold", 50, 65)
        p["rsi_entry_threshold"] = trial.suggest_float("rsi_entry_threshold", 45, 55)
        p["higher_tf"] = trial.suggest_categorical("higher_tf", ["4H","1D"])
        p["sl_atr_mult"] = trial.suggest_float("sl_atr_mult", 1.0, 2.5)
        p["tp_atr_mult"] = trial.suggest_float("tp_atr_mult", 1.5, 4.0)
        p["atr_period"] = trial.suggest_int("atr_period", 10, 20)
        p["rsi_exit_overbought"] = trial.suggest_float("rsi_exit_overbought", 70, 85)
        p["rsi_exit_oversold"] = trial.suggest_float("rsi_exit_oversold", 15, 30)
        p["cooldown_bars"] = trial.suggest_int("cooldown_bars", 0, 5)
    elif strategy == "S11_Supertrend":
        p["st_period"] = trial.suggest_int("st_period", 7, 14)
        p["st_multiplier"] = trial.suggest_float("st_multiplier", 2.0, 4.0)
        p["sl_atr_mult"] = trial.suggest_float("sl_atr_mult", 1.0, 3.0)
        p["tp_atr_mult"] = trial.suggest_float("tp_atr_mult", 2.0, 5.0)
        p["atr_period"] = trial.suggest_int("atr_period", 10, 20)
        p["trend_filter_ema"] = trial.suggest_categorical("trend_filter_ema", [0,50,100,200])
        p["use_trailing_stop"] = trial.suggest_categorical("use_trailing_stop", [True, False])
        p["trailing_atr_mult"] = trial.suggest_float("trailing_atr_mult", 1.5, 3.0)
    elif strategy == "S02_Inverted_Hammer_Long":
        p["body_ratio_max"] = trial.suggest_float("body_ratio_max", 0.2, 0.5)
        p["upper_wick_ratio_min"] = trial.suggest_float("upper_wick_ratio_min", 1.5, 3.0)
        p["lower_wick_ratio_max"] = trial.suggest_float("lower_wick_ratio_max", 0.3, 0.8)
        p["adx_period"] = trial.suggest_int("adx_period", 10, 20)
        p["adx_threshold"] = trial.suggest_float("adx_threshold", 15, 30)
        p["sl_atr_mult"] = trial.suggest_float("sl_atr_mult", 0.5, 2.0)
        p["tp_atr_mult"] = trial.suggest_float("tp_atr_mult", 1.0, 3.0)
        p["atr_period"] = trial.suggest_int("atr_period", 10, 20)
        p["volume_confirm_mult"] = trial.suggest_float("volume_confirm_mult", 0.8, 1.5)
    return p

def default_params(strategy):
    if strategy == "S30_Seasonality":
        return {"entry_day_long":4,"entry_day_short":3,"rsi_period":14,
                "rsi_long_threshold":50,"rsi_short_threshold":50,
                "macd_fast":12,"macd_slow":26,"macd_signal":9,
                "hold_bars":2,"sl_atr_mult":1.5,"tp_atr_mult":2.5,"atr_period":14}
    if strategy == "S13_Hull_MA_Trend":
        return {"hma_period":21,"slope_lookback":2,"sl_atr_mult":2.0,
                "tp_atr_mult":3.0,"atr_period":14,"trend_filter_ema":0,
                "min_slope_threshold":0.0}
    if strategy == "S28_MTF_RSI_Align":
        return {"rsi_period":14,"rsi_higher_tf_threshold":55,"rsi_entry_threshold":50,
                "higher_tf":"4H","sl_atr_mult":1.5,"tp_atr_mult":2.5,"atr_period":14,
                "rsi_exit_overbought":75,"rsi_exit_oversold":25,"cooldown_bars":0}
    if strategy == "S11_Supertrend":
        return {"st_period":10,"st_multiplier":3.0,"sl_atr_mult":2.0,"tp_atr_mult":3.0,
                "atr_period":14,"trend_filter_ema":0,"use_trailing_stop":False,
                "trailing_atr_mult":2.0}
    if strategy == "S02_Inverted_Hammer_Long":
        return {"body_ratio_max":0.33,"upper_wick_ratio_min":2.0,"lower_wick_ratio_max":0.5,
                "adx_period":14,"adx_threshold":20,"sl_atr_mult":1.0,"tp_atr_mult":1.5,
                "atr_period":14,"volume_confirm_mult":1.0}
    return {}

# ============================================================
# CV FOLDS
# ============================================================
CV_FOLDS_SPEC = [
    ("2022-01-01","2023-12-31","2024-01-01","2024-12-31"),
    ("2022-01-01","2024-12-31","2025-01-01","2025-12-31"),
    ("2023-01-01","2024-12-31","2025-01-01","2025-12-31"),
]

WF_WINDOWS = [
    ("2022-01-01","2022-06-30","2022-07-01","2022-12-31"),
    ("2022-01-01","2022-12-31","2023-01-01","2023-06-30"),
    ("2022-01-01","2023-06-30","2023-07-01","2023-12-31"),
    ("2022-01-01","2023-12-31","2024-01-01","2024-06-30"),
    ("2022-01-01","2024-06-30","2024-07-01","2025-06-30"),
    ("2022-01-01","2025-06-30","2025-07-01","2025-12-31"),
]

def slice_df(df, s, e):
    s = pd.Timestamp(s); e = pd.Timestamp(e + " 23:59:59") if len(e)==10 else pd.Timestamp(e)
    return df[(df.index >= s) & (df.index <= e)].copy()

# ============================================================
# METHOD 1: OPTUNA BAYESIAN
# ============================================================
def method1_optuna(strategy, symbol, tf, df, n_trials=OPTUNA_TRIALS, verbose=True):
    if df is None or len(df) < 300: return None
    fold_dfs = [(slice_df(df, s1, e1), slice_df(df, s2, e2)) for s1,e1,s2,e2 in CV_FOLDS_SPEC]
    fold_dfs = [(tr,te) for tr,te in fold_dfs if len(tr)>50 and len(te)>50]
    if len(fold_dfs) < 2: return None

    def objective(trial):
        p = sample_params(trial, strategy)
        # Validation: check hasn't collapsed the search space
        if strategy == "S30_Seasonality" and p["macd_fast"] >= p["macd_slow"]:
            return -10
        try:
            fold_sharpes = []
            for tr_df, te_df in fold_dfs:
                # Use validation fold only for scoring
                trades_v, eq_v = run_strategy(strategy, symbol, te_df, p)
                m_v = compute_metrics(trades_v, eq_v)
                if m_v["trades"] < 8 or m_v["max_dd"] < -25 or m_v["pf"] < 1.1:
                    fold_sharpes.append(-1.0)
                else:
                    fold_sharpes.append(m_v["sharpe"])
            avg = float(np.mean(fold_sharpes))
            var = float(np.var(fold_sharpes))
            if var > 0.5: avg -= 0.5*var
            return avg
        except Exception:
            return -10

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    class ProgressCB:
        def __init__(self, total): self.total=total; self.last=0
        def __call__(self, study, trial):
            n = trial.number + 1
            if verbose and (n - self.last >= 50 or n == self.total):
                self.last = n
                best = study.best_value if study.best_trial else 0
                print(f"    [Optuna {strategy}] Trial {n}/{self.total} | Best Sharpe: {best:.3f}")
    try:
        study.optimize(objective, n_trials=n_trials, callbacks=[ProgressCB(n_trials)], show_progress_bar=False)
        return {"method":"Optuna", "best_params":study.best_params,
                "best_value":float(study.best_value), "n_trials":len(study.trials)}
    except Exception as e:
        print(f"    {C_RED}[Optuna ERR] {e}{S_RS}")
        return None

# ============================================================
# METHOD 2: WALK-FORWARD
# ============================================================
def method2_walkforward(strategy, symbol, tf, df, verbose=True):
    if df is None or len(df) < 300: return None
    test_sharpes = []; best_params_list = []
    for wi, (s1,e1,s2,e2) in enumerate(WF_WINDOWS):
        tr_df = slice_df(df, s1, e1); te_df = slice_df(df, s2, e2)
        if len(tr_df) < 50 or len(te_df) < 30: continue

        def obj(trial):
            p = sample_params(trial, strategy)
            if strategy == "S30_Seasonality" and p["macd_fast"] >= p["macd_slow"]: return -10
            try:
                trades, eq = run_strategy(strategy, symbol, tr_df, p)
                m = compute_metrics(trades, eq)
                if m["trades"] < 5 or m["max_dd"] < -25: return -1
                return m["sharpe"]
            except Exception: return -10

        try:
            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=42+wi))
            study.optimize(obj, n_trials=WALK_FORWARD_TRIALS, show_progress_bar=False)
            best_p = study.best_params
            trades_te, eq_te = run_strategy(strategy, symbol, te_df, best_p)
            m_te = compute_metrics(trades_te, eq_te)
            test_sharpes.append(m_te["sharpe"])
            best_params_list.append(best_p)
            if verbose:
                print(f"    [WF {strategy}] Window {wi+1}/{len(WF_WINDOWS)} | "
                      f"Train Sharpe: {study.best_value:.2f} | Test Sharpe: {m_te['sharpe']:.2f}")
        except Exception as e:
            if verbose: print(f"    {C_RED}[WF ERR window {wi}] {e}{S_RS}")

    if not test_sharpes: return None
    avg_test = float(np.mean(test_sharpes))
    consistency = 1.0 - min(float(np.std(test_sharpes))/max(abs(avg_test),0.01), 1.0)
    # Aggregate best params: mode/median across windows
    if best_params_list:
        agg = {}
        keys = best_params_list[-1].keys()
        for k in keys:
            vals = [bp[k] for bp in best_params_list if k in bp]
            if not vals: continue
            if isinstance(vals[0], (int, np.integer)):
                agg[k] = int(np.median(vals))
            elif isinstance(vals[0], (float, np.floating)):
                agg[k] = float(np.median(vals))
            else:
                # Categorical: most common
                agg[k] = max(set(vals), key=vals.count)
        best_params = agg
    else:
        best_params = default_params(strategy)
    return {"method":"WalkForward", "best_params":best_params,
            "best_value":avg_test, "consistency":consistency,
            "windows":len(test_sharpes)}

# ============================================================
# METHOD 3: STABILITY HEATMAP
# ============================================================
def method3_stability(strategy, symbol, tf, df, verbose=True):
    if df is None or len(df) < 300: return None

    # Choose top 2 tunable params per strategy for grid
    if strategy == "S30_Seasonality":
        grid = {"sl_atr_mult": np.linspace(1.0, 3.0, GRID_POINTS),
                "tp_atr_mult": np.linspace(1.5, 4.0, GRID_POINTS)}
    elif strategy == "S13_Hull_MA_Trend":
        grid = {"hma_period": [9, 15, 21, 30, 40, 55],
                "sl_atr_mult": np.linspace(1.0, 3.0, GRID_POINTS)}
    elif strategy == "S28_MTF_RSI_Align":
        grid = {"rsi_higher_tf_threshold": np.linspace(50, 65, GRID_POINTS),
                "sl_atr_mult": np.linspace(1.0, 2.5, GRID_POINTS)}
    elif strategy == "S11_Supertrend":
        grid = {"st_multiplier": np.linspace(2.0, 4.0, GRID_POINTS),
                "sl_atr_mult": np.linspace(1.0, 3.0, GRID_POINTS)}
    elif strategy == "S02_Inverted_Hammer_Long":
        grid = {"adx_threshold": np.linspace(15, 30, GRID_POINTS),
                "sl_atr_mult": np.linspace(0.5, 2.0, GRID_POINTS)}
    else:
        return None

    keys = list(grid.keys())
    kx, ky = keys[0], keys[1]
    heatmap = np.zeros((len(grid[kx]), len(grid[ky])))
    base = default_params(strategy)

    for i, vx in enumerate(grid[kx]):
        for j, vy in enumerate(grid[ky]):
            p = dict(base)
            if isinstance(vx, (np.floating,)): vx = float(vx)
            elif isinstance(vx, (np.integer,)): vx = int(vx)
            if isinstance(vy, (np.floating,)): vy = float(vy)
            elif isinstance(vy, (np.integer,)): vy = int(vy)
            p[kx] = vx; p[ky] = vy
            try:
                trades, eq = run_strategy(strategy, symbol, df, p)
                m = compute_metrics(trades, eq)
                sh = m["sharpe"] if m["trades"] >= 15 else -1
                heatmap[i,j] = sh
            except Exception:
                heatmap[i,j] = -1

    # Find largest plateau: region where sharpe > 1.0
    plateau_mask = heatmap > 1.0
    plateau_size = int(plateau_mask.sum())

    if plateau_size >= 2:
        # Center of mass of plateau
        idx = np.argwhere(plateau_mask)
        cx = int(round(np.mean(idx[:,0])))
        cy = int(round(np.mean(idx[:,1])))
        cx = max(0, min(cx, len(grid[kx])-1))
        cy = max(0, min(cy, len(grid[ky])-1))
        val_x = grid[kx][cx]; val_y = grid[ky][cy]
        if isinstance(val_x, np.floating): val_x = float(val_x)
        elif isinstance(val_x, np.integer): val_x = int(val_x)
        if isinstance(val_y, np.floating): val_y = float(val_y)
        elif isinstance(val_y, np.integer): val_y = int(val_y)
        best_params = dict(base)
        best_params[kx] = val_x; best_params[ky] = val_y
        best_val = float(heatmap[cx,cy])
        robustness = plateau_size / (len(grid[kx]) * len(grid[ky]))
        if verbose:
            print(f"    [Stab {strategy}] Plateau size: {plateau_size}/{len(grid[kx])*len(grid[ky])} | "
                  f"Center Sharpe: {best_val:.2f} | Robustness: {robustness:.2%}")
        return {"method":"Stability", "best_params":best_params,
                "best_value":best_val, "plateau_size":plateau_size,
                "robustness":robustness, "heatmap":heatmap.tolist(),
                "kx":kx, "ky":ky}
    else:
        # No plateau, pick global best
        idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        val_x = grid[kx][idx[0]]; val_y = grid[ky][idx[1]]
        if isinstance(val_x, np.floating): val_x = float(val_x)
        elif isinstance(val_x, np.integer): val_x = int(val_x)
        if isinstance(val_y, np.floating): val_y = float(val_y)
        elif isinstance(val_y, np.integer): val_y = int(val_y)
        best_params = dict(base)
        best_params[kx] = val_x; best_params[ky] = val_y
        if verbose:
            print(f"    [Stab {strategy}] No wide plateau; best point Sharpe: {heatmap[idx]:.2f}")
        return {"method":"Stability", "best_params":best_params,
                "best_value":float(heatmap[idx]),
                "plateau_size":plateau_size, "robustness":0.0,
                "heatmap":heatmap.tolist(), "kx":kx, "ky":ky}

# ============================================================
# SELECTION LOGIC
# ============================================================
def select_best_method(r1, r2, r3):
    """Returns (selected_result, method_name, rationale)"""
    if r1 is None and r2 is None and r3 is None:
        return None, "NONE", "All methods failed"
    if r2 is not None and r1 is not None:
        # If WF sharpe >= 70% of Optuna sharpe, prefer WF (more robust)
        if r2["best_value"] >= 0.7 * r1["best_value"] and r2["best_value"] > 0:
            return r2, "WalkForward", f"WF Sharpe ({r2['best_value']:.2f}) >= 70% of Optuna ({r1['best_value']:.2f})"
    if r3 is not None and r3.get("robustness", 0) > 0.30:
        return r3, "Stability", f"Wide plateau ({r3['plateau_size']} cells, robustness {r3['robustness']:.0%})"
    if r1 is not None and r1["best_value"] > 0:
        return r1, "Optuna", f"Highest Sharpe: {r1['best_value']:.2f}"
    if r2 is not None:
        return r2, "WalkForward", f"Fallback to WF (avg Sharpe {r2['best_value']:.2f})"
    if r3 is not None:
        return r3, "Stability", "Fallback to stability"
    return None, "NONE", "No positive result"

# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    box("HYPERPARAMETER OPTIMIZATION ENGINE v1.0", C_CYAN)
    print(f"{C_CYAN}  Run       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{C_CYAN}  IS Window : {IS_START.date()} -> {IS_END.date()}")
    print(f"{C_CYAN}  OOS Locked: {OOS_START.date()} -> 2026-08-31")
    print(f"{C_YEL}  [!] DATE GATE ACTIVE: 2026 data is sealed{S_RS}")
    print(f"{C_CYAN}  Optuna trials: {OPTUNA_TRIALS} | WF trials per window: {WALK_FORWARD_TRIALS}")
    print(f"{C_CYAN}  Grid points  : {GRID_POINTS}x{GRID_POINTS}")

    all_results = []
    best_params_all = {}

    for idx, edge in enumerate(TOP10, 1):
        strategy = edge["strategy"]; symbol = edge["symbol"]; tf = edge["tf"]
        section(idx, f"EDGE #{idx}/10: {strategy} | {symbol} | {tf}", C_MAG)
        df = load_data(symbol, tf)
        if df is None:
            print(f"  {C_RED}[!] Data not found for {symbol} @ {tf}. Skipping.{S_RS}")
            continue

        # Baseline
        p_def = default_params(strategy)
        t_def, eq_def = run_strategy(strategy, symbol, df, p_def)
        m_def = compute_metrics(t_def, eq_def)
        print(f"  {C_CYAN}Baseline (default params): Trades={m_def['trades']} Sharpe={m_def['sharpe']:.2f} "
              f"PF={m_def['pf']:.2f} MaxDD={m_def['max_dd']:.2f}%{S_RS}")

        # METHOD 1
        print(f"\n  {C_MAG}>>> METHOD 1: Optuna Bayesian ({OPTUNA_TRIALS} trials, 3-fold CV){S_RS}")
        r1 = method1_optuna(strategy, symbol, tf, df)
        if r1: print(f"     Best CV Sharpe: {r1['best_value']:.3f}")

        # METHOD 2
        print(f"\n  {C_MAG}>>> METHOD 2: Walk-Forward ({len(WF_WINDOWS)} windows x {WALK_FORWARD_TRIALS} trials){S_RS}")
        r2 = method2_walkforward(strategy, symbol, tf, df)
        if r2: print(f"     Avg Test Sharpe: {r2['best_value']:.3f} | Consistency: {r2.get('consistency',0):.2%}")

        # METHOD 3
        print(f"\n  {C_MAG}>>> METHOD 3: Stability Grid ({GRID_POINTS}x{GRID_POINTS}){S_RS}")
        r3 = method3_stability(strategy, symbol, tf, df)

        # Selection
        selected, method_name, rationale = select_best_method(r1, r2, r3)
        print(f"\n  {C_GRN}[SELECTED] Method: {method_name} | Rationale: {rationale}{S_RS}")

        # Run tuned backtest on full IS
        if selected:
            tuned_params = selected["best_params"]
            t_tuned, eq_tuned = run_strategy(strategy, symbol, df, tuned_params)
            m_tuned = compute_metrics(t_tuned, eq_tuned)
        else:
            tuned_params = p_def
            m_tuned = m_def

        # Guard: if tuned is worse than default, revert
        keep_tuned = m_tuned["sharpe"] > m_def["sharpe"] and m_tuned["trades"] >= 15
        if not keep_tuned:
            print(f"  {C_YEL}[!] Tuned Sharpe ({m_tuned['sharpe']:.2f}) <= default ({m_def['sharpe']:.2f}). Reverting to default.{S_RS}")
            tuned_params = p_def
            m_tuned = m_def
            selection = "REVERTED_TO_DEFAULT"
        else:
            selection = method_name

        # Detect suspicious tuning
        orig_sharpe = edge["orig"]["sharpe"]
        overfit_flag = "LOW"
        if m_tuned["sharpe"] > 2.5 * orig_sharpe:
            overfit_flag = "HIGH"
        elif m_tuned["sharpe"] > 1.5 * orig_sharpe:
            overfit_flag = "MEDIUM"

        entry = {
            "idx": idx, "strategy": strategy, "symbol": symbol, "tf": tf,
            "orig": edge["orig"], "tuned": m_tuned,
            "params_default": p_def, "params_tuned": tuned_params,
            "method": selection, "rationale": rationale,
            "m1_sharpe": r1["best_value"] if r1 else None,
            "m2_sharpe": r2["best_value"] if r2 else None,
            "m2_consistency": r2.get("consistency") if r2 else None,
            "m3_sharpe": r3["best_value"] if r3 else None,
            "m3_robustness": r3.get("robustness") if r3 else None,
            "overfit_flag": overfit_flag
        }
        all_results.append(entry)
        best_params_all[f"{strategy}|{symbol}|{tf}"] = tuned_params

    # ==== SECTION 2: BEFORE vs AFTER individual ====
    section(2, "BEFORE vs AFTER: INDIVIDUAL STRATEGY COMPARISON", C_CYAN)
    for e in all_results:
        header = f"{e['strategy']} | {e['symbol']} | {e['tf']}"
        print(f"\n  {C_CYAN}{S_BR}+{'='*70}+{S_RS}")
        print(f"  {C_CYAN}{S_BR}| {header:<68} |{S_RS}")
        print(f"  {C_CYAN}{S_BR}+{'='*70}+{S_RS}")
        o = e["orig"]; t = e["tuned"]
        rows = []
        for label, key, better_high in [
            ("Trades","trades",True),("Sharpe","sharpe",True),
            ("Sortino","sortino",True),("Calmar","calmar",True),
            ("Profit Factor","pf",True),("Max DD %","max_dd",False),
            ("Win Rate %","win_rate",True),("CAGR %","cagr",True),
            ("Total Return %","total_return",True),("Avg Trade %","avg_trade",True),
            ("Expectancy","expectancy",True),("Monthly Freq","monthly_freq",True),
            ("Max Loss Streak","max_loss_streak",False),
        ]:
            ov = o.get(key, 0); tv = t.get(key, 0)
            try:
                delta = float(tv) - float(ov)
                delta_col = color_delta(delta, better_higher=better_high)
            except Exception:
                delta_col = "-"
            rows.append([label, ov, tv, delta_col])
        print(tabulate(rows, headers=["Metric","BEFORE","AFTER","Change"], tablefmt="simple"))

        # Param changes
        print(f"\n  {C_YEL}Parameters Changed:{S_RS}")
        changed = 0
        for k, v_new in e["params_tuned"].items():
            v_def = e["params_default"].get(k, "N/A")
            if str(v_def) != str(v_new):
                print(f"    {k}: {v_def} -> {v_new}")
                changed += 1
        if changed == 0:
            print(f"    (no changes - defaults were optimal)")
        print(f"\n  Optimization Method: {C_GRN}{e['method']}{S_RS}")
        print(f"  Rationale: {e['rationale']}")
        risk_col = C_GRN if e['overfit_flag']=='LOW' else (C_YEL if e['overfit_flag']=='MEDIUM' else C_RED)
        print(f"  Overfitting Risk: {risk_col}{e['overfit_flag']}{S_RS}")

    # ==== SECTION 3: PORTFOLIO-LEVEL ====
    section(3, "PORTFOLIO-LEVEL COMPARISON (THE MONEY TABLE)", C_CYAN)

    # Aggregate portfolio metrics (equal weight sum equity curves)
    def portfolio_agg(source_key):
        all_eq = []
        combined_trades = []
        for e in all_results:
            strategy = e["strategy"]; symbol = e["symbol"]; tf = e["tf"]
            df = load_data(symbol, tf)
            if df is None: continue
            params = e["params_default"] if source_key=="orig" else e["params_tuned"]
            trades, eq = run_strategy(strategy, symbol, df, params)
            if eq is None or len(eq) < 5: continue
            try:
                d = eq["equity"].resample("D").last().ffill()
                all_eq.append(d)
                combined_trades.extend(trades)
            except Exception:
                continue
        if not all_eq: return None
        combined = pd.concat(all_eq, axis=1).ffill().fillna(STARTING_CAPITAL)
        combined.columns = [f"e{i}" for i in range(len(all_eq))]
        port_eq = combined.mean(axis=1)
        return port_eq, combined_trades

    orig_port = portfolio_agg("orig")
    tuned_port = portfolio_agg("tuned")

    def port_metrics(res):
        if res is None: return None
        eq, tr = res
        if len(eq) < 5: return None
        m = {"start_eq":float(eq.iloc[0]), "end_eq":float(eq.iloc[-1]),
             "trades":len(tr)}
        days = (eq.index[-1]-eq.index[0]).days
        years = max(days/365.25, 0.1)
        m["cagr"] = ((eq.iloc[-1]/eq.iloc[0])**(1/years)-1)*100 if eq.iloc[0]>0 else 0
        dr = eq.pct_change().dropna()
        m["sharpe"] = float(dr.mean()/dr.std()*np.sqrt(365)) if dr.std()>0 else 0
        neg = dr[dr<0]
        m["sortino"] = float(dr.mean()/neg.std()*np.sqrt(365)) if len(neg)>0 and neg.std()>0 else 0
        peak = eq.cummax(); dd = (eq-peak)/peak
        m["max_dd"] = float(dd.min()*100)
        m["calmar"] = m["cagr"]/abs(m["max_dd"]) if m["max_dd"]<0 else 0
        if tr:
            pnls = np.array([t["pnl"] for t in tr])
            wins = (pnls>0).sum()
            m["win_rate"] = 100*wins/len(pnls)
            gp = pnls[pnls>0].sum(); gl = abs(pnls[pnls<0].sum())
            m["pf"] = gp/gl if gl>0 else 999
        else:
            m["win_rate"]=0; m["pf"]=0
        months = days/30.44 if days>0 else 1
        m["monthly_trades"] = len(tr)/max(months,1)
        try:
            monthly = dr.resample("ME").apply(lambda x: (1+x).prod()-1)
            m["worst_month"] = float(monthly.min())*100 if len(monthly)>0 else 0
            m["best_month"] = float(monthly.max())*100 if len(monthly)>0 else 0
            m["pct_profitable_months"] = float((monthly>0).mean()*100) if len(monthly)>0 else 0
        except Exception:
            m["worst_month"]=0; m["best_month"]=0; m["pct_profitable_months"]=0
        m["net_profit"] = m["end_eq"] - m["start_eq"]
        for k in ["cagr","sharpe","sortino","calmar","pf","max_dd","win_rate",
                  "monthly_trades","worst_month","best_month","pct_profitable_months"]:
            m[k] = round(float(m[k]), 3)
        return m

    mo = port_metrics(orig_port)
    mt = port_metrics(tuned_port)

    if mo and mt:
        rows = []
        pairs = [
            ("Portfolio CAGR %","cagr",True),
            ("Portfolio Sharpe","sharpe",True),
            ("Portfolio Sortino","sortino",True),
            ("Portfolio Max DD %","max_dd",False),
            ("Portfolio Calmar","calmar",True),
            ("Portfolio PF","pf",True),
            ("Portfolio Win Rate %","win_rate",True),
            ("Total Trades","trades",True),
            ("Monthly Trades","monthly_trades",True),
            ("% Profitable Months","pct_profitable_months",True),
            ("Worst Month %","worst_month",True),
            ("Best Month %","best_month",True),
            ("Start Equity","start_eq",True),
            ("End Equity","end_eq",True),
            ("Net Profit $","net_profit",True),
        ]
        for label, key, hi in pairs:
            ov = mo.get(key,0); tv = mt.get(key,0)
            delta = float(tv) - float(ov)
            delta_col = color_delta(delta, better_higher=hi)
            rows.append([label, f"{ov:.2f}", f"{tv:.2f}", delta_col])
        print(tabulate(rows, headers=["Portfolio Metric","ORIGINAL","TUNED","Change"], tablefmt="grid"))

        verdict = "IMPROVED" if mt["sharpe"] > mo["sharpe"] else "NO IMPROVEMENT"
        vc = C_GRN if verdict=="IMPROVED" else C_YEL
        print(f"\n  {vc}{S_BR}VERDICT: {verdict}{S_RS}")

    # ==== SECTION 4: PARAM CHANGE SUMMARY ====
    section(4, "PARAMETER CHANGE SUMMARY", C_CYAN)
    param_changes = []
    for e in all_results:
        for k, v_new in e["params_tuned"].items():
            v_def = e["params_default"].get(k)
            if v_def is None or str(v_def) == str(v_new): continue
            delta_sharpe = e["tuned"]["sharpe"] - e["orig"]["sharpe"]
            param_changes.append([e["strategy"][:20], k, v_def, v_new, round(delta_sharpe,2)])
    if param_changes:
        param_changes.sort(key=lambda x: -abs(x[4]))
        print(tabulate(param_changes[:30], headers=["Strategy","Param","Default","Tuned","dSharpe"], tablefmt="simple"))
    else:
        print(f"  {C_YEL}No parameters were tuned away from defaults.{S_RS}")

    # ==== SECTION 5: OVERFITTING DIAGNOSTICS ====
    section(5, "OVERFITTING DIAGNOSTICS", C_CYAN)
    diag = []
    for e in all_results:
        diag.append([
            e["strategy"][:20], e["symbol"][:12], e["tf"],
            e["orig"]["sharpe"], round(e["tuned"]["sharpe"],2),
            round(e["m2_sharpe"],2) if e["m2_sharpe"] is not None else "-",
            round(e["m2_consistency"],2) if e["m2_consistency"] is not None else "-",
            round(e["m3_robustness"],2) if e["m3_robustness"] is not None else "-",
            e["method"], e["overfit_flag"]
        ])
    print(tabulate(diag, headers=["Strategy","Symbol","TF","OrigSh","TunedSh","WFSh","WFCons","M3Rob","Method","Risk"], tablefmt="grid"))

    # ==== SECTION 6: RECOMMENDATIONS ====
    section(6, "RECOMMENDATIONS", C_CYAN)
    improved = [e for e in all_results if e["tuned"]["sharpe"] > e["orig"]["sharpe"]]
    no_change = [e for e in all_results if e["tuned"]["sharpe"] <= e["orig"]["sharpe"]]

    print(f"\n  {C_GRN}Strategies IMPROVED by tuning ({len(improved)}):{S_RS}")
    for e in sorted(improved, key=lambda x: -(x["tuned"]["sharpe"]-x["orig"]["sharpe"]))[:10]:
        d = e["tuned"]["sharpe"] - e["orig"]["sharpe"]
        print(f"    {e['strategy']} | {e['symbol']} | {e['tf']}: "
              f"Sharpe {e['orig']['sharpe']:.2f} -> {e['tuned']['sharpe']:.2f} (+{d:.2f})")

    print(f"\n  {C_YEL}Strategies where tuning did NOT improve ({len(no_change)}):{S_RS}")
    for e in no_change:
        print(f"    {e['strategy']} | {e['symbol']} | {e['tf']}: kept defaults")

    print(f"\n  {C_CYAN}Final Tuned Portfolio Composition:{S_RS}")
    for e in all_results:
        print(f"    #{e['idx']} {e['strategy']:<26} {e['symbol']:<22} {e['tf']:<3} "
              f"Sharpe: {e['tuned']['sharpe']:.2f} | Method: {e['method']} | Risk: {e['overfit_flag']}")

    print(f"\n  {C_MAG}Next Steps:{S_RS}")
    print(f"    1. Walk-Forward validation on 2026 sealed OOS data")
    print(f"    2. Monte Carlo trade-sequence stress testing (2000 randomizations)")
    print(f"    3. Live paper trading on Bybit with tuned parameters (30 days minimum)")

    # ==== SECTION 7: EXPORT ====
    section(7, "EXPORT", C_CYAN)

    # tuned_portfolio_top10.csv
    rows = []
    for e in all_results:
        row = {"idx":e["idx"], "strategy":e["strategy"], "symbol":e["symbol"], "tf":e["tf"],
               "method":e["method"], "overfit_flag":e["overfit_flag"]}
        for k, v in e["tuned"].items(): row[f"tuned_{k}"] = v
        for k, v in e["orig"].items(): row[f"orig_{k}"] = v
        rows.append(row)
    tuned_csv = os.path.join(RESULTS_PATH, "tuned_portfolio_top10.csv")
    pd.DataFrame(rows).to_csv(tuned_csv, index=False)
    print(f"  {C_GRN}[SAVED] {tuned_csv}{S_RS}")

    # optimization_comparison.csv
    comp_rows = []
    for e in all_results:
        for metric in ["sharpe","sortino","calmar","pf","max_dd","win_rate","cagr","total_return","trades"]:
            comp_rows.append({
                "strategy": e["strategy"], "symbol": e["symbol"], "tf": e["tf"],
                "metric": metric,
                "before": e["orig"].get(metric),
                "after": e["tuned"].get(metric),
                "delta": (e["tuned"].get(metric,0) or 0) - (e["orig"].get(metric,0) or 0)
            })
    comp_csv = os.path.join(RESULTS_PATH, "optimization_comparison.csv")
    pd.DataFrame(comp_rows).to_csv(comp_csv, index=False)
    print(f"  {C_GRN}[SAVED] {comp_csv}{S_RS}")

    # best_parameters.json
    def _json_safe(v):
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.bool_,)): return bool(v)
        return v
    json_output = {}
    for key, params in best_params_all.items():
        json_output[key] = {k: _json_safe(v) for k, v in params.items()}
    json_path = os.path.join(RESULTS_PATH, "best_parameters.json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2, default=str)
    print(f"  {C_GRN}[SAVED] {json_path}{S_RS}")

    elapsed = time.time() - t0
    print()
    box(f"OPTIMIZATION COMPLETE - Runtime: {elapsed/60:.2f} minutes", C_GRN)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C_RED}Interrupted by user{S_RS}")
    except Exception as e:
        print(f"\n{C_RED}FATAL: {e}{S_RS}")
        traceback.print_exc()
