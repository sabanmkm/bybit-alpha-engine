import os
import sys
import time
import hmac
import hashlib
import requests
import inspect
import pandas as pd
import numpy as np
from datetime import datetime

# --- Manual .env parser (No python-dotenv required) ---
env_path = r"C:\BybitLiveBot\.env"
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("'").strip('"')

sys.path.insert(0, r"C:\BybitLiveBot")

# Dynamically import strategies module
try:
    import strategies as strat_module
except Exception as e:
    print(f"[ERROR] Could not import strategies: {e}")
    sys.exit(1)

def get_strategy_instance(pattern):
    for name, cls in inspect.getmembers(strat_module, inspect.isclass):
        if pattern.lower() in name.lower():
            try:
                return cls()
            except Exception:
                pass
    return None

API_KEY = os.getenv("BYBIT_API_KEY", "")
API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BASE_URL = "https://api.bybit.com"

# Live bot startup timestamp
STARTUP_TS = pd.Timestamp("2026-09-19 05:39:00")

def bybit_get(endpoint, params=None):
    if params is None: params = {}
    timestamp = str(int(time.time() * 1000))
    recv_window = "20000"
    query_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
    sign_str = timestamp + API_KEY + recv_window + query_str
    signature = hmac.new(API_SECRET.encode("utf-8"), sign_str.encode("utf-8"), hashlib.sha256).hexdigest()
    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-SIGN": signature,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "Content-Type": "application/json"
    }
    try:
        r = requests.get(f"{BASE_URL}{endpoint}?{query_str}", headers=headers, timeout=15)
        return r.json()
    except Exception as e:
        return {"retCode": -1, "retMsg": str(e)}

print("\n[1/4] Fetching Closed Trades from Bybit API...")
pnl_res = bybit_get("/v5/position/closed-pnl", {"category": "linear", "limit": "50"})

live_trades = []
if pnl_res.get("retCode") == 0:
    raw_list = pnl_res.get("result", {}).get("list", [])
    for t in raw_list:
        symbol = t.get("symbol")
        closing_side = t.get("side")
        pos_dir = "LONG" if closing_side == "Sell" else "SHORT"
        
        entry_p = float(t.get("avgEntryPrice", 0))
        exit_p = float(t.get("avgExitPrice", 0))
        closed_pnl = float(t.get("closedPnl", 0))
        created_time = pd.to_datetime(int(t.get("createdTime", 0)), unit="ms")
        
        if created_time >= STARTUP_TS:
            live_trades.append({
                "symbol": symbol,
                "direction": pos_dir,
                "exit_time": created_time,
                "entry_price": entry_p,
                "exit_price": exit_p,
                "pnl": closed_pnl
            })
    print(f"      [OK] Loaded {len(live_trades)} live closed trades from Bybit.")
else:
    print(f"      [WARN] API Error: {pnl_res.get('retMsg')}")

print("\n[2/4] Downloading Live Candle Data from Bybit...")

def fetch_klines(symbol, interval, limit=200):
    url = f"{BASE_URL}/v5/market/kline"
    params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("retCode") == 0:
            rows = data.get("result", {}).get("list", [])
            if not rows: return pd.DataFrame()
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df
    except Exception:
        pass
    return pd.DataFrame()

active_edge_configs = [
    ("ATOMUSDT", "240", "S30a"),
    ("ADAUSDT",  "240", "S30a"),
    ("BSVUSDT",  "240", "S30a"),
    ("AVAXUSDT", "240", "S30a"),
    ("QTUMUSDT", "240", "S30a"),
    ("ETCUSDT",  "240", "S30a"),
    ("TWTUSDT",  "240", "S30b"),
    ("DOTUSDT",  "60",  "S30d"),
    ("XLMUSDT",  "60",  "S30d"),
    ("ETHUSDT",  "240", "R13"),
    ("SANDUSDT", "240", "R15"),
    ("OPUSDT",   "240", "R01"),
    ("OPUSDT",   "240", "R15"),
]

print("\n[3/4] Running Backtest Simulation on Live Candle Data...")

backtest_trades = []

for symbol, interval, strat_pattern in active_edge_configs:
    strat = get_strategy_instance(strat_pattern)
    if strat is None:
        continue
        
    df = fetch_klines(symbol, interval, limit=200)
    if df.empty or len(df) < 30:
        continue
    
    try:
        params = getattr(strat, "default_params", {})
        signals, sl_series, tp_series = strat.generate_signals(df, params)
    except Exception:
        continue
    
    for i in range(len(df) - 1):
        sig_time = df.loc[i, "timestamp"]
        sig = signals.iloc[i]
        
        if sig != 0 and sig_time >= STARTUP_TS:
            entry_time = df.loc[i + 1, "timestamp"]
            entry_price = df.loc[i + 1, "open"]
            direction = "LONG" if sig > 0 else "SHORT"
            sl = sl_series.iloc[i]
            tp = tp_series.iloc[i]
            
            exit_price = None
            exit_time = None
            for j in range(i + 1, len(df)):
                cur_bar = df.loc[j]
                if direction == "LONG":
                    if cur_bar["low"] <= sl:
                        exit_price = sl
                        exit_time = cur_bar["timestamp"]
                        break
                    elif cur_bar["high"] >= tp:
                        exit_price = tp
                        exit_time = cur_bar["timestamp"]
                        break
                else:
                    if cur_bar["high"] >= sl:
                        exit_price = sl
                        exit_time = cur_bar["timestamp"]
                        break
                    elif cur_bar["low"] <= tp:
                        exit_price = tp
                        exit_time = cur_bar["timestamp"]
                        break
            
            if exit_price is not None:
                qty = 5.0 / entry_price
                gross_pnl = (exit_price - entry_price) * qty if direction == "LONG" else (entry_price - exit_price) * qty
                fees = (5.0 * 0.00055) + (qty * exit_price * 0.00055)
                net_pnl = gross_pnl - fees
                
                backtest_trades.append({
                    "symbol": symbol,
                    "direction": direction,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": net_pnl
                })

print(f"      [OK] Simulated {len(backtest_trades)} backtest trades in this period.")

print("\n[4/4] Generating Side-by-Side Comparison Table...")

comparison_rows = []
matched_live_indices = set()

for bt in backtest_trades:
    bt_sym = bt["symbol"]
    bt_dir = bt["direction"]
    bt_entry_p = bt["entry_price"]
    bt_exit_p = bt["exit_price"]
    bt_pnl = bt["pnl"]
    
    best_match = None
    best_idx = None
    for idx, lt in enumerate(live_trades):
        if idx in matched_live_indices: continue
        if lt["symbol"] == bt_sym and lt["direction"] == bt_dir:
            if abs(lt["entry_price"] - bt_entry_p) / bt_entry_p < 0.03:
                best_match = lt
                best_idx = idx
                break
    
    if best_match:
        matched_live_indices.add(best_idx)
        pnl_diff = abs(best_match["pnl"] - bt_pnl)
        status = "EXACT MATCH" if pnl_diff < 0.05 else "MATCH (SLIPPAGE)"
        comparison_rows.append({
            "Symbol": bt_sym,
            "Dir": bt_dir,
            "Signal Date": str(bt["entry_time"])[:16],
            "BT Entry": f"${bt_entry_p:.4f}",
            "Live Entry": f"${best_match['entry_price']:.4f}",
            "BT Exit": f"${bt_exit_p:.4f}",
            "Live Exit": f"${best_match['exit_price']:.4f}",
            "BT PnL": f"${bt_pnl:+.4f}",
            "Live PnL": f"${best_match['pnl']:+.4f}",
            "Match Status": status
        })
    else:
        comparison_rows.append({
            "Symbol": bt_sym,
            "Dir": bt_dir,
            "Signal Date": str(bt["entry_time"])[:16],
            "BT Entry": f"${bt_entry_p:.4f}",
            "Live Entry": "N/A",
            "BT Exit": f"${bt_exit_p:.4f}",
            "Live Exit": "N/A",
            "BT PnL": f"${bt_pnl:+.4f}",
            "Live PnL": "N/A",
            "Match Status": "MISSED BY LIVE"
        })

for idx, lt in enumerate(live_trades):
    if idx not in matched_live_indices:
        comparison_rows.append({
            "Symbol": lt["symbol"],
            "Dir": lt["direction"],
            "Signal Date": str(lt["exit_time"])[:16],
            "BT Entry": "N/A",
            "Live Entry": f"${lt['entry_price']:.4f}",
            "BT Exit": "N/A",
            "Live Exit": f"${lt['exit_price']:.4f}",
            "BT PnL": "N/A",
            "Live PnL": f"${lt['pnl']:+.4f}",
            "Match Status": "EXTRA LIVE"
        })

df_comp = pd.DataFrame(comparison_rows)
print("\n" + "=" * 110)
print("  SIDE-BY-SIDE RECONCILIATION TABLE")
print("=" * 110)
print(df_comp.to_string(index=False))

matches = sum(1 for r in comparison_rows if "MATCH" in r["Match Status"])
total = len(comparison_rows)
match_rate = (matches / total * 100) if total > 0 else 0

print("\n" + "-" * 60)
print(f"  Total Trades Evaluated : {total}")
print(f"  Successfully Matched   : {matches} / {total}")
print(f"  Reconciliation Match % : {match_rate:.1f}%")
print("-" * 60)
