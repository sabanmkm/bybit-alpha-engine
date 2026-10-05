"""
BYBIT LIVE BOT - BYBIT V5 API WRAPPER
REST + WebSocket with retries, backoff, rate limiting.
"""
import time
import hmac
import hashlib
import json
import logging
import threading
from typing import Optional, Dict, List, Callable
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np

import config

log = logging.getLogger(__name__)

class BybitAPI:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = config.BYBIT_TESTNET_URL if testnet else config.BYBIT_MAINNET_URL
        self.testnet = testnet
        self._time_offset_ms = 0
        self._last_time_sync = 0
        self._session = self._build_session()
        self._instrument_cache = {}

    def _build_session(self) -> requests.Session:
        s = requests.Session()
        retry = Retry(total=config.API_MAX_RETRIES, backoff_factor=1.5,
                      status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        s.mount("https://", adapter)
        s.headers.update({"User-Agent": "BybitLiveBot/1.0"})
        return s

    def _sign(self, params: Dict, timestamp_ms: int, recv_window: int = 5000) -> str:
        param_str = f"{timestamp_ms}{self.api_key}{recv_window}"
        if params:
            param_str += json.dumps(params, separators=(",", ":"))
        signature = hmac.new(
            self.api_secret.encode(), param_str.encode(), hashlib.sha256
        ).hexdigest()
        return signature

    def _sign_get(self, params: Dict, timestamp_ms: int, recv_window: int = 5000) -> str:
        # For GET requests, params are query string
        sorted_params = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        param_str = f"{timestamp_ms}{self.api_key}{recv_window}{sorted_params}"
        return hmac.new(
            self.api_secret.encode(), param_str.encode(), hashlib.sha256
        ).hexdigest()

    def sync_server_time(self):
        if time.time() - self._last_time_sync < 300:  # sync every 5 min
            return
        try:
            r = self._session.get(f"{self.base_url}/v5/market/time", timeout=10)
            if r.status_code == 200:
                data = r.json()
                srv_ms = int(data["result"]["timeNano"]) // 1_000_000
                local_ms = int(time.time() * 1000)
                self._time_offset_ms = srv_ms - local_ms
                self._last_time_sync = time.time()
                log.debug(f"Time sync: offset {self._time_offset_ms}ms")
        except Exception as e:
            log.warning(f"Time sync failed: {e}")

    def _get_timestamp_ms(self) -> int:
        return int(time.time() * 1000) + self._time_offset_ms

    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None,
                 signed: bool = False, max_retries: int = None) -> Optional[Dict]:
        if max_retries is None: max_retries = config.API_MAX_RETRIES
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        for attempt in range(max_retries):
            try:
                if signed:
                    self.sync_server_time()
                    ts_ms = self._get_timestamp_ms()
                    recv_window = 5000
                    if method == "GET":
                        sig = self._sign_get(params, ts_ms, recv_window)
                    else:
                        sig = self._sign(params, ts_ms, recv_window)
                    headers = {
                        "X-BAPI-API-KEY": self.api_key,
                        "X-BAPI-SIGN": sig,
                        "X-BAPI-TIMESTAMP": str(ts_ms),
                        "X-BAPI-RECV-WINDOW": str(recv_window),
                        "Content-Type": "application/json",
                    }
                    if method == "POST":
                        r = self._session.post(url, json=params, headers=headers, timeout=config.API_TIMEOUT_SEC)
                    else:
                        r = self._session.get(url, params=params, headers=headers, timeout=config.API_TIMEOUT_SEC)
                else:
                    if method == "GET":
                        r = self._session.get(url, params=params, timeout=config.API_TIMEOUT_SEC)
                    else:
                        r = self._session.post(url, json=params, timeout=config.API_TIMEOUT_SEC)

                if r.status_code == 200:
                    data = r.json()
                    if data.get("retCode") == 0:
                        return data
                    else:
                        log.warning(f"API error {endpoint}: {data.get('retCode')} - {data.get('retMsg')}")
                        if data.get("retCode") in (10002, 10004):  # timestamp/signature error
                            self._last_time_sync = 0  # force re-sync
                            time.sleep(1); continue
                        return data  # return error dict for caller
                elif r.status_code == 429:
                    sleep_time = config.API_BACKOFF_BASE_SEC ** (attempt + 1)
                    log.warning(f"Rate limited, sleeping {sleep_time}s")
                    time.sleep(sleep_time)
                    continue
                else:
                    log.warning(f"HTTP {r.status_code} on {endpoint}: {r.text[:200]}")
            except Exception as e:
                sleep_time = config.API_BACKOFF_BASE_SEC ** (attempt + 1)
                log.warning(f"Request exception on {endpoint}: {e}. Retry {attempt+1}/{max_retries} after {sleep_time}s")
                time.sleep(sleep_time)
        log.error(f"All retries failed for {endpoint}")
        return None

    # ============================================================
    # PUBLIC ENDPOINTS
    # ============================================================
    def get_server_time(self) -> Optional[int]:
        r = self._request("GET", "/v5/market/time")
        if r and r.get("retCode") == 0:
            return int(r["result"]["timeNano"]) // 1_000_000
        return None

    def get_instrument_info(self, symbol: str) -> Optional[Dict]:
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]
        r = self._request("GET", "/v5/market/instruments-info",
                          params={"category": "linear", "symbol": symbol})
        if r and r.get("retCode") == 0 and r["result"].get("list"):
            info = r["result"]["list"][0]
            parsed = {
                "symbol": info["symbol"],
                "min_order_qty": float(info["lotSizeFilter"]["minOrderQty"]),
                "qty_step": float(info["lotSizeFilter"]["qtyStep"]),
                "min_notional_value": float(info["lotSizeFilter"].get("minNotionalValue", 5)),
                "tick_size": float(info["priceFilter"]["tickSize"]),
                "max_leverage": float(info["leverageFilter"]["maxLeverage"]),
                "status": info["status"],
            }
            self._instrument_cache[symbol] = parsed
            return parsed
        return None

    def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        r = self._request("GET", "/v5/market/tickers",
                          params={"category": "linear", "symbol": symbol})
        if r and r.get("retCode") == 0 and r["result"].get("list"):
            t = r["result"]["list"][0]
            return {
                "symbol": symbol,
                "funding_rate": float(t.get("fundingRate", 0)),
                "next_funding_time": int(t.get("nextFundingTime", 0)),
                "last_price": float(t.get("lastPrice", 0)),
                "bid": float(t.get("bid1Price", 0)) if t.get("bid1Price") else 0,
                "ask": float(t.get("ask1Price", 0)) if t.get("ask1Price") else 0,
                "volume_24h": float(t.get("turnover24h", 0)),
            }
        return None

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Returns DataFrame with columns: open, high, low, close, volume; indexed by open_time UTC."""
        r = self._request("GET", "/v5/market/kline",
                          params={"category": "linear", "symbol": symbol,
                                  "interval": interval, "limit": min(limit, 1000)})
        if r and r.get("retCode") == 0 and r["result"].get("list"):
            rows = r["result"]["list"]
            # Bybit returns descending; reverse
            rows = list(reversed(rows))
            df = pd.DataFrame(rows, columns=["open_time","open","high","low","close","volume","turnover"])
            df["open_time"] = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms", utc=True).dt.tz_localize(None)
            df = df.set_index("open_time")
            for c in ["open","high","low","close","volume"]:
                df[c] = df[c].astype(float)
            return df[["open","high","low","close","volume"]]
        return None

    # ============================================================
    # PRIVATE ENDPOINTS
    # ============================================================
    def get_wallet_balance(self, account_type: str = "UNIFIED") -> Optional[float]:
        r = self._request("GET", "/v5/account/wallet-balance",
                          params={"accountType": account_type, "coin": "USDT"},
                          signed=True)
        if r and r.get("retCode") == 0 and r["result"].get("list"):
            wallets = r["result"]["list"]
            for w in wallets:
                for c in w.get("coin", []):
                    if c["coin"] == "USDT":
                        # Prefer availableToWithdraw; fallback to equity/walletBalance
                        for key in ("availableToWithdraw", "walletBalance", "equity"):
                            v = c.get(key, "")
                            if v not in ("", None):
                                try:
                                    return float(v)
                                except: pass
        return None

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        params = {"category": "linear", "settleCoin": "USDT"}
        if symbol: params["symbol"] = symbol
        r = self._request("GET", "/v5/position/list", params=params, signed=True)
        if r and r.get("retCode") == 0:
            positions = []
            for p in r["result"].get("list", []):
                size = float(p.get("size", 0))
                if size > 0:
                    positions.append({
                        "symbol": p["symbol"],
                        "side": p["side"],  # "Buy" or "Sell"
                        "size": size,
                        "avg_price": float(p.get("avgPrice", 0)),
                        "leverage": float(p.get("leverage", 1)),
                        "unrealized_pnl": float(p.get("unrealisedPnl", 0)),
                        "position_value": float(p.get("positionValue", 0)),
                        "stop_loss": float(p.get("stopLoss", 0)) if p.get("stopLoss") else 0,
                        "take_profit": float(p.get("takeProfit", 0)) if p.get("takeProfit") else 0,
                    })
            return positions
        return []

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        r = self._request("POST", "/v5/position/set-leverage",
                          params={"category": "linear", "symbol": symbol,
                                  "buyLeverage": str(leverage), "sellLeverage": str(leverage)},
                          signed=True)
        # retCode 110043 = leverage not modified (already set)
        return r is not None and r.get("retCode") in (0, 110043)

    def set_margin_mode(self, symbol: str, mode: str = "ISOLATED", leverage: int = 10) -> bool:
        # mode: "ISOLATED_MARGIN" or "REGULAR_MARGIN" (cross)
        r = self._request("POST", "/v5/position/switch-isolated",
                          params={"category": "linear", "symbol": symbol,
                                  "tradeMode": 1 if mode == "ISOLATED" else 0,
                                  "buyLeverage": str(leverage), "sellLeverage": str(leverage)},
                          signed=True)
        # 110026 = already in that mode
        return r is not None and r.get("retCode") in (0, 110026, 110043)

    def place_order(self, symbol: str, side: str, qty: float,
                    order_type: str = "Market", price: Optional[float] = None,
                    stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                    reduce_only: bool = False) -> Optional[str]:
        """
        side: "Buy" or "Sell"
        Returns order_id on success, None on failure.
        """
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
        }
        if price is not None and order_type == "Limit":
            params["price"] = str(price)
        if stop_loss is not None:
            params["stopLoss"] = str(stop_loss)
        if take_profit is not None:
            params["takeProfit"] = str(take_profit)
        if reduce_only:
            params["reduceOnly"] = True

        r = self._request("POST", "/v5/order/create", params=params, signed=True)
        if r and r.get("retCode") == 0:
            order_id = r["result"].get("orderId")
            log.info(f"Order placed: {symbol} {side} qty={qty} SL={stop_loss} TP={take_profit} -> {order_id}")
            return order_id
        else:
            err = r.get("retMsg", "Unknown") if r else "No response"
            log.error(f"Order failed: {symbol} {side} qty={qty}: {err}")
            return None

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        r = self._request("POST", "/v5/order/cancel",
                          params={"category": "linear", "symbol": symbol, "orderId": order_id},
                          signed=True)
        return r is not None and r.get("retCode") == 0

    def cancel_all_orders(self, symbol: Optional[str] = None) -> bool:
        params = {"category": "linear", "settleCoin": "USDT"}
        if symbol: params["symbol"] = symbol
        r = self._request("POST", "/v5/order/cancel-all", params=params, signed=True)
        return r is not None and r.get("retCode") == 0

    def get_order_status(self, symbol: str, order_id: str) -> Optional[Dict]:
        r = self._request("GET", "/v5/order/realtime",
                          params={"category": "linear", "symbol": symbol, "orderId": order_id},
                          signed=True)
        if r and r.get("retCode") == 0 and r["result"].get("list"):
            o = r["result"]["list"][0]
            return {
                "order_id": o["orderId"],
                "symbol": o["symbol"],
                "status": o["orderStatus"],
                "avg_price": float(o.get("avgPrice", 0)) if o.get("avgPrice") else 0,
                "cum_exec_qty": float(o.get("cumExecQty", 0)),
                "cum_exec_value": float(o.get("cumExecValue", 0)),
            }
        return None

    def close_position(self, symbol: str, side_to_close: str, qty: float) -> Optional[str]:
        """Close position by placing opposite market order with reduceOnly."""
        opposite = "Sell" if side_to_close == "Buy" else "Buy"
        return self.place_order(symbol, opposite, qty, order_type="Market", reduce_only=True)
