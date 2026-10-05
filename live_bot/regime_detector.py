"""
BYBIT LIVE BOT - REGIME DETECTOR
BTC-anchored BULL/BEAR/CHOP/CRISIS classifier.
"""
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta
from engine_v2 import EMA, ATR, ADX
import logging

log = logging.getLogger(__name__)

class RegimeDetector:
    def __init__(self, api):
        self.api = api
        self._btc_cache: Optional[pd.DataFrame] = None
        self._regime_cache: dict = {}
        self._last_fetch: Optional[datetime] = None

    def _fetch_btc_daily(self) -> Optional[pd.DataFrame]:
        # Cache for 4 hours to avoid excessive API calls
        if self._btc_cache is not None and self._last_fetch is not None:
            if (datetime.utcnow() - self._last_fetch).total_seconds() < 4 * 3600:
                return self._btc_cache
        log.info("Fetching BTC 1D data for regime classification...")
        try:
            df = self.api.get_klines("BTCUSDT", "D", limit=500)
            if df is None or len(df) < 200:
                log.warning(f"Insufficient BTC data: {len(df) if df is not None else 0}")
                return None
            self._btc_cache = df
            self._last_fetch = datetime.utcnow()
            self._regime_cache = {}  # invalidate
            return df
        except Exception as e:
            log.error(f"BTC fetch failed: {e}")
            return None

    def _compute_features(self, btc: pd.DataFrame) -> pd.DataFrame:
        close = btc["close"]
        feats = pd.DataFrame(index=btc.index)
        feats["close"] = close
        feats["ema_200"] = EMA(close, 200)
        feats["ema_50"] = EMA(close, 50)
        feats["ema_50_slope"] = (feats["ema_50"] - feats["ema_50"].shift(20)) / feats["ema_50"].shift(20)
        adx_v, _, _ = ADX(btc, 14)
        feats["adx_14"] = adx_v
        atr_v = ATR(btc, 14)
        feats["atr_pct"] = 100 * atr_v / close
        feats["atr_percentile"] = feats["atr_pct"].rolling(90, min_periods=30).rank(pct=True) * 100
        return feats

    def _label(self, row) -> str:
        if pd.isna(row["ema_200"]) or pd.isna(row["adx_14"]) or pd.isna(row["atr_percentile"]):
            return "UNKNOWN"
        if row["atr_percentile"] > 95:
            return "CRISIS"
        if row["close"] > row["ema_200"] and row["ema_50_slope"] > 0.02 and row["adx_14"] > 20:
            return "BULL"
        if row["close"] < row["ema_200"] and row["ema_50_slope"] < -0.02 and row["adx_14"] > 20:
            return "BEAR"
        return "CHOP"

    def classify(self, ts: Optional[pd.Timestamp] = None) -> str:
        if ts is None:
            ts = pd.Timestamp.utcnow().tz_localize(None)
        day_key = ts.strftime("%Y-%m-%d")
        if day_key in self._regime_cache:
            return self._regime_cache[day_key]
        btc = self._fetch_btc_daily()
        if btc is None: return "UNKNOWN"
        subset = btc[btc.index <= ts]
        if len(subset) < 200:
            regime = "UNKNOWN"
        else:
            feats = self._compute_features(subset)
            regime = self._label(feats.iloc[-1])
        self._regime_cache[day_key] = regime
        return regime

    def classify_series(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Returns regime for every day between start and end. Used for backtest parity."""
        btc = self._fetch_btc_daily()
        if btc is None:
            return pd.Series(dtype=str)
        feats = self._compute_features(btc)
        labels = feats.apply(self._label, axis=1)
        return labels[(labels.index >= start) & (labels.index <= end)]
