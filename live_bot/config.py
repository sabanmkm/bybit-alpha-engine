"""
BYBIT LIVE BOT - CONFIGURATION
All constants and settings in one place.
"""
import os
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
BASE_PATH = r"C:\BybitLiveBot"
LOG_PATH = os.path.join(BASE_PATH, "logs")
DATA_PATH = os.path.join(BASE_PATH, "data")
STATE_PATH = os.path.join(BASE_PATH, "state")
LOCK_FILE = os.path.join(BASE_PATH, "bot.lock")
SYMBOL_COSTS_FILE = os.path.join(BASE_PATH, "symbol_costs.json")
STATE_FILE = os.path.join(STATE_PATH, "bot_state.json")

os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(STATE_PATH, exist_ok=True)

# ============================================================
# ACCOUNT & RISK
# ============================================================
ACCOUNT_SIZE = 15.0                     # $15 USDT starting capital
LEVERAGE = 10                            # 10x isolated per position
RISK_PER_TRADE = 0.01                    # 1% of equity per trade = $0.15
FEES = 0.00055                           # Bybit taker fee per side
SLIPPAGE = 0.0008                        # 0.08% per side (conservative)
MAX_GROSS_LEVERAGE = 3.0                 # Total open notional <= 3x equity

# ============================================================
# SAFETY LIMITS
# ============================================================
DAILY_LOSS_LIMIT = 13.50                 # Halt trading if equity < $13.50
HWM_DRAWDOWN_LIMIT = 0.85                # Halt if equity < 85% of high-water mark
LOSS_STREAK_LIMIT = 3                    # 3+ losses in 24h -> pause 12h
PAUSE_DURATION_HOURS = 12                # How long to pause on loss streak
DAILY_HALT_HOURS = 24                    # Halt duration on daily loss trigger
MIN_EDGE_THRESHOLD = 0.0022              # 2x round-trip fees (2 * 0.11%)
FUNDING_CHECK_THRESHOLD = 0.0005         # Skip LONG if funding > 0.05%

# ============================================================
# 12 DEPLOY-READY EDGES
# ============================================================
DEPLOY_SYMBOLS = [
    {"id":1, "strategy":"S30a_DOW_Seasonality",     "symbol":"ATOMUSDT", "tf":"4H", "priority":"DEPLOY"},
    {"id":2, "strategy":"S30a_DOW_Seasonality",     "symbol":"ADAUSDT",  "tf":"4H", "priority":"DEPLOY"},
    {"id":3, "strategy":"S30a_DOW_Seasonality",     "symbol":"BSVUSDT",  "tf":"4H", "priority":"DEPLOY"},
    {"id":4, "strategy":"S30a_DOW_Seasonality",     "symbol":"AVAXUSDT", "tf":"4H", "priority":"DEPLOY"},
    {"id":5, "strategy":"S30a_DOW_Seasonality",     "symbol":"QTUMUSDT", "tf":"4H", "priority":"DEPLOY"},
    {"id":6, "strategy":"R13_ATR_Range_Expansion",  "symbol":"ETHUSDT",  "tf":"4H", "priority":"DEPLOY"},
    {"id":7, "strategy":"R01_EMA_Cross",            "symbol":"OPUSDT",   "tf":"4H", "priority":"DEPLOY"},
    {"id":8, "strategy":"R15_Volume_Breakout",      "symbol":"OPUSDT",   "tf":"4H", "priority":"DEPLOY"},
]
MONITOR_SYMBOLS = [
    {"id":9,  "strategy":"S30a_DOW_Seasonality",       "symbol":"ETCUSDT",  "tf":"4H", "priority":"MONITOR"},
    {"id":10, "strategy":"S30b_MidWeek_MeanReversion", "symbol":"TWTUSDT",  "tf":"4H", "priority":"MONITOR"},
    {"id":11, "strategy":"S30d_NY_London_Momentum",    "symbol":"DOTUSDT",  "tf":"1H", "priority":"MONITOR"},
    {"id":12, "strategy":"S30d_NY_London_Momentum",    "symbol":"XLMUSDT",  "tf":"1H", "priority":"MONITOR"},
]
ALL_EDGES = DEPLOY_SYMBOLS + MONITOR_SYMBOLS
ALL_SYMBOLS = sorted(set(e["symbol"] for e in ALL_EDGES))
BTC_ANCHOR_SYMBOL = "BTCUSDT"
BTC_ANCHOR_TF = "D"

# ============================================================
# EXECUTION TIMING
# ============================================================
CANDLE_CHECK_INTERVAL_SEC = 30           # Check for new candle every 30s
HEARTBEAT_INTERVAL_SEC = 300             # Log heartbeat every 5 min
KLINE_LIMIT = 300                        # Number of candles to fetch for indicators

# ============================================================
# BYBIT API
# ============================================================
BYBIT_MAINNET_URL = "https://api.bybit.com"
BYBIT_TESTNET_URL = "https://api-testnet.bybit.com"
API_TIMEOUT_SEC = 60
API_MAX_RETRIES = 5
API_BACKOFF_BASE_SEC = 2

# ============================================================
# TIMEFRAME MAPPING (Bybit v5 interval codes)
# ============================================================
TF_MAP = {
    "15m": "15",
    "30m": "30",
    "1H": "60",
    "4H": "240",
    "1D": "D",
    "D": "D",
}
TF_MINUTES = {
    "15m": 15, "30m": 30, "1H": 60, "4H": 240, "1D": 1440, "D": 1440
}
