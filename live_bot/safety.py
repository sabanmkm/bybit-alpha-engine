"""
BYBIT LIVE BOT - SAFETY MECHANISMS
Kill switches, lock file, duplicate check, edge threshold, sleep prevention.
"""
import os
import sys
import time
import json
import logging
import signal
import ctypes
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import config

log = logging.getLogger(__name__)

# ============================================================
# LOCK FILE (prevent multiple bot instances)
# ============================================================
class LockFile:
    def __init__(self, path: str = None):
        self.path = path or config.LOCK_FILE
        self.acquired = False

    def acquire(self) -> bool:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                pid = data.get("pid")
                # Check if process is still running
                if pid and self._is_pid_running(pid):
                    log.error(f"Bot already running (PID {pid}). Aborting.")
                    return False
                else:
                    log.warning(f"Stale lock file (PID {pid} not running). Removing.")
                    os.remove(self.path)
            except Exception as e:
                log.warning(f"Corrupt lock file, removing: {e}")
                try: os.remove(self.path)
                except: pass
        try:
            with open(self.path, "w") as f:
                json.dump({"pid": os.getpid(), "started_at": datetime.utcnow().isoformat()}, f)
            self.acquired = True
            log.info(f"Lock file acquired: PID {os.getpid()}")
            return True
        except Exception as e:
            log.error(f"Failed to create lock file: {e}")
            return False

    def release(self):
        if self.acquired and os.path.exists(self.path):
            try:
                os.remove(self.path)
                log.info("Lock file released")
            except Exception as e:
                log.warning(f"Failed to release lock file: {e}")
            self.acquired = False

    def _is_pid_running(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
        except Exception:
            return False

# ============================================================
# STATE PERSISTENCE
# ============================================================
class BotState:
    def __init__(self, path: str = None):
        self.path = path or config.STATE_FILE
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"State load failed, using defaults: {e}")
        return {
            "high_water_mark": config.ACCOUNT_SIZE,
            "daily_start_equity": config.ACCOUNT_SIZE,
            "daily_start_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "recent_trades": [],  # list of {timestamp, pnl, symbol, edge_id}
            "halt_until": None,
            "halt_reason": None,
            "started_at": datetime.utcnow().isoformat(),
        }

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            log.error(f"State save failed: {e}")

    def update_hwm(self, equity: float):
        if equity > self.data["high_water_mark"]:
            self.data["high_water_mark"] = equity
            self.save()

    def rotate_daily(self, current_equity: float):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if self.data.get("daily_start_date") != today:
            self.data["daily_start_date"] = today
            self.data["daily_start_equity"] = current_equity
            self.save()
            log.info(f"Daily rotation: new start equity ${current_equity:.2f}")

    def add_trade(self, pnl: float, symbol: str, edge_id: int):
        self.data["recent_trades"].append({
            "ts": datetime.utcnow().isoformat(),
            "pnl": pnl,
            "symbol": symbol,
            "edge_id": edge_id,
        })
        # Keep only last 100
        self.data["recent_trades"] = self.data["recent_trades"][-100:]
        self.save()

    def set_halt(self, hours: int, reason: str):
        until = datetime.utcnow() + timedelta(hours=hours)
        self.data["halt_until"] = until.isoformat()
        self.data["halt_reason"] = reason
        self.save()
        log.warning(f"HALT SET: {reason} until {until.isoformat()}")

    def is_halted(self) -> Tuple[bool, str]:
        halt_until_str = self.data.get("halt_until")
        if not halt_until_str: return False, ""
        try:
            halt_until = datetime.fromisoformat(halt_until_str.replace("Z", ""))
            if datetime.utcnow() < halt_until:
                return True, self.data.get("halt_reason", "Unknown")
            else:
                # Halt expired
                self.data["halt_until"] = None
                self.data["halt_reason"] = None
                self.save()
                return False, ""
        except Exception:
            return False, ""

# ============================================================
# SAFETY CHECKS
# ============================================================
def check_daily_loss(state: BotState, current_equity: float) -> Tuple[bool, str]:
    """Returns (halt_required, reason)"""
    if current_equity < config.DAILY_LOSS_LIMIT:
        return True, f"Equity ${current_equity:.2f} < daily limit ${config.DAILY_LOSS_LIMIT}"
    return False, ""

def check_hwm_loss(state: BotState, current_equity: float) -> Tuple[bool, str]:
    hwm = state.data.get("high_water_mark", config.ACCOUNT_SIZE)
    threshold = hwm * config.HWM_DRAWDOWN_LIMIT
    if current_equity < threshold:
        return True, f"Equity ${current_equity:.2f} < HWM {hwm:.2f} * {config.HWM_DRAWDOWN_LIMIT} = ${threshold:.2f}"
    return False, ""

def check_loss_streak(state: BotState) -> Tuple[bool, str]:
    recent = state.data.get("recent_trades", [])
    if not recent: return False, ""
    cutoff = datetime.utcnow() - timedelta(hours=24)
    recent_24h = [t for t in recent if datetime.fromisoformat(t["ts"].replace("Z","")) > cutoff]
    # Find last N consecutive losses
    consec_losses = 0
    for t in reversed(recent_24h):
        if t["pnl"] < 0:
            consec_losses += 1
        else:
            break
    if consec_losses >= config.LOSS_STREAK_LIMIT:
        return True, f"{consec_losses} consecutive losses in 24h"
    return False, ""

def check_duplicate_position(symbol: str, open_positions: List[Dict]) -> bool:
    """Returns True if duplicate exists (should SKIP)"""
    for p in open_positions:
        if p["symbol"] == symbol:
            return True
    return False

def check_min_notional(notional: float, symbol_info: Dict) -> Tuple[bool, str]:
    min_n = symbol_info.get("min_notional_value", 5.0)
    if notional < min_n:
        return False, f"Notional ${notional:.4f} < min ${min_n}"
    return True, "OK"

def check_edge_threshold(entry_price: float, target_price: float, direction: int,
                         min_edge: float = None) -> Tuple[bool, str]:
    if min_edge is None: min_edge = config.MIN_EDGE_THRESHOLD
    expected_pct = abs(target_price - entry_price) / entry_price
    if expected_pct < min_edge:
        return False, f"Expected edge {expected_pct*100:.3f}% < threshold {min_edge*100:.3f}%"
    return True, f"Edge {expected_pct*100:.3f}% OK"

def check_funding_rate(funding_info: Dict, direction: int) -> Tuple[bool, str]:
    """Skip LONG if funding > threshold (LONG pays). SHORTs are fine."""
    if funding_info is None: return True, "No funding info"
    rate = funding_info.get("funding_rate", 0)
    if direction == 1 and rate > config.FUNDING_CHECK_THRESHOLD:
        return False, f"Funding {rate*100:.4f}% > threshold {config.FUNDING_CHECK_THRESHOLD*100:.4f}% for LONG"
    return True, f"Funding {rate*100:.4f}% OK"

# ============================================================
# WINDOWS SLEEP PREVENTION
# ============================================================
def prevent_windows_sleep():
    if sys.platform != "win32":
        log.info("Not on Windows, sleep prevention skipped")
        return
    try:
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        )
        log.info("Windows sleep prevention enabled")
    except Exception as e:
        log.warning(f"Failed to prevent Windows sleep: {e}")

def allow_windows_sleep():
    if sys.platform != "win32": return
    try:
        ES_CONTINUOUS = 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        log.info("Windows sleep prevention released")
    except Exception:
        pass

# ============================================================
# GRACEFUL SHUTDOWN HANDLER
# ============================================================
_shutdown_handlers = []

def register_shutdown_handler(fn):
    _shutdown_handlers.append(fn)

def handle_shutdown(signum, frame):
    log.warning(f"Shutdown signal {signum} received. Running cleanup...")
    for fn in _shutdown_handlers:
        try: fn()
        except Exception as e: log.error(f"Shutdown handler failed: {e}")
    allow_windows_sleep()
    log.info("Bot exited cleanly.")
    sys.exit(0)

def install_signal_handlers():
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    if sys.platform == "win32":
        try:
            signal.signal(signal.SIGBREAK, handle_shutdown)
        except Exception: pass
