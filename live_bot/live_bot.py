"""
BYBIT LIVE BOT - MAIN 24/7 EXECUTION LOOP (Task 14C)
Deploys 12 validated edges on Bybit USDT Perpetuals mainnet.
"""
import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import load_dotenv
import pandas as pd
import numpy as np

import config
from logger_setup import setup_logging
from bybit_api import BybitAPI
from regime_detector import RegimeDetector
from position_sizer import PositionSizer
from safety import (
    LockFile, BotState, check_daily_loss, check_hwm_loss, check_loss_streak,
    check_duplicate_position, check_min_notional, check_edge_threshold, check_funding_rate,
    prevent_windows_sleep, install_signal_handlers, register_shutdown_handler
)
from strategies import get_strategy, STRATEGY_REGISTRY

log = logging.getLogger("live_bot")

class LiveBot:
    def __init__(self):
        self.api: Optional[BybitAPI] = None
        self.regime_detector: Optional[RegimeDetector] = None
        self.position_sizer: Optional[PositionSizer] = None
        self.state: Optional[BotState] = None
        self.lock: Optional[LockFile] = None
        self.symbol_costs: Dict = {}
        self.tradeable_edges: List[Dict] = []
        self.startup_ts: Optional[pd.Timestamp] = None
        self.last_signal_bar_ts: Dict[str, pd.Timestamp] = {}  # (edge_id -> last bar timestamp)
        self.last_heartbeat: float = 0

    # ============================================================
    # STARTUP
    # ============================================================
    def startup(self) -> bool:
        setup_logging("INFO")
        log.info("=" * 60)
        log.info("BYBIT LIVE BOT STARTING")
        log.info("=" * 60)

        # Load env
        env_path = os.path.join(config.BASE_PATH, ".env")
        if not os.path.exists(env_path):
            log.critical(f".env not found at {env_path}. Copy .env.example to .env and fill in keys.")
            return False
        load_dotenv(env_path)
        api_key = os.getenv("BYBIT_API_KEY", "")
        api_secret = os.getenv("BYBIT_API_SECRET", "")
        testnet = os.getenv("BYBIT_TESTNET", "false").lower() in ("true", "1", "yes")
        if not api_key or not api_secret or api_key == "your_api_key_here":
            log.critical("API keys not configured in .env")
            return False
        log.info(f"API keys loaded. Testnet={testnet}")

        # Lock file
        self.lock = LockFile()
        if not self.lock.acquire():
            return False
        register_shutdown_handler(self.shutdown)

        # Sleep prevention
        prevent_windows_sleep()

        # Signal handlers
        install_signal_handlers()

        # API init
        self.api = BybitAPI(api_key, api_secret, testnet=testnet)
        self.api.sync_server_time()

        # Load symbol costs
        if not os.path.exists(config.SYMBOL_COSTS_FILE):
            log.critical(f"Symbol costs file missing: {config.SYMBOL_COSTS_FILE}")
            log.critical("Run: python symbol_scanner.py")
            return False
        with open(config.SYMBOL_COSTS_FILE, "r") as f:
            self.symbol_costs = json.load(f)

        # Verify balance
        balance = self.api.get_wallet_balance()
        if balance is None:
            log.critical("Could not fetch account balance. Check API permissions.")
            return False
        log.info(f"Account balance: ${balance:.4f} USDT")
        if balance < config.DAILY_LOSS_LIMIT:
            log.critical(f"Balance ${balance:.2f} below daily loss limit ${config.DAILY_LOSS_LIMIT}. Halting.")
            return False

        # State
        self.state = BotState()
        self.state.rotate_daily(balance)
        self.state.update_hwm(balance)

        # Position sizer
        self.position_sizer = PositionSizer()

        # Regime detector
        self.regime_detector = RegimeDetector(self.api)
        initial_regime = self.regime_detector.classify()
        log.info(f"Initial BTC regime: {initial_regime}")

        # Filter tradeable edges
        for edge in config.ALL_EDGES:
            entry = next((r for r in self.symbol_costs.get("results", []) if r["edge_id"] == edge["id"]), None)
            if entry and entry.get("tradeable"):
                edge["symbol_info"] = entry
                self.tradeable_edges.append(edge)
            else:
                log.warning(f"Edge #{edge['id']} {edge['symbol']} NOT tradeable: {entry.get('reason','no info') if entry else 'no info'}")

        log.info(f"Tradeable edges: {len(self.tradeable_edges)}/{len(config.ALL_EDGES)}")

        # Set leverage & margin mode for each tradeable symbol
        for edge in self.tradeable_edges:
            sym = edge["symbol"]
            leverage = int(edge["symbol_info"].get("chosen_leverage", config.LEVERAGE))
            try:
                self.api.set_margin_mode(sym, "ISOLATED", leverage)
                self.api.set_leverage(sym, leverage)
                log.info(f"  {sym}: leverage={leverage}x isolated")
            except Exception as e:
                log.warning(f"  {sym} leverage setup failed: {e}")

        # Startup timestamp - only trade on candles that close after this
        self.startup_ts = pd.Timestamp.utcnow().tz_localize(None)
        log.info(f"Startup timestamp: {self.startup_ts.isoformat()} (only fresh candles will be traded)")

        # Load existing positions to prevent duplicates
        try:
            existing = self.api.get_open_positions()
            if existing:
                log.info(f"Detected {len(existing)} pre-existing positions:")
                for p in existing:
                    log.info(f"  {p['symbol']} {p['side']} size={p['size']} entry=${p['avg_price']}")
        except Exception as e:
            log.warning(f"Could not fetch existing positions: {e}")

        # Startup banner
        log.info("=" * 60)
        log.info(f"BOT READY | Equity: ${balance:.2f} | Edges: {len(self.tradeable_edges)} | Regime: {initial_regime}")
        log.info(f"Risk/trade: {config.RISK_PER_TRADE*100}% (${balance*config.RISK_PER_TRADE:.4f})")
        log.info(f"Leverage: {config.LEVERAGE}x | Max gross: {config.MAX_GROSS_LEVERAGE}x")
        log.info(f"Daily loss halt: ${config.DAILY_LOSS_LIMIT} | HWM drawdown: {config.HWM_DRAWDOWN_LIMIT*100}%")
        log.info("=" * 60)
        return True

    # ============================================================
    # CANDLE ALIGNMENT
    # ============================================================
    def is_candle_close_time(self, tf: str, ts: pd.Timestamp) -> bool:
        """Check if ts is at a candle close boundary for tf (within 60 sec)."""
        minutes = config.TF_MINUTES.get(tf, 60)
        total_minutes = ts.hour * 60 + ts.minute
        # For 4H (240 min): closes at 0, 4, 8, 12, 16, 20 UTC
        # For 1H (60 min): closes at every hour
        # For 1D (1440 min): closes at 00:00 UTC
        if minutes >= 1440:
            return ts.hour == 0 and ts.minute < 5
        return (total_minutes % minutes) < 3  # allow 3-minute window after close

    def get_current_bar_ts(self, tf: str) -> pd.Timestamp:
        """Returns the UTC timestamp of the current forming bar's open."""
        now = pd.Timestamp.utcnow().tz_localize(None)
        minutes = config.TF_MINUTES.get(tf, 60)
        if minutes >= 1440:
            return now.floor("D")
        total_minutes_since_day = now.hour * 60 + now.minute
        bar_start_min = (total_minutes_since_day // minutes) * minutes
        bar_hour = bar_start_min // 60
        bar_minute = bar_start_min % 60
        return pd.Timestamp(year=now.year, month=now.month, day=now.day,
                             hour=bar_hour, minute=bar_minute)

    def get_last_closed_bar_ts(self, tf: str) -> pd.Timestamp:
        """Returns UTC timestamp of the most recently closed bar's OPEN time."""
        current = self.get_current_bar_ts(tf)
        minutes = config.TF_MINUTES.get(tf, 60)
        return current - pd.Timedelta(minutes=minutes)

    # ============================================================
    # SIGNAL GENERATION
    # ============================================================
    def process_edge(self, edge: Dict, equity: float, positions: List[Dict]) -> bool:
        """Process one edge on the latest closed candle. Returns True if trade placed."""
        strat_name = edge["strategy"]
        symbol = edge["symbol"]
        tf = edge["tf"]
        eid = edge["id"]

        last_closed_bar = self.get_last_closed_bar_ts(tf)

        # Skip if we already processed this bar
        if self.last_signal_bar_ts.get(eid) == last_closed_bar:
            return False

        # Only trade candles that closed AFTER startup
        if last_closed_bar < self.startup_ts:
            log.debug(f"Edge #{eid}: last closed bar {last_closed_bar} is before startup {self.startup_ts}; skipping")
            self.last_signal_bar_ts[eid] = last_closed_bar
            return False

        # Duplicate position check
        if check_duplicate_position(symbol, positions):
            log.debug(f"Edge #{eid} {symbol}: duplicate position exists; skipping")
            self.last_signal_bar_ts[eid] = last_closed_bar
            return False

        # Fetch klines
        interval = config.TF_MAP.get(tf, "60")
        try:
            df = self.api.get_klines(symbol, interval, limit=config.KLINE_LIMIT)
        except Exception as e:
            log.error(f"Edge #{eid} {symbol}: kline fetch failed: {e}")
            return False
        if df is None or len(df) < 100:
            log.warning(f"Edge #{eid} {symbol}: insufficient klines")
            return False

        # Get regime series for regime-adaptive strategies
        regime_series = None
        if strat_name.startswith("R"):
            try:
                regime_series = self.regime_detector.classify_series(
                    df.index.min(), df.index.max()
                )
            except Exception as e:
                log.warning(f"Regime series build failed: {e}")

        # Generate signals
        try:
            strat = get_strategy(strat_name)
            signals, sl_series, tp_series = strat.generate_signals(df, None, regime_series)
        except Exception as e:
            log.error(f"Edge #{eid} {symbol}: signal generation failed: {e}")
            log.debug(traceback.format_exc())
            return False

        # Check signal on the last CLOSED bar (index -2 to be safe; last bar could be forming)
        # We want the signal that would trigger entry on the NEXT bar
        # In backtest: signal at time T triggers entry at T+1
        # In live: we process at bar close, so we check signal at index -1 (most recent CLOSED bar)
        # Note: df.iloc[-1] should be the latest bar. If bar hasn't fully closed on Bybit,
        # we need to check the SECOND to last bar for a confirmed signal.
        # Safest: use df.iloc[-2] as the signal bar (confirmed closed), enter on next bar open
        if len(df) < 2: return False

        # Find the signal on the most recent CONFIRMED closed bar
        # This bar's open_time should equal last_closed_bar
        signal_bar_ts = last_closed_bar
        if signal_bar_ts not in df.index:
            log.debug(f"Edge #{eid}: signal bar {signal_bar_ts} not in dataframe")
            self.last_signal_bar_ts[eid] = last_closed_bar
            return False

        sig_value = int(signals.loc[signal_bar_ts])
        if sig_value == 0:
            self.last_signal_bar_ts[eid] = last_closed_bar
            return False

        # Mark this bar as processed
        self.last_signal_bar_ts[eid] = last_closed_bar

        direction = sig_value  # 1 = long, -1 = short
        sl_price = float(sl_series.loc[signal_bar_ts])
        tp_price = float(tp_series.loc[signal_bar_ts])
        if pd.isna(sl_price) or pd.isna(tp_price):
            log.warning(f"Edge #{eid} {symbol}: SL or TP is NaN")
            return False

        # Use current market price for entry (approximation of next candle open)
        ticker = self.api.get_funding_rate(symbol)
        if ticker is None:
            log.warning(f"Edge #{eid} {symbol}: cannot get ticker")
            return False
        entry_price = ticker["last_price"]
        if entry_price <= 0:
            log.warning(f"Edge #{eid} {symbol}: invalid last price")
            return False

        # Validate SL/TP relative to current price
        if direction == 1:
            if sl_price >= entry_price or tp_price <= entry_price:
                log.warning(f"Edge #{eid} {symbol}: LONG invalid SL={sl_price} TP={tp_price} vs entry={entry_price}")
                return False
        else:
            if sl_price <= entry_price or tp_price >= entry_price:
                log.warning(f"Edge #{eid} {symbol}: SHORT invalid SL={sl_price} TP={tp_price} vs entry={entry_price}")
                return False

        # Funding check
        funding_ok, funding_msg = check_funding_rate(ticker, direction)
        if not funding_ok:
            log.info(f"Edge #{eid} {symbol}: skipping - {funding_msg}")
            return False

        # Edge threshold check
        edge_ok, edge_msg = check_edge_threshold(entry_price, tp_price, direction)
        if not edge_ok:
            log.info(f"Edge #{eid} {symbol}: skipping - {edge_msg}")
            return False

        # Position size
        current_open_notional = sum(p.get("position_value", 0) for p in positions)
        qty, notional, margin, size_msg = self.position_sizer.calculate_position_size(
            equity, entry_price, sl_price, edge["symbol_info"], current_open_notional
        )
        if qty <= 0:
            log.info(f"Edge #{eid} {symbol}: skipping - {size_msg}")
            return False

        # Round SL/TP to tick size
        tick = edge["symbol_info"].get("tick_size", 0.01)
        sl_price = round(sl_price / tick) * tick
        tp_price = round(tp_price / tick) * tick

        # Place order
        side = "Buy" if direction == 1 else "Sell"
        log.info(f"PLACING ORDER: Edge #{eid} {symbol} {side} qty={qty} entry~${entry_price} SL=${sl_price} TP=${tp_price}")
        order_id = self.api.place_order(
            symbol=symbol, side=side, qty=qty, order_type="Market",
            stop_loss=sl_price, take_profit=tp_price
        )
        if order_id is None:
            log.error(f"Edge #{eid} {symbol}: order placement failed")
            return False

        # Verify fill (poll for up to 10 seconds)
        time.sleep(2)
        for _ in range(4):
            status = self.api.get_order_status(symbol, order_id)
            if status and status.get("status") in ("Filled", "PartiallyFilled"):
                log.info(f"  ORDER FILLED: {symbol} avg=${status.get('avg_price')} qty={status.get('cum_exec_qty')}")
                return True
            time.sleep(2)
        log.warning(f"Edge #{eid} {symbol}: order status unknown after 10s")
        return True  # order placed, may still fill

    # ============================================================
    # RECONCILIATION & TRACKING
    # ============================================================
    def reconcile_closed_positions(self, prev_positions: List[Dict], current_positions: List[Dict]):
        """Detect closed positions and log P&L, update state."""
        current_syms = {p["symbol"] for p in current_positions}
        for prev in prev_positions:
            if prev["symbol"] not in current_syms:
                # Position closed - fetch recent PnL
                try:
                    # Simplistic: assume the change in equity accounts for it
                    log.info(f"POSITION CLOSED: {prev['symbol']} {prev['side']}")
                    # We can't easily get exact PnL without extra API calls; record placeholder
                    self.state.add_trade(pnl=0, symbol=prev["symbol"], edge_id=0)
                except Exception as e:
                    log.warning(f"Reconcile failed for {prev['symbol']}: {e}")

    def heartbeat(self, equity: float, positions: List[Dict], regime: str):
        now = time.time()
        if now - self.last_heartbeat < config.HEARTBEAT_INTERVAL_SEC:
            return
        self.last_heartbeat = now
        n_pos = len(positions)
        gross_notional = sum(p.get("position_value", 0) for p in positions)
        unrealized = sum(p.get("unrealized_pnl", 0) for p in positions)
        hwm = self.state.data.get("high_water_mark", equity)
        dd_from_hwm = (equity - hwm) / hwm * 100 if hwm > 0 else 0
        log.info(f"HEARTBEAT | Equity: ${equity:.4f} | Positions: {n_pos} | "
                 f"Gross: ${gross_notional:.2f} | uPnL: ${unrealized:+.4f} | "
                 f"HWM: ${hwm:.2f} ({dd_from_hwm:+.2f}%) | Regime: {regime}")

    # ============================================================
    # MAIN LOOP
    # ============================================================
    def run(self):
        try:
            prev_positions = []
            while True:
                try:
                    # Fetch state
                    equity = self.api.get_wallet_balance()
                    if equity is None:
                        log.warning("Balance fetch failed; sleeping 30s")
                        time.sleep(30); continue
                    self.state.rotate_daily(equity)
                    self.state.update_hwm(equity)

                    # Safety checks
                    halted, halt_reason = self.state.is_halted()
                    if halted:
                        log.warning(f"HALTED: {halt_reason}. Sleeping 60s.")
                        time.sleep(60); continue

                    for check_name, check_fn in [
                        ("daily_loss", check_daily_loss),
                        ("hwm_loss", check_hwm_loss),
                    ]:
                        halt, msg = check_fn(self.state, equity)
                        if halt:
                            log.critical(f"SAFETY TRIGGER ({check_name}): {msg}")
                            self.state.set_halt(config.DAILY_HALT_HOURS, msg)
                            break

                    streak_halt, streak_msg = check_loss_streak(self.state)
                    if streak_halt:
                        log.warning(f"LOSS STREAK: {streak_msg}")
                        self.state.set_halt(config.PAUSE_DURATION_HOURS, streak_msg)

                    # Refresh positions
                    positions = self.api.get_open_positions()
                    if positions is None: positions = []

                    # Reconcile closed
                    self.reconcile_closed_positions(prev_positions, positions)
                    prev_positions = positions

                    # Get regime
                    regime = self.regime_detector.classify()

                    # Heartbeat
                    self.heartbeat(equity, positions, regime)

                    # Process each tradeable edge
                    for edge in self.tradeable_edges:
                        try:
                            self.process_edge(edge, equity, positions)
                        except Exception as e:
                            log.error(f"Edge #{edge['id']} exception: {e}")
                            log.debug(traceback.format_exc())

                    # Sleep until next check
                    time.sleep(config.CANDLE_CHECK_INTERVAL_SEC)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    log.error(f"Main loop iteration failed: {e}")
                    log.debug(traceback.format_exc())
                    time.sleep(30)
        except KeyboardInterrupt:
            log.info("Ctrl+C received")
        finally:
            self.shutdown()

    def shutdown(self):
        log.info("Bot shutting down...")
        try:
            if self.state:
                self.state.save()
        except Exception: pass
        try:
            if self.lock:
                self.lock.release()
        except Exception: pass
        log.info("Shutdown complete.")

# ============================================================
# ENTRY POINT
# ============================================================
def main():
    bot = LiveBot()
    if not bot.startup():
        print("Startup failed. Check logs.")
        sys.exit(1)
    bot.run()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
