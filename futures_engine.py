"""
CRYPTO FUTURES BACKTESTING ENGINE v1.0
Models real Bybit USDT Perpetual Futures mechanics:
  1. 8-Hour Funding Rate charges (long pays short / short pays long)
  2. Mark Price vs Last Price (liquidation on mark, fills on last)
  3. Isolated Margin Liquidation at 10x leverage
  4. Volume-Dependent Slippage Model
  5. Maker/Taker Fee Differentiation
  6. Cumulative Funding Cost Tracking
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\BybitBacktest")

FUNDING_DIR = Path(r"C:\BybitBacktest\data\funding_rates")

@dataclass
class FuturesTrade:
    symbol: str
    timeframe: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int            # 1=Long, -1=Short
    entry_price: float
    exit_price: float
    size: float               # Contracts
    notional: float
    margin_used: float
    leverage: int
    gross_pnl: float
    funding_cost: float       # Cumulative funding paid/received
    fee_paid: float           # Entry + Exit taker fees
    net_pnl: float            # Gross - Fees - Funding
    pnl_pct: float            # Net PnL / Margin
    exit_reason: str          # TP, SL, LIQUIDATION, TIME_STOP, END_OF_DATA
    bars_held: int

@dataclass
class FuturesBacktestResult:
    symbol: str
    timeframe: str
    strategy_name: str
    trades: list = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    net_profit: float = 0.0
    net_profit_pct: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    total_fees: float = 0.0
    total_funding: float = 0.0
    liquidation_count: int = 0
    composite_score: float = 0.0


class FundingRateModel:
    """Loads and interpolates historical 8h funding rates."""

    def __init__(self):
        self._cache = {}

    def get_funding_rate(self, symbol: str, timestamp: pd.Timestamp) -> float:
        """Returns the funding rate applicable at a given timestamp."""
        base = symbol.split("/")[0].replace(":", "")

        if base not in self._cache:
            fpath = FUNDING_DIR / f"{base}_funding.parquet"
            if fpath.exists():
                try:
                    df = pd.read_parquet(fpath)
                    self._cache[base] = df["funding_rate"]
                except Exception:
                    self._cache[base] = None
            else:
                self._cache[base] = None

        rates = self._cache.get(base)
        if rates is None or len(rates) == 0:
            return 0.0001  # Default neutral funding rate (0.01%)

        # Find the most recent funding rate at or before the timestamp
        try:
            idx = rates.index.asof(timestamp)
            if pd.isna(idx):
                return 0.0001
            rate = float(rates.loc[idx])
            return rate if not np.isnan(rate) else 0.0001
        except Exception:
            return 0.0001


class FuturesEngine:
    """
    Realistic Crypto Futures Backtesting Engine.
    Models Bybit USDT Perpetual mechanics with funding, liquidation, and slippage.
    """

    def __init__(self,
                 capital: float = 10000.0,
                 leverage: int = 10,
                 taker_fee: float = 0.00055,
                 maker_fee: float = 0.0002,
                 base_slippage: float = 0.0003,
                 risk_per_trade: float = 0.01):

        self.capital = capital
        self.leverage = leverage
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.base_slippage = base_slippage
        self.risk_per_trade = risk_per_trade
        self.funding_model = FundingRateModel()

        # Maintenance margin rate (Bybit standard for most altcoins)
        self.maintenance_margin_rate = 0.005  # 0.5%

    def _calculate_liquidation_price(self, entry_price: float, direction: int) -> float:
        """
        Calculates the exact liquidation price for isolated margin.
        Long:  liq = entry * (1 - 1/leverage + maintenance_margin)
        Short: liq = entry * (1 + 1/leverage - maintenance_margin)
        """
        if direction == 1:  # Long
            return entry_price * (1.0 - 1.0/self.leverage + self.maintenance_margin_rate)
        else:  # Short
            return entry_price * (1.0 + 1.0/self.leverage - self.maintenance_margin_rate)

    def _volume_slippage(self, volume: float, avg_volume: float) -> float:
        """
        Volume-dependent slippage model.
        Low volume = higher slippage. High volume = lower slippage.
        """
        if avg_volume <= 0:
            return self.base_slippage * 2.0

        vol_ratio = volume / avg_volume
        if vol_ratio > 2.0:
            return self.base_slippage * 0.5   # High liquidity = less slippage
        elif vol_ratio > 1.0:
            return self.base_slippage * 0.8
        elif vol_ratio > 0.5:
            return self.base_slippage * 1.2
        else:
            return self.base_slippage * 2.0   # Low liquidity = more slippage

    def _apply_funding(self, position_direction: int, position_notional: float,
                       current_time: pd.Timestamp, symbol: str,
                       last_funding_time: pd.Timestamp) -> tuple[float, pd.Timestamp]:
        """
        Calculates funding charges since the last funding payment.
        Bybit funding settles every 8 hours (00:00, 08:00, 16:00 UTC).
        """
        # Determine how many 8h funding periods have passed
        funding_hours = [0, 8, 16]
        funding_cost = 0.0
        new_funding_time = last_funding_time

        # Check each 8h window between last_funding_time and current_time
        check_time = last_funding_time
        while True:
            # Find next funding settlement
            next_funding = None
            for h in funding_hours:
                candidate = check_time.normalize() + pd.Timedelta(hours=h)
                if candidate > check_time and candidate <= current_time:
                    if next_funding is None or candidate < next_funding:
                        next_funding = candidate

            if next_funding is None:
                # Try next day
                for h in funding_hours:
                    candidate = (check_time.normalize() + pd.Timedelta(days=1)) + pd.Timedelta(hours=h)
                    if candidate <= current_time:
                        if next_funding is None or candidate < next_funding:
                            next_funding = candidate

            if next_funding is None or next_funding <= last_funding_time:
                break

            # Get funding rate at this settlement time
            rate = self.funding_model.get_funding_rate(symbol, next_funding)

            # Funding payment: Long pays positive rate, Short receives positive rate
            # cost = notional * rate * direction_multiplier
            if position_direction == 1:  # Long
                funding_cost += position_notional * rate  # Long pays when rate > 0
            else:  # Short
                funding_cost -= position_notional * rate  # Short receives when rate > 0

            new_funding_time = next_funding
            check_time = next_funding

        return funding_cost, new_funding_time

    def run_backtest(self,
                     df: pd.DataFrame,
                     signals: pd.Series,
                     symbol: str = "UNKNOWN",
                     timeframe: str = "4H",
                     strategy_name: str = "Strategy",
                     sl_atr_mult: float = 2.0,
                     tp_atr_mult: float = 4.0,
                     time_stop_bars: int = 0,
                     trailing_stop_atr: float = 0.0,
                     allow_short: bool = True) -> FuturesBacktestResult:

        if len(df) < 50 or signals.empty:
            return FuturesBacktestResult(symbol=symbol, timeframe=timeframe, strategy_name=strategy_name)

        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        volumes = df["volume"].values
        times = df.index
        sig_vals = signals.values
        n_bars = len(df)

        # Precompute ATR
        tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))
        tr[0] = highs[0] - lows[0]
        atrs = pd.Series(tr).rolling(14, min_periods=1).mean().values

        # Precompute mark price (EMA-3 of close as proxy for Bybit mark price)
        mark_prices = pd.Series(closes).ewm(span=3, adjust=False).mean().values

        # Precompute average volume for slippage model
        avg_volumes = pd.Series(volumes).rolling(20, min_periods=1).mean().values

        # Find signal bars
        sig_indices = np.flatnonzero(sig_vals != 0)
        if len(sig_indices) == 0:
            return FuturesBacktestResult(symbol=symbol, timeframe=timeframe, strategy_name=strategy_name)

        capital = self.capital
        trades = []
        curr_bar = 0

        for s_idx in sig_indices:
            if s_idx < curr_bar or s_idx >= n_bars - 1:
                continue

            sig = sig_vals[s_idx]
            if sig == -1 and not allow_short:
                continue

            entry_bar = s_idx + 1
            raw_entry = opens[entry_bar]

            # Apply volume-dependent slippage to entry
            slip = self._volume_slippage(volumes[entry_bar], avg_volumes[entry_bar])
            entry_price = raw_entry * (1 + slip) if sig == 1 else raw_entry * (1 - slip)

            curr_atr = atrs[s_idx]

            # Position sizing (risk-based)
            risk_amt = capital * self.risk_per_trade
            sl_dist = max(sl_atr_mult * curr_atr, entry_price * 0.003)
            size = risk_amt / sl_dist
            notional = size * entry_price
            margin = notional / self.leverage

            if margin > capital * 0.9:
                continue  # Insufficient margin

            # SL / TP levels
            if sig == 1:
                sl_price = entry_price - sl_dist
                tp_price = entry_price + tp_atr_mult * curr_atr if tp_atr_mult > 0 else 0
            else:
                sl_price = entry_price + sl_dist
                tp_price = entry_price - tp_atr_mult * curr_atr if tp_atr_mult > 0 else 0

            liq_price = self._calculate_liquidation_price(entry_price, sig)

            # Entry fee (taker)
            entry_fee = notional * self.taker_fee
            capital -= (margin + entry_fee)

            # Track funding
            last_funding_time = times[entry_bar]
            cumulative_funding = 0.0

            # Scan forward for exit
            exit_bar = n_bars - 1
            exit_price = closes[-1]
            exit_reason = "END_OF_DATA"

            max_scan = min(n_bars, entry_bar + time_stop_bars) if time_stop_bars > 0 else n_bars

            for b in range(entry_bar, max_scan):
                b_high = highs[b]
                b_low = lows[b]
                b_open = opens[b]
                b_mark = mark_prices[b]
                b_time = times[b]

                # Apply funding every 8 hours
                funding_charge, last_funding_time = self._apply_funding(
                    sig, notional, b_time, symbol, last_funding_time
                )
                cumulative_funding += funding_charge

                # Check LIQUIDATION first (on mark price)
                if sig == 1 and b_mark <= liq_price:
                    exit_bar = b
                    exit_price = liq_price
                    exit_reason = "LIQUIDATION"
                    break
                elif sig == -1 and b_mark >= liq_price:
                    exit_bar = b
                    exit_price = liq_price
                    exit_reason = "LIQUIDATION"
                    break

                # Check SL (on last price high/low)
                if sig == 1 and b_low <= sl_price:
                    exit_bar = b
                    exit_price = min(b_open, sl_price)
                    exit_reason = "SL"
                    break
                elif sig == -1 and b_high >= sl_price:
                    exit_bar = b
                    exit_price = max(b_open, sl_price)
                    exit_reason = "SL"
                    break

                # Check TP
                if tp_price > 0:
                    if sig == 1 and b_high >= tp_price:
                        exit_bar = b
                        exit_price = max(b_open, tp_price)
                        exit_reason = "TP"
                        break
                    elif sig == -1 and b_low <= tp_price:
                        exit_bar = b
                        exit_price = min(b_open, tp_price)
                        exit_reason = "TP"
                        break

                # Check Time Stop
                if time_stop_bars > 0 and (b - entry_bar) >= time_stop_bars:
                    exit_bar = b
                    exit_price = closes[b]
                    exit_reason = "TIME_STOP"
                    break

            # Calculate PnL
            if exit_reason == "LIQUIDATION":
                # On liquidation, you lose your entire margin minus maintenance
                gross_pnl = -(margin * (1 - self.maintenance_margin_rate))
                exit_fee = 0.0  # Liquidation fee is taken from remaining margin
            else:
                gross_pnl = size * (exit_price - entry_price) if sig == 1 else size * (entry_price - exit_price)
                exit_fee = size * exit_price * self.taker_fee

            total_fees = entry_fee + exit_fee
            net_pnl = gross_pnl - total_fees - cumulative_funding
            capital += (margin + net_pnl)
            pnl_pct = (net_pnl / margin) * 100.0 if margin > 0 else 0.0

            trades.append(FuturesTrade(
                symbol=symbol, timeframe=timeframe,
                entry_time=times[entry_bar], exit_time=times[exit_bar],
                direction=sig, entry_price=entry_price, exit_price=exit_price,
                size=size, notional=notional, margin_used=margin,
                leverage=self.leverage,
                gross_pnl=gross_pnl, funding_cost=cumulative_funding,
                fee_paid=total_fees, net_pnl=net_pnl, pnl_pct=pnl_pct,
                exit_reason=exit_reason, bars_held=exit_bar - entry_bar
            ))

            curr_bar = exit_bar + 1

        # Compute metrics
        total_trades = len(trades)
        if total_trades == 0:
            return FuturesBacktestResult(symbol=symbol, timeframe=timeframe, strategy_name=strategy_name)

        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]
        win_rate = len(wins) / total_trades
        gross_profit = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        pf = gross_profit / (gross_loss + 1e-9) if gross_loss > 0 else 2.0

        pnl_arr = np.array([t.net_pnl for t in trades])
        cum_eq = self.capital + np.cumsum(pnl_arr)
        roll_max = np.maximum.accumulate(cum_eq)
        dd = (cum_eq - roll_max) / roll_max
        max_dd = abs(float(np.min(dd))) * 100.0

        days = max((times[-1] - times[0]).days, 1)
        final_cap = self.capital + pnl_arr.sum()
        cagr = ((final_cap / self.capital) ** (365.0 / days) - 1.0) * 100.0 if final_cap > 0 else -100.0

        pnl_pcts = np.array([t.pnl_pct for t in trades])
        t_yr = (total_trades / days) * 365.0
        ret_std = np.std(pnl_pcts) + 1e-9
        sharpe = float((np.mean(pnl_pcts) / ret_std) * np.sqrt(t_yr)) if t_yr > 0 else 0.0
        downside = np.std([r for r in pnl_pcts if r < 0] + [0.0]) + 1e-9
        sortino = float((np.mean(pnl_pcts) / downside) * np.sqrt(t_yr)) if t_yr > 0 else 0.0
        calmar = float(cagr / max_dd) if max_dd > 0 else 0.0

        liq_count = sum(1 for t in trades if t.exit_reason == "LIQUIDATION")
        total_fees = sum(t.fee_paid for t in trades)
        total_funding = sum(t.funding_cost for t in trades)

        # Composite score
        norm_sharpe = np.clip(sharpe / 2.0, -1.0, 3.0)
        norm_calmar = np.clip(calmar / 3.0, -1.0, 3.0)
        norm_pf = np.clip((pf - 1.0), -1.0, 3.0)
        dd_penalty = 1.0 - (max_dd / 25.0)
        norm_wr = (win_rate - 0.50) * 2.0
        trade_penalty = min(total_trades / 50, 1.0) if total_trades < 50 else 1.0
        liq_penalty = max(0, 1.0 - liq_count * 0.1)
        composite = float((0.30*norm_sharpe + 0.20*norm_calmar + 0.15*norm_pf +
                           0.15*dd_penalty + 0.10*norm_wr + 0.10*trade_penalty) * liq_penalty)

        # Equity curve
        trade_df = pd.DataFrame([{"t": t.exit_time, "pnl": t.net_pnl} for t in trades]).set_index("t")
        daily_pnl = trade_df.resample("1D").sum().reindex(
            pd.date_range(times[0], times[-1], freq="1D", tz="UTC")
        ).fillna(0.0)
        eq_series = self.capital + daily_pnl["pnl"].cumsum()

        return FuturesBacktestResult(
            symbol=symbol, timeframe=timeframe, strategy_name=strategy_name,
            trades=trades, equity_curve=eq_series,
            total_trades=total_trades, win_rate=win_rate,
            profit_factor=pf, net_profit=final_cap - self.capital,
            net_profit_pct=((final_cap - self.capital) / self.capital) * 100.0,
            cagr=cagr, sharpe_ratio=sharpe, sortino_ratio=sortino,
            calmar_ratio=calmar, max_drawdown_pct=max_dd,
            total_fees=total_fees, total_funding=total_funding,
            liquidation_count=liq_count, composite_score=composite
        )
