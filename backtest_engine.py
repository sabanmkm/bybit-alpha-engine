"""
backtest_engine.py
==================
Improved realistic backtesting engine with fixed Trade dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from rich.console import Console

from config import BacktestConfig, BT, assert_no_2026

console = Console()


class Side(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class Trade:
    symbol: str
    side: Side
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    size: float = 0.0
    qty: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    time_stop_bars: Optional[int] = None
    bars_held: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fee_paid: float = 0.0
    exit_reason: str = ""
    mae: float = 0.0
    mfe: float = 0.0


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    strategy_name: str
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    metrics: Dict[str, float] = field(default_factory=dict)
    config: BacktestConfig = field(default_factory=lambda: BT)


def _apply_slippage(price: float, side: Side, is_entry: bool, slip: float) -> float:
    if side == Side.LONG:
        return price * (1 + slip) if is_entry else price * (1 - slip)
    return price * (1 - slip) if is_entry else price * (1 + slip)


def _calc_metrics(trades: List[Trade], equity: pd.Series, starting_capital: float, timeframe: str) -> Dict[str, float]:
    if not trades or equity.empty:
        return {
            "n_trades": 0, "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
            "profit_factor": 0.0, "max_dd": 1.0, "win_rate": 0.0,
            "avg_trade": 0.0, "total_return": 0.0, "final_equity": starting_capital,
            "max_consec_losses": 0, "ulcer_index": 0.0,
        }

    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    n_trades = len(trades)
    win_rate = len(wins) / n_trades if n_trades else 0.0
    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 1e-12
    profit_factor = gross_profit / gross_loss

    total_return = (equity.iloc[-1] / starting_capital) - 1.0

    rets = equity.pct_change().dropna()
    if len(rets) < 10 or rets.std() == 0:
        sharpe = sortino = 0.0
    else:
        bars_per_year = {"15m": 365*96, "30m": 365*48, "1H": 365*24, "4H": 365*6, "1D": 365}.get(timeframe, 365*24)
        mean_ret = rets.mean()
        std_ret = rets.std()
        sharpe = float(mean_ret / std_ret * np.sqrt(bars_per_year))
        downside = rets[rets < 0]
        sortino = float(mean_ret / downside.std() * np.sqrt(bars_per_year)) if len(downside) > 1 else 0.0

    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(abs(dd.min()))
    calmar = (total_return / max_dd) if max_dd > 1e-8 else 0.0
    ulcer_index = float(np.sqrt((dd**2).mean())) if len(dd) > 0 else 0.0

    loss_streak = 0
    max_streak = 0
    for t in trades:
        if t.pnl <= 0:
            loss_streak += 1
            max_streak = max(max_streak, loss_streak)
        else:
            loss_streak = 0

    return {
        "n_trades": float(n_trades),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "profit_factor": profit_factor,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "avg_trade": float(pnls.mean()),
        "total_return": total_return,
        "final_equity": float(equity.iloc[-1]),
        "max_consec_losses": max_streak,
        "ulcer_index": ulcer_index,
    }


def composite_score(metrics: Dict[str, float], cfg: BacktestConfig = BT) -> float:
    """Stricter composite scoring."""
    sharpe = max(metrics.get("sharpe", 0.0), -5.0)
    sortino = max(metrics.get("sortino", 0.0), -5.0)
    calmar = max(metrics.get("calmar", 0.0), -5.0)
    pf = min(metrics.get("profit_factor", 0.0), 10.0)
    max_dd = metrics.get("max_dd", 1.0)
    win_rate = metrics.get("win_rate", 0.0)
    n_trades = metrics.get("n_trades", 0.0)
    ulcer = metrics.get("ulcer_index", 0.0)

    trade_score = min(n_trades / 500.0, 1.0)
    dd_score = max(0.0, 1.0 - (max_dd / 0.25)) ** 2
    ulcer_score = max(0.0, 1.0 - ulcer / 0.15)

    noise_penalty = 1.0
    if n_trades > 1500 and pf < 1.35:
        noise_penalty = 0.55
    elif n_trades > 800 and pf < 1.25:
        noise_penalty = 0.75

    score = (
        cfg.w_sharpe * (sharpe / 2.0)
        + cfg.w_calmar * (calmar / 4.0)
        + 0.10 * (sortino / 3.0)
        + cfg.w_pf * (pf / 2.0)
        + cfg.w_maxdd * dd_score
        + cfg.w_winrate * (win_rate ** 1.5)
        + cfg.w_trades * trade_score
        + 0.05 * ulcer_score
    ) * noise_penalty

    return float(max(score, 0.0))


class BacktestEngine:
    def __init__(self, cfg: BacktestConfig = BT):
        self.cfg = cfg

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        symbol: str,
        timeframe: str,
        strategy_name: str,
        sl_pct: Optional[float] = None,
        tp_pct: Optional[float] = None,
        trailing_pct: Optional[float] = None,
        time_stop: Optional[int] = None,
        enforce_date_gate: bool = True,
    ) -> BacktestResult:
        if enforce_date_gate:
            assert_no_2026(df)

        df = df.copy()
        signals = signals.reindex(df.index).fillna(0).astype(int)
        entry_signal = signals.shift(1).fillna(0).astype(int)

        capital = self.cfg.starting_capital
        equity_list: List[float] = [capital]
        equity_ts: List[pd.Timestamp] = [df.index[0]]
        trades: List[Trade] = []
        open_trade: Optional[Trade] = None

        fee = self.cfg.fee_rate
        slip = self.cfg.slippage_rate
        risk = self.cfg.risk_per_trade

        for i in range(1, len(df)):
            ts = df.index[i]
            o = df["open"].iloc[i]
            h = df["high"].iloc[i]
            l = df["low"].iloc[i]
            c = df["close"].iloc[i]
            sig = entry_signal.iloc[i]

            if open_trade is not None:
                open_trade.bars_held += 1
                side = open_trade.side
                entry_px = open_trade.entry_price
                exit_px = None
                reason = ""

                if side == Side.LONG:
                    open_trade.mae = min(open_trade.mae, (l - entry_px) / entry_px)
                    open_trade.mfe = max(open_trade.mfe, (h - entry_px) / entry_px)
                else:
                    open_trade.mae = min(open_trade.mae, (entry_px - h) / entry_px)
                    open_trade.mfe = max(open_trade.mfe, (entry_px - l) / entry_px)

                if open_trade.stop_loss is not None:
                    if (side == Side.LONG and l <= open_trade.stop_loss) or (side == Side.SHORT and h >= open_trade.stop_loss):
                        exit_px = open_trade.stop_loss
                        reason = "sl"

                if exit_px is None and open_trade.take_profit is not None:
                    if (side == Side.LONG and h >= open_trade.take_profit) or (side == Side.SHORT and l <= open_trade.take_profit):
                        exit_px = open_trade.take_profit
                        reason = "tp"

                if exit_px is None and trailing_pct is not None:
                    trail = trailing_pct
                    if side == Side.LONG:
                        new_stop = h * (1 - trail)
                        if open_trade.stop_loss is None or new_stop > open_trade.stop_loss:
                            open_trade.stop_loss = new_stop
                        if l <= open_trade.stop_loss:
                            exit_px = open_trade.stop_loss
                            reason = "trail"
                    else:
                        new_stop = l * (1 + trail)
                        if open_trade.stop_loss is None or new_stop < open_trade.stop_loss:
                            open_trade.stop_loss = new_stop
                        if h >= open_trade.stop_loss:
                            exit_px = open_trade.stop_loss
                            reason = "trail"

                if exit_px is None and time_stop is not None and open_trade.bars_held >= time_stop:
                    exit_px = c
                    reason = "time"

                if exit_px is None and i == len(df) - 1:
                    exit_px = c
                    reason = "eod"

                if exit_px is not None:
                    exit_px = _apply_slippage(exit_px, side, is_entry=False, slip=slip)
                    fee_exit = exit_px * open_trade.qty * fee
                    raw_pnl = (exit_px - entry_px) * open_trade.qty if side == Side.LONG else (entry_px - exit_px) * open_trade.qty
                    pnl = raw_pnl - open_trade.fee_paid - fee_exit

                    open_trade.exit_time = ts
                    open_trade.exit_price = exit_px
                    open_trade.pnl = pnl
                    open_trade.pnl_pct = pnl / (entry_px * open_trade.qty) if open_trade.qty else 0.0
                    open_trade.fee_paid += fee_exit
                    open_trade.exit_reason = reason

                    capital += pnl
                    trades.append(open_trade)
                    open_trade = None

            if open_trade is None and sig != 0 and i < len(df) - 1:
                side = Side.LONG if sig > 0 else Side.SHORT
                entry_px = _apply_slippage(o, side, is_entry=True, slip=slip)

                if sl_pct and sl_pct > 0:
                    risk_amount = capital * risk
                    stop_dist = entry_px * sl_pct
                    qty = risk_amount / stop_dist if stop_dist > 0 else 0.0
                else:
                    qty = (capital * risk) / entry_px

                qty = min(qty, (capital * self.cfg.leverage) / entry_px)
                if qty <= 0:
                    equity_list.append(capital)
                    equity_ts.append(ts)
                    continue

                fee_entry = entry_px * qty * fee
                capital -= fee_entry

                stop_loss = entry_px * (1 - sl_pct) if sl_pct and side == Side.LONG else entry_px * (1 + sl_pct) if sl_pct else None
                take_profit = entry_px * (1 + tp_pct) if tp_pct and side == Side.LONG else entry_px * (1 - tp_pct) if tp_pct else None

                open_trade = Trade(
                    symbol=symbol,
                    side=side,
                    entry_time=ts,
                    entry_price=entry_px,
                    size=entry_px * qty,
                    qty=qty,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_stop=trailing_pct,
                    time_stop_bars=time_stop,
                    fee_paid=fee_entry,
                )

            equity_list.append(capital)
            equity_ts.append(ts)

        if open_trade is not None:
            c = df["close"].iloc[-1]
            ts = df.index[-1]
            side = open_trade.side
            entry_px = open_trade.entry_price
            exit_px = _apply_slippage(c, side, is_entry=False, slip=slip)
            fee_exit = exit_px * open_trade.qty * fee
            raw_pnl = (exit_px - entry_px) * open_trade.qty if side == Side.LONG else (entry_px - exit_px) * open_trade.qty
            pnl = raw_pnl - open_trade.fee_paid - fee_exit

            open_trade.exit_time = ts
            open_trade.exit_price = exit_px
            open_trade.pnl = pnl
            open_trade.pnl_pct = pnl / (entry_px * open_trade.qty) if open_trade.qty else 0.0
            open_trade.fee_paid += fee_exit
            open_trade.exit_reason = "eod"
            capital += pnl
            trades.append(open_trade)

        equity_curve = pd.Series(equity_list, index=pd.DatetimeIndex(equity_ts), name="equity")
        metrics = _calc_metrics(trades, equity_curve, self.cfg.starting_capital, timeframe)
        metrics["composite"] = composite_score(metrics, self.cfg)

        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            strategy_name=strategy_name,
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
            config=self.cfg,
        )


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    symbol: str,
    timeframe: str,
    strategy_name: str,
    sl_pct: Optional[float] = 0.02,
    tp_pct: Optional[float] = 0.04,
    trailing_pct: Optional[float] = 0.015,
    time_stop: Optional[int] = None,
    cfg: BacktestConfig = BT,
    enforce_date_gate: bool = True,
) -> BacktestResult:
    engine = BacktestEngine(cfg)
    return engine.run(df, signals, symbol, timeframe, strategy_name,
                      sl_pct, tp_pct, trailing_pct, time_stop, enforce_date_gate)
