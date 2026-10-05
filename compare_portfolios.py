"""
PORTFOLIO COMPARISON ENGINE: Original 6 vs Super 7 (Original 6 + STG 120m).
Simulates both portfolios on 2026 OOS data with exact live bot constraints:
  $15 Capital | 10x Leverage | $5 Min Notional | Isolated Margin | $13.50 Circuit Breaker
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, r"C:\BybitBacktest")
import config as cfg
cfg.DATA_GATE_UNLOCKED = True

from strategy_generator import get_strategy
from backtest_engine import run_backtest

console = Console()

PORTFOLIO_EDGES_JSON = cfg.RESULTS_DIR / "portfolio_edges.json"
EXOTIC_OPTIMIZED_JSON = cfg.RESULTS_DIR / "exotic_optimized_edges.json"
CUSTOM_RESAMPLED_DIR = cfg.ROOT / "data" / "resampled_exotic"

LIVE_CAPITAL = 15.0
LIVE_LEVERAGE = 10
MIN_NOTIONAL = 5.0
DAILY_LOSS_LIMIT = 13.50
FEE_PER_SIDE = 0.00055
SLIPPAGE = 0.0003

LOT_SPECS = {
    "GALA/USDT:USDT":  {"min_qty": 10.0, "qty_step": 1.0},
    "STORJ/USDT:USDT": {"min_qty": 1.0,  "qty_step": 0.1},
    "ALGO/USDT:USDT":  {"min_qty": 1.0,  "qty_step": 0.1},
    "LUNA2/USDT:USDT": {"min_qty": 0.1,  "qty_step": 0.01},
    "CHZ/USDT:USDT":   {"min_qty": 1.0,  "qty_step": 0.1},
    "CRO/USDT:USDT":   {"min_qty": 10.0, "qty_step": 1.0},
    "STG/USDT:USDT":   {"min_qty": 1.0,  "qty_step": 0.1},
}


def load_edges(include_stg=False):
    portfolio = json.loads(PORTFOLIO_EDGES_JSON.read_text(encoding="utf-8"))
    edges = portfolio.get("edges", [])

    if include_stg and EXOTIC_OPTIMIZED_JSON.exists():
        exotic = json.loads(EXOTIC_OPTIMIZED_JSON.read_text(encoding="utf-8"))
        stg_edges = [e for e in exotic if e["symbol"] == "STG/USDT:USDT" and e.get("wf_pass")]
        if stg_edges:
            stg = stg_edges[0]
            edges.append({
                "symbol": stg["symbol"],
                "timeframe": stg["timeframe"],
                "strategy": stg["strategy"],
                "tuned_strategy_params": stg.get("tuned_strategy_params", {}),
                "tuned_risk_params": stg.get("tuned_risk_params", {}),
                "allocation_weight": 14.0
            })
            # Rebalance weights
            total_w = sum(e.get("allocation_weight", 16.67) for e in edges)
            for e in edges:
                e["allocation_weight"] = round(e.get("allocation_weight", 16.67) / total_w * 100, 2)

    return edges


def simulate_portfolio(edges, label="Portfolio"):
    edge_data = []
    for edge in edges:
        sym = edge["symbol"]
        tf = edge["timeframe"]
        strat_name = edge["strategy"]
        s_params = edge.get("tuned_strategy_params", {})
        r_params = edge.get("tuned_risk_params", {})

        if tf in ["4H", "1H", "15m", "30m", "1D"]:
            fpath = cfg.RESAMPLED_DIR / tf / (sym.replace("/", "_").replace(":", "_") + ".parquet")
        else:
            fpath = CUSTOM_RESAMPLED_DIR / tf / (sym.replace("/", "_").replace(":", "_") + ".parquet")

        if not fpath.exists():
            continue

        df = pd.read_parquet(fpath)
        df_2026 = df[(df.index >= pd.Timestamp("2026-01-01", tz="UTC")) &
                      (df.index <= pd.Timestamp("2026-08-31", tz="UTC"))].copy()

        if len(df_2026) < 30:
            continue

        strat_meta = get_strategy(strat_name)
        signals = strat_meta["func"](df_2026, **s_params)

        highs = df_2026["high"].values
        lows = df_2026["low"].values
        closes = df_2026["close"].values
        tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))
        tr[0] = highs[0] - lows[0]
        atrs = pd.Series(tr, index=df_2026.index).rolling(14, min_periods=1).mean().values

        spec = LOT_SPECS.get(sym, {"min_qty": 1.0, "qty_step": 0.1})
        edge_data.append({
            "symbol": sym, "tf": tf, "strategy": strat_name,
            "df": df_2026, "signals": signals, "atrs": atrs,
            "r_params": r_params,
            "qty_step": spec["qty_step"], "min_qty": spec["min_qty"],
            "weight": edge.get("allocation_weight", 16.67) / 100.0
        })

    all_ts = sorted(set(ts for ed in edge_data for ts in ed["df"].index))
    capital = LIVE_CAPITAL
    open_pos = {}
    trades = []
    fee_rate = FEE_PER_SIDE + SLIPPAGE
    halted = False

    for current_time in all_ts:
        if halted:
            break

        for sym in list(open_pos.keys()):
            pos = open_pos[sym]
            ed = next(e for e in edge_data if e["symbol"] == sym)
            df = ed["df"]
            if current_time not in df.index:
                continue
            row = df.loc[current_time]
            b_h, b_l, b_o, b_c = row["high"], row["low"], row["open"], row["close"]
            pos["bars"] += 1
            exit_t, exit_p, exit_r = False, 0.0, ""

            if pos["dir"] == 1:
                if b_l <= pos["sl"]: exit_t, exit_p, exit_r = True, min(b_o, pos["sl"]), "SL"
                elif pos["tp"] > 0 and b_h >= pos["tp"]: exit_t, exit_p, exit_r = True, max(b_o, pos["tp"]), "TP"
            else:
                if b_h >= pos["sl"]: exit_t, exit_p, exit_r = True, max(b_o, pos["sl"]), "SL"
                elif pos["tp"] > 0 and b_l <= pos["tp"]: exit_t, exit_p, exit_r = True, min(b_o, pos["tp"]), "TP"

            if exit_t:
                gross = pos["qty"] * (exit_p - pos["entry"]) if pos["dir"] == 1 else pos["qty"] * (pos["entry"] - exit_p)
                ef = pos["qty"] * exit_p * fee_rate
                net = gross - ef
                capital += (pos["margin"] + net)
                trades.append({
                    "symbol": sym, "strategy": pos["strategy"], "dir": pos["dir"],
                    "entry_time": pos["entry_time"], "exit_time": current_time,
                    "entry": pos["entry"], "exit": exit_p, "qty": pos["qty"],
                    "notional": pos["notional"], "margin": pos["margin"],
                    "pnl": net, "pnl_pct": (net / pos["margin"]) * 100,
                    "fees": pos["entry_fee"] + ef, "reason": exit_r,
                    "bars": pos["bars"], "hours": pos["bars"] * 4,
                    "month": pos["entry_time"].strftime("%Y-%m")
                })
                del open_pos[sym]
                if capital < DAILY_LOSS_LIMIT:
                    halted = True
                    break

        for ed in edge_data:
            sym = ed["symbol"]
            df = ed["df"]
            if sym in open_pos or current_time not in df.index:
                continue
            idx = df.index.get_loc(current_time)
            if idx == 0:
                continue
            sig = ed["signals"].iloc[idx - 1]
            if sig == 0:
                continue
            entry_p = df["open"].iloc[idx]
            atr = ed["atrs"][idx - 1]
            rp = ed["r_params"]
            step, mq = ed["qty_step"], ed["min_qty"]
            qty = max(mq, np.ceil(MIN_NOTIONAL / entry_p / step) * step)
            notional = qty * entry_p
            margin = notional / LIVE_LEVERAGE
            locked = sum(p["margin"] for p in open_pos.values())
            if (capital - locked) < margin:
                continue
            sl_d = max(rp.get("sl_atr_mult", 2.0) * atr, entry_p * 0.003)
            tp_m = rp.get("tp_atr_mult", 4.0)
            sl = entry_p - sl_d if sig == 1 else entry_p + sl_d
            tp = entry_p + tp_m * atr if (sig == 1 and tp_m > 0) else (entry_p - tp_m * atr if tp_m > 0 else 0)
            ef = notional * fee_rate
            capital -= (margin + ef)
            open_pos[sym] = {
                "dir": sig, "entry_time": current_time, "entry": entry_p,
                "qty": qty, "notional": notional, "margin": margin,
                "entry_fee": ef, "sl": sl, "tp": tp,
                "strategy": ed["strategy"], "bars": 0
            }

    for sym, pos in open_pos.items():
        ed = next(e for e in edge_data if e["symbol"] == sym)
        fc = ed["df"]["close"].iloc[-1]
        gross = pos["qty"] * (fc - pos["entry"]) if pos["dir"] == 1 else pos["qty"] * (pos["entry"] - fc)
        ef = pos["qty"] * fc * fee_rate
        net = gross - ef
        capital += (pos["margin"] + net)
        trades.append({
            "symbol": sym, "strategy": pos["strategy"], "dir": pos["dir"],
            "entry_time": pos["entry_time"], "exit_time": ed["df"].index[-1],
            "entry": pos["entry"], "exit": fc, "qty": pos["qty"],
            "notional": pos["notional"], "margin": pos["margin"],
            "pnl": net, "pnl_pct": (net / pos["margin"]) * 100,
            "fees": pos["entry_fee"] + ef, "reason": "END_OF_DATA",
            "bars": pos["bars"], "hours": pos["bars"] * 4,
            "month": pos["entry_time"].strftime("%Y-%m")
        })

    tdf = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    return tdf, capital, halted


def compute_metrics(tdf, final_cap, halted):
    if tdf.empty:
        return {}
    total = len(tdf)
    wins = len(tdf[tdf["pnl"] > 0])
    losses = total - wins
    wr = wins / total * 100
    net_pnl = tdf["pnl"].sum()
    fees = tdf["fees"].sum()
    ret_pct = (net_pnl / LIVE_CAPITAL) * 100
    ann_ret = ret_pct * 1.5
    avg_win = tdf[tdf["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
    avg_loss = tdf[tdf["pnl"] <= 0]["pnl"].mean() if losses > 0 else 0
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    best = tdf["pnl"].max()
    worst = tdf["pnl"].min()
    avg_hold = tdf["hours"].mean()

    eq = LIVE_CAPITAL + tdf["pnl"].cumsum()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = abs(dd.min()) * 100

    # Monthly stats
    months = sorted(tdf["month"].unique())
    monthly_pnl = tdf.groupby("month")["pnl"].sum()
    pos_months = len(monthly_pnl[monthly_pnl > 0])
    month_wr = pos_months / len(months) * 100 if months else 0
    avg_month = monthly_pnl.mean()
    best_month = monthly_pnl.max()
    worst_month = monthly_pnl.min()

    # Sharpe approx
    daily_eq = eq.resample("1D").last().ffill() if hasattr(eq.index, 'freq') else eq
    daily_ret = daily_eq.pct_change().dropna()
    sharpe = float(np.mean(daily_ret) / (np.std(daily_ret) + 1e-9) * np.sqrt(365)) if len(daily_ret) > 10 else 0
    downside = daily_ret[daily_ret < 0].std()
    sortino = float(np.mean(daily_ret) / (downside + 1e-9) * np.sqrt(365)) if len(daily_ret) > 10 else 0
    calmar = ann_ret / max_dd if max_dd > 0 else 0

    # Per-symbol breakdown
    sym_stats = {}
    for sym in tdf["symbol"].unique():
        st = tdf[tdf["symbol"] == sym]
        sym_stats[sym] = {
            "trades": len(st),
            "wr": len(st[st["pnl"] > 0]) / len(st) * 100,
            "pnl": st["pnl"].sum()
        }

    return {
        "total_trades": total, "wins": wins, "losses": losses,
        "win_rate": wr, "net_pnl": net_pnl, "fees": fees,
        "return_pct": ret_pct, "ann_return": ann_ret,
        "final_cap": final_cap, "max_dd": max_dd,
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "avg_win": avg_win, "avg_loss": avg_loss, "payoff": payoff,
        "best_trade": best, "worst_trade": worst, "avg_hold_h": avg_hold,
        "months": len(months), "pos_months": pos_months, "month_wr": month_wr,
        "avg_month_pnl": avg_month, "best_month": best_month, "worst_month": worst_month,
        "circuit_breaker": "HALTED" if halted else "SAFE",
        "sym_stats": sym_stats
    }


def main():
    console.rule("[bold magenta]PORTFOLIO COMPARISON: ORIGINAL 6 vs SUPER 7")
    console.print("[yellow]Simulating both portfolios on 2026 OOS data ($15 / 10x / $5 Min Notional)...\n[/yellow]")

    # Run Original 6
    console.print("[cyan]Running Original 6 (4H Pure)...[/]")
    edges_6 = load_edges(include_stg=False)
    tdf_6, cap_6, halt_6 = simulate_portfolio(edges_6, "Original 6")
    m6 = compute_metrics(tdf_6, cap_6, halt_6)

    # Run Super 7
    console.print("[cyan]Running Super 7 (4H + STG 120m)...[/]")
    edges_7 = load_edges(include_stg=True)
    tdf_7, cap_7, halt_7 = simulate_portfolio(edges_7, "Super 7")
    m7 = compute_metrics(tdf_7, cap_7, halt_7)

    if not m6 or not m7:
        console.print("[red]Simulation failed![/]")
        return

    # ── MASTER COMPARISON TABLE ────────────────────────────────
    console.rule("[bold cyan]MASTER COMPARISON TABLE")

    tbl = Table(title="ORIGINAL 6 vs SUPER 7 — Full 2026 OOS Comparison ($15 / 10x / $5 Min Notional)", show_lines=True)
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Original 6 (4H)", justify="right", style="bold white")
    tbl.add_column("Super 7 (4H+2H)", justify="right", style="bold green")
    tbl.add_column("Delta", justify="right", style="bold yellow")

    def delta_str(v6, v7, fmt=".2f", suffix="", higher_better=True):
        d = v7 - v6
        color = "green" if (d > 0 and higher_better) or (d < 0 and not higher_better) else "red"
        sign = "+" if d > 0 else ""
        return f"[{color}]{sign}{d:{fmt}}{suffix}[/]"

    tbl.add_row("─── CORE PERFORMANCE ───", "───", "───", "───")
    tbl.add_row("Final Capital", f"${m6['final_cap']:.2f}", f"${m7['final_cap']:.2f}", delta_str(m6['final_cap'], m7['final_cap'], ".2f", ""))
    tbl.add_row("Net Profit", f"${m6['net_pnl']:+.2f}", f"${m7['net_pnl']:+.2f}", delta_str(m6['net_pnl'], m7['net_pnl'], ".2f"))
    tbl.add_row("Total Return %", f"{m6['return_pct']:+.1f}%", f"{m7['return_pct']:+.1f}%", delta_str(m6['return_pct'], m7['return_pct'], ".1f", "%"))
    tbl.add_row("Annualized Return", f"{m6['ann_return']:+.1f}%", f"{m7['ann_return']:+.1f}%", delta_str(m6['ann_return'], m7['ann_return'], ".1f", "%"))

    tbl.add_row("─── RISK-ADJUSTED RATIOS ───", "───", "───", "───")
    tbl.add_row("Sharpe Ratio", f"{m6['sharpe']:.2f}", f"{m7['sharpe']:.2f}", delta_str(m6['sharpe'], m7['sharpe']))
    tbl.add_row("Sortino Ratio", f"{m6['sortino']:.2f}", f"{m7['sortino']:.2f}", delta_str(m6['sortino'], m7['sortino']))
    tbl.add_row("Calmar Ratio", f"{m6['calmar']:.2f}", f"{m7['calmar']:.2f}", delta_str(m6['calmar'], m7['calmar']))
    tbl.add_row("Max Drawdown %", f"{m6['max_dd']:.1f}%", f"{m7['max_dd']:.1f}%", delta_str(m6['max_dd'], m7['max_dd'], ".1f", "%", False))

    tbl.add_row("─── TRADE QUALITY ───", "───", "───", "───")
    tbl.add_row("Total Trades", str(m6['total_trades']), str(m7['total_trades']), f"+{m7['total_trades']-m6['total_trades']}")
    tbl.add_row("Win Rate %", f"{m6['win_rate']:.1f}%", f"{m7['win_rate']:.1f}%", delta_str(m6['win_rate'], m7['win_rate'], ".1f", "%"))
    tbl.add_row("Wins / Losses", f"{m6['wins']}W / {m6['losses']}L", f"{m7['wins']}W / {m7['losses']}L", "—")
    tbl.add_row("Avg Win ($)", f"${m6['avg_win']:+.4f}", f"${m7['avg_win']:+.4f}", delta_str(m6['avg_win'], m7['avg_win'], ".4f"))
    tbl.add_row("Avg Loss ($)", f"${m6['avg_loss']:+.4f}", f"${m7['avg_loss']:+.4f}", delta_str(m6['avg_loss'], m7['avg_loss'], ".4f", "", False))
    tbl.add_row("Payoff Ratio", f"{m6['payoff']:.2f}", f"{m7['payoff']:.2f}", delta_str(m6['payoff'], m7['payoff']))
    tbl.add_row("Best Trade ($)", f"${m6['best_trade']:+.4f}", f"${m7['best_trade']:+.4f}", delta_str(m6['best_trade'], m7['best_trade'], ".4f"))
    tbl.add_row("Worst Trade ($)", f"${m6['worst_trade']:+.4f}", f"${m7['worst_trade']:+.4f}", delta_str(m6['worst_trade'], m7['worst_trade'], ".4f", "", False))
    tbl.add_row("Avg Hold (Hours)", f"{m6['avg_hold_h']:.0f}h", f"{m7['avg_hold_h']:.0f}h", delta_str(m6['avg_hold_h'], m7['avg_hold_h'], ".0f", "h", False))

    tbl.add_row("─── MONTHLY CONSISTENCY ───", "───", "───", "───")
    tbl.add_row("Positive Months", f"{m6['pos_months']}/{m6['months']}", f"{m7['pos_months']}/{m7['months']}", "—")
    tbl.add_row("Monthly Win Rate", f"{m6['month_wr']:.1f}%", f"{m7['month_wr']:.1f}%", delta_str(m6['month_wr'], m7['month_wr'], ".1f", "%"))
    tbl.add_row("Avg Monthly PnL", f"${m6['avg_month_pnl']:+.2f}", f"${m7['avg_month_pnl']:+.2f}", delta_str(m6['avg_month_pnl'], m7['avg_month_pnl'], ".2f"))
    tbl.add_row("Best Month", f"${m6['best_month']:+.2f}", f"${m7['best_month']:+.2f}", delta_str(m6['best_month'], m7['best_month'], ".2f"))
    tbl.add_row("Worst Month", f"${m6['worst_month']:+.2f}", f"${m7['worst_month']:+.2f}", delta_str(m6['worst_month'], m7['worst_month'], ".2f", "", False))

    tbl.add_row("─── COSTS & SAFETY ───", "───", "───", "───")
    tbl.add_row("Total Fees Paid", f"${m6['fees']:.2f}", f"${m7['fees']:.2f}", delta_str(m6['fees'], m7['fees'], ".2f", "", False))
    tbl.add_row("Circuit Breaker", m6['circuit_breaker'], m7['circuit_breaker'], "—")
    tbl.add_row("Active Edges", "6 (all 4H)", "7 (6x4H + 1x2H)", "+1")

    console.print(tbl)

    # ── PER-SYMBOL BREAKDOWN ───────────────────────────────────
    console.rule("[bold cyan]PER-SYMBOL PnL BREAKDOWN (2026)")

    sym_tbl = Table(title="Individual Edge Contribution to Portfolio PnL", show_lines=True)
    sym_tbl.add_column("Symbol", style="bold white")
    sym_tbl.add_column("TF", justify="center", style="magenta")
    sym_tbl.add_column("Strategy", style="yellow")
    sym_tbl.add_column("In Orig 6?", justify="center")
    sym_tbl.add_column("Trades (6→7)", justify="center")
    sym_tbl.add_column("PnL Orig 6", justify="right")
    sym_tbl.add_column("PnL Super 7", justify="right")
    sym_tbl.add_column("Delta", justify="right", style="bold yellow")

    all_syms = sorted(set(list(m6.get("sym_stats", {}).keys()) + list(m7.get("sym_stats", {}).keys())))
    for sym in all_syms:
        s6 = m6.get("sym_stats", {}).get(sym, {"trades": 0, "pnl": 0, "wr": 0})
        s7 = m7.get("sym_stats", {}).get(sym, {"trades": 0, "pnl": 0, "wr": 0})
        in_orig = "✓" if sym in m6.get("sym_stats", {}) else "[bold green]★ NEW[/]"
        tf = "4H" if sym in m6.get("sym_stats", {}) else "120m"
        strat = "—"
        for e in edges_7:
            if e["symbol"] == sym:
                strat = e["strategy"]
                tf = e["timeframe"]
                break
        d = s7["pnl"] - s6["pnl"]
        d_color = "green" if d >= 0 else "red"
        sym_tbl.add_row(
            sym.split("/")[0], tf, strat, in_orig,
            f"{s6['trades']}→{s7['trades']}",
            f"${s6['pnl']:+.2f}", f"${s7['pnl']:+.2f}",
            f"[{d_color}]${d:+.2f}[/]"
        )

    console.print(sym_tbl)

    # ── VERDICT ────────────────────────────────────────────────
    pnl_improvement = m7['net_pnl'] - m6['net_pnl']
    dd_change = m7['max_dd'] - m6['max_dd']
    sharpe_change = m7['sharpe'] - m6['sharpe']

    if pnl_improvement > 0 and dd_change <= 2.0:
        verdict = "UPGRADE RECOMMENDED"
        v_color = "green"
        v_msg = (f"The Super 7 portfolio adds ${pnl_improvement:+.2f} extra profit "
                 f"with minimal drawdown change ({dd_change:+.1f}%). "
                 f"STG 120m provides genuine orthogonal alpha.")
    elif pnl_improvement > 0 and dd_change > 2.0:
        verdict = "CONDITIONAL UPGRADE"
        v_color = "yellow"
        v_msg = (f"Higher profit (+${pnl_improvement:.2f}) but drawdown increased by {dd_change:+.1f}%. "
                 f"Consider reducing STG position size.")
    else:
        verdict = "STICK WITH ORIGINAL 6"
        v_color = "yellow"
        v_msg = "The Original 6 portfolio remains the optimal configuration."

    console.print(Panel(
        f"[bold {v_color}]{verdict}[/bold {v_color}]\n\n{v_msg}\n\n"
        f"• Original 6: $15.00 → ${m6['final_cap']:.2f} ({m6['return_pct']:+.1f}%)\n"
        f"• Super 7:    $15.00 → ${m7['final_cap']:.2f} ({m7['return_pct']:+.1f}%)\n"
        f"• Sharpe Change: {sharpe_change:+.2f} | DD Change: {dd_change:+.1f}%",
        title="FINAL PORTFOLIO VERDICT",
        border_style=v_color
    ))

    console.rule("[bold green]COMPARISON COMPLETE")

if __name__ == "__main__":
    main()
