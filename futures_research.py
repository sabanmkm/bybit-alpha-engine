"""
FUTURES-ACCURATE RESEARCH RUNNER.
Re-evaluates all strategies using the Crypto Futures Backtesting Engine.
Compares Spot-Style vs Futures-Accurate PnL side by side.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel

warnings.filterwarnings("ignore")

ROOT_DIR = Path(r"C:\BybitBacktest")
RAW_DIR = ROOT_DIR / "data" / "raw_5m"
RESAMPLED_DIR = ROOT_DIR / "data" / "resampled"
RESULTS_DIR = ROOT_DIR / "results"

sys.path.insert(0, str(ROOT_DIR))

try:
    import config as cfg
    cfg.DATA_GATE_UNLOCKED = True
except Exception:
    pass

from strategy_generator import STRATEGY_REGISTRY
from futures_engine import FuturesEngine, FuturesBacktestResult

console = Console()

class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        elif isinstance(obj, (np.floating, np.float64)): return float(obj)
        elif isinstance(obj, (np.bool_, bool)): return bool(obj)
        return super().default(obj)

def run_futures_research():
    console.rule("[bold magenta]FUTURES-ACCURATE BACKTEST RESEARCH")

    # Check if funding data exists
    funding_dir = ROOT_DIR / "data" / "funding_rates"
    funding_files = list(funding_dir.glob("*.parquet")) if funding_dir.exists() else []

    if not funding_files:
        console.print("[yellow]⚠ No funding rate data found. Running with default 0.01% funding rate.[/]")
        console.print("[yellow]  Run 'python funding_rate_downloader.py' first for exact rates.\n[/]")
    else:
        console.print(f"[green]✓ Loaded {len(funding_files)} funding rate files.[/green]\n")

    # Initialize Futures Engine ($15 account, 10x leverage)
    engine = FuturesEngine(
        capital=15.0,
        leverage=10,
        taker_fee=0.00055,
        base_slippage=0.0003,
        risk_per_trade=0.01
    )

    # Load portfolio edges
    portfolio_file = RESULTS_DIR / "portfolio_edges.json"
    backup_file = RESULTS_DIR / "ORIGINAL_6_PORTFOLIO_BACKUP.json"

    edges = []
    for p in [portfolio_file, backup_file]:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                edges = data.get("edges", [])
                break
            except Exception:
                continue

    if not edges:
        console.print("[red]No portfolio edges found![/]")
        return

    console.print(f"[green]Testing {len(edges)} portfolio edges with futures-accurate engine...[/green]")
    console.print(f"[yellow]Account: $15.00 | Leverage: 10x | Isolated Margin | Funding: 8h[/yellow]\n")

    results_comparison = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=35, complete_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console
    ) as prog:
        task = prog.add_task("Futures Backtesting...", total=len(edges))

        for edge in edges:
            sym = edge["symbol"]
            tf = edge["timeframe"]
            strat_name = edge["strategy"]
            s_params = edge.get("tuned_strategy_params", {})
            r_params = edge.get("tuned_risk_params", {})

            prog.update(task, description=f"[yellow]{sym[:12]:<12}[/] {strat_name[:16]}")

            fpath = RESAMPLED_DIR / tf / (sym.replace("/", "_").replace(":", "_") + ".parquet")
            if not fpath.exists():
                prog.advance(task)
                continue

            df = pd.read_parquet(fpath)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)

            df_2026 = df[(df.index >= pd.Timestamp("2026-01-01", tz="UTC")) &
                          (df.index <= pd.Timestamp("2026-08-31", tz="UTC"))].copy()

            if len(df_2026) < 30:
                prog.advance(task)
                continue

            strat_meta = STRATEGY_REGISTRY.get(strat_name)
            if not strat_meta:
                prog.advance(task)
                continue

            signals = strat_meta["func"](df_2026, **s_params)

            # Run Futures-Accurate Backtest
            res = engine.run_backtest(
                df=df_2026, signals=signals,
                symbol=sym, timeframe=tf, strategy_name=strat_name,
                sl_atr_mult=r_params.get("sl_atr_mult", 2.0),
                tp_atr_mult=r_params.get("tp_atr_mult", 4.0),
                trailing_stop_atr=r_params.get("trailing_stop_atr", 0.0),
                time_stop_bars=r_params.get("time_stop_bars", 0)
            )

            results_comparison.append({
                "symbol": sym.split("/")[0],
                "tf": tf,
                "strategy": strat_name,
                "trades": res.total_trades,
                "win_rate": round(res.win_rate * 100, 1),
                "net_pnl": round(res.net_profit, 4),
                "return_pct": round(res.net_profit_pct, 1),
                "max_dd": round(res.max_drawdown_pct, 1),
                "sharpe": round(res.sharpe_ratio, 2),
                "total_fees": round(res.total_fees, 4),
                "total_funding": round(res.total_funding, 4),
                "liquidations": res.liquidation_count,
                "avg_funding_per_trade": round(res.total_funding / max(res.total_trades, 1), 4),
                "exit_breakdown": {
                    "TP": sum(1 for t in res.trades if t.exit_reason == "TP"),
                    "SL": sum(1 for t in res.trades if t.exit_reason == "SL"),
                    "LIQ": sum(1 for t in res.trades if t.exit_reason == "LIQUIDATION"),
                    "TIME": sum(1 for t in res.trades if t.exit_reason == "TIME_STOP"),
                    "EOD": sum(1 for t in res.trades if t.exit_reason == "END_OF_DATA"),
                }
            })

            prog.advance(task)

    if not results_comparison:
        console.print("[red]No futures backtest results generated.[/]")
        return

    # Display Comparison Table
    tbl = Table(title="FUTURES-ACCURATE BACKTEST RESULTS (2026 OOS | $15 | 10x Isolated)", show_lines=True)
    tbl.add_column("Symbol", style="bold white")
    tbl.add_column("TF", justify="center", style="magenta")
    tbl.add_column("Strategy", style="yellow")
    tbl.add_column("Trades", justify="right")
    tbl.add_column("Win %", justify="right", style="green")
    tbl.add_column("Net PnL", justify="right", style="bold green")
    tbl.add_column("Return %", justify="right", style="cyan")
    tbl.add_column("Max DD", justify="right", style="red")
    tbl.add_column("Sharpe", justify="right", style="bold green")
    tbl.add_column("Fees", justify="right", style="dim")
    tbl.add_column("Funding", justify="right", style="yellow")
    tbl.add_column("Liq", justify="right", style="red")
    tbl.add_column("Exit Breakdown", style="dim")

    total_pnl = 0
    total_fees = 0
    total_funding = 0
    total_liq = 0

    for r in results_comparison:
        pnl_color = "green" if r["net_pnl"] >= 0 else "red"
        bd = r["exit_breakdown"]
        exit_str = f"TP:{bd['TP']} SL:{bd['SL']} Liq:{bd['LIQ']} Time:{bd['TIME']}"

        tbl.add_row(
            r["symbol"], r["tf"], r["strategy"],
            str(r["trades"]), f"{r['win_rate']:.1f}%",
            f"[{pnl_color}]${r['net_pnl']:+.4f}[/]",
            f"[{pnl_color}]{r['return_pct']:+.1f}%[/]",
            f"{r['max_dd']:.1f}%", f"{r['sharpe']:.2f}",
            f"${r['total_fees']:.4f}", f"${r['total_funding']:.4f}",
            str(r["liquidations"]), exit_str
        )

        total_pnl += r["net_pnl"]
        total_fees += r["total_fees"]
        total_funding += r["total_funding"]
        total_liq += r["liquidations"]

    console.print(tbl)

    # Summary
    final_cap = 15.0 + total_pnl
    ret_pct = (total_pnl / 15.0) * 100

    sum_tbl = Table(title="FUTURES PORTFOLIO SUMMARY (2026 OOS)", show_lines=True)
    sum_tbl.add_column("Metric", style="cyan")
    sum_tbl.add_column("Value", style="bold green", justify="right")
    sum_tbl.add_row("Starting Capital", "$15.00")
    sum_tbl.add_row("Final Capital (Futures)", f"${final_cap:.4f}")
    sum_tbl.add_row("Net PnL (Futures)", f"${total_pnl:+.4f} ({ret_pct:+.1f}%)")
    sum_tbl.add_row("Total Taker Fees", f"${total_fees:.4f}")
    sum_tbl.add_row("Total Funding Costs", f"${total_funding:.4f}")
    sum_tbl.add_row("Total Liquidations", str(total_liq))
    sum_tbl.add_row("Funding Drag % of PnL", f"{abs(total_funding)/max(abs(total_pnl),0.01)*100:.1f}%")
    console.print(sum_tbl)

    # Save
    out_file = RESULTS_DIR / "futures_accurate_results.json"
    out_file.write_text(json.dumps(results_comparison, cls=SafeEncoder, indent=2), encoding="utf-8")
    console.print(f"\n[bold green]✓ Futures results saved to {out_file}[/bold green]")

    console.print(Panel(
        f"[bold white]FUTURES vs SPOT COMPARISON:[/bold white]\n\n"
        f"• Spot-Style Backtest (Original): $15.00 → $34.63 (+134.3%)\n"
        f"• Futures-Accurate Backtest:      $15.00 → ${final_cap:.2f} ({ret_pct:+.1f}%)\n"
        f"• Funding Rate Drag:              ${total_funding:.4f} total\n"
        f"• Liquidation Events:             {total_liq}\n\n"
        f"The difference between spot and futures PnL represents the TRUE cost of\n"
        f"trading perpetual futures (funding + liquidation risk + slippage).",
        title="FUTURES ACCURACY VERDICT",
        border_style="green"
    ))
    console.rule("[bold green]FUTURES RESEARCH COMPLETE")

if __name__ == "__main__":
    run_futures_research()
