"""
Phase 9 — Exhaustive Walk-Forward Validation Engine (2026 Data Unlock 🔓).
Evaluates all tuned candidate edges on locked 2026-01-01 -> 2026-08-31 data.
Identifies and exports all robust, passing edges for portfolio assembly.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
)
from rich.table import Table

sys.path.insert(0, r"C:\BybitBacktest")
import config as cfg

# 🔓 UNLOCK 2026 DATA GATE FOR PHASE 9
cfg.DATA_GATE_UNLOCKED = True

from strategy_generator import get_strategy
from backtest_engine import run_backtest, BacktestResult

console = Console()
TUNED_STRATEGIES_JSON    = cfg.RESULTS_DIR / "tuned_strategies.json"
VALIDATED_EDGES_JSON     = cfg.RESULTS_DIR / "validated_edges.json"
WALKFORWARD_RESULTS_JSON = cfg.RESULTS_DIR / "walkforward_results.json"


def evaluate_edge_walkforward(edge: dict) -> dict:
    sym = edge["symbol"]
    tf = edge["timeframe"]
    strat_name = edge["strategy"]
    fpath = cfg.RESAMPLED_DIR / tf / (sym.replace("/", "_").replace(":", "_") + ".parquet")

    if not fpath.exists():
        return {"edge": edge, "status": "FILE_NOT_FOUND"}

    df = pd.read_parquet(fpath)
    strat_meta = get_strategy(strat_name)
    strat_func = strat_meta["func"]
    s_params = edge.get("tuned_strategy_params", {})
    r_params = edge.get("tuned_risk_params", {})

    # 1. In-Sample Slice (2022 - 2025)
    df_is = df[df.index <= pd.Timestamp(cfg.IS_END)].copy()
    sig_is = strat_func(df_is, **s_params)
    res_is: BacktestResult = run_backtest(
        df=df_is,
        signals=sig_is,
        symbol=sym,
        timeframe=tf,
        strategy_name=strat_name,
        sl_atr_mult=r_params.get("sl_atr_mult", 2.0),
        tp_atr_mult=r_params.get("tp_atr_mult", 4.0),
        trailing_stop_atr=r_params.get("trailing_stop_atr", 0.0),
        time_stop_bars=r_params.get("time_stop_bars", 0)
    )

    # 2. Out-of-Sample Slice (2026-01-01 -> 2026-08-31)
    df_oos = df[(df.index >= pd.Timestamp(cfg.OOS_START)) & (df.index <= pd.Timestamp(cfg.END_DATE))].copy()
    if len(df_oos) < 20:
        return {"edge": edge, "status": "INSUFFICIENT_OOS_DATA"}

    sig_oos = strat_func(df_oos, **s_params)
    res_oos: BacktestResult = run_backtest(
        df=df_oos,
        signals=sig_oos,
        symbol=sym,
        timeframe=tf,
        strategy_name=strat_name,
        sl_atr_mult=r_params.get("sl_atr_mult", 2.0),
        tp_atr_mult=r_params.get("tp_atr_mult", 4.0),
        trailing_stop_atr=r_params.get("trailing_stop_atr", 0.0),
        time_stop_bars=r_params.get("time_stop_bars", 0)
    )

    is_sharpe = max(res_is.sharpe_ratio, 0.01)
    oos_sharpe = res_oos.sharpe_ratio
    sharpe_deg_ratio = oos_sharpe / is_sharpe

    is_dd = max(res_is.max_drawdown_pct, 1.0)
    oos_dd = res_oos.max_drawdown_pct
    dd_ratio = oos_dd / is_dd

    # Robust Pass Criteria: OOS Sharpe > 0.35, Net Profit > 0, Max DD < 15%, PF >= 1.05
    pass_sharpe = oos_sharpe >= 0.35
    pass_profit = res_oos.net_profit > 0
    pass_dd = oos_dd < 15.0
    pass_pf = res_oos.profit_factor >= 1.05

    passed = pass_sharpe and pass_profit and pass_dd and pass_pf
    status = "PASS" if passed else "FAIL"

    return {
        "symbol": sym,
        "timeframe": tf,
        "strategy": strat_name,
        "category": edge.get("category", "General"),
        "status": status,
        "is_trades": res_is.total_trades,
        "is_sharpe": round(res_is.sharpe_ratio, 2),
        "is_calmar": round(res_is.calmar_ratio, 2),
        "is_max_dd": round(res_is.max_drawdown_pct, 1),
        "is_win_rate": round(res_is.win_rate * 100, 1),
        "is_pf": round(res_is.profit_factor, 2),
        "is_net_pct": round(res_is.net_profit_pct, 1),
        "oos_trades": res_oos.total_trades,
        "oos_sharpe": round(res_oos.sharpe_ratio, 2),
        "oos_calmar": round(res_oos.calmar_ratio, 2),
        "oos_max_dd": round(res_oos.max_drawdown_pct, 1),
        "oos_win_rate": round(res_oos.win_rate * 100, 1),
        "oos_pf": round(res_oos.profit_factor, 2),
        "oos_net_pct": round(res_oos.net_profit_pct, 1),
        "sharpe_degradation": round(sharpe_deg_ratio, 2),
        "dd_ratio": round(dd_ratio, 2),
        "tuned_strategy_params": s_params,
        "tuned_risk_params": r_params,
        "after_sharpe": round(res_is.sharpe_ratio, 2),
        "after_calmar": round(res_is.calmar_ratio, 2),
        "after_max_dd": round(res_is.max_drawdown_pct, 1),
        "after_win_rate": round(res_is.win_rate * 100, 1)
    }


def run_walkforward():
    console.rule("[bold cyan]PHASE 9 — Walk-Forward Validation on 2026 OOS Data 🔓")

    if not TUNED_STRATEGIES_JSON.exists():
        console.print("[bold red]No tuned strategies found![/] Run Phase 7 first.")
        return

    tuned_candidates = json.loads(TUNED_STRATEGIES_JSON.read_text(encoding="utf-8"))
    console.print(f"[green]Evaluating all {len(tuned_candidates)} tuned candidates on 2026 data...[/green]\n")

    all_results = []
    passing_edges = []

    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=35, complete_style="green"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Validating Candidates...", total=len(tuned_candidates))

        for edge in tuned_candidates:
            sym_tf = f"{edge['symbol']} ({edge['timeframe']})"
            progress.update(task, description=f"Testing [yellow]{sym_tf:<18}[/] [cyan]{edge['strategy'][:18]}[/]")

            wf_res = evaluate_edge_walkforward(edge)
            all_results.append(wf_res)

            if wf_res.get("status") == "PASS":
                passing_edges.append(wf_res)
                console.print(
                    f"  [bold green]✓ PASS[/] {sym_tf} {edge['strategy']} "
                    f"| IS Sharpe: {wf_res.get('is_sharpe', 0.0)} → [bold green]OOS Sharpe: {wf_res.get('oos_sharpe', 0.0)}[/] "
                    f"| OOS Return: [bold green]+{wf_res.get('oos_net_pct', 0.0)}%[/] "
                    f"| OOS DD: [cyan]{wf_res.get('oos_max_dd', 0.0)}%[/]"
                )
            else:
                console.print(
                    f"  [dim red]✗ FAIL[/] {sym_tf} {edge['strategy']} "
                    f"| OOS Sharpe: {wf_res.get('oos_sharpe', 0.0)} | OOS Return: {wf_res.get('oos_net_pct', 0.0)}%"
                )

            progress.advance(task)

    # Save outputs
    VALIDATED_EDGES_JSON.write_text(json.dumps(passing_edges, indent=2), encoding="utf-8")
    
    out_payload = {
        "total_tested": len(tuned_candidates),
        "total_passed": len(passing_edges),
        "edges_validation": all_results
    }
    WALKFORWARD_RESULTS_JSON.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    console.print(f"\n[bold green]✓ Found {len(passing_edges)} Walk-Forward VALIDATED edges suitable for live portfolio![/bold green]\n")

    # Display Table
    tbl = Table(title="Walk-Forward Validation: In-Sample (2022–2025) vs. Out-of-Sample (2026)", show_lines=True)
    tbl.add_column("Symbol", style="bold white")
    tbl.add_column("TF", justify="center", style="magenta")
    tbl.add_column("Strategy", style="yellow")
    tbl.add_column("IS Sharpe", justify="right", style="green")
    tbl.add_column("2026 OOS Sharpe", justify="right", style="bold green")
    tbl.add_column("2026 OOS PnL %", justify="right", style="bold cyan")
    tbl.add_column("2026 OOS Max DD", justify="right", style="bold red")
    tbl.add_column("OOS PF", justify="right")
    tbl.add_column("Verdict", justify="center")

    for r in all_results:
        status_styled = "[bold green]PASS[/]" if r.get("status") == "PASS" else "[dim red]FAIL[/]"
        pnl_str = f"+{r.get('oos_net_pct', 0.0):.1f}%" if r.get('oos_net_pct', 0.0) >= 0 else f"{r.get('oos_net_pct', 0.0):.1f}%"
        tbl.add_row(
            r["symbol"],
            r["timeframe"],
            r["strategy"],
            f"{r.get('is_sharpe', 0.0):.2f}",
            f"{r.get('oos_sharpe', 0.0):.2f}",
            pnl_str,
            f"{r.get('oos_max_dd', 0.0):.1f}%",
            f"{r.get('oos_pf', 0.0):.2f}",
            status_styled
        )
    console.print(tbl)
    console.rule("[bold green]PHASE 9 COMPLETE")

if __name__ == "__main__":
    run_walkforward()
