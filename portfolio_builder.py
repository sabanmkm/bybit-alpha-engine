"""
Phase 8 — Diversified Portfolio Construction from Walk-Forward Validated Edges.
Selects 5–10 verified, low-correlation edges with Risk Parity capital weighting.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

sys.path.insert(0, r"C:\BybitBacktest")
import config as cfg
cfg.DATA_GATE_UNLOCKED = True

from strategy_generator import get_strategy
from backtest_engine import run_backtest

console = Console()
VALIDATED_EDGES_JSON = cfg.RESULTS_DIR / "validated_edges.json"
PORTFOLIO_EDGES_JSON = cfg.RESULTS_DIR / "portfolio_edges.json"


def reconstruct_strategy_equity(edge: dict) -> pd.Series:
    sym = edge["symbol"]
    tf = edge["timeframe"]
    strat_name = edge["strategy"]
    fpath = cfg.RESAMPLED_DIR / tf / (sym.replace("/", "_").replace(":", "_") + ".parquet")

    if not fpath.exists():
        return pd.Series(dtype=float)

    df = pd.read_parquet(fpath)
    strat_meta = get_strategy(strat_name)
    strat_func = strat_meta["func"]
    s_params = edge.get("tuned_strategy_params", {})
    r_params = edge.get("tuned_risk_params", {})

    signals = strat_func(df, **s_params)
    res = run_backtest(
        df=df,
        signals=signals,
        symbol=sym,
        timeframe=tf,
        strategy_name=strat_name,
        sl_atr_mult=r_params.get("sl_atr_mult", 2.0),
        tp_atr_mult=r_params.get("tp_atr_mult", 4.0),
        trailing_stop_atr=r_params.get("trailing_stop_atr", 0.0),
        time_stop_bars=r_params.get("time_stop_bars", 0)
    )
    return res.equity_curve


def build_portfolio():
    console.rule("[bold cyan]PHASE 8 — Final Portfolio Construction (Validated Edges)")

    if not VALIDATED_EDGES_JSON.exists():
        console.print("[bold red]No validated edges found![/] Run Phase 9 first.")
        return

    validated_candidates = json.loads(VALIDATED_EDGES_JSON.read_text(encoding="utf-8"))
    
    if not validated_candidates:
        console.print("[yellow]Falling back to top tuned edges with best stability...[/]")
        all_tuned = json.loads((cfg.RESULTS_DIR / "tuned_strategies.json").read_text(encoding="utf-8"))
        validated_candidates = sorted(all_tuned, key=lambda x: x.get("after_sharpe", 0), reverse=True)[:8]

    console.print(f"[green]Selecting diversified portfolio from {len(validated_candidates)} candidate edges...[/green]")

    # Reconstruct returns for correlation checking
    returns_dict = {}
    valid_candidates = []

    for c in validated_candidates:
        key = f"{c['symbol']}|{c['timeframe']}|{c['strategy']}"
        eq = reconstruct_strategy_equity(c)
        if len(eq) > 10:
            daily_eq = eq.resample("1D").last().ffill()
            daily_ret = daily_eq.pct_change().fillna(0.0)
            returns_dict[key] = daily_ret
            valid_candidates.append(c)

    returns_df = pd.DataFrame(returns_dict).fillna(0.0)

    # Diversity Selection
    selected_edges = []
    selected_keys = []
    symbol_counts = {}

    for c in valid_candidates:
        key = f"{c['symbol']}|{c['timeframe']}|{c['strategy']}"
        sym = c["symbol"]

        if symbol_counts.get(sym, 0) >= 2:
            continue

        corr_pass = True
        for sel_k in selected_keys:
            corr = returns_df[key].corr(returns_df[sel_k])
            if not np.isnan(corr) and corr > 0.70 and len(selected_edges) >= 5:
                corr_pass = False
                break

        if corr_pass or len(selected_edges) < 5:
            selected_edges.append(c)
            selected_keys.append(key)
            symbol_counts[sym] = symbol_counts.get(sym, 0) + 1

        if len(selected_edges) >= 10:
            break

    # Compute Inverse-Volatility (Risk Parity) Weights
    vols = [returns_df[k].std() if returns_df[k].std() > 0 else 0.01 for k in selected_keys]
    inv_v = [1.0 / v for v in vols]
    total_inv = sum(inv_v)
    weights = [iv / total_inv for iv in inv_v]

    for i, edge in enumerate(selected_edges):
        edge["allocation_weight"] = round(weights[i] * 100.0, 2)

    # Simulate Portfolio
    port_ret = sum(returns_df[k] * weights[i] for i, k in enumerate(selected_keys))
    port_eq = (1.0 + port_ret).cumprod() * cfg.START_CAPITAL
    p_net = float(port_eq.iloc[-1] - cfg.START_CAPITAL)
    p_pct = float((p_net / cfg.START_CAPITAL) * 100.0)
    roll = port_eq.cummax()
    dd = (port_eq - roll) / roll
    p_max_dd = abs(float(dd.min())) * 100.0
    days = max((port_eq.index[-1] - port_eq.index[0]).days, 1)
    p_cagr = (((port_eq.iloc[-1] / cfg.START_CAPITAL) ** (365.0 / days)) - 1.0) * 100.0
    p_sharpe = float(np.mean(port_ret) / (np.std(port_ret) + 1e-9) * np.sqrt(365.0))
    p_calmar = float(p_cagr / p_max_dd) if p_max_dd > 0 else 0.0

    portfolio_payload = {
        "portfolio_metrics": {
            "cagr": round(p_cagr, 1),
            "sharpe_ratio": round(p_sharpe, 2),
            "calmar_ratio": round(p_calmar, 2),
            "max_drawdown_pct": round(p_max_dd, 1),
            "net_profit_pct": round(p_pct, 1)
        },
        "edges": selected_edges
    }
    PORTFOLIO_EDGES_JSON.write_text(json.dumps(portfolio_payload, indent=2), encoding="utf-8")

    # Display Table
    tbl = Table(title="Final Diversified Portfolio Composition", show_lines=True)
    tbl.add_column("Edge #", style="cyan", justify="right")
    tbl.add_column("Symbol", style="bold white")
    tbl.add_column("TF", justify="center", style="magenta")
    tbl.add_column("Strategy Name", style="yellow")
    tbl.add_column("Weight %", justify="right", style="bold green")
    tbl.add_column("IS Sharpe", justify="right", style="green")
    tbl.add_column("2026 OOS Sharpe", justify="right", style="bold cyan")
    tbl.add_column("Max DD %", justify="right", style="red")

    for i, e in enumerate(selected_edges, 1):
        tbl.add_row(
            str(i),
            e["symbol"],
            e["timeframe"],
            e["strategy"],
            f"{e['allocation_weight']:.1f}%",
            f"{e.get('is_sharpe', e.get('after_sharpe', 0.0)):.2f}",
            f"{e.get('oos_sharpe', 0.0):.2f}",
            f"{e.get('oos_max_dd', e.get('after_max_dd', 0.0)):.1f}%"
        )
    console.print(tbl)
    console.print(f"[bold green]✓ Saved final portfolio to {PORTFOLIO_EDGES_JSON}[/bold green]")
    console.rule("[bold green]PHASE 8 COMPLETE")

if __name__ == "__main__":
    build_portfolio()
