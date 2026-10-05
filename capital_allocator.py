"""
Phase 9.7 — Capital Allocation & Position Sizing Engine.
Data-gate unlocked & timezone compatible.
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
PORTFOLIO_EDGES_JSON    = cfg.RESULTS_DIR / "portfolio_edges.json"
CAPITAL_ALLOCATION_JSON = cfg.RESULTS_DIR / "capital_allocation.json"


def run_capital_allocation():
    console.rule("[bold cyan]PHASE 9.7 — Capital Allocation & Position Sizing Engine")

    if not PORTFOLIO_EDGES_JSON.exists():
        console.print("[bold red]No portfolio edges found![/]")
        return

    portfolio_payload = json.loads(PORTFOLIO_EDGES_JSON.read_text(encoding="utf-8"))
    edges = portfolio_payload.get("edges", [])
    console.print(f"[green]Evaluating position sizing across {len(edges)} portfolio edges...[/green]\n")

    returns_dict = {}
    kelly_fractions = []
    vols = []

    for edge in edges:
        key = f"{edge['symbol']}|{edge['timeframe']}|{edge['strategy']}"
        fpath = cfg.RESAMPLED_DIR / edge["timeframe"] / (edge["symbol"].replace("/", "_").replace(":", "_") + ".parquet")
        if not fpath.exists():
            continue

        df = pd.read_parquet(fpath)
        strat_meta = get_strategy(edge["strategy"])
        s_params = edge.get("tuned_strategy_params", {})
        r_params = edge.get("tuned_risk_params", {})
        signals = strat_meta["func"](df, **s_params)

        res = run_backtest(
            df=df,
            signals=signals,
            symbol=edge["symbol"],
            timeframe=edge["timeframe"],
            sl_atr_mult=r_params.get("sl_atr_mult", 2.0),
            tp_atr_mult=r_params.get("tp_atr_mult", 4.0),
            trailing_stop_atr=r_params.get("trailing_stop_atr", 0.0),
            time_stop_bars=r_params.get("time_stop_bars", 0)
        )

        daily_ret = res.equity_curve.resample("1D").last().ffill().pct_change().fillna(0.0)
        returns_dict[key] = daily_ret

        wr = max(res.win_rate, 0.50)
        pf = max(res.profit_factor, 1.2)
        f_star = float(np.clip((wr * pf - (1 - wr)) / pf, 0.05, 0.40))
        kelly_fractions.append(f_star)

        vol = daily_ret.std() * np.sqrt(365.0)
        vols.append(vol if vol > 0 else 0.20)

    ret_df = pd.DataFrame(returns_dict).fillna(0.0)
    n = len(edges)

    w_equal = np.ones(n) / n
    w_quarter_kelly = (np.array(kelly_fractions) * 0.25) / sum(np.array(kelly_fractions) * 0.25)
    inv_v = 1.0 / np.array(vols)
    w_risk_parity = inv_v / sum(inv_v)
    w_max_div = (w_risk_parity + w_quarter_kelly) / 2.0

    methods = [
        ("Equal Weight", w_equal, "Baseline balance"),
        ("Quarter Kelly (★ Recommended)", w_quarter_kelly, "Optimal risk-adjusted compound growth"),
        ("Inverse Volatility (Risk Parity)", w_risk_parity, "Equal risk contribution"),
        ("Max Diversification Ensemble", w_max_div, "Blended risk parity + Kelly efficiency")
    ]

    allocation_comparison = []

    for name, weights, desc in methods:
        p_ret = ret_df.dot(weights)
        p_eq = (1.0 + p_ret).cumprod() * cfg.START_CAPITAL
        p_net = float(p_eq.iloc[-1] - cfg.START_CAPITAL)
        p_pct = float((p_net / cfg.START_CAPITAL) * 100.0)
        roll = p_eq.cummax()
        dd = (p_eq - roll) / roll
        p_max_dd = abs(float(dd.min())) * 100.0
        days = max((p_eq.index[-1] - p_eq.index[0]).days, 1)
        cagr = (((p_eq.iloc[-1] / cfg.START_CAPITAL) ** (365.0 / days)) - 1.0) * 100.0
        sharpe = float(np.mean(p_ret) / (np.std(p_ret) + 1e-9) * np.sqrt(365.0))
        calmar = float(cagr / p_max_dd) if p_max_dd > 0 else 0.0

        allocation_comparison.append({
            "method": name,
            "description": desc,
            "sharpe_ratio": round(sharpe, 2),
            "calmar_ratio": round(calmar, 2),
            "cagr_pct": round(cagr, 1),
            "max_dd_pct": round(p_max_dd, 1),
            "net_profit_pct": round(p_pct, 1),
            "weights": [round(float(w) * 100, 2) for w in weights]
        })

    out_payload = {
        "recommended_method": "Quarter Kelly (★ Recommended)",
        "allocation_models": allocation_comparison
    }
    CAPITAL_ALLOCATION_JSON.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    tbl = Table(title="Capital Allocation Models Comparison", show_lines=True)
    tbl.add_column("Sizing Method", style="bold white")
    tbl.add_column("Description", style="dim")
    tbl.add_column("Sharpe", justify="right", style="bold green")
    tbl.add_column("Calmar", justify="right", style="cyan")
    tbl.add_column("CAGR %", justify="right", style="green")
    tbl.add_column("Max DD %", justify="right", style="bold red")

    for m in allocation_comparison:
        tbl.add_row(m["method"], m["description"], f"{m['sharpe_ratio']:.2f}", f"{m['calmar_ratio']:.2f}", f"{m['cagr_pct']:.1f}%", f"{m['max_dd_pct']:.1f}%")
    console.print(tbl)

    d_tbl = Table(title="Dollar Deployment Breakdown (Quarter-Kelly Weights)", show_lines=True)
    d_tbl.add_column("Edge #", justify="center", style="cyan")
    d_tbl.add_column("Symbol", style="bold white")
    d_tbl.add_column("TF", justify="center", style="magenta")
    d_tbl.add_column("Strategy", style="yellow")
    d_tbl.add_column("Weight %", justify="right", style="bold green")
    d_tbl.add_column("$10,000 Account", justify="right")
    d_tbl.add_column("$50,000 Account", justify="right")
    d_tbl.add_column("$100,000 Account", justify="right")

    for i, edge in enumerate(edges):
        w = w_quarter_kelly[i]
        d_tbl.add_row(str(i + 1), edge["symbol"], edge["timeframe"], edge["strategy"], f"{w * 100.0:.1f}%", f"${10_000 * w:,.2f}", f"${50_000 * w:,.2f}", f"${100_000 * w:,.2f}")
    console.print(d_tbl)
    console.print(f"[bold green]✓ Sizing models saved to {CAPITAL_ALLOCATION_JSON}[/bold green]")
    console.rule("[bold green]PHASE 9.7 COMPLETE")

if __name__ == "__main__":
    run_capital_allocation()
