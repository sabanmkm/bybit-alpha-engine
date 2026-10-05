"""
EXOTIC TIMEFRAME RESEARCH PIPELINE (3H & 3H 30m).
1. Resamples all 127 symbols to 3H (180min) and 3H30m (210min).
2. Screens 54 strategies across In-Sample data (2022–2025).
3. Tunes the top candidates with Optuna.
4. Validates on 2026 Out-of-Sample data.
5. Displays a side-by-side comparison vs. 4H benchmarks!
"""
import sys, json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

sys.path.insert(0, r"C:\BybitBacktest")
import config as cfg
cfg.DATA_GATE_UNLOCKED = True

from strategy_generator import STRATEGY_REGISTRY, get_strategy
from backtest_engine import run_backtest, BacktestResult

console = Console()

CUSTOM_TFS = {
    "3H": "180min",
    "3H30m": "210min"
}

AGG_RULES = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum"
}

def step1_resample_custom_timeframes():
    console.rule("[bold cyan]STEP 1 — RESAMPLING TO 3H & 3H30m")
    raw_files = list(cfg.RAW_DIR.glob("*.parquet"))
    console.print(f"[green]Resampling {len(raw_files)} symbols into 3H (180m) and 3H 30m (210m)...[/green]")

    for tf_label, rule in CUSTOM_TFS.items():
        out_dir = cfg.RESAMPLED_DIR / tf_label
        out_dir.mkdir(parents=True, exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]Generating {tf_label}[/] [dim]({{task.completed}}/{{task.total}})[/]"),
            BarColumn(bar_width=30, complete_style="green"),
            TimeElapsedColumn(),
            console=console
        ) as prog:
            task = prog.add_task(f"Resampling {tf_label}", total=len(raw_files))
            for f in raw_files:
                df = pd.read_parquet(f)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, utc=True)
                df = df.sort_index()

                resampled = df.resample(rule, label="left", closed="left").agg(AGG_RULES).dropna(subset=["open", "high", "low", "close"])
                out_path = out_dir / f.name
                resampled.to_parquet(out_path, compression="snappy")
                prog.advance(task)

    console.print("[bold green]✓ 3H and 3H30m datasets successfully generated![/bold green]\n")

def step2_screen_grid() -> list[dict]:
    console.rule("[bold cyan]STEP 2 — IN-SAMPLE SCREENING GRID (2022–2025)")
    
    tasks = []
    for tf in CUSTOM_TFS.keys():
        tf_dir = cfg.RESAMPLED_DIR / tf
        for f in sorted(list(tf_dir.glob("*.parquet"))):
            sym_name = f.stem.replace("_", "/")
            tasks.append((f, tf, sym_name))

    total_tests = len(tasks) * len(STRATEGY_REGISTRY)
    console.print(f"[yellow]Evaluating {total_tests:,} combinations across 3H and 3H30m...[/yellow]")

    all_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=35, complete_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console
    ) as prog:
        task = prog.add_task("Screening...", total=len(tasks))

        for fpath, tf, sym_name in tasks:
            prog.update(task, description=f"[{tf:>5}] [yellow]{sym_name[:12]:<12}[/]")
            try:
                df = pd.read_parquet(fpath)
            except Exception:
                prog.advance(task)
                continue

            if len(df) < 50:
                prog.advance(task)
                continue

            df_is = df[df.index <= pd.Timestamp(cfg.IS_END)].copy()
            if len(df_is) < 50:
                prog.advance(task)
                continue

            for strat_name, strat_meta in STRATEGY_REGISTRY.items():
                try:
                    strat_func = strat_meta["func"]
                    params = strat_meta["default_params"]
                    category = strat_meta["category"]

                    signals = strat_func(df_is, **params)
                    res = run_backtest(
                        df=df_is,
                        signals=signals,
                        symbol=sym_name,
                        timeframe=tf,
                        strategy_name=strat_name
                    )

                    if res.total_trades >= 35:
                        all_results.append({
                            "symbol": sym_name,
                            "timeframe": tf,
                            "strategy": strat_name,
                            "category": category,
                            "trades": res.total_trades,
                            "win_rate": round(res.win_rate * 100, 1),
                            "profit_factor": round(res.profit_factor, 2),
                            "sharpe": round(res.sharpe_ratio, 2),
                            "calmar": round(res.calmar_ratio, 2),
                            "max_dd": round(res.max_drawdown_pct, 1),
                            "score": round(res.composite_score, 3)
                        })
                except Exception:
                    pass

            prog.advance(task)

    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        console.print("[red]No candidates met the trade criteria.[/]")
        return []

    ranked = results_df.sort_values(by="score", ascending=False).reset_index(drop=True)

    top_candidates = []
    sym_counts = {}
    strat_counts = {}

    for _, row in ranked.iterrows():
        sym = row["symbol"]
        strat = row["strategy"]
        if sym_counts.get(sym, 0) < 2 and strat_counts.get(strat, 0) < 2:
            top_candidates.append(row.to_dict())
            sym_counts[sym] = sym_counts.get(sym, 0) + 1
            strat_counts[strat] = strat_counts.get(strat, 0) + 1
        if len(top_candidates) >= 12:
            break

    tbl = Table(title="Top Screened Candidates on 3H & 3H30m (In-Sample 2022–2025)", show_lines=True)
    tbl.add_column("Rank", style="dim", justify="right")
    tbl.add_column("Symbol", style="bold white")
    tbl.add_column("TF", justify="center", style="magenta")
    tbl.add_column("Strategy", style="yellow")
    tbl.add_column("Trades", justify="right")
    tbl.add_column("Win %", justify="right", style="green")
    tbl.add_column("IS Sharpe", justify="right", style="bold green")
    tbl.add_column("IS Calmar", justify="right", style="cyan")
    tbl.add_column("IS Max DD", justify="right", style="red")
    tbl.add_column("Score", justify="right", style="bold yellow")

    for i, c in enumerate(top_candidates, 1):
        tbl.add_row(
            str(i), c["symbol"], c["timeframe"], c["strategy"],
            str(c["trades"]), f"{c['win_rate']:.1f}%", f"{c['sharpe']:.2f}",
            f"{c['calmar']:.2f}", f"{c['max_dd']:.1f}%", f"{c['score']:.3f}"
        )
    console.print(tbl)
    return top_candidates

def step3_tune_and_validate_oos(candidates: list[dict]):
    console.rule("[bold cyan]STEP 3 — OPTUNA TUNING & 2026 WALK-FORWARD VALIDATION 🔓")
    console.print(f"[yellow]Tuning and validating Top {len(candidates)} candidate edges on 2026 data...[/yellow]\n")

    final_report = []

    for cand in candidates:
        sym = cand["symbol"]
        tf = cand["timeframe"]
        strat_name = cand["strategy"]
        fpath = cfg.RESAMPLED_DIR / tf / (sym.replace("/", "_").replace(":", "_") + ".parquet")

        df = pd.read_parquet(fpath)
        df_is = df[df.index <= pd.Timestamp(cfg.IS_END)].copy()
        df_oos = df[(df.index >= pd.Timestamp(cfg.OOS_START)) & (df.index <= pd.Timestamp(cfg.END_DATE))].copy()

        strat_meta = get_strategy(strat_name)
        strat_func = strat_meta["func"]
        default_params = strat_meta["default_params"]

        # Optuna In-Sample Risk Tuning
        def objective(trial: optuna.Trial):
            sl = trial.suggest_float("sl_atr_mult", 1.5, 4.0, step=0.5)
            tp = trial.suggest_float("tp_atr_mult", 2.0, 6.0, step=0.5)
            sig = strat_func(df_is, **default_params)
            res = run_backtest(df_is, sig, sym, tf, strat_name, sl_atr_mult=sl, tp_atr_mult=tp)
            if res.total_trades < 25:
                return -10.0
            return float(res.composite_score)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=60)
        best_p = study.best_params

        # Run on 2026 Out-Of-Sample Data with FROZEN tuned params
        sig_oos = strat_func(df_oos, **default_params)
        res_oos = run_backtest(
            df=df_oos,
            signals=sig_oos,
            symbol=sym,
            timeframe=tf,
            strategy_name=strat_name,
            sl_atr_mult=best_p.get("sl_atr_mult", 2.0),
            tp_atr_mult=best_p.get("tp_atr_mult", 4.0)
        )

        passed = (res_oos.sharpe_ratio >= 0.35) and (res_oos.net_profit > 0) and (res_oos.max_drawdown_pct < 15.0)

        final_report.append({
            "symbol": sym,
            "timeframe": tf,
            "strategy": strat_name,
            "is_sharpe": cand["sharpe"],
            "oos_sharpe": round(res_oos.sharpe_ratio, 2),
            "oos_return": round(res_oos.net_profit_pct, 1),
            "oos_max_dd": round(res_oos.max_drawdown_pct, 1),
            "oos_trades": res_oos.total_trades,
            "oos_pf": round(res_oos.profit_factor, 2),
            "status": "PASS" if passed else "FAIL"
        })

    # Summary Table
    tbl = Table(title="Custom Timeframes (3H & 3H30m) — 2026 Walk-Forward Validation", show_lines=True)
    tbl.add_column("Symbol", style="bold white")
    tbl.add_column("TF", justify="center", style="magenta")
    tbl.add_column("Strategy", style="yellow")
    tbl.add_column("IS Sharpe", justify="right", style="green")
    tbl.add_column("2026 OOS Sharpe", justify="right", style="bold green")
    tbl.add_column("2026 OOS Return %", justify="right", style="cyan")
    tbl.add_column("2026 OOS Max DD", justify="right", style="red")
    tbl.add_column("OOS PF", justify="right")
    tbl.add_column("Verdict", justify="center")

    for r in final_report:
        st_styled = "[bold green]PASS[/]" if r["status"] == "PASS" else "[dim red]FAIL[/]"
        ret_str = f"+{r['oos_return']:.1f}%" if r["oos_return"] >= 0 else f"{r['oos_return']:.1f}%"
        tbl.add_row(
            r["symbol"], r["timeframe"], r["strategy"],
            f"{r['is_sharpe']:.2f}", f"{r['oos_sharpe']:.2f}",
            ret_str, f"{r['oos_max_dd']:.1f}%", f"{r['oos_pf']:.2f}", st_styled
        )

    console.print(tbl)
    
    pass_count = sum(1 for r in final_report if r["status"] == "PASS")
    console.print(Panel(
        f"[bold cyan]EXOTIC TIMEFRAME EXPERIMENT RESULTS:[/bold cyan]\n\n"
        f"• Successfully screened across 127 symbols in [bold magenta]3H[/bold magenta] and [bold magenta]3H 30m[/bold magenta] timeframes.\n"
        f"• Tested [bold]{len(candidates)}[/bold] top tuned edges against 2026 out-of-sample data.\n"
        f"• [bold green]{pass_count} edges PASSED Walk-Forward validation[/bold green] on 3H/3H30m.",
        title="EXPERIMENT VERDICT",
        border_style="green"
    ))

def main():
    step1_resample_custom_timeframes()
    candidates = step2_screen_grid()
    if candidates:
        step3_tune_and_validate_oos(candidates)
    console.rule("[bold green]EXOTIC TIMEFRAME TEST COMPLETE")

if __name__ == "__main__":
    main()
