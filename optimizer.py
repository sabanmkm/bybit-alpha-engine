"""
Phase 7 — Two-Round Optuna Optimizer with 3-Fold Time-Series Cross-Validation.
- Round 1: Risk Parameters (SL / TP / Trailing Stop / Time Stop ATR multiples)
- Round 2: Full Strategy Parameters + Indicator Lookbacks + Rolling Time-Series CV
- Enforces strict In-Sample boundaries (2022-2025) and outputs tuned candidates for portfolio building.
"""
from __future__ import annotations
import json, sys, os, warnings
from pathlib import Path
import optuna
import pandas as pd
import numpy as np
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
)
from rich.table import Table

# Suppress Optuna chatty output
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

sys.path.insert(0, r"C:\BybitBacktest")
import config as cfg
from strategy_generator import STRATEGY_REGISTRY, get_strategy
from backtest_engine import run_backtest, BacktestResult

console = Console()
TOP_CANDIDATES_JSON = cfg.RESULTS_DIR / "top_candidates.json"
TUNED_STRATEGIES_JSON = cfg.RESULTS_DIR / "tuned_strategies.json"


def get_cv_folds(df: pd.DataFrame, n_folds: int = 3) -> list[pd.DataFrame]:
    """Splits in-sample data into n sequential rolling/expanding folds for cross-validation."""
    total_len = len(df)
    fold_size = total_len // n_folds
    folds = []
    for k in range(n_folds):
        start_idx = k * (fold_size // 2)  # overlapping rolling slices
        end_idx = min(start_idx + fold_size * 2, total_len)
        folds.append(df.iloc[start_idx:end_idx].copy())
    return folds


def suggest_strategy_params(trial: optuna.Trial, strat_name: str, default_params: dict) -> dict:
    """Dynamically suggests parameter search space based on strategy parameter names."""
    params = {}
    for p_name, p_val in default_params.items():
        if isinstance(p_val, int):
            if "fast" in p_name or "tenkan" in p_name:
                params[p_name] = trial.suggest_int(p_name, 5, 25)
            elif "slow" in p_name or "kijun" in p_name or "mid" in p_name:
                params[p_name] = trial.suggest_int(p_name, 20, 70)
            elif "trend" in p_name or "filter" in p_name or "slow_p" in p_name or "senkou" in p_name:
                params[p_name] = trial.suggest_int(p_name, 50, 200, step=10)
            elif "period" in p_name or "lookback" in p_name or "window" in p_name:
                params[p_name] = trial.suggest_int(p_name, 5, 50)
            elif "bars" in p_name:
                params[p_name] = trial.suggest_int(p_name, 3, 20)
            elif "thresh" in p_name or "bound" in p_name:
                params[p_name] = trial.suggest_int(p_name, 15, 85)
            else:
                params[p_name] = trial.suggest_int(p_name, max(2, int(p_val * 0.5)), int(p_val * 2.0))
        elif isinstance(p_val, float):
            if "mult" in p_name or "std" in p_name or "ratio" in p_name:
                params[p_name] = trial.suggest_float(p_name, 1.0, 4.5, step=0.25)
            elif "pct" in p_name or "tolerance" in p_name:
                params[p_name] = trial.suggest_float(p_name, 0.005, 0.05, step=0.005)
            elif "step" in p_name:
                params[p_name] = trial.suggest_float(p_name, 0.01, 0.05, step=0.01)
            else:
                params[p_name] = trial.suggest_float(p_name, p_val * 0.5, p_val * 2.0)
    return params


def optimize_single_candidate(candidate: dict) -> dict:
    """Performs Round 1 (Risk Tuning) and Round 2 (Full Parameter + CV Tuning) on one candidate."""
    sym = candidate["symbol"]
    tf = candidate["timeframe"]
    strat_name = candidate["strategy"]
    fpath = cfg.RESAMPLED_DIR / tf / (sym.replace("/", "_").replace(":", "_") + ".parquet")

    if not fpath.exists():
        return candidate

    df_full = pd.read_parquet(fpath)
    df_is = df_full[df_full.index <= pd.Timestamp(cfg.IS_END)].copy()
    cfg.assert_no_future_data(df_is, phase_name=f"Optuna_{sym}_{strat_name}")

    strat_meta = get_strategy(strat_name)
    strat_func = strat_meta["func"]
    default_params = strat_meta["default_params"]

    # -------------------------------------------------------------
    # ROUND 1: Risk Parameter Optimization
    # -------------------------------------------------------------
    def r1_objective(trial: optuna.Trial) -> float:
        sl_atr = trial.suggest_float("sl_atr_mult", 1.0, 5.0, step=0.5)
        tp_atr = trial.suggest_float("tp_atr_mult", 1.5, 10.0, step=0.5)
        trail_atr = trial.suggest_categorical("trailing_stop_atr", [0.0, 1.5, 2.5, 3.5])
        time_stop = trial.suggest_int("time_stop_bars", 0, 80, step=10)

        signals = strat_func(df_is, **default_params)
        res = run_backtest(
            df=df_is,
            signals=signals,
            symbol=sym,
            timeframe=tf,
            strategy_name=strat_name,
            sl_atr_mult=sl_atr,
            tp_atr_mult=tp_atr,
            trailing_stop_atr=trail_atr,
            time_stop_bars=time_stop
        )

        if res.total_trades < 30 or res.max_drawdown_pct > 30.0:
            return -10.0
        return float(res.composite_score)

    study_r1 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_r1.optimize(r1_objective, n_trials=min(cfg.OPTUNA_TRIALS_ROUND1, 150))
    best_risk_params = study_r1.best_params

    # -------------------------------------------------------------
    # ROUND 2: Full Strategy Tuning with 3-Fold Time-Series CV
    # -------------------------------------------------------------
    cv_folds = get_cv_folds(df_is, n_folds=cfg.CV_FOLDS)

    def r2_objective(trial: optuna.Trial) -> float:
        # Suggest Strategy Parameters
        s_params = suggest_strategy_params(trial, strat_name, default_params)
        # Suggest Risk Parameters centered around Round 1 best
        sl_atr = trial.suggest_float("sl_atr_mult", max(1.0, best_risk_params.get("sl_atr_mult", 2.0) - 1.0), best_risk_params.get("sl_atr_mult", 2.0) + 1.5, step=0.5)
        tp_atr = trial.suggest_float("tp_atr_mult", max(1.5, best_risk_params.get("tp_atr_mult", 4.0) - 1.5), best_risk_params.get("tp_atr_mult", 4.0) + 2.0, step=0.5)
        trail_atr = trial.suggest_categorical("trailing_stop_atr", [0.0, 1.5, 2.5, 3.5])
        time_stop = trial.suggest_int("time_stop_bars", 0, 80, step=10)

        fold_sharpes = []
        fold_dds = []

        for fold_df in cv_folds:
            try:
                sig_fold = strat_func(fold_df, **s_params)
                res_fold = run_backtest(
                    df=fold_df,
                    signals=sig_fold,
                    symbol=sym,
                    timeframe=tf,
                    strategy_name=strat_name,
                    sl_atr_mult=sl_atr,
                    tp_atr_mult=tp_atr,
                    trailing_stop_atr=trail_atr,
                    time_stop_bars=time_stop
                )
                if res_fold.total_trades < 10:
                    return -10.0
                fold_sharpes.append(res_fold.sharpe_ratio)
                fold_dds.append(res_fold.max_drawdown_pct)
            except Exception:
                return -10.0

        mean_sharpe = float(np.mean(fold_sharpes))
        std_sharpe = float(np.std(fold_sharpes))
        max_fold_dd = float(np.max(fold_dds))

        # Penalize cross-fold instability & heavy drawdown
        if max_fold_dd > 25.0:
            return -10.0

        cv_score = mean_sharpe - (0.5 * std_sharpe)
        return cv_score

    study_r2 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_r2.optimize(r2_objective, n_trials=min(cfg.OPTUNA_TRIALS_ROUND2, 400))

    best_combined = study_r2.best_params
    best_s_params = {k: v for k, v in best_combined.items() if k in default_params}
    if not best_s_params:
        best_s_params = default_params

    # Run full In-Sample test with final tuned parameters
    final_signals = strat_func(df_is, **best_s_params)
    final_res = run_backtest(
        df=df_is,
        signals=final_signals,
        symbol=sym,
        timeframe=tf,
        strategy_name=strat_name,
        sl_atr_mult=best_combined.get("sl_atr_mult", best_risk_params.get("sl_atr_mult", 2.0)),
        tp_atr_mult=best_combined.get("tp_atr_mult", best_risk_params.get("tp_atr_mult", 4.0)),
        trailing_stop_atr=best_combined.get("trailing_stop_atr", best_risk_params.get("trailing_stop_atr", 0.0)),
        time_stop_bars=best_combined.get("time_stop_bars", best_risk_params.get("time_stop_bars", 0))
    )

    return {
        "symbol": sym,
        "timeframe": tf,
        "strategy": strat_name,
        "category": candidate.get("category", "General"),
        "tuned_strategy_params": best_s_params,
        "tuned_risk_params": {
            "sl_atr_mult": best_combined.get("sl_atr_mult", best_risk_params.get("sl_atr_mult", 2.0)),
            "tp_atr_mult": best_combined.get("tp_atr_mult", best_risk_params.get("tp_atr_mult", 4.0)),
            "trailing_stop_atr": best_combined.get("trailing_stop_atr", best_risk_params.get("trailing_stop_atr", 0.0)),
            "time_stop_bars": best_combined.get("time_stop_bars", best_risk_params.get("time_stop_bars", 0))
        },
        # Before Tuning Metrics
        "before_sharpe": candidate.get("sharpe", 0.0),
        "before_calmar": candidate.get("calmar", 0.0),
        "before_max_dd": candidate.get("max_dd", 0.0),
        "before_win_rate": candidate.get("win_rate", 0.0),
        # After Tuning Metrics
        "after_trades": final_res.total_trades,
        "after_sharpe": round(final_res.sharpe_ratio, 2),
        "after_calmar": round(final_res.calmar_ratio, 2),
        "after_sortino": round(final_res.sortino_ratio, 2),
        "after_profit_factor": round(final_res.profit_factor, 2),
        "after_max_dd": round(final_res.max_drawdown_pct, 2),
        "after_win_rate": round(final_res.win_rate * 100, 2),
        "after_cagr": round(final_res.cagr, 2),
        "after_composite_score": round(final_res.composite_score, 4)
    }


def main():
    console.rule("[bold cyan]PHASE 7 — Hyperparameter Optimization & Cross-Validation")

    if not TOP_CANDIDATES_JSON.exists():
        console.print("[bold red]No top candidates found![/] Please run [yellow]python run_all_backtests.py[/] first.")
        return

    candidates = json.loads(TOP_CANDIDATES_JSON.read_text(encoding="utf-8"))
    console.print(f"[green]Loaded {len(candidates)} candidates for 2-round Optuna optimization...[/green]\n")

    tuned_results = []

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
        task = progress.add_task("Tuning candidates...", total=len(candidates))

        for cand in candidates:
            desc = f"Tuning [{cand['timeframe']}] [yellow]{cand['symbol'][:10]}[/] [cyan]{cand['strategy'][:18]}[/]"
            progress.update(task, description=desc)

            try:
                res = optimize_single_candidate(cand)
                tuned_results.append(res)
                console.print(
                    f"  [bold green]✓ TUNED[/] {cand['symbol']} ({cand['timeframe']}) {cand['strategy']} "
                    f"| Sharpe: {cand.get('sharpe', 0.0)} -> [bold green]{res['after_sharpe']}[/] "
                    f"| MaxDD: {cand.get('max_dd', 0.0)}% -> [bold cyan]{res['after_max_dd']}%[/]"
                )
            except Exception as e:
                console.print(f"  [red]Failed {cand['symbol']} {cand['strategy']}: {e}[/]")

            progress.advance(task)

    # Save all tuned strategies
    TUNED_STRATEGIES_JSON.write_text(json.dumps(tuned_results, indent=2), encoding="utf-8")
    console.print(f"\n[bold green]✓ Saved {len(tuned_results)} tuned edges to {TUNED_STRATEGIES_JSON}[/bold green]\n")

    # ── Comparison Summary Table ─────────────────────────────────────
    tbl = Table(title="Phase 7 Tuning Comparison (Before vs. After)", show_lines=True)
    tbl.add_column("Symbol", style="bold white")
    tbl.add_column("TF", justify="center", style="magenta")
    tbl.add_column("Strategy", style="yellow")
    tbl.add_column("Sharpe (B4 → After)", justify="center", style="bold green")
    tbl.add_column("Calmar (B4 → After)", justify="center", style="cyan")
    tbl.add_column("Max DD % (B4 → After)", justify="center", style="bold red")
    tbl.add_column("Win %", justify="right", style="green")
    tbl.add_column("Trades", justify="right")
    tbl.add_column("Score", justify="right", style="bold yellow")

    for r in sorted(tuned_results, key=lambda x: x.get("after_composite_score", 0), reverse=True):
        b_sh, a_sh = r.get("before_sharpe", 0.0), r.get("after_sharpe", 0.0)
        b_dd, a_dd = r.get("before_max_dd", 0.0), r.get("after_max_dd", 0.0)
        b_cal, a_cal = r.get("before_calmar", 0.0), r.get("after_calmar", 0.0)

        tbl.add_row(
            r["symbol"],
            r["timeframe"],
            r["strategy"],
            f"{b_sh:.2f} → [bold]{a_sh:.2f}[/]",
            f"{b_cal:.2f} → {a_cal:.2f}",
            f"{b_dd:.1f}% → [bold]{a_dd:.1f}%[/]",
            f"{r.get('after_win_rate', 0.0):.1f}%",
            str(r.get("after_trades", 0)),
            f"{r.get('after_composite_score', 0):.3f}"
        )

    console.print(tbl)
    console.rule("[bold green]PHASE 7 COMPLETE")

if __name__ == "__main__":
    main()
