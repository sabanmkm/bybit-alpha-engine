"""
ISOLATED SECONDARY RESEARCH: FULL EXOTIC CANDIDATE OPTIMIZATION & AUDIT (v2).
Fixes numpy type serialization and renders the full integration scorecard.
Does NOT modify or overwrite the Primary Original 6 Portfolio.
"""
import sys, json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from scipy import stats
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

sys.path.insert(0, r"C:\BybitBacktest")
import config as cfg
cfg.DATA_GATE_UNLOCKED = True

from strategy_generator import get_strategy, STRATEGY_REGISTRY
from backtest_engine import run_backtest, BacktestResult

console = Console()

MINING_RESULTS_JSON   = cfg.RESULTS_DIR / "exotic_timeframe_discoveries.json"
CUSTOM_RESAMPLED_DIR  = cfg.ROOT / "data" / "resampled_exotic"
EXOTIC_OPTIMIZED_JSON = cfg.RESULTS_DIR / "exotic_optimized_edges.json"
PORTFOLIO_EDGES_JSON  = cfg.RESULTS_DIR / "portfolio_edges.json"

# Robust JSON converter for NumPy data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Load existing Original 6 4H portfolio for correlation comparison
def load_original_portfolio_returns() -> dict[str, pd.Series]:
    if not PORTFOLIO_EDGES_JSON.exists():
        return {}
    portfolio = json.loads(PORTFOLIO_EDGES_JSON.read_text(encoding="utf-8"))
    returns_dict = {}
    for edge in portfolio.get("edges", []):
        sym = edge["symbol"]
        tf = edge["timeframe"]
        fpath = cfg.RESAMPLED_DIR / tf / (sym.replace("/", "_").replace(":", "_") + ".parquet")
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        strat_meta = get_strategy(edge["strategy"])
        s_params = edge.get("tuned_strategy_params", {})
        signals = strat_meta["func"](df, **s_params)
        res = run_backtest(df, signals, sym, tf, edge["strategy"],
                           sl_atr_mult=edge.get("tuned_risk_params", {}).get("sl_atr_mult", 2.0),
                           tp_atr_mult=edge.get("tuned_risk_params", {}).get("tp_atr_mult", 4.0))
        daily_ret = res.equity_curve.resample("1D").last().ffill().pct_change().fillna(0.0)
        returns_dict[f"{sym}_{tf}"] = daily_ret
    return returns_dict


def suggest_strategy_params(trial, strat_name, default_params):
    params = {}
    for p_name, p_val in default_params.items():
        if isinstance(p_val, int):
            if "fast" in p_name or "tenkan" in p_name:
                params[p_name] = trial.suggest_int(p_name, 5, 25)
            elif "slow" in p_name or "kijun" in p_name or "mid" in p_name:
                params[p_name] = trial.suggest_int(p_name, 20, 70)
            elif "trend" in p_name or "filter" in p_name or "slow_p" in p_name:
                params[p_name] = trial.suggest_int(p_name, 50, 200, step=10)
            elif "period" in p_name or "lookback" in p_name or "window" in p_name:
                params[p_name] = trial.suggest_int(p_name, 5, 50)
            elif "bars" in p_name or "range" in p_name:
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
            else:
                params[p_name] = trial.suggest_float(p_name, p_val * 0.5, p_val * 2.0)
    return params


def optimize_single_exotic_edge(edge: dict) -> dict | None:
    sym = edge["symbol"]
    tf = edge["timeframe"]
    strat_name = edge["strategy"]

    fpath = CUSTOM_RESAMPLED_DIR / tf / (sym.replace("/", "_").replace(":", "_") + ".parquet")
    if not fpath.exists():
        return None

    df_full = pd.read_parquet(fpath)
    df_is = df_full[df_full.index <= pd.Timestamp(cfg.IS_END)].copy()
    df_oos = df_full[(df_full.index >= pd.Timestamp(cfg.OOS_START)) &
                      (df_full.index <= pd.Timestamp(cfg.END_DATE))].copy()

    if len(df_is) < 60 or len(df_oos) < 20:
        return None

    strat_meta = get_strategy(strat_name)
    strat_func = strat_meta["func"]
    default_params = strat_meta["default_params"]

    # ── ROUND 1: Risk Parameter Tuning (300 trials) ──────────────
    def r1_objective(trial):
        sl = trial.suggest_float("sl_atr_mult", 1.0, 5.0, step=0.5)
        tp = trial.suggest_float("tp_atr_mult", 1.5, 8.0, step=0.5)
        trail = trial.suggest_categorical("trailing_stop_atr", [0.0, 1.5, 2.5, 3.5])
        tstop = trial.suggest_int("time_stop_bars", 0, 60, step=10)
        sig = strat_func(df_is, **default_params)
        res = run_backtest(df_is, sig, sym, tf, strat_name,
                           sl_atr_mult=sl, tp_atr_mult=tp,
                           trailing_stop_atr=trail, time_stop_bars=tstop)
        if res.total_trades < 25 or res.max_drawdown_pct > 30:
            return -10.0
        return float(res.composite_score)

    study_r1 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_r1.optimize(r1_objective, n_trials=300)
    best_risk = study_r1.best_params

    # ── ROUND 2: Full Strategy + Risk Tuning with 3-Fold CV (800 trials) ──
    total_len = len(df_is)
    fold_size = total_len // 3
    folds = [
        df_is.iloc[:fold_size * 2].copy(),
        df_is.iloc[fold_size // 2:fold_size * 2 + fold_size // 2].copy(),
        df_is.iloc[fold_size:].copy()
    ]

    def r2_objective(trial):
        s_params = suggest_strategy_params(trial, strat_name, default_params)
        sl = trial.suggest_float("sl_atr_mult",
                                  max(1.0, best_risk.get("sl_atr_mult", 2.0) - 1.0),
                                  best_risk.get("sl_atr_mult", 2.0) + 1.5, step=0.5)
        tp = trial.suggest_float("tp_atr_mult",
                                  max(1.5, best_risk.get("tp_atr_mult", 4.0) - 1.5),
                                  best_risk.get("tp_atr_mult", 4.0) + 2.0, step=0.5)
        trail = trial.suggest_categorical("trailing_stop_atr", [0.0, 1.5, 2.5, 3.5])
        tstop = trial.suggest_int("time_stop_bars", 0, 60, step=10)

        fold_sharpes = []
        for fold_df in folds:
            if len(fold_df) < 30:
                return -10.0
            try:
                sig = strat_func(fold_df, **s_params)
                res = run_backtest(fold_df, sig, sym, tf, strat_name,
                                   sl_atr_mult=sl, tp_atr_mult=tp,
                                   trailing_stop_atr=trail, time_stop_bars=tstop)
                if res.total_trades < 8:
                    return -10.0
                fold_sharpes.append(res.sharpe_ratio)
            except Exception:
                return -10.0

        mean_sharpe = float(np.mean(fold_sharpes))
        std_sharpe = float(np.std(fold_sharpes))
        return mean_sharpe - (0.5 * std_sharpe)

    study_r2 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_r2.optimize(r2_objective, n_trials=800)

    best_combined = study_r2.best_params
    best_s_params = {k: v for k, v in best_combined.items() if k in default_params}
    if not best_s_params:
        best_s_params = default_params

    final_risk = {
        "sl_atr_mult": float(best_combined.get("sl_atr_mult", best_risk.get("sl_atr_mult", 2.0))),
        "tp_atr_mult": float(best_combined.get("tp_atr_mult", best_risk.get("tp_atr_mult", 4.0))),
        "trailing_stop_atr": float(best_combined.get("trailing_stop_atr", 0.0)),
        "time_stop_bars": int(best_combined.get("time_stop_bars", 0))
    }

    # ── Full IS Backtest with Tuned Params ────────────────────────
    sig_is = strat_func(df_is, **best_s_params)
    res_is = run_backtest(df_is, sig_is, sym, tf, strat_name, **final_risk)

    # ── 2026 OOS Walk-Forward with FROZEN Params ──────────────────
    sig_oos = strat_func(df_oos, **best_s_params)
    res_oos = run_backtest(df_oos, sig_oos, sym, tf, strat_name, **final_risk)

    is_sharpe = max(res_is.sharpe_ratio, 0.01)
    oos_sharpe = res_oos.sharpe_ratio
    sharpe_retention = float(oos_sharpe / is_sharpe)
    is_dd = max(res_is.max_drawdown_pct, 1.0)
    oos_dd = res_oos.max_drawdown_pct
    dd_expansion = float(oos_dd / is_dd)

    wf_pass = bool((oos_sharpe >= 0.50 * is_sharpe or oos_sharpe >= 1.0) and (oos_dd <= 1.5 * is_dd) and (res_oos.profit_factor >= 1.05))

    # ── Monte Carlo 10,000x ───────────────────────────────────────
    trade_pnls = [t.pnl for t in res_is.trades]
    mc_max_dds = []
    if len(trade_pnls) >= 10:
        arr = np.array(trade_pnls)
        for _ in range(10000):
            shuffled = np.random.permutation(arr)
            eq = cfg.START_CAPITAL + np.cumsum(shuffled)
            peak = np.maximum.accumulate(eq)
            dd = abs(np.min((eq - peak) / peak)) * 100.0
            mc_max_dds.append(dd)

    mc_95_dd = float(np.percentile(mc_max_dds, 95)) if mc_max_dds else 0.0

    return {
        "symbol": sym,
        "timeframe": tf,
        "strategy": strat_name,
        "tuned_strategy_params": best_s_params,
        "tuned_risk_params": final_risk,
        "is_trades": int(res_is.total_trades),
        "is_sharpe": round(float(res_is.sharpe_ratio), 2),
        "is_calmar": round(float(res_is.calmar_ratio), 2),
        "is_max_dd": round(float(res_is.max_drawdown_pct), 1),
        "is_win_rate": round(float(res_is.win_rate * 100), 1),
        "is_pf": round(float(res_is.profit_factor), 2),
        "oos_trades": int(res_oos.total_trades),
        "oos_sharpe": round(float(res_oos.sharpe_ratio), 2),
        "oos_return_pct": round(float(res_oos.net_profit_pct), 1),
        "oos_max_dd": round(float(res_oos.max_drawdown_pct), 1),
        "oos_win_rate": round(float(res_oos.win_rate * 100), 1),
        "oos_pf": round(float(res_oos.profit_factor), 2),
        "sharpe_retention": round(sharpe_retention, 2),
        "dd_expansion": round(dd_expansion, 2),
        "wf_pass": wf_pass,
        "mc_95_dd": round(mc_95_dd, 1),
        "mc_safe": bool(mc_95_dd < 25.0)
    }


def main():
    console.rule("[bold magenta]ISOLATED EXPERIMENTAL OPTIMIZATION: 5 CONFIRMED EXOTIC EDGES")

    if not MINING_RESULTS_JSON.exists():
        console.print("[bold red]No mining results found! Run deep_timeframe_miner.py first.[/]")
        return

    discoveries = json.loads(MINING_RESULTS_JSON.read_text(encoding="utf-8"))
    confirmed = [d for d in discoveries if d.get("verdict") == "CONFIRMED_EDGE"]

    if not confirmed:
        console.print("[yellow]No confirmed exotic edges from the miner.[/]")
        return

    console.print(f"[green]Optimizing {len(confirmed)} confirmed edges: Round 1 (300) + Round 2 (800) + 10k MC...[/green]\n")

    optimized_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=35, complete_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console
    ) as prog:
        task = prog.add_task("Optimizing...", total=len(confirmed))

        for edge in confirmed:
            sym_tf = f"{edge['symbol']} ({edge['timeframe']})"
            prog.update(task, description=f"Tuning [yellow]{sym_tf:<18}[/] [cyan]{edge['strategy'][:16]}[/]")

            try:
                result = optimize_single_exotic_edge(edge)
                if result:
                    optimized_results.append(result)
                    status = "[bold green]PASS[/]" if result["wf_pass"] else "[bold red]FAIL[/]"
                    console.print(
                        f"  {status} {sym_tf} {edge['strategy']} "
                        f"| IS: {result['is_sharpe']:.2f} → OOS: {result['oos_sharpe']:.2f} "
                        f"| Retain: {int(result['sharpe_retention']*100)}% "
                        f"| MC95 DD: {result['mc_95_dd']:.1f}%"
                    )
            except Exception as e:
                console.print(f"  [red]Error on {sym_tf}: {e}[/]")

            prog.advance(task)

    if not optimized_results:
        console.print("[red]No optimization results generated.[/]")
        return

    # Safe JSON dump using custom encoder
    EXOTIC_OPTIMIZED_JSON.write_text(json.dumps(optimized_results, cls=NumpyEncoder, indent=2), encoding="utf-8")

    # ── Correlation Audit vs Original 6 ──────────────────────────
    console.print("\n[bold yellow]Calculating pairwise return correlation vs. Original 6 Portfolio...[/]")
    original_returns = load_original_portfolio_returns()

    for r in optimized_results:
        sym = r["symbol"]
        tf = r["timeframe"]
        fpath = CUSTOM_RESAMPLED_DIR / tf / (sym.replace("/", "_").replace(":", "_") + ".parquet")
        if not fpath.exists():
            r["max_corr_with_orig6"] = 1.0
            continue

        df = pd.read_parquet(fpath)
        strat_meta = get_strategy(r["strategy"])
        s_params = r.get("tuned_strategy_params", {})
        risk = r.get("tuned_risk_params", {})
        signals = strat_meta["func"](df, **s_params)
        res = run_backtest(df, signals, sym, tf, r["strategy"], **risk)
        daily_ret = res.equity_curve.resample("1D").last().ffill().pct_change().fillna(0.0)

        max_corr = 0.0
        for orig_key, orig_ret in original_returns.items():
            common = daily_ret.index.intersection(orig_ret.index)
            if len(common) > 30:
                corr = abs(float(daily_ret.loc[common].corr(orig_ret.loc[common])))
                max_corr = max(max_corr, corr)

        r["max_corr_with_orig6"] = round(float(max_corr), 3)

    # ── Comparison & Integration Scorecard ────────────────────────
    console.rule("[bold magenta]EXOTIC CANDIDATE SCORECARD (COMPARED AGAINST ORIGINAL 6)")

    tbl = Table(title="Exotic Candidate Full Audit (2-Round Optuna + Walk-Forward + 10k MC + Correlation)", show_lines=True)
    tbl.add_column("Symbol", style="bold white")
    tbl.add_column("TF", justify="center", style="bold magenta")
    tbl.add_column("Strategy", style="yellow")
    tbl.add_column("IS Sharpe", justify="right", style="green")
    tbl.add_column("2026 OOS Sharpe", justify="right", style="bold green")
    tbl.add_column("Retention", justify="right", style="cyan")
    tbl.add_column("2026 OOS DD", justify="right", style="red")
    tbl.add_column("2026 OOS PF", justify="right")
    tbl.add_column("MC 95% DD", justify="right", style="red")
    tbl.add_column("Max Corr vs Orig 6", justify="right", style="yellow")
    tbl.add_column("WF Pass", justify="center")
    tbl.add_column("Recommended?", justify="center")

    recommended_edges = []

    for r in sorted(optimized_results, key=lambda x: x.get("oos_sharpe", 0), reverse=True):
        wf = "[bold green]YES[/]" if r["wf_pass"] else "[bold red]NO[/]"
        low_corr = r["max_corr_with_orig6"] < 0.55
        mc_safe = r["mc_safe"]
        is_recommended = r["wf_pass"] and low_corr and mc_safe and r["oos_sharpe"] >= 1.0
        rec_str = "[bold green]★ TOP CANDIDATE[/]" if is_recommended else "[dim]NO[/]"

        if is_recommended:
            recommended_edges.append(r)

        tbl.add_row(
            r["symbol"], r["timeframe"], r["strategy"],
            f"{r['is_sharpe']:.2f}", f"{r['oos_sharpe']:.2f}",
            f"{int(r['sharpe_retention']*100)}%",
            f"{r['oos_max_dd']:.1f}%", f"{r['oos_pf']:.2f}",
            f"{r['mc_95_dd']:.1f}%", f"{r['max_corr_with_orig6']:.3f}",
            wf, rec_str
        )

    console.print(tbl)

    console.print(Panel(
        f"[bold white]ORIGINAL 6 VAULT STATUS:[/bold white] [bold green]100% UNTOUCHED & PROTECTED[/bold green]\n"
        f"• All experimental results saved separately to [bold cyan]results/exotic_optimized_edges.json[/bold cyan].\n"
        f"• Original portfolio remains intact in [bold green]results/ORIGINAL_6_PORTFOLIO_BACKUP.json[/bold green].",
        title="[bold green]SAFETY & ISOLATION CONFIRMATION[/bold green]",
        border_style="green"
    ))

    console.rule("[bold green]EXOTIC OPTIMIZATION COMPLETE")

if __name__ == "__main__":
    main()
