"""
Master Orchestrator — Bybit Quantitative Backtesting & Walk-Forward Pipeline.
Runs the entire quantitative research pipeline end-to-end or lets you execute individual phases.
"""
from __future__ import annotations
import sys, os, argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

sys.path.insert(0, r"C:\BybitBacktest")
import config as cfg

console = Console()

def print_banner():
    banner_text = """
 [bold cyan]========================================================================[/]
 [bold green]       BYBIT QUANTITATIVE BACKTESTING & WALK-FORWARD PIPELINE           [/]
 [bold yellow]        100% Anti-Lookahead • 2-Round Optuna • Monte Carlo • Kelly     [/]
 [bold cyan]========================================================================[/]
    """
    console.print(banner_text)

def run_phase(phase_name: str, script_name: str):
    console.rule(f"[bold magenta]STARTING {phase_name}[/]")
    exit_code = os.system(f"python {cfg.ROOT / script_name}")
    if exit_code != 0:
        console.print(f"[bold red]Error in {script_name} (Exit code: {exit_code})[/]")
    else:
        console.print(f"[bold green]✓ {phase_name} completed successfully[/]\n")

def run_end_to_end():
    console.print("[bold yellow]Executing Complete Quantitative Pipeline (Phases 6 to 10)...[/]\n")
    # Phase 6: Initial Screening Grid
    run_phase("Phase 6: Exhaustive Backtest Grid", "run_all_backtests.py")
    # Phase 7: Optuna Hyperparameter Optimization
    run_phase("Phase 7: Two-Round Optuna Tuning", "optimizer.py")
    # Phase 8: Portfolio Construction
    run_phase("Phase 8: Diversified Portfolio Builder", "portfolio_builder.py")
    # Phase 9: Walk-Forward Validation (2026 OOS)
    run_phase("Phase 9: Walk-Forward Validation (2026)", "walkforward.py")
    # Phase 9.5: Robustness Suite
    run_phase("Phase 9.5: Robustness & Monte Carlo", "robustness.py")
    # Phase 9.7: Capital Sizing
    run_phase("Phase 9.7: Capital Allocation", "capital_allocator.py")
    # Phase 10: Final Comprehensive Report
    run_phase("Phase 10: Final HTML & Graphic Report", "report_generator.py")

def interactive_menu():
    while True:
        print_banner()
        console.print("[bold white]Select a Pipeline Phase to Execute:[/]")
        console.print("  [cyan]1[/] - Phase 1: Download 5m Data from Bybit")
        console.print("  [cyan]2[/] - Phase 2: Multi-Timeframe Resampling (15m, 30m, 1H, 4H, 1D)")
        console.print("  [cyan]3[/] - Phase 6: Run Exhaustive In-Sample Backtest Grid (2022-2025)")
        console.print("  [cyan]4[/] - Phase 7: Run Two-Round Optuna Optimizer with Rolling CV")
        console.print("  [cyan]5[/] - Phase 8: Build Diversified Low-Correlation Portfolio")
        console.print("  [cyan]6[/] - Phase 9: Run Walk-Forward Validation on 2026 OOS Data 🔓")
        console.print("  [cyan]7[/] - Phase 9.5: Run Robustness, Monte Carlo & Deflated Sharpe Suite")
        console.print("  [cyan]8[/] - Phase 9.7: Run Capital Allocation & Kelly Sizing Engine")
        console.print("  [cyan]9[/] - Phase 10: Generate Final Executive HTML Report & Charts")
        console.print("  [bold green]A[/] - [bold green]RUN ALL PHASES END-TO-END (Phases 6 -> 10)[/]")
        console.print("  [red]Q[/] - Exit\n")

        choice = Prompt.ask("[bold yellow]Enter your choice[/]", choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "a", "Q", "q"], default="A")

        if choice in ["Q", "q"]:
            console.print("[bold red]Exiting pipeline.[/]")
            break
        elif choice in ["A", "a"]:
            run_end_to_end()
            break
        elif choice == "1":
            run_phase("Phase 1: Bybit Downloader", "download_data.py")
        elif choice == "2":
            run_phase("Phase 2: Multi-TF Resampler", "resample_data.py")
        elif choice == "3":
            run_phase("Phase 6: Backtest Grid", "run_all_backtests.py")
        elif choice == "4":
            run_phase("Phase 7: Optuna Optimizer", "optimizer.py")
        elif choice == "5":
            run_phase("Phase 8: Portfolio Builder", "portfolio_builder.py")
        elif choice == "6":
            run_phase("Phase 9: Walk-Forward Validation", "walkforward.py")
        elif choice == "7":
            run_phase("Phase 9.5: Robustness Suite", "robustness.py")
        elif choice == "8":
            run_phase("Phase 9.7: Capital Sizing", "capital_allocator.py")
        elif choice == "9":
            run_phase("Phase 10: Report Generator", "report_generator.py")

        Prompt.ask("\n[dim]Press Enter to return to menu...[/]")

def main():
    parser = argparse.ArgumentParser(description="Bybit Quantitative Backtesting Pipeline")
    parser.add_argument("--all", action="store_true", help="Run entire pipeline end-to-end (Phases 6 to 10)")
    parser.add_argument("--phase", type=int, choices=[1, 2, 6, 7, 8, 9, 10], help="Run a specific phase number")
    args = parser.parse_args()

    if args.all:
        print_banner()
        run_end_to_end()
    elif args.phase:
        mapping = {
            1: ("Phase 1: Downloader", "download_data.py"),
            2: ("Phase 2: Resampler", "resample_data.py"),
            6: ("Phase 6: Grid Screening", "run_all_backtests.py"),
            7: ("Phase 7: Optimizer", "optimizer.py"),
            8: ("Phase 8: Portfolio Builder", "portfolio_builder.py"),
            9: ("Phase 9: Walk-Forward Validation", "walkforward.py"),
            10: ("Phase 10: Report Generator", "report_generator.py"),
        }
        p_name, s_name = mapping[args.phase]
        run_phase(p_name, s_name)
    else:
        interactive_menu()

if __name__ == "__main__":
    main()
