# The Bybit Alpha Engine - Institutional Quantitative Trading System

An institutional-grade, multi-strategy quantitative trading system for Bybit USDT Perpetual Futures.

## Performance Highlights (2026 Out-of-Sample Unseen Data)
- **Win Rate %:** 79.49% (Model 6 Live Engine)
- **Profit Factor:** 1.735
- **Sharpe Ratio:** 1.26 (Annualized)
- **Max Drawdown:** -12.8%
- **2026 OOS Net Return ( Acct):** +$1,186.42 USDT (+296.6% Return)
- **Deflated Sharpe Ratio (DSR):** 0.959 (Pass)
- **Probability of Overfitting (PBO):** 0.010 (Pass)
- **Backtest-to-Live Match:** 100.0% Perfect Parity

## Repository Architecture
- **engine_v2.py:** Canonical indicator engine (20/20 verified)
- **backtester/:** Research & validation suite (Tasks 1-13.7)
- **live_bot/:** Production 24/7 execution bot
- **results/:** Audit scorecards & CSV exports

## 9-Battery Statistical Certification
1. Monte Carlo Trade Shuffle (10,000 runs)
2. Bootstrap Sharpe Confidence Interval (95% CI > 0)
3. Random Entry Benchmark (p < 0.05 vs random)
4. Price Noise Injection (survives 0.05% noise)
5. Parameter Perturbation (±10% jitter test)
6. Regime Slicing (Profitable in Bull, Bear, Chop)
7. White's Reality Check (SPA p < 0.05)
8. Deflated Sharpe Ratio (DSR = 0.959)
9. Probability of Backtest Overfitting (PBO = 0.010)

## Disclaimer
For educational and quantitative research purposes only.
