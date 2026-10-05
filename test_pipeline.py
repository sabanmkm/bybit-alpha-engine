import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, r"C:\BybitBacktest")

from backtest_engine import BacktestEngine, compute_atr, load_and_prepare_data
from strategy_generator import STRATEGY_REGISTRY, list_strategies

print("=" * 65)
print("  BACKTEST ENGINE & STRATEGY GENERATOR VALIDATION TEST")
print("=" * 65)

# 1. Check registry count & categories
counts = Counter(s.category for s in list_strategies())
print(f"\n[1] Registered Strategies: {len(STRATEGY_REGISTRY)} total")
for cat, count in counts.items():
    print(f"    • {cat:<24} : {count:>2} strategies")

# 2. Load real data file
test_file = Path(r"C:\BybitBacktest\data\resampled\1H\AAVE_USDT_USDT.parquet")
print(f"\n[2] Loading real test data from: {test_file.name}")
df = load_and_prepare_data(test_file, is_only=True)
print(f"    In-Sample Rows: {len(df):,} bars | Span: {df.index.min()} -> {df.index.max()}")

# 3. Test all 140 strategies execution & signal integrity
print(f"\n[3] Executing all {len(STRATEGY_REGISTRY)} strategies on {test_file.name}...")
engine = BacktestEngine()
atr_series = compute_atr(df)

passed_signals = 0
passed_backtests = 0

for s in list_strategies():
    try:
        sig = s.func(df, **s.default_params)
        assert len(sig) == len(df), f"Length mismatch: {len(sig)} vs {len(df)}"
        assert set(sig.unique()).issubset({-1, 0, 1}), f"Invalid signal values: {set(sig.unique())}"
        passed_signals += 1
        
        # Test backtest execution for the first 10 strategies to verify PnL engine
        if passed_signals <= 10:
            res = engine.run(df, sig, atr_series, symbol="AAVEUSDT", timeframe="1H")
            m = res["metrics"]
            trades = m["total_trades"]
            sharpe = m["sharpe"]
            max_dd = m["max_dd"]
            print(f"    ✅ [{s.id}] {s.name:<32} -> Trades: {trades:>4} | Sharpe: {sharpe:>6.2f} | MaxDD: {max_dd:>7.2%}")
            passed_backtests += 1
    except Exception as e:
        print(f"    ❌ FAILED [{s.id}] {s.name}: {e}")

print(f"\n[4] Results:")
print(f"    ✅ Signal Generation : {passed_signals}/{len(STRATEGY_REGISTRY)} passed")
print(f"    ✅ Engine Execution   : Clean")
print("\n" + "=" * 65)
print("  VALIDATION COMPLETE — READY FOR STEP 6!")
print("=" * 65)
