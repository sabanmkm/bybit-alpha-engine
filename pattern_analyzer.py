import numpy as np
import pandas as pd
from config import Config

class PatternAnalyzer:
    
    def __init__(self):
        self.pattern_results = {}
    
    def analyze_all_patterns(self, df, symbol='', timeframe=''):
        results = {}
        
        # Single candle patterns
        results['doji'] = self._detect_doji(df)
        results['hammer'] = self._detect_hammer(df)
        results['inverted_hammer'] = self._detect_inverted_hammer(df)
        results['shooting_star'] = self._detect_shooting_star(df)
        results['marubozu_bull'] = self._detect_marubozu(df, bullish=True)
        results['marubozu_bear'] = self._detect_marubozu(df, bullish=False)
        results['spinning_top'] = self._detect_spinning_top(df)
        results['dragonfly_doji'] = self._detect_dragonfly_doji(df)
        results['gravestone_doji'] = self._detect_gravestone_doji(df)
        results['long_legged_doji'] = self._detect_long_legged_doji(df)
        
        # Two candle patterns
        results['bullish_engulfing'] = self._detect_engulfing(df, bullish=True)
        results['bearish_engulfing'] = self._detect_engulfing(df, bullish=False)
        results['piercing_line'] = self._detect_piercing_line(df)
        results['dark_cloud_cover'] = self._detect_dark_cloud(df)
        results['tweezer_top'] = self._detect_tweezer(df, top=True)
        results['tweezer_bottom'] = self._detect_tweezer(df, top=False)
        results['harami_bull'] = self._detect_harami(df, bullish=True)
        results['harami_bear'] = self._detect_harami(df, bullish=False)
        
        # Three candle patterns
        results['morning_star'] = self._detect_morning_star(df)
        results['evening_star'] = self._detect_evening_star(df)
        results['three_white_soldiers'] = self._detect_three_soldiers(df, bullish=True)
        results['three_black_crows'] = self._detect_three_soldiers(df, bullish=False)
        results['three_inside_up'] = self._detect_three_inside(df, bullish=True)
        results['three_inside_down'] = self._detect_three_inside(df, bullish=False)
        
        # Advanced patterns
        results['gap_up'] = self._detect_gap(df, up=True)
        results['gap_down'] = self._detect_gap(df, up=False)
        results['inside_bar'] = self._detect_inside_bar(df)
        results['outside_bar'] = self._detect_outside_bar(df)
        results['pin_bar_bull'] = self._detect_pin_bar(df, bullish=True)
        results['pin_bar_bear'] = self._detect_pin_bar(df, bullish=False)
        results['volume_spike'] = self._detect_volume_spike(df)
        results['high_wave'] = self._detect_high_wave(df)
        
        # Calculate forward returns for each pattern
        pattern_stats = self._calculate_pattern_statistics(df, results)
        
        key = f'{timeframe}_{symbol}'
        self.pattern_results[key] = {
            'signals': results,
            'statistics': pattern_stats
        }
        
        return pattern_stats
    
    # ==================== SINGLE CANDLE PATTERNS ====================
    
    def _detect_doji(self, df, threshold=0.1):
        return df['body_pct'].fillna(1) < threshold
    
    def _detect_hammer(self, df):
        body_small = df['body_pct'].fillna(1) < 0.35
        lower_long = df['lower_shadow_pct'].fillna(0) > 0.6
        upper_small = df['upper_shadow_pct'].fillna(1) < 0.1
        return body_small & lower_long & upper_small
    
    def _detect_inverted_hammer(self, df):
        body_small = df['body_pct'].fillna(1) < 0.35
        upper_long = df['upper_shadow_pct'].fillna(0) > 0.6
        lower_small = df['lower_shadow_pct'].fillna(1) < 0.1
        return body_small & upper_long & lower_small
    
    def _detect_shooting_star(self, df):
        body_small = df['body_pct'].fillna(1) < 0.35
        upper_long = df['upper_shadow_pct'].fillna(0) > 0.6
        lower_small = df['lower_shadow_pct'].fillna(1) < 0.1
        bearish_body = df['body'] < 0
        return body_small & upper_long & lower_small & bearish_body
    
    def _detect_marubozu(self, df, bullish=True, shadow_max=0.05):
        no_shadows = (df['upper_shadow_pct'].fillna(1) < shadow_max) & \
                     (df['lower_shadow_pct'].fillna(1) < shadow_max)
        large_body = df['body_pct'].fillna(0) > 0.9
        if bullish:
            return no_shadows & large_body & (df['body'] > 0)
        else:
            return no_shadows & large_body & (df['body'] < 0)
    
    def _detect_spinning_top(self, df):
        small_body = df['body_pct'].fillna(1).between(0.1, 0.35)
        has_shadows = (df['upper_shadow_pct'].fillna(0) > 0.25) & \
                      (df['lower_shadow_pct'].fillna(0) > 0.25)
        return small_body & has_shadows
    
    def _detect_dragonfly_doji(self, df):
        doji = df['body_pct'].fillna(1) < 0.1
        long_lower = df['lower_shadow_pct'].fillna(0) > 0.7
        no_upper = df['upper_shadow_pct'].fillna(1) < 0.05
        return doji & long_lower & no_upper
    
    def _detect_gravestone_doji(self, df):
        doji = df['body_pct'].fillna(1) < 0.1
        long_upper = df['upper_shadow_pct'].fillna(0) > 0.7
        no_lower = df['lower_shadow_pct'].fillna(1) < 0.05
        return doji & long_upper & no_lower
    
    def _detect_long_legged_doji(self, df):
        doji = df['body_pct'].fillna(1) < 0.1
        long_shadows = (df['upper_shadow_pct'].fillna(0) > 0.35) & \
                       (df['lower_shadow_pct'].fillna(0) > 0.35)
        large_range = df['range'] > df['atr'] * 1.5
        return doji & long_shadows & large_range
    
    # ==================== TWO CANDLE PATTERNS ====================
    
    def _detect_engulfing(self, df, bullish=True):
        signals = pd.Series(False, index=df.index)
        if bullish:
            signals.iloc[1:] = (
                (df['body'].iloc[:-1].values < 0) &
                (df['body'].iloc[1:].values > 0) &
                (df['open'].iloc[1:].values <= df['close'].iloc[:-1].values) &
                (df['close'].iloc[1:].values >= df['open'].iloc[:-1].values) &
                (df['body_abs'].iloc[1:].values > df['body_abs'].iloc[:-1].values)
            )
        else:
            signals.iloc[1:] = (
                (df['body'].iloc[:-1].values > 0) &
                (df['body'].iloc[1:].values < 0) &
                (df['open'].iloc[1:].values >= df['close'].iloc[:-1].values) &
                (df['close'].iloc[1:].values <= df['open'].iloc[:-1].values) &
                (df['body_abs'].iloc[1:].values > df['body_abs'].iloc[:-1].values)
            )
        return signals
    
    def _detect_piercing_line(self, df):
        signals = pd.Series(False, index=df.index)
        prev_bear = df['body'].shift(1) < 0
        curr_bull = df['body'] > 0
        gap_down = df['open'] < df['close'].shift(1)
        close_above_mid = df['close'] > (df['open'].shift(1) + df['close'].shift(1)) / 2
        close_below_open = df['close'] < df['open'].shift(1)
        signals = prev_bear & curr_bull & gap_down & close_above_mid & close_below_open
        return signals
    
    def _detect_dark_cloud(self, df):
        prev_bull = df['body'].shift(1) > 0
        curr_bear = df['body'] < 0
        gap_up = df['open'] > df['close'].shift(1)
        close_below_mid = df['close'] < (df['open'].shift(1) + df['close'].shift(1)) / 2
        close_above_open = df['close'] > df['open'].shift(1)
        return prev_bull & curr_bear & gap_up & close_below_mid & close_above_open
    
    def _detect_tweezer(self, df, top=True):
        if top:
            same_high = (df['high'] - df['high'].shift(1)).abs() / df['atr'] < 0.05
            first_bull = df['body'].shift(1) > 0
            second_bear = df['body'] < 0
            return same_high & first_bull & second_bear
        else:
            same_low = (df['low'] - df['low'].shift(1)).abs() / df['atr'] < 0.05
            first_bear = df['body'].shift(1) < 0
            second_bull = df['body'] > 0
            return same_low & first_bear & second_bull
    
    def _detect_harami(self, df, bullish=True):
        if bullish:
            prev_bear = df['body'].shift(1) < 0
            curr_bull = df['body'] > 0
            inside = (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1))
            smaller = df['body_abs'] < df['body_abs'].shift(1)
            return prev_bear & curr_bull & inside & smaller
        else:
            prev_bull = df['body'].shift(1) > 0
            curr_bear = df['body'] < 0
            inside = (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1))
            smaller = df['body_abs'] < df['body_abs'].shift(1)
            return prev_bull & curr_bear & inside & smaller
    
    # ==================== THREE CANDLE PATTERNS ====================
    
    def _detect_morning_star(self, df):
        first_bear = df['body'].shift(2) < 0
        first_large = df['body_abs'].shift(2) > df['atr'].shift(2) * 0.5
        second_small = df['body_abs'].shift(1) < df['atr'].shift(1) * 0.3
        third_bull = df['body'] > 0
        third_large = df['body_abs'] > df['atr'] * 0.5
        third_closes_above = df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2
        return first_bear & first_large & second_small & third_bull & third_large & third_closes_above
    
    def _detect_evening_star(self, df):
        first_bull = df['body'].shift(2) > 0
        first_large = df['body_abs'].shift(2) > df['atr'].shift(2) * 0.5
        second_small = df['body_abs'].shift(1) < df['atr'].shift(1) * 0.3
        third_bear = df['body'] < 0
        third_large = df['body_abs'] > df['atr'] * 0.5
        third_closes_below = df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2
        return first_bull & first_large & second_small & third_bear & third_large & third_closes_below
    
    def _detect_three_soldiers(self, df, bullish=True):
        if bullish:
            c1 = df['body'].shift(2) > 0
            c2 = df['body'].shift(1) > 0
            c3 = df['body'] > 0
            rising = (df['close'].shift(1) > df['close'].shift(2)) & (df['close'] > df['close'].shift(1))
            large = (df['body_pct'].shift(2).fillna(0) > 0.6) & \
                    (df['body_pct'].shift(1).fillna(0) > 0.6) & \
                    (df['body_pct'].fillna(0) > 0.6)
            return c1 & c2 & c3 & rising & large
        else:
            c1 = df['body'].shift(2) < 0
            c2 = df['body'].shift(1) < 0
            c3 = df['body'] < 0
            falling = (df['close'].shift(1) < df['close'].shift(2)) & (df['close'] < df['close'].shift(1))
            large = (df['body_pct'].shift(2).fillna(0) > 0.6) & \
                    (df['body_pct'].shift(1).fillna(0) > 0.6) & \
                    (df['body_pct'].fillna(0) > 0.6)
            return c1 & c2 & c3 & falling & large
    
    def _detect_three_inside(self, df, bullish=True):
        if bullish:
            harami = self._detect_harami(df.shift(1), bullish=True)
            confirm = df['close'] > df['close'].shift(1)
            return harami & confirm
        else:
            harami = self._detect_harami(df.shift(1), bullish=False)
            confirm = df['close'] < df['close'].shift(1)
            return harami & confirm
    
    # ==================== ADVANCED PATTERNS ====================
    
    def _detect_gap(self, df, up=True):
        if up:
            return df['low'] > df['high'].shift(1)
        else:
            return df['high'] < df['low'].shift(1)
    
    def _detect_inside_bar(self, df):
        return (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
    
    def _detect_outside_bar(self, df):
        return (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
    
    def _detect_pin_bar(self, df, bullish=True):
        if bullish:
            long_lower = df['lower_shadow_pct'].fillna(0) > 0.65
            small_body = df['body_pct'].fillna(1) < 0.25
            small_upper = df['upper_shadow_pct'].fillna(1) < 0.15
            return long_lower & small_body & small_upper
        else:
            long_upper = df['upper_shadow_pct'].fillna(0) > 0.65
            small_body = df['body_pct'].fillna(1) < 0.25
            small_lower = df['lower_shadow_pct'].fillna(1) < 0.15
            return long_upper & small_body & small_lower
    
    def _detect_volume_spike(self, df, multiplier=3.0):
        return df['volume_ratio'].fillna(0) > multiplier
    
    def _detect_high_wave(self, df):
        small_body = df['body_pct'].fillna(1) < 0.2
        big_shadows = (df['upper_shadow_pct'].fillna(0) > 0.3) & \
                      (df['lower_shadow_pct'].fillna(0) > 0.3)
        large_range = df['range'] > df['atr'] * 1.5
        return small_body & big_shadows & large_range
    
    # ==================== STATISTICS ====================
    
    def _calculate_pattern_statistics(self, df, pattern_signals, forward_periods=[1, 3, 5, 10, 20]):
        stats = []
        for name, signals in pattern_signals.items():
            count = signals.sum()
            if count < 3:
                continue
            
            row = {
                'pattern': name,
                'count': int(count),
                'frequency_pct': round(count / len(df) * 100, 3),
            }
            
            for fp in forward_periods:
                fwd_ret = df['returns'].shift(-fp).rolling(fp).sum()
                pattern_returns = fwd_ret[signals].dropna()
                if len(pattern_returns) > 0:
                    row[f'avg_return_{fp}bar'] = round(pattern_returns.mean() * 100, 4)
                    row[f'win_rate_{fp}bar'] = round((pattern_returns > 0).mean() * 100, 2)
                    row[f'avg_win_{fp}bar'] = round(pattern_returns[pattern_returns > 0].mean() * 100, 4) if (pattern_returns > 0).any() else 0
                    row[f'avg_loss_{fp}bar'] = round(pattern_returns[pattern_returns < 0].mean() * 100, 4) if (pattern_returns < 0).any() else 0
                    row[f'expectancy_{fp}bar'] = round(
                        (pattern_returns > 0).mean() * pattern_returns[pattern_returns > 0].mean() +
                        (pattern_returns <= 0).mean() * pattern_returns[pattern_returns <= 0].mean()
                        if (pattern_returns > 0).any() and (pattern_returns <= 0).any() else 0, 6
                    ) * 100
                else:
                    row[f'avg_return_{fp}bar'] = None
                    row[f'win_rate_{fp}bar'] = None
            
            stats.append(row)
        
        return pd.DataFrame(stats)
