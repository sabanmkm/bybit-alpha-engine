import os

class Config:
    BASE_DIR = r'C:\BybitBacktest\data\resampled'
    OUTPUT_DIR = r'C:\BybitBacktest\analysis_results'
    TIMEFRAMES = ['15m', '30m', '1h', '4h']
    
    # Analysis parameters
    MIN_CANDLES = 500
    PATTERN_LOOKBACK = 5
    TREND_WINDOW_SHORT = 20
    TREND_WINDOW_MEDIUM = 50
    TREND_WINDOW_LONG = 200
    CORRELATION_WINDOW = 50
    CYCLE_MAX_PERIOD = 100
    ANOMALY_ZSCORE_THRESHOLD = 3.0
    OUTLIER_IQR_MULTIPLIER = 2.5
    SIMILARITY_TOP_N = 10
    SEQUENCE_LENGTH = 10
    
    # Visualization
    SAVE_PLOTS = True
    PLOT_DPI = 150
    PLOT_FORMAT = 'png'
    
    @classmethod
    def ensure_dirs(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        for sub in ['patterns', 'trends', 'correlations', 'anomalies',
                     'cycles', 'classifications', 'similarities',
                     'sequences', 'comparisons', 'reports', 'plots']:
            os.makedirs(os.path.join(cls.OUTPUT_DIR, sub), exist_ok=True)
