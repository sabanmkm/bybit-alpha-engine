import os
import pandas as pd
import numpy as np
from config import Config
from rich.console import Console
from rich.progress import track

console = Console()

class DataLoader:
    def __init__(self):
        self.data = {}   # {timeframe: {symbol: DataFrame}}
        
    def load_all(self):
        console.print('[bold cyan]Loading all data...[/bold cyan]')
        for tf in Config.TIMEFRAMES:
            tf_path = os.path.join(Config.BASE_DIR, tf)
            if not os.path.exists(tf_path):
                console.print(f'[yellow]Skipping {tf} - folder not found[/yellow]')
                continue
            self.data[tf] = {}
            files = [f for f in os.listdir(tf_path) if f.endswith('.parquet')]
            for f in track(files, description=f'Loading {tf}'):
                symbol = f.replace('.parquet', '')
                try:
                    df = pd.read_parquet(os.path.join(tf_path, f))
                    df = self._standardize(df)
                    if len(df) >= Config.MIN_CANDLES:
                        self.data[tf][symbol] = df
                except Exception as e:
                    console.print(f'[red]Error loading {f}: {e}[/red]')
            console.print(f'[green]{tf}: {len(self.data[tf])} symbols loaded[/green]')
        return self.data
    
    def _standardize(self, df):
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ['open', 'o']:
                col_map[c] = 'open'
            elif cl in ['high', 'h']:
                col_map[c] = 'high'
            elif cl in ['low', 'l']:
                col_map[c] = 'low'
            elif cl in ['close', 'c']:
                col_map[c] = 'close'
            elif cl in ['volume', 'v', 'vol']:
                col_map[c] = 'volume'
            elif cl in ['timestamp', 'time', 'date', 'datetime']:
                col_map[c] = 'timestamp'
        df = df.rename(columns=col_map)
        
        required = ['open', 'high', 'low', 'close']
        for r in required:
            if r not in df.columns:
                raise ValueError(f'Missing column: {r}')
        
        if 'volume' not in df.columns:
            df['volume'] = 0
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp') if df.index.name != 'timestamp' else df
        elif not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                pass
        
        df = df.sort_index()
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Add derived columns
        df['body'] = df['close'] - df['open']
        df['body_abs'] = df['body'].abs()
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['range'] = df['high'] - df['low']
        df['body_pct'] = df['body_abs'] / df['range'].replace(0, np.nan)
        df['upper_shadow_pct'] = df['upper_shadow'] / df['range'].replace(0, np.nan)
        df['lower_shadow_pct'] = df['lower_shadow'] / df['range'].replace(0, np.nan)
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, np.nan)
        df['atr'] = df['range'].rolling(14).mean()
        
        return df

    def get_symbols(self, timeframe):
        return list(self.data.get(timeframe, {}).keys())
    
    def get_df(self, timeframe, symbol):
        return self.data[timeframe][symbol]
    
    def summary(self):
        rows = []
        for tf in self.data:
            for sym in self.data[tf]:
                df = self.data[tf][sym]
                rows.append({
                    'timeframe': tf,
                    'symbol': sym,
                    'candles': len(df),
                    'start': str(df.index[0])[:19],
                    'end': str(df.index[-1])[:19],
                    'avg_volume': df['volume'].mean(),
                    'avg_range_pct': (df['range'] / df['close']).mean() * 100
                })
        return pd.DataFrame(rows)
