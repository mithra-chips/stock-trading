import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class TechnicalAnalysis:
    """
    技术分析类，用于计算各种技术指标
    """
    
    def __init__(self, data=None, open_price=None, close_price=None, high_price=None, low_price=None, volume=None,
                 open_col='Open', close_col='Close', high_col='High', 
                 low_col='Low', volume_col='Volume'):
        """
        初始化技术分析类
        
        Args:
            data: DataFrame对象，包含价格和成交量数据
            open_price: 开盘价序列 (如果data为None时使用)
            close_price: 收盘价序列 (如果data为None时使用)
            high_price: 最高价序列 (如果data为None时使用)
            low_price: 最低价序列 (如果data为None时使用)
            volume: 成交量序列 (如果data为None时使用)
            open_col: DataFrame中开盘价列名
            close_col: DataFrame中收盘价列名
            high_col: DataFrame中最高价列名
            low_col: DataFrame中最低价列名
            volume_col: DataFrame中成交量列名
        """
        if data is not None:
            # 从DataFrame中提取数据
            self.open_price = data[open_col]
            self.close_price = data[close_col]
            self.high_price = data[high_col]
            self.low_price = data[low_col]
            self.volume = data[volume_col]
        else:
            # 使用直接传入的数据
            self.open_price = pd.Series(open_price) if not isinstance(open_price, pd.Series) else open_price
            self.close_price = pd.Series(close_price) if not isinstance(close_price, pd.Series) else close_price
            self.high_price = pd.Series(high_price) if not isinstance(high_price, pd.Series) else high_price
            self.low_price = pd.Series(low_price) if not isinstance(low_price, pd.Series) else low_price
            self.volume = pd.Series(volume) if not isinstance(volume, pd.Series) else volume
            data[open_col] = self.open_price
            data[close_col] = self.close_price
            data[high_col] = self.high_price
            data[low_col] = self.low_price
            data[volume_col] = self.volume
        
    def cal_bollinger_bands(self, window=20, num_std=2):
        """
        计算布林带指标
        
        Args:
            window: 移动平均窗口期，默认20
            num_std: 标准差倍数，默认2
            查找正态分布的置信区间、num_std = 2相当于95.5%左右
            
        Returns:
            tuple: (rolling_mean, upper_band, lower_band)
        """
        rolling_mean = self.close_price.rolling(window=window).mean()
        rolling_std = self.close_price.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        return rolling_mean, upper_band, lower_band
    
    def cal_ma(self):
        """
        计算MA50和MA200的移动平均线
        
        Returns:
            tuple: (ma50, ma200)
        """
        ma50 = self.close_price.rolling(window=50, min_periods=1).mean()
        ma200 = self.close_price.rolling(window=200, min_periods=40).mean()
        
        return ma50, ma200
    
    def cal_macd(self, short_window=12, long_window=26, signal_window=9):
        """
        计算短期和长期EMA以及MACD指标
        
        Args:
            short_window: 短期EMA周期，默认12
            long_window: 长期EMA周期，默认26  
            signal_window: 信号线周期，默认9
            
        Returns:
            tuple: (short_ema, long_ema, macd, signal)
        """
        short_ema = self.close_price.ewm(span=short_window, adjust=False).mean()
        long_ema = self.close_price.ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        
        return short_ema, long_ema, macd, signal
    
    def cal_volume(self, window=63):
        """
        计算平均成交量和相对成交量
        约3个月的交易日数据
        
        Args:
            window: 计算平均成交量的窗口期，默认63天
            
        Returns:
            tuple: (average_volume, relative_volume)
        """
        # 计算移动平均成交量
        average_volume = self.volume.rolling(window=window).mean()
        
        # 去除空值后的平均成交量
        average_volume_cleaned = average_volume.dropna()
        
        # 计算相对成交量（当前成交量与平均成交量的比值）
        relative_volume = self.volume[average_volume_cleaned.index] / average_volume_cleaned
        
        return average_volume, relative_volume
    
    def cal_rs(self, windows=14):
        """
        计算相对动力(Relative Strength) = average gain / average loss
        Args:
            window: 计算平均量的窗口期
        Returns:
            float: rs
        """
        diff = self.close_price.diff()
        gain = diff.mask(diff < 0, 0)
        loss = -diff.mask(diff > 0, 0)
        avg_gain = gain.ewm(com=windows - 1, min_periods=windows).mean()
        avg_loss = loss.ewm(com=windows - 1, min_periods=windows).mean()
        rs = avg_gain / avg_loss
        return rs
    
    def cal_rsi(self,windows=14):
        """
        计算相对动力指数(Relative Strength Index) = 100 - 100/(1 + RS)
        Args:
            window: 计算平均量的窗口期
        Returns:

        """
        return 100 - 100 / (1 + self.cal_rs(windows))
    

    def cal_atr(self, windows=14):
        '''
        Volatility of the current market.
        The higher the value, the more volatile it is currently.
        Suitable for risk management.
        '''
        # Calculate True Range (TR)
        tr = self.data.apply(lambda row: max(row['High'] - row['Low'], abs(row['High'] - self.data['Close'].shift(1).loc[row.name]), abs(row['Low'] - self.data['Close'].shift(1).loc[row.name])), axis=1)
        return tr.rolling(window=windows).mean()

    def plot_ma(self):
        """
        绘制移动平均线图
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        MA50, MA200 = self.cal_ma()
        
        MA50.plot(ax=ax, label='MA50')
        MA200.plot(ax=ax, label='MA200')
        self.close_price.plot(ax=ax, label='Close Price')
        
        ax.set_title('MA vs MA200')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (JPY)')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_bollinger_bands(self):
        """
        绘制布林带图
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        self.close_price.plot(ax=ax, label='Close Price')
        rolling_mean, upper_band, lower_band = self.cal_bollinger_bands()
        upper_band.plot(ax=ax, label='Upper', color='gray', linestyle='--')
        lower_band.plot(ax=ax, label='Lower', color='gray', linestyle='--')
        rolling_mean.plot(ax=ax, label='MA20', color='orange', linestyle='-')
        
        ax.set_title('Bollinger Bands')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (JPY)')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 填充布林带区域
        if hasattr(plt, 'fill_between'):
            plt.fill_between(upper_band.index, upper_band, lower_band, 
                           color='gray', alpha=0.2)
        plt.show()
    
    def plot_macd(self):
        """
        绘制MACD图
        """
        _, _, macd, signal = self.cal_macd()
        
        _, ax = plt.subplots(figsize=(12, 6))
        
        macd.plot(ax=ax, label='MACD')
        signal.plot(ax=ax, label='Signal')
        
        ax.set_title('MACD')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (JPY)')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
   
    # Calculate average and relative volume
    def plot_volume(self):
        avg_volume, relative_volume = self.cal_volume()

        # Create subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=('Average Daily Volume', 'Relative Volume'),
                            vertical_spacing=0.3) # Increased vertical spacing

        # Add average volume bar chart
        fig.add_trace(go.Bar(x=avg_volume.index, y=avg_volume.values, name='Average Volume'), row=1, col=1)

        # Add relative volume line chart
        fig.add_trace(go.Scatter(x=relative_volume.index, y=relative_volume.values, mode='lines', name='Relative Volume'), row=2, col=1)

        # Update layout
        fig.update_layout(title_text='Volume Analysis',
                            xaxis=dict(title='Date'),
                            yaxis=dict(title='Volume'),
                            yaxis2=dict(title='Relative Volume'))

        # Show plot
        fig.show()

    def plot_rsi(self):
        _, ax = plt.subplots(figsize=(12, 6))
        rsi = self.cal_rsi()
        
        rsi.plot(ax=ax, label='RSI(Relative Strength Index)')
        
        ax.set_title('RSI(Relative Strength Index)')
        ax.set_xlabel('Date')
        ax.set_ylabel('RSI')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Highlight the area between 30 and 70
        ax.axhspan(30, 70, color='gray', alpha=0.2, label='30-70 Range')
        plt.show()