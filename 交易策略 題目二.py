import time
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def get_futures_klines(symbol="BTCUSDT", interval="4h", start_str="2021-01-01"):
    """
    分批獲取更多的歷史數據，不設置總數限制
    """
    url = "https://fapi.binance.com/fapi/v1/klines"
    all_data = []
    start_time = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_time = int(pd.Timestamp.now().timestamp() * 1000)
    
    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "limit": 1500
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if not data:
                break
                
            all_data.extend(data)
            
            # 更新下一批數據的起始時間
            start_time = data[-1][0] + 1
            
            # 添加延遲以避免觸發API限制
            time.sleep(0.1)
            
        except Exception as e:
            print(f"獲取數據時出錯: {e}")
            break
    
    if not all_data:
        raise Exception("未能獲取任何數據")
    
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", 
                                       "close_time", "quote_asset_volume", "trades", "taker_base", "taker_quote", "ignore"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    
    print(f"獲取到的數據範圍: {df.index[0]} 到 {df.index[-1]}")
    print(f"總共獲取 {len(df)} 根K線")
    
    return df[["open", "high", "low", "close", "volume"]]
def calculate_vwap(df, window=20):
    """計算VWAP"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
    return vwap

def calculate_atr(df, period=14):
    """計算ATR"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calculate_rsi(df, period=14):
    """計算RSI指標"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """計算布林通道"""
    ma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper_band = ma + (std * std_dev)
    lower_band = ma - (std * std_dev)
    return upper_band, ma, lower_band

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """計算MACD"""
    fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_market_regime(df, long_window=200):
    """計算市場狀態 (牛市/熊市)"""
    long_ma = df['close'].rolling(window=long_window).mean()
    return df['close'] > long_ma

class EnhancedStrategy:
    def __init__(self, df, initial_capital=10000, commission=0.0005, risk_per_trade=0.02):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.risk_per_trade = risk_per_trade
        self.trades = []
        self.equity_curve = None
        
    def run_strategy(self, vwap_period=20, atr_period=14, rsi_period=14, 
                    bb_period=20, bb_std=2, trend_period=50, atr_multiplier=2.0,
                    rsi_oversold=30, rsi_overbought=70, macd_fast=12, macd_slow=26, macd_signal=9):
        # 計算指標
        self.df['VWAP'] = calculate_vwap(self.df, vwap_period)
        self.df['ATR'] = calculate_atr(self.df, atr_period)
        self.df['RSI'] = calculate_rsi(self.df, rsi_period)
        self.df['BB_Upper'], self.df['BB_Middle'], self.df['BB_Lower'] = calculate_bollinger_bands(self.df, bb_period, bb_std)
        self.df['MACD'], self.df['MACD_Signal'], self.df['MACD_Hist'] = calculate_macd(self.df, macd_fast, macd_slow, macd_signal)
        self.df['Market_Regime'] = calculate_market_regime(self.df)
        self.df['Volume_MA'] = self.df['volume'].rolling(window=vwap_period).mean()
        
        # 計算價格與VWAP的偏離率
        self.df['VWAP_Deviation'] = (self.df['close'] - self.df['VWAP']) / self.df['VWAP']
        
        # 計算短期趨勢和長期趨勢
        self.df['Short_Trend'] = self.df['close'].rolling(window=20).mean() > self.df['close'].rolling(window=50).mean()
        self.df['Long_Trend'] = calculate_trend(self.df, trend_period)
        
        position = 0
        capital = self.initial_capital
        stop_loss = 0
        take_profit = 0
        entry_price = 0
        position_size = 0
        
        # 確保我們有足夠的資料用於指標計算
        start_idx = max(trend_period, bb_period, rsi_period, macd_slow + macd_signal, 200)
        
        for i in range(start_idx, len(self.df)):
            current_price = self.df['close'].iloc[i]
            current_atr = self.df['ATR'].iloc[i]
            
            if position == 0:  # 無倉位
                # 多頭條件
                # 使用更寬鬆的條件以增加交易機會
                long_condition = (
                    self.df['Market_Regime'].iloc[i] and  # 市場處於牛市
                    current_price > self.df['VWAP'].iloc[i] and  # 價格在VWAP之上
                    self.df['RSI'].iloc[i] > 40 and self.df['RSI'].iloc[i] < 70 and  # RSI不超買但有上升動能
                    self.df['MACD_Hist'].iloc[i] > self.df['MACD_Hist'].iloc[i-1] and  # MACD柱狀體上升
                    self.df['volume'].iloc[i] > self.df['Volume_MA'].iloc[i]  # 成交量確認
                )
                
                # 空頭條件
                short_condition = (
                    not self.df['Market_Regime'].iloc[i] and  # 市場處於熊市
                    current_price < self.df['VWAP'].iloc[i] and  # 價格在VWAP之下
                    self.df['RSI'].iloc[i] < 60 and self.df['RSI'].iloc[i] > 30 and  # RSI不超賣但有下降動能
                    self.df['MACD_Hist'].iloc[i] < self.df['MACD_Hist'].iloc[i-1] and  # MACD柱狀體下降
                    self.df['volume'].iloc[i] > self.df['Volume_MA'].iloc[i]  # 成交量確認
                )
                
                if long_condition:
                    position = 1
                    entry_price = current_price
                    # 風險管理: 設置動態止損和止盈
                    stop_loss = entry_price - current_atr * atr_multiplier
                    take_profit = entry_price + current_atr * atr_multiplier * 1.5  # 風險報酬比 1:1.5
                    
                    # 計算倉位大小 (基於風險)
                    risk_amount = capital * self.risk_per_trade
                    position_size = risk_amount / (entry_price - stop_loss) if entry_price > stop_loss else 0
                    if position_size <= 0:
                        position = 0
                        continue
                    
                    entry_time = self.df.index[i]
                    
                elif short_condition:
                    position = -1
                    entry_price = current_price
                    # 風險管理: 設置動態止損和止盈
                    stop_loss = entry_price + current_atr * atr_multiplier
                    take_profit = entry_price - current_atr * atr_multiplier * 1.5  # 風險報酬比 1:1.5
                    
                    # 計算倉位大小 (基於風險)
                    risk_amount = capital * self.risk_per_trade
                    position_size = risk_amount / (stop_loss - entry_price) if stop_loss > entry_price else 0
                    if position_size <= 0:
                        position = 0
                        continue
                    
                    entry_time = self.df.index[i]
            
            elif position == 1:  # 持有多倉
                # 止損、止盈或反轉條件
                if (current_price <= stop_loss or  # 止損
                    current_price >= take_profit or  # 止盈
                    (self.df['MACD'].iloc[i] < self.df['MACD_Signal'].iloc[i] and 
                     self.df['MACD'].iloc[i-1] > self.df['MACD_Signal'].iloc[i-1])):  # MACD死叉
                    
                    exit_price = current_price
                    profit_pct = (exit_price - entry_price) / entry_price
                    profit_amount = position_size * entry_price * profit_pct
                    commission_cost = self.commission * 2 * position_size * entry_price
                    net_profit = profit_amount - commission_cost
                    capital += net_profit
                    
                    self.trades.append({
                        'entry_time': entry_time,
                        'exit_time': self.df.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': 'LONG',
                        'position_size': position_size,
                        'profit_pct': profit_pct * 100,
                        'profit_amount': net_profit,
                        'commission': commission_cost
                    })
                    position = 0
                else:
                    # 更新追蹤止損 (移動止損)
                    if current_price - entry_price > current_atr * atr_multiplier:  # 已有盈利
                        new_stop = current_price - current_atr * (atr_multiplier * 0.75)  # 收緊止損
                        stop_loss = max(stop_loss, new_stop)
            
            elif position == -1:  # 持有空倉
                # 止損、止盈或反轉條件
                if (current_price >= stop_loss or  # 止損
                    current_price <= take_profit or  # 止盈
                    (self.df['MACD'].iloc[i] > self.df['MACD_Signal'].iloc[i] and 
                     self.df['MACD'].iloc[i-1] < self.df['MACD_Signal'].iloc[i-1])):  # MACD金叉
                    
                    exit_price = current_price
                    profit_pct = (entry_price - exit_price) / entry_price
                    profit_amount = position_size * entry_price * profit_pct
                    commission_cost = self.commission * 2 * position_size * entry_price
                    net_profit = profit_amount - commission_cost
                    capital += net_profit
                    
                    self.trades.append({
                        'entry_time': entry_time,
                        'exit_time': self.df.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': 'SHORT',
                        'position_size': position_size,
                        'profit_pct': profit_pct * 100,
                        'profit_amount': net_profit,
                        'commission': commission_cost
                    })
                    position = 0
                else:
                    # 更新追蹤止損 (移動止損)
                    if entry_price - current_price > current_atr * atr_multiplier:  # 已有盈利
                        new_stop = current_price + current_atr * (atr_multiplier * 0.75)  # 收緊止損
                        stop_loss = min(stop_loss, new_stop)
        
        self.trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        self.final_capital = capital
        self.calculate_metrics()
        
        return len(self.trades_df) > 0  # 返回是否有交易發生
    
    def calculate_metrics(self):
        self.total_return = (self.final_capital - self.initial_capital) / self.initial_capital
        trading_days = len(self.df) / 6  # 假設每天6個4小時K線
        self.annual_return = (1 + self.total_return) ** (365 / trading_days) - 1 if trading_days > 0 else 0
        
        if len(self.trades_df) > 0:
            # 計算累積資金曲線
            self.equity_curve = self.calculate_equity_curve()
            
            trade_returns = self.trades_df['profit_amount'].values / self.initial_capital
            
            self.sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(len(trade_returns)) if np.std(trade_returns) != 0 else 0
            
            # 計算最大回撤
            rolling_max = np.maximum.accumulate(self.equity_curve)
            drawdowns = (self.equity_curve - rolling_max) / rolling_max
            self.max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
            
            # 計算勝率與盈虧比
            wins = self.trades_df[self.trades_df['profit_amount'] > 0]
            losses = self.trades_df[self.trades_df['profit_amount'] <= 0]
            
            self.win_rate = len(wins) / len(self.trades_df) if len(self.trades_df) > 0 else 0
            self.profit_factor = abs(wins['profit_amount'].sum() / losses['profit_amount'].sum()) if losses['profit_amount'].sum() != 0 and len(losses) > 0 else float('inf')
            
            # 計算平均每筆交易利潤
            self.avg_profit_per_trade = self.trades_df['profit_amount'].mean()
            
            # 計算最大連續虧損次數
            self.trades_df['win'] = self.trades_df['profit_amount'] > 0
            self.max_consecutive_losses = self.calculate_max_consecutive(~self.trades_df['win'])
            
            # 計算卡爾馬比率 (CalmarRatio)
            self.calmar_ratio = self.annual_return / self.max_drawdown if self.max_drawdown != 0 else float('inf')
        else:
            self.sharpe_ratio = 0
            self.max_drawdown = 0
            self.win_rate = 0
            self.profit_factor = 0
            self.avg_profit_per_trade = 0
            self.max_consecutive_losses = 0
            self.calmar_ratio = 0
            self.equity_curve = pd.Series(index=self.df.index, data=1)
    
    def calculate_max_consecutive(self, series):
        """計算最大連續True的次數"""
        max_streak = 0
        current_streak = 0
        for val in series:
            if val:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak
    
    def calculate_monthly_returns(self):
        """計算每月報酬率"""
        if len(self.trades_df) == 0:
            return pd.Series()
        
        # 將每筆交易的退出時間和盈利分組
        monthly_returns = self.trades_df.set_index('exit_time')['profit_amount'].resample('M').sum() / self.initial_capital
        return monthly_returns
    
    def calculate_equity_curve(self):
        """計算完整的權益曲線"""
        if len(self.trades_df) == 0:
            return pd.Series(index=self.df.index, data=1)
        
        # 初始化權益曲線
        equity = pd.Series(index=self.df.index, data=self.initial_capital)
        
        # 根據交易記錄填充權益曲線
        for _, trade in self.trades_df.iterrows():
            try:
                exit_idx = self.df.index.get_indexer([trade['exit_time']])[0]
                equity.iloc[exit_idx:] += trade['profit_amount']
            except:
                continue  # 如果索引有问题，跳过此交易
        
        # 轉換為百分比變化
        equity = equity / self.initial_capital
        
        return equity

def calculate_trend(df, period=50):
    """計算趨勢"""
    ma = df['close'].rolling(window=period).mean()
    return df['close'] > ma

def optimize_strategy(df, param_grid, min_trades=5):
    """使用網格搜索優化策略參數"""
    best_sharpe = -np.inf
    best_params = None
    results = []
    
    total_combs = len(list(ParameterGrid(param_grid)))
    print(f"開始優化 {total_combs} 種參數組合...")
    
    for i, params in enumerate(ParameterGrid(param_grid)):
        if i % 10 == 0:
            print(f"進度: {i}/{total_combs}")
            
        strategy = EnhancedStrategy(df)
        has_trades = strategy.run_strategy(**params)
        
        result = {
            'params': params,
            'sharpe_ratio': strategy.sharpe_ratio,
            'annual_return': strategy.annual_return,
            'max_drawdown': strategy.max_drawdown,
            'win_rate': strategy.win_rate,
            'trade_count': len(strategy.trades_df)
        }
        results.append(result)
        
        if has_trades and strategy.sharpe_ratio > best_sharpe and len(strategy.trades_df) >= min_trades:
            best_sharpe = strategy.sharpe_ratio
            best_params = params
    
    results_df = pd.DataFrame(results)
    
    # 如果沒有找到好的參數，則選擇一個默認參數
    if best_params is None:
        print("沒有找到滿足條件的參數組合，使用預設參數...")
        best_params = {
            'vwap_period': 20,
            'atr_period': 14,
            'rsi_period': 14,
            'bb_period': 20,
            'bb_std': 2.0,
            'trend_period': 50,
            'atr_multiplier': 2.0,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }
    
    return best_params, results_df

def plot_results(strategy, df):
    """繪製策略結果圖表"""
    plt.figure(figsize=(15, 15))
    
    # 繪製價格與VWAP
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df.index, df['close'], label='價格')
    ax1.plot(df.index, strategy.df['VWAP'], label='VWAP', alpha=0.7)
    
    # 標記交易點
    if len(strategy.trades_df) > 0:
        for _, trade in strategy.trades_df.iterrows():
            try:
                if trade['position'] == 'LONG':
                    ax1.scatter(trade['entry_time'], trade['entry_price'], color='green', marker='^', s=100)
                    ax1.scatter(trade['exit_time'], trade['exit_price'], color='red', marker='v', s=100)
                else:
                    ax1.scatter(trade['entry_time'], trade['entry_price'], color='red', marker='v', s=100)
                    ax1.scatter(trade['exit_time'], trade['exit_price'], color='green', marker='^', s=100)
            except:
                continue  # 如果畫點出錯則跳過
    
    ax1.set_title('價格與交易點')
    ax1.grid(True)
    ax1.legend()
    
    # 繪製權益曲線
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(strategy.equity_curve)
    ax2.set_title('權益曲線 (倍數)')
    ax2.grid(True)
    
    # 繪製回撤曲線
    ax3 = plt.subplot(3, 1, 3)
    if len(strategy.equity_curve) > 0:
        rolling_max = np.maximum.accumulate(strategy.equity_curve)
        drawdowns = (strategy.equity_curve - rolling_max) / rolling_max
        ax3.fill_between(strategy.equity_curve.index, drawdowns, 0, color='red', alpha=0.3)
        ax3.set_title('回撤曲線')
        ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('strategy_performance.png')
    plt.show()
    
    # 繪製月度收益分佈
    plt.figure(figsize=(15, 6))
    monthly_returns = strategy.calculate_monthly_returns()
    if not monthly_returns.empty:
        monthly_returns.plot(kind='bar')
        plt.title('月度收益分佈')
        plt.grid(True)
        plt.savefig('monthly_returns.png')
        plt.show()

# 主程式
if __name__ == "__main__":
    # 獲取資料
    print("獲取交易資料中...")
    df = get_futures_klines(symbol="BTCUSDT", interval="4h", start_str="2021-01-01")
    print(f"獲取到 {len(df)} 筆資料，從 {df.index[0]} 到 {df.index[-1]}")
    
    # 參數優化（更寬鬆的條件，增加交易機會）
    param_grid = {
        'vwap_period': [15, 20, 25],
        'atr_period': [10, 14, 18],
        'rsi_period': [14],
        'bb_period': [20],
        'bb_std': [2.0],
        'trend_period': [50, 100],
        'atr_multiplier': [1.5, 2.0, 2.5],
        'rsi_oversold': [30],
        'rsi_overbought': [70],
        'macd_fast': [12],
        'macd_slow': [26],
        'macd_signal': [9]
    }
    
    print("進行參數優化中...")
    best_params, results_df = optimize_strategy(df, param_grid, min_trades=5)
    print(f"最佳參數: {best_params}")
    
    # 使用最佳參數運行策略
    print("使用最佳參數運行策略...")
    strategy = EnhancedStrategy(df)
    strategy.run_strategy(**best_params)
    
    # 輸出結果
    print(f"總報酬率: {strategy.total_return:.2%}")
    print(f"年化報酬率: {strategy.annual_return:.2%}")
    print(f"夏普比率: {strategy.sharpe_ratio:.2f}")
    print(f"最大回撤: {strategy.max_drawdown:.2%}")
    print(f"勝率: {strategy.win_rate:.2%}")
    
    if hasattr(strategy, 'profit_factor') and strategy.profit_factor != float('inf'):
        print(f"盈虧比: {strategy.profit_factor:.2f}")
    else:
        print(f"盈虧比: N/A")
    
    if hasattr(strategy, 'calmar_ratio') and strategy.calmar_ratio != float('inf'):
        print(f"卡爾馬比率: {strategy.calmar_ratio:.2f}")
    else:
        print(f"卡爾馬比率: N/A")
        
    print(f"平均每筆交易利潤: ${strategy.avg_profit_per_trade:.2f}")
    print(f"最大連續虧損次數: {strategy.max_consecutive_losses}")
    print(f"總交易次數: {len(strategy.trades_df)}")
    
    # 保存交易記錄
    if len(strategy.trades_df) > 0:
        strategy.trades_df.to_csv('enhanced_trading_details.csv', index=False)
        print("交易記錄已保存至 enhanced_trading_details.csv")
    
    # 繪製圖表
    print("生成績效分析圖表...")
    plot_results(strategy, df)
    
    import os
    
    if len(strategy.trades_df) > 0:
        csv_path = os.path.abspath("enhanced_trading_details.csv")  # 取得 CSV 絕對路徑
        print(f"CSV 檔案路徑: {csv_path}")
        try:
            os.startfile(csv_path)  # 讓 Windows 自動打開 CSV
            print("已開啟交易記錄檔案")
        except:
            print("無法自動開啟檔案，請手動開啟")