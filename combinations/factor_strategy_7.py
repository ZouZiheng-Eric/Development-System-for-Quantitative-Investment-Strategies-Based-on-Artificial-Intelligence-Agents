"""
成交量异常因子策略 - Factor Strategy 7
基于成交量的异常波动检测，捕捉放量突破和缩量回调信号
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

class DataHandler:
    """数据处理器类"""
    def __init__(self, data_path):
        self.data = pd.read_pickle(data_path)
        self.data = self.data.sort_index()
        self.data['returns'] = self.data['Close'].pct_change()
        
    def calculate_volume_anomaly_factor(self, short_period=5, long_period=20):
        """计算成交量异常因子"""
        # 成交量移动平均
        self.data['Volume_SMA_Short'] = self.data['Volume'].rolling(window=short_period).mean()
        self.data['Volume_SMA_Long'] = self.data['Volume'].rolling(window=long_period).mean()
        
        # 成交量比率
        self.data['Volume_Ratio_Short'] = self.data['Volume'] / self.data['Volume_SMA_Short']
        self.data['Volume_Ratio_Long'] = self.data['Volume'] / self.data['Volume_SMA_Long']
        
        # 成交量标准化得分
        volume_mean = self.data['Volume'].rolling(window=60).mean()
        volume_std = self.data['Volume'].rolling(window=60).std()
        self.data['Volume_Z_Score'] = (self.data['Volume'] - volume_mean) / (volume_std + 1e-8)
        
        # 成交量异常检测
        self._calculate_volume_anomaly_signals()
        
    def _calculate_volume_anomaly_signals(self):
        """计算成交量异常信号"""
        # 成交量突破阈值检测
        volume_percentile_80 = self.data['Volume'].rolling(window=60).quantile(0.8)
        volume_percentile_20 = self.data['Volume'].rolling(window=60).quantile(0.2)
        
        # 异常放量
        self.data['Volume_Spike'] = (self.data['Volume'] > volume_percentile_80).astype(int)
        
        # 异常缩量
        self.data['Volume_Drop'] = (self.data['Volume'] < volume_percentile_20).astype(int)
        
        # 价格动量
        self.data['Price_Momentum_1d'] = self.data['Close'].pct_change()
        self.data['Price_Momentum_3d'] = self.data['Close'].pct_change(3)
        
        # 成交量与价格动量的关系
        self.data['Volume_Price_Momentum'] = self.data['Volume_Ratio_Long'] * np.sign(self.data['Price_Momentum_1d'])
        
        # 成交量连续性检测
        self.data['Volume_Trend'] = np.where(
            self.data['Volume_Ratio_Short'] > 1.2, 1,  # 连续放量
            np.where(self.data['Volume_Ratio_Short'] < 0.8, -1, 0)  # 连续缩量
        )
        
        # 成交量异常因子
        # 放量上涨看多，放量下跌看空，缩量时保持中性或反向
        self.data['Volume_Anomaly_Factor'] = (
            0.4 * self.data['Volume_Price_Momentum'] +  # 量价配合
            0.3 * np.sign(self.data['Volume_Z_Score']) * np.sign(self.data['Price_Momentum_1d']) +  # Z-score量价信号
            0.2 * self.data['Volume_Trend'] * np.sign(self.data['Price_Momentum_3d']) +  # 趋势量价配合
            -0.1 * self.data['Volume_Drop'] * np.sign(self.data['Price_Momentum_1d'])  # 缩量反向信号
        )
        
        # 平滑处理
        self.data['Volume_Anomaly_Factor'] = self.data['Volume_Anomaly_Factor'].rolling(window=3).mean()

class VolumeAnomalyStrategy:
    """成交量异常策略"""
    def __init__(self, data, threshold=0.6):
        self.data = data
        self.threshold = threshold
        self.positions = pd.Series(index=data.index, dtype=float).fillna(0)
        
    def generate_signals(self):
        """生成交易信号"""
        # 基于成交量异常因子生成信号
        conditions = [
            (self.data['Volume_Anomaly_Factor'] > self.threshold) & 
            (self.data['Volume_Ratio_Long'] > 1.5) & 
            (self.data['Price_Momentum_1d'] > 0),  # 强放量上涨
            
            (self.data['Volume_Anomaly_Factor'] > 0.3) & 
            (self.data['Volume_Ratio_Long'] > 1.2),  # 一般放量
            
            (self.data['Volume_Anomaly_Factor'] < -self.threshold) & 
            (self.data['Volume_Ratio_Long'] > 1.5) & 
            (self.data['Price_Momentum_1d'] < 0),  # 强放量下跌
            
            (self.data['Volume_Anomaly_Factor'] < -0.3) & 
            (self.data['Volume_Ratio_Long'] > 1.2) & 
            (self.data['Price_Momentum_1d'] < 0),  # 一般放量下跌
        ]
        choices = [1, 0.5, -1, -0.5]  # 满仓做多、半仓做多、满仓做空、半仓做空
        
        self.positions = np.select(conditions, choices, default=0)
        self.positions = pd.Series(self.positions, index=self.data.index).shift(1).fillna(0)

class Performance:
    """业绩分析类"""
    def __init__(self, data, positions):
        self.data = data
        self.positions = positions
        self._calculate_performance()
        
    def _calculate_performance(self):
        """计算策略业绩"""
        strategy_returns = self.data['returns'] * self.positions
        self.data['strategy_returns'] = strategy_returns
        self.data['cumulative_returns'] = (1 + strategy_returns).cumprod()
        self.data['benchmark_returns'] = (1 + self.data['returns']).cumprod()
        
    def calculate_metrics(self):
        """计算业绩指标"""
        strategy_returns = self.data['strategy_returns'].dropna()
        
        annual_return = strategy_returns.mean() * 252
        annual_volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        cumulative = self.data['cumulative_returns']
        max_dd = (cumulative / cumulative.expanding().max() - 1).min()
        
        excess_returns = strategy_returns - self.data['returns']
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0
        
        win_rate = (strategy_returns > 0).sum() / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'information_ratio': information_ratio,
            'win_rate': win_rate
        }
        
    def plot_results(self):
        """绘制回测结果"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 累计收益率对比
        ax1.plot(self.data.index, self.data['cumulative_returns'], label='成交量异常策略', linewidth=2, color='blue')
        ax1.plot(self.data.index, self.data['benchmark_returns'], label='基准(买入持有)', linewidth=2, color='red')
        ax1.set_title('累计收益率对比')
        ax1.legend()
        ax1.grid(True)
        
        # 价格和成交量
        ax2.plot(self.data.index, self.data['Close'], label='价格', color='black', linewidth=1.5)
        ax2_twin = ax2.twinx()
        ax2_twin.bar(self.data.index, self.data['Volume'], label='成交量', alpha=0.3, color='gray')
        ax2_twin.plot(self.data.index, self.data['Volume_SMA_Long'], label='成交量均线', color='blue', linewidth=1)
        ax2.set_title('价格与成交量')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True)
        
        # 成交量比率和异常信号
        ax3.plot(self.data.index, self.data['Volume_Ratio_Long'], label='成交量比率', color='purple', linewidth=1.5)
        ax3.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='放量阈值')
        ax3.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5, label='基准线')
        ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='缩量阈值')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(self.data.index, self.positions, label='仓位', color='orange', alpha=0.7, linewidth=2)
        ax3.set_title('成交量比率与仓位')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True)
        
        # 成交量异常因子
        ax4.plot(self.data.index, self.data['Volume_Anomaly_Factor'], label='成交量异常因子', color='navy', linewidth=1.5)
        ax4.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='做多阈值')
        ax4.axhline(y=-0.6, color='green', linestyle='--', alpha=0.7, label='做空阈值')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('成交量异常因子')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('./volume_anomaly_strategy_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    data_path = 'data/data_202410.pkl'
    
    data_handler = DataHandler(data_path)
    data_handler.calculate_volume_anomaly_factor()
    
    strategy = VolumeAnomalyStrategy(data_handler.data)
    strategy.generate_signals()
    
    performance = Performance(data_handler.data, strategy.positions)
    metrics = performance.calculate_metrics()
    
    print("=== 成交量异常因子策略回测结果 ===")
    print(f"年化收益率: {metrics['annual_return']:.4f}")
    print(f"年化波动率: {metrics['annual_volatility']:.4f}")
    print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
    print(f"最大回撤: {metrics['max_drawdown']:.4f}")
    print(f"信息比率: {metrics['information_ratio']:.4f}")
    print(f"胜率: {metrics['win_rate']:.4f}")
    
    performance.plot_results()
    
    results = {
        'strategy_name': '成交量异常因子策略',
        'metrics': metrics,
        'data': data_handler.data,
        'positions': strategy.positions
    }
    
    with open('./volume_anomaly_results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main() 