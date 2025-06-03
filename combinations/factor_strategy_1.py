"""
RSI反转因子策略 - Factor Strategy 1
基于RSI超买超卖区间的反转逻辑，当RSI超买时做空，超卖时做多
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
        
    def calculate_rsi(self, period=14):
        """计算RSI指标"""
        delta = self.data['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        self.data['RSI'] = np.where(avg_loss == 0, 100, self.data['RSI'])
        self.data['RSI'] = np.where(avg_gain == 0, 0, self.data['RSI'])
        
        # 计算RSI反转因子
        self._calculate_rsi_reversal_factor()
        
    def _calculate_rsi_reversal_factor(self):
        """计算RSI反转因子"""
        # RSI的变化率
        self.data['RSI_Change'] = self.data['RSI'].diff()
        self.data['RSI_Change_3d'] = self.data['RSI'].diff(3)
        
        # RSI的标准化（基于历史分布）
        rsi_mean = self.data['RSI'].rolling(window=60).mean()
        rsi_std = self.data['RSI'].rolling(window=60).std()
        self.data['RSI_Norm'] = (self.data['RSI'] - rsi_mean) / (rsi_std + 1e-8)
        
        # RSI超买超卖强度
        self.data['RSI_Overbought_Strength'] = np.maximum(0, self.data['RSI'] - 70) / 30
        self.data['RSI_Oversold_Strength'] = np.maximum(0, 30 - self.data['RSI']) / 30
        
        # RSI动量背离检测
        price_momentum_3d = self.data['Close'].pct_change(3)
        self.data['RSI_Price_Divergence'] = np.sign(price_momentum_3d) - np.sign(self.data['RSI_Change_3d'])
        
        # RSI持续性
        oversold_periods = (self.data['RSI'] < 30).astype(int)
        overbought_periods = (self.data['RSI'] > 70).astype(int)
        self.data['RSI_Oversold_Persistence'] = oversold_periods.rolling(window=5).sum()
        self.data['RSI_Overbought_Persistence'] = overbought_periods.rolling(window=5).sum()
        
        # 综合RSI反转因子
        self.data['RSI_Reversal_Factor'] = (
            0.4 * (self.data['RSI_Oversold_Strength'] - self.data['RSI_Overbought_Strength']) +  # 主要反转信号
            0.3 * np.tanh(self.data['RSI_Price_Divergence']) +  # 背离信号
            0.2 * (self.data['RSI_Oversold_Persistence'] - self.data['RSI_Overbought_Persistence']) / 5 +  # 持续性
            -0.1 * np.tanh(self.data['RSI_Norm'])  # 标准化位置（反向）
        )

class RSIReversalStrategy:
    """RSI反转策略"""
    def __init__(self, data, threshold=0.3):
        self.data = data
        self.threshold = threshold
        self.positions = pd.Series(index=data.index, dtype=float).fillna(0)
        
    def generate_signals(self):
        """生成交易信号"""
        # 基于RSI反转因子生成信号，结合RSI极值条件
        conditions = [
            (self.data['RSI_Reversal_Factor'] > self.threshold) & 
            (self.data['RSI'] < 35) & 
            (self.data['RSI_Oversold_Persistence'] >= 1),  # 超卖反转，满仓做多
            
            (self.data['RSI_Reversal_Factor'] > 0.15) & 
            (self.data['RSI'] < 40),  # 弱超卖，半仓做多
            
            (self.data['RSI_Reversal_Factor'] < -self.threshold) & 
            (self.data['RSI'] > 65) & 
            (self.data['RSI_Overbought_Persistence'] >= 1),  # 超买反转，满仓做空
            
            (self.data['RSI_Reversal_Factor'] < -0.15) & 
            (self.data['RSI'] > 60),  # 弱超买，半仓做空
        ]
        choices = [1, 0.5, -1, -0.5]  # 满仓做多、半仓做多、满仓做空、半仓做空
        
        self.positions = np.select(conditions, choices, default=0)
        self.positions = pd.Series(self.positions, index=self.data.index).shift(1).fillna(0)

class Performance:
    """业绩分析类"""
    def __init__(self, data, positions):
        self.data = data
        self.positions = positions
        self.cum_returns = None
        self._calculate_performance()
        
    def _calculate_performance(self):
        """计算策略业绩"""
        # 计算策略收益
        strategy_returns = self.data['returns'] * self.positions
        self.data['strategy_returns'] = strategy_returns
        self.data['cumulative_returns'] = (1 + strategy_returns).cumprod()
        self.data['benchmark_returns'] = (1 + self.data['returns']).cumprod()
        
    def calculate_metrics(self):
        """计算业绩指标"""
        strategy_returns = self.data['strategy_returns'].dropna()
        
        # 年化收益率
        annual_return = strategy_returns.mean() * 252
        
        # 年化波动率
        annual_volatility = strategy_returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # 最大回撤
        cumulative = self.data['cumulative_returns']
        max_dd = (cumulative / cumulative.expanding().max() - 1).min()
        
        # 信息比率
        excess_returns = strategy_returns - self.data['returns']
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0
        
        # 胜率
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
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 累计收益率对比
        ax1.plot(self.data.index, self.data['cumulative_returns'], label='RSI反转策略', linewidth=2)
        ax1.plot(self.data.index, self.data['benchmark_returns'], label='基准(买入持有)', linewidth=2)
        ax1.set_title('累计收益率对比')
        ax1.legend()
        ax1.grid(True)
        
        # RSI指标和交易信号
        ax2.plot(self.data.index, self.data['RSI'], label='RSI', color='purple')
        ax2.axhline(y=70, color='r', linestyle='--', label='超买线(70)')
        ax2.axhline(y=30, color='g', linestyle='--', label='超卖线(30)')
        ax2.set_title('RSI指标')
        ax2.legend()
        ax2.grid(True)
        
        # 价格和仓位
        ax3.plot(self.data.index, self.data['Close'], label='价格', color='black')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(self.data.index, self.positions, label='仓位', color='orange', alpha=0.7)
        ax3.set_title('价格与仓位')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True)
        
        # 策略日收益率分布
        ax4.plot(self.data.index, self.data['RSI_Reversal_Factor'], label='RSI反转因子', color='navy', linewidth=1.5)
        ax4.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='做多阈值')
        ax4.axhline(y=-0.3, color='green', linestyle='--', alpha=0.7, label='做空阈值')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('RSI反转因子')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('./rsi_reversal_strategy_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    # 数据路径
    data_path = 'code_1/300638_2024.pkl'
    
    # 初始化数据处理器
    data_handler = DataHandler(data_path)
    data_handler.calculate_rsi()
    
    # 初始化策略
    strategy = RSIReversalStrategy(data_handler.data)
    strategy.generate_signals()
    
    # 业绩分析
    performance = Performance(data_handler.data, strategy.positions)
    metrics = performance.calculate_metrics()
    
    # 打印结果
    print("=== RSI反转因子策略回测结果 ===")
    print(f"年化收益率: {metrics['annual_return']:.4f}")
    print(f"年化波动率: {metrics['annual_volatility']:.4f}")
    print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
    print(f"最大回撤: {metrics['max_drawdown']:.4f}")
    print(f"信息比率: {metrics['information_ratio']:.4f}")
    print(f"胜率: {metrics['win_rate']:.4f}")
    
    # 绘制结果
    performance.plot_results()
    
    # 保存结果
    results = {
        'strategy_name': 'RSI反转因子策略',
        'metrics': metrics,
        'data': data_handler.data,
        'positions': strategy.positions
    }
    
    with open('./rsi_reversal_results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main() 