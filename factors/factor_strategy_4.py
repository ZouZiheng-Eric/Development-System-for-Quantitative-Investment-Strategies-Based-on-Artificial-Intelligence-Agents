"""
威廉指标反转因子策略 - Factor Strategy 4
基于Williams %R的反转信号，结合超买超卖判断生成交易信号
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
        
    def calculate_williams_r(self, period=14):
        """计算威廉指标"""
        # 计算最高价和最低价的滚动窗口
        highest_high = self.data['High'].rolling(window=period).max()
        lowest_low = self.data['Low'].rolling(window=period).min()
        
        # 计算Williams %R
        self.data['Williams_R'] = (highest_high - self.data['Close']) / (highest_high - lowest_low) * (-100)
        
        # 计算Williams %R的移动平均
        self.data['Williams_R_MA'] = self.data['Williams_R'].rolling(window=5).mean()
        
        # 计算威廉指标反转因子
        self._calculate_williams_reversal_factor()
        
    def _calculate_williams_reversal_factor(self):
        """计算威廉指标反转因子"""
        # Williams %R的变化率
        self.data['Williams_R_Change'] = self.data['Williams_R'].diff()
        
        # Williams %R的动量
        self.data['Williams_R_Momentum'] = self.data['Williams_R'].diff(3)
        
        # 标准化Williams %R
        wr_mean = self.data['Williams_R'].rolling(window=60).mean()
        wr_std = self.data['Williams_R'].rolling(window=60).std()
        self.data['Williams_R_Norm'] = (self.data['Williams_R'] - wr_mean) / (wr_std + 1e-8)
        
        # 威廉指标反转因子
        # 当Williams %R < -80时（超卖），产生做多信号
        # 当Williams %R > -20时（超买），产生做空信号
        self.data['Williams_Reversal_Factor'] = (
            np.where(self.data['Williams_R'] < -80, (-80 - self.data['Williams_R']) / 20, 0) +  # 超卖信号
            np.where(self.data['Williams_R'] > -20, (-20 - self.data['Williams_R']) / 20, 0) +  # 超买信号
            0.3 * np.sign(self.data['Williams_R_Change']) +  # 变化方向
            0.2 * np.tanh(self.data['Williams_R_Norm'])  # 标准化强度
        )

class WilliamsReversalStrategy:
    """威廉指标反转策略"""
    def __init__(self, data, threshold=0.5):
        self.data = data
        self.threshold = threshold
        self.positions = pd.Series(index=data.index, dtype=float).fillna(0)
        
    def generate_signals(self):
        """生成交易信号"""
        # 基于威廉指标反转因子生成信号
        conditions = [
            (self.data['Williams_Reversal_Factor'] > self.threshold) & (self.data['Williams_R'] < -70),  # 强烈超卖
            (self.data['Williams_Reversal_Factor'] < -self.threshold) & (self.data['Williams_R'] > -30),  # 强烈超买
        ]
        choices = [1, -1]  # 做多、做空
        
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
        ax1.plot(self.data.index, self.data['cumulative_returns'], label='威廉指标反转策略', linewidth=2, color='blue')
        ax1.plot(self.data.index, self.data['benchmark_returns'], label='基准(买入持有)', linewidth=2, color='red')
        ax1.set_title('累计收益率对比')
        ax1.legend()
        ax1.grid(True)
        
        # Williams %R指标
        ax2.plot(self.data.index, self.data['Williams_R'], label='Williams %R', color='purple', linewidth=1.5)
        ax2.axhline(y=-20, color='red', linestyle='--', alpha=0.7, label='超买线(-20)')
        ax2.axhline(y=-50, color='gray', linestyle='-', alpha=0.5, label='中线(-50)')
        ax2.axhline(y=-80, color='green', linestyle='--', alpha=0.7, label='超卖线(-80)')
        ax2.set_title('Williams %R指标')
        ax2.legend()
        ax2.grid(True)
        
        # 价格和仓位
        ax3.plot(self.data.index, self.data['Close'], label='价格', color='black', linewidth=1.5)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(self.data.index, self.positions, label='仓位', color='orange', alpha=0.7, linewidth=2)
        ax3.set_title('价格与仓位')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True)
        
        # 威廉指标反转因子
        ax4.plot(self.data.index, self.data['Williams_Reversal_Factor'], label='威廉指标反转因子', color='navy', linewidth=1.5)
        ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='做多阈值')
        ax4.axhline(y=-0.5, color='green', linestyle='--', alpha=0.7, label='做空阈值')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('威廉指标反转因子')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('./williams_reversal_strategy_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    data_path = 'data/data_202410.pkl'
    
    data_handler = DataHandler(data_path)
    data_handler.calculate_williams_r()
    
    strategy = WilliamsReversalStrategy(data_handler.data)
    strategy.generate_signals()
    
    performance = Performance(data_handler.data, strategy.positions)
    metrics = performance.calculate_metrics()
    
    print("=== 威廉指标反转因子策略回测结果 ===")
    print(f"年化收益率: {metrics['annual_return']:.4f}")
    print(f"年化波动率: {metrics['annual_volatility']:.4f}")
    print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
    print(f"最大回撤: {metrics['max_drawdown']:.4f}")
    print(f"信息比率: {metrics['information_ratio']:.4f}")
    print(f"胜率: {metrics['win_rate']:.4f}")
    
    performance.plot_results()
    
    results = {
        'strategy_name': '威廉指标反转因子策略',
        'metrics': metrics,
        'data': data_handler.data,
        'positions': strategy.positions
    }
    
    with open('./williams_reversal_results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main() 