"""
CCI商品通道指数因子策略 - Factor Strategy 5
基于CCI指标的超买超卖判断，结合价格动量生成交易信号
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
        
    def calculate_cci_factor(self, period=20):
        """计算CCI指标和因子"""
        # 计算典型价格 (Typical Price)
        self.data['TP'] = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        
        # 计算移动平均
        self.data['SMA_TP'] = self.data['TP'].rolling(window=period).mean()
        
        # 计算平均偏差
        self.data['Mean_Deviation'] = self.data['TP'].rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        # 计算CCI
        self.data['CCI'] = (self.data['TP'] - self.data['SMA_TP']) / (0.015 * self.data['Mean_Deviation'])
        
        # 计算CCI因子
        self._calculate_cci_momentum_factor()
        
    def _calculate_cci_momentum_factor(self):
        """计算CCI动量因子"""
        # CCI的变化率
        self.data['CCI_Change'] = self.data['CCI'].diff()
        
        # CCI的动量
        self.data['CCI_Momentum'] = self.data['CCI'].diff(3)
        
        # CCI的标准化
        cci_mean = self.data['CCI'].rolling(window=60).mean()
        cci_std = self.data['CCI'].rolling(window=60).std()
        self.data['CCI_Norm'] = (self.data['CCI'] - cci_mean) / (cci_std + 1e-8)
        
        # CCI位置强度
        self.data['CCI_Position_Strength'] = np.where(
            self.data['CCI'] > 100, (self.data['CCI'] - 100) / 100,
            np.where(self.data['CCI'] < -100, (self.data['CCI'] + 100) / 100, 0)
        )
        
        # 综合CCI因子
        # CCI > 100为超买，< -100为超卖
        self.data['CCI_Factor'] = (
            -0.4 * np.tanh(self.data['CCI'] / 100) +  # 主要的反转信号
            -0.3 * np.sign(self.data['CCI_Position_Strength']) +  # 位置强度信号
            0.2 * np.sign(self.data['CCI_Change']) +  # 变化方向
            0.1 * np.tanh(self.data['CCI_Norm'])  # 标准化强度
        )

class CCIStrategy:
    """CCI商品通道指数策略"""
    def __init__(self, data, threshold=0.4):
        self.data = data
        self.threshold = threshold
        self.positions = pd.Series(index=data.index, dtype=float).fillna(0)
        
    def generate_signals(self):
        """生成交易信号"""
        # 基于CCI因子和CCI值生成信号
        conditions = [
            (self.data['CCI_Factor'] > self.threshold) & (self.data['CCI'] < -100),  # 超卖反转
            (self.data['CCI_Factor'] > 0.2) & (self.data['CCI'] < -50),  # 弱超卖
            (self.data['CCI_Factor'] < -self.threshold) & (self.data['CCI'] > 100),  # 超买反转
            (self.data['CCI_Factor'] < -0.2) & (self.data['CCI'] > 50),  # 弱超买
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
        ax1.plot(self.data.index, self.data['cumulative_returns'], label='CCI策略', linewidth=2, color='blue')
        ax1.plot(self.data.index, self.data['benchmark_returns'], label='基准(买入持有)', linewidth=2, color='red')
        ax1.set_title('累计收益率对比')
        ax1.legend()
        ax1.grid(True)
        
        # CCI指标
        ax2.plot(self.data.index, self.data['CCI'], label='CCI', color='purple', linewidth=1.5)
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='超买线(100)')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5, label='中线(0)')
        ax2.axhline(y=-100, color='green', linestyle='--', alpha=0.7, label='超卖线(-100)')
        ax2.set_title('CCI指标')
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
        
        # CCI因子
        ax4.plot(self.data.index, self.data['CCI_Factor'], label='CCI因子', color='navy', linewidth=1.5)
        ax4.axhline(y=0.4, color='red', linestyle='--', alpha=0.7, label='做多阈值')
        ax4.axhline(y=-0.4, color='green', linestyle='--', alpha=0.7, label='做空阈值')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('CCI因子')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('./cci_strategy_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    data_path = 'data/data_202410.pkl'
    
    data_handler = DataHandler(data_path)
    data_handler.calculate_cci_factor()
    
    strategy = CCIStrategy(data_handler.data)
    strategy.generate_signals()
    
    performance = Performance(data_handler.data, strategy.positions)
    metrics = performance.calculate_metrics()
    
    print("=== CCI商品通道指数因子策略回测结果 ===")
    print(f"年化收益率: {metrics['annual_return']:.4f}")
    print(f"年化波动率: {metrics['annual_volatility']:.4f}")
    print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
    print(f"最大回撤: {metrics['max_drawdown']:.4f}")
    print(f"信息比率: {metrics['information_ratio']:.4f}")
    print(f"胜率: {metrics['win_rate']:.4f}")
    
    performance.plot_results()
    
    results = {
        'strategy_name': 'CCI商品通道指数因子策略',
        'metrics': metrics,
        'data': data_handler.data,
        'positions': strategy.positions
    }
    
    with open('./cci_results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main() 