"""
VWAP偏离因子策略 - Factor Strategy 10
基于成交量加权平均价格的偏离分析，捕捉价格回归机会
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
        
    def calculate_vwap_factor(self, period=20):
        """计算VWAP偏离因子"""
        # 计算VWAP
        self.data['Typical_Price'] = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        self.data['Price_Volume'] = self.data['Typical_Price'] * self.data['Volume']
        
        # 滚动VWAP
        self.data['VWAP'] = (
            self.data['Price_Volume'].rolling(window=period).sum() / 
            self.data['Volume'].rolling(window=period).sum()
        )
        
        # VWAP偏离
        self.data['VWAP_Deviation'] = (self.data['Close'] - self.data['VWAP']) / self.data['VWAP']
        
        # 计算VWAP因子
        self._calculate_vwap_signals()
        
    def _calculate_vwap_signals(self):
        """计算VWAP信号"""
        # VWAP偏离的标准化
        vwap_dev_mean = self.data['VWAP_Deviation'].rolling(window=60).mean()
        vwap_dev_std = self.data['VWAP_Deviation'].rolling(window=60).std()
        self.data['VWAP_Deviation_Norm'] = (
            (self.data['VWAP_Deviation'] - vwap_dev_mean) / (vwap_dev_std + 1e-8)
        )
        
        # VWAP偏离的分位数排名
        self.data['VWAP_Deviation_Rank'] = self.data['VWAP_Deviation'].rolling(window=60).rank(pct=True)
        
        # VWAP偏离的持续性
        positive_dev = (self.data['VWAP_Deviation'] > 0).astype(int)
        negative_dev = (self.data['VWAP_Deviation'] < 0).astype(int)
        
        self.data['Positive_Dev_Persistence'] = positive_dev.rolling(window=5).sum()
        self.data['Negative_Dev_Persistence'] = negative_dev.rolling(window=5).sum()
        
        # VWAP偏离变化率
        self.data['VWAP_Deviation_Change'] = self.data['VWAP_Deviation'].diff()
        
        # 多周期VWAP偏离
        # 短期VWAP (5日)
        vwap_short = (
            self.data['Price_Volume'].rolling(window=5).sum() / 
            self.data['Volume'].rolling(window=5).sum()
        )
        self.data['VWAP_Short_Deviation'] = (self.data['Close'] - vwap_short) / vwap_short
        
        # 长期VWAP (40日)
        vwap_long = (
            self.data['Price_Volume'].rolling(window=40).sum() / 
            self.data['Volume'].rolling(window=40).sum()
        )
        self.data['VWAP_Long_Deviation'] = (self.data['Close'] - vwap_long) / vwap_long
        
        # VWAP偏离趋势
        self.data['VWAP_Deviation_Trend'] = (
            self.data['VWAP_Short_Deviation'] - self.data['VWAP_Long_Deviation']
        )
        
        # 成交量权重调整
        volume_weight = self.data['Volume'] / self.data['Volume'].rolling(window=20).mean()
        volume_weight = np.clip(volume_weight, 0.5, 2.0)  # 限制权重范围
        
        # 综合VWAP偏离因子
        # 当价格高于VWAP时看空（均值回归），低于VWAP时看多
        # 但要考虑偏离的持续性和趋势
        self.data['VWAP_Factor'] = (
            -0.4 * np.tanh(self.data['VWAP_Deviation_Norm'] * 2) +  # 主要的均值回归信号
            -0.3 * (self.data['VWAP_Deviation_Rank'] - 0.5) * 2 +  # 偏离排名信号
            -0.2 * np.tanh(self.data['VWAP_Deviation_Trend'] * 5) +  # 偏离趋势
            0.1 * np.sign(self.data['VWAP_Deviation_Change'])  # 偏离变化方向（反向）
        )
        
        # 成交量调整：高成交量时信号更可靠
        self.data['VWAP_Factor'] = self.data['VWAP_Factor'] * volume_weight

class VWAPStrategy:
    """VWAP偏离策略"""
    def __init__(self, data, threshold=0.6):
        self.data = data
        self.threshold = threshold
        self.positions = pd.Series(index=data.index, dtype=float).fillna(0)
        
    def generate_signals(self):
        """生成交易信号"""
        # 基于VWAP偏离因子生成信号
        conditions = [
            (self.data['VWAP_Factor'] > self.threshold) & 
            (self.data['VWAP_Deviation'] < -0.02) & 
            (self.data['Negative_Dev_Persistence'] >= 2),  # 持续低于VWAP，看多
            
            (self.data['VWAP_Factor'] > 0.3) & 
            (self.data['VWAP_Deviation_Rank'] < 0.3),  # 偏离程度较大，看多
            
            (self.data['VWAP_Factor'] < -self.threshold) & 
            (self.data['VWAP_Deviation'] > 0.02) & 
            (self.data['Positive_Dev_Persistence'] >= 2),  # 持续高于VWAP，看空
            
            (self.data['VWAP_Factor'] < -0.3) & 
            (self.data['VWAP_Deviation_Rank'] > 0.7),  # 偏离程度较大，看空
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
        ax1.plot(self.data.index, self.data['cumulative_returns'], label='VWAP偏离策略', linewidth=2, color='blue')
        ax1.plot(self.data.index, self.data['benchmark_returns'], label='基准(买入持有)', linewidth=2, color='red')
        ax1.set_title('累计收益率对比')
        ax1.legend()
        ax1.grid(True)
        
        # 价格与VWAP
        ax2.plot(self.data.index, self.data['Close'], label='价格', color='black', linewidth=1.5)
        ax2.plot(self.data.index, self.data['VWAP'], label='VWAP', color='blue', linewidth=1.5, alpha=0.7)
        ax2.fill_between(self.data.index, self.data['Close'], self.data['VWAP'], 
                        where=(self.data['Close'] > self.data['VWAP']), 
                        alpha=0.3, color='red', label='价格>VWAP')
        ax2.fill_between(self.data.index, self.data['Close'], self.data['VWAP'], 
                        where=(self.data['Close'] <= self.data['VWAP']), 
                        alpha=0.3, color='green', label='价格≤VWAP')
        ax2.set_title('价格与VWAP')
        ax2.legend()
        ax2.grid(True)
        
        # VWAP偏离和排名
        ax3.plot(self.data.index, self.data['VWAP_Deviation'] * 100, label='VWAP偏离(%)', color='purple', linewidth=1.5)
        ax3.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='上偏离阈值(2%)')
        ax3.axhline(y=-2, color='green', linestyle='--', alpha=0.7, label='下偏离阈值(-2%)')
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(self.data.index, self.positions, label='仓位', color='orange', alpha=0.7, linewidth=2)
        ax3.set_title('VWAP偏离与仓位')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True)
        
        # VWAP因子
        ax4.plot(self.data.index, self.data['VWAP_Factor'], label='VWAP因子', color='navy', linewidth=1.5)
        ax4.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='做多阈值')
        ax4.axhline(y=-0.6, color='green', linestyle='--', alpha=0.7, label='做空阈值')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('VWAP因子')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('./vwap_strategy_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    data_path = 'code_1/300638_2024.pkl'
    
    data_handler = DataHandler(data_path)
    data_handler.calculate_vwap_factor()
    
    strategy = VWAPStrategy(data_handler.data)
    strategy.generate_signals()
    
    performance = Performance(data_handler.data, strategy.positions)
    metrics = performance.calculate_metrics()
    
    print("=== VWAP偏离因子策略回测结果 ===")
    print(f"年化收益率: {metrics['annual_return']:.4f}")
    print(f"年化波动率: {metrics['annual_volatility']:.4f}")
    print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
    print(f"最大回撤: {metrics['max_drawdown']:.4f}")
    print(f"信息比率: {metrics['information_ratio']:.4f}")
    print(f"胜率: {metrics['win_rate']:.4f}")
    
    performance.plot_results()
    
    results = {
        'strategy_name': 'VWAP偏离因子策略',
        'metrics': metrics,
        'data': data_handler.data,
        'positions': strategy.positions
    }
    
    with open('./vwap_results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main() 