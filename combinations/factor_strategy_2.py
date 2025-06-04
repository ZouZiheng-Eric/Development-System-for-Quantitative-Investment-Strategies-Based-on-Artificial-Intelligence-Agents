"""
MACD动量因子策略 - Factor Strategy 2
基于MACD柱状图和信号线的动量分析，捕捉中期趋势动量
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
        
    def calculate_macd(self, fast_period=12, slow_period=26, signal_period=9):
        """计算MACD指标"""
        # 计算快速和慢速EMA
        exp_fast = self.data['Close'].ewm(span=fast_period).mean()
        exp_slow = self.data['Close'].ewm(span=slow_period).mean()
        
        # MACD线
        self.data['MACD'] = exp_fast - exp_slow
        
        # 信号线
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=signal_period).mean()
        
        # MACD柱状图
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['MACD_Signal']
        
        # 计算MACD动量因子
        self._calculate_macd_momentum_factor()
        
    def _calculate_macd_momentum_factor(self):
        """计算MACD动量因子"""
        # MACD柱状图的变化率
        self.data['MACD_Hist_Change'] = self.data['MACD_Histogram'].diff()
        self.data['MACD_Hist_Change_3d'] = self.data['MACD_Histogram'].diff(3)
        
        # MACD线的变化率
        self.data['MACD_Change'] = self.data['MACD'].diff()
        
        # 标准化处理（增加数值稳定性）
        macd_hist_mean = self.data['MACD_Histogram'].rolling(30).mean()
        macd_hist_std = self.data['MACD_Histogram'].rolling(30).std()
        self.data['MACD_Hist_Norm'] = (self.data['MACD_Histogram'] - macd_hist_mean) / (macd_hist_std + 1e-8)
        
        # MACD金叉死叉检测
        self.data['MACD_Cross'] = np.where(
            (self.data['MACD'] > self.data['MACD_Signal']) & 
            (self.data['MACD'].shift(1) <= self.data['MACD_Signal'].shift(1)), 1,  # 金叉
            np.where(
                (self.data['MACD'] < self.data['MACD_Signal']) & 
                (self.data['MACD'].shift(1) >= self.data['MACD_Signal'].shift(1)), -1, 0  # 死叉
            )
        )
        
        # MACD背离检测
        price_momentum_3d = self.data['Close'].pct_change(3)
        macd_momentum_3d = self.data['MACD'].diff(3)
        self.data['MACD_Price_Divergence'] = np.sign(price_momentum_3d) - np.sign(macd_momentum_3d)
        
        # MACD信号持续性
        bullish_periods = (self.data['MACD_Histogram'] > 0).astype(int)
        bearish_periods = (self.data['MACD_Histogram'] < 0).astype(int)
        self.data['MACD_Bullish_Persistence'] = bullish_periods.rolling(window=5).sum()
        self.data['MACD_Bearish_Persistence'] = bearish_periods.rolling(window=5).sum()
        
        # MACD柱状图强度
        self.data['MACD_Hist_Strength'] = np.abs(self.data['MACD_Histogram']) / (self.data['MACD_Histogram'].rolling(20).std() + 1e-8)
        
        # 综合MACD动量因子
        self.data['MACD_Momentum_Factor'] = (
            0.3 * np.tanh(self.data['MACD_Hist_Norm']) +  # 标准化柱状图强度
            0.25 * np.sign(self.data['MACD_Hist_Change']) +  # 柱状图变化方向
            0.2 * self.data['MACD_Cross'] +  # 金叉死叉信号
            0.15 * np.tanh(self.data['MACD_Price_Divergence']) +  # 背离信号
            0.1 * (self.data['MACD_Bullish_Persistence'] - self.data['MACD_Bearish_Persistence']) / 5  # 持续性
        )

class MACDMomentumStrategy:
    """MACD动量策略"""
    def __init__(self, data, threshold=0.4):
        self.data = data
        self.threshold = threshold
        self.positions = pd.Series(index=data.index, dtype=float).fillna(0)
        
    def generate_signals(self):
        """生成交易信号"""
        # 基于MACD动量因子生成信号，结合多种MACD条件
        conditions = [
            (self.data['MACD_Momentum_Factor'] > self.threshold) & 
            (self.data['MACD_Cross'] == 1) & 
            (self.data['MACD_Bullish_Persistence'] >= 2),  # 强金叉信号，满仓做多
            
            (self.data['MACD_Momentum_Factor'] > 0.2) & 
            (self.data['MACD_Histogram'] > 0) & 
            (self.data['MACD_Hist_Change'] > 0),  # 一般看多信号，半仓做多
            
            (self.data['MACD_Momentum_Factor'] < -self.threshold) & 
            (self.data['MACD_Cross'] == -1) & 
            (self.data['MACD_Bearish_Persistence'] >= 2),  # 强死叉信号，满仓做空
            
            (self.data['MACD_Momentum_Factor'] < -0.2) & 
            (self.data['MACD_Histogram'] < 0) & 
            (self.data['MACD_Hist_Change'] < 0),  # 一般看空信号，半仓做空
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
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 累计收益率对比
        ax1.plot(self.data.index, self.data['cumulative_returns'], label='MACD动量策略', linewidth=2, color='blue')
        ax1.plot(self.data.index, self.data['benchmark_returns'], label='基准(买入持有)', linewidth=2, color='red')
        ax1.set_title('累计收益率对比')
        ax1.legend()
        ax1.grid(True)
        
        # MACD指标
        ax2.plot(self.data.index, self.data['MACD'], label='MACD', color='blue', linewidth=1.5)
        ax2.plot(self.data.index, self.data['MACD_Signal'], label='信号线', color='red', linewidth=1.5)
        ax2.bar(self.data.index, self.data['MACD_Histogram'], label='MACD柱状图', alpha=0.6, color='green')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('MACD指标')
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
        
        # MACD动量因子
        ax4.plot(self.data.index, self.data['MACD_Momentum_Factor'], label='MACD动量因子', color='purple', linewidth=1.5)
        ax4.axhline(y=0.4, color='red', linestyle='--', alpha=0.7, label='做多阈值')
        ax4.axhline(y=-0.4, color='green', linestyle='--', alpha=0.7, label='做空阈值')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('MACD动量因子（增强版）')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('./macd_momentum_strategy_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    # 数据路径
    data_path = 'data/data_202410.pkl'
    
    # 初始化数据处理器
    data_handler = DataHandler(data_path)
    data_handler.calculate_macd()
    
    # 初始化策略
    strategy = MACDMomentumStrategy(data_handler.data)
    strategy.generate_signals()
    
    # 业绩分析
    performance = Performance(data_handler.data, strategy.positions)
    metrics = performance.calculate_metrics()
    
    # 打印结果
    print("=== MACD动量因子策略回测结果 ===")
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
        'strategy_name': 'MACD动量因子策略',
        'metrics': metrics,
        'data': data_handler.data,
        'positions': strategy.positions
    }
    
    with open('./macd_momentum_results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main() 