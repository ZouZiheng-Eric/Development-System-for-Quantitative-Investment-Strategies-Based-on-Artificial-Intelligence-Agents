"""
布林带位置因子策略 - Factor Strategy 3
基于%B位置指标的均值回归策略，利用价格相对于布林带的位置判断超买超卖
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
        
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """计算布林带指标"""
        # 中轨（移动平均线）
        self.data['BB_Middle'] = self.data['Close'].rolling(window=period).mean()
        
        # 标准差
        rolling_std = self.data['Close'].rolling(window=period).std()
        
        # 上轨和下轨
        self.data['BB_Upper'] = self.data['BB_Middle'] + (std_dev * rolling_std)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (std_dev * rolling_std)
        
        # %B指标：价格在布林带中的相对位置
        self.data['BB_Percent_B'] = (self.data['Close'] - self.data['BB_Lower']) / (self.data['BB_Upper'] - self.data['BB_Lower'])
        
        # 布林带宽度
        self.data['BB_Width'] = (self.data['BB_Upper'] - self.data['BB_Lower']) / self.data['BB_Middle']
        
        # 计算布林带位置因子
        self._calculate_bollinger_position_factor()
        
    def _calculate_bollinger_position_factor(self):
        """计算布林带位置因子"""
        # %B的移动平均，平滑处理
        self.data['BB_Percent_B_MA'] = self.data['BB_Percent_B'].rolling(window=5).mean()
        
        # %B的标准化，基于历史分布
        bb_mean = self.data['BB_Percent_B'].rolling(window=60).mean()
        bb_std = self.data['BB_Percent_B'].rolling(window=60).std()
        self.data['BB_Percent_B_Norm'] = (self.data['BB_Percent_B'] - bb_mean) / (bb_std + 1e-8)
        
        # 价格相对于中轨的位置
        self.data['Price_to_Middle'] = (self.data['Close'] - self.data['BB_Middle']) / self.data['BB_Middle']
        
        # 布林带挤压检测（低波动环境）
        self.data['BB_Squeeze'] = self.data['BB_Width'].rolling(window=20).rank(pct=True)
        
        # 综合布林带位置因子
        # 当%B接近1时（接近上轨），产生做空信号
        # 当%B接近0时（接近下轨），产生做多信号
        # 同时考虑布林带宽度和价格动量
        self.data['Bollinger_Position_Factor'] = (
            -2.0 * (self.data['BB_Percent_B'] - 0.5) +  # 主要的均值回归信号
            -0.5 * np.tanh(self.data['BB_Percent_B_Norm']) +  # 标准化的位置信号
            0.3 * (0.5 - self.data['BB_Squeeze']) +  # 在挤压期给予更强信号
            -0.2 * np.sign(self.data['Price_to_Middle'])  # 价格相对中轨位置
        )

class BollingerPositionStrategy:
    """布林带位置策略"""
    def __init__(self, data, threshold=0.6):
        self.data = data
        self.threshold = threshold
        self.positions = pd.Series(index=data.index, dtype=float).fillna(0)
        
    def generate_signals(self):
        """生成交易信号"""
        # 基于布林带位置因子生成信号
        # 结合%B位置和因子得分
        conditions = [
            (self.data['Bollinger_Position_Factor'] > self.threshold) & (self.data['BB_Percent_B'] <= 0.2),  # 强烈超卖
            (self.data['Bollinger_Position_Factor'] > 0.3) & (self.data['BB_Percent_B'] <= 0.3),  # 超卖
            (self.data['Bollinger_Position_Factor'] < -self.threshold) & (self.data['BB_Percent_B'] >= 0.8),  # 强烈超买
            (self.data['Bollinger_Position_Factor'] < -0.3) & (self.data['BB_Percent_B'] >= 0.7),  # 超买
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
        
        # 卡尔马比率
        calmar_ratio = annual_return / (-max_dd) if max_dd < 0 else 0
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'information_ratio': information_ratio,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio
        }
        
    def plot_results(self):
        """绘制回测结果"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 累计收益率对比
        ax1.plot(self.data.index, self.data['cumulative_returns'], label='布林带位置策略', linewidth=2, color='blue')
        ax1.plot(self.data.index, self.data['benchmark_returns'], label='基准(买入持有)', linewidth=2, color='red')
        ax1.set_title('累计收益率对比')
        ax1.legend()
        ax1.grid(True)
        
        # 布林带和价格
        ax2.plot(self.data.index, self.data['Close'], label='价格', color='black', linewidth=2)
        ax2.plot(self.data.index, self.data['BB_Upper'], label='上轨', color='red', alpha=0.7)
        ax2.plot(self.data.index, self.data['BB_Middle'], label='中轨', color='blue', alpha=0.7)
        ax2.plot(self.data.index, self.data['BB_Lower'], label='下轨', color='green', alpha=0.7)
        ax2.fill_between(self.data.index, self.data['BB_Upper'], self.data['BB_Lower'], alpha=0.1, color='gray')
        ax2.set_title('布林带与价格')
        ax2.legend()
        ax2.grid(True)
        
        # %B指标和仓位
        ax3.plot(self.data.index, self.data['BB_Percent_B'], label='%B', color='purple', linewidth=1.5)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='上轨(1.0)')
        ax3.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='超买(0.8)')
        ax3.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label='中位(0.5)')
        ax3.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='超卖(0.2)')
        ax3.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='下轨(0.0)')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(self.data.index, self.positions, label='仓位', color='brown', alpha=0.8, linewidth=2)
        ax3.set_title('%B指标与仓位')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True)
        
        # 布林带位置因子
        ax4.plot(self.data.index, self.data['Bollinger_Position_Factor'], label='布林带位置因子', color='navy', linewidth=1.5)
        ax4.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='做多阈值')
        ax4.axhline(y=-0.6, color='green', linestyle='--', alpha=0.7, label='做空阈值')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('布林带位置因子')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('./bollinger_position_strategy_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    data_path = 'data/data_202410.pkl'
    
    data_handler = DataHandler(data_path)
    data_handler.calculate_bollinger_bands()
    
    strategy = BollingerPositionStrategy(data_handler.data)
    strategy.generate_signals()
    
    performance = Performance(data_handler.data, strategy.positions)
    metrics = performance.calculate_metrics()
    
    print("=== 布林带位置因子策略回测结果 ===")
    print(f"年化收益率: {metrics['annual_return']:.4f}")
    print(f"年化波动率: {metrics['annual_volatility']:.4f}")
    print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
    print(f"最大回撤: {metrics['max_drawdown']:.4f}")
    print(f"信息比率: {metrics['information_ratio']:.4f}")
    print(f"胜率: {metrics['win_rate']:.4f}")
    
    performance.plot_results()
    
    results = {
        'strategy_name': '布林带位置因子策略',
        'metrics': metrics,
        'data': data_handler.data,
        'positions': strategy.positions
    }
    
    with open('./bollinger_position_results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main() 