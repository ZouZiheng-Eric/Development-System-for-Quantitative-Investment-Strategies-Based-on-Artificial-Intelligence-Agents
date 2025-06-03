"""
价量背离因子策略 - Factor Strategy 6
基于价格和成交量的背离关系，捕捉市场情绪变化信号
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
        
    def calculate_price_volume_divergence_factor(self, period=20):
        """计算价量背离因子"""
        # 价格动量
        self.data['Price_Momentum'] = self.data['Close'].pct_change(period)
        
        # 成交量动量
        self.data['Volume_Change'] = self.data['Volume'].pct_change(period)
        
        # 价格趋势强度（使用RSI思想）
        price_change = self.data['Close'].diff()
        price_gain = price_change.where(price_change > 0, 0.0)
        price_loss = -price_change.where(price_change < 0, 0.0)
        
        avg_gain = price_gain.rolling(window=period).mean()
        avg_loss = price_loss.rolling(window=period).mean()
        
        price_strength = avg_gain / (avg_gain + avg_loss)
        self.data['Price_Strength'] = price_strength.fillna(0.5)
        
        # 成交量趋势强度
        volume_ma = self.data['Volume'].rolling(window=period).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / volume_ma
        
        # 计算价量背离因子
        self._calculate_divergence_signals()
        
    def _calculate_divergence_signals(self):
        """计算价量背离信号"""
        # 价格相对强度排名
        self.data['Price_Rank'] = self.data['Price_Strength'].rolling(window=60).rank(pct=True)
        
        # 成交量相对强度排名
        self.data['Volume_Rank'] = self.data['Volume_Ratio'].rolling(window=60).rank(pct=True)
        
        # 价量相关性
        self.data['PV_Correlation'] = self.data['Close'].rolling(window=20).corr(self.data['Volume'])
        
        # 背离信号计算
        # 当价格走强但成交量萎缩时，产生看空信号（顶背离）
        # 当价格走弱但成交量放大时，产生看多信号（底背离）
        price_volume_diff = self.data['Price_Rank'] - self.data['Volume_Rank']
        
        # 标准化背离强度
        pv_diff_mean = price_volume_diff.rolling(window=60).mean()
        pv_diff_std = price_volume_diff.rolling(window=60).std()
        self.data['PV_Divergence_Norm'] = (price_volume_diff - pv_diff_mean) / (pv_diff_std + 1e-8)
        
        # 动量背离
        price_momentum_rank = self.data['Price_Momentum'].rolling(window=40).rank(pct=True)
        volume_momentum_rank = self.data['Volume_Change'].rolling(window=40).rank(pct=True)
        momentum_divergence = price_momentum_rank - volume_momentum_rank
        
        # 综合价量背离因子
        self.data['Price_Volume_Divergence_Factor'] = (
            -0.5 * np.tanh(self.data['PV_Divergence_Norm']) +  # 主要背离信号（反转）
            -0.3 * np.tanh(momentum_divergence) +  # 动量背离
            0.2 * (0.5 - np.abs(self.data['PV_Correlation'])) +  # 相关性弱化时加强信号
            0.1 * np.sign(self.data['Volume_Change'])  # 成交量变化方向
        )

class PriceVolumeDivergenceStrategy:
    """价量背离策略"""
    def __init__(self, data, threshold=0.5):
        self.data = data
        self.threshold = threshold
        self.positions = pd.Series(index=data.index, dtype=float).fillna(0)
        
    def generate_signals(self):
        """生成交易信号"""
        # 基于价量背离因子生成信号
        # 结合价格强度和成交量比率进行过滤
        conditions = [
            (self.data['Price_Volume_Divergence_Factor'] > self.threshold) & 
            (self.data['Price_Rank'] < 0.3) & (self.data['Volume_Rank'] > 0.6),  # 底背离：价格弱但量大
            
            (self.data['Price_Volume_Divergence_Factor'] > 0.3) & 
            (self.data['Price_Rank'] < 0.4),  # 弱底背离
            
            (self.data['Price_Volume_Divergence_Factor'] < -self.threshold) & 
            (self.data['Price_Rank'] > 0.7) & (self.data['Volume_Rank'] < 0.4),  # 顶背离：价格强但量小
            
            (self.data['Price_Volume_Divergence_Factor'] < -0.3) & 
            (self.data['Price_Rank'] > 0.6),  # 弱顶背离
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
        ax1.plot(self.data.index, self.data['cumulative_returns'], label='价量背离策略', linewidth=2, color='blue')
        ax1.plot(self.data.index, self.data['benchmark_returns'], label='基准(买入持有)', linewidth=2, color='red')
        ax1.set_title('累计收益率对比')
        ax1.legend()
        ax1.grid(True)
        
        # 价格和成交量
        ax2.plot(self.data.index, self.data['Close'], label='价格', color='black', linewidth=1.5)
        ax2_twin = ax2.twinx()
        ax2_twin.bar(self.data.index, self.data['Volume'], label='成交量', alpha=0.3, color='gray')
        ax2.set_title('价格与成交量')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True)
        
        # 价格强度与成交量比率
        ax3.plot(self.data.index, self.data['Price_Rank'], label='价格强度排名', color='red', linewidth=1.5)
        ax3.plot(self.data.index, self.data['Volume_Rank'], label='成交量比率排名', color='blue', linewidth=1.5)
        ax3.axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=0.3, color='green', linestyle='--', alpha=0.5)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(self.data.index, self.positions, label='仓位', color='orange', alpha=0.7, linewidth=2)
        ax3.set_title('价格强度与成交量比率')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True)
        
        # 价量背离因子
        ax4.plot(self.data.index, self.data['Price_Volume_Divergence_Factor'], label='价量背离因子', color='purple', linewidth=1.5)
        ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='做多阈值')
        ax4.axhline(y=-0.5, color='green', linestyle='--', alpha=0.7, label='做空阈值')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('价量背离因子')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('./price_volume_divergence_strategy_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    data_path = 'code_1/300638_2024.pkl'
    
    data_handler = DataHandler(data_path)
    data_handler.calculate_price_volume_divergence_factor()
    
    strategy = PriceVolumeDivergenceStrategy(data_handler.data)
    strategy.generate_signals()
    
    performance = Performance(data_handler.data, strategy.positions)
    metrics = performance.calculate_metrics()
    
    print("=== 价量背离因子策略回测结果 ===")
    print(f"年化收益率: {metrics['annual_return']:.4f}")
    print(f"年化波动率: {metrics['annual_volatility']:.4f}")
    print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
    print(f"最大回撤: {metrics['max_drawdown']:.4f}")
    print(f"信息比率: {metrics['information_ratio']:.4f}")
    print(f"胜率: {metrics['win_rate']:.4f}")
    
    performance.plot_results()
    
    results = {
        'strategy_name': '价量背离因子策略',
        'metrics': metrics,
        'data': data_handler.data,
        'positions': strategy.positions
    }
    
    with open('./price_volume_divergence_results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main() 