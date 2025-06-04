"""
换手率因子策略 - Factor Strategy 8
基于换手率的流动性分析，捕捉市场活跃度变化信号
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
        
    def calculate_turnover_factor(self, period=20):
        """计算换手率因子"""
        # 计算换手率（如果没有流通股本数据，用成交量代替）
        # 这里假设成交量已经反映了换手情况
        self.data['Turnover'] = self.data['Volume']
        
        # 换手率移动平均
        self.data['Turnover_MA'] = self.data['Turnover'].rolling(window=period).mean()
        
        # 换手率比率
        self.data['Turnover_Ratio'] = self.data['Turnover'] / self.data['Turnover_MA']
        
        # 换手率标准化
        turnover_mean = self.data['Turnover'].rolling(window=60).mean()
        turnover_std = self.data['Turnover'].rolling(window=60).std()
        self.data['Turnover_Z_Score'] = (self.data['Turnover'] - turnover_mean) / (turnover_std + 1e-8)
        
        # 计算换手率因子
        self._calculate_turnover_signals()
        
    def _calculate_turnover_signals(self):
        """计算换手率信号"""
        # 换手率分位数排名
        self.data['Turnover_Rank'] = self.data['Turnover'].rolling(window=60).rank(pct=True)
        
        # 换手率变化率
        self.data['Turnover_Change'] = self.data['Turnover'].pct_change()
        self.data['Turnover_Change_3d'] = self.data['Turnover'].pct_change(3)
        
        # 价格与换手率相关性
        self.data['Price_Turnover_Corr'] = self.data['Close'].rolling(window=20).corr(self.data['Turnover'])
        
        # 换手率趋势
        turnover_ma_short = self.data['Turnover'].rolling(window=5).mean()
        turnover_ma_long = self.data['Turnover'].rolling(window=20).mean()
        self.data['Turnover_Trend'] = (turnover_ma_short - turnover_ma_long) / turnover_ma_long
        
        # 高换手率持续性
        high_turnover = (self.data['Turnover_Rank'] > 0.8).astype(int)
        self.data['High_Turnover_Persistence'] = high_turnover.rolling(window=5).sum()
        
        # 低换手率持续性
        low_turnover = (self.data['Turnover_Rank'] < 0.2).astype(int)
        self.data['Low_Turnover_Persistence'] = low_turnover.rolling(window=5).sum()
        
        # 换手率异常检测
        self.data['Turnover_Anomaly'] = np.where(
            np.abs(self.data['Turnover_Z_Score']) > 2, 
            np.sign(self.data['Turnover_Z_Score']), 0
        )
        
        # 综合换手率因子
        # 高换手率配合上涨趋势看多，低换手率看空
        # 但要考虑换手率的持续性和异常情况
        self.data['Turnover_Factor'] = (
            0.3 * np.tanh(self.data['Turnover_Trend']) +  # 换手率趋势
            0.25 * (self.data['Turnover_Rank'] - 0.5) * 2 +  # 换手率水平（中性化）
            0.2 * np.sign(self.data['Turnover_Change_3d']) +  # 换手率变化方向
            0.15 * (self.data['High_Turnover_Persistence'] / 5 - 0.5) * 2 +  # 高换手持续性
            -0.1 * (self.data['Low_Turnover_Persistence'] / 5 - 0.5) * 2  # 低换手持续性（反向）
        )
        
        # 价格动量过滤
        price_momentum = self.data['Close'].pct_change(5)
        self.data['Turnover_Factor'] = self.data['Turnover_Factor'] * (1 + 0.3 * np.tanh(price_momentum * 10))

class TurnoverStrategy:
    """换手率策略"""
    def __init__(self, data, threshold=0.4):
        self.data = data
        self.threshold = threshold
        self.positions = pd.Series(index=data.index, dtype=float).fillna(0)
        
    def generate_signals(self):
        """生成交易信号"""
        # 基于换手率因子生成信号
        conditions = [
            (self.data['Turnover_Factor'] > self.threshold) & 
            (self.data['Turnover_Rank'] > 0.6) & 
            (self.data['High_Turnover_Persistence'] >= 2),  # 高换手率持续看多
            
            (self.data['Turnover_Factor'] > 0.2) & 
            (self.data['Turnover_Trend'] > 0.1),  # 换手率上升趋势
            
            (self.data['Turnover_Factor'] < -self.threshold) & 
            (self.data['Turnover_Rank'] < 0.4) & 
            (self.data['Low_Turnover_Persistence'] >= 2),  # 低换手率持续看空
            
            (self.data['Turnover_Factor'] < -0.2) & 
            (self.data['Turnover_Trend'] < -0.1),  # 换手率下降趋势
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
        ax1.plot(self.data.index, self.data['cumulative_returns'], label='换手率策略', linewidth=2, color='blue')
        ax1.plot(self.data.index, self.data['benchmark_returns'], label='基准(买入持有)', linewidth=2, color='red')
        ax1.set_title('累计收益率对比')
        ax1.legend()
        ax1.grid(True)
        
        # 价格和换手率
        ax2.plot(self.data.index, self.data['Close'], label='价格', color='black', linewidth=1.5)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(self.data.index, self.data['Turnover'], label='换手率', color='blue', alpha=0.7)
        ax2_twin.plot(self.data.index, self.data['Turnover_MA'], label='换手率均线', color='red', linewidth=1)
        ax2.set_title('价格与换手率')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True)
        
        # 换手率排名和趋势
        ax3.plot(self.data.index, self.data['Turnover_Rank'], label='换手率排名', color='purple', linewidth=1.5)
        ax3.plot(self.data.index, self.data['Turnover_Trend'], label='换手率趋势', color='green', linewidth=1.5)
        ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='高换手阈值')
        ax3.axhline(y=0.2, color='blue', linestyle='--', alpha=0.7, label='低换手阈值')
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(self.data.index, self.positions, label='仓位', color='orange', alpha=0.7, linewidth=2)
        ax3.set_title('换手率排名与趋势')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True)
        
        # 换手率因子
        ax4.plot(self.data.index, self.data['Turnover_Factor'], label='换手率因子', color='navy', linewidth=1.5)
        ax4.axhline(y=0.4, color='red', linestyle='--', alpha=0.7, label='做多阈值')
        ax4.axhline(y=-0.4, color='green', linestyle='--', alpha=0.7, label='做空阈值')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('换手率因子')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('./turnover_strategy_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    data_path = 'data/data_202410.pkl'
    
    data_handler = DataHandler(data_path)
    data_handler.calculate_turnover_factor()
    
    strategy = TurnoverStrategy(data_handler.data)
    strategy.generate_signals()
    
    performance = Performance(data_handler.data, strategy.positions)
    metrics = performance.calculate_metrics()
    
    print("=== 换手率因子策略回测结果 ===")
    print(f"年化收益率: {metrics['annual_return']:.4f}")
    print(f"年化波动率: {metrics['annual_volatility']:.4f}")
    print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
    print(f"最大回撤: {metrics['max_drawdown']:.4f}")
    print(f"信息比率: {metrics['information_ratio']:.4f}")
    print(f"胜率: {metrics['win_rate']:.4f}")
    
    performance.plot_results()
    
    results = {
        'strategy_name': '换手率因子策略',
        'metrics': metrics,
        'data': data_handler.data,
        'positions': strategy.positions
    }
    
    with open('./turnover_results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main() 