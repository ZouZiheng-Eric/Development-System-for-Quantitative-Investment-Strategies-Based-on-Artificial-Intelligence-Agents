"""
历史波动率因子策略 - Factor Strategy 9
基于历史波动率的风险分析，捕捉波动率变化带来的交易机会
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
        
    def calculate_volatility_factor(self, short_period=10, long_period=30):
        """计算历史波动率因子"""
        # 计算历史波动率（年化）
        self.data['Volatility_Short'] = self.data['returns'].rolling(window=short_period).std() * np.sqrt(252)
        self.data['Volatility_Long'] = self.data['returns'].rolling(window=long_period).std() * np.sqrt(252)
        
        # 波动率比率
        self.data['Volatility_Ratio'] = self.data['Volatility_Short'] / (self.data['Volatility_Long'] + 1e-8)
        
        # 波动率排名
        self.data['Volatility_Rank'] = self.data['Volatility_Long'].rolling(window=60).rank(pct=True)
        
        # 计算波动率因子
        self._calculate_volatility_signals()
        
    def _calculate_volatility_signals(self):
        """计算波动率信号"""
        # 波动率变化
        self.data['Volatility_Change'] = self.data['Volatility_Long'].pct_change()
        
        # 波动率趋势
        vol_ma_short = self.data['Volatility_Long'].rolling(window=5).mean()
        vol_ma_long = self.data['Volatility_Long'].rolling(window=15).mean()
        self.data['Volatility_Trend'] = (vol_ma_short - vol_ma_long) / (vol_ma_long + 1e-8)
        
        # 波动率异常检测
        vol_mean = self.data['Volatility_Long'].rolling(window=60).mean()
        vol_std = self.data['Volatility_Long'].rolling(window=60).std()
        self.data['Volatility_Z_Score'] = (self.data['Volatility_Long'] - vol_mean) / (vol_std + 1e-8)
        
        # 高波动率持续性
        high_vol = (self.data['Volatility_Rank'] > 0.8).astype(int)
        self.data['High_Vol_Persistence'] = high_vol.rolling(window=5).sum()
        
        # 低波动率持续性
        low_vol = (self.data['Volatility_Rank'] < 0.2).astype(int)
        self.data['Low_Vol_Persistence'] = low_vol.rolling(window=5).sum()
        
        # GARCH效应：高波动率后的均值回归 (修复：直接使用Z分数，高波动率时为正值)
        self.data['Vol_Mean_Reversion'] = np.tanh(self.data['Volatility_Z_Score'])
        
        # 波动率突破
        self.data['Vol_Breakout'] = np.where(
            self.data['Volatility_Ratio'] > 1.5, 1,  # 短期波动率突破
            np.where(self.data['Volatility_Ratio'] < 0.7, -1, 0)
        )
        
        # 价格动量计算 (修复：存储到data中)
        self.data['Price_Momentum'] = self.data['Close'].pct_change(5)
        
        # 波动率与收益率关系
        self.data['Vol_Return_Correlation'] = self.data['returns'].rolling(window=20).corr(self.data['Volatility_Short'])
        
        # 综合波动率因子 (修复：简化逻辑，低波动率看多，高波动率看空)
        self.data['Volatility_Factor'] = (
            -0.5 * self.data['Vol_Mean_Reversion'] +  # 高波动率看空（均值回归）
            -0.3 * (self.data['Volatility_Rank'] - 0.5) +  # 波动率水平（高波动看空）
            0.2 * np.sign(self.data['Vol_Breakout']) * np.sign(self.data['Price_Momentum']).fillna(0)  # 波动率突破配合价格动量
        )
        
        # 标准化因子值到[-1, 1]区间
        factor_std = self.data['Volatility_Factor'].rolling(window=60).std()
        self.data['Volatility_Factor'] = np.tanh(self.data['Volatility_Factor'] / (factor_std + 1e-8))

class VolatilityStrategy:
    """历史波动率策略"""
    def __init__(self, data, threshold=0.5):
        self.data = data
        self.threshold = threshold
        self.positions = pd.Series(index=data.index, dtype=float).fillna(0)
        
    def generate_signals(self):
        """生成交易信号"""
        # 修复：简化信号生成逻辑，使其更容易触发
        conditions = [
            # 强烈看多：低波动率环境 + 因子值为正
            (self.data['Volatility_Factor'] > 0.3) & 
            (self.data['Volatility_Rank'] < 0.4),
            
            # 温和看多：因子值适中正值
            (self.data['Volatility_Factor'] > 0.1) & 
            (self.data['Volatility_Factor'] <= 0.3),
            
            # 强烈看空：高波动率环境 + 因子值为负  
            (self.data['Volatility_Factor'] < -0.3) & 
            (self.data['Volatility_Rank'] > 0.6),
            
            # 温和看空：因子值适中负值
            (self.data['Volatility_Factor'] < -0.1) & 
            (self.data['Volatility_Factor'] >= -0.3),
        ]
        choices = [1.0, 0.5, -1.0, -0.5]  # 满仓做多、半仓做多、满仓做空、半仓做空
        
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
        
        # 修复：信息比率应该是策略收益的信息比率，而不是相对基准的主动收益
        information_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
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
        ax1.plot(self.data.index, self.data['cumulative_returns'], label='波动率策略', linewidth=2, color='blue')
        ax1.plot(self.data.index, self.data['benchmark_returns'], label='基准(买入持有)', linewidth=2, color='red')
        ax1.set_title('累计收益率对比')
        ax1.legend()
        ax1.grid(True)
        
        # 价格和历史波动率
        ax2.plot(self.data.index, self.data['Close'], label='价格', color='black', linewidth=1.5)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(self.data.index, self.data['Volatility_Long'], label='历史波动率', color='red', alpha=0.7)
        ax2_twin.plot(self.data.index, self.data['Volatility_Short'], label='短期波动率', color='blue', alpha=0.7)
        ax2.set_title('价格与历史波动率')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True)
        
        # 波动率排名和比率
        ax3.plot(self.data.index, self.data['Volatility_Rank'], label='波动率排名', color='purple', linewidth=1.5)
        ax3.plot(self.data.index, self.data['Volatility_Ratio'], label='波动率比率', color='green', linewidth=1.5)
        ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='高波动阈值')
        ax3.axhline(y=0.2, color='blue', linestyle='--', alpha=0.7, label='低波动阈值')
        ax3.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5, label='基准线')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(self.data.index, self.positions, label='仓位', color='orange', alpha=0.7, linewidth=2)
        ax3.set_title('波动率排名与比率')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True)
        
        # 波动率因子
        ax4.plot(self.data.index, self.data['Volatility_Factor'], label='波动率因子', color='navy', linewidth=1.5)
        ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='做多阈值')
        ax4.axhline(y=-0.5, color='green', linestyle='--', alpha=0.7, label='做空阈值')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('波动率因子')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('./volatility_strategy_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    data_path = 'data/data_202410.pkl'
    
    data_handler = DataHandler(data_path)
    data_handler.calculate_volatility_factor()
    
    strategy = VolatilityStrategy(data_handler.data)
    strategy.generate_signals()
    
    performance = Performance(data_handler.data, strategy.positions)
    metrics = performance.calculate_metrics()
    
    print("=== 历史波动率因子策略回测结果 ===")
    print(f"年化收益率: {metrics['annual_return']:.4f}")
    print(f"年化波动率: {metrics['annual_volatility']:.4f}")
    print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
    print(f"最大回撤: {metrics['max_drawdown']:.4f}")
    print(f"信息比率: {metrics['information_ratio']:.4f}")
    print(f"胜率: {metrics['win_rate']:.4f}")
    
    performance.plot_results()
    
    results = {
        'strategy_name': '历史波动率因子策略',
        'metrics': metrics,
        'data': data_handler.data,
        'positions': strategy.positions
    }
    
    with open('./volatility_results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main() 