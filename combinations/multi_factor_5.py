"""
多因子组合策略 - 5因子组合
基于前5个最优因子的多种权重组合策略，包括等权重、IC加权、风险平价等方法
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 添加因子策略路径以导入单因子模块
sys.path.append('../factors')

# 导入单因子策略类
from factor_strategy_1 import RSIReversalStrategy, DataHandler as DataHandler1
from factor_strategy_2 import MACDMomentumStrategy, DataHandler as DataHandler2
from factor_strategy_3 import BollingerPositionStrategy, DataHandler as DataHandler3
from factor_strategy_4 import WilliamsReversalStrategy, DataHandler as DataHandler4
from factor_strategy_5 import CCIStrategy, DataHandler as DataHandler5

class FactorDataProcessor:
    """因子数据处理器 - 统一处理多个因子的计算和标准化"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.factors = {}
        self.returns = None
        
    def load_and_calculate_factors(self):
        """加载数据并计算前5个因子"""
        print("正在计算前5个因子...")
        
        # 1. RSI反转因子
        data_handler1 = DataHandler1(self.data_path)
        data_handler1.calculate_rsi()
        self.factors['RSI_Reversal'] = data_handler1.data['RSI_Reversal_Factor'].fillna(0)
        
        # 2. MACD动量因子
        data_handler2 = DataHandler2(self.data_path)
        data_handler2.calculate_macd()
        self.factors['MACD_Momentum'] = data_handler2.data['MACD_Momentum_Factor'].fillna(0)
        
        # 3. 布林带因子
        data_handler3 = DataHandler3(self.data_path)
        data_handler3.calculate_bollinger_bands()
        self.factors['Bollinger_Position'] = data_handler3.data['Bollinger_Position_Factor'].fillna(0)
        
        # 4. 威廉指标因子
        data_handler4 = DataHandler4(self.data_path)
        data_handler4.calculate_williams_r()
        self.factors['Williams_Reversal'] = data_handler4.data['Williams_Reversal_Factor'].fillna(0)
        
        # 5. CCI因子
        data_handler5 = DataHandler5(self.data_path)
        data_handler5.calculate_cci_factor()
        self.factors['CCI'] = data_handler5.data['CCI_Factor'].fillna(0)
        
        # 使用第一个数据处理器的收益率作为基准
        self.returns = data_handler1.data['returns'].fillna(0)
        
        print(f"成功计算{len(self.factors)}个因子")
        
    def standardize_factors(self, window=60):
        """标准化因子 - Z-score方法"""
        for name, factor in self.factors.items():
            rolling_mean = factor.rolling(window=window, min_periods=1).mean()
            rolling_std = factor.rolling(window=window, min_periods=1).std()
            self.factors[name] = (factor - rolling_mean) / (rolling_std + 1e-8)
        print("因子标准化完成")
        
    def calculate_factor_ic(self, window=60):
        """计算因子IC（信息系数）"""
        factor_ics = {}
        for name, factor in self.factors.items():
            # 计算滚动IC
            rolling_ic = []
            for i in range(window, len(factor)):
                window_factor = factor.iloc[i-window:i]
                window_returns = self.returns.iloc[i-window:i]
                ic = window_factor.corr(window_returns)
                rolling_ic.append(ic if not np.isnan(ic) else 0)
            
            factor_ics[name] = pd.Series(rolling_ic, 
                                       index=factor.index[window:], 
                                       name=f'{name}_IC')
        
        return factor_ics

class MultiFactor5Strategy:
    """5因子组合策略"""
    def __init__(self, factors, returns, combination_method='equal_weight'):
        self.factors = factors
        self.returns = returns
        self.combination_method = combination_method
        self.combined_factor = None
        self.positions = None
        self.factor_weights = None
        
    def combine_factors(self):
        """根据不同方法组合因子"""
        if self.combination_method == 'equal_weight':
            self._equal_weight_combination()
        elif self.combination_method == 'ic_weight':
            self._ic_weight_combination()
        elif self.combination_method == 'risk_parity':
            self._risk_parity_combination()
        elif self.combination_method == 'dynamic_weight':
            self._dynamic_weight_combination()
        elif self.combination_method == 'ml_weight':
            self._ml_weight_combination()
        else:
            raise ValueError(f"未知的组合方法: {self.combination_method}")
            
    def _equal_weight_combination(self):
        """等权重组合"""
        self.factor_weights = {name: 0.2 for name in self.factors.keys()}  # 每个因子权重0.2
        
        factor_df = pd.DataFrame(self.factors)
        self.combined_factor = factor_df.mean(axis=1)
        print("使用等权重方法组合因子")
        
    def _ic_weight_combination(self, window=60):
        """基于IC的权重组合"""
        factor_df = pd.DataFrame(self.factors)
        self.combined_factor = pd.Series(0, index=factor_df.index)
        self.factor_weights = pd.DataFrame(index=factor_df.index, columns=factor_df.columns)
        
        # 计算滚动IC权重
        for i in range(window, len(factor_df)):
            window_data = factor_df.iloc[i-window:i]
            window_returns = self.returns.iloc[i-window:i]
            
            # 计算各因子与收益的相关性（IC）
            ics = []
            for col in window_data.columns:
                ic = window_data[col].corr(window_returns)
                ics.append(abs(ic) if not np.isnan(ic) else 0)
            
            # 权重标准化
            total_ic = sum(ics) if sum(ics) > 0 else 1
            weights = [ic / total_ic for ic in ics]
            
            # 存储权重
            self.factor_weights.iloc[i] = weights
            
            # 计算加权因子
            weighted_factor = sum(window_data.iloc[-1, j] * weights[j] 
                                for j in range(len(weights)))
            self.combined_factor.iloc[i] = weighted_factor
        
        # 填充前面的值
        self.combined_factor.iloc[:window] = 0
        self.factor_weights.iloc[:window] = 0.2  # 前期使用等权重
        
        print("使用IC加权方法组合因子")
        
    def _risk_parity_combination(self, window=60):
        """风险平价组合"""
        factor_df = pd.DataFrame(self.factors)
        self.combined_factor = pd.Series(0, index=factor_df.index)
        self.factor_weights = pd.DataFrame(index=factor_df.index, columns=factor_df.columns)
        
        for i in range(window, len(factor_df)):
            window_data = factor_df.iloc[i-window:i]
            
            # 计算各因子风险（标准差）
            risks = []
            for col in window_data.columns:
                risk = window_data[col].std()
                risks.append(1/risk if risk > 0 else 1)
            
            # 权重标准化（风险反比）
            total_inv_risk = sum(risks)
            weights = [risk / total_inv_risk for risk in risks]
            
            # 存储权重
            self.factor_weights.iloc[i] = weights
            
            # 计算加权因子
            weighted_factor = sum(factor_df.iloc[i, j] * weights[j] 
                                for j in range(len(weights)))
            self.combined_factor.iloc[i] = weighted_factor
        
        # 填充前面的值
        self.combined_factor.iloc[:window] = 0
        self.factor_weights.iloc[:window] = 0.2
        
        print("使用风险平价方法组合因子")
        
    def _dynamic_weight_combination(self, window=60):
        """动态权重组合 - 基于滚动表现（修正版）"""
        factor_df = pd.DataFrame(self.factors)
        self.combined_factor = pd.Series(0, index=factor_df.index)
        self.factor_weights = pd.DataFrame(index=factor_df.index, columns=factor_df.columns)
        
        for i in range(window, len(factor_df)):
            window_data = factor_df.iloc[i-window:i]
            window_returns = self.returns.iloc[i-window:i]
            
            performances = []
            for col in window_data.columns:
                factor_series = window_data[col]
                
                # 1. 相关性表现（IC）
                ic = abs(factor_series.corr(window_returns))
                ic_score = ic if not np.isnan(ic) else 0
                
                # 2. 稳定性表现（改进计算方式）
                factor_std = factor_series.std()
                factor_mean = factor_series.mean()
                # 避免除零，用变异系数的倒数
                if factor_std > 0 and abs(factor_mean) > 1e-6:
                    stability_score = 1 / (1 + abs(factor_std / factor_mean))
                else:
                    stability_score = 0.5  # 默认中等稳定性
                
                # 3. 方向一致性（检查因子与收益的方向一致性）
                factor_sign = np.sign(factor_series)
                return_sign = np.sign(window_returns)
                direction_consistency = (factor_sign == return_sign).mean()
                
                # 综合表现评分（调整权重分配）
                performance = (0.5 * ic_score +           # IC权重50%
                             0.3 * stability_score +      # 稳定性权重30%
                             0.2 * direction_consistency)  # 方向一致性权重20%
                
                performances.append(max(performance, 0.01))  # 确保最小权重
            
            # 权重标准化
            total_perf = sum(performances)
            weights = [perf / total_perf for perf in performances]
            
            # 权重平滑化（避免权重剧烈变化）
            if i > window:
                prev_weights = self.factor_weights.iloc[i-1].values
                smooth_factor = 0.7  # 平滑系数
                weights = [smooth_factor * w + (1-smooth_factor) * pw 
                          for w, pw in zip(weights, prev_weights)]
                # 重新标准化
                weights = [w / sum(weights) for w in weights]
            
            # 存储权重
            self.factor_weights.iloc[i] = weights
            
            # 计算加权因子
            weighted_factor = sum(factor_df.iloc[i, j] * weights[j] 
                                for j in range(len(weights)))
            self.combined_factor.iloc[i] = weighted_factor
        
        # 填充前面的值
        self.combined_factor.iloc[:window] = 0
        self.factor_weights.iloc[:window] = 0.2
        
        print("使用改进的动态权重方法组合因子")
        
    def _ml_weight_combination(self):
        """机器学习权重组合 - 简化版线性回归"""
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score
        
        factor_df = pd.DataFrame(self.factors).fillna(0)
        returns_clean = self.returns.fillna(0)
        
        # 使用线性回归拟合因子权重
        X = factor_df.values
        y = returns_clean.values
        
        # 滚动训练
        window = 120
        self.combined_factor = pd.Series(0, index=factor_df.index)
        self.factor_weights = pd.DataFrame(index=factor_df.index, columns=factor_df.columns)
        
        for i in range(window, len(factor_df)):
            # 训练数据
            X_train = X[i-window:i]
            y_train = y[i-window:i]
            
            # 训练模型
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # 权重归一化
            weights = model.coef_
            abs_weights = np.abs(weights)
            weights_normalized = abs_weights / (abs_weights.sum() + 1e-8)
            
            # 存储权重
            self.factor_weights.iloc[i] = weights_normalized
            
            # 计算加权因子
            weighted_factor = np.dot(factor_df.iloc[i].values, weights_normalized)
            self.combined_factor.iloc[i] = weighted_factor
        
        # 填充前面的值
        self.combined_factor.iloc[:window] = 0
        self.factor_weights.iloc[:window] = 0.2
        
        print("使用机器学习权重方法组合因子")
        
    def generate_signals(self, threshold=0.5):
        """生成交易信号"""
        if self.combined_factor is None:
            raise ValueError("请先调用combine_factors()方法")
            
        # 多层信号生成
        conditions = [
            self.combined_factor > threshold,      # 强看多
            self.combined_factor > threshold * 0.5, # 弱看多
            self.combined_factor < -threshold,     # 强看空
            self.combined_factor < -threshold * 0.5, # 弱看空
        ]
        choices = [1, 0.5, -1, -0.5]  # 满仓、半仓、满仓空、半仓空
        
        self.positions = np.select(conditions, choices, default=0)
        self.positions = pd.Series(self.positions, index=self.combined_factor.index).shift(1).fillna(0)
        
        return self.positions

class PerformanceAnalyzer:
    """性能分析器"""
    def __init__(self, returns, positions, factors=None, factor_weights=None, combined_factor=None):
        self.returns = returns
        self.positions = positions
        self.factors = factors
        self.factor_weights = factor_weights
        self.combined_factor = combined_factor
        self._calculate_performance()
    
    def _calculate_performance(self):
        """计算策略表现"""
        # 策略收益
        self.strategy_returns = self.returns * self.positions
        self.cumulative_returns = (1 + self.strategy_returns).cumprod()
        self.benchmark_returns = (1 + self.returns).cumprod()
        
    def calculate_metrics(self):
        """计算性能指标（增强版）"""
        returns_clean = self.strategy_returns.dropna()
        
        if len(returns_clean) == 0:
            return {}
        
        # 基础指标
        annual_return = returns_clean.mean() * 252
        annual_volatility = returns_clean.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # 最大回撤和回撤持续时间
        cumulative = self.cumulative_returns
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max - 1)
        max_dd = drawdown.min()
        
        # 回撤持续时间
        dd_duration = 0
        current_dd_duration = 0
        for dd in drawdown:
            if dd < 0:
                current_dd_duration += 1
                dd_duration = max(dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0
        
        # 胜率和盈亏比
        win_rate = (returns_clean > 0).sum() / len(returns_clean)
        profit_trades = returns_clean[returns_clean > 0]
        loss_trades = returns_clean[returns_clean < 0]
        
        profit_loss_ratio = (profit_trades.mean() / abs(loss_trades.mean()) 
                           if len(loss_trades) > 0 else np.inf)
        
        # 信息比率
        excess_returns = self.strategy_returns - self.returns
        information_ratio = (excess_returns.mean() / excess_returns.std() * np.sqrt(252) 
                           if excess_returns.std() != 0 else 0)
        
        # 卡尔马比率
        calmar_ratio = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        # Sortino比率（只考虑下行波动）
        downside_returns = returns_clean[returns_clean < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annual_return / downside_volatility if downside_volatility != 0 else 0
        
        # 换手率
        position_changes = np.abs(np.diff(self.positions))
        turnover = position_changes.sum() / len(position_changes) if len(position_changes) > 0 else 0
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_dd,
            'max_dd_duration': dd_duration,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'information_ratio': information_ratio,
            'calmar_ratio': calmar_ratio,
            'turnover': turnover,
            'total_trades': np.sum(np.abs(np.diff(self.positions)))
        }
    
    def plot_results(self, combination_method, save_path=None):
        """绘制结果图表（增强版）"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        
        # 1. 累积收益对比
        axes[0, 0].plot(self.cumulative_returns.index, self.cumulative_returns.values, 
                       label=f'5因子组合策略({combination_method})', linewidth=2)
        axes[0, 0].plot(self.benchmark_returns.index, self.benchmark_returns.values, 
                       label='基准(买入持有)', linewidth=2)
        axes[0, 0].set_title('累积收益率对比')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 组合因子时序
        if self.combined_factor is not None:
            axes[0, 1].plot(self.combined_factor.index, self.combined_factor.values, 
                           label='组合因子', color='purple', linewidth=1.5)
            axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[0, 1].set_title('组合因子时序')
            axes[0, 1].legend()
        else:
            axes[0, 1].text(0.5, 0.5, '组合因子数据不可用', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('组合因子时序')
        axes[0, 1].grid(True)
        
        # 3. 仓位变化
        axes[1, 0].plot(self.positions.index, self.positions.values, 
                       label='仓位', color='orange', alpha=0.7)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].set_title('仓位变化')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. 因子权重变化
        if self.factor_weights is not None and isinstance(self.factor_weights, pd.DataFrame) and not self.factor_weights.empty:
            # 显示所有5个因子的权重变化
            for i, col in enumerate(self.factor_weights.columns[:5]):
                if col in self.factor_weights.columns:
                    weight_data = self.factor_weights[col].dropna()
                    if len(weight_data) > 0:
                        axes[1, 1].plot(weight_data.index, weight_data.values, 
                                       label=col, alpha=0.8)
            axes[1, 1].set_title('因子权重变化')
            axes[1, 1].legend(fontsize=9)
        elif isinstance(self.factor_weights, dict):
            # 如果是字典格式（等权重情况）
            factor_names = list(self.factor_weights.keys())
            weights = list(self.factor_weights.values())
            axes[1, 1].bar(factor_names, weights, alpha=0.7, color='skyblue')
            axes[1, 1].set_title('因子权重分布')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, '因子权重数据不可用', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('因子权重变化')
        axes[1, 1].grid(True)
        
        # 5. 收益分布
        strategy_returns_clean = self.strategy_returns.dropna()
        if len(strategy_returns_clean) > 0:
            axes[2, 0].hist(strategy_returns_clean, bins=50, alpha=0.7, color='skyblue')
            axes[2, 0].axvline(x=strategy_returns_clean.mean(), color='red', linestyle='--', 
                              label=f'均值: {strategy_returns_clean.mean():.4f}')
            axes[2, 0].set_title('策略日收益分布')
            axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # 6. 回撤分析
        running_max = self.cumulative_returns.expanding().max()
        drawdown = (self.cumulative_returns / running_max - 1) * 100
        axes[2, 1].fill_between(drawdown.index, drawdown.values, 0, 
                               color='red', alpha=0.3, label='回撤')
        axes[2, 1].set_title('回撤分析 (%)')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表保存至: {save_path}")
        
        plt.close()  # 关闭图表释放内存

def run_multi_factor_5_strategy(data_path, combination_methods=None):
    """运行5因子组合策略的主函数"""
    if combination_methods is None:
        combination_methods = ['equal_weight', 'ic_weight', 'risk_parity', 'dynamic_weight']
    
    print("=== 5因子组合策略回测开始 ===")
    
    # 1. 加载和处理因子数据
    processor = FactorDataProcessor(data_path)
    processor.load_and_calculate_factors()
    processor.standardize_factors()
    
    results = {}
    
    # 2. 测试不同的组合方法
    for method in combination_methods:
        print(f"\n--- 测试{method}组合方法 ---")
        
        # 创建策略
        strategy = MultiFactor5Strategy(processor.factors, processor.returns, method)
        strategy.combine_factors()
        strategy.generate_signals()
        
        # 性能分析
        analyzer = PerformanceAnalyzer(processor.returns, strategy.positions, 
                                     processor.factors, strategy.factor_weights, strategy.combined_factor)
        metrics = analyzer.calculate_metrics()
        
        # 保存结果
        results[method] = {
            'metrics': metrics,
            'strategy': strategy,
            'analyzer': analyzer
        }
        
        # 打印关键指标
        print(f"年化收益率: {metrics['annual_return']:.4f}")
        print(f"年化波动率: {metrics['annual_volatility']:.4f}")
        print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
        print(f"Sortino比率: {metrics['sortino_ratio']:.4f}")
        print(f"最大回撤: {metrics['max_drawdown']:.4f}")
        print(f"胜率: {metrics['win_rate']:.4f}")
        print(f"信息比率: {metrics['information_ratio']:.4f}")
        print(f"卡尔马比率: {metrics['calmar_ratio']:.4f}")
        print(f"换手率: {metrics['turnover']:.4f}")
        print(f"总交易次数: {metrics['total_trades']:.0f}")
        
        # 绘制结果
        os.makedirs('./results/backtest_results', exist_ok=True)
        save_path = f'./results/backtest_results/multi_factor_5_{method}_results.png'
        analyzer.plot_results(method, save_path)
    
    # 3. 结果对比
    print("\n=== 各方法性能对比 ===")
    comparison_df = pd.DataFrame({method: result['metrics'] 
                                for method, result in results.items()}).T
    print(comparison_df)
    
    # 保存对比结果
    comparison_df.to_csv('./results/backtest_results/multi_factor_5_comparison.csv')
    
    print("=== 5因子组合策略回测完成 ===")
    return results

def main():
    """主函数"""
    # 数据路径
    data_path = 'data/data_202410.pkl'
    
    try:
        # 运行5因子组合策略
        results = run_multi_factor_5_strategy(data_path)
        
        # 选择最佳方法
        best_method = max(results.keys(), 
                         key=lambda x: results[x]['metrics']['sharpe_ratio'])
        
        print(f"\n最佳组合方法: {best_method}")
        print(f"最佳夏普比率: {results[best_method]['metrics']['sharpe_ratio']:.4f}")
        
    except Exception as e:
        print(f"策略运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 