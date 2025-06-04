"""
多因子组合策略 - 10因子组合
基于前10个最优因子的多种权重组合策略，包括等权重、IC加权、风险平价等方法
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
from factor_strategy_6 import PriceVolumeDivergenceStrategy, DataHandler as DataHandler6
from factor_strategy_7 import VolumeAnomalyStrategy, DataHandler as DataHandler7
from factor_strategy_8 import TurnoverStrategy, DataHandler as DataHandler8
from factor_strategy_9 import VolatilityStrategy, DataHandler as DataHandler9
from factor_strategy_10 import VWAPStrategy, DataHandler as DataHandler10

class FactorDataProcessor:
    """因子数据处理器 - 统一处理多个因子的计算和标准化"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.factors = {}
        self.returns = None
        
    def load_and_calculate_factors(self):
        """加载数据并计算前10个因子"""
        print("正在计算前10个因子...")
        
        # 1-5 因子（与5因子组合相同）
        data_handler1 = DataHandler1(self.data_path)
        data_handler1.calculate_rsi()
        self.factors['RSI_Reversal'] = data_handler1.data['RSI_Reversal_Factor'].fillna(0)
        
        data_handler2 = DataHandler2(self.data_path)
        data_handler2.calculate_macd()
        self.factors['MACD_Momentum'] = data_handler2.data['MACD_Momentum_Factor'].fillna(0)
        
        data_handler3 = DataHandler3(self.data_path)
        data_handler3.calculate_bollinger_bands()
        self.factors['Bollinger_Position'] = data_handler3.data['Bollinger_Position_Factor'].fillna(0)
        
        data_handler4 = DataHandler4(self.data_path)
        data_handler4.calculate_williams_r()
        self.factors['Williams_Reversal'] = data_handler4.data['Williams_Reversal_Factor'].fillna(0)
        
        data_handler5 = DataHandler5(self.data_path)
        data_handler5.calculate_cci_factor()
        self.factors['CCI'] = data_handler5.data['CCI_Factor'].fillna(0)
        
        # 6-10 因子（新增）
        data_handler6 = DataHandler6(self.data_path)
        data_handler6.calculate_price_volume_divergence_factor()
        self.factors['Price_Volume_Divergence'] = data_handler6.data['Price_Volume_Divergence_Factor'].fillna(0)
        
        data_handler7 = DataHandler7(self.data_path)
        data_handler7.calculate_volume_anomaly_factor()
        self.factors['Volume_Anomaly'] = data_handler7.data['Volume_Anomaly_Factor'].fillna(0)
        
        data_handler8 = DataHandler8(self.data_path)
        data_handler8.calculate_turnover_factor()
        self.factors['Turnover'] = data_handler8.data['Turnover_Factor'].fillna(0)
        
        data_handler9 = DataHandler9(self.data_path)
        data_handler9.calculate_volatility_factor()
        self.factors['Volatility'] = data_handler9.data['Volatility_Factor'].fillna(0)
        
        data_handler10 = DataHandler10(self.data_path)
        data_handler10.calculate_vwap_factor()
        self.factors['VWAP'] = data_handler10.data['VWAP_Factor'].fillna(0)
        
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
        
    def calculate_factor_correlation(self):
        """计算因子相关性矩阵"""
        factor_df = pd.DataFrame(self.factors)
        correlation_matrix = factor_df.corr()
        
        print("因子相关性矩阵:")
        print(correlation_matrix)
        
        # 找出高相关性的因子对
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = abs(correlation_matrix.iloc[i, j])
                if corr_val > 0.7:  # 高相关性阈值
                    high_corr_pairs.append((correlation_matrix.columns[i], 
                                          correlation_matrix.columns[j], 
                                          corr_val))
        
        if high_corr_pairs:
            print("\n发现高相关性因子对 (|相关系数| > 0.7):")
            for pair in high_corr_pairs:
                print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")
        
        return correlation_matrix

class MultiFactor10Strategy:
    """10因子组合策略"""
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
        elif self.combination_method == 'pca_weight':
            self._pca_weight_combination()
        else:
            raise ValueError(f"未知的组合方法: {self.combination_method}")
            
    def _equal_weight_combination(self):
        """等权重组合"""
        self.factor_weights = {name: 0.1 for name in self.factors.keys()}  # 每个因子权重0.1
        
        factor_df = pd.DataFrame(self.factors)
        self.combined_factor = factor_df.mean(axis=1)
        print("使用等权重方法组合因子")
        
    def _ic_weight_combination(self, window=60):
        """基于IC的权重组合（改进版）"""
        factor_df = pd.DataFrame(self.factors)
        self.combined_factor = pd.Series(0, index=factor_df.index)
        self.factor_weights = pd.DataFrame(index=factor_df.index, columns=factor_df.columns)
        
        # 计算滚动IC权重
        for i in range(window, len(factor_df)):
            window_data = factor_df.iloc[i-window:i]
            window_returns = self.returns.iloc[i-window:i]
            
            # 计算各因子的滚动IC均值和稳定性
            ics = []
            ic_stabilities = []
            
            for col in window_data.columns:
                # 计算子窗口IC
                sub_window = 20
                sub_ics = []
                for j in range(sub_window, window, 5):  # 每5天计算一次IC
                    sub_factor = window_data[col].iloc[j-sub_window:j]
                    sub_return = window_returns.iloc[j-sub_window:j]
                    sub_ic = sub_factor.corr(sub_return)
                    if not np.isnan(sub_ic):
                        sub_ics.append(sub_ic)
                
                if sub_ics:
                    ic_mean = np.mean([abs(ic) for ic in sub_ics])
                    ic_std = np.std(sub_ics)
                    ic_stability = ic_mean / (ic_std + 1e-8)  # IC的信噪比
                else:
                    ic_mean = 0
                    ic_stability = 0
                    
                ics.append(ic_mean)
                ic_stabilities.append(ic_stability)
            
            # 综合考虑IC均值和稳定性
            combined_scores = [ic * (1 + stability) for ic, stability in zip(ics, ic_stabilities)]
            
            # 权重标准化
            total_score = sum(combined_scores) if sum(combined_scores) > 0 else 1
            weights = [score / total_score for score in combined_scores]
            
            # 存储权重
            self.factor_weights.iloc[i] = weights
            
            # 计算加权因子
            weighted_factor = sum(factor_df.iloc[i, j] * weights[j] 
                                for j in range(len(weights)))
            self.combined_factor.iloc[i] = weighted_factor
        
        # 填充前面的值
        self.combined_factor.iloc[:window] = 0
        self.factor_weights.iloc[:window] = 0.1  # 前期使用等权重
        
        print("使用改进IC加权方法组合因子")
        
    def _risk_parity_combination(self, window=60):
        """风险平价组合（改进版）"""
        factor_df = pd.DataFrame(self.factors)
        self.combined_factor = pd.Series(0, index=factor_df.index)
        self.factor_weights = pd.DataFrame(index=factor_df.index, columns=factor_df.columns)
        
        for i in range(window, len(factor_df)):
            window_data = factor_df.iloc[i-window:i]
            
            # 计算各因子的风险贡献度（考虑相关性）
            cov_matrix = window_data.cov()
            volatilities = window_data.std()
            
            # 使用风险平价算法计算权重
            num_factors = len(window_data.columns)
            weights = np.ones(num_factors) / num_factors  # 初始等权重
            
            # 迭代优化风险平价权重
            for iteration in range(50):  # 最多迭代50次
                # 计算风险贡献度
                portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
                marginal_contrib = np.dot(cov_matrix, weights)
                contrib = weights * marginal_contrib / portfolio_var
                
                # 计算目标风险贡献度（等权重）
                target_contrib = 1.0 / num_factors
                
                # 更新权重
                for j in range(num_factors):
                    if contrib[j] > 0:
                        weights[j] *= (target_contrib / contrib[j]) ** 0.1
                
                # 标准化权重
                weights = weights / weights.sum()
                
                # 检查收敛
                if np.max(np.abs(contrib - target_contrib)) < 1e-6:
                    break
            
            # 存储权重
            self.factor_weights.iloc[i] = weights
            
            # 计算加权因子
            weighted_factor = np.dot(factor_df.iloc[i].values, weights)
            self.combined_factor.iloc[i] = weighted_factor
        
        # 填充前面的值
        self.combined_factor.iloc[:window] = 0
        self.factor_weights.iloc[:window] = 0.1
        
        print("使用改进风险平价方法组合因子")
        
    def _dynamic_weight_combination(self, window=60):
        """动态权重组合 - 基于多维度表现评估"""
        factor_df = pd.DataFrame(self.factors)
        self.combined_factor = pd.Series(0, index=factor_df.index)
        self.factor_weights = pd.DataFrame(index=factor_df.index, columns=factor_df.columns)
        
        for i in range(window, len(factor_df)):
            window_data = factor_df.iloc[i-window:i]
            window_returns = self.returns.iloc[i-window:i]
            
            performances = []
            for col in window_data.columns:
                factor_series = window_data[col]
                
                # 1. 相关性表现
                corr = abs(factor_series.corr(window_returns))
                corr = corr if not np.isnan(corr) else 0
                
                # 2. 稳定性表现
                stability = 1 - (factor_series.std() / (abs(factor_series.mean()) + 1e-8))
                
                # 3. 单调性表现（检查因子的方向一致性）
                sign_consistency = abs(factor_series.diff().fillna(0).apply(np.sign).mean())
                
                # 4. 预测能力（延迟相关性）
                if len(factor_series) > 1 and len(window_returns) > 1:
                    future_corr = abs(factor_series[:-1].corr(window_returns[1:]))
                    future_corr = future_corr if not np.isnan(future_corr) else 0
                else:
                    future_corr = 0
                
                # 综合表现得分
                performance = (0.4 * corr + 
                             0.3 * max(stability, 0) + 
                             0.2 * sign_consistency + 
                             0.1 * future_corr)
                performances.append(performance)
            
            # 权重标准化
            total_perf = sum(performances) if sum(performances) > 0 else 1
            weights = [perf / total_perf for perf in performances]
            
            # 权重平滑化（避免权重剧烈变化）
            if i > window:
                prev_weights = self.factor_weights.iloc[i-1].values
                smooth_factor = 0.8
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
        self.factor_weights.iloc[:window] = 0.1
        
        print("使用多维度动态权重方法组合因子")
        
    def _ml_weight_combination(self):
        """机器学习权重组合 - 使用多种算法集成"""
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor
        
        factor_df = pd.DataFrame(self.factors).fillna(0)
        returns_clean = self.returns.fillna(0)
        
        X = factor_df.values
        y = returns_clean.values
        
        window = 120
        self.combined_factor = pd.Series(0, index=factor_df.index)
        self.factor_weights = pd.DataFrame(index=factor_df.index, columns=factor_df.columns)
        
        for i in range(window, len(factor_df)):
            X_train = X[i-window:i]
            y_train = y[i-window:i]
            
            # 多个模型的权重集成
            models = {
                'linear': LinearRegression(),
                'ridge': Ridge(alpha=0.1),
                'lasso': Lasso(alpha=0.01),
            }
            
            model_weights = []
            model_scores = []
            
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    score = model.score(X_train, y_train)
                    weights = getattr(model, 'coef_', np.zeros(len(factor_df.columns)))
                    
                    model_weights.append(np.abs(weights))
                    model_scores.append(max(score, 0))
                except:
                    model_weights.append(np.ones(len(factor_df.columns)))
                    model_scores.append(0.01)
            
            # 基于模型表现加权平均
            total_score = sum(model_scores) if sum(model_scores) > 0 else 1
            final_weights = np.zeros(len(factor_df.columns))
            
            for weights, score in zip(model_weights, model_scores):
                final_weights += weights * (score / total_score)
            
            # 权重归一化
            weights_normalized = final_weights / (final_weights.sum() + 1e-8)
            
            # 存储权重
            self.factor_weights.iloc[i] = weights_normalized
            
            # 计算加权因子
            weighted_factor = np.dot(factor_df.iloc[i].values, weights_normalized)
            self.combined_factor.iloc[i] = weighted_factor
        
        # 填充前面的值
        self.combined_factor.iloc[:window] = 0
        self.factor_weights.iloc[:window] = 0.1
        
        print("使用机器学习集成权重方法组合因子")
        
    def _pca_weight_combination(self, n_components=5):
        """PCA降维权重组合"""
        from sklearn.decomposition import PCA
        
        factor_df = pd.DataFrame(self.factors).fillna(0)
        
        # 使用PCA提取主成分
        pca = PCA(n_components=n_components)
        
        window = 120
        self.combined_factor = pd.Series(0, index=factor_df.index)
        self.factor_weights = pd.DataFrame(index=factor_df.index, columns=factor_df.columns)
        
        for i in range(window, len(factor_df)):
            window_data = factor_df.iloc[i-window:i].values
            
            # 拟合PCA
            pca.fit(window_data)
            
            # 获取第一主成分的权重（解释方差最大的成分）
            first_component = pca.components_[0]
            
            # 考虑解释方差比例调整权重
            explained_variance_ratio = pca.explained_variance_ratio_[0]
            
            # 权重归一化
            weights = np.abs(first_component)
            weights = weights / weights.sum()
            
            # 存储权重
            self.factor_weights.iloc[i] = weights
            
            # 计算加权因子
            weighted_factor = np.dot(factor_df.iloc[i].values, weights)
            self.combined_factor.iloc[i] = weighted_factor
        
        # 填充前面的值
        self.combined_factor.iloc[:window] = 0
        self.factor_weights.iloc[:window] = 0.1
        
        print(f"使用PCA降维方法组合因子（保留{n_components}个主成分）")
        
    def generate_signals(self, threshold=0.5):
        """生成交易信号"""
        if self.combined_factor is None:
            raise ValueError("请先调用combine_factors()方法")
            
        # 分层信号生成（更细粒度）
        conditions = [
            self.combined_factor > threshold * 1.2,      # 强看多
            self.combined_factor > threshold,            # 中等看多
            self.combined_factor > threshold * 0.5,      # 弱看多
            self.combined_factor < -threshold * 1.2,     # 强看空
            self.combined_factor < -threshold,           # 中等看空
            self.combined_factor < -threshold * 0.5,     # 弱看空
        ]
        choices = [1, 0.8, 0.4, -1, -0.8, -0.4]  # 对应仓位
        
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
        """绘制结果图表"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(3, 2, figsize=(18, 18))
        
        # 1. 累积收益对比
        axes[0, 0].plot(self.cumulative_returns.index, self.cumulative_returns.values, 
                       label=f'10因子组合策略({combination_method})', linewidth=2)
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
            # 只显示前10个因子的权重变化
            for i, col in enumerate(self.factor_weights.columns[:10]):
                if col in self.factor_weights.columns:
                    weight_data = self.factor_weights[col].dropna()
                    if len(weight_data) > 0:
                        axes[1, 1].plot(weight_data.index, weight_data.values, 
                                       label=col, alpha=0.8)
            axes[1, 1].set_title('因子权重变化')
            axes[1, 1].legend(fontsize=8)
        elif isinstance(self.factor_weights, dict):
            # 如果是字典格式（等权重情况）
            factor_names = list(self.factor_weights.keys())
            weights = list(self.factor_weights.values())
            axes[1, 1].bar(factor_names, weights, alpha=0.7)
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
        
        # 6. 月度收益分布（替换为简单的分组）
        period_size = 21  # 大约一个月的交易日
        period_returns = []
        strategy_returns_clean = self.strategy_returns.dropna()
        
        for i in range(0, len(strategy_returns_clean), period_size):
            period_data = strategy_returns_clean.iloc[i:i+period_size]
            if len(period_data) > 0:
                period_returns.append(period_data.sum())
        
        if period_returns:
            axes[2, 1].hist(period_returns, bins=min(20, len(period_returns)), alpha=0.7, color='lightgreen')
            mean_return = np.mean(period_returns)
            axes[2, 1].axvline(x=mean_return, color='red', linestyle='--', 
                              label=f'均值: {mean_return:.4f}')
            axes[2, 1].set_title('月度收益分布')
            axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表保存至: {save_path}")
        
        plt.close()  # 关闭图表释放内存

def run_multi_factor_10_strategy(data_path, combination_methods=None):
    """运行10因子组合策略的主函数"""
    if combination_methods is None:
        combination_methods = ['equal_weight', 'ic_weight', 'risk_parity', 
                             'dynamic_weight', 'ml_weight', 'pca_weight']
    
    print("=== 10因子组合策略回测开始 ===")
    
    # 1. 加载和处理因子数据
    processor = FactorDataProcessor(data_path)
    processor.load_and_calculate_factors()
    processor.standardize_factors()
    
    # 分析因子相关性
    correlation_matrix = processor.calculate_factor_correlation()
    
    results = {}
    
    # 2. 测试不同的组合方法
    for method in combination_methods:
        print(f"\n--- 测试{method}组合方法 ---")
        
        try:
            # 创建策略
            strategy = MultiFactor10Strategy(processor.factors, processor.returns, method)
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
            print(f"换手率: {metrics['turnover']:.4f}")
            
            # 绘制结果
            os.makedirs('./results/backtest_results', exist_ok=True)
            save_path = f'./results/backtest_results/multi_factor_10_{method}_results.png'
            analyzer.plot_results(method, save_path)
            
        except Exception as e:
            print(f"方法 {method} 运行失败: {e}")
            continue
    
    # 3. 结果对比
    if results:
        print("\n=== 各方法性能对比 ===")
        comparison_df = pd.DataFrame({method: result['metrics'] 
                                    for method, result in results.items()}).T
        print(comparison_df)
        
        # 保存对比结果
        comparison_df.to_csv('./results/backtest_results/multi_factor_10_comparison.csv')
        
        # 保存因子相关性矩阵
        correlation_matrix.to_csv('./results/backtest_results/factor_correlation_matrix_10.csv')
    
    print("=== 10因子组合策略回测完成 ===")
    return results

def main():
    """主函数"""
    # 数据路径
    data_path = 'data/data_202410.pkl'
    
    try:
        # 运行10因子组合策略
        results = run_multi_factor_10_strategy(data_path)
        
        if results:
            # 选择最佳方法
            best_method = max(results.keys(), 
                             key=lambda x: results[x]['metrics']['sharpe_ratio'])
            
            print(f"\n最佳组合方法: {best_method}")
            print(f"最佳夏普比率: {results[best_method]['metrics']['sharpe_ratio']:.4f}")
            
            # 显示各方法排名
            print("\n各方法夏普比率排名:")
            sorted_methods = sorted(results.items(), 
                                  key=lambda x: x[1]['metrics']['sharpe_ratio'], 
                                  reverse=True)
            for i, (method, result) in enumerate(sorted_methods, 1):
                print(f"{i}. {method}: {result['metrics']['sharpe_ratio']:.4f}")
        
    except Exception as e:
        print(f"策略运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 