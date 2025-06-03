"""
多因子组合策略 - 20因子组合
基于前20个因子的大规模多因子组合策略，采用最先进的权重分配和风险控制方法
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 添加因子策略路径
sys.path.append('../factors')

# 导入基础因子处理器
from factor_strategy_1 import DataHandler as DataHandler1
from factor_strategy_2 import DataHandler as DataHandler2
from factor_strategy_3 import DataHandler as DataHandler3
from factor_strategy_4 import DataHandler as DataHandler4
from factor_strategy_5 import DataHandler as DataHandler5
from factor_strategy_6 import DataHandler as DataHandler6
from factor_strategy_7 import DataHandler as DataHandler7
from factor_strategy_8 import DataHandler as DataHandler8
from factor_strategy_9 import DataHandler as DataHandler9
from factor_strategy_10 import DataHandler as DataHandler10

class EnhancedFactorHandlers:
    """增强因子处理器 - 为20因子组合创建更多高级因子"""
    
    @staticmethod
    def create_advanced_factors(data):
        """创建10个高级因子"""
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        returns = close.pct_change()
        
        factors = {}
        
        # 11. 动量反转因子
        momentum_5d = close.pct_change(5)
        momentum_20d = close.pct_change(20)
        factors['Momentum_Reversal'] = -0.6 * momentum_5d + 0.4 * momentum_20d
        
        # 12. 量价背离因子
        price_trend = (close.rolling(5).mean() / close.rolling(20).mean() - 1)
        volume_trend = (volume.rolling(5).mean() / volume.rolling(20).mean() - 1)
        factors['Volume_Price_Divergence'] = np.sign(price_trend) - np.sign(volume_trend)
        
        # 13. 波动率偏度因子
        volatility = returns.rolling(20).std()
        skewness = returns.rolling(20).skew()
        factors['Volatility_Skew'] = volatility * skewness
        
        # 14. 高低价位置因子
        high_20 = high.rolling(20).max()
        low_20 = low.rolling(20).min()
        factors['Price_Position'] = (close - low_20) / (high_20 - low_20 + 1e-8) - 0.5
        
        # 15. 换手率异常因子
        turnover = volume / volume.rolling(60).mean()
        factors['Turnover_Anomaly'] = (turnover - turnover.rolling(20).mean()) / turnover.rolling(20).std()
        
        # 16. 跳空因子
        gap = (close - close.shift(1)) / close.shift(1)
        factors['Gap_Factor'] = gap.rolling(10).sum()
        
        # 17. 价格加速度因子
        acceleration = returns.diff()
        factors['Price_Acceleration'] = acceleration.rolling(5).mean()
        
        # 18. 成交量能量因子
        volume_energy = volume * abs(returns)
        factors['Volume_Energy'] = (volume_energy / volume_energy.rolling(20).mean() - 1)
        
        # 19. 趋势强度因子
        trend_strength = abs(close.rolling(20).corr(pd.Series(range(20), index=close.index[-20:])))
        factors['Trend_Strength'] = trend_strength
        
        # 20. 市场微观结构因子
        microstructure = (high + low + 2 * close) / 4 - close
        factors['Microstructure'] = microstructure.rolling(10).mean()
        
        # 填充缺失值
        for name, factor in factors.items():
            factors[name] = factor.fillna(0)
        
        return factors

class FactorDataProcessor:
    """20因子数据处理器"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.factors = {}
        self.returns = None
        self.raw_data = None
        
    def load_and_calculate_factors(self):
        """加载数据并计算20个因子"""
        print("正在计算20个因子...")
        
        # 基础10个因子
        handlers = [DataHandler1, DataHandler2, DataHandler3, DataHandler4, DataHandler5,
                   DataHandler6, DataHandler7, DataHandler8, DataHandler9, DataHandler10]
        
        factor_methods = ['calculate_rsi', 'calculate_macd', 'calculate_bollinger_bands',
                         'calculate_williams_r', 'calculate_cci_factor', 'calculate_price_volume_divergence_factor',
                         'calculate_volume_anomaly_factor', 'calculate_turnover_factor', 
                         'calculate_volatility_factor', 'calculate_vwap_factor']
        
        factor_names = ['RSI_Reversal_Factor', 'MACD_Momentum_Factor', 'Bollinger_Position_Factor',
                       'Williams_Reversal_Factor', 'CCI_Factor', 'Price_Volume_Divergence_Factor',
                       'Volume_Anomaly_Factor', 'Turnover_Factor', 'Volatility_Factor',
                       'VWAP_Factor']
        
        factor_labels = ['RSI_Reversal', 'MACD_Momentum', 'Bollinger_Position', 'Williams_Reversal',
                        'CCI', 'Price_Volume_Divergence', 'Volume_Anomaly', 'Turnover', 'Volatility',
                        'VWAP']
        
        # 计算基础因子
        for i, (handler_class, method, factor_name, label) in enumerate(zip(handlers, factor_methods, factor_names, factor_labels)):
            try:
                handler = handler_class(self.data_path)
                getattr(handler, method)()
                self.factors[label] = handler.data[factor_name].fillna(0)
                
                if i == 0:  # 保存第一个处理器的数据
                    self.returns = handler.data['returns'].fillna(0)
                    self.raw_data = handler.data
            except Exception as e:
                print(f"计算因子 {label} 失败: {e}")
                # 如果失败，创建虚拟因子
                if self.raw_data is not None:
                    self.factors[label] = pd.Series(0, index=self.raw_data.index)
                continue
        
        # 计算高级因子(11-20)
        if self.raw_data is not None:
            advanced_factors = EnhancedFactorHandlers.create_advanced_factors(self.raw_data)
            self.factors.update(advanced_factors)
        
        print(f"成功计算{len(self.factors)}个因子")
        
    def standardize_factors(self, window=60):
        """多方法标准化因子"""
        for name, factor in self.factors.items():
            # Robust Z-score标准化
            rolling_median = factor.rolling(window=window, min_periods=1).median()
            mad = (factor - rolling_median).abs().rolling(window=window, min_periods=1).median()
            self.factors[name] = (factor - rolling_median) / (mad * 1.4826 + 1e-8)
        
        print("因子标准化完成（Robust Z-score）")

class MultiFactor20Strategy:
    """20因子组合策略"""
    def __init__(self, factors, returns, combination_method='advanced_ensemble'):
        self.factors = factors
        self.returns = returns
        self.combination_method = combination_method
        self.combined_factor = None
        self.positions = None
        self.factor_weights = None
        
    def combine_factors(self):
        """组合因子"""
        methods = {
            'equal_weight': self._equal_weight_combination,
            'advanced_ensemble': self._advanced_ensemble_combination,
            'hierarchical_clustering': self._hierarchical_clustering_combination,
            'neural_network': self._neural_network_combination,
            'adaptive_weight': self._adaptive_weight_combination
        }
        
        if self.combination_method in methods:
            methods[self.combination_method]()
        else:
            self._advanced_ensemble_combination()
            
    def _equal_weight_combination(self):
        """等权重组合"""
        weight = 1.0 / len(self.factors)
        self.factor_weights = {name: weight for name in self.factors.keys()}
        
        factor_df = pd.DataFrame(self.factors)
        self.combined_factor = factor_df.mean(axis=1)
        print(f"使用等权重方法，每个因子权重: {weight:.4f}")
        
    def _advanced_ensemble_combination(self):
        """高级集成组合方法"""
        factor_df = pd.DataFrame(self.factors).fillna(0)
        self.combined_factor = pd.Series(0, index=factor_df.index)
        self.factor_weights = pd.DataFrame(index=factor_df.index, columns=factor_df.columns)
        
        window = 120
        
        for i in range(window, len(factor_df)):
            window_data = factor_df.iloc[i-window:i]
            window_returns = self.returns.iloc[i-window:i]
            
            # 多维度评估
            ic_scores = []
            stability_scores = []
            monotonicity_scores = []
            
            for col in window_data.columns:
                factor_series = window_data[col]
                
                # IC得分
                ic = abs(factor_series.corr(window_returns))
                ic_scores.append(ic if not np.isnan(ic) else 0)
                
                # 稳定性得分
                factor_returns = factor_series.shift(1) * window_returns
                ir = abs(factor_returns.mean() / (factor_returns.std() + 1e-8))
                stability_scores.append(ir if not np.isnan(ir) else 0)
                
                # 单调性得分
                monotonicity = 1 - abs(factor_series.diff().fillna(0).apply(np.sign).mean())
                monotonicity_scores.append(max(monotonicity, 0))
            
            # 综合评分
            combined_scores = []
            for ic, stability, monotonicity in zip(ic_scores, stability_scores, monotonicity_scores):
                score = 0.5 * ic + 0.3 * stability + 0.2 * monotonicity
                combined_scores.append(score)
            
            # Softmax权重
            exp_scores = np.exp(np.array(combined_scores) * 5)  # 温度参数
            weights = exp_scores / exp_scores.sum()
            
            # 权重约束
            max_weight = 0.15
            min_weight = 0.01
            weights = np.clip(weights, min_weight, max_weight)
            weights = weights / weights.sum()
            
            self.factor_weights.iloc[i] = weights
            
            # 计算组合因子
            weighted_factor = np.dot(factor_df.iloc[i].values, weights)
            self.combined_factor.iloc[i] = weighted_factor
        
        # 填充前期值
        equal_weight = 1.0 / len(self.factors)
        self.combined_factor.iloc[:window] = 0
        self.factor_weights.iloc[:window] = equal_weight
        
        print("使用高级集成方法组合因子")
    
    def _hierarchical_clustering_combination(self):
        """分层聚类组合"""
        from sklearn.cluster import AgglomerativeClustering
        
        factor_df = pd.DataFrame(self.factors).fillna(0)
        
        # 计算因子相关性
        correlation_matrix = factor_df.corr()
        distance_matrix = 1 - abs(correlation_matrix)
        
        # 分层聚类
        n_clusters = min(8, len(self.factors) // 3)
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        factor_clusters = clustering.fit_predict(distance_matrix.values)
        
        # 为每个聚类分配权重
        cluster_weights = {}
        unique_clusters = np.unique(factor_clusters)
        
        for cluster_id in unique_clusters:
            cluster_factors = [factor_df.columns[i] for i in range(len(factor_clusters)) 
                             if factor_clusters[i] == cluster_id]
            
            # 聚类内最优因子权重分配
            cluster_ics = []
            for factor_name in cluster_factors:
                ic = abs(factor_df[factor_name].corr(self.returns))
                cluster_ics.append(ic if not np.isnan(ic) else 0)
            
            # 聚类权重 = 1/聚类数 * 聚类内IC权重
            cluster_base_weight = 1.0 / len(unique_clusters)
            total_cluster_ic = sum(cluster_ics) if sum(cluster_ics) > 0 else 1
            
            for i, factor_name in enumerate(cluster_factors):
                factor_weight_in_cluster = cluster_ics[i] / total_cluster_ic
                cluster_weights[factor_name] = cluster_base_weight * factor_weight_in_cluster
        
        self.factor_weights = cluster_weights
        
        # 计算组合因子
        self.combined_factor = pd.Series(0, index=factor_df.index)
        for factor_name, weight in self.factor_weights.items():
            self.combined_factor += factor_df[factor_name] * weight
        
        print(f"使用分层聚类方法组合因子（{n_clusters}个聚类）")
    
    def _neural_network_combination(self):
        """神经网络权重组合"""
        from sklearn.neural_network import MLPRegressor
        
        factor_df = pd.DataFrame(self.factors).fillna(0)
        returns_clean = self.returns.fillna(0)
        
        window = 150
        self.combined_factor = pd.Series(0, index=factor_df.index)
        self.factor_weights = pd.DataFrame(index=factor_df.index, columns=factor_df.columns)
        
        for i in range(window, len(factor_df)):
            X_train = factor_df.iloc[i-window:i].values
            y_train = returns_clean.iloc[i-window:i].values
            
            try:
                # 使用MLP预测权重
                mlp = MLPRegressor(hidden_layer_sizes=(50, 20), max_iter=200, random_state=42)
                mlp.fit(X_train, y_train)
                
                # 获取第一层权重作为因子重要性
                first_layer_weights = abs(mlp.coefs_[0]).mean(axis=1)
                weights = first_layer_weights / first_layer_weights.sum()
                
            except:
                # 如果训练失败，使用等权重
                weights = np.ones(len(factor_df.columns)) / len(factor_df.columns)
            
            self.factor_weights.iloc[i] = weights
            
            # 计算组合因子
            weighted_factor = np.dot(factor_df.iloc[i].values, weights)
            self.combined_factor.iloc[i] = weighted_factor
        
        # 填充前期值
        equal_weight = 1.0 / len(self.factors)
        self.combined_factor.iloc[:window] = 0
        self.factor_weights.iloc[:window] = equal_weight
        
        print("使用神经网络方法组合因子")
    
    def _adaptive_weight_combination(self):
        """自适应权重组合"""
        factor_df = pd.DataFrame(self.factors).fillna(0)
        
        window = 60
        self.combined_factor = pd.Series(0, index=factor_df.index)
        self.factor_weights = pd.DataFrame(index=factor_df.index, columns=factor_df.columns)
        
        # 初始化权重
        prev_weights = np.ones(len(factor_df.columns)) / len(factor_df.columns)
        
        for i in range(window, len(factor_df)):
            window_data = factor_df.iloc[i-window:i]
            window_returns = self.returns.iloc[i-window:i]
            
            # 计算每个因子的表现
            performances = []
            for j, col in enumerate(window_data.columns):
                factor_series = window_data[col]
                
                # 加权表现评估
                recent_ic = abs(factor_series[-20:].corr(window_returns[-20:]))
                recent_ic = recent_ic if not np.isnan(recent_ic) else 0
                
                overall_ic = abs(factor_series.corr(window_returns))
                overall_ic = overall_ic if not np.isnan(overall_ic) else 0
                
                # 自适应权重：70%近期表现 + 30%整体表现
                performance = 0.7 * recent_ic + 0.3 * overall_ic
                performances.append(performance)
            
            # 权重更新（带动量）
            momentum = 0.8
            new_weights = np.array(performances)
            new_weights = new_weights / (new_weights.sum() + 1e-8)
            
            weights = momentum * prev_weights + (1 - momentum) * new_weights
            weights = weights / weights.sum()
            
            # 权重约束
            max_weight = 0.12
            min_weight = 0.02
            weights = np.clip(weights, min_weight, max_weight)
            weights = weights / weights.sum()
            
            self.factor_weights.iloc[i] = weights
            prev_weights = weights
            
            # 计算组合因子
            weighted_factor = np.dot(factor_df.iloc[i].values, weights)
            self.combined_factor.iloc[i] = weighted_factor
        
        # 填充前期值
        equal_weight = 1.0 / len(self.factors)
        self.combined_factor.iloc[:window] = 0
        self.factor_weights.iloc[:window] = equal_weight
        
        print("使用自适应权重方法组合因子")
        
    def generate_signals(self, threshold=0.3):
        """生成交易信号（20因子版本）"""
        if self.combined_factor is None:
            raise ValueError("请先调用combine_factors()方法")
            
        # 动态阈值
        rolling_std = self.combined_factor.rolling(60).std()
        rolling_mean = self.combined_factor.rolling(60).mean()
        
        # 标准化组合因子
        normalized_factor = (self.combined_factor - rolling_mean) / (rolling_std + 1e-8)
        
        # 精细化信号分层
        conditions = [
            normalized_factor > threshold * 2.0,     # 强看多
            normalized_factor > threshold * 1.2,     # 中强看多
            normalized_factor > threshold * 0.6,     # 中等看多
            normalized_factor > threshold * 0.2,     # 弱看多
            normalized_factor < -threshold * 2.0,    # 强看空
            normalized_factor < -threshold * 1.2,    # 中强看空
            normalized_factor < -threshold * 0.6,    # 中等看空
            normalized_factor < -threshold * 0.2,    # 弱看空
        ]
        choices = [1.0, 0.8, 0.5, 0.2, -1.0, -0.8, -0.5, -0.2]
        
        self.positions = np.select(conditions, choices, default=0)
        self.positions = pd.Series(self.positions, index=self.combined_factor.index).shift(1).fillna(0)
        
        return self.positions

class PerformanceAnalyzer:
    """20因子专用性能分析器"""
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
        """计算性能指标"""
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
        
        # Sortino比率
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
        """绘制结果图表（20因子版本）"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 16))
        
        # 1. 累积收益对比
        axes[0, 0].plot(self.cumulative_returns.index, self.cumulative_returns.values, 
                       label=f'20因子组合策略({combination_method})', linewidth=2)
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
        
        # 4. 因子权重变化（显示前8个因子）
        if self.factor_weights is not None and isinstance(self.factor_weights, pd.DataFrame) and not self.factor_weights.empty:
            for i, col in enumerate(self.factor_weights.columns[:8]):
                if col in self.factor_weights.columns:
                    weight_data = self.factor_weights[col].dropna()
                    if len(weight_data) > 0:
                        axes[1, 1].plot(weight_data.index, weight_data.values, 
                                       label=col, alpha=0.8)
            axes[1, 1].set_title('主要因子权重变化')
            axes[1, 1].legend(fontsize=7)
        elif isinstance(self.factor_weights, dict):
            factor_names = list(self.factor_weights.keys())
            weights = list(self.factor_weights.values())
            axes[1, 1].bar(factor_names, weights, alpha=0.7)
            axes[1, 1].set_title('因子权重分布')
            axes[1, 1].tick_params(axis='x', rotation=90)
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

def run_multi_factor_20_strategy(data_path, combination_methods=None):
    """运行20因子组合策略"""
    if combination_methods is None:
        combination_methods = ['equal_weight', 'advanced_ensemble', 'hierarchical_clustering', 
                             'adaptive_weight']
    
    print("=== 20因子组合策略回测开始 ===")
    
    # 加载数据
    processor = FactorDataProcessor(data_path)
    processor.load_and_calculate_factors()
    processor.standardize_factors()
    
    results = {}
    
    for method in combination_methods:
        print(f"\n--- 测试{method}组合方法 ---")
        
        try:
            strategy = MultiFactor20Strategy(processor.factors, processor.returns, method)
            strategy.combine_factors()
            strategy.generate_signals()
            
            # 使用与其他多因子相同的性能分析器
            analyzer = PerformanceAnalyzer(processor.returns, strategy.positions, 
                                         processor.factors, strategy.factor_weights, strategy.combined_factor)
            metrics = analyzer.calculate_metrics()
            
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
            save_path = f'./results/backtest_results/multi_factor_20_{method}_results.png'
            analyzer.plot_results(method, save_path)
            
        except Exception as e:
            print(f"方法 {method} 运行失败: {e}")
            continue
    
    # 结果对比
    if results:
        print("\n=== 各方法性能对比 ===")
        comparison_df = pd.DataFrame({method: result['metrics'] 
                                    for method, result in results.items()}).T
        print(comparison_df)
        comparison_df.to_csv('./results/backtest_results/multi_factor_20_comparison.csv')
    
    print("=== 20因子组合策略回测完成 ===")
    return results

def main():
    """主函数"""
    data_path = 'code_1/300638_2024.pkl'
    
    try:
        results = run_multi_factor_20_strategy(data_path)
        
        if results:
            # 最佳方法
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