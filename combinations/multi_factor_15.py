"""
多因子组合策略 - 15因子组合
基于前15个因子的大规模多因子组合策略，采用更加复杂和精细的权重分配方法
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

# 导入所有15个因子策略类
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

# 由于只有10个因子策略文件，我们需要创建额外的因子
class AdditionalFactorHandlers:
    """额外因子处理器 - 为15因子组合创建更多因子"""
    
    @staticmethod
    def create_momentum_reversal_factor(data):
        """动量反转因子"""
        close = data['Close']
        volume = data['Volume']
        
        # 短期动量
        momentum_5d = close.pct_change(5)
        momentum_10d = close.pct_change(10)
        momentum_20d = close.pct_change(20)
        
        # 反转信号：短期动量过强时反转
        factor = -0.5 * momentum_5d - 0.3 * momentum_10d + 0.2 * momentum_20d
        
        return factor.fillna(0)
    
    @staticmethod
    def create_volume_price_trend_factor(data):
        """量价趋势因子"""
        close = data['Close']
        volume = data['Volume']
        
        # 价格趋势
        price_ma_5 = close.rolling(5).mean()
        price_ma_20 = close.rolling(20).mean()
        price_trend = (price_ma_5 / price_ma_20 - 1)
        
        # 成交量趋势
        volume_ma_5 = volume.rolling(5).mean()
        volume_ma_20 = volume.rolling(20).mean()
        volume_trend = (volume_ma_5 / volume_ma_20 - 1)
        
        # 量价背离检测
        factor = np.sign(price_trend) * volume_trend
        
        return factor.fillna(0)
    
    @staticmethod
    def create_volatility_skew_factor(data):
        """波动率偏度因子"""
        returns = data['Close'].pct_change()
        
        # 计算滚动波动率
        volatility = returns.rolling(20).std()
        
        # 计算收益偏度
        skewness = returns.rolling(20).skew()
        
        # 波动率偏度因子
        factor = volatility * skewness
        
        return factor.fillna(0)
    
    @staticmethod
    def create_high_low_ratio_factor(data):
        """高低价比率因子"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # 当日高低价差
        daily_range = (high - low) / close
        
        # 滚动最高最低价
        high_20 = high.rolling(20).max()
        low_20 = low.rolling(20).min()
        
        # 相对位置因子
        position_factor = (close - low_20) / (high_20 - low_20 + 1e-8)
        
        # 结合当日波动率
        factor = daily_range * (0.5 - position_factor)  # 中位数回归
        
        return factor.fillna(0)
    
    @staticmethod
    def create_turnover_momentum_factor(data):
        """换手率动量因子"""
        volume = data['Volume']
        close = data['Close']
        
        # 估算换手率（假设流通股本固定）
        turnover = volume / volume.rolling(60).mean()
        
        # 换手率变化
        turnover_change = turnover.pct_change(5)
        
        # 价格动量
        price_momentum = close.pct_change(5)
        
        # 换手率动量因子
        factor = turnover_change * np.sign(price_momentum)
        
        return factor.fillna(0)

class FactorDataProcessor:
    """因子数据处理器 - 处理15个因子"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.factors = {}
        self.returns = None
        self.raw_data = None
        
    def load_and_calculate_factors(self):
        """加载数据并计算前15个因子"""
        print("正在计算前15个因子...")
        
        # 1-10 因子（与10因子组合相同）
        data_handler1 = DataHandler1(self.data_path)
        data_handler1.calculate_rsi()
        self.factors['RSI_Reversal'] = data_handler1.data['RSI_Reversal_Factor'].fillna(0)
        self.raw_data = data_handler1.data  # 保存原始数据用于计算额外因子
        
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
        
        # 11-15 因子（新增的复合因子）
        self.factors['Momentum_Reversal'] = AdditionalFactorHandlers.create_momentum_reversal_factor(self.raw_data)
        self.factors['Volume_Price_Trend'] = AdditionalFactorHandlers.create_volume_price_trend_factor(self.raw_data)
        self.factors['Volatility_Skew'] = AdditionalFactorHandlers.create_volatility_skew_factor(self.raw_data)
        self.factors['High_Low_Ratio'] = AdditionalFactorHandlers.create_high_low_ratio_factor(self.raw_data)
        self.factors['Turnover_Momentum'] = AdditionalFactorHandlers.create_turnover_momentum_factor(self.raw_data)
        
        # 使用第一个数据处理器的收益率作为基准
        self.returns = data_handler1.data['returns'].fillna(0)
        
        print(f"成功计算{len(self.factors)}个因子")
        
    def standardize_factors(self, window=60):
        """标准化因子 - 多种方法结合"""
        for name, factor in self.factors.items():
            # Z-score标准化
            rolling_mean = factor.rolling(window=window, min_periods=1).mean()
            rolling_std = factor.rolling(window=window, min_periods=1).std()
            z_score = (factor - rolling_mean) / (rolling_std + 1e-8)
            
            # 分位数标准化（Robust Scaling）
            rolling_median = factor.rolling(window=window, min_periods=1).median()
            rolling_iqr = (factor.rolling(window=window, min_periods=1).quantile(0.75) - 
                          factor.rolling(window=window, min_periods=1).quantile(0.25))
            robust_score = (factor - rolling_median) / (rolling_iqr + 1e-8)
            
            # 组合标准化（70% Z-score + 30% Robust）
            self.factors[name] = 0.7 * z_score + 0.3 * robust_score
            
        print("因子标准化完成（Z-score + Robust scaling）")
        
    def calculate_factor_statistics(self):
        """计算因子统计特征"""
        factor_df = pd.DataFrame(self.factors)
        
        stats = {}
        for factor_name in factor_df.columns:
            factor_data = factor_df[factor_name]
            stats[factor_name] = {
                'mean': factor_data.mean(),
                'std': factor_data.std(),
                'skew': factor_data.skew(),
                'kurt': factor_data.kurtosis(),
                'ic': factor_data.corr(self.returns),
                'ic_abs': abs(factor_data.corr(self.returns))
            }
        
        stats_df = pd.DataFrame(stats).T
        print("因子统计特征:")
        print(stats_df)
        
        return stats_df

class MultiFactor15Strategy:
    """15因子组合策略"""
    def __init__(self, factors, returns, combination_method='equal_weight'):
        self.factors = factors
        self.returns = returns
        self.combination_method = combination_method
        self.combined_factor = None
        self.positions = None
        self.factor_weights = None
        
    def combine_factors(self):
        """根据不同方法组合因子"""
        methods = {
            'equal_weight': self._equal_weight_combination,
            'ic_weight': self._ic_weight_combination,
            'risk_parity': self._risk_parity_combination,
            'dynamic_weight': self._dynamic_weight_combination,
            'ml_weight': self._ml_weight_combination,
            'pca_weight': self._pca_weight_combination,
            'cluster_weight': self._cluster_weight_combination,
            'hierarchical_weight': self._hierarchical_weight_combination
        }
        
        if self.combination_method in methods:
            methods[self.combination_method]()
        else:
            raise ValueError(f"未知的组合方法: {self.combination_method}")
            
    def _equal_weight_combination(self):
        """等权重组合"""
        weight = 1.0 / len(self.factors)
        self.factor_weights = {name: weight for name in self.factors.keys()}
        
        factor_df = pd.DataFrame(self.factors)
        self.combined_factor = factor_df.mean(axis=1)
        print(f"使用等权重方法组合因子，每个因子权重: {weight:.4f}")
        
    def _ic_weight_combination(self, window=60):
        """基于IC的权重组合（增强版）"""
        factor_df = pd.DataFrame(self.factors)
        self.combined_factor = pd.Series(0, index=factor_df.index)
        self.factor_weights = pd.DataFrame(index=factor_df.index, columns=factor_df.columns)
        
        for i in range(window, len(factor_df)):
            window_data = factor_df.iloc[i-window:i]
            window_returns = self.returns.iloc[i-window:i]
            
            # 计算多期IC
            ics_1d = []
            ics_3d = []
            ics_5d = []
            
            for col in window_data.columns:
                factor_series = window_data[col]
                
                # 1日IC
                ic_1d = abs(factor_series.corr(window_returns))
                ics_1d.append(ic_1d if not np.isnan(ic_1d) else 0)
                
                # 3日IC
                if len(window_returns) >= 3:
                    ic_3d = abs(factor_series[:-2].corr(window_returns[2:]))
                    ics_3d.append(ic_3d if not np.isnan(ic_3d) else 0)
                else:
                    ics_3d.append(0)
                
                # 5日IC
                if len(window_returns) >= 5:
                    ic_5d = abs(factor_series[:-4].corr(window_returns[4:]))
                    ics_5d.append(ic_5d if not np.isnan(ic_5d) else 0)
                else:
                    ics_5d.append(0)
            
            # 加权组合多期IC
            combined_ics = [0.5 * ic1 + 0.3 * ic3 + 0.2 * ic5 
                           for ic1, ic3, ic5 in zip(ics_1d, ics_3d, ics_5d)]
            
            # 权重标准化
            total_ic = sum(combined_ics) if sum(combined_ics) > 0 else 1
            weights = [ic / total_ic for ic in combined_ics]
            
            # 权重平滑
            if i > window:
                prev_weights = self.factor_weights.iloc[i-1].values
                alpha = 0.7  # 平滑系数
                weights = [alpha * w + (1-alpha) * pw for w, pw in zip(weights, prev_weights)]
                weights = [w / sum(weights) for w in weights]
            
            self.factor_weights.iloc[i] = weights
            
            # 计算加权因子
            weighted_factor = sum(factor_df.iloc[i, j] * weights[j] 
                                for j in range(len(weights)))
            self.combined_factor.iloc[i] = weighted_factor
        
        # 填充前面的值
        equal_weight = 1.0 / len(self.factors)
        self.combined_factor.iloc[:window] = 0
        self.factor_weights.iloc[:window] = equal_weight
        
        print("使用多期IC加权方法组合因子")
        
    def _cluster_weight_combination(self, n_clusters=5):
        """基于聚类的权重组合"""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        factor_df = pd.DataFrame(self.factors).fillna(0)
        
        # 计算因子相关性矩阵作为聚类特征
        correlation_matrix = factor_df.corr()
        
        # 使用K-means对因子进行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        factor_clusters = kmeans.fit_predict(correlation_matrix.values)
        
        # 为每个聚类分配权重
        cluster_weights = {}
        unique_clusters = np.unique(factor_clusters)
        cluster_base_weight = 1.0 / len(unique_clusters)
        
        for cluster_id in unique_clusters:
            cluster_factors = [factor_df.columns[i] for i in range(len(factor_clusters)) 
                             if factor_clusters[i] == cluster_id]
            
            # 计算聚类内因子的IC权重
            cluster_ics = []
            for factor_name in cluster_factors:
                ic = abs(factor_df[factor_name].corr(self.returns))
                cluster_ics.append(ic if not np.isnan(ic) else 0)
            
            # 聚类内权重分配
            total_cluster_ic = sum(cluster_ics) if sum(cluster_ics) > 0 else 1
            for i, factor_name in enumerate(cluster_factors):
                factor_weight_in_cluster = cluster_ics[i] / total_cluster_ic
                cluster_weights[factor_name] = cluster_base_weight * factor_weight_in_cluster
        
        # 标准化权重
        total_weight = sum(cluster_weights.values())
        self.factor_weights = {name: weight / total_weight 
                              for name, weight in cluster_weights.items()}
        
        # 计算组合因子
        self.combined_factor = pd.Series(0, index=factor_df.index)
        for factor_name, weight in self.factor_weights.items():
            self.combined_factor += factor_df[factor_name] * weight
        
        print(f"使用聚类权重方法组合因子（{n_clusters}个聚类）")
        
    def _hierarchical_weight_combination(self):
        """分层权重组合 - 基于因子类型分层"""
        factor_df = pd.DataFrame(self.factors)
        
        # 定义因子分组
        factor_groups = {
            'trend_factors': ['RSI_Reversal', 'MACD_Momentum', 'Price_Volume_Divergence', 
                             'Williams_Reversal', 'CCI'],
            'volatility_factors': ['Bollinger_Position', 'Volatility', 'Volatility_Skew', 
                                 'High_Low_Ratio'],
            'volume_factors': ['Volume_Anomaly', 'Volume_Price_Trend', 'Turnover_Momentum'],
            'oscillator_factors': ['Williams_Reversal', 'CCI'],
            'relative_factors': ['VWAP']
        }
        
        # 为每个组分配基础权重
        group_base_weights = {
            'trend_factors': 0.35,
            'volatility_factors': 0.25,
            'volume_factors': 0.20,
            'oscillator_factors': 0.15,
            'relative_factors': 0.05
        }
        
        # 计算每个组内的因子权重
        final_weights = {}
        
        for group_name, factors_in_group in factor_groups.items():
            # 过滤存在的因子
            existing_factors = [f for f in factors_in_group if f in self.factors]
            
            if not existing_factors:
                continue
                
            group_base_weight = group_base_weights[group_name]
            
            # 计算组内因子的IC
            group_ics = []
            for factor_name in existing_factors:
                ic = abs(factor_df[factor_name].corr(self.returns))
                group_ics.append(ic if not np.isnan(ic) else 0)
            
            # 组内权重分配
            total_group_ic = sum(group_ics) if sum(group_ics) > 0 else 1
            for i, factor_name in enumerate(existing_factors):
                factor_weight_in_group = group_ics[i] / total_group_ic
                final_weights[factor_name] = group_base_weight * factor_weight_in_group
        
        # 标准化权重
        total_weight = sum(final_weights.values())
        self.factor_weights = {name: weight / total_weight 
                              for name, weight in final_weights.items()}
        
        # 计算组合因子
        self.combined_factor = pd.Series(0, index=factor_df.index)
        for factor_name, weight in self.factor_weights.items():
            if factor_name in factor_df.columns:
                self.combined_factor += factor_df[factor_name] * weight
        
        print("使用分层权重方法组合因子")
        
    def _risk_parity_combination(self, window=60):
        """风险平价组合（改进版，适用于更多因子）"""
        factor_df = pd.DataFrame(self.factors)
        self.combined_factor = pd.Series(0, index=factor_df.index)
        self.factor_weights = pd.DataFrame(index=factor_df.index, columns=factor_df.columns)
        
        for i in range(window, len(factor_df)):
            window_data = factor_df.iloc[i-window:i]
            
            # 计算协方差矩阵
            cov_matrix = window_data.cov().values
            
            # 处理奇异矩阵
            try:
                # 添加正则化项避免奇异矩阵
                regularization = 1e-6 * np.eye(cov_matrix.shape[0])
                cov_matrix_reg = cov_matrix + regularization
                
                # 风险平价权重计算
                num_factors = len(window_data.columns)
                weights = np.ones(num_factors) / num_factors
                
                # 迭代优化
                for iteration in range(30):
                    portfolio_var = np.dot(weights, np.dot(cov_matrix_reg, weights))
                    if portfolio_var <= 1e-10:
                        break
                        
                    marginal_contrib = np.dot(cov_matrix_reg, weights)
                    contrib = weights * marginal_contrib / portfolio_var
                    
                    target_contrib = 1.0 / num_factors
                    
                    # 更新权重
                    for j in range(num_factors):
                        if contrib[j] > 1e-10:
                            weights[j] *= (target_contrib / contrib[j]) ** 0.05
                    
                    # 标准化权重
                    weights = weights / weights.sum()
                    
                    # 检查收敛
                    if np.max(np.abs(contrib - target_contrib)) < 1e-4:
                        break
                        
            except:
                # 如果风险平价失败，使用等权重
                weights = np.ones(num_factors) / num_factors
            
            self.factor_weights.iloc[i] = weights
            
            # 计算加权因子
            weighted_factor = np.dot(factor_df.iloc[i].values, weights)
            self.combined_factor.iloc[i] = weighted_factor
        
        # 填充前面的值
        equal_weight = 1.0 / len(self.factors)
        self.combined_factor.iloc[:window] = 0
        self.factor_weights.iloc[:window] = equal_weight
        
        print("使用改进风险平价方法组合因子")
    
    def _dynamic_weight_combination(self, window=60):
        """动态权重组合 - 多维度评估（15因子版本）"""
        factor_df = pd.DataFrame(self.factors)
        self.combined_factor = pd.Series(0, index=factor_df.index)
        self.factor_weights = pd.DataFrame(index=factor_df.index, columns=factor_df.columns)
        
        for i in range(window, len(factor_df)):
            window_data = factor_df.iloc[i-window:i]
            window_returns = self.returns.iloc[i-window:i]
            
            performances = []
            for col in window_data.columns:
                factor_series = window_data[col]
                
                # 1. 预测能力（IC）
                ic = abs(factor_series.corr(window_returns))
                ic_score = ic if not np.isnan(ic) else 0
                
                # 2. 稳定性（信息比率）
                factor_returns = factor_series.shift(1) * window_returns
                ir = (factor_returns.mean() / factor_returns.std() 
                     if factor_returns.std() > 0 else 0)
                stability_score = abs(ir) if not np.isnan(ir) else 0
                
                # 3. 单调性（因子值变化的一致性）
                monotonicity = abs(factor_series.diff().fillna(0).apply(np.sign).mean())
                
                # 4. 非线性关系检测（Spearman相关系数）
                try:
                    spearman_corr = factor_series.rank().corr(window_returns.rank())
                    nonlinear_score = abs(spearman_corr) if not np.isnan(spearman_corr) else 0
                except:
                    nonlinear_score = 0
                
                # 5. 尾部收益预测能力
                extreme_returns = window_returns[abs(window_returns) > window_returns.std()]
                if len(extreme_returns) > 5:
                    tail_ic = abs(factor_series.loc[extreme_returns.index].corr(extreme_returns))
                    tail_score = tail_ic if not np.isnan(tail_ic) else 0
                else:
                    tail_score = 0
                
                # 综合得分
                performance = (0.30 * ic_score +           # IC权重
                             0.25 * stability_score +      # 稳定性权重
                             0.15 * monotonicity +         # 单调性权重
                             0.20 * nonlinear_score +      # 非线性关系权重
                             0.10 * tail_score)            # 尾部预测权重
                
                performances.append(performance)
            
            # 权重标准化
            total_perf = sum(performances) if sum(performances) > 0 else 1
            weights = [perf / total_perf for perf in performances]
            
            # 权重约束（避免过度集中）
            max_weight = 0.15  # 单个因子最大权重15%
            min_weight = 0.02  # 单个因子最小权重2%
            
            weights = [max(min(w, max_weight), min_weight) for w in weights]
            weights = [w / sum(weights) for w in weights]  # 重新标准化
            
            # 权重平滑
            if i > window:
                prev_weights = self.factor_weights.iloc[i-1].values
                alpha = 0.75
                weights = [alpha * w + (1-alpha) * pw for w, pw in zip(weights, prev_weights)]
                weights = [w / sum(weights) for w in weights]
            
            self.factor_weights.iloc[i] = weights
            
            # 计算加权因子
            weighted_factor = sum(factor_df.iloc[i, j] * weights[j] 
                                for j in range(len(weights)))
            self.combined_factor.iloc[i] = weighted_factor
        
        # 填充前面的值
        equal_weight = 1.0 / len(self.factors)
        self.combined_factor.iloc[:window] = 0
        self.factor_weights.iloc[:window] = equal_weight
        
        print("使用多维度动态权重方法组合因子")
    
    def _ml_weight_combination(self):
        """机器学习权重组合（15因子增强版）"""
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
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
            
            # 多种机器学习模型
            models = {
                'linear': LinearRegression(),
                'ridge': Ridge(alpha=0.1),
                'lasso': Lasso(alpha=0.01),
                'elastic': ElasticNet(alpha=0.01, l1_ratio=0.5),
                'rf': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            }
            
            model_weights = []
            model_scores = []
            
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    
                    # 获取模型权重
                    if hasattr(model, 'coef_'):
                        weights = model.coef_
                    elif hasattr(model, 'feature_importances_'):
                        weights = model.feature_importances_
                    else:
                        weights = np.ones(len(factor_df.columns))
                    
                    # 计算模型得分
                    train_score = model.score(X_train, y_train)
                    
                    model_weights.append(np.abs(weights))
                    model_scores.append(max(train_score, 0.01))
                    
                except Exception as e:
                    print(f"模型 {name} 训练失败: {e}")
                    model_weights.append(np.ones(len(factor_df.columns)))
                    model_scores.append(0.01)
            
            # 基于模型表现的集成权重
            total_score = sum(model_scores)
            final_weights = np.zeros(len(factor_df.columns))
            
            for weights, score in zip(model_weights, model_scores):
                final_weights += weights * (score / total_score)
            
            # 权重标准化
            weights_normalized = final_weights / (final_weights.sum() + 1e-8)
            
            # 权重约束
            max_weight = 0.2
            min_weight = 0.01
            weights_normalized = np.clip(weights_normalized, min_weight, max_weight)
            weights_normalized = weights_normalized / weights_normalized.sum()
            
            self.factor_weights.iloc[i] = weights_normalized
            
            # 计算加权因子
            weighted_factor = np.dot(factor_df.iloc[i].values, weights_normalized)
            self.combined_factor.iloc[i] = weighted_factor
        
        # 填充前面的值
        equal_weight = 1.0 / len(self.factors)
        self.combined_factor.iloc[:window] = 0
        self.factor_weights.iloc[:window] = equal_weight
        
        print("使用机器学习集成权重方法组合因子")
    
    def _pca_weight_combination(self, n_components=8):
        """PCA降维权重组合（适应15因子）"""
        from sklearn.decomposition import PCA
        
        factor_df = pd.DataFrame(self.factors).fillna(0)
        
        window = 120
        self.combined_factor = pd.Series(0, index=factor_df.index)
        self.factor_weights = pd.DataFrame(index=factor_df.index, columns=factor_df.columns)
        
        for i in range(window, len(factor_df)):
            window_data = factor_df.iloc[i-window:i].values
            
            # 拟合PCA
            pca = PCA(n_components=min(n_components, len(factor_df.columns)))
            pca.fit(window_data)
            
            # 计算权重（基于主成分贡献度）
            components = pca.components_
            explained_variance_ratio = pca.explained_variance_ratio_
            
            # 加权主成分
            weighted_components = components.T * explained_variance_ratio
            final_weights = np.abs(weighted_components.sum(axis=1))
            
            # 权重标准化
            weights_normalized = final_weights / final_weights.sum()
            
            self.factor_weights.iloc[i] = weights_normalized
            
            # 计算加权因子
            weighted_factor = np.dot(factor_df.iloc[i].values, weights_normalized)
            self.combined_factor.iloc[i] = weighted_factor
        
        # 填充前面的值
        equal_weight = 1.0 / len(self.factors)
        self.combined_factor.iloc[:window] = 0
        self.factor_weights.iloc[:window] = equal_weight
        
        print(f"使用PCA降维方法组合因子（保留{n_components}个主成分）")
        
    def generate_signals(self, threshold=0.4):
        """生成交易信号（15因子版本 - 更精细的信号分层）"""
        if self.combined_factor is None:
            raise ValueError("请先调用combine_factors()方法")
            
        # 计算动态阈值
        rolling_std = self.combined_factor.rolling(60).std()
        dynamic_threshold = threshold * (1 + rolling_std / rolling_std.mean())
        
        # 多层信号生成
        conditions = [
            self.combined_factor > dynamic_threshold * 1.5,     # 强看多
            self.combined_factor > dynamic_threshold,           # 中等看多
            self.combined_factor > dynamic_threshold * 0.5,     # 弱看多
            self.combined_factor < -dynamic_threshold * 1.5,    # 强看空
            self.combined_factor < -dynamic_threshold,          # 中等看空
            self.combined_factor < -dynamic_threshold * 0.5,    # 弱看空
        ]
        choices = [1.0, 0.7, 0.3, -1.0, -0.7, -0.3]  # 对应仓位
        
        self.positions = np.select(conditions, choices, default=0)
        self.positions = pd.Series(self.positions, index=self.combined_factor.index).shift(1).fillna(0)
        
        return self.positions

class PerformanceAnalyzer:
    """15因子专用性能分析器"""
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
        """绘制结果图表（15因子版本）"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        
        # 1. 累积收益对比
        axes[0, 0].plot(self.cumulative_returns.index, self.cumulative_returns.values, 
                       label=f'15因子组合策略({combination_method})', linewidth=2)
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
            for i, col in enumerate(self.factor_weights.columns[:10]):  # 显示前10个因子
                if col in self.factor_weights.columns:
                    weight_data = self.factor_weights[col].dropna()
                    if len(weight_data) > 0:
                        axes[1, 1].plot(weight_data.index, weight_data.values, 
                                       label=col, alpha=0.8)
            axes[1, 1].set_title('主要因子权重变化')
            axes[1, 1].legend(fontsize=8)
        elif isinstance(self.factor_weights, dict):
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
        
        # 6. 月度收益分布
        period_size = 21
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

def run_multi_factor_15_strategy(data_path, combination_methods=None):
    """运行15因子组合策略的主函数"""
    if combination_methods is None:
        combination_methods = ['equal_weight', 'ic_weight', 'risk_parity', 
                             'dynamic_weight', 'ml_weight', 'pca_weight',
                             'cluster_weight', 'hierarchical_weight']
    
    print("=== 15因子组合策略回测开始 ===")
    
    # 1. 加载和处理因子数据
    processor = FactorDataProcessor(data_path)
    processor.load_and_calculate_factors()
    processor.standardize_factors()
    
    # 分析因子统计特征
    factor_stats = processor.calculate_factor_statistics()
    
    results = {}
    
    # 2. 测试不同的组合方法
    for method in combination_methods:
        print(f"\n--- 测试{method}组合方法 ---")
        
        try:
            # 创建策略
            strategy = MultiFactor15Strategy(processor.factors, processor.returns, method)
            strategy.combine_factors()
            strategy.generate_signals()
            
            # 性能分析
            # 使用与10因子相同的性能分析器
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
            save_path = f'./results/backtest_results/multi_factor_15_{method}_results.png'
            # 修改plot_results方法标题
            analyzer.plot_results(method, save_path)
            
        except Exception as e:
            print(f"方法 {method} 运行失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 3. 结果对比
    if results:
        print("\n=== 各方法性能对比 ===")
        comparison_df = pd.DataFrame({method: result['metrics'] 
                                    for method, result in results.items()}).T
        print(comparison_df)
        
        # 保存对比结果
        comparison_df.to_csv('./results/backtest_results/multi_factor_15_comparison.csv')
        
        # 保存因子统计特征
        factor_stats.to_csv('./results/backtest_results/factor_statistics_15.csv')
    
    print("=== 15因子组合策略回测完成 ===")
    return results

def main():
    """主函数"""
    # 数据路径
    data_path = 'code_1/300638_2024.pkl'
    
    try:
        # 运行15因子组合策略
        results = run_multi_factor_15_strategy(data_path)
        
        if results:
            # 综合评估（不仅仅看夏普比率）
            print("\n=== 综合性能评估 ===")
            
            # 计算综合得分
            for method, result in results.items():
                metrics = result['metrics']
                # 综合得分 = 夏普比率 * 0.4 + Sortino比率 * 0.3 + 卡尔马比率 * 0.2 + 胜率 * 0.1
                comprehensive_score = (metrics['sharpe_ratio'] * 0.4 + 
                                     metrics['sortino_ratio'] * 0.3 + 
                                     metrics['calmar_ratio'] * 0.2 + 
                                     metrics['win_rate'] * 0.1)
                results[method]['comprehensive_score'] = comprehensive_score
            
            # 按综合得分排序
            sorted_results = sorted(results.items(), 
                                  key=lambda x: x[1]['comprehensive_score'], 
                                  reverse=True)
            
            print("各方法综合得分排名:")
            for i, (method, result) in enumerate(sorted_results, 1):
                print(f"{i}. {method}: {result['comprehensive_score']:.4f}")
                
            # 最佳方法
            best_method = sorted_results[0][0]
            print(f"\n最佳组合方法: {best_method}")
            print(f"综合得分: {sorted_results[0][1]['comprehensive_score']:.4f}")
        
    except Exception as e:
        print(f"策略运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 