#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能模型选择系统
基于数据特征自动选择最优模型并进行超参数优化
支持缺少可选依赖包的情况
"""

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

# 基础依赖包
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# 可选依赖包检查
HAS_XGBOOST = False
HAS_LIGHTGBM = False
HAS_CATBOOST = False
HAS_OPTUNA = False
HAS_TENSORFLOW = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    xgb = None
    print("信息: XGBoost未安装，将跳过XGBoost模型")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    print("信息: LightGBM未安装，将跳过LightGBM模型")

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    cb = None
    print("信息: CatBoost未安装，将跳过CatBoost模型")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    optuna = None
    print("信息: Optuna未安装，将使用默认参数")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    tf = None
    Sequential = None
    print("信息: TensorFlow未安装，将跳过深度学习模型")

import logging
import pickle
from datetime import datetime
import json
from scipy import stats

# 确保目录存在
os.makedirs('logs', exist_ok=True)
os.makedirs('results', exist_ok=True)

# 配置日志 - 修复编码问题
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_selection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataProfiler:
    """数据特征分析器"""
    
    def __init__(self):
        self.profile = {}
    
    def analyze_data_characteristics(self, features, returns):
        """
        分析数据特征
        
        Args:
            features: 特征数据 DataFrame
            returns: 目标变量（收益率）Series
            
        Returns:
            dict: 数据特征描述
        """
        logger.info("开始分析数据特征...")
        
        n_samples, n_features = features.shape
        
        # 基本统计信息
        self.profile['n_samples'] = n_samples
        self.profile['n_features'] = n_features
        self.profile['feature_to_sample_ratio'] = n_features / n_samples
        
        # 特征相关性分析
        corr_matrix = features.corr()
        high_corr_pairs = np.sum(np.abs(corr_matrix) > 0.8) - n_features
        self.profile['high_correlation_pairs'] = high_corr_pairs
        
        # 避免空的相关性矩阵
        upper_triangle = np.triu_indices_from(corr_matrix.values, k=1)
        if len(upper_triangle[0]) > 0:
            self.profile['avg_feature_correlation'] = np.mean(np.abs(corr_matrix.values[upper_triangle]))
        else:
            self.profile['avg_feature_correlation'] = 0
        
        # 目标变量分析
        self.profile['target_mean'] = returns.mean()
        self.profile['target_std'] = returns.std()
        self.profile['target_skewness'] = returns.skew()
        self.profile['target_kurtosis'] = returns.kurtosis()
        
        # 缺失值分析
        self.profile['missing_rate'] = features.isnull().sum().sum() / (n_samples * n_features)
        
        # 时间序列特性
        returns_clean = returns.dropna()
        if len(returns_clean) > 1:
            try:
                autocorr_1 = returns_clean.autocorr(lag=1)
                self.profile['autocorr_lag1'] = autocorr_1 if not np.isnan(autocorr_1) else 0
            except:
                self.profile['autocorr_lag1'] = 0
        else:
            self.profile['autocorr_lag1'] = 0
            
        # 特征分布特性
        numerical_features = features.select_dtypes(include=[np.number])
        if not numerical_features.empty:
            self.profile['feature_skewness_mean'] = numerical_features.skew().mean()
            self.profile['feature_kurtosis_mean'] = numerical_features.kurtosis().mean()
        else:
            self.profile['feature_skewness_mean'] = 0
            self.profile['feature_kurtosis_mean'] = 0
        
        logger.info(f"数据特征分析完成: {n_samples}样本, {n_features}特征")
        return self.profile

class ModelRecommender:
    """模型推荐器"""
    
    def __init__(self):
        self.model_scores = {}
    
    def recommend_models_based_on_data(self, data_profile):
        """
        基于数据特征推荐合适的模型
        
        Args:
            data_profile: 数据特征字典
            
        Returns:
            list: 推荐的模型列表，按推荐程度排序
        """
        logger.info("基于数据特征推荐模型...")
        
        recommendations = {}
        
        # 线性模型适用条件 - 总是可用
        if data_profile['feature_to_sample_ratio'] < 0.5 and data_profile['avg_feature_correlation'] < 0.7:
            recommendations['ridge'] = 0.8
            recommendations['lasso'] = 0.7
            recommendations['elastic_net'] = 0.75
        else:
            # 即使条件不完全满足，也保留线性模型作为基准
            recommendations['ridge'] = 0.6
        
        # 树模型适用条件
        if data_profile['n_samples'] > 50:  # 降低样本要求
            recommendations['random_forest'] = 0.85
            
            # 只有在包可用时才推荐
            if HAS_XGBOOST:
                recommendations['xgboost'] = 0.9
            if HAS_LIGHTGBM:
                recommendations['lightgbm'] = 0.95
            if HAS_CATBOOST:
                recommendations['catboost'] = 0.88
        
        # 支持向量机适用条件
        if data_profile['n_samples'] < 2000 and data_profile['n_features'] < 20:
            recommendations['svr'] = 0.6
        
        # 深度学习模型适用条件
        if HAS_TENSORFLOW and data_profile['n_samples'] > 200:
            recommendations['mlp'] = 0.7
            if abs(data_profile.get('autocorr_lag1', 0)) > 0.1:
                recommendations['lstm'] = 0.8
                recommendations['gru'] = 0.75
        
        # 确保至少有一些推荐模型
        if not recommendations:
            recommendations['ridge'] = 0.8
            recommendations['random_forest'] = 0.7
        
        # 按推荐分数排序
        sorted_models = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        recommended_models = [model for model, score in sorted_models if score > 0.5]
        
        logger.info(f"推荐模型: {recommended_models}")
        return recommended_models

class ModelOptimizer:
    """模型优化器"""
    
    def __init__(self):
        self.best_params = {}
    
    def optimize_hyperparameters(self, model_type, features, returns, n_trials=30):
        """
        超参数优化
        
        Args:
            model_type: 模型类型
            features: 特征数据
            returns: 目标变量
            n_trials: 优化试验次数
            
        Returns:
            dict: 最优超参数
        """
        logger.info(f"开始优化{model_type}模型超参数...")
        
        if not HAS_OPTUNA:
            logger.warning("Optuna未安装，使用默认参数")
            return self._get_default_params(model_type)
        
        # 检查模型是否可用
        if model_type == 'xgboost' and not HAS_XGBOOST:
            logger.warning("XGBoost不可用，使用随机森林替代")
            return self._get_default_params('random_forest')
        
        if model_type == 'lightgbm' and not HAS_LIGHTGBM:
            logger.warning("LightGBM不可用，使用随机森林替代")
            return self._get_default_params('random_forest')
        
        if model_type == 'catboost' and not HAS_CATBOOST:
            logger.warning("CatBoost不可用，使用随机森林替代")
            return self._get_default_params('random_forest')
        
        def objective(trial):
            try:
                # 获取模型和参数
                model, params = self._get_model_with_params(model_type, trial)
                
                if model is None:
                    return -np.inf
                
                # 时间序列交叉验证
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                
                for train_idx, val_idx in tscv.split(features):
                    X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
                    y_train, y_val = returns.iloc[train_idx], returns.iloc[val_idx]
                    
                    # 数据预处理
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    if model_type in ['lstm', 'gru']:
                        score = self._train_deep_model(model, X_train_scaled, y_train, X_val_scaled, y_val)
                    else:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_val_scaled)
                        score = -mean_squared_error(y_val, y_pred)
                    
                    scores.append(score)
                
                return np.mean(scores)
            except Exception as e:
                logger.warning(f"参数优化过程中出现错误: {e}")
                return -np.inf
        
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, timeout=180)  # 3分钟超时
            
            best_params = study.best_params
            logger.info(f"{model_type}最优参数: {best_params}")
            return best_params
        except Exception as e:
            logger.error(f"超参数优化失败: {e}")
            return self._get_default_params(model_type)
    
    def _get_model_with_params(self, model_type, trial):
        """根据模型类型和trial获取模型和参数"""
        try:
            if model_type == 'ridge':
                alpha = trial.suggest_float('alpha', 1e-4, 100, log=True)
                return Ridge(alpha=alpha, random_state=2025), {'alpha': alpha}
                
            elif model_type == 'lasso':
                alpha = trial.suggest_float('alpha', 1e-4, 10, log=True)
                return Lasso(alpha=alpha, random_state=2025), {'alpha': alpha}
                
            elif model_type == 'elastic_net':
                alpha = trial.suggest_float('alpha', 1e-4, 10, log=True)
                l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
                return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=2025), {'alpha': alpha, 'l1_ratio': l1_ratio}
                
            elif model_type == 'random_forest':
                n_estimators = trial.suggest_int('n_estimators', 50, 200)
                max_depth = trial.suggest_int('max_depth', 3, 15)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                return RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=2025
                ), {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split}
                
            elif model_type == 'xgboost' and HAS_XGBOOST:
                n_estimators = trial.suggest_int('n_estimators', 50, 200)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                max_depth = trial.suggest_int('max_depth', 2, 8)
                subsample = trial.suggest_float('subsample', 0.6, 1.0)
                return xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    random_state=2025
                ), {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth, 'subsample': subsample}
                
            elif model_type == 'lightgbm' and HAS_LIGHTGBM:
                n_estimators = trial.suggest_int('n_estimators', 50, 200)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                num_leaves = trial.suggest_int('num_leaves', 10, 100)
                subsample = trial.suggest_float('subsample', 0.6, 1.0)
                return lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    subsample=subsample,
                    random_state=2025,
                    verbose=-1
                ), {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'num_leaves': num_leaves, 'subsample': subsample}
                
            elif model_type == 'catboost' and HAS_CATBOOST:
                depth = trial.suggest_int('depth', 3, 8)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                iterations = trial.suggest_int('iterations', 50, 200)
                return cb.CatBoostRegressor(
                    depth=depth,
                    learning_rate=learning_rate,
                    iterations=iterations,
                    random_seed=2025,
                    verbose=False
                ), {'depth': depth, 'learning_rate': learning_rate, 'iterations': iterations}
                
            elif model_type == 'svr':
                C = trial.suggest_float('C', 1e-3, 100, log=True)
                gamma = trial.suggest_float('gamma', 1e-4, 10, log=True)
                return SVR(C=C, gamma=gamma), {'C': C, 'gamma': gamma}
                
            else:
                return None, {}
                
        except Exception as e:
            logger.warning(f"创建模型{model_type}失败: {e}")
            return None, {}
    
    def _train_deep_model(self, model_config, X_train, y_train, X_val, y_val):
        """训练深度学习模型"""
        if not HAS_TENSORFLOW:
            return -np.inf
            
        try:
            # 重塑数据用于LSTM/GRU
            if len(X_train.shape) == 2:
                X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
                X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
            
            model = model_config
            early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
            
            model.fit(X_train, y_train, 
                     validation_data=(X_val, y_val),
                     epochs=50, 
                     batch_size=32, 
                     verbose=0,
                     callbacks=[early_stopping])
            
            y_pred = model.predict(X_val, verbose=0)
            return -mean_squared_error(y_val, y_pred.flatten())
        except Exception as e:
            logger.warning(f"深度学习模型训练失败: {e}")
            return -np.inf
    
    def _get_default_params(self, model_type):
        """获取默认参数"""
        default_params = {
            'ridge': {'alpha': 1.0},
            'lasso': {'alpha': 1.0},
            'elastic_net': {'alpha': 1.0, 'l1_ratio': 0.5},
            'random_forest': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
            'xgboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.8},
            'lightgbm': {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31, 'subsample': 0.8},
            'catboost': {'depth': 6, 'learning_rate': 0.1, 'iterations': 100},
            'svr': {'C': 1.0, 'gamma': 'scale'},
            'mlp': {'hidden_layers': [64, 32], 'dropout': 0.2},
            'lstm': {'units': 50, 'dropout': 0.2},
            'gru': {'units': 50, 'dropout': 0.2}
        }
        return default_params.get(model_type, {})

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def cross_validate_models(self, best_models, features, returns):
        """
        交叉验证评估模型
        
        Args:
            best_models: 模型和参数列表
            features: 特征数据
            returns: 目标变量
            
        Returns:
            dict: 模型评估结果
        """
        logger.info("开始交叉验证评估模型...")
        
        results = {}
        tscv = TimeSeriesSplit(n_splits=3)  # 减少折数以加快速度
        scaler = StandardScaler()
        
        for model_type, params in best_models:
            logger.info(f"评估模型: {model_type}")
            
            # 检查模型可用性
            if not self._is_model_available(model_type):
                logger.warning(f"模型{model_type}不可用，跳过")
                continue
            
            try:
                scores = {
                    'mse': [],
                    'r2': [],
                    'mae': [],
                    'directional_accuracy': []
                }
                
                for train_idx, val_idx in tscv.split(features):
                    X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
                    y_train, y_val = returns.iloc[train_idx], returns.iloc[val_idx]
                    
                    # 数据标准化
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # 创建并训练模型
                    model = self._create_model(model_type, params)
                    
                    if model is None:
                        continue
                    
                    if model_type in ['lstm', 'gru', 'mlp'] and HAS_TENSORFLOW:
                        y_pred = self._train_and_predict_deep_model(
                            model, X_train_scaled, y_train, X_val_scaled, y_val
                        )
                    else:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_val_scaled)
                    
                    if y_pred is not None and len(y_pred) > 0:
                        # 计算评估指标
                        scores['mse'].append(mean_squared_error(y_val, y_pred))
                        scores['r2'].append(max(r2_score(y_val, y_pred), -1))  # 限制R2下限
                        scores['mae'].append(mean_absolute_error(y_val, y_pred))
                        
                        # 方向准确率
                        directional_acc = np.mean(np.sign(y_pred) == np.sign(y_val))
                        scores['directional_accuracy'].append(directional_acc)
                
                # 计算平均分数
                if scores['mse']:
                    results[model_type] = {
                        'mse_mean': np.mean(scores['mse']),
                        'mse_std': np.std(scores['mse']),
                        'r2_mean': np.mean(scores['r2']),
                        'r2_std': np.std(scores['r2']),
                        'mae_mean': np.mean(scores['mae']),
                        'mae_std': np.std(scores['mae']),
                        'directional_accuracy_mean': np.mean(scores['directional_accuracy']),
                        'directional_accuracy_std': np.std(scores['directional_accuracy']),
                        'params': params
                    }
                    
                    logger.info(f"{model_type} - R2: {results[model_type]['r2_mean']:.4f}, "
                              f"MSE: {results[model_type]['mse_mean']:.6f}")
                
            except Exception as e:
                logger.error(f"评估模型{model_type}时出错: {e}")
                continue
        
        self.evaluation_results = results
        return results
    
    def _is_model_available(self, model_type):
        """检查模型是否可用"""
        if model_type == 'xgboost':
            return HAS_XGBOOST
        elif model_type == 'lightgbm':
            return HAS_LIGHTGBM
        elif model_type == 'catboost':
            return HAS_CATBOOST
        elif model_type in ['mlp', 'lstm', 'gru']:
            return HAS_TENSORFLOW
        else:
            return True  # 基础模型总是可用
    
    def _create_model(self, model_type, params):
        """创建模型实例"""
        try:
            if model_type == 'ridge':
                return Ridge(**params, random_state=2025)
            elif model_type == 'lasso':
                return Lasso(**params, random_state=2025)
            elif model_type == 'elastic_net':
                return ElasticNet(**params, random_state=2025)
            elif model_type == 'random_forest':
                return RandomForestRegressor(**params, random_state=2025)
            elif model_type == 'xgboost' and HAS_XGBOOST:
                return xgb.XGBRegressor(**params, random_state=2025)
            elif model_type == 'lightgbm' and HAS_LIGHTGBM:
                return lgb.LGBMRegressor(**params, random_state=2025, verbose=-1)
            elif model_type == 'catboost' and HAS_CATBOOST:
                return cb.CatBoostRegressor(**params, random_seed=2025, verbose=False)
            elif model_type == 'svr':
                return SVR(**params)
            elif model_type in ['mlp', 'lstm', 'gru'] and HAS_TENSORFLOW:
                return self._create_deep_model(model_type, params)
            else:
                return None
        except Exception as e:
            logger.warning(f"创建模型{model_type}失败: {e}")
            return None
    
    def _create_deep_model(self, model_type, params):
        """创建深度学习模型"""
        if not HAS_TENSORFLOW:
            return None
            
        try:
            model = Sequential()
            
            if model_type == 'mlp':
                model.add(Dense(params.get('hidden_layers', [64, 32])[0], activation='relu', input_dim=10))
                model.add(Dropout(params.get('dropout', 0.2)))
                for units in params.get('hidden_layers', [64, 32])[1:]:
                    model.add(Dense(units, activation='relu'))
                    model.add(Dropout(params.get('dropout', 0.2)))
                model.add(Dense(1))
                
            elif model_type == 'lstm':
                model.add(LSTM(params.get('units', 50), input_shape=(1, 10), return_sequences=False))
                model.add(Dropout(params.get('dropout', 0.2)))
                model.add(Dense(1))
                
            elif model_type == 'gru':
                model.add(GRU(params.get('units', 50), input_shape=(1, 10), return_sequences=False))
                model.add(Dropout(params.get('dropout', 0.2)))
                model.add(Dense(1))
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            return model
        except Exception as e:
            logger.warning(f"创建深度学习模型{model_type}失败: {e}")
            return None
    
    def _train_and_predict_deep_model(self, model, X_train, y_train, X_val, y_val):
        """训练并预测深度学习模型"""
        if not HAS_TENSORFLOW or model is None:
            return None
            
        try:
            # 调整数据形状
            if len(X_train.shape) == 2 and 'lstm' in str(type(model)).lower():
                X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
                X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
            
            early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
            
            model.fit(X_train, y_train,
                     validation_data=(X_val, y_val),
                     epochs=50,
                     batch_size=32,
                     verbose=0,
                     callbacks=[early_stopping])
            
            y_pred = model.predict(X_val, verbose=0)
            return y_pred.flatten()
        except Exception as e:
            logger.warning(f"深度学习模型训练失败: {e}")
            return None
    
    def select_best_model(self, model_performance):
        """
        选择最佳模型
        
        Args:
            model_performance: 模型性能字典
            
        Returns:
            tuple: (最佳模型类型, 最佳模型参数, 综合得分)
        """
        logger.info("选择最佳模型...")
        
        if not model_performance:
            logger.error("没有可用的模型性能数据")
            return None, None, 0
        
        best_model = None
        best_params = None
        best_score = -np.inf
        
        for model_type, performance in model_performance.items():
            # 综合评分：R2占60%，方向准确率占30%，稳定性（低标准差）占10%
            r2_score = max(performance['r2_mean'], 0)  # 确保非负
            directional_score = performance['directional_accuracy_mean']
            stability_score = 1 - min(performance['r2_std'], 1)  # 限制标准差影响
            
            score = (0.6 * r2_score + 
                    0.3 * directional_score + 
                    0.1 * stability_score)
            
            logger.info(f"{model_type} 综合得分: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model_type
                best_params = performance['params']
        
        logger.info(f"最佳模型: {best_model}, 得分: {best_score:.4f}")
        return best_model, best_params, best_score

class IntelligentModelSelection:
    """智能模型选择主类"""
    
    def __init__(self):
        self.profiler = DataProfiler()
        self.recommender = ModelRecommender()
        self.optimizer = ModelOptimizer()
        self.evaluator = ModelEvaluator()
        self.results = {}
    
    def run(self, features, returns, n_trials=20):
        """
        运行完整的智能模型选择流程
        
        Args:
            features: 特征数据 DataFrame
            returns: 目标变量 Series
            n_trials: 超参数优化试验次数
            
        Returns:
            dict: 完整的选择结果
        """
        logger.info("=== 开始智能模型选择流程 ===")
        
        try:
            # 1. 数据特征分析
            data_profile = self.profiler.analyze_data_characteristics(features, returns)
            
            # 2. 模型推荐
            recommended_models = self.recommender.recommend_models_based_on_data(data_profile)
            
            if not recommended_models:
                logger.warning("没有推荐的模型，使用默认模型")
                recommended_models = ['ridge', 'random_forest']
            
            # 3. 超参数优化
            best_models = []
            for model_type in recommended_models[:5]:  # 限制最多5个模型以节省时间
                try:
                    best_params = self.optimizer.optimize_hyperparameters(
                        model_type, features, returns, n_trials
                    )
                    best_models.append((model_type, best_params))
                except Exception as e:
                    logger.warning(f"优化{model_type}参数失败: {e}")
                    # 使用默认参数
                    default_params = self.optimizer._get_default_params(model_type)
                    best_models.append((model_type, default_params))
            
            if not best_models:
                logger.error("没有成功配置的模型")
                return None
            
            # 4. 交叉验证评估
            model_performance = self.evaluator.cross_validate_models(best_models, features, returns)
            
            # 5. 最优模型选择
            best_model, best_params, best_score = self.evaluator.select_best_model(model_performance)
            
            # 保存结果
            self.results = {
                'data_profile': data_profile,
                'recommended_models': recommended_models,
                'model_performance': model_performance,
                'best_model': best_model,
                'best_params': best_params,
                'best_score': best_score,
                'timestamp': datetime.now().isoformat(),
                'available_packages': {
                    'xgboost': HAS_XGBOOST,
                    'lightgbm': HAS_LIGHTGBM,
                    'catboost': HAS_CATBOOST,
                    'optuna': HAS_OPTUNA,
                    'tensorflow': HAS_TENSORFLOW
                }
            }
            
            # 保存到文件
            self.save_results()
            
            logger.info("=== 智能模型选择流程完成 ===")
            return self.results
            
        except Exception as e:
            logger.error(f"智能模型选择流程失败: {e}")
            return None
    
    def save_results(self, filename=None):
        """保存结果到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'results/model_selection_results_{timestamp}.pkl'
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.results, f)
            logger.info(f"结果已保存到: {filename}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")

def demo_run():
    """演示运行"""
    logger.info("=== 智能模型选择系统演示 ===")
    
    # 打印可用包信息
    print("\n可用包检查:")
    print(f"XGBoost: {'✓' if HAS_XGBOOST else '✗'}")
    print(f"LightGBM: {'✓' if HAS_LIGHTGBM else '✗'}")
    print(f"CatBoost: {'✓' if HAS_CATBOOST else '✗'}")
    print(f"Optuna: {'✓' if HAS_OPTUNA else '✗'}")
    print(f"TensorFlow: {'✓' if HAS_TENSORFLOW else '✗'}")
    
    # 生成示例数据
    np.random.seed(2025)
    n_samples = 500
    n_features = 8
    
    # 创建有意义的特征
    X = np.random.randn(n_samples, n_features)
    
    # 创建目标变量（带有一些真实的关系）
    y = (0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + 
         0.1 * X[:, 3] * X[:, 4] + 0.05 * np.random.randn(n_samples))
    
    features = pd.DataFrame(X, columns=[f'factor_{i+1}' for i in range(n_features)])
    returns = pd.Series(y, name='returns')
    
    # 运行智能模型选择
    selector = IntelligentModelSelection()
    results = selector.run(features, returns, n_trials=10)
    
    if results:
        print(f"\n=== 模型选择结果 ===")
        print(f"最佳模型: {results['best_model']}")
        print(f"最佳参数: {results['best_params']}")
        print(f"综合得分: {results['best_score']:.4f}")
        print(f"\n推荐模型列表: {results['recommended_models']}")
        
        if results['model_performance']:
            print(f"\n=== 模型性能对比 ===")
            for model, perf in results['model_performance'].items():
                print(f"{model}: R2={perf['r2_mean']:.4f}±{perf['r2_std']:.4f}, "
                      f"方向准确率={perf['directional_accuracy_mean']:.4f}")
    else:
        print("模型选择失败")

if __name__ == "__main__":
    demo_run()