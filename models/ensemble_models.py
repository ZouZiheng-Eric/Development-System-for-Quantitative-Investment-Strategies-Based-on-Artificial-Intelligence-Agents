#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成模型系统
实现多种集成学习方法，提高预测性能和稳定性
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import lightgbm as lgb

try:
    import catboost as cb
except ImportError:
    cb = None

try:
    from sklearn.neural_network import MLPRegressor
except ImportError:
    MLPRegressor = None

import logging
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ensemble_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StackingEnsemble(BaseEstimator, RegressorMixin):
    """Stacking集成模型"""
    
    def __init__(self, base_models, meta_model=None, cv_folds=5):
        """
        初始化Stacking集成模型
        
        Args:
            base_models: 基础模型列表
            meta_model: 元学习器，默认为Ridge回归
            cv_folds: 交叉验证折数
        """
        self.base_models = base_models
        self.meta_model = meta_model if meta_model else Ridge(alpha=1.0)
        self.cv_folds = cv_folds
        self.fitted_base_models = []
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        训练Stacking模型
        
        Args:
            X: 特征数据
            y: 目标变量
        """
        logger.info(f"开始训练Stacking集成模型，包含{len(self.base_models)}个基础模型")
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # 生成meta特征
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for i, (model_name, model) in enumerate(self.base_models.items()):
            logger.info(f"训练基础模型: {model_name}")
            
            fold_predictions = np.zeros(len(X))
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # 训练基础模型
                model_clone = self._clone_model(model)
                model_clone.fit(X_train, y_train)
                
                # 预测验证集
                val_predictions = model_clone.predict(X_val)
                fold_predictions[val_idx] = val_predictions
            
            meta_features[:, i] = fold_predictions
        
        # 训练最终基础模型（在全部数据上）
        self.fitted_base_models = []
        for model_name, model in self.base_models.items():
            model_clone = self._clone_model(model)
            model_clone.fit(X, y)
            self.fitted_base_models.append((model_name, model_clone))
        
        # 训练元学习器
        self.meta_model.fit(meta_features, y)
        
        self.is_fitted = True
        logger.info("Stacking集成模型训练完成")
        return self
    
    def predict(self, X):
        """
        使用Stacking模型进行预测
        
        Args:
            X: 特征数据
            
        Returns:
            预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        # 获取基础模型预测
        base_predictions = np.zeros((len(X), len(self.fitted_base_models)))
        
        for i, (model_name, model) in enumerate(self.fitted_base_models):
            base_predictions[:, i] = model.predict(X)
        
        # 元学习器预测
        final_predictions = self.meta_model.predict(base_predictions)
        
        return final_predictions
    
    def _clone_model(self, model):
        """克隆模型"""
        from sklearn.base import clone
        try:
            return clone(model)
        except:
            # 对于一些不支持clone的模型，重新创建
            return type(model)(**model.get_params() if hasattr(model, 'get_params') else {})

class WeightedEnsemble(BaseEstimator, RegressorMixin):
    """加权集成模型"""
    
    def __init__(self, models, weight_method='performance'):
        """
        初始化加权集成模型
        
        Args:
            models: 模型字典
            weight_method: 权重计算方法 ('equal', 'performance', 'inverse_error')
        """
        self.models = models
        self.weight_method = weight_method
        self.weights = None
        self.fitted_models = []
        self.is_fitted = False
    
    def fit(self, X, y):
        """训练加权集成模型"""
        logger.info(f"开始训练加权集成模型，权重方法: {self.weight_method}")
        
        # 训练所有模型
        self.fitted_models = []
        for model_name, model in self.models.items():
            logger.info(f"训练模型: {model_name}")
            model_clone = self._clone_model(model)
            model_clone.fit(X, y)
            self.fitted_models.append((model_name, model_clone))
        
        # 计算权重
        self.weights = self._calculate_weights(X, y)
        
        self.is_fitted = True
        logger.info(f"加权集成模型训练完成，权重: {dict(zip([name for name, _ in self.fitted_models], self.weights))}")
        return self
    
    def predict(self, X):
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        predictions = np.zeros((len(X), len(self.fitted_models)))
        
        for i, (model_name, model) in enumerate(self.fitted_models):
            predictions[:, i] = model.predict(X)
        
        # 加权平均
        weighted_predictions = np.average(predictions, axis=1, weights=self.weights)
        
        return weighted_predictions
    
    def _calculate_weights(self, X, y):
        """计算模型权重"""
        if self.weight_method == 'equal':
            return np.ones(len(self.fitted_models)) / len(self.fitted_models)
        
        elif self.weight_method == 'performance':
            # 基于交叉验证性能计算权重
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for model_name, model in self.fitted_models:
                model_scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model_clone = self._clone_model(model)
                    model_clone.fit(X_train, y_train)
                    y_pred = model_clone.predict(X_val)
                    
                    # 使用R2作为性能指标
                    score = r2_score(y_val, y_pred)
                    model_scores.append(max(score, 0))  # 确保非负
                
                scores.append(np.mean(model_scores))
            
            # 归一化权重
            scores = np.array(scores)
            weights = scores / (scores.sum() + 1e-8)
            return weights
        
        elif self.weight_method == 'inverse_error':
            # 基于误差的倒数计算权重
            errors = []
            
            for model_name, model in self.fitted_models:
                y_pred = model.predict(X)
                mse = mean_squared_error(y, y_pred)
                errors.append(mse + 1e-8)  # 避免除零
            
            weights = 1.0 / np.array(errors)
            weights = weights / weights.sum()
            return weights
        
        else:
            return np.ones(len(self.fitted_models)) / len(self.fitted_models)
    
    def _clone_model(self, model):
        """克隆模型"""
        from sklearn.base import clone
        try:
            return clone(model)
        except:
            return type(model)(**model.get_params() if hasattr(model, 'get_params') else {})

class DynamicEnsemble(BaseEstimator, RegressorMixin):
    """动态集成模型"""
    
    def __init__(self, models, window_size=252, rebalance_freq=21):
        """
        初始化动态集成模型
        
        Args:
            models: 模型字典
            window_size: 性能评估窗口大小
            rebalance_freq: 权重重新平衡频率
        """
        self.models = models
        self.window_size = window_size
        self.rebalance_freq = rebalance_freq
        self.fitted_models = []
        self.weight_history = []
        self.is_fitted = False
    
    def fit(self, X, y):
        """训练动态集成模型"""
        logger.info("开始训练动态集成模型")
        
        # 训练所有基础模型
        self.fitted_models = []
        for model_name, model in self.models.items():
            model_clone = self._clone_model(model)
            model_clone.fit(X, y)
            self.fitted_models.append((model_name, model_clone))
        
        self.is_fitted = True
        logger.info("动态集成模型训练完成")
        return self
    
    def predict(self, X):
        """动态预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        predictions = []
        
        for i in range(len(X)):
            # 获取当前时点的权重
            if i % self.rebalance_freq == 0 or i == 0:
                # 重新计算权重（基于历史表现）
                current_weights = self._calculate_dynamic_weights(X.iloc[:i+1] if i > 0 else X.iloc[:1])
            
            # 预测当前时点
            current_predictions = np.array([model.predict(X.iloc[i:i+1])[0] 
                                         for _, model in self.fitted_models])
            
            weighted_prediction = np.average(current_predictions, weights=current_weights)
            predictions.append(weighted_prediction)
        
        return np.array(predictions)
    
    def _calculate_dynamic_weights(self, X_history):
        """动态计算权重"""
        if len(X_history) < 10:  # 历史数据不足时使用等权重
            return np.ones(len(self.fitted_models)) / len(self.fitted_models)
        
        # 基于最近的表现计算权重
        recent_window = min(self.window_size, len(X_history))
        X_recent = X_history.tail(recent_window)
        
        # 这里简化处理，实际应用中需要更复杂的逻辑
        return np.ones(len(self.fitted_models)) / len(self.fitted_models)
    
    def _clone_model(self, model):
        """克隆模型"""
        from sklearn.base import clone
        try:
            return clone(model)
        except:
            return type(model)(**model.get_params() if hasattr(model, 'get_params') else {})

class EnsembleModelManager:
    """集成模型管理器"""
    
    def __init__(self):
        self.base_models = {}
        self.ensemble_models = {}
        self.results = {}
    
    def add_base_model(self, name, model):
        """添加基础模型"""
        self.base_models[name] = model
        logger.info(f"添加基础模型: {name}")
    
    def create_default_models(self):
        """创建默认的基础模型集合"""
        logger.info("创建默认基础模型集合...")
        
        # 线性模型
        self.add_base_model('Ridge', Ridge(alpha=1.0, random_state=2025))
        self.add_base_model('Lasso', Lasso(alpha=1.0, random_state=2025))
        self.add_base_model('ElasticNet', ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=2025))
        
        # 树模型
        self.add_base_model('RandomForest', RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=2025
        ))
        self.add_base_model('XGBoost', xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=2025
        ))
        self.add_base_model('LightGBM', lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, num_leaves=31, 
            random_state=2025, verbose=-1
        ))
        
        # CatBoost（如果可用）
        if cb is not None:
            self.add_base_model('CatBoost', cb.CatBoostRegressor(
                iterations=100, learning_rate=0.1, depth=6, 
                random_seed=2025, verbose=False
            ))
        
        # 神经网络（如果可用）
        if MLPRegressor is not None:
            self.add_base_model('MLP', MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=500, random_state=2025
            ))
    
    def create_ensemble_models(self):
        """创建集成模型"""
        logger.info("创建集成模型...")
        
        if len(self.base_models) < 2:
            logger.warning("基础模型数量不足，无法创建集成模型")
            return
        
        # Stacking集成
        self.ensemble_models['Stacking'] = StackingEnsemble(
            base_models=self.base_models.copy(),
            meta_model=Ridge(alpha=1.0),
            cv_folds=3
        )
        
        # 等权重集成
        self.ensemble_models['EqualWeight'] = WeightedEnsemble(
            models=self.base_models.copy(),
            weight_method='equal'
        )
        
        # 性能加权集成
        self.ensemble_models['PerformanceWeight'] = WeightedEnsemble(
            models=self.base_models.copy(),
            weight_method='performance'
        )
        
        # 误差倒数加权集成
        self.ensemble_models['InverseErrorWeight'] = WeightedEnsemble(
            models=self.base_models.copy(),
            weight_method='inverse_error'
        )
        
        # 动态集成
        self.ensemble_models['Dynamic'] = DynamicEnsemble(
            models=self.base_models.copy(),
            window_size=252,
            rebalance_freq=21
        )
        
        logger.info(f"创建了{len(self.ensemble_models)}个集成模型")
    
    def train_and_evaluate(self, X, y, test_size=0.3):
        """训练和评估所有模型"""
        logger.info("开始训练和评估所有模型...")
        
        # 分割数据（时间序列分割）
        split_point = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # 训练和评估基础模型
        logger.info("评估基础模型...")
        for name, model in self.base_models.items():
            try:
                model_clone = self._clone_model(model)
                model_clone.fit(X_train_scaled, y_train)
                y_pred = model_clone.predict(X_test_scaled)
                
                self.results[name] = {
                    'type': 'base',
                    'model': model_clone,
                    'r2': r2_score(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'predictions': y_pred,
                    'y_true': y_test.values
                }
                
                logger.info(f"{name} - R2: {self.results[name]['r2']:.4f}, MSE: {self.results[name]['mse']:.6f}")
                
            except Exception as e:
                logger.error(f"训练基础模型{name}失败: {e}")
                continue
        
        # 训练和评估集成模型
        logger.info("评估集成模型...")
        for name, ensemble in self.ensemble_models.items():
            try:
                ensemble.fit(X_train_scaled, y_train)
                y_pred = ensemble.predict(X_test_scaled)
                
                self.results[f'Ensemble_{name}'] = {
                    'type': 'ensemble',
                    'model': ensemble,
                    'r2': r2_score(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'predictions': y_pred,
                    'y_true': y_test.values
                }
                
                logger.info(f"Ensemble_{name} - R2: {self.results[f'Ensemble_{name}']['r2']:.4f}, "
                          f"MSE: {self.results[f'Ensemble_{name}']['mse']:.6f}")
                
            except Exception as e:
                logger.error(f"训练集成模型{name}失败: {e}")
                continue
        
        logger.info("模型训练和评估完成")
        return self.results
    
    def get_performance_summary(self):
        """获取性能汇总"""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'Model': name,
                'Type': result['type'],
                'R2': result['r2'],
                'MSE': result['mse'],
                'RMSE': np.sqrt(result['mse'])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('R2', ascending=False)
        
        return summary_df
    
    def plot_performance_comparison(self, save_path='results/'):
        """绘制性能对比图"""
        if not self.results:
            logger.warning("没有结果数据用于绘图")
            return
        
        summary_df = self.get_performance_summary()
        
        # 创建对比图
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # R2对比
        colors = ['lightblue' if t == 'base' else 'lightcoral' for t in summary_df['Type']]
        bars1 = axes[0].bar(range(len(summary_df)), summary_df['R2'], color=colors)
        axes[0].set_title('R² 性能对比')
        axes[0].set_ylabel('R²')
        axes[0].set_xticks(range(len(summary_df)))
        axes[0].set_xticklabels(summary_df['Model'], rotation=45, ha='right')
        
        # 添加数值标签
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # MSE对比
        bars2 = axes[1].bar(range(len(summary_df)), summary_df['MSE'], color=colors)
        axes[1].set_title('MSE 对比 (越小越好)')
        axes[1].set_ylabel('MSE')
        axes[1].set_xticks(range(len(summary_df)))
        axes[1].set_xticklabels(summary_df['Model'], rotation=45, ha='right')
        
        # 添加数值标签
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightblue', label='基础模型'),
                          Patch(facecolor='lightcoral', label='集成模型')]
        fig.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_path}ensemble_performance_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"性能对比图已保存: {filename}")
    
    def save_best_model(self, save_path='models/'):
        """保存最佳模型"""
        if not self.results:
            logger.warning("没有结果数据")
            return
        
        # 找到最佳模型
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
        best_model = self.results[best_model_name]['model']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_path}best_ensemble_model_{timestamp}.pkl"
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'model': best_model,
                    'model_name': best_model_name,
                    'performance': {
                        'r2': self.results[best_model_name]['r2'],
                        'mse': self.results[best_model_name]['mse']
                    },
                    'timestamp': datetime.now().isoformat()
                }, f)
            
            logger.info(f"最佳模型已保存: {filename}")
            logger.info(f"最佳模型: {best_model_name}, R2: {self.results[best_model_name]['r2']:.4f}")
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
    
    def _clone_model(self, model):
        """克隆模型"""
        from sklearn.base import clone
        try:
            return clone(model)
        except:
            return type(model)(**model.get_params() if hasattr(model, 'get_params') else {})

def demo_ensemble():
    """演示集成模型功能"""
    logger.info("=== 集成模型系统演示 ===")
    
    # 生成示例数据
    np.random.seed(2025)
    n_samples = 1000
    n_features = 10
    
    # 创建有意义的特征和目标变量
    X = np.random.randn(n_samples, n_features)
    y = (0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + 
         0.1 * X[:, 3] * X[:, 4] + 0.02 * np.random.randn(n_samples))
    
    features = pd.DataFrame(X, columns=[f'factor_{i+1}' for i in range(n_features)])
    returns = pd.Series(y, name='returns')
    
    # 创建集成模型管理器
    manager = EnsembleModelManager()
    
    # 创建基础模型
    manager.create_default_models()
    
    # 创建集成模型
    manager.create_ensemble_models()
    
    # 训练和评估
    results = manager.train_and_evaluate(features, returns, test_size=0.3)
    
    # 显示结果
    summary = manager.get_performance_summary()
    print("\n=== 模型性能汇总 ===")
    print(summary.to_string(index=False))
    
    # 绘制对比图
    manager.plot_performance_comparison()
    
    # 保存最佳模型
    manager.save_best_model()
    
    # 显示最佳集成模型
    ensemble_results = {k: v for k, v in results.items() if v['type'] == 'ensemble'}
    if ensemble_results:
        best_ensemble = max(ensemble_results.keys(), key=lambda x: ensemble_results[x]['r2'])
        print(f"\n最佳集成模型: {best_ensemble}")
        print(f"R2: {ensemble_results[best_ensemble]['r2']:.4f}")
        print(f"MSE: {ensemble_results[best_ensemble]['mse']:.6f}")

if __name__ == "__main__":
    demo_ensemble()