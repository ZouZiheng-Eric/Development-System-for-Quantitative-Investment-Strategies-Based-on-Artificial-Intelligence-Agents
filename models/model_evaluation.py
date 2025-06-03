 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估框架
提供全面的模型性能评估、对比分析和可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import logging

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('final_project/logs/model_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """性能指标计算器"""
    
    @staticmethod
    def regression_metrics(y_true, y_pred):
        """
        回归任务评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            dict: 各种回归指标
        """
        metrics = {}
        
        # 基础指标
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # 方向准确率
        direction_true = np.sign(y_true)
        direction_pred = np.sign(y_pred)
        metrics['directional_accuracy'] = np.mean(direction_true == direction_pred)
        
        # 信息比率相关指标
        residuals = y_true - y_pred
        metrics['tracking_error'] = np.std(residuals)
        if metrics['tracking_error'] > 0:
            metrics['information_ratio'] = np.mean(residuals) / metrics['tracking_error']
        else:
            metrics['information_ratio'] = 0
        
        # 分位数指标
        metrics['mean_abs_percentage_error'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # IC指标 (Information Coefficient)
        ic, p_value = stats.pearsonr(y_true, y_pred)
        metrics['ic'] = ic if not np.isnan(ic) else 0
        metrics['ic_p_value'] = p_value if not np.isnan(p_value) else 1
        
        # Rank IC
        rank_ic, rank_p_value = stats.spearmanr(y_true, y_pred)
        metrics['rank_ic'] = rank_ic if not np.isnan(rank_ic) else 0
        metrics['rank_ic_p_value'] = rank_p_value if not np.isnan(rank_p_value) else 1
        
        return metrics
    
    @staticmethod
    def classification_metrics(y_true, y_pred, y_pred_proba=None):
        """
        分类任务评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_pred_proba: 预测概率
            
        Returns:
            dict: 各种分类指标
        """
        metrics = {}
        
        # 基础分类指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 如果有概率预测，计算AUC等指标
        if y_pred_proba is not None:
            try:
                from sklearn.metrics import roc_auc_score, log_loss
                if len(np.unique(y_true)) == 2:  # 二分类
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            except ImportError:
                pass
        
        return metrics
    
    @staticmethod
    def trading_metrics(returns, positions=None, benchmark_returns=None):
        """
        交易相关指标
        
        Args:
            returns: 策略收益率序列
            positions: 仓位序列
            benchmark_returns: 基准收益率
            
        Returns:
            dict: 交易指标
        """
        metrics = {}
        
        if len(returns) == 0:
            return metrics
            
        # 收益指标
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = (1 + returns.mean()) ** 252 - 1
        metrics['volatility'] = returns.std() * np.sqrt(252)
        
        # 风险调整指标
        if metrics['volatility'] > 0:
            metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility']
        else:
            metrics['sharpe_ratio'] = 0
            
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        
        # 卡尔马比率
        if abs(metrics['max_drawdown']) > 1e-6:
            metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = 0
            
        # 胜率
        if len(returns) > 0:
            metrics['win_rate'] = np.mean(returns > 0)
        else:
            metrics['win_rate'] = 0
            
        # 平均盈亏比
        profits = returns[returns > 0]
        losses = returns[returns < 0]
        if len(losses) > 0 and len(profits) > 0:
            metrics['profit_loss_ratio'] = profits.mean() / abs(losses.mean())
        else:
            metrics['profit_loss_ratio'] = 0
            
        # 换手率（如果提供仓位数据）
        if positions is not None:
            position_changes = np.abs(np.diff(positions))
            metrics['turnover_rate'] = position_changes.mean()
        
        # 相对基准指标
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            excess_returns = returns - benchmark_returns
            metrics['excess_return'] = excess_returns.mean() * 252
            metrics['tracking_error'] = excess_returns.std() * np.sqrt(252)
            if metrics['tracking_error'] > 0:
                metrics['information_ratio'] = metrics['excess_return'] / metrics['tracking_error']
            else:
                metrics['information_ratio'] = 0
                
            # Beta
            if benchmark_returns.std() > 0:
                metrics['beta'] = np.cov(returns, benchmark_returns)[0, 1] / benchmark_returns.var()
            else:
                metrics['beta'] = 0
        
        return metrics

class ModelComparator:
    """模型比较器"""
    
    def __init__(self):
        self.comparison_results = {}
    
    def compare_models(self, model_results, metric_type='regression'):
        """
        比较多个模型的性能
        
        Args:
            model_results: 字典，键为模型名称，值为包含预测结果的字典
            metric_type: 'regression' 或 'classification'
            
        Returns:
            DataFrame: 模型比较结果表
        """
        logger.info(f"开始比较{len(model_results)}个模型...")
        
        comparison_data = []
        
        for model_name, results in model_results.items():
            try:
                y_true = results['y_true']
                y_pred = results['y_pred']
                
                if metric_type == 'regression':
                    metrics = PerformanceMetrics.regression_metrics(y_true, y_pred)
                elif metric_type == 'classification':
                    y_pred_proba = results.get('y_pred_proba', None)
                    metrics = PerformanceMetrics.classification_metrics(y_true, y_pred, y_pred_proba)
                else:
                    continue
                
                # 添加模型名称
                metrics['model'] = model_name
                
                # 如果有额外的指标，添加进去
                if 'additional_metrics' in results:
                    metrics.update(results['additional_metrics'])
                
                comparison_data.append(metrics)
                
            except Exception as e:
                logger.warning(f"比较模型{model_name}时出错: {e}")
                continue
        
        if not comparison_data:
            logger.error("没有有效的模型结果用于比较")
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('model')
        
        # 按主要指标排序
        if metric_type == 'regression':
            comparison_df = comparison_df.sort_values('r2', ascending=False)
        elif metric_type == 'classification':
            comparison_df = comparison_df.sort_values('f1', ascending=False)
        
        self.comparison_results = comparison_df
        logger.info("模型比较完成")
        return comparison_df
    
    def rank_models(self, weights=None):
        """
        对模型进行综合排名
        
        Args:
            weights: 指标权重字典
            
        Returns:
            Series: 模型排名
        """
        if self.comparison_results.empty:
            logger.error("没有比较结果用于排名")
            return pd.Series()
        
        df = self.comparison_results.copy()
        
        # 默认权重（针对回归任务）
        if weights is None:
            weights = {
                'r2': 0.3,
                'directional_accuracy': 0.3,
                'ic': 0.2,
                'sharpe_ratio': 0.1,
                'information_ratio': 0.1
            }
        
        # 标准化指标（0-1范围）
        normalized_df = pd.DataFrame()
        for col in df.columns:
            if col in weights:
                if col in ['mse', 'rmse', 'mae', 'max_drawdown']:  # 越小越好的指标
                    normalized_df[col] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
                else:  # 越大越好的指标
                    normalized_df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
        
        # 计算加权综合得分
        scores = pd.Series(0, index=df.index)
        for metric, weight in weights.items():
            if metric in normalized_df.columns:
                scores += weight * normalized_df[metric].fillna(0)
        
        # 排序
        rankings = scores.sort_values(ascending=False)
        
        logger.info("模型排名完成")
        return rankings

class TimeSeriesEvaluator:
    """时间序列模型评估器"""
    
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
    
    def rolling_window_evaluation(self, model, X, y, window_size=252, step_size=21):
        """
        滚动窗口评估
        
        Args:
            model: 训练好的模型
            X: 特征数据
            y: 目标变量
            window_size: 训练窗口大小
            step_size: 步长
            
        Returns:
            DataFrame: 滚动评估结果
        """
        logger.info("开始滚动窗口评估...")
        
        results = []
        
        for start_idx in range(window_size, len(X) - step_size, step_size):
            try:
                # 训练集
                train_start = start_idx - window_size
                train_end = start_idx
                X_train = X.iloc[train_start:train_end]
                y_train = y.iloc[train_start:train_end]
                
                # 测试集
                test_start = start_idx
                test_end = min(start_idx + step_size, len(X))
                X_test = X.iloc[test_start:test_end]
                y_test = y.iloc[test_start:test_end]
                
                # 训练模型
                model.fit(X_train, y_train)
                
                # 预测
                y_pred = model.predict(X_test)
                
                # 计算指标
                metrics = PerformanceMetrics.regression_metrics(y_test, y_pred)
                metrics['period_start'] = X_test.index[0] if hasattr(X_test.index[0], 'strftime') else test_start
                metrics['period_end'] = X_test.index[-1] if hasattr(X_test.index[-1], 'strftime') else test_end - 1
                
                results.append(metrics)
                
            except Exception as e:
                logger.warning(f"滚动窗口评估中出错: {e}")
                continue
        
        if not results:
            logger.error("滚动窗口评估失败")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        logger.info(f"滚动窗口评估完成，共{len(results_df)}个时间段")
        return results_df
    
    def stability_analysis(self, rolling_results):
        """
        稳定性分析
        
        Args:
            rolling_results: 滚动评估结果
            
        Returns:
            dict: 稳定性指标
        """
        if rolling_results.empty:
            return {}
        
        stability_metrics = {}
        
        # 收益稳定性
        r2_values = rolling_results['r2'].dropna()
        if len(r2_values) > 0:
            stability_metrics['r2_mean'] = r2_values.mean()
            stability_metrics['r2_std'] = r2_values.std()
            stability_metrics['r2_min'] = r2_values.min()
            stability_metrics['r2_max'] = r2_values.max()
            stability_metrics['r2_consistency'] = np.mean(r2_values > 0)  # 正R2的比例
        
        # 方向预测稳定性
        dir_acc_values = rolling_results['directional_accuracy'].dropna()
        if len(dir_acc_values) > 0:
            stability_metrics['dir_acc_mean'] = dir_acc_values.mean()
            stability_metrics['dir_acc_std'] = dir_acc_values.std()
            stability_metrics['dir_acc_consistency'] = np.mean(dir_acc_values > 0.5)
        
        # IC稳定性
        ic_values = rolling_results['ic'].dropna()
        if len(ic_values) > 0:
            stability_metrics['ic_mean'] = ic_values.mean()
            stability_metrics['ic_std'] = ic_values.std()
            stability_metrics['ic_positive_rate'] = np.mean(ic_values > 0)
            stability_metrics['ic_significant_rate'] = np.mean(rolling_results['ic_p_value'] < 0.05)
        
        return stability_metrics

class ModelEvaluationReport:
    """模型评估报告生成器"""
    
    def __init__(self):
        self.report_data = {}
    
    def generate_comprehensive_report(self, model_results, save_path='final_project/reports/'):
        """
        生成综合评估报告
        
        Args:
            model_results: 模型结果字典
            save_path: 保存路径
        """
        logger.info("开始生成模型评估报告...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 模型比较表
        comparator = ModelComparator()
        comparison_df = comparator.compare_models(model_results)
        
        if not comparison_df.empty:
            # 保存比较结果
            comparison_file = f"{save_path}model_comparison_{timestamp}.xlsx"
            comparison_df.to_excel(comparison_file)
            logger.info(f"模型比较表已保存: {comparison_file}")
            
            # 2. 生成可视化图表
            self._create_comparison_charts(comparison_df, save_path, timestamp)
            
            # 3. 模型排名
            rankings = comparator.rank_models()
            if not rankings.empty:
                print("\n=== 模型综合排名 ===")
                for i, (model, score) in enumerate(rankings.items(), 1):
                    print(f"{i}. {model}: {score:.4f}")
            
            # 4. 生成详细报告
            self._create_detailed_report(comparison_df, rankings, save_path, timestamp)
        
        logger.info("模型评估报告生成完成")
    
    def _create_comparison_charts(self, comparison_df, save_path, timestamp):
        """创建比较图表"""
        try:
            # 设置图表样式
            plt.style.use('seaborn-v0_8')
            
            # 1. 主要指标对比图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('模型性能对比', fontsize=16, fontweight='bold')
            
            # R2对比
            if 'r2' in comparison_df.columns:
                comparison_df['r2'].plot(kind='bar', ax=axes[0, 0], color='skyblue')
                axes[0, 0].set_title('R² 对比')
                axes[0, 0].set_ylabel('R²')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 方向准确率对比
            if 'directional_accuracy' in comparison_df.columns:
                comparison_df['directional_accuracy'].plot(kind='bar', ax=axes[0, 1], color='lightgreen')
                axes[0, 1].set_title('方向准确率对比')
                axes[0, 1].set_ylabel('准确率')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # IC对比
            if 'ic' in comparison_df.columns:
                comparison_df['ic'].plot(kind='bar', ax=axes[1, 0], color='orange')
                axes[1, 0].set_title('IC对比')
                axes[1, 0].set_ylabel('IC')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # MSE对比（越小越好）
            if 'mse' in comparison_df.columns:
                comparison_df['mse'].plot(kind='bar', ax=axes[1, 1], color='lightcoral')
                axes[1, 1].set_title('MSE对比 (越小越好)')
                axes[1, 1].set_ylabel('MSE')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            chart_file = f"{save_path}model_comparison_charts_{timestamp}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 雷达图
            self._create_radar_chart(comparison_df, save_path, timestamp)
            
            logger.info(f"比较图表已保存: {chart_file}")
            
        except Exception as e:
            logger.error(f"创建比较图表失败: {e}")
    
    def _create_radar_chart(self, comparison_df, save_path, timestamp):
        """创建雷达图"""
        try:
            from math import pi
            
            # 选择主要指标
            radar_metrics = ['r2', 'directional_accuracy', 'ic', 'information_ratio']
            available_metrics = [m for m in radar_metrics if m in comparison_df.columns]
            
            if len(available_metrics) < 3:
                return
            
            # 标准化数据到0-1范围
            radar_data = comparison_df[available_metrics].copy()
            for col in radar_data.columns:
                if col == 'mse':  # MSE越小越好
                    radar_data[col] = 1 - (radar_data[col] - radar_data[col].min()) / (radar_data[col].max() - radar_data[col].min() + 1e-8)
                else:
                    radar_data[col] = (radar_data[col] - radar_data[col].min()) / (radar_data[col].max() - radar_data[col].min() + 1e-8)
            
            # 创建雷达图
            angles = [n / float(len(available_metrics)) * 2 * pi for n in range(len(available_metrics))]
            angles += angles[:1]  # 闭合
            
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
            
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            for i, (model_name, values) in enumerate(radar_data.iterrows()):
                values = values.tolist()
                values += values[:1]  # 闭合
                
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=model_name, color=colors[i % len(colors)])
                ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(available_metrics, fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_title('模型性能雷达图', size=16, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            radar_file = f"{save_path}model_radar_chart_{timestamp}.png"
            plt.savefig(radar_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"雷达图已保存: {radar_file}")
            
        except Exception as e:
            logger.error(f"创建雷达图失败: {e}")
    
    def _create_detailed_report(self, comparison_df, rankings, save_path, timestamp):
        """创建详细报告"""
        try:
            report_content = []
            report_content.append("# 模型评估详细报告\n")
            report_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 执行摘要
            report_content.append("## 执行摘要\n")
            if not rankings.empty:
                best_model = rankings.index[0]
                best_score = rankings.iloc[0]
                report_content.append(f"- **最佳模型**: {best_model}\n")
                report_content.append(f"- **综合得分**: {best_score:.4f}\n")
                
                if best_model in comparison_df.index:
                    best_metrics = comparison_df.loc[best_model]
                    if 'r2' in best_metrics:
                        report_content.append(f"- **R²**: {best_metrics['r2']:.4f}\n")
                    if 'directional_accuracy' in best_metrics:
                        report_content.append(f"- **方向准确率**: {best_metrics['directional_accuracy']:.4f}\n")
            
            # 模型排名
            report_content.append("\n## 模型排名\n")
            for i, (model, score) in enumerate(rankings.items(), 1):
                report_content.append(f"{i}. **{model}**: {score:.4f}\n")
            
            # 详细指标
            report_content.append("\n## 详细性能指标\n")
            report_content.append(comparison_df.round(4).to_string())
            
            # 建议
            report_content.append("\n\n## 建议\n")
            if not rankings.empty:
                top_models = rankings.head(3).index.tolist()
                report_content.append(f"- 推荐使用前三名模型: {', '.join(top_models)}\n")
                report_content.append("- 建议进行集成学习以进一步提升性能\n")
                report_content.append("- 定期重新评估模型性能，确保稳定性\n")
            
            # 保存报告
            report_file = f"{save_path}model_evaluation_report_{timestamp}.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.writelines(report_content)
            
            logger.info(f"详细报告已保存: {report_file}")
            
        except Exception as e:
            logger.error(f"创建详细报告失败: {e}")

def demo_evaluation():
    """演示评估流程"""
    logger.info("=== 模型评估框架演示 ===")
    
    # 生成示例数据
    np.random.seed(2025)
    n_samples = 500
    
    # 模拟真实值
    y_true = np.random.randn(n_samples) * 0.02
    
    # 模拟不同模型的预测结果
    model_results = {
        'LightGBM': {
            'y_true': y_true,
            'y_pred': y_true + np.random.randn(n_samples) * 0.01,  # 较好的预测
        },
        'RandomForest': {
            'y_true': y_true,
            'y_pred': y_true + np.random.randn(n_samples) * 0.015,  # 中等预测
        },
        'LinearRegression': {
            'y_true': y_true,
            'y_pred': y_true + np.random.randn(n_samples) * 0.02,  # 较差的预测
        }
    }
    
    # 生成评估报告
    report_generator = ModelEvaluationReport()
    report_generator.generate_comprehensive_report(model_results)

if __name__ == "__main__":
    demo_evaluation()