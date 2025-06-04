"""
多因子组合策略综合性能对比
运行并比较5、10、15、20因子组合策略的表现，生成详细的对比报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 导入各个多因子策略
from multi_factor_5 import run_multi_factor_5_strategy
from multi_factor_10 import run_multi_factor_10_strategy
from multi_factor_15 import run_multi_factor_15_strategy
from multi_factor_20 import run_multi_factor_20_strategy

class MultiFactorPerformanceComparison:
    """多因子组合策略综合性能对比器"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.results = {}
        self.comparison_summary = None
        
    def run_all_strategies(self):
        """运行所有多因子策略"""
        print("=" * 60)
        print("开始运行多因子组合策略综合对比")
        print("=" * 60)
        
        # 定义要测试的组合方法（选择较优的方法）
        test_methods = ['equal_weight', 'ic_weight', 'dynamic_weight']
        
        strategies = {
            '5_factor': run_multi_factor_5_strategy,
            '10_factor': run_multi_factor_10_strategy,
            '15_factor': run_multi_factor_15_strategy,
            '20_factor': run_multi_factor_20_strategy
        }
        
        for strategy_name, strategy_func in strategies.items():
            print(f"\n运行 {strategy_name} 策略...")
            try:
                strategy_results = strategy_func(self.data_path, test_methods)
                self.results[strategy_name] = strategy_results
                print(f"{strategy_name} 策略运行完成")
            except Exception as e:
                print(f"{strategy_name} 策略运行失败: {e}")
                self.results[strategy_name] = {}
                continue
        
        print("\n所有策略运行完成！")
    
    def generate_comprehensive_comparison(self):
        """生成综合对比报告"""
        if not self.results:
            print("没有可用的结果数据")
            return
        
        print("\n" + "=" * 60)
        print("生成综合对比报告")
        print("=" * 60)
        
        # 创建综合对比数据框
        comparison_data = []
        
        for strategy_name, strategy_results in self.results.items():
            if not strategy_results:
                continue
                
            for method_name, method_result in strategy_results.items():
                if 'metrics' not in method_result:
                    continue
                    
                metrics = method_result['metrics']
                
                # 计算综合得分
                comprehensive_score = self._calculate_comprehensive_score(metrics)
                
                comparison_data.append({
                    'Strategy': strategy_name,
                    'Method': method_name,
                    'Annual_Return': metrics.get('annual_return', 0),
                    'Annual_Volatility': metrics.get('annual_volatility', 0),
                    'Sharpe_Ratio': metrics.get('sharpe_ratio', 0),
                    'Sortino_Ratio': metrics.get('sortino_ratio', 0),
                    'Max_Drawdown': metrics.get('max_drawdown', 0),
                    'Win_Rate': metrics.get('win_rate', 0),
                    'Information_Ratio': metrics.get('information_ratio', 0),
                    'Calmar_Ratio': metrics.get('calmar_ratio', 0),
                    'Turnover': metrics.get('turnover', 0),
                    'Comprehensive_Score': comprehensive_score
                })
        
        if not comparison_data:
            print("没有有效的对比数据")
            return
        
        self.comparison_summary = pd.DataFrame(comparison_data)
        
        # 保存详细对比结果
        self.comparison_summary.to_csv('./results/backtest_results/comprehensive_comparison.csv', index=False)
        
        # 打印对比摘要
        self._print_comparison_summary()
        
        # 生成可视化图表
        self._generate_comparison_charts()
        
        return self.comparison_summary
    
    def _calculate_comprehensive_score(self, metrics):
        """计算综合得分"""
        # 权重设置
        weights = {
            'sharpe_ratio': 0.25,
            'sortino_ratio': 0.20,
            'calmar_ratio': 0.20,
            'annual_return': 0.15,
            'win_rate': 0.10,
            'information_ratio': 0.10
        }
        
        score = 0
        for metric, weight in weights.items():
            value = metrics.get(metric, 0)
            if not np.isnan(value) and not np.isinf(value):
                score += weight * value
        
        return score
    
    def _print_comparison_summary(self):
        """打印对比摘要"""
        if self.comparison_summary is None or self.comparison_summary.empty:
            return
        
        print("\n【综合性能排名】（按综合得分排序）")
        print("-" * 80)
        
        # 按综合得分排序
        sorted_results = self.comparison_summary.sort_values('Comprehensive_Score', ascending=False)
        
        print(f"{'排名':<4} {'策略':<12} {'方法':<15} {'年化收益':<8} {'夏普比率':<8} {'最大回撤':<8} {'综合得分':<8}")
        print("-" * 80)
        
        for i, (_, row) in enumerate(sorted_results.head(10).iterrows(), 1):
            print(f"{i:<4} {row['Strategy']:<12} {row['Method']:<15} "
                  f"{row['Annual_Return']:<8.4f} {row['Sharpe_Ratio']:<8.4f} "
                  f"{row['Max_Drawdown']:<8.4f} {row['Comprehensive_Score']:<8.4f}")
        
        print("\n【各策略最佳方法】")
        print("-" * 50)
        
        for strategy in sorted_results['Strategy'].unique():
            strategy_data = sorted_results[sorted_results['Strategy'] == strategy]
            best_method = strategy_data.iloc[0]
            print(f"{strategy}: {best_method['Method']} "
                  f"(综合得分: {best_method['Comprehensive_Score']:.4f})")
        
        # 统计分析
        print("\n【统计分析】")
        print("-" * 30)
        print(f"平均年化收益率: {sorted_results['Annual_Return'].mean():.4f}")
        print(f"平均夏普比率: {sorted_results['Sharpe_Ratio'].mean():.4f}")
        print(f"平均最大回撤: {sorted_results['Max_Drawdown'].mean():.4f}")
        print(f"平均胜率: {sorted_results['Win_Rate'].mean():.4f}")
    
    def _generate_comparison_charts(self):
        """生成对比图表"""
        if self.comparison_summary is None or self.comparison_summary.empty:
            return
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 夏普比率对比
        strategy_sharpe = self.comparison_summary.groupby('Strategy')['Sharpe_Ratio'].max()
        axes[0, 0].bar(strategy_sharpe.index, strategy_sharpe.values, color='skyblue')
        axes[0, 0].set_title('各策略最佳夏普比率对比')
        axes[0, 0].set_ylabel('夏普比率')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 年化收益率对比
        strategy_return = self.comparison_summary.groupby('Strategy')['Annual_Return'].max()
        axes[0, 1].bar(strategy_return.index, strategy_return.values, color='lightgreen')
        axes[0, 1].set_title('各策略最佳年化收益率对比')
        axes[0, 1].set_ylabel('年化收益率')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 最大回撤对比
        strategy_drawdown = self.comparison_summary.groupby('Strategy')['Max_Drawdown'].min()
        axes[0, 2].bar(strategy_drawdown.index, abs(strategy_drawdown.values), color='lightcoral')
        axes[0, 2].set_title('各策略最小最大回撤对比')
        axes[0, 2].set_ylabel('最大回撤')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. 胜率对比
        strategy_winrate = self.comparison_summary.groupby('Strategy')['Win_Rate'].max()
        axes[1, 0].bar(strategy_winrate.index, strategy_winrate.values, color='gold')
        axes[1, 0].set_title('各策略最佳胜率对比')
        axes[1, 0].set_ylabel('胜率')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. 综合得分对比
        strategy_score = self.comparison_summary.groupby('Strategy')['Comprehensive_Score'].max()
        axes[1, 1].bar(strategy_score.index, strategy_score.values, color='plum')
        axes[1, 1].set_title('各策略最佳综合得分对比')
        axes[1, 1].set_ylabel('综合得分')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. 风险收益散点图
        for strategy in self.comparison_summary['Strategy'].unique():
            strategy_data = self.comparison_summary[self.comparison_summary['Strategy'] == strategy]
            axes[1, 2].scatter(strategy_data['Annual_Volatility'], strategy_data['Annual_Return'], 
                             label=strategy, alpha=0.7, s=60)
        
        axes[1, 2].set_xlabel('年化波动率')
        axes[1, 2].set_ylabel('年化收益率')
        axes[1, 2].set_title('风险收益散点图')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 确保目录存在
        os.makedirs('./results/backtest_results', exist_ok=True)
        save_path = './results/backtest_results/multi_factor_comprehensive_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图表释放内存
        
        print(f"对比图表已保存至: {save_path}")
    
    def generate_summary_report(self):
        """生成总结报告"""
        if self.comparison_summary is None or self.comparison_summary.empty:
            return
        
        print("\n" + "=" * 60)
        print("多因子组合策略总结报告")
        print("=" * 60)
        
        # 最佳策略推荐
        best_overall = self.comparison_summary.loc[self.comparison_summary['Comprehensive_Score'].idxmax()]
        
        print(f"\n【最佳策略推荐】")
        print(f"策略: {best_overall['Strategy']}")
        print(f"方法: {best_overall['Method']}")
        print(f"综合得分: {best_overall['Comprehensive_Score']:.4f}")
        print(f"年化收益率: {best_overall['Annual_Return']:.4f}")
        print(f"夏普比率: {best_overall['Sharpe_Ratio']:.4f}")
        print(f"最大回撤: {best_overall['Max_Drawdown']:.4f}")
        print(f"胜率: {best_overall['Win_Rate']:.4f}")
        
        # 策略特点分析
        print(f"\n【策略特点分析】")
        
        strategy_analysis = {}
        for strategy in self.comparison_summary['Strategy'].unique():
            strategy_data = self.comparison_summary[self.comparison_summary['Strategy'] == strategy]
            strategy_analysis[strategy] = {
                'avg_sharpe': strategy_data['Sharpe_Ratio'].mean(),
                'avg_return': strategy_data['Annual_Return'].mean(),
                'avg_drawdown': strategy_data['Max_Drawdown'].mean(),
                'best_method': strategy_data.loc[strategy_data['Comprehensive_Score'].idxmax(), 'Method']
            }
        
        for strategy, analysis in strategy_analysis.items():
            print(f"\n{strategy}:")
            print(f"  - 平均夏普比率: {analysis['avg_sharpe']:.4f}")
            print(f"  - 平均年化收益: {analysis['avg_return']:.4f}")
            print(f"  - 平均最大回撤: {analysis['avg_drawdown']:.4f}")
            print(f"  - 最佳方法: {analysis['best_method']}")
        
        # 生成Markdown报告
        self._generate_markdown_report()
    
    def _generate_markdown_report(self):
        """生成Markdown格式报告"""
        if self.comparison_summary is None:
            return
        
        report_content = []
        report_content.append("# 多因子组合策略综合评估报告\n")
        report_content.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 执行摘要
        best_overall = self.comparison_summary.loc[self.comparison_summary['Comprehensive_Score'].idxmax()]
        report_content.append("## 执行摘要\n")
        report_content.append(f"本次评估测试了5因子、10因子、15因子和20因子四种组合策略，")
        report_content.append(f"共计{len(self.comparison_summary)}个策略-方法组合。\n")
        report_content.append(f"**最佳组合**: {best_overall['Strategy']} + {best_overall['Method']}")
        report_content.append(f"（综合得分: {best_overall['Comprehensive_Score']:.4f}）\n")
        
        # 性能排名表
        report_content.append("## 性能排名\n")
        sorted_results = self.comparison_summary.sort_values('Comprehensive_Score', ascending=False)
        
        report_content.append("| 排名 | 策略 | 方法 | 年化收益率 | 夏普比率 | 最大回撤 | 综合得分 |")
        report_content.append("|------|------|------|------------|----------|----------|----------|")
        
        for i, (_, row) in enumerate(sorted_results.head(10).iterrows(), 1):
            report_content.append(f"| {i} | {row['Strategy']} | {row['Method']} | "
                                f"{row['Annual_Return']:.4f} | {row['Sharpe_Ratio']:.4f} | "
                                f"{row['Max_Drawdown']:.4f} | {row['Comprehensive_Score']:.4f} |")
        
        report_content.append("")
        
        # 策略建议
        report_content.append("## 策略建议\n")
        report_content.append("### 1. 因子数量选择\n")
        
        factor_analysis = self.comparison_summary.groupby('Strategy').agg({
            'Sharpe_Ratio': 'mean',
            'Annual_Return': 'mean',
            'Max_Drawdown': 'mean',
            'Comprehensive_Score': 'mean'
        }).round(4)
        
        best_strategy = factor_analysis['Comprehensive_Score'].idxmax()
        report_content.append(f"- **推荐**: {best_strategy}在综合表现上最优")
        report_content.append(f"- 平均综合得分: {factor_analysis.loc[best_strategy, 'Comprehensive_Score']:.4f}\n")
        
        report_content.append("### 2. 组合方法选择\n")
        method_analysis = self.comparison_summary.groupby('Method').agg({
            'Comprehensive_Score': 'mean'
        }).round(4)
        
        best_method = method_analysis['Comprehensive_Score'].idxmax()
        report_content.append(f"- **推荐**: {best_method}方法整体表现最佳")
        report_content.append(f"- 平均综合得分: {method_analysis.loc[best_method, 'Comprehensive_Score']:.4f}\n")
        
        # 风险提示
        report_content.append("## 风险提示\n")
        report_content.append("1. 本评估基于历史数据，未来表现可能有所不同")
        report_content.append("2. 所有策略均存在最大回撤风险，投资者应注意风险控制")
        report_content.append("3. 建议结合市场环境动态调整策略参数")
        report_content.append("4. 实盘应用时需考虑交易成本、流动性等因素\n")
        
        # 保存报告
        with open('./results/backtest_results/comprehensive_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print("详细报告已保存至: ./results/backtest_results/comprehensive_analysis_report.md")

def main():
    """主函数"""
    # 数据路径
    data_path = 'data/data_202410.pkl'
    
    try:
        # 创建对比器
        comparator = MultiFactorPerformanceComparison(data_path)
        
        # 运行所有策略
        comparator.run_all_strategies()
        
        # 生成综合对比
        comparison_df = comparator.generate_comprehensive_comparison()
        
        # 生成总结报告
        comparator.generate_summary_report()
        
        print("\n" + "=" * 60)
        print("多因子组合策略综合评估完成！")
        print("=" * 60)
        print("结果文件:")
        print("1. comprehensive_comparison.csv - 详细对比数据")
        print("2. multi_factor_comprehensive_comparison.png - 对比图表") 
        print("3. comprehensive_analysis_report.md - 分析报告")
        
    except Exception as e:
        print(f"综合评估运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 