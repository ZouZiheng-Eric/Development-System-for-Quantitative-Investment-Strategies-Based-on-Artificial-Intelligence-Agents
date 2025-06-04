import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys
import os
from datetime import datetime

class FactorEvaluator:
    """
    因子评估器
    用于批量评估所有单因子策略的表现
    """
    def __init__(self, data_path=None):
        # 如果没有指定数据路径，则使用data目录中的数据
        if data_path is None:
            self.data_path = 'data/data_202410.pkl'
        else:
            self.data_path = data_path
        self.factor_strategies = {}
        self.evaluation_results = []
        # 获取当前脚本所在目录
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
    def load_strategy_module(self, strategy_file):
        """
        动态加载策略模块
        """
        try:
            # 构建完整的文件路径
            full_path = os.path.join(self.script_dir, strategy_file)
            spec = importlib.util.spec_from_file_location("strategy", full_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"加载策略文件 {strategy_file} 失败: {e}")
            return None
    
    def run_single_strategy(self, strategy_name, strategy_file):
        """
        运行单个策略并收集结果
        """
        print(f"\n正在评估: {strategy_name}")
        
        try:
            # 动态加载策略模块
            module = self.load_strategy_module(strategy_file)
            if module is None:
                return None
                
            # 执行策略
            data_handler = module.DataHandler(self.data_path)
            
            # 根据策略文件调用对应的计算方法
            if 'factor_strategy_1.py' in strategy_file:
                data_handler.calculate_rsi()
            elif 'factor_strategy_2.py' in strategy_file:
                data_handler.calculate_macd()
            elif 'factor_strategy_3.py' in strategy_file:
                data_handler.calculate_bollinger_bands()
            elif 'factor_strategy_4.py' in strategy_file:
                data_handler.calculate_williams_r()
            elif 'factor_strategy_5.py' in strategy_file:
                data_handler.calculate_cci_factor()
            elif 'factor_strategy_6.py' in strategy_file:
                data_handler.calculate_price_volume_divergence_factor()
            elif 'factor_strategy_7.py' in strategy_file:
                data_handler.calculate_volume_anomaly_factor()
            elif 'factor_strategy_8.py' in strategy_file:
                data_handler.calculate_turnover_factor()
            elif 'factor_strategy_9.py' in strategy_file:
                data_handler.calculate_volatility_factor()
            elif 'factor_strategy_10.py' in strategy_file:
                data_handler.calculate_vwap_factor()
            else:
                print(f"未找到对应的计算方法在 {strategy_file}")
                return None
            
            # 动态找到策略类
            strategy_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    attr_name not in ['DataHandler', 'Performance'] and
                    not attr_name.startswith('_')):
                    # 检查是否是策略类
                    try:
                        # 检查类是否有generate_signals方法或者名称包含Strategy
                        if (hasattr(attr, '__init__') and 
                            ('Strategy' in attr_name or attr_name == 'Strategy')):
                            strategy_class = attr
                            print(f"找到策略类: {attr_name}")
                            break
                    except:
                        continue
            
            if strategy_class is None:
                print(f"无法找到策略类在文件 {strategy_file}")
                print(f"可用的类: {[name for name in dir(module) if isinstance(getattr(module, name), type)]}")
                return None
            
            # 执行策略
            strategy = strategy_class(data_handler.data)
            strategy.generate_signals()
            
            # 计算业绩
            perf = module.Performance(data_handler.data, strategy.positions)
            metrics = perf.calculate_metrics()
            
            # 收集结果
            result = {
                'strategy_name': strategy_name,
                'annual_return': metrics.get('annual_return', 0),
                'annual_volatility': metrics.get('annual_volatility', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'total_return': data_handler.data['cumulative_returns'].iloc[-1] if 'cumulative_returns' in data_handler.data.columns else 0,
                'num_trades': (strategy.positions.diff() != 0).sum(),
                'avg_holding_period': self._calculate_avg_holding_period(strategy.positions),
                'metrics': metrics
            }
            
            print(f"{strategy_name} 评估完成")
            return result
            
        except Exception as e:
            print(f"评估 {strategy_name} 时出错: {e}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def _calculate_avg_holding_period(self, positions):
        """
        计算平均持仓周期
        """
        if len(positions) == 0:
            return 0
            
        position_changes = positions.diff()
        trades = position_changes[position_changes != 0]
        
        if len(trades) <= 1:
            return len(positions)
            
        holding_periods = []
        current_start = None
        
        for i, pos in enumerate(positions):
            if i == 0 or positions.iloc[i] != positions.iloc[i-1]:
                if current_start is not None:
                    holding_periods.append(i - current_start)
                current_start = i
        
        # 处理最后一个持仓期
        if current_start is not None:
            holding_periods.append(len(positions) - current_start)
            
        return np.mean(holding_periods) if holding_periods else 0
    
    def evaluate_all_factors(self):
        """
        评估所有因子策略
        """
        print("开始批量评估所有因子策略...")
        
        # 定义所有策略文件
        strategy_files = {
            'RSI反转因子': 'factor_strategy_1.py',
            'MACD动量因子': 'factor_strategy_2.py',
            '布林带位置因子': 'factor_strategy_3.py',
            '威廉指标反转因子': 'factor_strategy_4.py',
            'CCI商品通道指数因子': 'factor_strategy_5.py',
            '价量背离因子': 'factor_strategy_6.py',
            '成交量异常因子': 'factor_strategy_7.py',
            '换手率因子': 'factor_strategy_8.py',
            '历史波动率因子': 'factor_strategy_9.py',
            'VWAP偏离因子': 'factor_strategy_10.py'
        }
        
        # 运行所有策略
        for strategy_name, strategy_file in strategy_files.items():
            # 构建完整的文件路径
            full_path = os.path.join(self.script_dir, strategy_file)
            if os.path.exists(full_path):
                result = self.run_single_strategy(strategy_name, strategy_file)
                if result:
                    self.evaluation_results.append(result)
            else:
                print(f"策略文件 {strategy_file} 不存在于 {self.script_dir}")
        
        print(f"\n批量评估完成！共成功评估 {len(self.evaluation_results)} 个策略")
        
    def generate_ranking_report(self):
        """
        生成因子排名报告
        """
        if not self.evaluation_results:
            print("没有评估结果，无法生成报告")
            return
            
        # 创建结果DataFrame
        df = pd.DataFrame(self.evaluation_results)
        
        # 按不同指标排名
        rankings = {
            'sharpe_ratio': df.sort_values('sharpe_ratio', ascending=False),
            'annual_return': df.sort_values('annual_return', ascending=False),
            'max_drawdown': df.sort_values('max_drawdown', ascending=False),  # 最大回撤越小越好
            'win_rate': df.sort_values('win_rate', ascending=False)
        }
        
        # 生成综合评分
        df['综合得分'] = self._calculate_composite_score(df)
        overall_ranking = df.sort_values('综合得分', ascending=False)
        
        # 打印报告
        print("\n" + "="*80)
        print("                     因子策略评估报告")
        print("="*80)
        
        print("\n1. 综合排名（基于综合得分）:")
        print("-" * 60)
        for i, (_, row) in enumerate(overall_ranking.iterrows(), 1):
            print(f"{i:2d}. {row['strategy_name']:20s} | 得分: {row['综合得分']:6.3f} | "
                  f"夏普: {row['sharpe_ratio']:6.3f} | 年化收益: {row['annual_return']:7.2%}")
        
        print("\n2. 夏普比率排名:")
        print("-" * 60)
        for i, (_, row) in enumerate(rankings['sharpe_ratio'].iterrows(), 1):
            print(f"{i:2d}. {row['strategy_name']:20s} | 夏普比率: {row['sharpe_ratio']:6.3f}")
        
        print("\n3. 年化收益率排名:")
        print("-" * 60)
        for i, (_, row) in enumerate(rankings['annual_return'].iterrows(), 1):
            print(f"{i:2d}. {row['strategy_name']:20s} | 年化收益: {row['annual_return']:7.2%}")
        
        print("\n4. 最大回撤排名（越小越好）:")
        print("-" * 60)
        for i, (_, row) in enumerate(rankings['max_drawdown'].iterrows(), 1):
            print(f"{i:2d}. {row['strategy_name']:20s} | 最大回撤: {row['max_drawdown']:7.2%}")
        
        print("\n5. 胜率排名:")
        print("-" * 60)
        for i, (_, row) in enumerate(rankings['win_rate'].iterrows(), 1):
            print(f"{i:2d}. {row['strategy_name']:20s} | 胜率: {row['win_rate']:7.2%}")
        
        # 保存详细报告到Excel
        self.save_excel_report(overall_ranking)
        
        return overall_ranking
    
    def _calculate_composite_score(self, df):
        """
        计算综合得分
        """
        # 标准化各个指标
        sharpe_norm = (df['sharpe_ratio'] - df['sharpe_ratio'].min()) / (df['sharpe_ratio'].max() - df['sharpe_ratio'].min() + 1e-8)
        return_norm = (df['annual_return'] - df['annual_return'].min()) / (df['annual_return'].max() - df['annual_return'].min() + 1e-8)
        drawdown_norm = 1 - (df['max_drawdown'] - df['max_drawdown'].max()) / (df['max_drawdown'].min() - df['max_drawdown'].max() + 1e-8)
        winrate_norm = (df['win_rate'] - df['win_rate'].min()) / (df['win_rate'].max() - df['win_rate'].min() + 1e-8)
        
        # 加权综合得分
        composite_score = (sharpe_norm * 0.4 + 
                          return_norm * 0.3 + 
                          drawdown_norm * 0.2 + 
                          winrate_norm * 0.1)
        
        return composite_score
    
    def save_excel_report(self, ranking_df):
        """
        保存详细报告到Excel文件
        """
        try:
            # 确保目录存在
            os.makedirs('./factors/results', exist_ok=True)
            
            # 准备详细数据
            detailed_data = []
            for _, row in ranking_df.iterrows():
                detailed_data.append({
                    '策略名称': row['strategy_name'],
                    '综合得分': round(row['综合得分'], 4),
                    '年化收益率': f"{row['annual_return']:.2%}",
                    '年化波动率': f"{row['annual_volatility']:.2%}",
                    '夏普比率': round(row['sharpe_ratio'], 4),
                    '最大回撤': f"{row['max_drawdown']:.2%}",
                    '胜率': f"{row['win_rate']:.2%}",
                    '总收益率': f"{row['total_return']:.2%}",
                    '交易次数': int(row['num_trades']),
                    '平均持仓周期': round(row['avg_holding_period'], 2)
                })
            
            # 创建DataFrame并保存
            report_df = pd.DataFrame(detailed_data)
            
            filename = f'./factors/results/factor_evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
            report_df.to_excel(filename, index=False, sheet_name='因子评估报告')
            
            print(f"\n详细报告已保存至: {filename}")
            
        except Exception as e:
            print(f"保存Excel报告时出错: {e}")
    
    def plot_comparison_chart(self, ranking_df):
        """
        绘制因子对比图表
        """
        # 确保目录存在
        os.makedirs('./factors/results', exist_ok=True)
        
        plt.figure(figsize=(20, 12))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 综合得分对比
        plt.subplot(2, 3, 1)
        strategy_names = [name[:8] + '...' if len(name) > 8 else name for name in ranking_df['strategy_name']]
        plt.bar(strategy_names, ranking_df['综合得分'], color='skyblue', alpha=0.7)
        plt.title('综合得分对比')
        plt.ylabel('综合得分')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 2. 夏普比率对比
        plt.subplot(2, 3, 2)
        plt.bar(strategy_names, ranking_df['sharpe_ratio'], color='lightgreen', alpha=0.7)
        plt.title('夏普比率对比')
        plt.ylabel('夏普比率')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. 年化收益率对比
        plt.subplot(2, 3, 3)
        plt.bar(strategy_names, ranking_df['annual_return']*100, color='orange', alpha=0.7)
        plt.title('年化收益率对比')
        plt.ylabel('年化收益率(%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 4. 最大回撤对比
        plt.subplot(2, 3, 4)
        plt.bar(strategy_names, ranking_df['max_drawdown']*100, color='red', alpha=0.7)
        plt.title('最大回撤对比')
        plt.ylabel('最大回撤(%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 5. 胜率对比
        plt.subplot(2, 3, 5)
        plt.bar(strategy_names, ranking_df['win_rate']*100, color='purple', alpha=0.7)
        plt.title('胜率对比')
        plt.ylabel('胜率(%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 6. 收益风险散点图
        plt.subplot(2, 3, 6)
        plt.scatter(ranking_df['annual_volatility']*100, ranking_df['annual_return']*100, 
                   s=100, alpha=0.7, c=ranking_df['综合得分'], cmap='viridis')
        plt.xlabel('年化波动率(%)')
        plt.ylabel('年化收益率(%)')
        plt.title('收益-风险散点图')
        plt.colorbar(label='综合得分')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # 修复：确保图表保存到正确的目录
        plt.savefig('./factors/results/factor_comparison_charts.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("对比图表已保存至: ./factors/results/factor_comparison_charts.png")

if __name__ == "__main__":
    print("开始因子策略批量评估...")
    
    # 创建评估器
    evaluator = FactorEvaluator()
    
    # 评估所有因子
    evaluator.evaluate_all_factors()
    
    # 生成排名报告
    if evaluator.evaluation_results:
        ranking_df = evaluator.generate_ranking_report()
        
        # 绘制对比图表
        evaluator.plot_comparison_chart(ranking_df)
        
        print("\n因子评估完成！请查看生成的报告和图表。")
    else:
        print("没有成功评估的策略，请检查策略文件。") 