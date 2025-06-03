#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块三运行脚本（修复编码问题版本）
智能模型构建系统一键运行
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import logging
import subprocess
warnings.filterwarnings('ignore')

# 设置控制台编码
if sys.platform == "win32":
    import locale
    try:
        # 设置控制台编码为UTF-8
        os.system('chcp 65001 >nul')
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

def setup_logging():
    """设置日志配置，解决编码问题"""
    # 确保目录存在
    os.makedirs('final_project/logs', exist_ok=True)
    
    # 清除之前的logging配置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 创建自定义的文件处理器，强制使用UTF-8编码
    file_handler = logging.FileHandler(
        'final_project/logs/module3_run.log', 
        mode='w',  # 覆盖模式，避免追加乱码
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 配置root logger
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)
    
    return logging.getLogger(__name__)

def check_and_install_packages():
    """检查并安装必需的包，解决编码问题"""
    logger = logging.getLogger(__name__)
    logger.info("开始检查依赖包...")
    
    required_packages = {
        'pandas': 'pandas>=1.3.0',
        'numpy': 'numpy>=1.20.0', 
        'sklearn': 'scikit-learn>=1.0.0',
        'matplotlib': 'matplotlib>=3.3.0',
        'seaborn': 'seaborn>=0.11.0'
    }
    
    missing_packages = []
    
    # 检查必需包
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            logger.info(f"✓ {package} 已安装")
        except ImportError:
            logger.warning(f"✗ 缺少 {package}")
            missing_packages.append(pip_name.split('>=')[0])
    
    # 安装缺失的包
    if missing_packages:
        logger.info(f"正在安装缺失的包: {', '.join(missing_packages)}")
        try:
            # 使用subprocess安装包，避免编码问题
            cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                encoding='utf-8',
                errors='ignore'  # 忽略编码错误
            )
            
            if result.returncode == 0:
                logger.info("✓ 依赖包安装成功")
            else:
                logger.warning(f"包安装可能有问题，但继续运行: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"包安装失败: {e}，但程序将继续运行")
    
    # 检查可选包
    optional_packages = ['xgboost', 'lightgbm', 'optuna']
    available_optional = 0
    
    for package in optional_packages:
        try:
            __import__(package)
            logger.info(f"✓ 可选包 {package} 可用")
            available_optional += 1
        except ImportError:
            logger.info(f"○ 可选包 {package} 未安装")
    
    logger.info(f"可选包可用数量: {available_optional}/{len(optional_packages)}")
    return len(missing_packages) == 0

def create_directories():
    """创建必要的目录结构"""
    directories = [
        'final_project/models',
        'final_project/results',
        'final_project/reports', 
        'final_project/logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def generate_sample_data():
    """生成示例数据"""
    logger = logging.getLogger(__name__)
    logger.info("生成示例数据...")
    
    np.random.seed(2025)
    n_samples = 800
    n_features = 8
    
    # 生成特征数据
    X = np.random.randn(n_samples, n_features)
    
    # 生成目标变量（模拟真实的因子收益关系）
    y = (0.4 * X[:, 0] +           # 主要因子
         0.3 * X[:, 1] +           # 次要因子
         -0.2 * X[:, 2] +          # 反向因子
         0.15 * X[:, 3] * X[:, 4] + # 交互因子
         0.1 * np.sin(X[:, 5]) +   # 非线性因子
         0.02 * np.random.randn(n_samples))  # 噪声
    
    feature_names = [
        'momentum_factor', 'reversal_factor', 'volatility_factor',
        'volume_factor', 'sentiment_factor', 'macro_factor',
        'technical_factor', 'fundamental_factor'
    ]
    
    features = pd.DataFrame(X, columns=feature_names)
    returns = pd.Series(y, name='returns')
    
    logger.info(f"数据生成完成: {len(features)}行 x {len(features.columns)}列")
    return features, returns

def run_basic_model_comparison(features, returns):
    """运行基础模型对比（避免复杂的依赖）"""
    logger = logging.getLogger(__name__)
    logger.info("开始基础模型对比...")
    
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    
    # 准备数据
    split_point = int(len(features) * 0.7)
    X_train, X_test = features.iloc[:split_point], features.iloc[split_point:]
    y_train, y_test = returns.iloc[:split_point], returns.iloc[split_point:]
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 定义模型
    models = {
        'Ridge回归': Ridge(alpha=1.0, random_state=2025),
        'Lasso回归': Lasso(alpha=1.0, random_state=2025),
        '随机森林': RandomForestRegressor(n_estimators=100, random_state=2025)
    }
    
    results = {}
    
    # 训练和评估模型
    for name, model in models.items():
        try:
            logger.info(f"训练模型: {name}")
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # 计算指标
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # 方向准确率
            direction_accuracy = np.mean(np.sign(y_pred) == np.sign(y_test))
            
            results[name] = {
                'R2': r2,
                'MSE': mse,
                '方向准确率': direction_accuracy
            }
            
            logger.info(f"{name} - R2: {r2:.4f}, MSE: {mse:.6f}, 方向准确率: {direction_accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"模型 {name} 训练失败: {e}")
            continue
    
    return results

def create_simple_ensemble(features, returns, base_results):
    """创建简单的集成模型"""
    logger = logging.getLogger(__name__)
    logger.info("创建简单集成模型...")
    
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, VotingRegressor
    
    try:
        # 准备数据
        split_point = int(len(features) * 0.7)
        X_train, X_test = features.iloc[:split_point], features.iloc[split_point:]
        y_train, y_test = returns.iloc[:split_point], returns.iloc[split_point:]
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 创建集成模型
        estimators = [
            ('ridge', Ridge(alpha=1.0, random_state=2025)),
            ('rf', RandomForestRegressor(n_estimators=50, random_state=2025))
        ]
        
        ensemble = VotingRegressor(estimators=estimators)
        ensemble.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred = ensemble.predict(X_test_scaled)
        
        # 计算指标
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        direction_accuracy = np.mean(np.sign(y_pred) == np.sign(y_test))
        
        ensemble_result = {
            'R2': r2,
            'MSE': mse,
            '方向准确率': direction_accuracy
        }
        
        logger.info(f"集成模型 - R2: {r2:.4f}, MSE: {mse:.6f}, 方向准确率: {direction_accuracy:.4f}")
        
        return ensemble_result, ensemble
        
    except Exception as e:
        logger.error(f"集成模型创建失败: {e}")
        return None, None

def generate_simple_report(base_results, ensemble_result):
    """生成简单的报告"""
    logger = logging.getLogger(__name__)
    logger.info("生成总结报告...")
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f'final_project/reports/模块三简化报告_{timestamp}.txt'
        
        # 使用UTF-8编码写入文件
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("模块三：智能模型构建系统 - 简化执行报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 基础模型结果
            f.write("基础模型性能对比:\n")
            f.write("-" * 30 + "\n")
            
            for model_name, metrics in base_results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  R²: {metrics['R2']:.4f}\n")
                f.write(f"  MSE: {metrics['MSE']:.6f}\n")
                f.write(f"  方向准确率: {metrics['方向准确率']:.4f}\n")
            
            # 集成模型结果
            if ensemble_result:
                f.write(f"\n集成模型:\n")
                f.write(f"  R²: {ensemble_result['R2']:.4f}\n")
                f.write(f"  MSE: {ensemble_result['MSE']:.6f}\n")
                f.write(f"  方向准确率: {ensemble_result['方向准确率']:.4f}\n")
            
            # 找到最佳模型
            all_results = base_results.copy()
            if ensemble_result:
                all_results['集成模型'] = ensemble_result
            
            best_model = max(all_results.keys(), key=lambda x: all_results[x]['R2'])
            best_r2 = all_results[best_model]['R2']
            
            f.write(f"\n推荐模型: {best_model}\n")
            f.write(f"最佳R²: {best_r2:.4f}\n")
            
            f.write("\n总结:\n")
            f.write("- 成功完成了基础模型对比和集成模型构建\n")
            f.write("- 建议使用R²最高的模型进行实际应用\n")
            f.write("- 可以根据需要进一步优化模型参数\n")
        
        logger.info(f"报告已保存: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"报告生成失败: {e}")
        return None

def main():
    """主函数"""
    # 设置日志
    logger = setup_logging()
    
    print("🚀 模块三：智能模型构建系统（简化版）")
    print("=" * 50)
    print("解决编码问题，提供稳定的基础功能")
    print("=" * 50)
    
    start_time = datetime.now()
    
    try:
        # 1. 检查并安装依赖
        logger.info("步骤1: 检查依赖包")
        packages_ok = check_and_install_packages()
        
        # 2. 创建目录
        logger.info("步骤2: 创建目录结构")
        create_directories()
        
        # 3. 生成数据
        logger.info("步骤3: 生成示例数据")
        features, returns = generate_sample_data()
        
        # 4. 运行基础模型对比
        logger.info("步骤4: 运行基础模型对比")
        base_results = run_basic_model_comparison(features, returns)
        
        if not base_results:
            logger.error("基础模型运行失败")
            return False
        
        # 5. 创建集成模型
        logger.info("步骤5: 创建集成模型")
        ensemble_result, ensemble_model = create_simple_ensemble(features, returns, base_results)
        
        # 6. 生成报告
        logger.info("步骤6: 生成报告")
        report_path = generate_simple_report(base_results, ensemble_result)
        
        # 7. 显示结果
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 50)
        print("🎉 模块三执行完成!")
        print("=" * 50)
        print(f"⏱️  总耗时: {duration}")
        
        print("\n📊 模型性能排名:")
        all_results = base_results.copy()
        if ensemble_result:
            all_results['集成模型'] = ensemble_result
        
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['R2'], reverse=True)
        
        for i, (name, metrics) in enumerate(sorted_results, 1):
            print(f"  {i}. {name}: R²={metrics['R2']:.4f}, 方向准确率={metrics['方向准确率']:.4f}")
        
        best_model = sorted_results[0][0]
        print(f"\n🏆 推荐模型: {best_model}")
        
        print(f"\n📁 输出文件:")
        print(f"   - 运行日志: final_project/logs/module3_run.log")
        if report_path:
            print(f"   - 详细报告: {report_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        print(f"\n❌ 程序执行失败: {e}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ 程序正常结束")
            input("\n按回车键退出...")
        else:
            print("\n❌ 程序异常结束")
            input("\n按回车键退出...")
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断执行")
    except Exception as e:
        print(f"\n💥 程序崩溃: {e}")
        input("\n按回车键退出...")