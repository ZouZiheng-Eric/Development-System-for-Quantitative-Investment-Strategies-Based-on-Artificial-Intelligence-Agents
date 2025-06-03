#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块三快速测试脚本
检查系统是否能正常运行
"""

import os
import sys
import numpy as np
import pandas as pd

def check_basic_imports():
    """检查基础依赖包"""
    print("=== 检查基础依赖包 ===")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (请安装: pip install {pip_name})")
            missing_packages.append(pip_name)
    
    return missing_packages

def check_optional_imports():
    """检查可选依赖包"""
    print("\n=== 检查可选依赖包 ===")
    
    optional_packages = {
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'catboost': 'catboost',
        'optuna': 'optuna',
        'tensorflow': 'tensorflow'
    }
    
    available_count = 0
    
    for package, pip_name in optional_packages.items():
        try:
            __import__(package)
            print(f"✓ {package}")
            available_count += 1
        except ImportError:
            print(f"○ {package} (可选: pip install {pip_name})")
    
    print(f"\n可选包可用数量: {available_count}/{len(optional_packages)}")
    return available_count

def test_model_selection():
    """测试智能模型选择"""
    print("\n=== 测试智能模型选择 ===")
    
    try:
        # 添加模块路径
        sys.path.insert(0, 'final_project/models')
        
        from intelligent_model_selection import IntelligentModelSelection
        
        # 生成测试数据
        np.random.seed(2025)
        n_samples = 200
        n_features = 5
        
        X = np.random.randn(n_samples, n_features)
        y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.1 * np.random.randn(n_samples)
        
        features = pd.DataFrame(X, columns=[f'factor_{i+1}' for i in range(n_features)])
        returns = pd.Series(y, name='returns')
        
        # 运行模型选择（快速测试）
        selector = IntelligentModelSelection()
        results = selector.run(features, returns, n_trials=3)
        
        if results:
            print("✓ 智能模型选择测试通过")
            print(f"  最佳模型: {results['best_model']}")
            print(f"  综合得分: {results['best_score']:.4f}")
            return True
        else:
            print("✗ 智能模型选择测试失败")
            return False
            
    except Exception as e:
        print(f"✗ 智能模型选择测试出错: {e}")
        return False

def test_model_evaluation():
    """测试模型评估"""
    print("\n=== 测试模型评估 ===")
    
    try:
        from model_evaluation import PerformanceMetrics
        
        # 生成测试数据
        np.random.seed(2025)
        y_true = np.random.randn(100)
        y_pred = y_true + 0.1 * np.random.randn(100)
        
        # 测试回归指标
        metrics = PerformanceMetrics.regression_metrics(y_true, y_pred)
        
        if 'r2' in metrics and 'mse' in metrics:
            print("✓ 模型评估测试通过")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  MSE: {metrics['mse']:.6f}")
            return True
        else:
            print("✗ 模型评估测试失败")
            return False
            
    except Exception as e:
        print(f"✗ 模型评估测试出错: {e}")
        return False

def test_ensemble_models():
    """测试集成模型"""
    print("\n=== 测试集成模型 ===")
    
    try:
        from ensemble_models import EnsembleModelManager
        
        # 生成测试数据
        np.random.seed(2025)
        n_samples = 200
        n_features = 5
        
        X = np.random.randn(n_samples, n_features)
        y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.1 * np.random.randn(n_samples)
        
        features = pd.DataFrame(X, columns=[f'factor_{i+1}' for i in range(n_features)])
        returns = pd.Series(y, name='returns')
        
        # 创建集成模型管理器
        manager = EnsembleModelManager()
        manager.create_default_models()
        manager.create_ensemble_models()
        
        print("✓ 集成模型创建测试通过")
        print(f"  基础模型数量: {len(manager.base_models)}")
        print(f"  集成模型数量: {len(manager.ensemble_models)}")
        return True
        
    except Exception as e:
        print(f"✗ 集成模型测试出错: {e}")
        return False

def create_directories():
    """创建必要的目录"""
    directories = [
        'final_project/models',
        'final_project/results', 
        'final_project/reports',
        'final_project/logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """主测试函数"""
    print("模块三系统测试")
    print("=" * 50)
    
    # 创建目录
    create_directories()
    
    # 检查依赖包
    missing_basic = check_basic_imports()
    available_optional = check_optional_imports()
    
    if missing_basic:
        print(f"\n❌ 缺少必需的依赖包: {', '.join(missing_basic)}")
        print("请先安装必需的依赖包后再运行测试")
        return False
    
    print(f"\n✅ 基础依赖包检查通过")
    
    # 运行功能测试
    test_results = []
    
    test_results.append(test_model_selection())
    test_results.append(test_model_evaluation()) 
    test_results.append(test_ensemble_models())
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    test_names = ["智能模型选择", "模型评估", "集成模型"]
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n通过率: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！系统可以正常运行")
        print("\n下一步:")
        print("  1. 运行完整测试: python final_project/run_module3.py")
        print("  2. 查看运行指南: final_project/models/模块三运行指南.md")
    else:
        print(f"\n⚠️  {total_tests - passed_tests}个测试失败")
        print("请检查错误信息并安装相应的依赖包")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)