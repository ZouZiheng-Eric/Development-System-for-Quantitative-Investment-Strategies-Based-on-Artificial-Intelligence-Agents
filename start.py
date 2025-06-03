#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块三快速启动脚本
提供用户友好的菜单选择界面
"""

import os
import sys
import subprocess

def setup_encoding():
    """设置编码环境"""
    try:
        if sys.platform == 'win32':
            # Windows系统设置控制台编码为UTF-8
            os.system('chcp 65001 >nul 2>&1')
            
        # 设置Python IO编码
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
            sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
    except:
        pass  # 静默处理编码设置失败

def print_banner():
    """打印欢迎横幅"""
    print("=" * 60)
    print("🎯 量化投资概论 - 模块三：智能模型构建系统")
    print("=" * 60)
    print("📊 功能包括:")
    print("   • 智能模型选择与优化")
    print("   • 多模型性能评估")  
    print("   • 集成学习模型构建")
    print("   • 自动报告生成")
    print("=" * 60)

def show_menu():
    """显示主菜单"""
    print("\n📋 请选择要执行的操作:")
    print("1. 🔧 检查并安装依赖包")
    print("2. 🚀 运行完整的模块三系统")
    print("3. 📊 仅运行智能模型选择")
    print("4. 🔗 仅运行集成模型训练")
    print("5. 📝 查看使用说明")
    print("6. 🚪 退出")
    print("-" * 30)

def check_basic_requirements():
    """检查基本环境"""
    required = ['pandas', 'numpy', 'sklearn']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def option_install_dependencies():
    """选项1：安装依赖包"""
    print("\n🔧 开始检查和安装依赖包...")
    try:
        # 使用subprocess执行安装脚本，避免编码问题
        import subprocess
        
        # 设置环境变量处理编码
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run([sys.executable, 'final_project/install_dependencies.py'], 
                              env=env,
                              encoding='utf-8',
                              errors='ignore')
        
        if result.returncode == 0:
            print("✅ 依赖包安装脚本执行完成")
        else:
            print("⚠️ 安装脚本执行可能遇到问题，但已尝试处理")
            
    except FileNotFoundError:
        print("❌ 安装脚本不存在")
        print("请手动安装: pip install pandas numpy scikit-learn matplotlib seaborn")
    except Exception as e:
        print(f"❌ 安装过程出错: {e}")
        print("💡 尝试手动执行: python final_project/install_dependencies.py")

def option_run_full_system():
    """选项2：运行完整系统"""
    print("\n🚀 启动完整的模块三系统...")
    
    missing = check_basic_requirements()
    if missing:
        print(f"❌ 缺少必需依赖包: {', '.join(missing)}")
        print("请先选择选项1安装依赖包")
        return
    
    try:
        # 使用subprocess执行主程序，避免编码问题
        import subprocess
        
        # 设置环境变量处理编码
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run([sys.executable, 'final_project/run_module3.py'], 
                              env=env,
                              encoding='utf-8',
                              errors='ignore')
        
        if result.returncode == 0:
            print("✅ 模块三系统执行完成")
        else:
            print("⚠️ 系统执行可能遇到问题，请检查日志")
            
    except FileNotFoundError:
        print("❌ 主程序文件不存在")
    except Exception as e:
        print(f"❌ 运行过程出错: {e}")
        print("💡 尝试手动执行: python final_project/run_module3.py")

def option_model_selection_only():
    """选项3：仅运行模型选择"""
    print("\n📊 启动智能模型选择...")
    
    missing = check_basic_requirements()
    if missing:
        print(f"❌ 缺少必需依赖包: {', '.join(missing)}")
        return
    
    try:
        sys.path.insert(0, 'final_project/models')
        from intelligent_model_selection import demo_run
        demo_run()
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
    except Exception as e:
        print(f"❌ 运行出错: {e}")

def option_ensemble_only():
    """选项4：仅运行集成模型"""
    print("\n🔗 启动集成模型训练...")
    
    missing = check_basic_requirements()
    if missing:
        print(f"❌ 缺少必需依赖包: {', '.join(missing)}")
        return
    
    try:
        sys.path.insert(0, 'final_project/models')
        from ensemble_models import demo_ensemble
        demo_ensemble()
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
    except Exception as e:
        print(f"❌ 运行出错: {e}")

def option_show_instructions():
    """选项5：显示使用说明"""
    print("\n📝 使用说明")
    print("=" * 40)
    print("🎯 模块三系统功能介绍:")
    print()
    print("1. 智能模型选择:")
    print("   • 基于数据特征自动推荐合适的模型")
    print("   • 支持线性模型、树模型、神经网络等")
    print("   • 自动超参数优化")
    print()
    print("2. 模型性能评估:")
    print("   • 多指标综合评估")
    print("   • 时间序列交叉验证")
    print("   • 可视化性能对比")
    print()
    print("3. 集成学习:")
    print("   • Stacking集成")
    print("   • 加权平均集成")
    print("   • 动态权重调整")
    print()
    print("📦 依赖包说明:")
    print("   必需: pandas, numpy, scikit-learn, matplotlib")
    print("   可选: xgboost, lightgbm, optuna (提升性能)")
    print("   高级: catboost, tensorflow (深度学习)")
    print()
    print("🚀 快速开始:")
    print("   1. 选择菜单选项1安装依赖包")
    print("   2. 选择菜单选项2运行完整系统")
    print("   3. 查看final_project/目录下的输出文件")
    print()
    print("❓ 常见问题:")
    print("   • 如果出现包缺失错误，请安装相应依赖包")
    print("   • 系统会自动跳过不可用的模型类型")
    print("   • 生成的文件保存在final_project/目录下")

def create_project_structure():
    """创建项目目录结构"""
    directories = [
        'final_project/models',
        'final_project/results',
        'final_project/reports',
        'final_project/logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """主函数"""
    # 设置编码环境
    setup_encoding()
    
    # 创建目录结构
    create_project_structure()
    
    # 显示欢迎信息
    print_banner()
    
    # 检查基本环境
    missing = check_basic_requirements()
    if missing:
        print(f"\n⚠️  检测到缺少基础依赖包: {', '.join(missing)}")
        print("💡 建议先选择选项1安装依赖包")
    else:
        print("\n✅ 基础环境检查通过")
    
    # 主循环
    while True:
        try:
            show_menu()
            choice = input("请输入选项号码 (1-6): ").strip()
            
            if choice == '1':
                option_install_dependencies()
            elif choice == '2':
                option_run_full_system()
            elif choice == '3':
                option_model_selection_only()
            elif choice == '4':
                option_ensemble_only()
            elif choice == '5':
                option_show_instructions()
            elif choice == '6':
                print("\n👋 谢谢使用，再见！")
                break
            else:
                print("❌ 无效选项，请输入1-6之间的数字")
            
            if choice != '6':
                input("\n按回车键继续...")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，程序退出")
            break
        except Exception as e:
            print(f"\n❌ 程序出错: {e}")
            input("按回车键继续...")

if __name__ == "__main__":
    main()