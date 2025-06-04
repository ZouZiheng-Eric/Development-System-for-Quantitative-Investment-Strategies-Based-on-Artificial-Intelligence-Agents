#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块三依赖包安装脚本
自动安装所需的依赖包
"""

import subprocess
import sys
import os
import locale
from datetime import datetime

def get_system_encoding():
    """获取系统编码"""
    try:
        return locale.getpreferredencoding()
    except:
        return 'utf-8'

def install_package(package):
    """安装单个包"""
    try:
        print(f"正在安装 {package}...")
        
        # 设置环境变量以处理编码问题
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                              capture_output=True, 
                              text=True,
                              encoding='utf-8',
                              errors='ignore',  # 忽略编码错误
                              env=env)
        
        if result.returncode == 0:
            print(f"✅ {package} 安装成功")
            return True
        else:
            # 安全地处理错误输出
            error_msg = result.stderr if result.stderr else "未知错误"
            if isinstance(error_msg, bytes):
                try:
                    error_msg = error_msg.decode('utf-8', errors='ignore')
                except:
                    error_msg = str(error_msg)
            print(f"❌ {package} 安装失败: {error_msg}")
            return False
    except Exception as e:
        print(f"❌ {package} 安装出错: {e}")
        return False

def check_package(package):
    """检查包是否已安装"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def main():
    """主安装函数"""
    # 设置控制台编码
    try:
        if sys.platform == 'win32':
            # Windows系统设置控制台编码
            os.system('chcp 65001 >nul')  # 设置为UTF-8
    except:
        pass
    
    print("🔧 模块三依赖包自动安装")
    print("=" * 40)
    
    # 基础必需包
    required_packages = [
        "pandas",
        "numpy", 
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "scipy"
    ]
    
    # 可选包（建议安装）
    optional_packages = [
        "xgboost",
        "lightgbm", 
        "optuna"
    ]
    
    # 高级可选包
    advanced_packages = [
        "catboost",
        "tensorflow"
    ]
    
    success_count = 0
    total_count = 0
    
    # 安装必需包
    print("\n📦 安装基础必需包...")
    for package in required_packages:
        total_count += 1
        if check_package(package.replace('-', '_')):
            print(f"✅ {package} 已安装")
            success_count += 1
        else:
            if install_package(package):
                success_count += 1
    
    # 安装建议包
    print("\n🎯 安装建议包（提升性能）...")
    for package in optional_packages:
        total_count += 1
        if check_package(package):
            print(f"✅ {package} 已安装")
            success_count += 1
        else:
            if install_package(package):
                success_count += 1
    
    # 询问是否安装高级包
    print("\n🚀 高级包安装（可选）")
    print("以下包可以提供额外功能，但不是必需的：")
    for package in advanced_packages:
        print(f"  - {package}")
    
    try:
        install_advanced = input("\n是否安装高级包？(y/N): ").lower().strip()
        if install_advanced in ['y', 'yes']:
            for package in advanced_packages:
                total_count += 1
                if check_package(package):
                    print(f"✅ {package} 已安装")
                    success_count += 1
                else:
                    if install_package(package):
                        success_count += 1
    except KeyboardInterrupt:
        print("\n跳过高级包安装")
    
    # 显示结果
    print("\n" + "=" * 40)
    print("📊 安装结果:")
    print(f"   成功: {success_count}/{total_count}")
    print(f"   成功率: {success_count/total_count*100:.1f}%")
    
    if success_count >= len(required_packages):
        print("\n✅ 基础环境配置成功！")
        print("🚀 现在可以运行: python run_module3.py")
        
        # 保存安装记录 
        try:
            with open("installed_packages.txt", "w", encoding='utf-8') as f:
                f.write("已安装的Python包列表\n")
                f.write("=" * 30 + "\n")
                f.write(f"安装时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"成功安装: {success_count}个包\n")
                f.write(f"失败: {total_count - success_count}个包\n\n")
                
                f.write("成功安装的包:\n")
                for pkg in required_packages + optional_packages + advanced_packages:
                    if check_package(pkg.replace('-', '_')):
                        f.write(f"- {pkg}\n")
        
            print("📝 已安装包列表保存到: installed_packages.txt")
        except Exception as e:
            print(f"保存安装记录时出错: {e}")
    else:
        print("\n⚠️ 部分基础包安装失败")
        print("请手动安装失败的包或检查网络连接")

if __name__ == "__main__":
    main()