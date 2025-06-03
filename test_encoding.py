#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编码问题测试脚本
"""

import sys
import os
import subprocess
import locale

def test_encoding():
    """测试编码设置"""
    print("🔧 编码环境测试")
    print("=" * 30)
    
    # 测试系统编码
    try:
        print(f"系统首选编码: {locale.getpreferredencoding()}")
        print(f"系统默认编码: {sys.getdefaultencoding()}")
        print(f"文件系统编码: {sys.getfilesystemencoding()}")
    except Exception as e:
        print(f"获取编码信息失败: {e}")
    
    # 测试中文输出
    try:
        print("中文输出测试: ✅ 成功")
        print("特殊字符测试: 📊 🎯 🚀 ⚠️")
    except Exception as e:
        print(f"中文输出测试失败: {e}")
    
    # 测试subprocess
    try:
        print("\n测试subprocess执行...")
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run([sys.executable, "-c", "print('subprocess中文测试成功')"], 
                              capture_output=True, 
                              text=True,
                              encoding='utf-8',
                              errors='ignore',
                              env=env)
        
        if result.returncode == 0:
            print(f"✅ subprocess输出: {result.stdout.strip()}")
        else:
            print(f"❌ subprocess失败: {result.stderr}")
            
    except Exception as e:
        print(f"❌ subprocess测试失败: {e}")
    
    print("\n🎉 编码测试完成！")

if __name__ == "__main__":
    # Windows系统设置控制台编码
    if sys.platform == 'win32':
        try:
            os.system('chcp 65001 >nul 2>&1')
        except:
            pass
    
    test_encoding()