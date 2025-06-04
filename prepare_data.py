"""
数据准备脚本
从data_202410目录整合股票数据，创建适合因子策略的数据格式
支持用户选择不同的时间间隔：分钟、小时、每日
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def show_timeframe_menu():
    """显示时间间隔选择菜单"""
    print("\n" + "="*50)
    print("📊 请选择数据分析的时间间隔:")
    print("="*50)
    print("1. 🕐 分钟数据 (1分钟K线) - 适合短线交易策略")
    print("2. ⏰ 小时数据 (1小时K线) - 适合中短线策略") 
    print("3. 📅 每日数据 (日K线) - 适合中长线策略")
    print("4. 🔧 自定义时间间隔")
    print("-"*50)
    
    while True:
        try:
            choice = input("请输入选项号码 (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return choice
            else:
                print("❌ 无效选项，请输入1-4之间的数字")
        except KeyboardInterrupt:
            print("\n👋 用户中断，程序退出")
            exit()

def get_timeframe_config(choice):
    """根据用户选择获取时间间隔配置"""
    if choice == '1':
        return {
            'resample_rule': '1T',  # 1分钟
            'name': '分钟数据',
            'min_periods': 1000,  # 至少需要1000个分钟数据点
            'description': '1分钟K线数据，适合高频交易和短线策略'
        }
    elif choice == '2':
        return {
            'resample_rule': '1H',  # 1小时
            'name': '小时数据', 
            'min_periods': 100,   # 至少需要100个小时数据点
            'description': '1小时K线数据，适合日内和短期策略'
        }
    elif choice == '3':
        return {
            'resample_rule': '1D',  # 1天
            'name': '每日数据',
            'min_periods': 30,    # 至少需要30个交易日数据点
            'description': '日K线数据，适合中长期策略分析'
        }
    elif choice == '4':
        return get_custom_timeframe()

def get_custom_timeframe():
    """获取自定义时间间隔配置"""
    print("\n🔧 自定义时间间隔设置:")
    print("支持的时间单位:")
    print("  T 或 min: 分钟")
    print("  H: 小时") 
    print("  D: 天")
    print("示例: 5T (5分钟), 30T (30分钟), 4H (4小时), 1D (1天)")
    
    while True:
        try:
            rule = input("请输入时间间隔规则: ").strip().upper()
            if not rule:
                print("❌ 时间间隔不能为空")
                continue
                
            # 简单验证
            if any(unit in rule for unit in ['T', 'MIN', 'H', 'D']):
                name = input("请输入此时间间隔的描述名称: ").strip()
                if not name:
                    name = f"{rule}数据"
                    
                min_periods = 50  # 默认最小周期数
                try:
                    periods_input = input(f"请输入最小数据点数量 (默认{min_periods}): ").strip()
                    if periods_input:
                        min_periods = int(periods_input)
                except ValueError:
                    print("⚠️ 使用默认最小数据点数量")
                
                return {
                    'resample_rule': rule,
                    'name': name,
                    'min_periods': min_periods,
                    'description': f'自定义{rule}时间间隔数据'
                }
            else:
                print("❌ 无效的时间间隔格式，请重新输入")
        except KeyboardInterrupt:
            print("\n👋 用户中断，程序退出")
            exit()

def load_and_merge_data(stock_code='000001'):
    """
    加载并合并OHLCV数据
    """
    print(f"正在处理股票代码: {stock_code}")
    
    # 查找可用的日期文件
    data_dir = 'data_202410'
    if not os.path.exists(data_dir):
        raise ValueError(f"数据目录 {data_dir} 不存在")
        
    files = os.listdir(data_dir)
    dates = sorted(list(set([f[:8] for f in files if f.endswith('.pkl')])))
    
    print(f"找到以下日期的数据: {dates}")
    
    all_data = []
    
    for date in dates:
        try:
            # 加载当日的OHLCV数据
            close_file = f'{data_dir}/{date}_Close.pkl'
            open_file = f'{data_dir}/{date}_Open.pkl'
            high_file = f'{data_dir}/{date}_High.pkl'
            low_file = f'{data_dir}/{date}_Low.pkl'
            volume_file = f'{data_dir}/{date}_Volume.pkl'
            amount_file = f'{data_dir}/{date}_Amount.pkl'
            
            # 检查文件是否存在
            required_files = [close_file, open_file, high_file, low_file, volume_file, amount_file]
            if not all(os.path.exists(f) for f in required_files):
                print(f"跳过日期 {date}，缺少必要文件")
                continue
            
            # 读取数据
            close_data = pd.read_pickle(close_file)
            open_data = pd.read_pickle(open_file)
            high_data = pd.read_pickle(high_file)
            low_data = pd.read_pickle(low_file)
            volume_data = pd.read_pickle(volume_file)
            amount_data = pd.read_pickle(amount_file)
            
            # 检查股票代码是否存在
            if stock_code not in close_data.columns:
                print(f"股票代码 {stock_code} 不存在，可用的前10个代码:")
                print(list(close_data.columns[:10]))
                # 如果指定股票不存在，使用第一个可用的股票
                stock_code = close_data.columns[0]
                print(f"改用股票代码: {stock_code}")
            
            # 提取指定股票的数据
            daily_data = pd.DataFrame({
                'Open': open_data[stock_code],
                'High': high_data[stock_code],
                'Low': low_data[stock_code],
                'Close': close_data[stock_code],
                'Volume': volume_data[stock_code],
                'Amount': amount_data[stock_code]
            })
            
            # 去除空值
            daily_data = daily_data.dropna()
            
            if len(daily_data) > 0:
                all_data.append(daily_data)
                print(f"成功加载 {date} 的数据，共 {len(daily_data)} 条记录")
        
        except Exception as e:
            print(f"处理日期 {date} 时出错: {e}")
            continue
    
    if not all_data:
        raise ValueError("没有成功加载任何数据")
    
    # 合并所有日期的数据
    combined_data = pd.concat(all_data, axis=0)
    combined_data = combined_data.sort_index()
    
    print(f"合并后数据形状: {combined_data.shape}")
    print(f"时间范围: {combined_data.index[0]} 到 {combined_data.index[-1]}")
    
    return combined_data, stock_code

def resample_data(data, timeframe_config):
    """
    根据指定的时间间隔重采样数据
    """
    resample_rule = timeframe_config['resample_rule']
    name = timeframe_config['name']
    
    print(f"将数据重采样为{name}({resample_rule})...")
    
    # 重采样数据
    resampled_data = data.resample(resample_rule).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min', 
        'Close': 'last',
        'Volume': 'sum',
        'Amount': 'sum'
    }).dropna()
    
    print(f"{name}数据形状: {resampled_data.shape}")
    print(f"数据时间范围: {resampled_data.index[0]} 到 {resampled_data.index[-1]}")
    
    return resampled_data

def add_technical_indicators(data, timeframe_config):
    """
    添加技术指标计算所需的基础数据
    根据时间间隔调整指标参数
    """
    print("添加技术指标...")
    
    # 计算收益率
    data['returns'] = data['Close'].pct_change()
    
    # 根据时间间隔调整技术指标参数
    resample_rule = timeframe_config['resample_rule']
    
    if 'T' in resample_rule or 'MIN' in resample_rule:
        # 分钟数据 - 使用较短的周期
        short_ma, medium_ma, long_ma = 5, 15, 30
        volume_ma = 10
        price_position_window = 20
    elif 'H' in resample_rule:
        # 小时数据 - 使用中等周期
        short_ma, medium_ma, long_ma = 3, 8, 20
        volume_ma = 8
        price_position_window = 15
    else:
        # 日数据 - 使用标准周期
        short_ma, medium_ma, long_ma = 5, 10, 20
        volume_ma = 10
        price_position_window = 10
    
    # 计算移动平均线
    data[f'SMA_{short_ma}'] = data['Close'].rolling(window=short_ma).mean()
    data[f'SMA_{medium_ma}'] = data['Close'].rolling(window=medium_ma).mean()
    data[f'SMA_{long_ma}'] = data['Close'].rolling(window=long_ma).mean()
    
    # 价格相对位置
    data['Price_Position'] = (data['Close'] - data['Low'].rolling(window=price_position_window).min()) / \
                           (data['High'].rolling(window=price_position_window).max() - 
                            data['Low'].rolling(window=price_position_window).min() + 1e-8)
    
    # 成交量移动平均
    data['Volume_MA'] = data['Volume'].rolling(window=volume_ma).mean()
    
    # 价格变化率
    data['Price_Change'] = data['Close'].pct_change()
    
    # 波动率（根据时间间隔调整年化因子）
    if 'T' in resample_rule or 'MIN' in resample_rule:
        # 分钟数据：一年约有252*390分钟
        volatility_factor = np.sqrt(252 * 390)
    elif 'H' in resample_rule:
        # 小时数据：一年约有252*6.5小时
        volatility_factor = np.sqrt(252 * 6.5)
    else:
        # 日数据：一年252个交易日
        volatility_factor = np.sqrt(252)
    
    data['Volatility'] = data['returns'].rolling(window=20).std() * volatility_factor
    
    print(f"✅ 已添加适用于{timeframe_config['name']}的技术指标")
    
    return data

def validate_data(data, timeframe_config):
    """验证数据质量和数量"""
    min_periods = timeframe_config['min_periods']
    name = timeframe_config['name']
    
    print(f"\n📋 验证{name}数据质量...")
    
    if len(data) < min_periods:
        print(f"⚠️ 警告: {name}数据量较少 ({len(data)} < {min_periods})，可能影响分析效果")
        response = input("是否继续使用此数据？(y/n): ").strip().lower()
        if response != 'y':
            return False
    
    # 检查数据完整性
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        print(f"⚠️ 数据中存在缺失值:")
        for col, count in missing_data[missing_data > 0].items():
            print(f"   {col}: {count} 个缺失值")
    
    # 检查价格数据的合理性
    if (data['High'] < data['Low']).any():
        print("❌ 错误: 发现高价低于低价的异常数据")
        return False
    
    if (data['High'] < data['Close']).any() or (data['Low'] > data['Close']).any():
        print("❌ 错误: 发现收盘价超出高低价范围的异常数据") 
        return False
    
    print("✅ 数据验证通过")
    return True

def main():
    """
    主函数
    """
    print("🚀 数据准备系统启动")
    print("本系统支持多种时间间隔的数据处理，适配不同的交易策略需求")
    
    try:
        # 用户选择时间间隔
        choice = show_timeframe_menu()
        timeframe_config = get_timeframe_config(choice)
        
        print(f"\n✅ 已选择: {timeframe_config['name']}")
        print(f"📝 说明: {timeframe_config['description']}")
        
        # 加载和合并数据
        print("\n📥 开始加载原始数据...")
        combined_data, stock_code = load_and_merge_data()
        
        # 重采样数据
        print(f"\n🔄 重采样为{timeframe_config['name']}...")
        resampled_data = resample_data(combined_data, timeframe_config)
        
        # 验证数据
        if not validate_data(resampled_data, timeframe_config):
            print("❌ 数据验证失败，程序退出")
            return
        
        # 添加技术指标
        print("\n📊 添加技术指标...")
        final_data = add_technical_indicators(resampled_data, timeframe_config)
        
        # 创建目录
        os.makedirs('data', exist_ok=True)
        
        # 生成文件名（包含时间间隔信息）
        timeframe_suffix = timeframe_config['resample_rule'].replace('T', 'min').replace('H', 'hour').replace('D', 'day')
        output_file = f'data/data_202410.pkl'
        
        # 保存数据
        final_data.to_pickle(output_file)
        
        # 保存配置信息
        config_file = f'data/config_{timeframe_suffix}.json'
        import json
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timeframe_config': timeframe_config,
                'stock_code': stock_code,
                'data_shape': final_data.shape,
                'time_range': {
                    'start': str(final_data.index[0]),
                    'end': str(final_data.index[-1])
                },
                'created_at': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        
        # 显示结果摘要
        print("\n" + "="*60)
        print("🎉 数据准备完成!")
        print("="*60)
        print(f"📈 股票代码: {stock_code}")
        print(f"⏱️ 时间间隔: {timeframe_config['name']} ({timeframe_config['resample_rule']})")
        print(f"💾 数据文件: {output_file}")
        print(f"🔧 配置文件: {config_file}")
        print(f"📐 数据形状: {final_data.shape}")
        print(f"📅 时间范围: {final_data.index[0]} 到 {final_data.index[-1]}")
        
        # 数据摘要
        print(f"\n📊 数据摘要:")
        summary = final_data[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
        print(summary)
        
        # 收益率统计
        print(f"\n💰 收益率统计:")
        returns = final_data['returns'].dropna()
        if len(returns) > 0:
            print(f"   平均收益率: {returns.mean():.6f}")
            print(f"   收益率标准差: {returns.std():.6f}")
            print(f"   年化波动率: {final_data['Volatility'].iloc[-1]:.4f}" if 'Volatility' in final_data.columns else "")
            print(f"   收益率范围: {returns.min():.4f} 到 {returns.max():.4f}")
        
        print(f"\n📋 前5行数据预览:")
        print(final_data.head())
        
        print(f"\n💡 提示: 您现在可以使用此{timeframe_config['name']}运行相应的量化策略")
        
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 