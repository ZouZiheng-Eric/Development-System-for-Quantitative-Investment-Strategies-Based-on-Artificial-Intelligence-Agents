# 量化投资概论 - Final Project
## 基于 AI 智能体的量化投资策略开发系统

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub repo](https://img.shields.io/badge/GitHub-Repository-blue.svg)](https://github.com/ZouZiheng-Eric/Development-System-for-Quantitative-Investment-Strategies-Based-on-Artificial-Intelligence-Agents)

[![Stars](https://img.shields.io/github/stars/ZouZiheng-Eric/Development-System-for-Quantitative-Investment-Strategies-Based-on-Artificial-Intelligence-Agents?style=social)](https://github.com/ZouZiheng-Eric/Development-System-for-Quantitative-Investment-Strategies-Based-on-Artificial-Intelligence-Agents/stargazers)
[![Forks](https://img.shields.io/github/forks/ZouZiheng-Eric/Development-System-for-Quantitative-Investment-Strategies-Based-on-Artificial-Intelligence-Agents?style=social)](https://github.com/ZouZiheng-Eric/Development-System-for-Quantitative-Investment-Strategies-Based-on-Artificial-Intelligence-Agents/network/members)

[![GitHub Issues](https://img.shields.io/github/issues/ZouZiheng-Eric/Development-System-for-Quantitative-Investment-Strategies-Based-on-Artificial-Intelligence-Agents)](https://github.com/ZouZiheng-Eric/Development-System-for-Quantitative-Investment-Strategies-Based-on-Artificial-Intelligence-Agents/issues)
[![GitHub Release](https://img.shields.io/github/v/release/ZouZiheng-Eric/Development-System-for-Quantitative-Investment-Strategies-Based-on-Artificial-Intelligence-Agents)](https://github.com/ZouZiheng-Eric/Development-System-for-Quantitative-Investment-Strategies-Based-on-Artificial-Intelligence-Agents/releases)
[![GitHub Discussions](https://img.shields.io/github/discussions/ZouZiheng-Eric/Development-System-for-Quantitative-Investment-Strategies-Based-on-Artificial-Intelligence-Agents)](https://github.com/ZouZiheng-Eric/Development-System-for-Quantitative-Investment-Strategies-Based-on-Artificial-Intelligence-Agents/discussions)

[![GitHub](https://img.shields.io/badge/GitHub-ZouZiheng--Eric-blue.svg)](https://zouziheng-eric.github.io/)

**🇨🇳 [中文](README.md) | 🇺🇸 [English](README_EN.md)**

</div>

---

## 📋 项目概述

本项目是一个基于人工智能的量化投资策略开发平台，集成了**因子挖掘**、**多因子组合优化**和**智能模型构建**三大核心模块。采用现代机器学习技术栈，通过对A股市场高频数据的深度分析，构建了一套从数据处理、因子开发、策略回测到模型部署的全流程量化投资解决方案。

### 🎯 核心技术特性

- **🤖 AI驱动的因子挖掘**: 基于机器学习自动发现alpha因子
- **📊 多因子组合优化**: 智能权重分配和风险控制
- **🧠 智能模型选择**: 自适应模型推荐和超参数优化
- **📈 全链条回测系统**: 从单因子到组合策略的完整验证
- **⚡ 高性能计算**: 向量化计算和并行处理
- **📝 可视化报告**: 详细的策略分析和性能报告

---

## 🏗️ 技术架构

```
project/
├── 📁 factors/              # 因子策略模块
│   ├── factor_strategy_*.py     # 10个单因子策略
│   └── factor_evaluation_report.py # 因子评估系统
├── 📁 combinations/         # 多因子组合模块  
│   ├── multi_factor_*.py        # 5/10/15/20因子组合
│   └── performance_comparison.py # 性能对比分析
├── 📁 models/              # 智能模型模块
│   ├── intelligent_model_selection.py # 智能模型选择
│   ├── ensemble_models.py       # 集成学习系统
│   └── model_evaluation.py     # 模型评估框架
├── 📁 results/             # 结果输出
├── 📁 reports/             # 分析报告
├── 📁 logs/               # 运行日志
├── 📁 configs/            # 配置文件
├── start.py               # 启动脚本
└── run_module3.py         # 主程序
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 确保Python版本 >= 3.7
python --version
```

### 2. 一键启动

```bash
# 运行启动脚本
python start.py
```

### 3. 选择功能

启动后会看到以下菜单：

```
📋 请选择要执行的操作:
1. 🔧 检查并安装依赖包
2. 🚀 运行完整的模块三系统  
3. 📊 仅运行智能模型选择
4. 🔗 仅运行集成模型训练
5. 📝 查看使用说明
6. 🚪 退出
```

**推荐流程**：
1. 首次使用先选择 `1` 安装依赖
2. 然后选择 `2` 运行完整系统
3. 查看 `results/` 和 `reports/` 目录中的输出文件

---

## 📦 模块详解

### 模块一：智能因子挖掘系统

#### 🎯 功能目标
- 基于AI智能体自动挖掘A股市场中的alpha因子
- 构建多维度、多类型的因子策略库
- 评估因子的有效性和创新性

#### 🛠️ 技术实现

**单因子策略库** (10个策略文件)：

| 因子类型 | 策略文件 | 核心算法 |
|---------|---------|---------|
| RSI反转因子 | `factor_strategy_1.py` | 相对强弱指数反转 |
| MACD动量因子 | `factor_strategy_2.py` | 指数平滑移动平均收敛散度 |
| 布林带位置因子 | `factor_strategy_3.py` | 价格带状通道位置 |
| 威廉指标因子 | `factor_strategy_4.py` | Williams %R反转信号 |
| CCI商品通道因子 | `factor_strategy_5.py` | 商品通道指数 |
| 价量背离因子 | `factor_strategy_6.py` | 价格与成交量背离检测 |
| 成交量异常因子 | `factor_strategy_7.py` | 成交量异常识别 |
| VWAP偏离因子 | `factor_strategy_8.py` | 成交量加权平均价偏离 |
| 历史波动率因子 | `factor_strategy_9.py` | 历史波动率计算 |
| 跳跃检测因子 | `factor_strategy_10.py` | 价格跳跃检测算法 |

**核心代码示例**：
```python
# factor_strategy_1.py - RSI反转因子
def calculate_rsi_factor(data, period=14):
    """RSI反转因子计算"""
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # 反转信号：RSI > 70卖出，RSI < 30买入
    signal = np.where(rsi > 70, -1, 
                     np.where(rsi < 30, 1, 0))
    return signal, rsi
```

### 模块二：多因子组合优化系统

#### 🎯 功能目标
- 构建风险调整后收益最大化的因子组合
- 实现动态权重分配和风险控制
- 对比不同规模因子组合的表现

#### 🛠️ 技术实现

**组合策略递进**：

| 组合类型 | 策略文件 | 因子数量 |
|---------|---------|---------|
| 精选组合 | `multi_factor_5.py` | 5个 |
| 标准组合 | `multi_factor_10.py` | 10个 |
| 增强组合 | `multi_factor_15.py` | 15个 |
| 全量组合 | `multi_factor_20.py` | 20个 |

**权重优化算法**：
```python
# 动态权重分配示例
def dynamic_weight_allocation(factor_returns, window=60):
    """基于滚动窗口的动态权重分配"""
    weights = []
    for i in range(window, len(factor_returns)):
        # 计算历史IC值
        recent_ic = factor_returns.iloc[i-window:i].corr()
        
        # 基于信息系数分配权重
        ic_scores = recent_ic.mean()
        weight = ic_scores / ic_scores.sum()
        weights.append(weight)
    
    return pd.DataFrame(weights)
```

### 模块三：智能模型构建系统

#### 🎯 功能目标
- 智能模型选择和超参数优化
- 集成学习和模型融合
- 自动化模型评估和报告生成

#### 🛠️ 技术架构

**模型候选池**：

| 模型类别 | 具体算法 | 适用场景 |
|---------|---------|---------|
| 线性模型 | Ridge, Lasso, ElasticNet | 线性关系，高解释性 |
| 树模型 | RandomForest, XGBoost, LightGBM | 非线性，特征重要性 |
| 集成模型 | Stacking, Voting, Blending | 综合性能最优 |
| 神经网络 | MLP, LSTM, GRU | 复杂非线性模式 |

**智能选择流程**：
```python
# 智能模型选择系统
class IntelligentModelSelector:
    def __init__(self):
        self.models = self.init_model_pool()
    
    def auto_select(self, X, y):
        """基于数据特征自动选择最优模型"""
        # 1. 数据特征分析
        data_profile = self.analyze_data(X, y)
        
        # 2. 模型预筛选
        candidate_models = self.filter_models(data_profile)
        
        # 3. 交叉验证评估
        results = self.cross_validate(candidate_models, X, y)
        
        # 4. 返回最优模型
        return self.select_best_model(results)
```

---

## 📊 核心功能

### 1. 因子挖掘与评估

- **自动因子挖掘**: 基于技术指标、价量关系、波动率等维度
- **因子有效性测试**: IC值、分层收益率、胜率分析
- **因子稳定性检验**: 子期间分析、滚动测试
- **因子创新性评估**: 与传统因子的相关性分析

### 2. 多因子组合构建

- **权重优化算法**: 等权重、IC加权、风险平价、机器学习权重
- **风险控制模型**: 行业中性、市值中性、波动率控制
- **组合构建流程**: 因子筛选 → 权重分配 → 风险调整 → 绩效归因

### 3. 智能模型系统

- **自适应模型选择**: 基于数据特征推荐最优算法
- **超参数自动优化**: 贝叶斯优化、网格搜索、随机搜索
- **集成学习框架**: Stacking、Voting、Blending多种策略
- **模型解释性分析**: 特征重要性、SHAP值分析

---

## 📈 性能指标

### 回测评估指标

| 指标类别 | 具体指标 | 计算方式 |
|---------|---------|---------|
| 收益指标 | 年化收益率、累积收益率 | 复合增长率 |
| 风险指标 | 年化波动率、最大回撤 | 标准差、峰谷值 |
| 风险调整指标 | 夏普比率、卡尔马比率 | 收益风险比 |
| 信息指标 | 信息比率、跟踪误差 | 超额收益稳定性 |
| 交易指标 | 换手率、交易成本 | 实盘可执行性 |

### 因子评估维度

- **📊 收益性**: 因子收益率分布、分层收益差
- **🎯 稳定性**: 月度胜率、IC值序列稳定性  
- **🔄 换手率**: 因子信号变化频率
- **💰 容量**: 策略资金容量估算
- **🆕 创新性**: 与已知因子的正交性

---

## 🛠️ 依赖环境

### 必备依赖

```python
# 核心数据处理
pandas >= 1.3.0
numpy >= 1.20.0

# 机器学习
scikit-learn >= 1.0.0
lightgbm >= 3.0.0

# 数据可视化  
matplotlib >= 3.3.0
seaborn >= 0.11.0
```

### 可选依赖

```python
# 高级机器学习
xgboost >= 1.4.0      # 梯度提升
catboost >= 1.0.0     # 分类提升
optuna >= 2.0.0       # 超参数优化

# 深度学习
tensorflow >= 2.6.0   # 神经网络
torch >= 1.9.0        # PyTorch

# 高性能计算
numba >= 0.53.0       # JIT编译加速
```

### 安装方式

```bash
# 方式1：自动安装（推荐）
python start.py  # 选择选项1

# 方式2：手动安装基础包
pip install pandas numpy scikit-learn matplotlib seaborn

# 方式3：安装完整环境
pip install -r requirements.txt
```

---

## 📚 使用指南

### 基础使用

#### 1. 运行单个因子策略
```bash
cd factors/
python factor_strategy_1.py
```

#### 2. 运行多因子组合
```bash  
cd combinations/
python multi_factor_10.py
```

#### 3. 运行智能模型系统
```bash
python run_module3.py
```

### 高级配置

#### 自定义因子参数
```python
# 在factor_strategy_*.py中修改参数
FACTOR_PARAMS = {
    'rsi_period': 14,      # RSI计算周期
    'ma_period': 20,       # 移动平均周期
    'threshold_buy': 30,   # 买入阈值
    'threshold_sell': 70   # 卖出阈值
}
```

#### 调整模型配置
```yaml
# configs/model_config.yaml
models:
  lightgbm:
    n_estimators: [100, 200, 500]
    max_depth: [3, 5, 7]
    learning_rate: [0.01, 0.05, 0.1]
    
evaluation:
  cv_folds: 5
  test_size: 0.2
  metrics: ['mse', 'r2', 'sharpe']
```

---

## 📋 输出说明

### 文件结构

```
project/
├── results/                    # 策略结果
│   ├── backtest_results/       # 回测结果
│   ├── factor_performance/     # 因子表现
│   ├── model_predictions/      # 模型预测
│   └── performance_charts/     # 性能图表
├── reports/                    # 分析报告
│   ├── factor_evaluation_report.xlsx  # 因子评估
│   ├── model_comparison_report.html   # 模型对比
│   └── final_analysis_report.md       # 综合分析
└── logs/                      # 运行日志
    ├── factor_logs/           # 因子计算日志
    ├── model_logs/            # 模型训练日志
    └── error_logs/            # 错误日志
```

### 关键输出文件

#### 1. 因子评估报告 (`factor_evaluation_report.xlsx`)

| 因子名称 | 年化收益率 | 夏普比率 | 最大回撤 | IC均值 | 胜率 |
|---------|-----------|---------|---------|--------|------|
| RSI反转因子 | 15.2% | 1.23 | -8.5% | 0.045 | 56.3% |
| MACD动量因子 | 12.8% | 1.15 | -12.1% | 0.038 | 54.2% |
| ... | ... | ... | ... | ... | ... |

#### 2. 模型对比报告 (`model_comparison_report.html`)

包含：
- 模型性能对比表格
- 特征重要性分析图
- 预测误差分布图
- 模型稳定性测试结果

#### 3. 综合分析报告 (`final_analysis_report.md`)

包含：
- 项目执行总结
- 关键发现和洞察
- 模型部署建议
- 风险提示和局限性

---

## ⚠️ 注意事项

### 数据要求

- **格式要求**: 标准OHLCV格式，包含时间戳
- **数据质量**: 需要处理缺失值、异常值、停牌数据
- **时间频率**: 支持分钟级、小时级、日级数据
- **股票池**: 建议使用流动性好的主板股票

### 系统限制

- **内存要求**: 建议8GB以上内存用于大规模回测
- **计算时间**: 完整系统运行需要30分钟到2小时
- **Python版本**: 需要Python 3.7或更高版本
- **操作系统**: Windows/macOS/Linux均支持

### 风险提示

⚠️ **重要声明**：
- 本系统仅用于学术研究和教学目的
- 历史表现不代表未来收益
- 实盘交易前请充分测试和验证
- 量化策略存在模型风险和市场风险
- 建议结合基本面分析进行投资决策

---

## 🤝 贡献与支持

### 问题反馈

如遇到问题，请按以下顺序排查：

1. **查看日志文件**: `logs/` 目录下的错误信息
2. **检查依赖环境**: 查看`requirements.txt` 文档
3. **参考使用说明**: 查看 `USAGE.md` 文档
4. **简化测试**: 使用小数据集进行功能验证

### 开发贡献

欢迎贡献代码和改进建议：

- **因子策略**: 在 `factors/` 目录添加新的因子算法
- **组合优化**: 在 `combinations/` 目录实现新的权重分配方法
- **模型算法**: 在 `models/` 目录集成新的机器学习模型
- **性能优化**: 提升计算效率和内存使用

---

## 📄 许可证

本项目遵循 MIT License 开源协议。

---

## 🎓 学术引用

如果本项目对您的研究有帮助，请考虑引用：

```
量化投资概论课程 Final Project
基于 AI 智能体的量化投资策略开发系统
https://github.com/ZouZiheng-Eric/Development-System-for-Quantitative-Investment-Strategies-Based-on-Artificial-Intelligence-Agents
```

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给个Star支持一下！⭐**

**🚀 开始您的量化投资之旅吧！🚀**

</div> 