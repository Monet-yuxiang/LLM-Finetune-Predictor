# 数据集特征提取工具包 (src)

## 📖 概述

这是一个全面的数据集质量分析工具包，支持从多个维度提取数据集特征，包括基础特征、静态特征和动态特征。工具包采用模块化设计，支持多卡并行和批量处理，大幅提升特征提取效率。

## 🏗️ 架构设计

```
src/
├── __init__.py                 # 包入口，导出主要接口
├── data_parsers.py            # 数据解析模块
├── static_features.py         # 静态特征提取模块
├── dynamic_probes.py          # 动态模型探针模块
├── feature_dispatcher.py      # 特征总调度模块
├── optimized_feature_extractor.py  # 优化特征提取器
└── README.md                  # 本文件
```

## 🚀 快速开始

### 1. 基础使用

```python
from src import extract_all_features, HyperParams

# 创建超参数
hyperparams = HyperParams(
    learning_rate=1e-4,
    lora_r=8,
    lora_alpha=16
)

# 提取所有特征
features = extract_all_features(
    dataset=your_dataset,
    hyperparams=hyperparams,
    base_model_name="your_model_path",
    static_sample_size=100,
    dynamic_probe_steps=100,
    dynamic_sample_size=50,
    save_to_csv=True,
    csv_filename="features.csv"
)
```

### 2. 优化版本使用

```python
from src.optimized_feature_extractor import OptimizedFeatureExtractor

# 初始化优化特征提取器
extractor = OptimizedFeatureExtractor(
    base_model_name="your_model_path",
    batch_size=8,      # 批处理大小
    num_gpus=None      # 自动检测GPU数量
)

# 提取特征
features = extractor.extract_all_features_optimized(
    dataset=your_dataset,
    hyperparams=hyperparams,
    static_sample_size=100,
    dynamic_probe_steps=100,
    dynamic_sample_size=50,
    save_to_csv=True
)

# 清理资源
extractor.cleanup()
```

## 📊 特征类型

### 1. 基础特征 (5个)
- `learning_rate`: 学习率
- `lora_r`: LoRA rank
- `lora_alpha`: LoRA alpha
- `dataset_size`: 数据集大小
- `initial_loss`: 初始损失

### 2. 文本统计特征 (10个)
- `avg_input_length`: 平均输入长度
- `avg_output_length`: 平均输出长度
- `io_length_ratio`: 输入输出长度比
- `input_length_std`: 输入长度标准差
- `output_length_std`: 输出长度标准差
- `input_ttr`: 输入TTR (类符/形符比)
- `output_ttr`: 输出TTR
- `output_ngram_repetition`: 输出n-gram重复率
- `approximate_duplicates`: 近似重复样本比例
- `vocab_complexity`: 词汇复杂度

### 3. 语义特征 (3个)
- `semantic_diversity`: 语义多样性
- `io_similarity`: 输入输出相似度
- `semantic_consistency`: 语义一致性

### 4. 困惑度特征 (5个)
- `reference_perplexity`: 参考模型困惑度
- `base_model_perplexity`: 基础模型困惑度
- `perplexity_change_rate`: 困惑度变化率
- `reference_perplexity_std`: 参考困惑度标准差
- `base_perplexity_std`: 基础困惑度标准差

### 5. 动态特征 (3个)
- `loss_decay_rate`: 损失下降率
- `avg_grad_norm`: 平均梯度范数
- `gradient_consistency`: 梯度一致性

## 🔧 模块详解

### data_parsers.py
**功能**: 数据解析和格式化
- `HyperParams`: 超参数配置类
- `DatasetAnalyzer`: 数据集分析器基类
- 支持多种数据格式 (Qwen2.5格式、简单QA对格式)
- 提供基础数据解析功能

### static_features.py
**功能**: 静态特征提取
- `StaticFeatureExtractor`: 静态特征提取器
- 继承自 `DatasetAnalyzer`
- 提取不需要模型训练的静态特征
- 包括文本统计、语义特征、困惑度特征

### dynamic_probes.py
**功能**: 动态模型探针
- `DynamicProbeAnalyzer`: 动态探针分析器
- 基于模型微调的动态特征分析
- 计算损失下降率、梯度范数、梯度一致性
- 需要模型训练，计算成本较高

### feature_dispatcher.py
**功能**: 特征总调度
- `extract_all_features()`: 提取所有特征的主函数
- `save_features_to_csv()`: 保存特征到CSV文件
- 统一调度所有模块特征提取
- 优化策略：将动态探针放在最后，只使用一个模型实例

### optimized_feature_extractor.py
**功能**: 优化特征提取器
- `OptimizedFeatureExtractor`: 优化特征提取器
- 支持多卡并行和批量处理
- 主进程串行调度多卡，避免并发问题
- 大幅提升特征提取速度

## ⚡ 性能优化

### 1. 多卡并行
- 自动检测可用GPU数量
- 每个GPU独立加载模型实例
- 批次轮流分配到不同GPU上处理

### 2. 批量处理
- 支持可配置的批处理大小
- 减少模型调用次数，提升效率
- 根据显存情况调整batch_size

### 3. 显存优化
- 使用float16减少显存使用
- 及时清理显存
- 模型共享，避免重复加载

## 📝 使用示例

### 示例1: 基础特征提取
```python
from src import extract_all_features, HyperParams

hyperparams = HyperParams(learning_rate=1e-4, lora_r=8, lora_alpha=16)
features = extract_all_features(
    dataset=dataset,
    hyperparams=hyperparams,
    base_model_name="Qwen2.5-7B-Instruct",
    static_sample_size=50,
    dynamic_probe_steps=50,
    dynamic_sample_size=20
)
```

### 示例2: 优化特征提取
```python
from src.optimized_feature_extractor import OptimizedFeatureExtractor

extractor = OptimizedFeatureExtractor(
    base_model_name="Qwen2.5-7B-Instruct",
    batch_size=4,
    num_gpus=None
)

features = extractor.extract_all_features_optimized(
    dataset=dataset,
    hyperparams=hyperparams,
    static_sample_size=50,
    dynamic_probe_steps=50,
    dynamic_sample_size=20
)
```

### 示例3: 保存特征到CSV
```python
features = extract_all_features(
    dataset=dataset,
    hyperparams=hyperparams,
    base_model_name="Qwen2.5-7B-Instruct",
    save_to_csv=True,
    csv_filename="dataset_features.csv"
)
```

## 🔍 参数调优建议

### 1. 静态特征参数
- `static_sample_size`: 建议100-500，根据数据集大小调整
- 样本数越多，特征越准确，但计算时间越长

### 2. 动态特征参数
- `dynamic_probe_steps`: 建议50-200步
- `dynamic_sample_size`: 建议20-100个样本
- 步数和样本数越多，动态特征越稳定

### 3. 优化参数
- `batch_size`: 根据显存大小调整，建议4-16
- `num_gpus`: 自动检测，或手动指定

### 4. 超参数建议
```python
# 标准配置
HyperParams(learning_rate=1e-4, lora_r=8, lora_alpha=16)

# 高精度配置
HyperParams(learning_rate=5e-5, lora_r=16, lora_alpha=32)

# 快速配置
HyperParams(learning_rate=5e-4, lora_r=4, lora_alpha=8)
```

## 🐛 常见问题

### 1. 显存不足
**解决方案**:
- 减少 `batch_size`
- 减少 `static_sample_size` 和 `dynamic_sample_size`
- 使用 `torch.float16`

### 2. 动态特征为0
**可能原因**:
- 动态探针步数太少
- 学习率设置不当
- 样本数量不足
- LoRA配置问题

**解决方案**:
- 增加 `dynamic_probe_steps` 到100-200
- 调整学习率到1e-4或1e-5
- 增加样本数量到20-50

### 3. 并发错误 (Already borrowed)
**解决方案**:
- 使用 `OptimizedFeatureExtractor` 而不是多线程
- 确保使用主进程串行调度多卡

## 📈 性能对比

| 方法 | 速度 | 稳定性 | 显存使用 | 适用场景 |
|------|------|--------|----------|----------|
| 基础方法 | 中等 | 高 | 低 | 小数据集，单卡 |
| 优化方法 | 高 | 高 | 中等 | 大数据集，多卡 |

## 🔄 版本历史

- **v1.0.0**: 初始版本，包含基础特征提取功能
- **v1.1.0**: 添加优化特征提取器，支持多卡并行
- **v1.2.0**: 修复并发问题，优化显存使用

## 📞 技术支持

如有问题或建议，请查看：
1. 测试脚本: `test_optimized_feature_extraction.py`
2. 性能对比: `performance_comparison.py`
3. 动态特征测试: `test_dynamic_features.py`

## 📄 许可证

本项目采用 MIT 许可证。 