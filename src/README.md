# 数据集特征提取与分析工具包

## 📖 概述

本工具包专为NLP数据集的质量分析与特征提取设计，支持多种主流数据格式，适配大模型静态/动态特征分析，助力数据治理、模型微调和数据集评估。  
支持批量推理、多卡适配、自动格式识别，适合科研、工业和数据治理场景。

---

## 🏗️ 目录结构

```
src/
├── __init__.py                 # 包入口，导出主要接口
├── data_parsers.py             # 数据解析与基础特征
├── static_features.py          # 静态特征批量提取
├── dynamic_probes.py           # 动态特征（梯度探针）分析
├── feature_dispatcher.py       # 特征统一调度与CSV保存
└── README.md                   # 本文件
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备模型与数据

- 下载Qwen等大模型权重至`models/`目录
- 准备或转换为支持的标准数据格式（如Qwen微调格式、最终格式等）

### 3. 运行特征提取测试

```bash
python data_progress/test_final_format_features.py
```

---

## 🔧 主要模块说明

### data_parsers.py
- 多格式数据解析与标准化
- 超参数管理（如LoRA参数、学习率等）
- 数据采样、基础统计等通用能力

### static_features.py
- 静态特征批量提取（文本长度、TTR、困惑度、语义多样性等）
- 支持批量推理，自动适配“最终格式”数据

### dynamic_probes.py
- 基于LoRA微调的梯度探针，分析损失下降率、梯度范数等动态特征
- 支持批量训练，自动显存管理

### feature_dispatcher.py
- 一键提取基础、静态、动态全量特征
- 支持特征保存为CSV，便于后续分析

---

## 📊 支持的特征类型

- **基础特征**：学习率、LoRA参数、数据集大小、初始损失等
- **文本统计特征**：长度、TTR、n-gram重复率、词汇复杂度等
- **语义特征**：多样性、输入输出相似度、语义一致性
- **困惑度特征**：参考模型困惑度、基础模型困惑度、变化率等
- **动态特征**：损失下降率、平均梯度范数、梯度一致性

---

## 📝 典型用法

```python
from src.static_features import StaticFeatureExtractor
from src.dynamic_probes import DynamicProbeAnalyzer
from src.data_parsers import HyperParams

# 加载模型和分词器
model = ...  # 参见测试脚本
tokenizer = ...
device = "cuda"

# 构造最终格式数据
data = [
    {"context_text": "...", "qa_pairs": [{"question": "...", "output": "..."}]}
]

# 静态特征提取
static_extractor = StaticFeatureExtractor(model, tokenizer, device=device)
static_features = static_extractor.extract_all_static_features(data, sample_size=8, batch_size=4)

# 动态特征提取
hyperparams = HyperParams(learning_rate=1e-4, lora_r=8, lora_alpha=16)
dynamic_analyzer = DynamicProbeAnalyzer(model, tokenizer, device=device)
dynamic_features = dynamic_analyzer.extract_all_dynamic_features(
    data, hyperparams, probe_steps=5, sample_size=8, batch_size=1
)
```

---

## ⚡ 性能优化建议

- **批量推理**：合理设置`batch_size`，充分利用GPU
- **显存管理**：动态特征提取时建议减小`batch_size`，避免OOM
- **多卡支持**：可扩展为多卡并行（需进一步开发）
- **采样参数**：`sample_size`、`probe_steps`等参数可根据数据量和硬件灵活调整

---

## 🐛 常见问题

- **动态特征为0或OOM**：请减小`batch_size`，或减少采样/步数
- **格式不兼容**：请用`data_parsers.py`中的方法转换为标准格式
- **模型加载慢**：建议本地缓存模型权重，或使用更小的模型测试

---

## 📄 许可证

本项目采用 MIT License。

---

如需更详细的API文档或二次开发指导，请查阅各模块源码或联系作者团队。

---

如需英文版或更详细的团队/贡献说明，也可随时告知！ 