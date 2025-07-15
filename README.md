# 数据集特征提取与格式转换工具包

## 📦 项目结构

```
src/
├── __init__.py                # 包入口
├── data_parsers.py            # 数据解析与基础特征
├── static_features.py         # 静态特征批量提取
├── dynamic_probes.py          # 动态特征（梯度探针）分析
├── feature_dispatcher.py      # 特征统一调度与CSV保存
├── README.md                  # 详细说明文档

data_progress/
├── dataset_to_qwen_converter.py   # 多格式数据集转Qwen格式
├── qwen_to_final_converter.py     # Qwen格式转最终格式
├── test_detailed_output.py        # 转换流程详细测试脚本
```

## ✨ 主要功能

### 1. 特征提取（src/）
- **data_parsers.py**：支持多格式数据解析、超参数管理、基础统计等。
- **static_features.py**：批量提取文本长度、TTR、困惑度、语义多样性等静态特征。
- **dynamic_probes.py**：基于LoRA微调的梯度探针，分析损失下降率、梯度范数等动态特征。
- **feature_dispatcher.py**：一键提取全部特征，并支持保存为CSV，便于后续分析。

### 2. 数据格式转换（data_progress/）
- **dataset_to_qwen_converter.py**：支持Alpaca、Dolly15k、QA对、SQuAD等主流格式转为Qwen微调格式。
- **qwen_to_final_converter.py**：将Qwen格式进一步转为最终分析格式（context_text + qa_pairs）。
- **test_detailed_output.py**：详细演示和测试数据格式转换流程，便于理解和验证。

## 🚀 快速开始

1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

2. 数据格式转换
   - 支持多种主流NLP数据集格式，统一转为Qwen微调格式，再转为最终分析格式。
   - 参考 `data_progress/test_detailed_output.py` 查看详细用法和测试流程。

3. 特征提取
   - 参考 `src/README.md` 或直接调用 `feature_dispatcher.py` 中的接口，批量提取静态/动态/基础特征。

## 📝 典型用法示例

```python
from src.static_features import StaticFeatureExtractor
from src.dynamic_probes import DynamicProbeAnalyzer
from src.data_parsers import HyperParams

# 加载模型和分词器
model = ...
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

## 💡 适用场景

- 科研、工业、数据治理场景下的大模型数据集质量分析与特征工程
- 支持批量推理、多卡适配、自动格式识别

## 📄 许可证

本项目采用 MIT License。

---

如需更详细的API文档或二次开发指导，请查阅各模块源码或联系作者团队。 