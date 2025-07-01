"""
数据解析模块

负责处理不同格式的数据集，包括：
- 超参数定义
- 数据格式化
- 基础数据解析功能
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

@dataclass
class HyperParams:
    """超参数配置类"""
    learning_rate: float
    lora_r: int
    lora_alpha: float
    # 可以添加其他超参数

class DatasetAnalyzer:
    """数据集分析器基类
    
    提供基础的数据解析和模型加载功能
    """
    
    def __init__(
        self,
        base_model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """初始化数据集分析器
        
        Args:
            base_model_name: 预训练模型名称
            device: 计算设备
        """
        self.device = device
        # 首先在 CPU 上加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=None,  # 不使用自动设备映射
            trust_remote_code=True,
            torch_dtype=torch.float16,  # 使用 float16 减少显存使用
        )
        # 然后手动将模型移动到指定设备
        self.model = self.model.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        self.model.eval()  # 设置为评估模式
        
    def format_qa_pair(self, item: Dict) -> str:
        """将 QA 对格式化为模型输入格式
        
        Args:
            item: 数据项，支持多种格式
            
        Returns:
            格式化后的文本
        """
        if "conversations" in item:
            # 处理 Qwen2.5 格式
            conversations = item["conversations"]
            formatted_text = ""
            for msg in conversations:
                if msg["role"] == "system":
                    formatted_text += f"{msg['content']}\n"
                elif msg["role"] == "user":
                    formatted_text += f"问：{msg['content']}\n"
                elif msg["role"] == "assistant":
                    formatted_text += f"答：{msg['content']}\n"
            return formatted_text.strip()
        else:
            # 处理简单 QA 对格式
            return f"问：{item['input']}\n答：{item['output']}"
    
    def extract_qa_from_item(self, item: Dict) -> Tuple[str, str]:
        """从数据项中提取问题和答案
        
        Args:
            item: 数据项
            
        Returns:
            (question, answer): 问题和答案的元组
        """
        if "conversations" in item:
            # 处理 Qwen2.5 格式
            question = next(msg["content"] for msg in item["conversations"] 
                          if msg["role"] == "user")
            answer = next(msg["content"] for msg in item["conversations"] 
                        if msg["role"] == "assistant")
        else:
            # 处理简单 QA 对格式
            question = item["input"]
            answer = item["output"]
        
        return question, answer
    
    def get_hyperparams_features(self, hyperparams: HyperParams) -> Dict[str, float]:
        """提取超参数特征
        
        Args:
            hyperparams: 超参数对象
            
        Returns:
            超参数特征字典
        """
        return {
            "learning_rate": hyperparams.learning_rate,
            "lora_r": hyperparams.lora_r,
            "lora_alpha": hyperparams.lora_alpha
        }
    
    def get_dataset_size(self, dataset: List[Dict]) -> int:
        """计算数据集大小
        
        Args:
            dataset: 数据集列表
            
        Returns:
            数据集大小
        """
        return len(dataset)
    
    def sample_dataset(self, dataset: List[Dict], sample_size: int) -> List[Dict]:
        """从数据集中采样
        
        Args:
            dataset: 原始数据集
            sample_size: 采样大小
            
        Returns:
            采样后的数据集
        """
        if len(dataset) < sample_size:
            sample_size = len(dataset)
        return np.random.choice(dataset, sample_size, replace=False)
    
    def calculate_initial_loss(
        self,
        dataset: List[Dict],
        sample_size: int = 100
    ) -> float:
        """计算初始平均损失
        
        Args:
            dataset: 数据集
            sample_size: 采样大小
            
        Returns:
            初始平均损失
        """
        sampled_data = self.sample_dataset(dataset, sample_size)
        total_loss = 0.0
        
        with torch.no_grad():
            for item in tqdm(sampled_data, desc="计算数据集初始损失"):
                # 格式化输入
                formatted_text = self.format_qa_pair(item)
                inputs = self.tokenizer(
                    formatted_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # 计算损失
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item()
                
        return total_loss / len(sampled_data)

    def extract_all_basic_features(self, dataset: List[Dict], hyperparams: HyperParams, sample_size: int = 100) -> Dict[str, float]:
        """一次性提取所有基础特征"""
        features = {}
        features.update(self.get_hyperparams_features(hyperparams))
        features["dataset_size"] = self.get_dataset_size(dataset)
        features["initial_loss"] = self.calculate_initial_loss(dataset, sample_size)
        return features 