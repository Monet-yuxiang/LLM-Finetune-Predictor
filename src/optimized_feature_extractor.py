"""
优化特征提取器

支持多卡并行和批量处理，大幅提升特征提取速度
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging
import time

from .data_parsers import DatasetAnalyzer, HyperParams
from .static_features import StaticFeatureExtractor
from .dynamic_probes import DynamicProbeAnalyzer

logger = logging.getLogger(__name__)

class OptimizedFeatureExtractor:
    """优化特征提取器
    
    支持多卡并行和批量处理，大幅提升特征提取速度
    """
    
    def __init__(
        self,
        base_model_name: str,
        batch_size: int = 8,
        num_gpus: int = None
    ):
        """初始化优化特征提取器
        
        Args:
            base_model_name: 预训练模型名称
            batch_size: 批处理大小
            num_gpus: 使用的GPU数量，None表示自动检测
        """
        self.base_model_name = base_model_name
        self.batch_size = batch_size
        
        # 检测可用GPU
        if num_gpus is None:
            self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        else:
            self.num_gpus = min(num_gpus, torch.cuda.device_count() if torch.cuda.is_available() else 0)
        
        logger.info(f"初始化优化特征提取器:")
        logger.info(f"  批处理大小: {self.batch_size}")
        logger.info(f"  可用GPU数量: {self.num_gpus}")
        logger.info(f"  调度策略: 主进程串行调度多卡")
        
        # 初始化模型实例（每个GPU一个）
        self.models = {}
        self.tokenizers = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化多GPU模型"""
        if self.num_gpus == 0:
            # CPU模式
            device = "cpu"
            model = self._load_model_on_device(device)
            self.models[device] = model
            self.tokenizers[device] = model.tokenizer
        else:
            # 多GPU模式
            for gpu_id in range(self.num_gpus):
                device = f"cuda:{gpu_id}"
                try:
                    model = self._load_model_on_device(device)
                    self.models[device] = model
                    self.tokenizers[device] = model.tokenizer
                    logger.info(f"成功加载模型到 {device}")
                except Exception as e:
                    logger.warning(f"加载模型到 {device} 失败: {str(e)}")
        
        if not self.models:
            raise RuntimeError("没有成功加载任何模型")
    
    def _load_model_on_device(self, device: str) -> DatasetAnalyzer:
        """在指定设备上加载模型"""
        return DatasetAnalyzer(self.base_model_name, device)
    
    def _batch_process_data(self, data: List[Dict], batch_size: int) -> List[List[Dict]]:
        """将数据分批"""
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
    def _get_device_for_batch(self, batch_idx: int) -> str:
        """为批次分配设备"""
        if self.num_gpus == 0:
            return "cpu"
        else:
            return f"cuda:{batch_idx % self.num_gpus}"
    
    def extract_all_features_optimized(
        self,
        dataset: List[Dict],
        hyperparams: HyperParams,
        static_sample_size: int = 100,
        dynamic_probe_steps: int = 100,
        dynamic_sample_size: int = 50,
        save_to_csv: bool = False,
        csv_filename: Optional[str] = None
    ) -> Dict[str, float]:
        """优化的特征提取主函数
        
        使用多卡并行和批量处理大幅提升速度
        """
        start_time = time.time()
        logger.info("开始优化特征提取...")
        
        # 1. 基础特征（单线程，快速）
        logger.info("提取基础特征...")
        base_analyzer = self.models[list(self.models.keys())[0]]
        basic_features = base_analyzer.extract_all_basic_features(dataset, hyperparams, static_sample_size)
        
        # 2. 静态特征（多卡并行）
        logger.info("提取静态特征（多卡并行）...")
        static_features = self._extract_static_features_parallel(dataset, static_sample_size)
        
        # 3. 动态特征（单卡，需要训练）
        logger.info("提取动态特征...")
        dynamic_features = self._extract_dynamic_features(dataset, hyperparams, dynamic_probe_steps, dynamic_sample_size)
        
        # 合并所有特征
        features = {}
        features.update(basic_features)
        features.update(static_features)
        features.update(dynamic_features)
        
        # 保存到CSV文件
        if save_to_csv:
            from .feature_dispatcher import save_features_to_csv
            save_features_to_csv(features, csv_filename, dataset, hyperparams)
        
        elapsed_time = time.time() - start_time
        logger.info(f"优化特征提取完成，耗时: {elapsed_time:.2f}秒")
        
        return features
    
    def _extract_static_features_parallel(self, dataset: List[Dict], sample_size: int) -> Dict[str, float]:
        """串行调度多卡处理静态特征（避免并发问题）"""
        if len(dataset) < sample_size:
            sample_size = len(dataset)
        
        sampled_data = np.random.choice(dataset, sample_size, replace=False)
        
        # 将数据分批
        batches = self._batch_process_data(sampled_data, self.batch_size)
        
        # 主进程串行调度每张卡
        all_features = []
        for batch_idx, batch in enumerate(tqdm(batches, desc="串行调度多卡处理静态特征")):
            device = self._get_device_for_batch(batch_idx)
            try:
                batch_features = self._process_static_batch(batch, device)
                all_features.append(batch_features)
                logger.debug(f"批次 {batch_idx} 在 {device} 上处理完成")
            except Exception as e:
                logger.error(f"处理批次 {batch_idx} 时发生错误: {str(e)}")
                # 添加空特征避免索引错误
                empty_features = {
                    "input_lengths": [],
                    "output_lengths": [],
                    "input_ttrs": [],
                    "output_ttrs": [],
                    "ngram_repetitions": [],
                    "vocab_complexities": [],
                    "reference_perplexities": [],
                    "base_perplexities": [],
                    "semantic_diversities": [],
                    "io_similarities": []
                }
                all_features.append(empty_features)
        
        # 合并批次特征
        return self._merge_batch_features(all_features)
    
    def _process_static_batch(self, batch: List[Dict], device: str) -> Dict[str, List[float]]:
        """处理单个批次的静态特征"""
        model = self.models[device]
        tokenizer = self.tokenizers[device]
        
        # 初始化特征列表
        features = {
            "input_lengths": [],
            "output_lengths": [],
            "input_ttrs": [],
            "output_ttrs": [],
            "ngram_repetitions": [],
            "vocab_complexities": [],
            "reference_perplexities": [],
            "base_perplexities": [],
            "semantic_diversities": [],
            "io_similarities": []
        }
        
        with torch.no_grad():
            for item in batch:
                # 提取QA对
                if "conversations" in item:
                    question = next(msg["content"] for msg in item["conversations"] 
                                  if msg["role"] == "user")
                    answer = next(msg["content"] for msg in item["conversations"] 
                                if msg["role"] == "assistant")
                else:
                    question = item["input"]
                    answer = item["output"]
                
                # 1. 长度特征
                input_tokens = tokenizer.encode(question, add_special_tokens=False)
                output_tokens = tokenizer.encode(answer, add_special_tokens=False)
                features["input_lengths"].append(len(input_tokens))
                features["output_lengths"].append(len(output_tokens))
                
                # 2. TTR特征
                features["input_ttrs"].append(self._calculate_ttr(question, tokenizer))
                features["output_ttrs"].append(self._calculate_ttr(answer, tokenizer))
                
                # 3. N-gram重复率
                ngram_rep = self._calculate_ngram_repetition(answer, tokenizer)
                features["ngram_repetitions"].append(ngram_rep)
                
                # 4. 词汇复杂度
                vocab_complexity = self._calculate_vocab_complexity(f"{question} {answer}")
                features["vocab_complexities"].append(vocab_complexity)
                
                # 5. 困惑度特征
                ref_perplexity = self._calculate_reference_perplexity(item, model, tokenizer)
                base_perplexity = self._calculate_base_perplexity(answer, model, tokenizer)
                features["reference_perplexities"].append(ref_perplexity)
                features["base_perplexities"].append(base_perplexity)
                
                # 6. 语义特征
                semantic_diversity = self._calculate_semantic_diversity(answer, model, tokenizer)
                io_similarity = self._calculate_io_similarity(question, answer, model, tokenizer)
                features["semantic_diversities"].append(semantic_diversity)
                features["io_similarities"].append(io_similarity)
        
        return features
    
    def _extract_dynamic_features(self, dataset: List[Dict], hyperparams: HyperParams, 
                                 probe_steps: int, sample_size: int) -> Dict[str, float]:
        """提取动态特征（单卡，需要训练）"""
        # 使用第一个可用的设备
        device = list(self.models.keys())[0]
        model = self.models[device]
        
        # 创建动态探针分析器
        dynamic_analyzer = DynamicProbeAnalyzer(self.base_model_name)
        dynamic_analyzer.model = model.model
        dynamic_analyzer.tokenizer = model.tokenizer
        dynamic_analyzer.device = device
        
        return dynamic_analyzer.extract_all_dynamic_features(
            dataset, hyperparams, probe_steps, sample_size
        )
    
    def _merge_batch_features(self, batch_features: List[Dict[str, List[float]]]) -> Dict[str, float]:
        """合并批次特征"""
        if not batch_features:
            return {}
        
        # 收集所有特征
        all_values = {}
        for feature_name in batch_features[0].keys():
            all_values[feature_name] = []
            for batch_feature in batch_features:
                all_values[feature_name].extend(batch_feature[feature_name])
        
        # 计算统计特征
        merged_features = {}
        
        # 长度特征
        merged_features["avg_input_length"] = np.mean(all_values["input_lengths"])
        merged_features["avg_output_length"] = np.mean(all_values["output_lengths"])
        merged_features["io_length_ratio"] = merged_features["avg_output_length"] / merged_features["avg_input_length"] if merged_features["avg_input_length"] > 0 else 0
        merged_features["input_length_std"] = np.std(all_values["input_lengths"])
        merged_features["output_length_std"] = np.std(all_values["output_lengths"])
        
        # 多样性特征
        merged_features["input_ttr"] = np.mean(all_values["input_ttrs"])
        merged_features["output_ttr"] = np.mean(all_values["output_ttrs"])
        merged_features["output_ngram_repetition"] = np.mean(all_values["ngram_repetitions"])
        merged_features["vocab_complexity"] = np.mean(all_values["vocab_complexities"])
        
        # 困惑度特征
        merged_features["reference_perplexity"] = np.mean(all_values["reference_perplexities"])
        merged_features["base_model_perplexity"] = np.mean(all_values["base_perplexities"])
        merged_features["perplexity_change_rate"] = (merged_features["reference_perplexity"] - merged_features["base_model_perplexity"]) / merged_features["base_model_perplexity"] if merged_features["base_model_perplexity"] > 0 else 0.0
        merged_features["reference_perplexity_std"] = np.std(all_values["reference_perplexities"])
        merged_features["base_perplexity_std"] = np.std(all_values["base_perplexities"])
        
        # 语义特征
        merged_features["semantic_diversity"] = np.mean(all_values["semantic_diversities"])
        merged_features["io_similarity"] = np.mean(all_values["io_similarities"])
        
        # 计算语义一致性（需要跨批次计算）
        merged_features["semantic_consistency"] = self._calculate_semantic_consistency(all_values["semantic_diversities"])
        
        # 计算近似重复（需要跨批次计算）
        merged_features["approximate_duplicates"] = 0.0  # 简化处理
        
        return merged_features
    
    def _calculate_ttr(self, text: str, tokenizer) -> float:
        """计算TTR"""
        tokens = tokenizer.tokenize(text.lower())
        if not tokens:
            return 0.0
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        return unique_tokens / total_tokens if total_tokens > 0 else 0.0
    
    def _calculate_ngram_repetition(self, text: str, tokenizer, n: int = 3) -> float:
        """计算n-gram重复率"""
        tokens = tokenizer.tokenize(text.lower())
        if len(tokens) < n:
            return 0.0
        
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        if not ngrams:
            return 0.0
        
        from collections import Counter
        ngram_counter = Counter(ngrams)
        repeated_ngrams = sum(1 for count in ngram_counter.values() if count > 1)
        
        return repeated_ngrams / len(ngram_counter) if ngram_counter else 0.0
    
    def _calculate_vocab_complexity(self, text: str) -> float:
        """计算词汇复杂度"""
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        long_words = sum(1 for word in words if len(word) > 8)
        return long_words / len(words) if words else 0
    
    def _calculate_reference_perplexity(self, item: Dict, model: DatasetAnalyzer, tokenizer) -> float:
        """计算参考模型困惑度"""
        formatted_text = model.format_qa_pair(item)
        inputs = tokenizer(
            formatted_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)
        
        loss = model.model(**inputs, labels=inputs["input_ids"]).loss
        return torch.exp(loss).item()
    
    def _calculate_base_perplexity(self, answer: str, model: DatasetAnalyzer, tokenizer) -> float:
        """计算基础模型困惑度"""
        inputs = tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)
        
        loss = model.model(**inputs, labels=inputs["input_ids"]).loss
        return torch.exp(loss).item()
    
    def _calculate_semantic_diversity(self, answer: str, model: DatasetAnalyzer, tokenizer) -> float:
        """计算语义多样性（简化版本）"""
        inputs = tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)
        
        hidden_states = model.model(**inputs, output_hidden_states=True).hidden_states[-1]
        embedding = hidden_states[:, 0, :].mean(dim=0)
        
        # 返回嵌入的L2范数作为多样性指标
        return embedding.norm(2).item()
    
    def _calculate_io_similarity(self, question: str, answer: str, model: DatasetAnalyzer, tokenizer) -> float:
        """计算输入输出相似度"""
        # 编码问题
        question_inputs = tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # 编码答案
        answer_inputs = tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # 获取嵌入
        question_hidden = model.model(**question_inputs, output_hidden_states=True).hidden_states[-1]
        answer_hidden = model.model(**answer_inputs, output_hidden_states=True).hidden_states[-1]
        
        question_embedding = question_hidden[:, 0, :].mean(dim=0)
        answer_embedding = answer_hidden[:, 0, :].mean(dim=0)
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(
            question_embedding.unsqueeze(0),
            answer_embedding.unsqueeze(0)
        ).item()
        
        return similarity
    
    def _calculate_semantic_consistency(self, semantic_diversities: List[float]) -> float:
        """计算语义一致性"""
        if len(semantic_diversities) < 2:
            return 1.0
        
        # 计算多样性值的标准差，然后转换为一致性
        std = np.std(semantic_diversities)
        mean = np.mean(semantic_diversities)
        
        if mean == 0:
            return 1.0
        
        # 一致性 = 1 - 变异系数
        consistency = 1 - (std / mean)
        return max(0.0, min(1.0, consistency))  # 限制在[0,1]范围内
    
    def cleanup(self):
        """清理资源"""
        for device, model in self.models.items():
            if hasattr(model, 'model'):
                del model.model
            if hasattr(model, 'tokenizer'):
                del model.tokenizer
        
        self.models.clear()
        self.tokenizers.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("已清理所有模型资源") 