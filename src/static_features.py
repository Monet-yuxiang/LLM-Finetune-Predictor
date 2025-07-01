"""
静态特征提取模块

负责提取不需要模型训练的静态特征，包括：
- 文本统计特征（长度、TTR、n-gram等）
- 语义特征（基于预训练模型的嵌入）
- 困惑度特征
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import torch.nn.functional as F
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import re
import logging

from .data_parsers import DatasetAnalyzer

logger = logging.getLogger(__name__)

class StaticFeatureExtractor(DatasetAnalyzer):
    """静态特征提取器
    
    继承自DatasetAnalyzer，专门用于提取静态特征
    """
    
    def calculate_semantic_diversity(
        self,
        dataset: List[Dict],
        sample_size: int = 100
    ) -> float:
        """计算输出语义多样性
        
        Args:
            dataset: 数据集
            sample_size: 采样大小
            
        Returns:
            语义多样性分数
        """
        sampled_data = self.sample_dataset(dataset, sample_size)
        embeddings = []
        
        with torch.no_grad():
            for item in tqdm(sampled_data, desc="计算数据集输出语义多样性"):
                # 获取答案部分
                _, answer = self.extract_qa_from_item(item)
                
                outputs = self.tokenizer(
                    answer,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # 获取最后一层的隐藏状态
                hidden_states = self.model(**outputs, output_hidden_states=True).hidden_states[-1]
                # 使用[CLS]token的嵌入作为句子表示
                embedding = hidden_states[:, 0, :].mean(dim=0)
                embeddings.append(embedding)
        
        embeddings = torch.stack(embeddings)
        # 计算所有嵌入对之间的余弦相似度
        similarity_matrix = F.cosine_similarity(
            embeddings.unsqueeze(1),
            embeddings.unsqueeze(0),
            dim=2
        )
        # 计算平均相异度
        diversity = 1 - similarity_matrix.mean().item()
        return diversity
    
    def calculate_io_similarity(
        self,
        dataset: List[Dict],
        sample_size: int = 100
    ) -> float:
        """计算输入-输出语义相似度
        
        Args:
            dataset: 数据集
            sample_size: 采样大小
            
        Returns:
            输入输出相似度
        """
        sampled_data = self.sample_dataset(dataset, sample_size)
        similarities = []
        
        with torch.no_grad():
            for item in tqdm(sampled_data, desc="计算数据集输入-输出语义IO相似度"):
                # 获取问题和答案
                question, answer = self.extract_qa_from_item(item)
                
                # 分别编码问题和答案
                question_tokens = self.tokenizer(
                    question,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                answer_tokens = self.tokenizer(
                    answer,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # 获取语义表示
                question_hidden = self.model(**question_tokens, output_hidden_states=True).hidden_states[-1]
                answer_hidden = self.model(**answer_tokens, output_hidden_states=True).hidden_states[-1]
                
                question_embedding = question_hidden[:, 0, :].mean(dim=0)
                answer_embedding = answer_hidden[:, 0, :].mean(dim=0)
                
                # 计算相似度
                similarity = F.cosine_similarity(
                    question_embedding.unsqueeze(0),
                    answer_embedding.unsqueeze(0)
                ).item()
                similarities.append(similarity)
                
        return np.mean(similarities)
    
    def calculate_reference_perplexity(
        self,
        dataset: List[Dict],
        sample_size: int = 100
    ) -> float:
        """计算参考模型困惑度
        
        Args:
            dataset: 数据集
            sample_size: 采样大小
            
        Returns:
            参考模型困惑度
        """
        sampled_data = self.sample_dataset(dataset, sample_size)
        total_perplexity = 0.0
        
        with torch.no_grad():
            for item in tqdm(sampled_data, desc="计算困惑度"):
                # 获取答案部分
                _, answer = self.extract_qa_from_item(item)
                
                outputs = self.tokenizer(
                    answer,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                loss = self.model(**outputs, labels=outputs["input_ids"]).loss
                perplexity = torch.exp(loss).item()
                total_perplexity += perplexity
                
        return total_perplexity / len(sampled_data)
    
    def calculate_base_model_perplexity(
        self,
        dataset: List[Dict],
        sample_size: int = 100
    ) -> float:
        """计算基础模型对输出文本的平均困惑度
        
        Args:
            dataset: 数据集
            sample_size: 采样大小
            
        Returns:
            基础模型困惑度
        """
        sampled_data = self.sample_dataset(dataset, sample_size)
        total_perplexity = 0.0
        
        with torch.no_grad():
            for item in tqdm(sampled_data, desc="计算基础模型困惑度"):
                # 获取答案部分
                _, answer = self.extract_qa_from_item(item)
                
                # 对答案进行编码
                inputs = self.tokenizer(
                    answer,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # 计算困惑度
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                total_perplexity += perplexity
                
        return total_perplexity / len(sampled_data)

    def calculate_length_features(self, dataset: List[Dict], sample_size: int = 100) -> Dict[str, float]:
        """计算长度相关特征
        
        Args:
            dataset: 数据集
            sample_size: 采样大小
            
        Returns:
            长度特征字典
        """
        sampled_data = self.sample_dataset(dataset, sample_size)
        input_lengths = []
        output_lengths = []
        
        for item in tqdm(sampled_data, desc="计算长度特征"):
            question, answer = self.extract_qa_from_item(item)
            
            # 计算输入输出token长度
            input_tokens = self.tokenizer.encode(question, add_special_tokens=False)
            output_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
            
            input_lengths.append(len(input_tokens))
            output_lengths.append(len(output_tokens))
        
        # 计算统计特征
        avg_input_length = np.mean(input_lengths)
        avg_output_length = np.mean(output_lengths)
        io_length_ratio = avg_output_length / avg_input_length if avg_input_length > 0 else 0
        input_length_std = np.std(input_lengths)
        output_length_std = np.std(output_lengths)
        
        return {
            "avg_input_length": avg_input_length,
            "avg_output_length": avg_output_length,
            "io_length_ratio": io_length_ratio,
            "input_length_std": input_length_std,
            "output_length_std": output_length_std
        }
    
    def calculate_ttr(self, text: str) -> float:
        """计算文本的类符/形符比 (TTR)
        
        Args:
            text: 输入文本
            
        Returns:
            TTR值
        """
        # 分词并转换为小写
        tokens = self.tokenizer.tokenize(text.lower())
        if not tokens:
            return 0.0
        
        # 计算唯一词数（类符）和总词数（形符）
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        
        return unique_tokens / total_tokens if total_tokens > 0 else 0.0
    
    def calculate_ngram_repetition(self, text: str, n: int = 3) -> float:
        """计算n-gram重复率
        
        Args:
            text: 输入文本
            n: n-gram的n值
            
        Returns:
            n-gram重复率
        """
        tokens = self.tokenizer.tokenize(text.lower())
        if len(tokens) < n:
            return 0.0
        
        # 生成n-grams
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        if not ngrams:
            return 0.0
        
        # 计算重复的n-gram比例
        ngram_counter = Counter(ngrams)
        repeated_ngrams = sum(1 for count in ngram_counter.values() if count > 1)
        
        return repeated_ngrams / len(ngram_counter) if ngram_counter else 0.0
    
    def calculate_diversity_features(self, dataset: List[Dict], sample_size: int = 100) -> Dict[str, float]:
        """计算多样性相关特征
        
        Args:
            dataset: 数据集
            sample_size: 采样大小
            
        Returns:
            多样性特征字典
        """
        sampled_data = self.sample_dataset(dataset, sample_size)
        
        # 计算TTR
        input_ttrs = []
        output_ttrs = []
        output_ngram_reps = []
        texts = []
        
        for item in tqdm(sampled_data, desc="计算多样性特征"):
            question, answer = self.extract_qa_from_item(item)
            
            # 计算TTR
            input_ttrs.append(self.calculate_ttr(question))
            output_ttrs.append(self.calculate_ttr(answer))
            
            # 计算n-gram重复率 (使用3-gram和4-gram的平均值)
            ngram_rep_3 = self.calculate_ngram_repetition(answer, n=3)
            ngram_rep_4 = self.calculate_ngram_repetition(answer, n=4)
            output_ngram_reps.append((ngram_rep_3 + ngram_rep_4) / 2)
            
            # 收集文本用于计算近似重复
            texts.append(f"{question} {answer}")
        
        # 计算近似重复样本
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity_matrix = sklearn_cosine_similarity(tfidf_matrix)
            # 将对角线设为0（排除自身）
            np.fill_diagonal(similarity_matrix, 0)
            # 计算相似度大于0.8的样本对数量
            duplicate_pairs = np.sum(similarity_matrix > 0.8)
            approximate_duplicates = duplicate_pairs / (len(texts) * (len(texts) - 1)) if len(texts) > 1 else 0
        except Exception as e:
            logger.warning(f"计算近似重复样本时发生错误: {str(e)}")
            approximate_duplicates = 0.0
        
        return {
            "input_ttr": np.mean(input_ttrs),
            "output_ttr": np.mean(output_ttrs),
            "output_ngram_repetition": np.mean(output_ngram_reps),
            "approximate_duplicates": approximate_duplicates
        }
    
    def calculate_text_statistics_features(self, dataset: List[Dict], sample_size: int = 100) -> Dict[str, float]:
        """计算文本统计特征（组合方法）
        
        Args:
            dataset: 数据集
            sample_size: 采样大小
            
        Returns:
            文本统计特征字典
        """
        sampled_data = self.sample_dataset(dataset, sample_size)
        
        # 初始化所有特征列表
        input_lengths = []
        output_lengths = []
        input_ttrs = []
        output_ttrs = []
        output_ngram_reps = []
        texts = []
        vocab_complexity = []
        
        for item in tqdm(sampled_data, desc="计算文本统计特征"):
            question, answer = self.extract_qa_from_item(item)
            
            # 1. 计算长度特征
            input_tokens = self.tokenizer.encode(question, add_special_tokens=False)
            output_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
            input_lengths.append(len(input_tokens))
            output_lengths.append(len(output_tokens))
            
            # 2. 计算TTR
            input_ttrs.append(self.calculate_ttr(question))
            output_ttrs.append(self.calculate_ttr(answer))
            
            # 3. 计算n-gram重复率
            ngram_rep_3 = self.calculate_ngram_repetition(answer, n=3)
            ngram_rep_4 = self.calculate_ngram_repetition(answer, n=4)
            output_ngram_reps.append((ngram_rep_3 + ngram_rep_4) / 2)
            
            # 4. 收集文本用于计算近似重复
            texts.append(f"{question} {answer}")
            
            # 5. 计算词汇复杂度（长词比例）
            combined_text = f"{question} {answer}".lower()
            words = re.findall(r'\b\w+\b', combined_text)
            long_words = sum(1 for word in words if len(word) > 8)
            complexity = long_words / len(words) if words else 0
            vocab_complexity.append(complexity)
        
        # 计算长度统计特征
        avg_input_length = np.mean(input_lengths)
        avg_output_length = np.mean(output_lengths)
        io_length_ratio = avg_output_length / avg_input_length if avg_input_length > 0 else 0
        input_length_std = np.std(input_lengths)
        output_length_std = np.std(output_lengths)
        
        # 计算多样性特征
        avg_input_ttr = np.mean(input_ttrs)
        avg_output_ttr = np.mean(output_ttrs)
        avg_ngram_repetition = np.mean(output_ngram_reps)
        avg_vocab_complexity = np.mean(vocab_complexity)
        
        # 计算近似重复样本
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity_matrix = sklearn_cosine_similarity(tfidf_matrix)
            np.fill_diagonal(similarity_matrix, 0)
            duplicate_pairs = np.sum(similarity_matrix > 0.8)
            approximate_duplicates = duplicate_pairs / (len(texts) * (len(texts) - 1)) if len(texts) > 1 else 0
        except Exception as e:
            logger.warning(f"计算近似重复样本时发生错误: {str(e)}")
            approximate_duplicates = 0.0
        
        return {
            # 长度特征
            "avg_input_length": avg_input_length,
            "avg_output_length": avg_output_length,
            "io_length_ratio": io_length_ratio,
            "input_length_std": input_length_std,
            "output_length_std": output_length_std,
            # 多样性特征
            "input_ttr": avg_input_ttr,
            "output_ttr": avg_output_ttr,
            "output_ngram_repetition": avg_ngram_repetition,
            "approximate_duplicates": approximate_duplicates,
            # 新增词汇复杂度特征
            "vocab_complexity": avg_vocab_complexity
        }
    
    def calculate_perplexity_features(self, dataset: List[Dict], sample_size: int = 100) -> Dict[str, float]:
        """计算困惑度相关特征（组合方法）
        
        Args:
            dataset: 数据集
            sample_size: 采样大小
            
        Returns:
            困惑度特征字典
        """
        sampled_data = self.sample_dataset(dataset, sample_size)
        reference_perplexities = []
        base_perplexities = []
        
        with torch.no_grad():
            for item in tqdm(sampled_data, desc="计算困惑度特征"):
                # 获取答案部分
                _, answer = self.extract_qa_from_item(item)
                
                # 计算参考模型困惑度（对整个QA对）
                formatted_text = self.format_qa_pair(item)
                ref_inputs = self.tokenizer(
                    formatted_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                ref_loss = self.model(**ref_inputs, labels=ref_inputs["input_ids"]).loss
                ref_perplexity = torch.exp(ref_loss).item()
                reference_perplexities.append(ref_perplexity)
                
                # 计算基础模型困惑度（仅对答案）
                base_inputs = self.tokenizer(
                    answer,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                base_loss = self.model(**base_inputs, labels=base_inputs["input_ids"]).loss
                base_perplexity = torch.exp(base_loss).item()
                base_perplexities.append(base_perplexity)
        
        # 计算统计特征
        avg_ref_perplexity = np.mean(reference_perplexities)
        avg_base_perplexity = np.mean(base_perplexities)
        
        # 计算困惑度变化率（参考模型 vs 基础模型）
        perplexity_change_rate = (avg_ref_perplexity - avg_base_perplexity) / avg_base_perplexity if avg_base_perplexity > 0 else 0.0
        
        # 计算困惑度稳定性（标准差）
        ref_perplexity_std = np.std(reference_perplexities)
        base_perplexity_std = np.std(base_perplexities)
        
        return {
            "reference_perplexity": avg_ref_perplexity,
            "base_model_perplexity": avg_base_perplexity,
            "perplexity_change_rate": perplexity_change_rate,
            "reference_perplexity_std": ref_perplexity_std,
            "base_perplexity_std": base_perplexity_std
        }
    
    def calculate_semantic_features(self, dataset: List[Dict], sample_size: int = 100) -> Dict[str, float]:
        """计算语义相关特征（组合方法）
        
        Args:
            dataset: 数据集
            sample_size: 采样大小
            
        Returns:
            语义特征字典
        """
        sampled_data = self.sample_dataset(dataset, sample_size)
        embeddings = []
        similarities = []
        semantic_consistencies = []
        
        with torch.no_grad():
            for item in tqdm(sampled_data, desc="计算语义特征"):
                question, answer = self.extract_qa_from_item(item)
                
                # 1. 计算语义多样性（基于答案嵌入）
                answer_outputs = self.tokenizer(
                    answer,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                answer_hidden = self.model(**answer_outputs, output_hidden_states=True).hidden_states[-1]
                answer_embedding = answer_hidden[:, 0, :].mean(dim=0)
                embeddings.append(answer_embedding)
                
                # 2. 计算输入输出相似度
                question_outputs = self.tokenizer(
                    question,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                question_hidden = self.model(**question_outputs, output_hidden_states=True).hidden_states[-1]
                question_embedding = question_hidden[:, 0, :].mean(dim=0)
                
                similarity = F.cosine_similarity(
                    question_embedding.unsqueeze(0),
                    answer_embedding.unsqueeze(0)
                ).item()
                similarities.append(similarity)
                
                # 3. 计算语义一致性（基于答案内部一致性）
                sentences = re.split(r'[.!?]+', answer)
                if len(sentences) > 1:
                    sentence_embeddings = []
                    for sentence in sentences:
                        if sentence.strip():
                            sent_outputs = self.tokenizer(
                                sentence.strip(),
                                return_tensors="pt",
                                truncation=True,
                                max_length=256
                            ).to(self.device)
                            sent_hidden = self.model(**sent_outputs, output_hidden_states=True).hidden_states[-1]
                            sent_embedding = sent_hidden[:, 0, :].mean(dim=0)
                            sentence_embeddings.append(sent_embedding)
                    
                    if len(sentence_embeddings) > 1:
                        # 计算句子间的平均相似度
                        sent_similarities = []
                        for i in range(len(sentence_embeddings)):
                            for j in range(i + 1, len(sentence_embeddings)):
                                sim = F.cosine_similarity(
                                    sentence_embeddings[i].unsqueeze(0),
                                    sentence_embeddings[j].unsqueeze(0)
                                ).item()
                                sent_similarities.append(sim)
                        semantic_consistencies.append(np.mean(sent_similarities))
                    else:
                        semantic_consistencies.append(1.0)  # 单句子的情况
                else:
                    semantic_consistencies.append(1.0)  # 单句子的情况
        
        # 计算语义多样性
        embeddings = torch.stack(embeddings)
        similarity_matrix = F.cosine_similarity(
            embeddings.unsqueeze(1),
            embeddings.unsqueeze(0),
            dim=2
        )
        diversity = 1 - similarity_matrix.mean().item()
        
        # 计算统计特征
        avg_similarity = np.mean(similarities)
        avg_semantic_consistency = np.mean(semantic_consistencies)
        
        return {
            "semantic_diversity": diversity,
            "io_similarity": avg_similarity,
            "semantic_consistency": avg_semantic_consistency
        }

    def extract_all_static_features(self, dataset: List[Dict], sample_size: int = 100) -> Dict[str, float]:
        """一次性提取所有静态特征"""
        features = {}
        features.update(self.calculate_text_statistics_features(dataset, sample_size))
        features.update(self.calculate_perplexity_features(dataset, sample_size))
        features.update(self.calculate_semantic_features(dataset, sample_size))
        return features 