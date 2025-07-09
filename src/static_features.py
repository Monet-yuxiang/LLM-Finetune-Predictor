"""
静态特征提取模块

负责提取不需要模型训练的静态特征，包括：
- 文本统计特征（长度、TTR、n-gram等）
- 语义特征（基于预训练模型的嵌入）
- 困惑度特征

支持最终格式数据：context_text + qa_pairs
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import torch.nn.functional as F
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import re
import logging

logger = logging.getLogger(__name__)

class StaticFeatureExtractor:
    """静态特征提取器
    
    直接接收模型和tokenizer，支持最终格式数据
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[str] = None
    ):
        """初始化静态特征提取器
        
        Args:
            model: 预训练模型
            tokenizer: 分词器
            device: 计算设备，如果为None则自动检测
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 确保模型在正确的设备上
        if next(self.model.parameters()).device != torch.device(self.device):
            self.model = self.model.to(self.device)
    
    def is_final_format(self, dataset: List[Dict]) -> bool:
        """判断是否为最终格式数据"""
        if not dataset:
            return False
        sample = dataset[0]
        return "context_text" in sample and "qa_pairs" in sample
    
    def extract_qa_from_final_format(self, item: Dict) -> List[Tuple[str, str]]:
        """从最终格式数据中提取所有问答对
        
        Args:
            item: 最终格式数据项 {"context_text": str, "qa_pairs": List[Dict]}
            
        Returns:
            List[Tuple[str, str]]: 问答对列表，每个元素为 (question, answer)
        """
        context = item.get("context_text", "")
        qa_pairs = item.get("qa_pairs", [])
        
        result = []
        for qa in qa_pairs:
            question = qa.get("question", "")
            answer = qa.get("output", "")
            
            if question and answer:
                # 将context和question组合
                full_question = f"{context}\n\n{question}" if context else question
                result.append((full_question, answer))
        
        return result
    
    def format_qa_pair(self, question: str, answer: str) -> str:
        """将问答对格式化为模型输入格式
        
        Args:
            question: 问题
            answer: 答案
            
        Returns:
            格式化后的文本
        """
        return f"问：{question}\n答：{answer}"
    
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

    def extract_all_static_features(self, dataset: List[Dict], sample_size: int = 100, batch_size: int = 8) -> Dict[str, float]:
        """一次性提取所有静态特征（支持最终格式，批量推理）"""
        logger.info("开始提取所有静态特征（支持最终格式，批量推理）...")
        is_final_format = self.is_final_format(dataset)
        logger.info(f"检测到数据格式: {'最终格式' if is_final_format else '标准格式'}")
        sampled_data = self.sample_dataset(dataset, sample_size)
        # 收集所有QA对
        qa_list = []
        if is_final_format:
            for item in sampled_data:
                qa_list.extend(self.extract_qa_from_final_format(item))
        else:
            for item in sampled_data:
                qa_list.append(self._extract_qa_from_standard_format(item))
        # 文本统计特征
        input_texts = [q for q, a in qa_list]
        output_texts = [a for q, a in qa_list]
        input_lengths = [len(self.tokenizer.encode(q, add_special_tokens=False)) for q in input_texts]
        output_lengths = [len(self.tokenizer.encode(a, add_special_tokens=False)) for a in output_texts]
        input_ttrs = [self.calculate_ttr(q) for q in input_texts]
        output_ttrs = [self.calculate_ttr(a) for a in output_texts]
        output_ngram_reps = [(self.calculate_ngram_repetition(a, n=3) + self.calculate_ngram_repetition(a, n=4)) / 2 for a in output_texts]
        texts = [f"{q} {a}" for q, a in qa_list]
        vocab_complexity = []
        for q, a in qa_list:
            combined_text = f"{q} {a}".lower()
            words = re.findall(r'\b\w+\b', combined_text)
            long_words = sum(1 for word in words if len(word) > 8)
            complexity = long_words / len(words) if words else 0
            vocab_complexity.append(complexity)
        # 批量困惑度特征
        reference_perplexities = []
        base_perplexities = []
        with torch.no_grad():
            for i in range(0, len(qa_list), batch_size):
                batch_qa = qa_list[i:i+batch_size]
                batch_formatted = [self.format_qa_pair(q, a) for q, a in batch_qa]
                ref_inputs = self.tokenizer(
                    batch_formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                ref_outputs = self.model(**ref_inputs, labels=ref_inputs["input_ids"])
                ref_loss = ref_outputs.loss
                if hasattr(ref_outputs, 'loss') and ref_outputs.loss is not None:
                    # loss是batch均值，需还原每条
                    logits = ref_outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = ref_inputs["input_ids"][..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = loss.view(shift_labels.size())
                    seq_loss = loss.mean(dim=1)  # 每条样本的loss
                    perplexity = torch.exp(seq_loss).tolist()
                    reference_perplexities.extend(perplexity)
                else:
                    reference_perplexities.extend([float('nan')] * len(batch_qa))
                # 基础模型困惑度（仅对答案）
                base_inputs = self.tokenizer(
                    [a for q, a in batch_qa],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                base_outputs = self.model(**base_inputs, labels=base_inputs["input_ids"])
                base_loss = base_outputs.loss
                if hasattr(base_outputs, 'loss') and base_outputs.loss is not None:
                    logits = base_outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = base_inputs["input_ids"][..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = loss.view(shift_labels.size())
                    seq_loss = loss.mean(dim=1)
                    perplexity = torch.exp(seq_loss).tolist()
                    base_perplexities.extend(perplexity)
                else:
                    base_perplexities.extend([float('nan')] * len(batch_qa))
        # 批量语义特征（embedding）
        embeddings = []
        similarities = []
        semantic_consistencies = []
        with torch.no_grad():
            for i in range(0, len(qa_list), batch_size):
                batch_qa = qa_list[i:i+batch_size]
                batch_answers = [a for q, a in batch_qa]
                answer_inputs = self.tokenizer(
                    batch_answers,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                answer_outputs = self.model(**answer_inputs, output_hidden_states=True)
                answer_hidden = answer_outputs.hidden_states[-1]
                answer_embedding = answer_hidden[:, 0, :].mean(dim=1)
                embeddings.extend(answer_embedding)
                # 输入输出相似度
                batch_questions = [q for q, a in batch_qa]
                question_inputs = self.tokenizer(
                    batch_questions,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                question_outputs = self.model(**question_inputs, output_hidden_states=True)
                question_hidden = question_outputs.hidden_states[-1]
                question_embedding = question_hidden[:, 0, :].mean(dim=1)
                # 修正：保证为二维
                if question_embedding.dim() == 1:
                    question_embedding = question_embedding.unsqueeze(0)
                if answer_embedding.dim() == 1:
                    answer_embedding = answer_embedding.unsqueeze(0)
                sim = F.cosine_similarity(question_embedding, answer_embedding, dim=1)
                similarities.extend(sim.cpu().tolist())
                # 语义一致性（这里简单用1.0填充，实际可自定义）
                semantic_consistencies.extend([1.0] * len(batch_qa))
        # ===== 计算所有统计特征 =====
        features = {}
        avg_input_length = np.mean(input_lengths)
        avg_output_length = np.mean(output_lengths)
        io_length_ratio = avg_output_length / avg_input_length if avg_input_length > 0 else 0
        input_length_std = np.std(input_lengths)
        output_length_std = np.std(output_lengths)
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
        features.update({
            "avg_input_length": avg_input_length,
            "avg_output_length": avg_output_length,
            "io_length_ratio": io_length_ratio,
            "input_length_std": input_length_std,
            "output_length_std": output_length_std,
            "input_ttr": avg_input_ttr,
            "output_ttr": avg_output_ttr,
            "output_ngram_repetition": avg_ngram_repetition,
            "approximate_duplicates": approximate_duplicates,
            "vocab_complexity": avg_vocab_complexity
        })
        # 2. 困惑度特征
        avg_ref_perplexity = np.mean(reference_perplexities)
        avg_base_perplexity = np.mean(base_perplexities)
        perplexity_change_rate = (avg_ref_perplexity - avg_base_perplexity) / avg_base_perplexity if avg_base_perplexity > 0 else 0.0
        ref_perplexity_std = np.std(reference_perplexities)
        base_perplexity_std = np.std(base_perplexities)
        features.update({
            "reference_perplexity": avg_ref_perplexity,
            "base_model_perplexity": avg_base_perplexity,
            "perplexity_change_rate": perplexity_change_rate,
            "reference_perplexity_std": ref_perplexity_std,
            "base_perplexity_std": base_perplexity_std
        })
        # 3. 语义特征
        if embeddings and len(embeddings) > 1:
            # 如果embeddings是list，每个元素shape为[hidden]，stack后为[N, hidden]
            if isinstance(embeddings, list):
                embeddings = torch.stack(embeddings)
            # 如果embeddings已经是[N, hidden]的张量，直接用
            if embeddings.dim() == 1:
                # 只有一个样本，直接返回0
                diversity = 0.0
            else:
                similarity_matrix = F.cosine_similarity(
                    embeddings.unsqueeze(1),
                    embeddings.unsqueeze(0),
                    dim=-1
                )
                diversity = 1 - similarity_matrix.mean().item()
        else:
            diversity = 0.0
        avg_similarity = np.mean(similarities)
        avg_semantic_consistency = np.mean(semantic_consistencies)
        features.update({
            "semantic_diversity": diversity,
            "io_similarity": avg_similarity,
            "semantic_consistency": avg_semantic_consistency
        })
        logger.info(f"静态特征提取完成，共提取 {len(features)} 个特征")
        return features
    
    def _extract_qa_from_standard_format(self, item: Dict) -> Tuple[str, str]:
        """从标准格式数据中提取问答对（保持原有逻辑）"""
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
    
    def _process_single_qa(self, question: str, answer: str, input_lengths, output_lengths,
                          input_ttrs, output_ttrs, output_ngram_reps, texts, vocab_complexity,
                          reference_perplexities, base_perplexities, embeddings, similarities,
                          semantic_consistencies):
        """处理单个问答对的特征提取"""
        # ===== 1. 文本统计特征 =====
        # 计算长度特征
        input_tokens = self.tokenizer.encode(question, add_special_tokens=False)
        output_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
        input_lengths.append(len(input_tokens))
        output_lengths.append(len(output_tokens))
        
        # 计算TTR
        input_ttrs.append(self.calculate_ttr(question))
        output_ttrs.append(self.calculate_ttr(answer))
        
        # 计算n-gram重复率
        ngram_rep_3 = self.calculate_ngram_repetition(answer, n=3)
        ngram_rep_4 = self.calculate_ngram_repetition(answer, n=4)
        output_ngram_reps.append((ngram_rep_3 + ngram_rep_4) / 2)
        
        # 收集文本用于计算近似重复
        texts.append(f"{question} {answer}")
        
        # 计算词汇复杂度（长词比例）
        combined_text = f"{question} {answer}".lower()
        words = re.findall(r'\b\w+\b', combined_text)
        long_words = sum(1 for word in words if len(word) > 8)
        complexity = long_words / len(words) if words else 0
        vocab_complexity.append(complexity)
        
        # ===== 2. 困惑度特征 =====
        # 计算参考模型困惑度（对整个QA对）
        formatted_text = self.format_qa_pair(question, answer)
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
        
        # ===== 3. 语义特征 =====
        # 计算语义多样性（基于答案嵌入）
        answer_outputs = self.tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        answer_hidden = self.model(**answer_outputs, output_hidden_states=True).hidden_states[-1]
        answer_embedding = answer_hidden[:, 0, :].mean(dim=0)
        embeddings.append(answer_embedding)
        
        # 计算输入输出相似度
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
        
        # 计算语义一致性（基于答案内部一致性）
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