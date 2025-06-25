import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import cosine_similarity
from dataclasses import dataclass
from tqdm import tqdm
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType
import torch.optim as optim
import logging
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import re

logger = logging.getLogger(__name__)

@dataclass
class HyperParams:
    learning_rate: float
    lora_r: int
    lora_alpha: float
    # 可以添加其他超参数

class DatasetAnalyzer:
    def __init__(
        self,
        base_model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
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
        """将 QA 对格式化为模型输入格式"""
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
        
    def get_hyperparams_features(self, hyperparams: HyperParams) -> Dict[str, float]:
        """提取超参数特征"""
        return {
            "learning_rate": hyperparams.learning_rate,
            "lora_r": hyperparams.lora_r,
            "lora_alpha": hyperparams.lora_alpha
        }
    
    def get_dataset_size(self, dataset: List[Dict]) -> int:
        """计算数据集大小"""
        return len(dataset)
    
    def calculate_initial_loss(
        self,
        dataset: List[Dict],
        sample_size: int = 100
    ) -> float:
        """计算初始平均损失"""
        if len(dataset) < sample_size:
            sample_size = len(dataset)
            
        sampled_data = np.random.choice(dataset, sample_size, replace=False)
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
                
        return total_loss / sample_size
    
    def calculate_semantic_diversity(
        self,
        dataset: List[Dict],
        sample_size: int = 100
    ) -> float:
        """计算输出语义多样性"""
        if len(dataset) < sample_size:
            sample_size = len(dataset)
            
        sampled_data = np.random.choice(dataset, sample_size, replace=False)
        embeddings = []
        
        with torch.no_grad():
            for item in tqdm(sampled_data, desc="计算数据集输出语义多样性"):
                # 获取答案部分
                if "conversations" in item:
                    answer = next(msg["content"] for msg in item["conversations"] 
                                if msg["role"] == "assistant")
                else:
                    answer = item["output"]
                
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
        """计算输入-输出语义相似度"""
        if len(dataset) < sample_size:
            sample_size = len(dataset)
            
        sampled_data = np.random.choice(dataset, sample_size, replace=False)
        similarities = []
        
        with torch.no_grad():
            for item in tqdm(sampled_data, desc="计算数据集输入-输出语义IO相似度"):
                # 获取问题和答案
                if "conversations" in item:
                    question = next(msg["content"] for msg in item["conversations"] 
                                  if msg["role"] == "user")
                    answer = next(msg["content"] for msg in item["conversations"] 
                                if msg["role"] == "assistant")
                else:
                    question = item["input"]
                    answer = item["output"]
                
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
        """计算参考模型困惑度"""
        if len(dataset) < sample_size:
            sample_size = len(dataset)
            
        sampled_data = np.random.choice(dataset, sample_size, replace=False)
        total_perplexity = 0.0
        
        with torch.no_grad():
            for item in tqdm(sampled_data, desc="计算困惑度"):
                # 获取答案部分
                if "conversations" in item:
                    answer = next(msg["content"] for msg in item["conversations"] 
                                if msg["role"] == "assistant")
                else:
                    answer = item["output"]
                
                outputs = self.tokenizer(
                    answer,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                loss = self.model(**outputs, labels=outputs["input_ids"]).loss
                perplexity = torch.exp(loss).item()
                total_perplexity += perplexity
                
        return total_perplexity / sample_size
    
    def calculate_base_model_perplexity(
        self,
        dataset: List[Dict],
        sample_size: int = 100
    ) -> float:
        """计算基础模型对输出文本的平均困惑度"""
        if len(dataset) < sample_size:
            sample_size = len(dataset)
            
        sampled_data = np.random.choice(dataset, sample_size, replace=False)
        total_perplexity = 0.0
        
        with torch.no_grad():
            for item in tqdm(sampled_data, desc="计算基础模型困惑度"):
                # 获取答案部分
                if "conversations" in item:
                    answer = next(msg["content"] for msg in item["conversations"] 
                                if msg["role"] == "assistant")
                else:
                    answer = item["output"]
                
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
                
        return total_perplexity / sample_size

    def calculate_loss_decay_rate(
        self,
        dataset: List[Dict],
        hyperparams: HyperParams,
        probe_steps: int = 100,
        sample_size: int = 50
    ) -> Tuple[float, float, float]:
        """计算损失下降率、平均梯度范数和梯度一致性
        
        实现目标公式：
        - 损失下降率：(初始损失-最终损失)/初始损失
        - 平均梯度范数：1/T * Σ|∇_θ L_t|_2，其中T=探针步数
        - 梯度一致性：跨批次余弦相似度，识别数据噪声
        
        这是对"可学性"最直接的探测：
        - 损失下降率直接反映了数据提供的学习信号是否清晰有效
        - 学习稳定性：过大→震荡，过小→停滞
        - 梯度一致性：识别数据噪声和训练稳定性
        
        Returns:
            Tuple[float, float, float]: (损失下降率, 平均梯度范数, 梯度一致性)
        """
        if len(dataset) < sample_size:
            sample_size = len(dataset)
            
        sampled_data = np.random.choice(dataset, sample_size, replace=False)
        
        # 配置 LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=hyperparams.lora_r,
            lora_alpha=hyperparams.lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # 明确指定目标模块
        )
        
        try:
            # 创建 PEFT 模型
            model = get_peft_model(self.model, peft_config)
            # 确保模型处于训练模式
            model.train()
            
            # 准备优化器
            optimizer = optim.AdamW(model.parameters(), lr=hyperparams.learning_rate)
            
            # 记录初始损失和每步的梯度范数
            initial_loss = 0.0
            step_grad_norms = []  # 存储每步的梯度范数
            batch_gradients = []  # 存储每批次的梯度向量（用于计算一致性）
            
            # 准备训练数据
            train_inputs = []
            for item in sampled_data:
                formatted_text = self.format_qa_pair(item)
                inputs = self.tokenizer(
                    formatted_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                # 确保所有输入都在正确的设备上
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                train_inputs.append(inputs)
            
            # 将数据分成批次用于梯度一致性计算
            batch_size = max(1, len(train_inputs) // 5)  # 5个批次
            num_batches = len(train_inputs) // batch_size
            
            # 进行探针训练
            for step in tqdm(range(probe_steps), desc="执行梯度探针"):
                total_loss = 0.0
                valid_samples = 0
                
                # 清空梯度
                optimizer.zero_grad()
                
                # 计算当前步骤的总损失
                for inputs in train_inputs:
                    try:
                        outputs = model(**inputs, labels=inputs["input_ids"])
                        loss = outputs.loss
                        
                        # 检查损失值是否有效
                        if not torch.isfinite(loss):
                            logger.warning(f"检测到无效的损失值: {loss.item()}")
                            continue
                            
                        total_loss += loss.item()
                        valid_samples += 1
                        
                    except RuntimeError as e:
                        logger.warning(f"在训练步骤中发生错误: {str(e)}")
                        continue
                
                # 如果有有效样本，计算梯度
                if valid_samples > 0:
                    # 计算平均损失并反向传播
                    avg_loss = total_loss / valid_samples
                    loss_tensor = torch.tensor(avg_loss, requires_grad=True, device=self.device)
                    loss_tensor.backward()
                    
                    # 计算当前步骤的梯度范数：|∇_θ L_t|_2
                    step_grad_norm = 0.0
                    for param in model.parameters():
                        if param.grad is not None:
                            step_grad_norm += param.grad.norm(2).item() ** 2
                    step_grad_norm = step_grad_norm ** 0.5  # L2范数
                    
                    # 检查梯度是否有效
                    if torch.isfinite(torch.tensor(step_grad_norm)):
                        step_grad_norms.append(step_grad_norm)
                        
                        if step == 0:
                            initial_loss = avg_loss
                        
                        # 每10步计算一次批次梯度一致性
                        if step % 10 == 0 and num_batches >= 2:
                            batch_grad_vectors = []
                            
                            # 计算每个批次的梯度向量
                            for batch_idx in range(num_batches):
                                start_idx = batch_idx * batch_size
                                end_idx = start_idx + batch_size
                                batch_inputs = train_inputs[start_idx:end_idx]
                                
                                # 清空梯度
                                optimizer.zero_grad()
                                
                                # 计算批次损失
                                batch_loss = 0.0
                                batch_valid = 0
                                for inputs in batch_inputs:
                                    try:
                                        outputs = model(**inputs, labels=inputs["input_ids"])
                                        loss = outputs.loss
                                        if torch.isfinite(loss):
                                            batch_loss += loss.item()
                                            batch_valid += 1
                                    except:
                                        continue
                                
                                if batch_valid > 0:
                                    # 计算批次梯度
                                    avg_batch_loss = batch_loss / batch_valid
                                    batch_loss_tensor = torch.tensor(avg_batch_loss, requires_grad=True, device=self.device)
                                    batch_loss_tensor.backward()
                                    
                                    # 提取梯度向量
                                    grad_vector = []
                                    for param in model.parameters():
                                        if param.grad is not None:
                                            grad_vector.extend(param.grad.flatten().cpu().numpy())
                                    
                                    if grad_vector:
                                        batch_grad_vectors.append(np.array(grad_vector))
                            
                            # 计算批次间的梯度一致性
                            if len(batch_grad_vectors) >= 2:
                                batch_similarities = []
                                for i in range(len(batch_grad_vectors)):
                                    for j in range(i + 1, len(batch_grad_vectors)):
                                        # 确保两个梯度向量长度相同
                                        min_len = min(len(batch_grad_vectors[i]), len(batch_grad_vectors[j]))
                                        if min_len > 0:
                                            vec1 = batch_grad_vectors[i][:min_len]
                                            vec2 = batch_grad_vectors[j][:min_len]
                                            
                                            # 计算余弦相似度
                                            dot_product = np.dot(vec1, vec2)
                                            norm1 = np.linalg.norm(vec1)
                                            norm2 = np.linalg.norm(vec2)
                                            
                                            if norm1 > 0 and norm2 > 0:
                                                similarity = dot_product / (norm1 * norm2)
                                                batch_similarities.append(similarity)
                                
                                if batch_similarities:
                                    batch_gradients.append(np.mean(batch_similarities))
                        
                        # 执行优化步骤
                        optimizer.step()
                    else:
                        logger.warning(f"步骤 {step} 检测到无效的梯度范数: {step_grad_norm}")
                        step_grad_norms.append(0.0)
                else:
                    step_grad_norms.append(0.0)
            
            # 计算最终损失
            final_loss = avg_loss if 'avg_loss' in locals() else initial_loss
            
            # 计算损失下降率
            loss_decay_rate = (initial_loss - final_loss) / initial_loss if initial_loss != 0 else 0.0
            
            # 计算平均梯度范数：1/T * Σ|∇_θ L_t|_2
            avg_grad_norm = np.mean(step_grad_norms) if step_grad_norms else 0.0
            
            # 计算梯度一致性：跨批次余弦相似度的平均值
            gradient_consistency = np.mean(batch_gradients) if batch_gradients else 0.0
            
        except Exception as e:
            logger.error(f"在计算损失下降率时发生错误: {str(e)}")
            # 发生错误时返回默认值
            loss_decay_rate = 0.0
            avg_grad_norm = 0.0
            gradient_consistency = 0.0
        finally:
            # 恢复模型到评估模式
            if 'model' in locals():
                model.eval()
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return loss_decay_rate, avg_grad_norm, gradient_consistency
    
    def calculate_length_features(self, dataset: List[Dict], sample_size: int = 100) -> Dict[str, float]:
        """计算长度相关特征
        
        Returns:
            Dict[str, float]: 包含以下特征:
            - avg_input_length: 平均输入长度
            - avg_output_length: 平均输出长度
            - io_length_ratio: 输入输出长度比
            - input_length_std: 输入长度标准差
            - output_length_std: 输出长度标准差
        """
        if len(dataset) < sample_size:
            sample_size = len(dataset)
            
        sampled_data = np.random.choice(dataset, sample_size, replace=False)
        input_lengths = []
        output_lengths = []
        
        for item in tqdm(sampled_data, desc="计算长度特征"):
            if "conversations" in item:
                # 处理 Qwen2.5 格式
                question = next(msg["content"] for msg in item["conversations"] 
                              if msg["role"] == "user")
                answer = next(msg["content"] for msg in item["conversations"] 
                            if msg["role"] == "assistant")
            else:
                question = item["input"]
                answer = item["output"]
            
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
        """计算文本的类符/形符比 (TTR)"""
        # 分词并转换为小写
        tokens = self.tokenizer.tokenize(text.lower())
        if not tokens:
            return 0.0
        
        # 计算唯一词数（类符）和总词数（形符）
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        
        return unique_tokens / total_tokens if total_tokens > 0 else 0.0
    
    def calculate_ngram_repetition(self, text: str, n: int = 3) -> float:
        """计算n-gram重复率"""
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
        
        Returns:
            Dict[str, float]: 包含以下特征:
            - input_ttr: 输入TTR
            - output_ttr: 输出TTR
            - output_ngram_repetition: 输出n-gram重复率
            - approximate_duplicates: 近似重复样本比例
        """
        if len(dataset) < sample_size:
            sample_size = len(dataset)
            
        sampled_data = np.random.choice(dataset, sample_size, replace=False)
        
        # 计算TTR
        input_ttrs = []
        output_ttrs = []
        output_ngram_reps = []
        texts = []
        
        for item in tqdm(sampled_data, desc="计算多样性特征"):
            if "conversations" in item:
                question = next(msg["content"] for msg in item["conversations"] 
                              if msg["role"] == "user")
                answer = next(msg["content"] for msg in item["conversations"] 
                            if msg["role"] == "assistant")
            else:
                question = item["input"]
                answer = item["output"]
            
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
        
        一次性计算所有文本统计特征，提高效率：
        - 长度特征：输入输出长度、长度比、标准差
        - 多样性特征：TTR、n-gram重复率、近似重复
        - 词汇特征：词汇覆盖度、复杂度
        
        Returns:
            Dict[str, float]: 包含所有文本统计特征
        """
        if len(dataset) < sample_size:
            sample_size = len(dataset)
            
        sampled_data = np.random.choice(dataset, sample_size, replace=False)
        
        # 初始化所有特征列表
        input_lengths = []
        output_lengths = []
        input_ttrs = []
        output_ttrs = []
        output_ngram_reps = []
        texts = []
        vocab_complexity = []
        
        for item in tqdm(sampled_data, desc="计算文本统计特征"):
            if "conversations" in item:
                question = next(msg["content"] for msg in item["conversations"] 
                              if msg["role"] == "user")
                answer = next(msg["content"] for msg in item["conversations"] 
                            if msg["role"] == "assistant")
            else:
                question = item["input"]
                answer = item["output"]
            
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
        
        一次性计算所有困惑度特征：
        - 参考模型困惑度
        - 基础模型困惑度
        - 困惑度变化率
        
        Returns:
            Dict[str, float]: 包含所有困惑度特征
        """
        if len(dataset) < sample_size:
            sample_size = len(dataset)
            
        sampled_data = np.random.choice(dataset, sample_size, replace=False)
        reference_perplexities = []
        base_perplexities = []
        
        with torch.no_grad():
            for item in tqdm(sampled_data, desc="计算困惑度特征"):
                # 获取答案部分
                if "conversations" in item:
                    answer = next(msg["content"] for msg in item["conversations"] 
                                if msg["role"] == "assistant")
                else:
                    answer = item["output"]
                
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
        
        一次性计算所有语义特征：
        - 语义多样性
        - 输入输出相似度
        - 语义一致性
        
        Returns:
            Dict[str, float]: 包含所有语义特征
        """
        if len(dataset) < sample_size:
            sample_size = len(dataset)
            
        sampled_data = np.random.choice(dataset, sample_size, replace=False)
        embeddings = []
        similarities = []
        semantic_consistencies = []
        
        with torch.no_grad():
            for item in tqdm(sampled_data, desc="计算语义特征"):
                if "conversations" in item:
                    question = next(msg["content"] for msg in item["conversations"] 
                                  if msg["role"] == "user")
                    answer = next(msg["content"] for msg in item["conversations"] 
                                if msg["role"] == "assistant")
                else:
                    question = item["input"]
                    answer = item["output"]
                
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

    def analyze_dataset(
        self,
        dataset: List[Dict],
        hyperparams: HyperParams,
        sample_size: int = 100
    ) -> Dict[str, float]:
        """综合分析数据集特征"""
        features = {}
        
        # 第一优先级特征
        features.update(self.get_hyperparams_features(hyperparams))
        features["dataset_size"] = self.get_dataset_size(dataset)
        features["initial_loss"] = self.calculate_initial_loss(dataset, sample_size)
        
        # 组合语义特征
        semantic_features = self.calculate_semantic_features(dataset, sample_size)
        features.update(semantic_features)
        
        # 组合困惑度特征
        perplexity_features = self.calculate_perplexity_features(dataset, sample_size)
        features.update(perplexity_features)
        
        # 组合文本统计特征
        text_features = self.calculate_text_statistics_features(dataset, sample_size)
        features.update(text_features)
        
        # 计算损失下降率、梯度范数和梯度一致性（使用较小的样本量以加快计算）
        probe_sample_size = min(50, sample_size)
        loss_decay_rate, avg_grad_norm, gradient_consistency = self.calculate_loss_decay_rate(
            dataset, 
            hyperparams,
            probe_steps=100,
            sample_size=probe_sample_size
        )
        features["loss_decay_rate"] = loss_decay_rate
        features["avg_grad_norm"] = avg_grad_norm
        features["gradient_consistency"] = gradient_consistency
        
        return features
