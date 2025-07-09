"""
动态模型探针模块

负责基于模型微调的动态特征分析，包括：
- 损失下降率
- 平均梯度范数
- 梯度一致性

支持最终格式数据：context_text + qa_pairs
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import torch.optim as optim
from peft import get_peft_model, LoraConfig, TaskType
import logging

from .data_parsers import HyperParams

logger = logging.getLogger(__name__)

class DynamicProbeAnalyzer:
    """动态探针分析器（不再继承DatasetAnalyzer）
    
    专门用于动态特征分析，支持最终格式数据
    """
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def sample_dataset(self, dataset: List[Dict], sample_size: int) -> List[Dict]:
        """从数据集中采样"""
        if len(dataset) < sample_size:
            sample_size = len(dataset)
        return list(np.random.choice(dataset, sample_size, replace=False))

    def is_final_format(self, dataset: List[Dict]) -> bool:
        """判断是否为最终格式数据"""
        if not dataset:
            return False
        sample = dataset[0]
        return "context_text" in sample and "qa_pairs" in sample
    
    def extract_qa_from_final_format(self, item: Dict) -> List[Tuple[str, str]]:
        """从最终格式数据中提取所有问答对"""
        context = item.get("context_text", "")
        qa_pairs = item.get("qa_pairs", [])
        result = []
        for qa in qa_pairs:
            question = qa.get("question", "")
            answer = qa.get("output", "")
            if question and answer:
                full_question = f"{context}\n\n{question}" if context else question
                result.append((full_question, answer))
        return result

    def format_qa_pair(self, question: str, answer: str) -> str:
        """将问答对格式化为模型输入格式"""
        return f"问：{question}\n答：{answer}"

    def calculate_loss_decay_rate(
        self,
        dataset: List[Dict],
        hyperparams: HyperParams,
        probe_steps: int = 100,
        sample_size: int = 50,
        batch_size: int = 8
    ) -> Tuple[float, float, float]:
        """计算损失下降率、平均梯度范数和梯度一致性（支持最终格式，批量）"""
        if len(dataset) < sample_size:
            sample_size = len(dataset)
        sampled_data = self.sample_dataset(dataset, sample_size)
        is_final_format = self.is_final_format(dataset)
        logger.info(f"检测到数据格式: {'最终格式' if is_final_format else '标准格式'}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=hyperparams.lora_r,
            lora_alpha=hyperparams.lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        try:
            model = get_peft_model(self.model, peft_config)
            model.train()
            optimizer = optim.AdamW(model.parameters(), lr=hyperparams.learning_rate)
            initial_loss = 0.0
            step_grad_norms = []
            batch_gradients = []
            qa_list = []
            for item in sampled_data:
                if is_final_format:
                    qa_list.extend(self.extract_qa_from_final_format(item))
                else:
                    qa_list.append(item)
            # 组装训练输入
            train_batches = []
            for i in range(0, len(qa_list), batch_size):
                batch_qa = qa_list[i:i+batch_size]
                if is_final_format:
                    batch_formatted = [self.format_qa_pair(q, a) for q, a in batch_qa]
                else:
                    batch_formatted = [self.format_qa_pair(item) for item in batch_qa]
                inputs = self.tokenizer(
                    batch_formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                train_batches.append(inputs)
            num_batches = len(train_batches)
            for step in tqdm(range(probe_steps), desc="执行梯度探针(batch)"):
                total_loss = 0.0
                valid_samples = 0
                optimizer.zero_grad()
                for batch_inputs in train_batches:
                    try:
                        outputs = model(**batch_inputs, labels=batch_inputs["input_ids"])
                        loss = outputs.loss
                        if not torch.isfinite(loss):
                            logger.warning(f"检测到无效的损失值: {loss.item()}")
                            continue
                        total_loss += loss.item() * batch_inputs["input_ids"].size(0)
                        valid_samples += batch_inputs["input_ids"].size(0)
                        loss.backward()
                    except RuntimeError as e:
                        logger.warning(f"在训练步骤中发生错误: {str(e)}")
                        continue
                if valid_samples > 0:
                    avg_loss = total_loss / valid_samples
                    step_grad_norm = 0.0
                    for param in model.parameters():
                        if param.grad is not None:
                            step_grad_norm += param.grad.norm(2).item() ** 2
                    step_grad_norm = step_grad_norm ** 0.5
                    if torch.isfinite(torch.tensor(step_grad_norm)):
                        step_grad_norms.append(step_grad_norm)
                        if step == 0:
                            initial_loss = avg_loss
                        if step % 10 == 0 and num_batches >= 2:
                            batch_grad_vectors = []
                            for batch_inputs in train_batches:
                                optimizer.zero_grad()
                                try:
                                    outputs = model(**batch_inputs, labels=batch_inputs["input_ids"])
                                    loss = outputs.loss
                                    if torch.isfinite(loss):
                                        loss.backward()
                                        grad_vector = []
                                        for param in model.parameters():
                                            if param.grad is not None:
                                                grad_vector.extend(param.grad.flatten().cpu().numpy())
                                        if grad_vector:
                                            batch_grad_vectors.append(np.array(grad_vector))
                                except:
                                    continue
                            if len(batch_grad_vectors) >= 2:
                                batch_similarities = []
                                for i in range(len(batch_grad_vectors)):
                                    for j in range(i + 1, len(batch_grad_vectors)):
                                        min_len = min(len(batch_grad_vectors[i]), len(batch_grad_vectors[j]))
                                        if min_len > 0:
                                            vec1 = batch_grad_vectors[i][:min_len]
                                            vec2 = batch_grad_vectors[j][:min_len]
                                            dot_product = np.dot(vec1, vec2)
                                            norm1 = np.linalg.norm(vec1)
                                            norm2 = np.linalg.norm(vec2)
                                            if norm1 > 0 and norm2 > 0:
                                                similarity = dot_product / (norm1 * norm2)
                                                batch_similarities.append(similarity)
                                if batch_similarities:
                                    batch_gradients.append(np.mean(batch_similarities))
                        optimizer.step()
                    else:
                        logger.warning(f"步骤 {step} 检测到无效的梯度范数: {step_grad_norm}")
                        step_grad_norms.append(0.0)
                else:
                    step_grad_norms.append(0.0)
            final_loss = avg_loss if 'avg_loss' in locals() else initial_loss
            loss_decay_rate = (initial_loss - final_loss) / initial_loss if initial_loss != 0 else 0.0
            avg_grad_norm = np.mean(step_grad_norms) if step_grad_norms else 0.0
            gradient_consistency = np.mean(batch_gradients) if batch_gradients else 0.0
        except Exception as e:
            logger.error(f"在计算损失下降率时发生错误: {str(e)}")
            loss_decay_rate = 0.0
            avg_grad_norm = 0.0
            gradient_consistency = 0.0
        finally:
            if 'model' in locals():
                model.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return loss_decay_rate, avg_grad_norm, gradient_consistency 

    def extract_all_dynamic_features(self, dataset: List[Dict], hyperparams: HyperParams, probe_steps: int = 100, sample_size: int = 50, batch_size: int = 8) -> Dict[str, float]:
        """一次性提取所有动态特征（支持最终格式，批量）"""
        loss_decay_rate, avg_grad_norm, gradient_consistency = self.calculate_loss_decay_rate(
            dataset, hyperparams, probe_steps, sample_size, batch_size
        )
        return {
            "loss_decay_rate": loss_decay_rate,
            "avg_grad_norm": avg_grad_norm,
            "gradient_consistency": gradient_consistency
        } 