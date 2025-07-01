"""
动态模型探针模块

负责基于模型微调的动态特征分析，包括：
- 损失下降率
- 平均梯度范数
- 梯度一致性
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import torch.optim as optim
from peft import get_peft_model, LoraConfig, TaskType
import logging

from .data_parsers import DatasetAnalyzer, HyperParams

logger = logging.getLogger(__name__)

class DynamicProbeAnalyzer(DatasetAnalyzer):
    """动态探针分析器
    
    继承自DatasetAnalyzer，专门用于动态特征分析
    """
    
    def calculate_loss_decay_rate(
        self,
        dataset: List[Dict],
        hyperparams: HyperParams,
        probe_steps: int = 100,
        sample_size: int = 50
    ) -> Tuple[float, float, float]:
        """计算损失下降率、平均梯度范数和梯度一致性
        
        Returns:
            Tuple[float, float, float]: (损失下降率, 平均梯度范数, 梯度一致性)
        """
        if len(dataset) < sample_size:
            sample_size = len(dataset)
        
        sampled_data = self.sample_dataset(dataset, sample_size)
        
        # 配置 LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=hyperparams.lora_r,
            lora_alpha=hyperparams.lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        try:
            # 创建 PEFT 模型
            model = get_peft_model(self.model, peft_config)
            model.train()
            optimizer = optim.AdamW(model.parameters(), lr=hyperparams.learning_rate)
            initial_loss = 0.0
            step_grad_norms = []
            batch_gradients = []
            train_inputs = []
            for item in sampled_data:
                formatted_text = self.format_qa_pair(item)
                inputs = self.tokenizer(
                    formatted_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                train_inputs.append(inputs)
            batch_size = max(1, len(train_inputs) // 5)
            num_batches = len(train_inputs) // batch_size
            for step in tqdm(range(probe_steps), desc="执行梯度探针"):
                total_loss = 0.0
                valid_samples = 0
                optimizer.zero_grad()
                for inputs in train_inputs:
                    try:
                        outputs = model(**inputs, labels=inputs["input_ids"])
                        loss = outputs.loss
                        if not torch.isfinite(loss):
                            logger.warning(f"检测到无效的损失值: {loss.item()}")
                            continue
                        total_loss += loss.item()
                        valid_samples += 1
                    except RuntimeError as e:
                        logger.warning(f"在训练步骤中发生错误: {str(e)}")
                        continue
                if valid_samples > 0:
                    avg_loss = total_loss / valid_samples
                    loss_tensor = torch.tensor(avg_loss, requires_grad=True, device=self.device)
                    loss_tensor.backward()
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
                            for batch_idx in range(num_batches):
                                start_idx = batch_idx * batch_size
                                end_idx = start_idx + batch_size
                                batch_inputs = train_inputs[start_idx:end_idx]
                                optimizer.zero_grad()
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
                                    avg_batch_loss = batch_loss / batch_valid
                                    batch_loss_tensor = torch.tensor(avg_batch_loss, requires_grad=True, device=self.device)
                                    batch_loss_tensor.backward()
                                    grad_vector = []
                                    for param in model.parameters():
                                        if param.grad is not None:
                                            grad_vector.extend(param.grad.flatten().cpu().numpy())
                                    if grad_vector:
                                        batch_grad_vectors.append(np.array(grad_vector))
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

    def extract_all_dynamic_features(self, dataset: List[Dict], hyperparams: HyperParams, probe_steps: int = 100, sample_size: int = 50) -> Dict[str, float]:
        """一次性提取所有动态特征"""
        loss_decay_rate, avg_grad_norm, gradient_consistency = self.calculate_loss_decay_rate(
            dataset, hyperparams, probe_steps, sample_size
        )
        return {
            "loss_decay_rate": loss_decay_rate,
            "avg_grad_norm": avg_grad_norm,
            "gradient_consistency": gradient_consistency
        } 