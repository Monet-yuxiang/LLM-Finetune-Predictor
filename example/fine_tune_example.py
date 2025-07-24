"""
基于 LoRA 的模型微调脚本
主要功能：
1. 数据集处理：加载json格式
2. 模型训练：使用 LoRA 进行模型微调
3. 模型保存：保存训练后的模型到本地
notice:
1. 超参数硬编码，需要根据实际情况调整
2. 数据集路径需要根据实际情况调整
"""

import os
import json
import ast
import logging
import random
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
import torch
import numpy as np


class FineTuner:
    def __init__(self):
        # 设置日志
        self.setup_logging()
        
        # 加载环境变量
        load_dotenv()
        self.model_name = os.getenv('MODEL_NAME')

        # 设置随机种子
        self.seed = 42
        self.setup_seed(self.seed)
        self.logger.info(f"设置随机种子: {self.seed}")

        
        # 创建输出目录
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("fine_tune_outputs", self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化 dataset_path
        self.dataset_path = None
        
        # 验证环境变量
        if not self.model_name:
            raise ValueError("请在 .env 文件中设置 MODEL_NAME")

    def setup_logging(self):
        """设置日志记录"""
        self.logger = logging.getLogger('fine_tune')
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        fh = logging.FileHandler('fine_tune.log')
        fh.setLevel(logging.INFO)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def setup_seed(self, seed):
        """设置所有随机种子以确保可重复性"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.logger.info(f"seed {seed}")
        # 某些操作可能会变慢，但为了复现性这是必要的
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
            
    def prepare_context_dataset(self, data_path: str):
        """处理 context-QA JSON 格式数据"""
        self.logger.info("开始准备 context-QA JSON 格式数据集...")
        
        try:
            # 保存数据集路径（保持与原函数一致）
            self.dataset_path = data_path
    
            # 加载 JSON 数据
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证数据格式
            if not isinstance(data, list):
                raise ValueError("数据集应为 JSON 数组格式")
    
            # 初始化分词器
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
    
            def process_message(item):
                """处理单个消息对"""
                try:
                    messages = item.get("messages", [])
                    if len(messages) != 2:
                        return None
                        
                    # 直接使用messages中的内容
                    text = f"{messages[0]['content']}\n\nAnswer: {messages[1]['content']}"
                    
                    # 分词处理
                    return tokenizer(
                        text,
                        truncation=True,
                        max_length=512,
                        padding="max_length",
                        return_tensors=None,
                    )
                except Exception as e:
                    self.logger.warning(f"处理消息时出错: {str(e)}")
                    return None
    
            # 处理所有数据条目
            processed_data = []
            for idx, item in enumerate(data, 1):
                result = process_message(item)
                if result is not None:
                    processed_data.append(result)
                else:
                    self.logger.warning(f"跳过第 {idx} 条数据，格式不符合要求")
            
            if not processed_data:
                raise ValueError("没有成功处理任何数据条目")
            
            self.logger.info(f"context-QA JSON 数据集处理完成，共 {len(processed_data)} 个样本")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"准备 context-QA JSON 数据集时发生错误: {str(e)}")
            raise


    def prepare_model(self):
        """准备模型"""
        self.logger.info(f"开始加载模型: {self.model_name}")
        
        try:
            # 加载分词器
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
            
            # 记录 LoRA 配置
            lora_r = 8
            lora_alpha = 16
            lora_dropout = 0.05
            #h1 的配置是0.05，h2的配置是0.1
            # lora_dropout = 0.1
            target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']  # 通用的Transformer注意力模块名称
            
            self.logger.info("LoRA 配置:")
            self.logger.info(f"  Dataset: {os.path.basename(self.dataset_path)}")
            self.logger.info(f"  LoRA rank (r): {lora_r}")
            self.logger.info(f"  LoRA alpha: {lora_alpha}")
            self.logger.info(f"  LoRA dropout: {lora_dropout}")
            self.logger.info(f"  Target modules: {target_modules}")
            
            # LoRA 配置
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=target_modules
            )
            
            # 准备模型
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"准备模型时发生错误: {str(e)}")
            raise

    def train(self, dataset, model, tokenizer):
        """训练模型"""
        self.logger.info("开始训练...")
        
        try:
            # 训练超参数h1
            num_train_epochs = 3
            batch_size = 2
            gradient_accumulation_steps = 8
            learning_rate = 1e-4
            max_grad_norm = 0.3
            warmup_ratio = 0.03
            

            # 训练超参数h2
            # num_train_epochs = 3
            # batch_size = 8
            # gradient_accumulation_steps = 2
            # learning_rate = 3e-5
            # max_grad_norm = 0.3
            # warmup_ratio = 0.05

            
            # 优化器参数
            optimizer_kwargs = {
                "betas": (0.9, 0.999),    # β=(0.9, 0.999)
                "weight_decay": 0.01       # weight_decay=0.01
            }
            
            # 记录训练超参数
            self.logger.info("训练超参数:")
            self.logger.info(f"  训练轮数: {num_train_epochs}")
            self.logger.info(f"  批次大小: {batch_size}")
            self.logger.info(f"  梯度累积步数: {gradient_accumulation_steps}")
            self.logger.info(f"  学习率: {learning_rate}")
            self.logger.info(f"  最大梯度范数: {max_grad_norm}")
            self.logger.info(f"  预热比例: {warmup_ratio}")
            self.logger.info("优化器参数:")
            self.logger.info(f"  Beta: {optimizer_kwargs['betas']}")
            self.logger.info(f"  Weight Decay: {optimizer_kwargs['weight_decay']}")
           
            
            # 训练参数
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                max_grad_norm=max_grad_norm,
                warmup_ratio=warmup_ratio,
                logging_steps=10,
                save_steps=100,
                save_total_limit=2,
                fp16=True,
                report_to="none",

                
                # AdamW优化器参数-
                optim="adamw_torch",
                adam_beta1=optimizer_kwargs["betas"][0],
                adam_beta2=optimizer_kwargs["betas"][1],
                weight_decay=optimizer_kwargs["weight_decay"],
                 # 添加cosine退火学习率调度器
                lr_scheduler_type="cosine",
                lr_scheduler_kwargs={"num_cycles": 0.5}  # 半个周期的cosine退火
            )
            
            # 创建训练器
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            )
            
            # 开始训练
            self.logger.info("开始训练过程...")
            train_result = trainer.train()
            
            # 记录训练结果
            self.logger.info("训练完成，训练指标:")
            self.logger.info(f"  训练时长: {train_result.metrics['train_runtime']:.2f} 秒")
            self.logger.info(f"  每秒训练样本数: {train_result.metrics.get('train_samples_per_second', 0):.2f}")
            self.logger.info(f"  每秒训练步数: {train_result.metrics.get('train_steps_per_second', 0):.2f}")
            self.logger.info(f"  训练损失: {train_result.metrics.get('train_loss', 0):.4f}")
            
            # 保存模型
            trainer.save_model()
            self.logger.info(f"模型已保存到: {self.output_dir}")
            
            return trainer
            
        except Exception as e:
            self.logger.error(f"训练过程中发生错误: {str(e)}")
            raise

def main():
    # 创建基础日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('fine_tune_main')
    logger.info("开始新的训练任务...")
    
    
    try:
        # 初始化
        fine_tuner = FineTuner()
        
        # 准备数据集
        data_path='input_data_path'

        logger.info(f"使用数据集:{data_path}")
        
        dataset = fine_tuner.prepare_context_dataset(data_path)
        
        # 准备模型
        model, tokenizer = fine_tuner.prepare_model()
        
        # 训练模型
        trainer = fine_tuner.train(dataset, model, tokenizer)
        
        logger.info("训练流程完成")
        
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()