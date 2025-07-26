"""
基于 LoRA 的模型微调训练模块
主要功能：
1. 数据集处理：加载json格式，处理messages结构
2. 模型训练：使用 LoRA 进行模型微调
3. 训练摘要：返回详细的训练结果信息
4. 内存管理：训练结束后自动清理资源
"""

import os
import json
import time
import gc
import torch
import logging
import random
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
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

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def setup_seed(seed: int = 42):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"设置随机种子: {seed}")

def load_training_dataset(dataset_path: str) -> list:
    """
    加载训练数据集
    
    Args:
        dataset_path: 数据集路径
        
    Returns:
        处理后的数据集列表
    """
    try:
        logger.info(f"正在加载训练数据集: {dataset_path}")
        
        # 加载 JSON 数据
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 验证数据格式
        if not isinstance(data, list):
            raise ValueError("数据集应为 JSON 数组格式")
        
        logger.info(f"成功加载 {len(data)} 个训练样本")
        return data
        
    except Exception as e:
        logger.error(f"加载训练数据集失败: {e}")
        raise

def load_lora_config(lora_config_dir: str, config_name: str) -> LoraConfig:
    """
    加载LoRA配置
    
    Args:
        lora_config_dir: LoRA配置文件目录路径
        config_name: 配置文件名（可带或不带.json后缀）
        
    Returns:
        LoRA配置对象
    """
    try:
        # 处理配置文件名
        if not config_name.endswith('.json'):
            config_name += '.json'
        
        lora_config_path = os.path.join(lora_config_dir, config_name)
        logger.info(f"正在加载LoRA配置: {lora_config_path}")
        
        if os.path.exists(lora_config_path):
            # 从文件加载配置
            with open(lora_config_path, 'r') as f:
                config_dict = json.load(f)
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config_dict.get('r', 8),
                lora_alpha=config_dict.get('lora_alpha', 16),
                lora_dropout=config_dict.get('lora_dropout', 0.05),
                target_modules=config_dict.get('target_modules', ['q_proj', 'v_proj', 'k_proj', 'o_proj']),
                bias=config_dict.get('bias', 'none'),
                inference_mode=False
            )
        else:
            # 使用默认配置
            logger.warning(f"LoRA配置文件不存在，使用默认配置: {lora_config_path}")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                bias='none',
                inference_mode=False
            )
        
        logger.info("LoRA配置加载成功")
        logger.info(f"  LoRA rank (r): {lora_config.r}")
        logger.info(f"  LoRA alpha: {lora_config.lora_alpha}")
        logger.info(f"  LoRA dropout: {lora_config.lora_dropout}")
        logger.info(f"  Target modules: {lora_config.target_modules}")
        
        return lora_config
        
    except Exception as e:
        logger.error(f"加载LoRA配置失败: {e}")
        raise

def list_available_configs(lora_config_dir: str) -> list:
    """
    列出可用的LoRA配置文件
    
    Args:
        lora_config_dir: LoRA配置文件目录路径
        
    Returns:
        可用配置文件名列表
    """
    try:
        if not os.path.exists(lora_config_dir):
            logger.warning(f"配置目录不存在: {lora_config_dir}")
            return []
        
        config_files = []
        for file in os.listdir(lora_config_dir):
            if file.endswith('.json'):
                config_files.append(file)
        
        logger.info(f"在 {lora_config_dir} 中找到 {len(config_files)} 个配置文件:")
        for file in config_files:
            logger.info(f"  - {file}")
        
        return config_files
        
    except Exception as e:
        logger.error(f"列出配置文件失败: {e}")
        return []

def process_message(item, tokenizer):
    """
    处理单个消息对，返回分词结果或None
    """
    try:
        messages = item.get("messages", [])
        if len(messages) != 2:
            return None
        text = f"{messages[0]['content']}\n\nAnswer: {messages[1]['content']}"
        return tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors=None,
        )
    except Exception as e:
        logger.warning(f"处理消息时出错: {str(e)}")
        return None

def prepare_context_dataset(data: list, tokenizer) -> list:
    """
    处理 context-QA JSON 格式数据
    """
    logger.info("开始准备 context-QA JSON 格式数据集...")
    processed_data = []
    for idx, item in enumerate(tqdm(data, desc="数据预处理")):
        result = process_message(item, tokenizer)
        if result is not None:
            processed_data.append(result)
        else:
            logger.warning(f"跳过第 {idx} 条数据，格式不符合要求")
    if not processed_data:
        raise ValueError("没有成功处理任何数据条目")
    logger.info(f"context-QA JSON 数据集处理完成，共 {len(processed_data)} 个样本")
    return processed_data

def train_model_on_dataset(
    model,
    tokenizer,
    train_dataset_path: str,
    lora_config_dir: str,
    config_name: str,
    training_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    在指定数据集上训练模型
    
    Args:
        model: 已加载的模型实例
        tokenizer: 已加载的分词器实例
        train_dataset_path: 训练数据集路径
        lora_config_dir: LoRA配置文件目录路径
        config_name: 配置文件名（可带或不带.json后缀）
        training_args: 训练参数（可选）
        
    Returns:
        训练摘要字典
    """
    start_time = time.time()
    
    try:
        # 1. 设置随机种子
        setup_seed(42)
        
        # 2. 加载训练数据集
        logger.info("=" * 50)
        logger.info("开始训练流程")
        logger.info("=" * 50)
        
        data = load_training_dataset(train_dataset_path)
        dataset_name = os.path.basename(train_dataset_path)
        
        # 3. 加载LoRA配置并应用到模型
        lora_config = load_lora_config(lora_config_dir, config_name)
        
        # 确保模型已准备好训练
        if not hasattr(model, 'is_loaded_in_8bit') or not model.is_loaded_in_8bit:
            logger.info("准备模型进行8bit训练...")
            model = prepare_model_for_kbit_training(model)
        
        # 应用LoRA配置
        logger.info("正在应用LoRA配置...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # 4. 数据预处理
        processed_dataset = prepare_context_dataset(data, tokenizer)
        
        # 5. 设置训练参数
        if training_args is None:
            # 训练超参数
            num_train_epochs = 3
            batch_size = 8
            gradient_accumulation_steps = 8
            learning_rate = 1e-4
            max_grad_norm = 0.3
            warmup_ratio = 0.03
            
            # 优化器参数
            optimizer_kwargs = {
                "betas": (0.9, 0.999),
                "weight_decay": 0.01
            }
            
            training_args = TrainingArguments(
                output_dir="./temp_training_output",  # 临时目录
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                max_grad_norm=max_grad_norm,
                warmup_ratio=warmup_ratio,
                logging_steps=10,
                save_steps=0,  # 不保存检查点
                save_total_limit=0,  # 不保存任何检查点
                fp16=True,
                report_to="none",
                optim="adamw_torch",
                adam_beta1=optimizer_kwargs["betas"][0],
                adam_beta2=optimizer_kwargs["betas"][1],
                weight_decay=optimizer_kwargs["weight_decay"],
                lr_scheduler_type="cosine",
                lr_scheduler_kwargs={"num_cycles": 0.5}
            )
        
        # 6. 记录训练参数
        logger.info("训练超参数:")
        logger.info(f"  训练轮数: {training_args.num_train_epochs}")
        logger.info(f"  批次大小: {training_args.per_device_train_batch_size}")
        logger.info(f"  梯度累积步数: {training_args.gradient_accumulation_steps}")
        logger.info(f"  学习率: {training_args.learning_rate}")
        logger.info(f"  最大梯度范数: {training_args.max_grad_norm}")
        logger.info(f"  预热比例: {training_args.warmup_ratio}")
        
        # 7. 创建训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        )
        
        # 8. 开始训练
        logger.info(f"开始训练数据集: {dataset_name}")
        logger.info(f"训练样本数: {len(processed_dataset)}")
        
        # 记录初始loss
        initial_loss = None
        try:
            # 计算初始loss
            eval_results = trainer.evaluate()
            initial_loss = eval_results.get('eval_loss', 0.0)
            logger.info(f"初始loss: {initial_loss:.4f}")
        except:
            logger.warning("无法计算初始loss")
        
        # 开始训练
        train_result = trainer.train()
        
        # 记录最终loss
        final_loss = train_result.metrics.get('train_loss', 0.0)
        
        # 9. 训练摘要
        training_time = time.time() - start_time
        train_summary = {
            "dataset_name": dataset_name,
            "training_time": training_time,
            "train_samples": len(processed_dataset),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "training_metrics": {
                "train_runtime": train_result.metrics.get('train_runtime', 0),
                "train_samples_per_second": train_result.metrics.get('train_samples_per_second', 0),
                "train_steps_per_second": train_result.metrics.get('train_steps_per_second', 0),
                "train_loss": final_loss
            },
            "training_args": {
                "num_train_epochs": training_args.num_train_epochs,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "learning_rate": training_args.learning_rate,
                "max_grad_norm": training_args.max_grad_norm,
                "warmup_ratio": training_args.warmup_ratio
            },
            "lora_config": {
                "config_name": config_name,
                "config_path": os.path.join(lora_config_dir, config_name + ('' if config_name.endswith('.json') else '.json')),
                "r": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "lora_dropout": lora_config.lora_dropout,
                "target_modules": lora_config.target_modules
            }
        }
        
        logger.info("=" * 50)
        logger.info("训练完成！")
        logger.info(f"数据集: {dataset_name}")
        logger.info(f"LoRA配置: {config_name}")
        logger.info(f"训练时间: {training_time:.2f}秒")
        logger.info(f"训练样本数: {len(processed_dataset)}")
        logger.info(f"初始loss: {initial_loss:.4f}" if initial_loss else "未知")
        logger.info(f"最终loss: {final_loss:.4f}")
        logger.info(f"训练时长: {train_result.metrics.get('train_runtime', 0):.2f} 秒")
        logger.info(f"每秒训练样本数: {train_result.metrics.get('train_samples_per_second', 0):.2f}")
        logger.info(f"每秒训练步数: {train_result.metrics.get('train_steps_per_second', 0):.2f}")
        logger.info("=" * 50)
        
        return train_summary
        
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        raise
        
    finally:
        # 清理内存
        logger.info("正在清理训练资源...")
        
        # 清理数据集
        if 'data' in locals():
            del data
        if 'processed_dataset' in locals():
            del processed_dataset
        
        # 清理训练器
        if 'trainer' in locals():
            del trainer
        
        # 清理LoRA配置
        if 'lora_config' in locals():
            del lora_config
        
        # 清理训练参数
        if 'training_args' in locals():
            del training_args
        
        # 清理模型中的LoRA权重（重要！）
        if 'model' in locals() and hasattr(model, 'peft_config'):
            try:
                # 移除LoRA配置，恢复原始模型状态
                from peft import PeftModel
                if isinstance(model, PeftModel):
                    logger.info("正在移除LoRA配置...")
                    # 这里不合并权重，只是移除LoRA配置
                    model = model.get_base_model()
            except Exception as e:
                logger.warning(f"移除LoRA配置时出错: {e}")
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"清理后GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        logger.info("训练资源清理完成")

def train_model_on_dataset_with_progress(
    model,
    tokenizer,
    train_dataset_path: str,
    lora_config_dir: str,
    config_name: str,
    dataset_index: int = 1,
    total_datasets: int = 1,
    training_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    带进度显示的训练函数
    
    Args:
        model: 已加载的模型实例
        tokenizer: 已加载的分词器实例
        train_dataset_path: 训练数据集路径
        lora_config_dir: LoRA配置文件目录路径
        config_name: 配置文件名（可带或不带.json后缀）
        dataset_index: 当前数据集序号
        total_datasets: 总数据集数量
        training_args: 训练参数（可选）
        
    Returns:
        训练摘要字典
    """
    dataset_name = os.path.basename(train_dataset_path)
    
    print(f"\n{'='*60}")
    print(f"正在训练: {dataset_name} ({dataset_index}/{total_datasets})")
    print(f"使用LoRA配置: {config_name}")
    print(f"{'='*60}")
    
    return train_model_on_dataset(model, tokenizer, train_dataset_path, lora_config_dir, config_name, training_args)

if __name__ == "__main__":
    # 测试代码
    setup_logging()
    
    # 示例用法
    model_path = "/home/haonan/data_decision/models/Qwen/Qwen2.5-7B-Instruct-1M"
    train_dataset_path = "/home/haonan/data_decision/example/qwen_dataset/qa_dataset_20250709_1_25.json"
    lora_config_dir = "/path/to/lora_configs" # 假设配置文件放在这里
    config_name = "qwen_lora_config" # 假设配置文件名为 qwen_lora_config.json
    
    try:
        # 加载模型和分词器（在主程序中完成）
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map="auto"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # 训练
        summary = train_model_on_dataset_with_progress(
            model=model,
            tokenizer=tokenizer,
            train_dataset_path=train_dataset_path,
            lora_config_dir=lora_config_dir,
            config_name=config_name,
            dataset_index=1,
            total_datasets=1
        )
        
        print(f"训练摘要: {json.dumps(summary, indent=2, ensure_ascii=False)}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
    finally:
        # 清理模型
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 