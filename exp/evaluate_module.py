"""
评估模块 - 严格按照evaluation_example.py的实现
主要功能：
1. 数据集处理：加载评估数据集
2. 模型评估：使用传入的模型进行推理
3. 结果计算：计算EM和F1分数
4. 内存管理：评估结束后自动清理资源
"""

import os
import json
import time
import gc
import torch
import logging
import random
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load as load_metric

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('evaluation.log'),
            logging.StreamHandler()
        ]
    )

def load_evaluation_dataset(dataset_path: str) -> List[Dict]:
    """
    加载评估数据集
    
    Args:
        dataset_path: 评估数据集路径
        
    Returns:
        处理后的评估数据列表
    """
    try:
        logger.info(f"正在加载评估数据集: {dataset_path}")
        
        # 支持多种数据集格式
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif dataset_path.endswith('.jsonl'):
            data = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        else:
            # 尝试用datasets库加载
            dataset = load_dataset('json', data_files=dataset_path)
            data = dataset['train'] if 'train' in dataset else list(dataset.values())[0]
        
        # 数据清理和标准化
        processed_data = []
        for item in data:
            if isinstance(item, dict):
                # 标准化字段名
                processed_item = {}
                for key, value in item.items():
                    if isinstance(value, str):
                        processed_item[key] = value.strip()
                    else:
                        processed_item[key] = value
                processed_data.append(processed_item)
        
        logger.info(f"成功加载 {len(processed_data)} 个评估样本")
        return processed_data
        
    except Exception as e:
        logger.error(f"加载评估数据集失败: {e}")
        raise

def build_prompt(context: str, question: str) -> str:
    """
    构建评估prompt - 严格按照evaluation_example.py的格式
    
    Args:
        context: 上下文
        question: 问题
        
    Returns:
        构建的prompt
    """
    return (
        "Extract the exact answer from context. Do not explain. only exact phrase\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

def normalize_answer(s: str) -> str:
    """
    答案标准化 - 严格按照evaluation_example.py的实现
    
    Args:
        s: 原始答案字符串
        
    Returns:
        标准化后的答案
    """
    s = s.lower()
    s = re.sub(r'[\W_]+', ' ', s)
    s = ' '.join([word for word in s.split() if word not in ['a', 'an', 'the']])
    s = ' '.join(s.split())
    return s

def batch_generate(
    model, 
    tokenizer, 
    prompts: List[str], 
    batch_size: int = 16, 
    generation_kwargs: Optional[Dict] = None,
    logger=None
) -> List[str]:
    """
    批量生成答案 - 严格按照evaluation_example.py的实现
    
    Args:
        model: 模型实例
        tokenizer: 分词器实例
        prompts: prompt列表
        batch_size: 批次大小
        generation_kwargs: 生成参数
        logger: 日志对象
        
    Returns:
        生成的答案列表
    """
    logger = logger or logging.getLogger('evaluation')
    all_results = []
    model.eval()
    
    # 修复生成参数，避免probability tensor错误
    generation_kwargs = generation_kwargs or {
        'max_new_tokens': 40,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'repetition_penalty': 1.1,  # 稍微增加重复惩罚
        'use_cache': True,
        'do_sample': False,  # 使用贪婪解码，避免概率问题
        'temperature': 1.0,  # 设置温度
        'top_p': 1.0,  # 设置top_p
        'top_k': 50,  # 设置top_k
        'num_beams': 1,  # 使用单束搜索
        # 移除early_stopping，因为num_beams=1时不需要
    }
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="推理进度"):
            batch_prompts = prompts[i:i+batch_size]
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024
            ).to(model.device)
            
            try:
                outputs = model.generate(**inputs, **generation_kwargs)
                
                for j, output in enumerate(outputs):
                    input_len = inputs['input_ids'].shape[1]
                    gen_text = tokenizer.decode(output[input_len:], skip_special_tokens=True)
                    all_results.append(gen_text.strip())
                    
            except Exception as e:
                logger.error(f"生成过程中出错: {e}")
                # 如果出错，为这个批次添加空字符串
                for _ in range(len(batch_prompts)):
                    all_results.append("")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return all_results

def recommend_batch_size(
    data: List[Dict], 
    tokenizer, 
    max_new_tokens: int = 40, 
    reserved_mem_gb: int = 4,  # 增加预留内存
    logger=None
) -> int:
    """
    推荐batch_size - 严格按照evaluation_example.py的实现
    
    Args:
        data: 数据集
        tokenizer: 分词器
        max_new_tokens: 最大生成token数
        reserved_mem_gb: 预留显存GB
        logger: 日志对象
        
    Returns:
        推荐的batch_size
    """
    logger = logger or logging.getLogger('evaluation')
    sample_num = min(20, len(data))
    samples = random.sample(data, sample_num)
    
    prompts = []
    for item in samples:
        if 'context' in item and 'question' in item:
            prompts.append(build_prompt(item['context'], item['question']))
        elif 'messages' in item and len(item['messages']) >= 2:
            # 处理messages格式
            user_content = item['messages'][0]['content']
            
            # 解析user content中的context和question
            if 'Context:' in user_content and 'Question:' in user_content:
                # 提取context和question
                parts = user_content.split('Question:')
                if len(parts) == 2:
                    context_part = parts[0].replace('Context:', '').strip()
                    question_part = parts[1].strip()
                    prompts.append(build_prompt(context_part, question_part))
                else:
                    # 如果格式不标准，直接使用整个content作为context
                    prompts.append(build_prompt(user_content, ""))
            else:
                # 如果没有明确的Context:和Question:标记，直接使用整个content
                prompts.append(build_prompt(user_content, ""))
        else:
            prompts.append("")
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    token_lens = [len(ids) for ids in inputs['input_ids']]
    total_tokens = sum([l + max_new_tokens for l in token_lens])
    avg_tokens = total_tokens / sample_num
    
    # 经验公式：每token消耗（GB），7B模型约0.00005
    model_gb_factor = 0.00005
    per_sample_gb = avg_tokens * model_gb_factor
    
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        used = torch.cuda.memory_allocated(0) / 1024**3
        free = total - used - reserved_mem_gb
        
        # 更保守的batch_size计算
        batch_size = max(1, min(16, int(free // per_sample_gb)))  # 限制最大batch_size为16
        
        logger.info(f"[推荐参数] 数据集均值token数≈{avg_tokens:.1f}，单样本估算显存≈{per_sample_gb:.2f}GB")
        logger.info(f"[推荐参数] 当前GPU总显存={total:.1f}GB，已用={used:.1f}GB，预留={reserved_mem_gb}GB")
        logger.info(f"[推荐参数] 推荐batch_size≈{batch_size}（保守估计，避免内存不足）")
        
        return batch_size
    else:
        logger.info(f"[推荐参数] CPU环境，建议batch_size=1")
        return 1

def evaluate_model_on_dataset(
    model,
    tokenizer,
    eval_dataset_path: str,
    batch_size: Optional[int] = None,
    generation_kwargs: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    在指定数据集上评估模型
    
    Args:
        model: 已加载的模型实例
        tokenizer: 已加载的分词器实例
        eval_dataset_path: 评估数据集路径
        batch_size: 批次大小（可选，会自动推荐）
        generation_kwargs: 生成参数（可选）
        
    Returns:
        评估结果字典
    """
    start_time = time.time()
    
    try:
        # 1. 加载评估数据集

        
        data = load_evaluation_dataset(eval_dataset_path)
        dataset_name = os.path.basename(eval_dataset_path)
        
        # 2. 构建prompts

        prompts = []
        for item in tqdm(data, desc="构建prompts"):
            if 'context' in item and 'question' in item:
                prompt = build_prompt(item['context'], item['question'])
            elif 'messages' in item and len(item['messages']) >= 2:
                # 处理messages格式
                user_content = item['messages'][0]['content']
                
                # 解析user content中的context和question
                if 'Context:' in user_content and 'Question:' in user_content:
                    # 提取context和question
                    parts = user_content.split('Question:')
                    if len(parts) == 2:
                        context_part = parts[0].replace('Context:', '').strip()
                        question_part = parts[1].strip()
                        prompt = build_prompt(context_part, question_part)
                    else:
                        # 如果格式不标准，直接使用整个content作为context
                        prompt = build_prompt(user_content, "")
                else:
                    # 如果没有明确的Context:和Question:标记，直接使用整个content
                    prompt = build_prompt(user_content, "")
            else:
                logger.warning(f"跳过格式不符合要求的数据项")
                continue
            prompts.append(prompt)
        
        # 3. 推荐batch_size
        if batch_size is None:
            batch_size = recommend_batch_size(data, tokenizer)
        
        # 4. 批量生成答案
        logger.info(f"开始推理，数据集: {dataset_name}")
        logger.info(f"评估样本数: {len(prompts)}")
        logger.info(f"批次大小: {batch_size}")
        
        predictions = batch_generate(
            model, 
            tokenizer, 
            prompts, 
            batch_size=batch_size,
            generation_kwargs=generation_kwargs
        )
        
        # 5. 计算评估指标 - 严格按照evaluation_example.py的方式
        logger.info("正在计算评估指标...")
        
        # 准备预测和参考数据
        pred_list, ref_list = [], []
        
        for i, (item, pred) in enumerate(zip(data, predictions)):
            pred_norm = normalize_answer(pred)
            
            # 处理不同格式的参考答案
            if 'answers' in item and 'text' in item['answers']:
                # SQuAD格式
                gold_list = item['answers']['text']
            elif 'messages' in item and len(item['messages']) >= 2:
                # messages格式，取assistant的回答作为参考答案
                gold_list = [item['messages'][1]['content']]
            else:
                logger.warning(f"第{i}条数据格式不符合要求，跳过")
                continue
            
            gold_norm_list = [normalize_answer(g) for g in gold_list]
            
            # 构建预测和参考格式
            pred_list.append({
                'id': str(i), 
                'prediction_text': pred_norm
            })
            ref_list.append({
                'id': str(i), 
                'answers': {
                    'text': gold_norm_list, 
                    'answer_start': [0] * len(gold_norm_list)
                }
            })
        
        # 使用squad metric计算EM和F1
        squad_metric = load_metric("squad")
        results = squad_metric.compute(predictions=pred_list, references=ref_list)
        
        # 6. 评估摘要
        evaluation_time = time.time() - start_time
        eval_summary = {
            "dataset_name": dataset_name,
            "evaluation_time": evaluation_time,
            "eval_samples": len(predictions),
            "batch_size": batch_size,
            "metrics": {
                "exact_match": results['exact_match'],
                "f1": results['f1']
            },
            "generation_kwargs": generation_kwargs or {
                'max_new_tokens': 40,
                'pad_token_id': tokenizer.pad_token_id,
                'eos_token_id': tokenizer.eos_token_id,
                'repetition_penalty': 1.0,
                'use_cache': True
            }
        }
        
        logger.info("=" * 50)
        logger.info("评估完成！")
        logger.info(f"数据集: {dataset_name}")
        logger.info(f"评估时间: {evaluation_time:.2f}秒")
        logger.info(f"评估样本数: {len(predictions)}")
        logger.info(f"EM Score: {results['exact_match']:.4f}")
        logger.info(f"F1 Score: {results['f1']:.4f}")
        logger.info("=" * 50)
        
        return eval_summary
        
    except Exception as e:
        logger.error(f"评估过程中出错: {e}")
        raise
        
    finally:
        # 清理内存
        logger.info("正在清理评估资源...")
        
        # 清理数据集
        if 'data' in locals():
            del data
        if 'prompts' in locals():
            del prompts
        if 'predictions' in locals():
            del predictions
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("评估资源清理完成")

def evaluate_model_on_dataset_with_progress(
    model,
    tokenizer,
    eval_dataset_path: str,
    dataset_index: int = 1,
    total_datasets: int = 1,
    batch_size: Optional[int] = None,
    generation_kwargs: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    带进度显示的评估函数
    
    Args:
        model: 已加载的模型实例
        tokenizer: 已加载的分词器实例
        eval_dataset_path: 评估数据集路径
        dataset_index: 当前数据集序号
        total_datasets: 总数据集数量
        batch_size: 批次大小（可选）
        generation_kwargs: 生成参数（可选）
        
    Returns:
        评估结果字典
    """
    dataset_name = os.path.basename(eval_dataset_path)
    
    print(f"\n{'='*60}")
    print(f"正在评估: {dataset_name} ({dataset_index}/{total_datasets})")
    print(f"{'='*60}")
    
    return evaluate_model_on_dataset(
        model, 
        tokenizer, 
        eval_dataset_path, 
        batch_size, 
        generation_kwargs
    )

def load_model_and_tokenizer(model_path: str):
    """
    加载模型和分词器
    
    Args:
        model_path: 模型路径
        
    Returns:
        model, tokenizer: 模型和分词器实例
    """
    try:
        logger.info(f"正在加载模型: {model_path}")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map="auto"
        )
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'  # 修复decoder-only架构的padding问题
        
        logger.info("模型和分词器加载成功")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise

if __name__ == "__main__":
    # 测试代码
    setup_logging()
    
    # 示例用法
    model_path = "/home/haonan/data_decision/models/Qwen/Qwen2.5-7B-Instruct-1M"
    eval_dataset_path = "/home/haonan/data_decision/example/qwen_dataset/qa_dataset_20250709_1_25.json"
    
    try:
        # 加载模型和分词器（在主程序中完成）
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        # 评估
        summary = evaluate_model_on_dataset_with_progress(
            model=model,
            tokenizer=tokenizer,
            eval_dataset_path=eval_dataset_path,
            dataset_index=1,
            total_datasets=1
        )
        
        print(f"评估摘要: {json.dumps(summary, indent=2, ensure_ascii=False)}")
        
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