"""
总包文件 - 整合训练和评估模块
主要功能：
1. 数据集循环：对每个数据集进行训练和评估
2. 模型复用：确保同一模型实例用于训练和评估
3. 内存管理：循环结束后彻底释放所有资源
4. 进度显示：实时显示当前循环进度和状态
5. 结果汇总：将所有关键信息写入JSON文件
"""

import os
import json
import time
import gc
import torch
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# 导入训练和评估模块
from train_module import train_model_on_dataset_with_progress
from evaluate_module import evaluate_model_on_dataset_with_progress

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
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
        
        logger.info("模型和分词器加载成功")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise

def run_training_evaluation_pipeline(
    dataset_paths: List[str],
    model_path: str,
    lora_config_dir: str,
    config_name: str,
    result_output_path: str,
    training_args: Optional[Dict[str, Any]] = None,
    evaluation_batch_size: Optional[int] = None,
    evaluation_generation_kwargs: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    运行训练-评估流水线
    
    Args:
        dataset_paths: 数据集路径列表
        model_path: 模型路径
        lora_config_dir: LoRA配置文件目录路径
        config_name: 配置文件名
        result_output_path: 结果输出文件路径
        training_args: 训练参数（可选）
        evaluation_batch_size: 评估批次大小（可选）
        evaluation_generation_kwargs: 评估生成参数（可选）
        
    Returns:
        流水线执行摘要
    """
    start_time = time.time()
    pipeline_results = []
    
    try:
        # 1. 设置日志
        setup_logging()
        logger.info("=" * 80)
        logger.info("开始训练-评估流水线")
        logger.info("=" * 80)
        logger.info(f"模型路径: {model_path}")
        logger.info(f"LoRA配置目录: {lora_config_dir}")
        logger.info(f"LoRA配置名称: {config_name}")
        logger.info(f"数据集数量: {len(dataset_paths)}")
        logger.info(f"结果输出路径: {result_output_path}")
        
        # 2. 开始数据集循环
        total_datasets = len(dataset_paths)
        
        for dataset_index, dataset_path in enumerate(dataset_paths, 1):
            dataset_name = os.path.basename(dataset_path)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"处理数据集 {dataset_index}/{total_datasets}: {dataset_name}")
            logger.info(f"{'='*80}")
            
            try:
                # 每轮都重新加载模型和分词器（确保使用原始模型）
                logger.info("正在加载模型和分词器...")
                model, tokenizer = load_model_and_tokenizer(model_path)
                
                # 3.1 训练阶段
                logger.info(f"\n--- 开始训练阶段 ---")
                train_summary = train_model_on_dataset_with_progress(
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset_path=dataset_path,
                    lora_config_dir=lora_config_dir,
                    config_name=config_name,
                    dataset_index=dataset_index,
                    total_datasets=total_datasets,
                    training_args=training_args
                )
                
                # 3.2 评估阶段
                logger.info(f"\n--- 开始评估阶段 ---")
                eval_summary = evaluate_model_on_dataset_with_progress(
                    model=model,
                    tokenizer=tokenizer,
                    eval_dataset_path=dataset_path,
                    dataset_index=dataset_index,
                    total_datasets=total_datasets,
                    batch_size=evaluation_batch_size,
                    generation_kwargs=evaluation_generation_kwargs
                )
                
                # 3.3 合并结果
                dataset_result = {
                    "dataset_index": dataset_index,
                    "dataset_name": dataset_name,
                    "dataset_path": dataset_path,
                    "training": train_summary,
                    "evaluation": eval_summary,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                pipeline_results.append(dataset_result)
                
                # 3.4 显示当前数据集结果摘要
                logger.info(f"\n--- 数据集 {dataset_index}/{total_datasets} 完成 ---")
                logger.info(f"训练时间: {train_summary['training_time']:.2f}秒")
                logger.info(f"训练样本数: {train_summary['train_samples']}")
                logger.info(f"初始loss: {train_summary.get('initial_loss', 'N/A')}")
                logger.info(f"最终loss: {train_summary['final_loss']:.4f}")
                logger.info(f"评估时间: {eval_summary['evaluation_time']:.2f}秒")
                logger.info(f"评估样本数: {eval_summary['eval_samples']}")
                logger.info(f"EM Score: {eval_summary['metrics']['exact_match']:.4f}")
                logger.info(f"F1 Score: {eval_summary['metrics']['f1']:.4f}")
                
            except Exception as e:
                logger.error(f"处理数据集 {dataset_name} 时出错: {e}")
                # 记录错误但继续处理下一个数据集
                error_result = {
                    "dataset_index": dataset_index,
                    "dataset_name": dataset_name,
                    "dataset_path": dataset_path,
                    "error": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                pipeline_results.append(error_result)
                continue
            
            # 3.5 清理当前轮次的所有资源（包括模型）
            logger.info("清理当前轮次资源...")
            
            # 更彻底的内存清理
            if torch.cuda.is_available():
                logger.info(f"清理前GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            # 清理模型和分词器
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            
            # 强制垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"清理后GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 4. 生成最终结果摘要
        pipeline_time = time.time() - start_time
        successful_datasets = [r for r in pipeline_results if 'error' not in r]
        failed_datasets = [r for r in pipeline_results if 'error' in r]
        
        final_summary = {
            "pipeline_info": {
                "model_path": model_path,
                "lora_config_dir": lora_config_dir,
                "config_name": config_name,
                "total_datasets": total_datasets,
                "successful_datasets": len(successful_datasets),
                "failed_datasets": len(failed_datasets),
                "pipeline_time": pipeline_time,
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "dataset_results": pipeline_results,
            "summary_statistics": {
                "avg_training_time": sum(r['training']['training_time'] for r in successful_datasets) / len(successful_datasets) if successful_datasets else 0,
                "avg_evaluation_time": sum(r['evaluation']['evaluation_time'] for r in successful_datasets) / len(successful_datasets) if successful_datasets else 0,
                "avg_em_score": sum(r['evaluation']['metrics']['exact_match'] for r in successful_datasets) / len(successful_datasets) if successful_datasets else 0,
                "avg_f1_score": sum(r['evaluation']['metrics']['f1'] for r in successful_datasets) / len(successful_datasets) if successful_datasets else 0,
                "best_em_score": max(r['evaluation']['metrics']['exact_match'] for r in successful_datasets) if successful_datasets else 0,
                "best_f1_score": max(r['evaluation']['metrics']['f1'] for r in successful_datasets) if successful_datasets else 0
            }
        }
        
        # 5. 保存结果到JSON文件
        logger.info(f"\n正在保存结果到: {result_output_path}")
        os.makedirs(os.path.dirname(result_output_path), exist_ok=True)
        
        with open(result_output_path, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        
        # 6. 显示最终摘要
        logger.info("=" * 80)
        logger.info("流水线执行完成！")
        logger.info("=" * 80)
        logger.info(f"总执行时间: {pipeline_time:.2f}秒")
        logger.info(f"成功处理数据集: {len(successful_datasets)}/{total_datasets}")
        logger.info(f"失败数据集: {len(failed_datasets)}")
        if successful_datasets:
            logger.info(f"平均EM分数: {final_summary['summary_statistics']['avg_em_score']:.4f}")
            logger.info(f"平均F1分数: {final_summary['summary_statistics']['avg_f1_score']:.4f}")
            logger.info(f"最佳EM分数: {final_summary['summary_statistics']['best_em_score']:.4f}")
            logger.info(f"最佳F1分数: {final_summary['summary_statistics']['best_f1_score']:.4f}")
        logger.info(f"结果已保存到: {result_output_path}")
        logger.info("=" * 80)
        
        return final_summary
        
    except Exception as e:
        logger.error(f"流水线执行过程中出错: {e}")
        raise
        
    finally:
        # 7. 彻底清理所有资源
        logger.info("正在清理所有资源...")
        
        # 清理结果列表
        if 'pipeline_results' in locals():
            del pipeline_results
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("所有资源清理完成")

def run_pipeline_with_progress(
    dataset_paths: List[str],
    model_path: str,
    lora_config_dir: str,
    config_name: str,
    result_output_path: str,
    training_args: Optional[Dict[str, Any]] = None,
    evaluation_batch_size: Optional[int] = None,
    evaluation_generation_kwargs: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    带进度显示的流水线执行函数
    
    Args:
        dataset_paths: 数据集路径列表
        model_path: 模型路径
        lora_config_dir: LoRA配置文件目录路径
        config_name: 配置文件名
        result_output_path: 结果输出文件路径
        training_args: 训练参数（可选）
        evaluation_batch_size: 评估批次大小（可选）
        evaluation_generation_kwargs: 评估生成参数（可选）
        
    Returns:
        流水线执行摘要
    """
    print(f"\n{'='*80}")
    print(f"🚀 开始训练-评估流水线")
    print(f"{'='*80}")
    print(f"📁 模型路径: {model_path}")
    print(f"⚙️  LoRA配置: {config_name}")
    print(f"📊 数据集数量: {len(dataset_paths)}")
    print(f"💾 结果输出: {result_output_path}")
    print(f"{'='*80}\n")
    
    return run_training_evaluation_pipeline(
        dataset_paths=dataset_paths,
        model_path=model_path,
        lora_config_dir=lora_config_dir,
        config_name=config_name,
        result_output_path=result_output_path,
        training_args=training_args,
        evaluation_batch_size=evaluation_batch_size,
        evaluation_generation_kwargs=evaluation_generation_kwargs
    )

if __name__ == "__main__":
    # 测试代码
    setup_logging()
    
    # 示例用法
    dataset_paths = [
        "/home/haonan/data_decision/example/qwen_dataset/qa_dataset_20250709_1_25.json",
        # 可以添加更多数据集路径
    ]
    
    model_path = "/home/haonan/data_decision/models/Qwen/Qwen2.5-7B-Instruct-1M"
    lora_config_dir = "/path/to/lora_configs"  # 需要设置实际的LoRA配置目录
    config_name = "qwen_lora_config"
    result_output_path = "/home/haonan/data_decision/results/pipeline_results.json"
    
    try:
        # 运行流水线
        summary = run_pipeline_with_progress(
            dataset_paths=dataset_paths,
            model_path=model_path,
            lora_config_dir=lora_config_dir,
            config_name=config_name,
            result_output_path=result_output_path
        )
        
        print(f"\n✅ 流水线执行完成！")
        print(f"📊 结果摘要: {json.dumps(summary['pipeline_info'], indent=2, ensure_ascii=False)}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"❌ 流水线执行失败: {e}") 