"""
加载微调后的模型 
加载测试数据集，取其中的validation数据集

评估指标：
- Exact Match (EM): 预测答案与标准答案完全匹配的比例
- F1 Score: 预测答案和标准答案的词重叠率
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
from datasets import load_dataset
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import torch
from tqdm import tqdm
import evaluate
from dotenv import load_dotenv

class ModelTester:
    def __init__(self, adapter_path: str, test_dataset: str = "squad"):
        # 设置日志
        self.setup_logging()
        
        load_dotenv()
        self.model_name = os.getenv('MODEL_NAME')
        self.adapter_path = adapter_path
        self.test_dataset = test_dataset
        
        # 创建输出目录
        self.output_dir = "test_outputs"
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # 加载 SQuAD 评估指标
        try:
            self.squad_metric = evaluate.load("squad")
        except Exception as e:
            self.logger.error(f"加载 squad 指标时发生错误: {str(e)}")
            raise

        # 设置设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"使用设备: {self.device}")

        # 设置阈值
        self.retrieval_threshold = 0.5  

    def setup_logging(self):
        """设置日志记录"""
        self.logger = logging.getLogger('model_test')
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        fh = logging.FileHandler('model_test.log')
        fh.setLevel(logging.INFO)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def load_model(self):
        """加载并合并基础模型和 LoRA 权重"""
        self.logger.info("加载模型和分词器...")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        # 设置padding_side='left'，这对decoder-only模型很重要
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map="auto"
        )
        
        # 加载 LoRA 配置
        peft_config = PeftConfig.from_pretrained(self.adapter_path)
        
        # 加载 LoRA 模型
        model = PeftModel.from_pretrained(base_model, self.adapter_path)
        
        # 合并模型权重
        self.logger.info("正在合并 LoRA 权重...")
        try:
            model = model.merge_and_unload()
            self.logger.info("LoRA 权重合并成功")
        except Exception as e:
            self.logger.warning(f"LoRA 权重合并失败: {str(e)}，将使用未合并的模型继续")
        
        model.eval()
        return model, tokenizer

    def preprocess_validation_data(self, examples, tokenizer):
        """预处理验证集数据"""
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )
        return inputs

    def evaluate_model(self, model, tokenizer, validation_dataset):
        """使用批处理方式评估模型在SQuAD数据集上的性能"""
        self.logger.info(f"开始评估模型，总样本数: {len(validation_dataset)}")
        
        predictions = []
        batch_size = 16  # 可以根据GPU显存调整
        
        # Qwen模型特定的生成配置
        generation_kwargs = {
            'max_new_tokens': 40,     # 最大新生成的token数
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'repetition_penalty': 1.0, # 重复惩罚系数
            'use_cache': True         # 使用KV缓存加速生成
        }
        
        try:
            # 分批处理数据
            for i in tqdm(range(0, len(validation_dataset), batch_size), desc="评估进度"):
               
                batch = validation_dataset[i:i + batch_size]
                
                # 构建输入
                contexts = batch['context']
                questions = batch['question']
                ids = batch['id']
                
                # 构建提示模板
                prompts = []
                for ctx, q in zip(contexts, questions):
                    # prompt = (
                    #     "Answer the question based on the given context.\n\n"
                    #     f"Context: {ctx}\n\n"
                    #     f"Question: {q}\n\n"
                    #     "Answer:"
                    # )
                    prompt = (
                        "Extract the exact answer from context. Do not explain. only exact phrase\n\n"
                        f"Context: {ctx}\n\n"
                        f"Question: {q}\n\n"
                        "Answer:"
                    )

                    prompts.append(prompt)
                
                # 编码输入
                inputs = tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt"
                ).to(model.device)
                
                # 生成答案
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        **generation_kwargs
                    )
                
                # 解码答案
                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # 处理生成的文本，提取答案部分
                for text, sample_id in zip(generated_texts, ids):
                    # 提取Answer:后面的内容作为答案
                    answer = text.split("Answer:")[-1].strip()
                    # 只取第一行作为答案
                    answer = answer.split("\n")[0].strip()
                    
                    predictions.append({
                        "id": sample_id,
                        "prediction_text": answer
                    })
                
                # 定期清理显存
                if torch.cuda.is_available() and (i + batch_size) % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
            
            # 准备参考答案
            references = [{
                "id": ex["id"],
                "answers": {
                    "text": ex["answers"]["text"],
                    "answer_start": ex["answers"]["answer_start"]
                }
            } for ex in validation_dataset]
            
            # 计算评估指标
            results = self.squad_metric.compute(
                predictions=predictions,
                references=references
            )
            
            # 记录评估结果
            self.logger.info("\n========== 评估结果 ==========")
            self.logger.info(f"样本数: {len(validation_dataset)}")
            self.logger.info(f"Exact Match: {results['exact_match']:.2f}")
            self.logger.info(f"F1 Score: {results['f1']:.2f}")
            self.logger.info("==============================\n")
            
            return results
            
        except Exception as e:
            self.logger.error(f"评估过程发生错误: {str(e)}")
            self.logger.error(f"错误类型: {type(e)}")
            self.logger.error(f"错误位置: {e.__traceback__.tb_lineno}")
            raise

    def load_validation_dataset(self):
        """加载验证集数据"""
        self.logger.info("加载验证集...")
        try:
            dataset = load_dataset(self.test_dataset,split='validation')
            # return dataset.select(range(1000))
            return dataset
            
        except Exception as e:
            self.logger.error(f"加载数据集时发生错误: {str(e)}")
            raise

    def process_validation_data(self, validation_dataset):
        """处理验证集数据： squad格式
        
        Args:
            validation_dataset: 原始验证集数据
            
        Returns:
            processed_dataset: 处理后的验证集数据
        """
        self.logger.info("开始处理验证集数据...")
        
        processed_data = []
        for example in validation_dataset:
            try:
                # 验证必要字段
                required_fields = ["context", "question", "answers", "id"]
                if not all(field in example for field in required_fields):
                    self.logger.warning(f"样本 {example.get('id', 'unknown')} 缺少必要字段，跳过")
                    continue
                
                # 处理答案格式
                if not isinstance(example["answers"], dict) or "text" not in example["answers"]:
                    self.logger.warning(f"样本 {example['id']} 答案格式不正确，跳过")
                    continue
                
                # 清理文本
                processed_example = {
                    "id": example["id"],
                    "context": example["context"].strip(),
                    "question": example["question"].strip(),
                    "answers": {
                        "text": [ans.strip() for ans in example["answers"]["text"]],
                        "answer_start": example["answers"].get("answer_start", [])
                    }
                }
                
                processed_data.append(processed_example)
                
            except Exception as e:
                self.logger.error(f"处理样本时发生错误: {str(e)}")
                continue
        
        self.logger.info(f"验证集数据处理完成，共处理 {len(processed_data)} 个样本")
        return processed_data

def main():
    try:
        adapter_path = 'fine_tune_outputs/20250618_235327'
        test_dataset = "squad"
        
        # 初始化测试器
        tester = ModelTester(adapter_path, test_dataset)
        tester.logger.info("开始测试流程")
        tester.logger.info(f"使用适配器: {adapter_path}")
        tester.logger.info(f"测试数据集: {test_dataset}")
        
        # 加载模型
        try:
            model, tokenizer = tester.load_model()
        except Exception as e:
            tester.logger.error(f"加载模型失败: {str(e)}")
            raise
        
        # 加载验证集
        try:
            validation_dataset = tester.load_validation_dataset()
            tester.logger.info(f"成功加载验证集，样本数量: {len(validation_dataset)}")
        except Exception as e:
            tester.logger.error(f"加载验证集失败: {str(e)}")
            raise
        
        # 评估模型
        try:
            results = tester.evaluate_model(model, tokenizer, validation_dataset)
        except Exception as e:
            tester.logger.error(f"评估模型失败: {str(e)}")
            raise
        
        # 记录评估结果
        tester.logger.info("\n========== SQuAD 评估结果 ==========")
        tester.logger.info(f"基座模型: {tester.model_name}")
        tester.logger.info(f"适配器路径: {adapter_path if adapter_path else '未使用适配器'}")
        tester.logger.info(f"验证集样本数: {len(validation_dataset)}")
        tester.logger.info(f"Exact Match: {results['exact_match']:.2f}%")
        tester.logger.info(f"F1 Score: {results['f1']:.2f}%")
        tester.logger.info("===================================\n")
        
        # 保存结果到 JSON 文件
        result_file = os.path.join(tester.run_dir, "squad_results.json")
        with open(result_file, "w", encoding='utf-8') as f:
            json.dump({
                "model_info": {
                    "base_model": tester.model_name,
                    "adapter_path": adapter_path
                },
                "evaluation": {
                    "dataset": "SQuAD",
                    "sample_count": len(validation_dataset),
                    "metrics": {
                        "exact_match": float(results['exact_match']),
                        "f1": float(results['f1'])
                    }
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2, ensure_ascii=False)
        
        tester.logger.info(f"评估结果已保存到: {result_file}")
        tester.logger.info("测试流程完成")
        
    except Exception as e:
        tester.logger.error(f"测试过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()
