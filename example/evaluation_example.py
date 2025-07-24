import os
import json
import logging
from datetime import datetime
from typing import List, Dict
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load as load_metric
import random

# ========== 日志设置 ==========
def setup_logging(log_file='evaluation.log'):
    logger = logging.getLogger('evaluation')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

# ========== 模型加载 ==========
def load_lora_model(base_model_path: str, adapter_path: str = None, logger=None):
    """
    加载本地LoRA微调模型，支持LoRA权重合并。
    base_model_path: 基座模型路径
    adapter_path: LoRA权重路径（可选）
    """
    from peft import PeftModel, PeftConfig
    logger = logger or logging.getLogger('evaluation')
    logger.info(f"加载分词器: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"加载基座模型: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True
    )
    if adapter_path:
        logger.info(f"加载LoRA权重: {adapter_path}")
        peft_config = PeftConfig.from_pretrained(adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        try:
            model = model.merge_and_unload()
            logger.info("LoRA权重合并成功")
        except Exception as e:
            logger.warning(f"LoRA权重合并失败: {str(e)}，将使用未合并的模型继续")
    model.eval()
    return model, tokenizer

# ========== 数据集加载 ==========
def load_squad_validation(logger=None) -> List[Dict]:
    logger = logger or logging.getLogger('evaluation')
    logger.info("加载SQuAD1.1验证集...")
    dataset = load_dataset("squad", split="validation")
    processed_data = []
    for example in dataset:
        # 清理文本
        processed_data.append({
            "id": example["id"],
            "context": example["context"].strip(),
            "question": example["question"].strip(),
            "answers": {
                "text": [ans.strip() for ans in example["answers"]["text"]],
                "answer_start": example["answers"].get("answer_start", [])
            }
        })
    logger.info(f"验证集加载完成，共{len(processed_data)}条样本")
    return processed_data

# ========== Prompt模板 ==========
def build_prompt(context, question):
    return (
        "Extract the exact answer from context. Do not explain. only exact phrase\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

# ========== 答案标准化 ==========
def normalize_answer(s: str) -> str:
    import re
    s = s.lower()
    s = re.sub(r'[\W_]+', ' ', s)
    s = ' '.join([word for word in s.split() if word not in ['a', 'an', 'the']])
    s = ' '.join(s.split())
    return s

# ========== 批量生成答案 ==========
def batch_generate(model, tokenizer, prompts, batch_size=16, generation_kwargs=None, logger=None):
    logger = logger or logging.getLogger('evaluation')
    all_results = []
    model.eval()
    generation_kwargs = generation_kwargs or {
        'max_new_tokens': 40,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'repetition_penalty': 1.0,
        'use_cache': True
    }
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="推理进度"):
            batch_prompts = prompts[i:i+batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
            outputs = model.generate(**inputs, **generation_kwargs)
            for j, output in enumerate(outputs):
                input_len = inputs['input_ids'].shape[1]
                gen_text = tokenizer.decode(output[input_len:], skip_special_tokens=True)
                all_results.append(gen_text.strip())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return all_results

# ========== 推荐 batch_size ==========
def recommend_batch_size(data, tokenizer, max_new_tokens=40, reserved_mem_gb=2, logger=None):
    """
    根据数据集、模型、环境，推荐batch_size，仅供参考。
    """
    logger = logger or logging.getLogger('evaluation')
    sample_num = min(20, len(data))
    samples = random.sample(data, sample_num)
    prompts = [build_prompt(item['context'], item['question']) for item in samples]
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
        batch_size = max(1, int(free // per_sample_gb))
        logger.info(f"[推荐参数] 数据集均值token数≈{avg_tokens:.1f}，单样本估算显存≈{per_sample_gb:.2f}GB")
        logger.info(f"[推荐参数] 当前GPU总显存={total:.1f}GB，已用={used:.1f}GB，预留={reserved_mem_gb}GB")
        logger.info(f"[推荐参数] 推荐batch_size≈{batch_size}（仅供参考，实际可根据报错/速度调整）")
    else:
        logger.info(f"[推荐参数] CPU环境，建议batch_size=1")

# ========== 评测主流程 ==========
def evaluate_qa(base_model_path, adapter_path=None, batch_size=16):
    logger = setup_logging()
    logger.info("========== 开始评测流程 ==========")
    logger.info(f"基座模型: {base_model_path}")
    if adapter_path:
        logger.info(f"LoRA权重: {adapter_path}")
    else:
        logger.info("未使用LoRA权重")
    # 1. 加载模型
    model, tokenizer = load_lora_model(base_model_path, adapter_path, logger)
    # 2. 加载数据集
    data = load_squad_validation(logger)
    # data = data[:1000]  
    # logger.info("测试前1000条")
    # 3. 构建prompts
    prompts = [build_prompt(item['context'], item['question']) for item in data]
    # 4. 推荐 batch_size
    recommend_batch_size(data, tokenizer, max_new_tokens=40, reserved_mem_gb=2, logger=logger)
    # 5. 批量生成
    preds = batch_generate(model, tokenizer, prompts, batch_size=batch_size, logger=logger)
    # 6. 评测
    predictions, references = [], []
    for item, pred in zip(data, preds):
        pred_norm = normalize_answer(pred)
        gold_list = item['answers']['text']
        gold_norm_list = [normalize_answer(g) for g in gold_list]
        predictions.append({'id': item['id'], 'prediction_text': pred_norm})
        references.append({'id': item['id'], 'answers': {'text': gold_norm_list, 'answer_start': [0]*len(gold_norm_list)}})
    squad_metric = load_metric("squad")
    results = squad_metric.compute(predictions=predictions, references=references)
    logger.info(f"EM: {results['exact_match']:.2f}, F1: {results['f1']:.2f}")
    # 7. 保存结果
    run_dir = f"eval_outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    result_file = os.path.join(run_dir, "squad_results.json")
    with open(result_file, "w", encoding='utf-8') as f:
        json.dump({
            "base_model": base_model_path,
            "adapter_path": adapter_path,
            "metrics": results,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"评测结果已保存到: {result_file}")
    logger.info("========== 评测流程结束 ==========")

# ========== 主程序入口 ==========
if __name__ == "__main__":
    # 修改为你的本地模型和LoRA权重路径
    base_model_path = os.getenv('MODEL_NAME')   # 基座模型路径
    adapter_path = 'fine_tune_outputs/20250710_221451'  # LoRA权重路径（如无LoRA可设为None）
     

    # 
    evaluate_qa(base_model_path, adapter_path, batch_size=16)