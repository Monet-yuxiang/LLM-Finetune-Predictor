"""
Qwen微调格式转换器 - 简化版本
"""

import json
import logging
from typing import Dict, List
from collections import defaultdict

logger = logging.getLogger(__name__)

def extract_context_and_question(user_content: str) -> tuple:
    """从用户内容中提取context和question"""
    # 处理SQuAD格式: "Context: xxx\n\nQuestion: xxx"
    if "Context:" in user_content and "Question:" in user_content:
        parts = user_content.split("Question:", 1)
        if len(parts) == 2:
            context = parts[0].replace("Context:", "").strip()
            question = parts[1].strip()
            return context, question
    
    # 处理Alpaca格式: "instruction\n\ninput"
    if "\n\n" in user_content:
        parts = user_content.split("\n\n", 1)
        if len(parts) == 2:
            instruction = parts[0].strip()
            input_text = parts[1].strip()
            if input_text:
                return input_text, instruction
    
    # 默认情况
    return user_content, user_content

def qwen_to_custom_format(qwen_data: List[Dict]) -> List[Dict]:
    """将Qwen格式转换为自定义格式"""
    context_dict = defaultdict(list)
    
    for item in qwen_data:
        conversations = item.get("conversations", [])
        
        if len(conversations) < 2:
            continue
            
        user_msg = next((msg["content"] for msg in conversations if msg["role"] == "user"), "")
        assistant_msg = next((msg["content"] for msg in conversations if msg["role"] == "assistant"), "")
        
        if not user_msg or not assistant_msg:
            continue
            
        context, question = extract_context_and_question(user_msg)
        context_dict[context].append({
            "question": question,
            "output": assistant_msg
        })
    
    # 组装目标格式
    return [
        {
            "context_text": context,
            "qa_pairs": qa_pairs
        }
        for context, qa_pairs in context_dict.items()
    ]

def convert_file(input_file: str, output_file: str) -> str:
    """从文件转换"""
    with open(input_file, 'r', encoding='utf-8') as f:
        qwen_data = json.load(f)
    
    custom_data = qwen_to_custom_format(qwen_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(custom_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"文件转换完成: {input_file} -> {output_file}")
    return output_file

if __name__ == "__main__":
    # 示例用法
    with open("input_qwen.json", "r", encoding="utf-8") as f:
        qwen_data = json.load(f)
    
    custom_data = qwen_to_custom_format(qwen_data)
    
    with open("output_custom.json", "w", encoding="utf-8") as f:
        json.dump(custom_data, f, ensure_ascii=False, indent=2)
    
    print("转换完成，输出文件：output_custom.json") 