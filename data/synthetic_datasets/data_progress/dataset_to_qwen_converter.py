"""
数据集格式转换工具 - 简洁版本

支持将四种数据集格式转换为Qwen2.5-7B的LoRA微调格式
"""

import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseParser(ABC):
    """数据集解析器基类"""
    
    def __init__(self, system_prompt: str = "你是一个有用的AI助手。"):
        self.system_prompt = system_prompt
    
    @abstractmethod
    def parse_item(self, item: Dict) -> Optional[Dict]:
        """解析单个数据项"""
        pass
    
    def convert(self, raw_data: List[Dict]) -> List[Dict]:
        """转换数据格式"""
        if not isinstance(raw_data, list) or not raw_data:
            raise ValueError("数据格式无效")
        
        result = []
        skipped = 0
        
        for item in raw_data:
            parsed = self.parse_item(item)
            if parsed:
                result.append(parsed)
            else:
                skipped += 1
        
        logger.info(f"{self.__class__.__name__}转换完成: {len(result)} 条有效数据, {skipped} 条跳过")
        return result

class AlpacaParser(BaseParser):
    """Alpaca格式解析器"""
    
    def parse_item(self, item: Dict) -> Optional[Dict]:
        instruction = item.get("instruction", "")
        context = item.get("input", "")
        answer = item.get("output", "")
        
        if not instruction or not answer:
            return None
        
        user_content = f"{instruction}\n\n{context}" if context else instruction
        
        return {
            "conversations": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer}
            ]
        }

class Dolly15kParser(BaseParser):
    """Dolly15k格式解析器"""
    
    def parse_item(self, item: Dict) -> Optional[Dict]:
        instruction = item.get("instruction", "")
        context = item.get("context", "")
        answer = item.get("response", "")
        
        if not instruction or not answer:
            return None
        
        user_content = f"Context: {context}\n\nQuestion: {instruction}" if context else instruction
        
        return {
            "conversations": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer}
            ]
        }

class QAPairParser(BaseParser):
    """QA对格式解析器"""
    
    def parse_item(self, item: Dict) -> Optional[Dict]:
        # 支持多种键名
        question = (item.get("question") or item.get("prompt") or 
                   item.get("input") or item.get("instruction"))
        answer = (item.get("answer") or item.get("completion") or 
                 item.get("output") or item.get("response"))
        
        if not question or not answer:
            return None
        
        return {
            "conversations": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }

class SquadParser(BaseParser):
    """SQuAD格式解析器"""
    
    def parse_item(self, item: Dict) -> List[Dict]:
        """解析单个数据项，返回多个对话"""
        result = []
        
        # 处理标准SQuAD格式
        if "paragraphs" in item:
            for paragraph in item["paragraphs"]:
                context = paragraph.get("context", "")
                for qa in paragraph.get("qas", []):
                    question = qa.get("question", "")
                    answers = qa.get("answers", [])
                    
                    if question and answers:
                        answer = answers[0].get("text", "")
                        if answer:
                            result.append({
                                "conversations": [
                                    {"role": "system", "content": self.system_prompt},
                                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
                                    {"role": "assistant", "content": answer}
                                ]
                            })
        
        # 处理简化SQuAD格式
        elif "question" in item and "context" in item:
            question = item.get("question", "")
            context = item.get("context", "")
            answer = item.get("answer", "")
            
            if question and answer:
                result.append({
                    "conversations": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
                        {"role": "assistant", "content": answer}
                    ]
                })
        
        return result
    
    def convert(self, raw_data: List[Dict]) -> List[Dict]:
        """转换数据格式"""
        if not isinstance(raw_data, list) or not raw_data:
            raise ValueError("数据格式无效")
        
        result = []
        skipped = 0
        
        for item in raw_data:
            parsed_items = self.parse_item(item)
            if parsed_items:
                result.extend(parsed_items)
            else:
                skipped += 1
        
        logger.info(f"{self.__class__.__name__}转换完成: {len(result)} 条有效数据, {skipped} 条跳过")
        return result

class DatasetConverter:
    """数据集格式转换器"""
    
    def __init__(self, system_prompt: str = "你是一个有用的AI助手。"):
        self.system_prompt = system_prompt
        self.parsers = {
            "alpaca": AlpacaParser(system_prompt),
            "dolly15k": Dolly15kParser(system_prompt),
            "qa_pair": QAPairParser(system_prompt),
            "squad": SquadParser(system_prompt)
        }
    
    def detect_format(self, raw_data: List[Dict]) -> str:
        """自动检测数据格式"""
        if not raw_data:
            raise ValueError("数据为空")
        
        sample = raw_data[0]
        
        # 检测规则
        if "instruction" in sample and "output" in sample:
            return "alpaca"
        elif "instruction" in sample and "response" in sample and "context" in sample:
            return "dolly15k"
        elif "paragraphs" in sample or ("question" in sample and "context" in sample):
            return "squad"
        elif any(key in sample for key in ["question", "prompt", "input"]) and \
             any(key in sample for key in ["answer", "completion", "output", "response"]):
            return "qa_pair"
        
        raise ValueError(f"无法识别的数据格式: {list(sample.keys())}")
    
    def convert(self, raw_data: List[Dict], format_type: Optional[str] = None) -> List[Dict]:
        """转换数据格式"""
        if format_type is None:
            format_type = self.detect_format(raw_data)
            logger.info(f"自动检测到数据格式: {format_type}")
        
        if format_type not in self.parsers:
            raise ValueError(f"不支持的格式类型: {format_type}")
        
        return self.parsers[format_type].convert(raw_data)
    
    def convert_file(self, input_file: str, output_file: str, format_type: Optional[str] = None) -> str:
        """从文件转换数据格式"""
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        converted_data = self.convert(raw_data, format_type)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"转换完成: {input_file} -> {output_file} ({len(converted_data)} 条数据)")
        return output_file
    
    def convert_batch(self, input_files: List[str], output_dir: str, format_type: Optional[str] = None) -> List[str]:
        """批量转换文件"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = []
        for input_file in input_files:
            try:
                input_path = Path(input_file)
                output_file = output_dir / f"{input_path.stem}_qwen.json"
                result_file = self.convert_file(str(input_path), str(output_file), format_type)
                output_files.append(result_file)
            except Exception as e:
                logger.error(f"转换文件 {input_file} 失败: {e}")
        
        return output_files

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据集格式转换工具")
    parser.add_argument("input", help="输入文件或目录")
    parser.add_argument("output", help="输出文件或目录")
    parser.add_argument("--format", choices=["alpaca", "dolly15k", "qa_pair", "squad"], 
                       help="数据格式类型（可选，会自动检测）")
    parser.add_argument("--system-prompt", default="你是一个有用的AI助手。",
                       help="系统提示词")
    parser.add_argument("--batch", action="store_true", help="批量处理模式")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    converter = DatasetConverter(args.system_prompt)
    
    if args.batch:
        input_path = Path(args.input)
        input_files = [str(input_path)] if input_path.is_file() else [str(f) for f in input_path.glob("*.json")]
        output_files = converter.convert_batch(input_files, args.output, args.format)
        print(f"批量转换完成，共处理 {len(output_files)} 个文件")
    else:
        output_file = converter.convert_file(args.input, args.output, args.format)
        print(f"转换完成: {output_file}")

if __name__ == "__main__":
    main() 