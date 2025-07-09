"""
测试完整转换输出结果：显示详细的转换过程
"""

import json
import tempfile
import os
from dataset_to_qwen_converter import DatasetConverter
from qwen_to_final_converter import qwen_to_custom_format, convert_file

def create_test_data():
    """创建四种格式的测试数据"""
    
    # 1. Alpaca格式测试数据 - 修正逻辑
    alpaca_data = [
        {
            "instruction": "基于以下信息回答人工智能是什么？",
            "input": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "output": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"
        },
        {
            "instruction": "总结北京的主要特点",
            "input": "北京是中国的首都，是政治、文化、国际交往中心，拥有悠久的历史文化。",
            "output": "北京是中国的首都，是政治、文化、国际交往中心。"
        }
    ]
    
    # 2. Dolly15k格式测试数据 - 修正逻辑
    dolly15k_data = [
        {
            "instruction": "什么是机器学习？",
            "context": "机器学习是人工智能的一个子领域，通过算法让计算机从数据中学习，而不需要明确编程。",
            "response": "机器学习是人工智能的一个子领域，通过算法让计算机从数据中学习。"
        },
        {
            "instruction": "上海的主要特点是什么？",
            "context": "上海是中国最大的城市，是经济、金融、贸易中心，拥有国际化大都市的特点。",
            "response": "上海是中国最大的城市，是经济、金融、贸易中心。"
        }
    ]
    
    # 3. QA对格式测试数据 - 保持合理
    qa_pair_data = [
        {
            "question": "什么是深度学习？",
            "answer": "深度学习是机器学习的一个分支，使用多层神经网络进行学习。"
        },
        {
            "question": "什么是自然语言处理？",
            "answer": "自然语言处理是人工智能的一个分支，致力于让计算机理解和生成人类语言。"
        }
    ]
    
    # 4. SQuAD格式测试数据 - 修正逻辑
    squad_data = [
        {
            "paragraphs": [
                {
                    "context": "人工智能是计算机科学的一个分支，致力于创建智能系统。它涵盖了机器学习、自然语言处理、计算机视觉等多个子领域。",
                    "qas": [
                        {
                            "question": "什么是人工智能？",
                            "answers": [{"text": "人工智能是计算机科学的一个分支，致力于创建智能系统。"}]
                        },
                        {
                            "question": "人工智能包含哪些子领域？",
                            "answers": [{"text": "人工智能涵盖了机器学习、自然语言处理、计算机视觉等多个子领域。"}]
                        }
                    ]
                }
            ]
        },
        {
            "paragraphs": [
                {
                    "context": "机器学习是AI的子领域，通过数据训练模型。它使用算法让计算机从数据中学习，而不需要明确编程。",
                    "qas": [
                        {
                            "question": "机器学习是什么？",
                            "answers": [{"text": "机器学习是AI的子领域，通过数据训练模型。"}]
                        },
                        {
                            "question": "机器学习如何工作？",
                            "answers": [{"text": "机器学习使用算法让计算机从数据中学习，而不需要明确编程。"}]
                        }
                    ]
                }
            ]
        }
    ]
    
    return {
        "alpaca": alpaca_data,
        "dolly15k": dolly15k_data,
        "qa_pair": qa_pair_data,
        "squad": squad_data
    }

def print_json_pretty(data, title):
    """美化打印JSON数据"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"{'='*60}")

def test_detailed_conversion():
    """测试详细转换过程"""
    print("开始测试详细转换过程...")
    
    # 创建测试数据
    test_data = create_test_data()
    converter = DatasetConverter()
    
    for format_name, data in test_data.items():
        print(f"\n{'#'*80}")
        print(f"测试 {format_name.upper()} 格式转换")
        print(f"{'#'*80}")
        
        # 显示原始数据
        print_json_pretty(data, f"原始 {format_name} 格式数据")
        
        try:
            # 第一步：转换为Qwen格式
            qwen_data = converter.convert(data, format_name)
            print_json_pretty(qwen_data, f"转换为Qwen格式数据")
            
            # 第二步：转换为最终格式
            final_data = qwen_to_custom_format(qwen_data)
            print_json_pretty(final_data, f"最终格式数据")
            
            # 统计信息
            print(f"\n转换统计:")
            print(f"  原始数据: {len(data)} 条")
            print(f"  Qwen格式: {len(qwen_data)} 条")
            print(f"  最终格式: {len(final_data)} 条")
            
            # 详细分析每个条目
            for i, item in enumerate(final_data):
                print(f"\n  条目 {i+1} 详情:")
                print(f"    context_text: {item['context_text']}")
                print(f"    qa_pairs数量: {len(item['qa_pairs'])}")
                for j, qa in enumerate(item['qa_pairs']):
                    print(f"      问答对 {j+1}:")
                    print(f"        问题: {qa['question']}")
                    print(f"        答案: {qa['output']}")
            
        except Exception as e:
            print(f"转换失败: {e}")

def test_mixed_context_detailed():
    """测试混合context场景的详细输出"""
    print(f"\n{'#'*80}")
    print("测试混合Context场景")
    print(f"{'#'*80}")
    
    # 创建包含相同context的测试数据 - 修正逻辑
    mixed_data = [
        {
            "instruction": "什么是人工智能？",
            "input": "Context: 人工智能是计算机科学的一个分支，致力于创建智能系统。",
            "output": "人工智能是计算机科学的一个分支，致力于创建智能系统。"
        },
        {
            "instruction": "人工智能有哪些应用领域？",
            "input": "Context: 人工智能是计算机科学的一个分支，致力于创建智能系统。",
            "output": "人工智能的应用领域包括机器学习、自然语言处理、计算机视觉等。"
        },
        {
            "instruction": "什么是机器学习？",
            "input": "Context: 机器学习是AI的子领域，通过数据训练模型。",
            "output": "机器学习是AI的子领域，通过算法让计算机从数据中学习。"
        }
    ]
    
    print_json_pretty(mixed_data, "原始混合数据")
    
    converter = DatasetConverter()
    
    try:
        # 转换为Qwen格式
        qwen_data = converter.convert(mixed_data, "alpaca")
        print_json_pretty(qwen_data, "转换为Qwen格式")
        
        # 转换为最终格式
        final_data = qwen_to_custom_format(qwen_data)
        print_json_pretty(final_data, "最终格式（按context分组）")
        
        print(f"\n混合Context分析:")
        print(f"  原始数据: {len(mixed_data)} 条")
        print(f"  Qwen格式: {len(qwen_data)} 条")
        print(f"  最终格式: {len(final_data)} 条")
        
        # 分析context合并
        for i, item in enumerate(final_data):
            print(f"\n  条目 {i+1}:")
            print(f"    context: {item['context_text']}")
            print(f"    合并的问答对数量: {len(item['qa_pairs'])}")
            for j, qa in enumerate(item['qa_pairs']):
                print(f"      问答对 {j+1}:")
                print(f"        问题: {qa['question']}")
                print(f"        答案: {qa['output']}")
        
    except Exception as e:
        print(f"混合context测试失败: {e}")

def test_file_output_detailed():
    """测试文件输出的详细内容"""
    print(f"\n{'#'*80}")
    print("测试文件输出详细内容")
    print(f"{'#'*80}")
    
    # 使用Alpaca格式作为示例
    test_data = create_test_data()["alpaca"]
    converter = DatasetConverter()
    
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
            input_file = f.name
        
        qwen_file = input_file.replace('.json', '_qwen.json')
        final_file = input_file.replace('.json', '_final.json')
        
        # 第一步：转换为Qwen格式文件
        converter.convert_file(input_file, qwen_file, "alpaca")
        
        # 读取并显示Qwen格式文件内容
        with open(qwen_file, 'r', encoding='utf-8') as f:
            qwen_file_data = json.load(f)
        print_json_pretty(qwen_file_data, "Qwen格式文件内容")
        
        # 第二步：转换为最终格式文件
        convert_file(qwen_file, final_file)
        
        # 读取并显示最终格式文件内容
        with open(final_file, 'r', encoding='utf-8') as f:
            final_file_data = json.load(f)
        print_json_pretty(final_file_data, "最终格式文件内容")
        
        # 清理临时文件
        os.unlink(input_file)
        os.unlink(qwen_file)
        os.unlink(final_file)
        
    except Exception as e:
        print(f"文件输出测试失败: {e}")

def main():
    """运行详细测试"""
    print("开始详细转换输出测试...")
    
    # 测试详细转换过程
    test_detailed_conversion()
    
    # 测试混合context场景
    test_mixed_context_detailed()
    
    # 测试文件输出
    test_file_output_detailed()
    
    print(f"\n{'#'*80}")
    print("详细测试完成")
    print(f"{'#'*80}")

if __name__ == "__main__":
    main() 