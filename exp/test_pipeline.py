"""
测试文件 - 验证完整的训练-评估流水线
主要功能：
1. 测试单个模块功能
2. 测试完整流水线
3. 验证内存管理
4. 检查结果格式
"""

import os
import json
import sys
import time
import torch
import gc
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模块
from train_module import train_model_on_dataset_with_progress
from evaluate_module import evaluate_model_on_dataset_with_progress
from pipeline_module import run_pipeline_with_progress

def test_individual_modules():
    """测试单个模块功能"""
    print("=" * 80)
    print("🧪 测试单个模块功能")
    print("=" * 80)
    
    # 测试配置
    model_path = "/home/haonan/data_decision/models/Qwen/Qwen2.5-7B-Instruct-1M"
    dataset_path = "/home/haonan/data_decision/example/qwen_dataset/qa_dataset_20250709_1_25.json"
    lora_config_dir = "/home/haonan/data_decision/lora_configs"  # 需要创建这个目录
    config_name = "qwen_lora_config"
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return False
    
    # 创建LoRA配置目录和文件
    os.makedirs(lora_config_dir, exist_ok=True)
    lora_config_file = os.path.join(lora_config_dir, f"{config_name}.json")
    
    if not os.path.exists(lora_config_file):
        # 创建默认LoRA配置
        default_config = {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "bias": "none"
        }
        with open(lora_config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"✅ 创建默认LoRA配置: {lora_config_file}")
    
    try:
        # 加载模型和分词器
        print("📥 加载模型和分词器...")
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
        
        print("✅ 模型和分词器加载成功")
        
        # 测试训练模块
        print("\n🔧 测试训练模块...")
        try:
            train_result = train_model_on_dataset_with_progress(
                model=model,
                tokenizer=tokenizer,
                train_dataset_path=dataset_path,
                lora_config_dir=lora_config_dir,
                config_name=config_name,
                dataset_index=1,
                total_datasets=1
            )
            
            print("✅ 训练模块测试成功")
            print(f"   训练时间: {train_result['training_time']:.2f}秒")
            print(f"   训练样本数: {train_result['train_samples']}")
            print(f"   最终loss: {train_result['final_loss']:.4f}")
            
        except Exception as e:
            print(f"❌ 训练模块测试失败: {e}")
            return False
        
        # 测试评估模块
        print("\n🔍 测试评估模块...")
        try:
            eval_result = evaluate_model_on_dataset_with_progress(
                model=model,
                tokenizer=tokenizer,
                eval_dataset_path=dataset_path,
                dataset_index=1,
                total_datasets=1
            )
            
            print("✅ 评估模块测试成功")
            print(f"   评估时间: {eval_result['evaluation_time']:.2f}秒")
            print(f"   评估样本数: {eval_result['eval_samples']}")
            print(f"   EM Score: {eval_result['metrics']['exact_match']:.4f}")
            print(f"   F1 Score: {eval_result['metrics']['f1']:.4f}")
            
        except Exception as e:
            print(f"❌ 评估模块测试失败: {e}")
            return False
        
        # 清理资源
        print("\n🧹 清理资源...")
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ 单个模块测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 模块测试过程中出错: {e}")
        return False

def test_full_pipeline():
    """测试完整流水线"""
    print("\n" + "=" * 80)
    print("🚀 测试完整流水线")
    print("=" * 80)
    
    # 测试配置
    dataset_paths = [
        "/home/haonan/data_decision/example/qwen_dataset/qa_dataset_20250709_1_25.json"
    ]
    
    model_path = "/home/haonan/data_decision/models/Qwen/Qwen2.5-7B-Instruct-1M"
    lora_config_dir = "/home/haonan/data_decision/lora_configs"
    config_name = "qwen_lora_config"
    result_output_path = "/home/haonan/data_decision/results/test_pipeline_results.json"
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    for dataset_path in dataset_paths:
        if not os.path.exists(dataset_path):
            print(f"❌ 数据集路径不存在: {dataset_path}")
            return False
    
    try:
        # 运行完整流水线
        print("🔄 开始运行完整流水线...")
        start_time = time.time()
        
        summary = run_pipeline_with_progress(
            dataset_paths=dataset_paths,
            model_path=model_path,
            lora_config_dir=lora_config_dir,
            config_name=config_name,
            result_output_path=result_output_path
        )
        
        pipeline_time = time.time() - start_time
        
        # 验证结果
        print("\n📊 验证流水线结果...")
        
        # 检查结果文件
        if os.path.exists(result_output_path):
            print(f"✅ 结果文件已保存: {result_output_path}")
            
            # 读取并显示结果摘要
            with open(result_output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            pipeline_info = results['pipeline_info']
            summary_stats = results['summary_statistics']
            
            print(f"   总执行时间: {pipeline_info['pipeline_time']:.2f}秒")
            print(f"   成功数据集: {pipeline_info['successful_datasets']}/{pipeline_info['total_datasets']}")
            print(f"   失败数据集: {pipeline_info['failed_datasets']}")
            
            if pipeline_info['successful_datasets'] > 0:
                print(f"   平均EM分数: {summary_stats['avg_em_score']:.4f}")
                print(f"   平均F1分数: {summary_stats['avg_f1_score']:.4f}")
                print(f"   最佳EM分数: {summary_stats['best_em_score']:.4f}")
                print(f"   最佳F1分数: {summary_stats['best_f1_score']:.4f}")
            
            # 显示详细结果
            print("\n📋 详细结果:")
            for i, dataset_result in enumerate(results['dataset_results'], 1):
                if 'error' not in dataset_result:
                    training = dataset_result['training']
                    evaluation = dataset_result['evaluation']
                    print(f"   数据集 {i}: {dataset_result['dataset_name']}")
                    print(f"     训练时间: {training['training_time']:.2f}秒")
                    print(f"     训练样本: {training['train_samples']}")
                    print(f"     最终loss: {training['final_loss']:.4f}")
                    print(f"     评估时间: {evaluation['evaluation_time']:.2f}秒")
                    print(f"     评估样本: {evaluation['eval_samples']}")
                    print(f"     EM Score: {evaluation['metrics']['exact_match']:.4f}")
                    print(f"     F1 Score: {evaluation['metrics']['f1']:.4f}")
                else:
                    print(f"   数据集 {i}: {dataset_result['dataset_name']} - 错误: {dataset_result['error']}")
        
        else:
            print(f"❌ 结果文件未生成: {result_output_path}")
            return False
        
        print(f"\n✅ 完整流水线测试成功！")
        print(f"   实际执行时间: {pipeline_time:.2f}秒")
        return True
        
    except Exception as e:
        print(f"❌ 完整流水线测试失败: {e}")
        return False

def test_memory_management():
    """测试内存管理"""
    print("\n" + "=" * 80)
    print("🧠 测试内存管理")
    print("=" * 80)
    
    if torch.cuda.is_available():
        print("🔍 检查GPU内存使用情况...")
        
        # 记录初始内存
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"   初始GPU内存使用: {initial_memory:.2f} GB")
        
        # 运行一个简单的测试
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_path = "/home/haonan/data_decision/models/Qwen/Qwen2.5-7B-Instruct-1M"
            
            if os.path.exists(model_path):
                print("📥 加载模型测试内存...")
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    device_map="auto"
                )
                
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                tokenizer.pad_token = tokenizer.eos_token
                
                # 记录加载后内存
                loaded_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"   加载后GPU内存使用: {loaded_memory:.2f} GB")
                print(f"   内存增长: {loaded_memory - initial_memory:.2f} GB")
                
                # 清理资源
                print("🧹 清理资源...")
                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                
                # 记录清理后内存
                final_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"   清理后GPU内存使用: {final_memory:.2f} GB")
                print(f"   内存释放: {loaded_memory - final_memory:.2f} GB")
                
                if final_memory <= initial_memory + 0.1:  # 允许0.1GB的误差
                    print("✅ 内存管理测试通过")
                    return True
                else:
                    print("⚠️  内存可能未完全释放")
                    return False
            else:
                print("⚠️  模型路径不存在，跳过内存测试")
                return True
        except Exception as e:
            print(f"❌ 内存管理测试失败: {e}")
            return False
    else:
        print("⚠️  未检测到GPU，跳过内存测试")
        return True

def main():
    """主测试函数"""
    print("🎯 开始完整流程测试")
    print("=" * 80)
    
    # 创建结果目录
    results_dir = "/home/haonan/data_decision/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 测试结果
    test_results = {
        "individual_modules": False,
        "full_pipeline": False,
        "memory_management": False
    }
    
    # 1. 测试单个模块
    print("\n📋 测试项目:")
    print("1. 单个模块功能测试")
    print("2. 完整流水线测试")
    print("3. 内存管理测试")
    
    test_results["individual_modules"] = test_individual_modules()
    
    # 2. 测试完整流水线
    test_results["full_pipeline"] = test_full_pipeline()
    
    # 3. 测试内存管理
    test_results["memory_management"] = test_memory_management()
    
    # 显示最终结果
    print("\n" + "=" * 80)
    print("📊 测试结果汇总")
    print("=" * 80)
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(test_results.values())
    
    if all_passed:
        print("\n🎉 所有测试通过！系统可以正常使用。")
    else:
        print("\n⚠️  部分测试失败，请检查相关配置。")
    
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程中出现未预期的错误: {e}")
        sys.exit(1) 