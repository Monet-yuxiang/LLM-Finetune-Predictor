"""
特征总调度模块

统一调度所有特征提取，返回完整特征字典。
优化策略：将动态探针放在最后，只使用一个模型实例
"""
from .data_parsers import DatasetAnalyzer, HyperParams
from .static_features import StaticFeatureExtractor
from .dynamic_probes import DynamicProbeAnalyzer
from typing import Dict, List, Optional
import torch
import logging
import csv
import os
from datetime import datetime

logger = logging.getLogger(__name__)

def extract_all_features(
    dataset: List[Dict],
    hyperparams: HyperParams,
    base_model_name: str,
    static_sample_size: int = 100,
    dynamic_probe_steps: int = 100,
    dynamic_sample_size: int = 50,
    save_to_csv: bool = False,
    csv_filename: Optional[str] = None
) -> Dict[str, float]:
    """
    一口气提取所有特征（基础+静态+动态）
    
    优化策略：将动态探针放在最后，只使用一个模型实例
    """
    # 创建基础分析器（只加载一次模型）
    base_analyzer = DatasetAnalyzer(base_model_name)
    
    try:
        # 基础特征
        logger.info("提取基础特征...")
        basic_features = base_analyzer.extract_all_basic_features(dataset, hyperparams, static_sample_size)
        
        # 静态特征（复用基础分析器的模型）
        logger.info("提取静态特征...")
        static_analyzer = StaticFeatureExtractor(base_model_name)
        static_analyzer.model = base_analyzer.model
        static_analyzer.tokenizer = base_analyzer.tokenizer
        static_features = static_analyzer.extract_all_static_features(dataset, static_sample_size)
        
        # 动态特征（复用基础分析器的模型，但会进行训练）
        logger.info("提取动态特征...")
        dynamic_analyzer = DynamicProbeAnalyzer(base_model_name)
        dynamic_analyzer.model = base_analyzer.model
        dynamic_analyzer.tokenizer = base_analyzer.tokenizer
        dynamic_features = dynamic_analyzer.extract_all_dynamic_features(
            dataset, hyperparams, dynamic_probe_steps, dynamic_sample_size
        )
        
        # 合并所有特征
        features = {}
        features.update(basic_features)
        features.update(static_features)
        features.update(dynamic_features)
        
        # 保存到CSV文件
        if save_to_csv:
            save_features_to_csv(features, csv_filename, dataset, hyperparams)
        
        return features
        
    finally:
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("已清理GPU显存")

def save_features_to_csv(
    features: Dict[str, float], 
    csv_filename: Optional[str] = None,
    dataset: Optional[List[Dict]] = None,
    hyperparams: Optional[HyperParams] = None
) -> str:
    """
    将特征保存到CSV文件
    
    Args:
        features: 特征字典
        csv_filename: CSV文件名，如果为None则自动生成
        dataset: 数据集（用于记录数据集信息）
        hyperparams: 超参数（用于记录超参数信息）
    
    Returns:
        保存的文件路径
    """
    
    # 生成文件名
    if csv_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"dataset_features_{timestamp}.csv"
    
    # 确保文件扩展名正确
    if not csv_filename.endswith('.csv'):
        csv_filename += '.csv'
    
    # 准备CSV数据
    csv_data = []
    
    # 添加数据集信息
    if dataset is not None:
        csv_data.append(["dataset_size", len(dataset)])
    
    # 添加超参数信息
    if hyperparams is not None:
        csv_data.append(["learning_rate", hyperparams.learning_rate])
        csv_data.append(["lora_r", hyperparams.lora_r])
        csv_data.append(["lora_alpha", hyperparams.lora_alpha])
    
    # 添加特征数据
    for feature_name, feature_value in features.items():
        csv_data.append([feature_name, feature_value])
    
    # 写入CSV文件
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Feature_Name", "Feature_Value"])  # 表头
            writer.writerows(csv_data)
        
        logger.info(f"特征已保存到CSV文件: {csv_filename}")
        logger.info(f"总共保存了 {len(csv_data)} 个特征")
        
        return csv_filename
        
    except Exception as e:
        logger.error(f"保存CSV文件时发生错误: {str(e)}")
        raise

def extract_and_save_features(
    dataset: List[Dict],
    hyperparams: HyperParams,
    base_model_name: str,
    static_sample_size: int = 100,
    dynamic_probe_steps: int = 100,
    dynamic_sample_size: int = 50,
    use_model_sharing: bool = True,
    csv_filename: Optional[str] = None
) -> tuple[Dict[str, float], str]:
    """
    提取特征并保存到CSV文件的便捷函数
    
    Returns:
        (features_dict, csv_file_path): 特征字典和CSV文件路径
    """
    # 生成文件名
    if csv_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"dataset_features_{timestamp}.csv"
    if not csv_filename.endswith('.csv'):
        csv_filename += '.csv'
    
    features = extract_all_features(
        dataset=dataset,
        hyperparams=hyperparams,
        base_model_name=base_model_name,
        static_sample_size=static_sample_size,
        dynamic_probe_steps=dynamic_probe_steps,
        dynamic_sample_size=dynamic_sample_size,
        save_to_csv=True,
        csv_filename=csv_filename
    )
    
    return features, csv_filename 