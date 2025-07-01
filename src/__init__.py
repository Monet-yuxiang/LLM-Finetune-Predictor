"""
数据集特征提取工具包

这个包提供了全面的数据集质量分析功能，包括：
- 数据解析和格式化
- 静态特征提取（文本统计、语义特征等）
- 动态模型探针（学习动力学分析）
- 优化特征提取器（多卡并行、批量处理）
- CSV文件保存功能
"""

from .data_parsers import HyperParams, DatasetAnalyzer
from .static_features import StaticFeatureExtractor
from .dynamic_probes import DynamicProbeAnalyzer
from .feature_dispatcher import extract_all_features, save_features_to_csv, extract_and_save_features
from .optimized_feature_extractor import OptimizedFeatureExtractor

__version__ = "1.0.0"
__author__ = "Dataset Analysis Team"

__all__ = [
    "HyperParams",
    "DatasetAnalyzer", 
    "StaticFeatureExtractor",
    "DynamicProbeAnalyzer",
    "OptimizedFeatureExtractor",
    "extract_all_features",
    "save_features_to_csv",
    "extract_and_save_features"
] 