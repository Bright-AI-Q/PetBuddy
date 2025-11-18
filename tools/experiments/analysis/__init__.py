"""
分析工具模块 - 包含实验结果分析和可视化功能
"""

from pathlib import Path

# 模块版本
__version__ = "0.1.0"

# 导出主要功能
try:
    from .analyze_ablation_results import AblationAnalyzer, main as analyze_main
except ImportError:
    pass

# 模块说明
__all__ = ['AblationAnalyzer', 'analyze_main']

def get_module_info():
    """获取模块信息"""
    return {
        "name": "PetBuddy Analysis Module",
        "version": __version__,
        "description": "实验结果分析和可视化工具",
        "author": "PetBuddy Team",
        "module_path": str(Path(__file__).parent)
    }