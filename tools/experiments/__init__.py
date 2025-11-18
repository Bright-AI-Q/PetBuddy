"""
实验工具模块 - 包含A/B测试实验运行和分析功能
"""

from pathlib import Path

# 模块版本
__version__ = "0.1.0"

# 导出主要功能
try:
    from .run_ab_experiments import ExperimentRunner, main as run_experiments_main
except ImportError:
    pass

# 模块说明
__all__ = ['ExperimentRunner', 'run_experiments_main']

def get_module_info():
    """获取模块信息"""
    return {
        "name": "PetBuddy Experiments Module",
        "version": __version__,
        "description": "A/B测试实验运行和管理工具",
        "author": "PetBuddy Team",
        "module_path": str(Path(__file__).parent)
    }