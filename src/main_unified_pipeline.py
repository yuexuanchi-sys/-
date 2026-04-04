# 兼容性导入：实际实现在项目根目录 main_kggen_pipeline.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main_kggen_pipeline import *  # noqa: F401,F403
from main_kggen_pipeline import KGGenMathKnowledgePipeline  # noqa: F401
