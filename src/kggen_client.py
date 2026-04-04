# 兼容性导入：实际实现在项目根目录
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from kggen_client import *  # noqa: F401,F403
from kggen_client import KGGenClient  # noqa: F401
