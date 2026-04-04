# 兼容性导入：实际实现在项目根目录
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from neo4j_manager import *  # noqa: F401,F403
from neo4j_manager import Neo4jManager  # noqa: F401
