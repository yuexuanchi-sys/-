# 配置文件
import os

class Config:
    # 数据路径配置
    DATA_DIR = "D:\\数据"  # 数学教材数据目录
    OUTPUT_DIR = "./output"
    MODEL_DIR = "./models"
    
    # 实体类型定义
    ENTITY_TYPES = {
        "CONCEPT": "数学概念",
        "FORMULA": "数学公式", 
        "THEOREM": "定理定律",
        "EXAMPLE": "例题示例",
        "METHOD": "解题方法",
        "PROPERTY": "性质特征",
        "DEFINITION": "定义",
        "CHAPTER": "章节",
        "MODULE": "模块"
    }
    
    # 关系类型定义
    RELATION_TYPES = {
        "BELONGS_TO": "属于",           # 章节属于模块
        "PREREQUISITE": "前置知识",      # A依赖于B
        "RELATED": "相关概念",          # A与B相关
        "DERIVES": "推导关系",          # A推导自B
        "APPLIES": "应用",              # A应用于B
        "CONTAINS": "包含"              # A包含B
    }
    
    # 模型配置
    BERT_MODEL = "C:\\Users\\xiejiang\\models\\bert-base-chinese"
    MAX_SEQ_LENGTH = 512
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 3
    
    # Neo4j配置
    NEO4J_URI = "neo4j://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "C99VYLXMvEVanXXyv6wSTLxuXfn7IODfBfGK7Dgennk"
    
    # 正则表达式模式
    FORMULA_PATTERNS = [
        r'[a-zA-Zα-ωΑ-Ω]\s*=\s*[^=]+',  # 变量赋值
        r'[a-zA-Zα-ωΑ-Ω]\s*\+\s*[a-zA-Zα-ωΑ-Ω]',  # 变量相加
        r'[a-zA-Zα-ωΑ-Ω]\s*\-\s*[a-zA-Zα-ωΑ-Ω]',  # 变量相减
        r'[a-zA-Zα-ωΑ-Ω]\s*\*\s*[a-zA-Zα-ωΑ-Ω]',  # 变量相乘
        r'[a-zA-Zα-ωΑ-Ω]\s*/\s*[a-zA-Zα-ωΑ-Ω]',   # 变量相除
        r'[a-zA-Zα-ωΑ-Ω]\s*\^\s*\d+',             # 变量幂次
        r'√[a-zA-Zα-ωΑ-Ω\d]+',                    # 平方根
        r'[a-zA-Zα-ωΑ-Ω]\([^)]+\)',               # 函数形式
    ]
    
    THEOREM_PATTERNS = [
        r'[《》]([^《》]+)[定规定律]',
        r'([^，。]+)[定规定律]',
        r'[《》]([^《》]+)公式',
        r'([^，。]+)公式'
    ]
    
    # KGGen配置 - 已切换到Volcengine ARK API
    KGGen_API_KEY = "b05528ee-73ba-4d5f-b15e-937b5993b53b"  # 直接硬编码API密钥
    KGGen_MODEL = "doubao-seed-1-6-250615"  # 模型选择，添加volcengine提供商前缀
    KGGen_MAX_TOKENS = 4000  # 最大token数
    KGGen_TEMPERATURE = 0.1  # 温度参数
    KGGen_ENTITY_TYPES = ["CONCEPT", "FORMULA", "THEOREM", "METHOD", "EXAMPLE"]  # 实体类型
    KGGen_RELATION_TYPES = ["BELONGS_TO", "PREREQUISITE", "RELATED", "DERIVES", "APPLIES"]  # 关系类型
    
    KGGEN_API_URL = os.getenv('KGGEN_API_URL', 'https://ark.cn-beijing.volces.com/api/v3/chat/completions')  # KGGen API完整端点URL
    KGGEN_TIMEOUT = int(os.getenv('KGGEN_TIMEOUT', '60'))  # 请求超时时间（秒），增加到60秒
    KGGEN_MAX_RETRIES = int(os.getenv('KGGEN_MAX_RETRIES', '5'))  # 最大重试次数，增加到5次

    # LTP 模型路径配置
    LTP_MODEL_PATH = os.getenv('LTP_MODEL_PATH', r"C:\Users\xiejiang\Desktop\knowledge_graph_project\small-main")

    @staticmethod
    def ensure_dirs():
        """确保必要的目录存在"""
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.MODEL_DIR, exist_ok=True)

    @staticmethod
    def validate_kggen_config():
        """验证KGGen配置"""
        if not Config.KGGen_API_KEY or Config.KGGen_API_KEY == "your_api_key_here":
            print("警告: KGGen_API_KEY 未设置，KGGen功能可能无法使用")
            print("请设置环境变量: export KGGEN_API_KEY=your_actual_api_key")
        return bool(Config.KGGen_API_KEY and Config.KGGen_API_KEY != "your_api_key_here")

# 初始化目录
Config.ensure_dirs()
# 验证KGGen配置
Config.validate_kggen_config()