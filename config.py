# 配置文件
import os

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class Config:
    # 数据路径配置（使用环境变量，跨平台兼容）
    DATA_DIR = os.getenv('DATA_DIR', os.path.join(os.path.dirname(__file__), 'data'))
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', './output')
    MODEL_DIR = os.getenv('MODEL_DIR', './models')

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

    # 模型配置（使用环境变量，支持跨平台路径）
    BERT_MODEL = os.getenv('BERT_MODEL_PATH', 'bert-base-chinese')
    MAX_SEQ_LENGTH = 512
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 3

    # Neo4j配置（从环境变量读取，不再硬编码密码）
    NEO4J_URI = os.getenv('NEO4J_URI', 'neo4j://localhost:7687')
    NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', '')

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

    # KGGen配置（从环境变量读取，不再硬编码密钥）
    KGGen_API_KEY = os.getenv('KGGEN_API_KEY', '')
    KGGen_MODEL = os.getenv('KGGEN_MODEL', 'doubao-seed-1-6-250615')
    KGGen_MAX_TOKENS = 4000
    KGGen_TEMPERATURE = 0.1
    KGGen_ENTITY_TYPES = ["CONCEPT", "FORMULA", "THEOREM", "METHOD", "EXAMPLE"]
    KGGen_RELATION_TYPES = ["BELONGS_TO", "PREREQUISITE", "RELATED", "DERIVES", "APPLIES"]

    KGGEN_API_URL = os.getenv('KGGEN_API_URL', 'https://ark.cn-beijing.volces.com/api/v3/chat/completions')
    KGGEN_TIMEOUT = int(os.getenv('KGGEN_TIMEOUT', '60'))
    KGGEN_MAX_RETRIES = int(os.getenv('KGGEN_MAX_RETRIES', '5'))

    # LTP 模型路径配置
    LTP_MODEL_PATH = os.getenv('LTP_MODEL_PATH', '')

    @staticmethod
    def ensure_dirs():
        """确保必要的目录存在"""
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.MODEL_DIR, exist_ok=True)

    @staticmethod
    def validate_kggen_config():
        """验证KGGen配置"""
        if not Config.KGGen_API_KEY:
            print("警告: KGGEN_API_KEY 未设置，KGGen功能可能无法使用")
            print("请在 .env 文件中设置: KGGEN_API_KEY=your_actual_api_key")
        return bool(Config.KGGen_API_KEY)

    @staticmethod
    def validate_neo4j_config():
        """验证Neo4j配置"""
        if not Config.NEO4J_PASSWORD:
            print("警告: NEO4J_PASSWORD 未设置")
            print("请在 .env 文件中设置: NEO4J_PASSWORD=your_password")
        return bool(Config.NEO4J_PASSWORD)

# 初始化目录
Config.ensure_dirs()
# 验证配置
Config.validate_kggen_config()
