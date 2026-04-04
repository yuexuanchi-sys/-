import os
import time
import litellm
import logging
from typing import List, Dict, Optional
from config import Config
from kg_gen import KGGen

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KGGenClient:
    """真实接入 kg_gen 库的客户端，专为数学知识图谱定制"""
    
    def __init__(self):
        # 1. 强制覆盖环境变量，解决 LiteLLM 路由 DeepSeek 的底层 BUG
        os.environ["OPENAI_API_KEY"] = Config.KGGen_API_KEY
        os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"
        os.environ["MAX_TOKENS"] = str(Config.KGGen_MAX_TOKENS)

        # 2. 注册模型（骗过 LiteLLM 的校验机制）
        try:
            litellm.register_model({
                "deepseek-chat": {
                    "max_tokens": Config.KGGen_MAX_TOKENS,
                    "litellm_provider": "openai", 
                    "mode": "chat"
                }
            })
        except Exception as e:
            logger.debug(f"模型已注册或注册跳过: {e}")

        # 3. 初始化真正的 KGGen 核心
        self.kg = KGGen(
            model="custom/deepseek-chat", 
            temperature=Config.KGGen_TEMPERATURE
        )
        
        # 4. 固化数学领域的强约束上下文 (上帝指令)
        self.system_context = """
        你是一个严谨的初中数学教育专家。当前输入的是初中数学教材。
        请基于文本抽取核心的数学知识图谱，必须严格遵守以下规则：
        
        【规则 1：实体极简与去变量】
        实体必须是数学概念名词。绝对不要提取具体的点、线、面字母（如 A, B, O, OA, x, y）。
        
        【规则 2：关系（边）必须标准化】
        边的名称必须极度精简，尽量从以下词库中选择或组合：
        [包含, 属于, 定义为, 计算法则, 性质, 前提条件, 符号表示, 等于]
        """
        
        logger.info("KGGenClient (DeepSeek 核心) 初始化成功！")

    def extract_entities_and_relations(self, text: str) -> Dict:
        """
        从文本中提取实体和关系 (对上层 Pipeline 保持接口兼容)
        """
        if not text or not text.strip():
            return {"entities": [], "relations": []}

        try:
            # 核心调用：使用 kg_gen 生成图谱
            graph = self.kg.generate(
                input_data=text,
                context=self.system_context
            )
            
            # 解析并格式化输出，以匹配 Pipeline 的期望格式
            return self._format_graph_to_dict(graph, text)
            
        except Exception as e:
            logger.error(f"KGGen 抽取异常: {e}")
            return {"entities": [], "relations": []}

    def _format_graph_to_dict(self, graph, original_text: str) -> Dict:
        """
        适配器：将 kg_gen 的输出对象转换为 Pipeline 需要的 JSON 结构
        """
        result = {
            "entities": [],
            "relations": []
        }
        
        if not graph or not hasattr(graph, 'relations'):
            return result
            
        # 提取实体并附带位置信息
        for entity_name in graph.entities:
            # 过滤掉只有1-2个字母的垃圾变量 (后处理清洗)
            if len(entity_name) <= 2 and entity_name.encode('utf-8').isalpha():
                continue
                
            start_pos = original_text.find(entity_name)
            if start_pos != -1:
                result["entities"].append({
                    "text": entity_name,
                    "type": "数学概念", # 默认类型，后续可由 BERT 细化
                    "start": start_pos,
                    "end": start_pos + len(entity_name)
                })
            else:
                # 如果文本中找不到完全匹配的字面量（可能是模型归一化了），记录但无坐标
                result["entities"].append({
                    "text": entity_name,
                    "type": "数学概念",
                    "start": 0,
                    "end": 0
                })

        # 提取关系
        for subj, edge, obj in graph.relations:
            if len(subj) <= 2 and subj.encode('utf-8').isalpha(): continue
            if len(obj) <= 2 and obj.encode('utf-8').isalpha(): continue
            
            result["relations"].append({
                "subject": subj,
                "relation": edge,
                "object": obj
            })
            
        return result
    
    def batch_extract(self, texts: List[str], batch_size: int = 5) -> List[Dict]:
        """批量提取实体和关系"""
        results = []
        for i, text in enumerate(texts):
            logger.info(f"KGGen 正在处理第 {i+1}/{len(texts)} 个分块...")
            result = self.extract_entities_and_relations(text)
            results.append(result)
            # 添加小幅延迟，防止触发高频请求限制
            time.sleep(0.5)
        return results

# 测试用例
if __name__ == "__main__":
    # 模拟 Config
    class Config:
        KGGen_API_KEY = "sk-43738d2df5e84ea2b86d43fd53bf044b" # 替换为你的真实 Key
        KGGen_MAX_TOKENS = 8192
        KGGen_TEMPERATURE = 0.0

    client = KGGenClient()
    test_text = "勾股定理指出：在直角三角形中，两直角边的平方和等于斜边的平方。例如线段AB和BC。"
    result = client.extract_entities_and_relations(test_text)
    
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))