"""
Relation Extractor (KGGen) - LLM-powered relation extraction.

Uses the cloud LLM (via kggen_client.py) to identify and classify
relationships between math entities extracted by the BERT module.

Relation types include:
    - 属于 (belongs_to): e.g., 勾股定理 -> 属于 -> 直角三角形
    - 前置知识 (prerequisite): e.g., 一元一次方程 -> 前置知识 -> 二元一次方程
    - 包含 (contains): e.g., 几何 -> 包含 -> 三角形
    - 应用于 (applied_to): e.g., 勾股定理 -> 应用于 -> 距离计算
    - 等价于 (equivalent_to), 推导出 (derives), etc.

Output format:
    [
        {"head": "勾股定理", "relation": "属于", "tail": "直角三角形"},
        ...
    ]
"""

# TODO: Implement relation extraction logic
# Key components needed:
# - Accept entities from entity_recognizer_enhanced.py
# - Format prompt for cloud LLM with entity context
# - Call kggen_client.py to get LLM response
# - Parse and validate extracted relations
# - Deduplicate and normalize relation triples
