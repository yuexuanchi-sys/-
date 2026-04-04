import os
import json
import re
from typing import List, Dict, Tuple, Set
from tqdm import tqdm
from data_loader import MathDataLoader
from entity_extractor_kggen import KGGenEnhancedEntityExtractor
from relation_extractor_kggen import KGGenEnhancedRelationExtractor
from neo4j_manager import Neo4jManager
from config import Config
from performance_optimizer import PerformanceMonitor, time_it
import torch
from kggen_client import KGGenClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# 关系名称统一映射表
# 规则提取器用英文键名，KGGen 返回中文 → 全部映射为中文标准名
# ============================================================
RELATION_NORMALIZE_MAP = {
    # 英文 → 中文标准名
    'BELONGS_TO': '属于',
    'PREREQUISITE': '前置知识',
    'RELATED': '相关概念',
    'DERIVES': '推导关系',
    'APPLIES': '应用',
    'CONTAINS': '包含',
    # KGGen 可能返回的近义词 → 收敛为标准名
    '包括': '包含',
    '含有': '包含',
    '分为': '包含',
    '定义为': '定义',
    '描述': '定义',
    '计算法则': '计算法则',
    '性质': '性质',
    '前提条件': '前置知识',
    '符号表示': '符号表示',
    '等于': '等于',
    '应用于': '应用',
    '相关属性': '相关概念',
}


def normalize_relation(relation_name: str) -> str:
    """将各种来源的关系名归一到统一的中文名称"""
    return RELATION_NORMALIZE_MAP.get(relation_name, relation_name)


class KGGenEnhancedKnowledgeGraphBuilder:
    """KGGen增强的知识图谱构建器 (修复版)"""

    # ── 文本分块参数 ──────────────────────────────
    CHUNK_SIZE = 400          # 每块最大字符数 (中文 1 字 ≈ 1-2 token)
    CHUNK_OVERLAP = 60        # 前后重叠字符数，防止实体/关系被切断

    def __init__(self, device: str = None, use_kggen: bool = True):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.data_loader = MathDataLoader()
        # 创建实体提取器时 **禁用** 其内部的 KGGen 调用，
        # 由 builder 统一调度 KGGen，避免重复请求
        self.entity_extractor = KGGenEnhancedEntityExtractor(use_kggen=False)
        self.relation_extractor = KGGenEnhancedRelationExtractor()
        self.neo4j_manager = Neo4jManager()
        self.device = device
        self.use_kggen = use_kggen

        self.kggen_client = None
        if use_kggen:
            try:
                self.kggen_client = KGGenClient()
                logger.info("[OK] KGGen客户端初始化成功")
            except Exception as e:
                logger.warning(f"KGGen客户端初始化失败: {e}")
                self.use_kggen = False

        logger.info(f"运行设备: {device}")

    # ================================================================
    #  公共入口
    # ================================================================

    def build_knowledge_graph(self) -> Tuple[List[Dict], List[Dict]]:
        """构建完整的知识图谱（混合模式）"""
        monitor = PerformanceMonitor()
        monitor.start_timer('total_build_process')

        logger.info("开始构建数学知识图谱...")

        # 1. 加载数据
        logger.info("步骤1: 加载数学教材数据")
        monitor.start_timer('data_loading')
        data = self.data_loader.load_all_data()
        monitor.end_timer('data_loading')
        logger.info(f"成功加载 {len(data)} 个文件")

        # 2. 逐文件 → 分块 → 提取
        logger.info("步骤2: 分块 + 实体识别 + 关系抽取")
        all_entities: List[Dict] = []
        all_relations: List[Dict] = []

        monitor.start_timer('entity_relation_extraction')
        for item in tqdm(data, desc="处理文件"):
            try:
                raw_text = item['content']
                grade = item.get('grade')
                chapter = item.get('chapter')
                file_name = item.get('file_name')

                # ★ 修复1: 先预处理文本
                text = self.data_loader.preprocess_text(raw_text)
                if len(text.strip()) < 10:
                    continue

                # ★ 修复2: 按滑动窗口分块，而不是把整篇丢给 BERT
                chunks = self._split_into_chunks(text)
                logger.info(f"  {file_name}: {len(text)} 字 → {len(chunks)} 块")

                for chunk in chunks:
                    result = self._extract_from_chunk(chunk)

                    # 打上文件级元数据
                    for entity in result['entities']:
                        entity.update(grade=grade, chapter=chapter,
                                      source_file=file_name)
                    for rel in result['relations']:
                        rel.update(grade=grade, chapter=chapter,
                                   source_file=file_name)

                    all_entities.extend(result['entities'])
                    all_relations.extend(result['relations'])

            except Exception as e:
                logger.error(f"处理文件 {item.get('file_name')} 失败: {e}")
                continue

        monitor.end_timer('entity_relation_extraction')
        logger.info(f"原始提取: {len(all_entities)} 个实体, {len(all_relations)} 个关系")

        # 3. 关系名归一化
        logger.info("步骤3: 关系名称归一化")
        for rel in all_relations:
            rel['relation'] = normalize_relation(rel['relation'])

        # 4. 后处理
        logger.info("步骤4: 实体/关系后处理")
        monitor.start_timer('postprocessing')
        processed_entities = self._postprocess_entities(all_entities)
        processed_relations = self._postprocess_relations(
            all_relations, {e['text'] for e in processed_entities}
        )
        monitor.end_timer('postprocessing')

        # 5. 去重
        logger.info("步骤5: 数据去重")
        monitor.start_timer('deduplication')
        unique_entities = self._deduplicate_entities(processed_entities)
        unique_relations = self._deduplicate_relations(processed_relations)
        monitor.end_timer('deduplication')
        logger.info(f"去重后: {len(unique_entities)} 个实体, {len(unique_relations)} 个关系")

        # 6. 构建层级结构
        logger.info("步骤6: 构建教材层级结构")
        monitor.start_timer('hierarchy_building')
        self._add_hierarchical_relations(unique_entities, unique_relations, data)
        monitor.end_timer('hierarchy_building')

        # 7. 导入 Neo4j
        logger.info("步骤7: 导入Neo4j图数据库")
        monitor.start_timer('neo4j_import')
        self.neo4j_manager.import_knowledge_graph(unique_entities, unique_relations)
        monitor.end_timer('neo4j_import')

        # 8. 保存
        logger.info("步骤8: 保存处理结果")
        monitor.start_timer('saving_results')
        self._save_results(unique_entities, unique_relations)
        monitor.end_timer('saving_results')

        monitor.end_timer('total_build_process')
        monitor.print_stats()
        logger.info("知识图谱构建完成！")

        return unique_entities, unique_relations

    def build_knowledge_graph_with_kggen_direct(self) -> Tuple[List[Dict], List[Dict]]:
        """使用KGGen直接模式构建知识图谱（跳过本地模型）"""
        if not self.use_kggen or not self.kggen_client:
            raise RuntimeError("KGGen 客户端不可用，无法使用直接模式")

        logger.info("使用KGGen直接模式构建知识图谱...")
        data = self.data_loader.load_all_data()
        all_entities, all_relations = [], []

        for item in tqdm(data, desc="KGGen直接处理"):
            try:
                text = self.data_loader.preprocess_text(item['content'])
                chunks = self._split_into_chunks(text)
                grade = item.get('grade')
                chapter = item.get('chapter')
                file_name = item.get('file_name')

                for chunk in chunks:
                    result = self.kggen_client.extract_entities_and_relations(chunk)
                    for e in result.get('entities', []):
                        e.update(source='kggen_direct', grade=grade,
                                 chapter=chapter, source_file=file_name)
                    for r in result.get('relations', []):
                        r.update(source='kggen_direct', grade=grade,
                                 chapter=chapter, source_file=file_name)
                    all_entities.extend(result.get('entities', []))
                    all_relations.extend(result.get('relations', []))
            except Exception as e:
                logger.error(f"KGGen直接处理失败: {e}")
                continue

        # 归一化 + 后处理 + 去重
        for rel in all_relations:
            rel['relation'] = normalize_relation(rel['relation'])
        processed_entities = self._postprocess_entities(all_entities)
        processed_relations = self._postprocess_relations(
            all_relations, {e['text'] for e in processed_entities}
        )
        unique_entities = self._deduplicate_entities(processed_entities)
        unique_relations = self._deduplicate_relations(processed_relations)

        self.neo4j_manager.import_knowledge_graph(unique_entities, unique_relations)
        self._save_results(unique_entities, unique_relations)
        return unique_entities, unique_relations

    # ================================================================
    #  核心: 单块提取 (修复重复调用 + 混合关系)
    # ================================================================

    def _extract_from_chunk(self, chunk: str) -> Dict:
        """
        对单个文本块执行提取。

        修复：
        - KGGen 只调用一次 (而不是实体提取器 + builder 各调一次)
        - 本地关系提取器始终被调用 (之前 KGGen 可用时被跳过)
        - 两个来源的关系合并后去重
        """
        # ── 第 1 步: 本地实体提取 (规则 + 词性 + BERT, 不含 KGGen) ──
        local_entities = self.entity_extractor.extract_entities(chunk, use_ensemble=True)

        # ── 第 2 步: KGGen 提取 (仅调用一次) ──
        kggen_entities: List[Dict] = []
        kggen_relations: List[Dict] = []
        if self.use_kggen and self.kggen_client:
            try:
                kggen_result = self.kggen_client.extract_entities_and_relations(chunk)
                kggen_entities = kggen_result.get('entities', [])
                kggen_relations = kggen_result.get('relations', [])
            except Exception as e:
                logger.debug(f"KGGen 调用失败: {e}")

        # ── 第 3 步: 合并实体 ──
        merged_entities = self._merge_entities(local_entities, kggen_entities)

        # ── 第 4 步: 本地关系提取 (始终执行) ──
        local_relations = self.relation_extractor.extract_relations(
            chunk, merged_entities
        )

        # ── 第 5 步: 合并关系 ──
        # 验证 KGGen 关系 (subject/object 必须在实体集合中, 允许模糊匹配)
        entity_texts = {e['text'] for e in merged_entities}
        validated_kggen_rels = []
        for rel in kggen_relations:
            subj_match = self._fuzzy_match_entity(rel.get('subject', ''), entity_texts)
            obj_match = self._fuzzy_match_entity(rel.get('object', ''), entity_texts)
            if subj_match and obj_match:
                validated_kggen_rels.append({
                    'subject': subj_match,
                    'relation': rel.get('relation', '相关概念'),
                    'object': obj_match,
                    'source': 'kggen',
                    'confidence': rel.get('confidence', 0.9),
                })

        all_relations = local_relations + validated_kggen_rels

        return {'entities': merged_entities, 'relations': all_relations}

    # ================================================================
    #  文本分块 (滑动窗口)
    # ================================================================

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        滑动窗口分块。优先在句号/问号/分号处断开，
        避免把一个完整的定义切成两半。
        """
        if len(text) <= self.CHUNK_SIZE:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.CHUNK_SIZE, len(text))

            # 如果不是末尾，尝试在句子边界处截断
            if end < len(text):
                # 在 [start + CHUNK_SIZE*0.6, end] 区间内寻找最后一个句号
                search_start = start + int(self.CHUNK_SIZE * 0.6)
                best_break = -1
                for sep in '。！？；\n':
                    pos = text.rfind(sep, search_start, end)
                    if pos > best_break:
                        best_break = pos
                if best_break > search_start:
                    end = best_break + 1  # 包含句号本身

            chunk = text[start:end].strip()
            if len(chunk) >= 10:  # 过滤过短的块
                chunks.append(chunk)

            # 下一块起点 = 当前结束点 - 重叠区
            start = max(start + 1, end - self.CHUNK_OVERLAP)

        return chunks

    # ================================================================
    #  实体合并
    # ================================================================

    def _merge_entities(self, local_entities: List[Dict],
                        kggen_entities: List[Dict]) -> List[Dict]:
        """合并本地和 KGGen 实体，以本地为主、KGGen 补充"""
        local_texts = {e['text'] for e in local_entities}
        merged = list(local_entities)

        for ke in kggen_entities:
            text = ke.get('text', '')
            if not text or len(text) < 2:
                continue
            # 只添加本地没有的实体
            if text not in local_texts:
                ke.setdefault('source', 'kggen')
                ke.setdefault('confidence', 0.85)
                merged.append(ke)
                local_texts.add(text)

        return merged

    def _fuzzy_match_entity(self, name: str, entity_texts: Set[str]) -> str:
        """模糊匹配实体名称: 精确匹配 > 包含匹配 > None"""
        if not name:
            return None
        name = name.strip()
        # 精确匹配
        if name in entity_texts:
            return name
        # 包含匹配 (KGGen 返回的名称可能多/少几个字)
        for et in entity_texts:
            if name in et or et in name:
                return et
        return None

    # ================================================================
    #  后处理
    # ================================================================

    def _postprocess_entities(self, entities: List[Dict]) -> List[Dict]:
        """实体后处理: 过滤噪声"""
        processed = []
        for e in entities:
            text = e.get('text', '').strip()
            if len(text) < 2:
                continue
            # 纯标点 / 纯数字 → 跳过
            if all(c in '，。；：！？（）《》「」""\'\'0123456789 \t\n' for c in text):
                continue
            # 单个英文字母变量 (a, b, x, y) → 跳过
            if re.fullmatch(r'[A-Za-z]{1,2}', text):
                continue

            e['text'] = text
            # 如果类型无效，尝试重新分类
            if not self._validate_entity_type(text, e.get('type', '')):
                new_type = self._reclassify_entity(text)
                if new_type:
                    e['type'] = new_type
                else:
                    continue

            processed.append(e)
        return processed

    def _postprocess_relations(self, relations: List[Dict],
                               entity_texts: Set[str]) -> List[Dict]:
        """关系后处理: 过滤无效关系"""
        processed = []
        for rel in relations:
            subj = rel.get('subject', '').strip()
            obj = rel.get('object', '').strip()
            # 自环 / 空值 → 跳过
            if not subj or not obj or subj == obj:
                continue
            # subject 和 object 必须在实体集合中
            subj_match = self._fuzzy_match_entity(subj, entity_texts)
            obj_match = self._fuzzy_match_entity(obj, entity_texts)
            if not subj_match or not obj_match:
                continue
            rel['subject'] = subj_match
            rel['object'] = obj_match
            processed.append(rel)
        return processed

    def _validate_entity_type(self, text: str, entity_type: str) -> bool:
        if entity_type == 'FORMULA':
            return (any(c.isalpha() for c in text) or
                    any(c in '²³⁰⁺⁻⁼∫∑∏√∞πθφΔ∇∂' for c in text))
        if entity_type == 'THEOREM':
            return any(kw in text for kw in ['定理', '公式', '法则', '原理', '定律'])
        return True

    def _reclassify_entity(self, text: str) -> str:
        if any(c in '²³⁰⁺⁻⁼∫∑∏√∞πθφΔ∇∂' for c in text):
            return 'FORMULA'
        if any(kw in text for kw in ['定理', '公式', '法则', '原理', '定律']):
            return 'THEOREM'
        if any(kw in text for kw in ['方法', '解法', '算法', '步骤']):
            return 'METHOD'
        if len(text) >= 2:
            return 'CONCEPT'
        return None

    # ================================================================
    #  去重
    # ================================================================

    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """实体去重: 同名同类型合并，保留最高置信度"""
        best: Dict[Tuple, Dict] = {}
        for e in entities:
            key = (e['text'], e['type'])
            if key not in best:
                best[key] = e
            else:
                # 保留置信度更高的那条
                if e.get('confidence', 0) > best[key].get('confidence', 0):
                    best[key] = e
        return list(best.values())

    def _deduplicate_relations(self, relations: List[Dict]) -> List[Dict]:
        """关系去重: 同一三元组只保留置信度最高的"""
        best: Dict[Tuple, Dict] = {}
        for rel in relations:
            key = (rel['subject'], rel['relation'], rel['object'])
            if key not in best:
                best[key] = rel
            else:
                if rel.get('confidence', 0) > best[key].get('confidence', 0):
                    best[key] = rel
        return list(best.values())

    # ================================================================
    #  层级关系构建 (修复: 基于文件元数据而非实体类型)
    # ================================================================

    def _add_hierarchical_relations(self, entities: List[Dict],
                                     relations: List[Dict],
                                     file_data: List[Dict]):
        """
        基于教材的 grade/chapter 元数据构建层级结构。

        修复: 之前依赖 CHAPTER_TITLE 类型实体 (没有任何提取器生成),
              现在直接从文件元数据推导层级。

        生成结构:
          初中数学 -[包含]→ 第N章
          第N章    -[包含]→ 各知识点实体
        """
        # 收集所有出现过的 (grade, chapter) 组合
        chapters_seen: Dict[Tuple, str] = {}
        grades_seen: Dict[int, str] = {}

        for item in file_data:
            grade = item.get('grade')
            chapter = item.get('chapter')
            file_name = item.get('file_name', '')
            if grade is not None:
                grade_name = f"{grade}年级数学"
                grades_seen[grade] = grade_name
            if grade is not None and chapter is not None:
                chapter_name = f"第{chapter}章"
                chapters_seen[(grade, chapter)] = chapter_name

        # 创建年级根节点
        for grade, grade_name in grades_seen.items():
            entities.append({
                'text': grade_name,
                'type': 'MODULE',
                'source': 'hierarchy',
                'confidence': 1.0,
                'grade': grade,
            })

        # 创建章节节点 + 年级→章节关系
        for (grade, chapter), chapter_name in chapters_seen.items():
            grade_name = grades_seen.get(grade, f"{grade}年级数学")
            entities.append({
                'text': chapter_name,
                'type': 'CHAPTER',
                'source': 'hierarchy',
                'confidence': 1.0,
                'grade': grade,
                'chapter': chapter,
            })
            relations.append({
                'subject': grade_name,
                'relation': '包含',
                'object': chapter_name,
                'source': 'hierarchy',
                'confidence': 1.0,
            })

        # 将知识点实体挂到对应章节下
        for entity in entities:
            if entity.get('source') == 'hierarchy':
                continue  # 跳过刚创建的层级节点
            grade = entity.get('grade')
            chapter = entity.get('chapter')
            chapter_key = (grade, chapter)
            if chapter_key in chapters_seen:
                relations.append({
                    'subject': chapters_seen[chapter_key],
                    'relation': '包含',
                    'object': entity['text'],
                    'source': 'hierarchy',
                    'confidence': 1.0,
                })

    # ================================================================
    #  保存结果
    # ================================================================

    def _save_results(self, entities: List[Dict], relations: List[Dict]):
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        entities_path = os.path.join(Config.OUTPUT_DIR, "kggen_entities.json")
        with open(entities_path, 'w', encoding='utf-8') as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)

        relations_path = os.path.join(Config.OUTPUT_DIR, "kggen_relations.json")
        with open(relations_path, 'w', encoding='utf-8') as f:
            json.dump(relations, f, ensure_ascii=False, indent=2)

        # 统计信息
        stats = {
            'total_entities': len(entities),
            'total_relations': len(relations),
            'entity_types': {},
            'relation_types': {},
            'sources': {'entities': {}, 'relations': {}},
        }
        for e in entities:
            t = e.get('type', 'unknown')
            stats['entity_types'][t] = stats['entity_types'].get(t, 0) + 1
            s = e.get('source', 'unknown')
            stats['sources']['entities'][s] = stats['sources']['entities'].get(s, 0) + 1
        for r in relations:
            t = r.get('relation', 'unknown')
            stats['relation_types'][t] = stats['relation_types'].get(t, 0) + 1
            s = r.get('source', 'unknown')
            stats['sources']['relations'][s] = stats['sources']['relations'].get(s, 0) + 1

        stats_path = os.path.join(Config.OUTPUT_DIR, "kggen_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"结果已保存到 {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    builder = KGGenEnhancedKnowledgeGraphBuilder()
    try:
        entities, relations = builder.build_knowledge_graph()
        print(f"\n构建完成: {len(entities)} 个实体, {len(relations)} 个关系")
    except Exception as e:
        print(f"构建失败: {e}")
        import traceback
        traceback.print_exc()
