"""
KGGen增强的关系提取器

提取方法:
1. 基于规则的模板匹配 (40+ 数学领域模式)
2. 基于共现的关系推断 (同句/近邻共现)
3. 基于模型的关系分类 (BERT, 可选)
4. KGGen API 关系提取 (可选)
"""

import os
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import re
import logging
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KGGenEnhancedRelationExtractor:
    """KGGen增强的关系提取器"""

    def __init__(self, use_kggen: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 关系类型映射
        self.relation2id = {k: i for i, k in enumerate(Config.RELATION_TYPES.keys())}
        self.id2relation = {v: k for k, v in self.relation2id.items()}

        # KGGen客户端 (默认禁用, 由 builder 统一调度)
        self.kggen_client = None
        if use_kggen:
            try:
                from kggen_client import KGGenClient
                self.kggen_client = KGGenClient()
                logger.info("[OK] KGGen关系提取客户端初始化成功")
            except Exception as e:
                logger.warning(f"KGGen客户端初始化失败: {e}")

        # BERT关系分类模型 (仅当有训练好的权重时才加载)
        self.model_loaded = False
        self.tokenizer = None
        self.model = None
        rel_model_path = os.path.join(Config.MODEL_DIR, "relation_model", "pytorch_model.bin")
        if os.path.exists(rel_model_path):
            try:
                self.tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL)
                self.model = BERTRelationExtractor(len(self.relation2id))
                self.model.load_state_dict(torch.load(rel_model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.model_loaded = True
                logger.info("[OK] 关系抽取BERT模型加载成功 (已训练权重)")
            except Exception as e:
                logger.warning(f"关系抽取BERT模型加载失败: {e}")
        else:
            logger.info("[INFO] 关系抽取BERT模型未训练，使用规则+KGGen方法")

        # ============================================================
        # 规则模板 — 覆盖初中数学教材的常见表达
        # ============================================================
        self.rule_patterns = {
            'BELONGS_TO': [
                r'([^，。；]+?)属于([^，。；]+)',
                r'([^，。；]+?)是([^，。；]+?)的一种',
                r'([^，。；]+?)是([^，。；]+?)的特例',
                r'([^，。；]+?)是特殊的([^，。；]+)',
                r'([^，。；]+?)包含在([^，。；]+?)中',
            ],
            'PREREQUISITE': [
                r'学习([^，。；]+?)之前.*?掌握([^，。；]+)',
                r'([^，。；]+?)是学习([^，。；]+?)的基础',
                r'([^，。；]+?)以([^，。；]+?)为基础',
                r'([^，。；]+?)依赖于([^，。；]+)',
                r'由([^，。；]+?)引入([^，。；]+)',
                r'从([^，。；]+?)推广到([^，。；]+)',
                r'([^，。；]+?)建立在([^，。；]+?)基础上',
            ],
            'DERIVES': [
                r'由([^，。；]+?)可[以得]([^，。；]+)',
                r'根据([^，。；]+?)[,，].*?得到([^，。；]+)',
                r'([^，。；]+?)推导出([^，。；]+)',
                r'若([^，。；]+?)[则就]([^，。；]+)',
                r'([^，。；]+?)可以[推导]出([^，。；]+)',
                r'利用([^，。；]+?)可以[求计]算([^，。；]+)',
                r'([^，。；]+?)等价于([^，。；]+)',
            ],
            'RELATED': [
                r'([^，。；]+?)与([^，。；]+?)的关系',
                r'([^，。；]+?)和([^，。；]+?)[有存]在.*?关系',
                r'([^，。；]+?)类似于([^，。；]+)',
                r'([^，。；]+?)对应([^，。；]+)',
                r'([^，。；]+?)互为([^，。；]+)',
                r'([^，。；]+?)的逆运算是([^，。；]+)',
                r'([^，。；]+?)的逆.*?是([^，。；]+)',
            ],
            'APPLIES': [
                r'([^，。；]+?)应用于([^，。；]+)',
                r'用([^，。；]+?)解决([^，。；]+)',
                r'([^，。；]+?)可以解决([^，。；]+)',
                r'([^，。；]+?)用于([^，。；]+)',
                r'利用([^，。；]+?)[解求]([^，。；]+)',
                r'运用([^，。；]+?)[解求]([^，。；]+)',
            ],
            'CONTAINS': [
                r'([^，。；]+?)包含([^，。；]+)',
                r'([^，。；]+?)包括([^，。；]+)',
                r'([^，。；]+?)由([^，。；]+?)组成',
                r'([^，。；]+?)分为([^，。；]+)',
                r'([^，。；]+?)有.*?([^，。；]+?)等',
            ],
        }

        # ============================================================
        # 数学领域的定义/性质模式 (补充关系)
        # ============================================================
        self.definition_patterns = [
            # "XXX叫做YYY" / "XXX称为YYY"
            (r'([^，。；]+?)叫做([^，。；]+)', '定义'),
            (r'([^，。；]+?)称为([^，。；]+)', '定义'),
            (r'([^，。；]+?)就是([^，。；]+)', '定义'),
            (r'([^，。；]+?)简称([^，。；]+)', '定义'),
            # 性质
            (r'([^，。；]+?)的性质[是为：:]([^，。；]+)', '性质'),
            (r'([^，。；]+?)具有([^，。；]+?)性', '性质'),
            # 计算
            (r'([^，。；]+?)[的之]计算[法公]则[是为：:]([^，。；]+)', '计算法则'),
            (r'([^，。；]+?)的[计运]算[法规]则', '计算法则'),
            # 符号
            (r'([^，。；]+?)用([^，。；]+?)表示', '符号表示'),
            (r'([^，。；]+?)记[作为]([^，。；]+)', '符号表示'),
        ]

    # ================================================================
    #  公共接口
    # ================================================================

    def extract_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """提取实体之间的关系"""
        if not entities or len(entities) < 2:
            return []

        relations = []

        # 1. 规则模板匹配
        relations.extend(self._rule_based_extraction(text, entities))

        # 2. 定义/性质模式
        relations.extend(self._definition_pattern_extraction(text, entities))

        # 3. 共现关系推断
        relations.extend(self._cooccurrence_extraction(text, entities))

        # 4. BERT模型 (如果可用)
        if self.model_loaded and len(entities) >= 2:
            try:
                relations.extend(self._model_based_extraction(text, entities))
            except Exception as e:
                logger.debug(f"模型关系抽取失败: {e}")

        # 5. KGGen API (如果可用)
        if self.kggen_client:
            try:
                relations.extend(self._kggen_extraction(text, entities))
            except Exception as e:
                logger.debug(f"KGGen关系抽取失败: {e}")

        return relations

    # ================================================================
    #  规则模板匹配
    # ================================================================

    def _rule_based_extraction(self, text: str, entities: List[Dict]) -> List[Dict]:
        """基于规则模板的关系抽取"""
        relations = []
        entity_texts = [e['text'] for e in entities]

        for rel_type, patterns in self.rule_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    if match.groups() and len(match.groups()) >= 2:
                        e1_text = match.group(1).strip()
                        e2_text = match.group(2).strip()

                        e1 = self._find_best_match(e1_text, entities)
                        e2 = self._find_best_match(e2_text, entities)

                        if e1 and e2 and e1['text'] != e2['text']:
                            relations.append({
                                'subject': e1['text'],
                                'object': e2['text'],
                                'relation': rel_type,
                                'source': 'rule',
                                'confidence': 0.9,
                            })
        return relations

    def _definition_pattern_extraction(self, text: str, entities: List[Dict]) -> List[Dict]:
        """定义/性质模式提取"""
        relations = []
        for pattern, rel_name in self.definition_patterns:
            for match in re.finditer(pattern, text):
                if match.groups() and len(match.groups()) >= 2:
                    e1_text = match.group(1).strip()
                    e2_text = match.group(2).strip()

                    e1 = self._find_best_match(e1_text, entities)
                    e2 = self._find_best_match(e2_text, entities)

                    if e1 and e2 and e1['text'] != e2['text']:
                        relations.append({
                            'subject': e1['text'],
                            'object': e2['text'],
                            'relation': rel_name,
                            'source': 'rule',
                            'confidence': 0.85,
                        })
        return relations

    # ================================================================
    #  共现关系推断
    # ================================================================

    def _cooccurrence_extraction(self, text: str, entities: List[Dict]) -> List[Dict]:
        """
        基于共现的关系推断 (保守策略):
        仅对 CONCEPT/THEOREM/METHOD 类型实体建立共现关系,
        且要求上下文中存在明确的关系线索词。
        FORMULA 类型不参与共现 (噪声太多)。
        """
        relations = []
        # 只对非FORMULA实体做共现
        valid_types = {'CONCEPT', 'THEOREM', 'METHOD', 'PROPERTY'}
        concept_entities = [e for e in entities if e.get('type') in valid_types]

        if len(concept_entities) < 2:
            return relations

        sentences = re.split(r'[。！？；\n]', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            sentence_entities = []
            for e in concept_entities:
                pos = sentence.find(e['text'])
                if pos >= 0:
                    sentence_entities.append((e, pos))

            # 限制每句最多10个实体对
            if len(sentence_entities) > 10:
                sentence_entities = sentence_entities[:10]

            for i, (e1, pos1) in enumerate(sentence_entities):
                for j, (e2, pos2) in enumerate(sentence_entities):
                    if i >= j or e1['text'] == e2['text']:
                        continue
                    distance = abs(pos2 - pos1)
                    if distance > 40:
                        continue

                    if pos1 <= pos2:
                        between_start = pos1 + len(e1['text'])
                        between_end = pos2
                    else:
                        between_start = pos2 + len(e2['text'])
                        between_end = pos1
                    between_text = sentence[between_start:between_end] if between_start < between_end else ''

                    # 必须有明确的关系线索词 (不再默认推断 RELATED)
                    rel_type = self._infer_relation_from_context_strict(between_text, e1, e2)
                    if rel_type:
                        relations.append({
                            'subject': e1['text'],
                            'object': e2['text'],
                            'relation': rel_type,
                            'source': 'cooccurrence',
                            'confidence': 0.7,
                        })

        return relations

    def _infer_relation_from_context_strict(self, between_text: str,
                                             e1: Dict, e2: Dict) -> Optional[str]:
        """严格模式: 只在上下文有明确关系词时推断"""
        bt = between_text.strip()

        if any(kw in bt for kw in ['包含', '包括', '分为', '组成']):
            return 'CONTAINS'
        if any(kw in bt for kw in ['推导', '得到', '可得', '推出', '因此', '所以']):
            return 'DERIVES'
        if any(kw in bt for kw in ['基础', '前提', '先学', '需要', '依赖']):
            return 'PREREQUISITE'
        if any(kw in bt for kw in ['应用', '解决', '求解', '利用']):
            return 'APPLIES'
        if any(kw in bt for kw in ['属于', '是一种', '是特殊的']):
            return 'BELONGS_TO'
        if any(kw in bt for kw in ['叫做', '称为', '就是', '定义为']):
            return 'RELATED'
        if any(kw in bt for kw in ['与', '类似', '对应', '互为', '逆']):
            return 'RELATED'

        # 不做纯类型推断，避免无上下文信号的虚假关系
        return None

    # ================================================================
    #  BERT模型关系分类
    # ================================================================

    def _model_based_extraction(self, text: str, entities: List[Dict]) -> List[Dict]:
        """基于BERT模型的关系抽取 (仅在有训练好的模型时使用)"""
        relations = []

        # 限制实体对数量避免 O(n²) 爆炸
        max_pairs = min(len(entities) * (len(entities) - 1), 200)
        pair_count = 0

        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i >= j or pair_count >= max_pairs:
                    continue
                pair_count += 1

                try:
                    context = self._get_entity_context(text, entity1, entity2)
                    input_text = f"{context} [SEP] {entity1['text']} [SEP] {entity2['text']}"

                    encoded = self.tokenizer(
                        input_text, padding=True, truncation=True,
                        max_length=Config.MAX_SEQ_LENGTH, return_tensors="pt"
                    )

                    tokenized = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
                    entity1_pos = self._find_entity_position(tokenized, entity1['text'])
                    entity2_pos = self._find_entity_position(tokenized, entity2['text'])

                    if entity1_pos is None or entity2_pos is None:
                        continue

                    self.model.eval()
                    with torch.no_grad():
                        logits = self.model(
                            encoded['input_ids'].to(self.device),
                            encoded['attention_mask'].to(self.device),
                            torch.tensor([entity1_pos]).to(self.device),
                            torch.tensor([entity2_pos]).to(self.device),
                        )
                        probs = torch.softmax(logits, dim=-1)
                        confidence, pred_id = torch.max(probs, dim=-1)

                        if confidence.item() > 0.7:
                            rel_type = self.id2relation[pred_id.item()]
                            relations.append({
                                'subject': entity1['text'],
                                'object': entity2['text'],
                                'relation': rel_type,
                                'source': 'model',
                                'confidence': round(confidence.item(), 4),
                            })
                except Exception:
                    continue

        return relations

    # ================================================================
    #  KGGen API 关系提取
    # ================================================================

    def _kggen_extraction(self, text: str, entities: List[Dict]) -> List[Dict]:
        """使用 KGGen API 提取关系"""
        if not self.kggen_client:
            return []
        try:
            result = self.kggen_client.extract_entities_and_relations(text)
            entity_texts = {e['text'] for e in entities}
            valid = []
            for rel in result.get('relations', []):
                if rel['subject'] in entity_texts and rel['object'] in entity_texts:
                    rel['source'] = 'kggen'
                    rel.setdefault('confidence', 0.9)
                    valid.append(rel)
            return valid
        except Exception as e:
            logger.debug(f"KGGen关系提取失败: {e}")
            return []

    # ================================================================
    #  辅助方法
    # ================================================================

    def _find_best_match(self, matched_text: str, entities: List[Dict]) -> Optional[Dict]:
        """找到最佳匹配的实体"""
        matched_text = matched_text.strip()
        if not matched_text or len(matched_text) < 2:
            return None

        # 精确匹配
        for e in entities:
            if e['text'] == matched_text:
                return e

        # 包含匹配
        for e in entities:
            if e['text'] in matched_text or matched_text in e['text']:
                return e

        # Jaccard 相似度
        for e in entities:
            if self._text_similarity(e['text'], matched_text) > 0.6:
                return e

        return None

    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """Jaccard 字符相似度"""
        set1 = set(text1)
        set2 = set(text2)
        union = set1 | set2
        if not union:
            return 0.0
        return len(set1 & set2) / len(union)

    def _get_entity_context(self, text: str, e1: Dict, e2: Dict) -> str:
        """获取两个实体周围的上下文"""
        start = min(e1.get('start', 0), e2.get('start', 0))
        end = max(e1.get('end', 0), e2.get('end', 0))
        ctx_start = max(0, start - 80)
        ctx_end = min(len(text), end + 80)
        return text[ctx_start:ctx_end]

    def _find_entity_position(self, tokens: List[str], entity_text: str) -> Optional[int]:
        """在 token 序列中找到实体起始位置"""
        entity_tokens = self.tokenizer.tokenize(entity_text)
        for i in range(len(tokens) - len(entity_tokens) + 1):
            if tokens[i:i + len(entity_tokens)] == entity_tokens:
                return i
        return None


class BERTRelationExtractor(nn.Module):
    """BERT 关系分类模型"""

    def __init__(self, num_relations: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(Config.BERT_MODEL)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size * 3, num_relations)

    def forward(self, input_ids, attention_mask, entity1_pos, entity2_pos):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state
        batch_size = seq_out.size(0)

        entity1_repr = torch.zeros(batch_size, seq_out.size(-1), device=seq_out.device)
        entity2_repr = torch.zeros(batch_size, seq_out.size(-1), device=seq_out.device)

        for i in range(batch_size):
            pos1 = entity1_pos[i].item()
            pos2 = entity2_pos[i].item()
            # 确保位置在序列长度内
            pos1 = min(pos1, seq_out.size(1) - 1)
            pos2 = min(pos2, seq_out.size(1) - 1)
            entity1_repr[i] = seq_out[i, pos1]
            entity2_repr[i] = seq_out[i, pos2]

        cls_repr = seq_out[:, 0, :]
        combined = torch.cat([cls_repr, entity1_repr, entity2_repr], dim=-1)
        combined = self.dropout(combined)
        return self.classifier(combined)


if __name__ == "__main__":
    extractor = KGGenEnhancedRelationExtractor()

    test_text = "勾股定理是直角三角形的重要性质，它推导自平方差公式。学习勾股定理之前需要掌握直角三角形的基本概念。"
    test_entities = [
        {'text': '勾股定理', 'type': 'THEOREM', 'start': 0, 'end': 4},
        {'text': '直角三角形', 'type': 'CONCEPT', 'start': 5, 'end': 10},
        {'text': '平方差公式', 'type': 'FORMULA', 'start': 18, 'end': 23},
    ]

    relations = extractor.extract_relations(test_text, test_entities)
    print("提取到的关系:")
    for rel in relations:
        print(f"  {rel['subject']} --[{rel['relation']}]--> {rel['object']} "
              f"(来源: {rel['source']}, 置信度: {rel['confidence']:.2f})")
