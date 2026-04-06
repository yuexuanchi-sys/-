
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast
from TorchCRF import CRF
import re
import os
import json
from typing import List, Dict, Tuple, Optional, Set
from config import Config
import numpy as np
try:
    from seqeval.metrics import classification_report
except ImportError:
    classification_report = None
import jieba
import jieba.posseg as pseg
from collections import defaultdict
import warnings
from kggen_client import KGGenClient
warnings.filterwarnings('ignore')

class KGGenEnhancedEntityExtractor:
    """KGGen增强的实体提取器，集成本地模型和KGGen API"""
    
    def __init__(self, model_dir: str = None, use_pos_tagging: bool = True, use_kggen: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 实体标签定义（支持BIOES）
        self.label2id = {
            'O': 0,
            'B-CONCEPT': 1, 'I-CONCEPT': 2, 'E-CONCEPT': 3, 'S-CONCEPT': 4,
            'B-FORMULA': 5, 'I-FORMULA': 6, 'E-FORMULA': 7, 'S-FORMULA': 8,
            'B-THEOREM': 9, 'I-THEOREM': 10, 'E-THEOREM': 11, 'S-THEOREM': 12,
            'B-METHOD': 13, 'I-METHOD': 14, 'E-METHOD': 15, 'S-METHOD': 16,
            'B-EXAMPLE': 17, 'I-EXAMPLE': 18, 'E-EXAMPLE': 19, 'S-EXAMPLE': 20
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # 词性标注支持
        self.use_pos_tagging = use_pos_tagging
        if use_pos_tagging:
            self._initialize_pos_tagger()
        
        # KGGen客户端
        self.use_kggen = use_kggen
        self.kggen_client = None
        if use_kggen:
            try:
                self.kggen_client = KGGenClient()
                print("[OK] KGGen客户端初始化成功")
            except Exception as e:
                print(f"[WARN] KGGen客户端初始化失败: {e}")
                self.use_kggen = False
        
        # 模型加载
        self.model_loaded = False
        self.tokenizer = None
        self.model = None
        
        if model_dir is None:
            model_dir = os.path.join(Config.MODEL_DIR, "math_ner_advanced_model")
        
        if os.path.exists(model_dir):
            try:
                print("尝试加载高级BERT模型...")
                self._load_trained_model(model_dir)
            except Exception as e:
                print(f"训练模型加载失败: {e}")
                print("将使用基于规则和词性标注的方法进行实体识别")
        else:
            print("训练模型不存在，将使用基于规则和词性标注的方法")
    
    def _initialize_pos_tagger(self):
        """初始化词性标注器"""
        try:
            # 添加数学领域词汇到分词词典
            math_terms = [
                '勾股定理', '二次函数', '一元二次方程', '相似三角形', '圆周率',
                '平方根', '三角函数', '等差数列', '等比数列', '微积分', '导数',
                '积分', '矩阵', '行列式', '向量', '概率', '统计', '几何', '代数'
            ]
            
            for term in math_terms:
                jieba.add_word(term)
            
            print("词性标注器初始化完成")
        except Exception as e:
            print(f"词性标注器初始化失败: {e}")
            self.use_pos_tagging = False
    
    def _load_trained_model(self, model_dir: str):
        """加载训练好的模型"""
        # 加载tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        
        # 加载配置
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 更新标签映射
        self.label2id = config["label2id"]
        self.id2label = {int(k): v for k, v in config["id2label"].items()}
        
        # 初始化模型
        num_labels = config["num_labels"]
        model_config = config.get("model_config", {})
        
        self.model = AdvancedBERTCRFEntityRecognizer(
            num_labels=num_labels,
            dropout_rate=model_config.get("hidden_dropout_prob", 0.3),
            use_lstm=model_config.get("use_lstm", True),
            lstm_hidden_size=model_config.get("lstm_hidden_size", 256),
            lstm_layers=model_config.get("lstm_layers", 2)
        )
        
        # 加载模型权重
        model_path = os.path.join(model_dir, "pytorch_model.bin")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.model_loaded = True
        print("[OK] 高级BERT模型加载成功")
    
    def extract_entities(self, text: str, use_ensemble: bool = True) -> List[Dict]:
        """提取文本中的实体，支持多种技术融合"""
        entities = []
        
        # 1. 基于规则的提取
        rule_entities = self.rule_based_extraction(text)
        entities.extend(rule_entities)
        
        # 2. 基于词性标注的提取
        if self.use_pos_tagging:
            pos_entities = self.pos_based_extraction(text)
            entities.extend(pos_entities)
        
        # 3. 基于模型的提取（如果模型可用）
        if self.model_loaded and use_ensemble:
            try:
                model_entities = self.model_based_extraction(text)
                entities.extend(model_entities)
            except Exception as e:
                print(f"模型提取失败: {e}")
        
        # 4. KGGen API提取（如果可用）
        if self.use_kggen and use_ensemble:
            try:
                kggen_entities = self.kggen_based_extraction(text)
                entities.extend(kggen_entities)
            except Exception as e:
                print(f"KGGen提取失败: {e}")
        
        # 5. 合并和去重实体
        merged_entities = self.merge_and_deduplicate_entities(entities, text)
        
        # 6. 后处理：验证实体合理性
        final_entities = self.post_process_entities(merged_entities, text)
        
        return final_entities
    
    def kggen_based_extraction(self, text: str) -> List[Dict]:
        """使用KGGen API提取实体"""
        if not self.use_kggen or self.kggen_client is None:
            return []
        
        try:
            result = self.kggen_client.extract_entities_and_relations(text)
            entities = result.get("entities", [])
            
            # 添加来源标记
            for entity in entities:
                entity["source"] = "kggen"
                entity["confidence"] = entity.get("confidence", 0.95)  # KGGen置信度较高
            
            return entities
        except Exception as e:
            print(f"KGGen实体提取失败: {e}")
            return []
    
    def extract_entities_and_relations(self, text: str) -> Dict:
        """同时提取实体和关系（KGGen增强版）"""
        entities = self.extract_entities(text)
        relations = []
        
        # 使用KGGen提取关系
        if self.use_kggen and self.kggen_client:
            try:
                result = self.kggen_client.extract_entities_and_relations(text)
                kggen_relations = result.get("relations", [])
                
                # 验证关系中的实体是否存在
                valid_relations = []
                entity_texts = {e["text"] for e in entities}
                
                for rel in kggen_relations:
                    if (rel["subject"] in entity_texts and 
                        rel["object"] in entity_texts):
                        rel["source"] = "kggen"
                        rel["confidence"] = rel.get("confidence", 0.9)
                        valid_relations.append(rel)
                
                relations.extend(valid_relations)
            except Exception as e:
                print(f"KGGen关系提取失败: {e}")
        
        return {
            "entities": entities,
            "relations": relations
        }
    
    # 保留原有的规则提取、词性提取、模型提取等方法
    def rule_based_extraction(self, text: str) -> List[Dict]:
        """基于规则的实体提取，增强版"""
        # ... (保留原有的rule_based_extraction实现)
        entities = []
        
        # 数学公式模式（增强）
        formula_patterns = [
            r'[a-zA-Zα-ωΑ-Ω][²³⁴⁵⁶⁷⁸⁹⁰⁺⁻⁼⁽⁾]*(?:\s*[+\-*/^=<>≤≥≠≈]?\s*[a-zA-Zα-ωΑ-Ω0-9.²³⁴⁵⁶⁷⁸⁹⁰⁺⁻⁼⁽⁾]+)*',
            r'[∀∃∫∑∏√∞πθφ]',
            r'(?:sin|cos|tan|log|ln|exp|lim|max|min|det|rank|trace)\s*[\(（]',
            r'[Δ∇∂]',
            r'[ⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹ]'
        ]
        
        for pattern in formula_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'FORMULA',
                    'source': 'rule',
                    'confidence': 0.8
                })
        
        # 定理和概念模式
        theorem_patterns = [
            r'(勾股定理|毕达哥拉斯定理)',
            r'(费马大定理|费马最后定理)',
            r'(欧拉公式|欧拉恒等式)',
            r'(泰勒公式|泰勒展开)',
            r'(拉格朗日定理|拉格朗日中值定理)',
            r'(罗尔定理|罗尔中值定理)',
            r'(柯西定理|柯西中值定理)',
            r'(微积分基本定理)',
            r'(二项式定理)',
            r'(贝叶斯定理)'
        ]
        
        for pattern in theorem_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'text': match.group(1),
                    'start': match.start(1),
                    'end': match.end(1),
                    'type': 'THEOREM',
                    'source': 'rule',
                    'confidence': 0.9
                })
        
        # 方法模式
        method_patterns = [
            r'(因式分解法|配方法|公式法|图像法|代数法|几何法)',
            r'(代入法|消元法|加减法|乘法原理|加法原理)',
            r'(数学归纳法|反证法|构造法)',
            r'(求导法|积分法|微分法)'
        ]
        
        for pattern in method_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'text': match.group(1),
                    'start': match.start(1),
                    'end': match.end(1),
                    'type': 'METHOD',
                    'source': 'rule',
                    'confidence': 0.85
                })
        
        # 课标核心数学术语字典 (确保高覆盖率)
        core_math_terms = {
            'CONCEPT': [
                '有理数', '无理数', '整数', '分数', '负数', '正数', '实数',
                '数轴', '绝对值', '相反数', '倒数',
                '加法', '减法', '乘法', '除法', '乘方', '开方',
                '平方根', '立方根', '算术平方根',
                '代数式', '方程', '不等式', '等式',
                '一元一次方程', '一元二次方程', '二元一次方程组',
                '分式', '分式方程', '二次根式',
                '函数', '一次函数', '二次函数', '反比例函数', '正比例函数',
                '几何体', '棱柱', '棱锥', '圆柱', '圆锥',
                '直线', '射线', '线段', '角', '平行线', '垂直',
                '三角形', '直角三角形', '等腰三角形', '等边三角形',
                '全等三角形', '相似三角形',
                '平行四边形', '矩形', '菱形', '正方形', '梯形',
                '圆', '弧', '弦', '圆心角', '圆周角',
                '概率', '随机事件', '频率',
                '数据分析', '中位数', '众数', '方差', '平均数',
                '投影', '视图', '展开图',
                '坐标', '坐标系', '原点',
                '锐角三角函数', '正弦', '余弦', '正切',
            ],
            'THEOREM': [
                '勾股定理', '勾股定理的逆定理',
            ]
        }
        for entity_type, terms in core_math_terms.items():
            for term in terms:
                start = 0
                while True:
                    pos = text.find(term, start)
                    if pos == -1:
                        break
                    start = pos + len(term)
                    entities.append({
                        'text': term,
                        'start': pos,
                        'end': pos + len(term),
                        'type': entity_type,
                        'source': 'rule',
                        'confidence': 1.0,
                    })

        # 基于关键词的实体识别（增强）
        keyword_patterns = {
            'CONCEPT': ['概念', '定义', '含义', '意义', '性质', '特征', '特点', '本质'],
            'EXAMPLE': ['例子', '示例', '例题', '实例', '案例', '典型', '示范'],
            'METHOD': ['方法', '解法', '算法', '步骤', '技巧', '策略', '方案', '途径']
        }
        
        for entity_type, keywords in keyword_patterns.items():
            for keyword in keywords:
                # 修复: 查找所有出现位置，而不是只找第一个
                search_start = 0
                while True:
                    start = text.find(keyword, search_start)
                    if start == -1:
                        break
                    search_start = start + len(keyword)

                    # 提取关键词周围的上下文
                    context_start = max(0, start - 10)
                    context_end = min(len(text), start + len(keyword) + 20)
                    context = text[context_start:context_end]

                    concept = self._extract_concept_from_context(context, keyword)
                    if concept:
                        # 修复: 在原文中定位概念，而不是在截取的 context 中
                        concept_pos = text.find(concept, context_start)
                        if concept_pos != -1:
                            entities.append({
                                'text': concept,
                                'start': concept_pos,
                                'end': concept_pos + len(concept),
                                'type': entity_type,
                                'source': 'rule+context',
                                'confidence': 0.7,
                            })
        
        return entities
    
    def _extract_concept_from_context(self, context: str, keyword: str) -> str:
        """从上下文中提取概念"""
        # 简单的启发式规则
        patterns = [
            r'["，。；：]\s*([^"，。；：]{2,10}?)' + re.escape(keyword),
            r'([^"，。；：]{2,10}?)\s*' + re.escape(keyword),
            keyword + r'\s*[：:]\s*([^"，。；：]{2,15})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, context)
            if match and match.group(1):
                concept = match.group(1).strip()
                if len(concept) >= 2 and len(concept) <= 15:
                    return concept
        
        return None
    
    def pos_based_extraction(self, text: str) -> List[Dict]:
        """基于词性标注的实体提取 (修复: 追踪实际字符偏移量)"""
        if not self.use_pos_tagging:
            return []

        entities = []

        try:
            words = pseg.cut(text)

            # 追踪当前在原文中的偏移量，而不是用 text.find()
            offset = 0
            for word, flag in words:
                # 在原文中定位当前词的实际位置
                start = text.find(word, offset)
                if start == -1:
                    continue
                end = start + len(word)
                offset = end  # 下次从这里开始找

                if flag.startswith('n') and len(word) > 1:
                    if any(t in word for t in ['定理', '公式', '法则', '原理', '定义']):
                        entity_type = 'THEOREM'
                    elif any(t in word for t in ['函数', '方程', '不等式', '矩阵', '向量']):
                        entity_type = 'CONCEPT'
                    else:
                        entity_type = 'CONCEPT'

                    entities.append({
                        'text': word, 'start': start, 'end': end,
                        'type': entity_type, 'source': 'pos',
                        'confidence': 0.6,
                    })

                elif flag == 'eng' and len(word) > 1:
                    entities.append({
                        'text': word, 'start': start, 'end': end,
                        'type': 'FORMULA', 'source': 'pos',
                        'confidence': 0.5,
                    })

        except Exception as e:
            print(f"词性标注提取失败: {e}")

        return entities
    
    def model_based_extraction(self, text: str) -> List[Dict]:
        """基于模型的实体提取"""
        if not self.model_loaded or self.tokenizer is None:
            return []
        
        try:
            # Tokenize文本
            encoded = self.tokenize_and_pad([text])
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            offset_mapping = encoded['offset_mapping'][0].tolist()
            
            # 预测
            with torch.no_grad():
                predictions = self.model(input_ids, attention_mask)
            
            # 转换预测结果为实体
            entities = []
            
            if predictions and len(predictions) > 0:
                pred_labels = [self.id2label.get(int(pred), 'O') for pred in predictions[0]]
            else:
                return []
            
            current_entity = None
            for i, (start_char, end_char) in enumerate(offset_mapping):
                # 跳过特殊token和padding
                if start_char == 0 and end_char == 0:
                    continue
                
                # 获取当前字符的标签
                if i < len(pred_labels):
                    label = pred_labels[i]
                else:
                    label = 'O'
                
                # 处理实体边界（支持BIOES）
                if label.startswith('S-'):
                    # S-标签: 单token实体，立即结束
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
                    entity_type = label[2:]
                    entities.append({
                        'text': text[start_char:end_char],
                        'start': start_char,
                        'end': end_char,
                        'type': entity_type,
                        'source': 'model',
                        'confidence': 0.9
                    })
                elif label.startswith('B-'):
                    if current_entity:
                        entities.append(current_entity)
                    entity_type = label[2:]
                    current_entity = {
                        'text': text[start_char:end_char],
                        'start': start_char,
                        'end': end_char,
                        'type': entity_type,
                        'source': 'model',
                        'confidence': 0.9
                    }
                elif (label.startswith('I-') or label.startswith('E-')) and current_entity and current_entity['type'] == label[2:]:
                    # 扩展当前实体
                    current_entity['text'] += text[start_char:end_char]
                    current_entity['end'] = end_char
                    current_entity['confidence'] = 0.8  # 降低连续实体的置信度
                elif label == 'O' and current_entity:
                    entities.append(current_entity)
                    current_entity = None
            
            if current_entity:
                entities.append(current_entity)
            
            return entities
            
        except Exception as e:
            print(f"模型提取失败: {e}")
            return []
    
    def tokenize_and_pad(self, texts: List[str], max_length: int = None):
        """对文本进行tokenize和padding"""
        if max_length is None:
            max_length = Config.MAX_SEQ_LENGTH
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        return encoded
    
    def merge_and_deduplicate_entities(self, entities: List[Dict], text: str) -> List[Dict]:
        """合并和去重实体"""
        if not entities:
            return []
        
        # 按起始位置排序
        entities.sort(key=lambda x: x['start'])
        
        merged = []
        current = entities[0]
        
        for entity in entities[1:]:
            # 检查重叠
            if (entity['start'] <= current['end'] and 
                entity['end'] >= current['start']):
                
                # 重叠实体，选择置信度高的
                if entity.get('confidence', 0) > current.get('confidence', 0):
                    current = entity
                # 如果置信度相同，优先选择模型提取的结果
                elif (entity.get('confidence', 0) == current.get('confidence', 0) and
                      entity.get('source') == 'model'):
                    current = entity
            else:
                merged.append(current)
                current = entity
        
        merged.append(current)
        
        # 验证实体文本与原始文本匹配
        for entity in merged:
            actual_text = text[entity['start']:entity['end']]
            if entity['text'] != actual_text:
                entity['text'] = actual_text
        
        return merged
    
    def post_process_entities(self, entities: List[Dict], text: str) -> List[Dict]:
        """后处理：验证实体合理性"""
        processed_entities = []
        
        for entity in entities:
            # 过滤太短的实体（除非是公式）
            if (len(entity['text']) < 2 and 
                not entity['type'] == 'FORMULA' and
                not any(c in entity['text'] for c in 'αβγΔπθφ∫∑∏')):
                continue
            
            # 过滤纯标点符号的实体
            if all(c in '，。；：！？（）《》「」‘’""\'\'' for c in entity['text']):
                continue
            
            # 验证实体类型合理性
            if not self._validate_entity_type(entity['text'], entity['type']):
                # 尝试重新分类
                new_type = self._reclassify_entity(entity['text'])
                if new_type:
                    entity['type'] = new_type
                    entity['confidence'] = entity.get('confidence', 0.5) * 0.8  # 降低置信度
                else:
                    continue
            
            processed_entities.append(entity)
        
        return processed_entities
    
    def _validate_entity_type(self, text: str, entity_type: str) -> bool:
        """验证实体类型是否合理"""
        # 公式应该包含数学符号或字母
        if entity_type == 'FORMULA':
            return (any(c.isalpha() for c in text) or
                    any(c in '²³⁴⁵⁶⁷⁸⁹⁰⁺⁻⁼⁽⁾∀∃∫∑∏√∞πθφΔ∇∂' for c in text))
        
        # 定理应该包含"定理"、"公式"等关键词
        if entity_type == 'THEOREM':
            return any(keyword in text for keyword in ['定理', '公式', '法则', '原理'])
        
        # 概念应该是名词性短语
        if entity_type == 'CONCEPT':
            return len(text) >= 2 and not any(c in '+-*/=<>' for c in text)
        
        return True
    
    def _reclassify_entity(self, text: str) -> Optional[str]:
        """重新分类实体"""
        # 包含数学符号 -> FORMULA
        if any(c in '²³⁴⁵⁶⁷⁸⁹⁰⁺⁻⁼⁽⁾∀∃∫∑∏√∞πθφΔ∇∂' for c in text):
            return 'FORMULA'
        
        # 包含定理相关词 -> THEOREM
        if any(keyword in text for keyword in ['定理', '公式', '法则', '原理']):
            return 'THEOREM'
        
        # 包含方法相关词 -> METHOD
        if any(keyword in text for keyword in ['方法', '解法', '算法', '步骤']):
            return 'METHOD'
        
        # 较长的中文文本 -> CONCEPT
        if len(text) >= 3 and all('\u4e00' <= c <= '\u9fff' for c in text):
            return 'CONCEPT'
        
        return None
    
    def batch_extract(self, texts: List[str], batch_size: int = 8) -> List[List[Dict]]:
        """批量提取实体"""
        all_entities = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_entities = []
            
            for text in batch_texts:
                entities = self.extract_entities(text)
                batch_entities.append(entities)
            
            all_entities.extend(batch_entities)
        
        return all_entities

class AdvancedBERTCRFEntityRecognizer(nn.Module):
    """高级BERT+CRF实体识别模型，支持多种优化技术"""
    
    def __init__(self, num_labels: int, bert_model_name: str = None, 
                 dropout_rate: float = 0.3, use_lstm: bool = True,
                 lstm_hidden_size: int = 256, lstm_layers: int = 2):
        super(AdvancedBERTCRFEntityRecognizer, self).__init__()
        if bert_model_name is None:
            bert_model_name = Config.BERT_MODEL
        
        # 加载BERT模型
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 可选的LSTM层
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=self.bert.config.hidden_size,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout_rate if lstm_layers > 1 else 0
            )
            classifier_input_size = lstm_hidden_size * 2
        else:
            self.lstm = None
            classifier_input_size = self.bert.config.hidden_size
        
        # 分类层
        self.classifier = nn.Linear(classifier_input_size, num_labels)
        self.crf = CRF(num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # LSTM处理（如果启用）
        if self.use_lstm:
            lstm_output, _ = self.lstm(sequence_output)
            sequence_output = self.dropout(lstm_output)
        
        # 分类层
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.bool())
            return loss
        else:
            return self.crf.viterbi_decode(logits, mask=attention_mask.bool())

# 使用示例
if __name__ == "__main__":
    try:
        # 初始化KGGen增强实体提取器
        extractor = KGGenEnhancedEntityExtractor(use_pos_tagging=True, use_kggen=True)
        
        # 示例文本
        test_texts = [
            "勾股定理指出：在直角三角形中，两直角边的平方和等于斜边的平方，即 a² + b² = c²",
            "二次函数的一般形式是 y = ax² + bx + c，其中a、b、c是常数且a ≠ 0",
            "解一元二次方程可以使用求根公式 x = [-b ± √(b² - 4ac)] / 2a",
            "微积分基本定理建立了微分和积分之间的关系"
        ]
        
        for i, text in enumerate(test_texts):
            print(f"\n文本 {i+1}: {text}")
            entities = extractor.extract_entities(text)
            
            if entities:
                print("提取到的实体:")
                for entity in entities:
                    print(f"  {entity['type']}: '{entity['text']}' "
                          f"(位置: {entity['start']}-{entity['end']}, "
                          f"来源: {entity['source']}, 置信度: {entity.get('confidence', 1.0):.2f})")
            else:
                print("未提取到实体")
                
    except Exception as e:
        print(f"初始化错误: {e}")
        import traceback
        traceback.print_exc()