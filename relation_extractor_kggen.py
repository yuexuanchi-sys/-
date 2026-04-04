import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import re
from typing import List, Dict, Tuple
from config import Config
from kggen_client import KGGenClient

class KGGenEnhancedRelationExtractor:
    """KGGen增强的关系提取器，集成本地模型和KGGen API"""
    
    def __init__(self, use_kggen: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 关系类型映射
        self.relation2id = {k: i for i, k in enumerate(Config.RELATION_TYPES.keys())}
        self.id2relation = {v: k for k, v in self.relation2id.items()}

        # KGGen客户端 (默认禁用，由 builder 统一调度避免重复调用)
        self.kggen_client = None
        if use_kggen:
            try:
                self.kggen_client = KGGenClient()
                print("[OK] KGGen关系提取客户端初始化成功")
            except Exception as e:
                print(f"[WARN] KGGen客户端初始化失败: {e}")
        
        # 尝试加载BERT模型，如果失败则仅使用规则方法
        self.model_loaded = False
        self.tokenizer = None
        self.model = None
        
        try:
            self.tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL)
            self.model = BERTRelationExtractor(len(self.relation2id))
            self.model.to(self.device)
            self.model_loaded = True
            print("[OK] 关系抽取BERT模型加载成功")
        except Exception as e:
            print(f"[WARN] 关系抽取BERT模型加载失败: {e}")
            print("[WARN] 将使用基于规则和KGGen的方法进行关系抽取")
        
        # 规则模板 - 扩展更多数学教材中的常见表达
        self.rule_patterns = {
            'BELONGS_TO': [
                r'([^，。]+)属于([^，。]+)',
                r'([^，。]+)是([^，。]+)的一部分',
                r'([^，。]+)包含在([^，。]+)中',
                r'([^，。]+)包括([^，。]+)',
                r'([^，。]+)分为([^，。]+)',
                r'([^，。]+)可以分为([^，。]+)'
            ],
            'PREREQUISITE': [
                r'([^，。]+)需要([^，。]+)',
                r'([^，。]+)依赖于([^，。]+)',
                r'([^，。]+)的前提是([^，。]+)',
                r'([^，。]+)必须掌握([^，。]+)',
                r'([^，。]+)要求([^，。]+)',
                r'([^，。]+)基于([^，。]+)',
                r'([^，。]+)建立在([^，。]+)基础上'
            ],
            'DERIVES': [
                r'([^，。]+)推导出([^，。]+)',
                r'([^，。]+)推出([^，。]+)',
                r'由([^，。]+)可得([^，。]+)',
                r'若([^，。]+)则([^，。]+)',
                r'([^，。]+)导出([^，。]+)',
                r'([^，。]+)得到([^，。]+)',
                r'根据([^，。]+)得到([^，。]+)'
            ],
            'RELATED': [
                r'([^，。]+)与([^，。]+)相关',
                r'([^，。]+)和([^，。]+)有联系',
                r'([^，。]+)类似于([^，。]+)',
                r'([^，。]+)对应([^，。]+)',
                r'([^，。]+)相当于([^，。]+)',
                r'([^，。]+)等于([^，。]+)',
                r'([^，。]+)不同于([^，。]+)'
            ],
            'APPLIES': [
                r'([^，。]+)应用于([^，。]+)',
                r'([^，。]+)用来([^，。]+)',
                r'([^，。]+)可以解决([^，。]+)',
                r'([^，。]+)用于([^，。]+)'
            ],
            'CONTAINS': [
                r'([^，。]+)包含([^，。]+)',
                r'([^，。]+)有([^，。]+)',
                r'([^，。]+)包括([^，。]+)',
                r'([^，。]+)由([^，。]+)组成'
            ]
        }
    
    def extract_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """提取实体之间的关系，集成KGGen增强"""
        relations = []
        
        # 1. 基于规则的关系抽取
        rule_based_relations = self.rule_based_extraction(text, entities)
        relations.extend(rule_based_relations)
        
        # 2. 基于模型的关系抽取（如果有足够的实体对且模型已加载）
        if len(entities) >= 2 and self.model_loaded:
            try:
                model_based_relations = self.model_based_extraction(text, entities)
                relations.extend(model_based_relations)
            except Exception as e:
                print(f"模型关系抽取失败，继续使用规则方法: {e}")
        
        # 3. KGGen API关系抽取
        if self.kggen_client:
            try:
                kggen_based_relations = self.kggen_based_extraction(text, entities)
                relations.extend(kggen_based_relations)
            except Exception as e:
                print(f"KGGen关系抽取失败: {e}")
        
        return relations
    
    def kggen_based_extraction(self, text: str, entities: List[Dict]) -> List[Dict]:
        """使用KGGen API提取关系"""
        if not self.kggen_client:
            return []
        
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
            
            return valid_relations
        except Exception as e:
            print(f"KGGen关系提取失败: {e}")
            return []
    
    def rule_based_extraction(self, text: str, entities: List[Dict]) -> List[Dict]:
        """基于规则的关系抽取"""
        relations = []
        entity_texts = [e['text'] for e in entities]
        
        for rel_type, patterns in self.rule_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    if match.groups() and len(match.groups()) >= 2:
                        entity1_text = match.group(1).strip()
                        entity2_text = match.group(2).strip()
                        
                        # 检查是否在实体列表中（使用部分匹配）
                        entity1 = self._find_best_match_entity(entity1_text, entities)
                        entity2 = self._find_best_match_entity(entity2_text, entities)
                        
                        if entity1 and entity2:
                            relations.append({
                                'subject': entity1['text'],
                                'object': entity2['text'],
                                'relation': rel_type,
                                'source': 'rule',
                                'confidence': 0.9
                            })
        
        return relations
    
    def _find_best_match_entity(self, matched_text: str, entities: List[Dict]) -> Dict:
        """找到最佳匹配的实体"""
        # 首先尝试完全匹配
        for entity in entities:
            if entity['text'] == matched_text:
                return entity
        
        # 然后尝试部分匹配（匹配文本包含在实体文本中，或实体文本包含在匹配文本中）
        for entity in entities:
            if entity['text'] in matched_text or matched_text in entity['text']:
                return entity
        
        # 最后尝试相似度匹配（简单的字符串相似度）
        for entity in entities:
            if self._text_similarity(entity['text'], matched_text) > 0.6:
                return entity
        
        return None
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简单的Jaccard相似度）"""
        set1 = set(text1)
        set2 = set(text2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        if not union:
            return 0.0
        return len(intersection) / len(union)
    
    def model_based_extraction(self, text: str, entities: List[Dict]) -> List[Dict]:
        """基于模型的关系抽取"""
        relations = []
        
        # 为每对实体创建样本
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i != j:  # 避免自环
                    try:
                        # 使用实体周围的上下文窗口而不是整个文本
                        context_window = self._get_entity_context(text, entity1, entity2)
                        
                        # 准备输入
                        input_text = f"{context_window} [SEP] {entity1['text']} [SEP] {entity2['text']}"
                        
                        # Tokenize
                        encoded = self.tokenizer(
                            input_text,
                            padding=True,
                            truncation=True,
                            max_length=Config.MAX_SEQ_LENGTH,
                            return_tensors="pt"
                        )
                        
                        # 获取截断后的token序列
                        tokenized_text = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
                        # 根据attention mask移除padding tokens
                        attention_mask = encoded['attention_mask'][0].bool()
                        tokenized_text = [token for i, token in enumerate(tokenized_text) if attention_mask[i]]
                        
                        # 确保实体文本是字符串
                        if not isinstance(entity1['text'], str) or not isinstance(entity2['text'], str):
                            continue
                        
                        # 获取实体token序列
                        entity1_tokens = self.tokenizer.tokenize(entity1['text'])
                        entity2_tokens = self.tokenizer.tokenize(entity2['text'])
                        
                        # 检查实体token是否完全在截断序列中
                        if not all(token in tokenized_text for token in entity1_tokens) or not all(token in tokenized_text for token in entity2_tokens):
                            continue  # 跳过如果实体token不完整
                        
                        # 在截断的token序列中找到实体位置
                        entity1_pos = self._find_entity_position(tokenized_text, entity1['text'])
                        entity2_pos = self._find_entity_position(tokenized_text, entity2['text'])
                        
                        if entity1_pos is None or entity2_pos is None:
                            continue  # 跳过如果无法找到实体位置
                        
                        # 确保位置不超过序列长度
                        seq_len = len(tokenized_text)
                        if entity1_pos >= seq_len or entity2_pos >= seq_len:
                            continue  # 跳过如果位置无效
                            
                        # 预测关系
                        self.model.eval()
                        with torch.no_grad():
                            logits = self.model(
                                encoded['input_ids'].to(self.device),
                                encoded['attention_mask'].to(self.device),
                                torch.tensor([entity1_pos]).to(self.device),
                                torch.tensor([entity2_pos]).to(self.device)
                            )
                            
                            probs = torch.softmax(logits, dim=-1)
                            confidence, pred_id = torch.max(probs, dim=-1)
                            
                            if confidence.item() > 0.7:  # 置信度阈值
                                relation_type = self.id2relation[pred_id.item()]
                                
                                relations.append({
                                    'subject': entity1['text'],
                                    'object': entity2['text'],
                                    'relation': relation_type,
                                    'source': 'model',
                                    'confidence': confidence.item()
                                })
                    except Exception as e:
                        print(f"关系抽取模型处理失败: {e}")
                        continue
        
        return relations

    def _get_entity_context(self, text: str, entity1: Dict, entity2: Dict) -> str:
        """获取实体周围的上下文窗口"""
        # 确定两个实体的最小和最大位置
        start_pos = min(entity1['start'], entity2['start'])
        end_pos = max(entity1['end'], entity2['end'])
        
        # 扩展上下文窗口
        context_start = max(0, start_pos - 100)  # 向前扩展100字符
        context_end = min(len(text), end_pos + 100)  # 向后扩展100字符
        
        return text[context_start:context_end]
    
    def _find_entity_position(self, tokens: List[str], entity_text: str) -> int:
        """在token序列中找到实体的起始位置"""
        entity_tokens = self.tokenizer.tokenize(entity_text)
        
        # 在tokens中搜索entity_tokens的序列
        for i in range(len(tokens) - len(entity_tokens) + 1):
            if tokens[i:i+len(entity_tokens)] == entity_tokens:
                return i  # 返回起始位置
        
        return None  # 如果没有找到

class BERTRelationExtractor(nn.Module):
    def __init__(self, num_relations: int):
        super(BERTRelationExtractor, self).__init__()
        self.bert = BertModel.from_pretrained(Config.BERT_MODEL)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size * 3, num_relations)
        
    def forward(self, input_ids, attention_mask, entity1_pos, entity2_pos):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # 获取实体位置的表示
        batch_size = sequence_output.size(0)
        
        # 获取实体1和实体2的表示
        entity1_repr = torch.zeros(batch_size, sequence_output.size(-1)).to(sequence_output.device)
        entity2_repr = torch.zeros(batch_size, sequence_output.size(-1)).to(sequence_output.device)
        
        for i in range(batch_size):
            entity1_repr[i] = sequence_output[i, entity1_pos[i]].mean(dim=0)
            entity2_repr[i] = sequence_output[i, entity2_pos[i]].mean(dim=0)
        
        # 拼接实体表示和CLS表示
        cls_repr = sequence_output[:, 0, :]  # CLS token
        combined = torch.cat([cls_repr, entity1_repr, entity2_repr], dim=-1)
        combined = self.dropout(combined)
        
        logits = self.classifier(combined)
        return logits

# 使用示例
if __name__ == "__main__":
    extractor = KGGenEnhancedRelationExtractor()
    
    # 示例文本和实体
    test_text = "勾股定理是直角三角形的重要性质，它推导自平方差公式"
    test_entities = [
        {'text': '勾股定理', 'type': 'THEOREM', 'start': 0, 'end': 4},
        {'text': '直角三角形', 'type': 'CONCEPT', 'start': 6, 'end': 10},
        {'text': '平方差公式', 'type': 'FORMULA', 'start': 16, 'end': 20}
    ]
    
    relations = extractor.extract_relations(test_text, test_entities)
    print("提取到的关系:")
    for rel in relations:
        print(f"{rel['subject']} --{rel['relation']}--> {rel['object']} (来源: {rel['source']}, 置信度: {rel['confidence']:.2f})")