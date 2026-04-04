import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from TorchCRF import CRF
import re
import os
import json
from typing import List, Dict, Tuple
from config import Config

class BERTCRFEntityRecognizer(nn.Module):
    def __init__(self, num_labels: int):
        super(BERTCRFEntityRecognizer, self).__init__()
        self.bert = BertModel.from_pretrained(Config.BERT_MODEL)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.bool())
            return loss.mean()  # 确保返回标量损失
        else:
            return self.crf.viterbi_decode(logits, mask=attention_mask.bool())

class EntityRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        # 添加并行计算支持
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 实体标签定义
        self.label2id = {
            'O': 0,
            'B-CONCEPT': 1, 'I-CONCEPT': 2,
            'B-FORMULA': 3, 'I-FORMULA': 4,
            'B-THEOREM': 5, 'I-THEOREM': 6,
            'B-EXAMPLE': 7, 'I-EXAMPLE': 8,
            'B-METHOD': 9, 'I-METHOD': 10
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # 尝试加载训练好的BERT模型，如果失败则使用预训练模型
        self.model_loaded = False
        self.tokenizer = None
        self.model = None
        
        # 优先尝试加载训练好的模型
        trained_model_dir = "./models/math_ner_model"
        if os.path.exists(trained_model_dir):
            try:
                print("尝试加载训练好的BERT模型...")
                self.tokenizer = BertTokenizerFast.from_pretrained(trained_model_dir)
                
                # 加载配置获取标签数量
                config_path = os.path.join(trained_model_dir, "config.json")
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                num_labels = config["num_labels"]
                self.model = BERTCRFEntityRecognizer(num_labels)
                
                # 加载模型权重
                model_path = os.path.join(trained_model_dir, "pytorch_model.bin")
                self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
                self.model.to(self.device)
                
                # 更新标签映射
                self.label2id = config["label2id"]
                self.id2label = {int(k): v for k, v in config["id2label"].items()}
                
                self.model_loaded = True
                print("[OK] 训练好的BERT模型加载成功")
            except Exception as e:
                print(f"[WARN] 训练模型加载失败: {e}")
                # 回退到预训练模型
                self._load_pretrained_model()
        else:
            # 如果没有训练好的模型，加载预训练模型
            self._load_pretrained_model()
    
    def tokenize_and_pad(self, texts: List[str]):
        """对文本进行tokenize和padding"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=Config.MAX_SEQ_LENGTH,
            return_tensors="pt"
        )
        return encoded
    
    def predict_entities(self, text: str) -> List[Dict]:
        """预测文本中的实体"""
        # 使用规则匹配提取数学公式和定理
        rule_based_entities = self.rule_based_extraction(text)
        
        if self.model_loaded:
            # 使用BERT+CRF模型进行实体识别
            try:
                # 如果文本过长，分割成句子处理
                if len(text) > Config.MAX_SEQ_LENGTH * 0.8:  # 如果接近最大长度限制
                    model_based_entities = self._process_long_text(text)
                else:
                    model_based_entities = self.model_based_extraction(text)
                
                # 合并结果
                all_entities = rule_based_entities + model_based_entities
                merged_entities = self.merge_entities(all_entities)
                # 确保所有实体文本与原始文本完全匹配
                for entity in merged_entities:
                    entity['text'] = text[entity['start']:entity['end']]
                return merged_entities
            except Exception as e:
                print(f"模型预测失败: {e}")
                # 如果模型预测失败，只返回规则提取的结果
                return rule_based_entities
        else:
            # 如果没有加载模型，只使用规则方法
            return rule_based_entities
            
    def _process_long_text(self, text: str) -> List[Dict]:
        """处理长文本，分割成句子进行实体识别"""
        import re
        
        # 简单的中文句子分割
        sentences = re.split(r'[。！？!?;；]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        all_entities = []
        
        for sentence in sentences:
            if len(sentence) > 10:  # 只处理有意义的句子
                try:
                    entities = self.model_based_extraction(sentence)
                    all_entities.extend(entities)
                except Exception as e:
                    print(f"处理句子时出错: {e}")
                    continue
        
        return all_entities
    
    def rule_based_extraction(self, text: str) -> List[Dict]:
        """基于规则的实体提取"""
        entities = []
        
        # 提取数学公式
        for pattern in Config.FORMULA_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'FORMULA',
                    'source': 'rule'
                })
        
        # 提取定理名称
        for pattern in Config.THEOREM_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                if match.groups():
                    # 使用第一个分组的文本和位置
                    start = match.start(1)
                    end = match.end(1)
                    text_content = match.group(1)
                else:
                    start = match.start()
                    end = match.end()
                    text_content = match.group()
                
                entities.append({
                    'text': text_content,
                    'start': start,
                    'end': end,
                    'type': 'THEOREM',
                    'source': 'rule'
                })
        
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
            
            # 预测
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(input_ids, attention_mask)
            
            # 转换预测结果为实体
            entities = []
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            
            # viterbi_decode返回的是列表的列表，取第一个batch的结果
            if predictions and len(predictions) > 0:
                pred_labels = [self.id2label[pred] for pred in predictions[0]]
            else:
                return []
            
            # 获取原始文本到token的映射
            encoding = self.tokenizer(text, return_offsets_mapping=True)
            offset_mapping = encoding['offset_mapping']
            
            current_entity = None
            for i, (token, label) in enumerate(zip(tokens, pred_labels)):
                # 跳过特殊token
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    continue
                
                # 获取字符偏移量
                if i < len(offset_mapping):
                    start_char, end_char = offset_mapping[i]
                    
                    # 处理中文文本：确保连续字符正确合并
                    if label.startswith('B-'):
                        if current_entity:
                            entities.append(current_entity)
                        entity_type = label[2:]
                        current_entity = {
                            'text': text[start_char:end_char],
                            'start': start_char,
                            'end': end_char,
                            'type': entity_type,
                            'source': 'model'
                        }
                    elif label.startswith('I-') and current_entity and current_entity['type'] == label[2:]:
                        # 对于中文，直接拼接文本，不添加空格
                        current_entity['text'] += text[start_char:end_char]
                        current_entity['end'] = end_char
                    elif label == 'O' and current_entity:
                        entities.append(current_entity)
                        current_entity = None
                    elif current_entity and label.startswith('B-'):  # 新的B标签出现
                        entities.append(current_entity)
                        entity_type = label[2:]
                        current_entity = {
                            'text': text[start_char:end_char],
                            'start': start_char,
                            'end': end_char,
                            'type': entity_type,
                            'source': 'model'
                        }
            
            if current_entity:
                entities.append(current_entity)
            
            # 确保实体文本与原始文本完全匹配
            for entity in entities:
                entity['text'] = text[entity['start']:entity['end']]
            
            return entities
            
        except Exception as e:
            import traceback
            print(f"模型提取失败: {e}")
            print("详细错误信息:")
            traceback.print_exc()
            return []
    
    def merge_entities(self, entities: List[Dict]) -> List[Dict]:
        """合并重叠的实体"""
        if not entities:
            return []
        
        # 按起始位置排序
        entities.sort(key=lambda x: x['start'])
        
        merged = []
        current = entities[0]
        
        for entity in entities[1:]:
            if entity['start'] <= current['end']:
                # 重叠，合并
                current['end'] = max(current['end'], entity['end'])
                current['text'] = current['text'] + entity['text']
                # 优先保留规则提取的结果
                if entity['source'] == 'rule':
                    current['source'] = 'rule'
                    current['type'] = entity['type']
            else:
                merged.append(current)
                current = entity
        
        merged.append(current)
        return merged
    
    def _load_pretrained_model(self):
        """加载预训练模型"""
        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(Config.BERT_MODEL)
            self.model = BERTCRFEntityRecognizer(len(self.label2id))
            self.model.to(self.device)
            self.model_loaded = True
            print("[OK] 预训练BERT模型加载成功")
        except Exception as e:
            print(f"[WARN] 预训练BERT模型加载失败: {e}")
            print("[WARN] 将仅使用基于规则的方法进行实体识别")
    
    def train(self, train_data, val_data=None):
        """训练模型"""
        # 这里需要实现训练逻辑
        # 包括数据准备、训练循环、验证等
        pass
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label2id': self.label2id,
            'id2label': self.id2label
        }, path)
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.label2id = checkpoint['label2id']
        self.id2label = checkpoint['id2label']

# 使用示例
if __name__ == "__main__":
    try:
        recognizer = EntityRecognizer()
        
        # 示例文本
        test_text = "勾股定理指出：在直角三角形中，两直角边的平方和等于斜边的平方，即 a² + b² = c²"
        
        entities = recognizer.predict_entities(test_text)
        print("识别到的实体:")
        for entity in entities:
            print(f"{entity['type']}: {entity['text']} (来源: {entity['source']})")
            
    except Exception as e:
        print(f"初始化错误: {e}")
        print("这可能是由于网络问题无法下载BERT模型，或者缺少模型文件。")
        print("请检查网络连接，或者确保BERT模型已正确下载。")

class ImprovedEntityRecognizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载数学领域特定的NER模型（占位符，实际需要训练或加载预训练模型）
        self.model = self._load_math_ner_model()
        
        # 添加规则基础的方法作为后备
        self.math_keywords = {
            "THEOREM": ["定理", "定律", "引理", "公理"],
            "CONCEPT": ["概念", "定义", "含义", "意义"],
            "FORMULA": ["公式", "表达式", "方程", "不等式"],
            "METHOD": ["方法", "解法", "算法", "步骤"],
            "EXAMPLE": ["例子", "示例", "例题", "实例"],
            "CHAPTER_TITLE": ["章", "节", "部分", "章节"]
        }
    
    def _load_math_ner_model(self):
        """加载数学领域NER模型（占位符实现）"""
        # 这里应该是加载预训练模型或初始化新模型的逻辑
        # 由于没有实际模型，返回None，使用规则方法
        print("[INFO] 数学领域NER模型未实现，将使用规则方法")
        return None
    
    def predict_entities(self, text: str) -> List[Dict]:
        """预测文本中的实体"""
        # 先使用模型识别
        entities = self._model_predict(text)
        
        # 如果模型结果不理想，使用规则方法
        if not entities or self._low_confidence(entities):
            entities = self._rule_based_entity_recognition(text)
        
        return entities
    
    def _model_predict(self, text: str) -> List[Dict]:
        """模型预测（占位符实现）"""
        # 这里应该是实际模型预测的逻辑
        # 由于没有模型，返回空列表
        return []
    
    def _low_confidence(self, entities: List[Dict]) -> bool:
        """检查模型置信度是否较低"""
        # 简单的启发式方法：如果实体数量很少或文本很长但实体很少
        if len(entities) < 2 and len(entities) > 0:
            return True
        return False
    
    def _rule_based_entity_recognition(self, text: str) -> List[Dict]:
        """基于规则的实体识别"""
        entities = []
        
        # 基于关键词和模式的规则识别
        for entity_type, keywords in self.math_keywords.items():
            for keyword in keywords:
                start = text.find(keyword)
                if start != -1:
                    entities.append({
                        "text": keyword,
                        "start": start,
                        "end": start + len(keyword),
                        "type": entity_type,
                        "source": "rule_based"
                    })
        
        # 添加数学公式模式匹配（从现有EntityRecognizer中借鉴）
        formula_patterns = [
            r'[a-zA-Zα-ωΑ-Ω][²³⁰-⁹]?[\+\-\*\/\=\<\>][a-zA-Zα-ωΑ-Ω][²³⁰-⁹]?',  # 简单公式
            r'[a-zA-Zα-ωΑ-Ω]_?\{?[0-9]+\}?',  # 带下标的变量
            r'[∛∜√∑∏∫∂∇∆]',  # 数学符号
        ]
        
        for pattern in formula_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'FORMULA',
                    'source': 'rule'
                })
        
        return entities
    
    def _is_likely_misclassified(self, entity: Dict) -> bool:
        """判断实体是否可能被错误分类"""
        text = entity['text']
        entity_type = entity['type']
        
        # 章节标题不应被分类为定理
        if entity_type == "THEOREM" and ("章" in text or "节" in text):
            return True
            
        # 不完整的文本不应被分类为方法
        if entity_type == "METHOD" and text.endswith("了") and not text.endswith("方法"):
            return True
            
        return False
    
    def _correct_entity_type(self, entity: Dict) -> str:
        """修正实体类型"""
        text = entity['text']
        
        if "章" in text or "节" in text:
            return "CHAPTER_TITLE"
        elif "方法" in text or "解法" in text:
            return "METHOD"
        elif "定理" in text or "定律" in text:
            return "THEOREM"
        elif "公式" in text or "表达式" in text:
            return "FORMULA"
        else:
            return "CONCEPT"  # 默认归类为概念

# 使用示例
if __name__ == "__main__":
    # 测试ImprovedEntityRecognizer
    improved_recognizer = ImprovedEntityRecognizer()
    test_text = "勾股定理指出：在直角三角形中，两直角边的平方和等于斜边的平方，即 a² + b² = c²。本章介绍了解题方法。"
    
    entities = improved_recognizer.predict_entities(test_text)
    print("改进识别器识别到的实体:")
    for entity in entities:
        print(f"{entity['type']}: {entity['text']} (来源: {entity['source']})")