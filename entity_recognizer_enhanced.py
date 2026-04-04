import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from TorchCRF import CRF
import re
import os
import json
from typing import List, Dict, Tuple, Optional
from config import Config
import numpy as np
from seqeval.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

class CharLevelBERTCRFEntityRecognizer(nn.Module):
    """字级别的BERT+CRF实体识别模型"""
    
    def __init__(self, num_labels: int, bert_model_name: str = None, dropout_rate: float = 0.1):
        super(CharLevelBERTCRFEntityRecognizer, self).__init__()
        if bert_model_name is None:
            bert_model_name = Config.BERT_MODEL
        
        # 加载BERT模型
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.bool())
            return loss
        else:
            return self.crf.viterbi_decode(logits, mask=attention_mask.bool())

class EnhancedEntityRecognizer:
    """增强的实体识别器，支持字级别和词级别的实体识别"""
    
    def __init__(self, model_dir: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 实体标签定义
        self.label2id = {
            'O': 0,
            'B-CONCEPT': 1, 'I-CONCEPT': 2,
            'B-FORMULA': 3, 'I-FORMULA': 4,
            'B-THEOREM': 5, 'I-THEOREM': 6,
            'B-METHOD': 7, 'I-METHOD': 8,
            'B-EXAMPLE': 9, 'I-EXAMPLE': 10
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # 尝试加载训练好的模型
        self.model_loaded = False
        self.tokenizer = None
        self.model = None
        
        if model_dir is None:
            model_dir = os.path.join(Config.MODEL_DIR, "math_ner_char_model")
        
        if os.path.exists(model_dir):
            try:
                print("尝试加载训练好的字级别BERT模型...")
                self._load_trained_model(model_dir)
            except Exception as e:
                print(f"训练模型加载失败: {e}")
                print("将使用基于规则的方法进行实体识别")
        else:
            print("训练模型不存在，将使用基于规则的方法进行实体识别")
    
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
        self.model = CharLevelBERTCRFEntityRecognizer(num_labels)
        
        # 加载模型权重
        model_path = os.path.join(model_dir, "pytorch_model.bin")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.model_loaded = True
        print("[OK] 字级别BERT模型加载成功")
    
    def tokenize_and_pad(self, texts: List[str]):
        """对文本进行tokenize和padding"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=Config.MAX_SEQ_LENGTH,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        return encoded
    
    def predict_entities(self, text: str) -> List[Dict]:
        """预测文本中的实体"""
        # 使用规则匹配提取数学公式和定理
        rule_based_entities = self.rule_based_extraction(text)
        
        if self.model_loaded:
            # 使用BERT+CRF模型进行实体识别
            try:
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
    
    def model_based_extraction(self, text: str) -> List[Dict]:
        """基于模型的实体提取（字级别）"""
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
            
            # 获取预测标签
            if predictions and len(predictions) > 0:
                pred_labels = [self.id2label[pred] for pred in predictions[0]]
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
                
                # 处理实体边界
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
                    # 扩展当前实体
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
            
            return entities
            
        except Exception as e:
            import traceback
            print(f"模型提取失败: {e}")
            print("详细错误信息:")
            traceback.print_exc()
            return []
    
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
        
        # 基于关键词识别其他实体类型
        math_keywords = {
            'CONCEPT': ['概念', '定义', '含义', '意义', '性质', '特征'],
            'METHOD': ['方法', '解法', '算法', '步骤', '技巧', '策略'],
            'EXAMPLE': ['例子', '示例', '例题', '实例', '案例']
        }
        
        for entity_type, keywords in math_keywords.items():
            for keyword in keywords:
                start = text.find(keyword)
                if start != -1:
                    entities.append({
                        'text': keyword,
                        'start': start,
                        'end': start + len(keyword),
                        'type': entity_type,
                        'source': 'rule'
                    })
        
        return entities
    
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
    
    def evaluate(self, test_data_path: str):
        """评估模型性能"""
        if not self.model_loaded:
            print("模型未加载，无法进行评估")
            return
        
        # 加载测试数据
        test_data = []
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line))
        
        all_true_labels = []
        all_pred_labels = []
        
        for item in test_data:
            text = item['text']
            true_char_labels = item['char_labels']
            
            # 获取模型预测
            entities = self.model_based_extraction(text)
            
            # 将真实标签转换为BIO格式的序列
            true_labels = true_char_labels
            
            # 将预测结果转换为BIO格式的序列
            pred_labels = ['O'] * len(text)
            for entity in entities:
                for i in range(entity['start'], entity['end']):
                    if i == entity['start']:
                        pred_labels[i] = f'B-{entity["type"]}'
                    else:
                        pred_labels[i] = f'I-{entity["type"]}'
            
            # 确保长度一致
            min_len = min(len(true_labels), len(pred_labels))
            true_labels = true_labels[:min_len]
            pred_labels = pred_labels[:min_len]
            
            all_true_labels.append(true_labels)
            all_pred_labels.append(pred_labels)
        
        # 计算评估指标
        try:
            report = classification_report(all_true_labels, all_pred_labels, output_dict=True)
            print("评估结果:")
            print(f"精确率: {report['weighted avg']['precision']:.4f}")
            print(f"召回率: {report['weighted avg']['recall']:.4f}")
            print(f"F1分数: {report['weighted avg']['f1-score']:.4f}")
            print(f"准确率: {report['accuracy']:.4f}")
            
            return report
        except Exception as e:
            print(f"评估失败: {e}")
            return None

# 使用示例
if __name__ == "__main__":
    try:
        recognizer = EnhancedEntityRecognizer()
        
        # 示例文本
        test_text = "勾股定理指出：在直角三角形中，两直角边的平方和等于斜边的平方，即 a² + b² = c²"
        
        entities = recognizer.predict_entities(test_text)
        print("识别到的实体:")
        for entity in entities:
            print(f"{entity['type']}: {entity['text']} (来源: {entity['source']})")
            
    except Exception as e:
        print(f"初始化错误: {e}")
        print("这可能是由于模型文件不存在或格式不正确。")
        print("请确保已训练并保存了字级别BERT模型。")