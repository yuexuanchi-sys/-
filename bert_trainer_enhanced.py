import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from TorchCRF import CRF
import pandas as pd
from typing import List, Dict, Tuple, Optional
from config import Config
from tqdm import tqdm
import numpy as np
try:
    from seqeval.metrics import classification_report, f1_score
    SEQEVAL_AVAILABLE = True
except ImportError:
    SEQEVAL_AVAILABLE = False
    print("警告: seqeval 模块未安装，评估功能将受限")
import warnings
warnings.filterwarnings('ignore')

class CharLevelMathNERDataset(Dataset):
    """字级别的数学NER数据集类，使用BIO标注法"""
    
    def __init__(self, data_path: str, tokenizer: BertTokenizerFast, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)
        self.label2id = {
            'O': 0,
            'B-CONCEPT': 1, 'I-CONCEPT': 2,
            'B-FORMULA': 3, 'I-FORMULA': 4,
            'B-THEOREM': 5, 'I-THEOREM': 6,
            'B-METHOD': 7, 'I-METHOD': 8,
            'B-EXAMPLE': 9, 'I-EXAMPLE': 10
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def load_data(self, data_path: str) -> List[Dict]:
        """加载JSONL格式的训练数据"""
        data = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line)
                        # 验证数据格式
                        if 'text' in item and 'char_labels' in item:
                            if len(item['text']) == len(item['char_labels']):
                                data.append(item)
                            else:
                                print(f"警告: 第{line_num}行 文本和标签长度不匹配")
                        else:
                            print(f"警告: 第{line_num}行 缺少text或char_labels字段")
                    except json.JSONDecodeError:
                        print(f"错误: 第{line_num}行 JSON解析失败")
        except FileNotFoundError:
            print(f"错误: 文件不存在 {data_path}")
            raise
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        char_labels = item['char_labels']
        
        # Tokenize文本（按字分割）
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # 获取offset mapping用于标签对齐
        offset_mapping = encoding['offset_mapping'][0].tolist()
        
        # 创建标签序列
        label_ids = []
        for i, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:
                # 特殊token ([CLS], [SEP], [PAD])
                label_ids.append(0)  # O标签
            else:
                # 获取字符位置的标签
                if start < len(char_labels):
                    label = char_labels[start]
                    label_ids.append(self.label2id.get(label, 0))
                else:
                    label_ids.append(0)  # O标签
        
        # 确保标签长度匹配
        label_ids = label_ids[:self.max_length]
        if len(label_ids) < self.max_length:
            label_ids.extend([0] * (self.max_length - len(label_ids)))
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'offset_mapping': torch.tensor(offset_mapping, dtype=torch.long)
        }

class BERTLSTMCRFEntityRecognizer(nn.Module):
    """BERT+LSTM+CRF实体识别模型，集成NER-Chinese-master最佳实践"""
    
    def __init__(self, num_labels: int, bert_model_name: str = None,
                 dropout_rate: float = 0.1, lstm_hidden_size: int = 256,
                 lstm_layers: int = 1, use_bidirectional: bool = True):
        super(BERTLSTMCRFEntityRecognizer, self).__init__()
        if bert_model_name is None:
            bert_model_name = Config.BERT_MODEL
        
        # 加载BERT模型
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.bert_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=use_bidirectional,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )
        
        # 根据双向LSTM调整输出维度
        lstm_output_size = lstm_hidden_size * 2 if use_bidirectional else lstm_hidden_size
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分类层
        self.classifier = nn.Linear(lstm_output_size, num_labels)
        
        # CRF层
        self.crf = CRF(num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # LSTM处理
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)
        
        # 分类层
        logits = self.classifier(lstm_output)
        
        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.bool())
            return loss.mean()  # 确保返回标量损失
        else:
            return self.crf.viterbi_decode(logits, mask=attention_mask.bool())

class EnhancedBERTTrainer:
    """增强的BERT模型训练器，包含验证指标和早停机制"""
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = Config.BERT_MODEL
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def prepare_data(self, train_data_path: str, val_data_path: str = None, 
                    batch_size: int = 16, val_split: float = 0.1):
        """准备训练和验证数据，支持自动分割验证集"""
        full_dataset = CharLevelMathNERDataset(train_data_path, self.tokenizer, Config.MAX_SEQ_LENGTH)
        
        # 如果没有提供验证数据，自动分割
        if val_data_path is None:
            dataset_size = len(full_dataset)
            val_size = int(dataset_size * val_split)
            train_size = dataset_size - val_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
        else:
            train_dataset = CharLevelMathNERDataset(train_data_path, self.tokenizer, Config.MAX_SEQ_LENGTH)
            val_dataset = CharLevelMathNERDataset(val_data_path, self.tokenizer, Config.MAX_SEQ_LENGTH)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                      shuffle=True, num_workers=2)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                    shuffle=False, num_workers=2)
        
        print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    
    def initialize_model(self, num_labels: int, use_lstm: bool = True):
        """初始化模型，支持BERT+CRF或BERT+LSTM+CRF"""
        if use_lstm:
            self.model = BERTLSTMCRFEntityRecognizer(
                num_labels=num_labels,
                lstm_hidden_size=256,
                lstm_layers=1,
                use_bidirectional=True
            )
        else:
            # 回退到BERT+CRF
            self.model = BERTCRFEntityRecognizer(num_labels)
        
        self.model.to(self.device)
        
        # 多GPU支持
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
            self.model = nn.DataParallel(self.model)
    
    def train(self, epochs: int = 3, learning_rate: float = 2e-5, 
              warmup_steps: int = 100, early_stopping_patience: int = 3):
        """训练模型，包含早停机制和验证指标"""
        if self.model is None:
            raise ValueError("模型未初始化，请先调用initialize_model")
        if self.train_loader is None:
            raise ValueError("数据未准备，请先调用prepare_data")
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(self.train_loader) * epochs
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        early_stopping_counter = 0
        training_history = []
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            total_train_loss = 0
            train_progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [训练]")
            
            for batch in train_progress:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                loss = self.model(input_ids, attention_mask, labels)
                
                # 多GPU处理
                if isinstance(loss, tuple):
                    loss = loss[0]
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
                train_progress.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_train_loss / len(self.train_loader)
            
            # 验证阶段
            val_loss, val_metrics = self.validate()
            print(f"Epoch {epoch+1} - 训练损失: {avg_train_loss:.4f}, 验证损失: {val_loss:.4f}")
            print(f"验证指标: {val_metrics}")
            
            # 早停机制
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                early_stopping_counter = 0
                print("发现更好的模型，保存中...")
            else:
                early_stopping_counter += 1
                print(f"早停计数器: {early_stopping_counter}/{early_stopping_patience}")
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics
            })
            
            if early_stopping_counter >= early_stopping_patience:
                print("早停触发，停止训练")
                break
        
        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("已加载最佳模型")
        
        return training_history
    
    def validate(self):
        """验证模型，返回损失和评估指标，集成NER-Chinese-master的评估最佳实践"""
        if self.val_loader is None:
            return 0, {}
        
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                loss = self.model(input_ids, attention_mask, labels)
                total_loss += loss.item()
                
                # 获取预测结果
                predictions = self.model(input_ids, attention_mask)
                
                # 转换标签为可读格式
                batch_true_labels = []
                batch_predictions = []
                
                for i in range(len(labels)):
                    # 获取非padding部分的标签
                    mask = attention_mask[i].bool()
                    true_label_ids = labels[i][mask].cpu().numpy()
                    pred_label_ids = predictions[i][:len(true_label_ids)]
                    
                    # 转换ID为标签字符串
                    true_labels = [self.train_loader.dataset.dataset.id2label.get(id_, 'O')
                                  for id_ in true_label_ids]
                    pred_labels = [self.train_loader.dataset.dataset.id2label.get(id_, 'O')
                                  for id_ in pred_label_ids]
                    
                    batch_true_labels.append(true_labels)
                    batch_predictions.append(pred_labels)
                
                all_true_labels.extend(batch_true_labels)
                all_predictions.extend(batch_predictions)
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 计算详细的评估指标，包括每个实体类型的指标
        metrics = self._compute_detailed_metrics(all_true_labels, all_predictions)
        
        self.model.train()
        return avg_loss, metrics
    
    def _compute_detailed_metrics(self, true_labels, pred_labels):
        """计算详细的评估指标，包括每个实体类型的精确率、召回率和F1分数"""
        try:
            report = classification_report(true_labels, pred_labels, output_dict=True)
            
            # 提取总体指标
            overall_metrics = {
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1': report['weighted avg']['f1-score'],
                'accuracy': report['accuracy']
            }
            
            # 提取每个实体类型的指标
            entity_metrics = {}
            for key, value in report.items():
                if key not in ['micro avg', 'macro avg', 'weighted avg', 'accuracy']:
                    entity_metrics[key] = {
                        'precision': value['precision'],
                        'recall': value['recall'],
                        'f1': value['f1-score'],
                        'support': value['support']
                    }
            
            # 计算混淆矩阵相关的指标
            confusion_matrix = self._compute_confusion_matrix(true_labels, pred_labels)
            
            return {
                'overall': overall_metrics,
                'per_entity': entity_metrics,
                'confusion_matrix': confusion_matrix
            }
            
        except Exception as e:
            print(f"评估指标计算失败: {e}")
            return {
                'overall': {'f1': 0, 'precision': 0, 'recall': 0, 'accuracy': 0},
                'per_entity': {},
                'confusion_matrix': {}
            }
    
    def _compute_confusion_matrix(self, true_labels, pred_labels):
        """计算混淆矩阵相关的指标"""
        from seqeval.metrics import classification_report
        
        try:
            # 使用seqeval计算分类报告
            report = classification_report(true_labels, pred_labels, output_dict=True)
            
            # 提取混淆矩阵信息
            cm_info = {}
            for entity_type in report:
                if entity_type not in ['micro avg', 'macro avg', 'weighted avg', 'accuracy']:
                    cm_info[entity_type] = {
                        'true_positive': report[entity_type]['support'] * report[entity_type]['recall'],
                        'false_positive': (report[entity_type]['support'] * report[entity_type]['precision'] -
                                         report[entity_type]['support'] * report[entity_type]['recall']),
                        'false_negative': report[entity_type]['support'] * (1 - report[entity_type]['recall'])
                    }
            
            return cm_info
        except:
            return {}
    
    def save_model(self, output_dir: str, use_lstm: bool = True):
        """保存模型、tokenizer和配置，集成NER-Chinese-master的最佳实践"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存模型权重（完整模型，包括BERT权重）
        model_path = os.path.join(output_dir, "pytorch_model.bin")
        if isinstance(self.model, nn.DataParallel):
            torch.save(self.model.module.state_dict(), model_path)
        else:
            torch.save(self.model.state_dict(), model_path)
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存配置
        model_type = "bert-lstm-crf" if use_lstm else "bert-crf"
        config = {
            "model_type": model_type,
            "num_labels": self.model.classifier.out_features
            if not isinstance(self.model, nn.DataParallel)
            else self.model.module.classifier.out_features,
            "label2id": self.train_loader.dataset.dataset.label2id,
            "id2label": self.train_loader.dataset.dataset.id2label,
            "model_config": {
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1,
                "lstm_hidden_size": 256,
                "lstm_layers": 1,
                "use_bidirectional": True
            },
            "training_config": {
                "max_seq_length": Config.MAX_SEQ_LENGTH,
                "batch_size": Config.BATCH_SIZE,
                "learning_rate": Config.LEARNING_RATE,
                "epochs": Config.EPOCHS
            }
        }
        
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # 保存训练信息（包括评估指标）
        info_path = os.path.join(output_dir, "training_info.json")
        training_info = {
            "best_val_loss": self.best_val_loss,
            "device": str(self.device),
            "timestamp": pd.Timestamp.now().isoformat(),
            "training_history": self.training_history if hasattr(self, 'training_history') else []
        }
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, ensure_ascii=False, indent=2)
        
        # 保存词汇表（兼容NER-Chinese-master）
        vocab_path = os.path.join(output_dir, "vocab.txt")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for token in self.tokenizer.get_vocab():
                f.write(token + '\n')
        
        # 保存标签映射文件
        labels_path = os.path.join(output_dir, "labels.txt")
        with open(labels_path, 'w', encoding='utf-8') as f:
            for label, id_ in self.train_loader.dataset.dataset.label2id.items():
                f.write(f"{label}\t{id_}\n")
        
        print(f"模型已保存到: {output_dir}")
        print("包含文件: pytorch_model.bin, config.json, training_info.json, vocab.txt, labels.txt")

def prepare_char_level_training_data():
    """准备字级别的训练数据"""
    from enhanced_data_processor import EnhancedDataProcessor
    from data_loader import MathDataLoader
    
    processor = EnhancedDataProcessor()
    loader = MathDataLoader()
    
    # 检查训练数据是否已存在
    char_train_data_path = os.path.join(Config.OUTPUT_DIR, "math_ner_char_training_data.jsonl")
    
    if not os.path.exists(char_train_data_path):
        print("字级别训练数据不存在，正在生成...")
        data = loader.load_all_data()
        training_df = processor.create_char_level_ner_training_data(data)
        processor.save_char_level_training_data(training_df, char_train_data_path)
    else:
        print("字级别训练数据已存在，直接使用")
    
    return char_train_data_path

def main():
    """主训练函数"""
    print("开始字级别BERT模型训练...")
    print(f"使用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # 准备训练数据
    train_data_path = prepare_char_level_training_data()
    
    # 初始化训练器
    trainer = EnhancedBERTTrainer()
    
    # 准备数据
    trainer.prepare_data(train_data_path, batch_size=Config.BATCH_SIZE, val_split=0.1)
    
    # 初始化模型
    num_labels = len(trainer.train_loader.dataset.dataset.label2id)
    trainer.initialize_model(num_labels)
    
    # 训练模型
    history = trainer.train(
        epochs=Config.EPOCHS, 
        learning_rate=Config.LEARNING_RATE,
        warmup_steps=100,
        early_stopping_patience=3
    )
    
    # 保存模型
    model_dir = os.path.join(Config.MODEL_DIR, "math_ner_char_model")
    trainer.save_model(model_dir)
    
    # 保存训练历史
    history_path = os.path.join(Config.OUTPUT_DIR, "training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    print("训练完成！训练历史已保存")

if __name__ == "__main__":
    main()