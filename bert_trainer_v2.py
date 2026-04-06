import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW
from TorchCRF import CRF
import pandas as pd
from typing import List, Dict, Tuple, Optional
from config import Config
from tqdm import tqdm
import numpy as np
import random


class _NumpyEncoder(json.JSONEncoder):
    """处理numpy类型的JSON序列化"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# 可复现性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
try:
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    SEQEVAL_AVAILABLE = True
except ImportError:
    SEQEVAL_AVAILABLE = False
    print("警告: seqeval 模块未安装，评估功能将受限")
import warnings
warnings.filterwarnings('ignore')

class CharLevelMathNERDataset(Dataset):
    """字级别的数学NER数据集类，支持BIOES标注法"""
    
    def __init__(self, data_path: str, tokenizer: BertTokenizerFast, max_length: int = 128, 
                 use_bioes: bool = True, augment: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_bioes = use_bioes
        self.augment = augment
        self.data = self.load_data(data_path)
        
        # 定义标签映射
        if use_bioes:
            self.label2id = {
                'O': 0,
                'B-CONCEPT': 1, 'I-CONCEPT': 2, 'E-CONCEPT': 3, 'S-CONCEPT': 4,
                'B-FORMULA': 5, 'I-FORMULA': 6, 'E-FORMULA': 7, 'S-FORMULA': 8,
                'B-THEOREM': 9, 'I-THEOREM': 10, 'E-THEOREM': 11, 'S-THEOREM': 12,
                'B-METHOD': 13, 'I-METHOD': 14, 'E-METHOD': 15, 'S-METHOD': 16,
                'B-EXAMPLE': 17, 'I-EXAMPLE': 18, 'E-EXAMPLE': 19, 'S-EXAMPLE': 20
            }
        else:
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
                                # 如果需要数据增强，创建增强版本
                                if self.augment:
                                    augmented_items = self.augment_data(item)
                                    data.extend(augmented_items)
                                else:
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
    
    def augment_data(self, item: Dict) -> List[Dict]:
        """数据增强：随机替换、插入、删除"""
        augmented_items = [item]  # 包含原始数据
        
        text = item['text']
        char_labels = item['char_labels']
        
        # 1. 随机替换（10%的概率）
        if random.random() < 0.1:
            aug_text = list(text)
            aug_labels = char_labels.copy()
            
            # 选择非实体部分进行替换
            non_entity_indices = [i for i, label in enumerate(char_labels) if label == 'O']
            if non_entity_indices:
                idx = random.choice(non_entity_indices)
                aug_text[idx] = random.choice(['的', '是', '在', '和', '与', '或'])
                augmented_items.append({
                    'text': ''.join(aug_text),
                    'char_labels': aug_labels
                })
        
        # 2. 随机插入（5%的概率，仅在O标签位置插入以保持BIOES一致性）
        if random.random() < 0.05:
            # 只在O标签位置插入，避免破坏实体内部的BIOES序列
            o_positions = [i for i, label in enumerate(char_labels) if label == 'O']
            if o_positions:
                insert_pos = random.choice(o_positions)
                insert_char = random.choice(['，', '。', '；', '：'])

                aug_text = text[:insert_pos] + insert_char + text[insert_pos:]
                aug_labels = char_labels[:insert_pos] + ['O'] + char_labels[insert_pos:]

                augmented_items.append({
                    'text': aug_text,
                    'char_labels': aug_labels
                })
        
        # 3. 随机删除（5%的概率）
        if random.random() < 0.05:
            non_entity_indices = [i for i, label in enumerate(char_labels) if label == 'O']
            if non_entity_indices:
                idx = random.choice(non_entity_indices)
                aug_text = text[:idx] + text[idx+1:]
                aug_labels = char_labels[:idx] + char_labels[idx+1:]
                
                augmented_items.append({
                    'text': aug_text,
                    'char_labels': aug_labels
                })
        
        return augmented_items
    
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

class BERTBiLSTMCRFEntityRecognizer(nn.Module):
    """BERT+BiLSTM+CRF实体识别模型，支持BIOES标注"""
    
    def __init__(self, num_labels: int, bert_model_name: str = None,
                 dropout_rate: float = 0.3, lstm_hidden_size: int = 256,
                 lstm_layers: int = 2, use_bidirectional: bool = True):
        super(BERTBiLSTMCRFEntityRecognizer, self).__init__()
        if bert_model_name is None:
            bert_model_name = Config.BERT_MODEL
        
        # 加载BERT模型
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # BiLSTM层 (当lstm_hidden_size>0时启用)
        self.use_lstm = lstm_hidden_size > 0
        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=self.bert_hidden_size,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=use_bidirectional,
                dropout=dropout_rate if lstm_layers > 1 else 0
            )
            lstm_output_size = lstm_hidden_size * 2 if use_bidirectional else lstm_hidden_size
        else:
            lstm_output_size = self.bert_hidden_size

        # 分类层
        self.classifier = nn.Linear(lstm_output_size, num_labels)
        
        # CRF层
        self.crf = CRF(num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Dropout
        sequence_output = self.dropout(sequence_output)

        # LSTM处理 (可选)
        if self.use_lstm:
            lstm_output, _ = self.lstm(sequence_output)
            lstm_output = self.dropout(lstm_output)
        else:
            lstm_output = sequence_output
        
        # 分类层
        logits = self.classifier(lstm_output)
        
        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.bool())
            return loss.mean()
        else:
            return self.crf.viterbi_decode(logits, mask=attention_mask.bool())

    def forward_with_loss_and_decode(self, input_ids, attention_mask, labels):
        """单次前向传播同时返回loss和解码结果，避免双次推理"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        if self.use_lstm:
            lstm_output, _ = self.lstm(sequence_output)
            lstm_output = self.dropout(lstm_output)
        else:
            lstm_output = sequence_output
        logits = self.classifier(lstm_output)
        mask = attention_mask.bool()
        loss = -self.crf(logits, labels, mask=mask)
        predictions = self.crf.viterbi_decode(logits, mask=mask)
        return loss.mean(), predictions

class AdvancedBERTTrainer:
    """高级BERT模型训练器，支持多种优化技术"""
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = Config.BERT_MODEL
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0
        self.best_model_state = None
        self.training_history = []
        self.label2id = {}
        self.id2label = {}
        
    def prepare_data(self, train_data_path: str, val_data_path: str = None, 
                    test_data_path: str = None, batch_size: int = 16, 
                    val_split: float = 0.1, test_split: float = 0.1,
                    use_bioes: bool = True, augment: bool = False):
        """准备训练、验证和测试数据"""
        full_dataset = CharLevelMathNERDataset(
            train_data_path, self.tokenizer, Config.MAX_SEQ_LENGTH, use_bioes, augment
        )
        
        # 保存完整数据集引用以便后续使用
        self.full_dataset = full_dataset
        
        # 数据集分割 (使用torch.utils.data.random_split保持Dataset类型)
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * val_split)
        test_size = int(dataset_size * test_split)
        train_size = dataset_size - val_size - test_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                      shuffle=True, num_workers=2, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                    shuffle=False, num_workers=2, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                     shuffle=False, num_workers=2, pin_memory=True)
        
        print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}, 测试样本: {len(test_dataset)}")
    
    def initialize_model(self, num_labels: int, model_type: str = "bert_bilstm_crf"):
        """初始化模型，支持多种架构"""
        if model_type == "bert_bilstm_crf":
            self.model = BERTBiLSTMCRFEntityRecognizer(
                num_labels=num_labels,
                dropout_rate=0.3,
                lstm_hidden_size=256,
                lstm_layers=2,
                use_bidirectional=True
            )
        elif model_type == "bert_crf":
            # 简化的BERT+CRF模型
            self.model = BERTBiLSTMCRFEntityRecognizer(
                num_labels=num_labels,
                lstm_hidden_size=0,  # 不使用LSTM
                use_bidirectional=False
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        self.model.to(self.device)
        
        # 多GPU支持
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
            self.model = nn.DataParallel(self.model)
    
    def train(self, epochs: int = 3, learning_rate: float = 2e-5, 
              warmup_ratio: float = 0.1, early_stopping_patience: int = 5,
              gradient_accumulation_steps: int = 1, max_grad_norm: float = 1.0):
        """训练模型，支持梯度累积和早停"""
        if self.model is None:
            raise ValueError("模型未初始化，请先调用initialize_model")
        if self.train_loader is None:
            raise ValueError("数据未准备，请先调用prepare_data")
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(self.train_loader) * epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        early_stopping_counter = 0
        global_step = 0
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            total_train_loss = 0
            train_progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [训练]")
            
            for step, batch in enumerate(train_progress):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                loss = self.model(input_ids, attention_mask, labels)
                
                # 梯度累积
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                total_train_loss += loss.item() * gradient_accumulation_steps
                train_progress.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
            
            avg_train_loss = total_train_loss / len(self.train_loader)
            
            # 验证阶段
            val_loss, val_metrics = self.validate()
            val_f1 = val_metrics['overall']['f1']
            
            print(f"Epoch {epoch+1} - 训练损失: {avg_train_loss:.4f}, 验证损失: {val_loss:.4f}")
            print(f"验证F1: {val_f1:.4f}, 精确率: {val_metrics['overall']['precision']:.4f}, 召回率: {val_metrics['overall']['recall']:.4f}")
            
            # 保存训练历史
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            
            # 早停机制：基于F1分数
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_val_loss = val_loss
                self.best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                early_stopping_counter = 0
                print("发现更好的模型，保存中...")
            else:
                early_stopping_counter += 1
                print(f"早停计数器: {early_stopping_counter}/{early_stopping_patience}")
            
            if early_stopping_counter >= early_stopping_patience:
                print("早停触发，停止训练")
                break
        
        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("已加载最佳模型")
        
        return self.training_history
    
    def validate(self):
        """验证模型性能"""
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
                
                # DataParallel兼容: 获取原始模型
                raw_model = self.model.module if hasattr(self.model, 'module') else self.model
                loss, predictions = raw_model.forward_with_loss_and_decode(input_ids, attention_mask, labels)
                total_loss += loss.item()

                # 转换标签为可读格式
                batch_true_labels = []
                batch_predictions = []

                attention_mask_cpu = attention_mask.cpu()
                labels_cpu = labels.cpu()

                for i in range(len(labels)):
                    mask = attention_mask_cpu[i].bool()
                    true_label_ids = labels_cpu[i][mask].numpy()
                    pred_seq = predictions[i]
                    if hasattr(pred_seq, 'cpu'):
                        pred_label_ids = pred_seq[:len(true_label_ids)].cpu().numpy()
                    else:
                        pred_label_ids = pred_seq[:len(true_label_ids)]

                    true_labels = [self.id2label.get(int(id_), 'O')
                                  for id_ in true_label_ids]
                    pred_labels = [self.id2label.get(int(id_), 'O')
                                  for id_ in pred_label_ids]

                    batch_true_labels.append(true_labels)
                    batch_predictions.append(pred_labels)

                all_true_labels.extend(batch_true_labels)
                all_predictions.extend(batch_predictions)

        avg_loss = total_loss / len(self.val_loader)
        metrics = self._compute_detailed_metrics(all_true_labels, all_predictions)

        self.model.train()
        return avg_loss, metrics
    
    def test(self):
        """测试模型性能"""
        if self.test_loader is None:
            print("测试集未准备")
            return {}
        
        self.model.eval()
        all_predictions = []
        all_true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="测试"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                predictions = self.model(input_ids, attention_mask)
                
                # 将attention_mask和labels移动到CPU用于评估
                attention_mask_cpu = attention_mask.cpu()
                labels_cpu = batch['labels'].cpu()
                
                for i in range(len(predictions)):
                    mask = attention_mask_cpu[i].bool()  # mask在CPU上
                    true_label_ids = labels_cpu[i][mask].numpy()
                    # CRF decode返回list, 取对应长度
                    pred_seq = predictions[i]
                    if hasattr(pred_seq, 'cpu'):
                        pred_label_ids = pred_seq[:len(true_label_ids)].cpu().numpy()
                    else:
                        pred_label_ids = pred_seq[:len(true_label_ids)]

                    true_labels = [self.id2label.get(int(id_), 'O')
                                  for id_ in true_label_ids]
                    pred_labels = [self.id2label.get(int(id_), 'O')
                                  for id_ in pred_label_ids]
                    
                    all_true_labels.append(true_labels)
                    all_predictions.append(pred_labels)
        
        metrics = self._compute_detailed_metrics(all_true_labels, all_predictions)
        print("测试结果:")
        print(f"F1分数: {metrics['overall']['f1']:.4f}")
        print(f"精确率: {metrics['overall']['precision']:.4f}")
        print(f"召回率: {metrics['overall']['recall']:.4f}")
        
        return metrics
    
    def _compute_detailed_metrics(self, true_labels, pred_labels):
        """计算详细的评估指标"""
        if not SEQEVAL_AVAILABLE or not true_labels or not pred_labels:
            return {
                'overall': {'f1': 0, 'precision': 0, 'recall': 0, 'accuracy': 0},
                'per_entity': {},
                'confusion_matrix': {}
            }
        
        try:
            report = classification_report(true_labels, pred_labels, output_dict=True)
            
            # 安全地获取指标，避免KeyError
            overall_metrics = {
                'precision': report.get('weighted avg', {}).get('precision', 0),
                'recall': report.get('weighted avg', {}).get('recall', 0),
                'f1': report.get('weighted avg', {}).get('f1-score', 0),
                'accuracy': report.get('accuracy', 0)
            }
            
            entity_metrics = {}
            for key, value in report.items():
                if key not in ['micro avg', 'macro avg', 'weighted avg', 'accuracy'] and isinstance(value, dict):
                    entity_metrics[key] = {
                        'precision': value.get('precision', 0),
                        'recall': value.get('recall', 0),
                        'f1': value.get('f1-score', 0),
                        'support': value.get('support', 0)
                    }
            
            return {
                'overall': overall_metrics,
                'per_entity': entity_metrics
            }
            
        except Exception as e:
            print(f"评估指标计算失败: {e}")
            # 提供更详细的错误信息
            import traceback
            traceback.print_exc()
            return {
                'overall': {'f1': 0, 'precision': 0, 'recall': 0, 'accuracy': 0},
                'per_entity': {}
            }
    
    def save_model(self, output_dir: str, model_type: str = "bert_bilstm_crf"):
        """保存模型和配置"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存模型权重
        model_path = os.path.join(output_dir, "pytorch_model.bin")
        if isinstance(self.model, nn.DataParallel):
            torch.save(self.model.module.state_dict(), model_path)
        else:
            torch.save(self.model.state_dict(), model_path)
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存配置
        config = {
            "model_type": model_type,
            "num_labels": self.model.classifier.out_features
            if not isinstance(self.model, nn.DataParallel)
            else self.model.module.classifier.out_features,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "model_config": {
                "hidden_dropout_prob": 0.3,
                "attention_probs_dropout_prob": 0.1,
                "lstm_hidden_size": 256,
                "lstm_layers": 2,
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
            json.dump(config, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)

        # 保存训练信息
        info_path = os.path.join(output_dir, "training_info.json")
        training_info = {
            "best_val_loss": self.best_val_loss,
            "best_val_f1": self.best_val_f1,
            "device": str(self.device),
            "timestamp": pd.Timestamp.now().isoformat(),
            "training_history": self.training_history
        }
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
        
        print(f"模型已保存到: {output_dir}")

    @classmethod
    def load_model(cls, model_dir: str):
        """从保存的目录加载模型用于推理"""
        config_path = os.path.join(model_dir, "config.json")
        model_path = os.path.join(model_dir, "pytorch_model.bin")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        trainer = cls.__new__(cls)
        trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        trainer.label2id = config['label2id']
        trainer.id2label = {int(v): k for k, v in config['label2id'].items()}
        trainer.best_val_f1 = 0
        trainer.best_val_loss = float('inf')
        trainer.best_model_state = None
        trainer.training_history = []
        trainer.train_loader = None
        trainer.val_loader = None
        trainer.test_loader = None

        num_labels = config['num_labels']
        model_cfg = config.get('model_config', {})
        trainer.model = BERTBiLSTMCRFEntityRecognizer(
            num_labels=num_labels,
            lstm_hidden_size=model_cfg.get('lstm_hidden_size', 256),
            lstm_layers=model_cfg.get('lstm_layers', 2),
        )
        trainer.model.load_state_dict(torch.load(model_path, map_location=trainer.device))
        trainer.model.to(trainer.device)
        trainer.model.eval()
        print(f"模型已从 {model_dir} 加载")
        return trainer

    def predict(self, texts, max_length: int = None):
        """
        对文本列表进行 NER 预测。

        Args:
            texts: 字符串或字符串列表
            max_length: 最大序列长度

        Returns:
            列表, 每个元素是 [(实体文本, 实体类型, 起始位置, 结束位置), ...]
        """
        if self.model is None:
            raise ValueError("模型未加载")
        if isinstance(texts, str):
            texts = [texts]
        if max_length is None:
            max_length = Config.MAX_SEQ_LENGTH

        self.model.eval()
        all_entities = []

        with torch.no_grad():
            for text in texts:
                chars = list(text[:max_length - 2])
                encoding = self.tokenizer(
                    chars, is_split_into_words=True,
                    max_length=max_length, padding='max_length',
                    truncation=True, return_tensors='pt',
                )
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                predictions = self.model(input_ids, attention_mask)
                if isinstance(predictions, list):
                    pred_ids = predictions[0]
                else:
                    pred_ids = predictions[0].tolist()

                # 解码 BIOES 标签为实体
                labels = [self.id2label.get(pid, 'O') for pid in pred_ids]
                # 去掉 [CLS] 和 [SEP] 位置的标签
                labels = labels[1:len(chars) + 1]

                entities = []
                i = 0
                while i < len(labels):
                    tag = labels[i]
                    if tag.startswith('S-'):
                        etype = tag[2:]
                        entities.append((text[i], etype, i, i + 1))
                        i += 1
                    elif tag.startswith('B-'):
                        etype = tag[2:]
                        start = i
                        i += 1
                        while i < len(labels) and (labels[i].startswith('I-') or labels[i].startswith('E-')):
                            if labels[i].startswith('E-'):
                                i += 1
                                break
                            i += 1
                        entities.append((text[start:i], etype, start, i))
                    else:
                        i += 1

                all_entities.append(entities)

        return all_entities

def main():
    """主训练函数"""
    print("开始高级BERT模型训练...")
    print(f"使用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # 准备训练数据
    from enhanced_data_processor import EnhancedDataProcessor
    from data_loader import MathDataLoader
    
    processor = EnhancedDataProcessor()
    loader = MathDataLoader()
    
    # 生成字级别训练数据
    char_train_data_path = os.path.join(Config.OUTPUT_DIR, "math_ner_char_training_data.jsonl")
    if not os.path.exists(char_train_data_path):
        print("生成字级别训练数据...")
        data = loader.load_all_data()
        training_df = processor.create_char_level_ner_training_data(data)
        processor.save_char_level_training_data(training_df, char_train_data_path)
    
    # 初始化训练器
    trainer = AdvancedBERTTrainer()
    
    # 准备数据（使用数据增强和BIOES标签）
    trainer.prepare_data(
        char_train_data_path, 
        batch_size=Config.BATCH_SIZE, 
        val_split=0.1,
        test_split=0.1,
        use_bioes=True,
        augment=True
    )
    
    # 初始化模型
    # 从训练器的full_dataset中获取标签映射
    num_labels = len(trainer.full_dataset.label2id)
    trainer.initialize_model(num_labels, model_type="bert_bilstm_crf")
    
    # 保存标签映射到训练器，以便后续使用
    trainer.label2id = trainer.full_dataset.label2id
    trainer.id2label = trainer.full_dataset.id2label
    
    # 训练模型
    history = trainer.train(
        epochs=Config.EPOCHS, 
        learning_rate=Config.LEARNING_RATE,
        warmup_ratio=0.1,
        early_stopping_patience=5,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0
    )
    
    # 测试模型
    test_metrics = trainer.test()
    
    # 保存模型
    model_dir = os.path.join(Config.MODEL_DIR, "math_ner_advanced_model")
    trainer.save_model(model_dir)
    
    # 保存训练历史
    history_path = os.path.join(Config.OUTPUT_DIR, "advanced_training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    print("训练完成！")

if __name__ == "__main__":
    main()