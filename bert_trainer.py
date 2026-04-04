import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from TorchCRF import CRF
import pandas as pd
from typing import List, Dict
from config import Config
from tqdm import tqdm

class MathNERDataset(Dataset):
    """数学NER数据集类"""
    def __init__(self, data_path: str, tokenizer: BertTokenizer, max_length: int = 128):
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
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        labels = item['ner_tags']
        
        # Tokenize文本
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 创建标签序列
        label_ids = []
        for i, word in enumerate(tokens):
            word_tokens = self.tokenizer.tokenize(word)
            if i < len(labels):
                label = labels[i]
                label_id = self.label2id.get(label, 0)  # 默认O
                label_ids.extend([label_id] * len(word_tokens))
            else:
                label_ids.extend([0] * len(word_tokens))
        
        # 填充或截断标签序列
        label_ids = label_ids[:self.max_length]
        if len(label_ids) < self.max_length:
            label_ids.extend([0] * (self.max_length - len(label_ids)))
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

class BERTCRFEntityRecognizer(nn.Module):
    """BERT+CRF实体识别模型"""
    def __init__(self, num_labels: int, bert_model_name: str = None):
        super(BERTCRFEntityRecognizer, self).__init__()
        if bert_model_name is None:
            bert_model_name = Config.BERT_MODEL
        
        self.bert = BertModel.from_pretrained(bert_model_name)
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

class BERTTrainer:
    """BERT模型训练器"""
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = Config.BERT_MODEL
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None
        self.train_loader = None
        self.val_loader = None
        
    def prepare_data(self, train_data_path: str, val_data_path: str = None, batch_size: int = 16):
        """准备训练和验证数据"""
        train_dataset = MathNERDataset(train_data_path, self.tokenizer, Config.MAX_SEQ_LENGTH)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_data_path and os.path.exists(val_data_path):
            val_dataset = MathNERDataset(val_data_path, self.tokenizer, Config.MAX_SEQ_LENGTH)
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            self.val_loader = None
            print("警告: 未提供验证数据，将只进行训练")
    
    def initialize_model(self, num_labels: int):
        """初始化模型"""
        self.model = BERTCRFEntityRecognizer(num_labels)
        self.model.to(self.device)
    
    def train(self, epochs: int = 3, learning_rate: float = 2e-5):
        """训练模型"""
        if self.model is None:
            raise ValueError("模型未初始化，请先调用initialize_model")
        if self.train_loader is None:
            raise ValueError("数据未准备，请先调用prepare_data")
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(self.train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                loss = self.model(input_ids, attention_mask, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} 完成，平均损失: {avg_loss:.4f}")
            
            # 每个epoch后验证
            if self.val_loader:
                val_loss = self.validate()
                print(f"验证损失: {val_loss:.4f}")
    
    def validate(self):
        """验证模型"""
        if self.val_loader is None:
            return 0
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                loss = self.model(input_ids, attention_mask, labels)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.model.train()
        return avg_loss
    
    def save_model(self, output_dir: str):
        """保存模型"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存模型权重
        model_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存配置
        config = {
            "model_type": "bert-crf",
            "num_labels": self.model.classifier.out_features,
            "label2id": self.train_loader.dataset.label2id if self.train_loader else {},
            "id2label": self.train_loader.dataset.id2label if self.train_loader else {}
        }
        
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"模型已保存到: {output_dir}")

def prepare_training_data():
    """准备训练数据（如果尚未生成）"""
    from enhanced_data_processor import EnhancedDataProcessor
    from data_loader import MathDataLoader
    
    processor = EnhancedDataProcessor()
    loader = MathDataLoader()
    
    # 检查训练数据是否已存在
    train_data_path = os.path.join(Config.OUTPUT_DIR, "math_ner_training_data.jsonl")
    clean_data_path = os.path.join(Config.OUTPUT_DIR, "math_ner_training_data_clean.jsonl")
    
    if not os.path.exists(train_data_path):
        print("训练数据不存在，正在生成...")
        data = loader.load_all_data()
        training_df = processor.create_ner_training_data(data)
        processor.save_training_data(training_df, train_data_path)
    else:
        print("训练数据已存在，直接使用")
    
    # 优先使用清理后的数据
    if os.path.exists(clean_data_path):
        print("使用清理后的训练数据")
        return clean_data_path
    
    return train_data_path

def main():
    """主训练函数"""
    print("开始BERT模型训练...")
    print(f"使用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # 准备训练数据
    train_data_path = prepare_training_data()
    
    # 初始化训练器
    trainer = BERTTrainer()
    
    # 准备数据
    trainer.prepare_data(train_data_path, batch_size=Config.BATCH_SIZE)
    
    # 初始化模型（根据数据集的标签数量）
    num_labels = len(trainer.train_loader.dataset.label2id)
    trainer.initialize_model(num_labels)
    
    # 训练模型
    trainer.train(epochs=3, learning_rate=Config.LEARNING_RATE)
    
    # 保存模型
    model_dir = os.path.join(Config.MODEL_DIR, "math_ner_model")
    trainer.save_model(model_dir)
    
    print("训练完成！")

if __name__ == "__main__":
    main()