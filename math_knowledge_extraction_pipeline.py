import os
import argparse
from typing import Dict, List
from config import Config
from data_loader import MathDataLoader
from enhanced_data_processor import EnhancedDataProcessor
from bert_trainer_v2 import AdvancedBERTTrainer as BERTTrainer

class MathKnowledgeExtractionPipeline:
    """初中数学学科知识抽取完整管道"""
    
    def __init__(self):
        self.data_loader = MathDataLoader()
        self.data_processor = EnhancedDataProcessor()
        self.bert_trainer = BERTTrainer()  # AdvancedBERTTrainer
        
    def run_full_pipeline(self, train_bert: bool = True):
        """运行完整的数据处理和训练管道"""
        print("=" * 60)
        print("初中数学学科知识抽取管道")
        print("=" * 60)
        
        # 步骤1: 数据加载和预处理
        print("\n1. 数据加载和预处理...")
        data = self.data_loader.load_all_data()
        print(f"  成功加载 {len(data)} 个数学教材文件")
        
        # 步骤2: 数据清洗和增强处理
        print("\n2. 数据清洗和增强处理...")
        processed_data = []
        for item in data:
            cleaned_content = self.data_processor.advanced_text_clean(item['content'])
            item['cleaned_content'] = cleaned_content
            processed_data.append(item)
        print("  数据清洗完成")
        
        # 步骤3: 分词和词性标注
        print("\n3. 分词和词性标注...")
        sample_text = processed_data[0]['cleaned_content'][:200] if processed_data else ""
        if sample_text:
            words_tags = self.data_processor.segment_and_tag(sample_text)
            print(f"  示例分词结果: {' '.join([f'{w}/{t}' for w, t in words_tags[:10]])}...")
        
        # 步骤4: 构建训练数据集
        print("\n4. 构建NER训练数据集...")
        train_data_path = os.path.join(Config.OUTPUT_DIR, "math_ner_training_data.jsonl")
        
        if not os.path.exists(train_data_path):
            training_df = self.data_processor.create_ner_training_data(processed_data)
            self.data_processor.save_training_data(training_df, train_data_path)
            print(f"  训练数据集已创建: {len(training_df)} 条样本")
        else:
            print("  训练数据集已存在，跳过创建")
        
        # 步骤5: BERT模型训练
        if train_bert:
            print("\n5. BERT模型训练...")
            try:
                self.bert_trainer.prepare_data(train_data_path, batch_size=Config.BATCH_SIZE)
                
                # 初始化模型
                num_labels = len(self.bert_trainer.train_loader.dataset.label2id)
                self.bert_trainer.initialize_model(num_labels)
                
                # 训练模型
                self.bert_trainer.train(epochs=3, learning_rate=Config.LEARNING_RATE)
                
                # 保存模型
                model_dir = os.path.join(Config.MODEL_DIR, "math_ner_model")
                self.bert_trainer.save_model(model_dir)
                print(f"  BERT模型训练完成，保存到: {model_dir}")
                
            except Exception as e:
                print(f"  BERT训练出错: {e}")
                print("  将继续进行规则基础的实体识别")
                train_bert = False
        
        # 步骤6: 实体识别演示
        print("\n6. 实体识别演示...")
        if processed_data:
            demo_text = processed_data[0]['cleaned_content'][:300]  # 取前300字符演示
            print(f"  演示文本: {demo_text[:100]}...")
            
            # 使用规则方法识别实体
            entities = self.data_processor.extract_math_entities(demo_text)
            print(f"  识别到 {len(entities)} 个实体:")
            for i, entity in enumerate(entities[:5]):  # 显示前5个
                print(f"    {i+1}. {entity['text']} ({entity['type']})")
        
        print("\n" + "=" * 60)
        print("管道执行完成！")
        print("=" * 60)
        
        return {
            'data_loaded': len(data),
            'train_data_created': os.path.exists(train_data_path),
            'bert_trained': train_bert
        }

def main():
    parser = argparse.ArgumentParser(description='初中数学学科知识抽取管道')
    parser.add_argument('--no-train', action='store_true', help='跳过BERT训练')
    parser.add_argument('--data-dir', type=str, default=Config.DATA_DIR, help='数据目录路径')
    parser.add_argument('--output-dir', type=str, default=Config.OUTPUT_DIR, help='输出目录路径')
    
    args = parser.parse_args()
    
    # 更新配置
    if args.data_dir != Config.DATA_DIR:
        Config.DATA_DIR = args.data_dir
    if args.output_dir != Config.OUTPUT_DIR:
        Config.OUTPUT_DIR = args.output_dir
    
    # 确保目录存在
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # 运行管道
    pipeline = MathKnowledgeExtractionPipeline()
    results = pipeline.run_full_pipeline(train_bert=not args.no_train)
    
    # 输出总结
    print("\n执行总结:")
    print(f"  - 加载文件: {results['data_loaded']} 个")
    print(f"  - 训练数据: {'已创建' if results['train_data_created'] else '未创建'}")
    print(f"  - BERT训练: {'已完成' if results['bert_trained'] else '已跳过'}")

if __name__ == "__main__":
    main()