#!/usr/bin/env python3
"""
测试高级NER模型训练和实体识别
这个脚本用于测试新的BERT训练代码v2和实体抽取代码v2
"""

import os
import json
import torch
import pandas as pd
from enhanced_data_processor import EnhancedDataProcessor
from data_loader import MathDataLoader
from bert_trainer_v2 import AdvancedBERTTrainer
from entity_extractor_kggen import KGGenEnhancedEntityExtractor as AdvancedEntityExtractor
from config import Config
import shutil

def create_test_data():
    """使用真实数据文件创建测试数据"""
    print("使用真实数据文件创建测试数据...")
    
    # 指定要使用的数据文件
    data_files = ['7.1.docx', '7.2.docx', '7.11.docx']
    
    data_dir = Config.DATA_DIR
    
    # 详细检查数据文件是否存在
    print(f"检查数据目录: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        return create_fallback_test_data()
    
    missing_files = []
    existing_files = []
    for file_name in data_files:
        src_path = os.path.join(data_dir, file_name)
        if not os.path.exists(src_path):
            missing_files.append(file_name)
            print(f"警告: 文件不存在 {src_path}")
        else:
            existing_files.append(file_name)
            file_size = os.path.getsize(src_path)
            print(f"找到文件: {src_path} (大小: {file_size} 字节)")
    
    if missing_files:
        print(f"以下文件不存在: {missing_files}")
        if len(existing_files) == 0:
            print("没有找到任何文件，使用后备测试数据")
            return create_fallback_test_data()
        else:
            print("继续使用存在的文件")
    
    # 创建临时目录存放测试数据
    test_data_dir = os.path.join('output', 'test_data_advanced')
    
    # 彻底删除目录（如果存在）
    if os.path.exists(test_data_dir):
        try:
            shutil.rmtree(test_data_dir)
            print(f"已彻底删除目录: {test_data_dir}")
        except Exception as e:
            print(f"删除目录失败: {e}")
            return create_fallback_test_data()
    
    # 创建新目录
    try:
        os.makedirs(test_data_dir, exist_ok=True)
        print(f"创建干净临时目录: {test_data_dir}")
    except Exception as e:
        print(f"创建目录失败: {e}")
        return create_fallback_test_data()
    
    # 复制存在的文件到测试目录
    copied_files = []
    for file_name in existing_files:
        src_path = os.path.join(data_dir, file_name)
        dst_path = os.path.join(test_data_dir, file_name)
        
        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, dst_path)
                copied_files.append(file_name)
                print(f"已复制: {file_name} -> {dst_path}")
            except Exception as e:
                print(f"复制文件 {file_name} 失败: {e}")
        else:
            print(f"错误: 源文件不存在 {src_path}")
    
    # 使用数据加载器处理这些文件
    loader = MathDataLoader()
    loader.data_dir = test_data_dir
    
    try:
        # 加载数据
        print("使用MathDataLoader加载数据...")
        data = loader.load_all_data()
        print(f"共加载 {len(data)} 个文件")
        
        if len(data) == 0:
            print("警告: 没有加载到任何数据，使用后备数据")
            return create_fallback_test_data()
        
        # 使用增强处理器创建训练数据
        processor = EnhancedDataProcessor()
        processor.data_dir = test_data_dir
        
        # 创建字级别训练数据（使用BIOES标签）
        print("创建字级别训练数据（BIOES标签）...")
        training_df = processor.create_char_level_ner_training_data(data, use_bioes=True)
        
        if len(training_df) == 0:
            print("警告: 没有生成训练数据，使用后备数据")
            return create_fallback_test_data()
        
        # 保存训练数据
        test_data_path = os.path.join('output', 'test_char_data_advanced.jsonl')
        processor.save_char_level_training_data(training_df, test_data_path)
        
        print(f"测试数据已保存到: {test_data_path}, 样本数: {len(training_df)}")
        return test_data_path
        
    except Exception as e:
        print(f"创建测试数据失败: {e}")
        import traceback
        traceback.print_exc()
        print("使用后备测试数据")
        return create_fallback_test_data()

def create_fallback_test_data():
    """创建后备测试数据（当真实数据不可用时）"""
    print("创建后备测试数据（使用BIOES标签）...")
    
    # 示例数学文本数据
    test_texts = [
        "勾股定理指出：在直角三角形中，两直角边的平方和等于斜边的平方，即 a² + b² = c²",
        "二次函数的一般形式是 y = ax² + bx + c，其中a、b、c是常数",
        "解一元二次方程可以使用求根公式 x = [-b ± √(b² - 4ac)] / 2a",
        "相似三角形的性质：对应角相等，对应边成比例",
        "圆的面积公式是 S = πr²，其中r是半径，π是圆周率"
    ]
    
    # 创建字级别的标注数据（使用BIOES标签）
    test_data = []
    for text in test_texts:
        char_labels = ['O'] * len(text)
        
        # 简单规则标注（使用BIOES）
        if "勾股定理" in text:
            start = text.find("勾股定理")
            end = start + len("勾股定理")
            if end - start == 1:
                char_labels[start] = 'S-THEOREM'  # 单字实体
            else:
                char_labels[start] = 'B-THEOREM'  # 开始
                for i in range(start + 1, end - 1):
                    char_labels[i] = 'I-THEOREM'  # 中间
                char_labels[end - 1] = 'E-THEOREM'  # 结束
        
        if "a² + b² = c²" in text:
            start = text.find("a² + b² = c²")
            end = start + len("a² + b² = c²")
            char_labels[start] = 'B-FORMULA'
            for i in range(start + 1, end - 1):
                char_labels[i] = 'I-FORMULA'
            char_labels[end - 1] = 'E-FORMULA'
        
        if "二次函数" in text:
            start = text.find("二次函数")
            end = start + len("二次函数")
            char_labels[start] = 'B-CONCEPT'
            for i in range(start + 1, end - 1):
                char_labels[i] = 'I-CONCEPT'
            char_labels[end - 1] = 'E-CONCEPT'
        
        if "求根公式" in text:
            start = text.find("求根公式")
            end = start + len("求根公式")
            char_labels[start] = 'B-METHOD'
            for i in range(start + 1, end - 1):
                char_labels[i] = 'I-METHOD'
            char_labels[end - 1] = 'E-METHOD'
        
        test_data.append({
            'text': text,
            'char_labels': char_labels
        })
    
    # 保存测试数据
    test_data_path = os.path.join('output', 'test_char_data_advanced.jsonl')
    os.makedirs('output', exist_ok=True)
    
    with open(test_data_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"后备测试数据已保存到: {test_data_path}")
    return test_data_path

def test_advanced_training():
    """测试高级训练过程"""
    print("\n=== 测试高级训练过程 ===")
    
    # 创建测试数据
    test_data_path = create_test_data()
    
    if not os.path.exists(test_data_path):
        print(f"错误: 测试数据文件不存在: {test_data_path}")
        return None
    
    # 初始化高级训练器
    trainer = AdvancedBERTTrainer()
    
    try:
        # 准备数据（使用数据增强和BIOES标签）
        print("准备数据（使用数据增强和BIOES标签）...")
        trainer.prepare_data(
            test_data_path, 
            batch_size=2, 
            val_split=0.3,
            use_bioes=True,
            augment=True
        )
        
        # 初始化模型
        num_labels = len(trainer.train_loader.dataset.label2id)
        trainer.initialize_model(num_labels, model_type="bert_bilstm_crf")
        
        # 训练模型（小规模训练）
        print("开始训练模型...")
        history = trainer.train(
            epochs=2, 
            learning_rate=2e-5,
            warmup_ratio=0.1,
            early_stopping_patience=2
        )
        
        # 保存模型
        model_dir = os.path.join('models', 'test_ner_advanced_model')
        os.makedirs(model_dir, exist_ok=True)
        trainer.save_model(model_dir, model_type="bert_bilstm_crf")
        
        print("高级训练测试完成！")
        return model_dir
        
    except Exception as e:
        print(f"高级训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_advanced_entity_extraction(model_dir):
    """测试高级实体提取"""
    print("\n=== 测试高级实体提取 ===")
    
    try:
        # 初始化高级实体提取器（启用词性标注）
        extractor = AdvancedEntityExtractor(model_dir, use_pos_tagging=True)
        
        # 测试文本
        test_texts = [
            "勾股定理是一个重要的数学定理，公式为 a² + b² = c²",
            "二次函数的标准形式是 y = ax² + bx + c，其中a、b、c是常数且a ≠ 0",
            "使用求根公式可以解一元二次方程，这是代数学中的基本方法",
            "微积分基本定理建立了微分和积分之间的关系"
        ]
        
        for i, text in enumerate(test_texts):
            print(f"\n测试文本 {i+1}: {text}")
            entities = extractor.extract_entities(text)
            
            if entities:
                print("提取到的实体:")
                for entity in entities:
                    print(f"  {entity['type']}: '{entity['text']}' "
                          f"(位置: {entity['start']}-{entity['end']}, "
                          f"来源: {entity['source']}, 置信度: {entity.get('confidence', 1.0):.2f})")
            else:
                print("未提取到实体")
        
        return True
        
    except Exception as e:
        print(f"高级实体提取测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_extraction(model_dir):
    """测试批量实体提取"""
    print("\n=== 测试批量实体提取 ===")
    
    try:
        extractor = AdvancedEntityExtractor(model_dir)
        
        # 批量文本
        batch_texts = [
            "勾股定理：直角三角形斜边平方等于两直角边平方和",
            "二次函数图像为抛物线，顶点坐标为(-b/2a, c-b²/4a)",
            "求根公式用于解一元二次方程",
            "相似三角形对应角相等，对应边成比例"
        ]
        
        print("批量提取实体...")
        all_entities = extractor.batch_extract(batch_texts, batch_size=2)
        
        for i, entities in enumerate(all_entities):
            print(f"\n文本 {i+1}: {batch_texts[i]}")
            print(f"提取到 {len(entities)} 个实体:")
            for entity in entities:
                print(f"  {entity['type']}: '{entity['text']}'")
        
        return True
        
    except Exception as e:
        print(f"批量提取测试失败: {e}")
        return False

def test_without_model():
    """测试无模型时的实体提取（仅使用规则和词性标注）"""
    print("\n=== 测试无模型实体提取 ===")
    
    try:
        # 不指定模型目录，只使用规则和词性标注
        extractor = AdvancedEntityExtractor(use_pos_tagging=True)
        
        test_text = "勾股定理是数学中的重要定理，公式表示为 a² + b² = c²"
        print(f"测试文本: {test_text}")
        
        entities = extractor.extract_entities(test_text, use_ensemble=True)
        
        if entities:
            print("提取到的实体（仅规则和词性标注）:")
            for entity in entities:
                print(f"  {entity['type']}: '{entity['text']}' (来源: {entity['source']})")
        else:
            print("未提取到实体")
        
        return True
        
    except Exception as e:
        print(f"无模型提取测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("高级NER模型测试工具")
    print("=" * 50)
    
    # 检查CUDA可用性
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 测试无模型实体提取
    test_without_model()
    
    # 检查是否存在已训练的高级模型
    model_dir = os.path.join('models', 'test_ner_advanced_model')
    if os.path.exists(model_dir):
        print(f"\n找到已训练的高级模型: {model_dir}")
        print("直接使用现有模型进行测试...")
        
        # 测试实体提取
        test_advanced_entity_extraction(model_dir)
        
        # 测试批量提取
        test_batch_extraction(model_dir)
        
    else:
        print("\n未找到已训练的高级模型，开始训练过程...")
        
        # 训练模型
        model_dir = test_advanced_training()
        
        if model_dir:
            # 测试实体提取
            test_advanced_entity_extraction(model_dir)
            
            # 测试批量提取
            test_batch_extraction(model_dir)
        else:
            print("训练失败，无法进行实体提取测试")
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()