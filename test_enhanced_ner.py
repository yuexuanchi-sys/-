#!/usr/bin/env python3
"""
测试增强的NER模型训练和实体识别
这个脚本用于测试新的BERT训练代码和实体抽取代码
"""

import os
import json
import torch
import pandas as pd
from enhanced_data_processor import EnhancedDataProcessor
from data_loader import MathDataLoader
from bert_trainer_v2 import AdvancedBERTTrainer as EnhancedBERTTrainer
from entity_recognizer_enhanced import EnhancedEntityRecognizer
from config import Config
import shutil

def create_test_data():
    """使用真实数据文件创建测试数据"""
    print("使用真实数据文件创建测试数据...")
    
    # 指定要使用的数据文件：只用7.1.docx和7.2.docx
    data_files = ['7.11.docx', '8.11.docx']
    
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
    
    # 创建临时目录存放测试数据 - 彻底清理确保只有指定文件
    test_data_dir = os.path.join('output', 'test_data')
    
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
                # 检查复制后的文件
                if os.path.exists(dst_path):
                    copied_size = os.path.getsize(dst_path)
                    print(f"复制后文件大小: {copied_size} 字节")
                else:
                    print(f"错误: 复制后文件不存在 {dst_path}")
            except Exception as e:
                print(f"复制文件 {file_name} 失败: {e}")
        else:
            print(f"错误: 源文件不存在 {src_path}")
    
    # 验证临时目录中的文件
    print("验证临时目录中的文件:")
    files_in_test_dir = os.listdir(test_data_dir)
    if len(files_in_test_dir) == 0:
        print("错误: 临时目录为空")
        return create_fallback_test_data()
    
    for file in files_in_test_dir:
        print(f"  - {file}")
    
    # 检查是否只有我们需要的文件
    unexpected_files = set(files_in_test_dir) - set(copied_files)
    if unexpected_files:
        print(f"警告: 发现未预期的文件: {unexpected_files}")
        # 删除未预期的文件
        for file in unexpected_files:
            try:
                os.remove(os.path.join(test_data_dir, file))
                print(f"已删除未预期文件: {file}")
            except Exception as e:
                print(f"删除文件 {file} 失败: {e}")
    
    # 使用数据加载器处理这些文件
    loader = MathDataLoader()
    loader.data_dir = test_data_dir  # 设置数据目录
    print(f"设置数据加载器目录: {test_data_dir}")
    
    # 确保加载器只处理当前目录
    if hasattr(loader, 'data_dir'):
        print(f"加载器数据目录确认: {loader.data_dir}")
    
    # 重写get_file_list方法以确保只处理我们指定的文件
    original_get_file_list = loader.get_file_list
    
    def get_file_list_override():
        """重写文件列表获取方法，只返回我们指定的文件"""
        files = []
        for file_name in existing_files:
            file_path = os.path.join(test_data_dir, file_name)
            if os.path.exists(file_path):
                files.append(file_path)
        print(f"重写文件列表方法返回的文件: {[os.path.basename(f) for f in files]}")
        return sorted(files)
    
    loader.get_file_list = get_file_list_override
    
    try:
        # 加载数据
        print("使用MathDataLoader加载数据...")
        data = loader.load_all_data()
        print(f"共加载 {len(data)} 个文件")
        
        if len(data) == 0:
            print("警告: 没有加载到任何数据，使用后备数据")
            return create_fallback_test_data()
        
        # 打印加载的文件信息用于调试
        print("加载的文件详情:")
        for i, item in enumerate(data):
            filename = item.get('filename', '未知')
            text_length = len(item.get('text', ''))
            print(f"  文件 {i+1}: {filename}, 文本长度: {text_length}")
            if text_length == 0:
                print(f"    警告: 文件 {filename} 的文本内容为空")
        
        # 使用增强处理器创建训练数据
        processor = EnhancedDataProcessor()
        processor.data_dir = test_data_dir
        
        # 创建字级别训练数据
        print("创建字级别训练数据...")
        training_df = processor.create_char_level_ner_training_data(data)
        
        if len(training_df) == 0:
            print("警告: 没有生成训练数据，使用后备数据")
            return create_fallback_test_data()
        
        # 保存训练数据
        test_data_path = os.path.join('output', 'test_char_data.jsonl')
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
    print("创建后备测试数据...")
    
    # 示例数学文本数据
    test_texts = [
        "勾股定理指出：在直角三角形中，两直角边的平方和等于斜边的平方，即 a² + b² = c²",
        "二次函数的一般形式是 y = ax² + bx + c，其中a、b、c是常数",
        "解一元二次方程可以使用求根公式 x = [-b ± √(b² - 4ac)] / 2a",
        "相似三角形的性质：对应角相等，对应边成比例",
        "圆的面积公式是 S = πr²，其中r是半径，π是圆周率"
    ]
    
    # 创建字级别的标注数据
    test_data = []
    for text in test_texts:
        char_labels = ['O'] * len(text)
        
        # 简单规则标注
        if "勾股定理" in text:
            start = text.find("勾股定理")
            end = start + len("勾股定理")
            for i in range(start, end):
                if i == start:
                    char_labels[i] = 'B-THEOREM'
                else:
                    char_labels[i] = 'I-THEOREM'
        
        if "a² + b² = c²" in text:
            start = text.find("a² + b² = c²")
            end = start + len("a² + b² = c²")
            for i in range(start, end):
                if i == start:
                    char_labels[i] = 'B-FORMULA'
                else:
                    char_labels[i] = 'I-FORMULA'
        
        if "二次函数" in text:
            start = text.find("二次函数")
            end = start + len("二次函数")
            for i in range(start, end):
                if i == start:
                    char_labels[i] = 'B-CONCEPT'
                else:
                    char_labels[i] = 'I-CONCEPT'
        
        if "求根公式" in text:
            start = text.find("求根公式")
            end = start + len("求根公式")
            for i in range(start, end):
                if i == start:
                    char_labels[i] = 'B-METHOD'
                else:
                    char_labels[i] = 'I-METHOD'
        
        test_data.append({
            'text': text,
            'char_labels': char_labels
        })
    
    # 保存测试数据
    test_data_path = os.path.join('output', 'test_char_data.jsonl')
    os.makedirs('output', exist_ok=True)
    
    with open(test_data_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"后备测试数据已保存到: {test_data_path}")
    return test_data_path

def test_training():
    """测试训练过程"""
    print("\n=== 测试训练过程 ===")
    
    # 创建测试数据
    test_data_path = create_test_data()
    
    # 详细检查测试数据文件
    if not os.path.exists(test_data_path):
        print(f"错误: 测试数据文件不存在: {test_data_path}")
        return None, None
        
    # 检查文件大小和内容
    file_size = os.path.getsize(test_data_path)
    print(f"测试数据文件大小: {file_size} 字节")
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        line_count = len(lines)
        print(f"测试数据文件行数: {line_count}")
        
        # 打印前几行内容用于调试
        if line_count > 0:
            print("前3行数据内容:")
            for i, line in enumerate(lines[:3]):
                print(f"行 {i+1}: {line.strip()}")
        else:
            print("文件为空")
    
    if line_count == 0:
        print("错误: 测试数据文件为空")
        return None, None
    
    # 初始化训练器
    trainer = EnhancedBERTTrainer()
    
    try:
        # 准备数据 - 添加更多调试信息
        print("准备数据...")
        trainer.prepare_data(test_data_path, batch_size=2, val_split=0.3)
        
        # 检查数据加载器是否有效
        if trainer.train_loader is None:
            print("错误: 训练数据加载器未初始化")
            return None, None
            
        print(f"训练数据集大小: {len(trainer.train_loader.dataset)}")
        print(f"验证数据集大小: {len(trainer.val_loader.dataset) if trainer.val_loader else 0}")
        
        # 检查标签映射 - 从数据集中获取
        if hasattr(trainer.train_loader.dataset, 'dataset'):
            # 如果使用了random_split，需要访问.dataset属性
            label2id = trainer.train_loader.dataset.dataset.label2id
        else:
            label2id = trainer.train_loader.dataset.label2id
        print(f"标签映射: {label2id}")

        # 初始化模型
        num_labels = len(label2id)
        trainer.initialize_model(num_labels)
        
        # 训练模型（小规模训练）
        print("开始训练模型...")
        trainer.train(epochs=2, learning_rate=2e-5, early_stopping_patience=2)
        
        # 保存模型
        model_dir = os.path.join('models', 'test_ner_model')
        os.makedirs(model_dir, exist_ok=True)
        trainer.save_model(model_dir)
        
        print("训练测试完成！")
        return model_dir, None
        
    except Exception as e:
        print(f"训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_entity_recognition(model_dir):
    """测试实体识别"""
    print("\n=== 测试实体识别 ===")
    
    try:
        # 初始化实体识别器
        recognizer = EnhancedEntityRecognizer(model_dir)
        
        # 测试文本
        test_texts = [
            "勾股定理是一个重要的数学定理，公式为 a² + b² = c²",
            "二次函数的标准形式是 y = ax² + bx + c",
            "使用求根公式可以解一元二次方程"
        ]
        
        for i, text in enumerate(test_texts):
            print(f"\n测试文本 {i+1}: {text}")
            entities = recognizer.predict_entities(text)
            
            if entities:
                print("识别到的实体:")
                for entity in entities:
                    print(f"  {entity['type']}: '{entity['text']}' (位置: {entity['start']}-{entity['end']}, 来源: {entity['source']})")
            else:
                print("未识别到实体")
        
        return True
        
    except Exception as e:
        print(f"实体识别测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline():
    """测试完整流程"""
    print("开始测试增强的NER模型完整流程...")
    
    # 测试训练
    model_dir, history = test_training()
    
    if model_dir:
        # 测试实体识别
        success = test_entity_recognition(model_dir)
        
        if success:
            print("\n✅ 完整流程测试成功！")
        else:
            print("\n❌ 实体识别测试失败")
    else:
        print("\n❌ 训练测试失败")

def test_with_existing_data():
    """使用现有数据测试"""
    print("\n=== 使用现有数据测试 ===")
    
    try:
        # 检查是否有现有数据
        data_path = os.path.join('output', 'math_ner_training_data.json')
        if os.path.exists(data_path):
            print(f"使用现有数据: {data_path}")
            
            # 初始化实体识别器（使用预训练模型）
            recognizer = EnhancedEntityRecognizer()
            
            # 测试一些示例文本
            test_text = "勾股定理指出：在直角三角形中，两直角边的平方和等于斜边的平方"
            entities = recognizer.predict_entities(test_text)
            
            print(f"测试文本: {test_text}")
            if entities:
                print("识别到的实体:")
                for entity in entities:
                    print(f"  {entity['type']}: '{entity['text']}' (来源: {entity['source']})")
            else:
                print("未识别到实体")
                
            return True
        else:
            print("没有找到现有训练数据")
            return False
            
    except Exception as e:
        print(f"现有数据测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_entity_recognition_on_files(model_dir, files_to_test):
    """对指定文件进行实体识别测试并保存结果到文件"""
    print(f"\n=== 对指定文件进行实体识别测试 ===")
    
    # 创建结果目录
    results_dir = os.path.join('output', 'entity_recognition_results')
    os.makedirs(results_dir, exist_ok=True)
    print(f"结果将保存到目录: {results_dir}")
    
    try:
        # 初始化实体识别器
        recognizer = EnhancedEntityRecognizer(model_dir)
        
        # 使用数据加载器读取指定文件
        loader = MathDataLoader()
        loader.data_dir = Config.DATA_DIR  # 设置原始数据目录
        
        for file_name in files_to_test:
            print(f"\n处理文件: {file_name}")
            file_path = os.path.join(Config.DATA_DIR, file_name)
            
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                continue
                
            # 读取文件内容
            content = loader.read_file_content(file_path)
            if not content or len(content.strip()) == 0:
                print(f"文件内容为空或读取失败: {file_name}")
                continue
                
            print(f"文件内容长度: {len(content)} 字符")
            
            # 对内容进行实体识别
            print("开始实体识别...")
            entities = recognizer.predict_entities(content)
            
            # 准备结果数据
            result_data = {
                'file_name': file_name,
                'content_length': len(content),
                'entities': entities,
                'entity_count': {},
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            if entities:
                print(f"在 {file_name} 中识别到的实体:")
                entity_count = {}
                for entity in entities:
                    entity_type = entity['type']
                    entity_count[entity_type] = entity_count.get(entity_type, 0) + 1
                    print(f"  {entity_type}: '{entity['text']}' (来源: {entity['source']})")
                
                result_data['entity_count'] = entity_count
                
                print(f"\n实体统计:")
                for entity_type, count in entity_count.items():
                    print(f"  {entity_type}: {count} 个")
            else:
                print("未识别到实体")
            
            # 保存结果到JSON文件
            result_filename = f"entity_results_{file_name.replace('.', '_')}.json"
            result_path = os.path.join(results_dir, result_filename)
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            print(f"结果已保存到: {result_path}")
                
    except Exception as e:
        print(f"实体识别测试失败: {e}")
        import traceback
        traceback.print_exc()

def check_existing_model():
    """检查是否存在已训练的模型"""
    model_dir = os.path.join('models', 'test_ner_model')
    if os.path.exists(model_dir):
        # 检查模型文件是否存在
        model_files = ['pytorch_model.bin', 'config.json', 'vocab.txt']
        for file in model_files:
            if not os.path.exists(os.path.join(model_dir, file)):
                return None
        return model_dir
    return None

if __name__ == "__main__":
    # 检查CUDA可用性
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 检查是否存在已训练的模型
    model_dir = check_existing_model()
    
    if model_dir:
        print(f"找到已训练的模型: {model_dir}")
        print("直接使用现有模型进行实体识别...")
        files_to_test = ['7.1.docx', '7.2.docx']
        test_entity_recognition_on_files(model_dir, files_to_test)
    else:
        print("未找到已训练的模型，开始训练过程...")
        # 测试1: 完整流程测试
        model_dir, history = test_training()
        
        # 如果训练成功，对指定文件进行实体识别
        if model_dir and os.path.exists(model_dir):
            files_to_test = ['7.11.docx', '8.11.docx']
            test_entity_recognition_on_files(model_dir, files_to_test)
        else:
            print("训练失败，无法进行实体识别测试")
    
    print("\n测试完成！")