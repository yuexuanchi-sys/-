import json
import os
from config import Config

def clean_training_data():
    """清理训练数据，移除空token的样本"""
    input_path = os.path.join(Config.OUTPUT_DIR, "math_ner_training_data.jsonl")
    output_path = os.path.join(Config.OUTPUT_DIR, "math_ner_training_data_clean.jsonl")
    
    if not os.path.exists(input_path):
        print(f"训练数据文件不存在: {input_path}")
        return None
    
    print(f"清理训练数据: {input_path}")
    
    total_count = 0
    valid_count = 0
    empty_token_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            total_count += 1
            data = json.loads(line)
            
            # 检查tokens是否为空或只包含空字符串
            tokens = data.get('tokens', [])
            if not tokens or all(token.strip() == '' for token in tokens):
                empty_token_count += 1
                continue
            
            # 检查NER标签是否与tokens长度匹配
            ner_tags = data.get('ner_tags', [])
            if len(ner_tags) != len(tokens):
                # 如果不匹配，调整NER标签长度
                if len(ner_tags) > len(tokens):
                    data['ner_tags'] = ner_tags[:len(tokens)]
                else:
                    data['ner_tags'] = ner_tags + ['O'] * (len(tokens) - len(ner_tags))
            
            # 写入有效的样本
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            valid_count += 1
    
    print(f"清理完成:")
    print(f"  总样本数: {total_count}")
    print(f"  有效样本数: {valid_count}")
    print(f"  空token样本数: {empty_token_count}")
    print(f"  清理后数据保存到: {output_path}")
    
    return output_path

def validate_training_data(data_path):
    """验证训练数据质量"""
    print(f"\n验证训练数据: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        sample_count = 0
        for line in f:
            data = json.loads(line)
            tokens = data.get('tokens', [])
            ner_tags = data.get('ner_tags', [])
            
            if len(tokens) != len(ner_tags):
                print(f"警告: 样本 {sample_count} tokens和ner_tags长度不匹配")
                print(f"  tokens: {len(tokens)}, ner_tags: {len(ner_tags)}")
            
            if sample_count < 3:  # 显示前3个样本作为示例
                print(f"样本 {sample_count}:")
                print(f"  tokens: {tokens[:5]}...")  # 显示前5个token
                print(f"  ner_tags: {ner_tags[:5]}...")  # 显示前5个标签
                print(f"  句子: {data.get('sentence', '')[:50]}...")  # 显示前50个字符
                print()
            
            sample_count += 1
    
    print(f"总共验证了 {sample_count} 个样本")

if __name__ == "__main__":
    # 清理训练数据
    clean_data_path = clean_training_data()
    
    if clean_data_path:
        # 验证清理后的数据
        validate_training_data(clean_data_path)
        
        print("\n清理完成！现在可以运行BERT训练:")
        print("python math_knowledge_extraction_pipeline.py")