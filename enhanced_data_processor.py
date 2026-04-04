import os
import re
import pandas as pd
import jieba
import jieba.posseg as pseg
from typing import List, Dict, Tuple
from config import Config
import json
import torch
import numpy as np
from typing import Optional

class POSTagger:
    """词性标注器，集成Seg_Pos-master和LTP的最佳实践"""
    
    def __init__(self, use_ltp: bool = True):
        self.use_ltp = use_ltp
        self.ltp_model = None
        self._init_models()
        
    def _init_models(self):
        """初始化词性标注模型"""
        if self.use_ltp:
            try:
                from ltp import LTP
                # 使用配置中的LTP模型路径
                self.ltp_model = LTP(Config.LTP_MODEL_PATH)
                print(f"LTP词性标注模型加载成功，模型路径: {Config.LTP_MODEL_PATH}")
            except ImportError:
                print("LTP未安装，使用jieba进行词性标注")
                self.use_ltp = False
            except Exception as e:
                print(f"LTP加载失败: {e}，使用jieba进行词性标注")
                self.use_ltp = False
        else:
            print("使用jieba进行词性标注")
    
    def tag(self, text: str) -> List[Tuple[str, str]]:
        """对文本进行词性标注"""
        if not text or not text.strip():
            return []
            
        if self.use_ltp and self.ltp_model:
            try:
                # 使用LTP进行词性标注
                seg, hidden = self.ltp_model.seg([text])
                pos = self.ltp_model.pos(hidden)
                return list(zip(seg[0], pos[0]))
            except Exception as e:
                print(f"LTP标注失败: {e}，回退到jieba")
                return self._tag_with_jieba(text)
        else:
            return self._tag_with_jieba(text)
    
    def _tag_with_jieba(self, text: str) -> List[Tuple[str, str]]:
        """使用jieba进行词性标注"""
        words = pseg.cut(text)
        return [(word, flag) for word, flag in words]

class EnhancedDataProcessor:
    def __init__(self):
        self.data_dir = Config.DATA_DIR
        # 初始化jieba词典，添加数学领域词汇
        self._init_jieba_dict()
        # 初始化词性标注器
        self.pos_tagger = POSTagger(use_ltp=True)
        
    def _init_jieba_dict(self):
        """初始化jieba分词词典，添加数学领域专业词汇"""
        math_terms = [
            '勾股定理', '三角函数', '二次函数', '一元二次方程', '平行四边形',
            '等腰三角形', '直角三角形', '相似三角形', '全等三角形', '圆周率',
            '绝对值', '相反数', '有理数', '无理数', '整数', '分数', '小数',
            '代数式', '不等式', '方程组', '概率', '统计', '几何图形', '坐标系',
            '数轴', '体积', '面积', '周长', '角度', '平行', '垂直', '相似', '全等'
        ]
        
        for term in math_terms:
            jieba.add_word(term, freq=1000, tag='n')
        
        # 加载停用词表
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> set:
        """加载停用词表"""
        # 优先查找项目根目录下的 stopwords.txt
        stopwords_path = os.path.join(os.path.dirname(__file__), 'stopwords.txt')
        stopwords = set()
        if os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                for line in f:
                    stopwords.add(line.strip())
        else:
            # 默认停用词
            default_stopwords = {'的', '了', '在', '是', '我', '有', '和', '就',
                                '不', '人', '都', '一', '一个', '上', '也', '很',
                                '到', '说', '要', '去', '你', '会', '着', '没有',
                                '看', '好', '自己', '这', '那', '他', '她', '它'}
            stopwords = default_stopwords
        return stopwords

    def advanced_text_clean(self, text: str) -> str:
        """增强的文本清洗"""
        if not text:
            return ""
        
        # 1. 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 2. 去除特殊字符和多余空格
        text = re.sub(r'[^\w\s\u4e00-\u9fffα-ωΑ-Ω√∛∜±∓×÷∙∘°′″∞∝∼≈≅≠≡≤≥≪≫∈∉⊂⊃∪∩∅∀∃∴∵∶∷∥⊥∠△□○◇♢♡♤♧♠♣♥♦+−×÷=<>≤≥≈≠≡√∛∜∞∝∫∮∑∏∂∇∆δϵζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ，。！？；："\'()（）【】《》]', ' ', text)
        
        # 3. 修复中文文本中的异常空格
        text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)
        
        # 4. 规范化空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 5. 去除首尾空格
        text = text.strip()
        
        return text

    def segment_and_tag(self, text: str, filter_stopwords: bool = True) -> List[Tuple[str, str]]:
        """中文分词和词性标注，集成Seg_Pos-master的最佳实践"""
        words_tags = self.pos_tagger.tag(text)
        
        if filter_stopwords:
            # 过滤停用词和单字非实词
            filtered_words = []
            for word, flag in words_tags:
                if (word not in self.stopwords and
                    len(word) > 1 and
                    flag not in ['x', 'w', 'u']):  # 过滤标点、助词等
                    filtered_words.append((word, flag))
            return filtered_words
        else:
            return words_tags

    def extract_math_entities(self, text: str) -> List[Dict]:
        """提取数学实体（基于规则和词性）"""
        words_tags = self.segment_and_tag(text)
        entities = []
        
        # 基于词性模式识别实体
        for i, (word, tag) in enumerate(words_tags):
            # 名词大概率是概念
            if tag.startswith('n') and len(word) > 1:
                entities.append({
                    'text': word,
                    'type': 'CONCEPT',
                    'source': 'pos_based'
                })
            # 动词可能是方法
            elif tag.startswith('v') and '法' in word:
                entities.append({
                    'text': word,
                    'type': 'METHOD',
                    'source': 'pos_based'
                })
        
        return entities

    def create_ner_training_data(self, data: List[Dict]) -> pd.DataFrame:
        """创建NER训练数据（词级别）"""
        training_samples = []
        
        for item in data:
            text = item['content']
            cleaned_text = self.advanced_text_clean(text)
            
            # 分句处理
            sentences = self.split_into_sentences(cleaned_text)
            
            for sentence in sentences:
                if len(sentence) > 10:  # 过滤过短句子
                    # 分词和词性标注
                    words_tags = self.segment_and_tag(sentence)
                    words = [wt[0] for wt in words_tags]
                    tags = [wt[1] for wt in words_tags]
                    
                    # 简单的实体标注（BIO格式）
                    labels = ['O'] * len(words)
                    for i, (word, tag) in enumerate(words_tags):
                        if tag.startswith('n') and len(word) > 1:
                            labels[i] = 'B-CONCEPT'
                    
                    training_samples.append({
                        'tokens': words,
                        'pos_tags': tags,
                        'ner_tags': labels,
                        'sentence': sentence,
                        'grade': item.get('grade'),
                        'chapter': item.get('chapter'),
                        'source_file': item.get('file_name')
                    })
        
        return pd.DataFrame(training_samples)

    def create_char_level_ner_training_data(self, data: List[Dict]) -> pd.DataFrame:
        """创建字级别的NER训练数据，使用BIO标注法"""
        training_samples = []
        
        for item in data:
            text = item['content']
            cleaned_text = self.advanced_text_clean(text)
            
            # 分句处理
            sentences = self.split_into_sentences(cleaned_text)
            
            for sentence in sentences:
                if len(sentence) > 10:  # 过滤过短句子
                    # 按字分割文本
                    chars = list(sentence)
                    
                    # 初始化所有字符标签为'O'
                    char_labels = ['O'] * len(chars)
                    
                    # 基于规则识别数学实体并标注
                    # 1. 识别数学公式
                    for pattern in Config.FORMULA_PATTERNS:
                        matches = re.finditer(pattern, sentence)
                        for match in matches:
                            start, end = match.span()
                            # 标注BIO格式
                            for i in range(start, end):
                                if i == start:
                                    char_labels[i] = 'B-FORMULA'
                                else:
                                    char_labels[i] = 'I-FORMULA'
                    
                    # 2. 识别定理
                    for pattern in Config.THEOREM_PATTERNS:
                        matches = re.finditer(pattern, sentence)
                        for match in matches:
                            if match.groups():
                                # 使用第一个分组的文本和位置
                                start = match.start(1)
                                end = match.end(1)
                            else:
                                start = match.start()
                                end = match.end()
                            
                            # 标注BIO格式
                            for i in range(start, end):
                                if i == start:
                                    char_labels[i] = 'B-THEOREM'
                                else:
                                    char_labels[i] = 'I-THEOREM'
                    
                    # 3. 基于关键词识别其他实体类型
                    math_keywords = {
                        'CONCEPT': ['概念', '定义', '含义', '意义', '性质', '特征'],
                        'METHOD': ['方法', '解法', '算法', '步骤', '技巧', '策略'],
                        'EXAMPLE': ['例子', '示例', '例题', '实例', '案例']
                    }
                    
                    for entity_type, keywords in math_keywords.items():
                        for keyword in keywords:
                            start = sentence.find(keyword)
                            if start != -1:
                                end = start + len(keyword)
                                # 标注BIO格式
                                for i in range(start, end):
                                    if i == start:
                                        char_labels[i] = f'B-{entity_type}'
                                    else:
                                        char_labels[i] = f'I-{entity_type}'
                    
                    training_samples.append({
                        'text': sentence,
                        'char_labels': char_labels,
                        'grade': item.get('grade'),
                        'chapter': item.get('chapter'),
                        'source_file': item.get('file_name')
                    })
        
        return pd.DataFrame(training_samples)

    def split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        sentences = re.split(r'[。！？!?;；]', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    def save_training_data(self, df: pd.DataFrame, output_path: str):
        """保存训练数据（词级别）"""
        # 保存为JSONL格式，适合BERT训练
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                record = {
                    'tokens': row['tokens'],
                    'ner_tags': row['ner_tags'],
                    'pos_tags': row['pos_tags'],
                    'metadata': {
                        'grade': row['grade'],
                        'chapter': row['chapter'],
                        'source_file': row['source_file']
                    }
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"训练数据已保存到: {output_path}")

    def save_char_level_training_data(self, df: pd.DataFrame, output_path: str):
        """保存字级别训练数据"""
        # 保存为JSONL格式，适合BERT训练
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                record = {
                    'text': row['text'],
                    'char_labels': row['char_labels'],
                    'metadata': {
                        'grade': row['grade'],
                        'chapter': row['chapter'],
                        'source_file': row['source_file']
                    }
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"字级别训练数据已保存到: {output_path}")

    def save_conll_format_data(self, df: pd.DataFrame, output_path: str, level: str = 'char'):
        """保存CoNLL格式的训练数据，兼容NER-Chinese-master项目"""
        with open(output_path, 'w', encoding='utf-8') as f:
            if level == 'char':
                # 字级别的CoNLL格式
                for _, row in df.iterrows():
                    text = row['text']
                    char_labels = row['char_labels']
                    
                    # 每个字符一行，格式: 字符 O 实体标签
                    for i, char in enumerate(text):
                        if i < len(char_labels):
                            label = char_labels[i]
                        else:
                            label = 'O'
                        f.write(f"{char} O {label}\n")
                    f.write('\n')  # 句子间空行
            else:
                # 词级别的CoNLL格式
                for _, row in df.iterrows():
                    tokens = row['tokens']
                    ner_tags = row['ner_tags']
                    pos_tags = row['pos_tags']
                    
                    for token, pos_tag, ner_tag in zip(tokens, pos_tags, ner_tags):
                        f.write(f"{token} {pos_tag} {ner_tag}\n")
                    f.write('\n')  # 句子间空行
        
        print(f"CoNLL格式数据已保存到: {output_path}")

    def load_conll_format_data(self, file_path: str) -> pd.DataFrame:
        """加载CoNLL格式的数据文件"""
        data = []
        current_sentence = []
        current_labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        char = parts[0]
                        label = parts[2]
                        current_sentence.append(char)
                        current_labels.append(label)
                else:
                    # 空行表示句子结束
                    if current_sentence:
                        data.append({
                            'text': ''.join(current_sentence),
                            'char_labels': current_labels
                        })
                    current_sentence = []
                    current_labels = []
            
            # 处理最后一个句子
            if current_sentence:
                data.append({
                    'text': ''.join(current_sentence),
                    'char_labels': current_labels
                })
        
        return pd.DataFrame(data)

    def convert_to_conll2003_format(self, df: pd.DataFrame, output_path: str):
        """转换为标准的CoNLL-2003格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                text = row['text']
                char_labels = row['char_labels']
                
                # CoNLL-2003格式: 词 POS 块标签 实体标签
                # 由于我们是字级别，将每个字视为一个词
                for i, char in enumerate(text):
                    if i < len(char_labels):
                        label = char_labels[i]
                    else:
                        label = 'O'
                    
                    # 简单的POS标签（可以后续增强）
                    pos_tag = 'NN' if len(char) > 1 else 'CH'
                    f.write(f"{char} {pos_tag} O {label}\n")
                f.write('\n')
        
        print(f"CoNLL-2003格式数据已保存到: {output_path}")

# 使用示例
if __name__ == "__main__":
    from data_loader import MathDataLoader
    
    processor = EnhancedDataProcessor()
    loader = MathDataLoader()
    
    try:
        # 加载数据
        print("加载数据...")
        data = loader.load_all_data()
        print(f"共加载 {len(data)} 个文件")
        
        # 创建NER训练数据
        print("创建训练数据...")
        training_df = processor.create_ner_training_data(data)
        print(f"创建了 {len(training_df)} 条训练样本")
        
        # 保存训练数据
        output_path = os.path.join(Config.OUTPUT_DIR, "math_ner_training_data.jsonl")
        processor.save_training_data(training_df, output_path)
        
        # 示例：处理单个文档并显示结果
        if data:
            sample_text = data[0]['content'][:500]  # 前500字符
            cleaned = processor.advanced_text_clean(sample_text)
            print(f"\n原始文本前500字符: {sample_text}")
            print(f"\n清洗后文本: {cleaned}")
            
            words_tags = processor.segment_and_tag(cleaned)
            print(f"\n分词和词性标注结果:")
            for word, tag in words_tags[:10]:  # 显示前10个
                print(f"{word}/{tag}", end=" ")
            print()
            
            entities = processor.extract_math_entities(cleaned)
            print(f"\n提取的实体:")
            for entity in entities[:5]:  # 显示前5个
                print(f"{entity['text']} ({entity['type']})")
                
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()