"""
增强数据处理器 - 生成高质量 NER 训练数据

功能:
1. 文本清洗 (保留数学符号)
2. 分词和词性标注 (LTP / jieba)
3. 基于数学本体的自动标注 (BIOES 格式)
4. 多种输出格式 (JSONL / CoNLL / CoNLL-2003)
"""

import os
import re
import json
import logging
import jieba
import jieba.posseg as pseg
import pandas as pd
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 数学本体词典 — 用于自动标注 NER 训练数据
# ============================================================

MATH_ONTOLOGY = {
    'CONCEPT': [
        # 数与代数
        '有理数', '无理数', '实数', '整数', '正数', '负数', '分数', '小数', '自然数',
        '奇数', '偶数', '质数', '合数', '倒数', '相反数', '绝对值', '近似数', '有效数字',
        '数轴', '原点', '单位长度', '坐标', '坐标系', '坐标轴', '象限',
        '代数式', '单项式', '多项式', '整式', '分式', '二次根式', '根式',
        '系数', '次数', '常数项', '同类项',
        '方程', '等式', '不等式', '不等号', '解集',
        '一元一次方程', '二元一次方程组', '一元二次方程', '一元一次不等式',
        '分式方程', '方程组',
        '函数', '自变量', '因变量', '函数值', '定义域', '值域',
        '一次函数', '正比例函数', '反比例函数', '二次函数',
        '函数图象', '图象', '斜率', '截距',
        '算术平方根', '平方根', '立方根', '根号',
        '幂', '底数', '指数', '乘方', '科学记数法',
        # 图形与几何
        '点', '线', '面', '体', '直线', '射线', '线段', '曲线',
        '角', '锐角', '直角', '钝角', '平角', '周角', '余角', '补角',
        '对顶角', '同位角', '内错角', '同旁内角',
        '平行线', '垂线', '垂直', '平行', '相交',
        '三角形', '直角三角形', '等腰三角形', '等边三角形', '锐角三角形', '钝角三角形',
        '四边形', '平行四边形', '矩形', '菱形', '正方形', '梯形',
        '多边形', '正多边形', '五边形', '六边形',
        '圆', '半圆', '弧', '弦', '直径', '半径', '圆心', '圆弧',
        '扇形', '圆心角', '圆周角', '切线', '弧长',
        '棱柱', '棱锥', '棱台', '圆柱', '圆锥', '球',
        '长方体', '正方体', '长方形', '正方形',
        '面积', '周长', '体积', '表面积', '侧面积', '底面积',
        '对称', '轴对称', '中心对称', '对称轴', '对称中心',
        '全等', '相似', '全等三角形', '相似三角形',
        '旋转', '平移', '翻折', '位似',
        '投影', '视图', '三视图', '正视图', '侧视图', '俯视图', '展开图',
        '中点', '中线', '中位线', '高', '角平分线', '垂直平分线',
        '内角', '外角', '内角和', '外角和',
        '顶点', '边', '对角线',
        # 统计与概率
        '数据', '频数', '频率', '统计图', '条形图', '折线图', '扇形图', '直方图',
        '平均数', '中位数', '众数', '方差', '标准差', '极差',
        '概率', '随机事件', '必然事件', '不可能事件', '样本', '总体', '样本空间',
        '树状图', '列表法',
    ],
    'FORMULA': [
        '勾股定理公式', '二次公式', '求根公式', '韦达定理',
        '面积公式', '体积公式', '周长公式', '弧长公式', '扇形面积公式',
        '距离公式', '中点公式', '斜率公式',
        '完全平方公式', '平方差公式', '交叉相乘',
        'a²+b²=c²', 'S=πr²', 'C=2πr', 'V=lwh',
    ],
    'THEOREM': [
        '勾股定理', '逆定理', '勾股定理的逆定理',
        '三角形内角和定理', '外角定理',
        '平行线判定定理', '平行线性质定理',
        '等腰三角形性质定理', '等边三角形性质定理',
        '全等三角形判定', '相似三角形判定',
        'SSS', 'SAS', 'ASA', 'AAS', 'HL',
        'AA相似', 'SAS相似', 'SSS相似',
        '垂径定理', '圆周角定理', '切线长定理',
        '平行四边形判定定理', '矩形判定定理', '菱形判定定理',
    ],
    'METHOD': [
        '配方法', '公式法', '因式分解法', '十字相乘法',
        '代入消元法', '加减消元法', '换元法',
        '反证法', '数学归纳法', '数形结合',
        '作图法', '列表法', '树状图法',
        '解方程', '列方程', '移项', '合并同类项',
        '通分', '约分', '化简',
    ],
    'PROPERTY': [
        '交换律', '结合律', '分配律',
        '等式性质', '不等式性质',
        '乘法法则', '除法法则', '幂的运算法则',
        '三角形稳定性', '平行线性质',
    ],
}


class POSTagger:
    """词性标注器 (LTP / jieba 双引擎)"""

    def __init__(self, use_ltp: bool = True):
        self.use_ltp = use_ltp
        self.ltp_model = None
        self._init_models()

    def _init_models(self):
        if self.use_ltp:
            try:
                from ltp import LTP
                self.ltp_model = LTP(Config.LTP_MODEL_PATH)
                logger.info(f"LTP词性标注模型加载成功")
            except ImportError:
                logger.info("LTP未安装，使用jieba进行词性标注")
                self.use_ltp = False
            except Exception as e:
                logger.info(f"LTP加载失败: {e}，使用jieba进行词性标注")
                self.use_ltp = False
        else:
            logger.info("使用jieba进行词性标注")

    def tag(self, text: str) -> List[Tuple[str, str]]:
        if not text or not text.strip():
            return []
        if self.use_ltp and self.ltp_model:
            try:
                seg, hidden = self.ltp_model.seg([text])
                pos = self.ltp_model.pos(hidden)
                return list(zip(seg[0], pos[0]))
            except Exception:
                return self._tag_with_jieba(text)
        return self._tag_with_jieba(text)

    @staticmethod
    def _tag_with_jieba(text: str) -> List[Tuple[str, str]]:
        return [(word, flag) for word, flag in pseg.cut(text)]


class EnhancedDataProcessor:
    """增强数据处理器"""

    def __init__(self):
        self.data_dir = Config.DATA_DIR
        self._init_jieba_dict()
        self.pos_tagger = POSTagger(use_ltp=True)

        # 构建本体查找表 (用于自动标注)
        self.ontology_lookup: Dict[str, str] = {}
        for entity_type, terms in MATH_ONTOLOGY.items():
            for term in terms:
                self.ontology_lookup[term] = entity_type

        # 按长度降序排序，优先匹配长实体
        self.ontology_terms_sorted = sorted(
            self.ontology_lookup.keys(), key=len, reverse=True
        )

    def _init_jieba_dict(self):
        """初始化 jieba 数学领域词典"""
        all_terms = []
        for terms in MATH_ONTOLOGY.values():
            all_terms.extend(terms)

        for term in all_terms:
            jieba.add_word(term, freq=1000, tag='n')

        self.stopwords = self._load_stopwords()

    def _load_stopwords(self) -> Set[str]:
        stopwords_path = os.path.join(os.path.dirname(__file__), 'stopwords.txt')
        if os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                return {line.strip() for line in f}
        return {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
                '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
                '会', '着', '没有', '看', '好', '自己', '这', '那', '他', '她', '它'}

    # ================================================================
    #  文本清洗
    # ================================================================

    def advanced_text_clean(self, text: str) -> str:
        """增强文本清洗 (保留数学符号)"""
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', '', text)
        # 保留中文、字母、数字、数学符号、标点
        text = re.sub(
            r'[^\w\s\u4e00-\u9fffα-ωΑ-Ω√±×÷°′″∞≈≠≡≤≥∈⊂∪∩∅∀∃∴∵∥⊥∠△□○'
            r'+\-=<>()（）【】《》，。！？；：""'
            r"''／]",
            ' ', text
        )
        text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # ================================================================
    #  分词与标注
    # ================================================================

    def segment_and_tag(self, text: str, filter_stopwords: bool = True) -> List[Tuple[str, str]]:
        words_tags = self.pos_tagger.tag(text)
        if filter_stopwords:
            return [(w, t) for w, t in words_tags
                    if w not in self.stopwords and len(w) > 1 and t not in ('x', 'w', 'u')]
        return words_tags

    def extract_math_entities(self, text: str) -> List[Dict]:
        """基于规则和词性提取数学实体"""
        words_tags = self.segment_and_tag(text)
        entities = []
        for word, tag in words_tags:
            if tag.startswith('n') and len(word) > 1:
                entities.append({'text': word, 'type': 'CONCEPT', 'source': 'pos_based'})
            elif tag.startswith('v') and '法' in word:
                entities.append({'text': word, 'type': 'METHOD', 'source': 'pos_based'})
        return entities

    # ================================================================
    #  NER 训练数据生成 (BIOES 格式)
    # ================================================================

    def create_char_level_ner_training_data(self, data: List[Dict],
                                             scheme: str = 'bioes') -> pd.DataFrame:
        """
        生成字级别 NER 训练数据。

        标注策略:
        1. 本体词典精确匹配 (最优先)
        2. 正则模式匹配 (公式/定理)
        3. 未匹配字符标注为 O

        Args:
            data: [{content, grade, chapter, file_name}, ...]
            scheme: 'bioes' 或 'bio'
        """
        training_samples = []

        for item in data:
            text = item.get('content', '')
            cleaned = self.advanced_text_clean(text)
            sentences = self.split_into_sentences(cleaned)

            for sentence in sentences:
                if len(sentence) < 5:
                    continue

                labels = self._annotate_sentence(sentence, scheme)

                # 验证
                if len(labels) != len(sentence):
                    continue

                # 统计非O标签比例
                non_o = sum(1 for l in labels if l != 'O')
                if non_o == 0:
                    # 无实体句仍保留一部分作为负样本
                    if len(training_samples) % 3 != 0:
                        continue

                training_samples.append({
                    'text': sentence,
                    'char_labels': labels,
                    'grade': item.get('grade'),
                    'chapter': item.get('chapter'),
                    'source_file': item.get('file_name'),
                })

        logger.info(f"生成 {len(training_samples)} 条NER训练样本 (scheme={scheme})")
        return pd.DataFrame(training_samples)

    def _annotate_sentence(self, sentence: str, scheme: str = 'bioes') -> List[str]:
        """
        对单个句子进行自动标注

        优先级: 本体词典 > 公式模式 > 定理模式
        """
        n = len(sentence)
        labels = ['O'] * n
        occupied = [False] * n  # 防止重复标注

        # 1. 本体词典匹配 (优先匹配长实体)
        for term in self.ontology_terms_sorted:
            entity_type = self.ontology_lookup[term]
            start = 0
            while True:
                pos = sentence.find(term, start)
                if pos == -1:
                    break
                end = pos + len(term)
                # 检查是否已被标注
                if not any(occupied[pos:end]):
                    self._apply_labels(labels, occupied, pos, end, entity_type, scheme)
                start = pos + 1

        # 2. 公式模式
        for pattern in Config.FORMULA_PATTERNS:
            for match in re.finditer(pattern, sentence):
                start, end = match.span()
                if not any(occupied[start:end]):
                    self._apply_labels(labels, occupied, start, end, 'FORMULA', scheme)

        # 3. 定理模式
        for pattern in Config.THEOREM_PATTERNS:
            for match in re.finditer(pattern, sentence):
                if match.groups():
                    start, end = match.start(1), match.end(1)
                else:
                    start, end = match.span()
                if not any(occupied[start:end]):
                    self._apply_labels(labels, occupied, start, end, 'THEOREM', scheme)

        return labels

    @staticmethod
    def _apply_labels(labels: List[str], occupied: List[bool],
                      start: int, end: int, entity_type: str,
                      scheme: str = 'bioes'):
        """应用 BIOES 或 BIO 标注"""
        span_len = end - start
        if span_len <= 0:
            return

        if scheme == 'bioes':
            if span_len == 1:
                labels[start] = f'S-{entity_type}'
            else:
                labels[start] = f'B-{entity_type}'
                for i in range(start + 1, end - 1):
                    labels[i] = f'I-{entity_type}'
                labels[end - 1] = f'E-{entity_type}'
        else:  # BIO
            labels[start] = f'B-{entity_type}'
            for i in range(start + 1, end):
                labels[i] = f'I-{entity_type}'

        for i in range(start, end):
            occupied[i] = True

    def create_ner_training_data(self, data: List[Dict]) -> pd.DataFrame:
        """创建词级别NER训练数据 (兼容旧接口)"""
        training_samples = []
        for item in data:
            text = item['content']
            cleaned = self.advanced_text_clean(text)
            sentences = self.split_into_sentences(cleaned)
            for sentence in sentences:
                if len(sentence) > 10:
                    words_tags = self.segment_and_tag(sentence)
                    words = [wt[0] for wt in words_tags]
                    tags = [wt[1] for wt in words_tags]
                    labels = ['O'] * len(words)
                    for i, (word, tag) in enumerate(words_tags):
                        if word in self.ontology_lookup:
                            labels[i] = f'B-{self.ontology_lookup[word]}'
                        elif tag.startswith('n') and len(word) > 1:
                            labels[i] = 'B-CONCEPT'
                    training_samples.append({
                        'tokens': words, 'pos_tags': tags, 'ner_tags': labels,
                        'sentence': sentence, 'grade': item.get('grade'),
                        'chapter': item.get('chapter'), 'source_file': item.get('file_name'),
                    })
        return pd.DataFrame(training_samples)

    # ================================================================
    #  句子分割
    # ================================================================

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        sentences = re.split(r'[。！？!?;；\n]', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    # ================================================================
    #  保存
    # ================================================================

    def save_char_level_training_data(self, df: pd.DataFrame, output_path: str):
        """保存字级别训练数据为 JSONL"""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                record = {
                    'text': row['text'],
                    'char_labels': row['char_labels'],
                    'metadata': {
                        'grade': row.get('grade'),
                        'chapter': row.get('chapter'),
                        'source_file': row.get('source_file'),
                    }
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.info(f"字级别训练数据已保存到: {output_path} ({len(df)} 条)")

    def save_training_data(self, df: pd.DataFrame, output_path: str):
        """保存词级别训练数据为 JSONL"""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                record = {
                    'tokens': row['tokens'], 'ner_tags': row['ner_tags'],
                    'pos_tags': row['pos_tags'],
                    'metadata': {
                        'grade': row.get('grade'), 'chapter': row.get('chapter'),
                        'source_file': row.get('source_file'),
                    }
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.info(f"训练数据已保存到: {output_path} ({len(df)} 条)")

    def save_conll_format_data(self, df: pd.DataFrame, output_path: str, level: str = 'char'):
        """保存 CoNLL 格式"""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            if level == 'char':
                for _, row in df.iterrows():
                    text = row['text']
                    char_labels = row['char_labels']
                    for i, char in enumerate(text):
                        label = char_labels[i] if i < len(char_labels) else 'O'
                        f.write(f"{char} {label}\n")
                    f.write('\n')
            else:
                for _, row in df.iterrows():
                    for token, pos_tag, ner_tag in zip(
                            row['tokens'], row['pos_tags'], row['ner_tags']):
                        f.write(f"{token} {pos_tag} {ner_tag}\n")
                    f.write('\n')
        logger.info(f"CoNLL格式数据已保存到: {output_path}")

    def load_conll_format_data(self, file_path: str) -> pd.DataFrame:
        """加载 CoNLL 格式数据"""
        data = []
        current_sentence = []
        current_labels = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        current_sentence.append(parts[0])
                        current_labels.append(parts[-1])
                else:
                    if current_sentence:
                        data.append({'text': ''.join(current_sentence), 'char_labels': current_labels})
                    current_sentence, current_labels = [], []
            if current_sentence:
                data.append({'text': ''.join(current_sentence), 'char_labels': current_labels})
        return pd.DataFrame(data)


# ================================================================
#  入口
# ================================================================

if __name__ == "__main__":
    from data_loader import MathDataLoader

    processor = EnhancedDataProcessor()
    loader = MathDataLoader()

    try:
        data = loader.load_all_data()
        logger.info(f"共加载 {len(data)} 个文件")

        # 生成 BIOES 字级别训练数据
        char_df = processor.create_char_level_ner_training_data(data, scheme='bioes')
        logger.info(f"BIOES 训练样本: {len(char_df)} 条")

        # 统计标注分布
        from collections import Counter
        label_counter = Counter()
        for _, row in char_df.iterrows():
            label_counter.update(row['char_labels'])
        logger.info("标注分布:")
        for label, count in label_counter.most_common():
            logger.info(f"  {label}: {count}")

        # 保存
        output_dir = Config.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        processor.save_char_level_training_data(
            char_df, os.path.join(output_dir, 'ner_train_bioes.jsonl'))
        processor.save_conll_format_data(
            char_df, os.path.join(output_dir, 'ner_train_bioes.conll'), level='char')

        # 显示样例
        if len(char_df) > 0:
            sample = char_df.iloc[0]
            text = sample['text']
            labels = sample['char_labels']
            logger.info(f"\n样例句子: {text[:80]}...")
            for i, (ch, lb) in enumerate(zip(text[:60], labels[:60])):
                if lb != 'O':
                    print(f"  [{ch}] {lb}")

    except Exception as e:
        logger.error(f"处理出错: {e}")
        import traceback
        traceback.print_exc()
