import os
import re
import pandas as pd
from typing import List, Dict, Tuple
from config import Config

class MathDataLoader:
    def __init__(self):
        self.data_dir = Config.DATA_DIR
        
    def get_file_list(self) -> List[str]:
        """获取所有数学教材文件列表"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
            
        files = []
        for file_name in os.listdir(self.data_dir):
            # 匹配7.1, 7.2, 8.1, 8.2等命名格式
            if re.match(r'^\d+\.\d+', file_name):
                files.append(os.path.join(self.data_dir, file_name))
        
        return sorted(files)
    
    def read_file_content(self, file_path: str) -> str:
        """读取单个文件内容"""
        try:
            # 检查文件扩展名
            if file_path.endswith(('.docx', '.docm')):
                # 首先尝试使用OOXML提取（适用于.docx和.docm）
                ooxml_content = self._extract_text_from_ooxml(file_path)
                if ooxml_content and ooxml_content.strip():
                    return ooxml_content
                
                # 如果OOXML提取失败，回退到python-docx（仅适用于.docx）
                if file_path.endswith('.docx'):
                    try:
                        import docx
                        try:
                            doc = docx.Document(file_path)
                            content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                            if content.strip():  # 如果有内容，返回
                                return content
                            else:
                                raise Exception("python-docx无法提取文本内容")
                        except Exception as e:
                            print(f"警告: 使用python-docx读取Word文档失败 {file_path}: {e}")
                    except ImportError:
                        print(f"警告: 需要安装python-docx库来读取Word文档: {file_path}")
                
                # 最后尝试二进制提取
                return self._extract_text_from_binary(file_path)
            else:
                # 对于文本文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
                return content
            except:
                # 如果文本模式都失败，尝试二进制提取
                return self._extract_text_from_binary(file_path)
        except Exception as e:
            print(f"读取文件错误 {file_path}: {e}")
            return self._extract_text_from_binary(file_path)
            
    def _extract_text_from_ooxml(self, file_path: str) -> str:
        """从OOXML格式（.docx/.docm）文件中提取文本内容"""
        try:
            import zipfile
            import xml.etree.ElementTree as ET
            
            # OOXML文件实际上是ZIP压缩包
            with zipfile.ZipFile(file_path, 'r') as docx_zip:
                # 查找主要的文档XML文件
                if 'word/document.xml' in docx_zip.namelist():
                    with docx_zip.open('word/document.xml') as document_file:
                        xml_content = document_file.read()
                        
                        # 解析XML提取文本
                        namespace = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
                        root = ET.fromstring(xml_content)
                        
                        # 提取所有文本内容
                        text_parts = []
                        for elem in root.iter():
                            if elem.tag == f'{namespace}t' and elem.text:
                                text_parts.append(elem.text)
                            elif elem.tag == f'{namespace}p':
                                text_parts.append('\n')
                        
                        content = ''.join(text_parts)
                        return content.strip()
                
                return ""
                
        except Exception as e:
            print(f"OOXML文本提取失败 {file_path}: {e}")
            return ""
    
    def _extract_text_from_binary(self, file_path: str) -> str:
        """从二进制文件中提取文本内容"""
        try:
            with open(file_path, 'rb') as f:
                binary_data = f.read()
                
            # 尝试解码为文本（简单的文本提取）
            try:
                # 尝试UTF-8解码
                content = binary_data.decode('utf-8', errors='ignore')
            except:
                # 尝试其他常见编码
                try:
                    content = binary_data.decode('gbk', errors='ignore')
                except:
                    content = binary_data.decode('latin-1', errors='ignore')
            
            # 过滤非文本字符，保留中文、英文、数字和基本标点
            import re
            # 保留中文、英文、数字、基本数学符号和标点
            content = re.sub(r'[^\w\s\u4e00-\u9fffα-ωΑ-Ω√∛∜±∓×÷∙∘°′″∞∝∼≈≅≠≡≤≥≪≫∈∉⊂⊃∪∩∅∀∃∴∵∶∷∥⊥∠△□○◇♢♡♤♧♠♣♥♦+−×÷=<>≤≥≈≠≡√∛∜∞∝∫∮∑∏∂∇∆δϵζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ，。！？；："\'()（）【】《》]', ' ', content)
            
            # 去除过多的空白字符
            content = re.sub(r'\s+', ' ', content)
            
            return content.strip()
            
        except Exception as e:
            print(f"二进制文本提取失败 {file_path}: {e}")
            return ""
    
    def extract_metadata(self, file_name: str) -> Dict:
        """从文件名提取元数据"""
        match = re.match(r'^(\d+)\.(\d+)', os.path.basename(file_name))
        if match:
            grade = int(match.group(1))
            chapter = int(match.group(2))
            return {
                'grade': grade,
                'chapter': chapter,
                'file_name': os.path.basename(file_name)
            }
        return {}
    
    def load_all_data(self) -> List[Dict]:
        """加载所有数据"""
        files = self.get_file_list()
        all_data = []
        
        for file_path in files:
            content = self.read_file_content(file_path)
            metadata = self.extract_metadata(file_path)
            
            data_item = {
                'content': content,
                **metadata
            }
            all_data.append(data_item)
            
            print(f"已加载: {metadata.get('file_name', '未知文件')}")
        
        return all_data
    
    def preprocess_text(self, text: str) -> str:
        """增强的文本预处理"""
        # 修复异常空格（中文文本中不应有单词内部空格）
        text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)
        
        # 保留数学符号但清理其他特殊字符
        math_symbols = r'α-ωΑ-Ω√∛∜±∓×÷∙∘°′″∞∝∼≈≅≠≡≤≥≪≫∈∉⊂⊃∪∩∅∀∃∴∵∶∷∥⊥∠△□○◇♢♡♤♧♠♣♥♦+−×÷=<>≤≥≈≠≡√∛∜∞∝∫∮∑∏∂∇∆δϵζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ'
        text = re.sub(fr'[^\w\s{math_symbols}\u4e00-\u9fff]', ' ', text)
        
        # 规范化空白字符
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # 中文句子分割
        sentences = re.split(r'[。！？!?;；]', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_training_data(self, data: List[Dict]) -> pd.DataFrame:
        """创建训练数据格式"""
        training_data = []
        
        for item in data:
            content = item['content']
            sentences = self.split_into_sentences(content)
            
            for sentence in sentences:
                if len(sentence) > 10:  # 过滤过短的句子
                    training_data.append({
                        'text': sentence,
                        'grade': item.get('grade'),
                        'chapter': item.get('chapter'),
                        'source_file': item.get('file_name')
                    })
        
        return pd.DataFrame(training_data)
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """保存处理后的数据"""
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"数据已保存到: {output_path}")

# 使用示例
if __name__ == "__main__":
    loader = MathDataLoader()
    
    try:
        # 加载所有数据
        data = loader.load_all_data()
        print(f"共加载 {len(data)} 个文件")
        
        # 创建训练数据
        df = loader.create_training_data(data)
        print(f"创建了 {len(df)} 条训练样本")
        
        # 保存处理后的数据
        output_path = os.path.join(Config.OUTPUT_DIR, "processed_math_data.csv")
        loader.save_processed_data(df, output_path)
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保数据目录存在且包含数学教材文件")