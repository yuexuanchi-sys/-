# 初中数学知识图谱系统 - 用户指南

## 目录
1. [系统概述](#系统概述)
2. [安装配置](#安装配置)
3. [快速开始](#快速开始)
4. [问题解决](#问题解决)
5. [高级配置](#高级配置)
6. [性能优化](#性能优化)

## 系统概述

本系统是一个基于深度学习和图数据库的数学知识图谱构建与可视化系统，专门针对初中数学教材内容进行知识抽取和关系挖掘。

### 核心功能
- **智能实体识别**: BERT+CRF模型识别数学概念、公式、定理
- **关系抽取**: 结合规则匹配和深度学习模型
- **图数据库存储**: Neo4j存储和管理知识图谱
- **交互式可视化**: Web界面展示知识图谱
- **离线模式**: 支持无网络和无数据库环境运行

## 安装配置

### 系统要求
- Python 3.8+
- Neo4j 4.x+ (可选)
- 至少8GB内存
- 支持CUDA的GPU (可选，用于加速)

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <项目地址>
   cd knowledge_graph_project
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

   或者手动安装核心依赖：
   ```bash
   pip install torch transformers py2neo python-docx sentencepiece protobuf torchcrf
   ```

3. **配置Neo4j (可选)**
   - 安装Neo4j Desktop或Community Edition
   - 启动Neo4j服务
   - 修改 `config.py` 中的连接信息：
     ```python
     NEO4J_URI = "bolt://localhost:7687"
     NEO4J_USER = "neo4j"
     NEO4J_PASSWORD = "your_password"
     ```

4. **准备数据**
   - 创建数据目录: `D:\数据\`
   - 放入数学教材Word文档，命名格式: `7.1.docx`, `7.2.docx`, `8.1.docx` 等

## 快速开始

### 一键启动
```bash
# Windows
start.bat

# 或手动运行
python run.py all
```

### 分步运行
```bash
# 仅构建知识图谱
python run.py build

# 仅启动Web服务
python run.py web
```

访问 http://localhost:5000 查看可视化界面

### 测试系统功能
```python
# 简单测试
from entity_recognizer import EntityRecognizer
from neo4j_manager import Neo4jManager

# 测试实体识别
recognizer = EntityRecognizer()
text = "二次函数标准形式: y = ax² + bx + c"
entities = recognizer.predict_entities(text)
print("识别到的实体:", entities)

# 测试数据库连接
manager = Neo4jManager()
print("数据库连接状态:", manager.connected)
```

## 问题解决

### 常见问题

#### 1. 模块导入错误
**问题**: `ModuleNotFoundError: No module named 'torch'`
**解决**:
```bash
pip install torch transformers
```

#### 2. NumPy版本兼容性
**问题**: NumPy版本警告或错误
**解决**:
```bash
pip install numpy==1.26.4
```

#### 3. Neo4j连接失败
**问题**: `Neo4j连接失败: Cannot open connection`
**解决**:
- 检查Neo4j服务是否启动
- 验证config.py中的连接配置
- 系统会自动切换到离线模式，不影响基本功能

#### 4. BERT模型下载失败
**问题**: 网络问题导致模型下载失败
**解决**:
- 系统会自动使用规则方法进行实体识别
- 可手动下载模型到 `models/` 目录
- 或使用离线模式运行

#### 5. 长文本处理警告
**问题**: `Token indices sequence length is longer than the specified maximum`
**解决**:
- 系统已自动处理，无需干预
- 长文本会被分割成句子处理

#### 6. 编码问题
**问题**: Unicode编码错误
**解决**:
- 系统已内置编码处理
- 确保文件使用UTF-8编码

### 离线模式运行

系统支持完整的离线操作：

1. **无网络环境**
   ```bash
   # 使用规则方法进行实体识别
   # 无需下载BERT模型
   ```

2. **无数据库环境**
   ```bash
   # Neo4j离线模式
   # 识别结果显示在控制台
   # 不会保存到数据库
   ```

3. **完全离线**
   ```bash
   # 既无网络也无数据库
   # 系统仍能正常运行
   # 使用规则方法+控制台输出
   ```

## 高级配置

### 自定义实体类型

在 `config.py` 中修改实体类型定义：

```python
ENTITY_TYPES = {
    'CONCEPT': '数学概念',
    'FORMULA': '数学公式',
    'THEOREM': '定理定律',
    'EXAMPLE': '例题示例',
    'METHOD': '解题方法',
    # 添加新的实体类型
    'DEFINITION': '定义'
}
```

### 自定义模型路径

```python
# 使用本地模型
BERT_MODEL = "./models/bert-base-chinese"
```

### 调整处理参数

```python
# 最大序列长度
MAX_SEQ_LENGTH = 256

# 批处理大小
BATCH_SIZE = 8

# 置信度阈值
CONFIDENCE_THRESHOLD = 0.7
```

## 性能优化

### 硬件加速

1. **GPU加速**
   ```python
   # 自动检测GPU
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

2. **内存优化**
   - 调整批处理大小
   - 启用梯度检查点

### 处理优化

1. **文本预处理**
   - 自动清理无效字符
   - 智能句子分割

2. **并行处理**
   - 多文档并行处理
   - 异步I/O操作

### 数据库优化

1. **批量操作**
   - 使用事务批量提交
   - 优化索引和约束

2. **查询优化**
   - 使用参数化查询
   - 添加适当索引

## 扩展开发

### 添加新的关系类型

1. 在 `config.py` 中定义新关系
2. 在 `relation_extractor.py` 中添加抽取规则
3. 更新前端显示逻辑

### 自定义数据源

支持多种数据格式：
- Word文档 (.docx, .docm)
- 文本文件 (.txt)
- PDF文档 (需要额外库)

### API扩展

系统提供完整的RESTful API，可以轻松集成到其他系统：

```python
# 示例API调用
import requests

response = requests.get('http://localhost:5000/api/entities')
entities = response.json()
```

## 技术支持

如遇问题，请：
1. 查看本指南的"问题解决"部分
2. 检查控制台错误信息
3. 确认依赖包版本兼容性
4. 在GitHub提交Issue

## 版本更新

### v1.1 更新内容
- 添加完整的离线模式支持
- 解决NumPy版本兼容性问题
- 优化长文本处理能力
- 修复编码问题
- 增强错误处理和日志记录

---

**注意**: 本系统仍在积极开发中，建议定期更新代码以获取最新功能和修复。