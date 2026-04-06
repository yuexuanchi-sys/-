# 初中数学知识图谱系统 - 用户使用手册

> 本系统从人教版初中数学教材中自动提取知识实体与关系，构建结构化知识图谱，并提供交互式可视化界面。适用于教育研究、知识工程论文写作及教学辅助。

---

## 目录

1. [快速体验](#1-快速体验)
2. [完整安装与运行](#2-完整安装与运行)
3. [系统架构与模块说明](#3-系统架构与模块说明)
4. [运行模式详解](#4-运行模式详解)
5. [Web可视化使用指南](#5-web可视化使用指南)
6. [论文写作参考数据](#6-论文写作参考数据)
7. [评估框架使用说明](#7-评估框架使用说明)
8. [常见问题与故障排除](#8-常见问题与故障排除)
9. [项目文件结构](#9-项目文件结构)

---

## 1. 快速体验

**无需安装任何环境**，双击打开以下文件即可查看完整知识图谱：

```
results/knowledge_graph_standalone.html
```

该文件内嵌了 **646个知识节点 + 1575条关系**，支持：
- 点击「加载完整图谱」显示所有节点
- 搜索框输入关键词（如"三角形"、"方程"）筛选节点
- 点击节点查看详细信息和关联关系
- 鼠标滚轮缩放，拖拽平移画布
- 侧边栏浏览完整节点列表

> 可直接发送此文件给他人，用任意浏览器（Chrome/Edge/Firefox）打开即可。

---

## 2. 完整安装与运行

### 2.1 环境要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| Python | 3.8+ | 3.10+ |
| 内存 | 8GB | 16GB |
| GPU | 不需要 | CUDA兼容GPU（加速BERT） |
| Neo4j | 不需要（离线模式） | 4.x+（可选） |

### 2.2 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/yuexuanchi-sys/-.git
cd -

# 2. 安装Python依赖
pip install -r requirements.txt

# 3. 配置环境变量（可选）
cp .env.example .env
# 编辑 .env 文件，填入你的配置（不配置也可以离线运行）

# 4. 放置教材数据
# 将 .docx 格式的教材文件放入 data/ 目录
```

### 2.3 快速运行

```bash
# 运行知识图谱构建（本地模式，不需要API密钥）
python main_kggen_pipeline.py --mode local_only

# 启动Web可视化
python app.py
# 浏览器打开 http://127.0.0.1:5000

# 运行完整评估（生成论文数据）
python run_full_evaluation.py

# 运行测试套件
python test_pipeline.py
```

---

## 3. 系统架构与模块说明

### 3.1 技术架构

```
教材文本 (.docx)
    |
    v
[实体提取] ── 规则匹配 + jieba词性标注 + 核心术语词典 + BERT模型(可选)
    |
    v
[实体质量过滤] ── 多层过滤: 停用词/OCR噪声/句子片段/领域验证
    |
    v
[关系提取] ── 规则模板 + 共现分析 + 上下文语义推断 + LLM API(可选)
    |
    v
[知识图谱] ── 去重 + 层级构建 + Neo4j存储(可选) + JSON导出
    |
    v
[可视化] ── Flask Web应用 + Canvas力导向布局
```

### 3.2 核心模块

| 模块文件 | 功能 | 说明 |
|----------|------|------|
| `main_kggen_pipeline.py` | 主管道 | 统一调度所有模块，支持多种运行模式 |
| `entity_extractor_kggen.py` | 实体提取 | 规则+词性+BERT集成，含60+核心术语词典 |
| `relation_extractor_kggen.py` | 关系提取 | 规则模板+共现分析+LLM(可选) |
| `knowledge_graph_builder_kggen.py` | 图谱构建 | 实体过滤+去重+层级构建 |
| `bert_trainer_v2.py` | BERT训练 | BERT+BiLSTM+CRF，BIOES标注，数据增强 |
| `evaluation.py` | 评估框架 | NER指标+关系指标+图结构分析+消融实验 |
| `app.py` | Web应用 | Flask + 离线/在线双模式 |
| `neo4j_manager.py` | 数据库 | Neo4j CRUD，带注入防护 |
| `data_loader.py` | 数据加载 | .docx解析，文本预处理 |
| `enhanced_data_processor.py` | 数据处理 | BIOES NER训练数据生成 |
| `config.py` | 配置中心 | 环境变量管理 |
| `run_full_evaluation.py` | 一键评估 | 生成所有论文所需数据 |
| `test_pipeline.py` | 测试套件 | 9项功能测试 |

---

## 4. 运行模式详解

### 4.1 知识图谱构建

```bash
# 本地模式（推荐，不需要任何API密钥）
python main_kggen_pipeline.py --mode local_only

# 混合模式（需要DeepSeek API密钥）
python main_kggen_pipeline.py --mode hybrid

# 仅KGGen模式
python main_kggen_pipeline.py --mode kggen_direct
```

### 4.2 BERT模型训练

```bash
# 训练NER模型
python main_kggen_pipeline.py --mode train --epochs 10 --batch-size 16 --lr 2e-5
```

训练完成后模型保存在 `models/` 目录，后续构建管道会自动加载。

### 4.3 评估与论文数据生成

```bash
# 评估已有结果
python main_kggen_pipeline.py --mode evaluate

# 一键生成完整论文数据（推荐）
python run_full_evaluation.py
```

运行 `run_full_evaluation.py` 后，`results/` 目录将包含：

| 输出文件 | 内容 |
|----------|------|
| `evaluation_summary.txt` | 人类可读的完整评估报告 |
| `paper_metrics.json` | 结构化论文指标 |
| `kg_entities.json` | 全部实体数据 |
| `kg_relations.json` | 全部关系数据 |
| `ablation_results.json` | 消融实验对比 |
| `ner_evaluation.json` | NER评估验证 |
| `relation_evaluation.json` | 关系评估验证 |
| `test_report.json` | 测试套件报告 |
| `web_*.json` | Web API响应数据 |
| `knowledge_graph_standalone.html` | 独立可视化页面 |

### 4.4 消融实验

```bash
python main_kggen_pipeline.py --mode ablation
```

对比不同配置的效果（仅规则 vs 规则+词性 vs 混合模式）。

---

## 5. Web可视化使用指南

### 5.1 启动服务

```bash
python app.py
# 浏览器打开 http://127.0.0.1:5000
```

系统会自动检测Neo4j连接，不可用时切换为离线模式（从 `output/` 目录加载JSON数据）。

### 5.2 界面操作

| 操作 | 说明 |
|------|------|
| 「加载完整图谱」按钮 | 加载全部节点和关系 |
| 搜索框 | 输入关键词筛选节点（如"三角形"） |
| 点击节点 | 查看节点详情和关联关系 |
| 鼠标滚轮 | 缩放画布 |
| 拖拽画布 | 平移视图 |
| 侧边栏节点列表 | 点击跳转到对应节点 |

### 5.3 API接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/full-graph` | GET | 获取完整图谱数据 |
| `/api/chapters` | GET | 获取所有章节 |
| `/api/branch/<名称>` | GET | 获取某节点的知识分支 |
| `/api/search?q=关键词` | GET | 搜索知识点 |
| `/api/path?start=X&end=Y` | GET | 查找两点间路径 |
| `/health` | GET | 健康检查 |

### 5.4 离线分享

直接将 `results/knowledge_graph_standalone.html` 发送给他人，双击打开即可浏览完整图谱。

---

## 6. 论文写作参考数据

以下数据来自系统实际运行结果，可直接引用于论文。

### 6.1 知识图谱整体指标

| 指标 | 数值 |
|------|------|
| 实体总数 | 652 |
| 关系总数 | 1,575 |
| 实体类型数 | 7 |
| 关系类型数 | 9 |
| 图节点数 | 646 |
| 图边数 | 1,575 |
| 连通分量 | 1（全连通） |
| 孤立节点 | 0 |
| 平均节点度 | 4.88 |
| 图密度 | 0.003780 |
| 课标知识点覆盖率 | 94.74%（54/57） |

### 6.2 实体类型分布

| 类型 | 数量 | 占比 |
|------|------|------|
| CONCEPT（概念） | 608 | 93.25% |
| FORMULA（公式） | 19 | 2.91% |
| THEOREM（定理） | 9 | 1.38% |
| CHAPTER（章节） | 9 | 1.38% |
| METHOD（方法） | 3 | 0.46% |
| MODULE（模块） | 3 | 0.46% |
| EXAMPLE（示例） | 1 | 0.15% |

### 6.3 关系类型分布

| 关系类型 | 数量 | 占比 | 说明 |
|----------|------|------|------|
| 包含 | 748 | 47.49% | 上下位关系 |
| 相关概念 | 471 | 29.90% | 语义关联 |
| 推导关系 | 119 | 7.56% | 逻辑推导 |
| 应用 | 96 | 6.10% | 实际应用 |
| 符号表示 | 66 | 4.19% | 数学符号 |
| 定义 | 48 | 3.05% | 概念定义 |
| 前置知识 | 13 | 0.83% | 学习顺序 |
| 属于 | 12 | 0.76% | 分类归属 |
| 性质 | 2 | 0.13% | 数学性质 |

### 6.4 实体提取来源分布

| 提取方法 | 数量 | 说明 |
|----------|------|------|
| 词性标注（POS） | 534 | jieba分词+词性过滤 |
| 规则匹配 | 93 | 核心术语词典+正则 |
| 规则+上下文 | 13 | 关键词上下文提取 |
| 层级推断 | 12 | 教材结构推断 |

### 6.5 置信度分析

| 指标 | 实体 | 关系 |
|------|------|------|
| 均值 | 0.660 | 0.856 |
| 中位数 | 0.600 | 0.900 |
| 标准差 | 0.135 | 0.136 |
| 最小值 | 0.60 | 0.70 |
| 最大值 | 1.00 | 1.00 |

### 6.6 消融实验

| 配置 | 实体数 | 关系数 | 覆盖率 | 平均度 | 连通分量 |
|------|--------|--------|--------|--------|----------|
| 仅规则 | 27,445 | 0 | 98.25% | 0.00 | 4,271 |
| 规则+词性集成 | 652 | 1,575 | 94.74% | 4.88 | 1 |

**分析**：仅规则提取虽然覆盖率高（98.25%），但产生大量噪声实体（27,445个）且无法建立关系，图谱碎片化（4,271个连通分量）。集成方法通过多层过滤将实体精炼至652个，同时建立1,575条高质量关系，形成全连通图谱。

### 6.7 课标覆盖分析

**已覆盖（54/57）**：有理数、整数、分数、负数、正数、数轴、绝对值、相反数、加法、减法、乘法、除法、乘方、代数式、方程、一元一次方程、几何体、棱柱、圆柱、圆锥、直线、射线、线段、角、平行线、垂直、三角形、全等三角形、勾股定理、实数、平方根、立方根、二次根式、一次函数、正比例函数、平行四边形、矩形、菱形、正方形、分式、分式方程、数据分析、中位数、众数、方差、一元二次方程、二次函数、反比例函数、相似三角形、锐角三角函数、圆、概率、投影、视图

**未覆盖（3/57）**：弧、圆心角、随机事件

### 6.8 Hub节点（核心知识点）

| 排名 | 知识点 | 度数 | 入度 | 出度 |
|------|--------|------|------|------|
| 1 | 图形 | 67 | 26 | 41 |
| 2 | 直线 | 62 | 29 | 33 |
| 3 | 三角形 | 56 | 30 | 26 |
| 4 | 坐标 | 50 | 24 | 26 |
| 5 | 方程 | 41 | 21 | 20 |
| 6 | 平面 | 40 | 18 | 22 |
| 7 | 线段 | 39 | 20 | 19 |
| 8 | 顶点 | 36 | 12 | 24 |
| 9 | 正方形 | 26 | 12 | 14 |
| 10 | 函数 | 25 | 12 | 13 |

（排除章节节点，仅保留数学概念）

---

## 7. 评估框架使用说明

### 7.1 NER评估

```python
from evaluation import NERMetrics

gold = [['B-CONCEPT', 'E-CONCEPT', 'O', 'B-THEOREM', 'I-THEOREM', 'E-THEOREM']]
pred = [['B-CONCEPT', 'E-CONCEPT', 'O', 'B-THEOREM', 'I-THEOREM', 'E-THEOREM']]
metrics = NERMetrics.compute_span_metrics(gold, pred, scheme='bioes')
# metrics['overall'] -> {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
```

### 7.2 关系评估

```python
from evaluation import RelationMetrics

gold = [('三角形', '包含', '直角三角形')]
pred = [('三角形', '包含', '直角三角形')]
metrics = RelationMetrics.compute_relation_metrics(gold, pred)
```

### 7.3 知识图谱质量分析

```python
from evaluation import KnowledgeGraphAnalyzer

analyzer = KnowledgeGraphAnalyzer(entities, relations)
report = analyzer.full_report()
# report 包含: structural, entity_types, relation_types, sources, confidence, coverage, hubs
```

### 7.4 一键生成论文指标

```python
from evaluation import generate_paper_metrics

report = generate_paper_metrics(entities, relations)
# 自动保存到 output/paper_metrics.json
```

---

## 8. 常见问题与故障排除

### Q1: 启动Web时提示Neo4j连接失败

这是正常的。系统会自动切换到离线模式，从 `output/` 目录读取JSON数据。确保已运行过构建管道生成数据文件：

```bash
python main_kggen_pipeline.py --mode local_only
```

### Q2: 点击"加载完整图谱"没有显示节点

检查 `output/` 目录是否存在 `kggen_entities.json` 和 `kggen_relations.json`。如果没有，先运行构建管道。

### Q3: 运行时报 CUDA/GPU 相关错误

系统会自动检测GPU，没有GPU时使用CPU运行。如果GPU驱动有问题：

```bash
# 强制使用CPU
set CUDA_VISIBLE_DEVICES=""
python main_kggen_pipeline.py --mode local_only
```

### Q4: 教材数据格式要求

将 `.docx` 格式的教材文件放入 `data/` 目录。文件命名建议：`年级.册数.docx`（如 `7.1.docx` 表示七年级上册）。

### Q5: 如何提高覆盖率

1. 在 `entity_extractor_kggen.py` 的 `core_math_terms` 字典中添加缺失的术语
2. 在 `knowledge_graph_builder_kggen.py` 的过滤函数中调整阈值
3. 使用混合模式（`--mode hybrid`）接入LLM API提升提取质量

### Q6: 如何在论文中引用本系统

建议描述为：
> 基于规则匹配、词性标注与深度学习集成的初中数学知识图谱自动构建系统。系统采用多层实体质量过滤机制，从人教版初中数学教材中提取数学概念、公式、定理等实体及其语义关系，构建结构化知识图谱。

---

## 9. 项目文件结构

```
.
├── main_kggen_pipeline.py       # 主管道入口
├── run_full_evaluation.py       # 一键评估脚本
├── test_pipeline.py             # 测试套件
├── app.py                       # Flask Web应用
├── config.py                    # 配置中心
├── entity_extractor_kggen.py    # 实体提取器
├── relation_extractor_kggen.py  # 关系提取器
├── knowledge_graph_builder_kggen.py  # 图谱构建器
├── bert_trainer_v2.py           # BERT+BiLSTM+CRF训练器
├── evaluation.py                # 评估框架
├── neo4j_manager.py             # Neo4j管理器
├── data_loader.py               # 数据加载器
├── enhanced_data_processor.py   # 数据处理器
├── kggen_client.py              # LLM API客户端
├── performance_optimizer.py     # 性能监控
├── run.py                       # CLI入口
├── requirements.txt             # Python依赖
├── .env.example                 # 环境变量模板
├── data/                        # 教材数据目录
├── output/                      # 构建输出目录
├── results/                     # 评估结果目录
│   ├── evaluation_summary.txt   # 评估报告
│   ├── paper_metrics.json       # 论文指标
│   ├── kg_entities.json         # 实体数据
│   ├── kg_relations.json        # 关系数据
│   ├── ablation_results.json    # 消融实验
│   ├── knowledge_graph_standalone.html  # 独立可视化页面
│   └── ...
├── models/                      # 训练模型目录
└── templates/                   # Web页面模板
    └── math_knowledge.html
```
