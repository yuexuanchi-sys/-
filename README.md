# 初中数学知识图谱自动构建系统

基于 **规则匹配 + 词性标注 + BERT深度学习** 集成方法，从人教版初中数学教材中自动提取知识实体与语义关系，构建结构化知识图谱，并提供交互式Web可视化。

## 系统亮点

- **652个数学实体**，**1,575条语义关系**，**9种关系类型**
- **94.74%** 课程标准知识点覆盖率（54/57）
- **全连通图**（1个连通分量，0个孤立节点）
- 支持 **离线运行**，无需API密钥或数据库
- 提供 **独立HTML可视化**，可直接分享

## 快速体验

**无需安装任何环境**，下载后双击打开：

```
results/knowledge_graph_standalone.html
```

即可在浏览器中交互式浏览完整知识图谱（646节点 + 1575关系）。

## 安装与运行

```bash
# 安装依赖
pip install -r requirements.txt

# 构建知识图谱（本地模式，无需API密钥）
python main_kggen_pipeline.py --mode local_only

# 启动Web可视化
python app.py
# 浏览器打开 http://127.0.0.1:5000

# 一键生成完整论文数据
python run_full_evaluation.py

# 运行测试套件
python test_pipeline.py
```

## 系统架构

```
教材文本 (.docx)
    │
    ▼
 实体提取 ─── 规则匹配 + 词性标注 + 核心术语词典 + BERT(可选)
    │
    ▼
 质量过滤 ─── 停用词 / OCR噪声 / 句子片段 / 领域验证
    │
    ▼
 关系提取 ─── 规则模板 + 共现分析 + 语义推断 + LLM API(可选)
    │
    ▼
 知识图谱 ─── 去重 + 层级构建 + 可视化 + Neo4j(可选)
```

## 知识图谱指标

| 指标 | 数值 |
|------|------|
| 实体总数 | 652 |
| 关系总数 | 1,575 |
| 实体类型 | 7种（概念/公式/定理/方法/示例/模块/章节） |
| 关系类型 | 9种（包含/相关概念/推导/应用/符号/定义/前置知识/属于/性质） |
| 课标覆盖率 | 94.74% |
| 平均节点度 | 4.88 |
| 连通分量 | 1（全连通） |

## 消融实验

| 配置 | 实体数 | 关系数 | 覆盖率 | 连通分量 |
|------|--------|--------|--------|----------|
| 仅规则 | 27,445 | 0 | 98.25% | 4,271 |
| 规则+词性集成 | 652 | 1,575 | 94.74% | 1 |

## 运行模式

| 模式 | 命令 | 说明 |
|------|------|------|
| 本地构建 | `--mode local_only` | 仅使用本地模型，无需API |
| 混合构建 | `--mode hybrid` | 本地 + DeepSeek API |
| BERT训练 | `--mode train` | 训练NER模型 |
| 评估 | `--mode evaluate` | 生成论文指标 |
| 消融实验 | `--mode ablation` | 对比不同配置 |

## 输出文件

运行 `run_full_evaluation.py` 后，`results/` 目录包含：

- `evaluation_summary.txt` — 完整评估报告
- `paper_metrics.json` — 结构化论文指标
- `kg_entities.json` / `kg_relations.json` — 实体和关系数据
- `ablation_results.json` — 消融实验
- `knowledge_graph_standalone.html` — 独立可视化页面（可直接分享）

## 技术栈

- **NLP**: jieba分词, BERT+BiLSTM+CRF, BIOES标注
- **图数据库**: Neo4j（可选）
- **Web**: Flask + Canvas力导向布局
- **评估**: span-level P/R/F1, 图结构分析, 覆盖率分析

## 文档

- [用户使用手册](USER_GUIDE.md) — 完整的安装、运行、论文写作指南
- [Neo4j配置指南](NEO4J_SETUP_GUIDE.md) — 数据库安装与配置

## 环境要求

- Python 3.8+
- 8GB+ 内存
- Neo4j 4.x+（可选）
- CUDA GPU（可选，加速BERT）
