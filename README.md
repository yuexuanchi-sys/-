# 初中数学知识图谱系统

这是一个基于Flask和Neo4j的初中数学知识图谱可视化系统。

## 系统要求

- Python 3.7+
- Neo4j 4.0+ (可选，用于完整功能)
- 现代Web浏览器

## 安装和运行步骤

### 1. 安装依赖包

```bash
cd knowledge_graph_project
pip install -r requirements.txt
```

或者使用兼容性启动脚本（推荐）：
```bash
cd knowledge_graph_project
python run_app.py
```

### 2. 配置Neo4j数据库 (可选)

如果您有Neo4j数据库，请修改 [`app.py`](app.py) 中的数据库配置：

```python
# Neo4j配置
NEO4J_URI = "bolt://localhost:7687"  # 修改为您的Neo4j地址
NEO4J_USER = "neo4j"                 # 修改为您的用户名
NEO4J_PASSWORD = "password"          # 修改为您的密码
```

如果没有Neo4j数据库，应用仍然可以运行，但数据查询功能将无法使用。

### 3. 启动应用

**方法一：使用兼容性启动脚本（推荐）**
```bash
cd knowledge_graph_project
python run_app.py
```
此方法会自动检查并安装兼容的依赖版本，解决Flask版本冲突问题。

**方法二：直接运行**
```bash
cd knowledge_graph_project
python app.py
```
注意：如果遇到依赖版本冲突，请使用方法一。

**方法三：使用Flask命令行**
```bash
cd knowledge_graph_project
set FLASK_APP=app.py
flask run
```

### 4. 访问应用

应用启动后，在浏览器中打开：
```
http://localhost:5000
```

## 应用功能

### 主要功能
- **知识图谱可视化** - 使用ECharts动态展示数学知识点关系
- **章节导航** - 按教材章节结构浏览知识点
- **层级浏览** - 支持根分类、章节、小节、知识点四个层级
- **智能搜索** - 搜索知识点名称和ID
- **路径查找** - 查找两个知识点之间的关联路径
- **响应式设计** - 适配电脑和移动设备

### 界面说明
- **左侧侧边栏**：搜索、章节选择、层级筛选、路径查找
- **中间主区域**：知识图谱可视化图表
- **底部区域**：节点详情和路径结果显示

## API接口

应用提供以下REST API接口：

- `GET /` - 主页
- `GET /api/chapters` - 获取所有章节
- `GET /api/branch/<node_name>` - 获取知识分支
- `GET /api/path?start=X&end=Y` - 查找知识路径
- `GET /api/search?q=关键词` - 搜索知识点
- `GET /api/level/<level_type>` - 按层级获取知识
- `GET /api/chapter/<chapter_id>` - 获取章节结构
- `GET /api/health` - 健康检查

## 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 修改端口号
   python app.py --port 5001
   ```

2. **Neo4j连接失败**
   - 检查Neo4j服务是否启动
   - 验证用户名密码是否正确
   - 确认URI格式：`bolt://host:port`

3. **模块导入错误（特别是Flask/url_quote错误）**
   ```bash
   # 使用兼容性启动脚本自动解决
   python run_app.py
   # 或者手动安装指定版本
   pip install Flask==2.3.0 Werkzeug==2.3.0
   ```

4. **静态资源加载失败**
   - 检查static和templates目录结构
   - 确认文件路径正确

### 开发模式

启用开发模式（自动重载）：
```bash
set FLASK_ENV=development
python app.py
```

## 项目结构

```
knowledge_graph_project/
├── app.py                 # Flask主应用
├── run_app.py             # 兼容性启动脚本（推荐使用）
├── requirements.txt       # Python依赖
├── templates/
│   └── math_knowledge.html  # 主页面模板
├── static/
│   ├── css/
│   │   └── style.css     # 样式文件
│   └── js/
│       └── app.js        # 前端JavaScript
├── 数据加载指南.md        # Neo4j数据导入指南
└── README.md             # 说明文档
```

## 技术支持

如有问题，请检查：
1. 控制台错误信息
2. 浏览器开发者工具中的网络请求
3. Flask应用的日志输出