# Neo4j 数据库设置指南

## 问题诊断

当前遇到错误: `Cannot connect to any known routers`

这个错误通常表示:
1. Neo4j数据库服务没有运行
2. 连接配置不正确
3. 防火墙阻止了连接

## 解决方案

### 方法1: 使用Neo4j Desktop (推荐)

1. **下载并安装 Neo4j Desktop**
   - 访问 https://neo4j.com/download/
   - 下载 Neo4j Desktop for Windows
   - 安装并启动

2. **创建新数据库项目**
   - 打开 Neo4j Desktop
   - 点击 "New Project" → "Add Database" → "Local DBMS"
   - 设置数据库名称 (如: knowledge_graph)
   - 设置密码: `your_neo4j_password（请在.env文件中配置）` (与config.py中一致)
   - 点击 "Create"

3. **启动数据库**
   - 在数据库卡片上点击 "Start" 按钮
   - 等待状态变为 "Running"

4. **测试连接**
   - 点击 "Open" 打开 Neo4j Browser
   - 在浏览器中输入: `:server status` 检查状态
   - 或运行: `RETURN 1` 测试查询

### 方法2: 使用Docker (备选)

```bash
# 拉取Neo4j镜像
docker pull neo4j:latest

# 运行Neo4j容器
docker run \
    --name neo4j-knowledge-graph \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -v neo4j_data:/data \
    -v neo4j_logs:/logs \
    -v neo4j_import:/var/lib/neo4j/import \
    --env NEO4J_AUTH=neo4j/your_neo4j_password（请在.env文件中配置） \
    neo4j:latest
```

### 方法3: 直接安装Neo4j Server

1. **下载社区版**
   - https://neo4j.com/download-center/#community

2. **安装和配置**
   - 解压到 `C:\neo4j`
   - 编辑 `conf/neo4j.conf`:
     ```
     dbms.connector.bolt.listen_address=:7687
     dbms.connector.http.listen_address=:7474
     ```
   - 设置密码: `neo4j-admin set-initial-password your_neo4j_password（请在.env文件中配置）`

3. **启动服务**
   - `bin\neo4j console` (前台运行)
   - `bin\neo4j start` (后台服务)

## 连接测试

### 测试Neo4j Browser
访问: http://localhost:7474
- 用户名: `neo4j`
- 密码: `your_neo4j_password（请在.env文件中配置）`

### 测试Python连接
运行测试脚本:
```bash
python simple_neo4j_test.py
```

## 故障排除

### 常见问题

1. **端口冲突**
   - 检查7687和7474端口是否被占用
   - `netstat -ano | findstr :7687`

2. **防火墙阻止**
   - 允许Neo4j通过Windows防火墙
   - 或临时关闭防火墙测试

3. **密码错误**
   - 确认config.py中的密码与Neo4j设置一致
   - 如果需要重置: `neo4j-admin set-initial-password your_new_password`

4. **服务未启动**
   - 检查Neo4j服务状态
   - 重新启动Neo4j Desktop或服务

### 连接参数说明

- **URI**: `neo4j://localhost:7687`
  - `neo4j://` - Neo4j协议 (支持路由)
  - `localhost` - 本地主机
  - `7687` - Bolt协议默认端口

- **备用URI尝试**:
  - `bolt://localhost:7687` - 直接Bolt连接
  - `http://localhost:7474` - HTTP连接 (仅查询)

## 离线模式

如果无法连接Neo4j，程序会自动进入离线模式:
- 数据不会保存到数据库
- 实体和关系会在控制台显示
- 程序可以继续运行进行数据处理

要启用数据库存储，请确保Neo4j服务正常运行。

## 性能优化建议

1. **调整内存设置** (在neo4j.conf中):
   ```
   dbms.memory.heap.initial_size=2G
   dbms.memory.heap.max_size=4G
   dbms.memory.pagecache.size=2G
   ```

2. **定期维护**:
   ```cypher
   CALL db.indexes()  -- 检查索引
   CALL db.constraints()  -- 检查约束
   ```

3. **备份数据**:
   ```bash
   neo4j-admin dump --database=neo4j --to=backup.dump