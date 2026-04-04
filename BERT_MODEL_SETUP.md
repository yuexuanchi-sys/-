# BERT模型安装指南

## 问题描述
由于网络连接问题，无法从Hugging Face自动下载BERT模型。

## 解决方案

### 方案1: 使用下载脚本（推荐）
当网络连接可用时，运行下载脚本：

```bash
conda activate Empyrean
python download_model.py
```

这个脚本会自动：
1. 下载bert-base-chinese模型
2. 保存到本地目录 `./models/bert-base-chinese/`
3. 更新配置文件使用本地模型路径

### 方案2: 手动下载模型
如果自动下载失败，可以手动下载：

1. 访问Hugging Face模型库: https://huggingface.co/bert-base-chinese
2. 下载以下文件到 `./models/bert-base-chinese/` 目录：
   - `config.json`
   - `pytorch_model.bin`
   - `vocab.txt`
   - `tokenizer_config.json`
   - `special_tokens_map.json`

3. 手动更新配置文件：
```python
# 在 config.py 中修改
BERT_MODEL = "./models/bert-base-chinese"  # 使用相对路径
# 或者
BERT_MODEL = "D:/path/to/your/project/models/bert-base-chinese"  # 使用绝对路径
```

### 方案3: 使用离线模式
当前代码已经支持离线模式，即使没有BERT模型也能运行：

- 仅使用基于规则的实体识别
- 可以识别数学公式和定理名称
- 基本功能可用，但精度较低

## 验证安装
运行测试脚本确认模型安装成功：

```bash
python test_basic.py
```

## 网络问题解决建议
如果遇到网络连接问题：

1. **检查网络连接**: 确保可以访问 https://huggingface.co
2. **使用VPN**: 如果在中国大陆，可能需要VPN访问
3. **配置代理**: 设置HTTP代理
4. **使用镜像源**: 配置Hugging Face镜像

### 配置镜像源（中国大陆用户）
在终端中设置环境变量：
```bash
set HF_ENDPOINT=https://hf-mirror.com
```

或者修改Python代码：
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

## 文件结构
成功安装后，模型目录结构应为：
```
models/
└── bert-base-chinese/
    ├── config.json
    ├── pytorch_model.bin
    ├── vocab.txt
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

## 注意事项
1. 模型文件较大（约400MB），请确保有足够磁盘空间
2. 首次运行需要下载时间，请耐心等待
3. 如果下载中断，可以重新运行脚本，它会自动续传

## 技术支持
如果仍然遇到问题，请检查：
- 磁盘空间是否充足
- 网络连接是否稳定
- 防火墙是否阻止了下载