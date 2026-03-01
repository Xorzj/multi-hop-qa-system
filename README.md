# 基于多跳推理优化的垂域智能问答系统

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Package Manager](https://img.shields.io/badge/uv-enabled-brightgreen)
![API](https://img.shields.io/badge/FastAPI-async-009688)
![Graph](https://img.shields.io/badge/Neo4j-5.26%20community-blue)

本项目是毕业设计，目标是构建一套可落地的垂域智能问答系统。系统以 Neo4j 作为结构化知识图谱存储，结合多跳推理链路与大语言模型生成能力，输出可追溯的答案和证据链。文档处理采用 markitdown 进行 docx 到 markdown 的转换，并可对图像 base64 内容进行清理。

## 目录

- [项目特性](#项目特性)
- [架构概览](#架构概览)
- [技术栈](#技术栈)
- [前置条件](#前置条件)
- [安装与部署](#安装与部署)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用示例](#使用示例)
- [API 文档](#api-文档)
- [开发指南](#开发指南)
- [测试](#测试)
- [License](#license)

## 项目特性

- 多跳推理，基于知识图谱进行多步路径检索
- 领域无关，系统可从文档自动发现领域概念
- LLM 本地推理支持，必要时可使用 Zhipu API 作为外部模型
- FastAPI 异步接口，支持同步和异步问答
- markitdown 文档处理，保留结构化内容并可移除 base64 图片

## 架构概览

```
文档(docx)
   |
   v
DocumentLoader -> EntityExtractor/TripleExtractor -> GraphBuilder -> Neo4j
                                                            |
                                                            v
QuestionParser -> ReasoningOrchestrator -> ContextAssembler -> AnswerGenerator
                                                            |
                                                            v
                                                         FastAPI
```

说明

- 离线阶段负责文档加载、实体与关系抽取、图谱构建
- 在线阶段负责问题解析、多跳推理、上下文组装与答案生成

## 技术栈

- Python >= 3.12
- 包管理器: uv
- Web 框架: FastAPI
- 图数据库: Neo4j 5.26 community
- 文档处理: markitdown[docx]
- 训练与推理: transformers, peft, accelerate, torch

## 前置条件

- Python 3.12+
- Docker 和 Docker Compose
- uv 已安装

## 安装与部署

1. 安装依赖

```bash
uv sync --group dev --group dl --group llm --group graph --group api --group data
```

2. 启动 Neo4j

```bash
bash scripts/start_neo4j.sh
```

3. 配置环境变量

```bash
export NEO4J_PASSWORD=password123
export ZHIPU_API_KEY=your_api_key
```

说明

- NEO4J_PASSWORD 用于替换 config/config.yaml 中的 ${NEO4J_PASSWORD}
- 如果使用本地模型，可不设置 ZHIPU_API_KEY

4. 启动 API

```bash
uv run uvicorn api.main:app --reload --port 8000
```

## 快速开始

1. 启动 Neo4j

```bash
bash scripts/start_neo4j.sh
```

2. 启动 API

```bash
uv run uvicorn api.main:app --reload --port 8000
```

3. 发起同步问答请求

```bash
curl -X POST "http://localhost:8000/qa/sync" \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是多跳推理", "max_hops": 3, "include_evidence": true}'
```


## 配置说明

默认配置文件: `config/config.yaml`

```yaml
llm:
  provider: "local"        # local 或 zhipu
  model_path: "models/base"
  adapter_path: "models/adapters/default"
  generation:
    max_new_tokens: 512
    temperature: 0.7
    top_p: 0.9

graph:
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "${NEO4J_PASSWORD}"
  database: "neo4j"

api:
  host: "0.0.0.0"
  port: 8000

logging:
  level: "INFO"

data_processing:
  strip_base64_images: true
```

环境变量

- `APP_CONFIG_PATH` 自定义配置路径，默认 `config/config.yaml`
- `NEO4J_PASSWORD` Neo4j 密码
- `ZHIPU_API_KEY` 使用 zhipu provider 时必填

## 使用示例

### 异步问答

1. 提交任务

```bash
curl -X POST "http://localhost:8000/qa/submit" \
  -H "Content-Type: application/json" \
  -d '{"question": "多跳推理的优势是什么", "max_hops": 3, "include_evidence": true}'
```

2. 查询状态

```bash
curl "http://localhost:8000/qa/status/<task_id>"
```

3. 获取结果

```bash
curl "http://localhost:8000/qa/result/<task_id>"
```

### 同步问答

```bash
curl -X POST "http://localhost:8000/qa/sync" \
  -H "Content-Type: application/json" \
  -d '{"question": "多跳推理如何提升问答可靠性", "include_evidence": true}'
```

### 健康检查

```bash
curl "http://localhost:8000/health"
curl "http://localhost:8000/health/ready"
curl "http://localhost:8000/health/live"
```

## API 文档

基础地址: `http://localhost:8000`

| 端点 | 方法 | 说明 |
| --- | --- | --- |
| /health | GET | 健康检查 |
| /health/ready | GET | 就绪检查 |
| /health/live | GET | 存活检查 |
| /qa/submit | POST | 提交异步任务 |
| /qa/status/{task_id} | GET | 查询任务状态 |
| /qa/result/{task_id} | GET | 获取任务结果 |
| /qa/sync | POST | 同步问答 |

请求体示例

```json
{
  "question": "示例问题",
  "max_hops": 3,
  "include_evidence": true,
  "domain": null
}
```

返回示例

```json
{
  "answer": "...",
  "confidence": 0.72,
  "evidence": [
    {
      "path": ["实体A", "实体B"],
      "confidence": 0.8,
      "source": "实体A",
      "relation_type": "关联"
    }
  ],
  "reasoning_steps": ["..."],
  "latency_ms": 123.4
}
```

## 开发指南

### 代码结构

```
api/                     # FastAPI 服务
src/                     # 核心逻辑
  data_processing/       # 文档处理与抽取
  knowledge_graph/       # Neo4j 与图检索
  qa_engine/             # QA 组件
  reasoning/             # 多跳推理
  llm/                   # LLM 客户端
  inference/             # 本地推理引擎
scripts/                 # Neo4j 启停脚本
config/                  # 配置文件
```

### Neo4j 脚本说明

启动

```bash
bash scripts/start_neo4j.sh
```

停止

```bash
bash scripts/stop_neo4j.sh
```

停止并清理数据卷

```bash
bash scripts/stop_neo4j.sh --clean
```

脚本会自动检测 docker compose 版本，等待容器健康状态变为 healthy。

### 本地模型与 Zhipu 切换

- 本地模型: `llm.provider: local`
- Zhipu: `llm.provider: zhipu` 并设置 `ZHIPU_API_KEY`

OpenAI 客户端目前是占位实现，尚未接入真实 API。

## 测试

```bash
uv run pytest
```

常用检查

```bash
uv run ruff format .
uv run ruff check --fix .
uv run mypy src/ api/
```

## License

本仓库未附带 License 文件。如需开源，请补充合适的许可证。
