# 项目结构详解

> 本文档详细说明项目的目录结构和各模块职责。

## 目录树

```
PROJECT/
├── api/                      # FastAPI 接口层
│   ├── main.py               # 应用入口，生命周期管理
│   ├── dependencies.py       # 依赖注入工厂
│   ├── routers/              # 路由模块
│   │   ├── qa.py             # 问答接口
│   │   └── health.py         # 健康检查
│   └── schemas/              # 请求/响应模型
│       ├── request.py        # QuestionRequest
│       └── response.py       # AnswerResponse, TaskResponse
│
├── src/                      # 核心源代码
│   ├── common/               # 通用模块
│   │   ├── config.py         # 配置加载
│   │   ├── logger.py         # 日志工具
│   │   ├── exceptions.py     # 自定义异常
│   │   └── neo4j_docker.py   # Neo4j Docker 管理
│   │
│   ├── data_processing/      # 数据处理
│   │   ├── document_loader.py    # 文档加载器
│   │   ├── entity_extractor.py   # 实体抽取器
│   │   └── triple_extractor.py   # 三元组抽取器
│   │
│   ├── knowledge_graph/      # 知识图谱
│   │   ├── neo4j_client.py   # Neo4j 异步客户端
│   │   ├── graph_builder.py  # 图谱构建器
│   │   ├── graph_retriever.py    # 图谱检索器
│   │   ├── cypher_builder.py     # Cypher 查询生成
│   │   └── schema.py         # 图谱 Schema 定义
│   │
│   ├── llm/                  # 大语言模型
│   │   ├── base_client.py    # LLM 客户端基类
│   │   ├── client_factory.py # 客户端工厂
│   │   ├── local_client.py   # 本地模型客户端
│   │   └── zhipu_client.py   # 智谱 API 客户端
│   │
│   ├── inference/            # 推理引擎
│   │   ├── inference_engine.py   # 推理引擎
│   │   └── model_loader.py       # 模型加载器
│   │
│   ├── qa_engine/            # 问答引擎
│   │   ├── question_parser.py    # 问题解析器
│   │   ├── context_assembler.py  # 上下文组装器
│   │   └── answer_generator.py   # 答案生成器
│   │
│   ├── reasoning/            # 推理模块
│   │   ├── reasoning_orchestrator.py  # 推理编排器
│   │   └── evidence_chain.py          # 证据链
│   │
│   └── training/             # 模型训练
│       ├── dapt_trainer.py       # DAPT 预训练
│       ├── lora_trainer.py       # LoRA 微调
│       ├── data_collator.py      # 数据整理器
│       └── checkpoint_manager.py # 检查点管理
│
├── tests/                    # 测试代码
│   ├── unit/                 # 单元测试 (98个)
│   └── integration/          # 集成测试
│
├── config/                   # 配置文件
│   └── config.yaml           # 主配置文件
│
├── models/                   # 模型文件
│   └── base/                 # Qwen2.5-1.5B-Instruct
│
├── test_documents/           # 测试文档
│   ├── 01.docx               # 光网络规划 (9.4MB)
│   ├── 03.docx               # 光网络技术 (17.5MB, 24万字符)
│   └── ...
│
├── run_pipeline.py           # 完整流水线脚本
├── docker-compose.yml        # Neo4j 容器配置
└── pyproject.toml            # 项目配置 (uv)
```

## 模块详解

### 1. 数据处理模块 (`src/data_processing/`)

| 文件 | 类 | 职责 |
|------|-----|------|
| `document_loader.py` | `DocumentLoader` | 加载 .docx 文件，用 markitdown 转 Markdown |
| `entity_extractor.py` | `EntityExtractor` | 分块抽取实体，LLM + JSON 回退解析 |
| `triple_extractor.py` | `TripleExtractor` | 基于实体列表抽取三元组关系 |

**核心数据类**：
- `Document(source_path, content, metadata)`
- `Entity(name, entity_type, aliases, properties)`
- `Triple(subject, predicate, object, confidence)`

### 2. 知识图谱模块 (`src/knowledge_graph/`)

| 文件 | 类 | 职责 |
|------|-----|------|
| `neo4j_client.py` | `Neo4jClient` | 异步 Neo4j 驱动封装 |
| `graph_builder.py` | `GraphBuilder` | 将实体/三元组写入 Neo4j (MERGE 策略) |
| `graph_retriever.py` | `GraphRetriever` | 多跳图谱检索，返回 `HopResult` |
| `cypher_builder.py` | `CypherBuilder` | 安全生成 Cypher 查询语句 |
| `schema.py` | `GraphSchema` | 图谱 Schema 验证 |

**关键参数**：
- `GraphBuilder(auto_create_missing_nodes=True)` - 自动创建三元组中缺失的节点

### 3. LLM 模块 (`src/llm/` + `src/inference/`)

| 文件 | 类 | 职责 |
|------|-----|------|
| `base_client.py` | `BaseLLMClient` | 抽象基类，定义 `generate()`/`chat()` 接口 |
| `client_factory.py` | `create_llm_client()` | 根据配置创建客户端 |
| `local_client.py` | `LocalLLMClient` | 包装 InferenceEngine |
| `inference_engine.py` | `InferenceEngine` | 模型推理，支持 LoRA 热切换 |
| `model_loader.py` | `ModelLoader` | 4-bit 量化加载，PEFT 适配器 |

**使用方式**：
```python
from src.llm.client_factory import create_llm_client
from src.common.config import load_config

client = create_llm_client(load_config().llm)
await client.start()
result = await client.generate("你好")
await client.stop()
```

### 4. 问答与推理模块 (`src/qa_engine/` + `src/reasoning/`)

| 文件 | 类 | 职责 |
|------|-----|------|
| `question_parser.py` | `QuestionParser` | 解析问题意图和实体 |
| `reasoning_orchestrator.py` | `ReasoningOrchestrator` | 多跳推理编排 |
| `evidence_chain.py` | `EvidenceChain` | 证据链数据结构 |
| `context_assembler.py` | `ContextAssembler` | 组装 LLM 上下文 |
| `answer_generator.py` | `AnswerGenerator` | 生成最终答案 |

**问题意图类型** (`QueryIntent`):
- `FIND_RELATION` - 查找关系
- `EXPLAIN` - 解释概念
- `COMPARE` - 比较实体
- `LIST` - 列举
- `GENERAL` - 通用

### 5. API 模块 (`api/`)

| 文件 | 职责 |
|------|------|
| `main.py` | FastAPI 入口，lifespan 管理 Neo4j/LLM 初始化 |
| `dependencies.py` | 依赖注入工厂 |
| `routers/qa.py` | 问答接口 (同步/异步) |
| `routers/health.py` | 健康检查端点 |

## 重要文件速查

| 文件 | 说明 | 行数 |
|------|------|------|
| `checkpoint_manager.py` | 训练检查点管理 | 337 |
| `graph_builder.py` | 图谱构建核心逻辑 | 289 |
| `reasoning_orchestrator.py` | 多跳推理编排 | 288 |
| `lora_trainer.py` | LoRA 微调训练器 | 275 |
| `graph_retriever.py` | 图谱多跳检索 | 254 |
| `question_parser.py` | 问题解析 | 220 |
| `triple_extractor.py` | 三元组抽取 | 213 |
| `entity_extractor.py` | 实体抽取 | 190 |

## 配置文件

### `config/config.yaml`

```yaml
llm:
  provider: local
  model_path: models/base
  max_tokens: 512
  temperature: 0.7

graph:
  uri: bolt://localhost:7687
  user: neo4j
  password: ${NEO4J_PASSWORD}  # 环境变量插值

api:
  host: 0.0.0.0
  port: 8000

logging:
  level: INFO

data_processing:
  strip_base64_images: true
```

### 环境变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `NEO4J_PASSWORD` | Neo4j 密码 | `password123` |
