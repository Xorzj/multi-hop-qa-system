# AGENTS.md - 多跳推理知识图谱问答系统

> AI 编程助手在本项目中的工作指南。

## 项目概述

基于多跳推理优化的垂域智能问答系统。从 .docx 文档构建知识图谱，支持多跳推理问答。

## 当前进度

| 阶段 | 状态 | 描述 |
|------|------|------|
| 阶段 0 | ✅ 完成 | 项目初始化 (uv, 配置, 日志) |
| 阶段 1 | ✅ 完成 | 核心抽象层 (LLM客户端, Neo4j客户端) |
| 阶段 2 | ✅ 完成 | 离线流水线 (docx→实体→三元组→Neo4j) |
| 阶段 3 | ⏳ 待完成 | 模型训练 (DAPT, LoRA/QLoRA) |
| 阶段 4 | ✅ 完成 | 推理引擎 (本地模型加载) |
| 阶段 5 | ✅ 完成 | 多跳问答 (图谱检索, 推理链) |
| 阶段 6 | ✅ 完成 | API层 (FastAPI) |

## 快速开始

### 1. 环境配置
```bash
cd /home/xorzj/PROJECT
uv sync --group dev --group dl --group llm --group graph --group api
```

### 2. 启动 Neo4j
```bash
docker compose up -d
export NEO4J_PASSWORD=password123
```

### 3. 运行完整流水线
```bash
uv run python run_pipeline.py
```

### 4. 查看知识图谱
浏览器打开: http://localhost:7474 (用户名: neo4j / 密码: password123)
```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50
```

## 已验证的流水线 (2025-03-01)

**测试文档**: `test_documents/03.docx` (24万字符, 光网络技术)

| 步骤 | 结果 |
|------|------|
| 文档加载 | 240,312 字符 (markitdown) |
| 实体抽取 | 11 个实体 (DWDM, OADM, WDM, SDH...) |
| 三元组抽取 | 13 个三元组 |
| 图谱构建 | 18 节点, 13 关系 |

**性能基准** (RTX 5070 Ti, Qwen2.5-1.5B-Instruct 4-bit):
- 单个分块 (~2000字符): 5-8 秒
- 完整文档 (24万字符): 约 25-40 分钟

## 技术栈

| 组件 | 技术 |
|------|------|
| 包管理器 | uv |
| Python | >=3.12 |
| 图数据库 | Neo4j 5.26 (Docker) |
| 大语言模型 | Qwen2.5-1.5B-Instruct (4-bit量化) |
| 文档处理 | markitdown[docx] |
| API框架 | FastAPI |
| 训练框架 | PEFT + Transformers |

## 项目结构

```
src/
├── common/           # 配置、日志、异常
├── data_processing/  # 文档加载器、实体抽取器、三元组抽取器
├── knowledge_graph/  # Neo4j客户端、图谱构建器、图谱检索器
├── llm/              # LLM基类、本地LLM客户端、客户端工厂
├── inference/        # 推理引擎、模型加载器
├── qa_engine/        # 问题解析器、上下文组装器、答案生成器
├── reasoning/        # 证据链、推理编排器
└── training/         # DAPT训练器、LoRA训练器

api/
├── main.py           # FastAPI 应用入口
├── dependencies.py   # 依赖注入
├── routers/          # qa.py, health.py
└── schemas/          # request.py, response.py

tests/
├── unit/             # 98 个单元测试
└── integration/      # 集成测试
```

## 核心组件说明

### 实体抽取器 (EntityExtractor)
- 文本分块处理 (默认2000字符, 重叠200字符)
- 基于LLM抽取，带JSON解析回退机制 (正则提取)
- 按实体名称去重

### 三元组抽取器 (TripleExtractor)
- 上下文感知抽取，使用实体列表约束
- 相同的分块和JSON回退机制
- 支持 `predicate` 或 `relation` 字段

### 图谱构建器 (GraphBuilder)
- `auto_create_missing_nodes=True` 自动创建三元组中引用但实体列表缺失的节点
- 自动创建的节点标签为 "自动创建"

### 推理引擎 (InferenceEngine)
- Qwen ChatML 格式: `<|im_start|>{role}\n{content}<|im_end|>\n`
- 支持 LoRA 适配器热切换

## 常用命令

```bash
# 代码检查与格式化
uv run ruff format . && uv run ruff check --fix . && uv run mypy src/ api/

# 运行测试
uv run pytest                           # 全部测试
uv run pytest tests/unit/ -v            # 仅单元测试
uv run pytest --cov=src                 # 带覆盖率

# 启动开发服务器
uv run uvicorn api.main:app --reload --port 8000

# 快速推理测试
uv run python -m src.inference.inference_engine --prompt "测试问题"
```

## 代码规范 (强制)

- **类型注解**: 所有函数签名必须有类型注解
- **禁止类型压制**: 不允许使用 `as any`、`# type: ignore`
- **导入顺序**: 标准库 → 第三方库 → 本地模块 (空行分隔)
- **异步**: 所有I/O操作使用 `async/await`
- **异常处理**: 只捕获具体异常，禁止空 `except`

## Git 提交规范

- 提交格式: `<类型>: <描述>`
- 类型: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
- 每次提交只包含一个逻辑变更

## 后续工作 (TODO)

1. **模型训练** (阶段 3)
   - 实现领域语料 DAPT 预训练
   - QA 任务 LoRA 微调

2. **抽取质量优化**
   - 改进实体/三元组抽取提示词
   - 添加实体类型约束
   - 考虑升级到 7B 模型

3. **多跳问答演示**
   - 端到端图谱问答
   - 证据链可视化
