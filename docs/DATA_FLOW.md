# 数据流程详解

> 本文档说明系统中数据的流转过程。

## 概览

系统分为两个主要流程：

1. **离线流水线**：文档 → 知识图谱（批量处理）
2. **在线问答**：问题 → 答案（实时响应）

---

## 离线流水线

### 流程图

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│  .docx 文档  │───▶│  DocumentLoader │───▶│  Markdown 文本   │───▶│  文本分块      │
└─────────────┘    └──────────────┘    └─────────────────┘    └──────────────┘
                                                                      │
                        ┌─────────────────────────────────────────────┘
                        ▼
              ┌──────────────────┐
              │  EntityExtractor  │
              │  (LLM 抽取实体)   │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  TripleExtractor  │
              │  (LLM 抽取三元组) │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐    ┌──────────────┐
              │   GraphBuilder    │───▶│    Neo4j     │
              │  (构建知识图谱)   │    │   知识图谱    │
              └──────────────────┘    └──────────────┘
```

### 详细步骤

#### 1. 文档加载

```python
loader = DocumentLoader()
doc = loader.load('test_documents/03.docx')
content = doc.content  # Markdown 格式文本
```

- 输入：`.docx` 文件
- 处理：markitdown 转换
- 输出：`Document(source_path, content, metadata)`

#### 2. 文本分块

```python
# 内部实现
chunks = _chunk_text(content, chunk_size=2000, overlap=200)
```

- 每块最大 2000 字符
- 相邻块重叠 200 字符（避免截断实体）
- 24万字符 → ~134 个 chunks

#### 3. 实体抽取

```python
entity_extractor = EntityExtractor(llm_client)
entities = await entity_extractor.extract(content, chunk_size=2000)
```

- 每个 chunk 独立调用 LLM
- JSON 解析带正则回退
- 按实体名称去重合并

**输出示例**：
```json
[
  {"name": "DWDM", "type": "技术"},
  {"name": "OADM", "type": "设备"},
  {"name": "SDH", "type": "技术"}
]
```

#### 4. 三元组抽取

```python
triple_extractor = TripleExtractor(llm_client)
triples = await triple_extractor.extract(content, entities, chunk_size=2000)
```

- 使用实体列表作为约束
- 只在已知实体间建立关系
- 同样的分块 + JSON 回退机制

**输出示例**：
```json
[
  {"subject": "全光网络", "predicate": "使用", "object": "DWDM"},
  {"subject": "DWDM", "predicate": "采用", "object": "OADM"}
]
```

#### 5. 图谱构建

```python
builder = GraphBuilder(neo4j_client, auto_create_missing_nodes=True)
stats = await builder.build_from_extraction(entities, triples)
```

- MERGE 策略：节点存在则更新，不存在则创建
- 自动创建三元组中引用但实体列表缺失的节点
- 自动创建的节点标记为 "自动创建"

---

## 在线问答流程

### 流程图

```
┌──────────┐    ┌─────────────────┐    ┌────────────────────┐
│  用户问题  │───▶│  QuestionParser  │───▶│  ParsedQuestion    │
└──────────┘    │  (解析意图/实体)  │    │  (intent, entities) │
                └─────────────────┘    └──────────┬─────────┘
                                                  │
                                                  ▼
                                      ┌────────────────────┐
                                      │ ReasoningOrchestrator│
                                      │   (多跳推理编排)      │
                                      └──────────┬─────────┘
                                                 │
                         ┌───────────────────────┼───────────────────────┐
                         ▼                       ▼                       ▼
                   ┌──────────┐           ┌──────────┐           ┌──────────┐
                   │  Hop 1   │──────────▶│  Hop 2   │──────────▶│  Hop 3   │
                   │ 图谱检索  │           │ 图谱检索  │           │ 图谱检索  │
                   └──────────┘           └──────────┘           └──────────┘
                         │                       │                       │
                         └───────────────────────┼───────────────────────┘
                                                 ▼
                                      ┌────────────────────┐
                                      │   EvidenceChain    │
                                      │   (证据链)          │
                                      └──────────┬─────────┘
                                                 │
                                                 ▼
                                      ┌────────────────────┐
                                      │  ContextAssembler  │
                                      │  (组装 LLM 上下文)  │
                                      └──────────┬─────────┘
                                                 │
                                                 ▼
                                      ┌────────────────────┐    ┌──────────┐
                                      │  AnswerGenerator   │───▶│  最终答案  │
                                      │  (LLM 生成答案)     │    └──────────┘
                                      └────────────────────┘
```

### 详细步骤

#### 1. 问题解析

```python
parser = QuestionParser(llm_client)
parsed = await parser.parse("华为的 OptiX OSN 8800 支持什么协议？")
```

**输出**：
```python
ParsedQuestion(
    original="华为的 OptiX OSN 8800 支持什么协议？",
    intent=QueryIntent.FIND_RELATION,
    entities=["华为", "OptiX OSN 8800"],
    constraints={"relation_type": "支持"}
)
```

#### 2. 多跳推理

```python
orchestrator = ReasoningOrchestrator(graph_retriever, llm_client)
evidence_chain = await orchestrator.reason(parsed, max_hops=3)
```

**每跳操作**：
1. 从当前实体出发，查询邻居节点
2. LLM 决定是否继续（`_decide_next_hop()`）
3. 如果继续，选择下一跳实体
4. 重复直到达到 max_hops 或 LLM 判断足够

**证据链示例**：
```
华为 --[生产]--> OptiX OSN 8800 --[支持]--> SDH/MSTP --[遵循]--> ITU-T
```

#### 3. 上下文组装

```python
assembler = ContextAssembler()
context = assembler.assemble(parsed, evidence_chain)
```

将证据链转换为 LLM 可理解的 Markdown 格式：

```markdown
## 问题
华为的 OptiX OSN 8800 支持什么协议？

## 知识图谱证据
- 华为 生产 OptiX OSN 8800
- OptiX OSN 8800 支持 SDH/MSTP
- SDH/MSTP 遵循 ITU-T 标准

## 请根据以上证据回答问题
```

#### 4. 答案生成

```python
generator = AnswerGenerator(llm_client)
answer = await generator.generate(context)
```

**输出**：
```python
GeneratedAnswer(
    answer="华为的 OptiX OSN 8800 支持 SDH/MSTP 协议，该协议遵循 ITU-T 国际标准。",
    confidence=0.85,
    evidence_used=["OptiX OSN 8800 支持 SDH/MSTP", ...]
)
```

---

## 数据结构一览

### 离线流水线

| 阶段 | 输入 | 输出 |
|------|------|------|
| 文档加载 | `.docx` 文件路径 | `Document` |
| 实体抽取 | 文本字符串 | `list[Entity]` |
| 三元组抽取 | 文本 + 实体列表 | `list[Triple]` |
| 图谱构建 | 实体 + 三元组 | Neo4j 节点/关系 |

### 在线问答

| 阶段 | 输入 | 输出 |
|------|------|------|
| 问题解析 | 用户问题字符串 | `ParsedQuestion` |
| 多跳推理 | ParsedQuestion + max_hops | `EvidenceChain` |
| 上下文组装 | ParsedQuestion + EvidenceChain | `AssembledContext` |
| 答案生成 | AssembledContext | `GeneratedAnswer` |

---

## 性能参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | 2000 | 文本分块大小（字符） |
| `overlap` | 200 | 分块重叠（字符） |
| `max_hops` | 3 | 最大推理跳数 |
| `max_tokens` | 512 | LLM 最大生成长度 |
| `temperature` | 0.7 | LLM 采样温度 |
