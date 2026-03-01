# API 接口文档

> 本文档说明系统的 HTTP API 接口。

## 基础信息

| 项目 | 值 |
|------|-----|
| 基础 URL | `http://localhost:8000` |
| 内容类型 | `application/json` |
| 编码 | UTF-8 |

## 启动服务

```bash
cd /home/xorzj/PROJECT
export NEO4J_PASSWORD=password123
uv run uvicorn api.main:app --reload --port 8000
```

---

## 问答接口

### 1. 同步问答

同步方式提交问题并等待答案。

**请求**

```http
POST /qa/sync
Content-Type: application/json
```

```json
{
  "question": "华为的 OptiX OSN 8800 支持什么协议？",
  "max_hops": 3,
  "include_evidence": true
}
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `question` | string | ✅ | - | 用户问题 |
| `max_hops` | int | ❌ | 3 | 最大推理跳数 |
| `include_evidence` | bool | ❌ | true | 是否返回证据链 |

**响应** (200 OK)

```json
{
  "answer": "华为的 OptiX OSN 8800 支持 SDH/MSTP 协议，该协议遵循 ITU-T 国际标准。",
  "confidence": 0.85,
  "reasoning_steps": [
    {
      "hop": 1,
      "from_entity": "华为",
      "relation": "生产",
      "to_entity": "OptiX OSN 8800"
    },
    {
      "hop": 2,
      "from_entity": "OptiX OSN 8800",
      "relation": "支持",
      "to_entity": "SDH/MSTP"
    }
  ],
  "evidence": [
    {
      "subject": "华为",
      "predicate": "生产",
      "object": "OptiX OSN 8800",
      "source": "知识图谱"
    },
    {
      "subject": "OptiX OSN 8800",
      "predicate": "支持",
      "object": "SDH/MSTP",
      "source": "知识图谱"
    }
  ]
}
```

**错误响应**

| 状态码 | 说明 |
|--------|------|
| 400 | 请求参数错误 |
| 500 | 服务器内部错误 |
| 503 | 服务不可用（LLM 未就绪） |

---

### 2. 异步提交问题

提交问题后立即返回任务 ID，适合长时间推理。

**请求**

```http
POST /qa/submit
Content-Type: application/json
```

```json
{
  "question": "DWDM 和 SDH 有什么关系？",
  "max_hops": 5
}
```

**响应** (202 Accepted)

```json
{
  "task_id": "task_abc123",
  "status": "pending",
  "message": "任务已提交"
}
```

---

### 3. 查询任务状态

**请求**

```http
GET /qa/status/{task_id}
```

**响应** (200 OK)

```json
{
  "task_id": "task_abc123",
  "status": "running",
  "progress": {
    "current_hop": 2,
    "max_hops": 5
  }
}
```

| 状态值 | 说明 |
|--------|------|
| `pending` | 等待处理 |
| `running` | 正在推理 |
| `completed` | 已完成 |
| `failed` | 失败 |

---

### 4. 获取任务结果

**请求**

```http
GET /qa/result/{task_id}
```

**响应** (200 OK)

```json
{
  "task_id": "task_abc123",
  "status": "completed",
  "result": {
    "answer": "DWDM 和 SDH 是两种不同的光传输技术...",
    "confidence": 0.82,
    "reasoning_steps": [...],
    "evidence": [...]
  }
}
```

**错误响应**

| 状态码 | 说明 |
|--------|------|
| 404 | 任务不存在 |
| 425 | 任务尚未完成 |

---

## 健康检查接口

### 1. 基本健康检查

**请求**

```http
GET /health
```

**响应** (200 OK)

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

### 2. 就绪检查

检查所有依赖服务是否就绪。

**请求**

```http
GET /health/ready
```

**响应** (200 OK)

```json
{
  "status": "ready",
  "checks": {
    "neo4j": true,
    "llm": true
  }
}
```

**响应** (503 Service Unavailable)

```json
{
  "status": "not_ready",
  "checks": {
    "neo4j": true,
    "llm": false
  }
}
```

---

### 3. 存活检查

Kubernetes liveness probe 用。

**请求**

```http
GET /health/live
```

**响应** (200 OK)

```json
{
  "status": "alive"
}
```

---

## 请求/响应模型定义

### QuestionRequest

```python
class QuestionRequest(BaseModel):
    question: str                    # 用户问题（必填）
    max_hops: int = 3                # 最大推理跳数
    include_evidence: bool = True    # 是否返回证据链
```

### AnswerResponse

```python
class AnswerResponse(BaseModel):
    answer: str                      # 生成的答案
    confidence: float                # 置信度 (0-1)
    reasoning_steps: list[ReasoningStep]  # 推理步骤
    evidence: list[Evidence]         # 使用的证据
```

### TaskResponse

```python
class TaskResponse(BaseModel):
    task_id: str                     # 任务 ID
    status: str                      # 状态
    message: str | None = None       # 附加消息
```

### Evidence

```python
class Evidence(BaseModel):
    subject: str                     # 主语
    predicate: str                   # 谓语（关系）
    object: str                      # 宾语
    source: str = "知识图谱"          # 来源
```

---

## 使用示例

### Python

```python
import requests

# 同步问答
response = requests.post(
    "http://localhost:8000/qa/sync",
    json={
        "question": "什么是 DWDM？",
        "max_hops": 3
    }
)
print(response.json()["answer"])
```

### cURL

```bash
# 同步问答
curl -X POST http://localhost:8000/qa/sync \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是 DWDM？", "max_hops": 3}'

# 健康检查
curl http://localhost:8000/health/ready
```

### httpie

```bash
http POST localhost:8000/qa/sync question="什么是 DWDM？"
```

---

## 错误处理

所有错误响应遵循统一格式：

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "question 字段不能为空",
    "details": {}
  }
}
```

| 错误码 | 说明 |
|--------|------|
| `VALIDATION_ERROR` | 请求参数验证失败 |
| `NOT_FOUND` | 资源不存在 |
| `SERVICE_UNAVAILABLE` | 服务不可用 |
| `INTERNAL_ERROR` | 内部错误 |
