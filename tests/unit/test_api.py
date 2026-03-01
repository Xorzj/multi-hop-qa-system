from __future__ import annotations

# ruff: noqa: E402
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

fastapi_module = pytest.importorskip("fastapi")
fastapi_testclient_module = pytest.importorskip("fastapi.testclient")
pydantic_module = pytest.importorskip("pydantic")

from api.dependencies import (
    get_answer_generator,
    get_context_assembler,
    get_neo4j_client,
    get_question_parser,
    get_reasoning_orchestrator,
)
from api.routers import health as health_router
from api.routers import qa as qa_router
from api.schemas.request import BatchQuestionRequest, QuestionRequest
from api.schemas.response import (
    AnswerResponse,
    Evidence,
    HealthResponse,
    TaskResponse,
    TaskStatus,
)
from src.qa_engine.answer_generator import GeneratedAnswer
from src.qa_engine.context_assembler import AssembledContext
from src.reasoning.evidence_chain import EvidenceChain, EvidenceEdge, EvidenceNode

FastAPI = fastapi_module.FastAPI
TestClient = fastapi_testclient_module.TestClient
ValidationError = pydantic_module.ValidationError


@dataclass
class DummyNeo4jClient:
    should_fail: bool = False

    async def execute(self, query: str) -> int:
        if self.should_fail:
            raise RuntimeError("neo4j down")
        return 1


class DummyParser:
    async def parse(self, question: str) -> str:
        return f"parsed:{question}"


class DummyOrchestrator:
    async def reason(self, parsed: str) -> EvidenceChain:
        chain = EvidenceChain(
            nodes=[EvidenceNode(name="A", label="Node")],
            edges=[
                EvidenceEdge(
                    source="A",
                    target="B",
                    relation_type="REL",
                    confidence=0.8,
                )
            ],
            total_confidence=0.8,
        )
        return chain


class DummyAssembler:
    def assemble(
        self, question: str, evidence: EvidenceChain, include_reasoning: bool = True
    ) -> AssembledContext:
        return AssembledContext(
            question=question,
            evidence_summary="summary",
            evidence_confidence=evidence.total_confidence,
            reasoning_steps=["step"] if include_reasoning else [],
            entity_descriptions={},
            prompt="prompt",
        )


class DummyGenerator:
    async def generate(
        self, context: AssembledContext, include_reasoning: bool = True
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer="42",
            confidence=0.9,
            reasoning_steps=["reason"] if include_reasoning else None,
            latency_ms=12.5,
        )


@pytest.fixture(autouse=True)
def clear_tasks() -> None:
    qa_router.tasks.clear()


@pytest.fixture
def health_app():
    app = FastAPI()
    app.include_router(health_router.router, prefix="/health")
    return app


@pytest.fixture
def qa_app():
    app = FastAPI()
    app.include_router(qa_router.router, prefix="/qa")
    app.dependency_overrides[get_question_parser] = lambda: DummyParser()
    app.dependency_overrides[get_reasoning_orchestrator] = lambda: DummyOrchestrator()
    app.dependency_overrides[get_context_assembler] = lambda: DummyAssembler()
    app.dependency_overrides[get_answer_generator] = lambda: DummyGenerator()
    return app


@pytest.fixture
def qa_client(qa_app):
    return TestClient(qa_app)


def test_question_request_valid_defaults() -> None:
    payload = QuestionRequest(question="What is SDH?")
    assert payload.max_hops == 3
    assert payload.include_evidence is True
    assert payload.domain is None


@pytest.mark.parametrize("question", ["", " "])
def test_question_request_invalid_question(question: str) -> None:
    with pytest.raises(ValidationError):
        QuestionRequest(question=question)


@pytest.mark.parametrize("max_hops", [0, 6])
def test_question_request_invalid_max_hops(max_hops: int) -> None:
    with pytest.raises(ValidationError):
        QuestionRequest(question="hi", max_hops=max_hops)


def test_question_request_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        QuestionRequest(question="hi", extra_field="nope")


def test_batch_question_request_defaults() -> None:
    payload = BatchQuestionRequest(questions=[QuestionRequest(question="hi")])
    assert payload.max_concurrent == 5
    assert len(payload.questions) == 1


def test_batch_question_request_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        BatchQuestionRequest(
            questions=[QuestionRequest(question="hi")], extra_field="nope"
        )


def test_evidence_validation() -> None:
    Evidence(path=["A"], confidence=0.0, source="A", relation_type=None)
    Evidence(path=["A"], confidence=1.0, source="A", relation_type="REL")
    with pytest.raises(ValidationError):
        Evidence(path=["A"], confidence=-0.1, source="A")
    with pytest.raises(ValidationError):
        Evidence(path=["A"], confidence=1.1, source="A")


def test_answer_response_validation() -> None:
    payload = AnswerResponse(
        answer="ok",
        confidence=0.9,
        evidence=[Evidence(path=["A"], confidence=0.5, source="A", relation_type=None)],
        reasoning_steps=["step"],
        latency_ms=10.2,
    )
    assert payload.answer == "ok"


def test_task_response_validation() -> None:
    payload = TaskResponse(
        task_id="t1",
        status=TaskStatus.PENDING,
        result=None,
        error=None,
        created_at=datetime.now(tz=UTC),
        completed_at=None,
    )
    assert payload.status == TaskStatus.PENDING


def test_task_response_invalid_status() -> None:
    with pytest.raises(ValidationError):
        TaskResponse(
            task_id="t1",
            status="UNKNOWN",
            result=None,
            error=None,
            created_at=datetime.now(tz=UTC),
            completed_at=None,
        )


def test_health_response_validation() -> None:
    payload = HealthResponse(
        status="healthy",
        version="0.1.0",
        neo4j_connected=True,
        llm_available=False,
    )
    assert payload.neo4j_connected is True


def test_health_check_ok(health_app) -> None:
    health_app.dependency_overrides[get_neo4j_client] = lambda: DummyNeo4jClient()
    health_app.state.llm_client = object()
    client = TestClient(health_app)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"
    assert data["neo4j_connected"] is True
    assert data["llm_available"] is True


def test_readiness_check_degraded(health_app) -> None:
    health_app.dependency_overrides[get_neo4j_client] = lambda: DummyNeo4jClient(
        should_fail=True
    )
    health_app.state.llm_client = object()
    client = TestClient(health_app)

    response = client.get("/health/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["neo4j_connected"] is False
    assert data["llm_available"] is True


def test_liveness_check(health_app) -> None:
    health_app.state.llm_client = object()
    health_app.state.neo4j_connected = True
    client = TestClient(health_app)

    response = client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["neo4j_connected"] is True
    assert data["llm_available"] is True


def test_submit_question_creates_task(
    qa_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_op_task(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(qa_router, "run_qa_pipeline", no_op_task)

    response = qa_client.post("/qa/submit", json={"question": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["task_id"] in qa_router.tasks


def test_status_endpoint_returns_task(qa_client) -> None:
    task_id = "task-1"
    qa_router.tasks[task_id] = qa_router.TaskData(
        task_id=task_id,
        status=TaskStatus.PENDING,
        result=None,
        error=None,
        created_at=qa_router.datetime.now(tz=qa_router.UTC),
        completed_at=None,
    )

    response = qa_client.get(f"/qa/status/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == task_id
    assert data["status"] == TaskStatus.PENDING.value


def test_status_endpoint_missing_task(qa_client) -> None:
    response = qa_client.get("/qa/status/missing")
    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found"


def test_result_endpoint_returns_task(qa_client) -> None:
    task_id = "task-2"
    qa_router.tasks[task_id] = qa_router.TaskData(
        task_id=task_id,
        status=TaskStatus.COMPLETED,
        result=None,
        error=None,
        created_at=qa_router.datetime.now(tz=qa_router.UTC),
        completed_at=qa_router.datetime.now(tz=qa_router.UTC),
    )

    response = qa_client.get(f"/qa/result/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == task_id
    assert data["status"] == TaskStatus.COMPLETED.value


def test_sync_question_returns_answer(qa_client) -> None:
    response = qa_client.post("/qa/sync", json={"question": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "42"
    assert data["confidence"] == 0.9
    assert data["evidence"][0]["path"] == ["A", "B"]
    assert data["evidence"][0]["relation_type"] == "REL"
    assert data["reasoning_steps"] == ["reason"]


def test_sync_question_without_reasoning(qa_client) -> None:
    response = qa_client.post(
        "/qa/sync", json={"question": "hello", "include_evidence": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["reasoning_steps"] is None
