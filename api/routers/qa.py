from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from api.dependencies import (
    get_answer_generator,
    get_context_assembler,
    get_question_parser,
    get_reasoning_orchestrator,
)
from api.schemas.request import QuestionRequest
from api.schemas.response import AnswerResponse, Evidence, TaskResponse, TaskStatus
from src.common.logger import get_logger
from src.qa_engine.answer_generator import AnswerGenerator, GeneratedAnswer
from src.qa_engine.context_assembler import ContextAssembler
from src.qa_engine.question_parser import QuestionParser
from src.reasoning.evidence_chain import EvidenceChain
from src.reasoning.reasoning_orchestrator import ReasoningOrchestrator

logger = get_logger(__name__)

router = APIRouter()


@dataclass
class TaskData:
    task_id: str
    status: TaskStatus
    result: AnswerResponse | None
    error: str | None
    created_at: datetime
    completed_at: datetime | None


tasks: dict[str, TaskData] = {}


def _get_task_or_404(task_id: str) -> TaskData:
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


def _to_task_response(task: TaskData) -> TaskResponse:
    return TaskResponse(
        task_id=task.task_id,
        status=task.status,
        result=task.result,
        error=task.error,
        created_at=task.created_at,
        completed_at=task.completed_at,
    )


def _convert_evidence_chain(chain: EvidenceChain) -> list[Evidence]:
    if not chain.edges:
        if chain.nodes:
            first_node = chain.nodes[0]
            return [
                Evidence(
                    path=[first_node.name],
                    confidence=chain.total_confidence,
                    source=first_node.name,
                    relation_type=None,
                )
            ]
        return []
    return [
        Evidence(
            path=[edge.source, edge.target],
            confidence=edge.confidence,
            source=edge.source,
            relation_type=edge.relation_type,
        )
        for edge in chain.edges
    ]


def _map_generated_answer(
    generated: GeneratedAnswer, evidence: EvidenceChain
) -> AnswerResponse:
    return AnswerResponse(
        answer=generated.answer,
        confidence=generated.confidence,
        evidence=_convert_evidence_chain(evidence),
        reasoning_steps=generated.reasoning_steps,
        latency_ms=generated.latency_ms,
        evidence_xml=evidence.to_xml() if evidence.edges or evidence.nodes else None,
    )


async def run_qa_pipeline(
    task_id: str,
    request: QuestionRequest,
    parser: QuestionParser,
    orchestrator: ReasoningOrchestrator,
    assembler: ContextAssembler,
    generator: AnswerGenerator,
) -> None:
    task = tasks.get(task_id)
    if task is None:
        logger.warning("QA task missing", extra={"task_id": task_id})
        return
    task.status = TaskStatus.PROCESSING
    try:
        parsed = await parser.parse(request.question)
        evidence = await orchestrator.reason(parsed)
        include_reasoning = request.include_evidence
        context = assembler.assemble(
            question=request.question,
            evidence=evidence,
            include_reasoning=include_reasoning,
        )
        generated = await generator.generate(
            context=context, include_reasoning=include_reasoning
        )
        task.result = _map_generated_answer(generated, evidence)
        task.status = TaskStatus.COMPLETED
    except Exception as exc:  # noqa: BLE001
        logger.exception("QA task failed", extra={"task_id": task_id})
        task.status = TaskStatus.FAILED
        task.error = str(exc)
    task.completed_at = datetime.now(tz=UTC)


@router.post("/submit")
async def submit_question(
    request: QuestionRequest,
    background_tasks: BackgroundTasks,
    parser: QuestionParser = Depends(get_question_parser),
    orchestrator: ReasoningOrchestrator | None = Depends(get_reasoning_orchestrator),
    assembler: ContextAssembler = Depends(get_context_assembler),
    generator: AnswerGenerator = Depends(get_answer_generator),
) -> dict[str, str]:
    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Neo4j not available, QA features are disabled",
        )
    task_id = str(uuid4())
    tasks[task_id] = TaskData(
        task_id=task_id,
        status=TaskStatus.PENDING,
        result=None,
        error=None,
        created_at=datetime.now(tz=UTC),
        completed_at=None,
    )
    background_tasks.add_task(
        run_qa_pipeline,
        task_id,
        request,
        parser,
        orchestrator,
        assembler,
        generator,
    )
    return {"task_id": task_id}


@router.get("/status/{task_id}")
async def get_status(task_id: str) -> TaskResponse:
    task = _get_task_or_404(task_id)
    return _to_task_response(task)


@router.get("/result/{task_id}")
async def get_result(task_id: str) -> TaskResponse:
    task = _get_task_or_404(task_id)
    return _to_task_response(task)


@router.post("/sync")
async def sync_question(
    request: QuestionRequest,
    parser: QuestionParser = Depends(get_question_parser),
    orchestrator: ReasoningOrchestrator | None = Depends(get_reasoning_orchestrator),
    assembler: ContextAssembler = Depends(get_context_assembler),
    generator: AnswerGenerator = Depends(get_answer_generator),
) -> AnswerResponse:
    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Neo4j not available, QA features are disabled",
        )
    parsed = await parser.parse(request.question)
    evidence = await orchestrator.reason(parsed)
    include_reasoning = request.include_evidence
    context = assembler.assemble(
        question=request.question,
        evidence=evidence,
        include_reasoning=include_reasoning,
    )
    generated = await generator.generate(
        context=context, include_reasoning=include_reasoning
    )
    return _map_generated_answer(generated, evidence)
