from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Evidence(BaseModel):
    """Evidence returned from a knowledge graph traversal."""

    model_config = ConfigDict(extra="forbid")

    path: list[str]
    confidence: float = Field(..., ge=0, le=1)
    source: str
    relation_type: str | None = Field(default=None)


class AnswerResponse(BaseModel):
    """Answer payload for a completed QA request."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    confidence: float
    evidence: list[Evidence]
    reasoning_steps: list[str] | None = Field(default=None)
    latency_ms: float
    evidence_xml: str | None = Field(default=None)


class TaskStatus(StrEnum):
    """Status values for asynchronous QA tasks."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskResponse(BaseModel):
    """Response model for task status polling."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: TaskStatus
    result: AnswerResponse | None = Field(default=None)
    error: str | None = Field(default=None)
    created_at: datetime
    completed_at: datetime | None = Field(default=None)


class HealthResponse(BaseModel):
    """Health status response for dependencies and service."""

    model_config = ConfigDict(extra="forbid")

    status: str
    version: str
    neo4j_connected: bool
    llm_available: bool
