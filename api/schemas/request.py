from pydantic import BaseModel, ConfigDict, Field, field_validator


class QuestionRequest(BaseModel):
    """Request model for a single question."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1)
    max_hops: int = Field(default=3, ge=1, le=5)
    include_evidence: bool = Field(default=True)
    domain: str | None = Field(default=None)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            msg = "Question cannot be blank or whitespace only"
            raise ValueError(msg)
        return v


class BatchQuestionRequest(BaseModel):
    """Request model for a batch of questions."""

    model_config = ConfigDict(extra="forbid")

    questions: list[QuestionRequest]
    max_concurrent: int = Field(default=5)
