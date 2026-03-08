from __future__ import annotations

import re
import time
from dataclasses import dataclass

from src.common.logger import get_logger
from src.llm.base_client import BaseLLMClient, GenerationParams
from src.qa_engine.context_assembler import AssembledContext

logger = get_logger(__name__)


@dataclass
class GeneratedAnswer:
    answer: str
    confidence: float
    reasoning_steps: list[str] | None
    latency_ms: float
    tokens_used: int | None = None


class AnswerGenerator:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        default_max_tokens: int = 1024,
        default_temperature: float = 0.3,
    ) -> None:
        self._llm_client = llm_client
        self._default_max_tokens = default_max_tokens
        self._default_temperature = default_temperature
        self._logger = get_logger(__name__)

    async def generate(
        self, context: AssembledContext, include_reasoning: bool = True
    ) -> GeneratedAnswer:
        params = self._build_params(
            max_tokens=self._default_max_tokens,
            temperature=self._default_temperature,
        )
        prompt = context.prompt or context.question
        start = time.perf_counter()
        response = await self._llm_client.generate(prompt=prompt, params=params)
        latency_ms = (time.perf_counter() - start) * 1000
        answer_text, reasoning_steps = self._extract_reasoning(response)
        if not include_reasoning:
            reasoning_steps = None
        confidence = self._estimate_confidence(
            answer=answer_text, evidence_confidence=context.evidence_confidence
        )
        return GeneratedAnswer(
            answer=answer_text,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            latency_ms=latency_ms,
            tokens_used=None,
        )

    async def generate_simple(
        self, question: str, context_text: str
    ) -> GeneratedAnswer:
        prompt = "\n".join(
            [
                "你是一个专业的知识问答助手。请基于以下上下文回答问题。",
                "",
                "## 用户问题",
                question,
                "",
                "## 上下文",
                context_text,
                "",
                "请回答：",
            ]
        )
        start = time.perf_counter()
        response = await self._llm_client.generate(
            prompt=prompt, params=self._build_params(None, None)
        )
        latency_ms = (time.perf_counter() - start) * 1000
        answer_text, reasoning_steps = self._extract_reasoning(response)
        confidence = self._estimate_confidence(answer_text, evidence_confidence=0.6)
        return GeneratedAnswer(
            answer=answer_text,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            latency_ms=latency_ms,
            tokens_used=None,
        )

    def _estimate_confidence(self, answer: str, evidence_confidence: float) -> float:
        confidence = max(0.0, min(1.0, evidence_confidence))
        normalized = answer.strip()
        if any(token in normalized for token in ("可能", "不确定", "无法确定")):
            confidence -= 0.2
        if len(normalized) < 20:
            confidence -= 0.1
        return max(0.0, min(1.0, confidence))

    def _extract_reasoning(self, answer: str) -> tuple[str, list[str] | None]:
        cleaned = answer.strip()
        if not cleaned:
            return "", None

        for marker in ("推理过程:", "分析:"):
            if marker in cleaned:
                main, _, reasoning = cleaned.partition(marker)
                steps = self._split_reasoning_steps(reasoning)
                return main.strip(), steps
        return cleaned, None

    def _split_reasoning_steps(self, reasoning: str) -> list[str] | None:
        trimmed = reasoning.strip()
        if not trimmed:
            return None
        parts = [
            part.strip() for part in re.split(r"\n+|；|;", trimmed) if part.strip()
        ]
        return parts or None

    def _build_params(
        self, max_tokens: int | None, temperature: float | None
    ) -> GenerationParams:
        resolved_max_tokens = max_tokens or self._default_max_tokens
        resolved_temperature = temperature or self._default_temperature
        return GenerationParams(
            max_new_tokens=resolved_max_tokens,
            temperature=resolved_temperature,
        )
