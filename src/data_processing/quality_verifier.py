"""Optional quality verification pass for extracted entities and relations.

After extraction, an LLM reviews the results for completeness,
accuracy, and consistency — producing a quality score and
actionable suggestions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.common.logger import get_logger
from src.data_processing.entity_extractor import Entity
from src.data_processing.triple_extractor import Triple
from src.llm.base_client import BaseLLMClient, GenerationParams


@dataclass
class QualityReport:
    """Result of a quality verification pass."""

    quality_score: float  # 0.0 – 1.0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    missing_entities: list[str] = field(default_factory=list)
    missing_relations: list[str] = field(default_factory=list)


class QualityVerifier:
    """Post-extraction quality verification via LLM review.

    Sends the original text and extracted entities/relations to the LLM
    and asks it to evaluate completeness and accuracy.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> None:
        self._llm_client = llm_client
        self._logger = get_logger(__name__)
        self._generation_params = GenerationParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.1,
        )

    async def verify(
        self,
        text: str,
        entities: list[Entity],
        triples: list[Triple],
    ) -> QualityReport:
        """Verify extraction quality and return a report."""

        prompt = self._build_prompt(text, entities, triples)

        try:
            response = await self._llm_client.generate(
                prompt=prompt, params=self._generation_params
            )
            return self._parse_report(response)
        except Exception as exc:
            self._logger.warning(f"Quality verification failed: {exc}")
            return QualityReport(
                quality_score=0.0,
                issues=[f"验证失败: {exc}"],
            )

    def _build_prompt(
        self,
        text: str,
        entities: list[Entity],
        triples: list[Triple],
    ) -> str:
        """Build quality check prompt."""

        entity_lines = [f"- {e.name}（{e.entity_type}）" for e in entities]
        triple_lines = [f"- {t.subject} → {t.predicate} → {t.object}" for t in triples]

        return (
            "你是知识图谱质量审核专家。请评估以下抽取结果的质量。\n\n"
            f"原始文本：\n{text[:3000]}\n\n"
            f"已抽取的实体（{len(entities)} 个）：\n" + "\n".join(entity_lines) + "\n\n"
            f"已抽取的关系（{len(triples)} 个）：\n" + "\n".join(triple_lines) + "\n\n"
            "请从以下角度评估并以JSON格式输出：\n"
            "1. completeness: 实体和关系是否完整覆盖了文本中的关键知识\n"
            "2. accuracy: 抽取结果是否准确反映文本含义\n"
            "3. consistency: 实体命名和关系表述是否一致\n\n"
            "输出格式：\n"
            '{"quality_score": 0.85, "issues": ["遗漏了XX实体"], '
            '"suggestions": ["建议补充XX关系"], '
            '"missing_entities": ["实体名1"], '
            '"missing_relations": ["主语→谓语→宾语"]}\n\n'
            "请直接返回JSON对象："
        )

    def _parse_report(self, response: str) -> QualityReport:
        """Parse LLM quality verification response."""

        # Strip thinking tags
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", response).strip()
        if not cleaned:
            cleaned = response

        payload = self._extract_json_object(cleaned)
        if payload is None:
            self._logger.warning("Could not parse quality report JSON")
            return QualityReport(
                quality_score=0.0,
                issues=["无法解析质量报告"],
            )

        score = payload.get("quality_score", 0.0)
        if not isinstance(score, (int, float)):
            score = 0.0
        score = max(0.0, min(1.0, float(score)))

        def _str_list(key: str) -> list[str]:
            raw = payload.get(key, [])
            if isinstance(raw, list):
                return [str(item) for item in raw if item]
            return []

        return QualityReport(
            quality_score=score,
            issues=_str_list("issues"),
            suggestions=_str_list("suggestions"),
            missing_entities=_str_list("missing_entities"),
            missing_relations=_str_list("missing_relations"),
        )

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        """Extract JSON object from LLM response."""

        # Strategy 1: markdown code block
        code_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```", text)
        if code_match:
            try:
                result = json.loads(code_match.group(1).strip())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        # Strategy 2: direct parse
        try:
            result = json.loads(text.strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Strategy 3: find outermost { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                result = json.loads(text[start : end + 1])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        return None
