from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.common.logger import get_logger
from src.llm.base_client import BaseLLMClient, GenerationParams


class QueryIntent(StrEnum):
    FIND_ENTITY = "FIND_ENTITY"
    FIND_RELATION = "FIND_RELATION"
    FIND_PATH = "FIND_PATH"
    COMPARE = "COMPARE"
    EXPLAIN = "EXPLAIN"
    COUNT = "COUNT"
    LIST = "LIST"


@dataclass
class ParsedQuestion:
    original: str
    intent: QueryIntent
    entities: list[str] = field(default_factory=list)
    relation_hints: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)


class QuestionParser:
    def __init__(self, llm_client: BaseLLMClient) -> None:
        self._llm_client = llm_client
        self._logger = get_logger(__name__)
        self._generation_params = GenerationParams(temperature=0.2, max_new_tokens=256)

    async def parse(self, question: str) -> ParsedQuestion:
        if not question.strip():
            return ParsedQuestion(
                original=question,
                intent=QueryIntent.EXPLAIN,
                entities=[],
                relation_hints=[],
                constraints={},
            )

        prompt = self._build_prompt(question)
        try:
            response = await self._llm_client.generate(
                prompt=prompt, params=self._generation_params
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "LLM parsing failed, using fallback",
                extra={"error": str(exc)},
            )
            return self._fallback_parse(question)

        parsed = self._parse_response(response, original=question)
        if parsed.entities or parsed.relation_hints:
            return parsed
        return self._fallback_parse(question)

    def _build_prompt(self, question: str) -> str:
        return (
            "你是光通信/SDH领域的问题解析助手。"
            "从问题中抽取意图、实体、关系提示和约束，并仅返回JSON对象。"
            "意图枚举：FIND_ENTITY、FIND_RELATION、FIND_PATH、COMPARE、EXPLAIN、COUNT、LIST。"
            "示例：\n"
            "问题：SDH和STM-1什么关系\n"
            "返回："
            '{"intent":"FIND_RELATION","entities":["SDH","STM-1"],'
            '"relation_hints":["使用","包含"],"constraints":{"max_hops":2}}\n'
            "问题：SDH是什么\n"
            '返回：{"intent":"EXPLAIN","entities":["SDH"],'
            '"relation_hints":[],"constraints":{}}\n'
            "问题：哪些设备使用光纤\n"
            '返回：{"intent":"FIND_ENTITY","entities":["光纤"],'
            '"relation_hints":["使用"],"constraints":{}}\n'
            "问题：从SDH到PDH怎么走\n"
            '返回：{"intent":"FIND_PATH","entities":["SDH","PDH"],'
            '"relation_hints":["路径"],"constraints":{"max_hops":3}}\n'
            "问题：有多少种协议\n"
            '返回：{"intent":"COUNT","entities":["协议"],'
            '"relation_hints":[],"constraints":{}}\n'
            "现在请解析以下问题，仅返回JSON对象：\n"
            f"{question}"
        )

    def _parse_response(self, response: str, original: str) -> ParsedQuestion:
        try:
            payload = json.loads(self._strip_code_fences(response))
        except json.JSONDecodeError:
            self._logger.warning("Failed to parse question response")
            return self._fallback_parse(original)

        if not isinstance(payload, dict):
            self._logger.warning("Question response is not a JSON object")
            return self._fallback_parse(original)

        intent_raw = payload.get("intent")
        intent = self._parse_intent(intent_raw)
        entities = self._coerce_str_list(payload.get("entities"))
        relation_hints = self._coerce_str_list(payload.get("relation_hints"))
        constraints = payload.get("constraints")
        constraint_map = constraints if isinstance(constraints, dict) else {}
        return ParsedQuestion(
            original=original,
            intent=intent,
            entities=entities,
            relation_hints=relation_hints,
            constraints=constraint_map,
        )

    def _extract_entities_fallback(self, question: str) -> list[str]:
        entities = []
        quoted = re.findall(r"[\"'“”‘’]([^\"'“”‘’]{1,30})[\"'“”‘’]", question)
        for item in quoted:
            cleaned = item.strip()
            if cleaned and cleaned not in entities:
                entities.append(cleaned)

        patterns = [
            r"\b(?:SDH|PDH|WDM|OTN|SONET|MSTP|PTN)\b",
            r"\bSTM-\d+\b",
            r"\bVC-\d+\b",
            r"\bE\d+\b",
            r"\bTU-\d+\b",
            r"\bAU-\d+\b",
            r"\bODU\d+\b",
            r"\bOTU\d+\b",
            r"\bG\.\d{3}\b",
            r"\b\d+(?:\.\d+)?Gb/s\b",
            r"\b\d+(?:\.\d+)?Mb/s\b",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, question, flags=re.IGNORECASE):
                normalized = match.strip()
                if normalized and normalized not in entities:
                    entities.append(normalized)
        return entities

    def _fallback_parse(self, question: str) -> ParsedQuestion:
        entities = self._extract_entities_fallback(question)
        intent = self._infer_intent_from_text(question)
        relation_hints = self._infer_relation_hints(question)
        return ParsedQuestion(
            original=question,
            intent=intent,
            entities=entities,
            relation_hints=relation_hints,
            constraints=self._infer_constraints(question),
        )

    def _parse_intent(self, intent: Any) -> QueryIntent:
        if isinstance(intent, QueryIntent):
            return intent
        if isinstance(intent, str):
            normalized = intent.strip().upper()
            for candidate in QueryIntent:
                if candidate.value == normalized:
                    return candidate
        return QueryIntent.EXPLAIN

    def _coerce_str_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]

    def _infer_intent_from_text(self, question: str) -> QueryIntent:
        normalized = question.strip()
        if not normalized:
            return QueryIntent.EXPLAIN
        if any(token in normalized for token in ["多少", "几种", "数量", "有几"]):
            return QueryIntent.COUNT
        if "对比" in normalized or "比较" in normalized or "区别" in normalized:
            return QueryIntent.COMPARE
        if (
            "怎么走" in normalized
            or "路径" in normalized
            or "到" in normalized
            and "怎么" in normalized
        ):
            return QueryIntent.FIND_PATH
        if "关系" in normalized or "关联" in normalized:
            return QueryIntent.FIND_RELATION
        if "哪些" in normalized or "有哪些" in normalized or "列出" in normalized:
            return QueryIntent.FIND_ENTITY
        if (
            normalized.endswith("是什么")
            or normalized.endswith("是啥")
            or "解释" in normalized
        ):
            return QueryIntent.EXPLAIN
        return QueryIntent.LIST

    def _infer_relation_hints(self, question: str) -> list[str]:
        hints = []
        for hint in ["使用", "包含", "属于", "连接", "支持", "关联"]:
            if hint in question and hint not in hints:
                hints.append(hint)
        return hints

    def _infer_constraints(self, question: str) -> dict[str, Any]:
        constraints: dict[str, Any] = {}
        match = re.search(r"(\d+)跳", question)
        if match:
            constraints["max_hops"] = int(match.group(1))
        return constraints

    def _strip_code_fences(self, response: str) -> str:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        return cleaned
