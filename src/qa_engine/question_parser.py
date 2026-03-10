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
    def __init__(
        self,
        llm_client: BaseLLMClient,
        known_entities: list[str] | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._logger = get_logger(__name__)
        self._generation_params = GenerationParams(temperature=0.2, max_new_tokens=256)
        self._known_entities: list[str] = known_entities or []

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
            "你是知识图谱问题解析助手。"
            "从问题中抽取意图、实体、关系提示和约束，并仅返回JSON对象。\n"
            "意图枚举：FIND_ENTITY、FIND_RELATION、FIND_PATH、COMPARE、EXPLAIN、COUNT、LIST。\n"
            "\n"
            "重要规则：\n"
            "1. entities 必须包含问题中提到的所有"
            "专业术语、疾病名、器官名、技术名等实体\n"
            "2. 即使实体是中文词汇（如'高血压'、'冠心病'、'血管内皮'），也必须抽取\n"
            "3. 不要遗漏任何实体，宁可多抽取也不要漏掉\n"
            "\n"
            "示例：\n"
            "问题：A和B什么关系\n"
            '返回：{"intent":"FIND_RELATION","entities":["A","B"],'
            '"relation_hints":["关联","包含"],"constraints":{"max_hops":2}}\n'
            "问题：X是什么\n"
            '{"intent":"EXPLAIN","entities":["X"],'
            '"relation_hints":[],"constraints":{}}\n'
            "问题：高血压如何通过血管内皮损伤发展为冠心病？\n"
            '{"intent":"FIND_PATH","entities":["高血压","血管内皮","冠心病"],'
            '"relation_hints":["损伤","发展为"],"constraints":{"max_hops":3}}\n'
            "问题：动脉粥样硬化的发病机制是什么？\n"
            '{"intent":"EXPLAIN","entities":["动脉粥样硬化"],'
            '"relation_hints":["发病机制"],"constraints":{}}\n'
            "问题：DWDM和SDH是什么关系\n"
            '{"intent":"FIND_RELATION","entities":["DWDM","SDH"],'
            '"relation_hints":["关系"],"constraints":{}}\n'
            "问题：从A到C怎么走\n"
            '{"intent":"FIND_PATH","entities":["A","C"],'
            '"relation_hints":["路径"],"constraints":{"max_hops":3}}\n'
            "问题：有多少种类型\n"
            '{"intent":"COUNT","entities":["类型"],'
            '"relation_hints":[],"constraints":{}}\n'
            "\n"
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
        entities: list[str] = []

        # 1. Quoted strings
        quoted = re.findall(
            r"[\"'\u201c\u201d\u2018\u2019]"
            r"([^\"'\u201c\u201d\u2018\u2019]{1,30})"
            r"[\"'\u201c\u201d\u2018\u2019]",
            question,
        )
        for item in quoted:
            cleaned = item.strip()
            if cleaned and cleaned not in entities:
                entities.append(cleaned)

        # 2. Match against known graph entities (longest match first)
        if self._known_entities:
            sorted_known = sorted(self._known_entities, key=len, reverse=True)
            for entity_name in sorted_known:
                if entity_name in question and entity_name not in entities:
                    entities.append(entity_name)

        # 3. Domain-agnostic ASCII patterns
        _B = r"(?<![A-Za-z0-9])"
        _E = r"(?![A-Za-z0-9])"
        ascii_patterns = [
            _B + r"[A-Z]{2,}(?:-[A-Z0-9]+)*" + _E,
            _B + r"[A-Z][a-zA-Z]*[A-Z][a-zA-Z]*" + _E,
            _B + r"[A-Z]{1,}[a-z]*-\d+" + _E,
            _B + r"[A-Z]\.\d{2,}" + _E,
            _B + r"\d+(?:\.\d+)?\s*(?:Gb/s|Mb/s|GHz|MHz|kHz|TB|GB|MB|KB)" + _E,
        ]
        for pattern in ascii_patterns:
            for match in re.findall(pattern, question):
                normalized = match.strip()
                if normalized and normalized not in entities:
                    entities.append(normalized)

        # 4. Chinese noun phrase extraction (stop-word removal approach)
        #    Split on function words / punctuation, keep substantive fragments
        if not entities:
            stop_words = {
                "如何",
                "怎么",
                "怎样",
                "什么",
                "为什么",
                "是否",
                "是不是",
                "能否",
                "能不能",
                "可以",
                "通过",
                "之间",
                "有没有",
                "哪些",
                "多少",
                "有几",
                "有多少",
                "的",
                "了",
                "吗",
                "呢",
                "啊",
                "吧",
                "呀",
                "和",
                "与",
                "或",
                "及",
                "到",
                "在",
                "是",
                "将",
                "把",
                "被",
                "从",
                "对",
                "向",
                "为",
                "已",
                "也",
                "都",
                "就",
                "才",
                "又",
                "还",
            }
            # Split on punctuation and common function patterns
            fragments = re.split(
                r"[？?！!。，,、；;：:\s]|如何|怎么|怎样|什么|为什么|通过|之间|发展为|导致|引起",
                question,
            )
            for frag in fragments:
                frag = frag.strip()
                # Keep fragments that are >= 2 chars and not pure stop words
                if len(frag) >= 2 and frag not in stop_words:
                    # Strip leading/trailing single-char particles
                    _lead = r"^[的了吗呢啊吧呀和与或及到在是将把被从对向为]"
                    frag = re.sub(_lead, "", frag)
                    frag = re.sub(r"[的了吗呢啊吧呀]$", "", frag)
                    frag = frag.strip()
                    if len(frag) >= 2 and frag not in entities:
                        entities.append(frag)

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

        # COUNT
        if any(token in normalized for token in ["多少", "几种", "数量", "有几"]):
            return QueryIntent.COUNT

        # COMPARE
        if any(
            token in normalized
            for token in [
                "对比",
                "比较",
                "区别",
                "不同",
                "差异",
            ]
        ):
            return QueryIntent.COMPARE

        # FIND_PATH — fixed operator precedence
        if (
            "怎么走" in normalized
            or "路径" in normalized
            or ("到" in normalized and "怎么" in normalized)
            or "如何通过" in normalized
            or "发展为" in normalized
            or "演变" in normalized
            or "转变" in normalized
            or (
                "如何" in normalized
                and any(
                    token in normalized
                    for token in ["导致", "引起", "发展", "形成", "产生"]
                )
            )
        ):
            return QueryIntent.FIND_PATH

        # FIND_RELATION
        if any(token in normalized for token in ["关系", "关联", "有什么联系"]):
            return QueryIntent.FIND_RELATION

        # FIND_ENTITY
        if any(token in normalized for token in ["哪些", "有哪些", "列出", "列举"]):
            return QueryIntent.FIND_ENTITY

        # EXPLAIN
        if (
            normalized.endswith("是什么")
            or normalized.endswith("是啥")
            or "解释" in normalized
            or "机制" in normalized
            or "原理" in normalized
            or "原因" in normalized
        ):
            return QueryIntent.EXPLAIN

        # Default: EXPLAIN is more useful than LIST for domain QA
        return QueryIntent.EXPLAIN

    def _infer_relation_hints(self, question: str) -> list[str]:
        hints: list[str] = []
        relation_keywords = [
            "使用",
            "包含",
            "属于",
            "连接",
            "支持",
            "关联",
            "导致",
            "引起",
            "促进",
            "抑制",
            "损伤",
            "发展为",
            "诱发",
            "加剧",
            "缓解",
            "依赖",
            "影响",
        ]
        for hint in relation_keywords:
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
