"""QueryRewriter: rewrites a parsed question into a structured QueryPlan.

Distinguishes concrete entity names (e.g. "冠心病") from type-filter queries
(e.g. "症状") so the reasoning engine can apply Cypher label filters instead
of matching abstract concept nodes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from src.common.logger import get_logger
from src.data_processing.schema_inducer import DomainSchema
from src.llm.base_client import BaseLLMClient, GenerationParams
from src.qa_engine.question_parser import ParsedQuestion

logger = get_logger(__name__)


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class QueryStep:
    """A single retrieval step in the query plan."""

    action: Literal["find_neighbors", "find_by_path"] = "find_neighbors"
    target_type: str | None = None  # Neo4j label filter (e.g. "症状")
    relation_hint: str | None = None
    direction: Literal["out", "in", "both"] = "both"
    description: str = ""


@dataclass
class QueryPlan:
    """Structured query plan produced by QueryRewriter."""

    start_entities: list[str] = field(default_factory=list)
    steps: list[QueryStep] = field(default_factory=list)
    original_question: str = ""
    raw_entities: list[str] = field(default_factory=list)


# ── QueryRewriter ────────────────────────────────────────────────────


class QueryRewriter:
    """Rewrites a ParsedQuestion into a structured QueryPlan via LLM."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        domain_schema: DomainSchema | None = None,
    ) -> None:
        self._llm = llm_client
        self._schema = domain_schema
        self._logger = get_logger(__name__)
        self._generation_params = GenerationParams(temperature=0.2, max_new_tokens=512)

        # Pre-compute label set from schema for fallback classification
        self._schema_labels: set[str] = set()
        if domain_schema:
            for et in domain_schema.entity_types:
                self._schema_labels.add(et.label)
                self._schema_labels.add(et.name)

    async def rewrite(self, parsed: ParsedQuestion) -> QueryPlan:
        """Rewrite a parsed question into a QueryPlan."""
        if not parsed.entities:
            return QueryPlan(
                original_question=parsed.original,
                raw_entities=[],
            )

        prompt = self._build_rewrite_prompt(parsed)
        try:
            response = await self._llm.generate(
                prompt=prompt, params=self._generation_params
            )
            plan = self._parse_rewrite_response(response, parsed)
            if plan.start_entities:
                self._logger.info(
                    "QueryRewriter: start_entities=%s, steps=%d",
                    plan.start_entities,
                    len(plan.steps),
                )
                return plan
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("QueryRewriter LLM failed: %s", exc)

        # Fallback
        self._logger.info("QueryRewriter: using fallback plan")
        return self._fallback_plan(parsed)

    # ── Prompt construction ──────────────────────────────────

    def _build_rewrite_prompt(self, parsed: ParsedQuestion) -> str:
        type_list = ""
        if self._schema_labels:
            type_list = (
                "\n已知领域实体类型标签: "
                + "、".join(sorted(self._schema_labels))
                + "\n"
            )

        return (
            "你是知识图谱查询规划助手。将自然语言问题拆解为结构化查询计划。\n\n"
            f"问题: {parsed.original}\n"
            f"已抽取的实体: {parsed.entities}\n"
            f"关系提示: {parsed.relation_hints}\n"
            f"{type_list}\n"
            "规则:\n"
            "1. start_entities: 具体实体名（如'冠心病'、'高血压'、'SDH'），"
            "这些是图谱中的节点名称\n"
            "2. steps: 查询步骤列表，每步可以是:\n"
            "   - find_neighbors: 查找邻居节点\n"
            "   - find_by_path: 查找两个实体间的路径\n"
            "3. target_type: 如果查询的是某类实体（如'症状'、'药物'、'协议'），"
            "填写中文标签用于过滤邻居节点的类型。"
            "只有当词语是实体类型标签（而非具体实体）时才填写\n"
            "4. 判断标准: 词语是已知实体类型标签 → target_type；否则 → start_entity\n"
            "5. relation_hint: 可选的关系类型提示\n"
            "6. direction: out(出边)/in(入边)/both(双向)\n\n"
            "示例1 — 类型查询:\n"
            '问题: "冠心病会引起哪些症状？"\n'
            '已抽取的实体: ["冠心病", "症状"]\n'
            "```json\n"
            '{"start_entities": ["冠心病"], "steps": ['
            '{"action": "find_neighbors", "target_type": "症状", '
            '"relation_hint": "引起", "direction": "out", '
            '"description": "查找冠心病引起的症状"}'
            "]}\n"
            "```\n\n"
            "示例2 — 多跳分解:\n"
            '问题: "冠心病的症状用什么药物治疗？"\n'
            '已抽取的实体: ["冠心病", "症状", "药物"]\n'
            "```json\n"
            '{"start_entities": ["冠心病"], "steps": ['
            '{"action": "find_neighbors", "target_type": "症状", '
            '"direction": "out", "description": "找冠心病的症状"}, '
            '{"action": "find_neighbors", "target_type": "药物", '
            '"direction": "in", "description": "找治疗这些症状的药物"}'
            "]}\n"
            "```\n\n"
            "示例3 — 纯实体查询:\n"
            '问题: "SDH和WDM是什么关系？"\n'
            '已抽取的实体: ["SDH", "WDM"]\n'
            "```json\n"
            '{"start_entities": ["SDH", "WDM"], "steps": ['
            '{"action": "find_neighbors", "direction": "both", '
            '"description": "查找SDH和WDM的关联"}'
            "]}\n"
            "```\n\n"
            "示例4 — 路径查找:\n"
            '问题: "高血压如何通过血管内皮损伤发展为冠心病？"\n'
            '已抽取的实体: ["高血压", "血管内皮", "冠心病"]\n'
            "```json\n"
            '{"start_entities": ["高血压", "血管内皮", "冠心病"], "steps": ['
            '{"action": "find_by_path", "description": "查找从高血压到冠心病的路径"}'
            "]}\n"
            "```\n\n"
            "现在请为上述问题生成查询计划，仅返回JSON对象:"
        )

    # ── Response parsing ─────────────────────────────────────

    def _parse_rewrite_response(
        self, response: str, parsed: ParsedQuestion
    ) -> QueryPlan:
        # Strip thinking tags
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", response).strip()

        # Try code fence extraction
        code_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```", cleaned)
        if code_match:
            json_str = code_match.group(1).strip()
        else:
            json_str = cleaned

        # Try direct JSON parse
        data: dict[str, Any] | None = None
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Brace search fallback
            start = json_str.find("{")
            end = json_str.rfind("}")
            if start != -1 and end > start:
                try:
                    data = json.loads(json_str[start : end + 1])
                except json.JSONDecodeError:
                    pass

        if data is None or not isinstance(data, dict):
            self._logger.warning("Failed to parse rewrite response")
            return QueryPlan(
                original_question=parsed.original, raw_entities=parsed.entities
            )

        start_entities = self._coerce_str_list(data.get("start_entities"))
        raw_steps = data.get("steps", [])
        steps: list[QueryStep] = []
        if isinstance(raw_steps, list):
            for raw_step in raw_steps:
                if not isinstance(raw_step, dict):
                    continue
                action = raw_step.get("action", "find_neighbors")
                if action not in ("find_neighbors", "find_by_path"):
                    action = "find_neighbors"
                direction = raw_step.get("direction", "both")
                if direction not in ("out", "in", "both"):
                    direction = "both"
                raw_hint = raw_step.get("relation_hint")
                if isinstance(raw_hint, list):
                    raw_hint = raw_hint[0] if raw_hint else None
                elif not isinstance(raw_hint, str):
                    raw_hint = None
                steps.append(
                    QueryStep(
                        action=action,
                        target_type=raw_step.get("target_type"),
                        relation_hint=raw_hint,
                        direction=direction,
                        description=raw_step.get("description", ""),
                    )
                )

        return QueryPlan(
            start_entities=start_entities,
            steps=steps,
            original_question=parsed.original,
            raw_entities=parsed.entities,
        )

    # ── Fallback plan ────────────────────────────────────────

    def _fallback_plan(self, parsed: ParsedQuestion) -> QueryPlan:
        """Deterministic fallback when LLM fails.

        Entities matching a schema label → target_type filter.
        Remaining entities → start_entities.
        """
        start_entities: list[str] = []
        type_filters: list[str] = []

        for entity in parsed.entities:
            if entity in self._schema_labels:
                type_filters.append(entity)
            else:
                start_entities.append(entity)

        # If all entities are types (unlikely), use first as start
        if not start_entities and type_filters:
            start_entities = [type_filters.pop(0)]

        steps: list[QueryStep] = []
        if type_filters:
            for tf in type_filters:
                relation_hint = (
                    parsed.relation_hints[0] if parsed.relation_hints else None
                )
                steps.append(
                    QueryStep(
                        action="find_neighbors",
                        target_type=tf,
                        relation_hint=relation_hint,
                        direction="both",
                        description=f"查找类型为{tf}的邻居",
                    )
                )
        else:
            # No type filters → single generic step
            steps.append(
                QueryStep(
                    action="find_neighbors",
                    direction="both",
                    description="查找邻居节点",
                )
            )

        return QueryPlan(
            start_entities=start_entities,
            steps=steps,
            original_question=parsed.original,
            raw_entities=parsed.entities,
        )

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _coerce_str_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
