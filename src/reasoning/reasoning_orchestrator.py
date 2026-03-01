from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.common.logger import get_logger
from src.knowledge_graph.graph_retriever import (
    GraphNode,
    GraphRelation,
    GraphRetriever,
    HopResult,
)
from src.llm.base_client import BaseLLMClient, GenerationParams
from src.qa_engine.question_parser import ParsedQuestion
from src.reasoning.evidence_chain import (
    EvidenceChain,
    EvidenceChainBuilder,
    EvidenceEdge,
    EvidenceNode,
)

logger = get_logger(__name__)


@dataclass
class ReasoningConfig:
    max_hops: int = 3
    max_neighbors_per_hop: int = 10
    confidence_threshold: float = 0.3
    enable_cot: bool = True


@dataclass
class HopDecision:
    should_continue: bool
    next_entities: list[str]
    relation_filter: str | None
    reasoning: str


class ReasoningOrchestrator:
    def __init__(
        self,
        graph_retriever: GraphRetriever,
        llm_client: BaseLLMClient,
        config: ReasoningConfig | None = None,
    ) -> None:
        self._graph_retriever = graph_retriever
        self._llm_client = llm_client
        self._config = config or ReasoningConfig()
        self._generation_params = GenerationParams(temperature=0.2, max_new_tokens=256)
        self._logger = get_logger(__name__)

    async def reason(self, parsed_question: ParsedQuestion) -> EvidenceChain:
        if not parsed_question.entities:
            self._logger.info("No entities provided for reasoning")
            return EvidenceChainBuilder(start_entity="").finalize()

        current_entities = self._unique_entities(parsed_question.entities)
        relation_filter = (
            parsed_question.relation_hints[0]
            if parsed_question.relation_hints
            else None
        )
        builder = EvidenceChainBuilder(start_entity=current_entities[0])
        seen_entities = set(current_entities)

        for hop in range(self._config.max_hops):
            hop_number = hop + 1
            hop_result = await self._execute_hop(
                current_entities=current_entities,
                hop_number=hop_number,
                context=relation_filter or "",
            )
            evidence_chain = builder.finalize()
            decision = await self._decide_next_hop(
                question=parsed_question.original,
                current_evidence=evidence_chain,
                hop_result=hop_result,
            )
            nodes, edges = self._convert_hop_result(hop_result, hop_number)
            builder.add_hop(nodes=nodes, edges=edges, reasoning=decision.reasoning)
            evidence_chain = builder.finalize()

            if self._should_stop(decision, hop_number, evidence_chain):
                break

            next_entities = [
                entity
                for entity in decision.next_entities
                if entity and entity not in seen_entities
            ]
            for entity in next_entities:
                seen_entities.add(entity)

            if not next_entities:
                break

            current_entities = next_entities
            relation_filter = decision.relation_filter

        return builder.finalize()

    async def _execute_hop(
        self, current_entities: list[str], hop_number: int, context: str
    ) -> HopResult:
        relation_filter = context.strip() if context.strip() else None
        hop_nodes: dict[str, GraphNode] = {}
        hop_relations: list[GraphRelation] = []

        for entity in current_entities:
            if not entity:
                continue
            hop_result = await self._graph_retriever.get_neighbors(
                node_name=entity,
                relation_type=relation_filter,
                limit=self._config.max_neighbors_per_hop,
            )
            # 如果带关系过滤无结果，回退到无过滤查询
            if (
                hop_result is not None
                and not hop_result.nodes
                and relation_filter is not None
            ):
                self._logger.info(
                    "Relation filter '%s' empty for '%s', retrying without filter",
                    relation_filter,
                    entity,
                )
                hop_result = await self._graph_retriever.get_neighbors(
                    node_name=entity,
                    relation_type=None,
                    limit=self._config.max_neighbors_per_hop,
                )
            if hop_result is None:
                continue
            for node in hop_result.nodes:
                if node.name and node.name not in hop_nodes:
                    hop_nodes[node.name] = node
            hop_relations.extend(hop_result.relations)

        return HopResult(
            nodes=list(hop_nodes.values()),
            relations=hop_relations,
            hop_number=hop_number,
        )

    async def _decide_next_hop(
        self, question: str, current_evidence: EvidenceChain, hop_result: HopResult
    ) -> HopDecision:
        prompt = self._build_decision_prompt(question, current_evidence, hop_result)
        try:
            response = await self._llm_client.generate(
                prompt=prompt, params=self._generation_params
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "LLM decision failed, stopping",
                extra={"error": str(exc)},
            )
            return HopDecision(
                should_continue=False,
                next_entities=[],
                relation_filter=None,
                reasoning="",
            )

        return self._parse_decision(response)

    def _build_decision_prompt(
        self, question: str, evidence: EvidenceChain, hop_result: HopResult
    ) -> str:
        evidence_path = evidence.get_path_description() or "暂无"
        hop_entities = [node.name for node in hop_result.nodes if node.name]
        relation_types = [relation.relation_type for relation in hop_result.relations]
        relation_summary = ", ".join({rel for rel in relation_types if rel}) or "未知"
        reasoning_instruction = (
            "请给出简短的推理理由。"
            if self._config.enable_cot
            else "reasoning字段可为空字符串。"
        )

        return (
            "你是多跳知识图谱推理控制器。根据问题与当前证据，判断是否继续探索下一跳。"
            "请仅返回JSON对象，不要添加解释文字。"
            f"\n问题：{question}"
            f"\n已有路径：{evidence_path}"
            f"\n当前证据置信度：{evidence.total_confidence:.4f}"
            f"\n本跳候选实体：{hop_entities}"
            f"\n本跳关系类型：{relation_summary}"
            "\n输出字段：should_continue(boolean), next_entities(list[string]), "
            "relation_filter(string or null), reasoning(string)。"
            f"\n{reasoning_instruction}"
            '\n示例：{"should_continue": true, '
            '"next_entities": ["STM-1"], '
            '"relation_filter": "包含", '
            '"reasoning": "需要找到STM-1包含的具体结构"}'
        )

    def _parse_decision(self, response: str) -> HopDecision:
        try:
            payload = json.loads(self._strip_code_fences(response))
        except json.JSONDecodeError:
            self._logger.warning("Failed to parse hop decision")
            return HopDecision(
                should_continue=False,
                next_entities=[],
                relation_filter=None,
                reasoning="",
            )

        if not isinstance(payload, dict):
            self._logger.warning("Hop decision response is not an object")
            return HopDecision(
                should_continue=False,
                next_entities=[],
                relation_filter=None,
                reasoning="",
            )

        should_continue = bool(payload.get("should_continue"))
        next_entities = self._coerce_str_list(payload.get("next_entities"))
        relation_filter = payload.get("relation_filter")
        relation_value = (
            relation_filter.strip()
            if isinstance(relation_filter, str) and relation_filter.strip()
            else None
        )
        reasoning_value = payload.get("reasoning")
        reasoning = reasoning_value if isinstance(reasoning_value, str) else ""

        return HopDecision(
            should_continue=should_continue,
            next_entities=next_entities,
            relation_filter=relation_value,
            reasoning=reasoning,
        )

    def _should_stop(
        self, decision: HopDecision, hop_number: int, evidence: EvidenceChain
    ) -> bool:
        if hop_number >= self._config.max_hops:
            return True
        if not decision.should_continue:
            return True
        if not decision.next_entities:
            return True
        if evidence.calculate_confidence() < self._config.confidence_threshold:
            return True
        return False

    def _convert_hop_result(
        self, hop_result: HopResult, hop_number: int
    ) -> tuple[list[EvidenceNode], list[EvidenceEdge]]:
        nodes = [
            EvidenceNode(
                name=node.name,
                label=node.label,
                properties=dict(node.properties),
                hop=hop_number,
            )
            for node in hop_result.nodes
        ]
        edges = [
            EvidenceEdge(
                source=relation.source,
                target=relation.target,
                relation_type=relation.relation_type,
                confidence=self._extract_confidence(relation),
            )
            for relation in hop_result.relations
        ]
        return nodes, edges

    def _extract_confidence(self, relation: GraphRelation) -> float:
        value = relation.properties.get("confidence") if relation.properties else None
        if isinstance(value, (int, float)):
            return float(value)
        return 1.0

    def _strip_code_fences(self, response: str) -> str:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```")
            if cleaned.lstrip().startswith("json"):
                cleaned = cleaned.lstrip()[4:]
            cleaned = cleaned.strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        return cleaned

    def _coerce_str_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]

    def _unique_entities(self, entities: list[str]) -> list[str]:
        unique: list[str] = []
        for entity in entities:
            cleaned = entity.strip()
            if cleaned and cleaned not in unique:
                unique.append(cleaned)
        return unique
