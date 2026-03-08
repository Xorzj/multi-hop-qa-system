"""Reasoning orchestrator for multi-hop question answering."""

import json
import logging
import re
from dataclasses import dataclass

from src.knowledge_graph.graph_retriever import GraphRelation, GraphRetriever
from src.llm.base_client import BaseLLMClient, GenerationParams
from src.qa_engine.question_parser import ParsedQuestion

from .evidence_chain import (
    EvidenceChain,
    EvidenceChainBuilder,
    EvidenceEdge,
    EvidenceNode,
)

logger = logging.getLogger(__name__)


@dataclass
class HopResult:
    """Result of a single hop in the reasoning process."""

    nodes: list[EvidenceNode]
    relations: list[GraphRelation]


@dataclass
class ReasoningDecision:
    """LLM decision about next hop."""

    should_continue: bool
    next_entities: list[str]
    relation_filter: str | None
    reasoning: str


@dataclass
class ReasoningConfig:
    """Configuration for reasoning orchestrator."""

    max_hops: int = 3
    enable_cot: bool = True
    min_confidence: float = 0.5
    entity_resolve_max_retries: int = 3

class ReasoningOrchestrator:
    """Orchestrates multi-hop reasoning over knowledge graph."""

    def __init__(
        self,
        graph_retriever: GraphRetriever,
        llm_client: BaseLLMClient,
        config: ReasoningConfig | None = None,
    ):
        self._retriever = graph_retriever
        self._llm = llm_client
        self._config = config or ReasoningConfig()
        self._logger = logging.getLogger(self.__class__.__name__)

    async def reason(self, parsed_question: ParsedQuestion) -> EvidenceChain:
        if not parsed_question.entities:
            self._logger.info("No entities provided for reasoning")
            return EvidenceChainBuilder(start_entity="").finalize()

        # ── Agentic entity resolution ──────────────────────────
        entity_map = await self._resolve_entities(parsed_question.entities)
        resolved = [
            entity_map[e] for e in parsed_question.entities if e in entity_map
        ]
        if not resolved:
            self._logger.warning(
                "No entities could be resolved to graph nodes: %s",
                parsed_question.entities,
            )
            return EvidenceChainBuilder(start_entity="").finalize()

        current_entities = self._unique_entities(resolved)
        relation_filter = (
            parsed_question.relation_hints[0]
            if parsed_question.relation_hints
            else None
        )
        builder = EvidenceChainBuilder(start_entity=current_entities[0])
        # 分离起始实体和目标实体
        # 第一个实体作为起点，其余实体作为目标
        start_entities = current_entities[:1]
        goal_entities = (
            set(current_entities[1:])
            if len(current_entities) > 1
            else set()
        )
        current_entities = start_entities  # 从第一个实体开始查询
        seen_entities = set(start_entities)  # 标记起始实体为已见

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

            # 过滤 next_entities：只保留实际存在于 hop_result.nodes 的实体
            valid_neighbors = {node.name for node in hop_result.nodes if node.name}
            next_entities = [
                entity
                for entity in decision.next_entities
                if entity and entity not in seen_entities and entity in valid_neighbors
            ]
            # 记录被过滤掉的幻想实体
            filtered_out = [
                e for e in decision.next_entities if e and e not in valid_neighbors
            ]
            if filtered_out:
                self._logger.warning(
                    "LLM suggested entities not in graph neighbors, filtered out: %s",
                    filtered_out,
                )

            # 只添加连接到 next_entities 的边到证据链（保持路径连通性）
            nodes, edges = self._convert_hop_result(hop_result, hop_number)
            if next_entities:
                # 过滤边：只保留 target 在 next_entities 中的边
                path_edges = [e for e in edges if e.target in next_entities]
                # 过滤节点：只保留 next_entities 中的节点
                path_nodes = [n for n in nodes if n.name in next_entities]
            else:
                # 最后一跳：过滤到目标实体
                if goal_entities:
                    # 只保留连接到目标实体的边（检查双向）
                    path_edges = [
                        e for e in edges
                        if e.target in goal_entities or e.source in goal_entities
                    ]
                    path_nodes = [
                        n for n in nodes if n.name in goal_entities
                    ]

                    # 跳级重试：如果关系过滤器导致返回了不相关的边，
                    # 重新执行该跳（不使用关系过滤器）
                    if not path_edges and relation_filter:
                        self._logger.info(
                            "Goal filter yielded no relevant edges with"
                            " relation_filter='%s', retrying hop without"
                            " filter",
                            relation_filter,
                        )
                        hop_result = await self._execute_hop(
                            current_entities, hop_number, ""
                        )
                        nodes, edges = self._convert_hop_result(
                            hop_result, hop_number
                        )
                        path_edges = [
                        e for e in edges
                        if e.target in goal_entities or e.source in goal_entities
                        ]
                        path_nodes = [
                            n
                            for n in nodes
                            if n.name in goal_entities
                        ]
                else:
                    # 如果没有目标实体，保留所有边
                    path_edges = edges
                    path_nodes = nodes
            builder.add_hop(
                nodes=path_nodes, edges=path_edges, reasoning=decision.reasoning
            )
            evidence_chain = builder.finalize()

            if self._should_stop(decision, hop_number, evidence_chain):
                break

            for entity in next_entities:
                seen_entities.add(entity)

            # 更新当前实体为下一跳的实体
            if next_entities:
                current_entities = next_entities
                relation_filter = decision.relation_filter
            else:
                # 没有有效的下一跳实体，停止推理
                break
        return builder.finalize()

    async def _execute_hop(
        self, current_entities: list[str], hop_number: int, context: str
    ) -> HopResult:
        relation_filter = context.strip() if context.strip() else None
        hop_nodes: dict[str, EvidenceNode] = {}
        hop_relations: list[GraphRelation] = []

        for entity in current_entities:
            if not entity:
                continue
            result = await self._retriever.get_neighbors(
                node_name=entity,
                relation_type=relation_filter,
            )

            # Soft-hint fallback: if relation filter yields no results,
            # retry without filter (relation_hints are hints, not hard constraints)
            if not result.nodes and relation_filter:
                self._logger.info(
                    "No neighbors for '%s' with relation_type='%s', "
                    "retrying without filter",
                    entity,
                    relation_filter,
                )
                result = await self._retriever.get_neighbors(
                    node_name=entity,
                )

            for node in result.nodes:
                if node.name and node.name not in hop_nodes:
                    hop_nodes[node.name] = EvidenceNode(
                        name=node.name,
                        label=node.label,
                        properties=node.properties,
                        hop=hop_number,
                    )

            hop_relations.extend(result.relations)

        return HopResult(nodes=list(hop_nodes.values()), relations=hop_relations)

    async def _decide_next_hop(
        self,
        question: str,
        current_evidence: EvidenceChain,
        hop_result: HopResult,
    ) -> ReasoningDecision:
        if not hop_result.nodes:
            return ReasoningDecision(
                should_continue=False,
                next_entities=[],
                relation_filter=None,
                reasoning="No nodes found in this hop",
            )

        neighbors_text = self._format_neighbors(hop_result)
        prompt = self._build_decision_prompt(question, current_evidence, neighbors_text)
        response = await self._llm.generate(prompt)
        return self._parse_decision(response)

    def _format_neighbors(self, hop_result: HopResult) -> str:
        lines: list[str] = []
        for node in hop_result.nodes:
            lines.append(f"- {node.name} ({node.label})")
        for rel in hop_result.relations:
            lines.append(f"  关系: {rel.source} --{rel.relation_type}--> {rel.target}")
        return "\n".join(lines)

    def _build_decision_prompt(
        self,
        question: str,
        current_evidence: EvidenceChain,
        neighbors_text: str,
    ) -> str:
        path_desc = current_evidence.get_path_description()
        return f"""问题: {question}

当前推理路径: {path_desc if path_desc else "尚未开始"}

发现的相邻节点和关系:
{neighbors_text}

请分析这些信息，决定下一步:
1. 是否需要继续探索？(是/否)
2. 如果是，列出要探索的实体名称（用逗号分隔）
3. 有没有特定的关系类型需要关注？

请用JSON格式回答:
{{“continue”: true/false,
 “next_entities”: [“实体1”, “实体2”],
 "relation_filter": "关系类型或null",
 "reasoning": "推理说明"}}"""

    def _parse_decision(self, response: str) -> ReasoningDecision:

        # Try to extract JSON from response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return ReasoningDecision(
                    should_continue=data.get("continue", False),
                    next_entities=data.get("next_entities", []),
                    relation_filter=data.get("relation_filter"),
                    reasoning=data.get("reasoning", ""),
                )
            except json.JSONDecodeError:
                pass

        # Fallback: extract entities using regex
        entities: list[str] = []
        entity_match = re.search(r"next_entities[\"']?\s*:\s*\[(.*?)\]", response)
        if entity_match:
            entities = [
                e.strip().strip("\"'")
                for e in entity_match.group(1).split(",")
                if e.strip()
            ]

        return ReasoningDecision(
            should_continue=bool(entities),
            next_entities=entities,
            relation_filter=None,
            reasoning=response[:200],
        )

    def _should_stop(
        self,
        decision: ReasoningDecision,
        hop_number: int,
        evidence: EvidenceChain,
    ) -> bool:
        if not decision.should_continue:
            return True
        if hop_number >= self._config.max_hops:
            return True
        if evidence.total_confidence < self._config.min_confidence:
            return True
        return False

    def _convert_hop_result(
        self, hop_result: HopResult, hop_number: int
    ) -> tuple[list[EvidenceNode], list[EvidenceEdge]]:
        nodes = hop_result.nodes
        edges = [
            EvidenceEdge(
                source=rel.source,
                target=rel.target,
                relation_type=rel.relation_type,
                confidence=rel.properties.get("confidence", 1.0),
            )
            for rel in hop_result.relations
        ]
        return nodes, edges

    def _unique_entities(self, entities: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for entity in entities:
            if entity and entity not in seen:
                seen.add(entity)
                result.append(entity)
        return result

    # ── Agentic Entity Resolution ─────────────────────────────

    @dataclass
    class _EntityResolution:
        """Result of resolving a single entity."""

        original: str
        resolved: str | None
        method: str  # exact, substring, llm_iter_N, unresolved

    async def _resolve_entities(
        self, entities: list[str]
    ) -> dict[str, str]:
        """Resolve question entities to graph node names.

        Uses an iterative agentic loop:
        1. Exact match against graph nodes
        2. Bidirectional substring match
        3. LLM-assisted resolution (with retry / reformulation)
        """
        # Fetch all graph node names once
        all_nodes = await self._retriever.search_nodes("", limit=500)
        all_node_names = sorted({n.name for n in all_nodes if n.name})

        if not all_node_names:
            self._logger.warning("Graph is empty, skipping entity resolution")
            return {}

        self._logger.info(
            "Resolving %d entities against %d graph nodes",
            len(entities),
            len(all_node_names),
        )

        resolved: dict[str, str] = {}
        for entity in entities:
            if not entity:
                continue
            resolution = await self._resolve_single_entity(
                entity, all_node_names
            )
            if resolution.resolved:
                resolved[entity] = resolution.resolved
                if resolution.original != resolution.resolved:
                    self._logger.info(
                        "Entity resolved: '%s' → '%s' (method: %s)",
                        entity,
                        resolution.resolved,
                        resolution.method,
                    )
                else:
                    self._logger.info(
                        "Entity matched: '%s' (method: %s)",
                        entity,
                        resolution.method,
                    )
            else:
                self._logger.warning(
                    "Entity unresolved: '%s' — not found in graph", entity
                )

        return resolved

    async def _resolve_single_entity(
        self,
        entity: str,
        all_node_names: list[str],
    ) -> _EntityResolution:
        """Resolve one entity via iterative search + LLM."""
        max_retries = self._config.entity_resolve_max_retries

        # ── Phase 1: Exact match ──
        if entity in all_node_names:
            return self._EntityResolution(entity, entity, "exact")

        # ── Phase 2: Substring match ──
        candidates = self._substring_match(entity, all_node_names)
        if len(candidates) == 1:
            return self._EntityResolution(
                entity, candidates[0], "substring"
            )

        # ── Phase 3: Agentic LLM loop ──
        search_history: list[str] = [entity]
        for attempt in range(max_retries):
            llm_result = await self._llm_resolve_entity(
                original_entity=entity,
                all_node_names=all_node_names,
                search_history=search_history,
            )

            if not llm_result:
                break

            # LLM returned a valid graph node
            if llm_result in all_node_names:
                return self._EntityResolution(
                    entity, llm_result, f"llm_iter_{attempt + 1}"
                )

            # LLM suggested a name not in graph — use as next search term
            if llm_result not in search_history:
                search_history.append(llm_result)
                self._logger.debug(
                    "LLM reformulated '%s' → '%s', retrying",
                    entity,
                    llm_result,
                )

                # Try substring match with reformulated term
                candidates = self._substring_match(
                    llm_result, all_node_names
                )
                if len(candidates) == 1:
                    return self._EntityResolution(
                        entity,
                        candidates[0],
                        f"llm_reformulate_{attempt + 1}",
                    )
                # Multiple candidates or no match — loop again
                continue

            # LLM returned same thing it already tried — give up
            break

        return self._EntityResolution(entity, None, "unresolved")

    @staticmethod
    def _substring_match(
        query: str, node_names: list[str]
    ) -> list[str]:
        """Bidirectional substring matching.

        Returns nodes where query ⊂ node_name or node_name ⊂ query.
        """
        return [
            name
            for name in node_names
            if query in name or name in query
        ]

    async def _llm_resolve_entity(
        self,
        original_entity: str,
        all_node_names: list[str],
        search_history: list[str],
    ) -> str | None:
        """Ask LLM to map a question entity to a graph node."""
        history_text = (
            "\n搜索历史（已尝试但未匹配）: "
            + " → ".join(search_history)
            if len(search_history) > 1
            else ""
        )
        nodes_text = ", ".join(all_node_names)

        prompt = (
            "你是知识图谱实体解析助手。将问题中的实体匹配到图谱节点。\n\n"
            f"问题实体: \"{original_entity}\"\n"
            f"{history_text}\n"
            f"图谱可用节点: [{nodes_text}]\n\n"
            "匹配规则:\n"
            "- 缩写 ↔ 全称（如 WDM ↔ 波分复用）\n"
            "- 带后缀 ↔ 不带后缀（如 SDH网络 ↔ SDH）\n"
            "- 同义词 / 近义词\n"
            "- 如果没有匹配的节点，返回 NONE\n\n"
            "仅返回一个节点名称（从可用节点列表中选择，不要解释）:"
        )

        response = await self._llm.generate(
            prompt=prompt,
            params=GenerationParams(temperature=0.1, max_new_tokens=64),
        )

        result = response.strip().strip('"').strip("'")
        if result == "NONE" or not result:
            return None
        return result
