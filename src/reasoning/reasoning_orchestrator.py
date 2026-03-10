"""Reasoning orchestrator for multi-hop question answering."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.knowledge_graph.graph_retriever import GraphNode, GraphRelation, GraphRetriever
from src.knowledge_graph.graph_retriever import HopResult as GraphHopResult
from src.llm.base_client import BaseLLMClient, GenerationParams
from src.qa_engine.question_parser import ParsedQuestion

if TYPE_CHECKING:
    from src.qa_engine.query_rewriter import QueryPlan, QueryStep

from .evidence_chain import (
    EvidenceChain,
    EvidenceChainBuilder,
    EvidenceEdge,
    EvidenceNode,
)

logger = logging.getLogger(__name__)

_ABSTRACT_LABELS = frozenset(
    {"抽象概念", "治疗概念", "方法", "概念", "疾病类型", "疾病类别", "治疗链条"}
)


def _make_default_step() -> QueryStep:
    """Create a default find_neighbors step (no type/relation filter)."""
    from src.qa_engine.query_rewriter import QueryStep

    return QueryStep(
        action="find_neighbors", direction="both", description="默认查找邻居"
    )


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
class ReflectionResult:
    """Result of self-reflection after a hop."""

    action: str  # "continue", "backtrack", "switch"
    reasoning: str
    confidence: float
    suggested_entities: list[str]


@dataclass
class ReasoningConfig:
    """Configuration for reasoning orchestrator."""

    max_hops: int = 3
    enable_cot: bool = True
    min_confidence: float = 0.5
    entity_resolve_max_retries: int = 3
    enable_reflection: bool = True


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

    async def reason(
        self,
        parsed_question: ParsedQuestion,
        query_plan: QueryPlan | None = None,
    ) -> EvidenceChain:
        # When a QueryPlan is provided, use plan-based reasoning
        if query_plan is not None:
            return await self._reason_with_plan(query_plan, parsed_question)

        return await self._reason_original(parsed_question)

    # ── Plan-based reasoning ─────────────────────────────────

    async def _reason_with_plan(
        self, plan: QueryPlan, parsed_question: ParsedQuestion
    ) -> EvidenceChain:
        """Execute reasoning guided by a structured QueryPlan.

        Only resolves plan.start_entities (not type query words).
        Follows plan steps with Cypher label filtering.
        Falls back to original reasoning if plan yields no results.
        """
        if not plan.start_entities:
            self._logger.info("QueryPlan has no start_entities, falling back")
            return await self._reason_legacy(parsed_question)

        # Resolve only start_entities against graph
        entity_map = await self._resolve_entities(plan.start_entities)
        resolved = [entity_map[e] for e in plan.start_entities if e in entity_map]
        if not resolved:
            self._logger.warning(
                "No start_entities resolved from plan: %s", plan.start_entities
            )
            return await self._reason_legacy(parsed_question)

        current_entities = self._unique_entities(resolved)
        start_entities_original = list(current_entities)
        builder = EvidenceChainBuilder(start_entity=current_entities[0])

        if not plan.steps:
            # No steps defined — fall back to a single generic hop
            plan_steps = [_make_default_step()]
        else:
            plan_steps = plan.steps

        for hop_number, step in enumerate(plan_steps, start=1):
            if step.action == "find_by_path" and len(current_entities) >= 2:
                # Chain consecutive entity pairs so intermediate waypoints
                # are not skipped by shortestPath.
                all_path_hops: list[
                    tuple[list[EvidenceNode], list[EvidenceEdge]]
                ] = []
                for seg_idx in range(len(current_entities) - 1):
                    path_hops = await self._retriever.get_path(
                        current_entities[seg_idx],
                        current_entities[seg_idx + 1],
                        directed=False,
                    )
                    if path_hops:
                        for ph in path_hops:
                            seg_nodes = [
                                EvidenceNode(
                                    name=n.name,
                                    label=n.label,
                                    properties=n.properties,
                                    hop=hop_number,
                                )
                                for n in ph.nodes
                            ]
                            seg_edges = [
                                EvidenceEdge(
                                    source=r.source,
                                    target=r.target,
                                    relation_type=r.relation_type,
                                    confidence=r.properties.get(
                                        "confidence", 1.0
                                    ),
                                    source_chunk_id=str(
                                        r.properties.get("source_chunk_id", "")
                                    ),
                                    source_text=str(
                                        r.properties.get("source_text", "")
                                    ),
                                )
                                for r in ph.relations
                            ]
                            all_path_hops.append((seg_nodes, seg_edges))
                if all_path_hops:
                    for seg_nodes, seg_edges in all_path_hops:
                        builder.add_hop(
                            nodes=seg_nodes,
                            edges=seg_edges,
                            reasoning=step.description,
                        )
                continue

            # find_neighbors with cascade fallback
            hop_result = await self._execute_plan_hop(
                current_entities, step, hop_number
            )

            nodes, edges = self._convert_hop_result(hop_result, hop_number)
            builder.add_hop(nodes=nodes, edges=edges, reasoning=step.description)

            # All nodes from this hop become the next frontier
            next_frontier = [n.name for n in hop_result.nodes if n.name]
            if next_frontier:
                current_entities = self._unique_entities(
                    start_entities_original + next_frontier
                )
            else:
                self._logger.info("Plan hop %d yielded no nodes, stopping", hop_number)
                break

        result = builder.finalize()

        # If plan produced no edges, fall back to legacy reasoning
        if not result.edges:
            self._logger.info(
                "Plan-based reasoning found no edges, falling back to legacy"
            )
            return await self._reason_legacy(parsed_question)

        return result

    async def _execute_plan_hop(
        self,
        current_entities: list[str],
        step: QueryStep,
        hop_number: int,
    ) -> HopResult:
        """Execute a single plan step with cascade fallback.

        Fallback order:
        1. label + relation filter
        2. label only
        3. no filter
        """
        hop_nodes: dict[str, EvidenceNode] = {}
        hop_relations: list[GraphRelation] = []

        for entity in current_entities:
            if not entity:
                continue

            # Level 1: label + relation
            result = await self._retriever.get_neighbors(
                node_name=entity,
                relation_type=step.relation_hint,
                direction=step.direction,
                neighbor_label=step.target_type,
            )

            # Level 2: label only (drop relation filter)
            if not result.nodes and step.relation_hint and step.target_type:
                self._logger.info(
                    "Hop %d: no results for '%s' with label='%s' + "
                    "relation='%s', retrying with label only",
                    hop_number,
                    entity,
                    step.target_type,
                    step.relation_hint,
                )
                result = await self._retriever.get_neighbors(
                    node_name=entity,
                    direction=step.direction,
                    neighbor_label=step.target_type,
                )

            # Level 3: multi-hop label search (up to 3 hops away)
            if not result.nodes and step.target_type:
                self._logger.info(
                    "Hop %d: no 1-hop results for '%s' with label='%s', "
                    "trying multi-hop label search",
                    hop_number,
                    entity,
                    step.target_type,
                )
                nearby_nodes = await self._retriever.get_labeled_nearby(
                    entity, step.target_type, max_hops=3,
                )
                if nearby_nodes:
                    # Retrieve actual paths to get edges with source_text
                    path_nodes: list[GraphNode] = []
                    path_rels: list[GraphRelation] = []
                    for found in nearby_nodes:
                        path_hops = await self._retriever.get_path(
                            entity, found.name, max_hops=3, directed=False,
                        )
                        if path_hops:
                            for ph in path_hops:
                                path_nodes.extend(ph.nodes)
                                path_rels.extend(ph.relations)
                    all_nodes = nearby_nodes + path_nodes
                    result = GraphHopResult(
                        nodes=all_nodes,
                        relations=path_rels,
                        hop_number=hop_number,
                    )

            # Level 4: no filter
            if not result.nodes and step.target_type:
                self._logger.info(
                    "Hop %d: no results for '%s' with label='%s', "
                    "retrying without filter",
                    hop_number,
                    entity,
                    step.target_type,
                )
                result = await self._retriever.get_neighbors(
                    node_name=entity,
                    direction=step.direction,
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

    async def _reason_legacy(self, parsed_question: ParsedQuestion) -> EvidenceChain:
        """Run legacy reasoning without a QueryPlan (backward compat wrapper)."""
        # Temporarily set query_plan=None and call original logic
        return await self._reason_original(parsed_question)

    async def _reason_original(self, parsed_question: ParsedQuestion) -> EvidenceChain:
        """Original multi-hop reasoning logic (without QueryPlan)."""
        if not parsed_question.entities:
            self._logger.info("No entities provided for reasoning")
            return EvidenceChainBuilder(start_entity="").finalize()

        entity_map = await self._resolve_entities(parsed_question.entities)
        resolved = [entity_map[e] for e in parsed_question.entities if e in entity_map]
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
        start_entities = current_entities[:1]
        goal_entities = (
            set(current_entities[1:]) if len(current_entities) > 1 else set()
        )
        current_entities = start_entities
        seen_entities = set(start_entities)

        best_evidence: EvidenceChain | None = None
        best_confidence: float = 0.0

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

            valid_neighbors = {node.name for node in hop_result.nodes if node.name}
            next_entities = [
                entity
                for entity in decision.next_entities
                if entity and entity not in seen_entities and entity in valid_neighbors
            ]
            filtered_out = [
                e for e in decision.next_entities if e and e not in valid_neighbors
            ]
            if filtered_out:
                self._logger.warning(
                    "LLM suggested entities not in graph neighbors, filtered out: %s",
                    filtered_out,
                )

            nodes, edges = self._convert_hop_result(hop_result, hop_number)
            if next_entities:
                path_edges = [e for e in edges if e.target in next_entities]
                path_nodes = [n for n in nodes if n.name in next_entities]
            else:
                if goal_entities:
                    path_edges = [
                        e
                        for e in edges
                        if e.target in goal_entities or e.source in goal_entities
                    ]
                    path_nodes = [n for n in nodes if n.name in goal_entities]

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
                        nodes, edges = self._convert_hop_result(hop_result, hop_number)
                        path_edges = [
                            e
                            for e in edges
                            if e.target in goal_entities or e.source in goal_entities
                        ]
                        path_nodes = [n for n in nodes if n.name in goal_entities]
                else:
                    path_edges = edges
                    path_nodes = nodes
            builder.add_hop(
                nodes=path_nodes, edges=path_edges, reasoning=decision.reasoning
            )
            evidence_chain = builder.finalize()

            if self._config.enable_reflection and hop_number < self._config.max_hops:
                reflection = await self._reflect_on_hop(
                    question=parsed_question.original,
                    evidence=evidence_chain,
                    hop_number=hop_number,
                    goal_entities=goal_entities,
                )
                self._logger.info(
                    "Reflection at hop %d: action=%s confidence=%.2f — %s",
                    hop_number,
                    reflection.action,
                    reflection.confidence,
                    reflection.reasoning[:120],
                )

                if evidence_chain.total_confidence > best_confidence:
                    best_confidence = evidence_chain.total_confidence
                    best_evidence = evidence_chain

                if reflection.action == "backtrack":
                    self._logger.info(
                        "Reflection triggered backtrack at hop %d", hop_number
                    )
                    if reflection.suggested_entities:
                        switch_candidates = [
                            e
                            for e in reflection.suggested_entities
                            if e in valid_neighbors and e not in seen_entities
                        ]
                        if switch_candidates:
                            self._logger.info(
                                "Switching to reflection-suggested entities: %s",
                                switch_candidates,
                            )
                            current_entities = switch_candidates
                            for e in switch_candidates:
                                seen_entities.add(e)
                            relation_filter = None
                            continue
                    if best_evidence is not None:
                        return best_evidence
                    break

                if reflection.action == "switch" and reflection.suggested_entities:
                    chain_node_names = {n.name for n in evidence_chain.nodes if n.name}
                    switch_pool = chain_node_names | valid_neighbors
                    switch_candidates = [
                        e
                        for e in reflection.suggested_entities
                        if e in switch_pool and e not in seen_entities
                    ]
                    if switch_candidates:
                        self._logger.info(
                            "Reflection switched direction to: %s",
                            switch_candidates,
                        )
                        next_entities = switch_candidates

            if self._should_stop(decision, hop_number, evidence_chain):
                break

            for entity in next_entities:
                seen_entities.add(entity)

            if next_entities:
                current_entities = next_entities
                relation_filter = decision.relation_filter
            else:
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

    async def _reflect_on_hop(
        self,
        question: str,
        evidence: EvidenceChain,
        hop_number: int,
        goal_entities: set[str],
    ) -> ReflectionResult:
        """Self-reflection after a hop: evaluate path quality and decide action.

        Returns a ReflectionResult with action (continue/backtrack/switch),
        reasoning, confidence estimate, and optionally suggested entities
        to explore if switching direction.
        """
        path_desc = evidence.get_path_description()
        goals_text = "、".join(goal_entities) if goal_entities else "无明确目标"
        found_goals = goal_entities & {n.name for n in evidence.nodes}
        found_text = "、".join(found_goals) if found_goals else "尚未到达"

        prompt = (
            f"你是知识图谱推理质量评估专家。请评估当前推理路径的质量。\n\n"
            f"原始问题: {question}\n"
            f"目标实体: {goals_text}\n"
            f"已到达的目标: {found_text}\n"
            f"当前路径 (第{hop_number}跳后): {path_desc or '空'}\n"
            f"路径置信度: {evidence.total_confidence:.2f}\n\n"
            "请评估：\n"
            "1. 当前路径是否在接近回答问题？\n"
            "2. 应该继续当前方向、换一个方向探索、还是回溯？\n"
            "3. 给出置信度评分 (0-1)\n\n"
            '返回JSON: {{"action": "continue|backtrack|switch", '
            '"reasoning": "评估说明", "confidence": 0.8, '
            '"suggested_entities": ["如果switch，建议探索的实体"]}}'
        )

        try:
            response = await self._llm.generate(
                prompt=prompt,
                params=GenerationParams(temperature=0.2, max_new_tokens=256),
            )
            return self._parse_reflection(response)
        except Exception as exc:
            self._logger.warning("Reflection failed: %s", exc)
            return ReflectionResult(
                action="continue",
                reasoning="Reflection failed, continuing",
                confidence=0.5,
                suggested_entities=[],
            )

    def _parse_reflection(self, response: str) -> ReflectionResult:
        """Parse reflection LLM response into ReflectionResult."""
        # Strip thinking tags
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", response).strip()

        json_match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                action = data.get("action", "continue")
                if action not in ("continue", "backtrack", "switch"):
                    action = "continue"
                confidence = data.get("confidence", 0.5)
                if not isinstance(confidence, (int, float)):
                    confidence = 0.5
                suggested = data.get("suggested_entities", [])
                if not isinstance(suggested, list):
                    suggested = []
                return ReflectionResult(
                    action=action,
                    reasoning=data.get("reasoning", ""),
                    confidence=float(confidence),
                    suggested_entities=[
                        str(e) for e in suggested if isinstance(e, str)
                    ],
                )
            except json.JSONDecodeError:
                pass

        return ReflectionResult(
            action="continue",
            reasoning=cleaned[:200],
            confidence=0.5,
            suggested_entities=[],
        )

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
                source_chunk_id=str(rel.properties.get("source_chunk_id", "")),
                source_text=str(rel.properties.get("source_text", "")),
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

    async def _resolve_entities(self, entities: list[str]) -> dict[str, str]:
        """Resolve question entities to graph node names.

        Uses an iterative agentic loop:
        1. Exact match against graph nodes
        2. Bidirectional substring match
        3. LLM-assisted resolution (with retry / reformulation)

        Nodes with abstract labels (e.g. "抽象概念") are filtered out.
        """
        # Fetch all graph node names once
        all_nodes = await self._retriever.search_nodes("", limit=500)
        all_node_names = sorted({n.name for n in all_nodes if n.name})
        node_label_map: dict[str, str] = {n.name: n.label for n in all_nodes if n.name}

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
            resolution = await self._resolve_single_entity(entity, all_node_names)
            if resolution.resolved:
                # Filter out abstract nodes
                label = node_label_map.get(resolution.resolved, "")
                if label in _ABSTRACT_LABELS:
                    self._logger.info(
                        "Entity '%s' resolved to abstract node '%s' "
                        "(label=%s), skipping",
                        entity,
                        resolution.resolved,
                        label,
                    )
                    continue

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
            return self._EntityResolution(entity, candidates[0], "substring")

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
                candidates = self._substring_match(llm_result, all_node_names)
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
    def _substring_match(query: str, node_names: list[str]) -> list[str]:
        """Bidirectional substring matching.

        Returns nodes where query ⊂ node_name or node_name ⊂ query.
        """
        return [name for name in node_names if query in name or name in query]

    async def _llm_resolve_entity(
        self,
        original_entity: str,
        all_node_names: list[str],
        search_history: list[str],
    ) -> str | None:
        """Ask LLM to map a question entity to a graph node."""
        history_text = (
            "\n搜索历史（已尝试但未匹配）: " + " → ".join(search_history)
            if len(search_history) > 1
            else ""
        )
        nodes_text = ", ".join(all_node_names)

        prompt = (
            "你是知识图谱实体解析助手。将问题中的实体匹配到图谱节点。\n\n"
            f'问题实体: "{original_entity}"\n'
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
