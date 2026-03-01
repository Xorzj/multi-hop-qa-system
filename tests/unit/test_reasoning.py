from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from src.knowledge_graph.graph_retriever import GraphNode, GraphRelation, HopResult
from src.qa_engine.question_parser import ParsedQuestion, QueryIntent
from src.reasoning.evidence_chain import (
    EvidenceChain,
    EvidenceChainBuilder,
    EvidenceEdge,
    EvidenceNode,
    EvidenceStep,
)
from src.reasoning.reasoning_orchestrator import (
    HopDecision,
    ReasoningConfig,
    ReasoningOrchestrator,
)


@pytest.fixture
def sample_nodes() -> list[EvidenceNode]:
    return [
        EvidenceNode(name="A", label="Entity"),
        EvidenceNode(name="B", label="Entity"),
    ]


@pytest.fixture
def sample_edges() -> list[EvidenceEdge]:
    return [
        EvidenceEdge(source="A", target="B", relation_type="包含", confidence=0.5),
        EvidenceEdge(source="B", target="C", relation_type="关联", confidence=0.8),
    ]


@pytest.fixture
def sample_steps() -> list[EvidenceStep]:
    return [
        EvidenceStep(
            hop_number=1, action="expand_hop_1", nodes_explored=["A"], reasoning="ok"
        )
    ]


@pytest.fixture
def parsed_question() -> ParsedQuestion:
    return ParsedQuestion(
        original="A 与 B 有什么关系",
        intent=QueryIntent.FIND_RELATION,
        entities=["A"],
        relation_hints=["包含"],
        constraints={},
    )


def test_evidence_chain_add_and_path(sample_nodes: list[EvidenceNode]) -> None:
    chain = EvidenceChain()
    chain.add_node(sample_nodes[0])
    chain.add_node(sample_nodes[1])

    assert chain.get_path() == ["A", "B"]
    assert chain.get_path_description() == "A -- B"


def test_evidence_chain_edges_path(sample_edges: list[EvidenceEdge]) -> None:
    chain = EvidenceChain(edges=sample_edges)

    assert chain.get_path() == ["A", "B", "C"]
    assert chain.get_path_description() == "A --包含--> B --关联--> C"


def test_evidence_chain_confidence_and_serialization(
    sample_nodes: list[EvidenceNode],
    sample_edges: list[EvidenceEdge],
    sample_steps: list[EvidenceStep],
) -> None:
    chain = EvidenceChain(nodes=sample_nodes, edges=sample_edges, steps=sample_steps)

    confidence = chain.calculate_confidence()
    payload = chain.to_dict()

    assert confidence == pytest.approx(0.4)
    assert payload["total_confidence"] == pytest.approx(0.4)
    assert payload["path"] == ["A", "B", "C"]
    assert payload["path_description"] == "A --包含--> B --关联--> C"
    assert payload["nodes"][0]["name"] == "A"
    assert payload["edges"][0]["relation_type"] == "包含"


def test_evidence_chain_builder_add_hop_and_frontier() -> None:
    builder = EvidenceChainBuilder(start_entity="Start")
    nodes = [EvidenceNode(name="N1", label="Entity")]
    edges = [
        EvidenceEdge(source="Start", target="N1", relation_type="连接", confidence=0.9)
    ]

    assert builder.get_current_frontier() == ["Start"]

    builder.add_hop(nodes=nodes, edges=edges, reasoning="first hop")
    chain = builder.finalize(end_entity="N1")

    assert chain.start_entity == "Start"
    assert chain.end_entity == "N1"
    assert chain.nodes[0].hop == 1
    assert chain.steps[0].relation_used == "连接"
    assert builder.get_current_frontier() == ["N1"]


def test_reasoning_orchestrator_parse_decision_with_code_fences() -> None:
    orchestrator = ReasoningOrchestrator(
        graph_retriever=Mock(),
        llm_client=Mock(),
    )
    response = (
        "```json\n"
        '{"should_continue": true, "next_entities": ["B"], '
        '"relation_filter": "包含", "reasoning": "ok"}\n'
        "```"
    )

    decision = orchestrator._parse_decision(response)

    assert decision.should_continue is True
    assert decision.next_entities == ["B"]
    assert decision.relation_filter == "包含"
    assert decision.reasoning == "ok"


def test_reasoning_orchestrator_parse_decision_invalid_json() -> None:
    orchestrator = ReasoningOrchestrator(
        graph_retriever=Mock(),
        llm_client=Mock(),
    )

    decision = orchestrator._parse_decision("not json")

    assert decision == HopDecision(
        should_continue=False,
        next_entities=[],
        relation_filter=None,
        reasoning="",
    )


@pytest.mark.asyncio
async def test_reasoning_orchestrator_no_entities_returns_empty_chain() -> None:
    orchestrator = ReasoningOrchestrator(
        graph_retriever=Mock(),
        llm_client=Mock(),
    )
    question = ParsedQuestion(
        original="",
        intent=QueryIntent.EXPLAIN,
        entities=[],
        relation_hints=[],
        constraints={},
    )

    chain = await orchestrator.reason(question)

    assert chain.start_entity == ""
    assert chain.nodes == []
    assert chain.edges == []
    assert chain.steps == []


@pytest.mark.asyncio
async def test_reasoning_orchestrator_reasoning_flow(
    parsed_question: ParsedQuestion,
) -> None:
    graph_retriever = Mock()
    llm_client = Mock()

    graph_retriever.get_neighbors = AsyncMock(
        side_effect=[
            HopResult(
                nodes=[GraphNode(name="B", label="Entity")],
                relations=[
                    GraphRelation(
                        source="A",
                        target="B",
                        relation_type="包含",
                        properties={"confidence": 0.8},
                    )
                ],
                hop_number=1,
            ),
            HopResult(
                nodes=[GraphNode(name="C", label="Entity")],
                relations=[
                    GraphRelation(
                        source="B",
                        target="C",
                        relation_type="关联",
                        properties={"confidence": 0.6},
                    )
                ],
                hop_number=2,
            ),
        ]
    )

    llm_client.generate = AsyncMock(
        side_effect=[
            '{"should_continue": true, "next_entities": ["B"], '
            '"relation_filter": "关联", "reasoning": "继续"}',
            '{"should_continue": false, "next_entities": [], '
            '"relation_filter": null, "reasoning": "停止"}',
        ]
    )

    orchestrator = ReasoningOrchestrator(
        graph_retriever=graph_retriever,
        llm_client=llm_client,
        config=ReasoningConfig(max_hops=3),
    )

    chain = await orchestrator.reason(parsed_question)

    assert len(chain.nodes) == 2
    assert len(chain.edges) == 2
    assert len(chain.steps) == 2
    assert chain.edges[0].relation_type == "包含"
    assert chain.edges[1].relation_type == "关联"
    assert chain.total_confidence == pytest.approx(0.48)

    graph_retriever.get_neighbors.assert_any_call(
        node_name="A", relation_type="包含", limit=10
    )
    graph_retriever.get_neighbors.assert_any_call(
        node_name="B", relation_type="关联", limit=10
    )


def test_reasoning_orchestrator_should_stop_on_low_confidence() -> None:
    orchestrator = ReasoningOrchestrator(
        graph_retriever=Mock(),
        llm_client=Mock(),
        config=ReasoningConfig(max_hops=2, confidence_threshold=0.9),
    )
    decision = HopDecision(
        should_continue=True,
        next_entities=["B"],
        relation_filter=None,
        reasoning="",
    )
    chain = EvidenceChain(
        edges=[
            EvidenceEdge(source="A", target="B", relation_type="包含", confidence=0.2)
        ]
    )

    assert orchestrator._should_stop(decision, hop_number=1, evidence=chain) is True
