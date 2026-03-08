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
    ReasoningDecision,
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


def test_evidence_chain_to_xml() -> None:
    """to_xml() generates structured XML with edges grouped by hop."""
    builder = EvidenceChainBuilder(start_entity="SDH")

    # Hop 1
    hop1_nodes = [
        EvidenceNode(name="DWDM", label="技术"),
        EvidenceNode(name="时分复用", label="技术"),
    ]
    hop1_edges = [
        EvidenceEdge(
            source="SDH", target="DWDM", relation_type="协同", confidence=0.9
        ),
        EvidenceEdge(
            source="SDH", target="时分复用", relation_type="包含", confidence=0.85
        ),
    ]
    builder.add_hop(nodes=hop1_nodes, edges=hop1_edges, reasoning="第一跳探索")

    # Hop 2
    hop2_nodes = [EvidenceNode(name="波分复用", label="技术")]
    hop2_edges = [
        EvidenceEdge(
            source="DWDM", target="波分复用", relation_type="属于", confidence=0.95
        ),
    ]
    builder.add_hop(
        nodes=hop2_nodes, edges=hop2_edges, reasoning="第二跳探索"
    )

    chain = builder.finalize(end_entity="波分复用")
    xml = chain.to_xml()

    # Structural checks
    assert xml.startswith('<evidence_chain start="SDH"')
    assert 'end="\u6ce2\u5206\u590d\u7528"' in xml or 'end="波分复用"' in xml
    assert '<reasoning_path>' in xml
    assert '<hop number="1">' in xml
    assert '<hop number="2">' in xml
    assert '<entities>' in xml
    assert '<reasoning_steps>' in xml

    # Hop 1 edges
    assert '协同' in xml
    assert '包含' in xml

    # Hop 2 edges
    assert '属于' in xml
    assert '波分复用' in xml

    # Entities
    assert 'name="SDH"' not in xml  # start entity not in nodes list
    assert 'name="DWDM"' in xml
    assert 'hop="1"' in xml
    assert 'hop="2"' in xml

    # Edge hop tracking
    assert chain.edges[0].hop == 1  # hop1 edges tagged with hop=1
    assert chain.edges[1].hop == 1
    assert chain.edges[2].hop == 2  # hop2 edge tagged with hop=2


def test_reasoning_orchestrator_parse_decision_with_code_fences() -> None:
    orchestrator = ReasoningOrchestrator(
        graph_retriever=Mock(),
        llm_client=Mock(),
    )
    response = (
        "```json\n"
        '{"continue": true, "next_entities": ["B"], '
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

    assert decision == ReasoningDecision(
        should_continue=False,  # No entities extracted from plain text
        next_entities=[],
        relation_filter=None,
        reasoning="not json",  # Fallback uses first 200 chars
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

    # Entity resolution: search_nodes returns node "A" for exact match
    graph_retriever.search_nodes = AsyncMock(
        return_value=[GraphNode(name="A", label="Entity")]
    )
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
            '{"continue": true, "next_entities": ["B"], '
            '"relation_filter": "关联", "reasoning": "继续"}',
            '{"continue": false, "next_entities": [], '
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
        node_name="A", relation_type="包含"  # First hop uses relation_hints[0]
    )
    graph_retriever.get_neighbors.assert_any_call(
        node_name="B", relation_type="关联"  # Second hop uses decision's relation_filter
    )


def test_reasoning_orchestrator_should_stop_on_low_confidence() -> None:
    orchestrator = ReasoningOrchestrator(
        graph_retriever=Mock(),
        llm_client=Mock(),
        config=ReasoningConfig(max_hops=2, min_confidence=0.9),
    )
    decision = ReasoningDecision(
        should_continue=True,
        next_entities=["B"],
        relation_filter=None,
        reasoning="",
    )
    chain = EvidenceChain(
        edges=[
            EvidenceEdge(source="A", target="B", relation_type="包含", confidence=0.2)
        ],
        total_confidence=0.2,  # Explicitly set low confidence
    )

    assert orchestrator._should_stop(decision, hop_number=1, evidence=chain) is True


@pytest.mark.asyncio
async def test_reasoning_orchestrator_filters_invalid_next_entities(
    parsed_question: ParsedQuestion,
) -> None:
    """Test that next_entities not in hop_result.nodes are filtered out."""
    graph_retriever = Mock()
    llm_client = Mock()

    # Entity resolution: search_nodes returns node "A" for exact match
    graph_retriever.search_nodes = AsyncMock(
        return_value=[GraphNode(name="A", label="Entity")]
    )

    # Hop returns only node B, but LLM suggests both B and PHANTOM
    graph_retriever.get_neighbors = AsyncMock(
        return_value=HopResult(
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
        )
    )

    # LLM hallucinates "PHANTOM" entity that doesn't exist in graph neighbors
    llm_client.generate = AsyncMock(
        return_value='{"should_continue": true, "next_entities": ["B", "PHANTOM"], '
        '"relation_filter": null, "reasoning": "继续"}'
    )

    orchestrator = ReasoningOrchestrator(
        graph_retriever=graph_retriever,
        llm_client=llm_client,
        config=ReasoningConfig(max_hops=1),
    )

    chain = await orchestrator.reason(parsed_question)

    # Only B should be in the chain, PHANTOM should be filtered out
    node_names = [n.name for n in chain.nodes]
    assert "PHANTOM" not in node_names
    assert "B" in node_names


@pytest.mark.asyncio
async def test_reasoning_orchestrator_logs_warning_for_hallucinated_entities(
    parsed_question: ParsedQuestion,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that warning is logged when LLM suggests non-existent entities."""
    import logging

    graph_retriever = Mock()
    llm_client = Mock()

    # Entity resolution: search_nodes returns node "A" for exact match
    graph_retriever.search_nodes = AsyncMock(
        return_value=[GraphNode(name="A", label="Entity")]
    )

    graph_retriever.get_neighbors = AsyncMock(
        return_value=HopResult(
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
        )
    )

    # LLM only suggests entity that doesn't exist
    llm_client.generate = AsyncMock(
        return_value='{"should_continue": true, "next_entities": ["PHANTOM"], '
        '"relation_filter": null, "reasoning": "继续"}'
    )

    orchestrator = ReasoningOrchestrator(
        graph_retriever=graph_retriever,
        llm_client=llm_client,
        config=ReasoningConfig(max_hops=2),  # Need >= 2 to trigger filtering in loop
    )

    with caplog.at_level(logging.WARNING):
        await orchestrator.reason(parsed_question)

    assert any(
        "not in graph neighbors" in record.message for record in caplog.records
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entity Resolution Tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_resolve_entities_exact_match() -> None:
    """Entity that exactly matches a graph node needs no LLM call."""
    graph_retriever = Mock()
    llm_client = Mock()

    graph_retriever.search_nodes = AsyncMock(
        return_value=[
            GraphNode(name="SDH", label="Entity"),
            GraphNode(name="DWDM", label="Entity"),
            GraphNode(name="OTN", label="Entity"),
        ]
    )

    orchestrator = ReasoningOrchestrator(
        graph_retriever=graph_retriever, llm_client=llm_client
    )

    result = await orchestrator._resolve_entities(["SDH", "DWDM"])

    assert result == {"SDH": "SDH", "DWDM": "DWDM"}
    # LLM should NOT be called for exact matches
    llm_client.generate.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_entities_substring_match() -> None:
    """Entity 'SDH\u7f51\u7edc' should match graph node 'SDH' via substring."""
    graph_retriever = Mock()
    llm_client = Mock()

    graph_retriever.search_nodes = AsyncMock(
        return_value=[
            GraphNode(name="SDH", label="Entity"),
            GraphNode(name="DWDM", label="Entity"),
        ]
    )

    orchestrator = ReasoningOrchestrator(
        graph_retriever=graph_retriever, llm_client=llm_client
    )

    result = await orchestrator._resolve_entities(["SDH\u7f51\u7edc"])

    assert result == {"SDH\u7f51\u7edc": "SDH"}
    # Substring match should not require LLM
    llm_client.generate.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_entities_llm_resolve() -> None:
    """Entity 'WDM' should be resolved to '\u6ce2\u5206\u590d\u7528' by LLM."""
    graph_retriever = Mock()
    llm_client = Mock()

    graph_retriever.search_nodes = AsyncMock(
        return_value=[
            GraphNode(name="SDH", label="Entity"),
            GraphNode(name="\u6ce2\u5206\u590d\u7528", label="Entity"),
            GraphNode(name="OTN", label="Entity"),
        ]
    )

    # LLM correctly maps WDM -> \u6ce2\u5206\u590d\u7528
    llm_client.generate = AsyncMock(return_value="\u6ce2\u5206\u590d\u7528")

    orchestrator = ReasoningOrchestrator(
        graph_retriever=graph_retriever, llm_client=llm_client
    )

    result = await orchestrator._resolve_entities(["WDM"])

    assert result == {"WDM": "\u6ce2\u5206\u590d\u7528"}
    llm_client.generate.assert_called_once()


@pytest.mark.asyncio
async def test_resolve_entities_llm_reformulate_then_match() -> None:
    """LLM first suggests wrong name, then correct name on retry."""
    graph_retriever = Mock()
    llm_client = Mock()

    graph_retriever.search_nodes = AsyncMock(
        return_value=[
            GraphNode(name="\u5149\u7ea4\u901a\u4fe1", label="Entity"),
            GraphNode(name="SDH", label="Entity"),
        ]
    )

    # First call: LLM returns a wrong name; second call: correct name
    llm_client.generate = AsyncMock(
        side_effect=["\u5149\u7ea4", "\u5149\u7ea4\u901a\u4fe1"]
    )

    orchestrator = ReasoningOrchestrator(
        graph_retriever=graph_retriever, llm_client=llm_client
    )

    # Use entity that has NO substring match with any graph node
    # '\u5149\u7f51\u7edc\u6280\u672f' does NOT contain/is-contained-in '\u5149\u7ea4\u901a\u4fe1' or 'SDH'
    result = await orchestrator._resolve_entities(["\u5149\u7f51\u7edc\u6280\u672f"])

    assert result["\u5149\u7f51\u7edc\u6280\u672f"] == "\u5149\u7ea4\u901a\u4fe1"
    # First LLM call returns '\u5149\u7ea4' (not in graph, but substring of '\u5149\u7ea4\u901a\u4fe1')
    # so reformulation matches via substring on iter 1
    assert llm_client.generate.call_count == 1


@pytest.mark.asyncio
async def test_resolve_entities_unresolved() -> None:
    """Entity that cannot be resolved returns empty mapping."""
    graph_retriever = Mock()
    llm_client = Mock()

    graph_retriever.search_nodes = AsyncMock(
        return_value=[GraphNode(name="SDH", label="Entity")]
    )

    # LLM always returns NONE
    llm_client.generate = AsyncMock(return_value="NONE")

    orchestrator = ReasoningOrchestrator(
        graph_retriever=graph_retriever, llm_client=llm_client
    )

    result = await orchestrator._resolve_entities(["\u4e0d\u5b58\u5728\u7684\u5b9e\u4f53"])

    assert result == {}


@pytest.mark.asyncio
async def test_resolve_entities_mixed() -> None:
    """Mix of exact, substring, and LLM resolution."""
    graph_retriever = Mock()
    llm_client = Mock()

    graph_retriever.search_nodes = AsyncMock(
        return_value=[
            GraphNode(name="SDH", label="Entity"),
            GraphNode(name="OTN", label="Entity"),
            GraphNode(name="\u6ce2\u5206\u590d\u7528", label="Entity"),
        ]
    )

    # LLM only called for "WDM" (not exact or substring)
    llm_client.generate = AsyncMock(return_value="\u6ce2\u5206\u590d\u7528")

    orchestrator = ReasoningOrchestrator(
        graph_retriever=graph_retriever, llm_client=llm_client
    )

    result = await orchestrator._resolve_entities(
        ["SDH", "SDH\u7f51\u7edc", "WDM"]
    )

    assert result["SDH"] == "SDH"              # exact
    assert result["SDH\u7f51\u7edc"] == "SDH"    # substring
    assert result["WDM"] == "\u6ce2\u5206\u590d\u7528"   # LLM


def test_substring_match_bidirectional() -> None:
    """Test bidirectional substring matching."""
    orchestrator = ReasoningOrchestrator(
        graph_retriever=Mock(), llm_client=Mock()
    )
    nodes = ["SDH", "DWDM", "\u6ce2\u5206\u590d\u7528", "\u5149\u7ea4\u901a\u4fe1"]

    # Query is superstring of node name
    assert orchestrator._substring_match("SDH\u7f51\u7edc", nodes) == ["SDH"]

    # Query is substring of node name
    assert orchestrator._substring_match("\u5149\u7ea4", nodes) == ["\u5149\u7ea4\u901a\u4fe1"]

    # No match
    assert orchestrator._substring_match("\u4e0d\u5b58\u5728", nodes) == []

    # Multiple matches (ambiguous)
    nodes_with_dup = ["SDH", "SDH-NG"]
    assert orchestrator._substring_match("SDH", nodes_with_dup) == ["SDH", "SDH-NG"]


@pytest.mark.asyncio
async def test_resolve_entities_empty_graph() -> None:
    """Empty graph returns empty resolution."""
    graph_retriever = Mock()
    llm_client = Mock()

    graph_retriever.search_nodes = AsyncMock(return_value=[])

    orchestrator = ReasoningOrchestrator(
        graph_retriever=graph_retriever, llm_client=llm_client
    )

    result = await orchestrator._resolve_entities(["SDH"])

    assert result == {}



# ═══════════════════════════════════════════════════════════════════════════
# Soft-hint Relation Filter Fallback Tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_execute_hop_falls_back_when_relation_filter_returns_empty() -> None:
    """_execute_hop retries without relation_filter when filtered query returns empty."""
    graph_retriever = Mock()
    llm_client = Mock()

    # First call with relation_type='使用' -> empty
    # Second call without relation_type -> returns neighbors
    graph_retriever.get_neighbors = AsyncMock(
        side_effect=[
            HopResult(nodes=[], relations=[], hop_number=1),
            HopResult(
                nodes=[GraphNode(name="时分复用", label="技术")],
                relations=[
                    GraphRelation(
                        source="SDH",
                        target="时分复用",
                        relation_type="包含",
                        properties={"confidence": 0.9},
                    )
                ],
                hop_number=1,
            ),
        ]
    )

    orchestrator = ReasoningOrchestrator(
        graph_retriever=graph_retriever, llm_client=llm_client
    )

    result = await orchestrator._execute_hop(
        current_entities=["SDH"], hop_number=1, context="使用"
    )

    assert len(result.nodes) == 1
    assert result.nodes[0].name == "时分复用"
    assert len(result.relations) == 1
    # Verify two calls: first with filter, second without
    assert graph_retriever.get_neighbors.call_count == 2
    graph_retriever.get_neighbors.assert_any_call(
        node_name="SDH", relation_type="使用"
    )
    graph_retriever.get_neighbors.assert_any_call(node_name="SDH")


@pytest.mark.asyncio
async def test_execute_hop_no_fallback_when_filter_returns_results() -> None:
    """_execute_hop does NOT retry when filtered query returns results."""
    graph_retriever = Mock()
    llm_client = Mock()

    graph_retriever.get_neighbors = AsyncMock(
        return_value=HopResult(
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
        )
    )

    orchestrator = ReasoningOrchestrator(
        graph_retriever=graph_retriever, llm_client=llm_client
    )

    result = await orchestrator._execute_hop(
        current_entities=["A"], hop_number=1, context="包含"
    )

    assert len(result.nodes) == 1
    # Only one call — no fallback needed
    graph_retriever.get_neighbors.assert_called_once_with(
        node_name="A", relation_type="包含"
    )


@pytest.mark.asyncio
async def test_single_entity_reasoning_with_wrong_relation_hint() -> None:
    """Single-entity question with mismatched relation_hint still returns evidence.

    Simulates Q2: 'SDH网络中使用了哪些复用技术？'
    - relation_hints=['使用'] but graph has no '使用' edges
    - After fallback, SDH's neighbors are discovered via unfiltered query
    """
    graph_retriever = Mock()
    llm_client = Mock()

    # Entity resolution: search_nodes returns SDH
    graph_retriever.search_nodes = AsyncMock(
        return_value=[
            GraphNode(name="SDH", label="技术"),
            GraphNode(name="时分复用", label="技术"),
        ]
    )

    # _execute_hop calls:
    # 1st: get_neighbors(SDH, relation_type='使用') -> empty (triggers fallback)
    # 2nd: get_neighbors(SDH) -> returns neighbor
    graph_retriever.get_neighbors = AsyncMock(
        side_effect=[
            # Hop 1, first call with filter -> empty
            HopResult(nodes=[], relations=[], hop_number=1),
            # Hop 1, fallback without filter -> results
            HopResult(
                nodes=[GraphNode(name="时分复用", label="技术")],
                relations=[
                    GraphRelation(
                        source="SDH",
                        target="时分复用",
                        relation_type="包含",
                        properties={"confidence": 0.9},
                    )
                ],
                hop_number=1,
            ),
        ]
    )

    # LLM decision: stop after first hop (single entity, got results)
    llm_client.generate = AsyncMock(
        return_value='{"continue": false, "next_entities": [], '
        '"relation_filter": null, "reasoning": "找到了复用技术"}'
    )

    question = ParsedQuestion(
        original="SDH网络中使用了哪些复用技术？",
        intent=QueryIntent.LIST,
        entities=["SDH"],
        relation_hints=["使用"],
        constraints={},
    )

    orchestrator = ReasoningOrchestrator(
        graph_retriever=graph_retriever,
        llm_client=llm_client,
        config=ReasoningConfig(max_hops=3),
    )

    chain = await orchestrator.reason(question)

    # Should have evidence: SDH --包含--> 时分复用
    assert len(chain.edges) >= 1
    assert chain.edges[0].source == "SDH"
    assert chain.edges[0].target == "时分复用"
    assert chain.edges[0].relation_type == "包含"
    assert len(chain.nodes) >= 1


@pytest.mark.asyncio
async def test_goal_entities_edge_filter_bidirectional() -> None:
    """Goal-entities edge filter keeps edges where goal is source OR target.

    Simulates Q1: 'DWDM和WDM是什么关系？'
    Graph has: 波分复用 --包含--> DWDM (source=波分复用, target=DWDM)
    Goal entity = 波分复用. Edge must be kept even though target=DWDM, not 波分复用.
    """
    graph_retriever = Mock()
    llm_client = Mock()

    # Entity resolution: DWDM exact, WDM -> 波分复用 (substring)
    graph_retriever.search_nodes = AsyncMock(
        return_value=[
            GraphNode(name="DWDM", label="技术"),
            GraphNode(name="波分复用", label="技术"),
        ]
    )

    # Hop 1: DWDM's neighbors include 波分复用 via reverse edge
    graph_retriever.get_neighbors = AsyncMock(
        return_value=HopResult(
            nodes=[GraphNode(name="波分复用", label="技术")],
            relations=[
                GraphRelation(
                    source="波分复用",  # source is the GOAL entity!
                    target="DWDM",  # target is the START entity!
                    relation_type="包含",
                    properties={"confidence": 0.95},
                )
            ],
            hop_number=1,
        )
    )

    # LLM decision: suggests entities NOT in valid_neighbors
    # -> next_entities will be empty after filtering -> triggers goal_entities path
    llm_client.generate = AsyncMock(
        return_value='{"continue": false, "next_entities": ["WDM"], '
        '"relation_filter": null, "reasoning": "找到了目标关系"}'
    )

    question = ParsedQuestion(
        original="DWDM和WDM是什么关系？",
        intent=QueryIntent.FIND_RELATION,
        entities=["DWDM", "波分复用"],  # After entity resolution
        relation_hints=["包含"],
        constraints={},
    )

    orchestrator = ReasoningOrchestrator(
        graph_retriever=graph_retriever,
        llm_client=llm_client,
        config=ReasoningConfig(max_hops=3),
    )

    chain = await orchestrator.reason(question)

    # Edge 波分复用 --包含--> DWDM should be kept (goal entity = 波分复用 is source)
    assert len(chain.edges) >= 1
    found = any(
        e.source == "波分复用" and e.target == "DWDM" and e.relation_type == "包含"
        for e in chain.edges
    )
    assert found, (
        f"Expected edge 波分复用 --包含--> DWDM not found. Edges: {chain.edges}"
    )
    # Goal entity node should also be in evidence
    assert any(n.name == "波分复用" for n in chain.nodes)


@pytest.mark.asyncio
async def test_hop_level_retry_when_goal_filter_empty_due_to_wrong_edge() -> None:
    """Hop-level retry fires when relation_filter matches irrelevant edges.

    Simulates: DWDM has edge OTN--包含-->DWDM and DWDM--属于-->波分复用.
    relation_filter='包含' returns OTN (irrelevant). Goal filter rejects it.
    Retry without filter returns 波分复用 (the actual goal).
    """
    graph_retriever = Mock()
    llm_client = Mock()

    graph_retriever.search_nodes = AsyncMock(
        return_value=[
            GraphNode(name="DWDM", label="技术"),
            GraphNode(name="波分复用", label="技术"),
        ]
    )

    # Call sequence for get_neighbors:
    # 1. _execute_hop with filter '包含' -> OTN (irrelevant to goal)
    # 2. _execute_hop retry without filter -> 波分复用 (goal entity)
    graph_retriever.get_neighbors = AsyncMock(
        side_effect=[
            # Hop 1 first attempt: filter='包含' -> returns OTN
            HopResult(
                nodes=[GraphNode(name="OTN", label="技术")],
                relations=[
                    GraphRelation(
                        source="OTN",
                        target="DWDM",
                        relation_type="包含",
                        properties={"confidence": 0.9},
                    )
                ],
                hop_number=1,
            ),
            # Hop 1 retry: no filter -> returns 波分复用 (goal)
            HopResult(
                nodes=[
                    GraphNode(name="波分复用", label="技术"),
                    GraphNode(name="OTN", label="技术"),
                ],
                relations=[
                    GraphRelation(
                        source="DWDM",
                        target="波分复用",
                        relation_type="属于",
                        properties={"confidence": 0.95},
                    ),
                    GraphRelation(
                        source="OTN",
                        target="DWDM",
                        relation_type="包含",
                        properties={"confidence": 0.9},
                    ),
                ],
                hop_number=1,
            ),
        ]
    )

    # LLM decision for first hop (with OTN result) -> suggests wrong entities
    llm_client.generate = AsyncMock(
        return_value='{"continue": false, "next_entities": ["WDM"], '
        '"relation_filter": null, "reasoning": "探索关系"}'
    )

    question = ParsedQuestion(
        original="DWDM和WDM是什么关系？",
        intent=QueryIntent.FIND_RELATION,
        entities=["DWDM", "波分复用"],
        relation_hints=["包含"],
        constraints={},
    )

    orchestrator = ReasoningOrchestrator(
        graph_retriever=graph_retriever,
        llm_client=llm_client,
        config=ReasoningConfig(max_hops=3),
    )

    chain = await orchestrator.reason(question)

    # After hop-level retry, should find DWDM --属于--> 波分复用
    assert len(chain.edges) >= 1
    found = any(
        e.source == "DWDM"
        and e.target == "波分复用"
        and e.relation_type == "属于"
        for e in chain.edges
    )
    assert found, (
        f"Expected DWDM --属于--> 波分复用 after hop-level retry. "
        f"Got: {chain.edges}"
    )
    assert any(n.name == "波分复用" for n in chain.nodes)