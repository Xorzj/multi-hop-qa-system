import pytest

from src.data_processing.schema_inducer import DomainSchema, EntityTypeSpec
from src.llm.base_client import BaseLLMClient, GenerationParams
from src.qa_engine.answer_generator import AnswerGenerator, GeneratedAnswer
from src.qa_engine.context_assembler import AssembledContext, ContextAssembler
from src.qa_engine.query_rewriter import QueryPlan, QueryRewriter, QueryStep
from src.qa_engine.question_parser import ParsedQuestion, QueryIntent, QuestionParser
from src.reasoning.evidence_chain import (
    EvidenceChain,
    EvidenceEdge,
    EvidenceNode,
    EvidenceStep,
)


class FakeLLMClient(BaseLLMClient):
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[tuple[str, GenerationParams | None]] = []

    async def generate(
        self, prompt: str, params: GenerationParams | None = None
    ) -> str:
        self.prompts.append((prompt, params))
        return self.response

    async def chat(
        self, messages: list[dict[str, str]], params: GenerationParams | None = None
    ) -> str:
        self.prompts.append(
            ("\n".join(message["content"] for message in messages), params)
        )
        return self.response

    @property
    def provider(self) -> str:
        return "fake"


@pytest.fixture()
def llm_factory() -> callable:
    def _factory(response: str) -> FakeLLMClient:
        return FakeLLMClient(response)

    return _factory


@pytest.fixture()
def evidence_chain() -> EvidenceChain:
    nodes = [
        EvidenceNode(
            name="SDH", label="Tech", properties={"description": "同步数字体系"}
        ),
        EvidenceNode(name="STM-1", label="Signal", properties={"rate": "155Mb/s"}),
    ]
    edges = [
        EvidenceEdge(
            source="SDH",
            target="STM-1",
            relation_type="包含",
            confidence=0.8,
        )
    ]
    steps = [
        EvidenceStep(
            hop_number=1,
            action="expand",
            nodes_explored=["STM-1"],
            relation_used="包含",
            reasoning="SDH包含STM-1",
        )
    ]
    chain = EvidenceChain(
        nodes=nodes,
        edges=edges,
        steps=steps,
        start_entity="SDH",
    )
    chain.calculate_confidence()
    return chain


@pytest.fixture()
def context_assembler() -> ContextAssembler:
    return ContextAssembler(max_context_length=4000, include_cot=True)


def test_parsed_question_defaults() -> None:
    parsed = ParsedQuestion(original="Q", intent=QueryIntent.EXPLAIN)
    assert parsed.entities == []
    assert parsed.relation_hints == []
    assert parsed.constraints == {}


def test_question_parser_parse_response_valid_json(llm_factory: callable) -> None:
    parser = QuestionParser(llm_factory("{}"))
    response = (
        "```json\n"
        '{"intent":"FIND_RELATION",'
        '"entities":["SDH","STM-1"],'
        '"relation_hints":["包含"],'
        '"constraints":{"max_hops":2}}\n'
        "```"
    )
    parsed = parser._parse_response(response, original="SDH和STM-1什么关系")
    assert parsed.intent == QueryIntent.FIND_RELATION
    assert parsed.entities == ["SDH", "STM-1"]
    assert parsed.relation_hints == ["包含"]
    assert parsed.constraints == {"max_hops": 2}


def test_question_parser_parse_response_invalid_json_fallback(
    llm_factory: callable,
) -> None:
    parser = QuestionParser(llm_factory("not used"))
    parsed = parser._parse_response("not json", original="SDH是什么")
    assert parsed.intent == QueryIntent.EXPLAIN
    assert parsed.relation_hints == []
    assert parsed.constraints == {}


def test_question_parser_fallback_extracts_acronyms(llm_factory: callable) -> None:
    """Generic acronym patterns: TCP, HTTP-2, ACID."""
    parser = QuestionParser(llm_factory("not used"))
    parsed = parser._parse_response("not json", original="TCP和HTTP-2是什么关系")
    assert "TCP" in parsed.entities
    assert "HTTP-2" in parsed.entities


def test_question_parser_fallback_extracts_standards(llm_factory: callable) -> None:
    """Standards like G.709, X.25."""
    parser = QuestionParser(llm_factory("not used"))
    parsed = parser._parse_response("not json", original="G.709标准的作用是什么")
    assert "G.709" in parsed.entities


def test_question_parser_fallback_extracts_camelcase(llm_factory: callable) -> None:
    """CamelCase identifiers like PostgreSQL, FastAPI."""
    parser = QuestionParser(llm_factory("not used"))
    parsed = parser._parse_response("not json", original="PostgreSQL和MongoDB哪个好")
    camel_entities = [e for e in parsed.entities if e in ("PostgreSQL", "MongoDB")]
    assert len(camel_entities) >= 1


def test_question_parser_fallback_extracts_measurements(llm_factory: callable) -> None:
    """Measurement values like 10Gb/s, 100MHz."""
    parser = QuestionParser(llm_factory("not used"))
    parsed = parser._parse_response("not json", original="10Gb/s和100MHz的关系")
    # Measurements should be captured
    measurement_entities = [e for e in parsed.entities if "Gb/s" in e or "MHz" in e]
    assert len(measurement_entities) >= 1


def test_question_parser_fallback_no_optical_hardcode(llm_factory: callable) -> None:
    """Medical domain entities should work via generic acronym patterns."""
    parser = QuestionParser(llm_factory("not used"))
    parsed = parser._parse_response("not json", original="DNA和RNA的区别")
    assert "DNA" in parsed.entities
    assert "RNA" in parsed.entities


@pytest.mark.asyncio
async def test_question_parser_parse_empty_question(llm_factory: callable) -> None:
    parser = QuestionParser(llm_factory("{}"))
    parsed = await parser.parse("   ")
    assert parsed.intent == QueryIntent.EXPLAIN
    assert parsed.entities == []
    assert parsed.relation_hints == []
    assert parsed.constraints == {}


def test_assembled_context_defaults() -> None:
    context = AssembledContext(question="Q", evidence_summary="summary")
    assert context.evidence_confidence == 1.0
    assert context.reasoning_steps == []
    assert context.entity_descriptions == {}
    assert context.prompt == ""


def test_context_assembler_formats_prompt(
    context_assembler: ContextAssembler, evidence_chain: EvidenceChain
) -> None:
    question = "SDH和STM-1的关系是什么"
    assembled = context_assembler.assemble(
        question, evidence_chain, include_reasoning=True
    )
    assert assembled.question == question
    assert "<evidence_chain" in assembled.evidence_summary
    assert "SDH" in assembled.evidence_summary
    assert "STM-1" in assembled.evidence_summary
    assert assembled.evidence_confidence == pytest.approx(0.8)
    assert "## 用户问题" in assembled.prompt
    assert "## 知识图谱证据" in assembled.prompt
    assert "## 推理路径" in assembled.prompt
    assert "SDH包含STM-1" in "\n".join(assembled.reasoning_steps)


def test_context_assembler_without_reasoning(
    context_assembler: ContextAssembler, evidence_chain: EvidenceChain
) -> None:
    assembled = context_assembler.assemble("Q", evidence_chain, include_reasoning=False)
    assert assembled.reasoning_steps == []
    assert "## 推理路径" not in assembled.prompt


def test_generated_answer_dataclass() -> None:
    generated = GeneratedAnswer(
        answer="A",
        confidence=0.9,
        reasoning_steps=None,
        latency_ms=1.0,
    )
    assert generated.answer == "A"
    assert generated.confidence == 0.9
    assert generated.reasoning_steps is None
    assert generated.latency_ms == 1.0
    assert generated.tokens_used is None


@pytest.mark.asyncio
async def test_answer_generator_parses_reasoning(llm_factory: callable) -> None:
    response = "答案内容。推理过程: 第一步；第二步"
    generator = AnswerGenerator(llm_factory(response))
    context = AssembledContext(
        question="Q",
        evidence_summary="summary",
        evidence_confidence=0.9,
        prompt="PROMPT",
    )
    generated = await generator.generate(context, include_reasoning=True)
    assert generated.answer == "答案内容。"
    assert generated.reasoning_steps == ["第一步", "第二步"]
    assert generated.latency_ms >= 0


@pytest.mark.asyncio
async def test_answer_generator_omits_reasoning_when_disabled(
    llm_factory: callable,
) -> None:
    response = "可能需要更多信息"
    generator = AnswerGenerator(llm_factory(response))
    context = AssembledContext(
        question="Q",
        evidence_summary="summary",
        evidence_confidence=0.7,
        prompt="PROMPT",
    )
    generated = await generator.generate(context, include_reasoning=False)
    assert generated.reasoning_steps is None
    assert generated.answer == response
    assert 0.0 <= generated.confidence <= 1.0


# ===================== QueryRewriter tests =====================


def _make_schema_with_labels(*labels: str) -> DomainSchema:
    """Helper to create a DomainSchema with given entity type labels."""
    return DomainSchema(
        entity_types=[
            EntityTypeSpec(name=label, label=label, definition=f"{label}类型")
            for label in labels
        ]
    )


@pytest.mark.asyncio
async def test_query_rewriter_separates_entity_and_type(
    llm_factory: callable,
) -> None:
    """LLM correctly splits entities: 冠心病→start_entity, 症状→target_type."""
    response = (
        '{"start_entities": ["冠心病"], "steps": ['
        '{"action": "find_neighbors", "target_type": "症状", '
        '"relation_hint": "引起", "direction": "out", '
        '"description": "查找冠心病引起的症状"}'
        "]}"
    )
    schema = _make_schema_with_labels("症状", "药物", "疾病")
    rewriter = QueryRewriter(llm_factory(response), domain_schema=schema)
    parsed = ParsedQuestion(
        original="冠心病会引起哪些症状？",
        intent=QueryIntent.FIND_ENTITY,
        entities=["冠心病", "症状"],
        relation_hints=["引起"],
    )
    plan = await rewriter.rewrite(parsed)
    assert plan.start_entities == ["冠心病"]
    assert len(plan.steps) == 1
    assert plan.steps[0].target_type == "症状"
    assert plan.steps[0].action == "find_neighbors"


@pytest.mark.asyncio
async def test_query_rewriter_multihop_decomposition(
    llm_factory: callable,
) -> None:
    """Multi-hop query: 冠心病→症状→药物."""
    response = (
        '{"start_entities": ["冠心病"], "steps": ['
        '{"action": "find_neighbors", "target_type": "症状", '
        '"direction": "out", "description": "找症状"}, '
        '{"action": "find_neighbors", "target_type": "药物", '
        '"direction": "in", "description": "找药物"}'
        "]}"
    )
    schema = _make_schema_with_labels("症状", "药物")
    rewriter = QueryRewriter(llm_factory(response), domain_schema=schema)
    parsed = ParsedQuestion(
        original="冠心病的症状用什么药物治疗？",
        intent=QueryIntent.FIND_ENTITY,
        entities=["冠心病", "症状", "药物"],
    )
    plan = await rewriter.rewrite(parsed)
    assert plan.start_entities == ["冠心病"]
    assert len(plan.steps) == 2
    assert plan.steps[0].target_type == "症状"
    assert plan.steps[1].target_type == "药物"
    assert plan.steps[1].direction == "in"


@pytest.mark.asyncio
async def test_query_rewriter_all_concrete_entities(
    llm_factory: callable,
) -> None:
    """When all entities are concrete, no target_type filter is used."""
    response = (
        '{"start_entities": ["SDH", "WDM"], "steps": ['
        '{"action": "find_neighbors", "direction": "both", '
        '"description": "查找关联"}'
        "]}"
    )
    rewriter = QueryRewriter(llm_factory(response))
    parsed = ParsedQuestion(
        original="SDH和WDM是什么关系？",
        intent=QueryIntent.FIND_RELATION,
        entities=["SDH", "WDM"],
    )
    plan = await rewriter.rewrite(parsed)
    assert plan.start_entities == ["SDH", "WDM"]
    assert len(plan.steps) == 1
    assert plan.steps[0].target_type is None


@pytest.mark.asyncio
async def test_query_rewriter_llm_failure_fallback(llm_factory: callable) -> None:
    """When LLM returns garbage, fallback plan uses schema to classify."""
    schema = _make_schema_with_labels("症状", "药物")
    # Return unparseable text
    rewriter = QueryRewriter(llm_factory("not valid json!!"), domain_schema=schema)
    parsed = ParsedQuestion(
        original="冠心病会引起哪些症状？",
        intent=QueryIntent.FIND_ENTITY,
        entities=["冠心病", "症状"],
        relation_hints=["引起"],
    )
    plan = await rewriter.rewrite(parsed)
    # Fallback: "冠心病" is not a schema label → start_entity
    # "症状" is a schema label → target_type
    assert "冠心病" in plan.start_entities
    assert len(plan.steps) >= 1
    assert any(s.target_type == "症状" for s in plan.steps)


@pytest.mark.asyncio
async def test_query_rewriter_no_schema_fallback(llm_factory: callable) -> None:
    """Without domain schema, fallback puts all entities as start_entities."""
    rewriter = QueryRewriter(llm_factory("garbage"))
    parsed = ParsedQuestion(
        original="A和B的关系",
        intent=QueryIntent.FIND_RELATION,
        entities=["A", "B"],
    )
    plan = await rewriter.rewrite(parsed)
    # No schema → all entities become start_entities
    assert "A" in plan.start_entities
    assert "B" in plan.start_entities


def test_query_plan_dataclass_defaults() -> None:
    """QueryPlan defaults are sane."""
    plan = QueryPlan()
    assert plan.start_entities == []
    assert plan.steps == []
    assert plan.original_question == ""
    assert plan.raw_entities == []


def test_query_step_dataclass_defaults() -> None:
    """QueryStep defaults are sane."""
    step = QueryStep()
    assert step.action == "find_neighbors"
    assert step.target_type is None
    assert step.relation_hint is None
    assert step.direction == "both"


@pytest.mark.asyncio
async def test_query_rewriter_empty_entities(llm_factory: callable) -> None:
    """With no entities, returns empty plan immediately."""
    rewriter = QueryRewriter(llm_factory("should not be called"))
    parsed = ParsedQuestion(original="你好", intent=QueryIntent.EXPLAIN, entities=[])
    plan = await rewriter.rewrite(parsed)
    assert plan.start_entities == []
    assert plan.steps == []


@pytest.mark.asyncio
async def test_query_rewriter_code_fence_response(llm_factory: callable) -> None:
    """LLM response wrapped in markdown code fence is parsed correctly."""
    response = (
        "```json\n"
        '{"start_entities": ["X"], "steps": ['
        '{"action": "find_neighbors", "target_type": "Y", "direction": "out"}'
        "]}\n"
        "```"
    )
    rewriter = QueryRewriter(llm_factory(response))
    parsed = ParsedQuestion(
        original="X的Y有哪些",
        intent=QueryIntent.FIND_ENTITY,
        entities=["X", "Y"],
    )
    plan = await rewriter.rewrite(parsed)
    assert plan.start_entities == ["X"]
    assert plan.steps[0].target_type == "Y"
