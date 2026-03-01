import pytest

from src.llm.base_client import BaseLLMClient, GenerationParams
from src.qa_engine.answer_generator import AnswerGenerator, GeneratedAnswer
from src.qa_engine.context_assembler import AssembledContext, ContextAssembler
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
    assert "路径:" in assembled.evidence_summary
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
