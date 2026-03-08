from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.common.exceptions import NotFoundError, ValidationError
from src.data_processing.document_loader import DocumentLoader, Section
from src.data_processing.entity_extractor import (
    Entity,
    EntityExtractor,
    IncrementalRelation,
)
from src.data_processing.entity_merger import EntityMerger, MergeConfig
from src.data_processing.quality_verifier import QualityVerifier
from src.data_processing.relation_types import (
    CORRELATE_EDGE,
    DEFAULT_RELATION_TYPES,
    INCLUDE_EDGE,
    RelationType,
    build_relation_type_prompt,
    get_relation_type,
    register_relation_type,
)
from src.data_processing.triple_extractor import Triple, TripleExtractor


@pytest.fixture()
def fake_markitdown_module():
    class FakeResult:
        def __init__(self, content: str) -> None:
            self.text_content = content

    class FakeMarkItDown:
        def __init__(self) -> None:
            self.last_path: str | None = None

        def convert(self, path: str) -> FakeResult:
            self.last_path = path
            return FakeResult("Hello ![](data:image/png;base64,AAA) World")

    return SimpleNamespace(MarkItDown=FakeMarkItDown)


@pytest.fixture()
def fake_llm_client_factory():
    class FakeLLMClient:
        def __init__(self, response: str) -> None:
            self.response = response
            self.last_prompt: str | None = None
            self.last_params: Any | None = None
            self.call_count = 0

        async def generate(self, prompt: str, params=None) -> str:  # type: ignore[override]
            self.call_count += 1
            self.last_prompt = prompt
            self.last_params = params
            return self.response

    def _factory(response: str) -> FakeLLMClient:
        return FakeLLMClient(response)

    return _factory


@pytest.fixture()
def sample_entities() -> list[Entity]:
    return [
        Entity(name="Alice", entity_type="Person"),
        Entity(name="ACME", entity_type="Company"),
    ]


def test_document_loader_initialization_defaults() -> None:
    loader = DocumentLoader()
    assert loader.strip_images is True


def test_document_loader_load_converts_and_strips_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_markitdown_module,
) -> None:
    dir_path = tmp_path / "docs"
    dir_path.mkdir()
    file_path = dir_path / "sample.docx"
    file_path.write_text("dummy", encoding="utf-8")

    import src.data_processing.document_loader as document_loader

    monkeypatch.setattr(
        document_loader.importlib, "import_module", lambda name: fake_markitdown_module
    )

    loader = DocumentLoader(strip_images=True)
    document = loader.load(file_path)

    assert document.source_path == file_path
    assert document.content == "Hello [图片] World"
    assert document.metadata["filename"] == "sample.docx"
    assert document.metadata["file_size"] == file_path.stat().st_size
    assert isinstance(document.metadata["modified_time"], str)


def test_document_loader_load_rejects_non_docx(tmp_path: Path) -> None:
    dir_path = tmp_path / "docs"
    dir_path.mkdir()
    file_path = dir_path / "sample.txt"
    file_path.write_text("dummy", encoding="utf-8")

    loader = DocumentLoader()
    with pytest.raises(ValidationError, match="Unsupported file format"):
        loader.load(file_path)


def test_document_loader_load_directory_collects_docs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_markitdown_module,
) -> None:
    dir_path = tmp_path / "docs"
    dir_path.mkdir()
    first = dir_path / "a.docx"
    second = dir_path / "b.docx"
    first.write_text("dummy", encoding="utf-8")
    second.write_text("dummy", encoding="utf-8")

    import src.data_processing.document_loader as document_loader

    monkeypatch.setattr(
        document_loader.importlib, "import_module", lambda name: fake_markitdown_module
    )

    loader = DocumentLoader()
    documents = loader.load_directory(dir_path)

    assert [doc.source_path.name for doc in documents] == ["a.docx", "b.docx"]


def test_document_loader_load_directory_errors(tmp_path: Path) -> None:
    loader = DocumentLoader()
    missing = tmp_path / "missing"

    with pytest.raises(NotFoundError, match="Directory not found"):
        loader.load_directory(missing)

    file_path = tmp_path / "file.docx"
    file_path.write_text("dummy", encoding="utf-8")
    with pytest.raises(ValidationError, match="Path is not a directory"):
        loader.load_directory(file_path)


# ── Document chunking / splitting tests ──────────────────────────────────


def test_split_sections_single_newline_fallback() -> None:
    """Documents with only \n (no \n\n) should still split by line."""
    loader = DocumentLoader()
    # 6 lines * ~150 chars each = ~900 chars, well under 500 max_chunk_size
    text = "\n".join([f"第{i+1}段：" + "技术描述内容" * 15 for i in range(6)])
    sections = loader.split_into_sections(text, max_chunk_size=500)
    assert len(sections) > 1, "Should split on \\n when \\n\\n is absent"
    for s in sections:
        assert len(s.content) <= 500


def test_split_sections_double_newline_preferred() -> None:
    """Documents with \n\n should split on double newlines, not single."""
    loader = DocumentLoader()
    text = "\n\n".join([f"段落{i+1}。" + "填充内容。" * 30 for i in range(5)])
    sections = loader.split_into_sections(text, max_chunk_size=500)
    assert len(sections) >= 2
    for s in sections:
        assert len(s.content) <= 500


def test_split_sections_no_newlines_falls_to_chars() -> None:
    """Text with no newlines should still split by sentence/chars."""
    loader = DocumentLoader()
    text = "这是一段没有任何换行的长文本。" * 100  # ~1200 chars
    sections = loader.split_into_sections(text, max_chunk_size=500)
    assert len(sections) >= 2
    for s in sections:
        assert len(s.content) <= 500


def test_split_sections_heading_based() -> None:
    """Markdown headings should be primary split points."""
    loader = DocumentLoader()
    text = "# 第一章\n内容A。内容A。\n\n# 第二章\n内容B。内容B。"
    sections = loader.split_into_sections(text, max_chunk_size=1500)
    assert len(sections) == 2
    assert "第一章" in sections[0].heading_chain
    assert "第二章" in sections[1].heading_chain


def test_split_sections_large_single_newline_doc() -> None:
    """Regression: large doc with only \n should produce multiple sections."""
    loader = DocumentLoader()
    lines = [f"技术术语{i}: " + "补充描述" * 20 for i in range(30)]
    text = "\n".join(lines)
    assert "\n\n" not in text
    sections = loader.split_into_sections(text, max_chunk_size=1500)
    assert len(sections) >= 2, f"Expected >=2 sections, got {len(sections)}"
    total_chars = sum(len(s.content) for s in sections)
    # All content preserved (minus whitespace from stripping)
    assert total_chars >= len(text) * 0.9

def test_entity_dataclass_defaults() -> None:
    entity = Entity(name="Alice", entity_type="Person")
    assert entity.aliases == []
    assert entity.properties == {}
    assert entity.source_span is None


def test_entity_parse_response_valid(fake_llm_client_factory) -> None:
    extractor = EntityExtractor(fake_llm_client_factory("[]"))
    response = (
        '[{"name": " Alice ", "type": "Person", "aliases": ["A", 1], '
        '"properties": {"age": 30}}, '
        '{"name": 123, "type": "Bad"}]'
    )

    entities = extractor._parse_response(response)

    assert len(entities) == 1
    assert entities[0].name == "Alice"
    assert entities[0].entity_type == "Person"
    assert entities[0].aliases == ["A"]
    assert entities[0].properties == {"age": 30}


def test_entity_parse_response_invalid_json(fake_llm_client_factory) -> None:
    extractor = EntityExtractor(fake_llm_client_factory("[]"))
    assert extractor._parse_response("not json") == []


@pytest.mark.asyncio
async def test_entity_extract_section_calls_llm(fake_llm_client_factory) -> None:
    response = '[{"name": "Alice", "type": "Person"}]'
    llm_client = fake_llm_client_factory(response)
    extractor = EntityExtractor(llm_client)
    section = Section(
        content="Alice works at ACME.", heading_chain=[], level=0, index=0
    )

    entities = await extractor._extract_section(section, [])

    assert llm_client.call_count == 1
    assert llm_client.last_prompt is not None
    assert "Alice works at ACME." in llm_client.last_prompt
    assert len(entities) == 1
    assert entities[0].name == "Alice"


@pytest.mark.asyncio
async def test_entity_extract_empty_sections_returns_empty(
    fake_llm_client_factory,
) -> None:
    llm_client = fake_llm_client_factory("[]")
    extractor = EntityExtractor(llm_client)

    entities = await extractor.extract([])

    assert entities == []
    assert llm_client.call_count == 0


def test_triple_dataclass_defaults() -> None:
    triple = Triple(subject="A", predicate="rel", object="B")
    assert triple.confidence == 1.0
    assert triple.properties == {}
    assert triple.source is None


def test_triple_parse_response_valid(fake_llm_client_factory) -> None:
    extractor = TripleExtractor(fake_llm_client_factory("[]"))
    response = (
        '[{"subject": " A ", "predicate": "rel", "object": " B ", '
        '"confidence": "high", "properties": {"key": "value"}}, '
        '{"subject": 1, "predicate": "bad", "object": "skip"}]'
    )

    triples = extractor._parse_response(response)

    assert len(triples) == 1
    assert triples[0].subject == "A"
    assert triples[0].predicate == "rel"
    assert triples[0].object == "B"
    assert triples[0].confidence == 1.0
    assert triples[0].properties == {"key": "value"}


def test_triple_parse_response_invalid_json(fake_llm_client_factory) -> None:
    extractor = TripleExtractor(fake_llm_client_factory("[]"))
    assert extractor._parse_response("not json") == []


@pytest.mark.asyncio
async def test_triple_extract_section_calls_llm(
    fake_llm_client_factory, sample_entities
) -> None:
    response = '[{"subject": "Alice", "predicate": "works_at", "object": "ACME"}]'
    llm_client = fake_llm_client_factory(response)
    extractor = TripleExtractor(llm_client)
    section = Section(
        content="Alice works at ACME.", heading_chain=[], level=0, index=0
    )

    entity_names = {e.name for e in sample_entities}
    triples = await extractor._extract_section(section, sample_entities, entity_names)

    assert llm_client.call_count == 1
    assert llm_client.last_prompt is not None
    assert "Alice works at ACME." in llm_client.last_prompt
    assert "Alice" in llm_client.last_prompt
    assert "ACME" in llm_client.last_prompt
    assert len(triples) == 1
    assert triples[0].predicate == "works_at"


@pytest.mark.asyncio
async def test_triple_extract_requires_entities(fake_llm_client_factory) -> None:
    llm_client = fake_llm_client_factory("[]")
    extractor = TripleExtractor(llm_client)
    section = Section(
        content="Alice works at ACME.", heading_chain=[], level=0, index=0
    )

    triples = await extractor.extract([section], [])

    assert triples == []
    assert llm_client.call_count == 0


# ── Thinking model parsing tests ──────────────────────────────────────────


def test_entity_parse_response_markdown_code_block(fake_llm_client_factory) -> None:
    """Thinking models often wrap JSON in markdown code blocks."""
    extractor = EntityExtractor(fake_llm_client_factory("[]"))
    response = (
        "让我分析一下这段文本中的实体：\n\n"
        "1. 首先识别人名\n"
        "2. 然后识别组织\n\n"
        "```json\n"
        '[{"name": "Alice", "type": "Person"}, '
        '{"name": "ACME", "type": "Company"}]\n'
        "```"
    )
    entities = extractor._parse_response(response)
    assert len(entities) == 2
    assert entities[0].name == "Alice"
    assert entities[1].name == "ACME"


def test_entity_parse_response_think_tags(fake_llm_client_factory) -> None:
    """Strip <think>...</think> tags from DeepSeek-R1 style responses."""
    extractor = EntityExtractor(fake_llm_client_factory("[]"))
    response = (
        "<think>这段文本包含技术术语，我需要识别关键实体。"
        "WDM是波分复用技术，DWDM是密集波分复用。</think>"
        '[{"name": "WDM", "type": "技术"},'
        ' {"name": "DWDM", "type": "技术"}]'
    )
    entities = extractor._parse_response(response)
    assert len(entities) == 2
    assert entities[0].name == "WDM"
    assert entities[1].name == "DWDM"


def test_entity_parse_response_think_tags_with_code_block(
    fake_llm_client_factory,
) -> None:
    """Handle both <think> tags and markdown code blocks together."""
    extractor = EntityExtractor(fake_llm_client_factory("[]"))
    response = (
        "<think>分析文本内容...</think>\n"
        "```json\n"
        '[{"name": "TCP", "type": "协议"}]\n'
        "```"
    )
    entities = extractor._parse_response(response)
    assert len(entities) == 1
    assert entities[0].name == "TCP"


def test_entity_parse_response_json_in_thinking_text(
    fake_llm_client_factory,
) -> None:
    """JSON embedded at the end of chain-of-thought text (no code block)."""
    extractor = EntityExtractor(fake_llm_client_factory("[]"))
    response = (
        "我来分析一下这段关于网络协议的文本。\n"
        "文本中提到了HTTP和TCP两个协议。\n"
        "以下是提取的实体列表：\n"
        '[{"name": "HTTP", "type": "协议"}, '
        '{"name": "TCP", "type": "协议"}]'
    )
    entities = extractor._parse_response(response)
    assert len(entities) == 2
    assert entities[0].name == "HTTP"
    assert entities[1].name == "TCP"


def test_triple_parse_response_markdown_code_block(
    fake_llm_client_factory,
) -> None:
    """Thinking models often wrap JSON in markdown code blocks."""
    extractor = TripleExtractor(fake_llm_client_factory("[]"))
    response = (
        "分析实体之间的关系：\n\n"
        "```json\n"
        '[{"subject": "Alice", "predicate": "works_at", "object": "ACME"}]\n'
        "```"
    )
    triples = extractor._parse_response(response)
    assert len(triples) == 1
    assert triples[0].subject == "Alice"
    assert triples[0].predicate == "works_at"
    assert triples[0].object == "ACME"


def test_triple_parse_response_think_tags(fake_llm_client_factory) -> None:
    """Strip <think>...</think> tags from DeepSeek-R1 style responses."""
    extractor = TripleExtractor(fake_llm_client_factory("[]"))
    response = (
        "<think>这两个实体之间存在从属关系。</think>"
        '[{"subject": "DWDM", "predicate": "属于", "object": "WDM"}]'
    )
    triples = extractor._parse_response(response)
    assert len(triples) == 1
    assert triples[0].subject == "DWDM"
    assert triples[0].predicate == "属于"
    assert triples[0].object == "WDM"


def test_triple_parse_response_json_in_thinking_text(
    fake_llm_client_factory,
) -> None:
    """JSON embedded at the end of chain-of-thought text (no code block)."""
    extractor = TripleExtractor(fake_llm_client_factory("[]"))
    response = (
        "文本提到Alice在ACME工作。\n"
        "关系提取结果：\n"
        '[{"subject": "Alice", "predicate": "works_at", "object": "ACME"}]'
    )
    triples = extractor._parse_response(response)
    assert len(triples) == 1
    assert triples[0].predicate == "works_at"


def test_entity_parse_response_code_block_no_json_tag(
    fake_llm_client_factory,
) -> None:
    """Code block without 'json' language specifier."""
    extractor = EntityExtractor(fake_llm_client_factory("[]"))
    response = (
        "提取结果如下：\n"
        "```\n"
        '[{"name": "Neo4j", "type": "数据库"}]\n'
        "```"
    )
    entities = extractor._parse_response(response)
    assert len(entities) == 1
    assert entities[0].name == "Neo4j"


# ── Prompt Quality Tests ──────────────────────────────────────────────────


class TestEntityPromptQuality:
    """Verify the entity extraction prompt is domain-agnostic and thorough."""

    def test_prompt_includes_process_and_mechanism_types(
        self, fake_llm_client_factory
    ) -> None:
        extractor = EntityExtractor(fake_llm_client_factory("[]"))
        section = Section(
            content="高血压损伤血管内皮",
            heading_chain=[], level=0, index=0,
        )
        prompt = extractor._build_prompt(section, [])
        # Should mention abstract concepts / processes
        assert "过程" in prompt
        assert "机制" in prompt
        assert "现象" in prompt

    def test_prompt_includes_causal_chain_guidance(
        self, fake_llm_client_factory
    ) -> None:
        extractor = EntityExtractor(fake_llm_client_factory("[]"))
        section = Section(content="test", heading_chain=[], level=0, index=0)
        prompt = extractor._build_prompt(section, [])
        assert "因果链" in prompt
        assert "中间实体" in prompt or "中间节点" in prompt

    def test_prompt_uses_multidomain_examples(
        self, fake_llm_client_factory
    ) -> None:
        extractor = EntityExtractor(fake_llm_client_factory("[]"))
        section = Section(content="test", heading_chain=[], level=0, index=0)
        prompt = extractor._build_prompt(section, [])
        # Medical domain example
        assert "动脉粥样硬化" in prompt
        # Tech domain example
        assert "分布式数据库" in prompt or "光伏发电" in prompt

    def test_prompt_no_telecom_only_examples(
        self, fake_llm_client_factory
    ) -> None:
        """Prompt should NOT be biased toward telecom domain."""
        extractor = EntityExtractor(fake_llm_client_factory("[]"))
        section = Section(content="test", heading_chain=[], level=0, index=0)
        prompt = extractor._build_prompt(section, [])
        # Should NOT contain telecom-specific examples as the ONLY examples
        assert "STM-1" not in prompt

    def test_prompt_includes_entity_types_hint(
        self, fake_llm_client_factory
    ) -> None:
        extractor = EntityExtractor(
            fake_llm_client_factory("[]"),
            entity_types=["疾病", "药物"],
        )
        section = Section(content="test", heading_chain=[], level=0, index=0)
        prompt = extractor._build_prompt(section, [])
        assert "疾病" in prompt
        assert "药物" in prompt

    def test_prompt_includes_heading_context(
        self, fake_llm_client_factory
    ) -> None:
        extractor = EntityExtractor(fake_llm_client_factory("[]"))
        section = Section(
            content="content",
            heading_chain=["第一章", "心血管疾病"],
            level=0, index=0,
        )
        prompt = extractor._build_prompt(section, [])
        assert "第一章" in prompt
        assert "心血管疾病" in prompt

    def test_prompt_includes_context_entities(
        self, fake_llm_client_factory
    ) -> None:
        extractor = EntityExtractor(fake_llm_client_factory("[]"))
        section = Section(content="test", heading_chain=[], level=0, index=0)
        existing = [Entity(name="高血压", entity_type="疾病")]
        prompt = extractor._build_prompt(section, existing)
        assert "高血压" in prompt
        assert "已知实体" in prompt or "避免重复" in prompt


class TestTriplePromptQuality:
    """Verify the triple extraction prompt guides toward specific relations."""

    def test_prompt_discourages_generic_relations(
        self, fake_llm_client_factory
    ) -> None:
        extractor = TripleExtractor(fake_llm_client_factory("[]"))
        entities = [Entity(name="A", entity_type="T")]
        section = Section(content="test", heading_chain=[], level=0, index=0)
        prompt = extractor._build_prompt(section, entities)
        assert "相关" in prompt  # mentioned as example of what to avoid
        assert "避免" in prompt or "模糊" in prompt

    def test_prompt_suggests_causal_relation_types(
        self, fake_llm_client_factory
    ) -> None:
        extractor = TripleExtractor(fake_llm_client_factory("[]"))
        entities = [Entity(name="X", entity_type="T")]
        section = Section(content="test", heading_chain=[], level=0, index=0)
        prompt = extractor._build_prompt(section, entities)
        assert "导致" in prompt
        assert "损伤" in prompt
        assert "诱发" in prompt

    def test_prompt_allows_new_entity_discovery(
        self, fake_llm_client_factory
    ) -> None:
        """Prompt should NOT say 'only use entities from the list'."""
        extractor = TripleExtractor(fake_llm_client_factory("[]"))
        entities = [Entity(name="X", entity_type="T")]
        section = Section(content="test", heading_chain=[], level=0, index=0)
        prompt = extractor._build_prompt(section, entities)
        # Should say 'prioritize' not 'only use'
        assert "优先" in prompt
        # Should mention supplementing missing entities
        assert "遗漏" in prompt or "补充" in prompt

    def test_prompt_uses_multidomain_examples(
        self, fake_llm_client_factory
    ) -> None:
        extractor = TripleExtractor(fake_llm_client_factory("[]"))
        entities = [Entity(name="X", entity_type="T")]
        section = Section(content="test", heading_chain=[], level=0, index=0)
        prompt = extractor._build_prompt(section, entities)
        # Medical example
        assert "高血压" in prompt
        assert "血管内皮" in prompt
        # Tech example
        assert "分布式数据库" in prompt or "CAP定理" in prompt

    def test_prompt_warns_against_type_in_entity_names(
        self, fake_llm_client_factory
    ) -> None:
        """Prompt should tell LLM not to include type in entity names."""
        extractor = TripleExtractor(fake_llm_client_factory("[]"))
        entities = [Entity(name="X", entity_type="T")]
        section = Section(content="test", heading_chain=[], level=0, index=0)
        prompt = extractor._build_prompt(section, entities)
        assert "括号" in prompt or "不要加" in prompt or "不要带" in prompt


class TestTripleFilterRelaxed:
    """Verify _filter_triples allows new entities but blocks self-loops."""

    def test_keeps_triples_with_new_entities(
        self, fake_llm_client_factory
    ) -> None:
        """Triples with entities not in the known list should be kept."""
        extractor = TripleExtractor(fake_llm_client_factory("[]"))
        triples = [
            Triple(
                subject="高血压",
                predicate="损伤",
                object="血管内皮",  # NOT in entity_names
            )
        ]
        result = extractor._filter_triples(triples, {"高血压"})
        assert len(result) == 1
        assert result[0].object == "血管内皮"

    def test_filters_self_loops(
        self, fake_llm_client_factory
    ) -> None:
        extractor = TripleExtractor(fake_llm_client_factory("[]"))
        triples = [
            Triple(subject="A", predicate="导致", object="A")
        ]
        result = extractor._filter_triples(triples, {"A"})
        assert len(result) == 0

    def test_normalizes_entity_names_in_filter(
        self, fake_llm_client_factory
    ) -> None:
        """Type annotations should be stripped from entity names."""
        extractor = TripleExtractor(fake_llm_client_factory("[]"))
        triples = [
            Triple(
                subject="高血压（疾病）",
                predicate="损伤",
                object="血管内皮(组织)",
            )
        ]
        result = extractor._filter_triples(triples, {"高血压"})
        assert len(result) == 1
        assert result[0].subject == "高血压"
        assert result[0].object == "血管内皮"

    def test_filters_invalid_relation_types_when_defined(
        self, fake_llm_client_factory
    ) -> None:
        extractor = TripleExtractor(
            fake_llm_client_factory("[]"),
            relation_types=["导致", "包含"],
        )
        triples = [
            Triple(subject="A", predicate="导致", object="B"),
            Triple(subject="C", predicate="相关", object="D"),  # invalid
        ]
        result = extractor._filter_triples(triples, {"A", "B", "C", "D"})
        assert len(result) == 1
        assert result[0].predicate == "导致"

    def test_no_relation_type_filter_when_undefined(
        self, fake_llm_client_factory
    ) -> None:
        """When no relation_types defined, all predicates should pass."""
        extractor = TripleExtractor(fake_llm_client_factory("[]"))
        triples = [
            Triple(subject="A", predicate="任意关系", object="B"),
        ]
        result = extractor._filter_triples(triples, {"A", "B"})
        assert len(result) == 1

    def test_self_loop_after_normalization(
        self, fake_llm_client_factory
    ) -> None:
        """Self-loop check should happen after name normalization."""
        extractor = TripleExtractor(fake_llm_client_factory("[]"))
        triples = [
            Triple(
                subject="X（技术）",
                predicate="依赖",
                object="X",  # same entity after normalization
            )
        ]
        result = extractor._filter_triples(triples, {"X"})
        assert len(result) == 0


# ── RelationType tests ────────────────────────────────────────────────


class TestRelationType:
    def test_to_prompt_block_format(self) -> None:
        rt = RelationType(
            name="test",
            label="测试关系",
            definition="A测试B",
            examples=["例1", "例2"],
            direction_hint="A→B",
        )
        block = rt.to_prompt_block()
        assert "【测试关系】" in block
        assert "(test)" in block
        assert "A测试B" in block
        assert "A→B" in block
        assert "例1；例2" in block

    def test_to_prompt_block_minimal(self) -> None:
        rt = RelationType(name="x", label="X", definition="def")
        block = rt.to_prompt_block()
        assert "【X】" in block
        assert "方向" not in block
        assert "示例" not in block

    def test_default_relation_types_count(self) -> None:
        assert len(DEFAULT_RELATION_TYPES) >= 5

    def test_get_relation_type(self) -> None:
        rt = get_relation_type("include")
        assert rt is not None
        assert rt.name == "include"
        assert rt.label == "包含关系"

    def test_get_relation_type_missing(self) -> None:
        assert get_relation_type("nonexistent") is None

    def test_build_relation_type_prompt(self) -> None:
        prompt = build_relation_type_prompt()
        assert "关系类型定义" in prompt
        assert "include" in prompt
        assert "correlate" in prompt
        assert "derivative" in prompt
        assert "coordinate" in prompt
        assert "symbiotic" in prompt

    def test_build_relation_type_prompt_custom(self) -> None:
        custom = [RelationType(name="custom", label="自定义", definition="自定义关系")]
        prompt = build_relation_type_prompt(custom)
        assert "custom" in prompt
        assert "自定义" in prompt
        assert "include" not in prompt

    def test_register_relation_type(self) -> None:
        rt = RelationType(
            name="test_register",
            label="测试注册",
            definition="临时注册的关系",
        )
        register_relation_type(rt)
        assert get_relation_type("test_register") is rt


# ── EntityMerger tests ────────────────────────────────────────────────


class TestEntityMerger:
    def test_empty_list(self) -> None:
        merger = EntityMerger()
        assert merger.merge([]) == []

    def test_no_duplicates(self) -> None:
        entities = [
            Entity(name="A", entity_type="Type1"),
            Entity(name="B", entity_type="Type2"),
        ]
        result = merger_merge(entities)
        assert len(result) == 2

    def test_exact_name_dedup(self) -> None:
        entities = [
            Entity(name="高血压", entity_type="疾病"),
            Entity(name="高血压", entity_type="疾病"),
        ]
        result = merger_merge(entities)
        assert len(result) == 1
        assert result[0].name == "高血压"

    def test_case_insensitive_dedup(self) -> None:
        entities = [
            Entity(name="TCP", entity_type="协议"),
            Entity(name="tcp", entity_type="协议"),
        ]
        result = merger_merge(entities)
        assert len(result) == 1

    def test_synonym_group_merge(self) -> None:
        config = MergeConfig(
            synonym_groups=[["高血压", "HBP", "hypertension"]]
        )
        merger = EntityMerger(config=config)
        entities = [
            Entity(name="高血压", entity_type="疾病"),
            Entity(name="HBP", entity_type="疾病"),
        ]
        result = merger.merge(entities)
        assert len(result) == 1

    def test_alias_collected_on_merge(self) -> None:
        entities = [
            Entity(name="动脉粥样硬化", entity_type="疾病"),
            Entity(name="动脉粥样硬化", entity_type="病变", aliases=["AS"]),
        ]
        result = merger_merge(entities)
        assert len(result) == 1
        # The shorter-name entity's name should be in aliases
        assert "AS" in result[0].aliases

    def test_properties_merged_first_wins(self) -> None:
        entities = [
            Entity(name="X", entity_type="T", properties={"a": 1}),
            Entity(name="X", entity_type="T", properties={"a": 2, "b": 3}),
        ]
        result = merger_merge(entities)
        assert len(result) == 1
        assert result[0].properties["a"] == 1
        assert result[0].properties["b"] == 3

    def test_cross_group_similarity_merge(self) -> None:
        """Entities with high bigram similarity should merge."""
        config = MergeConfig(similarity_threshold=0.5)
        merger = EntityMerger(config=config)
        entities = [
            Entity(name="动脉粥样硬化", entity_type="疾病"),
            Entity(name="动脉粥样硬化症", entity_type="疾病"),
        ]
        result = merger.merge(entities)
        # Should merge due to high bigram similarity
        assert len(result) == 1

    def test_dissimilar_entities_not_merged(self) -> None:
        config = MergeConfig(similarity_threshold=0.9)
        merger = EntityMerger(config=config)
        entities = [
            Entity(name="高血压", entity_type="疾病"),
            Entity(name="糖尿病", entity_type="疾病"),
        ]
        result = merger.merge(entities)
        assert len(result) == 2

    def test_jaccard_bigram_empty(self) -> None:
        assert EntityMerger._jaccard_bigram("", "abc") == 0.0
        assert EntityMerger._jaccard_bigram("abc", "") == 0.0

    def test_jaccard_bigram_identical(self) -> None:
        assert EntityMerger._jaccard_bigram("hello", "hello") == 1.0

    def test_jaccard_bigram_single_char(self) -> None:
        assert EntityMerger._jaccard_bigram("a", "a") == 1.0
        assert EntityMerger._jaccard_bigram("a", "b") == 0.0


def merger_merge(entities: list[Entity]) -> list[Entity]:
    """Helper: merge with default config."""
    return EntityMerger().merge(entities)


# ── Incremental extraction tests ──────────────────────────────────────


class TestIncrementalRelation:
    def test_dataclass_defaults(self) -> None:
        rel = IncrementalRelation(
            subject="A", predicate="导致", object="B"
        )
        assert rel.relation_type == ""
        assert rel.confidence == 1.0

    def test_dataclass_full(self) -> None:
        rel = IncrementalRelation(
            subject="高血压",
            predicate="损伤",
            object="血管内皮",
            relation_type="correlate",
            confidence=0.95,
        )
        assert rel.subject == "高血压"
        assert rel.relation_type == "correlate"


class TestIncrementalExtraction:
    def test_parse_incremental_response_valid(
        self, fake_llm_client_factory
    ) -> None:
        response = json.dumps({
            "new_entities": [
                {"name": "高血压", "type": "疾病", "aliases": []},
                {"name": "血管内皮", "type": "组织", "aliases": ["内皮"]},
            ],
            "relations": [
                {
                    "subject": "高血压",
                    "predicate": "损伤",
                    "object": "血管内皮",
                    "relation_type": "correlate",
                    "confidence": 0.95,
                }
            ],
        })
        extractor = EntityExtractor(fake_llm_client_factory(response))
        entities, relations = extractor._parse_incremental_response(response)
        assert len(entities) == 2
        assert entities[0].name == "高血压"
        assert entities[1].aliases == ["内皮"]
        assert len(relations) == 1
        assert relations[0].subject == "高血压"
        assert relations[0].relation_type == "correlate"
        assert relations[0].confidence == 0.95

    def test_parse_incremental_response_markdown_block(
        self, fake_llm_client_factory
    ) -> None:
        inner = json.dumps({
            "new_entities": [{"name": "X", "type": "T"}],
            "relations": [],
        })
        response = f"```json\n{inner}\n```"
        extractor = EntityExtractor(fake_llm_client_factory(response))
        entities, relations = extractor._parse_incremental_response(response)
        assert len(entities) == 1
        assert len(relations) == 0

    def test_parse_incremental_response_fallback_to_entity_only(
        self, fake_llm_client_factory
    ) -> None:
        """When response is an array (not object), fall back to entity parse."""
        response = json.dumps([{"name": "X", "type": "T"}])
        extractor = EntityExtractor(fake_llm_client_factory(response))
        entities, relations = extractor._parse_incremental_response(response)
        assert len(entities) == 1
        assert len(relations) == 0

    def test_parse_incremental_response_with_think_tags(
        self, fake_llm_client_factory
    ) -> None:
        inner = json.dumps({
            "new_entities": [{"name": "A", "type": "T"}],
            "relations": [
                {"subject": "A", "predicate": "p", "object": "B"},
            ],
        })
        response = f"<think>thinking...</think>{inner}"
        extractor = EntityExtractor(fake_llm_client_factory(response))
        entities, relations = extractor._parse_incremental_response(response)
        assert len(entities) == 1
        assert len(relations) == 1

    def test_parse_incremental_response_invalid_entities_skipped(
        self, fake_llm_client_factory
    ) -> None:
        response = json.dumps({
            "new_entities": [
                {"name": "", "type": "T"},  # empty name
                {"name": 123, "type": "T"},  # non-str name
                {"name": "valid", "type": "T"},  # good
            ],
            "relations": [
                {"subject": 123, "predicate": "p", "object": "B"},  # bad
                {"subject": "A", "predicate": "p", "object": "B"},  # good
            ],
        })
        extractor = EntityExtractor(fake_llm_client_factory(response))
        entities, relations = extractor._parse_incremental_response(response)
        assert len(entities) == 1
        assert entities[0].name == "valid"
        assert len(relations) == 1

    @pytest.mark.asyncio()
    async def test_extract_incremental_accumulates_context(
        self, fake_llm_client_factory
    ) -> None:
        """Entities from section 0 should appear as context in section 1's prompt."""
        section_0_resp = json.dumps({
            "new_entities": [{"name": "E0", "type": "T"}],
            "relations": [],
        })
        section_1_resp = json.dumps({
            "new_entities": [{"name": "E1", "type": "T"}],
            "relations": [
                {"subject": "E0", "predicate": "rel", "object": "E1"},
            ],
        })

        call_idx = 0
        prompts: list[str] = []

        class FakeClient:
            async def generate(self, prompt: str, params=None) -> str:
                nonlocal call_idx
                prompts.append(prompt)
                resp = section_0_resp if call_idx == 0 else section_1_resp
                call_idx += 1
                return resp

        sections = [
            Section(content="text0", heading_chain=[], level=0, index=0),
            Section(content="text1", heading_chain=[], level=0, index=1),
        ]

        extractor = EntityExtractor(FakeClient())  # type: ignore[arg-type]
        entities, relations = await extractor.extract_incremental(sections)

        # E0 should appear as context entity in second call
        assert len(prompts) == 2
        assert "E0" in prompts[1]
        assert "已知实体" in prompts[1]

        assert len(entities) == 2
        assert len(relations) == 1

    def test_build_incremental_prompt_includes_relation_types(
        self, fake_llm_client_factory
    ) -> None:
        extractor = EntityExtractor(fake_llm_client_factory(""))
        section = Section(content="test text", heading_chain=[], level=0, index=0)
        prompt = extractor._build_incremental_prompt(
            section, [], DEFAULT_RELATION_TYPES
        )
        assert "关系类型定义" in prompt
        assert "include" in prompt
        assert "correlate" in prompt
        assert "new_entities" in prompt
        assert "relation_type" in prompt

    def test_build_incremental_prompt_includes_context(
        self, fake_llm_client_factory
    ) -> None:
        extractor = EntityExtractor(fake_llm_client_factory(""))
        section = Section(
            content="text", heading_chain=["第一章", "概述"],
            level=1, index=0,
        )
        context = [Entity(name="已有实体", entity_type="概念")]
        prompt = extractor._build_incremental_prompt(
            section, context, DEFAULT_RELATION_TYPES
        )
        assert "第一章 > 概述" in prompt
        assert "已有实体" in prompt
        assert "已知实体" in prompt


# ── Triple relation_type tests ────────────────────────────────────────


class TestTripleRelationType:
    def test_triple_with_relation_type(self) -> None:
        t = Triple(
            subject="A",
            predicate="包含",
            object="B",
            relation_type="include",
        )
        assert t.relation_type == "include"

    def test_triple_relation_type_default(self) -> None:
        t = Triple(subject="A", predicate="p", object="B")
        assert t.relation_type == ""

    def test_parse_response_includes_relation_type(
        self, fake_llm_client_factory
    ) -> None:
        response = json.dumps([
            {
                "subject": "高血压",
                "predicate": "损伤",
                "object": "血管",
                "relation_type": "correlate",
                "confidence": 0.9,
            }
        ])
        extractor = TripleExtractor(fake_llm_client_factory(response))
        triples = extractor._parse_response(response)
        assert len(triples) == 1
        assert triples[0].relation_type == "correlate"

    def test_filter_validates_relation_type_field(
        self, fake_llm_client_factory
    ) -> None:
        """When structured types used, filter checks relation_type field."""
        extractor = TripleExtractor(
            fake_llm_client_factory("[]"),
            relation_types=[INCLUDE_EDGE, CORRELATE_EDGE],
        )
        triples = [
            Triple(
                subject="A",
                predicate="包含",
                object="B",
                relation_type="include",  # valid
            ),
            Triple(
                subject="C",
                predicate="导致",
                object="D",
                relation_type="invalid_type",  # invalid
            ),
        ]
        result = extractor._filter_triples(triples, {"A", "B", "C", "D"})
        assert len(result) == 1
        assert result[0].relation_type == "include"


class TestTripleStructuredPrompt:
    def test_prompt_with_structured_types(
        self, fake_llm_client_factory
    ) -> None:
        extractor = TripleExtractor(
            fake_llm_client_factory("[]"),
            relation_types=[INCLUDE_EDGE, CORRELATE_EDGE],
        )
        section = Section(content="test text", heading_chain=[], level=0, index=0)
        entities = [Entity(name="A", entity_type="T")]
        prompt = extractor._build_prompt(section, entities)
        assert "关系类型定义" in prompt
        assert "include" in prompt
        assert "correlate" in prompt
        assert "relation_type" in prompt

    def test_prompt_with_string_types_legacy(
        self, fake_llm_client_factory
    ) -> None:
        extractor = TripleExtractor(
            fake_llm_client_factory("[]"),
            relation_types=["包含", "导致"],
        )
        section = Section(content="test text", heading_chain=[], level=0, index=0)
        entities = [Entity(name="A", entity_type="T")]
        prompt = extractor._build_prompt(section, entities)
        assert "包含" in prompt
        assert "导致" in prompt
        assert "关系类型限定" in prompt

    def test_prompt_with_no_types(
        self, fake_llm_client_factory
    ) -> None:
        extractor = TripleExtractor(fake_llm_client_factory("[]"))
        section = Section(content="test text", heading_chain=[], level=0, index=0)
        entities = [Entity(name="A", entity_type="T")]
        prompt = extractor._build_prompt(section, entities)
        assert "推荐的关系类型" in prompt


class TestCrossSectionExtraction:
    @pytest.mark.asyncio()
    async def test_extract_cross_section_basic(
        self, fake_llm_client_factory
    ) -> None:
        response = json.dumps([
            {"subject": "A", "predicate": "影响", "object": "B"},
        ])
        extractor = TripleExtractor(fake_llm_client_factory(response))
        sections = [
            Section(content="text1 about A", heading_chain=[], level=0, index=0),
            Section(content="text2 about B", heading_chain=[], level=0, index=1),
        ]
        entities = [
            Entity(name="A", entity_type="T"),
            Entity(name="B", entity_type="T"),
        ]
        result = await extractor.extract_cross_section(sections, entities)
        assert len(result) == 1
        assert result[0].subject == "A"

    @pytest.mark.asyncio()
    async def test_extract_cross_section_single_section_empty(
        self, fake_llm_client_factory
    ) -> None:
        extractor = TripleExtractor(fake_llm_client_factory("[]"))
        sections = [Section(content="text", heading_chain=[], level=0, index=0)]
        entities = [Entity(name="A", entity_type="T")]
        result = await extractor.extract_cross_section(sections, entities)
        assert result == []

    @pytest.mark.asyncio()
    async def test_extract_cross_section_no_entities_empty(
        self, fake_llm_client_factory
    ) -> None:
        extractor = TripleExtractor(fake_llm_client_factory("[]"))
        sections = [
            Section(content="t1", heading_chain=[], level=0, index=0),
            Section(content="t2", heading_chain=[], level=0, index=1),
        ]
        result = await extractor.extract_cross_section(sections, [])
        assert result == []

    def test_cross_section_prompt_structure(
        self, fake_llm_client_factory
    ) -> None:
        extractor = TripleExtractor(
            fake_llm_client_factory("[]"),
            relation_types=[INCLUDE_EDGE],
        )
        sections = [
            Section(
                content="text1",
                heading_chain=["章1"],
                level=1,
                index=0,
            ),
            Section(content="text2", heading_chain=[], level=0, index=1),
        ]
        entities = [Entity(name="X", entity_type="T")]
        prompt = extractor._build_cross_section_prompt(sections, entities)
        assert "跨章节" in prompt
        assert "章1" in prompt
        assert "X" in prompt
        assert "include" in prompt


# ── QualityVerifier tests ──────────────────────────────────────────────


class TestQualityVerifier:
    def test_parse_report_valid(
        self, fake_llm_client_factory
    ) -> None:
        response = json.dumps({
            "quality_score": 0.85,
            "issues": ["遗漏了某实体"],
            "suggestions": ["建议补充关系"],
            "missing_entities": ["实体A"],
            "missing_relations": ["A→导致→B"],
        })
        verifier = QualityVerifier(fake_llm_client_factory(response))
        report = verifier._parse_report(response)
        assert report.quality_score == 0.85
        assert len(report.issues) == 1
        assert len(report.suggestions) == 1
        assert "实体A" in report.missing_entities

    def test_parse_report_clamped_score(
        self, fake_llm_client_factory
    ) -> None:
        response = json.dumps({"quality_score": 1.5})
        verifier = QualityVerifier(fake_llm_client_factory(response))
        report = verifier._parse_report(response)
        assert report.quality_score == 1.0

    def test_parse_report_invalid_json(
        self, fake_llm_client_factory
    ) -> None:
        verifier = QualityVerifier(fake_llm_client_factory("garbage"))
        report = verifier._parse_report("garbage")
        assert report.quality_score == 0.0
        assert len(report.issues) > 0

    def test_parse_report_with_think_tags(
        self, fake_llm_client_factory
    ) -> None:
        inner = json.dumps({"quality_score": 0.7, "issues": []})
        response = f"<think>blah</think>{inner}"
        verifier = QualityVerifier(fake_llm_client_factory(response))
        report = verifier._parse_report(response)
        assert report.quality_score == 0.7

    @pytest.mark.asyncio()
    async def test_verify_success(
        self, fake_llm_client_factory
    ) -> None:
        response = json.dumps({
            "quality_score": 0.9,
            "issues": [],
            "suggestions": [],
            "missing_entities": [],
            "missing_relations": [],
        })
        verifier = QualityVerifier(fake_llm_client_factory(response))
        report = await verifier.verify(
            text="some text",
            entities=[Entity(name="A", entity_type="T")],
            triples=[Triple(subject="A", predicate="p", object="B")],
        )
        assert report.quality_score == 0.9

    @pytest.mark.asyncio()
    async def test_verify_llm_failure(self) -> None:
        class FailClient:
            async def generate(self, prompt: str, params=None) -> str:
                raise RuntimeError("LLM error")

        verifier = QualityVerifier(FailClient())  # type: ignore[arg-type]
        report = await verifier.verify(
            text="text",
            entities=[],
            triples=[],
        )
        assert report.quality_score == 0.0
        assert any("验证失败" in issue for issue in report.issues)
