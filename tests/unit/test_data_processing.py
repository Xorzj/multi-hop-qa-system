from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.common.exceptions import NotFoundError, ValidationError
from src.data_processing.document_loader import DocumentLoader, Section
from src.data_processing.entity_extractor import Entity, EntityExtractor
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
