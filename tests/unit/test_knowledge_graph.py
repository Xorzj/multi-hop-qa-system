import pytest

from src.knowledge_graph.cypher_builder import CypherBuilder
from src.knowledge_graph.schema import (
    DynamicGraphSchema,
    GraphSchema,
    NodeSchema,
    RelationSchema,
    validate_node,
    validate_relation,
)


@pytest.fixture()
def static_schema() -> GraphSchema:
    return GraphSchema(
        nodes={
            "Person": NodeSchema(
                label="Person",
                required_properties=["name"],
                optional_properties=["age"],
            )
        },
        relations={
            "KNOWS": RelationSchema(
                relation_type="KNOWS",
                source_labels=["Person"],
                target_labels=["Person"],
                properties=["since"],
            )
        },
    )


@pytest.fixture()
def dynamic_schema() -> DynamicGraphSchema:
    return DynamicGraphSchema()


@pytest.fixture()
def cypher_builder() -> CypherBuilder:
    return CypherBuilder()


def test_graph_schema_getters(static_schema: GraphSchema) -> None:
    assert static_schema.get_node_schema("Person") is not None
    assert static_schema.get_node_schema("Missing") is None
    assert static_schema.get_relation_schema("KNOWS") is not None
    assert static_schema.get_relation_schema("MISSING") is None


def test_dynamic_schema_records_node_label(dynamic_schema: DynamicGraphSchema) -> None:
    assert "Person" not in dynamic_schema.nodes
    node_schema = dynamic_schema.record_node_label("Person")
    assert node_schema.label == "Person"
    assert dynamic_schema.nodes["Person"] == node_schema


def test_dynamic_schema_records_relation_type(
    dynamic_schema: DynamicGraphSchema,
) -> None:
    assert "KNOWS" not in dynamic_schema.relations
    relation_schema = dynamic_schema.record_relation_type("KNOWS", "Person", "Person")
    assert relation_schema.relation_type == "KNOWS"
    assert "Person" in relation_schema.source_labels
    assert "Person" in relation_schema.target_labels
    assert dynamic_schema.relations["KNOWS"] == relation_schema


def test_validate_node_static_schema_success(static_schema: GraphSchema) -> None:
    assert validate_node("Person", {"name": "Ada", "age": 30}, static_schema)


def test_validate_node_static_schema_missing_required(
    static_schema: GraphSchema,
) -> None:
    assert not validate_node("Person", {"age": 30}, static_schema)


def test_validate_node_static_schema_invalid_property(
    static_schema: GraphSchema,
) -> None:
    assert not validate_node("Person", {"name": "Ada", "rank": 1}, static_schema)


def test_validate_node_static_schema_unknown_label(static_schema: GraphSchema) -> None:
    assert not validate_node("Unknown", {"name": "Ada"}, static_schema)


def test_validate_node_dynamic_schema_registers_label(
    dynamic_schema: DynamicGraphSchema,
) -> None:
    assert validate_node("Tool", {"any": "value"}, dynamic_schema)
    assert "Tool" in dynamic_schema.nodes


def test_validate_relation_static_schema_success(static_schema: GraphSchema) -> None:
    assert validate_relation("KNOWS", "Person", "Person", static_schema)


def test_validate_relation_static_schema_invalid_endpoints(
    static_schema: GraphSchema,
) -> None:
    assert not validate_relation("KNOWS", "Company", "Person", static_schema)


def test_validate_relation_static_schema_unknown_type(
    static_schema: GraphSchema,
) -> None:
    assert not validate_relation("MISSING", "Person", "Person", static_schema)


def test_validate_relation_dynamic_schema_registers_type(
    dynamic_schema: DynamicGraphSchema,
) -> None:
    assert validate_relation("BUILT_BY", "Tool", "Person", dynamic_schema)
    assert "BUILT_BY" in dynamic_schema.relations
    relation_schema = dynamic_schema.relations["BUILT_BY"]
    assert "Tool" in relation_schema.source_labels
    assert "Person" in relation_schema.target_labels


def test_find_node_with_label(cypher_builder: CypherBuilder) -> None:
    query = cypher_builder.find_node("Ada", label="Person")
    assert query.query == "MATCH (n:`Person` {name: $name}) RETURN n"
    assert query.parameters == {"name": "Ada"}


def test_find_node_without_label(cypher_builder: CypherBuilder) -> None:
    query = cypher_builder.find_node("Ada")
    assert query.query == "MATCH (n {name: $name}) RETURN n"
    assert query.parameters == {"name": "Ada"}


@pytest.mark.parametrize(
    ("direction", "pattern"),
    [
        ("out", "-[r:`KNOWS`]->"),
        ("in", "<-[r:`KNOWS`]-"),
        ("both", "-[r:`KNOWS`]-"),
    ],
)
def test_find_neighbors_direction(
    cypher_builder: CypherBuilder, direction: str, pattern: str
) -> None:
    query = cypher_builder.find_neighbors(
        "Ada", relation_type="KNOWS", direction=direction, limit=5
    )
    assert "MATCH (n {name: $name})" in query.query
    assert pattern in query.query
    assert "RETURN m, r, type(r) as rel_type" in query.query
    assert "startNode(r).name as rel_start" in query.query
    assert "endNode(r).name as rel_end" in query.query
    assert query.parameters == {"name": "Ada", "limit": 5}

def test_find_path(cypher_builder: CypherBuilder) -> None:
    query = cypher_builder.find_path("A", "B", max_hops=4)
    assert query.query == (
        "MATCH path = shortestPath((a {name: $start})-"
        "[*1..$max_hops]->(b {name: $end})) "
        "RETURN path"
    )
    assert query.parameters == {"start": "A", "end": "B", "max_hops": 4}

def test_find_by_property_escapes_property_name(
    cypher_builder: CypherBuilder,
) -> None:
    query = cypher_builder.find_by_property("Person", "na`me", "Ada")
    assert query.query == "MATCH (n:`Person`) WHERE n.`na``me` = $value RETURN n"
    assert query.parameters == {"value": "Ada"}


def test_get_node_properties(cypher_builder: CypherBuilder) -> None:
    query = cypher_builder.get_node_properties("Ada")
    assert query.query == "MATCH (n {name: $name}) RETURN properties(n) as properties"
    assert query.parameters == {"name": "Ada"}


def test_get_relation_properties(cypher_builder: CypherBuilder) -> None:
    query = cypher_builder.get_relation_properties("A", "B", "KNOWS")
    assert (
        query.query == "MATCH (a {name: $source})-[r:`KNOWS`]->(b {name: $target}) "
        "RETURN properties(r) as properties"
    )
    assert query.parameters == {"source": "A", "target": "B"}


def test_count_neighbors(cypher_builder: CypherBuilder) -> None:
    query = cypher_builder.count_neighbors("Ada", relation_type="KNOWS")
    assert (
        query.query
        == "MATCH (n {name: $name})-[r:`KNOWS`]-() RETURN count(r) as neighbor_count"
    )
    assert query.parameters == {"name": "Ada"}


# ===================== graph_retriever direction tests =====================


@pytest.mark.asyncio
class TestGraphRetrieverDirection:
    """Tests for correct edge direction handling in graph_retriever."""

    async def test_get_neighbors_uses_neo4j_direction_when_available(self) -> None:
        """When Cypher returns rel_start/rel_end, use them instead of inferring."""
        from unittest.mock import AsyncMock, MagicMock

        from src.knowledge_graph.graph_retriever import GraphRetriever

        mock_client = MagicMock()

        # Simulate Neo4j returning an incoming edge (OTN->SDH) when querying SDH
        mock_record = MagicMock()
        # Create relation mock with explicit start_node/end_node = None
        # (so _relation_endpoints uses the record's source/target instead)
        relation_mock = MagicMock()
        relation_mock.type = "依赖"
        relation_mock.start_node = None
        relation_mock.end_node = None

        mock_record.get.side_effect = lambda k: {
            "m": MagicMock(element_id="node:2", labels=frozenset({"Entity"})),
            "r": relation_mock,
            "rel_type": "依赖",
            "rel_start": "OTN",  # Real start of edge
            "rel_end": "SDH",  # Real end of edge
        }.get(k)
        mock_record.__getitem__ = mock_record.get

        # Configure properties for the neighbor node
        neighbor_node = mock_record.get("m")
        neighbor_node.__getitem__ = lambda self, k: "OTN" if k == "name" else None

        mock_client.execute = AsyncMock(return_value=[mock_record])

        retriever = GraphRetriever(mock_client)

        result = await retriever.get_neighbors("SDH", direction="both")
        # The relation should preserve the real direction: OTN -> SDH
        assert len(result.relations) == 1
        rel = result.relations[0]
        assert rel.source == "OTN", f"Expected source=OTN, got {rel.source}"
        assert rel.target == "SDH", f"Expected target=SDH, got {rel.target}"
