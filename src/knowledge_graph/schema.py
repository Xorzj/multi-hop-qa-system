"""Knowledge graph schema definitions and validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from src.common.logger import get_logger

logger = get_logger(__name__)


NodeLabel = str
RelationType = str


@dataclass(slots=True)
class NodeSchema:
    label: str
    required_properties: list[str]
    optional_properties: list[str]


@dataclass(slots=True)
class RelationSchema:
    relation_type: str
    source_labels: list[str]
    target_labels: list[str]
    properties: list[str]


@dataclass(slots=True)
class GraphSchema:
    nodes: dict[str, NodeSchema] = field(default_factory=dict)
    relations: dict[str, RelationSchema] = field(default_factory=dict)

    def get_node_schema(self, label: str) -> NodeSchema | None:
        return self.nodes.get(label)

    def get_relation_schema(self, relation_type: str) -> RelationSchema | None:
        return self.relations.get(relation_type)


@dataclass(slots=True)
class DynamicGraphSchema(GraphSchema):
    def record_node_label(self, label: str) -> NodeSchema:
        node_schema = self.nodes.get(label)
        if node_schema is None:
            node_schema = NodeSchema(
                label=label, required_properties=[], optional_properties=[]
            )
            self.nodes[label] = node_schema
        return node_schema

    def record_relation_type(
        self, relation_type: str, source_label: str, target_label: str
    ) -> RelationSchema:
        relation_schema = self.relations.get(relation_type)
        if relation_schema is None:
            relation_schema = RelationSchema(
                relation_type=relation_type,
                source_labels=[],
                target_labels=[],
                properties=[],
            )
            self.relations[relation_type] = relation_schema

        if source_label not in relation_schema.source_labels:
            relation_schema.source_labels.append(source_label)
        if target_label not in relation_schema.target_labels:
            relation_schema.target_labels.append(target_label)

        return relation_schema


def validate_node(
    label: str, properties: Mapping[str, object], schema: GraphSchema
) -> bool:
    if isinstance(schema, DynamicGraphSchema):
        schema.record_node_label(label)
        return True

    node_schema = schema.get_node_schema(label)
    if node_schema is None:
        logger.warning("Schema missing for node label", extra={"label": label})
        return False

    missing = [
        prop for prop in node_schema.required_properties if prop not in properties
    ]
    if missing:
        logger.info(
            "Node properties missing required fields",
            extra={"label": label, "missing": missing},
        )
        return False

    allowed = set(node_schema.required_properties + node_schema.optional_properties)
    invalid = [prop for prop in properties if prop not in allowed]
    if invalid:
        logger.info(
            "Node properties include unsupported fields",
            extra={"label": label, "invalid": invalid},
        )
        return False

    return True


def validate_relation(
    rel_type: str,
    source_label: str,
    target_label: str,
    schema: GraphSchema,
) -> bool:
    if isinstance(schema, DynamicGraphSchema):
        schema.record_relation_type(rel_type, source_label, target_label)
        return True

    relation_schema = schema.get_relation_schema(rel_type)
    if relation_schema is None:
        logger.warning(
            "Schema missing for relation type",
            extra={"relation_type": rel_type},
        )
        return False

    if (
        source_label not in relation_schema.source_labels
        or target_label not in relation_schema.target_labels
    ):
        logger.info(
            "Relation endpoints not allowed",
            extra={
                "relation_type": rel_type,
                "source_label": source_label,
                "target_label": target_label,
            },
        )
        return False

    return True
