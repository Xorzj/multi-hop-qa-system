from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.common.logger import get_logger
from src.data_processing.entity_extractor import Entity
from src.data_processing.triple_extractor import Triple
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.knowledge_graph.schema import (
    DynamicGraphSchema,
    GraphSchema,
    validate_node,
    validate_relation,
)

logger = get_logger(__name__)


@dataclass
class BuildStats:
    """Statistics returned after building the graph."""

    nodes_created: int = 0
    nodes_updated: int = 0
    relations_created: int = 0
    errors: list[str] = field(default_factory=list)


class GraphBuilder:
    """Build Neo4j graphs from extracted entities and triples."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        schema: GraphSchema | None = None,
        auto_create_missing_nodes: bool = True,
    ) -> None:
        """Initialize the graph builder with a Neo4j client and schema.

        Args:
            neo4j_client: The Neo4j client for database operations.
            schema: Optional schema for validation. Defaults to DynamicGraphSchema.
            auto_create_missing_nodes: If True, automatically create nodes referenced
                in triples but not in the entity list. Defaults to True.
        """
        self._neo4j_client = neo4j_client
        self._schema = schema or DynamicGraphSchema()
        self._auto_create_missing_nodes = auto_create_missing_nodes
    async def build_from_extraction(
        self,
        entities: list[Entity],
        triples: list[Triple],
        source_document: str | None = None,
    ) -> BuildStats:
        """Insert extracted entities and triples into Neo4j with schema validation."""

        stats = BuildStats()
        name_to_label: dict[str, str] = {}

        for entity in entities:
            name = entity.name.strip()
            if not name:
                stats.errors.append("Empty entity name encountered")
                logger.warning("Skipping entity with empty name")
                continue
            label = entity.entity_type.strip()
            properties = dict(entity.properties)
            if entity.aliases:
                properties["aliases"] = entity.aliases
            if source_document:
                properties["source_document"] = source_document
            properties["name"] = name

            if not validate_node(label, properties, self._schema):
                message = f"Invalid node schema for entity '{name}' ({label})"
                stats.errors.append(message)
                logger.warning(message)
                continue

            try:
                created = await self.create_node(
                    name=name,
                    label=label,
                    properties=properties,
                )
            except Exception as exc:
                message = f"Failed to create node '{name}': {exc}"
                stats.errors.append(message)
                logger.error(message)
                continue

            if created:
                stats.nodes_created += 1
            else:
                stats.nodes_updated += 1
            name_to_label[name] = label

        # Process triples and auto-create missing nodes if enabled
        for triple in triples:
            source_name = triple.subject.strip()
            target_name = triple.object.strip()
            source_label = name_to_label.get(source_name)
            target_label = name_to_label.get(target_name)

            # Auto-create missing source node
            if not source_label and self._auto_create_missing_nodes:
                source_label = "自动创建"
                properties = {"name": source_name, "auto_created": True}
                if source_document:
                    properties["source_document"] = source_document
                try:
                    created = await self.create_node(
                        name=source_name,
                        label=source_label,
                        properties=properties,
                    )
                    if created:
                        stats.nodes_created += 1
                    else:
                        stats.nodes_updated += 1
                    name_to_label[source_name] = source_label
                    logger.info(f"Auto-created missing node: {source_name}")
                except Exception as exc:
                    message = f"Failed to auto-create node '{source_name}': {exc}"
                    stats.errors.append(message)
                    logger.error(message)
                    continue

            # Auto-create missing target node
            if not target_label and self._auto_create_missing_nodes:
                target_label = "自动创建"
                properties = {"name": target_name, "auto_created": True}
                if source_document:
                    properties["source_document"] = source_document
                try:
                    created = await self.create_node(
                        name=target_name,
                        label=target_label,
                        properties=properties,
                    )
                    if created:
                        stats.nodes_created += 1
                    else:
                        stats.nodes_updated += 1
                    name_to_label[target_name] = target_label
                    logger.info(f"Auto-created missing node: {target_name}")
                except Exception as exc:
                    message = f"Failed to auto-create node '{target_name}': {exc}"
                    stats.errors.append(message)
                    logger.error(message)
                    continue

            # Skip if still missing after auto-create attempt
            if not source_label or not target_label:
                message = (
                    "Skipping relation with missing endpoint labels: "
                    f"{triple.subject} -[{triple.predicate}]-> {triple.object}"
                )
                stats.errors.append(message)
                logger.warning(message)
                continue

            if not validate_relation(
                triple.predicate,
                source_label,
                target_label,
                self._schema,
            ):
                message = (
                    "Invalid relation schema for triple: "
                    f"{triple.subject} -[{triple.predicate}]-> {triple.object}"
                )
                stats.errors.append(message)
                logger.warning(message)
                continue

            properties = dict(triple.properties)
            relation_source = source_document or triple.source
            if relation_source:
                properties["source_document"] = relation_source

            try:
                created = await self.create_relation(
                    source_name=triple.subject,
                    relation_type=triple.predicate,
                    target_name=triple.object,
                    properties=properties,
                )
            except Exception as exc:
                message = (
                    "Failed to create relation "
                    f"{triple.subject} -[{triple.predicate}]-> {triple.object}: {exc}"
                )
                stats.errors.append(message)
                logger.error(message)
                continue

            if created:
                stats.relations_created += 1

        return stats

    async def create_node(
        self,
        name: str,
        label: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """Create or update a single node via MERGE."""

        label_value = str(label)
        node_properties = dict(properties or {})
        node_properties.setdefault("name", name)

        exists_query = (
            f"MATCH (n:`{label_value}` {{name: $name}}) RETURN count(n) as count"
        )
        existing = await self._neo4j_client.execute(exists_query, {"name": name})
        already_exists = bool(existing and existing[0].get("count"))

        query, params = self._build_node_cypher(name, label_value, node_properties)
        await self._neo4j_client.execute(query, params)
        return not already_exists

    async def create_relation(
        self,
        source_name: str,
        relation_type: str,
        target_name: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """Create a relationship between existing nodes."""

        relation_value = str(relation_type)
        relation_properties = dict(properties or {})
        query, params = self._build_relation_cypher(
            source_name,
            relation_value,
            target_name,
            relation_properties,
        )
        records = await self._neo4j_client.execute(query, params)
        if not records:
            return False
        created = records[0].get("created")
        return bool(created)

    async def create_indexes(self) -> None:
        """Create indexes for common node lookups."""

        for label in self._schema.nodes:
            label_value = str(label)
            query = f"CREATE INDEX IF NOT EXISTS FOR (n:`{label_value}`) ON (n.name)"
            await self._neo4j_client.execute(query)

    async def clear_graph(self) -> None:
        """Delete all nodes and relationships from the graph."""

        await self._neo4j_client.execute("MATCH (n) DETACH DELETE n")

    def _build_node_cypher(
        self, name: str, label: str, properties: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Build the Cypher query for creating/updating a node."""

        query = f"MERGE (n:`{label}` {{name: $name}}) SET n += $properties"
        return query, {"name": name, "properties": properties}

    def _build_relation_cypher(
        self,
        source: str,
        rel_type: str,
        target: str,
        properties: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Build the Cypher query for creating a relationship."""

        query = (
            f"MATCH (a {{name: $source}}), (b {{name: $target}}) "
            f"CREATE (a)-[r:`{rel_type}`]->(b) "
            "SET r += $properties "
            "RETURN count(r) as created"
        )
        return query, {
            "source": source,
            "target": target,
            "properties": properties,
        }
