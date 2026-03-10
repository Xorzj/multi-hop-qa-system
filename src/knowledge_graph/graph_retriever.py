from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.common.logger import get_logger
from src.knowledge_graph.cypher_builder import CypherBuilder, CypherQuery
from src.knowledge_graph.neo4j_client import Neo4jClient

logger = get_logger(__name__)


@dataclass(slots=True)
class GraphNode:
    name: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphRelation:
    source: str
    target: str
    relation_type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HopResult:
    nodes: list[GraphNode]
    relations: list[GraphRelation]
    hop_number: int


class GraphRetriever:
    def __init__(
        self, neo4j_client: Neo4jClient, cypher_builder: CypherBuilder | None = None
    ) -> None:
        self._neo4j_client = neo4j_client
        self._cypher_builder = cypher_builder or CypherBuilder()

    async def get_node(self, name: str) -> GraphNode | None:
        query = self._cypher_builder.find_node(name)
        records = await self._execute(query)
        if not records:
            return None
        node = records[0].get("n")
        if node is None:
            return None
        return self._parse_node({"_node": node})

    async def get_neighbors(
        self,
        node_name: str,
        relation_type: str | None = None,
        direction: Literal["out", "in", "both"] = "both",
        limit: int = 20,
        neighbor_label: str | None = None,
    ) -> HopResult:
        direction_value: Literal["out", "in", "both"] = direction
        query = self._cypher_builder.find_neighbors(
            node_name=node_name,
            relation_type=relation_type,
            direction=direction_value,
            limit=limit,
            neighbor_label=neighbor_label,
        )
        records = await self._execute(query)
        nodes: list[GraphNode] = []
        relations: list[GraphRelation] = []
        for record in records:
            neighbor = record.get("m")
            relation = record.get("r")
            rel_type = record.get("rel_type")
            if neighbor is not None:
                nodes.append(self._parse_node({"_node": neighbor}))
            # 优先使用 Cypher 返回的真实方向
            rel_start = record.get("rel_start")
            rel_end = record.get("rel_end")
            if rel_start is not None and rel_end is not None:
                source, target = str(rel_start), str(rel_end)
            else:
                # 回退到推断方向（仅当 Cypher 不返回方向信息时）
                source, target = self._infer_relation_endpoints(
                    node_name=node_name,
                    neighbor=neighbor,
                    direction=direction,
                )
            if relation is not None:
                relation_record = {
                    "_relation": relation,
                    "relation_type": rel_type,
                    "source": source,
                    "target": target,
                }
                relations.append(self._parse_relation(relation_record))
        return HopResult(nodes=nodes, relations=relations, hop_number=1)

    async def get_labeled_nearby(
        self,
        node_name: str,
        target_label: str,
        max_hops: int = 3,
        limit: int = 20,
    ) -> list[GraphNode]:
        """Find nodes with a specific label within N hops (undirected)."""
        query = self._cypher_builder.find_labeled_nearby(
            node_name, target_label, max_hops, limit
        )
        records = await self._execute(query)
        return [self._parse_node({"_node": r.get("m")}) for r in records if r.get("m")]

    async def get_path(
        self, start: str, end: str, max_hops: int = 3, directed: bool = True,
    ) -> list[HopResult] | None:
        query = self._cypher_builder.find_path(start, end, max_hops, directed=directed)
        records = await self._execute(query)
        if not records:
            return None
        return self._parse_path_rows(records)

    async def search_nodes(self, query: str, limit: int = 10) -> list[GraphNode]:
        cypher = CypherQuery(
            query="MATCH (n) WHERE n.name CONTAINS $query RETURN n LIMIT $limit",
            parameters={"query": query, "limit": limit},
        )
        records = await self._execute(cypher)
        nodes: list[GraphNode] = []
        for record in records:
            node = record.get("n")
            if node is not None:
                nodes.append(self._parse_node({"_node": node}))
        return nodes

    async def read_context(self, name: str) -> dict[str, Any]:
        """Return node properties and source texts from connected edges.

        This is an atomic graph operation used by the reasoning plane to
        retrieve rich context for an entity, including the original chunk
        text that mentions it (stored as edge properties during graph build).

        Returns:
            Dict with ``node`` (GraphNode or None), ``source_texts`` (list
            of unique source text snippets from connected edges).
        """
        node = await self.get_node(name)
        # Fetch edges connected to this node and collect source_text properties
        cypher = CypherQuery(
            query=(
                "MATCH (n {name: $name})-[r]-() "
                "WHERE r.source_text IS NOT NULL "
                "RETURN DISTINCT r.source_text AS source_text, "
                "r.source_chunk_id AS chunk_id "
                "LIMIT 10"
            ),
            parameters={"name": name},
        )
        records = await self._execute(cypher)
        source_texts: list[dict[str, str]] = []
        seen: set[str] = set()
        for record in records:
            text = record.get("source_text")
            chunk_id = record.get("chunk_id", "")
            if isinstance(text, str) and text not in seen:
                seen.add(text)
                source_texts.append({"chunk_id": str(chunk_id), "text": text})
        return {"node": node, "source_texts": source_texts}

    async def get_node_context(self, name: str, depth: int = 1) -> list[HopResult]:
        if depth <= 0:
            return []
        results: list[HopResult] = []
        frontier = [name]
        seen: set[str] = {name}
        for hop in range(1, depth + 1):
            hop_nodes: dict[str, GraphNode] = {}
            hop_relations: list[GraphRelation] = []
            next_frontier: list[str] = []
            for node_name in frontier:
                hop_result = await self.get_neighbors(node_name=node_name)
                for node in hop_result.nodes:
                    if node.name and node.name not in hop_nodes:
                        hop_nodes[node.name] = node
                    if node.name and node.name not in seen:
                        seen.add(node.name)
                        next_frontier.append(node.name)
                hop_relations.extend(hop_result.relations)
            results.append(
                HopResult(
                    nodes=list(hop_nodes.values()),
                    relations=hop_relations,
                    hop_number=hop,
                )
            )
            if not next_frontier:
                break
            frontier = next_frontier
        return results

    def _parse_node(self, record: dict[str, Any]) -> GraphNode:
        node = record.get("_node", record)
        properties = self._extract_properties(node)
        labels = self._extract_labels(node)
        name_value = properties.get("name")
        name = str(name_value) if name_value is not None else ""
        label = labels[0] if labels else ""
        return GraphNode(name=name, label=label, properties=properties)

    def _parse_relation(self, record: dict[str, Any]) -> GraphRelation:
        relation = record.get("_relation", record)
        properties = self._extract_properties(relation)
        relation_type = record.get("relation_type") or record.get("rel_type")
        if relation_type is None:
            relation_type = getattr(relation, "type", "")
        source, target = self._relation_endpoints(relation, record)
        return GraphRelation(
            source=source,
            target=target,
            relation_type=str(relation_type) if relation_type else "",
            properties=properties,
        )

    async def _execute(self, query: CypherQuery) -> list[dict[str, Any]]:
        logger.debug("Executing Cypher query", extra={"query": query.query})
        return await self._neo4j_client.execute(query.query, query.parameters)

    def _extract_labels(self, node: Any) -> list[str]:
        if hasattr(node, "labels"):
            return [str(label) for label in node.labels]
        if isinstance(node, dict):
            labels = node.get("labels")
            if isinstance(labels, list):
                return [str(label) for label in labels]
            label = node.get("label")
            if label is not None:
                return [str(label)]
        return []

    def _extract_properties(self, entity: Any) -> dict[str, Any]:
        if isinstance(entity, dict):
            properties = entity.get("properties")
            if isinstance(properties, dict):
                return dict(properties)
            return dict(entity)
        try:
            return dict(entity)
        except (TypeError, ValueError):
            properties = getattr(entity, "_properties", None)
            if isinstance(properties, dict):
                return dict(properties)
        return {}

    def _relation_endpoints(
        self, relation: Any, record: dict[str, Any]
    ) -> tuple[str, str]:
        source = record.get("source")
        target = record.get("target")
        start_node = getattr(relation, "start_node", None)
        end_node = getattr(relation, "end_node", None)
        if start_node is not None and end_node is not None:
            source = self._node_name(start_node)
            target = self._node_name(end_node)
        return (
            str(source) if source is not None else "",
            str(target) if target is not None else "",
        )

    def _node_name(self, node: Any) -> str:
        properties = self._extract_properties(node)
        name_value = properties.get("name")
        return str(name_value) if name_value is not None else ""

    def _infer_relation_endpoints(
        self,
        node_name: str,
        neighbor: Any,
        direction: Literal["out", "in", "both"],
    ) -> tuple[str, str]:
        neighbor_name = self._node_name(neighbor) if neighbor is not None else ""
        if direction == "in":
            return neighbor_name, node_name
        return node_name, neighbor_name

    def _parse_path_rows(self, records: list[dict[str, Any]]) -> list[HopResult]:
        """Parse UNWIND-ed path rows (one record per hop edge)."""
        hop_results: list[HopResult] = []
        for record in records:
            hop_idx = record.get("hop_idx", 0)
            source_node = record.get("source_node")
            target_node = record.get("target_node")
            rel_type = record.get("rel_type", "")
            rel_props = record.get("rel_props") or {}
            rel_start = record.get("rel_start")
            rel_end = record.get("rel_end")

            hop_nodes: list[GraphNode] = []
            if source_node is not None:
                hop_nodes.append(self._parse_node({"_node": source_node}))
            if target_node is not None:
                hop_nodes.append(self._parse_node({"_node": target_node}))

            # Build relation with source/target from Neo4j direction
            if rel_start is not None and rel_end is not None:
                source, target = str(rel_start), str(rel_end)
            elif source_node and target_node:
                source = self._node_name(source_node)
                target = self._node_name(target_node)
            else:
                source, target = "", ""

            hop_relations = [
                GraphRelation(
                    source=source,
                    target=target,
                    relation_type=str(rel_type),
                    properties=dict(rel_props) if isinstance(rel_props, dict) else {},
                )
            ]
            hop_results.append(
                HopResult(
                    nodes=hop_nodes,
                    relations=hop_relations,
                    hop_number=hop_idx + 1,
                )
            )
        return hop_results
