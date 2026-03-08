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
    ) -> HopResult:
        direction_value: Literal["out", "in", "both"] = direction
        query = self._cypher_builder.find_neighbors(
            node_name=node_name,
            relation_type=relation_type,
            direction=direction_value,
            limit=limit,
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

    async def get_path(
        self, start: str, end: str, max_hops: int = 3
    ) -> list[HopResult] | None:
        query = self._cypher_builder.find_path(start, end, max_hops)
        records = await self._execute(query)
        if not records:
            return None
        path = records[0].get("path")
        if path is None:
            return None
        return self._parse_path(path)

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

    def _parse_path(self, path: Any) -> list[HopResult]:
        nodes: list[Any] = []
        relationships: list[Any] = []
        nodes_attr = getattr(path, "nodes", None)
        if nodes_attr is not None:
            nodes = list(nodes_attr)
        elif isinstance(path, dict) and "nodes" in path:
            nodes = list(path["nodes"])
        relationships_attr = getattr(path, "relationships", None)
        if relationships_attr is not None:
            relationships = list(relationships_attr)
        elif isinstance(path, dict):
            relationships = list(path.get("relationships", []))
        hop_results: list[HopResult] = []
        for idx, relation in enumerate(relationships):
            hop_nodes: list[GraphNode] = []
            if idx < len(nodes):
                hop_nodes.append(self._parse_node({"_node": nodes[idx]}))
            if idx + 1 < len(nodes):
                hop_nodes.append(self._parse_node({"_node": nodes[idx + 1]}))
            hop_relations = [self._parse_relation({"_relation": relation})]
            hop_results.append(
                HopResult(nodes=hop_nodes, relations=hop_relations, hop_number=idx + 1)
            )
        return hop_results
