from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.common.logger import get_logger
from src.knowledge_graph.schema import NodeLabel, RelationType

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CypherQuery:
    query: str
    parameters: dict[str, Any]


class CypherBuilder:
    def find_node(self, name: str, label: NodeLabel | str | None = None) -> CypherQuery:
        label_filter = self._build_label_filter(str(label) if label else None)
        query = f"MATCH (n{label_filter} {{name: $name}}) RETURN n"
        return CypherQuery(query=query, parameters={"name": name})

    def find_neighbors(
        self,
        node_name: str,
        relation_type: RelationType | str | None = None,
        direction: Literal["out", "in", "both"] = "both",
        limit: int = 20,
        neighbor_label: str | None = None,
    ) -> CypherQuery:
        left, right = self._build_direction(direction)
        rel_type = self._build_relation_filter(
            str(relation_type) if relation_type else None
        )
        label_filter = self._build_label_filter(neighbor_label)
        query = (
            "MATCH (n {name: $name})"
            f"{left}[r{rel_type}]{right}(m{label_filter}) "
            "RETURN m, r, type(r) as rel_type, "
            "startNode(r).name as rel_start, endNode(r).name as rel_end "
            "LIMIT $limit"
        )
        return CypherQuery(query=query, parameters={"name": node_name, "limit": limit})

    def find_labeled_nearby(
        self,
        node_name: str,
        target_label: str,
        max_hops: int = 3,
        limit: int = 20,
    ) -> CypherQuery:
        """Find nodes with a specific label within N hops (undirected)."""
        safe_label = self._escape_string(target_label)
        hops = int(max_hops)
        query = (
            "MATCH (n {name: $name})-[*1.." + str(hops) + "]-(m:`" + safe_label + "`) "
            "WHERE m.name IS NOT NULL AND m.name <> $name "
            "RETURN DISTINCT m "
            "LIMIT $limit"
        )
        return CypherQuery(query=query, parameters={"name": node_name, "limit": limit})

    def find_path(
        self,
        start_name: str,
        end_name: str,
        max_hops: int = 3,
        directed: bool = True,
    ) -> CypherQuery:
        # Neo4j shortestPath requires literal range bounds (no parameters).
        # UNWIND the path so result.data() returns per-hop rows with full
        # edge properties (source_text, etc.) instead of a flattened list.
        hops = int(max_hops)
        arrow = "->" if directed else "-"
        query = (
            "MATCH path = shortestPath("
            f"(a {{name: $start}})-[*1..{hops}]{arrow}(b {{name: $end}})"
            ") "
            "WITH nodes(path) AS ns, relationships(path) AS rs "
            "UNWIND range(0, size(rs)-1) AS i "
            "RETURN ns[i] AS source_node, ns[i+1] AS target_node, "
            "type(rs[i]) AS rel_type, properties(rs[i]) AS rel_props, "
            "startNode(rs[i]).name AS rel_start, "
            "endNode(rs[i]).name AS rel_end, "
            "i AS hop_idx"
        )
        return CypherQuery(
            query=query,
            parameters={"start": start_name, "end": end_name},
        )

    def find_by_property(
        self,
        label: NodeLabel | str,
        property_name: str,
        property_value: Any,
    ) -> CypherQuery:
        label_filter = self._build_label_filter(str(label))
        property_key = self._escape_string(property_name)
        query = f"MATCH (n{label_filter}) WHERE n.`{property_key}` = $value RETURN n"
        return CypherQuery(query=query, parameters={"value": property_value})

    def get_node_properties(self, name: str) -> CypherQuery:
        query = "MATCH (n {name: $name}) RETURN properties(n) as properties"
        return CypherQuery(query=query, parameters={"name": name})

    def get_relation_properties(
        self,
        source: str,
        target: str,
        relation_type: str,
    ) -> CypherQuery:
        rel_filter = self._build_relation_filter(relation_type)
        query = (
            "MATCH (a {name: $source})"
            f"-[r{rel_filter}]->"
            "(b {name: $target}) "
            "RETURN properties(r) as properties"
        )
        return CypherQuery(
            query=query,
            parameters={"source": source, "target": target},
        )

    def count_neighbors(
        self, node_name: str, relation_type: str | None = None
    ) -> CypherQuery:
        rel_filter = self._build_relation_filter(relation_type)
        query = (
            "MATCH (n {name: $name})"
            f"-[r{rel_filter}]-() "
            "RETURN count(r) as neighbor_count"
        )
        return CypherQuery(query=query, parameters={"name": node_name})

    def _escape_string(self, value: str) -> str:
        return value.replace("`", "``")

    def _build_label_filter(self, label: str | None) -> str:
        if not label:
            return ""
        safe_label = self._escape_string(label)
        return f":`{safe_label}`"

    def _build_relation_filter(self, relation_type: str | None) -> str:
        if not relation_type:
            return ""
        safe_rel = self._escape_string(relation_type)
        return f":`{safe_rel}`"

    def _build_direction(self, direction: str) -> tuple[str, str]:
        if direction == "out":
            return "-", "->"
        if direction == "in":
            return "<-", "-"
        return "-", "-"
