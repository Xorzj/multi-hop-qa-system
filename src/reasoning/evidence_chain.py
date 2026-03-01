from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvidenceNode:
    name: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)
    hop: int = 0


@dataclass
class EvidenceEdge:
    source: str
    target: str
    relation_type: str
    confidence: float = 1.0


@dataclass
class EvidenceStep:
    hop_number: int
    action: str
    nodes_explored: list[str] = field(default_factory=list)
    relation_used: str | None = None
    reasoning: str = ""


@dataclass
class EvidenceChain:
    nodes: list[EvidenceNode] = field(default_factory=list)
    edges: list[EvidenceEdge] = field(default_factory=list)
    steps: list[EvidenceStep] = field(default_factory=list)
    start_entity: str = ""
    end_entity: str | None = None
    total_confidence: float = 1.0

    def add_node(self, node: EvidenceNode) -> None:
        self.nodes.append(node)

    def add_edge(self, edge: EvidenceEdge) -> None:
        self.edges.append(edge)

    def add_step(self, step: EvidenceStep) -> None:
        self.steps.append(step)

    def get_path(self) -> list[str]:
        if not self.edges:
            return [node.name for node in self.nodes] if self.nodes else []

        ordered_names = [self.edges[0].source]
        for edge in self.edges:
            ordered_names.append(edge.target)
        return ordered_names

    def get_path_description(self) -> str:
        if not self.edges:
            return " -- ".join(self.get_path())

        parts: list[str] = [self.edges[0].source]
        for edge in self.edges:
            parts.append(f"--{edge.relation_type}-->")
            parts.append(edge.target)
        return " ".join(parts)

    def calculate_confidence(self) -> float:
        total = 1.0
        for edge in self.edges:
            total *= edge.confidence
        self.total_confidence = total
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "name": node.name,
                    "label": node.label,
                    "properties": node.properties,
                    "hop": node.hop,
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation_type": edge.relation_type,
                    "confidence": edge.confidence,
                }
                for edge in self.edges
            ],
            "steps": [
                {
                    "hop_number": step.hop_number,
                    "action": step.action,
                    "nodes_explored": step.nodes_explored,
                    "relation_used": step.relation_used,
                    "reasoning": step.reasoning,
                }
                for step in self.steps
            ],
            "start_entity": self.start_entity,
            "end_entity": self.end_entity,
            "total_confidence": self.total_confidence,
            "path": self.get_path(),
            "path_description": self.get_path_description(),
        }


class EvidenceChainBuilder:
    def __init__(self, start_entity: str) -> None:
        self.start_entity = start_entity
        self._nodes: list[EvidenceNode] = []
        self._edges: list[EvidenceEdge] = []
        self._steps: list[EvidenceStep] = []
        self._current_hop = 0

    def add_hop(
        self,
        nodes: list[EvidenceNode],
        edges: list[EvidenceEdge],
        reasoning: str,
    ) -> None:
        self._current_hop += 1
        for node in nodes:
            node.hop = self._current_hop
            self._nodes.append(node)

        self._edges.extend(edges)

        relation_used = edges[0].relation_type if edges else None
        step = EvidenceStep(
            hop_number=self._current_hop,
            action=f"expand_hop_{self._current_hop}",
            nodes_explored=[node.name for node in nodes],
            relation_used=relation_used,
            reasoning=reasoning,
        )
        self._steps.append(step)
        logger.debug(
            "Added hop",
            extra={
                "hop_number": self._current_hop,
                "nodes": [node.name for node in nodes],
                "edges": len(edges),
            },
        )

    def finalize(self, end_entity: str | None = None) -> EvidenceChain:
        chain = EvidenceChain(
            nodes=list(self._nodes),
            edges=list(self._edges),
            steps=list(self._steps),
            start_entity=self.start_entity,
            end_entity=end_entity,
        )
        chain.calculate_confidence()
        return chain

    def get_current_frontier(self) -> list[str]:
        if not self._nodes:
            return [self.start_entity]
        return [node.name for node in self._nodes if node.hop == self._current_hop]
