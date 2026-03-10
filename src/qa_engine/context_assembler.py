from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.common.logger import get_logger
from src.reasoning.evidence_chain import EvidenceChain

logger = get_logger(__name__)


@dataclass
class AssembledContext:
    question: str
    evidence_summary: str
    evidence_confidence: float = 1.0
    reasoning_steps: list[str] = field(default_factory=list)
    entity_descriptions: dict[str, str] = field(default_factory=dict)
    source_citations: list[dict[str, str]] = field(default_factory=list)
    prompt: str = ""


class ContextAssembler:
    def __init__(
        self, max_context_length: int = 4000, include_cot: bool = True
    ) -> None:
        self._max_context_length = max_context_length
        self._include_cot = include_cot
        self._logger = get_logger(__name__)

    def assemble(
        self, question: str, evidence: EvidenceChain, include_reasoning: bool = True
    ) -> AssembledContext:
        evidence_summary = self._summarize_evidence(evidence)
        reasoning_steps = (
            self._format_reasoning_steps(evidence)
            if include_reasoning and self._include_cot
            else []
        )
        entity_descriptions = self._describe_entities(evidence)
        source_citations = self._collect_source_citations(evidence)
        prompt = self._build_prompt(
            question, evidence_summary, reasoning_steps, source_citations
        )
        prompt = self._truncate_if_needed(prompt, self._max_context_length)
        return AssembledContext(
            question=question,
            evidence_summary=evidence_summary,
            evidence_confidence=evidence.total_confidence,
            reasoning_steps=reasoning_steps,
            entity_descriptions=entity_descriptions,
            source_citations=source_citations,
            prompt=prompt,
        )

    def _summarize_evidence(self, evidence: EvidenceChain) -> str:
        return evidence.to_xml()

    def _format_reasoning_steps(self, evidence: EvidenceChain) -> list[str]:
        steps: list[str] = []
        for step in evidence.steps:
            if step.reasoning:
                steps.append(step.reasoning)
                continue
            nodes = "、".join(step.nodes_explored)
            relation = step.relation_used or "未知关系"
            steps.append(f"第{step.hop_number}跳: 通过{relation}探索 {nodes}")
        return steps

    def _describe_entities(self, evidence: EvidenceChain) -> dict[str, str]:
        ordered_names = evidence.get_path()
        seen = set(ordered_names)
        for node in evidence.nodes:
            if node.name not in seen:
                ordered_names.append(node.name)
                seen.add(node.name)

        entity_descriptions: dict[str, str] = {}
        node_map = {node.name: node for node in evidence.nodes}
        for name in ordered_names:
            node_entry = node_map.get(name)
            if node_entry is None:
                entity_descriptions[name] = ""
                continue
            entity_descriptions[name] = self._format_node_description(
                node_entry.properties
            )
        return entity_descriptions

    def _collect_source_citations(
        self, evidence: EvidenceChain
    ) -> list[dict[str, str]]:
        """Extract unique source text citations from evidence edges."""
        citations: list[dict[str, str]] = []
        seen_chunks: set[str] = set()
        for edge in evidence.edges:
            if edge.source_text and edge.source_chunk_id:
                if edge.source_chunk_id not in seen_chunks:
                    seen_chunks.add(edge.source_chunk_id)
                    citations.append(
                        {
                            "chunk_id": edge.source_chunk_id,
                            "text": edge.source_text,
                            "relation": (
                                f"{edge.source} --{edge.relation_type}--> {edge.target}"
                            ),
                        }
                    )
        return citations

    def _build_prompt(
        self,
        question: str,
        evidence_summary: str,
        reasoning_steps: list[str],
        source_citations: list[dict[str, str]] | None = None,
    ) -> str:
        sections = [
            "你是一个专业的知识问答助手。请基于以下知识图谱证据回答用户问题。",
            "",
            "## 用户问题",
            question,
            "",
            "## 知识图谱证据（XML结构化格式）",
            evidence_summary,
        ]

        if reasoning_steps:
            sections.extend(["", "## 推理路径", "\n".join(reasoning_steps)])

        if source_citations:
            sections.extend(["", "## 原文引用"])
            for i, citation in enumerate(source_citations[:5], 1):
                sections.append(f"[{i}] ({citation['chunk_id']}) {citation['text']}")

        sections.extend(
            [
                "",
                "## 要求",
                "- 基于证据准确回答问题",
                "- 如果证据不足，明确说明",
                "- 使用专业但易懂的语言",
                "- 在回答末尾，引用证据链中的关键路径作为支撑"
                "（格式：[证据: 实体A --关系--> 实体B]）",
                "- 如果有原文引用，在关键结论后标注引用编号（格式：[1]、[2]等）",
                "",
                "请回答：",
            ]
        )
        return "\n".join(sections)

    def _truncate_if_needed(self, text: str, max_length: int) -> str:
        if max_length <= 0:
            return ""
        if len(text) <= max_length:
            return text
        truncated = text[: max_length - 1] + "…"
        self._logger.warning(
            "Prompt truncated to fit context length",
            extra={"max_length": max_length, "original_length": len(text)},
        )
        return truncated

    @staticmethod
    def _format_node_description(properties: dict[str, Any]) -> str:
        if not properties:
            return ""
        for key in ("description", "summary", "desc"):
            value = properties.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        parts = []
        for key, value in properties.items():
            if value is None:
                continue
            parts.append(f"{key}={value}")
        return "；".join(parts)
