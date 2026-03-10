"""Schema Induction: discover domain entity and relation types from samples.

Given a set of document sections from the same vertical domain, uses a Teacher LLM
to analyze representative samples and produce a structured domain schema (the "Domain
Constitution") that constrains all subsequent extraction.

Typical usage:

    inducer = SchemaInducer(llm_client)
    schema = await inducer.induce(sections, n_samples=20)
    schema.save("config/domain_schema.json")
    # Then pass schema.entity_types / schema.relation_types to extractors
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.logger import get_logger
from src.data_processing.document_loader import Section
from src.data_processing.relation_types import RelationType
from src.llm.base_client import BaseLLMClient, GenerationParams

logger = get_logger(__name__)


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class EntityTypeSpec:
    """A domain-specific entity type discovered during schema induction."""

    name: str  # English identifier, e.g. "protocol"
    label: str  # Chinese display label, e.g. "协议"
    definition: str  # One-sentence definition
    examples: list[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        block = f"  【{self.label}】({self.name})"
        block += f"\n    定义：{self.definition}"
        if self.examples:
            block += f"\n    示例：{'、'.join(self.examples)}"
        return block


@dataclass
class RelationTypeSpec:
    """A domain-specific relation type discovered during schema induction."""

    name: str  # English identifier
    label: str  # Chinese display label
    definition: str
    source_types: list[str] = field(default_factory=list)
    target_types: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)

    def to_relation_type(self) -> RelationType:
        """Convert to the existing RelationType dataclass for prompt building."""
        direction = ""
        if self.source_types and self.target_types:
            src = "/".join(self.source_types[:2])
            tgt = "/".join(self.target_types[:2])
            direction = f"{src} → {tgt}"
        return RelationType(
            name=self.name,
            label=self.label,
            definition=self.definition,
            examples=self.examples,
            direction_hint=direction,
        )


@dataclass
class ConstraintRule:
    """A valid (source_type, relation, target_type) combination."""

    source_type: str
    relation: str
    target_type: str


@dataclass
class DomainSchema:
    """The complete domain schema — the 'Domain Constitution'."""

    domain_name: str = ""
    entity_types: list[EntityTypeSpec] = field(default_factory=list)
    relation_types: list[RelationTypeSpec] = field(default_factory=list)
    constraints: list[ConstraintRule] = field(default_factory=list)

    # ── Convenience accessors ────────────────────────────────

    def entity_type_names(self) -> list[str]:
        return [et.name for et in self.entity_types]

    def entity_type_labels(self) -> list[str]:
        return [et.label for et in self.entity_types]

    def to_relation_types(self) -> list[RelationType]:
        """Convert relation type specs to RelationType objects for extractors."""
        return [rt.to_relation_type() for rt in self.relation_types]

    def get_entity_type_map(self) -> dict[str, EntityTypeSpec]:
        return {et.name: et for et in self.entity_types}

    def get_valid_combinations(self) -> set[tuple[str, str, str]]:
        return {(c.source_type, c.relation, c.target_type) for c in self.constraints}

    # ── Serialization ────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_name": self.domain_name,
            "entity_types": [
                {
                    "name": et.name,
                    "label": et.label,
                    "definition": et.definition,
                    "examples": et.examples,
                }
                for et in self.entity_types
            ],
            "relation_types": [
                {
                    "name": rt.name,
                    "label": rt.label,
                    "definition": rt.definition,
                    "source_types": rt.source_types,
                    "target_types": rt.target_types,
                    "examples": rt.examples,
                }
                for rt in self.relation_types
            ],
            "constraints": [
                {
                    "source_type": c.source_type,
                    "relation": c.relation,
                    "target_type": c.target_type,
                }
                for c in self.constraints
            ],
        }

    def save(self, path: str | Path) -> None:
        """Persist schema to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Domain schema saved to {p}")

    @classmethod
    def load(cls, path: str | Path) -> DomainSchema:
        """Load schema from a JSON file."""
        p = Path(path)
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainSchema:
        entity_types = [
            EntityTypeSpec(
                name=et["name"],
                label=et["label"],
                definition=et["definition"],
                examples=et.get("examples", []),
            )
            for et in data.get("entity_types", [])
        ]
        relation_types = [
            RelationTypeSpec(
                name=rt["name"],
                label=rt["label"],
                definition=rt["definition"],
                source_types=rt.get("source_types", []),
                target_types=rt.get("target_types", []),
                examples=rt.get("examples", []),
            )
            for rt in data.get("relation_types", [])
        ]
        constraints = [
            ConstraintRule(
                source_type=c["source_type"],
                relation=c["relation"],
                target_type=c["target_type"],
            )
            for c in data.get("constraints", [])
        ]
        return cls(
            domain_name=data.get("domain_name", ""),
            entity_types=entity_types,
            relation_types=relation_types,
            constraints=constraints,
        )

    # ── Prompt helpers for extractors ────────────────────────

    def build_entity_type_prompt(self) -> str:
        """Build a prompt section listing all entity types with definitions."""
        if not self.entity_types:
            return ""
        lines = ["【领域实体类型定义】（type 字段必须使用以下类型之一的 name 值）："]
        for et in self.entity_types:
            lines.append(et.to_prompt_block())
        lines.append("")
        return "\n".join(lines)

    def build_constraint_prompt(self) -> str:
        """Build a prompt section listing valid (source, relation, target) combos."""
        if not self.constraints:
            return ""
        lines = ["【合法三元组组合约束】（只允许以下类型组合）："]
        for c in self.constraints:
            lines.append(f"  {c.source_type} --[{c.relation}]--> {c.target_type}")
        lines.append("")
        return "\n".join(lines)


# ── Schema Inducer ───────────────────────────────────────────────────


class SchemaInducer:
    """Analyze document samples with a Teacher LLM to discover domain schema."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        max_new_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> None:
        self._llm_client = llm_client
        self._logger = get_logger(__name__)
        self._generation_params = GenerationParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            enable_thinking=False,
            system_message=(
                "你是领域知识建模专家。你的任务是分析垂直领域文本，"
                "归纳出该领域的实体类型和关系类型体系。"
                "请严格以JSON格式输出结果。"
            ),
        )

    # ── Public API ───────────────────────────────────────────

    async def induce(
        self,
        sections: list[Section],
        n_samples: int = 20,
        domain_hint: str = "",
    ) -> DomainSchema:
        """Analyze sampled sections and produce a domain schema.

        Args:
            sections: All sections from one or more domain documents.
            n_samples: Number of sections to sample for analysis.
            domain_hint: Optional hint about the domain (e.g. "计算机网络").

        Returns:
            A DomainSchema with induced entity types, relation types,
            and valid combination constraints.
        """
        samples = self._sample_sections(sections, n_samples)
        self._logger.info(
            f"Schema induction: analyzing {len(samples)} sample sections "
            f"from {len(sections)} total"
        )

        prompt = self._build_induction_prompt(samples, domain_hint)
        response = await self._llm_client.generate(
            prompt=prompt, params=self._generation_params
        )
        schema = self._parse_schema_response(response, domain_hint)

        self._logger.info(
            f"Schema induction complete: "
            f"{len(schema.entity_types)} entity types, "
            f"{len(schema.relation_types)} relation types, "
            f"{len(schema.constraints)} constraints"
        )
        return schema

    # ── Sampling strategy ────────────────────────────────────

    def _sample_sections(
        self, sections: list[Section], n_samples: int
    ) -> list[Section]:
        """Evenly sample sections from across the document(s)."""
        if len(sections) <= n_samples:
            return list(sections)

        # Stratified sampling: take evenly spaced sections
        step = len(sections) / n_samples
        indices = [int(i * step) for i in range(n_samples)]
        # Ensure we don't have duplicates at boundaries
        unique_indices = list(dict.fromkeys(indices))
        sampled = [sections[i] for i in unique_indices]

        # If we still need more, fill randomly from remaining
        remaining_indices = set(range(len(sections))) - set(unique_indices)
        if len(sampled) < n_samples and remaining_indices:
            extra = random.sample(
                list(remaining_indices),
                min(n_samples - len(sampled), len(remaining_indices)),
            )
            sampled.extend(sections[i] for i in sorted(extra))

        return sampled

    # ── Prompt construction ──────────────────────────────────

    def _build_induction_prompt(self, samples: list[Section], domain_hint: str) -> str:
        parts: list[str] = [
            "你是领域知识建模专家。我会给你多段来自同一垂直领域的文本样本，"
            "请分析这些文本，归纳出该领域的知识体系结构。\n",
        ]

        if domain_hint:
            parts.append(f"领域提示：{domain_hint}\n")

        parts.append(
            "请归纳以下内容：\n\n"
            "1. **实体类型**（5-15个），每个类型需要：\n"
            "   - name: 英文标识符（如 protocol, algorithm, device）\n"
            "   - label: 中文标签\n"
            "   - definition: 一句话定义该类型在该领域中的含义\n"
            "   - examples: 从给定文本中提取2-3个该类型的实例\n\n"
            "2. **关系类型**（5-15个），每个类型需要：\n"
            "   - name: 英文标识符（如 causes, contains, depends_on）\n"
            "   - label: 中文标签\n"
            "   - definition: 一句话定义\n"
            "   - source_types: 合法的头实体类型列表（使用实体类型的 name）\n"
            "   - target_types: 合法的尾实体类型列表\n"
            "   - examples: 从文本中提取的2-3个该关系的实例"
            "（格式：'实体A → 关系 → 实体B'）\n\n"
            "3. **约束规则**：列出所有合法的 "
            "(source_type, relation, target_type) 组合\n\n"
            "要求：\n"
            "- 实体类型要有区分度，不要过于宽泛（如'概念'、'事物'）"
            "也不要过于细碎\n"
            "- 关系类型要具体、有信息量，"
            "禁止使用'相关'、'有关'等模糊关系\n"
            "- 类型体系应覆盖给定文本中的绝大多数知识\n"
            "- 只返回JSON对象，不要有其他文字\n\n"
        )

        # Output format example
        parts.append(
            "输出格式：\n"
            "```json\n"
            "{\n"
            '  "domain_name": "领域名称",\n'
            '  "entity_types": [\n'
            '    {"name": "protocol", "label": "协议", '
            '"definition": "网络通信中的规则和标准", '
            '"examples": ["TCP", "HTTP", "BGP"]}\n'
            "  ],\n"
            '  "relation_types": [\n'
            '    {"name": "depends_on", "label": "依赖", '
            '"definition": "A的运行需要B的支持", '
            '"source_types": ["protocol", "system"], '
            '"target_types": ["protocol", "component"], '
            '"examples": ["TCP → 依赖 → IP协议"]}\n'
            "  ],\n"
            '  "constraints": [\n'
            '    {"source_type": "protocol", "relation": "depends_on", '
            '"target_type": "protocol"}\n'
            "  ]\n"
            "}\n"
            "```\n\n"
        )

        # Append sample texts
        parts.append("以下是文本样本：\n")
        for i, section in enumerate(samples):
            heading = " > ".join(section.heading_chain) if section.heading_chain else ""
            header = f"--- 样本 {i + 1}"
            if heading:
                header += f" [{heading}]"
            header += " ---"
            parts.append(header)
            # Truncate very long sections
            content = section.content
            text = content[:2000] if len(content) > 2000 else content
            parts.append(text)
            parts.append("")

        parts.append("请直接返回JSON对象：")
        return "\n".join(parts)

    # ── Response parsing ─────────────────────────────────────

    def _parse_schema_response(self, response: str, domain_hint: str) -> DomainSchema:
        """Parse the LLM response into a DomainSchema object."""

        # Strip thinking tags
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", response).strip()
        if not cleaned:
            cleaned = response

        payload = None

        # Try markdown code block
        code_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```", cleaned)
        if code_match:
            try:
                payload = json.loads(code_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Direct parse
        if payload is None:
            try:
                payload = json.loads(cleaned.strip())
            except json.JSONDecodeError:
                pass

        # Brace search
        if payload is None:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end > start:
                try:
                    payload = json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    pass

        if payload is None or not isinstance(payload, dict):
            self._logger.error(
                f"Failed to parse schema induction response: {cleaned[:300]}"
            )
            return DomainSchema(domain_name=domain_hint)

        try:
            schema = DomainSchema.from_dict(payload)
            if not schema.domain_name:
                schema.domain_name = domain_hint
            return schema
        except (KeyError, TypeError) as exc:
            self._logger.error(f"Schema parsing error: {exc}")
            return DomainSchema(domain_name=domain_hint)
