"""Predefined relation type ontology for knowledge graph extraction.

Provides a controlled vocabulary of 5 core relation types with formal
definitions, examples, and LLM-facing descriptions.  Users can extend
the default set via configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RelationType:
    """A single relation type with metadata for LLM guidance."""

    name: str
    label: str  # Chinese display label (used in prompts)
    definition: str  # Formal definition shown to LLM
    examples: list[str] = field(default_factory=list)
    direction_hint: str = ""  # e.g. "A→B" semantic description

    def to_prompt_block(self) -> str:
        """Format this relation type as a prompt-friendly block."""
        lines = [f"  【{self.label}】({self.name})"]
        lines.append(f"    定义：{self.definition}")
        if self.direction_hint:
            lines.append(f"    方向：{self.direction_hint}")
        if self.examples:
            lines.append(f"    示例：{'；'.join(self.examples)}")
        return "\n".join(lines)


# ── 5 Core relation types (matching reference repo ontology) ──────────

INCLUDE_EDGE = RelationType(
    name="include",
    label="包含关系",
    definition="上位概念包含下位概念，A是B的整体或上级分类",
    direction_hint="A 包含 B (A是更大/更高层的概念)",
    examples=[
        "心血管系统 包含 冠状动脉",
        "数据库 包含 索引",
        "可再生能源 包含 太阳能",
    ],
)

DERIVATIVE_EDGE = RelationType(
    name="derivative",
    label="派生关系",
    definition="A派生出B，B是A的衍生物、产物或后续发展",
    direction_hint="A 派生 B (B因A而产生)",
    examples=[
        "干细胞 派生 红细胞",
        "SQL 派生 NoSQL",
        "核裂变 派生 核废料",
    ],
)

CORRELATE_EDGE = RelationType(
    name="correlate",
    label="因果/关联关系",
    definition="A与B之间存在因果、促进、抑制或逻辑关联",
    direction_hint="A 作用于 B (A对B产生影响)",
    examples=[
        "高血压 导致 动脉粥样硬化",
        "索引 提升 查询性能",
        "温室效应 加剧 全球变暖",
    ],
)

COORDINATE_EDGE = RelationType(
    name="coordinate",
    label="并列关系",
    definition="A与B处于同一层级或类别，属于对等概念",
    direction_hint="A 与 B 并列 (无方向性)",
    examples=[
        "TCP 并列 UDP",
        "动脉 并列 静脉",
        "风能 并列 太阳能",
    ],
)

SYMBIOTIC_EDGE = RelationType(
    name="symbiotic",
    label="共生/依赖关系",
    definition="A与B经常共同出现，存在依赖或协作关系",
    direction_hint="A 依赖/协作 B",
    examples=[
        "TCP 依赖 IP协议",
        "血红蛋白 协作 氧气",
        "光伏板 依赖 逆变器",
    ],
)


# ── Registry ──────────────────────────────────────────────────────────

DEFAULT_RELATION_TYPES: list[RelationType] = [
    INCLUDE_EDGE,
    DERIVATIVE_EDGE,
    CORRELATE_EDGE,
    COORDINATE_EDGE,
    SYMBIOTIC_EDGE,
]

_RELATION_TYPE_MAP: dict[str, RelationType] = {
    rt.name: rt for rt in DEFAULT_RELATION_TYPES
}


def get_relation_type(name: str) -> RelationType | None:
    """Look up a relation type by name."""
    return _RELATION_TYPE_MAP.get(name)


def register_relation_type(rt: RelationType) -> None:
    """Register a custom relation type at runtime."""
    _RELATION_TYPE_MAP[rt.name] = rt
    if rt not in DEFAULT_RELATION_TYPES:
        DEFAULT_RELATION_TYPES.append(rt)


def build_relation_type_prompt(
    relation_types: list[RelationType] | None = None,
) -> str:
    """Build a formatted prompt section describing all relation types.

    Used by extractors to guide the LLM toward using the controlled vocabulary.
    """
    types = relation_types or DEFAULT_RELATION_TYPES
    lines = ["关系类型定义（必须使用以下类型之一作为 relation_type 字段的值）："]
    for rt in types:
        lines.append(rt.to_prompt_block())
    lines.append("")  # trailing newline
    return "\n".join(lines)


def get_relation_type_names(
    relation_types: list[RelationType] | None = None,
) -> list[str]:
    """Return the name strings of all registered relation types."""
    types = relation_types or DEFAULT_RELATION_TYPES
    return [rt.name for rt in types]
